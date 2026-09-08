// vkernels/kernels/dsa_kpool.hip
//
// DeepseekSparseAttn (DSA) kpool-cache compress/write kernels (gfx942 /
// MI300A), issue #60. gfx942 replacement for sglang's kpool_fp8_index.py
// Triton kernels ``kpool_assemble_softmax_rotate_write_cache`` (prefill) and
// ``kpool_decode_update_and_maybe_write_cache`` (decode), which declare
// ``tl.float8e4nv`` and therefore JIT-fail on SM80 (A100) on the first
// forward (job 82822). See dsa_kpool.hpp for the computation, the
// bf16-storage / no-fp8e4nv design, and the two-implementation model.
//
// Storage (mirrors the established gfx942 bf16 convention: gemm_bf16.hip,
// moe_device.hip, dsa.hip). The cache is ``[num_pages, slots_per_page,
// head_dim]`` bf16, carried as raw uint16 IEEE-754 bit patterns, converted
// to/from fp32 with bf16_to_f32 (zero-extend of the top 16 bits) and f2bf
// (round-to-nearest-even, the SAME helper the CPU oracle uses). bf16's
// 8-exponent-bit range is ample for the Hadamard-transformed softmax-weighted
// mean WITHOUT a per-vector scale, so the original fp8 K region + fp32 scale
// region collapses to flat bf16 (no scale region). The ONLY lossy step on the
// device is the f2bf store; everything else is fp32, so the device is
// bf16-tolerant against the CPU oracle exactly as dsa_sparse_fwd / gemm_bf16
// are.
//
// Mapping. head_dim is fixed at 128 (GLM-5.3 / DeepSeek-V3 index_head_dim).
// One warp (64 lanes) owns exactly one row: prefill -> one pool row r in
// [0, n_pools); decode -> one request row r in [0, batch). Lane l owns head
// elements {2l, 2l+1} (contiguous pairs -> coalesced loads); the online /
// two-pass softmax accumulates a per-lane pair (m, acc, denom), the weighted
// mean is staged to a 128-float shared buffer, and lane 0 runs the IDENTICAL
// staged Hadamard-128 the CPU oracle uses (copied verbatim below -- pure
// register arithmetic on the buffered vector) before the result is written
// back to shared and cooperatively stored as bf16. Running the same Hadamard
// construction on both sides guarantees the only device-vs-oracle residual is
// the bf16 store, which is exactly what test_dsa_kpool_correct.hip asserts.
//
// Two-implementation model (mirrors mla.{hpp,cpp,hip}, dsa.{hpp,cpp,hip}):
//   dsa_kpool.cpp  -- CPU reference (oracle), always compiled, stable fp32.
//   dsa_kpool.hip  -- this file, compiled with VKERNELS_HAS_HIP.
//                     Cross-checked against the oracle by
//                     meta/benchmarks/test_dsa_kpool_correct.hip.
#include <cstdint>
#include <cstdio>

#include "vkernels/kernels/dsa_kpool.hpp"
#include "vkernels/util/error.hpp"  // VK_EXPECTS (host: throws std::invalid_argument)

#if VKERNELS_HAS_HIP
#  include "vkernels/kernels/moe_device.hip"  // shared bf16<->f32 helpers
#  include <hip/hip_runtime.h>

namespace vkernels::kernels::hip {

namespace {

// The IDENTICAL staged 128-point Hadamard the CPU oracle uses
// (dsa_kpool.cpp::hadamard128), copied verbatim so device and oracle share
// one construction (the only residual between them is the bf16 store).
// Operates on a full 128-vector in registers; called by lane 0 on the
// shared-buffered weighted mean.
__device__ __forceinline__ void hadamard128(float* x) {
  struct Stage { int groups; int stride; };
  constexpr Stage kStages[] = {{64, 1}, {32, 2}, {16, 4}, {8, 8},
                               {4, 16}, {2, 32}, {1, 64}};
  for (const Stage st : kStages) {
    const int G = st.groups;
    const int S = st.stride;
    const int two_s = 2 * S;
    float tmp[128];
    for (int g = 0; g < G; ++g) {
      const int base = g * two_s;
      for (int k = 0; k < 2; ++k)
        for (int s = 0; s < S; ++s) {
          const float a = x[base + s];
          const float b = x[base + S + s];
          tmp[base + k * S + s] = (k == 0) ? (a + b) : (a - b);
        }
    }
    for (int i = 0; i < 128; ++i) x[i] = tmp[i];
  }
  constexpr float kScale = 0.08838834764831845f;  // 1/sqrt(128)
  for (int i = 0; i < 128; ++i) x[i] *= kScale;
}

// Round-trip through bf16 (RNE) -> back to fp32. Matches the CPU oracle's
// round_bf16 (dsa_kpool.cpp) and sglang's `.to(tl.bfloat16).to(tl.float32)`.
__device__ __forceinline__ float round_bf16(float v) {
  return bf16_to_f32(f2bf(v));
}

// fp32 -> fp8e4m3fn round-to-nearest-even. Pure IEEE bit manipulation, no
// native FP8 (SM80/A100 safe). Layout: 1 sign | 4 exponent (BIAS 7) |
// 3 mantissa; max finite = 448 (0x7E), NaN at 0x7F/0xFF. Mirrors the CPU
// oracle's f32_to_fp8e4m3fn_rne EXACTLY (same cases, subnormal handling,
// and tie-break rule) so device and oracle agree byte-for-byte.
__device__ __forceinline__ uint8_t f32_to_fp8e4m3fn_rne(float fv) {
  uint32_t bits;
  __builtin_memcpy(&bits, &fv, sizeof(bits));
  const uint32_t sign = (bits >> 31) & 1u;
  const uint32_t mag_bits = bits & 0x7FFFFFFFu;
  float amag;
  __builtin_memcpy(&amag, &mag_bits, sizeof(amag));
  if (amag != amag) return 0x7Fu;                  // NaN -> +NaN
  if (amag == 0.0f) return static_cast<uint8_t>(sign << 7);
  if (amag > 448.0f) return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // saturate
  int e;
  const float frac = frexpf(amag, &e);             // frac in [0.5,1), amag=frac*2^e
  const int e_unbiased = e - 1;                    // 2*frac in [1,2), amag=(2*frac)*2^e_unbiased
  if (e_unbiased < -9) return static_cast<uint8_t>(sign << 7);  // rounds to 0
  auto round_mant = [](float mf) -> uint32_t {     // RNE of mf in [0,8)
    uint32_t fl = static_cast<uint32_t>(floorf(mf));
    const float rem = mf - static_cast<float>(fl);
    if (rem < 0.5f) return fl;
    if (rem > 0.5f) return fl + 1u;
    return (fl & 1u) ? fl + 1u : fl;               // tie -> even
  };
  if (e_unbiased < -6) {                           // subnormal (stored exp=0)
    const uint32_t mant = round_mant(amag * 64.0f);  // amag / 2^-6 = amag*2^6
    if (mant >= 8u) return static_cast<uint8_t>((sign << 7) | 0x08u);  // -> smallest normal
    return static_cast<uint8_t>((sign << 7) | mant);
  }
  // normal: stored_exp = e_unbiased + 7 in [1, 15]; mant = round((2*frac-1)*8)
  uint32_t mant = round_mant((2.0f * frac - 1.0f) * 8.0f);
  uint32_t stored_exp = static_cast<uint32_t>(e_unbiased + 7);
  if (mant >= 8u) { mant = 0u; stored_exp += 1u; }  // carry into exponent
  if (stored_exp == 15u && mant == 7u) return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // NaN slot -> max finite
  if (stored_exp > 15u) return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // overflow -> max finite
  return static_cast<uint8_t>((sign << 7) | (stored_exp << 3) | mant);
}

// NOTE: the fp8 store epilogue needs TWO block-wide reductions -- an OR
// over "any element finite" (skip gate) and a max of |mean| (the single
// per-vector scale). These were first written with __shfl_xor (a 64-lane
// butterfly), but __shfl_xor is a HIP wavefront intrinsic that is undefined
// under nvcc, and a 64-lane butterfly is wrong on A100 (32-lane warps). They
// are now done inline in store_kpool_fp8_device via shared memory +
// __syncthreads -- the SAME block-wide pattern the bf16 kernels already use,
// which works for blockDim=64 whether that is one gfx942 wavefront or two
// sm_80 warps.

// FP8+scale store epilogue for ONE pool row. ``mean`` is the Hadamard-
// transformed fp32 128-vector already in shared memory. Lane l owns elements
// {2l, 2l+1}. Writes raw fp8 bytes to k_dst[0..127] and one fp32 scale to
// *scale_dst. Mirrors the CPU oracle's store_kpool_fp8 (dsa_kpool.cpp).
__device__ __forceinline__ void store_kpool_fp8_device(
    float* mean, int lane, int slots_per_page,
    uint8_t* k_dst, float* scale_dst, bool round_scale,
    bool finite0, bool finite1) {
  // Round mean -> bf16 -> fp32 (matches CPU round_bf16). Read BOTH owned
  // elements into registers FIRST: mean is reused below as the reduction
  // scratch, so every element read must complete before any scratch write
  // (writes touch mean[0..63] and mean[64..127] -- the whole buffer -- and
  // would race with other lanes' element reads without this barrier).
  const float x0 = round_bf16(mean[2 * lane]);
  const float x1 = round_bf16(mean[2 * lane + 1]);
  __syncthreads();  // all element reads done before reusing mean as scratch

  // Block-wide reductions via shared memory (portable across gfx942 64-lane
  // wavefronts and sm_80 32-lane warps: blockDim=64 is one wavefront on
  // gfx942 and two warps on sm_80, and __syncthreads is block-wide). Reuse
  // the 128-float mean buffer now that x0/x1 are in registers:
  //   mean[ 0..63]  -> per-lane absmax (0.0 for non-finite lanes)
  //   mean[64..127] -> per-lane any-finite flag (0.0 / 1.0)
  const bool any_local = (finite0 || finite1);
  mean[lane] = any_local ? fmaxf(fabsf(x0), fabsf(x1)) : 0.0f;
  mean[64 + lane] = any_local ? 1.0f : 0.0f;
  __syncthreads();
  if (lane == 0) {
    float mx = mean[0], fin = mean[64];
    for (int l = 1; l < blockDim.x; ++l) {
      mx = fmaxf(mx, mean[l]);
      fin = fmaxf(fin, mean[64 + l]);
    }
    mean[0] = mx;    // global absmax (0 iff no lane was finite)
    mean[64] = fin;  // global any-finite flag (0.0 / 1.0)
  }
  __syncthreads();

  const bool any_finite = (mean[64] > 0.5f);
  if (!any_finite) {
    if (lane == 0) *scale_dst = 0.0f;
    k_dst[2 * lane] = 0u;
    k_dst[2 * lane + 1] = 0u;
    return;
  }

  float absmax = mean[0];
  if (absmax < 1e-4f) absmax = 1e-4f;  // floored (Triton max(absmax, 1e-4))
  const float fp8_max_inv = 1.0f / 448.0f;
  float scale;
  if (round_scale) {
    scale = exp2f(ceilf(log2f(absmax * fp8_max_inv)));
  } else {
    scale = absmax * fp8_max_inv;
  }
  if (lane == 0) *scale_dst = scale;

  float q0 = x0 / scale;
  if (q0 > 448.0f) q0 = 448.0f;
  else if (q0 < -448.0f) q0 = -448.0f;
  float q1 = x1 / scale;
  if (q1 > 448.0f) q1 = 448.0f;
  else if (q1 < -448.0f) q1 = -448.0f;
  k_dst[2 * lane] = f32_to_fp8e4m3fn_rne(q0);
  k_dst[2 * lane + 1] = f32_to_fp8e4m3fn_rne(q1);
}

// ---------------------------------------------------------------------------
// Store-epilogue policies. The bf16 and fp8 kernels are the SAME kernel --
// identical gather, softmax and Hadamard; they differ only in how the
// completed pool vector is staged pre-Hadamard and stored post-Hadamard.
// One template kernel per path (assemble/decode) parameterized on an
// epilogue guarantees the two cache formats can never drift in the math
// that must be identical, and makes adding a third format a ~20-line
// epilogue instead of a 130-line kernel copy.
//
// Epilogue contract (both methods called by ALL lanes of the block):
//   stage(mean, lane, acc, denom)
//     pre-Hadamard staging of the lane's {2l, 2l+1} weighted-mean elements
//     into the 128-float shared buffer.
//   store(mean, lane, page, sip, slots_per_page, acc, denom)
//     post-Hadamard store of the staged vector at cache slot
//     [page, sip]. May contain __syncthreads() (all lanes reach it: the
//     row gates are block-uniform).
// ---------------------------------------------------------------------------
struct Bf16Epilogue {
  uint16_t* out;  // [num_pages, ssp, 128] bf16 device (zeroed first)

  __device__ void stage(float* mean, int lane, const float (&acc)[2],
                        const float (&denom)[2]) const {
    mean[2 * lane] = acc[0] / denom[0];
    mean[2 * lane + 1] = acc[1] / denom[1];
  }
  __device__ void store(float* mean, int lane, int page, int sip,
                        int slots_per_page, const float (&)[2],
                        const float (&)[2]) const {
    uint16_t* dst = out + ((size_t)page * slots_per_page + sip) * 128;
    dst[2 * lane] = f2bf(mean[2 * lane]);
    dst[2 * lane + 1] = f2bf(mean[2 * lane + 1]);
  }
};

struct Fp8Epilogue {
  uint8_t* cache_u8;               // [num_pages, ssp*(128+4)] (zeroed first)
  const int* round_scale_or_null;  // DEVICE int or null (graph-capturable)

  __device__ void stage(float* mean, int lane, const float (&acc)[2],
                        const float (&denom)[2]) const {
    const bool finite0 = (denom[0] > 0.0f && isfinite(denom[0]) && isfinite(acc[0]));
    const bool finite1 = (denom[1] > 0.0f && isfinite(denom[1]) && isfinite(acc[1]));
    mean[2 * lane] = finite0 ? round_bf16(acc[0] / denom[0]) : 0.0f;
    mean[2 * lane + 1] = finite1 ? round_bf16(acc[1] / denom[1]) : 0.0f;
  }
  __device__ void store(float* mean, int lane, int page, int sip,
                        int slots_per_page, const float (&acc)[2],
                        const float (&denom)[2]) const {
    const int page_bytes = slots_per_page * (128 + 4);
    // sglang packed layout (matches the CPU oracle
    // dsa_kpool_*_fp8_cpu): K region is ``ssp`` contiguous 128-B vectors at
    // ``page*page_bytes + sip*128``, the ``ssp`` fp32 scales occupy the
    // trailing ``ssp*4`` bytes. K is NOT interleaved with its scale (an
    // earlier ``(page*ssp+slot)*132`` stride wrote K at ``slot*132`` and
    // crashed the device-vs-CPU cross-check at high slot density -- it is
    // only valid for ``sip==0``).
    uint8_t* k_dst = cache_u8 + (size_t)page * page_bytes + (size_t)sip * 128;
    float* scale_dst = reinterpret_cast<float*>(
        cache_u8 + (size_t)page * page_bytes) + (slots_per_page * 128 / 4) + sip;
    // round_scale is a DEVICE int (see hip_capi.hpp): read *it > 0 on-device,
    // never on the host, so a prior graph node can produce it on this stream.
    const bool round_scale = round_scale_or_null && *round_scale_or_null > 0;
    const bool finite0 = (denom[0] > 0.0f && isfinite(denom[0]) && isfinite(acc[0]));
    const bool finite1 = (denom[1] > 0.0f && isfinite(denom[1]) && isfinite(acc[1]));
    store_kpool_fp8_device(mean, lane, slots_per_page, k_dst, scale_dst,
                           round_scale, finite0, finite1);
  }
};

// PREFILL: one warp per pool row. Lane l owns head elements {2l, 2l+1}.
// Online softmax over the pool's pool_size slots, then the staged Hadamard
// (lane 0 on the shared-buffered mean), then the epilogue's store at
// out[ loc[r]//ssp, loc[r]%ssp ] (write_mask[r]==0 -> warp returns early).
template <class Epi>
__global__ void dsa_kpool_assemble_kernel(
    Epi epi,                                  // store-epilogue (carries out)
    const uint16_t* __restrict__ chunk_k,    // [num_chunks, 128] bf16
    const uint16_t* __restrict__ chunk_score,// [num_chunks, 128] bf16
    const uint16_t* __restrict__ tail_k,     // [n_reqs, tail_size, 128] bf16
    const uint16_t* __restrict__ tail_score, // [n_reqs, tail_size, 128] bf16
    const float* __restrict__ ape,           // [pool_size, 128] fp32
    const int32_t* __restrict__ req_pool_idx,// [n_pools]
    const int32_t* __restrict__ n_from_tail, // [n_pools]
    const int32_t* __restrict__ chunk_src_start,// [n_pools]
    const int32_t* __restrict__ tail_logical_base,// [n_pools]
    const int32_t* __restrict__ loc,         // [n_pools]
    const int32_t* __restrict__ write_mask,  // [n_pools] or null
    int n_pools, int pool_size, int tail_size, int slots_per_page,
    int num_chunks, int n_reqs) {
  const int r = blockIdx.x;
  const int lane = threadIdx.x;             // 0..63
  if (r >= n_pools) return;
  if (write_mask && write_mask[r] == 0) return;  // early return row

  const int req = req_pool_idx[r];
  const int n_tail = n_from_tail[r];
  const int chunk_src = chunk_src_start[r];
  const int tail_base = tail_logical_base[r];
  const int lr = loc[r];

  // Per-lane online softmax state for elements {2l, 2l+1}.
  float m[2] = {-INFINITY, -INFINITY};
  float acc[2] = {0.0f, 0.0f};
  float denom[2] = {0.0f, 0.0f};
  for (int slot = 0; slot < pool_size; ++slot) {
    const int phys = ((tail_base + slot) % tail_size + tail_size) % tail_size;
    const int ci = chunk_src + (slot - n_tail);
    const uint16_t* spf = (slot < n_tail)
        ? tail_score + ((size_t)req * tail_size + phys) * 128
        : chunk_score + (size_t)ci * 128;
    const uint16_t* kpf = (slot < n_tail)
        ? tail_k + ((size_t)req * tail_size + phys) * 128
        : chunk_k + (size_t)ci * 128;
    for (int t = 0; t < 2; ++t) {
      const int d = 2 * lane + t;
      const float score = bf16_to_f32(spf[d]) + ape[(size_t)slot * 128 + d];
      const float new_m = fmaxf(m[t], score);
      const float rescale = expf(m[t] - new_m);
      const float prob = expf(score - new_m);
      denom[t] = denom[t] * rescale + prob;
      acc[t] = acc[t] * rescale + bf16_to_f32(kpf[d]) * prob;
      m[t] = new_m;
    }
  }

  extern __shared__ float smem[];
  float* mean = smem;  // 128
  epi.stage(mean, lane, acc, denom);
  __syncthreads();
  if (lane == 0) hadamard128(mean);
  __syncthreads();

  const int page = lr / slots_per_page;
  const int sip = lr % slots_per_page;
  epi.store(mean, lane, page, sip, slots_per_page, acc, denom);
}

// DECODE: one warp per request row. Lane l owns head elements {2l, 2l+1}.
// Two-pass softmax (max, then weighted sum under the fixed max) over the
// pool's pool_size slots with the current token's key/score substituted in
// at slot `pos % pool_size`; the compressed K is written (via the staged
// Hadamard + the epilogue) ONLY when pos_valid && slot == pool_size-1.
// UNCONDITIONALLY the current key/score is written to the live tail at
// [req, pos % tail_size] (masked to a no-op when !pos_valid).
template <class Epi>
__global__ void dsa_kpool_decode_kernel(
    Epi epi,                                  // store-epilogue (carries out)
    const uint16_t* __restrict__ key,        // [batch, 128] bf16
    const uint16_t* __restrict__ slot_score, // [batch, 128] bf16
    uint16_t* __restrict__ tail_k,           // [n_reqs, tail_size, 128] bf16 (IN-PLACE)
    uint16_t* __restrict__ tail_score,       // [n_reqs, tail_size, 128] bf16 (IN-PLACE)
    const float* __restrict__ ape,           // [pool_size, 128] fp32
    const int32_t* __restrict__ block_tables,// [batch, btc]
    const int32_t* __restrict__ req_pool_indices,  // [batch]
    const int32_t* __restrict__ positions,   // [batch]
    const int32_t* __restrict__ seq_lens,    // [batch]
    const int32_t* __restrict__ out_cache_loc,    // [batch]
    int batch, int pool_size, int tail_size, int slots_per_page,
    int block_table_cols, int n_reqs, int num_pages) {
  const int r = blockIdx.x;
  const int lane = threadIdx.x;
  if (r >= batch) return;

  const int req_raw = req_pool_indices[r];
  const bool req_valid = req_raw >= 0 && req_raw < n_reqs;
  const int req = req_valid ? req_raw : (req_raw < 0 ? 0 : n_reqs - 1);
  const int pos_raw = positions[r];
  const int safe_pos = pos_raw < 0 ? 0 : pos_raw;
  const int seq_len = seq_lens[r];
  const int cache_loc = out_cache_loc[r];
  const bool pos_valid = req_valid && cache_loc != 0 && pos_raw >= 0 &&
                         pos_raw < seq_len;
  const int slot = safe_pos % pool_size;

  const uint16_t* cur_k = key + (size_t)r * 128;
  const uint16_t* cur_s = slot_score + (size_t)r * 128;

  extern __shared__ float smem[];  // 128-float staging buffer for the Hadamard
  float* mean = smem;

  if (pos_valid && slot == pool_size - 1) {
    const int pool_logical_start = safe_pos - slot;
    // Pass 1: running max (each lane over its {2l, 2l+1} elements).
    float mx[2] = {-INFINITY, -INFINITY};
    for (int p = 0; p < pool_size; ++p) {
      const bool is_current = (p == slot);
      const int phys = ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
      const uint16_t* sb = tail_score + ((size_t)req * tail_size + phys) * 128;
      for (int t = 0; t < 2; ++t) {
        const int d = 2 * lane + t;
        const float s = (is_current ? bf16_to_f32(cur_s[d]) : bf16_to_f32(sb[d]))
                        + ape[(size_t)p * 128 + d];
        mx[t] = fmaxf(mx[t], s);
      }
    }
    // Pass 2: weighted sum under the fixed max.
    float acc[2] = {0.0f, 0.0f}, denom[2] = {0.0f, 0.0f};
    for (int p = 0; p < pool_size; ++p) {
      const bool is_current = (p == slot);
      const int phys = ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
      const uint16_t* sb = tail_score + ((size_t)req * tail_size + phys) * 128;
      const uint16_t* kb = tail_k + ((size_t)req * tail_size + phys) * 128;
      for (int t = 0; t < 2; ++t) {
        const int d = 2 * lane + t;
        const float s = (is_current ? bf16_to_f32(cur_s[d]) : bf16_to_f32(sb[d]))
                        + ape[(size_t)p * 128 + d];
        const float prob = expf(s - mx[t]);
        denom[t] += prob;
        acc[t] += (is_current ? bf16_to_f32(cur_k[d]) : bf16_to_f32(kb[d])) * prob;
      }
    }
    epi.stage(mean, lane, acc, denom);
    __syncthreads();
    if (lane == 0) hadamard128(mean);
    __syncthreads();

    const int pool_id = safe_pos / pool_size;
    const int pool_page_group = pool_id / slots_per_page;
    int tpr = pool_page_group * pool_size;
    if (tpr < 0) tpr = 0;
    if (tpr >= block_table_cols) tpr = block_table_cols - 1;
    const int packed_page = block_tables[(size_t)r * block_table_cols + tpr];
    const int loc_slot = pool_id % slots_per_page;
    epi.store(mean, lane, packed_page, loc_slot, slots_per_page, acc, denom);
  }

  // Unconditional live-tail write (no-op when !pos_valid). The tail is bf16
  // end-to-end in BOTH cache formats.
  if (pos_valid) {
    const int phys_slot = ((safe_pos % tail_size) + tail_size) % tail_size;
    uint16_t* tk = tail_k + ((size_t)req * tail_size + phys_slot) * 128;
    uint16_t* ts = tail_score + ((size_t)req * tail_size + phys_slot) * 128;
    tk[2 * lane] = cur_k[2 * lane];
    tk[2 * lane + 1] = cur_k[2 * lane + 1];
    ts[2 * lane] = cur_s[2 * lane];
    ts[2 * lane + 1] = cur_s[2 * lane + 1];
  }
}

}  // namespace

// Host launchers. Stream convention (dsa_kpool.hpp): trailing ``stream``
// (hipStream_t through void*), nullptr -> default stream. The bf16 and fp8
// launchers differ ONLY in the epilogue they pass to the shared kernels.
namespace {

// Shared launcher-side validation + launch for the PREFILL path.
template <class Epi>
void launch_assemble(int n_pools, int pool_size, int head_dim, int tail_size,
                     int slots_per_page, int num_pages, int num_chunks,
                     int n_reqs, const void* chunk_k, const void* chunk_score,
                     const void* tail_k, const void* tail_score,
                     const void* ape, const void* req_pool_idx,
                     const void* n_from_tail, const void* chunk_src_start,
                     const void* tail_logical_base, const void* loc,
                     const void* write_mask, Epi epi, void* stream) {
  VK_EXPECTS(n_pools >= 0 && pool_size > 0 && head_dim == 128 &&
                 tail_size > 0 && slots_per_page > 0 && num_pages > 0 &&
                 num_chunks >= 0 && n_reqs >= 0,
             "dsa_kpool_assemble: positive dims, head_dim==128, "
             "non-negative counts");
  if (n_pools == 0) return;
  const int shmem = 128 * sizeof(float);
  dim3 grid(n_pools, 1, 1);
  dim3 block(64);
  const hipStream_t s = stream ? static_cast<hipStream_t>(stream) : nullptr;
  dsa_kpool_assemble_kernel<<<grid, block, shmem, s>>>(
      epi,
      reinterpret_cast<const uint16_t*>(chunk_k),
      reinterpret_cast<const uint16_t*>(chunk_score),
      reinterpret_cast<const uint16_t*>(tail_k),
      reinterpret_cast<const uint16_t*>(tail_score),
      reinterpret_cast<const float*>(ape),
      reinterpret_cast<const int32_t*>(req_pool_idx),
      reinterpret_cast<const int32_t*>(n_from_tail),
      reinterpret_cast<const int32_t*>(chunk_src_start),
      reinterpret_cast<const int32_t*>(tail_logical_base),
      reinterpret_cast<const int32_t*>(loc),
      reinterpret_cast<const int32_t*>(write_mask),
      n_pools, pool_size, tail_size, slots_per_page, num_chunks, n_reqs);
}

// Shared launcher-side validation + launch for the DECODE path.
template <class Epi>
void launch_decode(int batch, int pool_size, int head_dim, int tail_size,
                   int slots_per_page, int block_table_cols, int n_reqs,
                   int num_pages, const void* key, const void* slot_score,
                   void* tail_k, void* tail_score, const void* ape,
                   const void* block_tables, const void* req_pool_indices,
                   const void* positions, const void* seq_lens,
                   const void* out_cache_loc, Epi epi, void* stream) {
  VK_EXPECTS(batch >= 0 && pool_size > 0 && head_dim == 128 &&
                 tail_size > 0 && slots_per_page > 0 && block_table_cols > 0 &&
                 n_reqs >= 0 && num_pages >= 0,
             "dsa_kpool_decode_update: positive dims, head_dim==128, "
             "non-negative counts");
  if (batch == 0) return;
  const int shmem = 128 * sizeof(float);
  dim3 grid(batch, 1, 1);
  dim3 block(64);
  const hipStream_t s = stream ? static_cast<hipStream_t>(stream) : nullptr;
  dsa_kpool_decode_kernel<<<grid, block, shmem, s>>>(
      epi,
      reinterpret_cast<const uint16_t*>(key),
      reinterpret_cast<const uint16_t*>(slot_score),
      reinterpret_cast<uint16_t*>(tail_k),
      reinterpret_cast<uint16_t*>(tail_score),
      reinterpret_cast<const float*>(ape),
      reinterpret_cast<const int32_t*>(block_tables),
      reinterpret_cast<const int32_t*>(req_pool_indices),
      reinterpret_cast<const int32_t*>(positions),
      reinterpret_cast<const int32_t*>(seq_lens),
      reinterpret_cast<const int32_t*>(out_cache_loc),
      batch, pool_size, tail_size, slots_per_page, block_table_cols, n_reqs,
      num_pages);
}

}  // namespace

void dsa_kpool_assemble(int n_pools, int pool_size, int head_dim,
                        int tail_size, int slots_per_page, int num_pages,
                        int num_chunks, int n_reqs,
                        const void* chunk_k, const void* chunk_score,
                        const void* tail_k, const void* tail_score,
                        const void* ape,
                        const void* req_pool_idx, const void* n_from_tail,
                        const void* chunk_src_start,
                        const void* tail_logical_base,
                        const void* loc, const void* write_mask, void* out,
                        void* stream) {
  launch_assemble(n_pools, pool_size, head_dim, tail_size, slots_per_page,
                  num_pages, num_chunks, n_reqs, chunk_k, chunk_score, tail_k,
                  tail_score, ape, req_pool_idx, n_from_tail, chunk_src_start,
                  tail_logical_base, loc, write_mask,
                  Bf16Epilogue{reinterpret_cast<uint16_t*>(out)}, stream);
}

void dsa_kpool_decode_update(int batch, int pool_size, int head_dim,
                             int tail_size, int slots_per_page,
                             int block_table_cols, int n_reqs, int num_pages,
                             const void* key, const void* slot_score,
                             void* tail_k, void* tail_score,
                             const void* ape,
                             const void* block_tables,
                             const void* req_pool_indices,
                             const void* positions, const void* seq_lens,
                             const void* out_cache_loc, void* out,
                             void* stream) {
  launch_decode(batch, pool_size, head_dim, tail_size, slots_per_page,
                block_table_cols, n_reqs, num_pages, key, slot_score, tail_k,
                tail_score, ape, block_tables, req_pool_indices, positions,
                seq_lens, out_cache_loc,
                Bf16Epilogue{reinterpret_cast<uint16_t*>(out)}, stream);
}

// Issue #61: fp8+scale store epilogue device launchers.
void dsa_kpool_assemble_fp8(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const void* chunk_k, const void* chunk_score, const void* tail_k,
    const void* tail_score, const void* ape, const void* req_pool_idx,
    const void* n_from_tail, const void* chunk_src_start,
    const void* tail_logical_base, const void* loc, const void* write_mask,
    void* cache_u8, const void* round_scale_or_null, void* stream) {
  VK_EXPECTS(cache_u8, "dsa_kpool_assemble_fp8: non-null cache_u8");
  // round_scale_or_null is a DEVICE int pointer (see hip_capi.hpp): the
  // kernel reads *it > 0 on-device so the value can be produced by a prior
  // graph node on this stream. The host must NOT dereference it -- a raw
  // CUDA device pointer is not host-accessible and only "worked" on ROCm
  // because its device memory happens to be host-visible (HMM).
  launch_assemble(n_pools, pool_size, head_dim, tail_size, slots_per_page,
                  num_pages, num_chunks, n_reqs, chunk_k, chunk_score, tail_k,
                  tail_score, ape, req_pool_idx, n_from_tail, chunk_src_start,
                  tail_logical_base, loc, write_mask,
                  Fp8Epilogue{reinterpret_cast<uint8_t*>(cache_u8),
                              reinterpret_cast<const int*>(round_scale_or_null)},
                  stream);
}

void dsa_kpool_decode_update_fp8(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const void* key, const void* slot_score, void* tail_k, void* tail_score,
    const void* ape, const void* block_tables, const void* req_pool_indices,
    const void* positions, const void* seq_lens, const void* out_cache_loc,
    void* cache_u8, const void* round_scale_or_null, void* stream) {
  VK_EXPECTS(cache_u8, "dsa_kpool_decode_update_fp8: non-null cache_u8");
  // round_scale_or_null is a DEVICE int pointer (see hip_capi.hpp): the
  // kernel reads *it > 0 on-device so the value can be produced by a prior
  // graph node on this stream. The host must NOT dereference it -- a raw
  // CUDA device pointer is not host-accessible and only "worked" on ROCm
  // because its device memory happens to be host-visible (HMM).
  launch_decode(batch, pool_size, head_dim, tail_size, slots_per_page,
                block_table_cols, n_reqs, num_pages, key, slot_score, tail_k,
                tail_score, ape, block_tables, req_pool_indices, positions,
                seq_lens, out_cache_loc,
                Fp8Epilogue{reinterpret_cast<uint8_t*>(cache_u8),
                            reinterpret_cast<const int*>(round_scale_or_null)},
                stream);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
