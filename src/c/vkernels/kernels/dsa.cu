// vkernels/kernels/dsa.cu -- CUDA wmma implementation (NVIDIA), a port of
// the gfx942 MFMA dsa_topk_logits kernels in dsa.hip (issue #51) for an
// NVIDIA deployment track (verified on GB10 / Grace Blackwell, sm_121).
//
//   out[b, t] = k_scale[page[b,i], j]
//             * Sum_h ( max(0, Sum_d Q[b,h,d] * K[page[b,i],j,d] )
//                       * gate[b, h] )
//   where t = i*block + j, t < seq_lens[b] (else left unwritten).
//
// Three device variants, mirroring dsa.hip exactly:
//   * dsa_topk_logits_kernel        -- fp32-Q scalar GEMV (Q staged once)
//   * dsa_topk_logits_kernel_fp8q   -- fp8-Q scalar GEMV (Q raw, dequant OTF)
//   * dsa_topk_logits_kernel_wmma   -- bf16 wmma (Q transposed once, K-tiled)
//
// The two scalar kernels are near-verbatim translations (64-lane warpfront
// -> 32-lane warp; the dequant + bf16 helpers are replicated here so the
// .cu TU is self-contained, exactly as gemm_bf16.cu replicated them). The
// wmma kernel replaces the AMD __builtin_amdgcn_mfma_f32_16x16x16bf16_1k
// (whose fragment layout drove a warp-shuffle gated H-sum) with nvcuda::
// wmma 16x16x16: sK[B][kBK] @ sQt[D][H] = [B,H] per K-tile, staged through
// shared sAcc[B][H] via store_matrix_sync, then a plain per-j gated H-sum
// (the AMD fragment-layout shuffle has no wmma analogue). The binding
// constraint is BLOCK THREADS = (B/16)*(H/16)*32 = B*H/8 <= 1024 (CUDA max
// threads/block; each warp owns ONE [16,16] c_frag, not the kNF
// fragments-per-lane the AMD 64-lane MFMA layout demands).
//
// GB10 vs gfx942 device cap. gfx942 has NO hipFuncSetAttribute opt-in past
// a 64 KB non-optin cap (see the KB note mi300a-dynamic-lds-no-optin), so
// dsa.hip refuses shapes the largest variant won't fit and falls back by
// staging Q as raw fp8. GB10 (sm_121) DOES honour cudaFuncSetAttribute(
// MaxDynamicSharedMemorySize) up to sharedMemPerBlockOptin = 101,376 B
// (verified on ds5), so the wmma variant (74,496 B at H=128) and the fp32-Q
// variant (99,072 B at H=128) both opt in and run there -- no fp8-Q
// fallback is needed on GB10, though the kernel is ported anyway (and
// exercised by the correctness harness) for parity with the HIP path.
//
// Same contract / dispatch as the HIP kernel (see dsa.hpp): cuda::
// dsa_topk_logits selects a variant via dsa_topk_logits_fits_lds{,_fp8q,
// _mfma}; cuda::dsa_topk_logits_with_variant is the explicit-variant hook
// the autotuner/correctness harness use to FORCE a specific kernel.
#include "vkernels/kernels/dsa.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>
#  include <mma.h>
#  include <cstdio>

namespace vkernels::kernels::cuda {

using namespace nvcuda::wmma;

static constexpr int kMmaK = 16;   // one bf16 wmma reduces K = 16
static constexpr int kBK   = 64;   // K-tile (fixed); D is a multiple of 64

// fp8 e4m3fnuz -> fp32. VERBATIM copy of fp8e4m3fnuz_to_f32 in
// moe_device.hip (__host__ __device__); the harness dequants on the host,
// the device kernel dequants on load via its own copy. Both MUST agree --
// the kernel's docstring asserts this, and test_dsa_topk_correct.cu
// cross-checks it.
static __device__ __forceinline__ float fp8e4m3fnuz_to_f32(uint8_t b) {
  const uint32_t s = static_cast<uint32_t>(b >> 7) & 1u;     // sign
  const uint32_t e = static_cast<uint32_t>(b >> 3) & 0xFu;   // exponent (bias 8)
  const uint32_t m = static_cast<uint32_t>(b) & 0x7u;        // mantissa (3)
  if ((b & 0x7Fu) == 0u) return 0.0f;                        // +0 (0x00 AND 0x80)
  float f;
  if (e == 15u && m == 7u) {                                 // 0x7F = NaN -> qNaN
    const uint32_t qnan = 0x7fc00000u;
    __builtin_memcpy(&f, &qnan, sizeof(f));
  } else if (e == 0u) {                                      // subnormal: m*2^-7
    const float v = static_cast<float>(m) * 0x1p-7f;         // m*2^(1-8), exact
    f = s ? -v : v;
  } else {                                                   // normal: 2^(e-8)*(1+m/8)
    const uint32_t bits = (s << 31) | ((e + 119u) << 23) | (m << 20);
    __builtin_memcpy(&f, &bits, sizeof(f));
  }
  return f;
}

// f32 -> bf16 (round-to-nearest-even). VERBATIM copy of f2bf in
// moe_device.hip; the wmma kernel uses it to stage Q once as bf16 sQt
// (fp8 -> fp32 -> bf16, lossless).
static __device__ __forceinline__ uint16_t f2bf(float v) {
  uint32_t b;
  __builtin_memcpy(&b, &v, sizeof(b));
  uint32_t lsb = (b >> 16) & 1u;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}

// ======================================================================
//  fp32-Q scalar GEMV kernel (port of dsa_topk_logits_kernel)
// ======================================================================
// One warp (32 threads) per (batch, split_kv) block; lane = KV token j.
// Stages Q (H*D fp32) and the gate (H fp32) once, then per page one K
// tile (B*D fp8->fp32) and B per-token scales (fp32, packed in the
// trailing B*4 bytes of each KV block). The scalar D-dot + sequential
// H-sum is the simple, correct baseline (mirrors dsa.hip exactly; the
// wmma variant below is the Matrix-Core fast path).
__global__ void dsa_topk_logits_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int H, int D, int B, int max_table_len, int max_seq_len, int split_kv) {
  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int lane = threadIdx.x;                // 0..B-1 -> KV token j

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  float* sQ = smem;                            // H * D
  float* sGate = sQ + (size_t)H * D;           // H
  float* sK = sGate + H;                       // B * D (reloaded per page)
  float* sKscale = sK + (size_t)B * D;         // B

  // --- cooperative load Q (H*D fp8 -> fp32) + gate (H fp32) once per block ---
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = lane; idx < H * D; idx += B) sQ[idx] = fp8e4m3fnuz_to_f32(qp[idx]);
  const float* gp = weight + (size_t)b * H;
  for (int h = lane; h < H; h += B) sGate[h] = gp[h];
  __syncthreads();

  for (int it = 0; it < n_iters; ++it) {
    const int i = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + i];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));
    // keys: B*D fp8 e4m3fnuz (bytes [0 : B*D]); scales: B fp32 (bytes
    // [B*D : B*(D+4)], 4-aligned). Cooperative load into shared.
    for (int idx = lane; idx < B * D; idx += B) sK[idx] = fp8e4m3fnuz_to_f32(kbase[idx]);
    for (int j = lane; j < B; j += B) sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];
    __syncthreads();

    // --- this lane owns KV token j = lane within the page ---
    const float* kj = sK + (size_t)lane * D;
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      const float* qh = sQ + (size_t)h * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += kj[d] * qh[d];
      acc += fmaxf(dot, 0.0f) * sGate[h];
    }
    const int t = i * B + lane;
    if (t < seq_len) out[(size_t)b * max_seq_len + t] = sKscale[lane] * acc;

    __syncthreads();                           // before reloading sK next iter
  }
}

// ======================================================================
//  fp8-Q scalar GEMV kernel (port of dsa_topk_logits_kernel_fp8q)
// ======================================================================
// Q staged as RAW fp8 (dequantised on the fly in the dot loop with the SAME
// helper -- bit-identical output to the fp32-Q kernel above). Staging drops
// to (H + B*D + B) * 4 + H*D bytes (the fp32-Q request minus the 3*H*D bytes
// raw-fp8 Q saves), so on gfx942 this is the fallback for H>=64; on GB10 the
// opt-in cap is high enough that the fp32-Q kernel runs instead, but this
// kernel is ported (and exercised by the correctness harness) for parity.
__global__ void dsa_topk_logits_kernel_fp8q(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int H, int D, int B, int max_table_len, int max_seq_len, int split_kv) {
  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int lane = threadIdx.x;                // 0..B-1 -> KV token j

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  // fp8-Q staging: the gate, one K tile and its per-token scales as fp32
  // (naturally aligned -- they come first), then Q as RAW fp8 bytes
  // (dequantised on the fly in the dot loop below).
  extern __shared__ float smem[];
  float* sGate = smem;                         // H
  float* sK = sGate + H;                       // B * D (reloaded per page)
  float* sKscale = sK + (size_t)B * D;         // B
  uint8_t* sQ_raw = reinterpret_cast<uint8_t*>(sKscale + B);  // H * D (fp8)

  // --- cooperative load Q (raw fp8) + gate (fp32) once per block ---
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = lane; idx < H * D; idx += B) sQ_raw[idx] = qp[idx];
  const float* gp = weight + (size_t)b * H;
  for (int h = lane; h < H; h += B) sGate[h] = gp[h];
  __syncthreads();

  for (int it = 0; it < n_iters; ++it) {
    const int i = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + i];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));
    // keys: B*D fp8 e4m3fnuz (bytes [0 : B*D]); scales: B fp32 (bytes
    // [B*D : B*(D+4)], 4-aligned). Cooperative load into shared.
    for (int idx = lane; idx < B * D; idx += B) sK[idx] = fp8e4m3fnuz_to_f32(kbase[idx]);
    for (int j = lane; j < B; j += B) sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];
    __syncthreads();

    // --- this lane owns KV token j = lane within the page ---
    const float* kj = sK + (size_t)lane * D;
    float acc = 0.0f;
    for (int h = 0; h < H; ++h) {
      const uint8_t* qh_raw = sQ_raw + (size_t)h * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) dot += kj[d] * fp8e4m3fnuz_to_f32(qh_raw[d]);
      acc += fmaxf(dot, 0.0f) * sGate[h];
    }
    const int t = i * B + lane;
    if (t < seq_len) out[(size_t)b * max_seq_len + t] = sKscale[lane] * acc;

    __syncthreads();                           // before reloading sK next iter
  }
}

// ======================================================================
//  bf16 wmma kernel (port of dsa_topk_logits_kernel_mfma)
// ======================================================================
// Replaces the AMD __builtin_amdgcn_mfma_f32_16x16x16bf16_1k (64-lane
// wavefront, fragment-layout warp-shuffle gated H-sum) with nvcuda::wmma
// 16x16x16 (32-lane warp). The [B,H] = sK[B][kBK] @ sQt[D][H] dot matrix
// per K-tile is computed with wmma (A=sK row_major, B=sQt row_major --
// exactly the gemm_bf16.cu pattern), accumulated across the D/kBK K-tiles
// in register fragments, then store_matrix_sync stages the full [B,H]
// result in shared sAcc[B][H]. The per-token gated reduce (ReLU + gate
// mul + H-sum + k_scale) is then a plain fp32 loop over the H heads, one
// row per lane in warp_c==0 (parallel, no redundant work) -- the AMD
// fragment-layout warp-shuffle reduction has no wmma analogue, so the
// shared staging is the bridge.
//
// Storage (mirrors dsa.hip, with sAcc added for the wmma epilogue):
//   sQt[D][H]   bf16  -- Q transposed ONCE per block (reused across every
//                        page in the split). fp8->bf16 is LOSSLESS.
//   sK[B][kBK]  bf16  -- one K-tile, reloaded per K-tile per page
//   sGate[H]    fp32  -- the per-head gate, loaded once per block
//   sKscale[B]  fp32  -- per-token scales, loaded once per page
//   sAcc[B][H]  fp32  -- the wmma output, staged per K-tile (see below)
// At the GLM-5.3 widths (kBK=64):
//   H=32:  (2048+2048)*2 + (32+64)*4 + 32*32*4 = 12,288 B
//   H=64:  (4096+4096)*2 + (64+64)*4 + 64*64*4 = 25,088 B
//   H=128: (8192+8192)*2 + (128+64)*4 + 128*128*4 = 74,496 B
// The H=128 case (74,496 B) exceeds GB10's 49,152 B non-optin cap, so the
// launcher opts in to the 101,376 B ceiling via cudaFuncSetAttribute (set
// ONCE per context, not per launch, to avoid the cudaEventElapsedTime==0
// glitch -- see gemm_bf16.cu for the same fix).
//
// ----------------------------------------------------------------------
// Thread + shared-memory caps for the wmma kernel (GB10 / sm_121, verified
// on ds5): each warp owns ONE [16,16] output tile (a single c_frag, not
// the kNF fragments-per-lane the AMD 64-lane MFMA layout demands), so the
// binding constraint is BLOCK THREADS = (B/16)*(H/16)*32 = B*H/8 <= 1024
// (CUDA max threads/block), NOT registers. At H=128,B=64 that is 1024
// threads (the max); larger H needs smaller B. Shared is
//   (D*H + B*kBK)*2 + (H+B)*4 + B*H*4
// (the last term is the sAcc[B][H] staging the store_matrix_sync epilogue
// needs -- the AMD fragment-layout warp-shuffle gated reduce has no wmma
// analogue). At H=128,D=128,B=64 that is 74,496 B < GB10's 101,376 B opt-in
// cap, so the launcher opts in via cudaFuncSetAttribute (set ONCE per
// context, not per launch, to avoid the cudaEventElapsedTime==0 glitch --
// see gemm_bf16.cu for the same fix). The accumulators are ZEROED PER PAGE
// -- each page is an independent [H,B]=Q@K^T dot matrix; carrying acc
// across pages would sum prior pages' K into the current output.
template <int H, int D, int B>
__global__ void dsa_topk_logits_kernel_wmma(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ out,
    int max_table_len, int max_seq_len, int split_kv) {
  static_assert(H % 16 == 0, "H must be a multiple of 16 (wmma fragment)");
  static_assert(D % kBK == 0, "D must be a multiple of kBK (K-tile)");
  static_assert(B % 16 == 0, "B must be a multiple of 16 (wmma fragment)");
  constexpr int kFragsM = B / 16;          // row fragments (KV tokens)
  constexpr int kFragsN = H / 16;          // col fragments (heads)
  constexpr int kWarps  = kFragsM * kFragsN;
  constexpr int kTh     = kWarps * 32;     // = B*H/8 <= 1024

  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int tid  = threadIdx.x;             // 0..kTh-1
  const int warp = tid >> 5;                // 0..kWarps-1
  const int warp_r = warp / kFragsN;        // which 16-row (KV) fragment
  const int warp_c = warp % kFragsN;        // which 16-col (head) fragment
  const int lane = tid & 31;

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;  // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  uint16_t* sQt = reinterpret_cast<uint16_t*>(smem);            // D * H
  uint16_t* sK = sQt + (size_t)D * H;                           // B * kBK
  float*    sGate = reinterpret_cast<float*>(sK + (size_t)B * kBK);   // H
  float*    sKscale = sGate + H;                               // B
  float*    sAcc = sKscale + B;                                // B * H

  // --- cooperative load Q transposed as bf16 sQt[D][H] once per block ---
  // Q is laid out [H][D] (q[h*D + d] = Q[h,d]); sQt is the TRANSPOSE
  // [D][H] so the wmma B fragment (sQt[K-row=d][N-col=h]) reads Q[h,d]
  // in one bf16 load. fp8 -> fp32 -> bf16 is lossless.
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  for (int idx = tid; idx < H * D; idx += kTh) {
    const int h = idx / D;
    const int d = idx % D;
    sQt[(size_t)d * H + h] = f2bf(fp8e4m3fnuz_to_f32(qp[(size_t)h * D + d]));
  }
  const float* gp = weight + (size_t)b * H;
  for (int h = tid; h < H; h += kTh) sGate[h] = gp[h];
  __syncthreads();

  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;

  for (int it = 0; it < n_iters; ++it) {
    const int pg = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + pg];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));

    // Per-token scales: load ONCE per page (the trailing B*4 bytes; they do
    // not change across K-tiles).
    for (int j = tid; j < B; j += kTh)
      sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];

    fill_fragment(c_frag, 0.0f);   // per-page: independent dot matrix

    // K-tiles: D/kBK of them, each kBK wide. A K-tile is consumed by
    // kBK/kMmaK = 4 wmma mma_syncs.
    for (int kt = 0; kt < D / kBK; ++kt) {
      // cooperative load sK[B][kBK] for THIS K-tile (bf16). col is the LOCAL
      // K-column; gcol = kt*kBK + col is the global D-column. (D is a
      // multiple of kBK so no D-bound check.)
      for (int idx = tid; idx < B * kBK; idx += kTh) {
        const int row = idx / kBK;          // 0..B-1 (KV token j)
        const int col = idx % kBK;          // 0..kBK-1 (local K-col)
        const int gcol = kt * kBK + col;
        sK[(size_t)row * kBK + col] =
            f2bf(fp8e4m3fnuz_to_f32(kbase[(size_t)row * D + gcol]));
      }
      __syncthreads();

      // --- kBK/kMmaK = 4 wmma mma_syncs consume the K-tile; c_frag is
      //     preserved across K-tiles -> the full D-dot per (j, h). ---
      #pragma unroll
      for (int mf = 0; mf < kBK / kMmaK; ++mf) {
        const int kk = mf * kMmaK;
        load_matrix_sync(a_frag, (const __nv_bfloat16*)&sK[(size_t)(warp_r * 16) * kBK + kk],
                         kBK);
        load_matrix_sync(b_frag, (const __nv_bfloat16*)&sQt[(size_t)(kt * kBK + kk) * H + warp_c * 16],
                         H);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      __syncthreads();                       // before reloading sK next tile
    }

    // Stage this warp's [16,16] fp32 result into sAcc[B][H] (row = warp_r*16
    // + r, col = warp_c*16 + c). The gated reduce below reads sAcc by (j,h):
    // every warp_c stores its 16-col slice, so sAcc[j][0..H) is the COMPLETE
    // dot row after the sync. (The AMD fragment-layout warp-shuffle gated
    // reduce has no wmma analogue -- this shared staging is the bridge.)
    store_matrix_sync(&sAcc[(size_t)warp_r * 16 * H + warp_c * 16], c_frag, H,
                      mem_row_major);
    __syncthreads();

    // --- per-token gated reduce (fp32) + k_scale, then write ---
    // Only warp_c==0 writes (it reads sAcc[j][0..H) -- the complete row,
    // since every warp_c already stored its 16-col slice above). The 16
    // lanes of the warp each own one of the warp's 16 rows, so the reduce
    // is parallel across rows with no redundant work.
    if (warp_c == 0) {
      const int j = warp_r * 16 + (lane & 15);   // 0..B-1 (each lane a row)
      if (lane < 16 && j < B) {
        float partial = 0.0f;
        for (int h = 0; h < H; ++h) {
          float d = sAcc[(size_t)j * H + h];
          if (d < 0.0f) d = 0.0f;              // ReLU (max(0, raw))
          partial += sGate[h] * d;
        }
        const int t = pg * B + j;
        if (t < seq_len)
          out[(size_t)b * max_seq_len + t] = sKscale[j] * partial;
      }
    }
    __syncthreads();                         // before next page's sK load
  }
}

// ---------------------------------------------------------------------------
//  Launchers
// ---------------------------------------------------------------------------
namespace {

// GB10 (sm_121) shared-memory caps, verified on ds5 (dgx-spark-05):
//   sharedMemPerBlock      = 49,152 B  (non-optin)
//   sharedMemPerBlockOptin = 101,376 B (opt-in; honoured, unlike gfx942)
// The fp32-Q kernel's largest admitted shape (H=128, D=128, B=64) stages
// (128*128 + 128 + 64*128 + 64) * 4 = 99,072 B -- under the opt-in cap, so
// GB10 runs the fp32-Q kernel at every width gfx942's dsa_topk_logits_fits_lds
// admits AND larger. The wmma kernel (74,496 B at H=128) also opts in.
static constexpr int kGb10LdsOptin = 101376;

// GB10 admission for the fp32-Q kernel (mirrors dsa_topk_logits_fits_lds but
// against the GB10 opt-in cap). Pure arithmetic on a documented device cap.
static bool fits_lds_fp32q(int H, int D, int B) {
  const int bytes = (H * D + H + B * D + B) * 4;
  return bytes > 0 && bytes <= kGb10LdsOptin;
}

// GB10 admission for the fp8-Q kernel (mirrors dsa_topk_logits_fits_lds_fp8q
// but against the GB10 opt-in cap). Pure arithmetic.
static bool fits_lds_fp8q(int H, int D, int B) {
  const int bytes = (H + B * D + B) * 4 + H * D;
  return bytes > 0 && bytes <= kGb10LdsOptin;
}

// GB10 admission for the wmma kernel (mirrors dsa_topk_logits_fits_lds_mfma's
// shape constraints -- H%16, D%64, B%16 -- plus the kTh=(B/16)*(H/16)*32
// <= 1024 CUDA max-threads/block limit and the opt-in LDS cap that includes
// the sAcc[B][H] staging). Pure arithmetic on documented device caps.
static bool fits_lds_wmma(int H, int D, int B) {
  if (H <= 0 || D <= 0 || B <= 0) return false;
  if (H % 16 != 0 || D % kBK != 0 || B % 16 != 0) return false;
  const int kTh = (B / 16) * (H / 16) * 32;   // threads per block (= B*H/8)
  if (kTh > 1024) return false;                // CUDA max threads/block
  const int bytes = (D * H + B * kBK) * 2 + (H + B) * 4 + B * H * 4;
  return bytes <= kGb10LdsOptin;
}

template <int H, int D, int B>
void launch_wmma(const uint8_t* q_fp8, const uint8_t* kvcache_u8,
                 const float* weight, const int32_t* seq_lens,
                 const int32_t* page_table, float* out,
                 int batch_size, int max_table_len, int max_seq_len,
                 int split_kv) {
  constexpr int kTh = (B / 16) * (H / 16) * 32;   // = B*H/8 <= 1024
  const int shmem =
      (D * H + B * kBK) * 2 + (H + B) * (int)sizeof(float) +
      B * H * (int)sizeof(float);
  // Opt in to the larger-than-default shared region ONCE per kernel
  // instantiation. cudaFuncAttributeMaxDynamicSharedMemorySize is a
  // per-context, per-function attribute that persists, so setting it on
  // every launch (as in a timing loop) perturbs the event clock and can
  // make cudaEventElapsedTime return 0. Needed when shmem exceeds the
  // 49,152 B non-optin cap (e.g. H=128 -> 74,496 B); harmless otherwise.
  static const bool inited = [&] {
    cudaFuncSetAttribute((const void*)dsa_topk_logits_kernel_wmma<H, D, B>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    return true;
  }();
  (void)inited;
  dim3 block(kTh);
  dim3 grid(batch_size, split_kv, 1);   // x=batch, y=split (mirrors HIP)
  dsa_topk_logits_kernel_wmma<H, D, B><<<grid, block, shmem, 0>>>(
      q_fp8, kvcache_u8, weight, seq_lens, page_table, out,
      max_table_len, max_seq_len, split_kv);
}

}  // namespace

// Dispatch shim shared by dsa_topk_logits (auto) and
// dsa_topk_logits_with_variant (explicit). `variant`: 0 = auto
// (wmma -> fp32q -> fp8q -> refuse); 1 = fp32-Q; 2 = fp8-Q; 3 = wmma.
// An explicit variant that does NOT fit GB10's opt-in LDS cap is refused
// with a stderr diagnostic + no-op (the caller zeroed `out`, which is what
// stays) rather than launch a block the driver silently drops -- mirrors
// the HIP path's contract.
namespace {
void dsa_topk_logits_dispatch(int batch_size, int num_heads, int head_dim,
                              int block, int max_table_len, int max_seq_len,
                              int split_kv, const void* q_fp8,
                              const void* kvcache_u8, const void* weight,
                              const void* seq_lens, const void* page_table,
                              void* out, int variant) {
  if (batch_size <= 0 || max_table_len <= 0) return;  // nothing to write
  if (num_heads <= 0 || head_dim <= 0 || block <= 0) return;  // bad dims

  const auto* q  = reinterpret_cast<const uint8_t*>(q_fp8);
  const auto* kv = reinterpret_cast<const uint8_t*>(kvcache_u8);
  const auto* w  = reinterpret_cast<const float*>(weight);
  const auto* sl = reinterpret_cast<const int32_t*>(seq_lens);
  const auto* pt = reinterpret_cast<const int32_t*>(page_table);
  auto* o        = reinterpret_cast<float*>(out);

  const int H = num_heads, D = head_dim, B = block;

  const bool want_wmma  = (variant == 3) || (variant == 0);
  const bool want_fp32q = (variant == 1) || (variant == 0);
  const bool want_fp8q  = (variant == 2) || (variant == 0);

  // Grid for the scalar kernels (x=batch, y=split). The wmma kernel builds
  // its own grid inside launch_wmma.
  dim3 grid(batch_size, split_kv, 1);

  // Fast path: the bf16 wmma kernel (Matrix-Core). Shapes need H%16==0,
  // D%64==0, B%16==0 and kTh=(B/16)*(H/16)*32 <= 1024 (CUDA max
  // threads/block). The wmma kernel is a template on (H,D,B); only the
  // GLM-5.3 family (H in {16,32,48,64,80,96,112,128}, D=128, B in
  // {16,32,48,64}) is instantiated below. Other fitting shapes fall
  // through to the scalar kernels.
  if (want_wmma && fits_lds_wmma(H, D, B)) {
    if (H == 32 && D == 128 && B == 64)  { launch_wmma<32,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 64 && D == 128 && B == 64)  { launch_wmma<64,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 128 && D == 128 && B == 64) { launch_wmma<128,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 32 && D == 128 && B == 16)  { launch_wmma<32,128,16>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 32 && D == 128 && B == 32)  { launch_wmma<32,128,32>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 16 && D == 128 && B == 64)  { launch_wmma<16,128,64>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 64 && D == 128 && B == 32)  { launch_wmma<64,128,32>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    if (H == 64 && D == 128 && B == 16)  { launch_wmma<64,128,16>(q,kv,w,sl,pt,o,batch_size,max_table_len,max_seq_len,split_kv); return; }
    // Shape fits the wmma gate but is not instantiated: fall through to the
    // scalar kernels (they admit every shape, so this is never a refuse).
  }

  // fp32-Q kernel (Q dequanted once into shared) -- GB10's default for any
  // shape the wmma kernel doesn't cover, up to the opt-in cap. Opt in to
  // the larger-than-default shared region ONCE per kernel (the attribute
  // persists; setting it per launch perturbs the event clock). Set to the
  // ceiling so every fits_lds_fp32q shape (<= kGb10LdsOptin) launches.
  if (want_fp32q && fits_lds_fp32q(H, D, B)) {
    static const bool inited_fp32q = [] {
      cudaFuncSetAttribute((const void*)dsa_topk_logits_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kGb10LdsOptin);
      return true;
    }();
    (void)inited_fp32q;
    const int shmem = (H * D + H + B * D + B) * (int)sizeof(float);
    dim3 blockdim(B);
    dsa_topk_logits_kernel<<<grid, blockdim, shmem, 0>>>(
        q, kv, w, sl, pt, o, H, D, B, max_table_len, max_seq_len, split_kv);
    return;
  }

  // fp8-Q kernel (Q staged raw, dequantised on the fly in the dot loop with
  // the SAME helper -- bit-identical output to the fp32-Q kernel above) for
  // shapes that fit the smaller fp8-Q footprint but not the fp32-Q one.
  // Same once-per-kernel opt-in as the fp32-Q path (the fp8-Q request also
  // exceeds the 49,152 B non-optin cap at the larger GLM-5.3 widths).
  if (want_fp8q && fits_lds_fp8q(H, D, B)) {
    static const bool inited_fp8q = [] {
      cudaFuncSetAttribute((const void*)dsa_topk_logits_kernel_fp8q,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kGb10LdsOptin);
      return true;
    }();
    (void)inited_fp8q;
    const int shmem = (H + B * D + B) * (int)sizeof(float) + H * D;
    dim3 blockdim(B);
    dsa_topk_logits_kernel_fp8q<<<grid, blockdim, shmem, 0>>>(
        q, kv, w, sl, pt, o, H, D, B, max_table_len, max_seq_len, split_kv);
    return;
  }

  // No admitted variant. REFUSE the shape rather than launch a block the
  // driver silently drops; the caller zeroed `out`, which is what stays.
  std::fprintf(stderr,
      "vk_cuda_dsa_topk_logits: indexer (H=%d D=%d B=%d) exceeds GB10's "
      "opt-in dynamic-LDS cap (%d B) under ALL of wmma (%d B), fp32-Q (%d B) "
      "and fp8-Q (%d B). Refusing (output left as the caller provided).\n",
      num_heads, head_dim, block, kGb10LdsOptin,
      (D * H + B * kBK) * 2 + (H + B) * 4 + B * H * 4,
      (H * D + H + B * D + B) * 4,
      (H + B * D + B) * 4 + H * D);
}
}  // namespace

void dsa_topk_logits_with_variant(int batch_size, int num_heads, int head_dim,
                                  int block, int max_table_len,
                                  int max_seq_len, int split_kv,
                                  const void* q_fp8, const void* kvcache_u8,
                                  const void* weight, const void* seq_lens,
                                  const void* page_table, void* out,
                                  int variant) {
  if (variant != 0 && variant != 1 && variant != 2 && variant != 3) {
    std::fprintf(stderr,
        "vk_cuda_dsa_topk_logits_with_variant: bad variant=%d "
        "(want 0=auto, 1=fp32q, 2=fp8q, 3=wmma). Refusing.\n", variant);
    return;
  }
  dsa_topk_logits_dispatch(batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8,
                           kvcache_u8, weight, seq_lens, page_table, out,
                           variant);
}

void dsa_topk_logits(int batch_size, int num_heads, int head_dim, int block,
                     int max_table_len, int max_seq_len, int split_kv,
                     const void* q_fp8, const void* kvcache_u8,
                     const void* weight, const void* seq_lens,
                     const void* page_table, void* out) {
  // Auto: wmma (Matrix-Core, if instantiated) -> fp32-Q -> fp8-Q -> refuse.
  dsa_topk_logits_dispatch(batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8,
                           kvcache_u8, weight, seq_lens, page_table, out, 0);
}

}  // namespace vkernels::kernels::cuda

#endif  // VKERNELS_HAS_CUDA
