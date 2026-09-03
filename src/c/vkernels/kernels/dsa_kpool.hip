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

// PREFILL: one warp per pool row. Lane l owns head elements {2l, 2l+1}.
// Online softmax over the pool's pool_size slots, then the staged Hadamard
// (lane 0 on the shared-buffered mean), then the bf16 store at
// out[ loc[r]//ssp, loc[r]%ssp ] (write_mask[r]==0 -> warp returns early).
__global__ void dsa_kpool_assemble_kernel(
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
    uint16_t* __restrict__ out,              // [num_pages, ssp, 128] bf16
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
  mean[2 * lane] = acc[0] / denom[0];
  mean[2 * lane + 1] = acc[1] / denom[1];
  __syncthreads();
  if (lane == 0) hadamard128(mean);
  __syncthreads();

  const int page = lr / slots_per_page;
  const int sip = lr % slots_per_page;
  uint16_t* dst = out + ((size_t)page * slots_per_page + sip) * 128;
  dst[2 * lane] = f2bf(mean[2 * lane]);
  dst[2 * lane + 1] = f2bf(mean[2 * lane + 1]);
}

// DECODE: one warp per request row. Lane l owns head elements {2l, 2l+1}.
// Two-pass softmax (max, then weighted sum under the fixed max) over the
// pool's pool_size slots with the current token's key/score substituted in
// at slot `pos % pool_size`; the compressed K is written (via the staged
// Hadamard) ONLY when pos_valid && slot == pool_size-1. UNCONDITIONALLY the
// current key/score is written to the live tail at [req, pos % tail_size]
// (masked to a no-op when !pos_valid).
__global__ void dsa_kpool_decode_kernel(
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
    uint16_t* __restrict__ out,              // [num_pages, ssp, 128] bf16
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
    mean[2 * lane] = acc[0] / denom[0];
    mean[2 * lane + 1] = acc[1] / denom[1];
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
    uint16_t* dst = out + ((size_t)packed_page * slots_per_page + loc_slot) * 128;
    dst[2 * lane] = f2bf(mean[2 * lane]);
    dst[2 * lane + 1] = f2bf(mean[2 * lane + 1]);
  }

  // Unconditional live-tail write (no-op when !pos_valid).
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

void dsa_kpool_assemble(int n_pools, int pool_size, int head_dim,
                        int tail_size, int slots_per_page, int num_pages,
                        int num_chunks, int n_reqs,
                        const void* chunk_k, const void* chunk_score,
                        const void* tail_k, const void* tail_score,
                        const void* ape,
                        const void* req_pool_idx, const void* n_from_tail,
                        const void* chunk_src_start,
                        const void* tail_logical_base,
                        const void* loc, const void* write_mask, void* out) {
  VK_EXPECTS(n_pools >= 0 && pool_size > 0 && head_dim == 128 &&
                 tail_size > 0 && slots_per_page > 0 && num_pages > 0 &&
                 num_chunks >= 0 && n_reqs >= 0,
             "dsa_kpool_assemble: positive dims, head_dim==128, "
             "non-negative counts");
  if (n_pools == 0) return;
  const int shmem = 128 * sizeof(float);
  dim3 grid(n_pools, 1, 1);
  dim3 block(64);
  dsa_kpool_assemble_kernel<<<grid, block, shmem, 0>>>(
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
      reinterpret_cast<uint16_t*>(out), n_pools, pool_size, tail_size,
      slots_per_page, num_chunks, n_reqs);
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
                             const void* out_cache_loc, void* out) {
  VK_EXPECTS(batch >= 0 && pool_size > 0 && head_dim == 128 &&
                 tail_size > 0 && slots_per_page > 0 && block_table_cols > 0 &&
                 n_reqs >= 0 && num_pages >= 0,
             "dsa_kpool_decode_update: positive dims, head_dim==128, "
             "non-negative counts");
  if (batch == 0) return;
  const int shmem = 128 * sizeof(float);
  dim3 grid(batch, 1, 1);
  dim3 block(64);
  dsa_kpool_decode_kernel<<<grid, block, shmem, 0>>>(
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
      reinterpret_cast<uint16_t*>(out), batch, pool_size, tail_size,
      slots_per_page, block_table_cols, n_reqs, num_pages);
}

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
