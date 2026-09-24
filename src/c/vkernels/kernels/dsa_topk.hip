// vkernels/kernels/dsa_topk.hip
//
// Radix top-k transform for the DSA kpool indexer (gfx942 / MI300A). The
// radix selector is a HIP port of sglang's kpool_topk_transform JIT kernel:
// an 8-bit fp16 coarse pass followed by fp32-key refinement. One workgroup
// owns one score row, selects pool groups in shared memory, expands them to
// token indices, then applies the optional page-table or ragged mapping.
//
// The radix machinery itself (coarse_float_key / ordered_float_key /
// radix_topk / transform_token / the transform row body) lives in the shared
// header dsa_topk_device.cuh -- one definition for the standalone kernel
// below, the fused cooperative tail, and the CUDA TU dsa.cu. This file also
// hosts the FUSED chain (VK_DSA_TOPK_FUSED=1): one cooperative grid that
// runs the scalar dsa_topk_logits GEMV (phase 1, the shared kernels from
// dsa_topk_device.cuh) and the transform row (phase 2) after a grid sync --
// the launch-count fusion of NOTES-fusion-dsa-kda-launches' candidate (i)
// for the top-k pair. The scalar logits kernels are vendor-intrinsic-free
// (no MFMA), so this TU shim-compiles with nvcc and the fused path is
// GB10-validatable; the MFMA fast paths stay in dsa.hip (documented shim
// negative) and are NOT part of the fused path.
#include "vkernels/kernels/dsa_topk.hpp"
#include "vkernels/kernels/dsa.hpp"  // dsa_topk_logits_fits_lds{,_fp8q}, kDsaTopkLdsCapGfx942
// The radix machinery + scalar logits kernels (shared with dsa.cu and the
// fused tail below).
#include "vkernels/kernels/dsa_topk_device.cuh"

#if VKERNELS_HAS_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
// Grid sync for the fused cooperative kernel. nvcc (the CUDA shim build)
// takes CUDA's <cooperative_groups.h> with the cg:: namespace; real HIP
// (ROCm 6.x, verified on beverin) ships the hipCG compat header
// <hip/hip_cooperative_groups.h> whose grid_group/this_grid() live in the
// GLOBAL namespace -- hence the VK_GRID_SYNC indirection below.
#if defined(__CUDACC__)
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define VK_GRID_SYNC() cg::this_grid().sync()
#else
#include <hip/hip_cooperative_groups.h>
#define VK_GRID_SYNC() ::this_grid().sync()
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace vkernels::kernels::hip {
namespace {

// ======================================================================
//  Fused phase 1: the scalar top-k logits GEMV, adapted to the
//  cooperative grid geometry.
// ======================================================================
// The shared kernels dsa_topk_logits_kernel (fp32-Q) and
// dsa_topk_logits_kernel_fp8q (raw-fp8-Q) in dsa_topk_device.cuh launch
// B-thread blocks (lane = KV token). The fused cooperative kernel must run
// kThreadsPerBlock = 1024-thread blocks -- phase 2's radix row selection is
// written for that block size -- so the phase-1 body runs here with only
// the first B lanes ACTIVE. For every active lane the statements, and
// therefore every fp32 result, are the SAME sequence as the standalone
// kernels (same element->lane assignment, same dequant, same D-dot and
// gated H-sum order); the __syncthreads() barriers are executed block-wide
// exactly as there (the `active` guards never wrap a barrier). The early
// `n_iters <= 0` return is uniform per block and returns from this DEVICE
// function, so every thread still reaches the grid sync below.
// Output is BIT-IDENTICAL to the standalone kernels -- verified by memcmp
// in tests/kernels/attn/test_dsa_topk_fused_gpu.cpp.
template<bool kFp8Q>
__device__ void dsa_topk_logits_phase1(const uint8_t* __restrict__ q_fp8,
                                       const uint8_t* __restrict__ kvcache_u8,
                                       const float* __restrict__ weight,
                                       const int32_t* __restrict__ seq_lens,
                                       const int32_t* __restrict__ page_table,
                                       float* __restrict__ out,
                                       int H, int D, int B, int max_table_len,
                                       int max_seq_len, int split_kv) {
  const int b = blockIdx.x;
  const int pid_split = blockIdx.y;
  const int tid = static_cast<int>(threadIdx.x);  // 0..1023; tid < B computes
  const bool active = tid < B;
  const int lane = tid;                           // lane = KV token j

  const int seq_len = seq_lens[b];
  const int np_total = (seq_len + B - 1) / B;     // pages with content
  const int stride = (np_total + split_kv - 1) / split_kv;
  const int i_start = pid_split * stride;
  const int rem = np_total - i_start;
  const int n_iters = (rem <= 0) ? 0 : (stride < rem ? stride : rem);
  if (n_iters <= 0) return;

  extern __shared__ float smem[];
  float* sGate;
  float* sK;
  float* sKscale;
  float* sQ = nullptr;
  uint8_t* sQ_raw = nullptr;
  if constexpr (kFp8Q) {
    // fp8-Q staging -- the dsa_topk_logits_kernel_fp8q layout: gate, one K
    // tile and its per-token scales as fp32 (naturally aligned -- they come
    // first), then Q as RAW fp8 bytes.
    sGate = smem;                                        // H
    sK = sGate + H;                                      // B * D
    sKscale = sK + (size_t)B * D;                        // B
    sQ_raw = reinterpret_cast<uint8_t*>(sKscale + B);    // H * D (fp8)
  } else {
    // fp32-Q staging -- the dsa_topk_logits_kernel layout.
    sQ = smem;                                           // H * D
    sGate = sQ + (size_t)H * D;                          // H
    sK = sGate + H;                                      // B * D
    sKscale = sK + (size_t)B * D;                        // B
  }

  // --- cooperative load Q + gate once per block (lanes < B only; the
  //     element->lane assignment matches the standalone kernels) ---
  const uint8_t* qp = q_fp8 + (size_t)b * H * D;
  if constexpr (kFp8Q) {
    if (active)
      for (int idx = lane; idx < H * D; idx += B) sQ_raw[idx] = qp[idx];
  } else {
    if (active)
      for (int idx = lane; idx < H * D; idx += B) sQ[idx] = fp8e4m3fnuz_to_f32(qp[idx]);
  }
  const float* gp = weight + (size_t)b * H;
  if (active)
    for (int h = lane; h < H; h += B) sGate[h] = gp[h];
  __syncthreads();

  for (int it = 0; it < n_iters; ++it) {
    const int i = i_start + it;
    const int32_t page = page_table[(size_t)b * max_table_len + i];
    const uint8_t* kbase = kvcache_u8 + (size_t)page * (B * (D + 4));
    // keys: B*D fp8 e4m3fnuz; scales: B fp32 (the trailing B*4 bytes).
    if (active) {
      for (int idx = lane; idx < B * D; idx += B) sK[idx] = fp8e4m3fnuz_to_f32(kbase[idx]);
      for (int j = lane; j < B; j += B) sKscale[j] = reinterpret_cast<const float*>(kbase + B * D)[j];
    }
    __syncthreads();

    // --- this lane owns KV token j = lane within the page ---
    if (active) {
      const float* kj = sK + (size_t)lane * D;
      float acc = 0.0f;
      for (int h = 0; h < H; ++h) {
        if constexpr (kFp8Q) {
          const uint8_t* qh_raw = sQ_raw + (size_t)h * D;
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += kj[d] * fp8e4m3fnuz_to_f32(qh_raw[d]);
          acc += fmaxf(dot, 0.0f) * sGate[h];
        } else {
          const float* qh = sQ + (size_t)h * D;
          float dot = 0.0f;
          for (int d = 0; d < D; ++d) dot += kj[d] * qh[d];
          acc += fmaxf(dot, 0.0f) * sGate[h];
        }
      }
      const int t = i * B + lane;
      if (t < seq_len) out[(size_t)b * max_seq_len + t] = sKscale[lane] * acc;
    }
    __syncthreads();                           // before reloading sK next iter
  }
}

// ======================================================================
//  Standalone kpool transform kernel: one row per block, the VERBATIM
//  row body from the shared header dsa_topk_device.cuh.
// ======================================================================
template<int K>
__global__ __launch_bounds__(kThreadsPerBlock) void dsa_topk_transform_kernel(
    const float* __restrict__ score,
    const int32_t* __restrict__ lengths,
    int32_t* __restrict__ dst_token_indices,
    int64_t score_stride,
    int32_t pool_size,
    int32_t out_cols,
    const int32_t* __restrict__ page_table,
    int64_t page_table_stride,
    const int32_t* __restrict__ page_table_row_index,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ seq_lens) {
  dsa_topk_transform_row<K>(score, lengths, dst_token_indices, score_stride, pool_size,
                            out_cols, page_table, page_table_stride, page_table_row_index,
                            topk_indices_offset, row_starts, seq_lens);
}

template<int K>
void launch(const float* score,
            const int32_t* lengths,
            int32_t* dst_token_indices,
            int32_t batch_size,
            int64_t score_stride,
            int32_t pool_size,
            int32_t out_cols,
            const int32_t* page_table,
            int64_t page_table_stride,
            const int32_t* page_table_row_index,
            const int32_t* topk_indices_offset,
            const int32_t* row_starts,
            const int32_t* seq_lens) {
  dsa_topk_transform_kernel<K>
      <<<batch_size, kThreadsPerBlock, kDynamicSmemBytes, 0>>>(score,
                                                               lengths,
                                                               dst_token_indices,
                                                               score_stride,
                                                               pool_size,
                                                               out_cols,
                                                               page_table,
                                                               page_table_stride,
                                                               page_table_row_index,
                                                               topk_indices_offset,
                                                               row_starts,
                                                               seq_lens);
}

// ======================================================================
//  Fused cooperative tail (VK_DSA_TOPK_FUSED=1)
// ======================================================================
// The device cap for the fits_lds arithmetic. gfx942 has NO dynamic-LDS
// opt-in (the 64 KB non-optin cap IS the ceiling -- hipFuncSetAttribute
// past it is a silent no-op, see the KB note mi300a-dynamic-lds-no-optin),
// so real HIP builds run the arithmetic against kDsaTopkLdsCapGfx942. The
// CUDA shim build queries the device's opt-in ceiling instead (101,376 B
// on GB10), mirroring dsa.cu::device_lds_optin_cap.
int dsa_topk_lds_cap() {
#if defined(VKERNELS_DSA_TOPK_CUDA_SHIM)
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return kDsaTopkLdsCapGfx942;
  int cap = 0;
  if (hipDeviceGetAttribute(&cap, hipDeviceAttributeMaxSharedMemoryPerBlockOptin, dev) !=
          hipSuccess ||
      cap <= 0) {
    return kDsaTopkLdsCapGfx942;
  }
  return cap;
#else
  return kDsaTopkLdsCapGfx942;
#endif
}

// The proven chain's logits arm: the scalar GEMV launches the AUTO
// dispatcher's scalar arms make (fp32-Q, else fp8-Q). `q_variant`:
// 0 = auto, 1 = fp32-Q, 2 = fp8-Q. Returns false when no scalar variant
// fits the device cap (the refuse contract: stderr diagnostic + no-op --
// the caller's zeroed/prior `logits` is what stays).
bool launch_scalar_logits(int q_variant, int batch_size, int num_heads, int head_dim,
                          int block, int max_table_len, int max_seq_len, int split_kv,
                          const void* q_fp8, const void* kvcache_u8, const void* weight,
                          const void* seq_lens, const void* page_table, void* logits) {
  if (batch_size <= 0 || max_table_len <= 0 || num_heads <= 0 || head_dim <= 0 || block <= 0)
    return false;
  if (batch_size > 0 && split_kv <= 0) return false;
  const int H = num_heads, D = head_dim, B = block;
  const int lds_cap = dsa_topk_lds_cap();
  const dim3 grid(batch_size, split_kv, 1);
  if ((q_variant == 1 || q_variant == 0) &&
      dsa_topk_logits_fits_lds(H, D, B, lds_cap)) {
    const int shmem = (H * D + H + B * D + B) * (int)sizeof(float);
    if (shmem > 48 * 1024)  // CUDA: over the 48 KB default -> opt in per launch
      hipFuncSetAttribute((const void*)dsa_topk_logits_kernel,
                          hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
    dsa_topk_logits_kernel<<<grid, dim3(B), shmem, 0>>>(
        reinterpret_cast<const uint8_t*>(q_fp8),
        reinterpret_cast<const uint8_t*>(kvcache_u8),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<const int32_t*>(page_table),
        reinterpret_cast<float*>(logits),
        H, D, B, max_table_len, max_seq_len, split_kv);
    return true;
  }
  if ((q_variant == 2 || q_variant == 0) &&
      dsa_topk_logits_fits_lds_fp8q(H, D, B, lds_cap)) {
    const int shmem = (H + B * D + B) * (int)sizeof(float) + H * D;
    if (shmem > 48 * 1024)
      hipFuncSetAttribute((const void*)dsa_topk_logits_kernel_fp8q,
                          hipFuncAttributeMaxDynamicSharedMemorySize, shmem);
    dsa_topk_logits_kernel_fp8q<<<grid, dim3(B), shmem, 0>>>(
        reinterpret_cast<const uint8_t*>(q_fp8),
        reinterpret_cast<const uint8_t*>(kvcache_u8),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<const int32_t*>(seq_lens),
        reinterpret_cast<const int32_t*>(page_table),
        reinterpret_cast<float*>(logits),
        H, D, B, max_table_len, max_seq_len, split_kv);
    return true;
  }
  std::fprintf(stderr,
               "vk_hip_dsa_topk_logits_transform_fused: indexer (H=%d D=%d B=%d) "
               "fits no scalar logits variant under the %d B device cap. Refusing.\n",
               H, D, B, lds_cap);
  return false;
}

// The fused cooperative kernel: ONE hipLaunchCooperativeKernel grid of
// (batch, split_kv) 1024-thread blocks. Phase 1 computes the logits row
// slice (block = (batch, split_kv), lane = KV token -- the AUTO
// dispatcher's scalar arms); cg::this_grid().sync() publishes every
// complete row; phase 2 (the blockIdx.y == 0 blocks) runs the VERBATIM
// transform row body on the complete logits row. REQUIRES all blocks
// co-resident -- checked in the launcher below.
template<int K, bool kFp8Q>
__global__ __launch_bounds__(kThreadsPerBlock) void dsa_topk_logits_transform_coop_kernel(
    const uint8_t* __restrict__ q_fp8,
    const uint8_t* __restrict__ kvcache_u8,
    const float* __restrict__ weight,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ page_table,
    float* __restrict__ logits_out,
    int H, int D, int B, int max_table_len, int max_seq_len, int split_kv,
    const int32_t* __restrict__ t_lengths,
    int32_t* __restrict__ dst_token_indices,
    int64_t score_stride,
    int32_t pool_size,
    int32_t out_cols,
    const int32_t* __restrict__ t_page_table,
    int64_t t_page_table_stride,
    const int32_t* __restrict__ page_table_row_index,
    const int32_t* __restrict__ topk_indices_offset,
    const int32_t* __restrict__ row_starts,
    const int32_t* __restrict__ t_seq_lens) {
  dsa_topk_logits_phase1<kFp8Q>(q_fp8, kvcache_u8, weight, seq_lens, page_table,
                                logits_out, H, D, B, max_table_len, max_seq_len,
                                split_kv);
  VK_GRID_SYNC();
  if (blockIdx.y == 0) {
    dsa_topk_transform_row<K>(logits_out, t_lengths, dst_token_indices, score_stride,
                              pool_size, out_cols, t_page_table, t_page_table_stride,
                              page_table_row_index, topk_indices_offset, row_starts,
                              t_seq_lens);
  }
}

// Launch attempt for one (K, Q-mode) instantiation. Returns false when the
// fused launch is not possible (no cooperative support, occupancy 0, grid
// larger than the co-residency capacity even after the split_kv clamp) or
// the launch itself errors -- the caller then runs the proven two-launch
// chain. On success the fused grid has run.
template<int K, bool kFp8Q>
bool launch_logits_transform_coop(int batch_size, int num_heads, int head_dim,
                                  int block, int max_table_len, int max_seq_len,
                                  int split_kv, const void* q_fp8,
                                  const void* kvcache_u8, const void* weight,
                                  const void* seq_lens, const void* page_table,
                                  void* logits, const int32_t* t_lengths,
                                  int32_t* dst_token_indices, int64_t score_stride,
                                  int32_t pool_size, int32_t out_cols,
                                  const int32_t* t_page_table,
                                  int64_t t_page_table_stride,
                                  const int32_t* page_table_row_index,
                                  const int32_t* topk_indices_offset,
                                  const int32_t* row_starts,
                                  const int32_t* t_seq_lens, int shmem) {
  // CUDA: dynamic smem over the 48 KB default needs a per-function opt-in
  // (per launch -- per-device state, mirroring dsa.cu). gfx942: the request
  // is <= 64 KB by construction (fits_lds admitted it) and the opt-in past
  // 64 KB is a silent no-op, so the call is harmless there.
  if (shmem > 48 * 1024)
    hipFuncSetAttribute(reinterpret_cast<const void*>(
                            &dsa_topk_logits_transform_coop_kernel<K, kFp8Q>),
                        hipFuncAttributeMaxDynamicSharedMemorySize, shmem);

  int blocks_per_sm = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm,
          reinterpret_cast<const void*>(&dsa_topk_logits_transform_coop_kernel<K, kFp8Q>),
          kThreadsPerBlock, (size_t)shmem) != hipSuccess ||
      blocks_per_sm <= 0)
    return false;
  int dev = 0;
  int sm_count = 0;
  int coop = 0;
  if (hipGetDevice(&dev) != hipSuccess ||
      hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, dev) !=
          hipSuccess)
    return false;
  if (hipDeviceGetAttribute(&coop, hipDeviceAttributeCooperativeLaunch, dev) != hipSuccess ||
      coop == 0)
    return false;
  const long long capacity = (long long)blocks_per_sm * sm_count;
  if (capacity < batch_size) return false;  // even one block per row won't fit
  // split_kv is perf-only -- the grouped logit is grouping-independent
  // (verified: test_dsa_topk_correct.{hip,cu} case split=2), so the fused
  // launcher CLAMPS it to the co-residency capacity instead of failing.
  // On MI300A the serving formula (dsa_topk_logits_split_for, NUM_CU=228)
  // already keeps batch*split_kv <= 228 = the CU count, i.e. the whole
  // serving matrix is co-resident at one 1024-thread block per CU; on GB10
  // (48 SMs, 1536 threads/SM -> 1 block/SM) the clamp engages.
  int split = split_kv < 1 ? 1 : split_kv;
  if ((long long)batch_size * split > capacity) split = (int)(capacity / batch_size);

  const uint8_t* q = reinterpret_cast<const uint8_t*>(q_fp8);
  const uint8_t* kv = reinterpret_cast<const uint8_t*>(kvcache_u8);
  const float* w = reinterpret_cast<const float*>(weight);
  const int32_t* sl = reinterpret_cast<const int32_t*>(seq_lens);
  const int32_t* pt = reinterpret_cast<const int32_t*>(page_table);
  float* lg = reinterpret_cast<float*>(logits);
  const int32_t* tpt = reinterpret_cast<const int32_t*>(t_page_table);
  int H = num_heads, D = head_dim, B = block, mtl = max_table_len, msl = max_seq_len;
  int sk = split;
  long long ss = static_cast<long long>(score_stride);
  int32_t ps = pool_size, oc = out_cols;
  long long tpts = static_cast<long long>(t_page_table_stride);
  void* args[] = {&q, &kv, &w, &sl, &pt, &lg, &H, &D, &B, &mtl, &msl, &sk,
                  &t_lengths, &dst_token_indices, &ss, &ps, &oc, &tpt, &tpts,
                  &page_table_row_index, &topk_indices_offset, &row_starts,
                  &t_seq_lens};
  const dim3 grid(batch_size, split, 1);
  const dim3 blockdim(kThreadsPerBlock, 1, 1);
  hipGetLastError();  // clear any sticky error so the post-launch check is ours
  hipLaunchCooperativeKernel(
      reinterpret_cast<const void*>(&dsa_topk_logits_transform_coop_kernel<K, kFp8Q>),
      grid, blockdim, args, (size_t)shmem, 0);
  return hipGetLastError() == hipSuccess;
}

} // namespace

void dsa_topk_transform(int32_t batch_size,
                        const float* score,
                        const int32_t* lengths,
                        int32_t* dst_token_indices,
                        int64_t score_stride,
                        int32_t pool_size,
                        int32_t token_topk,
                        int32_t out_cols,
                        const int32_t* page_table,
                        int64_t page_table_stride,
                        const int32_t* page_table_row_index,
                        const int32_t* topk_indices_offset,
                        const int32_t* row_starts,
                        const int32_t* seq_lens) {
  if (batch_size <= 0 || score == nullptr || lengths == nullptr || dst_token_indices == nullptr ||
      score_stride < 0 || pool_size <= 1 || token_topk <= 0 || token_topk % pool_size != 0 ||
      out_cols != token_topk + (seq_lens == nullptr ? 0 : pool_size - 1) ||
      (page_table != nullptr && topk_indices_offset != nullptr) ||
      (page_table != nullptr && page_table_stride <= 0))
    return;

  const int32_t group_topk = token_topk / pool_size;
#define VK_DSA_TOPK_LAUNCH(K)                                                                      \
  launch<K>(score,                                                                                 \
            lengths,                                                                               \
            dst_token_indices,                                                                     \
            batch_size,                                                                            \
            score_stride,                                                                          \
            pool_size,                                                                             \
            out_cols,                                                                              \
            page_table,                                                                            \
            page_table_stride,                                                                     \
            page_table_row_index,                                                                  \
            topk_indices_offset,                                                                   \
            row_starts,                                                                            \
            seq_lens)
  switch (group_topk) {
  case 128:
    VK_DSA_TOPK_LAUNCH(128);
    break;
  case 160:
    VK_DSA_TOPK_LAUNCH(160);
    break;
  case 192:
    VK_DSA_TOPK_LAUNCH(192);
    break;
  case 224:
    VK_DSA_TOPK_LAUNCH(224);
    break;
  case 256:
    VK_DSA_TOPK_LAUNCH(256);
    break;
  case 512:
    VK_DSA_TOPK_LAUNCH(512);
    break;
  default:
    break;
  }
#undef VK_DSA_TOPK_LAUNCH
}

// ======================================================================
//  The fused logits->transform entry (VK_DSA_TOPK_FUSED=1)
// ======================================================================
// One cooperative grid: scalar dsa_topk_logits (phase 1) + grid sync +
// dsa_topk_transform on the complete logits rows (phase 2) -- the top-k
// chain runs 1 launch instead of 2. With the gate OFF (the default) this
// entry IS the proven two-launch chain (the scalar logits launch the AUTO
// dispatcher makes + the standalone transform kernel on the default
// stream), so callers can wire it unconditionally and the env var decides;
// per-call getenv, the VK_MHC_PRE_STRICT pattern (no static caching).
//
// Fallback contract: any condition the fused path cannot honour (gate off,
// invalid transform parameters, an unsupported group_topk, no scalar logits
// variant under the device cap, no cooperative-launch support, occupancy 0,
// grid beyond the co-residency capacity) silently runs the proven chain
// instead. `logits` doubles as the transform's `score` buffer, so callers
// must pass score_stride == max_seq_len (exactly what the two-launch chain
// passes when it feeds the logits buffer into dsa_topk_transform) and zero
// or canary-fill the buffer up front -- the kernels write only
// t < seq_len[b] and the transform reads only [row_start, row_start +
// lengths[row]) per row, the same caller contract the two-launch chain has.
//
// `q_variant`: 0 = auto (fp32-Q, else fp8-Q -- the AUTO dispatcher's scalar
// arms; the MFMA fast paths are dsa.hip's and are NOT part of this fused
// path), 1 = fp32-Q, 2 = fp8-Q. On gfx942 H=32 picks fp32-Q and H=64
// picks fp8-Q (64 KB non-optin cap); on GB10 both fit fp32-Q (101,376 B
// opt-in ceiling). Both scalar variants produce BIT-IDENTICAL logits.
void dsa_topk_logits_transform_fused(int batch_size, int num_heads, int head_dim,
                                     int block, int max_table_len, int max_seq_len,
                                     int split_kv, const void* q_fp8,
                                     const void* kvcache_u8, const void* weight,
                                     const void* seq_lens, const void* page_table,
                                     void* logits, int q_variant,
                                     const int32_t* t_lengths,
                                     int32_t* dst_token_indices, int64_t score_stride,
                                     int32_t pool_size, int32_t token_topk,
                                     int32_t out_cols, const int32_t* t_page_table,
                                     int64_t t_page_table_stride,
                                     const int32_t* page_table_row_index,
                                     const int32_t* topk_indices_offset,
                                     const int32_t* row_starts,
                                     const int32_t* t_seq_lens) {
  const char* gate = std::getenv("VK_DSA_TOPK_FUSED");
  const bool want_fused = gate != nullptr && std::strcmp(gate, "1") == 0;
  if (q_variant != 0 && q_variant != 1 && q_variant != 2) {
    std::fprintf(stderr,
                 "vk_hip_dsa_topk_logits_transform_fused: bad q_variant=%d "
                 "(want 0=auto, 1=fp32q, 2=fp8q). Refusing.\n",
                 q_variant);
    return;
  }

  // Transform validity -- the SAME predicate dsa_topk_transform's silent
  // no-op tests, evaluated up front so an invalid tail falls back to the
  // proven chain (whose transform no-ops identically).
  const bool transform_valid =
      batch_size > 0 && t_lengths != nullptr && dst_token_indices != nullptr &&
      score_stride >= 0 && pool_size > 1 && token_topk > 0 &&
      token_topk % pool_size == 0 &&
      out_cols == token_topk + (t_seq_lens == nullptr ? 0 : pool_size - 1) &&
      (t_page_table == nullptr || topk_indices_offset == nullptr) &&
      (t_page_table == nullptr || t_page_table_stride > 0);
  const int32_t group_topk = transform_valid ? token_topk / pool_size : 0;

  bool fused = want_fused && transform_valid &&
               dsa_topk_transform_group_topk_supported(group_topk);
  if (fused) {
    // Scalar variant pick (mirrors the AUTO dispatcher's scalar arms) and
    // the fused dynamic-shared size: max(logits staging, the transform
    // row's kDynamicSmemBytes) -- the two phases ALIAS the same dynamic
    // window (extern __shared__ all start at offset 0), separated by the
    // grid sync.
    const int lds_cap = dsa_topk_lds_cap();
    int variant = 0;  // 1 = fp32-Q, 2 = fp8-Q
    if ((q_variant == 1 || q_variant == 0) &&
        dsa_topk_logits_fits_lds(num_heads, head_dim, block, lds_cap))
      variant = 1;
    else if ((q_variant == 2 || q_variant == 0) &&
             dsa_topk_logits_fits_lds_fp8q(num_heads, head_dim, block, lds_cap))
      variant = 2;
    if (variant != 0) {
      const int shmem_logits =
          variant == 1
              ? (num_heads * head_dim + num_heads + block * head_dim + block) *
                    (int)sizeof(float)
              : (num_heads + block * head_dim + block) * (int)sizeof(float) +
                    num_heads * head_dim;
      const int shmem = shmem_logits > (int)kDynamicSmemBytes ? shmem_logits
                                                              : (int)kDynamicSmemBytes;
#define VK_DSA_TOPK_COOP(K)                                                                        \
  fused = variant == 1                                                                             \
              ? launch_logits_transform_coop<K, false>(                                            \
                    batch_size, num_heads, head_dim, block, max_table_len, max_seq_len,            \
                    split_kv, q_fp8, kvcache_u8, weight, seq_lens, page_table, logits,             \
                    t_lengths, dst_token_indices, score_stride, pool_size, out_cols,               \
                    t_page_table, t_page_table_stride, page_table_row_index,                       \
                    topk_indices_offset, row_starts, t_seq_lens, shmem)                            \
              : launch_logits_transform_coop<K, true>(                                             \
                    batch_size, num_heads, head_dim, block, max_table_len, max_seq_len,            \
                    split_kv, q_fp8, kvcache_u8, weight, seq_lens, page_table, logits,             \
                    t_lengths, dst_token_indices, score_stride, pool_size, out_cols,               \
                    t_page_table, t_page_table_stride, page_table_row_index,                       \
                    topk_indices_offset, row_starts, t_seq_lens, shmem)
      switch (group_topk) {
      case 128:
        VK_DSA_TOPK_COOP(128);
        break;
      case 160:
        VK_DSA_TOPK_COOP(160);
        break;
      case 192:
        VK_DSA_TOPK_COOP(192);
        break;
      case 224:
        VK_DSA_TOPK_COOP(224);
        break;
      case 256:
        VK_DSA_TOPK_COOP(256);
        break;
      case 512:
        VK_DSA_TOPK_COOP(512);
        break;
      default:
        fused = false;
        break;
      }
#undef VK_DSA_TOPK_COOP
    } else {
      fused = false;
    }
  }
  if (fused) return;

  // Proven chain (also the gate-off default): scalar logits launch, then
  // the standalone transform kernel. On logits refusal nothing runs (the
  // caller-provided outputs stay, the refuse contract above).
  if (launch_scalar_logits(q_variant, batch_size, num_heads, head_dim, block,
                           max_table_len, max_seq_len, split_kv, q_fp8, kvcache_u8,
                           weight, seq_lens, page_table, logits)) {
    dsa_topk_transform(batch_size, reinterpret_cast<const float*>(logits), t_lengths,
                       dst_token_indices, score_stride, pool_size, token_topk, out_cols,
                       t_page_table, t_page_table_stride, page_table_row_index,
                       topk_indices_offset, row_starts, t_seq_lens);
  }
}

} // namespace vkernels::kernels::hip
#endif // VKERNELS_HAS_HIP
