// vkernels/kernels/dsa_topk_device.cuh
//
// The two scalar dsa_topk_logits device kernels, shared VERBATIM by the HIP
// TU (dsa.hip, gfx942) and the CUDA TU (dsa.cu, GB10 wmma port). They are
// plain scalar CUDA/HIP-C++ -- no vendor intrinsics -- so ONE definition
// compiles under hipcc and nvcc, exactly like device_numeric.cuh (which this
// header pulls in for the fp8 dequant). Both backends' launchers stage and
// index identically (lane = KV token, block = (batch, split_kv)), so the
// kernels are already byte-identical ports; keeping one copy makes the
// cross-backend bit-identical-output claim structural instead of a
// maintained coincidence.
//
// Only the two scalar GEMV variants live here. The Matrix-Core fast paths
// genuinely differ (AMD MFMA inline asm vs nvcuda::wmma fragments) and stay
// in their backend TUs. Include from a device-compiled TU, after the vendor
// runtime header (<hip/hip_runtime.h> or <cuda_runtime.h>); the kernels are
// `static`, so each including TU gets an internal-linkage copy and no
// device-link symbol can collide.
#pragma once

#include <cstdint>

// The kpool transform row below (new) is the first code in this header that
// needs the fp16 bit-cast intrinsics (__half_as_ushort / __float2half_rn);
// include the vendor fp16 header directly so every including TU gets them
// (the scalar logits kernels above deliberately avoid vendor intrinsics).
#if defined(VKERNELS_HAS_HIP)
#include <hip/hip_fp16.h>
// The kernels below use gridDim/blockIdx/threadIdx -- on real HIP these are
// only declared by <hip/hip_runtime.h> (hipcc does NOT pre-include it for
// header TUs), and on the CUDA shim build <hip/hip_runtime.h> resolves to
// cuda_compat/hip/hip_runtime.h via the -I path.
#include <hip/hip_runtime.h>
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include "vkernels/kernels/device_numeric.cuh"

namespace vkernels::kernels {

// ======================================================================
//  fp32-Q scalar GEMV kernel
// ======================================================================
// One warp (B lanes) per (batch, split_kv) block stages Q (H*D fp32) and
// the per-head gate (H fp32) once, then -- per page -- one K tile (B*D
// fp8->fp32) and the B per-token scales (fp32, packed in the trailing B*4
// bytes of each KV block). Total shared = (H*D + H + B*D + B) * 4 = 49,536 B
// at the GLM-5.3 config (H=32, D=128, B=64) -- UNDER the 64 KB non-optin LDS
// cap on gfx942, so NO hipFuncSetAttribute opt-in is needed there (gfx942's
// opt-in ceiling EQUALS the non-optin cap, 65,536 B, verified on a CSCS
// beverin node -- see the KB note mi300a-dynamic-lds-no-optin; the MHC 256 KB
// bug was a *static* over-allocation, not a dynamic-LDS opt-in). On GB10 the
// launcher opts in to the 101,376 B ceiling, so this kernel also serves the
// larger shapes (e.g. H=128 -> 99,072 B) that gfx942 must refuse.
//
// Larger H (e.g. 64 -> 66,048 B) exceeds gfx942's fp32-Q cap, so the launcher
// dispatches dsa_topk_logits_kernel_fp8q below. Each lane owns ONE KV token
// (j = lane), reuses the staged K across all H heads, and writes
// out[b, i*B+lane] only when t = i*B+lane < seq_len[b] (the caller ZEROES
// the output first). The grid's split_kv slices [0, ceildiv(seq_len, B))
// across the split blocks and is perf-only (the grouped logit is
// grouping-independent, mirroring the forward's block_I/inner_iter).
//
// This is the simple, correct baseline (one lane per key, sequential H-sum,
// K staged once per page and reused across all H heads) -- the topk's
// counterpart to dsa_sparse_fwd's "one query row per wavefront, one key at
// a time". A tiled-key / MFMA-fp8 prefetch is a follow-on optimisation, not
// required for correctness.
static __global__ void dsa_topk_logits_kernel(
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
//  fp8-Q scalar GEMV kernel (the raw-fp8-Q fallback)
// ======================================================================
// Q staged as RAW fp8 e4m3fnuz (1 byte/element), dequantised on the fly in
// the dot loop with the SAME fp8e4m3fnuz_to_f32 helper the fp32-Q kernel
// uses at load -- the dequanted Q values, and therefore the fp32 dot
// accumulator, the gated H-sum and the written output, are BIT-IDENTICAL to
// dsa_topk_logits_kernel, so the harness cross-check (max_rel < 1e-3 in
// test_dsa_topk_correct.{hip,cu}) carries over unchanged.
//
// On gfx942 this is the fallback for shapes whose fp32-Q staging
// (H*D + H + B*D + B) * 4 exceeds the 64 KB non-optin dynamic-LDS cap
// (e.g. H=64 -> 66,048 B): gfx942 has NO hipFuncSetAttribute opt-in past
// 64 KB (hipFuncSetAttribute(MaxDynamicSharedMemorySize, N>65536) returns
// hipSuccess but the launch silently never runs; verified on a CSCS beverin
// node -- see the KB note mi300a-dynamic-lds-no-optin), so instead of a
// driver opt-in this variant stages Q raw. Staging drops to
// (H + B*D + B) * 4 + H*D bytes = 41,472 B at H=64 (37,248 B at H=32, though
// the fp32-Q kernel remains the fast path there -- Q dequanted once, not per
// dot). On GB10 the opt-in cap is high enough that the fp32-Q kernel runs
// instead; this kernel is ported anyway (and exercised by the harnesses) for
// parity with the HIP dispatch. Each lane owns ONE KV token (j = lane) as in
// the fp32-Q kernel; only the Q storage layout differs.
static __global__ void dsa_topk_logits_kernel_fp8q(
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
//  DSA kpool transform: the radix top-k ROW (shared device code)
// ======================================================================
// The transform row body -- coarse fp16 binning, the 8-bit radix rounds and
// the group->token expansion -- moved here VERBATIM from dsa_topk.hip (no
// numeric change: the value-desc / index-asc tie-break selection is
// contract-critical, see NOTES-155) so BOTH launchers share one definition:
//
//   * dsa_topk.hip::dsa_topk_transform_kernel -- the standalone one-row-per-
//     block launch (gfx942 HIP and the HIP-on-NVIDIA shim), and
//   * dsa_topk.hip::dsa_topk_logits_transform_coop_kernel -- the fused
//     cooperative tail (VK_DSA_TOPK_FUSED=1), which runs the SAME row body
//     as phase 2 after a grid sync.
//
// `static` linkage as above: each including TU gets an internal copy.

constexpr int kThreadsPerBlock = 1024;
constexpr int kRadix = 256;
constexpr size_t kDynamicSmemBytes = 8 * 1024 * sizeof(uint32_t);
constexpr int kSmemInputSize = static_cast<int>(kDynamicSmemBytes / (2 * sizeof(int32_t)));

static __device__ __forceinline__ uint8_t coarse_float_key(float value) {
  const uint16_t bits = __half_as_ushort(__float2half_rn(value));
  const uint16_t key =
      (bits & 0x8000u) != 0 ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<uint8_t>(key >> 8);
}

static __device__ __forceinline__ uint32_t ordered_float_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

template<int K>
static __device__ void radix_topk(const float* __restrict__ input,
                                  int32_t* __restrict__ indices,
                                  int32_t row_start,
                                  int32_t length) {
  int32_t topk = K;

  __shared__ __align__(128) int32_t histogram_buf[2][kRadix + 128];
  __shared__ __align__(128) int32_t counter;
  __shared__ __align__(128) int32_t threshold_bin_id;
  __shared__ __align__(128) int32_t num_input[2];
  int32_t* histogram = histogram_buf[0];
  extern __shared__ int32_t staged_indices[];

  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  if (tid < kRadix + 1) histogram[tid] = 0;
  __syncthreads();

  for (int32_t idx = tid; idx < length; idx += kThreadsPerBlock)
    atomicAdd(&histogram[coarse_float_key(input[row_start + idx])], 1);
  __syncthreads();

  const auto run_reverse_cumsum = [&]() {
#pragma unroll 8
    for (int32_t i = 0; i < 8; ++i) {
      if (tid < kRadix) {
        const int32_t delta = 1 << i;
        const int32_t buffer = i & 1;
        int32_t value = histogram_buf[buffer][tid];
        if (tid < kRadix - delta) value += histogram_buf[buffer][tid + delta];
        histogram_buf[buffer ^ 1][tid] = value;
      }
      __syncthreads();
    }
  };

  run_reverse_cumsum();
  if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
    threshold_bin_id = tid;
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  int32_t threshold_bin = threshold_bin_id;
  topk -= histogram[threshold_bin + 1];
  if (topk == 0) {
    for (int32_t idx = tid; idx < length; idx += kThreadsPerBlock) {
      if (static_cast<int32_t>(coarse_float_key(input[row_start + idx])) > threshold_bin) {
        const int32_t pos = atomicAdd(&counter, 1);
        indices[pos] = idx;
      }
    }
    __syncthreads();
    return;
  }

  if (tid < kRadix + 1) histogram[tid] = 0;
  __syncthreads();
  for (int32_t idx = tid; idx < length; idx += kThreadsPerBlock) {
    const float value = input[row_start + idx];
    const int32_t bin = static_cast<int32_t>(coarse_float_key(value));
    if (bin > threshold_bin) {
      const int32_t pos = atomicAdd(&counter, 1);
      indices[pos] = idx;
    } else if (bin == threshold_bin) {
      const int32_t pos = atomicAdd(&num_input[0], 1);
      if (pos < kSmemInputSize) {
        staged_indices[pos] = idx;
        atomicAdd(&histogram[(ordered_float_key(value) >> 24) & 0xffu], 1);
      }
    }
  }
  __syncthreads();

#pragma unroll 4
  for (int32_t round = 0; round < 4; ++round) {
    __shared__ int32_t last_remain;
    const int32_t current = round & 1;
    const int32_t raw_count = num_input[current];
    const int32_t count = raw_count < kSmemInputSize ? raw_count : kSmemInputSize;

    run_reverse_cumsum();
    if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
      threshold_bin_id = tid;
      num_input[current ^ 1] = 0;
      last_remain = topk - histogram[tid + 1];
    }
    __syncthreads();

    threshold_bin = threshold_bin_id;
    topk -= histogram[threshold_bin + 1];
    const int32_t key_shift = 24 - round * 8;
    if (topk == 0) {
      for (int32_t i = tid; i < count; i += kThreadsPerBlock) {
        const int32_t idx = staged_indices[current * kSmemInputSize + i];
        const int32_t bin =
            static_cast<int32_t>((ordered_float_key(input[row_start + idx]) >> key_shift) & 0xffu);
        if (bin > threshold_bin) {
          const int32_t pos = atomicAdd(&counter, 1);
          indices[pos] = idx;
        }
      }
      __syncthreads();
      break;
    }

    if (tid < kRadix + 1) histogram[tid] = 0;
    __syncthreads();
    for (int32_t i = tid; i < count; i += kThreadsPerBlock) {
      const int32_t idx = staged_indices[current * kSmemInputSize + i];
      const float value = input[row_start + idx];
      const int32_t bin = static_cast<int32_t>((ordered_float_key(value) >> key_shift) & 0xffu);
      if (bin > threshold_bin) {
        const int32_t pos = atomicAdd(&counter, 1);
        indices[pos] = idx;
      } else if (bin == threshold_bin) {
        if (round == 3) {
          const int32_t pos = atomicAdd(&last_remain, -1);
          if (pos > 0) indices[K - pos] = idx;
        } else {
          const int32_t pos = atomicAdd(&num_input[current ^ 1], 1);
          if (pos < kSmemInputSize) {
            staged_indices[(current ^ 1) * kSmemInputSize + pos] = idx;
            const int32_t sub_bin =
                static_cast<int32_t>((ordered_float_key(value) >> (key_shift - 8)) & 0xffu);
            atomicAdd(&histogram[sub_bin], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

static __device__ __forceinline__ int32_t transform_token(int32_t raw_token,
                                                          const int32_t* __restrict__ page_table_entry,
                                                          int64_t page_table_stride,
                                                          const int32_t* __restrict__ topk_indices_offset,
                                                          int32_t offset) {
  if (page_table_entry != nullptr) {
    if (raw_token < 0 || raw_token >= page_table_stride) return -1;
    return page_table_entry[raw_token];
  }
  if (topk_indices_offset != nullptr) return raw_token + offset;
  return raw_token;
}

// One score row of the kpool transform: select K pool groups with radix_topk
// (blockIdx.x = row, one 1024-thread block per row), expand to token indices
// and apply the optional page-table / ragged remap. VERBATIM the body of
// dsa_topk.hip::dsa_topk_transform_kernel (see the file header above); the
// standalone kernel and the fused cooperative tail both call THIS.
template<int K>
static __device__ void dsa_topk_transform_row(const float* __restrict__ score,
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
  const int32_t row = static_cast<int32_t>(blockIdx.x);
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t length = lengths[row];
  const int32_t row_start = row_starts == nullptr ? 0 : row_starts[row];
  int32_t* dst = dst_token_indices + static_cast<size_t>(row) * out_cols;
  const int32_t page_row = page_table_row_index == nullptr ? row : page_table_row_index[row];
  const bool invalid = length < 0 || row_start < 0 ||
                       static_cast<int64_t>(row_start) + length > score_stride ||
                       length > INT32_MAX / pool_size ||
                       (page_table != nullptr &&
                        (page_row < 0 || page_row >= static_cast<int32_t>(gridDim.x))) ||
                       (seq_lens != nullptr && seq_lens[row] < 0);
  if (invalid) {
    for (int32_t col = tid; col < out_cols; col += kThreadsPerBlock) dst[col] = -1;
    return;
  }

  const float* score_row = score + static_cast<size_t>(row) * score_stride;
  const int32_t* page_table_entry =
      page_table == nullptr ? nullptr
                            : page_table + static_cast<size_t>(page_row) * page_table_stride;
  const int32_t offset = topk_indices_offset == nullptr ? 0 : topk_indices_offset[row];
  const int32_t valid_groups = length < K ? length : K;
  const int32_t history_len = valid_groups * pool_size;
  const int32_t tail_count = seq_lens == nullptr ? 0 : seq_lens[row] % pool_size;

  __shared__ int32_t selected_groups[K];
  // When length <= K the top-k winners are trivially [0, length): the CPU
  // oracle does the same (std::iota then skips nth_element), so the identity
  // fill makes the short-row output bit-identical to radix_topk's and lets a
  // single output loop replace the former fast/slow pair.
  if (length <= K) {
    for (int32_t i = tid; i < length; i += kThreadsPerBlock) selected_groups[i] = i;
    __syncthreads();
  } else {
    radix_topk<K>(score_row, selected_groups, row_start, length);
  }
  for (int32_t col = tid; col < out_cols; col += kThreadsPerBlock) {
    if (col < history_len) {
      const int32_t group_id = selected_groups[col / pool_size];
      const int32_t raw_token = group_id * pool_size + col % pool_size;
      dst[col] = transform_token(
          raw_token, page_table_entry, page_table_stride, topk_indices_offset, offset);
    } else if (col < history_len + tail_count) {
      const int32_t raw_token = length * pool_size + col - history_len;
      dst[col] = transform_token(
          raw_token, page_table_entry, page_table_stride, topk_indices_offset, offset);
    } else {
      dst[col] = -1;
    }
  }
}

}  // namespace vkernels::kernels
