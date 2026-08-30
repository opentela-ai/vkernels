// vkernels/kernels/dsa_topk.hip
//
// Radix top-k transform for the DSA kpool indexer (gfx942 / MI300A). The
// radix selector is a HIP port of sglang's kpool_topk_transform JIT kernel:
// an 8-bit fp16 coarse pass followed by fp32-key refinement. One workgroup
// owns one score row, selects pool groups in shared memory, expands them to
// token indices, then applies the optional page-table or ragged mapping.
#include "vkernels/kernels/dsa_topk.hpp"

#if VKERNELS_HAS_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace vkernels::kernels::hip {
namespace {

constexpr int kThreadsPerBlock = 1024;
constexpr int kRadix = 256;
constexpr size_t kDynamicSmemBytes = 8 * 1024 * sizeof(uint32_t);
constexpr int kSmemInputSize = static_cast<int>(kDynamicSmemBytes / (2 * sizeof(int32_t)));

__device__ __forceinline__ uint8_t coarse_float_key(float value) {
  const uint16_t bits = __half_as_ushort(__float2half_rn(value));
  const uint16_t key =
      (bits & 0x8000u) != 0 ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t ordered_float_key(float value) {
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

template<int K>
__device__ void radix_topk(const float* __restrict__ input,
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

__device__ __forceinline__ int32_t transform_token(int32_t raw_token,
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
    for (int32_t col = tid; col < out_cols; col += kThreadsPerBlock)
      dst[col] = -1;
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

  if (length <= K) {
    for (int32_t col = tid; col < out_cols; col += kThreadsPerBlock) {
      if (col < history_len) {
        dst[col] =
            transform_token(col, page_table_entry, page_table_stride, topk_indices_offset, offset);
      } else if (col < history_len + tail_count) {
        const int32_t raw_token = length * pool_size + col - history_len;
        dst[col] = transform_token(
            raw_token, page_table_entry, page_table_stride, topk_indices_offset, offset);
      } else {
        dst[col] = -1;
      }
    }
    return;
  }

  __shared__ int32_t selected_groups[K];
  radix_topk<K>(score_row, selected_groups, row_start, length);
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

} // namespace vkernels::kernels::hip
#endif // VKERNELS_HAS_HIP
