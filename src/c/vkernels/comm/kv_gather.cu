// vkernels/comm/kv_gather.cu — CUDA fused indexed K/V layer gather (issue #2).
//
// The host reference (kv_gather.cpp) is the correctness oracle and carries
// the full contract checks (null pointers, positive dimensions, BF16/FP16
// element size, non-negative and in-range source slots). This CUDA path is
// the performance implementation: the host-input entry point validates
// ONCE on the host, uploads `slot_ids` (int32 or int64) to a per-launch
// device buffer, and launches ONE kernel that reads every page's tokens
// from indexed local K/V sources and writes the packed
// [num_pages, page_size, 2, num_kv_heads, head_dim] destination. The
// device-slot entry point skips validation (it cannot read device memory
// without a D2H sync) and launches directly on the caller's slot buffer.
//
// The kernel is memory-bound (zero FLOP/byte — pure copy). It gathers
// [num_kv_heads, head_dim] per (page, token) from local indexed slots and
// writes K then V into the page. Design points (mirroring the donate gather):
//
//   * Grid-stride loop over pages with blockDim=256. Each thread copies
//     one uint4 (16 bytes) of K and one of V per iteration, vectorizing the
//     common case where per-slot bytes is a multiple of 16 (always true for
//     head_dim ∈ {64, 128, 256} with elem_size=2). A <16-byte scalar tail
//     path covers the general case.
//   * Destination writes are coalesced within each token (consecutive
//     16-byte chunks of the page). Source reads gather from arbitrary slots
//     — repeats are allowed (gather semantics) — so no two threads race is
//     guaranteed only by distinct *destinations*, which the page layout
//     ensures (every (page, token, K/V) is written exactly once).
//
// One kernel launch replaces the two separate advanced-index gathers (one
// for K, one for V) the KVAAS `pack_pages` path performs today. Enqueued on
// `stream`; returns without synchronising.
#include "vkernels/comm/kv_gather.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/kv_gather_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <cstdint>
#  include <type_traits>
#  include <vector>

namespace vkernels::comm::cuda {
namespace {

constexpr int kTargetBlocks = 264;

inline int launch_blocks(int num_pages) {
  return num_pages < kTargetBlocks ? num_pages : kTargetBlocks;
}

// Fused indexed K/V gather kernel. `SlotT` is `int` (int32) or `int64_t`.
// Each thread owns ONE 16-byte unit of ONE token of ONE page (the donate
// pattern): grid.x tiles the per-page unit range (page_size * units_per_token)
// with blockDim=256, grid.y strides the pages. When slot_bytes is not a
// multiple of 16 the unit is a single byte (general path). The output is
// byte-for-byte the host reference and the PyTorch two-gather result.
template <typename SlotT>
__global__ void kv_gather_kernel(unsigned char* __restrict__ dst,
                                 const unsigned char* __restrict__ k_src,
                                 const unsigned char* __restrict__ v_src,
                                 const SlotT* __restrict__ slot_ids,
                                 int num_pages, int page_size,
                                 int slot_bytes, int token_stride) {
  const bool aligned = (slot_bytes & 15) == 0;
  const int unit_bytes = aligned ? 16 : 1;
  const int units_per_token = aligned ? (slot_bytes >> 4) : slot_bytes;
  const int page_units = page_size * units_per_token;
  const int u_stride = gridDim.x * blockDim.x;

  for (int p = blockIdx.y; p < num_pages; p += gridDim.y) {
    const SlotT* slots = slot_ids + static_cast<long long>(p) * page_size;
    unsigned char* page =
        dst + static_cast<long long>(p) * page_size * token_stride;
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    for (; u < page_units; u += u_stride) {
      const int t = u / units_per_token;
      const int chunk = u - t * units_per_token;
      const int off = chunk * unit_bytes;
      const long long slot = static_cast<long long>(slots[t]);
      const unsigned char* src_k = k_src + slot * slot_bytes + off;
      const unsigned char* src_v = v_src + slot * slot_bytes + off;
      unsigned char* dst_k = page + static_cast<long long>(t) * token_stride + off;
      unsigned char* dst_v = dst_k + slot_bytes;
      if (unit_bytes == 16) {
        *reinterpret_cast<uint4*>(dst_k) =
            *reinterpret_cast<const uint4*>(src_k);
        *reinterpret_cast<uint4*>(dst_v) =
            *reinterpret_cast<const uint4*>(src_v);
      } else {
        *dst_k = *src_k;
        *dst_v = *src_v;
      }
    }
  }
}

template <typename SlotT>
void launch_typed(unsigned char* dst, const unsigned char* k_src,
                  const unsigned char* v_src, const SlotT* d_slot_ids,
                  int num_pages, int page_size, int slot_bytes,
                  int token_stride, cudaStream_t stream) {
  if (num_pages == 0) return;
  const bool aligned = (slot_bytes & 15) == 0;
  const int units_per_token = aligned ? (slot_bytes >> 4) : slot_bytes;
  const int page_units = page_size * units_per_token;
  int blocks_x = (page_units + 255) / 256;
  if (blocks_x > kTargetBlocks) blocks_x = kTargetBlocks;  // grid-stride covers the rest
  if (blocks_x < 1) blocks_x = 1;
  dim3 block(256);
  dim3 grid(blocks_x, launch_blocks(num_pages));
  kv_gather_kernel<SlotT><<<grid, block, 0, stream>>>(
      dst, k_src, v_src, d_slot_ids, num_pages, page_size, slot_bytes,
      token_stride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda kv_gather launch failed");
}

}  // namespace

void kv_gather_layer(void* dst,
                     const void* k_src, const void* v_src,
                     const void* slot_ids, bool slot_ids_int64,
                     std::size_t num_slots,
                     std::size_t num_pages, std::size_t page_size,
                     std::size_t num_kv_heads, std::size_t head_dim,
                     std::size_t elem_size,
                     cudaStream_t_kv stream) {
  VK_EXPECTS(num_pages == 0 || dst != nullptr, "dst must be non-null");
  VK_EXPECTS(num_pages == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_slots > 0 || num_pages == 0, "num_slots must be positive");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  // Bounds-check every source slot once on the host (gather semantics:
  // repeats and non-monotonic order are allowed, only the range matters).
  const std::size_t total_tokens = num_pages * page_size;
  if (slot_ids_int64) {
    const auto* s = static_cast<const std::int64_t*>(slot_ids);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      VK_EXPECTS(s[i] >= 0, "slot_ids must be non-negative");
      VK_EXPECTS(static_cast<std::uint64_t>(s[i]) <
                     static_cast<std::uint64_t>(num_slots),
                 "slot_ids must be < num_slots");
    }
  } else {
    const auto* s = static_cast<const int*>(slot_ids);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      VK_EXPECTS(s[i] >= 0, "slot_ids must be non-negative");
      VK_EXPECTS(static_cast<std::uint64_t>(s[i]) <
                     static_cast<std::uint64_t>(num_slots),
                 "slot_ids must be < num_slots");
    }
  }

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;
  auto* d = static_cast<unsigned char*>(dst);
  const auto* k = static_cast<const unsigned char*>(k_src);
  const auto* v = static_cast<const unsigned char*>(v_src);
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  // Upload slot_ids (int32 or int64) to a per-launch device buffer.
  if (slot_ids_int64) {
    std::int64_t* d_slots = nullptr;
    cudaError_t err = cudaMallocAsync(&d_slots,
                                      total_tokens * sizeof(std::int64_t), s);
    VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for slot_ids failed");
    err = cudaMemcpyAsync(d_slots, slot_ids, total_tokens * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice, s);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for slot_ids failed");
    launch_typed(d, k, v, d_slots, static_cast<int>(num_pages),
                 static_cast<int>(page_size), slot_bytes, token_stride, s);
    err = cudaFreeAsync(d_slots, s);
    VK_ENSURES(err == cudaSuccess, "cudaFreeAsync for slot_ids failed");
  } else {
    int* d_slots = nullptr;
    cudaError_t err = cudaMallocAsync(&d_slots, total_tokens * sizeof(int), s);
    VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for slot_ids failed");
    err = cudaMemcpyAsync(d_slots, slot_ids, total_tokens * sizeof(int),
                          cudaMemcpyHostToDevice, s);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for slot_ids failed");
    launch_typed(d, k, v, d_slots, static_cast<int>(num_pages),
                 static_cast<int>(page_size), slot_bytes, token_stride, s);
    err = cudaFreeAsync(d_slots, s);
    VK_ENSURES(err == cudaSuccess, "cudaFreeAsync for slot_ids failed");
  }
}

void kv_gather_layer_device_slots(void* dst,
                                  const void* k_src, const void* v_src,
                                  const void* slot_ids, bool slot_ids_int64,
                                  std::size_t num_slots,
                                  std::size_t num_pages, std::size_t page_size,
                                  std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size,
                                  cudaStream_t_kv stream) {
  // Shape validation only (null pointers, positive dimensions, elem_size).
  // Slot contents are on the device and are NOT validated -- reading them
  // would force a D2H sync, defeating the no-sync contract. The caller
  // guarantees non-negative, in-range slots (repeats allowed).
  (void)num_slots;
  VK_EXPECTS(num_pages == 0 || dst != nullptr, "dst must be non-null");
  VK_EXPECTS(num_pages == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;
  auto* d = static_cast<unsigned char*>(dst);
  const auto* k = static_cast<const unsigned char*>(k_src);
  const auto* v = static_cast<const unsigned char*>(v_src);
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  if (slot_ids_int64)
    launch_typed(d, k, v, static_cast<const std::int64_t*>(slot_ids),
                 static_cast<int>(num_pages), static_cast<int>(page_size),
                 slot_bytes, token_stride, s);
  else
    launch_typed(d, k, v, static_cast<const int*>(slot_ids),
                 static_cast<int>(num_pages), static_cast<int>(page_size),
                 slot_bytes, token_stride, s);
}

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
