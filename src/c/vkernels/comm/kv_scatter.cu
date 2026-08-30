// vkernels/comm/kv_scatter.cu — CUDA fused indexed K/V layer scatter (issue #1).
//
// The host reference (kv_scatter.cpp) is the correctness oracle and carries
// the full contract checks (null pointers, positive dimensions, BF16/FP16
// element size, non-negative, in-range AND unique destination slots). This
// CUDA path is the performance implementation: the host-input entry point
// validates ONCE on the host, uploads `slot_ids` (int32 or int64) to a
// per-launch device buffer, and launches ONE kernel that reads every page's
// tokens from the contiguous source and writes the indexed K/V destinations.
// The device-slot entry point skips validation (it cannot read device memory
// without a D2H sync) and launches directly on the caller's slot buffer.
//
// The kernel is memory-bound (zero FLOP/byte — pure copy). It scatters
// [num_kv_heads, head_dim] per (page, token) from a contiguous source into
// indexed destination slots. Design points (the mirror of kv_gather.cu):
//
//   * Grid-stride loop over pages with blockDim=256. Each thread copies
//     one uint4 (16 bytes) of K and one of V per iteration, vectorizing the
//     common case where per-slot bytes is a multiple of 16 (always true for
//     head_dim in {64, 128, 256} with elem_size=2). A <16-byte scalar tail
//     path covers the general case.
//   * Source reads are coalesced within each token (consecutive 16-byte
//     chunks of the page). Destination writes scatter to arbitrary slots —
//     UNIQUE by contract (the host validates this, the device path trusts
//     it) — so no two threads race the same destination byte.
//
// One kernel launch replaces the two separate advanced-index writes (one for
// K, one for V) the KVAAS restore performs today. Enqueued on `stream`;
// returns without synchronising.
#include "vkernels/comm/kv_scatter.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/kv_scatter_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <cstdint>
#  include <type_traits>
#  include <unordered_set>
#  include <vector>

namespace vkernels::comm::cuda {
namespace {

constexpr int kTargetBlocks = 264;

inline int launch_blocks(int num_pages) {
  return num_pages < kTargetBlocks ? num_pages : kTargetBlocks;
}

// Fused indexed K/V scatter kernel. `SlotT` is `int` (int32) or `int64_t`.
// Each thread owns ONE 16-byte unit of ONE token of ONE page (the mirror of
// the gather's donate pattern): grid.x tiles the per-page unit range
// (page_size * units_per_token) with blockDim=256, grid.y strides the pages.
// When slot_bytes is not a multiple of 16 the unit is a single byte (general
// path). The output is byte-for-byte the host reference and the PyTorch
// two-write result.
template <typename SlotT>
__global__ void kv_scatter_kernel(unsigned char* __restrict__ k_dst,
                                  unsigned char* __restrict__ v_dst,
                                  const SlotT* __restrict__ slot_ids,
                                  const unsigned char* __restrict__ src,
                                  int num_pages, int page_size,
                                  int slot_bytes, int token_stride) {
  const bool aligned = (slot_bytes & 15) == 0;
  const int unit_bytes = aligned ? 16 : 1;
  const int units_per_token = aligned ? (slot_bytes >> 4) : slot_bytes;
  const int page_units = page_size * units_per_token;
  const int u_stride = gridDim.x * blockDim.x;

  for (int p = blockIdx.y; p < num_pages; p += gridDim.y) {
    const SlotT* slots = slot_ids + static_cast<long long>(p) * page_size;
    const unsigned char* page =
        src + static_cast<long long>(p) * page_size * token_stride;
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    for (; u < page_units; u += u_stride) {
      const int t = u / units_per_token;
      const int chunk = u - t * units_per_token;
      const int off = chunk * unit_bytes;
      const long long slot = static_cast<long long>(slots[t]);
      const unsigned char* src_k = page + static_cast<long long>(t) * token_stride + off;
      const unsigned char* src_v = src_k + slot_bytes;
      unsigned char* dst_k = k_dst + slot * slot_bytes + off;
      unsigned char* dst_v = v_dst + slot * slot_bytes + off;
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
void launch_typed(unsigned char* k_dst, unsigned char* v_dst,
                  const SlotT* d_slot_ids, const unsigned char* d_src,
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
  kv_scatter_kernel<SlotT><<<grid, block, 0, stream>>>(
      k_dst, v_dst, d_slot_ids, d_src, num_pages, page_size, slot_bytes,
      token_stride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda kv_scatter launch failed");
}

}  // namespace

void kv_scatter_layer(void* k_dst, void* v_dst,
                      const void* slot_ids, bool slot_ids_int64,
                      std::size_t num_slots,
                      const void* src,
                      std::size_t num_pages, std::size_t page_size,
                      std::size_t num_kv_heads, std::size_t head_dim,
                      std::size_t elem_size,
                      cudaStream_t_kv stream) {
  VK_EXPECTS(num_pages == 0 || k_dst != nullptr, "k_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || v_dst != nullptr, "v_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || src != nullptr, "src must be non-null");
  VK_EXPECTS(num_slots > 0 || num_pages == 0, "num_slots must be positive");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  // Bounds-check every destination slot once on the host AND enforce
  // uniqueness (scatter writes disjoint destinations; a duplicate would race
  // the same bytes). Gather semantics allow repeats; scatter does not.
  const std::size_t total_tokens = num_pages * page_size;
  {
    std::unordered_set<std::int64_t> seen;
    seen.reserve(total_tokens);
    if (slot_ids_int64) {
      const auto* s = static_cast<const std::int64_t*>(slot_ids);
      for (std::size_t i = 0; i < total_tokens; ++i) {
        VK_EXPECTS(s[i] >= 0, "slot_ids must be non-negative");
        VK_EXPECTS(static_cast<std::uint64_t>(s[i]) <
                       static_cast<std::uint64_t>(num_slots),
                   "slot_ids must be < num_slots");
        VK_EXPECTS(seen.insert(s[i]).second, "slot_ids must be unique");
      }
    } else {
      const auto* s = static_cast<const int*>(slot_ids);
      for (std::size_t i = 0; i < total_tokens; ++i) {
        const std::int64_t v = static_cast<std::int64_t>(s[i]);
        VK_EXPECTS(v >= 0, "slot_ids must be non-negative");
        VK_EXPECTS(static_cast<std::uint64_t>(v) <
                       static_cast<std::uint64_t>(num_slots),
                   "slot_ids must be < num_slots");
        VK_EXPECTS(seen.insert(v).second, "slot_ids must be unique");
      }
    }
  }

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;
  auto* dk = static_cast<unsigned char*>(k_dst);
  auto* dv = static_cast<unsigned char*>(v_dst);
  const auto* dsrc = static_cast<const unsigned char*>(src);
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
    launch_typed(dk, dv, d_slots, dsrc, static_cast<int>(num_pages),
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
    launch_typed(dk, dv, d_slots, dsrc, static_cast<int>(num_pages),
                 static_cast<int>(page_size), slot_bytes, token_stride, s);
    err = cudaFreeAsync(d_slots, s);
    VK_ENSURES(err == cudaSuccess, "cudaFreeAsync for slot_ids failed");
  }
}

void kv_scatter_layer_device_slots(void* k_dst, void* v_dst,
                                   const void* slot_ids, bool slot_ids_int64,
                                   std::size_t num_slots,
                                   const void* src,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   cudaStream_t_kv stream) {
  // Shape validation only (null pointers, positive dimensions, elem_size).
  // Slot contents are on the device and are NOT validated -- reading them
  // would force a D2H sync, defeating the no-sync contract. The caller
  // guarantees non-negative, in-range AND UNIQUE slots.
  (void)num_slots;
  VK_EXPECTS(num_pages == 0 || k_dst != nullptr, "k_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || v_dst != nullptr, "v_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || src != nullptr, "src must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;
  auto* dk = static_cast<unsigned char*>(k_dst);
  auto* dv = static_cast<unsigned char*>(v_dst);
  const auto* dsrc = static_cast<const unsigned char*>(src);
  cudaStream_t s = static_cast<cudaStream_t>(stream);

  if (slot_ids_int64)
    launch_typed(dk, dv, static_cast<const std::int64_t*>(slot_ids), dsrc,
                 static_cast<int>(num_pages), static_cast<int>(page_size),
                 slot_bytes, token_stride, s);
  else
    launch_typed(dk, dv, static_cast<const int*>(slot_ids), dsrc,
                 static_cast<int>(num_pages), static_cast<int>(page_size),
                 slot_bytes, token_stride, s);
}

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
