// vkernels/comm/p2p_kv_restore.cu — CUDA fused peer-to-indexed-KV restore.
//
// The host reference (p2p_kv_restore.cpp) is the correctness oracle and
// carries the full contract checks. This CUDA path is the performance
// implementation: it validates the run list ONCE on the host (unique slots,
// non-null sources, positive dimensions), resolves effective page pointers,
// stages the resolved pointer array and slot_ids into per-launch device
// buffers, and launches ONE kernel that reads every page directly from peer
// memory over NVLink and writes into the indexed K/V destinations.
//
// The kernel is memory-bound (zero FLOP/byte — pure copy). It reads
// [page_size, 2, num_kv_heads, head_dim] per page from peer UVA and
// scatters K and V into separate indexed local buffers. The key design
// points:
//
//   * Grid-stride loop over pages with blockDim=256. Each thread copies
//     one uint4 (16 bytes) of K and one of V per iteration, vectorizing
//     the common case where per-slot bytes is a multiple of 16 (always
//     true for head_dim ∈ {64, 128, 256} with elem_size=2). A <16-byte
//     scalar tail path covers the general case.
//   * Source reads from peer UVA are coalesced within each token (all
//     threads in a block read consecutive 16-byte chunks from the page).
//     Destination writes scatter to arbitrary slots — the caller guarantees
//     unique slots so no two blocks write the same destination.
//   * One kernel launch replaces the two-stage path's N cudaMemcpyAsync
//     calls + one scatter kernel. The scratch buffer and its local-HBM
//     read/write pass are eliminated entirely.
//
// The two-stage GPU reference (p2p_kv_restore_layer_twostage) is provided
// for correctness baselines and scratch-cost measurement.
#include "vkernels/comm/p2p_kv_restore.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/p2p_kv_restore_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <algorithm>
#  include <cstddef>
#  include <cstdint>
#  include <vector>

namespace vkernels::comm {
namespace cuda {
namespace {

// Fixed-width page descriptor uploaded to the device. The resolved
// `src` pointer is the effective `peer_src_ptrs[p] + src_page_offsets[p]`
// computed on the host so the kernel does no per-page pointer arithmetic.
// `slot_base` points to the start of this page's slot_ids sub-array within
// the single device-side slot_ids buffer, so the kernel indexes it as
// `slot_base[t]` for token `t`.
struct PageDev {
  const unsigned char* src;  // peer UVA resolved pointer
  const int* slot_base;      // &slot_ids[p * page_size]
};

// ---------------------------------------------------------------------------
// Fused peer-to-indexed-KV restore kernel
// ---------------------------------------------------------------------------
//
// Grid: one block per page (or grid-stride if pages exceed grid limit).
// Block: 256 threads.
//
// Each block handles one page per grid-stride iteration. Within a page,
// threads collaboratively copy every token's K+V data:
//   - Source is [page_size, 2, num_kv_heads, head_dim] in row-major.
//   - For token t, slot = slot_ids[t], then:
//     K: src + t*token_stride → k_dst + slot*slot_bytes
//     V: src + t*token_stride + slot_bytes → v_dst + slot*slot_bytes
//   - Thread i copies K[16i : 16i+16) and V[16i : 16i+16) as uint4,
//     with a scalar tail for the <16-byte remainder.
__global__ void p2p_kv_restore_kernel(
    unsigned char* k_dst, unsigned char* v_dst,
    const PageDev* __restrict__ pages, int num_pages,
    int page_size, int slot_bytes, int token_stride) {

  for (int p = blockIdx.x; p < num_pages; p += gridDim.x) {
    PageDev page = pages[p];
    const unsigned char* __restrict__ src = page.src;
    const int* __restrict__ slots = page.slot_base;

    for (int t = 0; t < page_size; ++t) {
      int slot = slots[t];
      const unsigned char* src_k = src + t * token_stride;
      const unsigned char* src_v = src_k + slot_bytes;
      unsigned char* dst_k = k_dst + slot * slot_bytes;
      unsigned char* dst_v = v_dst + slot * slot_bytes;

      // Vectorized path: uint4 = 16 bytes. slot_bytes is typically a
      // multiple of 16 for realistic shapes (head_dim ∈ {64,128,256},
      // elem_size=2), but we keep the scalar tail for generality.
      const int vec_chunks = slot_bytes >> 4;  // slot_bytes / 16
      for (int i = threadIdx.x; i < vec_chunks; i += blockDim.x) {
        const int off = i << 4;  // i * 16
        *reinterpret_cast<uint4*>(dst_k + off) =
            *reinterpret_cast<const uint4*>(src_k + off);
        *reinterpret_cast<uint4*>(dst_v + off) =
            *reinterpret_cast<const uint4*>(src_v + off);
      }

      // Scalar tail for the <16-byte remainder.
      const int tail_off = vec_chunks << 4;
      const int tail_len = slot_bytes - tail_off;
      for (int i = threadIdx.x; i < tail_len; i += blockDim.x) {
        dst_k[tail_off + i] = src_k[tail_off + i];
        dst_v[tail_off + i] = src_v[tail_off + i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Indexed KV scatter kernel (used by the two-stage reference)
// ---------------------------------------------------------------------------
//
// Copies from a flat scratch buffer (same layout as the peer source:
// [page_size, 2, num_kv_heads, head_dim]) into indexed K/V destinations.
// Grid-stride over pages; identical thread mapping to the fused kernel.
__global__ void kv_scatter_kernel(unsigned char* k_dst, unsigned char* v_dst,
                                  const unsigned char* __restrict__ scratch,
                                  const int* __restrict__ slot_ids,
                                  int num_pages, int page_size,
                                  int slot_bytes, int token_stride) {
  for (int p = blockIdx.x; p < num_pages; p += gridDim.x) {
    const unsigned char* src = scratch + p * page_size * token_stride;
    const int* slots = slot_ids + p * page_size;

    for (int t = 0; t < page_size; ++t) {
      int slot = slots[t];
      const unsigned char* src_k = src + t * token_stride;
      const unsigned char* src_v = src_k + slot_bytes;
      unsigned char* dst_k = k_dst + slot * slot_bytes;
      unsigned char* dst_v = v_dst + slot * slot_bytes;

      const int vec_chunks = slot_bytes >> 4;
      for (int i = threadIdx.x; i < vec_chunks; i += blockDim.x) {
        const int off = i << 4;
        *reinterpret_cast<uint4*>(dst_k + off) =
            *reinterpret_cast<const uint4*>(src_k + off);
        *reinterpret_cast<uint4*>(dst_v + off) =
            *reinterpret_cast<const uint4*>(src_v + off);
      }
      const int tail_off = vec_chunks << 4;
      const int tail_len = slot_bytes - tail_off;
      for (int i = threadIdx.x; i < tail_len; i += blockDim.x) {
        dst_k[tail_off + i] = src_k[tail_off + i];
        dst_v[tail_off + i] = src_v[tail_off + i];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Launch helpers
// ---------------------------------------------------------------------------

// Number of blocks to launch: min(num_pages, a sensible SM-occupancy cap).
// More blocks than SMs gives the hardware warp schedulers room to overlap
// peer-read latency with execution. 2× the SM count (e.g. 264 on H100) is a
// safe default that keeps the grid small while still hiding NVLink latency.
inline int launch_blocks(int num_pages) {
  // Launch enough blocks to fill SMs twice (generous for H100's 132 SMs).
  constexpr int kTargetBlocks = 264;
  return std::min(num_pages, kTargetBlocks);
}

void launch_fused_kernel(unsigned char* k_dst, unsigned char* v_dst,
                         const PageDev* d_pages, int num_pages,
                         int page_size, int slot_bytes, int token_stride,
                         cudaStream_t stream) {
  if (num_pages == 0) return;
  dim3 block(256);
  dim3 grid(launch_blocks(num_pages));
  p2p_kv_restore_kernel<<<grid, block, 0, stream>>>(
      k_dst, v_dst, d_pages, num_pages, page_size, slot_bytes, token_stride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda p2p_kv_restore launch failed");
}

void launch_scatter_kernel(unsigned char* k_dst, unsigned char* v_dst,
                           const unsigned char* d_scratch, const int* d_slot_ids,
                           int num_pages, int page_size, int slot_bytes,
                           int token_stride, cudaStream_t stream) {
  if (num_pages == 0) return;
  dim3 block(256);
  dim3 grid(launch_blocks(num_pages));
  kv_scatter_kernel<<<grid, block, 0, stream>>>(
      k_dst, v_dst, d_scratch, d_slot_ids, num_pages, page_size,
      slot_bytes, token_stride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda kv_scatter launch failed");
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void p2p_kv_restore_layer(void* k_dst, void* v_dst,
                          const int* slot_ids,
                          const void* const* peer_src_ptrs,
                          const std::size_t* src_page_offsets,
                          std::size_t num_pages, std::size_t page_size,
                          std::size_t num_kv_heads, std::size_t head_dim,
                          std::size_t elem_size,
                          cudaStream_t stream) {
  // Contract checks (same as the host oracle, throws on violation).
  VK_EXPECTS(num_pages == 0 || k_dst != nullptr, "k_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || v_dst != nullptr, "v_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || peer_src_ptrs != nullptr, "peer_src_ptrs must be non-null");
  VK_EXPECTS(num_pages == 0 || src_page_offsets != nullptr,
             "src_page_offsets must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;

  // Resolve effective page pointers and build host-side descriptors.
  std::vector<PageDev> pages(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_src_ptrs[p] != nullptr, "peer_src_ptrs[p] must be non-null");
    pages[p].src =
        static_cast<const unsigned char*>(peer_src_ptrs[p]) + src_page_offsets[p];
    // slot_base is filled after we upload slot_ids to the device.
  }

  // Upload slot_ids to the device (one contiguous buffer).
  const std::size_t total_tokens = num_pages * page_size;
  const std::size_t slot_bytes_dev = total_tokens * sizeof(int);
  int* d_slot_ids = nullptr;
  cudaError_t err = cudaMallocAsync(&d_slot_ids, slot_bytes_dev, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for slot_ids failed");
  err = cudaMemcpyAsync(d_slot_ids, slot_ids, slot_bytes_dev,
                        cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for slot_ids failed");

  // Fill in the per-page slot_base pointers (now that we have d_slot_ids).
  for (std::size_t p = 0; p < num_pages; ++p)
    pages[p].slot_base = d_slot_ids + p * page_size;

  // Upload page descriptors to the device.
  const std::size_t pages_bytes = num_pages * sizeof(PageDev);
  PageDev* d_pages = nullptr;
  err = cudaMallocAsync(&d_pages, pages_bytes, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for page descriptors failed");
  err = cudaMemcpyAsync(d_pages, pages.data(), pages_bytes,
                        cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for page descriptors failed");

  // Launch the fused kernel.
  launch_fused_kernel(static_cast<unsigned char*>(k_dst),
                      static_cast<unsigned char*>(v_dst),
                      d_pages, static_cast<int>(num_pages),
                      static_cast<int>(page_size), slot_bytes, token_stride,
                      stream);

  // Free the staging buffers (the kernel is already enqueued behind them).
  cudaFreeAsync(d_slot_ids, stream);
  cudaFreeAsync(d_pages, stream);
}

void p2p_kv_restore_layer_twostage(void* k_dst, void* v_dst,
                                   const int* slot_ids,
                                   const void* const* peer_src_ptrs,
                                   const std::size_t* src_page_offsets,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   cudaStream_t stream) {
  if (num_pages == 0) return;

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;
  const int scratch_per_page = static_cast<int>(page_size) * token_stride;

  // Allocate scratch buffer (one page at a time, reused).
  unsigned char* d_scratch = nullptr;
  cudaError_t err = cudaMallocAsync(&d_scratch, scratch_per_page, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for scratch failed");

  // Upload slot_ids once.
  const std::size_t total_tokens = num_pages * page_size;
  int* d_slot_ids = nullptr;
  err = cudaMallocAsync(&d_slot_ids, total_tokens * sizeof(int), stream);
  VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for slot_ids failed");
  err = cudaMemcpyAsync(d_slot_ids, slot_ids, total_tokens * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for slot_ids failed");

  // Stage 1: copy each page from peer to scratch (one cudaMemcpyAsync per page).
  // Stage 2: scatter from scratch to indexed K/V (one kernel launch per page,
  // or we batch them all into one scatter at the end).
  //
  // We execute both stages page-by-page on the SAME stream so that the
  // scatter kernel for page p can reuse the same scratch buffer as page p-1.
  for (std::size_t p = 0; p < num_pages; ++p) {
    const auto* src = static_cast<const unsigned char*>(peer_src_ptrs[p]) +
                      src_page_offsets[p];
    // Stage 1: peer → scratch.
    err = cudaMemcpyAsync(d_scratch, src, scratch_per_page,
                          cudaMemcpyDefault, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync peer→scratch failed");
    // Stage 2: scratch → indexed K/V (scatter for this one page).
    launch_scatter_kernel(static_cast<unsigned char*>(k_dst),
                          static_cast<unsigned char*>(v_dst),
                          d_scratch, d_slot_ids + p * page_size,
                          1, static_cast<int>(page_size),
                          slot_bytes, token_stride, stream);
  }

  cudaFreeAsync(d_scratch, stream);
  cudaFreeAsync(d_slot_ids, stream);
}

}  // namespace cuda
}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
