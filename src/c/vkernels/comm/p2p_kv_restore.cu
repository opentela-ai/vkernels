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
#  include <memory>
#  include <unordered_set>
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

// Plan descriptor (issue #27). Unlike PageDev, `src` is the peer page BASE
// (no per-page offset): the per-layer `src_offset` is a scalar kernel
// argument added at execute time, so one descriptor buffer is reused across
// every layer. `slot_base` points into either the plan's own device slot map
// (host-input variant) or the caller's `device_indices` buffer
// (from_device_slots variant) at offset `p * page_size`.
struct PagePlanDev {
  const unsigned char* src;
  const int* slot_base;
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
      // The vec path is only safe when slot_bytes is a multiple of 16:
      // otherwise src_v = src_k + slot_bytes and dst_k = k_dst + slot*
      // slot_bytes are misaligned, and a 16-byte uint4 load/store from a
      // sub-16-aligned address is an illegal memory access on the GPU.
      const bool aligned = (slot_bytes & 15) == 0;
      const int vec_chunks = aligned ? (slot_bytes >> 4) : 0;  // slot_bytes / 16
      for (int i = threadIdx.x; i < vec_chunks; i += blockDim.x) {
        const int off = i << 4;  // i * 16
        *reinterpret_cast<uint4*>(dst_k + off) =
            *reinterpret_cast<const uint4*>(src_k + off);
        *reinterpret_cast<uint4*>(dst_v + off) =
            *reinterpret_cast<const uint4*>(src_v + off);
      }

      // Scalar tail for the <16-byte remainder (or the whole slot when
      // slot_bytes is not 16-aligned).
      const int tail_off = aligned ? (vec_chunks << 4) : 0;
      const int tail_len = aligned ? (slot_bytes - tail_off) : slot_bytes;
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

      // The vec path is only safe when slot_bytes is a multiple of 16:
      // otherwise src_v = src + slot_bytes and dst_v = v_dst + slot*slot_bytes
      // are misaligned, and a 16-byte uint4 load/store from a sub-16-aligned
      // address is an illegal memory access on the GPU.
      const bool aligned = (slot_bytes & 15) == 0;
      const int vec_chunks = aligned ? (slot_bytes >> 4) : 0;
      for (int i = threadIdx.x; i < vec_chunks; i += blockDim.x) {
        const int off = i << 4;
        *reinterpret_cast<uint4*>(dst_k + off) =
            *reinterpret_cast<const uint4*>(src_k + off);
        *reinterpret_cast<uint4*>(dst_v + off) =
            *reinterpret_cast<const uint4*>(src_v + off);
      }
      const int tail_off = aligned ? (vec_chunks << 4) : 0;
      const int tail_len = aligned ? (slot_bytes - tail_off) : slot_bytes;
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

// Public scatter entry point (issue #27): one launch over a contiguous
// [num_pages, page_size, 2, num_kv_heads, head_dim] scratch buffer that a
// caller already gathered (e.g. with a prepared P2PGatherPlan2D), using a
// persistent device slot map. This is the "prepared gather plus indexed
// scatter" baseline the plan is measured against.

// Grid-axis limit (matches the gather kernels).
constexpr unsigned kPlanMaxGridAxis = 65535u;

// One-time int64 -> int32 device conversion (issue #27). The int64
// device-slot plan owns its int32 slot map and fills it with this kernel so
// no D2H sync is needed; the caller may free the int64 buffer as soon as the
// constructor returns (the subsequent blocking cudaMemcpy for the page
// descriptors serialises after this launch on the default stream).
__global__ void convert_slots_i64_to_i32_kernel(
    const long long* __restrict__ src, int* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<int>(src[i]);
}

void launch_convert_i64(const long long* src, int* dst, int n) {
  if (n <= 0) return;
  dim3 block(256);
  unsigned long long grids =
      (static_cast<unsigned long long>(n) + 255ULL) / 256ULL;
  if (grids > kPlanMaxGridAxis) grids = kPlanMaxGridAxis;
  dim3 grid(static_cast<unsigned>(grids));
  convert_slots_i64_to_i32_kernel<<<grid, block>>>(src, dst, n);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda int64->int32 slot conversion launch failed");
}

// Plan restore kernel (issue #27). One block per (page, token-group) pair:
// grid.x tiles each page's unit range, grid.y = num_pages, blockDim = 256.
// A unit is one chunk of K plus one chunk of V for one token:
//   * vectorized path (slot_bytes % 16 == 0 and the base+offset is
//     16-byte aligned): unit = 16 bytes of K + 16 bytes of V (one uint4
//     each). units_per_token = slot_bytes / 16.
//   * scalar fallback: unit = 1 byte of K + 1 byte of V. units_per_token =
//     slot_bytes. This handles any slot_bytes / offset alignment and still
//     fills the SMs via the page*units grid (vs one block per page).
//
// The unit index u (blockIdx.x*blockDim.x + threadIdx.x) maps to
// (token, chunk) by t = u / units_per_token, chunk = u % units_per_token;
// K is at src + src_offset + t*token_stride + chunk*unit_bytes and V at the
// same offset plus slot_bytes. Slot lookups (pages[p].slot_base[t]) are
// broadcast across the threads of a token, so the small device-side slot
// map is read once per token from L2.
__global__ void p2p_kv_restore_plan_kernel(
    unsigned char* k_dst, unsigned char* v_dst,
    const PagePlanDev* __restrict__ pages, int num_pages,
    int page_size, int slot_bytes, int token_stride,
    int units_per_token, int unit_bytes,
    unsigned long long src_offset) {
  int p = blockIdx.y;
  if (p >= num_pages) return;
  PagePlanDev page = pages[p];
  const unsigned char* __restrict__ base = page.src;
  const int* __restrict__ slots = page.slot_base;

  unsigned long long u =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long page_units =
      static_cast<unsigned long long>(page_size) * units_per_token;
  if (u >= page_units) return;

  int t = static_cast<int>(u / static_cast<unsigned long long>(units_per_token));
  int chunk = static_cast<int>(u % static_cast<unsigned long long>(units_per_token));
  int slot = slots[t];
  unsigned long long off = static_cast<unsigned long long>(chunk) * unit_bytes;
  const unsigned char* src_k = base + src_offset + t * token_stride + off;
  const unsigned char* src_v = src_k + slot_bytes;
  unsigned char* dst_k = k_dst + static_cast<unsigned long long>(slot) * slot_bytes + off;
  unsigned char* dst_v = v_dst + static_cast<unsigned long long>(slot) * slot_bytes + off;
  if (unit_bytes == 16) {
    *reinterpret_cast<uint4*>(dst_k) = *reinterpret_cast<const uint4*>(src_k);
    *reinterpret_cast<uint4*>(dst_v) = *reinterpret_cast<const uint4*>(src_v);
  } else {
    *dst_k = *src_k;
    *dst_v = *src_v;
  }
}

// One launch for a whole prepared plan. tile = 256 units; grid.x tiles the
// page's unit range, grid.y = num_pages (capped at kPlanMaxGridAxis).
void launch_plan_kernel(unsigned char* k_dst, unsigned char* v_dst,
                        const PagePlanDev* d_pages, int num_pages,
                        int page_size, int slot_bytes, int token_stride,
                        unsigned long long src_offset, cudaStream_t stream) {
  if (num_pages == 0) return;
  const bool aligned = (slot_bytes % 16 == 0) && (src_offset % 16 == 0);
  const int units_per_token = aligned ? (slot_bytes / 16) : slot_bytes;
  const int unit_bytes = aligned ? 16 : 1;
  const unsigned long long page_units =
      static_cast<unsigned long long>(page_size) * units_per_token;
  const unsigned grid_x =
      page_units == 0 ? 1u
                      : static_cast<unsigned>((page_units + 255ull) / 256ull);
  const unsigned grid_y =
      static_cast<unsigned>(num_pages) < kPlanMaxGridAxis
          ? static_cast<unsigned>(num_pages)
          : kPlanMaxGridAxis;
  dim3 block(256);
  dim3 grid(grid_x, grid_y);
  p2p_kv_restore_plan_kernel<<<grid, block, 0, stream>>>(
      k_dst, v_dst, d_pages, num_pages, page_size, slot_bytes, token_stride,
      units_per_token, unit_bytes, src_offset);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda p2p_kv_restore_plan launch failed");
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

  // Validate slot contents on the host, mirroring the host reference
  // p2p_kv_restore_layer: every slot must be non-negative and unique, the
  // fused kernel's disjoint-destination assumption. `slot_ids` may be a
  // host OR a device pointer (the device C ABI tests pass a device
  // buffer), so mirror it into a host staging buffer with the synchronous
  // cudaMemcpy (cudaMemcpyDefault auto-detects the source kind under
  // UVA) before the kernel launch. This is the same per-call validation
  // the host one-shot performs; the prepared plan
  // (vkernels_p2p_kv_restore_plan_create) validates ONCE at create and
  // stays sync-free on execute(). The synchronous copy runs on the same
  // stream as the async ops below when `stream` is the default stream,
  // and otherwise reads only the caller's read-only `slot_ids`.
  const std::size_t total_tokens = num_pages * page_size;
  {
    std::vector<int> h_slots(total_tokens);
    cudaError_t val_err = cudaMemcpy(
        h_slots.data(), slot_ids, total_tokens * sizeof(int),
        cudaMemcpyDefault);
    VK_ENSURES(val_err == cudaSuccess,
               "cudaMemcpy for cross-node slot validation failed");
    std::unordered_set<int> seen;
    seen.reserve(total_tokens);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      int slot = h_slots[i];
      VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
      VK_EXPECTS(seen.insert(slot).second, "slot_ids must be unique");
    }
  }

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

// Public indexed-KV scatter (issue #27). `slot_ids` is a caller-owned
// DEVICE pointer (shape [num_pages * page_size]); `scratch` is a device
// buffer of [num_pages, page_size, 2, num_kv_heads, head_dim] gathered by a
// prepared P2PGatherPlan2D. One kernel launch, no upload, no validation —
// the caller MUST guarantee unique, non-negative, in-range slots and keep
// both `slot_ids` and `scratch` alive until the kernel completes on `stream`.
// This is the "prepared scatter" baseline the plan is measured against.
void kv_scatter(void* k_dst, void* v_dst, const void* scratch,
                const int* slot_ids, std::size_t num_pages,
                std::size_t page_size, std::size_t num_kv_heads,
                std::size_t head_dim, std::size_t elem_size,
                cudaStream_t stream) {
  VK_EXPECTS(num_pages == 0 || k_dst != nullptr, "k_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || v_dst != nullptr, "v_dst must be non-null");
  VK_EXPECTS(num_pages == 0 || scratch != nullptr, "scratch must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");
  if (num_pages == 0) return;

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;

  launch_scatter_kernel(static_cast<unsigned char*>(k_dst),
                        static_cast<unsigned char*>(v_dst),
                        static_cast<const unsigned char*>(scratch), slot_ids,
                        static_cast<int>(num_pages),
                        static_cast<int>(page_size), slot_bytes, token_stride,
                        stream);
}

// ---------------------------------------------------------------------------
// Prepared fused peer-to-indexed-KV restore plan (CUDA, issue #27)
// ---------------------------------------------------------------------------

struct P2PKvRestorePlan::Impl {
  int num_pages;
  int page_size;
  int slot_bytes;
  int token_stride;
  int num_slots;
  int num_kv_heads;
  int head_dim;
  int elem_size;
  std::size_t total_bytes;
  PagePlanDev* d_pages;      // persistent [num_pages] descriptor buffer
  const int* slot_ids;       // device pointer: owned or borrowed
  bool owns_slot_ids;        // true for host-input and int64; false for int32 borrow

  Impl() : num_pages(0), page_size(0), slot_bytes(0), token_stride(0),
           num_slots(0), num_kv_heads(0), head_dim(0), elem_size(0),
           total_bytes(0), d_pages(nullptr), slot_ids(nullptr),
           owns_slot_ids(false) {}
};

// Shared constructor body: validate metadata shape, resolve peer bases,
// allocate the persistent descriptor buffer and (for host-input / int64) the
// device slot map. The synchronous cudaMalloc/cudaMemcpy use stream 0 so the
// persistent buffers are not bound to any one stream and concurrent
// execute() on arbitrary streams is race-free (CUDA default stream is
// synchronising with respect to per-thread streams).
P2PKvRestorePlan::Impl* P2PKvRestorePlan::init(
    SlotSource mode, std::size_t num_slots, std::size_t num_kv_heads,
    std::size_t head_dim, std::size_t elem_size, const void* slot_ids,
    const void* const* peer_src_ptrs, std::size_t num_pages,
    std::size_t page_size) {
  VK_EXPECTS(num_pages == 0 || num_slots > 0, "num_slots must be positive");
  VK_EXPECTS(num_pages == 0 || page_size > 0, "page_size must be positive");
  VK_EXPECTS(num_pages == 0 || num_kv_heads > 0, "num_kv_heads must be positive");
  VK_EXPECTS(num_pages == 0 || head_dim > 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || peer_src_ptrs != nullptr,
             "peer_src_ptrs must be non-null");

  auto* impl = new Impl();
  impl->num_pages = static_cast<int>(num_pages);
  impl->page_size = static_cast<int>(page_size);
  impl->slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  impl->token_stride = 2 * impl->slot_bytes;
  impl->num_slots = static_cast<int>(num_slots);
  impl->num_kv_heads = static_cast<int>(num_kv_heads);
  impl->head_dim = static_cast<int>(head_dim);
  impl->elem_size = static_cast<int>(elem_size);
  impl->total_bytes = num_pages * page_size *
                      static_cast<std::size_t>(impl->token_stride);
  if (num_pages == 0) return impl;

  const std::size_t total_tokens = num_pages * page_size;
  if (mode == SlotSource::HostValidated) {
    const int* host_slots = static_cast<const int*>(slot_ids);
    std::unordered_set<int> seen;
    seen.reserve(total_tokens);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      int slot = host_slots[i];
      VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
      VK_EXPECTS(slot < static_cast<int>(num_slots),
                 "slot_ids must be < num_slots");
      VK_EXPECTS(seen.insert(slot).second, "slot_ids must be unique");
    }
  }

  // Build host-side page descriptors (resolved peer bases; slot base filled
  // after the device slot map is allocated).
  std::vector<PagePlanDev> pages(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_src_ptrs[p] != nullptr, "peer_src_ptrs[p] must be non-null");
    pages[p].src = static_cast<const unsigned char*>(peer_src_ptrs[p]);
  }

  // Device slot map: owned for host-input (H2D copy) and int64 (conversion
  // kernel), borrowed for int32 device-slots.
  const int* d_slot_ids = nullptr;
  int* owned = nullptr;
  if (mode == SlotSource::HostValidated) {
    cudaError_t err = cudaMalloc(&owned, total_tokens * sizeof(int));
    VK_ENSURES(err == cudaSuccess, "cudaMalloc for plan slot_ids failed");
    err = cudaMemcpy(owned, slot_ids, total_tokens * sizeof(int),
                     cudaMemcpyHostToDevice);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpy for plan slot_ids failed");
    d_slot_ids = owned;
    impl->owns_slot_ids = true;
  } else if (mode == SlotSource::DeviceInt64Converted) {
    cudaError_t err = cudaMalloc(&owned, total_tokens * sizeof(int));
    VK_ENSURES(err == cudaSuccess, "cudaMalloc for plan slot_ids failed");
    // Convert the caller's int64 device buffer in-place to an owned int32
    // buffer. Launched on the default stream; the blocking cudaMemcpy for
    // the page descriptors below serialises after it, so the caller may free
    // the int64 buffer as soon as the constructor returns.
    launch_convert_i64(static_cast<const long long*>(slot_ids), owned,
                       static_cast<int>(total_tokens));
    d_slot_ids = owned;
    impl->owns_slot_ids = true;
  } else {  // DeviceInt32Borrowed
    d_slot_ids = static_cast<const int*>(slot_ids);  // caller's device pointer
    impl->owns_slot_ids = false;
  }
  impl->slot_ids = d_slot_ids;

  // Persistent page-descriptor buffer (not stream-bound).
  PagePlanDev* d_pages = nullptr;
  cudaError_t err = cudaMalloc(&d_pages, num_pages * sizeof(PagePlanDev));
  VK_ENSURES(err == cudaSuccess, "cudaMalloc for plan page descriptors failed");
  for (std::size_t p = 0; p < num_pages; ++p)
    pages[p].slot_base = d_slot_ids + p * page_size;
  err = cudaMemcpy(d_pages, pages.data(), num_pages * sizeof(PagePlanDev),
                   cudaMemcpyHostToDevice);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpy for plan page descriptors failed");
  impl->d_pages = d_pages;

  return impl;
}

P2PKvRestorePlan::P2PKvRestorePlan(std::size_t num_slots,
                                   std::size_t num_kv_heads,
                                   std::size_t head_dim, std::size_t elem_size,
                                   const int* slot_ids,
                                   const void* const* peer_src_ptrs,
                                   std::size_t num_pages,
                                   std::size_t page_size)
    : impl_(init(SlotSource::HostValidated, num_slots, num_kv_heads,
                 head_dim, elem_size, slot_ids, peer_src_ptrs, num_pages,
                 page_size)) {}

P2PKvRestorePlan::P2PKvRestorePlan(vkernels::comm::from_device_slots_t,
                                   std::size_t num_slots,
                                   std::size_t num_kv_heads,
                                   std::size_t head_dim, std::size_t elem_size,
                                   const int* device_indices,
                                   const void* const* peer_src_ptrs,
                                   std::size_t num_pages,
                                   std::size_t page_size)
    : impl_(init(SlotSource::DeviceInt32Borrowed, num_slots, num_kv_heads,
                 head_dim, elem_size, device_indices, peer_src_ptrs, num_pages,
                 page_size)) {}

P2PKvRestorePlan::P2PKvRestorePlan(vkernels::comm::from_device_slots_int64_t,
                                   std::size_t num_slots,
                                   std::size_t num_kv_heads,
                                   std::size_t head_dim, std::size_t elem_size,
                                   const std::int64_t* device_indices,
                                   const void* const* peer_src_ptrs,
                                   std::size_t num_pages,
                                   std::size_t page_size)
    : impl_(init(SlotSource::DeviceInt64Converted, num_slots, num_kv_heads,
                 head_dim, elem_size, device_indices, peer_src_ptrs, num_pages,
                 page_size)) {}

P2PKvRestorePlan::~P2PKvRestorePlan() {
  if (!impl_) return;
  if (impl_->owns_slot_ids && impl_->slot_ids)
    cudaFree(const_cast<int*>(impl_->slot_ids));
  if (impl_->d_pages) cudaFree(impl_->d_pages);
  delete impl_;
}

std::size_t P2PKvRestorePlan::num_pages() const { return impl_->num_pages; }
std::size_t P2PKvRestorePlan::num_kv_heads() const { return impl_->num_kv_heads; }
std::size_t P2PKvRestorePlan::head_dim() const { return impl_->head_dim; }
std::size_t P2PKvRestorePlan::elem_size() const { return impl_->elem_size; }
std::size_t P2PKvRestorePlan::num_slots() const { return impl_->num_slots; }
std::size_t P2PKvRestorePlan::page_size() const { return impl_->page_size; }
std::size_t P2PKvRestorePlan::total_bytes() const { return impl_->total_bytes; }

void P2PKvRestorePlan::execute(void* k_dst, void* v_dst,
                               std::size_t source_layer_offset_bytes,
                               cudaStream_t stream) const {
  if (impl_->num_pages == 0) return;
  VK_EXPECTS(k_dst != nullptr, "k_dst must be non-null");
  VK_EXPECTS(v_dst != nullptr, "v_dst must be non-null");
  launch_plan_kernel(static_cast<unsigned char*>(k_dst),
                     static_cast<unsigned char*>(v_dst), impl_->d_pages,
                     impl_->num_pages, impl_->page_size, impl_->slot_bytes,
                     impl_->token_stride,
                     static_cast<unsigned long long>(source_layer_offset_bytes),
                     stream);
}

}  // namespace cuda
}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
