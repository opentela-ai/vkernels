// vkernels/comm/p2p_kv_donate.cu — CUDA fused indexed-KV-to-peer donation.
//
// The host reference (p2p_kv_donate.cpp) is the correctness oracle and
// carries the full contract checks. This CUDA path is the performance
// implementation: it validates the run list ONCE on the host (non-negative
// slots, non-null sources/destinations, positive dimensions), resolves
// effective destination pointers, stages the resolved pointer array and
// slot_ids into per-launch device buffers, and launches ONE kernel that
// reads every page's tokens directly from local paged-KV slots and writes
// into the peer destination over NVLink.
//
// The kernel is memory-bound (zero FLOP/byte — pure copy). It reads
// [num_kv_heads, head_dim] per (page, token) from local indexed slots and
// writes K and V into the [page_size, 2, num_kv_heads, head_dim] peer page.
// The key design points:
//
//   * Grid-stride loop over pages with blockDim=256. Each thread copies
//     one uint4 (16 bytes) of K and one of V per iteration, vectorizing
//     the common case where per-slot bytes is a multiple of 16 (always
//     true for head_dim ∈ {64, 128, 256} with elem_size=2). A <16-byte
//     scalar tail path covers the general case.
//   * Destination writes to peer UVA are coalesced within each token (all
//     threads in a block write consecutive 16-byte chunks to the page).
//     Source reads gather from arbitrary slots — repeats are allowed
//     (gather semantics), so no two threads racing is guaranteed only by
//     distinct *destinations*, which the page layout ensures.
//   * One kernel launch replaces the two-stage path's one gather kernel
//     plus N cudaMemcpyAsync per-page writes. The scratch buffer and its
//     local-HBM read/write pass are eliminated entirely (the prepared plan
//     allocates none, ever).
//
// The two-stage GPU reference (p2p_kv_donate_layer_twostage) is provided
// for correctness baselines, scratch-cost measurement, and as the
// copy-engine fallback when direct peer stores are unsupported or lose to
// the copy engine (issue #36).
#include "vkernels/comm/p2p_kv_donate.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/comm/p2p_kv_donate_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <algorithm>
#  include <cstddef>
#  include <cstdint>
#  include <vector>

namespace vkernels::comm {
namespace cuda {
namespace {

// Fixed-width page descriptor uploaded to the device. The resolved
// `dst` pointer is the effective `peer_dst_ptrs[p] + dst_page_offsets[p]`
// computed on the host so the kernel does no per-page pointer arithmetic.
// `slot_base` points to the start of this page's slot_ids sub-array within
// the single device-side slot_ids buffer, so the kernel indexes it as
// `slot_base[t]` for token `t`.
struct PageDev {
  unsigned char* dst;        // peer UVA resolved pointer
  const int* slot_base;      // &slot_ids[p * page_size]
};

// Plan descriptor (issue #36). Unlike PageDev, `dst` is the peer page BASE
// (no per-page offset): the per-layer `dst_offset` is a scalar kernel
// argument added at execute time, so one descriptor buffer is reused across
// every layer. `slot_base` points into either the plan's own device slot map
// (host-input variant) or the caller's `device_indices` buffer
// (from_device_slots variant) at offset `p * page_size`.
struct PagePlanDev {
  unsigned char* dst;
  const int* slot_base;
};

// ---------------------------------------------------------------------------
// Fused indexed-KV-to-peer donation kernel (one-shot)
// ---------------------------------------------------------------------------
//
// Grid: grid-stride over pages (launch_blocks), blockDim = 256.
//
// Each block handles one page per grid-stride iteration. Within a page,
// threads collaboratively copy every token's K+V data:
//   - For token t, slot = slot_ids[t], then:
//     K: k_src + slot*slot_bytes → dst + t*token_stride
//     V: v_src + slot*slot_bytes → dst + t*token_stride + slot_bytes
//   - Thread i copies K[16i : 16i+16) and V[16i : 16i+16) as uint4,
//     with a scalar tail for the <16-byte remainder.
//
// The vec path is only safe when slot_bytes is a multiple of 16: otherwise
// v_src = k_src + slot*slot_bytes and dst_v = dst_k + slot_bytes are
// misaligned, and a 16-byte uint4 load/store from a sub-16-aligned address
// is an illegal memory access on the GPU.
__global__ void p2p_kv_donate_kernel(
    const unsigned char* __restrict__ k_src,
    const unsigned char* __restrict__ v_src,
    const PageDev* __restrict__ pages, int num_pages,
    int page_size, int slot_bytes, int token_stride) {

  for (int p = blockIdx.x; p < num_pages; p += gridDim.x) {
    PageDev page = pages[p];
    unsigned char* __restrict__ dst = page.dst;
    const int* __restrict__ slots = page.slot_base;

    for (int t = 0; t < page_size; ++t) {
      int slot = slots[t];
      const unsigned char* src_k = k_src + static_cast<long long>(slot) * slot_bytes;
      const unsigned char* src_v = v_src + static_cast<long long>(slot) * slot_bytes;
      unsigned char* dst_k = dst + t * token_stride;
      unsigned char* dst_v = dst_k + slot_bytes;

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
// Indexed KV gather kernel (used by the two-stage reference / fallback)
// ---------------------------------------------------------------------------
//
// Copies from indexed local K/V sources into a flat scratch buffer laid out
// [num_pages, page_size, 2, num_kv_heads, head_dim] — the first stage of the
// two-path. Grid-stride over pages; identical thread mapping to the fused
// kernel, but the scratch is contiguous (coalesced writes) while the source
// reads are gathered (may repeat).
__global__ void kv_gather_kernel(unsigned char* __restrict__ scratch,
                                 const unsigned char* __restrict__ k_src,
                                 const unsigned char* __restrict__ v_src,
                                 const int* __restrict__ slot_ids,
                                 int num_pages, int page_size,
                                 int slot_bytes, int token_stride) {
  const int scratch_per_page = page_size * token_stride;
  for (int p = blockIdx.x; p < num_pages; p += gridDim.x) {
    unsigned char* dst = scratch + static_cast<long long>(p) * scratch_per_page;
    const int* slots = slot_ids + p * page_size;

    for (int t = 0; t < page_size; ++t) {
      int slot = slots[t];
      const unsigned char* src_k = k_src + static_cast<long long>(slot) * slot_bytes;
      const unsigned char* src_v = v_src + static_cast<long long>(slot) * slot_bytes;
      unsigned char* dst_k = dst + t * token_stride;
      unsigned char* dst_v = dst_k + slot_bytes;

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

inline int launch_blocks(int num_pages) {
  constexpr int kTargetBlocks = 264;
  return std::min(num_pages, kTargetBlocks);
}

void launch_fused_kernel(const unsigned char* k_src, const unsigned char* v_src,
                         const PageDev* d_pages, int num_pages,
                         int page_size, int slot_bytes, int token_stride,
                         cudaStream_t stream) {
  if (num_pages == 0) return;
  dim3 block(256);
  dim3 grid(launch_blocks(num_pages));
  p2p_kv_donate_kernel<<<grid, block, 0, stream>>>(
      k_src, v_src, d_pages, num_pages, page_size, slot_bytes, token_stride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda p2p_kv_donate launch failed");
}

void launch_gather_kernel(unsigned char* d_scratch, const unsigned char* k_src,
                          const unsigned char* v_src, const int* d_slot_ids,
                          int num_pages, int page_size, int slot_bytes,
                          int token_stride, cudaStream_t stream) {
  if (num_pages == 0) return;
  dim3 block(256);
  dim3 grid(launch_blocks(num_pages));
  kv_gather_kernel<<<grid, block, 0, stream>>>(
      d_scratch, k_src, v_src, d_slot_ids, num_pages, page_size,
      slot_bytes, token_stride);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda kv_gather launch failed");
}

// Public gather entry point (issue #36): one launch over indexed local
// sources into a contiguous [num_pages, page_size, 2, num_kv_heads,
// head_dim] scratch that a caller will then copy to peer, using a
// persistent device slot map. This is the "prepared gather" first stage.
constexpr unsigned kPlanMaxGridAxis = 65535u;

// One-time int64 -> int32 device conversion (issue #36). The int64
// device-slot plan owns its int32 slot map and fills it with this kernel so
// no D2H sync is needed.
__global__ void convert_slots_i64_to_i32_kernel(
    const long long* __restrict__ src, int* __restrict__ dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<int>(src[i]);
}

void launch_convert_i64(const long long* src, int* dst, int n) {
  if (n <= 0) return;
  dim3 block(256);
  unsigned long long grids = (static_cast<unsigned long long>(n) + 255ULL) / 256ULL;
  if (grids > kPlanMaxGridAxis) grids = kPlanMaxGridAxis;
  dim3 grid(static_cast<unsigned>(grids));
  convert_slots_i64_to_i32_kernel<<<grid, block>>>(src, dst, n);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda int64->int32 slot conversion launch failed");
}

// Plan donate kernel (issue #36). One block per (page, token-group) pair:
// grid.x tiles each page's unit range, grid.y = num_pages, blockDim = 256.
// A unit is one chunk of K plus one chunk of V for one token:
//   * vectorized path (slot_bytes % 16 == 0 and dst_offset is 16-aligned):
//     unit = 16 bytes of K + 16 bytes of V (one uint4 each).
//     units_per_token = slot_bytes / 16.
//   * scalar fallback: unit = 1 byte of K + 1 byte of V.
//     units_per_token = slot_bytes. Handles any alignment and still fills
//     the SMs via the page*units grid (vs one block per page).
//
// The unit index u (blockIdx.x*blockDim.x + threadIdx.x) maps to
// (token, chunk) by t = u / units_per_token, chunk = u % units_per_token;
// K is read from k_src + slot*slot_bytes + chunk*unit_bytes and written to
// dst + dst_offset + t*token_stride + chunk*unit_bytes; V is read from
// v_src + slot*slot_bytes + chunk*unit_bytes and written to the same
// destination plus slot_bytes. Slot lookups (pages[p].slot_base[t]) are
// broadcast across the threads of a token, so the small device-side slot
// map is read once per token from L2.
__global__ void p2p_kv_donate_plan_kernel(
    const unsigned char* __restrict__ k_src,
    const unsigned char* __restrict__ v_src,
    const PagePlanDev* __restrict__ pages, int num_pages,
    int page_size, int slot_bytes, int token_stride,
    int units_per_token, int unit_bytes,
    unsigned long long dst_offset) {
  int p = blockIdx.y;
  if (p >= num_pages) return;
  PagePlanDev page = pages[p];
  unsigned char* __restrict__ base = page.dst;
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
  unsigned long long slot_off = static_cast<unsigned long long>(slot) * slot_bytes;
  const unsigned char* src_k = k_src + slot_off + off;
  const unsigned char* src_v = v_src + slot_off + off;
  unsigned char* dst_k = base + dst_offset + static_cast<unsigned long long>(t) * token_stride + off;
  unsigned char* dst_v = dst_k + slot_bytes;
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
void launch_plan_kernel(const unsigned char* k_src, const unsigned char* v_src,
                        const PagePlanDev* d_pages, int num_pages,
                        int page_size, int slot_bytes, int token_stride,
                        unsigned long long dst_offset, cudaStream_t stream) {
  if (num_pages == 0) return;
  const bool aligned = (slot_bytes % 16 == 0) && (dst_offset % 16 == 0);
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
  p2p_kv_donate_plan_kernel<<<grid, block, 0, stream>>>(
      k_src, v_src, d_pages, num_pages, page_size, slot_bytes, token_stride,
      units_per_token, unit_bytes, dst_offset);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda p2p_kv_donate_plan launch failed");
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

void p2p_kv_donate_layer(const void* k_src, const void* v_src,
                         const int* slot_ids,
                         const void* const* peer_dst_ptrs,
                         const std::size_t* dst_page_offsets,
                         std::size_t num_pages, std::size_t page_size,
                         std::size_t num_kv_heads, std::size_t head_dim,
                         std::size_t elem_size,
                         cudaStream_t stream) {
  VK_EXPECTS(num_pages == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || peer_dst_ptrs != nullptr, "peer_dst_ptrs must be non-null");
  VK_EXPECTS(num_pages == 0 || dst_page_offsets != nullptr,
             "dst_page_offsets must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  const int slot_bytes = static_cast<int>(num_kv_heads * head_dim * elem_size);
  const int token_stride = 2 * slot_bytes;

  // Resolve effective destination pointers and build host-side descriptors.
  std::vector<PageDev> pages(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_dst_ptrs[p] != nullptr, "peer_dst_ptrs[p] must be non-null");
    pages[p].dst = static_cast<unsigned char*>(const_cast<void*>(
                          peer_dst_ptrs[p])) +
                   dst_page_offsets[p];
  }

  // Upload slot_ids to the device (one contiguous buffer).
  const std::size_t total_tokens = num_pages * page_size;
  int* d_slot_ids = nullptr;
  cudaError_t err = cudaMallocAsync(&d_slot_ids, total_tokens * sizeof(int), stream);
  VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for slot_ids failed");
  err = cudaMemcpyAsync(d_slot_ids, slot_ids, total_tokens * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
  VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync for slot_ids failed");

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

  // Adaptive dispatch: direct-store kernel by default; copy-engine fallback
  // (gather + per-page copy) when the model prefers it or the mode is forced.
  const std::size_t total_bytes = num_pages * page_size *
                                  static_cast<std::size_t>(token_stride);
  if (vkernels::comm::prefer_direct_store(num_pages, total_bytes)) {
    launch_fused_kernel(static_cast<const unsigned char*>(k_src),
                        static_cast<const unsigned char*>(v_src),
                        d_pages, static_cast<int>(num_pages),
                        static_cast<int>(page_size), slot_bytes, token_stride,
                        stream);
  } else {
    // Copy-engine fallback: gather all pages into a scratch, then one
    // cudaMemcpyAsync per page from scratch to peer.
    unsigned char* d_scratch = nullptr;
    err = cudaMallocAsync(&d_scratch, total_bytes, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMallocAsync for scratch failed");
    launch_gather_kernel(d_scratch, static_cast<const unsigned char*>(k_src),
                         static_cast<const unsigned char*>(v_src), d_slot_ids,
                         static_cast<int>(num_pages),
                         static_cast<int>(page_size), slot_bytes, token_stride,
                         stream);
    for (std::size_t p = 0; p < num_pages; ++p) {
      const std::size_t page_bytes = page_size * static_cast<std::size_t>(token_stride);
      err = cudaMemcpyAsync(pages[p].dst, d_scratch + p * page_bytes,
                            page_bytes, cudaMemcpyDeviceToDevice, stream);
      VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync scratch->peer failed");
    }
    cudaFreeAsync(d_scratch, stream);
  }

  // Free the staging buffers (the kernel/copies are already enqueued).
  cudaFreeAsync(d_slot_ids, stream);
  cudaFreeAsync(d_pages, stream);
}

void p2p_kv_donate_layer_twostage(const void* k_src, const void* v_src,
                                  const int* slot_ids,
                                  const void* const* peer_dst_ptrs,
                                  const std::size_t* dst_page_offsets,
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

  // Stage 1: gather one page from local slots into scratch (one kernel).
  // Stage 2: copy scratch to the peer destination (one cudaMemcpyAsync).
  // Both stages run page-by-page on the SAME stream so the scratch can be
  // reused across pages.
  for (std::size_t p = 0; p < num_pages; ++p) {
    unsigned char* dst = static_cast<unsigned char*>(const_cast<void*>(
                           peer_dst_ptrs[p])) +
                         dst_page_offsets[p];
    // Stage 1: local slots -> scratch.
    launch_gather_kernel(d_scratch, static_cast<const unsigned char*>(k_src),
                         static_cast<const unsigned char*>(v_src),
                         d_slot_ids + p * page_size,
                         1, static_cast<int>(page_size),
                         slot_bytes, token_stride, stream);
    // Stage 2: scratch -> peer.
    err = cudaMemcpyAsync(dst, d_scratch, scratch_per_page,
                          cudaMemcpyDeviceToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync scratch->peer failed");
  }

  cudaFreeAsync(d_scratch, stream);
  cudaFreeAsync(d_slot_ids, stream);
}

// Public indexed-KV gather (issue #36). `slot_ids` is a caller-owned DEVICE
// pointer (shape [num_pages * page_size]); `scratch` is a device buffer of
// [num_pages, page_size, 2, num_kv_heads, head_dim]. One kernel launch, no
// upload, no validation — the caller MUST keep `slot_ids` and the sources
// alive until the kernel completes on `stream`.
void kv_gather(void* scratch, const void* k_src, const void* v_src,
               const int* slot_ids, std::size_t num_pages,
               std::size_t page_size, std::size_t num_kv_heads,
               std::size_t head_dim, std::size_t elem_size,
               cudaStream_t stream) {
  VK_EXPECTS(num_pages == 0 || scratch != nullptr, "scratch must be non-null");
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

  launch_gather_kernel(static_cast<unsigned char*>(scratch),
                       static_cast<const unsigned char*>(k_src),
                       static_cast<const unsigned char*>(v_src), slot_ids,
                       static_cast<int>(num_pages),
                       static_cast<int>(page_size), slot_bytes, token_stride,
                       stream);
}

// ---------------------------------------------------------------------------
// Prepared fused indexed-KV-to-peer donation plan (CUDA, issue #36)
// ---------------------------------------------------------------------------

struct P2PKvDonatePlan::Impl {
  int num_pages;
  int page_size;
  int slot_bytes;
  int token_stride;
  int num_slots;
  int num_kv_heads;
  int head_dim;
  int elem_size;
  std::size_t total_bytes;
  PagePlanDev* d_pages;      // persistent [num_pages] descriptor buffer (device)
  const int* slot_ids;       // device pointer: owned or borrowed
  bool owns_slot_ids;        // true for host-input and int64; false for int32 borrow
  // Resolved peer page BASES kept on the host so execute_via_scratch()
  // can issue its per-page cudaMemcpyAsync without dereferencing the
  // device-only d_pages buffer. The per-layer offset is added at execute
  // time, so these are the un-offset bases.
  std::vector<unsigned char*> host_dsts;

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
P2PKvDonatePlan::Impl* P2PKvDonatePlan::init(
    SlotSource mode, std::size_t num_slots, std::size_t num_kv_heads,
    std::size_t head_dim, std::size_t elem_size, const void* slot_ids,
    const void* const* peer_dst_ptrs, std::size_t num_pages,
    std::size_t page_size) {
  VK_EXPECTS(num_pages == 0 || num_slots > 0, "num_slots must be positive");
  VK_EXPECTS(num_pages == 0 || page_size > 0, "page_size must be positive");
  VK_EXPECTS(num_pages == 0 || num_kv_heads > 0, "num_kv_heads must be positive");
  VK_EXPECTS(num_pages == 0 || head_dim > 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || peer_dst_ptrs != nullptr,
             "peer_dst_ptrs must be non-null");

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
    // Host-input: validate non-negativity and bounds. Uniqueness is NOT
    // required (gather semantics: a repeated slot re-reads the same memory).
    const int* host_slots = static_cast<const int*>(slot_ids);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      int slot = host_slots[i];
      VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
      VK_EXPECTS(slot < static_cast<int>(num_slots),
                 "slot_ids must be < num_slots");
    }
  }

  // Build host-side page descriptors (resolved peer bases; slot base filled
  // after the device slot map is allocated).
  std::vector<PagePlanDev> pages(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_dst_ptrs[p] != nullptr, "peer_dst_ptrs[p] must be non-null");
    pages[p].dst = static_cast<unsigned char*>(const_cast<void*>(
                        peer_dst_ptrs[p]));
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
  impl->host_dsts.resize(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p)
    impl->host_dsts[p] = pages[p].dst;

  return impl;
}

P2PKvDonatePlan::P2PKvDonatePlan(std::size_t num_slots,
                                 std::size_t num_kv_heads,
                                 std::size_t head_dim, std::size_t elem_size,
                                 const int* slot_ids,
                                 const void* const* peer_dst_ptrs,
                                 std::size_t num_pages,
                                 std::size_t page_size)
    : impl_(init(SlotSource::HostValidated, num_slots, num_kv_heads,
                 head_dim, elem_size, slot_ids, peer_dst_ptrs, num_pages,
                 page_size)) {}

P2PKvDonatePlan::P2PKvDonatePlan(vkernels::comm::from_device_slots_t,
                                 std::size_t num_slots,
                                 std::size_t num_kv_heads,
                                 std::size_t head_dim, std::size_t elem_size,
                                 const int* device_indices,
                                 const void* const* peer_dst_ptrs,
                                 std::size_t num_pages,
                                 std::size_t page_size)
    : impl_(init(SlotSource::DeviceInt32Borrowed, num_slots, num_kv_heads,
                 head_dim, elem_size, device_indices, peer_dst_ptrs, num_pages,
                 page_size)) {}

P2PKvDonatePlan::P2PKvDonatePlan(vkernels::comm::from_device_slots_int64_t,
                                 std::size_t num_slots,
                                 std::size_t num_kv_heads,
                                 std::size_t head_dim, std::size_t elem_size,
                                 const std::int64_t* device_indices,
                                 const void* const* peer_dst_ptrs,
                                 std::size_t num_pages,
                                 std::size_t page_size)
    : impl_(init(SlotSource::DeviceInt64Converted, num_slots, num_kv_heads,
                 head_dim, elem_size, device_indices, peer_dst_ptrs, num_pages,
                 page_size)) {}

P2PKvDonatePlan::~P2PKvDonatePlan() {
  if (!impl_) return;
  if (impl_->owns_slot_ids && impl_->slot_ids)
    cudaFree(const_cast<int*>(impl_->slot_ids));
  if (impl_->d_pages) cudaFree(impl_->d_pages);
  delete impl_;
}

std::size_t P2PKvDonatePlan::num_pages() const { return impl_->num_pages; }
std::size_t P2PKvDonatePlan::num_kv_heads() const { return impl_->num_kv_heads; }
std::size_t P2PKvDonatePlan::head_dim() const { return impl_->head_dim; }
std::size_t P2PKvDonatePlan::elem_size() const { return impl_->elem_size; }
std::size_t P2PKvDonatePlan::num_slots() const { return impl_->num_slots; }
std::size_t P2PKvDonatePlan::page_size() const { return impl_->page_size; }
std::size_t P2PKvDonatePlan::total_bytes() const { return impl_->total_bytes; }
std::size_t P2PKvDonatePlan::scratch_bytes() const { return impl_->total_bytes; }

void P2PKvDonatePlan::execute(const void* k_src, const void* v_src,
                              std::size_t destination_layer_offset_bytes,
                              cudaStream_t stream) const {
  if (impl_->num_pages == 0) return;
  VK_EXPECTS(k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(v_src != nullptr, "v_src must be non-null");
  launch_plan_kernel(static_cast<const unsigned char*>(k_src),
                     static_cast<const unsigned char*>(v_src),
                     impl_->d_pages, impl_->num_pages, impl_->page_size,
                     impl_->slot_bytes, impl_->token_stride,
                     static_cast<unsigned long long>(destination_layer_offset_bytes),
                     stream);
}

void P2PKvDonatePlan::execute_via_scratch(const void* k_src, const void* v_src,
                                          void* scratch,
                                          std::size_t destination_layer_offset_bytes,
                                          cudaStream_t stream) const {
  if (impl_->num_pages == 0) return;
  VK_EXPECTS(k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(scratch != nullptr, "scratch must be non-null");
  const int num_pages = impl_->num_pages;
  const int page_size = impl_->page_size;
  const int slot_bytes = impl_->slot_bytes;
  const int token_stride = impl_->token_stride;
  const std::size_t page_bytes =
      static_cast<std::size_t>(page_size) * token_stride;

  // Stage 1: one gather kernel (all pages) into the caller-owned scratch.
  launch_gather_kernel(static_cast<unsigned char*>(scratch),
                       static_cast<const unsigned char*>(k_src),
                       static_cast<const unsigned char*>(v_src),
                       impl_->slot_ids, num_pages, page_size, slot_bytes,
                       token_stride, stream);
  // Stage 2: one cudaMemcpyAsync per page from scratch to peer. The
  // destination bases live on the host (impl_->host_dsts); only the page
  // descriptors uploaded to the kernel live on the device.
  for (int p = 0; p < num_pages; ++p) {
    unsigned char* dst = impl_->host_dsts[p] +
                         static_cast<unsigned long long>(destination_layer_offset_bytes);
    cudaError_t err = cudaMemcpyAsync(dst,
                       static_cast<unsigned char*>(scratch) + p * page_bytes,
                       page_bytes, cudaMemcpyDeviceToDevice, stream);
    VK_ENSURES(err == cudaSuccess, "cudaMemcpyAsync scratch->peer failed");
  }
}

}  // namespace cuda
}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
