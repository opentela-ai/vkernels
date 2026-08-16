// vkernels/comm/p2p_kv_donate.cpp
//
// Host reference for the fused indexed-KV-to-peer donation kernel (issue
// #36). Always compiled and fully unit-tested; it is the byte-exact oracle
// for both the direct-store CUDA kernel and the copy-engine fallback. The
// data flow is the mirror of p2p_kv_restore.cpp: the restore reads peer
// pages and scatters into indexed local slots, the donate reads indexed
// local slots and gathers into peer pages.
//
//   restore: peer_page[p][l, t, 0|1, h, d]  -->  k_dst[slot[p,t], h, d]
//                                                   v_dst[slot[p,t], h, d]
//   donate:  k_src[slot[p,t], h, d]  -->  peer_dst[p] + l*layer_bytes
//                                           + t*token_stride + (0|1)*slot_bytes
//           v_src[slot[p,t], h, d]  -->  ... + slot_bytes
//
// No per-call allocations on the fused path; the two-stage reference
// allocates a per-page scratch (the documented copy-engine fallback shape).

#include "vkernels/comm/p2p_kv_donate.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <utility>
#include <vector>

#include "vkernels/util/annotations.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels::comm {

namespace {

inline std::size_t per_slot_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size) {
  return num_kv_heads * head_dim * elem_size;
}

inline std::size_t token_stride_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                      std::size_t elem_size) {
  return 2 * per_slot_bytes(num_kv_heads, head_dim, elem_size);
}

// Read local paged-KV slots and write K/V directly into one peer page.
// `page_base` is the effective peer destination (peer_dst_ptrs[p] +
// offset). For each token `t` in the page, slot = slot_ids[t]; the source K
// is at k_src + slot * slot_bytes and the source V at v_src + slot *
// slot_bytes; the destination K is at page_base + t * token_stride and the
// destination V at page_base + t * token_stride + slot_bytes.
inline void fused_donate_page(std::uint8_t* page_base,
                              const std::uint8_t* k_src,
                              const std::uint8_t* v_src,
                              const int* slot_ids, std::size_t page_size,
                              std::size_t slot_bytes, std::size_t token_stride) {
  for (std::size_t t = 0; t < page_size; ++t) {
    const int slot = slot_ids[t];
    const std::size_t src_off = static_cast<std::size_t>(slot) * slot_bytes;
    const std::size_t dst_off = t * token_stride;
    std::memcpy(page_base + dst_off, k_src + src_off, slot_bytes);
    std::memcpy(page_base + dst_off + slot_bytes, v_src + src_off, slot_bytes);
  }
}

// Gather local paged-KV slots into one contiguous scratch page laid out
// [page_size, 2, num_kv_heads, head_dim] (the first stage of the two-path).
inline void gather_kv_to_scratch(std::uint8_t* scratch_page,
                                 const std::uint8_t* k_src,
                                 const std::uint8_t* v_src,
                                 const int* slot_ids, std::size_t page_size,
                                 std::size_t slot_bytes,
                                 std::size_t token_stride) {
  for (std::size_t t = 0; t < page_size; ++t) {
    const int slot = slot_ids[t];
    const std::size_t src_off = static_cast<std::size_t>(slot) * slot_bytes;
    const std::size_t dst_off = t * token_stride;
    std::memcpy(scratch_page + dst_off, k_src + src_off, slot_bytes);
    std::memcpy(scratch_page + dst_off + slot_bytes, v_src + src_off,
                slot_bytes);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// One-shot primitives
// ---------------------------------------------------------------------------

void p2p_kv_donate_layer(const void* k_src, const void* v_src,
                         const int* slot_ids,
                         const void* const* peer_dst_ptrs,
                         const std::size_t* dst_page_offsets,
                         std::size_t num_pages, std::size_t page_size,
                         std::size_t num_kv_heads, std::size_t head_dim,
                         std::size_t elem_size,
                         Stream* stream) {
  // Validate arguments on the host (the CUDA kernel trusts these).
  VK_EXPECTS(num_pages == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || peer_dst_ptrs != nullptr,
             "peer_dst_ptrs must be non-null");
  VK_EXPECTS(num_pages == 0 || dst_page_offsets != nullptr,
             "dst_page_offsets must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  // Source slots must be non-negative. Unlike the restore (which scatters
  // and therefore requires UNIQUE destination slots), the donate gathers:
  // a repeated source slot simply re-reads the same memory, so uniqueness
  // is NOT required (and not checked). The one-shot does not take
  // `num_slots`; the caller guarantees the source buffers are large enough.
  const std::size_t total_tokens = num_pages * page_size;
  for (std::size_t i = 0; i < total_tokens; ++i)
    VK_EXPECTS(slot_ids[i] >= 0, "slot_ids must be non-negative");

  const std::size_t slot_bytes = per_slot_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t token_stride = token_stride_bytes(num_kv_heads, head_dim, elem_size);
  const auto* k = static_cast<const std::uint8_t*>(k_src);
  const auto* v = static_cast<const std::uint8_t*>(v_src);

  if (stream == nullptr) {
    // Synchronous path: one fused pass over all pages, no scratch.
    for (std::size_t p = 0; p < num_pages; ++p) {
      VK_EXPECTS(peer_dst_ptrs[p] != nullptr,
                 "peer_dst_ptrs[p] must be non-null");
      auto* page_base = static_cast<std::uint8_t*>(const_cast<void*>(
                          peer_dst_ptrs[p])) +
                        dst_page_offsets[p];
      fused_donate_page(page_base, k, v, slot_ids + p * page_size,
                        page_size, slot_bytes, token_stride);
    }
    return;
  }

  // Async path: capture validated metadata, enqueue one task. Resolve the
  // effective page pointers now so the task captures by value.
  std::vector<std::uint8_t*> resolved(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_dst_ptrs[p] != nullptr,
               "peer_dst_ptrs[p] must be non-null");
    resolved[p] = static_cast<std::uint8_t*>(const_cast<void*>(
                    peer_dst_ptrs[p])) +
                  dst_page_offsets[p];
  }
  // Capture slot_ids by copying into a vector (the caller may free the
  // original array as soon as this function returns).
  std::vector<int> slots_copy(slot_ids, slot_ids + total_tokens);

  stream->submit([k, v, slots = std::move(slots_copy), pages = std::move(resolved),
                  num_pages, page_size, slot_bytes, token_stride]() {
    for (std::size_t p = 0; p < num_pages; ++p)
      fused_donate_page(pages[p], k, v, slots.data() + p * page_size,
                        page_size, slot_bytes, token_stride);
  });
}

void p2p_kv_donate_layer_twostage(const void* k_src, const void* v_src,
                                  const int* slot_ids,
                                  const void* const* peer_dst_ptrs,
                                  const std::size_t* dst_page_offsets,
                                  std::size_t num_pages, std::size_t page_size,
                                  std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size,
                                  Stream* stream) {
  VK_EXPECTS(num_pages == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(num_pages == 0 || peer_dst_ptrs != nullptr,
             "peer_dst_ptrs must be non-null");
  VK_EXPECTS(num_pages == 0 || dst_page_offsets != nullptr,
             "dst_page_offsets must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  for (std::size_t i = 0, n = num_pages * page_size; i < n; ++i)
    VK_EXPECTS(slot_ids[i] >= 0, "slot_ids must be non-negative");

  const std::size_t slot_bytes = per_slot_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t token_stride = token_stride_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t scratch_per_page = page_size * token_stride;
  const auto* k = static_cast<const std::uint8_t*>(k_src);
  const auto* v = static_cast<const std::uint8_t*>(v_src);

  if (stream == nullptr) {
    std::vector<std::uint8_t> scratch(scratch_per_page);
    for (std::size_t p = 0; p < num_pages; ++p) {
      VK_EXPECTS(peer_dst_ptrs[p] != nullptr,
                 "peer_dst_ptrs[p] must be non-null");
      gather_kv_to_scratch(scratch.data(), k, v, slot_ids + p * page_size,
                           page_size, slot_bytes, token_stride);
      auto* dst = static_cast<std::uint8_t*>(const_cast<void*>(
                     peer_dst_ptrs[p])) +
                  dst_page_offsets[p];
      std::memcpy(dst, scratch.data(), scratch_per_page);
    }
    return;
  }

  // Async: each page is one stream task (gather -> scratch -> peer copy).
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_dst_ptrs[p] != nullptr,
               "peer_dst_ptrs[p] must be non-null");
    const int* page_slots = slot_ids + p * page_size;
    std::uint8_t* dst = static_cast<std::uint8_t*>(const_cast<void*>(
                          peer_dst_ptrs[p])) +
                        dst_page_offsets[p];
    stream->submit([k, v, page_slots, page_size, slot_bytes, token_stride,
                    scratch_per_page, dst]() {
      std::vector<std::uint8_t> scratch(scratch_per_page);
      gather_kv_to_scratch(scratch.data(), k, v, page_slots, page_size,
                           slot_bytes, token_stride);
      std::memcpy(dst, scratch.data(), scratch_per_page);
    });
  }
}

void kv_gather(void* scratch, const void* k_src, const void* v_src,
               const int* slot_ids, std::size_t num_pages,
               std::size_t page_size, std::size_t num_kv_heads,
               std::size_t head_dim, std::size_t elem_size,
               Stream* stream) {
  VK_EXPECTS(num_pages == 0 || scratch != nullptr, "scratch must be non-null");
  VK_EXPECTS(num_pages == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages == 0 || slot_ids != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");

  if (num_pages == 0) return;

  for (std::size_t i = 0, n = num_pages * page_size; i < n; ++i)
    VK_EXPECTS(slot_ids[i] >= 0, "slot_ids must be non-negative");

  const std::size_t slot_bytes = per_slot_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t token_stride = token_stride_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t scratch_per_page = page_size * token_stride;
  const auto* k = static_cast<const std::uint8_t*>(k_src);
  const auto* v = static_cast<const std::uint8_t*>(v_src);
  auto* base = static_cast<std::uint8_t*>(scratch);

  if (stream == nullptr) {
    for (std::size_t p = 0; p < num_pages; ++p)
      gather_kv_to_scratch(base + p * scratch_per_page, k, v,
                           slot_ids + p * page_size, page_size, slot_bytes,
                           token_stride);
    return;
  }

  // Capture the (unchanging) pointers and sizes; copy the slot map so the
  // caller may free it as soon as this returns. One task gathers all pages
  // (the scratch is contiguous), matching the fused restore's one task.
  std::vector<int> slots_copy(slot_ids, slot_ids + num_pages * page_size);
  stream->submit([base, k, v, slots = std::move(slots_copy), num_pages,
                  page_size, slot_bytes, token_stride, scratch_per_page]() {
    for (std::size_t p = 0; p < num_pages; ++p)
      gather_kv_to_scratch(base + p * scratch_per_page, k, v,
                           slots.data() + p * page_size, page_size, slot_bytes,
                           token_stride);
  });
}

// ---------------------------------------------------------------------------
// Adaptive dispatch (host-testable pure functions)
// ---------------------------------------------------------------------------

namespace {

// Runtime-tunable dispatch state. Atomically read by the CUDA path
// (p2p_kv_donate.cu) and written by set_donate_dispatch.
std::atomic<unsigned> g_dispatch_mode(static_cast<unsigned>(DonateDispatchMode::kAdaptive));
std::atomic<std::size_t> g_dispatch_min_pages(1);

// Cost-model constants fitted to H100 NVL measurements of the restore's
// mirror path (sgs-gpu07, CUDA 13 / driver 580.82.07, real NVLink peer
// writes GPU0->GPU1). The direct-store kernel writes peer memory from SMs
// over NVLink at ~4.20 us/MiB with a ~8.6 us launch floor (matching the
// restore's gather kernel, which reads the same link at the same rate);
// the copy-engine fallback pays the gather kernel PLUS one
// cudaMemcpyAsync per page (~201.7 us for 48 MiB in one call ≈ 4.20 us/MiB,
// plus ~7.37 us per extra page, with a ~20 us per-call floor). Because the
// fallback runs the gather kernel regardless, the direct store wins from
// one page; the model keeps the explicit fallback available for systems
// where direct peer stores are unsupported.
constexpr double kKernelPerMiBUs = 4.20;
constexpr double kKernelFixedUs = 8.6;
constexpr double kCopyPerMiBUs = 4.20;
constexpr double kCopyFixedUs = 20.0;
constexpr double kCopyPerPageUs = 7.37;

}  // namespace

void set_donate_dispatch(DonateDispatchMode mode, std::size_t min_pages_for_direct) {
  g_dispatch_mode.store(static_cast<unsigned>(mode));
  g_dispatch_min_pages.store(min_pages_for_direct);
}

std::pair<DonateDispatchMode, std::size_t> donate_dispatch_config() {
  return {static_cast<DonateDispatchMode>(g_dispatch_mode.load()),
          g_dispatch_min_pages.load()};
}

double est_direct_store_us(std::size_t num_pages, std::size_t total_bytes) {
  (void)num_pages;  // flat in page count below the grid cap
  if (total_bytes == 0) return 0.0;  // nothing to write: never pay a launch
  const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return std::max(kKernelFixedUs, kKernelPerMiBUs * mib);
}

double est_copy_engine_donate_us(std::size_t num_pages, std::size_t total_bytes) {
  if (total_bytes == 0) return 0.0;  // nothing to write
  const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  const double pages = static_cast<double>(num_pages);
  // One gather kernel (no NVLink) plus per-page peer copies.
  const double gather = std::max(kKernelFixedUs, kKernelPerMiBUs * mib);
  const double one_copy = std::max(kCopyFixedUs, kCopyPerMiBUs * mib);
  return gather + one_copy + kCopyPerPageUs * (pages > 1.0 ? pages - 1.0 : 0.0);
}

bool prefer_direct_store(std::size_t num_pages, std::size_t total_bytes) {
  if (total_bytes == 0) return false;  // nothing to write: never pay a launch
  switch (static_cast<DonateDispatchMode>(g_dispatch_mode.load())) {
    case DonateDispatchMode::kForceDirect: return true;
    case DonateDispatchMode::kForceCopyEngine: return false;
    case DonateDispatchMode::kAdaptive: break;
  }
  if (num_pages < g_dispatch_min_pages.load()) return false;
  return est_direct_store_us(num_pages, total_bytes) <
         est_copy_engine_donate_us(num_pages, total_bytes);
}

// ---------------------------------------------------------------------------
// Prepared plan (host reference)
// ---------------------------------------------------------------------------

void P2PKvDonatePlan::validate_shape(const void* const* peer_dst_ptrs) const {
  VK_EXPECTS(num_slots_ > 0 || num_pages_ == 0, "num_slots must be positive");
  VK_EXPECTS(page_size_ > 0 || num_pages_ == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads_ > 0 || num_pages_ == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim_ > 0 || num_pages_ == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size_ == 2, "elem_size must be 2 for BF16/FP16");
  if (num_pages_ == 0) return;
  VK_EXPECTS(slot_ids_ != nullptr, "slot_ids must be non-null");
  VK_EXPECTS(peer_dst_ptrs != nullptr, "peer_dst_ptrs must be non-null");
  for (std::size_t p = 0; p < num_pages_; ++p)
    VK_EXPECTS(peer_dst_ptrs[p] != nullptr, "peer_dst_ptrs[p] must be non-null");
}

P2PKvDonatePlan::P2PKvDonatePlan(std::size_t num_slots,
                                 std::size_t num_kv_heads,
                                 std::size_t head_dim, std::size_t elem_size,
                                 const int* slot_ids,
                                 const void* const* peer_dst_ptrs,
                                 std::size_t num_pages,
                                 std::size_t page_size)
    : num_slots_(num_slots),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim), elem_size_(elem_size),
      page_size_(page_size), num_pages_(num_pages),
      slot_bytes_(per_slot_bytes(num_kv_heads, head_dim, elem_size)),
      token_stride_(token_stride_bytes(num_kv_heads, head_dim, elem_size)),
      layer_bytes_(page_size * token_stride_bytes(num_kv_heads, head_dim, elem_size)),
      total_bytes_(0), scratch_bytes_(0), slot_ids_(slot_ids) {
  validate_shape(peer_dst_ptrs);
  if (num_pages_ == 0) return;

  // Validate slot contents once (non-negativity, bounds). Unlike the
  // restore, uniqueness is NOT required -- the donate gathers, so a
  // repeated source slot simply re-reads the same memory.
  const std::size_t total_tokens = num_pages_ * page_size_;
  for (std::size_t i = 0; i < total_tokens; ++i) {
    int slot = slot_ids[i];
    VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
    VK_EXPECTS(static_cast<std::size_t>(slot) < num_slots_,
               "slot_ids must be < num_slots");
  }

  // Own the metadata: peer bases + slot map. The caller may free the
  // original arrays as soon as the constructor returns.
  peer_bases_.assign(const_cast<void**>(peer_dst_ptrs),
                     const_cast<void**>(peer_dst_ptrs) + num_pages_);
  owned_slots_.assign(slot_ids, slot_ids + total_tokens);
  slot_ids_ = owned_slots_.data();
  total_bytes_ = num_pages_ * layer_bytes_;
  scratch_bytes_ = total_bytes_;
}

P2PKvDonatePlan::P2PKvDonatePlan(from_device_slots_t, std::size_t num_slots,
                                 std::size_t num_kv_heads,
                                 std::size_t head_dim, std::size_t elem_size,
                                 const int* device_indices,
                                 const void* const* peer_dst_ptrs,
                                 std::size_t num_pages,
                                 std::size_t page_size)
    : num_slots_(num_slots),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim), elem_size_(elem_size),
      page_size_(page_size), num_pages_(num_pages),
      slot_bytes_(per_slot_bytes(num_kv_heads, head_dim, elem_size)),
      token_stride_(token_stride_bytes(num_kv_heads, head_dim, elem_size)),
      layer_bytes_(page_size * token_stride_bytes(num_kv_heads, head_dim, elem_size)),
      total_bytes_(0), scratch_bytes_(0), slot_ids_(device_indices) {
  validate_shape(peer_dst_ptrs);
  if (num_pages_ == 0) {
    slot_ids_ = nullptr;
    return;
  }
  // The slot contents are on the device and are NOT validated here -- the
  // caller guarantees non-negative, in-range slots (repeats allowed) and
  // keeps `device_indices` alive until the plan is destroyed. Only peer
  // bases are copied (the slot pointer is borrowed).
  peer_bases_.assign(const_cast<void**>(peer_dst_ptrs),
                     const_cast<void**>(peer_dst_ptrs) + num_pages_);
  total_bytes_ = num_pages_ * layer_bytes_;
  scratch_bytes_ = total_bytes_;
}

// The int64 device-slot constructor (host reference): convert the caller's
// int64 indices to an OWNED int32 array once, mirroring the CUDA plan's
// device-side int64->int32 conversion. Like the int32 device-slot variant it
// does NOT validate slot contents, so the caller guarantees non-negative,
// in-range slots. The caller may free the int64 buffer as soon as the
// constructor returns (the host reference owns its int32 copy).
P2PKvDonatePlan::P2PKvDonatePlan(from_device_slots_int64_t,
                                 std::size_t num_slots,
                                 std::size_t num_kv_heads,
                                 std::size_t head_dim, std::size_t elem_size,
                                 const std::int64_t* device_indices,
                                 const void* const* peer_dst_ptrs,
                                 std::size_t num_pages,
                                 std::size_t page_size)
    : num_slots_(num_slots),
      num_kv_heads_(num_kv_heads), head_dim_(head_dim), elem_size_(elem_size),
      page_size_(page_size), num_pages_(num_pages),
      slot_bytes_(per_slot_bytes(num_kv_heads, head_dim, elem_size)),
      token_stride_(token_stride_bytes(num_kv_heads, head_dim, elem_size)),
      layer_bytes_(page_size * token_stride_bytes(num_kv_heads, head_dim, elem_size)),
      total_bytes_(0), scratch_bytes_(0), slot_ids_(nullptr) {
  // Convert the caller's int64 indices to an OWNED int32 array first so
  // slot_ids_ is non-null for validate_shape (mirrors the int32 device-slot
  // constructor borrowing the pointer).
  const std::size_t total_tokens = num_pages_ * page_size_;
  if (total_tokens > 0) {
    owned_slots_.resize(total_tokens);
    for (std::size_t i = 0; i < total_tokens; ++i)
      owned_slots_[i] = static_cast<int>(device_indices[i]);
    slot_ids_ = owned_slots_.data();
  }
  validate_shape(peer_dst_ptrs);
  if (num_pages_ == 0) return;

  peer_bases_.assign(const_cast<void**>(peer_dst_ptrs),
                     const_cast<void**>(peer_dst_ptrs) + num_pages_);
  total_bytes_ = num_pages_ * layer_bytes_;
  scratch_bytes_ = total_bytes_;
}

void P2PKvDonatePlan::execute(const void* k_src, const void* v_src,
                              std::size_t destination_layer_offset_bytes,
                              Stream* stream) const {
  VK_EXPECTS(num_pages_ == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages_ == 0 || v_src != nullptr, "v_src must be non-null");
  if (num_pages_ == 0) return;  // valid no-op plan

  const auto* k = static_cast<const std::uint8_t*>(k_src);
  const auto* v = static_cast<const std::uint8_t*>(v_src);
  const P2PKvDonatePlan* self = this;
  auto do_all = [self, k, v, destination_layer_offset_bytes] {
    const std::size_t sb = self->slot_bytes_;
    const std::size_t ts = self->token_stride_;
    const std::size_t ps = self->page_size_;
    const std::size_t np = self->num_pages_;
    for (std::size_t p = 0; p < np; ++p) {
      auto* page_base = static_cast<std::uint8_t*>(self->peer_bases_[p]) +
                        destination_layer_offset_bytes;
      fused_donate_page(page_base, k, v, self->slot_ids_ + p * ps,
                        ps, sb, ts);
    }
  };
  if (stream == nullptr) {
    do_all();
    return;
  }
  stream->submit(std::move(do_all));
}

void P2PKvDonatePlan::execute_via_scratch(const void* k_src, const void* v_src,
                                          void* scratch,
                                          std::size_t destination_layer_offset_bytes,
                                          Stream* stream) const {
  VK_EXPECTS(num_pages_ == 0 || k_src != nullptr, "k_src must be non-null");
  VK_EXPECTS(num_pages_ == 0 || v_src != nullptr, "v_src must be non-null");
  VK_EXPECTS(num_pages_ == 0 || scratch != nullptr, "scratch must be non-null");
  if (num_pages_ == 0) return;  // valid no-op plan

  const auto* k = static_cast<const std::uint8_t*>(k_src);
  const auto* v = static_cast<const std::uint8_t*>(v_src);
  auto* base = static_cast<std::uint8_t*>(scratch);
  const P2PKvDonatePlan* self = this;
  auto do_all = [self, k, v, base, destination_layer_offset_bytes] {
    const std::size_t sb = self->slot_bytes_;
    const std::size_t ts = self->token_stride_;
    const std::size_t ps = self->page_size_;
    const std::size_t np = self->num_pages_;
    const std::size_t spp = ps * ts;  // scratch_per_page == layer_bytes_
    for (std::size_t p = 0; p < np; ++p) {
      gather_kv_to_scratch(base + p * spp, k, v, self->slot_ids_ + p * ps,
                           ps, sb, ts);
      auto* dst = static_cast<std::uint8_t*>(self->peer_bases_[p]) +
                  destination_layer_offset_bytes;
      std::memcpy(dst, base + p * spp, spp);
    }
  };
  if (stream == nullptr) {
    do_all();
    return;
  }
  stream->submit(std::move(do_all));
}

}  // namespace vkernels::comm
