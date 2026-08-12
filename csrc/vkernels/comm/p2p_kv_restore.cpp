// vkernels/comm/p2p_kv_restore.cpp — host reference (oracle) implementation.
//
// The CPU reference is the correctness oracle: it is always compiled, fully
// unit-tested, and carries the contract checks. The CUDA path (p2p_kv_restore.cu)
// performs the same fused peer-read + indexed-scatter in a single kernel and
// trusts the already-validated metadata so the hot path stays free of checks.
#include "vkernels/comm/p2p_kv_restore.hpp"

#include "vkernels/util/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace vkernels::comm {
namespace {

// Byte stride per destination slot (K and V are the same size).
inline std::size_t per_slot_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size) {
  return num_kv_heads * head_dim * elem_size;
}

// Byte stride per token in the source: K + V for all heads.
inline std::size_t token_stride_bytes(std::size_t num_kv_heads, std::size_t head_dim,
                                      std::size_t elem_size) {
  return 2 * per_slot_bytes(num_kv_heads, head_dim, elem_size);
}

// Scatter K/V for one page from a flat scratch buffer to indexed slots.
// `scratch` is laid out as [page_size, 2, num_kv_heads, head_dim] — same as
// the peer source. This is the second stage of the two-stage reference.
void scatter_kv_from_scratch(const std::uint8_t* scratch, std::uint8_t* k_dst,
                             std::uint8_t* v_dst, const int* slot_ids,
                             std::size_t page_size, std::size_t slot_bytes) {
  const std::size_t token_stride = 2 * slot_bytes;  // K+V per token
  for (std::size_t t = 0; t < page_size; ++t) {
    int slot = slot_ids[t];
    const std::uint8_t* src_k = scratch + t * token_stride;
    const std::uint8_t* src_v = src_k + slot_bytes;
    std::memcpy(k_dst + slot * slot_bytes, src_k, slot_bytes);
    std::memcpy(v_dst + slot * slot_bytes, src_v, slot_bytes);
  }
}

// Perform the fused copy for one page: peer UVA → indexed K/V destinations.
// Same as scatter_kv_from_scratch but reads from a peer UVA pointer instead
// of a local scratch buffer.
void fused_copy_one_page(const std::uint8_t* page_base, std::uint8_t* k_dst,
                         std::uint8_t* v_dst, const int* slot_ids,
                         std::size_t page_size, std::size_t slot_bytes) {
  const std::size_t token_stride = 2 * slot_bytes;
  for (std::size_t t = 0; t < page_size; ++t) {
    int slot = slot_ids[t];
    const std::uint8_t* src_k = page_base + t * token_stride;
    const std::uint8_t* src_v = src_k + slot_bytes;
    std::memcpy(k_dst + slot * slot_bytes, src_k, slot_bytes);
    std::memcpy(v_dst + slot * slot_bytes, src_v, slot_bytes);
  }
}

}  // namespace

void p2p_kv_restore_layer(void* k_dst, void* v_dst,
                          const int* slot_ids,
                          const void* const* peer_src_ptrs,
                          const std::size_t* src_page_offsets,
                          std::size_t num_pages, std::size_t page_size,
                          std::size_t num_kv_heads, std::size_t head_dim,
                          std::size_t elem_size,
                          Stream* stream) {
  // Validate arguments on the host (the CUDA kernel trusts these).
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

  // Destination slots must be unique. We check this once on the host; the
  // CUDA kernel relies on it for correctness (no two pages write the same
  // destination slot).
  const std::size_t total_tokens = num_pages * page_size;
  {
    std::unordered_set<int> seen;
    seen.reserve(total_tokens);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      int slot = slot_ids[i];
      VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
      VK_EXPECTS(seen.insert(slot).second, "slot_ids must be unique");
    }
  }

  const std::size_t slot_bytes = per_slot_bytes(num_kv_heads, head_dim, elem_size);
  auto* k = static_cast<std::uint8_t*>(k_dst);
  auto* v = static_cast<std::uint8_t*>(v_dst);

  if (stream == nullptr) {
    // Synchronous path.
    for (std::size_t p = 0; p < num_pages; ++p) {
      VK_EXPECTS(peer_src_ptrs[p] != nullptr,
                 "peer_src_ptrs[p] must be non-null");
      const auto* page_base = static_cast<const std::uint8_t*>(peer_src_ptrs[p]) +
                              src_page_offsets[p];
      fused_copy_one_page(page_base, k, v, slot_ids + p * page_size,
                          page_size, slot_bytes);
    }
    return;
  }

  // Async path: capture validated metadata, enqueue one task.
  // Resolve the effective page pointers now so the task captures by value.
  std::vector<const std::uint8_t*> resolved(num_pages);
  for (std::size_t p = 0; p < num_pages; ++p) {
    VK_EXPECTS(peer_src_ptrs[p] != nullptr,
               "peer_src_ptrs[p] must be non-null");
    resolved[p] = static_cast<const std::uint8_t*>(peer_src_ptrs[p]) +
                  src_page_offsets[p];
  }
  // Capture slot_ids by copying into a vector (the caller may free the
  // original array as soon as this function returns).
  std::vector<int> slots_copy(slot_ids, slot_ids + total_tokens);

  stream->submit([k, v, slots = std::move(slots_copy), pages = std::move(resolved),
                  num_pages, page_size, slot_bytes]() {
    for (std::size_t p = 0; p < num_pages; ++p) {
      fused_copy_one_page(pages[p], k, v, slots.data() + p * page_size,
                          page_size, slot_bytes);
    }
  });
}

void p2p_kv_restore_layer_twostage(void* k_dst, void* v_dst,
                                   const int* slot_ids,
                                   const void* const* peer_src_ptrs,
                                   const std::size_t* src_page_offsets,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   Stream* stream) {
  if (num_pages == 0) return;

  const std::size_t slot_bytes = per_slot_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t tkn_stride = token_stride_bytes(num_kv_heads, head_dim, elem_size);
  const std::size_t scratch_per_page = page_size * tkn_stride;
  auto* k = static_cast<std::uint8_t*>(k_dst);
  auto* v = static_cast<std::uint8_t*>(v_dst);

  if (stream == nullptr) {
    std::vector<std::uint8_t> scratch(scratch_per_page);
    for (std::size_t p = 0; p < num_pages; ++p) {
      const auto* src = static_cast<const std::uint8_t*>(peer_src_ptrs[p]) +
                        src_page_offsets[p];
      std::memcpy(scratch.data(), src, scratch_per_page);
      scatter_kv_from_scratch(scratch.data(), k, v, slot_ids + p * page_size,
                              page_size, slot_bytes);
    }
    return;
  }

  // Async: each page is one stream task (peer → scratch → scatter).
  for (std::size_t p = 0; p < num_pages; ++p) {
    const auto* src = static_cast<const std::uint8_t*>(peer_src_ptrs[p]) +
                      src_page_offsets[p];
    const int* page_slots = slot_ids + p * page_size;
    stream->submit([k, v, src, page_slots, page_size, slot_bytes, scratch_per_page]() {
      std::vector<std::uint8_t> scratch(scratch_per_page);
      std::memcpy(scratch.data(), src, scratch_per_page);
      scatter_kv_from_scratch(scratch.data(), k, v, page_slots, page_size, slot_bytes);
    });
  }
}

}  // namespace vkernels::comm
