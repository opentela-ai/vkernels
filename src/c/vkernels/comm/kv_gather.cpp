// vkernels/comm/kv_gather.cpp — host reference (oracle) implementation.
//
// The CPU reference is the correctness oracle: it is always compiled, fully
// unit-tested, and carries the contract checks (null pointers, positive
// dimensions, BF16/FP16 element size, non-negative and in-range source
// slots). The CUDA path (kv_gather.cu) performs the same fused indexed
// gather in a single kernel and trusts the already-validated metadata so
// the hot path stays free of checks.
//
// The gather is memory-bound (zero FLOP/byte — pure copy). For each token
// `t` of page `p` it reads K and V from source slot `slot_ids[p*page_size+t]`
// and writes them into the packed [num_pages, page_size, 2, ...] destination,
// matching the PyTorch two-gather reference byte-for-byte:
//
//     dst[:, :, 0] = k_src[slot_ids]
//     dst[:, :, 1] = v_src[slot_ids]
#include "vkernels/comm/kv_gather.hpp"

#include "vkernels/util/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace vkernels::comm {
namespace {

// Read one slot id, honouring the int32/int64 selector. Validates
// non-negativity and bounds against `num_slots`.
inline void validate_slot(std::int64_t slot, std::size_t num_slots) {
  VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
  VK_EXPECTS(static_cast<std::uint64_t>(slot) <
                 static_cast<std::uint64_t>(num_slots),
             "slot_ids must be < num_slots");
}

// Byte stride of one slot element (int32 or int64).
inline std::size_t slot_elem_size(bool slot_ids_int64) {
  return slot_ids_int64 ? sizeof(std::int64_t) : sizeof(int);
}

// Pointer to the first slot of page `p` (element stride, not byte stride).
inline const void* page_slots(const void* slot_ids, std::size_t p,
                              std::size_t page_size, bool slot_ids_int64) {
  return static_cast<const char*>(slot_ids) +
         p * page_size * slot_elem_size(slot_ids_int64);
}

// Gather K/V for one page from indexed sources into the packed destination.
// `dst_page` is the start of page `p` in `dst` (page_bytes each).
// `slots` is &slot_ids[p * page_size]. `slot_bytes` is the per-slot extent;
// `token_stride` is 2 * slot_bytes (the [K, V] interleave per token).
inline void gather_page(std::uint8_t* dst_page, const std::uint8_t* k_src,
                        const std::uint8_t* v_src, const void* slots,
                        bool slot_ids_int64, std::size_t page_size,
                        std::size_t slot_bytes, std::size_t token_stride) {
  if (slot_ids_int64) {
    const auto* s = static_cast<const std::int64_t*>(slots);
    for (std::size_t t = 0; t < page_size; ++t) {
      const std::size_t slot = static_cast<std::size_t>(s[t]);
      const std::size_t src_off = slot * slot_bytes;
      const std::size_t dst_off = t * token_stride;
      std::memcpy(dst_page + dst_off, k_src + src_off, slot_bytes);
      std::memcpy(dst_page + dst_off + slot_bytes, v_src + src_off, slot_bytes);
    }
  } else {
    const auto* s = static_cast<const int*>(slots);
    for (std::size_t t = 0; t < page_size; ++t) {
      const std::size_t slot = static_cast<std::size_t>(s[t]);
      const std::size_t src_off = slot * slot_bytes;
      const std::size_t dst_off = t * token_stride;
      std::memcpy(dst_page + dst_off, k_src + src_off, slot_bytes);
      std::memcpy(dst_page + dst_off + slot_bytes, v_src + src_off, slot_bytes);
    }
  }
}

}  // namespace

void kv_gather_layer(void* dst,
                     const void* k_src, const void* v_src,
                     const void* slot_ids, bool slot_ids_int64,
                     std::size_t num_slots,
                     std::size_t num_pages, std::size_t page_size,
                     std::size_t num_kv_heads, std::size_t head_dim,
                     std::size_t elem_size,
                     Stream* stream) {
  // Validate arguments on the host (the CUDA kernel trusts these).
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
    for (std::size_t i = 0; i < total_tokens; ++i)
      validate_slot(s[i], num_slots);
  } else {
    const auto* s = static_cast<const int*>(slot_ids);
    for (std::size_t i = 0; i < total_tokens; ++i)
      validate_slot(s[i], num_slots);
  }

  const std::size_t slot_bytes = num_kv_heads * head_dim * elem_size;
  const std::size_t token_stride = 2 * slot_bytes;
  const std::size_t page_bytes = page_size * token_stride;
  auto* d = static_cast<std::uint8_t*>(dst);
  const auto* k = static_cast<const std::uint8_t*>(k_src);
  const auto* v = static_cast<const std::uint8_t*>(v_src);

  auto do_all = [&] {
    for (std::size_t p = 0; p < num_pages; ++p)
      gather_page(d + p * page_bytes, k, v,
                  page_slots(slot_ids, p, page_size, slot_ids_int64),
                  slot_ids_int64, page_size, slot_bytes, token_stride);
  };

  if (stream == nullptr) {
    do_all();
    return;
  }

  // Async: capture the pointers and sizes, copy the slot map into
  // reference-counted storage that lives until the task completes (the
  // worker may run after this function returns). One stream task for the
  // whole gather, matching the fused single-launch contract (the CUDA path
  // issues exactly one kernel on the same stream).
  auto owned = std::make_shared<std::vector<unsigned char>>(
      total_tokens * slot_elem_size(slot_ids_int64));
  std::memcpy(owned->data(), slot_ids, owned->size());

  stream->submit([d, k, v, owned, slot_ids_int64, num_pages, page_size,
                  page_bytes, slot_bytes, token_stride] {
    const void* owned_slots = owned->data();
    for (std::size_t p = 0; p < num_pages; ++p)
      gather_page(d + p * page_bytes, k, v,
                  page_slots(owned_slots, p, page_size, slot_ids_int64),
                  slot_ids_int64, page_size, slot_bytes, token_stride);
  });
}

}  // namespace vkernels::comm
