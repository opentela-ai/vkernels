// vkernels/comm/kv_scatter.cpp — host reference (oracle) implementation.
//
// The CPU reference is the correctness oracle: it is always compiled, fully
// unit-tested, and carries the contract checks (null pointers, positive
// dimensions, BF16/FP16 element size, non-negative, in-range AND unique
// destination slots). The CUDA path (kv_scatter.cu) performs the same fused
// indexed scatter in a single kernel and trusts the already-validated
// metadata so the hot path stays free of checks.
//
// The scatter is memory-bound (zero FLOP/byte — pure copy). For each token
// `t` of page `p` it reads K and V from the contiguous source and writes
// them into the indexed destination slot `slot_ids[p*page_size+t]`,
// matching the PyTorch two-write reference byte-for-byte:
//
//     k_dst[slot_ids] = src[:, :, 0]
//     v_dst[slot_ids] = src[:, :, 1]
//
// Because the destination slots are disjoint (uniqueness is enforced), every
// byte is written exactly once and the order of writes is irrelevant.
#include "vkernels/comm/kv_scatter.hpp"

#include "vkernels/util/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <unordered_set>
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

// Byte stride of one slot id (int32 or int64).
inline std::size_t slot_elem_size(bool slot_ids_int64) {
  return slot_ids_int64 ? sizeof(std::int64_t) : sizeof(int);
}

// Pointer to the first slot of page `p` (element stride, not byte stride).
inline const void* page_slots(const void* slot_ids, std::size_t p,
                              std::size_t page_size, bool slot_ids_int64) {
  return static_cast<const char*>(slot_ids) +
         p * page_size * slot_elem_size(slot_ids_int64);
}

// Scatter K/V for one page from the contiguous source into indexed
// destinations. `src_page` is the start of page `p` in `src`
// (page_bytes each). `slots` is &slot_ids[p * page_size]. `slot_bytes` is the
// per-slot extent; `token_stride` is 2 * slot_bytes (the [K, V] split per
// token in the source).
inline void scatter_page(const std::uint8_t* src_page, std::uint8_t* k_dst,
                         std::uint8_t* v_dst, const void* slots,
                         bool slot_ids_int64, std::size_t page_size,
                         std::size_t slot_bytes, std::size_t token_stride) {
  if (slot_ids_int64) {
    const auto* s = static_cast<const std::int64_t*>(slots);
    for (std::size_t t = 0; t < page_size; ++t) {
      const std::size_t slot = static_cast<std::size_t>(s[t]);
      const std::size_t src_off = t * token_stride;
      const std::size_t dst_off = slot * slot_bytes;
      std::memcpy(k_dst + dst_off, src_page + src_off, slot_bytes);
      std::memcpy(v_dst + dst_off, src_page + src_off + slot_bytes, slot_bytes);
    }
  } else {
    const auto* s = static_cast<const int*>(slots);
    for (std::size_t t = 0; t < page_size; ++t) {
      const std::size_t slot = static_cast<std::size_t>(s[t]);
      const std::size_t src_off = t * token_stride;
      const std::size_t dst_off = slot * slot_bytes;
      std::memcpy(k_dst + dst_off, src_page + src_off, slot_bytes);
      std::memcpy(v_dst + dst_off, src_page + src_off + slot_bytes, slot_bytes);
    }
  }
}

}  // namespace

void kv_scatter_layer(void* k_dst, void* v_dst,
                      const void* slot_ids, bool slot_ids_int64,
                      std::size_t num_slots,
                      const void* src,
                      std::size_t num_pages, std::size_t page_size,
                      std::size_t num_kv_heads, std::size_t head_dim,
                      std::size_t elem_size,
                      Stream* stream) {
  // Validate arguments on the host (the CUDA kernel trusts these).
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

  // Bounds-check every destination slot once on the host, AND enforce
  // uniqueness: the scatter writes disjoint destinations (unlike the
  // gather's repeatable sources), so a duplicate slot is a contract
  // violation that would race the same bytes.
  const std::size_t total_tokens = num_pages * page_size;
  std::unordered_set<std::int64_t> seen;
  seen.reserve(total_tokens);
  if (slot_ids_int64) {
    const auto* s = static_cast<const std::int64_t*>(slot_ids);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      validate_slot(s[i], num_slots);
      VK_EXPECTS(seen.insert(s[i]).second, "slot_ids must be unique");
    }
  } else {
    const auto* s = static_cast<const int*>(slot_ids);
    for (std::size_t i = 0; i < total_tokens; ++i) {
      validate_slot(static_cast<std::int64_t>(s[i]), num_slots);
      VK_EXPECTS(seen.insert(static_cast<std::int64_t>(s[i])).second,
                 "slot_ids must be unique");
    }
  }

  const std::size_t slot_bytes = num_kv_heads * head_dim * elem_size;
  const std::size_t token_stride = 2 * slot_bytes;
  const std::size_t page_bytes = page_size * token_stride;
  auto* k = static_cast<std::uint8_t*>(k_dst);
  auto* v = static_cast<std::uint8_t*>(v_dst);
  const auto* s = static_cast<const std::uint8_t*>(src);

  auto do_all = [&] {
    for (std::size_t p = 0; p < num_pages; ++p)
      scatter_page(s + p * page_bytes, k, v,
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
  // whole scatter, matching the fused single-launch contract (the CUDA path
  // issues exactly one kernel on the same stream).
  auto owned = std::make_shared<std::vector<unsigned char>>(
      total_tokens * slot_elem_size(slot_ids_int64));
  std::memcpy(owned->data(), slot_ids, owned->size());

  stream->submit([k, v, s, owned, slot_ids_int64, num_pages, page_size,
                  page_bytes, slot_bytes, token_stride] {
    const void* owned_slots = owned->data();
    for (std::size_t p = 0; p < num_pages; ++p)
      scatter_page(s + p * page_bytes, k, v,
                   page_slots(owned_slots, p, page_size, slot_ids_int64),
                   slot_ids_int64, page_size, slot_bytes, token_stride);
  });
}

}  // namespace vkernels::comm
