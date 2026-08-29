// vkernels/comm/slot_map.hpp
//
// Shared geometry + slot-map validation and byte accounting for the
// cross-node KV plans (issue #49). The restore / donate host reference
// (cross_node_kv.cpp) and the all-gather CUDA plan
// (cross_node_kv_allgather.cu) validate the SAME contract: positive
// dimensions (relaxed for the `num_pages == 0` no-op), elem_size == 2 for
// BF16/FP16, and a host int32 slot map that is non-negative and in
// [0, num_slots). The restore / all-gather paths additionally require the
// slots to be UNIQUE (scatter -> each destination written once); the
// donate path allows repeats (gather).
//
// The validators lived in cross_node_kv.cpp's anonymous namespace, which
// is unreachable from the CUDA TU -- so the all-gather plan duplicated
// them inline. They now live here, header-only, so there is exactly one
// definition of each contract.
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "vkernels/util/error.hpp"

namespace vkernels::comm {

// One page's worth of ONE layer: [page_size, 2, num_kv_heads, head_dim].
// The per-slot stride (2 for K+V) is realistic (num_kv_heads * head_dim *
// elem_size <= 128 * 128 * 2) and cannot overflow size_t, so it stays a
// plain product.
inline std::size_t per_slot_bytes(std::size_t num_kv_heads,
                                  std::size_t head_dim,
                                  std::size_t elem_size) {
  return num_kv_heads * head_dim * elem_size;
}

// One token's worth of BOTH K and V: [2, num_kv_heads, head_dim, elem_size].
inline std::size_t token_stride_bytes(std::size_t num_kv_heads,
                                      std::size_t head_dim,
                                      std::size_t elem_size) {
  return 2 * per_slot_bytes(num_kv_heads, head_dim, elem_size);
}

// a * b with an overflow check. Throws std::invalid_argument (which the C
// ABI maps to VKERNELS_FI_ERR_INVALID_ARGUMENT) instead of silently
// wrapping; used for the page * stride * num_pages products whose totals
// can realistically exceed size_t for adversarial geometries.
inline std::size_t checked_mul(std::size_t a, std::size_t b, const char* what) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument(std::string(what) + " overflows size_t");
  return a * b;
}

// Shared geometry validation, mirroring P2PKvRestorePlan::validate_shape and
// P2PKvDonatePlan::validate_shape: `num_pages == 0` is a valid no-op that
// relaxes the "positive dimensions" checks (otherwise every dimension must
// be positive and elem_size must be 2 for BF16/FP16). elem_size is checked
// unconditionally (the no-op plan still reports its element size).
inline void validate_kv_plan_shape(std::size_t num_slots,
                                   std::size_t num_kv_heads,
                                   std::size_t head_dim,
                                   std::size_t elem_size,
                                   std::size_t num_pages,
                                   std::size_t page_size) {
  VK_EXPECTS(num_slots > 0 || num_pages == 0, "num_slots must be positive");
  VK_EXPECTS(page_size > 0 || num_pages == 0, "page_size must be positive");
  VK_EXPECTS(num_kv_heads > 0 || num_pages == 0, "num_kv_heads must be positive");
  VK_EXPECTS(head_dim > 0 || num_pages == 0, "head_dim must be positive");
  VK_EXPECTS(elem_size == 2, "elem_size must be 2 for BF16/FP16");
}

// Validate a slot map for the RESTORE / all-gather (scatter -> UNIQUE
// destination slots, non-negative, in [0, num_slots)). Mirrors
// P2PKvRestorePlan's host-input validation so each plan validates once, at
// construction.
inline void validate_unique_slots(const int* slot_ids, std::size_t total_tokens,
                                  std::size_t num_slots) {
  std::unordered_set<int> seen;
  seen.reserve(total_tokens);
  for (std::size_t i = 0; i < total_tokens; ++i) {
    const int slot = slot_ids[i];
    VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
    VK_EXPECTS(static_cast<std::size_t>(slot) < num_slots,
               "slot_ids must be < num_slots");
    VK_EXPECTS(seen.insert(slot).second, "slot_ids must be unique");
  }
}

// Validate a slot map for the DONATE (gather -> repeats allowed, only
// non-negativity and bounds). Mirrors P2PKvDonatePlan's host-input
// validation.
inline void validate_slot_bounds(const int* slot_ids, std::size_t total_tokens,
                                 std::size_t num_slots) {
  for (std::size_t i = 0; i < total_tokens; ++i) {
    const int slot = slot_ids[i];
    VK_EXPECTS(slot >= 0, "slot_ids must be non-negative");
    VK_EXPECTS(static_cast<std::size_t>(slot) < num_slots,
               "slot_ids must be < num_slots");
  }
}

}  // namespace vkernels::comm
