// vkernels/comm/kv_scatter.hpp
//
// Fused indexed K/V layer scatter kernel (issue #1).
//
// The restore-side reverse of kv_gather (issue #2): KVAAS scatters a
// contiguous, already-gathered per-layer scratch buffer back into the paged
// KV pool. The destination token slots are arbitrary and non-contiguous, so
// a memcpy cannot place the data directly. The current fallback performs two
// PyTorch advanced-index writes (one for K, one for V); this kernel fuses
// both writes (and the K/V split) into a single launch that reads each slot
// id once and writes K and V together.
//
// Contract
// --------
//  * `k_dst` / `v_dst` are local allocations on the executing device, each
//    of capacity `num_slots * num_kv_heads * head_dim * elem_size` bytes,
//    laid out [num_slots, num_kv_heads, head_dim] row-major. `num_slots` is
//    the paged-KV cache capacity (every slot_id must be in [0, num_slots)).
//  * `slot_ids[p * page_size + t]` gives the DESTINATION slot for token `t`
//    of page `p`. Unlike the gather (which may repeat source slots), the
//    scatter writes disjoint destinations: slot_ids must be UNIQUE across
//    the whole `[num_pages, page_size]` map. Duplicates are a contract
//    violation (two threads would race the same destination); the host
//    reference rejects them, exactly as `p2p_kv_restore` does. Slots may be
//    in any order (non-monotonic).
//  * `slot_ids` is `int32` when `slot_ids_int64 == false` and `int64` when
//    `slot_ids_int64 == true` (SGLang's radix-tree indices are torch.int64).
//  * `src` is a packed [num_pages, page_size, 2, num_kv_heads, head_dim]
//    row-major buffer of `num_pages * page_size * 2 * num_kv_heads *
//    head_dim * elem_size` bytes. Index 0 of the "2" dimension is K and
//    index 1 is V, exactly:
//
//        k_dst[slot_ids] = src[:, :, 0]
//        v_dst[slot_ids] = src[:, :, 1]
//
//    (the PyTorch two-write reference the kernel replaces, byte-identical).
//  * All tensors are on one device. Strides are not part of the C API:
//    sources and destination are assumed C-contiguous row-major, and the
//    Python binding rejects non-default strides explicitly.
//
// Lifetime
// --------
//  * `k_dst`, `v_dst`, `src` and (for the device-slot variants) `slot_ids`
//    must outlive `stream`.
//  * For the host-input `kv_scatter_layer` the `slot_ids` array is read and
//    copied into owned storage before the function returns, so the caller
//    may free or mutate the original as soon as the call returns -- only
//    `k_dst`, `v_dst` and `src` must persist until the work completes.
//  * When `stream == nullptr` the work runs to completion before returning.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/core/stream.hpp"

namespace vkernels::comm {

// Fused indexed K/V scatter for one layer. Reads `num_pages * page_size`
// indexed destination slots from `slot_ids` and writes K and V from the
// packed `src` in a single operation. `slot_ids_int64` selects the element
// type of `slot_ids` (int32 or int64). `num_pages == 0` is a valid no-op.
// Throws std::invalid_argument on a contract violation (null pointers, zero
// dimensions, non-BF16/FP16 elem_size, negative, out-of-range, OR DUPLICATE
// slots).
//
// This host reference is the correctness oracle: it carries the full
// contract checks (including slot uniqueness) and is always compiled. The
// CUDA path (kv_scatter.cu / kv_scatter_cuda.hpp) performs the same fused
// scatter in one kernel launch and trusts the already-validated metadata on
// the hot path.
void kv_scatter_layer(void* k_dst, void* v_dst,
                      const void* slot_ids, bool slot_ids_int64,
                      std::size_t num_slots,
                      const void* src,
                      std::size_t num_pages, std::size_t page_size,
                      std::size_t num_kv_heads, std::size_t head_dim,
                      std::size_t elem_size,
                      Stream* stream = nullptr);

}  // namespace vkernels::comm
