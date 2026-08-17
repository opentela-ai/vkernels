// vkernels/comm/kv_gather.hpp
//
// Fused indexed K/V layer gather kernel (issue #2).
//
// The donation-side reverse of p2p_kv_restore (issue #4): KVAAS packs
// arbitrary SGLang KV-pool slots into a contiguous per-layer page buffer
// before donation/eviction to peer HBM. The current implementation performs
// separate advanced-index gathers for K and V and separate writes into the
// packed output; this kernel fuses both gathers (and the K/V interleave) into
// a single launch.
//
// Contract
// --------
//  * `k_src` / `v_src` are local allocations on the executing device, each
//    of capacity `num_slots * num_kv_heads * head_dim * elem_size` bytes,
//    laid out [num_slots, num_kv_heads, head_dim] row-major. `num_slots` is
//    the paged-KV cache capacity (every slot_id must be in [0, num_slots)).
//  * `slot_ids[p * page_size + t]` gives the SOURCE slot for token `t` of
//    page `p`. Source slots may REPEAT (gather semantics) and may be in any
//    order (non-monotonic): unlike the restore (which scatters and therefore
//    requires UNIQUE destination slots), the gather reads source slots and a
//    repeated index simply re-reads the same memory.
//  * `slot_ids` is `int32` when `slot_ids_int64 == false` and `int64` when
//    `slot_ids_int64 == true` (SGLang's radix-tree indices are torch.int64).
//  * `dst` is a packed [num_pages, page_size, 2, num_kv_heads, head_dim]
//    row-major buffer of `num_pages * page_size * 2 * num_kv_heads *
//    head_dim * elem_size` bytes. Index 0 of the "2" dimension receives K
//    and index 1 receives V, exactly:
//
//        dst[:, :, 0] = k_src[slot_ids]
//        dst[:, :, 1] = v_src[slot_ids]
//
//    (the PyTorch two-gather reference the kernel replaces, byte-identical).
//  * All tensors are on one device. Strides are not part of the C API:
//    sources and destination are assumed C-contiguous row-major, and the
//    Python binding rejects non-default strides explicitly.
//
// Lifetime
// --------
//  * `k_src`, `v_src`, `dst` and (for the device-slot variants) `slot_ids`
//    must outlive `stream`.
//  * For the host-input `kv_gather_layer` the `slot_ids` array is read and
//    copied into owned storage before the function returns, so the caller
//    may free or mutate the original as soon as the call returns -- only
//    `k_src`, `v_src` and `dst` must persist until the work completes.
//  * When `stream == nullptr` the work runs to completion before returning.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/core/stream.hpp"

namespace vkernels::comm {

// Fused indexed K/V gather for one layer. Reads `num_pages * page_size`
// indexed source slots from `k_src` / `v_src` and writes the packed
// [num_pages, page_size, 2, num_kv_heads, head_dim] `dst` in a single
// operation. `slot_ids_int64` selects the element type of `slot_ids`
// (int32 or int64). `num_pages == 0` is a valid no-op. Throws
// std::invalid_argument on a contract violation (null pointers, zero
// dimensions, non-BF16/FP16 elem_size, negative or out-of-range slot).
//
// This host reference is the correctness oracle: it carries the full
// contract checks and is always compiled. The CUDA path
// (kv_gather.cu / kv_gather_cuda.hpp) performs the same fused gather in one
// kernel launch and trusts the already-validated metadata on the hot path.
void kv_gather_layer(void* dst,
                     const void* k_src, const void* v_src,
                     const void* slot_ids, bool slot_ids_int64,
                     std::size_t num_slots,
                     std::size_t num_pages, std::size_t page_size,
                     std::size_t num_kv_heads, std::size_t head_dim,
                     std::size_t elem_size,
                     Stream* stream = nullptr);

}  // namespace vkernels::comm
