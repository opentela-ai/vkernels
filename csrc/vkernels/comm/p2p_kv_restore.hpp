// vkernels/comm/p2p_kv_restore.hpp
//
// Fused peer-to-indexed-KV restore kernel (issue #4).
//
// The two-stage peer restore path moves each layer twice:
//   1. peer HBM → contiguous local scratch (p2p_gather_runs)
//   2. local scratch → paged KV-pool slots (indexed scatter)
// This kernel fuses both steps: it reads peer UVA addresses directly and
// writes into the indexed K/V destinations, eliminating the scratch
// allocation, the extra local-HBM read/write pass, and the separate scatter
// launch.
//
// Contract
// --------
//  * `k_dst` / `v_dst` are local allocations on the executing device,
//    each of capacity `num_slots * num_kv_heads * head_dim * elem_size`
//    bytes, laid out as [num_slots, num_kv_heads, head_dim] row-major.
//  * `slot_ids[p * page_size + t]` gives the destination slot for token
//    `t` of page `p`. Destination slots must be unique across all pages
//    (the host reference validates this; the CUDA kernel trusts it).
//  * Each `peer_src_ptrs[p]` is a peer-accessible UVA (or IPC-mapped)
//    pointer. `src_page_offsets[p]` is a byte offset added to that
//    pointer so the effective source for page `p` is
//    `peer_src_ptrs[p] + src_page_offsets[p]`.
//  * The source data at that address is laid out as
//    [page_size, 2, num_kv_heads, head_dim] in row-major order with
//    element size `elem_size` bytes. The "2" dimension is [K, V]:
//    K data for all heads of a token comes first, then V data.
//  * Peer access and IPC mapping must be established by the caller before
//    the launch and held until `stream` completes — direct peer reads are
//    issued from inside the kernel.
//
// Lifetime
// --------
//  * `k_dst`, `v_dst`, and the peer memory behind every `peer_src_ptrs[p]`
//    must outlive `stream`.
//  * The metadata arrays (`slot_ids`, `peer_src_ptrs`, `src_page_offsets`)
//    are read and copied into owned storage before the function returns,
//    so the caller may free or mutate the originals as soon as the call
//    returns — only the IPC mappings, not the metadata, must persist.
//  * When `stream == nullptr` the work runs to completion before returning.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/core/stream.hpp"

namespace vkernels::comm {

// Fused peer-to-indexed-KV restore for one layer. Copies `num_pages` pages
// of KV data from peer UVA directly into indexed local slots.
//
// `k_dst`, `v_dst`: local buffers of `num_slots * num_kv_heads * head_dim * elem_size`
//                   bytes each, row-major [num_slots, num_kv_heads, head_dim].
// `slot_ids`:       [num_pages * page_size] int32 destination slot per token.
// `peer_src_ptrs`:  [num_pages] peer UVA base pointers.
// `src_page_offsets`:[num_pages] byte offset added to each base pointer.
// `num_pages`:      number of pages (0 is a valid no-op).
// `page_size`:      tokens per page.
// `num_kv_heads`:   KV heads (not query heads; for GQA this is < num_q_heads).
// `head_dim`:       elements per head.
// `elem_size`:      bytes per element (2 for BF16/FP16).
void p2p_kv_restore_layer(void* k_dst, void* v_dst,
                          const int* slot_ids,
                          const void* const* peer_src_ptrs,
                          const std::size_t* src_page_offsets,
                          std::size_t num_pages, std::size_t page_size,
                          std::size_t num_kv_heads, std::size_t head_dim,
                          std::size_t elem_size,
                          Stream* stream = nullptr);

// Two-stage reference for testing and fallback:
//   1. peer → contiguous scratch (emulated gather), then
//   2. scratch → indexed K/V scatter.
// The fused kernel must produce byte-identical results.
void p2p_kv_restore_layer_twostage(void* k_dst, void* v_dst,
                                   const int* slot_ids,
                                   const void* const* peer_src_ptrs,
                                   const std::size_t* src_page_offsets,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   Stream* stream = nullptr);

}  // namespace vkernels::comm
