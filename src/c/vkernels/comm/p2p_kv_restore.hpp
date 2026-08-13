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
#include <vector>

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

// Scatter a flat scratch buffer (one page at [page_size, 2, num_kv_heads,
// head_dim] row-major per page) into indexed K/V destinations — the second
// stage of the two-path, exposed so a caller that already gathered pages
// (e.g. a prepared P2PGatherPlan2D) can scatter them without re-reading peer
// memory. `slot_ids[page * page_size + t]` is the destination slot for token
// `t` of page `page`; slots must be unique (the host reference validates
// this, exactly as `p2p_kv_restore_layer` does).
void kv_scatter(void* k_dst, void* v_dst, const void* scratch,
                const int* slot_ids, std::size_t num_pages,
                std::size_t page_size, std::size_t num_kv_heads,
                std::size_t head_dim, std::size_t elem_size,
                Stream* stream = nullptr);

// ---------------------------------------------------------------------------
// Prepared fused peer-to-indexed-KV restore plan (issue #27)
// ---------------------------------------------------------------------------
//
// KVAAS restores one residency set across all model layers (40 for
// Qwen3-14B), reusing the same peer page bases and destination slot map for
// every layer. The one-shot `p2p_kv_restore_layer` repeats metadata setup —
// two `cudaMallocAsync` calls, two H2D copies and two `cudaFreeAsync` calls
// per layer (80 allocations / 80 H2D copies / 80 frees for 40 layers) — which
// gives back the kernel savings and shows up as an end-to-end regression.
//
// A plan moves all of that to a single prepare step: the constructor
// validates the metadata once, captures the destination K/V geometry, element
// type, destination capacity and maximum valid slot, copies the peer page
// bases (and, for the host-input variant, the slot map) into owned storage,
// and — on the CUDA path — uploads them to a persistent per-device buffer.
// `execute(source_layer_offset_bytes, stream)` then only enqueues: no
// validation, no allocation, no H2D copy. KVAAS has one common scalar
// source-layer offset for every peer page in a layer; the same plan is reused
// across all model layers by varying only that scalar.
//
// Two creation modes:
//  * Host input (default constructor): `slot_ids` is a host array. The plan
//    validates uniqueness, non-negativity and destination bounds
//    (slot < num_slots) once at construction, then owns a copy.
//  * Device slots (`from_device_slots` tag): `slot_ids` is a caller-owned
//    CUDA device pointer SGLang already holds (its radix-tree `device_indices`).
//    The plan does NOT copy it and does NOT validate its contents — reading
//    device memory to validate would defeat the point (a device-to-host
//    synchronization) — so the caller MUST guarantee unique, non-negative,
//    in-range slots and MUST keep the `device_indices` buffer alive until the
//    plan is destroyed and every stream it was executed on has completed.
//
// Concurrency: after prepare the plan's metadata is immutable, so one plan
// may be executed concurrently on several streams. Lifetime: the destination
// buffers, the peer memory behind every page base, and (for the device-slot
// variant) the `device_indices` buffer must outlive every execute() that uses
// the plan; destroy the plan only after every such stream has completed.

// Tag selecting the device-slot constructor (see P2PKvRestorePlan).
struct from_device_slots_t {};
// A constexpr instance for `P2PKvRestorePlan{from_device_slots{}, ...}`.
inline constexpr from_device_slots_t from_device_slots{};

// A prepared fused restore plan, bound to one (k_dst, v_dst) pair and one
// destination geometry. Created once; executed many times across layers.
class P2PKvRestorePlan {
 public:
  // Host-input plan: validate the slot map once, copy peer bases and slot_ids
  // into owned storage. `num_slots` is the destination capacity in slots
  // (every slot_id must be in [0, num_slots)); `num_kv_heads`, `head_dim`,
  // `elem_size` fix the per-slot byte size and (with `page_size`) the
  // per-token source stride. `num_pages == 0` is a valid no-op plan.
  // Throws std::invalid_argument on a contract violation (null destination,
  // zero dimensions, non-BF16/FP16 elem_size, duplicate/negative/out-of-range
  // slot).
  P2PKvRestorePlan(void* k_dst, void* v_dst, std::size_t num_slots,
                   std::size_t num_kv_heads, std::size_t head_dim,
                   std::size_t elem_size, const int* slot_ids,
                   const void* const* peer_src_ptrs, std::size_t num_pages,
                   std::size_t page_size);

  // Device-slot plan: borrow the caller's CUDA `device_indices` (shape
  // [num_pages * page_size]) directly. No copy, no host validation of slot
  // contents — the caller guarantees unique, non-negative, in-range slots and
  // keeps `device_indices` alive until the plan is destroyed and all streams
  // it ran on have completed. Metadata shape (null pointers, zero
  // dimensions, elem_size) IS still validated.
  P2PKvRestorePlan(from_device_slots_t, void* k_dst, void* v_dst,
                   std::size_t num_slots, std::size_t num_kv_heads,
                   std::size_t head_dim, std::size_t elem_size,
                   const int* device_indices,
                   const void* const* peer_src_ptrs, std::size_t num_pages,
                   std::size_t page_size);

  P2PKvRestorePlan(const P2PKvRestorePlan&) = delete;
  P2PKvRestorePlan& operator=(const P2PKvRestorePlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t num_kv_heads() const { return num_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t elem_size() const { return elem_size_; }
  std::size_t total_bytes() const { return total_bytes_; }

  // Enqueue the fused restore for one layer, adding `source_layer_offset_bytes`
  // to every peer page base before reading. Exactly one stream task
  // (host reference) regardless of page count. A null stream runs to
  // completion before returning. The plan must outlive the stream.
  void execute(std::size_t source_layer_offset_bytes,
               Stream* stream = nullptr) const;

 private:
  // Shared validation of metadata shape (used by both constructors; the
  // host-input constructor additionally validates slot contents).
  void validate_shape(const void* const* peer_src_ptrs) const;

  void* k_dst_;
  void* v_dst_;
  std::size_t num_slots_;
  std::size_t num_kv_heads_;
  std::size_t head_dim_;
  std::size_t elem_size_;
  std::size_t page_size_;
  std::size_t num_pages_;
  std::size_t slot_bytes_;
  std::size_t token_stride_;
  std::size_t total_bytes_;
  // Peer page bases, copied at construction (the caller may free the original
  // array as soon as the constructor returns).
  std::vector<const void*> peer_bases_;
  // Owned slot map (empty for the device-slot variant, which borrows the
  // caller's pointer).
  std::vector<int> owned_slots_;
  const int* slot_ids_;  // points into owned_slots_ or the caller's pointer
};

}  // namespace vkernels::comm
