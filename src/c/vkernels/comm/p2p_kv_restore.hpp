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
// every layer, but writing each layer into a DIFFERENT destination K/V
// buffer. The one-shot `p2p_kv_restore_layer` repeats metadata setup —
// two `cudaMallocAsync` calls, two H2D copies and two `cudaFreeAsync` calls
// per layer (80 allocations / 80 H2D copies / 80 frees for 40 layers) — which
// gives back the kernel savings and shows up as an end-to-end regression.
//
// A plan moves all of that to a single prepare step: the constructor
// validates the metadata once, captures the destination slot count, KV shape
// and element type, and copies the peer page bases (and, for the host-input
// variant, the slot map) into owned storage — on the CUDA path it uploads
// them to a persistent per-device buffer. Crucially the plan does NOT bind a
// destination: (k_dst, v_dst) is supplied at every `execute(k_dst, v_dst,
// source_layer_offset_bytes, stream)`, so one plan fans the same run list out
// into a distinct K/V layer buffer per layer. execute() then only launches
// one task (plus a single null-destination guard): no metadata validation,
// no allocation, no H2D copy. KVAAS has one common scalar source-layer offset
// for every peer page in a layer; the same plan is reused across all model
// layers by varying only that scalar and the destination pair.
//
// Three creation modes:
//  * Host input (default constructor): `slot_ids` is a host array. The plan
//    validates uniqueness, non-negativity and destination bounds
//    (slot < num_slots) once at construction, then owns a copy.
//  * Device slots, int32 (`from_device_slots` tag): `slot_ids` is a
//    caller-owned CUDA device pointer SGLang already holds (its radix-tree
//    `device_indices`). The plan does NOT copy it and does NOT validate its
//    contents — reading device memory to validate would defeat the point (a
//    device-to-host synchronization) — so the caller MUST guarantee unique,
//    non-negative, in-range slots and MUST keep the `device_indices` buffer
//    alive until the plan is destroyed and every stream it was executed on
//    has completed.
//  * Device slots, int64 (`from_device_slots_int64` tag): `slot_ids` is a
//    caller-owned CUDA device pointer of `int64` (SGLang's radix-tree indices
//    are torch.int64). The plan converts them in one device kernel to an
//    OWNED int32 buffer at construction — no D2H sync, and the caller may
//    free the int64 buffer as soon as the constructor returns. Slot contents
//    are NOT validated (same as the int32 device-slot variant).
//
// Concurrency: after prepare the plan's metadata is immutable, so one plan
// may be executed concurrently on several streams (each call supplies its
// own (k_dst, v_dst); the caller is responsible for non-overlapping
// destinations across concurrent executes). Lifetime: the per-execute
// destination buffers and the peer memory behind every page base must outlive
// every execute() that uses them; for the int32 device-slot variant the
// `device_indices` buffer must outlive the plan as well (the int64 variant
// owns its copy and imposes no such constraint). Destroy the plan only after
// every stream it was executed on has completed.

// Tag selecting the device-slot constructor that borrows the caller's int32
// CUDA `device_indices` pointer (see P2PKvRestorePlan).
struct from_device_slots_t {};
// A constexpr instance for `P2PKvRestorePlan{from_device_slots{}, ...}`.
inline constexpr from_device_slots_t from_device_slots{};

// Tag selecting the device-slot constructor that takes the caller's int64
// CUDA `device_indices` pointer (e.g. SGLang's radix-tree indices, which are
// torch.int64). The plan owns an int32 device copy filled by a one-time
// int64->int32 conversion kernel at construction -- no D2H sync, and the
// caller may free the int64 buffer as soon as the constructor returns.
struct from_device_slots_int64_t {};
// A constexpr instance for `P2PKvRestorePlan{from_device_slots_int64{}, ...}`.
inline constexpr from_device_slots_int64_t from_device_slots_int64{};

// A prepared fused restore plan, bound to one destination geometry (slot
// count, KV shape, element type) and one run list (peer page bases + slot
// map), but NOT to any destination buffer. Created once; the destination
// (k_dst, v_dst) pair is supplied at every execute() so a single plan can
// fan one run list out into a distinct K/V layer buffer per layer (the
// KVAAS restore pattern). All metadata is validated and staged once at
// construction; execute() only launches.
class P2PKvRestorePlan {
 public:
  // Host-input plan: validate the slot map once, copy peer bases and
  // slot_ids into owned storage. `num_slots` is the destination capacity in
  // slots (every slot_id must be in [0, num_slots)); `num_kv_heads`,
  // `head_dim`, `elem_size` fix the per-slot byte size and (with
  // `page_size`) the per-token source stride. `num_pages == 0` is a valid
  // no-op plan. Throws std::invalid_argument on a contract violation (zero
  // dimensions, non-BF16/FP16 elem_size, duplicate/negative/out-of-range
  // slot).
  P2PKvRestorePlan(std::size_t num_slots, std::size_t num_kv_heads,
                   std::size_t head_dim, std::size_t elem_size,
                   const int* slot_ids, const void* const* peer_src_ptrs,
                   std::size_t num_pages, std::size_t page_size);

  // Device-slot plan (int32): borrow the caller's CUDA `device_indices`
  // (shape [num_pages * page_size]) directly. No copy, no host validation
  // of slot contents -- the caller guarantees unique, non-negative, in-range
  // slots and keeps `device_indices` alive until the plan is destroyed and
  // all streams it ran on have completed. Metadata shape (null pointers,
  // zero dimensions, elem_size) IS still validated.
  P2PKvRestorePlan(from_device_slots_t, std::size_t num_slots,
                   std::size_t num_kv_heads, std::size_t head_dim,
                   std::size_t elem_size, const int* device_indices,
                   const void* const* peer_src_ptrs, std::size_t num_pages,
                   std::size_t page_size);

  // Device-slot plan (int64): take the caller's CUDA int64 `device_indices`
  // (shape [num_pages * page_size], e.g. SGLang's torch.int64 radix-tree
  // indices), convert them in one device kernel to an OWNED int32 buffer,
  // and keep that. No D2H sync (the conversion is device-to-device); the
  // caller may free the int64 buffer as soon as the constructor returns.
  // Slot contents are NOT validated (same as the int32 device-slot plan).
  P2PKvRestorePlan(from_device_slots_int64_t, std::size_t num_slots,
                   std::size_t num_kv_heads, std::size_t head_dim,
                   std::size_t elem_size, const std::int64_t* device_indices,
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

  // Enqueue the fused restore for one layer into (k_dst, v_dst), adding
  // `source_layer_offset_bytes` to every peer page base before reading.
  // Exactly one stream task (host reference) regardless of page count. A
  // null stream runs to completion before returning. (k_dst, v_dst) must be
  // non-null and each of capacity `num_slots * num_kv_heads * head_dim *
  // elem_size` bytes, laid out [num_slots, num_kv_heads, head_dim]; this is
  // the only execute-time check. The plan must outlive the stream.
  void execute(void* k_dst, void* v_dst,
               std::size_t source_layer_offset_bytes,
               Stream* stream = nullptr) const;

 private:
  // Shared validation of metadata shape (used by all constructors; the
  // host-input constructor additionally validates slot contents).
  void validate_shape(const void* const* peer_src_ptrs) const;

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
  // Owned int32 slot map. Empty for the int32 device-slot variant (which
  // borrows the caller's pointer); filled by the host-input constructor
  // (host copy) and the int64 device-slot constructor (int64->int32 convert).
  std::vector<int> owned_slots_;
  const int* slot_ids_;  // points into owned_slots_ or the caller's pointer
};

}  // namespace vkernels::comm
