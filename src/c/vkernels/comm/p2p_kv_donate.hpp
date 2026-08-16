// vkernels/comm/p2p_kv_donate.hpp
//
// Fused indexed-KV-to-peer donation kernel (issue #36).
//
// The donation-side mirror of p2p_kv_restore (issue #27). KVAAS still
// materializes a full all-layer packed scratch tensor before peer DMA:
// pack_pages allocates [pages, layers, page_size, 2, heads, dim], PyTorch
// advanced-index gathers K and V into that scratch for every layer, then
// KVAAS issues scratch-to-peer copies and pins the scratch until the
// completion ACK. This kernel fuses the gather and the peer write: it reads
// arbitrary local paged-KV slots and writes K/V directly into the
// layer-major peer-page destination through peer-accessible UVA pointers,
// eliminating the scratch allocation, the extra local-HBM read/write pass,
// and the separate peer copy.
//
// Contract
// --------
//  * `k_src` / `v_src` are local allocations on the executing device, each
//    of capacity `num_slots * num_kv_heads * head_dim * elem_size` bytes,
//    laid out [num_slots, num_kv_heads, head_dim] row-major. `num_slots`
//    is fixed by the caller (the paged-KV cache capacity); the one-shot
//    does not take it and only validates that source slots are
//    non-negative, while the host-input plan validates `slot < num_slots`.
//  * `slot_ids[p * page_size + t]` gives the SOURCE slot for token `t` of
//    page `p`. Source slots may REPEAT (gather semantics) and may be in any
//    order (non-monotonic): unlike the restore (which scatters and
//    therefore requires UNIQUE destination slots), the donate reads source
//    slots and a repeated index simply re-reads the same memory. The host
//    reference validates only non-negativity; the CUDA kernel trusts the
//    already-validated metadata.
//  * Each `peer_dst_ptrs[p]` is a peer-accessible UVA (or IPC-mapped)
//    pointer to a page laid out [layers, page_size, 2, num_kv_heads,
//    head_dim] in row-major order with element size `elem_size` bytes. The
//    "2" dimension is [K, V]: K data for all heads of a token comes first,
//    then V data.
//  * `dst_page_offsets[p]` is a byte offset added to `peer_dst_ptrs[p]` so
//    the effective destination for page `p` is
//    `peer_dst_ptrs[p] + dst_page_offsets[p]`. For the prepared plan this
//    becomes a single scalar `destination_layer_offset_bytes` added to
//    every page base before writing.
//  * Peer access and IPC mapping must be established by the caller before
//    the launch and held until `stream` completes -- direct peer stores are
//    issued from inside the kernel.
//  * Destination peer slots are already reserved by KVAAS and remain valid
//    through completion; the caller owns completion/publish ordering.
//
// Lifetime
// --------
//  * `k_src`, `v_src`, and the peer memory behind every `peer_dst_ptrs[p]`
//    must outlive `stream`.
//  * The metadata arrays (`slot_ids`, `peer_dst_ptrs`, `dst_page_offsets`)
//    are read and copied into owned storage before the one-shot returns
//    (the prepared plan owns them from construction), so the caller may
//    free or mutate the originals as soon as the call returns -- only the
//    IPC mappings, not the metadata, must persist.
//  * When `stream == nullptr` the work runs to completion before returning.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "vkernels/comm/p2p_kv_restore.hpp"  // from_device_slots_t/int64_t tags
#include "vkernels/core/stream.hpp"

namespace vkernels::comm {

// ---------------------------------------------------------------------------
// One-shot primitives
// ---------------------------------------------------------------------------

// Fused indexed-KV-to-peer donation for one layer. Reads `num_pages` pages
// of local paged KV (indexed by `slot_ids`) and writes K/V directly into the
// layer-major peer-page destination (peer_dst_ptrs[p] + dst_page_offsets[p])
// through one stream task. `num_pages == 0` is a valid no-op.
//
// On CUDA (p2p_kv_donate.cu) this dispatches ADAPTIVELY (see
// prefer_direct_store): by default a single SM kernel reads local slots and
// writes peer memory over NVLink with no scratch allocation; when the copy
// engine wins (few large pages) or direct peer stores are unsupported, it
// falls back to gather-then-per-page-copy (p2p_kv_donate_layer_twostage).
// The host reference is always the fused task (it cannot distinguish an SM
// kernel from a copy engine) and is the byte-exact oracle for both paths.
void p2p_kv_donate_layer(const void* k_src, const void* v_src,
                         const int* slot_ids,
                         const void* const* peer_dst_ptrs,
                         const std::size_t* dst_page_offsets,
                         std::size_t num_pages, std::size_t page_size,
                         std::size_t num_kv_heads, std::size_t head_dim,
                         std::size_t elem_size,
                         Stream* stream = nullptr);

// Two-stage reference for testing and fallback:
//   1. indexed local slots -> contiguous scratch (kv_gather), then
//   2. scratch -> peer page copy (one copy per page).
// The fused kernel must produce byte-identical results. This is also the
// copy-engine fallback used when direct peer stores are unsupported or lose
// to the copy-engine path (issue #36). The scratch is allocated per call.
void p2p_kv_donate_layer_twostage(const void* k_src, const void* v_src,
                                  const int* slot_ids,
                                  const void* const* peer_dst_ptrs,
                                  const std::size_t* dst_page_offsets,
                                  std::size_t num_pages, std::size_t page_size,
                                  std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size,
                                  Stream* stream = nullptr);

// Gather local paged-KV slots into a flat scratch buffer -- one
// [page_size, 2, num_kv_heads, head_dim] page per page, row-major -- the
// first stage of the two-path. Exposed so a caller that wants the
// copy-engine fallback can gather once (e.g. with a prepared plan) and then
// issue the peer copy separately. `slot_ids` may repeat (gather semantics).
void kv_gather(void* scratch, const void* k_src, const void* v_src,
               const int* slot_ids, std::size_t num_pages,
               std::size_t page_size, std::size_t num_kv_heads,
               std::size_t head_dim, std::size_t elem_size,
               Stream* stream = nullptr);

// ---------------------------------------------------------------------------
// Adaptive dispatch (host-testable pure functions; mirrors p2p_gather's
// GatherDispatchMode). The CUDA one-shot p2p_kv_donate_layer reads the
// runtime mode at launch time; the host one-shot is always the fused
// reference (it cannot distinguish an SM kernel from a copy engine).
// ---------------------------------------------------------------------------

enum class DonateDispatchMode { kAdaptive = 0, kForceDirect = 1, kForceCopyEngine = 2 };

// Set the dispatch mode and the minimum page count at which the direct
// store kernel is eligible. Defaults: kAdaptive with min_pages = 1 (the
// fused direct-store kernel wins on H100 NVL from one page: it pays one
// launch instead of one gather plus per-page copies, and SM-driven peer
// writes match the copy engine's NVLink bandwidth without occupying it).
// Thread-safe (atomics); typically set once before launching.
void set_donate_dispatch(DonateDispatchMode mode = DonateDispatchMode::kAdaptive,
                         std::size_t min_pages_for_direct = 1);
std::pair<DonateDispatchMode, std::size_t> donate_dispatch_config();

// Estimated device time (microseconds) of the single-launch direct-store
// kernel: SM-driven peer writes over NVLink, no per-page driver cost, plus
// a launch floor. Flat in page count below the grid cap. `total_bytes` is
// the bytes written to peer (num_pages * page_size * 2 * num_kv_heads *
// head_dim * elem_size); zero bytes estimates 0.
double est_direct_store_us(std::size_t num_pages, std::size_t total_bytes);

// Estimated device time (microseconds) of the copy-engine fallback: one
// gather kernel into a local scratch (no NVLink) plus per-page
// cudaMemcpyAsync writes to peer, with a per-call floor that dominates
// small payloads. `total_bytes` is the bytes written to peer.
double est_copy_engine_donate_us(std::size_t num_pages, std::size_t total_bytes);

// Pure dispatch decision: true -> single-launch direct-store kernel, false
// -> gather-then-per-page-copy fallback. Honours the configured mode and
// min-pages floor; zero bytes never takes the kernel.
bool prefer_direct_store(std::size_t num_pages, std::size_t total_bytes);

// ---------------------------------------------------------------------------
// Prepared fused indexed-KV-to-peer donation plan (issue #36)
// ---------------------------------------------------------------------------
//
// The host reference mirrors the CUDA plan's contract exactly: validation
// and metadata staging happen ONCE at construction, execute() only enqueues
// one stream task that adds `destination_layer_offset_bytes` to every peer
// page base before writing. The device-slot variant (from_device_slots)
// skips slot validation (the CUDA path cannot read device memory without a
// sync) and borrows the caller's pointer instead of copying it.
//
// Unlike the one-shot, the plan's execute() is ALWAYS the direct store
// kernel -- no scratch allocation, ever -- so one plan can be reused across
// all model layers (40 for Qwen3-14B) with no per-layer allocation, H2D
// copy, or local packed-KV scratch. The copy-engine fallback is available
// via execute_via_scratch() (caller-owned scratch, also no per-call
// allocation) and via the adaptive one-shot p2p_kv_donate_layer.
class P2PKvDonatePlan {
 public:
  // Host-input plan: validate the slot map once (non-negativity and bounds;
  // uniqueness is NOT required -- gather semantics), copy peer bases and
  // slot_ids into owned storage. `num_slots` is the source capacity in
  // slots (every slot_id must be in [0, num_slots)); `num_kv_heads`,
  // `head_dim`, `elem_size` fix the per-slot byte size and (with
  // `page_size`) the per-token source/destination stride. `num_pages == 0`
  // is a valid no-op plan. Throws std::invalid_argument on a contract
  // violation (zero dimensions, non-BF16/FP16 elem_size,
  // negative/out-of-range slot).
  P2PKvDonatePlan(std::size_t num_slots, std::size_t num_kv_heads,
                  std::size_t head_dim, std::size_t elem_size,
                  const int* slot_ids, const void* const* peer_dst_ptrs,
                  std::size_t num_pages, std::size_t page_size);

  // Device-slot plan (int32): borrow the caller's CUDA `device_indices`
  // (shape [num_pages * page_size]) directly. No copy, no host validation
  // of slot contents -- the caller guarantees non-negative, in-range slots
  // (repeats are allowed) and keeps `device_indices` alive until the plan
  // is destroyed and all streams it ran on have completed. Metadata shape
  // (null pointers, zero dimensions, elem_size) IS still validated.
  P2PKvDonatePlan(from_device_slots_t, std::size_t num_slots,
                  std::size_t num_kv_heads, std::size_t head_dim,
                  std::size_t elem_size, const int* device_indices,
                  const void* const* peer_dst_ptrs, std::size_t num_pages,
                  std::size_t page_size);

  // Device-slot plan (int64): take the caller's CUDA int64 `device_indices`
  // (shape [num_pages * page_size], e.g. SGLang's torch.int64 radix-tree
  // indices), convert them in one device kernel to an OWNED int32 buffer,
  // and keep that. No D2H sync (the conversion is device-to-device); the
  // caller may free the int64 buffer as soon as the constructor returns.
  // Slot contents are NOT validated (same as the int32 device-slot plan).
  P2PKvDonatePlan(from_device_slots_int64_t, std::size_t num_slots,
                  std::size_t num_kv_heads, std::size_t head_dim,
                  std::size_t elem_size, const std::int64_t* device_indices,
                  const void* const* peer_dst_ptrs, std::size_t num_pages,
                  std::size_t page_size);

  P2PKvDonatePlan(const P2PKvDonatePlan&) = delete;
  P2PKvDonatePlan& operator=(const P2PKvDonatePlan&) = delete;

  std::size_t num_pages() const { return num_pages_; }
  std::size_t page_size() const { return page_size_; }
  std::size_t num_slots() const { return num_slots_; }
  std::size_t num_kv_heads() const { return num_kv_heads_; }
  std::size_t head_dim() const { return head_dim_; }
  std::size_t elem_size() const { return elem_size_; }
  // Bytes written to peer per execute() (num_pages * page_size * token_stride).
  std::size_t total_bytes() const { return total_bytes_; }
  // Bytes needed for the copy-engine fallback scratch
  // (num_pages * page_size * token_stride). Same as total_bytes(); kept as a
  // distinct accessor to document the fallback's scratch size.
  std::size_t scratch_bytes() const { return scratch_bytes_; }

  // Enqueue the fused direct-store donate for one layer from (k_src, v_src),
  // adding `destination_layer_offset_bytes` to every peer page base before
  // writing. Exactly one stream task (host reference) regardless of page
  // count, with NO scratch allocation. A null stream runs to completion
  // before returning. (k_src, v_src) must be non-null and each of capacity
  // `num_slots * num_kv_heads * head_dim * elem_size` bytes, laid out
  // [num_slots, num_kv_heads, head_dim]; this is the only execute-time
  // check. The plan must outlive the stream.
  void execute(const void* k_src, const void* v_src,
               std::size_t destination_layer_offset_bytes,
               Stream* stream = nullptr) const;

  // Copy-engine fallback: gather indexed local slots into the CALLER-OWNED
  // `scratch` (capacity scratch_bytes()), then copy each scratch page to its
  // peer destination (peer_dst_ptrs[p] + destination_layer_offset_bytes).
  // No per-call allocation (the caller owns the scratch). Produces the same
  // bytes as execute() -- use it when direct peer stores are unsupported or
  // lose to the copy-engine path. The plan must outlive the stream.
  void execute_via_scratch(const void* k_src, const void* v_src,
                           void* scratch,
                           std::size_t destination_layer_offset_bytes,
                           Stream* stream = nullptr) const;

 private:
  // Shared validation of metadata shape (used by all constructors; the
  // host-input constructor additionally validates slot contents).
  void validate_shape(const void* const* peer_dst_ptrs) const;

  std::size_t num_slots_;
  std::size_t num_kv_heads_;
  std::size_t head_dim_;
  std::size_t elem_size_;
  std::size_t page_size_;
  std::size_t num_pages_;
  std::size_t slot_bytes_;    // num_kv_heads * head_dim * elem_size
  std::size_t token_stride_;  // 2 * slot_bytes  ([K, V] per token)
  std::size_t layer_bytes_;   // page_size * token_stride (one layer per page)
  std::size_t total_bytes_;   // num_pages * layer_bytes_
  std::size_t scratch_bytes_; // == total_bytes_ (fallback scratch)
  // Peer page bases (the destination), copied at construction.
  std::vector<void*> peer_bases_;
  // Owned int32 slot map. Empty for the int32 device-slot variant (which
  // borrows the caller's pointer); filled by the host-input constructor
  // (host copy) and the int64 device-slot constructor (int64->int32 convert).
  std::vector<int> owned_slots_;
  const int* slot_ids_;  // points into owned_slots_ or the caller's pointer
};

}  // namespace vkernels::comm
