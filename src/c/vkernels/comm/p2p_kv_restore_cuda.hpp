// vkernels/comm/p2p_kv_restore_cuda.hpp
//
// CUDA-only declarations for the fused peer-to-indexed-KV restore kernel.
// Kept separate from p2p_kv_restore.hpp because the CUDA entry points take
// `cudaStream_t`, which must not be exposed to host-only translation units.
// Included only when VKERNELS_HAS_CUDA; the definitions live in
// p2p_kv_restore.cu.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/comm/p2p_kv_restore.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_kv;  // avoid pulling cuda_runtime.h into this header

namespace vkernels::comm::cuda {

// Fused peer-to-indexed-KV restore for one layer. See p2p_kv_restore.hpp
// for the full contract. `k_dst` / `v_dst` are local device pointers;
// `peer_src_ptrs` are peer UVA pointers (raw `void*` from the caller).
// Enqueued on `stream`; returns without synchronising.
//
// The CUDA implementation resolves effective page pointers
// (`peer_src_ptrs[p] + src_page_offsets[p]`) on the host, stages them into
// a device-side `page_ptrs` array with a streaming H2D copy, and launches
// one kernel that reads peer memory over NVLink and writes directly into
// the indexed K/V slots. No scratch buffer is allocated.
void p2p_kv_restore_layer(void* k_dst, void* v_dst,
                          const int* slot_ids,
                          const void* const* peer_src_ptrs,
                          const std::size_t* src_page_offsets,
                          std::size_t num_pages, std::size_t page_size,
                          std::size_t num_kv_heads, std::size_t head_dim,
                          std::size_t elem_size,
                          cudaStream_t_kv stream);

// Two-stage reference on the GPU:
//   1. cudaMemcpyAsync per page (peer → scratch), then
//   2. indexed KV scatter kernel (scratch → slots).
// Useful as a correctness baseline and for measuring the scratch cost.
void p2p_kv_restore_layer_twostage(void* k_dst, void* v_dst,
                                   const int* slot_ids,
                                   const void* const* peer_src_ptrs,
                                   const std::size_t* src_page_offsets,
                                   std::size_t num_pages, std::size_t page_size,
                                   std::size_t num_kv_heads, std::size_t head_dim,
                                   std::size_t elem_size,
                                   cudaStream_t_kv stream);

// Scatter a flat scratch buffer (one [page_size, 2, num_kv_heads, head_dim]
// page per page, contiguous) into indexed K/V destinations — the second
// stage of the two-path, exposed so a caller that already gathered pages
// with a prepared P2PGatherPlan2D can scatter them in one launch. `slot_ids`
// is a caller-owned DEVICE pointer (shape [num_pages * page_size]); the
// caller MUST guarantee unique, non-negative, in-range slots and keep both
// `slot_ids` and `scratch` alive until the kernel completes — like
// p2p_kv_restore_layer the device path is check-free (the host reference
// kv_scatter is the validating oracle). One kernel launch, no upload.
void kv_scatter(void* k_dst, void* v_dst, const void* scratch,
                const int* slot_ids, std::size_t num_pages,
                std::size_t page_size, std::size_t num_kv_heads,
                std::size_t head_dim, std::size_t elem_size,
                cudaStream_t_kv stream);

// ---------------------------------------------------------------------------
// Prepared fused peer-to-indexed-KV restore plan (issue #27)
// ---------------------------------------------------------------------------
//
// Same semantics as vkernels::comm::P2PKvRestorePlan (validate once at
// construction, execute() only enqueues) with the CUDA specifics: the
// constructor also uploads the page descriptors — and, for the host-input
// variant, the slot map — to a persistent per-device buffer with a
// synchronous cudaMemcpy (one-time cost, no stream association so concurrent
// execute() on arbitrary streams is race-free). execute() launches ONE
// page-by-token-group kernel that adds `source_layer_offset_bytes` to every
// peer page base before reading, so the same plan is reused across all model
// layers with zero per-layer allocation or H2D copy.
//
// Kernel grid (vs the one-shot one-block-per-page kernel): grid.x tiles each
// page's unit range (one unit = one 16-byte K chunk + one 16-byte V chunk on
// the vectorized path, one byte of each on the scalar fallback), grid.y =
// num_pages, blockDim = 256. A 64-token / 8-head / dim-128 / BF16 page is
// 64 * 128 = 8192 units -> 32 blocks/page, so 16+ pages fill an H100's 132
// SMs and even 1 page launches 32 blocks (vs 1 before). grid.y is capped at
// 65535; for the KVAAS page counts (<=192) this is never reached.
//
// Lifetime: the plan owns the device descriptor buffer (and, for the
// host-input variant, the device slot map); destroy the plan only after
// every stream it was executed on has been synchronised. For the device-slot
// variant the caller's `device_indices` buffer must outlive the plan for the
// same reason. Read-only after construction, so concurrent execute() on
// several streams is safe.
class P2PKvRestorePlan {
 public:
  P2PKvRestorePlan(void* k_dst, void* v_dst, std::size_t num_slots,
                   std::size_t num_kv_heads, std::size_t head_dim,
                   std::size_t elem_size, const int* slot_ids,
                   const void* const* peer_src_ptrs, std::size_t num_pages,
                   std::size_t page_size);
  P2PKvRestorePlan(vkernels::comm::from_device_slots_t, void* k_dst, void* v_dst,
                   std::size_t num_slots, std::size_t num_kv_heads,
                   std::size_t head_dim, std::size_t elem_size,
                   const int* device_indices,
                   const void* const* peer_src_ptrs, std::size_t num_pages,
                   std::size_t page_size);
  ~P2PKvRestorePlan();
  P2PKvRestorePlan(const P2PKvRestorePlan&) = delete;
  P2PKvRestorePlan& operator=(const P2PKvRestorePlan&) = delete;

  std::size_t num_pages() const;
  std::size_t num_kv_heads() const;
  std::size_t head_dim() const;
  std::size_t elem_size() const;
  std::size_t num_slots() const;
  std::size_t page_size() const;
  std::size_t total_bytes() const;

  // Enqueue the fused restore for one layer, adding `source_layer_offset_bytes`
  // to every peer page base. One kernel launch; no validation, allocation or
  // H2D copy. When the offset is not a multiple of 16 the kernel falls back
  // to the scalar (one-byte) path, sized to the full slot width.
  void execute(std::size_t source_layer_offset_bytes,
               cudaStream_t_kv stream) const;

 private:
  struct Impl;
  // Shared construction body (see p2p_kv_restore.cu). Validating form
  // checks slot uniqueness/bounds and owns a device copy; non-validating
  // form borrows the caller's device pointer. A member so it may name the
  // private Impl type.
  template <bool ValidateSlots>
  Impl* init(void* k_dst, void* v_dst, std::size_t num_slots,
             std::size_t num_kv_heads, std::size_t head_dim,
             std::size_t elem_size, const int* slot_ids,
             const void* const* peer_src_ptrs, std::size_t num_pages,
             std::size_t page_size);
  Impl* impl_;
};

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
