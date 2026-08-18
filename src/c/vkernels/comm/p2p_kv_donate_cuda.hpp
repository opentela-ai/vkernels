// vkernels/comm/p2p_kv_donate_cuda.hpp
//
// CUDA-only declarations for the fused indexed-KV-to-peer donation kernel
// (issue #36). Kept separate from p2p_kv_donate.hpp because the CUDA entry
// points take `cudaStream_t`, which must not be exposed to host-only
// translation units. Included only when VKERNELS_HAS_CUDA; the definitions
// live in p2p_kv_donate.cu.
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/comm/p2p_kv_donate.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_kv;  // avoid pulling cuda_runtime.h into this header

namespace vkernels::comm::cuda {

// Fused indexed-KV-to-peer donation for one layer. See p2p_kv_donate.hpp
// for the full contract. `k_src` / `v_src` are local device pointers;
// `peer_dst_ptrs` are peer UVA pointers (raw `void*` from the caller). The
// call dispatches ADAPTIVELY (see vkernels::comm::prefer_direct_store):
// the default is one SM kernel that reads local slots and writes peer memory
// over NVLink with no scratch; the copy-engine fallback (gather into a
// per-call scratch plus per-page cudaMemcpyAsync) is taken when the model
// prefers it or when the mode is forced. Enqueued on `stream`; returns
// without synchronising.
void p2p_kv_donate_layer(const void* k_src, const void* v_src,
                         const int* slot_ids,
                         const void* const* peer_dst_ptrs,
                         const std::size_t* dst_page_offsets,
                         std::size_t num_pages, std::size_t page_size,
                         std::size_t num_kv_heads, std::size_t head_dim,
                         std::size_t elem_size,
                         cudaStream_t_kv stream);

// Two-stage reference on the GPU:
//   1. indexed KV gather kernel (local slots -> scratch), then
//   2. cudaMemcpyAsync per page (scratch -> peer).
// Useful as a correctness baseline and for measuring the scratch cost.
void p2p_kv_donate_layer_twostage(const void* k_src, const void* v_src,
                                  const int* slot_ids,
                                  const void* const* peer_dst_ptrs,
                                  const std::size_t* dst_page_offsets,
                                  std::size_t num_pages, std::size_t page_size,
                                  std::size_t num_kv_heads, std::size_t head_dim,
                                  std::size_t elem_size,
                                  cudaStream_t_kv stream);

// Gather a flat scratch buffer (one [page_size, 2, num_kv_heads, head_dim]
// page per page, contiguous) from indexed local K/V sources — the first
// stage of the two-path, exposed so a caller that wants the copy-engine
// fallback can gather once (e.g. with a prepared plan) and then issue the
// peer copy separately. `slot_ids` is a caller-owned DEVICE pointer
// (shape [num_pages * page_size]); the caller MUST keep `slot_ids` and the
// sources alive until the kernel completes — the device path is check-free
// (the host reference kv_gather is the validating oracle). One kernel
// launch, no upload.
void kv_gather(void* scratch, const void* k_src, const void* v_src,
               const int* slot_ids, std::size_t num_pages,
               std::size_t page_size, std::size_t num_kv_heads,
               std::size_t head_dim, std::size_t elem_size,
               cudaStream_t_kv stream);

// ---------------------------------------------------------------------------
// Prepared fused indexed-KV-to-peer donation plan (issue #36)
// ---------------------------------------------------------------------------
//
// Same semantics as vkernels::comm::P2PKvDonatePlan (validate once at
// construction, execute() only enqueues) with the CUDA specifics: the
// constructor also uploads the page descriptors — and, for the host-input
// variant, the slot map — to a persistent per-device buffer with a
// synchronous cudaMemcpy (one-time cost, no stream association so concurrent
// execute() on arbitrary streams is race-free). The int64 device-slot
// variant additionally runs a one-time int64->int32 conversion kernel (no
// D2H sync) and owns the int32 result. execute(k_src, v_src,
// destination_layer_offset_bytes, stream) supplies the per-layer source and
// launches ONE page-by-token-group kernel that adds the offset to every
// peer page base before writing, so the same plan is reused across all
// model layers with zero per-layer allocation or H2D copy. No scratch is
// allocated, ever.
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
// host-input and int64 variants, the device slot map); destroy the plan only
// after every stream it was executed on has been synchronised. For the int32
// device-slot variant the caller's `device_indices` buffer must outlive the
// plan for the same reason (the int64 variant owns its copy and imposes no
// such constraint). Read-only after construction, so concurrent execute() on
// several streams is safe (each call supplies its own source).
class P2PKvDonatePlan {
 public:
  P2PKvDonatePlan(std::size_t num_slots, std::size_t num_kv_heads,
                  std::size_t head_dim, std::size_t elem_size,
                  const int* slot_ids, const void* const* peer_dst_ptrs,
                  std::size_t num_pages, std::size_t page_size);
  P2PKvDonatePlan(vkernels::comm::from_device_slots_t, std::size_t num_slots,
                  std::size_t num_kv_heads, std::size_t head_dim,
                  std::size_t elem_size, const int* device_indices,
                  const void* const* peer_dst_ptrs, std::size_t num_pages,
                  std::size_t page_size);
  P2PKvDonatePlan(vkernels::comm::from_device_slots_int64_t,
                  std::size_t num_slots, std::size_t num_kv_heads,
                  std::size_t head_dim, std::size_t elem_size,
                  const std::int64_t* device_indices,
                  const void* const* peer_dst_ptrs, std::size_t num_pages,
                  std::size_t page_size);
  ~P2PKvDonatePlan();
  P2PKvDonatePlan(const P2PKvDonatePlan&) = delete;
  P2PKvDonatePlan& operator=(const P2PKvDonatePlan&) = delete;

  std::size_t num_pages() const;
  std::size_t num_kv_heads() const;
  std::size_t head_dim() const;
  std::size_t elem_size() const;
  std::size_t num_slots() const;
  std::size_t page_size() const;
  std::size_t total_bytes() const;
  std::size_t scratch_bytes() const;

  // Enqueue the fused direct-store donate for one layer from (k_src, v_src),
  // adding `destination_layer_offset_bytes` to every peer page base. One
  // kernel launch; no metadata validation, allocation or H2D copy (a single
  // null-source guard is the only execute-time check). When the offset is
  // not a multiple of 16 the kernel falls back to the scalar (one-byte)
  // path, sized to the full slot width.
  void execute(const void* k_src, const void* v_src,
               std::size_t destination_layer_offset_bytes,
               cudaStream_t_kv stream) const;

  // Copy-engine fallback: gather indexed local slots into the CALLER-OWNED
  // device `scratch` (capacity scratch_bytes()), then copy each scratch page
  // to its peer destination (peer_dst_ptrs[p] +
  // destination_layer_offset_bytes) with one cudaMemcpyAsync per page. No
  // per-call allocation. Produces the same bytes as execute() -- use it when
  // direct peer stores are unsupported or lose to the copy-engine path.
  void execute_via_scratch(const void* k_src, const void* v_src,
                           void* scratch,
                           std::size_t destination_layer_offset_bytes,
                           cudaStream_t_kv stream) const;

 private:
  enum class SlotSource {
    HostValidated,
    DeviceInt32Borrowed,
    DeviceInt64Converted
  };
  struct Impl;
  Impl* init(SlotSource mode, std::size_t num_slots,
             std::size_t num_kv_heads, std::size_t head_dim,
             std::size_t elem_size, const void* slot_ids,
             const void* const* peer_dst_ptrs, std::size_t num_pages,
             std::size_t page_size);
  Impl* impl_;
};

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
