// vkernels/comm/p2p_kv_donate_c.h
//
// C ABI for the fused indexed-KV-to-peer donation kernel (issue #36).
// Non-C++ consumers call these `extern "C"` entry points. Errors are
// RETURNED as codes -- no C++ exceptions cross the ABI boundary.
//
// CUDA only: the functions take device pointers and a raw `cudaStream_t`.
// The header is includable from both C and C++; when the CUDA runtime
// headers are present the entry points are declared, otherwise only the
// status codes (which are CUDA-independent) are visible.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    define VKERNELS_C_HAS_CUDA 1
#    include <cuda_runtime.h>
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Status codes mirroring vkernels::Code.
typedef enum {
  VKERNELS_OK = 0,
  VKERNELS_ERR_INVALID_ARGUMENT = 1,
  VKERNELS_ERR_OUT_OF_RANGE = 2,
  VKERNELS_ERR_UNSUPPORTED = 3,
  VKERNELS_ERR_INTERNAL = 4,
} vkernels_status_t;

#ifdef VKERNELS_C_HAS_CUDA

// Fused indexed-KV-to-peer donation for one layer. See p2p_kv_donate.hpp
// for the full contract. The CUDA implementation dispatches ADAPTIVELY:
// a single SM kernel reads local slots and writes peer memory over NVLink
// (no scratch) by default, falling back to gather-then-per-page-copy when
// the copy engine wins or direct peer stores are unsupported.
//
// Returns VKERNELS_OK on success. On contract violation (null pointers,
// zero dimensions, negative slots) returns VKERNELS_ERR_INVALID_ARGUMENT.
// On device allocation or launch failure returns VKERNELS_ERR_INTERNAL.
vkernels_status_t vkernels_p2p_kv_donate_layer(
    const void* k_src, const void* v_src,
    const int* slot_ids,
    const void* const* peer_dst_ptrs,
    const size_t* dst_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream);

// Two-stage reference (gather->scratch->peer copy) for correctness
// baselines and as the copy-engine fallback.
vkernels_status_t vkernels_p2p_kv_donate_layer_twostage(
    const void* k_src, const void* v_src,
    const int* slot_ids,
    const void* const* peer_dst_ptrs,
    const size_t* dst_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream);

// Indexed KV gather of local paged slots into a contiguous scratch buffer
// (one [page_size, 2, num_kv_heads, head_dim] page per page, row-major).
// This is the first stage of the two-path, exposed so a caller that wants
// the copy-engine fallback can gather once and then issue the peer copy
// separately. The caller MUST guarantee non-negative, in-range slots
// (repeats allowed) and keep both `slot_ids` and the sources alive until
// the kernel completes -- like vkernels_p2p_kv_donate_layer the device
// path is check-free.
vkernels_status_t vkernels_kv_gather(
    void* scratch, const void* k_src, const void* v_src,
    const int* slot_ids, size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Prepared fused indexed-KV-to-peer donation plan (issue #36)
// ---------------------------------------------------------------------------
//
// A plan validates the slot map and destination geometry ONCE at create and
// uploads the page descriptors (and, for the host-input variant, the slot
// map) to a persistent per-device buffer; execute_offset() only launches ONE
// page-by-token-group kernel that adds `destination_layer_offset_bytes` to
// every peer page base before writing. The (k_src, v_src) source is supplied
// at every execute_offset(), so one plan reads a distinct K/V layer buffer
// per layer (the KVAAS donate pattern) and writes into the peer pages. Reuse
// one plan across all model layers (40 for Qwen3-14B) with no per-layer
// allocation, H2D copy, or local packed-KV scratch.
//
// execute_via_scratch() is the copy-engine fallback: it gathers the indexed
// local slots into a CALLER-OWNED device scratch (capacity
// vkernels_p2p_kv_donate_plan_scratch_bytes) and then issues one
// cudaMemcpyAsync per page to peer. Use it when direct peer stores are
// unsupported or lose to the copy-engine path; it produces the same bytes.
//
// Three create entry points:
//  * vkernels_p2p_kv_donate_plan_create: `slot_ids` is a HOST int32 array.
//    The plan validates non-negativity and bounds (slot < num_slots) once
//    and copies the slot map to the device. Uniqueness is NOT required
//    (gather semantics).
//  * vkernels_p2p_kv_donate_plan_create_device_slots: `slot_ids` is a
//    caller-owned CUDA int32 DEVICE pointer (e.g. SGLang's radix-tree
//    `device_indices`). The plan does NOT copy it and does NOT validate its
//    contents (reading device memory to validate would force a D2H sync).
//    The caller MUST guarantee non-negative, in-range slots (repeats allowed)
//    and MUST keep `slot_ids` alive until the plan is destroyed and every
//    stream it was executed on has completed.
//  * vkernels_p2p_kv_donate_plan_create_device_slots_int64: `slot_ids` is a
//    caller-owned CUDA int64 DEVICE pointer (SGLang's indices are
//    torch.int64). The plan runs a one-time int64->int32 conversion kernel
//    and owns the int32 result (no D2H sync); the caller may free the int64
//    buffer as soon as the create call returns. Slot contents are NOT
//    validated (same as the int32 device-slot create).
//
// On success the create functions return a non-NULL handle and set
// *status_out to VKERNELS_OK; on a contract violation or device failure they
// return NULL and set *status_out. The plan is read-only after create, so
// execute_offset()/execute_via_scratch() may be called concurrently on
// several streams. Destroy the plan only after every such stream has been
// synchronised.
typedef struct vkernels_p2p_kv_donate_plan vkernels_p2p_kv_donate_plan_t;

vkernels_p2p_kv_donate_plan_t* vkernels_p2p_kv_donate_plan_create(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, const void* const* peer_dst_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out);

vkernels_p2p_kv_donate_plan_t* vkernels_p2p_kv_donate_plan_create_device_slots(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* device_indices, const void* const* peer_dst_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out);

vkernels_p2p_kv_donate_plan_t* vkernels_p2p_kv_donate_plan_create_device_slots_int64(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int64_t* device_indices, const void* const* peer_dst_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out);

void vkernels_p2p_kv_donate_plan_destroy(
    vkernels_p2p_kv_donate_plan_t* plan);

// Bytes written to peer per execute (num_pages * page_size * token_stride).
size_t vkernels_p2p_kv_donate_plan_total_bytes(
    const vkernels_p2p_kv_donate_plan_t* plan);

// Bytes needed for the copy-engine fallback scratch (== total_bytes).
size_t vkernels_p2p_kv_donate_plan_scratch_bytes(
    const vkernels_p2p_kv_donate_plan_t* plan);

// Fused direct-store donate for one layer. No scratch allocation.
vkernels_status_t vkernels_p2p_kv_donate_plan_execute_offset(
    vkernels_p2p_kv_donate_plan_t* plan,
    const void* k_src, const void* v_src,
    size_t destination_layer_offset_bytes,
    cudaStream_t stream);

// Copy-engine fallback: gather into the caller-owned device `scratch`
// (capacity scratch_bytes), then one cudaMemcpyAsync per page to peer.
// Produces the same bytes as execute_offset().
vkernels_status_t vkernels_p2p_kv_donate_plan_execute_via_scratch(
    vkernels_p2p_kv_donate_plan_t* plan,
    const void* k_src, const void* v_src,
    void* scratch,
    size_t destination_layer_offset_bytes,
    cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
