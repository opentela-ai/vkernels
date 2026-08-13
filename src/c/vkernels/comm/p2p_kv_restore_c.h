// vkernels/comm/p2p_kv_restore_c.h
//
// C ABI for the fused peer-to-indexed-KV restore kernel. Non-C++ consumers
// call these `extern "C"` entry points. Errors are RETURNED as codes — no
// C++ exceptions cross the ABI boundary.
//
// CUDA only: the functions take device pointers and a raw `cudaStream_t`.
// The header is includable from both C and C++; when the CUDA runtime headers
// are present the entry points are declared, otherwise only the status codes
// (which are CUDA-independent) are visible.
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

// Fused peer-to-indexed-KV restore for one layer. See p2p_kv_restore.hpp
// for the full contract.
//
// Returns VKERNELS_OK on success. On contract violation (null pointers,
// zero dimensions, non-unique slots) returns VKERNELS_ERR_INVALID_ARGUMENT.
// On device allocation or launch failure returns VKERNELS_ERR_INTERNAL.
vkernels_status_t vkernels_p2p_kv_restore_layer(
    void* k_dst, void* v_dst,
    const int* slot_ids,
    const void* const* peer_src_ptrs,
    const size_t* src_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream);

// Two-stage reference (peer→scratch→scatter) for correctness baselines.
vkernels_status_t vkernels_p2p_kv_restore_layer_twostage(
    void* k_dst, void* v_dst,
    const int* slot_ids,
    const void* const* peer_src_ptrs,
    const size_t* src_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream);

// Indexed KV scatter of an already-gathered contiguous scratch buffer (one
// [page_size, 2, num_kv_heads, head_dim] page per page, row-major). This is
// the second stage of the two-path, exposed so a caller that gathered pages
// with a prepared P2PGatherPlan2D can scatter them in one launch. The caller
// MUST guarantee unique, non-negative, in-range slots: like
// vkernels_p2p_kv_restore_layer the device path is check-free.
vkernels_status_t vkernels_p2p_kv_scatter(
    void* k_dst, void* v_dst, const void* scratch,
    const int* slot_ids, size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Prepared fused peer-to-indexed-KV restore plan (issue #27)
// ---------------------------------------------------------------------------
//
// A plan validates the slot map and destination geometry ONCE at create and
// uploads the page descriptors (and, for the host-input variant, the slot
// map) to a persistent per-device buffer; execute_offset() only launches ONE
// page-by-token-group kernel, adding `source_layer_offset_bytes` to every
// peer page base. The (k_dst, v_dst) destination is supplied at every
// execute_offset(), so one plan fans one run list out into a distinct K/V
// layer buffer per layer (the KVAAS restore pattern). Reuse one plan across
// all model layers (40 for Qwen3-14B) with no per-layer allocation or H2D
// copy.
//
// Three create entry points:
//  * vkernels_p2p_kv_restore_plan_create: `slot_ids` is a HOST int32 array.
//    The plan validates uniqueness, non-negativity and bounds
//    (slot < num_slots) once and copies the slot map to the device.
//  * vkernels_p2p_kv_restore_plan_create_device_slots: `slot_ids` is a
//    caller-owned CUDA int32 DEVICE pointer (e.g. SGLang's radix-tree
//    `device_indices`). The plan does NOT copy it and does NOT validate its
//    contents (reading device memory to validate would force a D2H sync).
//    The caller MUST guarantee unique, non-negative, in-range slots and MUST
//    keep `slot_ids` alive until the plan is destroyed and every stream it
//    was executed on has completed.
//  * vkernels_p2p_kv_restore_plan_create_device_slots_int64: `slot_ids` is a
//    caller-owned CUDA int64 DEVICE pointer (SGLang's indices are
//    torch.int64). The plan runs a one-time int64->int32 conversion kernel
//    and owns the int32 result (no D2H sync); the caller may free the int64
//    buffer as soon as the create call returns. Slot contents are NOT
//    validated (same as the int32 device-slot create).
//
// On success the create functions return a non-NULL handle and set
// *status_out to VKERNELS_OK; on a contract violation or device failure they
// return NULL and set *status_out. The plan is read-only after create, so
// execute_offset() may be called concurrently on several streams. Destroy
// the plan only after every such stream has been synchronised.
typedef struct vkernels_p2p_kv_restore_plan vkernels_p2p_kv_restore_plan_t;

vkernels_p2p_kv_restore_plan_t* vkernels_p2p_kv_restore_plan_create(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, const void* const* peer_src_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out);

vkernels_p2p_kv_restore_plan_t* vkernels_p2p_kv_restore_plan_create_device_slots(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* device_indices, const void* const* peer_src_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out);

vkernels_p2p_kv_restore_plan_t* vkernels_p2p_kv_restore_plan_create_device_slots_int64(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int64_t* device_indices, const void* const* peer_src_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out);

void vkernels_p2p_kv_restore_plan_destroy(
    vkernels_p2p_kv_restore_plan_t* plan);

vkernels_status_t vkernels_p2p_kv_restore_plan_execute_offset(
    vkernels_p2p_kv_restore_plan_t* plan, void* k_dst, void* v_dst,
    size_t source_layer_offset_bytes, cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
