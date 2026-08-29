// vkernels/comm/cross_node_kv_c.h
//
// C ABI for the cross-node prepared fused KV restore / donate plans
// (issue #49). Non-C++ consumers call these `extern "C"` entry points.
// Errors are RETURNED as codes -- no C++ exception crosses the ABI
// boundary.
//
// CUDA only: the plan wraps the existing fused P2PKvRestorePlan /
// P2PKvDonatePlan (issues #27, #36) over an imported device pointer (from
// fabric_import_c.h's vkernels_fabric_import_device_ptr) on a real
// `cudaStream_t`. kHostBounce (imported_device_ptr == nullptr) gathers /
// scatters through a PINNED host scratch (cudaMallocHost, caller-supplied
// for restore / caller-owned after execute for donate) producing the SAME
// bytes the direct-store kernel would.
//
// The header is includable from both C and C++; when the CUDA runtime
// headers are present the entry points are declared, otherwise only the
// status codes and the cost-model / transport types (which are
// CUDA-independent, in fabric_import_c.h) are visible.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vkernels/comm/fabric_import_c.h"

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    define VKERNELS_C_HAS_CUDA 1
#    include <cuda_runtime.h>
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef VKERNELS_C_HAS_CUDA

// Cross-node prepared fused KV RESTORE plan (issue #49). The plan lays the
// per-page peer bases out as `imported_device_ptr + p * page_layer_bytes`
// (kFabricMapped / kSameNodePeer) and runs the EXISTING fused
// P2PKvRestorePlan over it on a real cudaStream -- ONE kernel launch, the
// peer loads now naming cross-node memory instead of a same-node peer. The
// kernels run UNCHANGED; only the peer base they dereference is
// cross-node. kHostBounce (imported_device_ptr == nullptr): scatter a
// caller-supplied PINNED layer (recv'd over the host transport) into the
// indexed local K/V slots with the existing kv_scatter -- the SAME bytes
// the direct-store kernel would produce.
//
// `slot_ids` is a HOST int32 array, validated for uniqueness /
// non-negativity / bounds (slot < num_slots) once at create and copied to
// the device. The plan is read-only after create, so execute() may be
// called concurrently on several streams. Destroy the plan only after
// every such stream has been synchronised.
typedef struct vkernels_cross_node_kv_restore_plan
    vkernels_cross_node_kv_restore_plan_t;

vkernels_cross_node_kv_restore_plan_t* vkernels_cross_node_kv_restore_plan_create(
    size_t num_slots, size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, size_t num_pages, size_t page_size,
    int transport, void* imported_device_ptr,
    vkernels_fi_status_t* status_out);

void vkernels_cross_node_kv_restore_plan_destroy(
    vkernels_cross_node_kv_restore_plan_t* plan);

// Bytes transferred by one execute call.  `bounce_bytes` is the pinned-host
// receive capacity required by VKERNELS_FI_TRANSPORT_HOST_BOUNCE.  Both return
// zero for a null plan.
size_t vkernels_cross_node_kv_restore_plan_total_bytes(
    const vkernels_cross_node_kv_restore_plan_t* plan);
size_t vkernels_cross_node_kv_restore_plan_bounce_bytes(
    const vkernels_cross_node_kv_restore_plan_t* plan);

// kFabricMapped / kSameNodePeer: ONE kernel over the imported pointer.
// kHostBounce: scatter `pinned` (one [num_pages, page_size, 2, heads,
// head_dim] layer, caller-supplied) into local slots. Returns
// VKERNELS_FI_OK on success, VKERNELS_FI_ERR_INVALID_ARGUMENT on a null
// plan / null k_dst|v_dst (num_pages > 0) / null pinned (host-bounce).
vkernels_fi_status_t vkernels_cross_node_kv_restore_plan_execute(
    vkernels_cross_node_kv_restore_plan_t* plan,
    void* k_dst, void* v_dst, size_t source_layer_offset_bytes,
    const void* pinned, cudaStream_t stream);

// Cross-node prepared fused KV DONATE plan (issue #49). kFabricMapped /
// kSameNodePeer: ONE kernel writing cross-node memory over
// imported_device_ptr (laid out as base + p * page_layer_bytes). kHostBounce
// (imported_device_ptr == nullptr): gather local slots into a PINNED
// scratch allocated by the plan (cudaMallocHost), returned through
// *out_pinned -- the SAME bytes the direct-store kernel would produce. The
// caller owns *out_pinned after execute and releases it with
// vkernels_fabric_bounce_scratch_free.
typedef struct vkernels_cross_node_kv_donate_plan
    vkernels_cross_node_kv_donate_plan_t;

vkernels_cross_node_kv_donate_plan_t* vkernels_cross_node_kv_donate_plan_create(
    size_t num_slots, size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, size_t num_pages, size_t page_size,
    int transport, void* imported_device_ptr,
    vkernels_fi_status_t* status_out);

void vkernels_cross_node_kv_donate_plan_destroy(
    vkernels_cross_node_kv_donate_plan_t* plan);

// Per-execute payload and fallback capacities.  All are equal for the current
// packed [page, token, K/V, head, dim] layout and return zero for a null plan.
size_t vkernels_cross_node_kv_donate_plan_total_bytes(
    const vkernels_cross_node_kv_donate_plan_t* plan);
size_t vkernels_cross_node_kv_donate_plan_scratch_bytes(
    const vkernels_cross_node_kv_donate_plan_t* plan);
size_t vkernels_cross_node_kv_donate_plan_bounce_bytes(
    const vkernels_cross_node_kv_donate_plan_t* plan);

// kFabricMapped / kSameNodePeer: ONE kernel over the imported pointer.
// kHostBounce: gather into *out_pinned (caller frees with
// vkernels_fabric_bounce_scratch_free). Returns VKERNELS_FI_OK on success,
// VKERNELS_FI_ERR_INVALID_ARGUMENT on a null plan / null k_src|v_src
// (num_pages > 0) / null out_pinned (host-bounce), VKERNELS_FI_ERR_INTERNAL
// on a pinned allocation failure.
vkernels_fi_status_t vkernels_cross_node_kv_donate_plan_execute(
    vkernels_cross_node_kv_donate_plan_t* plan,
    const void* k_src, const void* v_src,
    size_t destination_layer_offset_bytes,
    void** out_pinned, cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
