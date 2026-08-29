// vkernels/comm/cross_node_kv_c.cu
//
// C ABI implementation for the cross-node prepared fused KV restore / donate
// plans (issue #49). Wraps the C++ `vkernels::comm::cuda::CrossNodeKv
// RestorePlan` / `CrossNodeKvDonatePlan` entry points, catching every C++
// exception and converting it to a `vkernels_fi_status_t` so nothing is
// thrown across the language boundary. Built only with a CUDA toolkit --
// the entry points take a raw `cudaStream_t`.
//
// The caller obtains the imported device pointer with
// vkernels_fabric_import_device_ptr (fabric_import_c.cu) and hands it to
// the plan; the EXISTING *_execute_offset kernels then run UNCHANGED over
// cross-node memory. kHostBounce (imported_device_ptr == nullptr) gathers /
// scatters through a pinned scratch.
#include "vkernels/comm/cross_node_kv_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/cross_node_kv_cuda.hpp"
#  include "vkernels/comm/fabric_import.hpp"

#  include <exception>
#  include <stdexcept>

namespace {

vkernels::comm::FabricImportTransport to_transport(int t) {
  return static_cast<vkernels::comm::FabricImportTransport>(t);
}

}  // namespace

// ---------------------------------------------------------------------------
// CrossNodeKvRestorePlan
// ---------------------------------------------------------------------------

extern "C" vkernels_cross_node_kv_restore_plan_t*
vkernels_cross_node_kv_restore_plan_create(
    size_t num_slots, size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, size_t num_pages, size_t page_size,
    int transport, void* imported_device_ptr,
    vkernels_fi_status_t* status_out) {
  if (status_out != nullptr) *status_out = VKERNELS_FI_OK;
  try {
    return reinterpret_cast<vkernels_cross_node_kv_restore_plan_t*>(
        new vkernels::comm::cuda::CrossNodeKvRestorePlan(
            num_slots, num_kv_heads, head_dim, elem_size, slot_ids,
            num_pages, page_size, to_transport(transport),
            imported_device_ptr));
  } catch (const std::exception& e) {
    if (status_out != nullptr) {
      if (dynamic_cast<const std::invalid_argument*>(&e))
        *status_out = VKERNELS_FI_ERR_INVALID_ARGUMENT;
      else
        *status_out = VKERNELS_FI_ERR_INTERNAL;
    }
    return nullptr;
  }
}

extern "C" void vkernels_cross_node_kv_restore_plan_destroy(
    vkernels_cross_node_kv_restore_plan_t* plan) {
  delete reinterpret_cast<vkernels::comm::cuda::CrossNodeKvRestorePlan*>(plan);
}

extern "C" size_t vkernels_cross_node_kv_restore_plan_total_bytes(
    const vkernels_cross_node_kv_restore_plan_t* plan) {
  if (plan == nullptr) return 0;
  return reinterpret_cast<const vkernels::comm::cuda::CrossNodeKvRestorePlan*>(
             plan)
      ->total_bytes();
}

extern "C" size_t vkernels_cross_node_kv_restore_plan_bounce_bytes(
    const vkernels_cross_node_kv_restore_plan_t* plan) {
  if (plan == nullptr) return 0;
  return reinterpret_cast<const vkernels::comm::cuda::CrossNodeKvRestorePlan*>(
             plan)
      ->bounce_bytes();
}

extern "C" vkernels_fi_status_t vkernels_cross_node_kv_restore_plan_execute(
    vkernels_cross_node_kv_restore_plan_t* plan,
    void* k_dst, void* v_dst, size_t source_layer_offset_bytes,
    const void* pinned, cudaStream_t stream) {
  if (plan == nullptr) return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  try {
    reinterpret_cast<vkernels::comm::cuda::CrossNodeKvRestorePlan*>(plan)
        ->execute(k_dst, v_dst, source_layer_offset_bytes, pinned, stream);
  } catch (const std::invalid_argument&) {
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VKERNELS_FI_ERR_INTERNAL;
  }
  return VKERNELS_FI_OK;
}

// ---------------------------------------------------------------------------
// CrossNodeKvDonatePlan
// ---------------------------------------------------------------------------

extern "C" vkernels_cross_node_kv_donate_plan_t*
vkernels_cross_node_kv_donate_plan_create(
    size_t num_slots, size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, size_t num_pages, size_t page_size,
    int transport, void* imported_device_ptr,
    vkernels_fi_status_t* status_out) {
  if (status_out != nullptr) *status_out = VKERNELS_FI_OK;
  try {
    return reinterpret_cast<vkernels_cross_node_kv_donate_plan_t*>(
        new vkernels::comm::cuda::CrossNodeKvDonatePlan(
            num_slots, num_kv_heads, head_dim, elem_size, slot_ids,
            num_pages, page_size, to_transport(transport),
            imported_device_ptr));
  } catch (const std::exception& e) {
    if (status_out != nullptr) {
      if (dynamic_cast<const std::invalid_argument*>(&e))
        *status_out = VKERNELS_FI_ERR_INVALID_ARGUMENT;
      else
        *status_out = VKERNELS_FI_ERR_INTERNAL;
    }
    return nullptr;
  }
}

extern "C" void vkernels_cross_node_kv_donate_plan_destroy(
    vkernels_cross_node_kv_donate_plan_t* plan) {
  delete reinterpret_cast<vkernels::comm::cuda::CrossNodeKvDonatePlan*>(plan);
}

extern "C" size_t vkernels_cross_node_kv_donate_plan_total_bytes(
    const vkernels_cross_node_kv_donate_plan_t* plan) {
  if (plan == nullptr) return 0;
  return reinterpret_cast<const vkernels::comm::cuda::CrossNodeKvDonatePlan*>(
             plan)
      ->total_bytes();
}

extern "C" size_t vkernels_cross_node_kv_donate_plan_scratch_bytes(
    const vkernels_cross_node_kv_donate_plan_t* plan) {
  if (plan == nullptr) return 0;
  return reinterpret_cast<const vkernels::comm::cuda::CrossNodeKvDonatePlan*>(
             plan)
      ->scratch_bytes();
}

extern "C" size_t vkernels_cross_node_kv_donate_plan_bounce_bytes(
    const vkernels_cross_node_kv_donate_plan_t* plan) {
  if (plan == nullptr) return 0;
  return reinterpret_cast<const vkernels::comm::cuda::CrossNodeKvDonatePlan*>(
             plan)
      ->bounce_bytes();
}

extern "C" vkernels_fi_status_t vkernels_cross_node_kv_donate_plan_execute(
    vkernels_cross_node_kv_donate_plan_t* plan,
    const void* k_src, const void* v_src,
    size_t destination_layer_offset_bytes,
    void** out_pinned, cudaStream_t stream) {
  if (plan == nullptr) return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  try {
    reinterpret_cast<vkernels::comm::cuda::CrossNodeKvDonatePlan*>(plan)
        ->execute(k_src, v_src, destination_layer_offset_bytes, out_pinned,
                  stream);
  } catch (const std::invalid_argument&) {
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VKERNELS_FI_ERR_INTERNAL;
  }
  return VKERNELS_FI_OK;
}

#endif  // VKERNELS_C_HAS_CUDA
