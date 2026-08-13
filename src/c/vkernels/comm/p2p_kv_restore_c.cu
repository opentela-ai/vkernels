// vkernels/comm/p2p_kv_restore_c.cu
//
// C ABI implementation for the fused peer-to-indexed-KV restore kernel.
// Wraps the C++ `vkernels::comm::cuda::p2p_kv_restore_layer` / `_twostage`
// entry points, catching every C++ exception and converting it to a
// `vkernels_status_t` so nothing is thrown across the language boundary.
#include "vkernels/comm/p2p_kv_restore_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/p2p_kv_restore_cuda.hpp"

#  include <cstddef>
#  include <exception>
#  include <functional>
#  include <stdexcept>

namespace {

vkernels_status_t wrap(const std::function<void()>& fn) {
  try {
    fn();
    return VKERNELS_OK;
  } catch (const std::invalid_argument&) {
    return VKERNELS_ERR_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VKERNELS_ERR_INTERNAL;
  } catch (...) {
    return VKERNELS_ERR_INTERNAL;
  }
}

}  // namespace

vkernels_status_t vkernels_p2p_kv_restore_layer(
    void* k_dst, void* v_dst,
    const int* slot_ids,
    const void* const* peer_src_ptrs,
    const size_t* src_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::p2p_kv_restore_layer(
        k_dst, v_dst, slot_ids, peer_src_ptrs, src_page_offsets,
        num_pages, page_size, num_kv_heads, head_dim, elem_size, stream);
  });
}

vkernels_status_t vkernels_p2p_kv_restore_layer_twostage(
    void* k_dst, void* v_dst,
    const int* slot_ids,
    const void* const* peer_src_ptrs,
    const size_t* src_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::p2p_kv_restore_layer_twostage(
        k_dst, v_dst, slot_ids, peer_src_ptrs, src_page_offsets,
        num_pages, page_size, num_kv_heads, head_dim, elem_size, stream);
  });
}

vkernels_status_t vkernels_p2p_kv_scatter(
    void* k_dst, void* v_dst, const void* scratch,
    const int* slot_ids, size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::kv_scatter(k_dst, v_dst, scratch, slot_ids,
                                     num_pages, page_size, num_kv_heads,
                                     head_dim, elem_size, stream);
  });
}

// Prepared plan: opaque handle over vkernels::comm::cuda::P2PKvRestorePlan.
// create maps a validation failure (std::invalid_argument) or device failure
// (std::runtime_error) to a status code and returns NULL; destroy and
// execute_offset never throw across the boundary.
extern "C" vkernels_p2p_kv_restore_plan_t* vkernels_p2p_kv_restore_plan_create(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, const void* const* peer_src_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out) {
  if (status_out) *status_out = VKERNELS_OK;
  try {
    return reinterpret_cast<vkernels_p2p_kv_restore_plan_t*>(
        new vkernels::comm::cuda::P2PKvRestorePlan(
            num_slots, num_kv_heads, head_dim, elem_size,
            slot_ids, peer_src_ptrs, num_pages, page_size));
  } catch (const std::exception& e) {
    if (status_out) {
      if (dynamic_cast<const std::invalid_argument*>(&e))
        *status_out = VKERNELS_ERR_INVALID_ARGUMENT;
      else
        *status_out = VKERNELS_ERR_INTERNAL;
    }
    return nullptr;
  }
}

extern "C" vkernels_p2p_kv_restore_plan_t*
vkernels_p2p_kv_restore_plan_create_device_slots(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* device_indices, const void* const* peer_src_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out) {
  if (status_out) *status_out = VKERNELS_OK;
  try {
    return reinterpret_cast<vkernels_p2p_kv_restore_plan_t*>(
        new vkernels::comm::cuda::P2PKvRestorePlan(
            vkernels::comm::from_device_slots, num_slots, num_kv_heads,
            head_dim, elem_size, device_indices, peer_src_ptrs, num_pages,
            page_size));
  } catch (const std::exception& e) {
    if (status_out) {
      if (dynamic_cast<const std::invalid_argument*>(&e))
        *status_out = VKERNELS_ERR_INVALID_ARGUMENT;
      else
        *status_out = VKERNELS_ERR_INTERNAL;
    }
    return nullptr;
  }
}

extern "C" vkernels_p2p_kv_restore_plan_t*
vkernels_p2p_kv_restore_plan_create_device_slots_int64(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int64_t* device_indices, const void* const* peer_src_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out) {
  if (status_out) *status_out = VKERNELS_OK;
  try {
    return reinterpret_cast<vkernels_p2p_kv_restore_plan_t*>(
        new vkernels::comm::cuda::P2PKvRestorePlan(
            vkernels::comm::from_device_slots_int64, num_slots, num_kv_heads,
            head_dim, elem_size, device_indices, peer_src_ptrs, num_pages,
            page_size));
  } catch (const std::exception& e) {
    if (status_out) {
      if (dynamic_cast<const std::invalid_argument*>(&e))
        *status_out = VKERNELS_ERR_INVALID_ARGUMENT;
      else
        *status_out = VKERNELS_ERR_INTERNAL;
    }
    return nullptr;
  }
}

extern "C" void vkernels_p2p_kv_restore_plan_destroy(
    vkernels_p2p_kv_restore_plan_t* plan) {
  delete reinterpret_cast<vkernels::comm::cuda::P2PKvRestorePlan*>(plan);
}

extern "C" vkernels_status_t vkernels_p2p_kv_restore_plan_execute_offset(
    vkernels_p2p_kv_restore_plan_t* plan, void* k_dst, void* v_dst,
    size_t source_layer_offset_bytes, cudaStream_t stream) {
  try {
    reinterpret_cast<vkernels::comm::cuda::P2PKvRestorePlan*>(plan)->execute(
        k_dst, v_dst, source_layer_offset_bytes, stream);
  } catch (const std::invalid_argument&) {
    return VKERNELS_ERR_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VKERNELS_ERR_INTERNAL;
  }
  return VKERNELS_OK;
}

#endif  // VKERNELS_C_HAS_CUDA
