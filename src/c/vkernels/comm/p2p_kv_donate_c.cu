// vkernels/comm/p2p_kv_donate_c.cu
//
// C ABI implementation for the fused indexed-KV-to-peer donation kernel
// (issue #36). Wraps the C++ `vkernels::comm::cuda::p2p_kv_donate_layer` /
// `_twostage` entry points and the `vkernels::comm::cuda::P2PKvDonatePlan`
// plan, catching every C++ exception and converting it to a
// `vkernels_status_t` so nothing is thrown across the language boundary.
#include "vkernels/comm/p2p_kv_donate_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/p2p_kv_donate_cuda.hpp"

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

vkernels_status_t vkernels_p2p_kv_donate_layer(
    const void* k_src, const void* v_src,
    const int* slot_ids,
    const void* const* peer_dst_ptrs,
    const size_t* dst_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::p2p_kv_donate_layer(
        k_src, v_src, slot_ids, peer_dst_ptrs, dst_page_offsets,
        num_pages, page_size, num_kv_heads, head_dim, elem_size, stream);
  });
}

vkernels_status_t vkernels_p2p_kv_donate_layer_twostage(
    const void* k_src, const void* v_src,
    const int* slot_ids,
    const void* const* peer_dst_ptrs,
    const size_t* dst_page_offsets,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim,
    size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::p2p_kv_donate_layer_twostage(
        k_src, v_src, slot_ids, peer_dst_ptrs, dst_page_offsets,
        num_pages, page_size, num_kv_heads, head_dim, elem_size, stream);
  });
}

vkernels_status_t vkernels_kv_gather(
    void* scratch, const void* k_src, const void* v_src,
    const int* slot_ids, size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::kv_gather(scratch, k_src, v_src, slot_ids,
                                    num_pages, page_size, num_kv_heads,
                                    head_dim, elem_size, stream);
  });
}

// Prepared plan: opaque handle over vkernels::comm::cuda::P2PKvDonatePlan.
// create maps a validation failure (std::invalid_argument) or device failure
// (std::runtime_error) to a status code and returns NULL; destroy and the
// execute helpers never throw across the boundary.
extern "C" vkernels_p2p_kv_donate_plan_t* vkernels_p2p_kv_donate_plan_create(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* slot_ids, const void* const* peer_dst_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out) {
  if (status_out) *status_out = VKERNELS_OK;
  try {
    return reinterpret_cast<vkernels_p2p_kv_donate_plan_t*>(
        new vkernels::comm::cuda::P2PKvDonatePlan(
            num_slots, num_kv_heads, head_dim, elem_size,
            slot_ids, peer_dst_ptrs, num_pages, page_size));
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

extern "C" vkernels_p2p_kv_donate_plan_t*
vkernels_p2p_kv_donate_plan_create_device_slots(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int* device_indices, const void* const* peer_dst_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out) {
  if (status_out) *status_out = VKERNELS_OK;
  try {
    return reinterpret_cast<vkernels_p2p_kv_donate_plan_t*>(
        new vkernels::comm::cuda::P2PKvDonatePlan(
            vkernels::comm::from_device_slots, num_slots, num_kv_heads,
            head_dim, elem_size, device_indices, peer_dst_ptrs, num_pages,
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

extern "C" vkernels_p2p_kv_donate_plan_t*
vkernels_p2p_kv_donate_plan_create_device_slots_int64(
    size_t num_slots,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    const int64_t* device_indices, const void* const* peer_dst_ptrs,
    size_t num_pages, size_t page_size,
    vkernels_status_t* status_out) {
  if (status_out) *status_out = VKERNELS_OK;
  try {
    return reinterpret_cast<vkernels_p2p_kv_donate_plan_t*>(
        new vkernels::comm::cuda::P2PKvDonatePlan(
            vkernels::comm::from_device_slots_int64, num_slots, num_kv_heads,
            head_dim, elem_size, device_indices, peer_dst_ptrs, num_pages,
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

extern "C" void vkernels_p2p_kv_donate_plan_destroy(
    vkernels_p2p_kv_donate_plan_t* plan) {
  delete reinterpret_cast<vkernels::comm::cuda::P2PKvDonatePlan*>(plan);
}

extern "C" size_t vkernels_p2p_kv_donate_plan_total_bytes(
    const vkernels_p2p_kv_donate_plan_t* plan) {
  return reinterpret_cast<const vkernels::comm::cuda::P2PKvDonatePlan*>(
      plan)->total_bytes();
}

extern "C" size_t vkernels_p2p_kv_donate_plan_scratch_bytes(
    const vkernels_p2p_kv_donate_plan_t* plan) {
  return reinterpret_cast<const vkernels::comm::cuda::P2PKvDonatePlan*>(
      plan)->scratch_bytes();
}

extern "C" vkernels_status_t vkernels_p2p_kv_donate_plan_execute_offset(
    vkernels_p2p_kv_donate_plan_t* plan,
    const void* k_src, const void* v_src,
    size_t destination_layer_offset_bytes,
    cudaStream_t stream) {
  try {
    reinterpret_cast<vkernels::comm::cuda::P2PKvDonatePlan*>(plan)->execute(
        k_src, v_src, destination_layer_offset_bytes, stream);
  } catch (const std::invalid_argument&) {
    return VKERNELS_ERR_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VKERNELS_ERR_INTERNAL;
  }
  return VKERNELS_OK;
}

extern "C" vkernels_status_t vkernels_p2p_kv_donate_plan_execute_via_scratch(
    vkernels_p2p_kv_donate_plan_t* plan,
    const void* k_src, const void* v_src,
    void* scratch,
    size_t destination_layer_offset_bytes,
    cudaStream_t stream) {
  try {
    reinterpret_cast<vkernels::comm::cuda::P2PKvDonatePlan*>(plan)->
        execute_via_scratch(k_src, v_src, scratch,
                            destination_layer_offset_bytes, stream);
  } catch (const std::invalid_argument&) {
    return VKERNELS_ERR_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VKERNELS_ERR_INTERNAL;
  }
  return VKERNELS_OK;
}

#endif  // VKERNELS_C_HAS_CUDA
