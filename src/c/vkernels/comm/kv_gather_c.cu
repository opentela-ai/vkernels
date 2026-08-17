// vkernels/comm/kv_gather_c.cu
//
// C ABI implementation for the fused indexed K/V layer gather kernel
// (issue #2). Wraps the C++ `vkernels::comm::cuda::kv_gather_layer` /
// `kv_gather_layer_device_slots` entry points, catching every C++ exception
// and converting it to a `vkernels_status_t` so nothing is thrown across the
// language boundary.
#include "vkernels/comm/kv_gather_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/kv_gather_cuda.hpp"

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

extern "C" vkernels_status_t vkernels_kv_gather_layer(
    void* dst, const void* k_src, const void* v_src,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::kv_gather_layer(
        dst, k_src, v_src, slot_ids, slot_ids_int64 != 0,
        num_slots, num_pages, page_size, num_kv_heads, head_dim, elem_size,
        stream);
  });
}

extern "C" vkernels_status_t vkernels_kv_gather_layer_device_slots(
    void* dst, const void* k_src, const void* v_src,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::kv_gather_layer_device_slots(
        dst, k_src, v_src, slot_ids, slot_ids_int64 != 0,
        num_slots, num_pages, page_size, num_kv_heads, head_dim, elem_size,
        stream);
  });
}

#endif  // VKERNELS_C_HAS_CUDA
