// vkernels/comm/kv_scatter_c.cu
//
// C ABI implementation for the fused indexed K/V layer scatter kernel
// (issue #1). Wraps the C++ `vkernels::comm::cuda::kv_scatter_layer` /
// `kv_scatter_layer_device_slots` entry points, catching every C++ exception
// and converting it to a `vkernels_status_t` so nothing is thrown across the
// language boundary.
#include "vkernels/comm/kv_scatter_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/kv_scatter_cuda.hpp"

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

extern "C" vkernels_status_t vkernels_kv_scatter_layer(
    void* k_dst, void* v_dst,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots,
    const void* src,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::kv_scatter_layer(
        k_dst, v_dst, slot_ids, slot_ids_int64 != 0,
        num_slots, src, num_pages, page_size,
        num_kv_heads, head_dim, elem_size, stream);
  });
}

extern "C" vkernels_status_t vkernels_kv_scatter_layer_device_slots(
    void* k_dst, void* v_dst,
    const void* slot_ids, int slot_ids_int64,
    size_t num_slots,
    const void* src,
    size_t num_pages, size_t page_size,
    size_t num_kv_heads, size_t head_dim, size_t elem_size,
    cudaStream_t stream) {
  return wrap([&] {
    vkernels::comm::cuda::kv_scatter_layer_device_slots(
        k_dst, v_dst, slot_ids, slot_ids_int64 != 0,
        num_slots, src, num_pages, page_size,
        num_kv_heads, head_dim, elem_size, stream);
  });
}

#endif  // VKERNELS_C_HAS_CUDA
