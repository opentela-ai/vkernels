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

#endif  // VKERNELS_C_HAS_CUDA
