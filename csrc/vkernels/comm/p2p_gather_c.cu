// vkernels/comm/p2p_gather_c.cu — thin `extern "C"` wrapper over the C++ API.
//
// Compiled only when a CUDA toolkit is available (it is a .cu so the host
// compiler still processes it, but the entry points are only declared when
// <cuda_runtime.h> was found by p2p_gather_c.h). The host reference in
// p2p_gather.cpp has no C ABI: a non-C++ consumer on a GPU host needs the
// single-launch primitive, not the host worker-thread seam. Every C++
// exception thrown by the staging validators (std::invalid_argument via
// VK_EXPECTS) or the launch path (std::runtime_error via VK_ENSURES) is
// caught here and folded into a status code, so no exception ever crosses
// the ABI boundary.
#include "vkernels/comm/p2p_gather_c.h"

#if defined(VKERNELS_C_HAS_CUDA)

// The wrapper functions below are HOST-ONLY: they use C++ exceptions
// (try/catch, std::exception, dynamic_cast) to fold vkernels contract
// violations into status codes. nvcc compiles a .cu file in both host and
// device modes; guard the whole body so the device pass emits nothing.
#  ifndef __CUDA_ARCH__

#  include "vkernels/comm/p2p_gather_cuda.hpp"

#  include <exception>
#  include <stdexcept>

namespace {

// Map a caught C++ exception to an ABI status code. Contract violations
// (VK_EXPECTS) arrive as std::invalid_argument; launch/post-condition
// failures (VK_ENSURES) arrive as std::runtime_error. Anything else is
// reported as the most conservative "something was wrong" code.
vkernels_status_t to_status(const std::exception& e) {
  if (dynamic_cast<const std::invalid_argument*>(&e)) return VKERNELS_ERR_INVALID_ARGUMENT;
  if (dynamic_cast<const std::runtime_error*>(&e)) return VKERNELS_ERR_INTERNAL;
  return VKERNELS_ERR_INTERNAL;
}

}  // namespace

extern "C" vkernels_status_t vkernels_p2p_gather_runs(
    uint8_t* dst, size_t dst_capacity, const void* const* src_ptrs,
    const size_t* dst_offsets, const size_t* lengths, size_t num_runs,
    cudaStream_t stream) {
  try {
    vkernels::comm::cuda::p2p_gather_runs(dst, dst_capacity, src_ptrs,
                                          dst_offsets, lengths, num_runs,
                                          stream);
  } catch (const std::exception& e) {
    return to_status(e);
  }
  return VKERNELS_OK;
}

extern "C" vkernels_status_t vkernels_p2p_gather_runs_2d(
    uint8_t* dst, size_t dst_capacity, const vkernels_gather_2d_run_t* runs,
    size_t num_runs, cudaStream_t stream) {
  try {
    // Re-wrap the POD descriptors as Gather2DRun. The two structs are layout-
    // compatible by construction (same members, same order, no padding
    // surprises: the largest member is a pointer, so alignment is uniform).
    vkernels::comm::cuda::p2p_gather_runs_2d(
        dst, dst_capacity,
        reinterpret_cast<const vkernels::comm::Gather2DRun*>(runs), num_runs,
        stream);
  } catch (const std::exception& e) {
    return to_status(e);
  }
  return VKERNELS_OK;
}

#  endif  // __CUDA_ARCH__
#endif  // defined(VKERNELS_C_HAS_CUDA)
