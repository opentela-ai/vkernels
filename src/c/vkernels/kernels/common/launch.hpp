// vkernels/kernels/common/launch.hpp — the CUDA/HIP launch-and-check idiom.
//
// Every device launcher in this tree ended with the same three lines:
// ``kernel<<<grid, block>>>(args...)`` then ``cudaGetLastError`` funneled
// into VK_ENSURES with a hand-written message. :func:`vkernels::kernels::common::launch`
// keeps the idiom in one place (the ``common/`` layering borrowed from
// DeepGEMM's include tree), so a launcher is one call and a missing check
// is impossible by construction.
//
// Runtime selection:
//   * ROCm HIP TUs (hip-clang defines __HIP__): HIP runtime, hipStream_t.
//   * CUDA TUs, including the HIP-on-CUDA nvcc build of .hip sources (the
//     cuda_compat shim maps hip* onto cuda*, so the same call sites work):
//     CUDA runtime, cudaStream_t.
// Host-only TUs see nothing. The two runtimes live in distinct inline
// namespaces so a binary that links both a .cu and a .hip TU has no ODR clash.
#pragma once

#include <cstddef>
#include <string>
#include <utility>

#if defined(__HIP__) || defined(__HIPCC__)
#  include <hip/hip_runtime.h>
#  define VKERNELS_LAUNCH_RT_HIP 1
#elif VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>
#  define VKERNELS_LAUNCH_RT_CUDA 1
#endif

#if defined(VKERNELS_LAUNCH_RT_HIP) || defined(VKERNELS_LAUNCH_RT_CUDA)

#  include "vkernels/util/error.hpp"

namespace vkernels::kernels::common {

inline std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Threads-per-block for a flat 1-D elementwise-style kernel.
inline int default_block_size() { return 256; }

#  if defined(VKERNELS_LAUNCH_RT_HIP)
inline namespace hip_rt {
using stream_t = hipStream_t;
inline void check_launch(const char* what) {
  const hipError_t err = hipGetLastError();
  VK_ENSURES(err == hipSuccess,
             std::string(what) + " launch failed: " + hipGetErrorString(err));
}
#  else
inline namespace cuda_rt {
using stream_t = cudaStream_t;
inline void check_launch(const char* what) {
  const cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess,
             std::string(what) + " launch failed: " + cudaGetErrorString(err));
}
#  endif

// Launch ``kernel`` with an explicit dynamic-shared-memory size and stream,
// then fail through VK_ENSURES on any launch error. ``args`` convert to the
// kernel's parameter types exactly as in a direct ``<<<>>>`` call.
template <typename... KArgs, typename... Args>
void launch_on(void (*kernel)(KArgs...), dim3 grid, dim3 block, std::size_t shmem,
               stream_t stream, const char* what, Args&&... args) {
  kernel<<<grid, block, shmem, stream>>>(std::forward<Args>(args)...);
  check_launch(what);
}

// Launch ``kernel`` with ``args...`` on the default stream and fail through
// VK_ENSURES on any launch error. ``what`` names the operation.
template <typename... Args>
void launch(void (*kernel)(Args...), dim3 grid, dim3 block, const char* what,
            Args... args) {
  kernel<<<grid, block>>>(args...);
  check_launch(what);
}

// 1-D convenience: grid sized for ``n`` elements at ``threads`` per block.
template <typename... Args>
void launch_1d(void (*kernel)(Args...), std::size_t n, const char* what,
               Args... args) {
  const int threads = default_block_size();
  const dim3 grid(static_cast<unsigned>(ceil_div(n, static_cast<std::size_t>(threads))));
  launch(kernel, grid, dim3(static_cast<unsigned>(threads)), what, args...);
}

}  // namespace hip_rt / cuda_rt
}  // namespace vkernels::kernels::common

#endif  // VKERNELS_LAUNCH_RT_HIP || VKERNELS_LAUNCH_RT_CUDA
