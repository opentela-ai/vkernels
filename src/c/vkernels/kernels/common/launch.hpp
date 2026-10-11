// vkernels/kernels/common/launch.hpp — the CUDA launch-and-check idiom.
//
// Every device launcher in this tree ended with the same three lines:
// ``kernel<<<grid, block>>>(args...)`` then ``cudaGetLastError`` funneled
// into VK_ENSURES with a hand-written message. :func:`vkernels::kernels::common::launch`
// keeps the idiom in one place (the ``common/`` layering borrowed from
// DeepGEMM's include tree), so a launcher is one call and a missing check
// is impossible by construction.
//
// CUDA-only content: include from translation units compiled with a
// toolkit (the .cu files guard themselves with VKERNELS_HAS_CUDA the same
// way).
#pragma once

#include <cstddef>

#if VKERNELS_HAS_CUDA

#  include <cuda_runtime.h>

#  include <string>

#  include "vkernels/util/error.hpp"

namespace vkernels::kernels::common {

inline std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Threads-per-block for a flat 1-D elementwise-style kernel.
inline int default_block_size() { return 256; }

// Launch ``kernel`` with ``args...`` and fail through VK_ENSURES on any
// launch error. ``what`` names the operation in the error message.
template <typename... Args>
void launch(void (*kernel)(Args...), dim3 grid, dim3 block, const char* what,
            Args... args) {
  kernel<<<grid, block>>>(args...);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, std::string(what) + " launch failed");
}

// 1-D convenience: grid sized for ``n`` elements at ``threads`` per block.
template <typename... Args>
void launch_1d(void (*kernel)(Args...), std::size_t n, const char* what,
               Args... args) {
  const int threads = default_block_size();
  const dim3 grid(static_cast<unsigned>(ceil_div(n, static_cast<std::size_t>(threads))));
  launch(kernel, grid, dim3(static_cast<unsigned>(threads)), what, args...);
}

}  // namespace vkernels::kernels::common

#endif  // VKERNELS_HAS_CUDA
