// vkernels/kernels/elementwise.cu — CUDA implementation (compiled with a toolkit).
#include "vkernels/kernels/elementwise.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/kernels/common/launch.hpp"
#  include "vkernels/util/error.hpp"

namespace vkernels::kernels {

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void scale_kernel(const float* x, float alpha, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = alpha * x[i];
}

__global__ void relu_kernel(const float* x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = fmaxf(x[i], 0.0f);
}

namespace cuda {

void add(Span<const float> a, Span<const float> b, Span<float> out) {
  VK_EXPECTS(a.size() == b.size(), "a and b must have equal length");
  VK_EXPECTS(a.size() == out.size(), "out must have the same length as inputs");
  int n = static_cast<int>(a.size());
  // NOTE: device pointers expected here; host launch path wired in a later change.
  common::launch_1d(add_kernel, static_cast<std::size_t>(n), "cuda add",
                    a.data(), b.data(), out.data(), n);
}

void scale(Span<const float> x, float alpha, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  int n = static_cast<int>(x.size());
  common::launch_1d(scale_kernel, static_cast<std::size_t>(n), "cuda scale",
                    x.data(), alpha, out.data(), n);
}

void relu(Span<const float> x, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  int n = static_cast<int>(x.size());
  common::launch_1d(relu_kernel, static_cast<std::size_t>(n), "cuda relu",
                    x.data(), out.data(), n);
}

}  // namespace cuda
}  // namespace vkernels::kernels

#endif  // VKERNELS_HAS_CUDA
