// vkernels/kernels/elementwise.cu — CUDA implementation (compiled with a toolkit).
#include "vkernels/kernels/elementwise.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

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
  check_same(a, b, out);  // reuse CPU contract
  int n = static_cast<int>(a.size());
  // NOTE: device pointers expected here; host launch path wired in a later change.
  add_kernel<<<(n + 255) / 256, 256>>>(a.data(), b.data(), out.data(), n);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda add launch failed");
}

void scale(Span<const float> x, float alpha, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  int n = static_cast<int>(x.size());
  scale_kernel<<<(n + 255) / 256, 256>>>(x.data(), alpha, out.data(), n);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda scale launch failed");
}

void relu(Span<const float> x, Span<float> out) {
  VK_EXPECTS(x.size() == out.size(), "x and out must have equal length");
  int n = static_cast<int>(x.size());
  relu_kernel<<<(n + 255) / 256, 256>>>(x.data(), out.data(), n);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda relu launch failed");
}

}  // namespace cuda
}  // namespace vkernels::kernels

#endif  // VKERNELS_HAS_CUDA
