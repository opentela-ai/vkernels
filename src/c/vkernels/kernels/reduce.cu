// CUDA reductions return a host scalar, so completion is synchronous.
#include "vkernels/kernels/reduce.hpp"
#if VKERNELS_HAS_CUDA
#include <cuda_runtime.h>
#include <math_constants.h>
#include <limits>
#include "vkernels/util/error.hpp"
namespace vkernels::kernels::cuda {
namespace {
template <bool Max>
__global__ void reduce(const float* x, float* result, std::size_t n) {
  __shared__ float partial[256];
  float value = Max ? -CUDART_INF_F : 0.0f;
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
    value = Max ? fmaxf(value, x[i]) : value + x[i];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step)
      partial[threadIdx.x] = Max ? fmaxf(partial[threadIdx.x], partial[threadIdx.x + step])
                                : partial[threadIdx.x] + partial[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = partial[0];
}
struct Scalar {
  float* ptr = nullptr;
  ~Scalar() { if (ptr) cudaFree(ptr); }
};
template <bool Max>
void run(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  Scalar result;
  VK_ENSURES(cudaMalloc(&result.ptr, sizeof(float)) == cudaSuccess, "CUDA reduction allocation failed");
  reduce<Max><<<1, 256>>>(x.data(), result.ptr, x.size());
  VK_ENSURES(cudaGetLastError() == cudaSuccess, "CUDA reduction launch failed");
  VK_ENSURES(cudaMemcpy(&out, result.ptr, sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess,
             "CUDA reduction completion failed");
}
}
void sum(Span<const float> x, float& out) { run<false>(x, out); }
void max(Span<const float> x, float& out) { run<true>(x, out); }
}
#endif
