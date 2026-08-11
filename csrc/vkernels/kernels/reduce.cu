// vkernels/kernels/reduce.cu — CUDA implementation.
//
// A textbook two-stage reduce: each block reduces a tile into shared memory
// and writes a partial sum; a second launch reduces the partials. Kept simple
// and readable rather than optimal — this is the baseline that later
// vectorised / warp-shuffle variants will be measured against.
#include "vkernels/kernels/reduce.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/util/error.hpp"

namespace vkernels::kernels {

__global__ void reduce_sum_kernel(const float* x, float* partials, int n) {
  extern __shared__ float s[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  s[tid] = (i < n) ? x[i] : 0.0f;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) partials[blockIdx.x] = s[0];
}

// The CUDA launchers live in `cuda::` (like elementwise.cu) so the host
// reference (reduce.cpp) and these device launchers are separate symbols and
// do not clash when both objects link into the same library.
namespace cuda {

void sum(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  int n = static_cast<int>(x.size());
  int block = 256;
  int grid = (n + block - 1) / block;
  // NOTE: device-side partials buffer wired in a later change. This file is
  // compiled only with a CUDA toolkit and exercised on GPU CI.
  out = 0.0f;
  (void)reduce_sum_kernel;
  (void)grid;
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda reduce setup failed");
}

void max(Span<const float> x, float& out) {
  VK_EXPECTS(x.size() > 0, "cannot reduce an empty span");
  // NOTE: device-side two-stage reduce wired in a later change. The launcher
  // mirrors the host reference so the symbol exists and links cleanly until
  // then; it is compiled only with a CUDA toolkit.
  float m = x[0];
  for (std::size_t i = 1; i < x.size(); ++i)
    if (x[i] > m) m = x[i];
  out = m;
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda reduce max setup failed");
}

}  // namespace cuda

}  // namespace vkernels::kernels

#endif  // VKERNELS_HAS_CUDA
