// verifier/gpu/harnesses/reduce.cu
//
// GPUVerify-dialect mirror of the reduction kernels in
// src/c/vkernels/kernels/reduce.cu.  The production kernel is a C++ template
// over `bool Max`; GPUVerify's 2018 frontend handles the template poorly, so
// the two instantiations are written out concretely.  Keep in sync with the
// production source.
//
// Invocation:
//   verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=1 \
//       verifier/gpu/harnesses/reduce.cu
#include "cuda.h"

// -FLT_MAX, spelled out: GPUVerify's stub does not provide CUDART_INF_F.
#define NEG_BIG (-3.402823466e38f)

__global__ void sum_reduce(const float* x, float* result, int n) {
  __shared__ float partial[256];
  float value = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) value = value + x[i];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step)
      partial[threadIdx.x] = partial[threadIdx.x] + partial[threadIdx.x + step];
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = partial[0];
}

__global__ void max_reduce(const float* x, float* result, int n) {
  __shared__ float partial[256];
  float value = NEG_BIG;
  for (int i = threadIdx.x; i < n; i += blockDim.x) value = fmaxf(value, x[i]);
  partial[threadIdx.x] = value;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step)
      partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + step]);
    __syncthreads();
  }
  if (threadIdx.x == 0) *result = partial[0];
}
