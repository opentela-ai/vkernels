// verifier/gpu/harnesses/elementwise.cu
//
// GPUVerify-dialect mirror of the elementwise kernels in
// src/c/vkernels/kernels/elementwise.cu.  GPUVerify cannot parse the production
// translation unit (it pulls in <cuda_runtime.h> and project headers the 2018
// frontend rejects), so the kernel bodies are restated here verbatim and
// verified against GPUVerify's own `cuda.h` stub.  Keep in sync with the
// production source.
//
// Invocation:
//   verifier/gpu/gpuverify.sh --blockDim=32 --gridDim=4 \
//       verifier/gpu/harnesses/elementwise.cu
#include "cuda.h"

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
