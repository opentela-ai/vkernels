// verifier/gpu/harnesses/comm_allreduce.cu
//
// GPUVerify-dialect mirror of the fused all-reduce stub kernel in
// src/c/vkernels/comm/allreduce.cu.  Body verbatim.  The stub is not yet
// launched by any production path (the source documents it as a placeholder
// keeping the device TU well-formed), so it is verified at a representative
// decode-shape launch.
//
// Invocation:
//   verifier/gpu/gpuverify.sh --blockDim=256 --gridDim=2 \
//       verifier/gpu/harnesses/comm_allreduce.cu
#include "cuda.h"

__global__ void fused_reduce_stub(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
