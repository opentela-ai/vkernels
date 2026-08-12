// vkernels/comm/allreduce.cu — fused GPU all-reduce (compiled with a toolkit).
//
// Placeholder for a kernel-fused ring all-reduce that keeps data resident on
// device and overlaps the reduce step with the next chunk's copy. Compiled
// only when VKERNELS_HAS_CUDA; exercised on GPU CI.
#include "vkernels/comm/allreduce.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

namespace vkernels::comm {

// Future: a single kernel performing a tree reduce on shared memory followed
// by an inter-block ring. Kept as a stub so the device translation unit is
// well-formed and the build path stays exercised.
__global__ void fused_reduce_stub(float* dst, const float* src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}

}  // namespace vkernels::comm

#endif  // VKERNELS_HAS_CUDA
