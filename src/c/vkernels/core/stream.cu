// vkernels/core/stream.cu — CUDA-only CudaStream implementation.
#include "vkernels/core/stream.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/util/error.hpp"

namespace vkernels {

CudaStream::CudaStream() : stream_(nullptr) {
  cudaError_t err = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_));
  VK_ENSURES(err == cudaSuccess, "cudaStreamCreate failed");
}

CudaStream::~CudaStream() {
  if (stream_ != nullptr) cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
}

void CudaStream::wait() {
  cudaError_t err = cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  VK_ENSURES(err == cudaSuccess, "cudaStreamSynchronize failed");
}

cudaStream_t_internal CudaStream::raw() const { return stream_; }

}  // namespace vkernels
#endif
