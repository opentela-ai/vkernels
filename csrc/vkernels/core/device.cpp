// vkernels/core/device.cpp — CUDA-only portions of the device abstraction.
#include "vkernels/core/device.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/util/error.hpp"

namespace vkernels {

void Device::set_current() const {
  int idx = index_ < 0 ? 0 : index_;
  cudaError_t err = cudaSetDevice(idx);
  VK_ENSURES(err == cudaSuccess, "cudaSetDevice failed");
}

void Device::sync() const {
  set_current();
  cudaError_t err = cudaDeviceSynchronize();
  VK_ENSURES(err == cudaSuccess, "cudaDeviceSynchronize failed");
}

bool Device::supports_peer(const Device& other) const {
  int can = 0;
  cudaError_t err = cudaDeviceCanAccessPeer(index_ < 0 ? 0 : index_,
                                            other.index_ < 0 ? 0 : other.index_, &can);
  if (err != cudaSuccess) return false;
  return can != 0;
}

}  // namespace vkernels
#endif
