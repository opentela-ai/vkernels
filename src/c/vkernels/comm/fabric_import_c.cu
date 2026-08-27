// vkernels/comm/fabric_import_c.cu
//
// C ABI implementation for the on-device fabric / VMM import (issue #49).
// Thin `extern "C"` forwards over the C++ `vkernels::comm::cuda::`
// `fabric_import_device_ptr` / `fabric_import_release` / `fabric_bounce_*`
// entry points. Those entry points return status codes (or are
// stream-ordered no-ops on null/zero) and never throw, so nothing is
// thrown across the language boundary. Built only with a CUDA toolkit --
// the entry points take a raw `cudaStream_t` and the real driver handles
// (CU_MEM_HANDLE_TYPE_FABRIC / cudaIpcMemHandle_t).
//
// The always-compiled host planning C ABI (classify, eager-break, cost
// model) is in fabric_import_c.cpp; this file is the on-device artifact,
// exactly as pipeline_boundary_c.cu is the device counterpart to
// pipeline_boundary_c.cpp.
#include "vkernels/comm/fabric_import_c.h"

#if defined(VKERNELS_C_HAS_CUDA) && !defined(__CUDA_ARCH__)

#  include "vkernels/comm/fabric_import.hpp"
#  include "vkernels/comm/fabric_import_cuda.hpp"

#  include <cstddef>

namespace {

vkernels::comm::FabricImportConfig to_cpp_cfg(
    const vkernels_fi_config_t* cfg) {
  vkernels::comm::FabricImportConfig cpp;
  if (cfg != nullptr) {
    cpp.same_node = cfg->same_node != 0;
    cpp.has_gpudirect_rdma = cfg->has_gpudirect_rdma != 0;
    cpp.dram_only_libfabric = cfg->dram_only_libfabric != 0;
  }
  return cpp;
}

vkernels::comm::cuda::RemoteFabricHandle to_cpp_handle(
    const vkernels_remote_fabric_handle_t* h) {
  vkernels::comm::cuda::RemoteFabricHandle cpp;
  if (h != nullptr) {
    cpp.remote_node = h->remote_node;
    cpp.token = h->token;
    cpp.size = h->size;
    cpp.handle_bytes = h->handle_bytes;
  }
  return cpp;
}

}  // namespace

extern "C" void* vkernels_fabric_import_device_ptr(
    const vkernels_remote_fabric_handle_t* handle,
    const vkernels_fi_config_t* cfg, int* status) {
  return vkernels::comm::cuda::fabric_import_device_ptr(
      to_cpp_handle(handle), to_cpp_cfg(cfg), status);
}

extern "C" void vkernels_fabric_import_release(void* imported_ptr,
                                               size_t mapped_size) {
  vkernels::comm::cuda::fabric_import_release(imported_ptr, mapped_size);
}

extern "C" void* vkernels_fabric_bounce_scratch_alloc(size_t size,
                                                      int* status) {
  return vkernels::comm::cuda::fabric_bounce_scratch_alloc(size, status);
}

extern "C" void vkernels_fabric_bounce_scratch_free(void* pinned) {
  vkernels::comm::cuda::fabric_bounce_scratch_free(pinned);
}

extern "C" void vkernels_fabric_bounce_device_to_pinned(
    void* pinned, const void* device, size_t size, cudaStream_t stream) {
  vkernels::comm::cuda::fabric_bounce_device_to_pinned(
      pinned, device, size, stream);
}

extern "C" void vkernels_fabric_bounce_pinned_to_device(
    void* device, const void* pinned, size_t size, cudaStream_t stream) {
  vkernels::comm::cuda::fabric_bounce_pinned_to_device(
      device, pinned, size, stream);
}

#endif  // VKERNELS_C_HAS_CUDA
