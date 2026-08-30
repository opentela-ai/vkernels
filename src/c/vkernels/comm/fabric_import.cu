// vkernels/comm/fabric_import.cu — CUDA device path for the cross-node
// fabric / VMM import (issue #49).
//
// The host reference (fabric_import.cpp) is the always-compiled,
// 100%-line-covered correctness oracle and carries the full contract
// (transport classification, the graph-capturable decision, an owned
// mirror model of the VMM import, and the per-hop cost model). This
// CUDA path is the device realization of the import: the real
// CU_MEM_HANDLE_TYPE_FABRIC sequence (cuMemAddressReserve ->
// cuMemImportFromShareableHandle -> cuMemMap -> cuMemSetAccess) yields a pointer
// the existing *_execute_offset kernels dereference UNCHANGED, and the
// pinned-host (cudaMallocHost) fallback ships the bytes over the network
// transport when no fabric-mapped device pointer is available.
//
// The host reference is the CI-verifiable surface (it runs on a machine
// with no GPU and proves the kernels produce byte-identical results over
// the imported pointer); this path is the on-device artifact built only
// where a CUDA toolkit is present, exactly as pipeline_boundary.cu /
// p2p_kv_restore.cu are. Neither is host-tested — the real-RDMA-fabric
// per-hop throughput is a hardware exercise on H-CLARIDEN / H-JSC.
//
// Lifetime: fabric_import_device_ptr owns the reserved VA range and the
// imported allocation; release it (or let it leave scope) only after
// every stream it was used on has been synchronised. The pinned scratch
// from fabric_bounce_scratch_alloc is owned by the caller and released
// with fabric_bounce_scratch_free.
#include "vkernels/comm/fabric_import.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda.h>
#  include <cuda_runtime.h>

#  include "vkernels/comm/fabric_import_cuda.hpp"
#  include "vkernels/util/error.hpp"

#  include <cstddef>
#  include <cstdint>
#  include <cstring>

namespace vkernels::comm::cuda {
namespace {

// Set *status (when non-null) to `code` and return `ptr`.
void* with_status(void* ptr, int code, int* status) {
  if (status) *status = code;
  return ptr;
}

}  // namespace

// Import the remote VRAM described by `handle` into the local device's
// address space. `handle.handle_bytes` carries the opaque descriptor the
// peer published over the control plane (NIXL/libfabric/CXI):
//
//   * cross-node VRAM with GPUDirect-RDMA -- a CUmemGenericAllocationHandle
//     re-exported as CU_MEM_HANDLE_TYPE_FABRIC, imported with
//     cuMemImportFromShareableHandle and mapped into a reserved VA range.
//   * same-node (kSameNodePeer) -- a cudaIpcMemHandle_t (CUDA-IPC),
//     opened with cudaIpcOpenMemHandle.
//
// Returns nullptr + VKERNELS_FI_ERR_UNSUPPORTED when no fabric-mapped
// device pointer is available (the caller takes the host-bounce fallback
// in cross_node_kv.hpp), or VKERNELS_FI_ERR_INTERNAL on a driver / handle
// error. The caller owns the returned pointer and releases it with
// fabric_import_release.
void* fabric_import_device_ptr(const RemoteFabricHandle& handle,
                               const FabricImportConfig& cfg,
                               int* status) {
  if (cfg.same_node) {
    // CUDA-IPC: cudaIpcOpenMemHandle turns the peer's published
    // cudaIpcMemHandle_t into a directly addressable device pointer. No
    // fabric import is needed; the handle bytes ARE the IPC handle.
    if (handle.size < sizeof(cudaIpcMemHandle_t) || handle.handle_bytes == nullptr)
      return with_status(nullptr, VKERNELS_FI_ERR_UNSUPPORTED, status);
    cudaIpcMemHandle_t ipc;
    std::memcpy(&ipc, handle.handle_bytes, sizeof(ipc));
    void* dev = nullptr;
    cudaError_t err = cudaIpcOpenMemHandle(&dev, ipc, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
      // cudaIpcOpenMemHandle failing on an invalid/unrelated handle is an
      // "unsupported" condition (caller falls back), not an internal bug.
      return with_status(nullptr,
                         err == cudaErrorInvalidValue ? VKERNELS_FI_ERR_UNSUPPORTED
                                                      : VKERNELS_FI_ERR_INTERNAL,
                         status);
    }
    return with_status(dev, VKERNELS_FI_OK, status);
  }

  // Cross-node: need a GPUDirect-RDMA / CU_MEM_HANDLE_TYPE_FABRIC path.
  // Without it the caller takes the host-bounce fallback.
  if (!cfg.has_gpudirect_rdma || handle.size == 0 || handle.handle_bytes == nullptr)
    return with_status(nullptr, VKERNELS_FI_ERR_UNSUPPORTED, status);

  CUcontext ctx;
  CUresult err = cuCtxGetCurrent(&ctx);
  if (err != CUDA_SUCCESS) return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);

  // Reserve a virtual address range for the import. Align up to the
  // allocation granularity so the mapping is always whole-granule.
  static constexpr std::size_t kGranule = 1u << 21;  // 2 MiB (VMM minimum)
  const std::size_t mapped = (handle.size + kGranule - 1u) & ~(kGranule - 1u);
  CUdeviceptr va = 0;
  err = cuMemAddressReserve(&va, mapped, 0ULL, 0ULL, 0ULL);
  if (err != CUDA_SUCCESS) return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);

  // Import the remote peer's published fabric descriptor (handle.handle_bytes
  // -- the opaque bytes cuMemExportToShareableHandle produced on the remote, shipped
  // over the control plane into a local CUmemGenericAllocationHandle.
  CUmemGenericAllocationHandle local_handle = 0;
  err = cuMemImportFromShareableHandle(&local_handle,
                                       const_cast<void*>(handle.handle_bytes),
                                       CU_MEM_HANDLE_TYPE_FABRIC);
  if (err != CUDA_SUCCESS) {
    cuMemAddressFree(va, mapped);
    return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);
  }

  // Map the imported allocation into the reserved range. cuMemSetAccess
  // then grants every local device read/write access to it, so the
  // existing *_execute_offset kernels dereference `va` UNCHANGED.
  err = cuMemMap(va, mapped, 0ULL, local_handle, 0ULL);
  // cuMemMap holds its own reference to local_handle; release ours so
  // fabric_import_release only needs cuMemUnmap + cuMemAddressFree.
  cuMemRelease(local_handle);
  if (err != CUDA_SUCCESS) {
    cuMemAddressFree(va, mapped);
    return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);
  }

  CUdevice dev;
  if (cuCtxGetDevice(&dev) != CUDA_SUCCESS) {
    cuMemUnmap(va, mapped);
    cuMemAddressFree(va, mapped);
    return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);
  }
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = dev;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (cuMemSetAccess(va, mapped, &access, 1ULL) != CUDA_SUCCESS) {
    cuMemUnmap(va, mapped);
    cuMemAddressFree(va, mapped);
    return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);
  }

  // The imported pointer IS the reserved VA; the existing kernels
  // dereference it UNCHANGED. The mapping holds its own reference to the
  // allocation (released above), so fabric_import_release only needs
  // cuMemUnmap + cuMemAddressFree over the granule-aligned size.
  (void)ctx;
  return with_status(reinterpret_cast<void*>(va), VKERNELS_FI_OK, status);
}

// Release a pointer previously returned by fabric_import_device_ptr for
// the kFabricMapped (VMM) path: cuMemUnmap + cuMemAddressFree over the
// granule-aligned `mapped_size` (the original RemoteFabricHandle::size).
// No-op on nullptr. The caller must have flushed every stream the pointer
// was used on BEFORE releasing (the VMM mapping is reference-counted by
// the driver; releasing while a kernel is in flight is undefined). The
// kSameNodePeer (CUDA-IPC) pointer is released with cudaIpcCloseMemHandle
// instead.
void fabric_import_release(void* imported_ptr, std::size_t mapped_size) {
  if (imported_ptr == nullptr) return;
  static constexpr std::size_t kGranule = 1u << 21;
  const std::size_t mapped =
      (mapped_size + kGranule - 1u) & ~(kGranule - 1u);
  CUdeviceptr va = reinterpret_cast<CUdeviceptr>(imported_ptr);
  cuMemUnmap(va, mapped);
  cuMemAddressFree(va, mapped);
}

// Pinned-host scratch for the bounce fallback (cudaMallocHost). The
// caller owns the scratch and releases it with fabric_bounce_scratch_free.
void* fabric_bounce_scratch_alloc(std::size_t size, int* status) {
  if (size == 0) return with_status(nullptr, VKERNELS_FI_OK, status);
  void* pinned = nullptr;
  cudaError_t err = cudaMallocHost(&pinned, size);
  if (err != cudaSuccess) return with_status(nullptr, VKERNELS_FI_ERR_INTERNAL, status);
  return with_status(pinned, VKERNELS_FI_OK, status);
}

void fabric_bounce_scratch_free(void* pinned) {
  if (pinned != nullptr) cudaFreeHost(pinned);
}

// device -> pinned (donate gather) and pinned -> device (restore scatter),
// stream-ordered. No-op on null pointers / zero size.
void fabric_bounce_device_to_pinned(void* pinned, const void* device,
                                    std::size_t size, cudaStream_t_fi stream) {
  if (pinned == nullptr || device == nullptr || size == 0) return;
  cudaMemcpyAsync(pinned, device, size, cudaMemcpyDeviceToHost, stream);
}

void fabric_bounce_pinned_to_device(void* device, const void* pinned,
                                    std::size_t size, cudaStream_t_fi stream) {
  if (device == nullptr || pinned == nullptr || size == 0) return;
  cudaMemcpyAsync(device, pinned, size, cudaMemcpyHostToDevice, stream);
}

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
