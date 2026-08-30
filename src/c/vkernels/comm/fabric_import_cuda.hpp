// vkernels/comm/fabric_import_cuda.hpp
//
// CUDA-only declarations for the cross-node fabric / VMM import (issue #49).
// Kept separate from fabric_import.hpp because the CUDA entry points take
// `cudaStream_t`, which must not be exposed to host-only translation units.
// Included only when VKERNELS_HAS_CUDA; the definitions live in
// fabric_import.cu and mirror the host reference (fabric_import.cpp).
//
// The real path is CU_MEM_HANDLE_TYPE_FABRIC: cuMemAddressReserve reserves
// a virtual address range on the local device, cuMemImportFromShareableHandle turns
// the remote peer's published allocation handle into a local handle, and
// cuMemMap + cuMemSetAccess map it into the reserved range so the existing
// *_execute_offset kernels dereference the imported pointer UNCHANGED.
//
// Lifetime: the FabricImport owns the reserved VA range and the mapping;
// release it (or let it leave scope) only after every stream it was used on
// has been synchronised (cuMemUnmap + cuMemAddressFree).
#pragma once

#include <cstddef>
#include <cstdint>

#include "vkernels/comm/fabric_import.hpp"
#include "vkernels/util/config.hpp"

#if VKERNELS_HAS_CUDA
struct CUstream_st;
typedef CUstream_st* cudaStream_t_fi;  // avoid pulling cuda_runtime.h here

namespace vkernels::comm::cuda {

// Status codes mirroring pipeline_boundary_c.h's VKERNELS_PP_ERR_*. 0 on
// success; the import yields nullptr + VKERNELS_FI_ERR_UNSUPPORTED when no
// fabric-mapped device pointer is available (the caller takes the host
// bounce); VKERNELS_FI_ERR_INTERNAL on a driver / handle / allocation error.
enum {
  VKERNELS_FI_OK = 0,
  VKERNELS_FI_ERR_UNSUPPORTED = 3,
  VKERNELS_FI_ERR_INTERNAL = 4
};

// A remote allocation published for fabric import. Mirrors the host
// reference's FabricHandle but carries the real driver handle + size +
// the opaque fabric descriptor bytes the peer sent over the control plane
// (NIXL/libfabric/CXI). For testing on a single device the bytes can be
// a CUDA-IPC handle (cudaIpcMemHandle_t) re-exported over the wire.
struct RemoteFabricHandle {
  std::uint64_t remote_node = 0;
  std::uint64_t token = 0;
  std::size_t size = 0;       // bytes of remote VRAM
  const void* handle_bytes = nullptr;  // opaque descriptor (CUmemGenericAllocationHandle-adjacent)
};

// Import the remote VRAM described by `handle` into the local device's
// address space via CU_MEM_HANDLE_TYPE_FABRIC and yield a directly
// device-addressable pointer the existing *_execute_offset kernels
// dereference UNCHANGED. Returns nullptr (and sets *status to
// VKERNELS_FI_ERR_UNSUPPORTED) when no fabric-mapped device pointer is
// available -- the caller then takes the host-bounce fallback
// (cross_node_kv.hpp). On a driver/handle error sets *status to
// VKERNELS_FI_ERR_INTERNAL.
//
// kSameNodePeer uses cudaIpcOpenMemHandle over the published
// cudaIpcMemHandle_t (handle.handle_bytes); release that pointer with
// cudaIpcCloseMemHandle, NOT fabric_import_release. kFabricMapped uses the
// VMM import (cuMemAddressReserve -> cuMemImportFromShareableHandle -> cuMemMap ->
// cuMemSetAccess); release that pointer with fabric_import_release.
//
// The import is a host operation done ONCE; subsequent device dereferences
// are graph-capturable (ties into #10). The returned pointer is owned by
// the caller and must be released after every stream it was used on has
// been synchronised.
void* fabric_import_device_ptr(const RemoteFabricHandle& handle,
                               const FabricImportConfig& cfg,
                               int* status);

// Release a pointer previously returned by fabric_import_device_ptr for
// the kFabricMapped (VMM) path: cuMemUnmap + cuMemAddressFree over the
// granule-aligned `mapped_size` (the original RemoteFabricHandle::size).
// No-op on nullptr. The kSameNodePeer (CUDA-IPC) pointer is released with
// cudaIpcCloseMemHandle instead.
void fabric_import_release(void* imported_ptr, std::size_t mapped_size);

// Host-bounce fallback on the GPU: copy `bytes` (device) into a PINNED
// host scratch (cudaMallocHost), sized `size`, ready to ship over the
// network transport. The caller owns the pinned scratch and releases it
// with fabric_bounce_scratch_free. Returns nullptr on a host allocation
// failure (sets *status to VKERNELS_FI_ERR_INTERNAL). `stream` orders the
// device->host copy; returns without synchronising.
void* fabric_bounce_scratch_alloc(std::size_t size, int* status);
void fabric_bounce_scratch_free(void* pinned);

// Copy device -> pinned (the gather step of the donate host-bounce) and
// pinned -> device (the scatter step of the restore host-bounce),
// stream-ordered. No-op on null pointers / zero size. `status` (when
// non-null) is set to VKERNELS_FI_OK on success, VKERNELS_FI_ERR_INTERNAL
// on a copy error.
void fabric_bounce_device_to_pinned(void* pinned, const void* device,
                                    std::size_t size, cudaStream_t_fi stream);
void fabric_bounce_pinned_to_device(void* device, const void* pinned,
                                    std::size_t size, cudaStream_t_fi stream);

}  // namespace vkernels::comm::cuda

#endif  // VKERNELS_HAS_CUDA
