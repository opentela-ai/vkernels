// vkernels/comm/fabric_import_c.h
//
// C ABI for the cross-node fabric / VMM import (issue #49). Non-C++
// consumers (kvaas, a Rust or Python runtime, ctypes) reach the cross-node
// mechanism through these `extern "C"` entry points. Errors are RETURNED
// as codes -- no C++ exception crosses the ABI boundary.
//
// Two layers, mirroring rccl_c (always-compiled host planning surface) and
// p2p_kv_restore_c (CUDA-only device primitive), exactly as pipeline_boundary_c
// (issue #10) splits the same-node boundary into a host planning C ABI
// (pipeline_boundary_c.{h,cpp}) and a CUDA device plan (pipeline_boundary_c.cu):
//   * The transport classification, the graph-capturable / eager-break
//     decision, the transport name, and the per-hop cost model are ALWAYS
//     visible and callable without a GPU (fabric_import_c.cpp). The host
//     CI job and its 100% line-coverage gate exercise the same planning
//     surface a non-C++ consumer reaches here.
//   * The on-device fabric import (CU_MEM_HANDLE_TYPE_FABRIC / CUDA-IPC),
//     the pinned-host bounce scratch, and the stream-ordered bounce copies
//     are visible only when the CUDA runtime headers are present
//     (fabric_import_c.cu), because they take a raw `cudaStream_t` and the
//     real driver handles.
//
// The cross-node prepared fused KV restore / donate plan C ABI is in
// cross_node_kv_c.h (CUDA-only, mirroring p2p_kv_restore_c / p2p_kv_donate_c):
// the caller obtains an imported device pointer here and hands it to that
// plan so the EXISTING *_execute_offset kernels run UNCHANGED over
// cross-node memory.
#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__has_include)
#  if __has_include(<cuda_runtime.h>)
#    define VKERNELS_C_HAS_CUDA 1
#    include <cuda_runtime.h>
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Status codes mirroring vkernels::Code (0 = ok). Integer values are
// compatible with the CUDA-device enum in fabric_import_cuda.hpp
// (VKERNELS_FI_OK=0, VKERNELS_FI_ERR_UNSUPPORTED=3,
// VKERNELS_FI_ERR_INTERNAL=4) so the C ABI and the device path agree on
// the values that cross the boundary.
typedef enum {
  VKERNELS_FI_OK = 0,
  VKERNELS_FI_ERR_INVALID_ARGUMENT = 1,
  VKERNELS_FI_ERR_OUT_OF_RANGE = 2,
  VKERNELS_FI_ERR_UNSUPPORTED = 3,
  VKERNELS_FI_ERR_INTERNAL = 4,
} vkernels_fi_status_t;

// Fabric import transport (mirrors vkernels::comm::FabricImportTransport;
// integer values are stable across the ABI). Passed and returned as a
// plain int (the enum only names the constants), mirroring
// pipeline_boundary_c.h's transport convention.
typedef enum {
  VKERNELS_FI_TRANSPORT_FABRIC_MAPPED = 0,  // VMM/fabric import -> device ptr
  VKERNELS_FI_TRANSPORT_SAME_NODE_PEER = 1,  // existing peer path, no import
  VKERNELS_FI_TRANSPORT_HOST_BOUNCE = 2,     // pinned-host copy both ways
} vkernels_fi_transport_t;

// Deployment facts a fabric import is classified from (mirrors
// vkernels::comm::FabricImportConfig; 0/1 booleans for C portability).
typedef struct {
  int same_node;             // peer stage on the same node (NVLink/C2C)
  int has_gpudirect_rdma;    // GPU can be named by the fabric plugin
  int dram_only_libfabric;   // GH200 constraint: plugin only carries DRAM
} vkernels_fi_config_t;

// Per-hop cost for one cross-node KV transfer (mirrors
// vkernels::comm::CrossNodeHopCost). `transport` is one of
// VKERNELS_FI_TRANSPORT_*.
typedef struct {
  int transport;                   // vkernels_fi_transport_t
  size_t total_bytes;              // bytes moved in one hop
  double per_hop_gbps;             // estimated throughput for this hop
  double per_hop_us;               // estimated time for this hop
  double same_node_roof_gbps;      // the same-node roofline for comparison
  double bulk_copy_fallback_gbps;  // the synchronous host-bounce fallback
  int gh200_dram_only_caveat;      // 1 when the host-bounce caveat applies
} vkernels_cross_node_kv_cost_t;

// Classify the fabric import transport from the deployment facts. On
// success returns one of VKERNELS_FI_TRANSPORT_* and, when status_out is
// non-null, sets it to VKERNELS_FI_OK. On a null config sets *status_out
// (when non-null) to VKERNELS_FI_ERR_INVALID_ARGUMENT and returns
// VKERNELS_FI_TRANSPORT_HOST_BOUNCE (the safe fallback).
int vkernels_fabric_import_classify(
    const vkernels_fi_config_t* cfg, vkernels_fi_status_t* status_out);

// Eager-break decision mirroring vLLM `eager_break_during_capture`: 1 when
// the cross-node transfer is NOT graph-capturable (the pinned-host send/recv
// must eager-break out of a captured decode segment and run between
// launches), 0 when it is capturable (fabric-mapped once imported, or
// same-node peer). Same status contract as vkernels_fabric_import_classify.
int vkernels_fabric_import_eager_break(
    const vkernels_fi_config_t* cfg, vkernels_fi_status_t* status_out);

// 1 iff `t` is a pure device transfer a captured CUDA/HIP graph segment can
// replay with no host progress (fabric-mapped once imported, or same-node
// peer); 0 otherwise (kHostBounce -- the pinned send/recv cannot be
// serviced during a replay). Mirrors vkernels::comm::is_import_graph_capturable.
int vkernels_fabric_import_is_graph_capturable(int t);

// Human-readable transport name for VKERNELS_FI_TRANSPORT_*, or "?" on an
// unknown value. Never returns null.
const char* vkernels_fabric_import_transport_name(int t);

// Same-node roofline (GB/s) for a given transport:
//   VKERNELS_FI_TRANSPORT_FABRIC_MAPPED / VKERNELS_FI_TRANSPORT_SAME_NODE_PEER
//       -> 88.5 (the fused restore roof -- the binding resource for the
//          prepared kernel; NVLink raw is 220-243).
//   VKERNELS_FI_TRANSPORT_HOST_BOUNCE -> the synchronous bulk-copy fallback
//       (much lower; dominated by the pinned-host copy in both directions).
double vkernels_fabric_import_same_node_roof_gbps(int t);

// Per-hop cost for `total_bytes` over `transport`. When `gh200_dram_only` is
// non-zero the GH200 DRAM-only / host-bounce caveat is flagged
// (VKERNELS_FI_TRANSPORT_FABRIC_MAPPED degrades to a host bounce on GH200
// because the C2C-attached GPU is invisible to the libfabric plugin). On
// success sets *out and, when status_out is non-null, *status_out to
// VKERNELS_FI_OK. On a null `out` sets *status_out (when non-null) to
// VKERNELS_FI_ERR_INVALID_ARGUMENT and returns it. Pure; zero bytes yields
// zero time.
vkernels_fi_status_t vkernels_cross_node_kv_throughput(
    int transport, size_t total_bytes, int gh200_dram_only,
    vkernels_cross_node_kv_cost_t* out, vkernels_fi_status_t* status_out);

#ifdef VKERNELS_C_HAS_CUDA

// The opaque descriptor the peer published over the control plane
// (NIXL/libfabric/CXI). For VKERNELS_FI_TRANSPORT_SAME_NODE_PEER this is a
// cudaIpcMemHandle_t; for VKERNELS_FI_TRANSPORT_FABRIC_MAPPED a
// CUmemGenericAllocationHandle re-exported as CU_MEM_HANDLE_TYPE_FABRIC.
// `handle_bytes` carries `size` bytes of that descriptor.
typedef struct {
  uint64_t remote_node;        // peer node id (informational)
  uint64_t token;              // caller-defined correlation token
  size_t size;                 // bytes of remote VRAM
  const void* handle_bytes;    // opaque descriptor (size >= handle size)
} vkernels_remote_fabric_handle_t;

// Import `handle` into the local device's address space via
// CU_MEM_HANDLE_TYPE_FABRIC (or cudaIpcOpenMemHandle for same-node) and
// yield a directly device-addressable pointer the existing
// *_execute_offset kernels dereference UNCHANGED. On success returns a
// non-null pointer and sets *status (when non-null) to VKERNELS_FI_OK;
// returns nullptr + VKERNELS_FI_ERR_UNSUPPORTED when no fabric-mapped
// device pointer is available (the caller takes the host-bounce
// fallback); VKERNELS_FI_ERR_INTERNAL on a driver / handle error.
//
// The import is a host operation done ONCE; subsequent device dereferences
// are graph-capturable. The caller owns the returned pointer and releases
// it with vkernels_fabric_import_release (kFabricMapped) or
// cudaIpcCloseMemHandle (kSameNodePeer) after every stream it was used on
// has been synchronised.
void* vkernels_fabric_import_device_ptr(
    const vkernels_remote_fabric_handle_t* handle,
    const vkernels_fi_config_t* cfg, int* status);

// Release a pointer previously returned by
// vkernels_fabric_import_device_ptr for the kFabricMapped (VMM) path
// (cuMemUnmap + cuMemAddressFree over the granule-aligned `mapped_size`,
// the original handle->size). No-op on nullptr. The kSameNodePeer
// (CUDA-IPC) pointer is released with cudaIpcCloseMemHandle instead.
void vkernels_fabric_import_release(void* imported_ptr, size_t mapped_size);

// Pinned-host (cudaMallocHost) scratch for the bounce fallback, ordered on
// `stream`. The caller owns the scratch and releases it with
// vkernels_fabric_bounce_scratch_free. Returns nullptr + VKERNELS_FI_OK on
// a zero size, or nullptr + VKERNELS_FI_ERR_INTERNAL on a host allocation
// failure (sets *status when non-null).
void* vkernels_fabric_bounce_scratch_alloc(size_t size, int* status);
void vkernels_fabric_bounce_scratch_free(void* pinned);

// device -> pinned (donate gather) and pinned -> device (restore scatter),
// stream-ordered. No-op on null pointers / zero size.
void vkernels_fabric_bounce_device_to_pinned(void* pinned, const void* device,
                                             size_t size, cudaStream_t stream);
void vkernels_fabric_bounce_pinned_to_device(void* device, const void* pinned,
                                             size_t size, cudaStream_t stream);

#endif  // VKERNELS_C_HAS_CUDA

#ifdef __cplusplus
}
#endif
