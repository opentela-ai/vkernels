// vkernels/comm/fabric_import.hpp
//
// Fabric / VMM import for cross-node KV reuse (issue #49).
//
// The prepared fused KV restore / donate kernels (issues #27, #36) are
// transport-agnostic at the pointer level: their *_execute_offset entry
// points dereference peer_src_ptrs / peer_dst_ptrs that must ALREADY be
// device-addressable on the local fabric (CUDA-IPC, same-node). On the
// sgs-gpu07 devbox that peaks at ~220-243 GB/s on NVLink pairs and
// ~54-55 GB/s cross-pair PCIe; the fused Stage-3 peer restore sustains
// ~88.5 GB/s. Cross-node movement existed only as a separate, synchronous
// host-staged bulk copy (kvaas NIXL/libfabric `transfer()`), which
// BYPASSES these kernels entirely.
//
// This module is the remaining slice: a vkernels-side mechanism that yields
// a DIRECTLY DEVICE-ADDRESSABLE handle to remote VRAM (CU_MEM_HANDLE_TYPE_FABRIC
// on CUDA, the IMEX equivalent on ROCm where applicable), so the existing
// *_execute_offset kernels run UNCHANGED over cross-node memory instead of
// being replaced by a bulk copy.
//
// Where a fabric-mapped device pointer is infeasible (no GPUDirect-RDMA, or
// the GH200 libfabric-only-DRAM constraint -- hwloc-PCIe discovery cannot
// see the C2C-attached H100 so VRAM is denied and cross-node VRAM must
// bounce through a pinned host buffer), the same plan ships the bytes over
// a host transport (see cross_node_kv.hpp): gather into a pinned
// device-accessible scratch, send over the wire, scatter on the far side --
// producing the SAME bytes as the direct-store kernel.
//
// Two-implementation model (mirrors p2p_kv_restore / pipeline_boundary):
//   * Host reference (fabric_import.cpp, always compiled) -- the import
//     classification, the graph-capturable decision, a host model of the
//     VMM import that maps a remote handle to an OWNED local mirror so the
//     existing kernels produce byte-identical results, and the per-hop
//     cost model vs the same-node roofline. Fully unit-tested with 100%
//     line coverage on a machine with no GPU.
//   * CUDA implementation (fabric_import_cuda.hpp + fabric_import.cu,
//     guarded by VKERNELS_HAS_CUDA) -- the real CU_MEM_HANDLE_TYPE_FABRIC
//     import: cuMemAddressReserve -> cuMemImportFromShareableHandle -> cuMemMap ->
//     cuMemSetAccess, yielding a device pointer the kernels dereference.
//     The host reference is the correctness oracle; the CUDA path mirrors
//     its API.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::comm {

// ---------------------------------------------------------------------------
// Fabric import transport
// ---------------------------------------------------------------------------
//
// The transport a cross-node fabric import resolves to. Mirrors the three
// transports pipeline_boundary.hpp uses for the PP boundary (#10), so a
// cross-node KV transfer is classified and graph-captured the same way.
//
// * kFabricMapped -- the remote VRAM is imported into the local address
//   space (CU_MEM_HANDLE_TYPE_FABRIC / IMEX), yielding a directly
//   device-addressable pointer. The existing *_execute_offset kernels run
//   UNCHANGED over it. The import is a host operation done ONCE, so the
//   subsequent device dereferences are graph-capturable (ties into #10).
// * kSameNodePeer -- the "remote" memory is on the same node (NVLink/HBM,
//   or ROCm IPC on MI300A): the existing same-node peer path. Also
//   graph-capturable.
// * kHostBounce  -- no GPUDirect-RDMA, or the GH200 DRAM-only libfabric
//   constraint: cross-node VRAM must bounce through a pinned host buffer.
//   NOT graph-capturable (the host send/recv cannot be serviced during a
//   replay); the framework must eager-break (mirrors vLLM
//   eager_break_during_capture, ties into #10).
enum class FabricImportTransport {
  kFabricMapped = 0,
  kSameNodePeer = 1,
  kHostBounce = 2
};

// Human-readable name (for logging / the bench / the cost model). Mirrors
// pipeline_transport_name.
inline constexpr const char* fabric_import_transport_name(FabricImportTransport t) {
  switch (t) {
    case FabricImportTransport::kFabricMapped: return "fabric-mapped";
    case FabricImportTransport::kSameNodePeer: return "same-node-peer";
    case FabricImportTransport::kHostBounce: return "host-bounce";
  }
  return "?";  // LCOV_EXCL_LINE (exhaustive switch)
}
std::ostream& operator<<(std::ostream& os, FabricImportTransport t);

// Deployment facts a cross-node fabric import is classified from, mirroring
// the knobs a serving runtime resolves before touching remote memory:
//
// * `same_node`           -- the "remote" memory is on the same node
//   (NVLink/HBM, ROCm IPC): the existing same-node peer path; no import
//   is needed and no host bounce occurs.
// * `has_gpudirect_rdma`  -- the local GPU + fabric expose a GPUDirect-RDMA
//   / CU_MEM_HANDLE_TYPE_FABRIC path, so remote VRAM can be imported into
//   the local address space and dereferenced directly by the existing
//   kernels.
// * `dram_only_libfabric` -- the GH200 libfabric constraint: hwloc-PCIe
//   discovery cannot see the C2C-attached H100, so the libfabric plugin
//   only carries DRAM<->DRAM. Cross-node VRAM must bounce through a pinned
//   host buffer EVEN when the GPU could otherwise import (the C2C-attached
//   device is invisible to the fabric). Forces kHostBounce.
//
// classify_fabric_import() precedence: same_node > dram_only_libfabric >
// has_gpudirect_rdma > kHostBounce. A cross-node deployment that is neither
// same-node, GPUDirect-RDMA, nor DRAM-only-libfabric also resolves to
// kHostBounce (no fabric path at all).
struct FabricImportConfig {
  bool same_node = false;
  bool has_gpudirect_rdma = false;
  bool dram_only_libfabric = false;
};

// Classify the fabric import transport from the deployment facts. Pure.
//
//   same_node            -> kSameNodePeer   (existing peer path, no import)
//   dram_only_libfabric  -> kHostBounce     (the GH200 constraint forces a
//                                           host bounce even with RDMA)
//   has_gpudirect_rdma   -> kFabricMapped   (VMM/fabric import -> device ptr)
//   otherwise            -> kHostBounce     (no fabric path)
FabricImportTransport classify_fabric_import(const FabricImportConfig& cfg);

// True iff `t` is a pure device transfer that a CUDA/HIP graph segment can
// contain and replay with no host progress (fabric-mapped once imported, or
// same-node peer). Host-bounce is false: the pinned-host send/recv cannot
// be serviced during a replay and must eager-break (ties into #10). Mirrors
// pipeline_boundary's is_graph_capturable.
bool is_import_graph_capturable(FabricImportTransport t);

// Eager-break decision for the fabric import: when the resolved transport
// is NOT graph-capturable, the framework must end the current capture
// segment, run the host-staged bounce transfer, and begin the next --
// exactly the PipelineBoundaryPlan eager-break contract (#10). Pure;
// equivalent to !is_import_graph_capturable(classify_fabric_import(cfg)).
bool eager_break_fabric_import(const FabricImportConfig& cfg);

// ---------------------------------------------------------------------------
// FabricHandle -- an opaque descriptor of remote memory
// ---------------------------------------------------------------------------
//
// A FabricHandle is the addressable "name" a peer node publishes for a
// range of its VRAM (or, on the DRAM-only path, its pinned host memory).
// On CUDA it carries a CUmemGenericAllocationHandle plus the size and the
// fabric-relevant fields; on ROCm the IMEX descriptor equivalent. The host
// reference carries the published bytes INLINE (it models a published
// handle as a copy the local node can read through its imported mirror),
// so the existing *_execute_offset kernels produce byte-identical results
// when they dereference the imported pointer -- that is the acceptance
// test. For a real deployment the handle bytes arrive over the control
// plane (NIXL/libfabric/CXI) and the CUDA/ROCm path turns them into a
// device pointer via the VMM import (fabric_import.cu).
class FabricHandle {
 public:
  // Publish `bytes` (capacity `size`) of remote memory under the given
  // `remote_node` id and a caller `token` (e.g. a NIXL descriptor id).
  // Throws std::invalid_argument on size==0 or null bytes.
  FabricHandle(std::uint64_t remote_node, std::uint64_t token,
               const void* bytes, std::size_t size);

  std::uint64_t remote_node() const { return remote_node_; }
  std::uint64_t token() const { return token_; }
  std::size_t size() const { return size_; }
  // The published bytes (host reference: inline copy; the local node reads
  // them through its imported mirror). For a real deployment these are the
  // opaque handle bytes the VMM import consumes.
  std::uint8_t* bytes() { return bytes_.data(); }
  const std::uint8_t* bytes() const { return bytes_.data(); }

 private:
  std::uint64_t remote_node_;
  std::uint64_t token_;
  std::size_t size_;
  std::vector<std::uint8_t> bytes_;
};

// ---------------------------------------------------------------------------
// FabricImport -- yields a directly device-addressable handle to remote VRAM
// ---------------------------------------------------------------------------
//
// Given a FabricHandle + FabricImportConfig, FabricImport performs the VMM
// import (CU_MEM_HANDLE_TYPE_FABRIC / IMEX) ONCE and yields a `void*` that
// the existing P2PKvRestorePlan / P2PKvDonatePlan execute()/execute_offset()
// dereference UNCHANGED. The imported pointer is owned by the FabricImport
// and must outlive the kernel stream; releasing the FabricImport unmaps.
//
// Host reference behaviour, per transport:
//
// * kSameNodePeer -- the imported pointer ALIASES the remote bytes (same
//   node, no copy). The existing kernel reads from / writes to the remote
//   buffer directly.
// * kFabricMapped -- the imported pointer is an OWNED local mirror of the
//   remote bytes, pre-populated from them at construction (the host model
//   of VMM import: the local address space now names the remote memory).
//   For a donate (local -> remote) the kernel writes the mirror; write_back()
//   copies the mirror into the remote FabricHandle (the fabric write-back).
//   For a restore (remote -> local) the kernel reads the pre-populated
//   mirror.
// * kHostBounce -- a fabric-mapped device pointer is UNAVAILABLE (the GH200
//   DRAM-only / no-GPUDirect-RDMA case). device_ptr() is nullptr so the
//   caller takes the host-bounce path (cross_node_kv.hpp); the import
//   still owns a mirror the fallback can use for correctness. has_device_ptr()
//   is the acceptance surface for "where a fabric-mapped device pointer is
//   unavailable".
class FabricImport {
 public:
  // Default: an empty placeholder (transport kHostBounce, no device
  // pointer, empty mirror). Lets a caller build a cross-node plan for the
  // host-bounce path without fabricating a meaningless remote handle.
  FabricImport() = default;

  // Import the remote memory described by `handle` for the given transport
  // (use classify_fabric_import on the caller's config). Throws
  // std::invalid_argument on an unknown transport. See the class comment
  // for the per-transport host model.
  FabricImport(FabricImportTransport transport, const FabricHandle& handle,
               FabricImportConfig cfg = {});

  FabricImport(const FabricImport&) = delete;
  FabricImport& operator=(const FabricImport&) = delete;
  // device_ptr_ points into mirror_, so a move must fix it up (otherwise it
  // dangles at the old mirror address). Implemented in fabric_import.cpp.
  FabricImport(FabricImport&& other) noexcept;
  FabricImport& operator=(FabricImport&& other) noexcept;

  FabricImportTransport transport() const { return transport_; }
  FabricImportConfig config() const { return cfg_; }
  bool is_graph_capturable() const { return is_import_graph_capturable(transport_); }
  std::size_t size() const { return size_; }

  // The directly device-addressable pointer the existing kernels
  // dereference, or nullptr when the fabric import is unavailable and the
  // caller must take the host-bounce path (cross_node_kv.hpp).
  void* device_ptr() { return device_ptr_; }
  const void* device_ptr() const { return device_ptr_; }

  // True iff a fabric-mapped device pointer is available (kFabricMapped or
  // kSameNodePeer). False for the default placeholder and kHostBounce.
  bool has_device_ptr() const { return device_ptr_ != nullptr; }

  // For a donate (local -> remote), copy the imported mirror back into the
  // remote FabricHandle's bytes (the host model of the fabric write-back).
  // No-op for the default placeholder and kSameNodePeer (the donate wrote
  // directly through the aliased peer pointer -- no separate mirror to
  // ship) and kHostBounce (the fallback writes the remote out-of-band).
  // The caller passes the remote FabricHandle the donate targeted; throws
  // std::invalid_argument if the remote handle is smaller than the mirror.
  void write_back(FabricHandle& remote) const;

 private:
  FabricImportTransport transport_ = FabricImportTransport::kHostBounce;
  FabricImportConfig cfg_{};
  std::size_t size_ = 0;
  std::vector<std::uint8_t> mirror_;    // the imported address space
  void* device_ptr_ = nullptr;          // points into mirror_, or nullptr
};

// ---------------------------------------------------------------------------
// Cost model -- per-hop estimated throughput vs the same-node roofline
// ---------------------------------------------------------------------------
//
// Reports the estimated per-hop throughput (GB/s) and the per-hop cost
// (microseconds) for a cross-node KV transfer of `total_bytes` over
// `transport`, compared against the same-node roofline. The GH200
// DRAM-only / host-bounce caveat is surfaced explicitly (see the doc):
// a host-bounce hop pays a pinned-host copy in BOTH directions plus the
// network, so it is never the same-node roofline.
//
// Roofline references (measured on the sgs-gpu07 devbox, see the issue and
// docs/operations): ~220-243 GB/s on NVLink pairs, ~88.5 GB/s fused
// restore, ~54-55 GB/s cross-pair PCIe. The synchronous bulk-copy fallback
// over the DRAM-only libfabric transport is much lower (the TCP fallback
// on Slingshot measured ~1.4 GB/s); a no-real-fabric NIXL/UCX loopback is
// ~0.34 GB/s (context only -- the issue requires a real RDMA fabric, not a
// loopback). The real-RDMA-fabric per-hop number is a hardware exercise on
// H-CLARIDEN / H-JSC (see docs/comm-cross-node-kv.md); the cost model uses
// a configurable fabric link rate (default HDR 400 Gb/s = 50 GB/s).
struct CrossNodeHopCost {
  FabricImportTransport transport;
  std::size_t total_bytes;
  double per_hop_gbps;            // estimated throughput for this hop
  double per_hop_us;              // estimated time for this hop
  double same_node_roof_gbps;     // the same-node roofline for comparison
  double bulk_copy_fallback_gbps; // the synchronous host-bounce fallback
  bool gh200_dram_only_caveat;    // true when the host-bounce caveat applies
};

// Compute the per-hop cost for `total_bytes` over `transport`. When
// `gh200_dram_only` is true the GH200 DRAM-only / host-bounce caveat is
// flagged (kFabricMapped degrades to a host bounce on GH200 because the
// C2C-attached GPU is invisible to the libfabric plugin). Pure; zero bytes
// yields zero time.
CrossNodeHopCost cross_node_kv_throughput(FabricImportTransport transport,
                                          std::size_t total_bytes,
                                          bool gh200_dram_only = false);

// Same-node roofline (GB/s) for a given transport, used by the cost model
// and the bench:
//   kFabricMapped / kSameNodePeer -> 88.5 (the fused restore roof -- the
//       binding resource for the prepared kernel; NVLink raw is 220-243).
//   kHostBounce                  -> the synchronous bulk-copy fallback
//       (much lower; dominated by the pinned-host copy).
double same_node_fabric_roof_gbps(FabricImportTransport transport);

}  // namespace vkernels::comm
