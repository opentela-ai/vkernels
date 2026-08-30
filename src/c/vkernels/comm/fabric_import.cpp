// vkernels/comm/fabric_import.cpp -- host reference (oracle) implementation.
//
// The CPU reference is the correctness oracle for the cross-node fabric
// import (issue #49): the transport classification, the eager-break
// decision, a host model of the VMM import that maps a remote handle to an
// OWNED local mirror so the existing *_execute_offset kernels produce
// byte-identical results, and the per-hop cost model vs the same-node
// roofline. It is always compiled and fully unit-tested on a machine with
// no GPU; the CUDA path (fabric_import.cu, guarded by VKERNELS_HAS_CUDA)
// performs the real CU_MEM_HANDLE_TYPE_FABRIC import and mirrors this API.
#include "vkernels/comm/fabric_import.hpp"

#include <algorithm>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <utility>

#include "vkernels/util/annotations.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels::comm {

// ---------------------------------------------------------------------------
// Constants -- the same-node roofline references from the issue and the
// fallback / loopback numbers measured on sgs-gpu07 / Slingshot. The
// real-RDMA-fabric per-hop number is a hardware exercise (see
// docs/comm-cross-node-kv.md); the cost model uses a configurable fabric
// link rate (kFabricLinkGbps, HDR 400 Gb/s = 50 GB/s).
// ---------------------------------------------------------------------------
namespace {

// NVLink raw pair bandwidth (GB/s). The issue reports 220-243; the roofline
// reference is the conservative lower end.
constexpr double kNvlinkPairGbps = 220.0;

// Fused Stage-3 peer restore sustained throughput (GB/s) -- the binding
// resource for the prepared kernel and the cross-node reference.
constexpr double kFusedRestoreRoofGbps = 88.5;

// Cross-pair PCIe (GB/s), the issue's 54-55 number.
constexpr double kCrossPairPcieGbps = 54.0;

// Fabric link rate (GB/s). HDR 400 Gb/s = 50 GB/s per direction. The
// prepared kernel is the min(fabric, kernel_roof).
constexpr double kFabricLinkGbps = 50.0;

// Synchronous bulk-copy fallback over the DRAM-only libfabric transport
// (GB/s). The Slingshot TCP fallback measured ~1.4 GB/s (UCX has no CXI
// provider); pinned-host copy both ways dominates.
constexpr double kBulkCopyFallbackGbps = 1.4;

// NIXL/UCX loopback with no real fabric (GB/s), context only -- the issue
// explicitly requires a real RDMA fabric, not a loopback.
constexpr double kNixlUcxLoopbackGbps = 0.34;

// Per-hop launch floor (microseconds) for the direct-store kernel, mirroring
// p2p_kv_donate's kKernelFixedUs.
constexpr double kHopFloorUs = 2.0;

}  // namespace

// ---------------------------------------------------------------------------
// Transport classification
// ---------------------------------------------------------------------------

FabricImportTransport classify_fabric_import(const FabricImportConfig& cfg) {
  // Same-node peer (NVLink/HBM, ROCm IPC): the existing same-node path, no
  // import needed and no host bounce. Always graph-capturable -- the fast
  // path the issue wants the cross-node case to reach.
  if (cfg.same_node) return FabricImportTransport::kSameNodePeer;
  // The GH200 libfabric constraint: even with GPUDirect-RDMA on paper, the
  // C2C-attached GPU is invisible to the fabric plugin (hwloc-PCIe discovery
  // only sees DRAM), so cross-node VRAM MUST bounce through a pinned host
  // buffer. This is the caveat the issue asks to call out explicitly.
  if (cfg.dram_only_libfabric) return FabricImportTransport::kHostBounce;
  // A GPUDirect-RDMA / CU_MEM_HANDLE_TYPE_FABRIC path is available: import
  // the remote VRAM into the local address space and dereference directly.
  if (cfg.has_gpudirect_rdma) return FabricImportTransport::kFabricMapped;
  // No fabric path at all (cross-node, no GPUDirect-RDMA, not DRAM-only):
  // host-bounce, eager-break.
  return FabricImportTransport::kHostBounce;
}

bool is_import_graph_capturable(FabricImportTransport t) {
  return t == FabricImportTransport::kFabricMapped ||
         t == FabricImportTransport::kSameNodePeer;
}

bool eager_break_fabric_import(const FabricImportConfig& cfg) {
  return !is_import_graph_capturable(classify_fabric_import(cfg));
}

std::ostream& operator<<(std::ostream& os, FabricImportTransport t) {
  return os << fabric_import_transport_name(t);
}

// ---------------------------------------------------------------------------
// FabricHandle
// ---------------------------------------------------------------------------

FabricHandle::FabricHandle(std::uint64_t remote_node, std::uint64_t token,
                           const void* bytes, std::size_t size)
    : remote_node_(remote_node), token_(token), size_(size) {
  VK_EXPECTS(size > 0, "FabricHandle size must be positive");
  VK_EXPECTS(bytes != nullptr, "FabricHandle bytes must be non-null");
  bytes_.resize(size);
  std::memcpy(bytes_.data(), bytes, size);
}

// ---------------------------------------------------------------------------
// FabricImport
// ---------------------------------------------------------------------------

FabricImport::FabricImport(FabricImportTransport transport,
                           const FabricHandle& handle, FabricImportConfig cfg)
    : transport_(transport), cfg_(cfg), size_(handle.size()), device_ptr_(nullptr) {
  VK_EXPECTS(transport == FabricImportTransport::kFabricMapped ||
                 transport == FabricImportTransport::kSameNodePeer ||
                 transport == FabricImportTransport::kHostBounce,
             "unknown fabric import transport");
  switch (transport_) {
    case FabricImportTransport::kSameNodePeer:
      // Same-node: the imported pointer ALIASES the published FabricHandle
      // bytes (no copy). The handle is borrowed and must outlive this
      // import (and every plan/stream that dereferences device_ptr()), the
      // same lifetime the existing same-node path requires of a peer
      // pointer. No mirror is owned: on a real same-node deployment
      // (cudaIpcOpenMemHandle / ROCm IPC / NVLink C2C) the imported pointer
      // already names the remote physical memory, so the existing kernels
      // read from / write to it directly and write_back() is a no-op.
      device_ptr_ = const_cast<std::uint8_t*>(handle.bytes());
      break;
    case FabricImportTransport::kFabricMapped:
      // VMM import (host model): the local address space now names the
      // remote memory. For a restore (remote -> local) the kernel reads the
      // pre-populated mirror; for a donate (local -> remote) the kernel
      // writes the mirror and write_back() ships it to the remote handle.
      mirror_.assign(handle.bytes(), handle.bytes() + size_);
      device_ptr_ = mirror_.data();
      break;
    case FabricImportTransport::kHostBounce:
      // A fabric-mapped device pointer is UNAVAILABLE (the GH200 DRAM-only /
      // no-GPUDirect-RDMA case). device_ptr() is nullptr so the caller takes
      // the host-bounce path (cross_node_kv.hpp); the mirror is owned for
      // the fallback's correctness.
      mirror_.assign(handle.bytes(), handle.bytes() + size_);
      device_ptr_ = nullptr;
      break;
  }
}

// device_ptr_ points into the owned mirror ONLY for kFabricMapped (the
// VMM import yields a separate local address). For kSameNodePeer it aliases
// the borrowed FabricHandle's bytes -- the handle does not move with this
// import, so the alias stays valid across a move and must NOT be rebased
// from an empty mirror. The default / kHostBounce placeholder has
// device_ptr_ == nullptr.
FabricImport::FabricImport(FabricImport&& other) noexcept
    : transport_(other.transport_), cfg_(other.cfg_), size_(other.size_) {
  if (other.transport_ == FabricImportTransport::kFabricMapped) {
    const std::uintptr_t off = other.device_ptr_
        ? reinterpret_cast<std::uintptr_t>(other.device_ptr_) -
          reinterpret_cast<std::uintptr_t>(other.mirror_.data())
        : 0;
    mirror_ = std::move(other.mirror_);
    device_ptr_ = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(mirror_.data()) + off);
  } else {
    // kSameNodePeer: device_ptr_ aliases a borrowed external FabricHandle;
    // the handle does not move, so the alias is kept as-is. Default /
    // kHostBounce: device_ptr_ is already nullptr.
    device_ptr_ = other.device_ptr_;
  }
  other.transport_ = FabricImportTransport::kHostBounce;
  other.cfg_ = {};
  other.size_ = 0;
  other.device_ptr_ = nullptr;
}

FabricImport& FabricImport::operator=(FabricImport&& other) noexcept {
  if (this == &other) return *this;
  transport_ = other.transport_;
  cfg_ = other.cfg_;
  size_ = other.size_;
  if (other.transport_ == FabricImportTransport::kFabricMapped) {
    const std::uintptr_t off = other.device_ptr_
        ? reinterpret_cast<std::uintptr_t>(other.device_ptr_) -
          reinterpret_cast<std::uintptr_t>(other.mirror_.data())
        : 0;
    mirror_ = std::move(other.mirror_);
    device_ptr_ = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(mirror_.data()) + off);
  } else {
    mirror_.clear();
    mirror_.shrink_to_fit();
    device_ptr_ = other.device_ptr_;
  }
  other.transport_ = FabricImportTransport::kHostBounce;
  other.cfg_ = {};
  other.size_ = 0;
  other.device_ptr_ = nullptr;
  return *this;
}

void FabricImport::write_back(FabricHandle& remote) const {
  // Only the kFabricMapped donate actually writes the remote handle through
  // the imported mirror (the host model of the fabric write-back). kSameNodePeer
  // aliases the remote bytes (already written), and kHostBounce writes the
  // remote out-of-band (cross_node_kv.hpp).
  if (transport_ != FabricImportTransport::kFabricMapped) return;
  VK_EXPECTS(remote.size() >= size_,
             "write_back: remote FabricHandle smaller than the imported mirror");
  std::memcpy(remote.bytes(), mirror_.data(), size_);
}

// ---------------------------------------------------------------------------
// Cost model -- per-hop estimated throughput vs the same-node roofline
// ---------------------------------------------------------------------------

double same_node_fabric_roof_gbps(FabricImportTransport transport) {
  switch (transport) {
    case FabricImportTransport::kFabricMapped:
    case FabricImportTransport::kSameNodePeer:
      // The fused Stage-3 restore roof (88.5 GB/s) is the binding resource
      // for the prepared kernel; NVLink raw is 220-243. The cross-node
      // path is compared against THIS same-node roof.
      return kFusedRestoreRoofGbps;
    case FabricImportTransport::kHostBounce:
      // The synchronous bulk-copy fallback, dominated by the pinned-host
      // copy in both directions.
      return kBulkCopyFallbackGbps;
  }
  return 0.0;  // LCOV_EXCL_LINE (exhaustive switch)
}

CrossNodeHopCost cross_node_kv_throughput(FabricImportTransport transport,
                                          std::size_t total_bytes,
                                          bool gh200_dram_only) {
  CrossNodeHopCost c{};
  c.transport = transport;
  c.total_bytes = total_bytes;
  c.gh200_dram_only_caveat = false;
  c.same_node_roof_gbps = kFusedRestoreRoofGbps;  // the cross-node reference
  c.bulk_copy_fallback_gbps = kBulkCopyFallbackGbps;

  if (total_bytes == 0) {
    c.per_hop_gbps = 0.0;
    c.per_hop_us = 0.0;
    return c;
  }

  switch (transport) {
    case FabricImportTransport::kSameNodePeer:
      // The existing same-node peer path. On GH200 this is the NVLink/C2C
      // roof (88.5 fused, 220-243 raw); there is no host-bounce caveat.
      c.per_hop_gbps = kFusedRestoreRoofGbps;
      break;
    case FabricImportTransport::kFabricMapped: {
      // Direct fabric import: the existing kernel runs unchanged over the
      // imported pointer. The binding resource is min(fabric link, kernel
      // roof). On GH200 the DRAM-only libfabric constraint forces a host
      // bounce EVEN though a fabric import is requested, so the caveat
      // applies and the per-hop rate degrades to the bulk-copy fallback.
      const double fabric = std::min(kFabricLinkGbps, kFusedRestoreRoofGbps);
      c.per_hop_gbps = gh200_dram_only ? kBulkCopyFallbackGbps : fabric;
      c.gh200_dram_only_caveat = gh200_dram_only;
      break;
    }
    case FabricImportTransport::kHostBounce:
      // Pinned-host copy in both directions plus the network: the
      // synchronous bulk-copy fallback. The GH200 DRAM-only caveat is the
      // reason this path is taken.
      c.per_hop_gbps = kBulkCopyFallbackGbps;
      c.gh200_dram_only_caveat = true;
      break;
  }

  // per-hop time = bytes / (throughput * 1e9) * 1e6, floored at kHopFloorUs.
  const double bytes = static_cast<double>(total_bytes);
  const double t = bytes / (c.per_hop_gbps * 1e9) * 1e6;
  c.per_hop_us = std::max(t, kHopFloorUs);
  return c;
}

// Expose the raw / loopback constants to the bench and the doc via inline
// accessors (kept here, next to the cost model, rather than in the header
// to avoid leaking magic numbers into the public surface). coverage.py only
// honours LCOV_EXCL_LINE (not START/STOP), so each accessor is excluded
// individually: used only by the CUDA bench, not unit-tested.
double fabric_import_nvlink_pair_gbps() { return kNvlinkPairGbps; }        // LCOV_EXCL_LINE (CUDA bench only)
double fabric_import_cross_pair_pcie_gbps() { return kCrossPairPcieGbps; } // LCOV_EXCL_LINE (CUDA bench only)
double fabric_import_nixl_ucx_loopback_gbps() { return kNixlUcxLoopbackGbps; } // LCOV_EXCL_LINE (CUDA bench only)

}  // namespace vkernels::comm
