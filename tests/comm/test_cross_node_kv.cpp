// tests/comm/test_cross_node_kv.cpp
//
// Host-reference tests for the cross-node KV transfer (issue #49). The CPU
// reference is the byte-exact oracle: these tests pin the ByteChannel
// transport, the FabricImport classification, the per-hop cost model, and
// -- the core acceptance -- that a cross-node prepared fused RESTORE
// round-trip (peer pages on B -> local K/V on A) and a cross-node DONATE
// round-trip (local K/V on A -> peer pages on B) produce byte-identical
// results to the same-node NVLink path, over BOTH the graph-capturable
// path (kFabricMapped / kSameNodePeer) and the host-bounce fallback
// (kHostBounce, where a fabric-mapped device pointer is unavailable).
//
// Acceptance (issue #49):
//   #1  A cross-node prepared fused restore round-trip and a cross-node
//       donate round-trip produce byte-identical results to the same-node
//       NVLink path. The direct path runs the existing *_execute_offset
//       kernels UNCHANGED over a FabricImport that mirrors the remote
//       bytes, so the host reference's memcpy IS the same memcpy the
//       same-node path issues. The host-bounce path runs the existing
//       kv_gather / kv_scatter primitives, which the two-stage oracles
//       already prove byte-identical to the fused kernels.
//   #2  Per-hop throughput is reported vs the same-node roofline and the
//       synchronous bulk-copy fallback, with the GH200 DRAM-only /
//       host-bounce caveat called out.
//   #3  The host-bounce fallback validates correctly where a fabric-mapped
//       device pointer is unavailable (kHostBounce: device_ptr() == nullptr,
//       execute() gathers + sends / recvs + scatters over a ByteChannel).
//
// The real-RDMA-fabric exercise (not UCX loopback) is a hardware step on
// H-CLARIDEN / H-JSC; the host oracle here is the byte-exact model the
// CUDA path (cross_node_kv.cu, guarded by VKERNELS_HAS_CUDA) mirrors.
#include "minitest.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "vkernels/comm/cross_node_kv.hpp"
#include "vkernels/comm/fabric_import.hpp"
#include "vkernels/comm/p2p_kv_donate.hpp"
#include "vkernels/comm/p2p_kv_restore.hpp"
#include "vkernels/comm/pipeline_boundary.hpp"
#include "vkernels/core/stream.hpp"

using vkernels::Stream;
using vkernels::comm::ByteBlockingQueue;
using vkernels::comm::ByteChannel;
using vkernels::comm::classify_fabric_import;
using vkernels::comm::cross_node_kv_throughput;
using vkernels::comm::CrossNodeHopCost;
using vkernels::comm::CrossNodeKvDonatePlan;
using vkernels::comm::CrossNodeKvRestorePlan;
using vkernels::comm::CrossNodeKvAccess;
using vkernels::comm::CrossNodeKvTransferKind;
using vkernels::comm::eager_break_fabric_import;
using vkernels::comm::FabricHandle;
using vkernels::comm::FabricImport;
using vkernels::comm::FabricImportConfig;
using vkernels::comm::FabricImportTransport;
using vkernels::comm::fabric_import_transport_name;
using vkernels::comm::GraphCapture;
using vkernels::comm::is_import_graph_capturable;
using vkernels::comm::make_byte_link;
using vkernels::comm::MockByteChannel;
using vkernels::comm::P2PKvDonatePlan;
using vkernels::comm::P2PKvRestorePlan;
using vkernels::comm::select_cross_node_kv_route;

namespace {

// Layout constants for every test (one shape, BF16). num_pages=4, page_size=3,
// heads=2, head_dim=4, elem=2, slots=16, layers=5 (a Qwen3-14B-style 40-layer
// workload scaled down). The slot map is unique for the restore (scatter)
// and reused for the donate (gather); a second non-unique map exercises the
// donate's repeat-allowed semantics.
constexpr std::size_t kPages = 4, kPageSize = 3, kHeads = 2, kHeadDim = 4;
constexpr std::size_t kElem = 2, kSlots = 16, kLayers = 5;
constexpr std::size_t kSlotBytes = kHeads * kHeadDim * kElem;      // 16
constexpr std::size_t kTokenStride = 2 * kSlotBytes;               // 32
constexpr std::size_t kPageLayerBytes = kPageSize * kTokenStride;  // 96
constexpr std::size_t kLayerBytes = kPages * kPageLayerBytes;      // 384

// A unique slot map for the restore (scatter). The same indices are a
// valid donate source map.
const int* unique_slots() {
  static const int s[kPages * kPageSize] = {3, 1, 14, 7, 0, 9, 6, 12, 2, 5, 8, 11};
  return s;
}

// A non-unique slot map (gather semantics) for the donate: tokens 0..2 of
// every page reuse slots {0, 1, 0} -- index 0 repeats, which the restore
// must reject and the donate must accept.
const int* repeat_slots() {
  static const int s[kPages * kPageSize] = {0, 1, 0,  4, 5, 4,  8, 9, 8,  12, 13, 12};
  return s;
}

// One full remote allocation laid out [num_layers, num_pages, page_size, 2,
// heads, head_dim], filled deterministically so mis-routed reads are
// obvious. This is the "peer pages on host B" buffer.
std::vector<std::uint8_t> make_remote(std::uint8_t seed = 0x40) {
  std::vector<std::uint8_t> r(kLayers * kLayerBytes);
  for (std::size_t i = 0; i < r.size(); ++i)
    r[i] = static_cast<std::uint8_t>(seed + i);
  return r;
}

// A local K/V pair ([num_slots, heads, head_dim] each), filled
// deterministically. This is "local K/V on host A".
struct LocalKv {
  std::vector<std::uint8_t> k, v;
};
LocalKv make_local(std::uint8_t kseed = 0x80, std::uint8_t vseed = 0xC0) {
  const std::size_t n = kSlots * kHeads * kHeadDim * kElem;
  LocalKv kv;
  kv.k.resize(n);
  kv.v.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    kv.k[i] = static_cast<std::uint8_t>(kseed + i);
    kv.v[i] = static_cast<std::uint8_t>(vseed + i);
  }
  return kv;
}

// Per-page peer base pointers into a remote allocation, laid out exactly as
// the same-node path: peer[p] = base + p * kPageLayerBytes. The plan adds
// the scalar layer offset at execute time.
std::vector<const void*> peer_bases_in(const void* base) {
  std::vector<const void*> p(kPages);
  const auto* b = static_cast<const std::uint8_t*>(base);
  for (std::size_t i = 0; i < kPages; ++i) p[i] = b + i * kPageLayerBytes;
  return p;
}
std::vector<void*> peer_bases_out(void* base) {
  std::vector<void*> p(kPages);
  auto* b = static_cast<std::uint8_t*>(base);
  for (std::size_t i = 0; i < kPages; ++i) p[i] = b + i * kPageLayerBytes;
  return p;
}

// Byte-wise equality of two uint8 buffers (minitest cannot stream a
// std::vector<uint8_t>, so the assertions compare element-by-element and
// this helper gives EXPECT_TRUE a single boolean with a tight message).
bool same_bytes(const std::vector<std::uint8_t>& a,
                const std::vector<std::uint8_t>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size()) == 0;
}

// Snapshot of a FabricHandle's owned bytes. FabricHandle makes its own copy
// of the published memory at construction and never references the caller
// buffer again, so the observable result of a cross-node donate (which
// write_back() ships to the handle, or aliases directly on kSameNodePeer)
// is read through h.bytes().
std::vector<std::uint8_t> handle_bytes(const FabricHandle& h) {
  return std::vector<std::uint8_t>(h.bytes(), h.bytes() + h.size());
}

}  // namespace

TEST(CrossNodeKvRoute, SingleConsumerAlwaysUsesPointToPoint) {
  CrossNodeKvAccess access{4, 1, true, true, true};
  FabricImportConfig fabric;
  fabric.has_gpudirect_rdma = true;
  const auto route = select_cross_node_kv_route(access, fabric);
  EXPECT_TRUE(route.kind == CrossNodeKvTransferKind::kPointToPoint);
  EXPECT_EQ(route.point_to_point_transport,
            FabricImportTransport::kFabricMapped);
  EXPECT_TRUE(route.graph_capturable);
}

TEST(CrossNodeKvRoute, FullEvenShardUsesAvailableCollective) {
  CrossNodeKvAccess access{4, 4, true, true, true};
  FabricImportConfig fabric;
  const auto route = select_cross_node_kv_route(access, fabric);
  EXPECT_TRUE(route.kind == CrossNodeKvTransferKind::kAllGather);
  EXPECT_EQ(route.point_to_point_transport,
            FabricImportTransport::kHostBounce);
  EXPECT_TRUE(route.graph_capturable);
}

TEST(CrossNodeKvRoute, RaggedOrUnavailableCollectiveFallsBackToP2P) {
  FabricImportConfig fabric;
  CrossNodeKvAccess ragged{4, 4, false, true, true};
  CrossNodeKvAccess unavailable{4, 4, true, false, true};
  EXPECT_TRUE(select_cross_node_kv_route(ragged, fabric).kind ==
              CrossNodeKvTransferKind::kPointToPoint);
  EXPECT_TRUE(select_cross_node_kv_route(unavailable, fabric).kind ==
              CrossNodeKvTransferKind::kPointToPoint);
  EXPECT_FALSE(select_cross_node_kv_route(ragged, fabric).graph_capturable);
}

TEST(CrossNodeKvRoute, RejectsInvalidAccessShape) {
  FabricImportConfig fabric;
  EXPECT_THROW(select_cross_node_kv_route({0, 1, true, true, true}, fabric),
               std::invalid_argument);
  EXPECT_THROW(select_cross_node_kv_route({2, 0, true, true, true}, fabric),
               std::invalid_argument);
  EXPECT_THROW(select_cross_node_kv_route({2, 3, true, true, true}, fabric),
               std::invalid_argument);
}

TEST(CrossNodeKvRoute, CollectiveGraphCapabilityIsIndependent) {
  FabricImportConfig fabric;
  CrossNodeKvAccess access{4, 4, true, true, false};
  const auto route = select_cross_node_kv_route(access, fabric);
  EXPECT_TRUE(route.kind == CrossNodeKvTransferKind::kAllGather);
  EXPECT_FALSE(route.graph_capturable);
}

// ---------------------------------------------------------------------------
// ByteChannel + make_byte_link (the byte-capable transport)
// ---------------------------------------------------------------------------

TEST(ByteChannel, MakeByteLinkFormsDirectedPair) {
  auto link = make_byte_link();
  auto& a = link.first;
  auto& b = link.second;
  EXPECT_FALSE(a->closed());
  EXPECT_FALSE(b->closed());

  // a sends -> b receives
  std::vector<std::uint8_t> msg{0xDE, 0xAD, 0xBE, 0xEF};
  a->send(msg);
  auto got = b->recv();
  ASSERT_EQ(got.size(), 4u);
  EXPECT_EQ(got[0], 0xDE);
  EXPECT_EQ(got[3], 0xEF);

  // b sends -> a receives (independent direction)
  std::vector<std::uint8_t> reply{0x11, 0x22};
  b->send(std::move(reply));
  auto back = a->recv();
  ASSERT_EQ(back.size(), 2u);
  EXPECT_EQ(back[0], 0x11);
}

TEST(ByteChannel, MockNeedsBothQueues) {
  EXPECT_THROW(MockByteChannel(nullptr, nullptr), std::invalid_argument);
}

TEST(ByteChannel, BlockingRecvWakesAfterSend) {
  // One in each direction; recv blocks until the peer pushes. Run the
  // producer on a thread so the recv is observed to block then wake.
  auto a_out = std::make_shared<ByteBlockingQueue>();  // a -> b
  auto b_out = std::make_shared<ByteBlockingQueue>();  // b -> a
  MockByteChannel a(a_out, b_out);
  MockByteChannel b(b_out, a_out);
  std::vector<std::uint8_t> got;
  std::thread consumer([&] { got = b.recv(); });
  std::vector<std::uint8_t> payload(kPageLayerBytes, 0x55);
  a.send(payload);
  consumer.join();
  ASSERT_EQ(got.size(), payload.size());
  EXPECT_EQ(got.front(), 0x55);
}

// ---------------------------------------------------------------------------
// Fabric import classification + helpers (the planning surface)
// ---------------------------------------------------------------------------

TEST(FabricImport, ClassifiesEveryPath) {
  // Same-node peer (NVLink / HBM / ROCm IPC): the existing same-node path,
  // no import needed, always graph-capturable.
  {
    FabricImportConfig cfg; cfg.same_node = true;
    EXPECT_EQ(classify_fabric_import(cfg), FabricImportTransport::kSameNodePeer);
  }
  // The GH200 libfabric constraint: even with GPUDirect-RDMA on paper the
  // C2C-attached GPU is invisible to the fabric plugin (DRAM only), so
  // cross-node VRAM MUST bounce through a pinned host buffer. Takes
  // precedence over has_gpudirect_rdma.
  {
    FabricImportConfig cfg; cfg.dram_only_libfabric = true; cfg.has_gpudirect_rdma = true;
    EXPECT_EQ(classify_fabric_import(cfg), FabricImportTransport::kHostBounce);
  }
  // A GPUDirect-RDMA / CU_MEM_HANDLE_TYPE_FABRIC path is available: import
  // the remote VRAM and dereference directly.
  {
    FabricImportConfig cfg; cfg.has_gpudirect_rdma = true;
    EXPECT_EQ(classify_fabric_import(cfg), FabricImportTransport::kFabricMapped);
  }
  // No fabric path at all: host-bounce, eager-break.
  {
    FabricImportConfig cfg;  // all defaults false
    EXPECT_EQ(classify_fabric_import(cfg), FabricImportTransport::kHostBounce);
  }
}

TEST(FabricImport, GraphCapturableAndEagerBreak) {
  EXPECT_TRUE(is_import_graph_capturable(FabricImportTransport::kFabricMapped));
  EXPECT_TRUE(is_import_graph_capturable(FabricImportTransport::kSameNodePeer));
  EXPECT_FALSE(is_import_graph_capturable(FabricImportTransport::kHostBounce));

  FabricImportConfig capturable; capturable.same_node = true;
  EXPECT_FALSE(eager_break_fabric_import(capturable));
  FabricImportConfig dram; dram.dram_only_libfabric = true;
  EXPECT_TRUE(eager_break_fabric_import(dram));
  FabricImportConfig rdma; rdma.has_gpudirect_rdma = true;
  EXPECT_FALSE(eager_break_fabric_import(rdma));
  FabricImportConfig none;
  EXPECT_TRUE(eager_break_fabric_import(none));
}

TEST(FabricImport, TransportNamePrints) {
  std::ostringstream os;
  os << FabricImportTransport::kFabricMapped << ' '
     << FabricImportTransport::kSameNodePeer << ' '
     << FabricImportTransport::kHostBounce;
  EXPECT_EQ(os.str(), std::string("fabric-mapped same-node-peer host-bounce"));
  (void)fabric_import_transport_name(FabricImportTransport::kFabricMapped);
}

// ---------------------------------------------------------------------------
// FabricHandle / FabricImport host model
// ---------------------------------------------------------------------------

TEST(FabricImport, HandleRejectsZeroAndNull) {
  std::uint8_t b = 1;
  EXPECT_THROW(FabricHandle(0, 0, &b, 0), std::invalid_argument);
  EXPECT_THROW(FabricHandle(0, 0, nullptr, 4), std::invalid_argument);
}

TEST(FabricImport, FabricMappedMirrorsAndExposesDevicePtr) {
  auto remote = make_remote();
  FabricHandle h(/*remote_node=*/1, /*token=*/42, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  EXPECT_TRUE(imp.has_device_ptr());
  EXPECT_EQ(imp.size(), remote.size());
  // The mirror is byte-identical to the remote handle (the local address
  // space now names the remote memory).
  EXPECT_EQ(std::memcmp(imp.device_ptr(), remote.data(), remote.size()), 0);
  EXPECT_TRUE(imp.is_graph_capturable());
}

TEST(FabricImport, SameNodePeerAliasesRemoteBytes) {
  auto remote = make_remote();
  FabricHandle h(1, 7, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kSameNodePeer, h);
  // Same-node: the imported pointer ALIASES the published FabricHandle
  // bytes (no copy), exactly as the same-node NVLink path aliases a peer
  // pointer. (FabricHandle owns its own copy of `remote`; the alias is
  // into h.bytes(), which is byte-identical to `remote`.)
  EXPECT_TRUE(imp.has_device_ptr());
  EXPECT_EQ(imp.device_ptr(), static_cast<void*>(h.bytes()));
  EXPECT_EQ(std::memcmp(imp.device_ptr(), remote.data(), remote.size()), 0);
}

TEST(FabricImport, HostBounceHasNoDevicePtr) {
  auto remote = make_remote();
  FabricHandle h(1, 9, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kHostBounce, h);
  EXPECT_FALSE(imp.has_device_ptr());
  EXPECT_EQ(imp.device_ptr(), nullptr);
  // The mirror is still owned for the fallback's correctness.
  EXPECT_EQ(imp.size(), remote.size());
}

TEST(FabricImport, DefaultIsEmptyBouncePlaceholder) {
  FabricImport imp;
  EXPECT_FALSE(imp.has_device_ptr());
  EXPECT_EQ(imp.transport(), FabricImportTransport::kHostBounce);
  EXPECT_EQ(imp.size(), 0u);
}

TEST(FabricImport, WriteBackShipsMirrorToHandle) {
  // A kFabricMapped donate writes the imported MIRROR (an owned copy of
  // the handle bytes); write_back() ships the mirror to the remote
  // FabricHandle. Mutate the mirror through device_ptr(), then write_back,
  // and confirm the HANDLE picked the mutation up (the original caller
  // buffer is never referenced after construction -- FabricHandle owns
  // its own copy).
  auto remote = make_remote();
  FabricHandle h(1, 1, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  auto* p = static_cast<std::uint8_t*>(imp.device_ptr());
  p[0] = 0xFE; p[1] = 0xED;
  // The handle is still pristine (the mirror is an OWNED copy).
  EXPECT_NE(h.bytes()[0], 0xFE);
  imp.write_back(h);
  EXPECT_EQ(h.bytes()[0], 0xFE);
  EXPECT_EQ(h.bytes()[1], 0xED);
}

TEST(FabricImport, WriteBackIsNoOpForSameNodeAndBounce) {
  // kSameNodePeer: the donate wrote directly through the aliased peer
  // pointer (no separate mirror), so write_back is a no-op -- the bytes
  // already landed in the handle.
  {
    auto remote = make_remote();
    FabricHandle h(1, 1, remote.data(), remote.size());
    FabricImport imp(FabricImportTransport::kSameNodePeer, h);
    static_cast<std::uint8_t*>(imp.device_ptr())[0] = 0xAB;
    imp.write_back(h);  // no-op
    EXPECT_EQ(h.bytes()[0], 0xAB);  // unchanged by the no-op
  }
  // kHostBounce: the fallback writes the remote out-of-band, so write_back
  // is a no-op here (the handle is untouched).
  {
    auto remote = make_remote();
    FabricHandle h(1, 1, remote.data(), remote.size());
    FabricImport imp(FabricImportTransport::kHostBounce, h);
    std::vector<std::uint8_t> pristine(h.bytes(), h.bytes() + h.size());
    imp.write_back(h);
    EXPECT_TRUE(same_bytes(std::vector<std::uint8_t>(h.bytes(), h.bytes() + h.size()),
                           pristine));
  }
}

TEST(FabricImport, MovePreservesDevicePtrIntoMirror) {
  // kFabricMapped owns a mirror; device_ptr_ points into it. A move
  // transfers the mirror buffer and re-bases device_ptr_ so the byte
  // round-trips and the moved-from is the empty placeholder.
  auto remote = make_remote();
  FabricHandle h(1, 1, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  std::uint8_t first = static_cast<std::uint8_t*>(imp.device_ptr())[0];
  FabricImport moved = std::move(imp);
  ASSERT_TRUE(moved.has_device_ptr());
  EXPECT_EQ(static_cast<std::uint8_t*>(moved.device_ptr())[0], first);
  EXPECT_FALSE(imp.has_device_ptr());
  EXPECT_EQ(imp.transport(), FabricImportTransport::kHostBounce);
}

TEST(FabricImport, MovePreservesSameNodePeerAlias) {
  // kSameNodePeer aliases a borrowed FabricHandle. The handle does not move
  // with the import, so the alias survives a move and still dereferences to
  // the same byte.
  auto remote = make_remote();
  FabricHandle h(1, 2, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kSameNodePeer, h);
  void* aliased = imp.device_ptr();
  EXPECT_EQ(aliased, static_cast<void*>(h.bytes()));
  FabricImport moved = std::move(imp);
  ASSERT_TRUE(moved.has_device_ptr());
  // Still aliases the same handle (external, did not move).
  EXPECT_EQ(moved.device_ptr(), static_cast<void*>(h.bytes()));
  EXPECT_EQ(static_cast<std::uint8_t*>(moved.device_ptr())[0], h.bytes()[0]);
  EXPECT_FALSE(imp.has_device_ptr());
}

TEST(FabricImport, MoveAssignmentPreservesDevicePtrIntoMirror) {
  // kFabricMapped move-ASSIGNMENT transfers the owned mirror and re-bases
  // device_ptr_ onto it (the same rebase as the move ctor), so the byte
  // round-trips and the moved-from is the empty placeholder.
  auto remote = make_remote();
  FabricHandle h(1, 1, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  std::uint8_t first = static_cast<std::uint8_t*>(imp.device_ptr())[0];
  FabricImport moved;  // empty placeholder
  moved = std::move(imp);
  ASSERT_TRUE(moved.has_device_ptr());
  EXPECT_EQ(static_cast<std::uint8_t*>(moved.device_ptr())[0], first);
  EXPECT_FALSE(imp.has_device_ptr());
  EXPECT_EQ(imp.transport(), FabricImportTransport::kHostBounce);
}

TEST(FabricImport, MoveAssignmentPreservesSameNodePeerAlias) {
  // kSameNodePeer move-ASSIGNMENT keeps the borrowed alias (the handle
  // does not move with the import) and clears the moved-from.
  auto remote = make_remote();
  FabricHandle h(1, 2, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kSameNodePeer, h);
  FabricImport moved;
  moved = std::move(imp);
  ASSERT_TRUE(moved.has_device_ptr());
  EXPECT_EQ(moved.device_ptr(), static_cast<void*>(h.bytes()));
  EXPECT_FALSE(imp.has_device_ptr());
}

TEST(FabricImport, MoveAssignmentSelfIsNoOp) {
  // Self move-assignment is the early `if (this == &other) return *this;`
  // path: the import is left untouched (mirror + device_ptr intact).
  auto remote = make_remote();
  FabricHandle h(1, 3, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  std::uint8_t first = static_cast<std::uint8_t*>(imp.device_ptr())[0];
  FabricImport& ref = imp;
  imp = std::move(ref);  // self-assignment: early return
  ASSERT_TRUE(imp.has_device_ptr());
  EXPECT_EQ(static_cast<std::uint8_t*>(imp.device_ptr())[0], first);
}

TEST(FabricImport, WriteBackRejectsUndersizedRemote) {
  auto remote = make_remote();
  FabricHandle big(1, 1, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, big);
  std::vector<std::uint8_t> small(8, 0);
  FabricHandle tiny(1, 1, small.data(), small.size());
  EXPECT_THROW(imp.write_back(tiny), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Cost model -- per-hop throughput vs the same-node roofline
// ---------------------------------------------------------------------------

TEST(CrossNodeCost, ZeroBytesIsZero) {
  auto c = cross_node_kv_throughput(FabricImportTransport::kFabricMapped, 0);
  EXPECT_EQ(c.per_hop_gbps, 0.0);
  EXPECT_EQ(c.per_hop_us, 0.0);
  EXPECT_EQ(c.total_bytes, 0u);
}

TEST(CrossNodeCost, FabricMappedBelowRoofline) {
  const std::size_t bytes = 1u << 20;  // 1 MiB
  auto c = cross_node_kv_throughput(FabricImportTransport::kFabricMapped, bytes);
  // The binding resource is min(fabric link 50, kernel roof 88.5) = 50 GB/s.
  EXPECT_NEAR(c.per_hop_gbps, 50.0, 1e-9);
  EXPECT_GT(c.per_hop_us, 0.0);
  EXPECT_NEAR(c.same_node_roof_gbps, 88.5, 1e-9);
  EXPECT_LT(c.per_hop_gbps, c.same_node_roof_gbps);
  EXPECT_FALSE(c.gh200_dram_only_caveat);
  // per_hop_us == bytes / (gbps * 1e9) * 1e6
  const double expect_us = static_cast<double>(bytes) / (50.0 * 1e9) * 1e6;
  EXPECT_NEAR(c.per_hop_us, expect_us, 1e-6);
}

TEST(CrossNodeCost, SameNodePeerHitsRoof) {
  const std::size_t bytes = 1u << 20;
  auto c = cross_node_kv_throughput(FabricImportTransport::kSameNodePeer, bytes);
  EXPECT_NEAR(c.per_hop_gbps, 88.5, 1e-9);
  EXPECT_NEAR(c.same_node_roof_gbps, 88.5, 1e-9);
  EXPECT_FALSE(c.gh200_dram_only_caveat);
}

TEST(CrossNodeCost, HostBounceFlagsCaveatAndUsesFallback) {
  const std::size_t bytes = 1u << 20;
  auto c = cross_node_kv_throughput(FabricImportTransport::kHostBounce, bytes);
  EXPECT_TRUE(c.gh200_dram_only_caveat);
  EXPECT_NEAR(c.per_hop_gbps, c.bulk_copy_fallback_gbps, 1e-9);
  EXPECT_LT(c.per_hop_gbps, c.same_node_roof_gbps);
}

TEST(CrossNodeCost, SameNodeRoofMatchesTransport) {
  // same_node_fabric_roof_gbps is the documented public roof for a given
  // transport (NOT the always-88.5 cross-node reference field on
  // CrossNodeHopCost -- kHostBounce's own roof is the bulk-copy fallback).
  using vkernels::comm::same_node_fabric_roof_gbps;
  // kFabricMapped / kSameNodePeer -> the fused-restore roof (88.5).
  EXPECT_NEAR(same_node_fabric_roof_gbps(FabricImportTransport::kFabricMapped),
              88.5, 1e-9);
  EXPECT_NEAR(same_node_fabric_roof_gbps(FabricImportTransport::kSameNodePeer),
              88.5, 1e-9);
  // kHostBounce -> the synchronous bulk-copy fallback (much lower).
  EXPECT_NEAR(same_node_fabric_roof_gbps(FabricImportTransport::kHostBounce),
              1.4, 1e-9);
  // Acceptance #2: the fallback is strictly below the same-node roof.
  EXPECT_GT(same_node_fabric_roof_gbps(FabricImportTransport::kFabricMapped),
            same_node_fabric_roof_gbps(FabricImportTransport::kHostBounce));
}

TEST(CrossNodeCost, Gh200DegradesFabricMappedToBounce) {
  const std::size_t bytes = 1u << 20;
  auto c = cross_node_kv_throughput(FabricImportTransport::kFabricMapped, bytes,
                                    /*gh200_dram_only=*/true);
  // On GH200 the DRAM-only libfabric constraint forces a host bounce EVEN
  // though a fabric import is requested -- the caveat the issue calls out.
  EXPECT_TRUE(c.gh200_dram_only_caveat);
  EXPECT_NEAR(c.per_hop_gbps, c.bulk_copy_fallback_gbps, 1e-9);
}

// ---------------------------------------------------------------------------
// CrossNodeKvRestorePlan -- byte-identical to the same-node NVLink path
// ---------------------------------------------------------------------------

TEST(CrossNodeRestore, DirectPathMatchesSameNode) {
  // The same remote allocation is read by (a) the same-node P2PKvRestorePlan
  // and (b) the cross-node plan over a fabric-mapped import. Both lay their
  // per-page peer bases as base + p * kPageLayerBytes and add the same
  // scalar layer offset, so both read the exact same bytes.
  auto remote = make_remote();
  const int* slots = unique_slots();
  auto pb = peer_bases_in(remote.data());

  auto same = make_local(0, 0);  // K/V destinations for the same-node path
  same.k.assign(same.k.size(), 0xCC);
  same.v.assign(same.v.size(), 0xCC);
  P2PKvRestorePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                             slots, pb.data(), kPages, kPageSize);

  FabricHandle h(/*remote_node=*/1, /*token=*/2, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  ASSERT_TRUE(imp.has_device_ptr());
  CrossNodeKvRestorePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               FabricImportTransport::kFabricMapped, &imp);

  EXPECT_EQ(xplan.num_pages(), kPages);
  EXPECT_EQ(xplan.total_bytes(), kLayerBytes);
  EXPECT_EQ(xplan.bounce_bytes(), kLayerBytes);
  EXPECT_TRUE(xplan.is_graph_capturable());

  // Restore every layer; the cross-node and same-node results must match
  // for each layer (source_layer_offset_bytes = L * kLayerBytes).
  for (std::size_t L = 0; L < kLayers; ++L) {
    auto ko = make_local(0, 0); ko.k.assign(ko.k.size(), 0); ko.v.assign(ko.v.size(), 0);
    auto xo = make_local(0, 0); xo.k.assign(xo.k.size(), 0); xo.v.assign(xo.v.size(), 0);
    same_plan.execute(ko.k.data(), ko.v.data(), L * kLayerBytes);
    xplan.execute(xo.k.data(), xo.v.data(), L * kLayerBytes);
    ASSERT_TRUE(same_bytes(ko.k, xo.k));
    ASSERT_TRUE(same_bytes(ko.v, xo.v));
  }
}

TEST(CrossNodeRestore, SameNodePeerMatchesSameNode) {
  // kSameNodePeer: the import aliases the published FabricHandle bytes
  // (byte-identical to the remote allocation), so the cross-node plan reads
  // the exact same memory as the same-node path.
  auto remote = make_remote();
  const int* slots = unique_slots();
  auto pb = peer_bases_in(remote.data());

  FabricHandle h(1, 3, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kSameNodePeer, h);
  ASSERT_EQ(imp.device_ptr(), static_cast<void*>(h.bytes()));
  ASSERT_EQ(std::memcmp(imp.device_ptr(), remote.data(), remote.size()), 0);
  CrossNodeKvRestorePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               FabricImportTransport::kSameNodePeer, &imp);

  P2PKvRestorePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                             slots, pb.data(), kPages, kPageSize);
  auto ko = make_local(0, 0); ko.k.assign(ko.k.size(), 0); ko.v.assign(ko.v.size(), 0);
  auto xo = make_local(0, 0); xo.k.assign(xo.k.size(), 0); xo.v.assign(xo.v.size(), 0);
  same_plan.execute(ko.k.data(), ko.v.data(), 2 * kLayerBytes);
  xplan.execute(xo.k.data(), xo.v.data(), 2 * kLayerBytes);
  ASSERT_TRUE(same_bytes(ko.k, xo.k));
  ASSERT_TRUE(same_bytes(ko.v, xo.v));
}

TEST(CrossNodeRestore, AsyncExecuteIsOneTaskAndCorrect) {
  auto remote = make_remote();
  const int* slots = unique_slots();
  FabricHandle h(1, 4, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  CrossNodeKvRestorePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               FabricImportTransport::kFabricMapped, &imp);

  Stream s;
  const std::size_t before = s.submitted();
  auto xo = make_local(0, 0); xo.k.assign(xo.k.size(), 0); xo.v.assign(xo.v.size(), 0);
  xplan.execute(xo.k.data(), xo.v.data(), kLayerBytes, &s);
  EXPECT_EQ(s.submitted(), before + 1u);
  s.wait();

  // Compare to the synchronous same-node result.
  auto pb = peer_bases_in(remote.data());
  P2PKvRestorePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                             slots, pb.data(), kPages, kPageSize);
  auto ko = make_local(0, 0); ko.k.assign(ko.k.size(), 0); ko.v.assign(ko.v.size(), 0);
  same_plan.execute(ko.k.data(), ko.v.data(), kLayerBytes);
  ASSERT_TRUE(same_bytes(ko.k, xo.k));
  ASSERT_TRUE(same_bytes(ko.v, xo.v));
}

TEST(CrossNodeRestore, HostBounceStreamIsOneTask) {
  // kHostBounce restore with a non-null stream (no graph) submits the
  // recv+scatter as ONE stream task (the `stream->submit(run)` path).
  auto link = make_byte_link();
  const int* slots = unique_slots();
  // Pre-send one layer so the restorer's recv is non-blocking.
  link.first->send(std::vector<std::uint8_t>(kLayerBytes, 0x77));

  CrossNodeKvRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem, slots,
                                  kPages, kPageSize,
                                  FabricImportTransport::kHostBounce, nullptr);
  Stream s;
  const std::size_t before = s.submitted();
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  restorer.execute(out.k.data(), out.v.data(), 0, &s,
                   /*graph=*/nullptr, link.second.get());
  EXPECT_EQ(s.submitted(), before + 1u);  // exactly one task
  s.wait();
  // The recv'd layer was scattered into the indexed local slots.
  EXPECT_TRUE(std::any_of(out.k.begin(), out.k.end(),
                          [](std::uint8_t b) { return b == 0x77; }));
}

TEST(CrossNodeRestore, ZeroPagesIsNoOp) {
  // num_pages == 0 is a valid no-op plan: no import needed, execute is a no-op.
  const int slots[1] = {0};
  CrossNodeKvRestorePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots, 0,
                               kPageSize, FabricImportTransport::kHostBounce,
                               nullptr);
  EXPECT_EQ(xplan.num_pages(), 0u);
  EXPECT_EQ(xplan.total_bytes(), 0u);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0x42); out.v.assign(out.v.size(), 0x42);
  xplan.execute(out.k.data(), out.v.data(), 0, nullptr, nullptr, nullptr);
  EXPECT_EQ(out.k.front(), 0x42);  // untouched
}

TEST(CrossNodeRestore, HugeNumPagesPageSizeOverflows) {
  // slot_map.hpp::checked_mul (issue #49): num_pages * page_size exceeds
  // size_t -> std::invalid_argument instead of silent wraparound. Validated
  // geometry (positive dims, elem_size == 2) with a non-graph-capturable
  // transport so the import is never touched; the throw happens before
  // slot_ids is read, so a dangling pointer is safe here.
  constexpr std::size_t kHuge = ~std::size_t{0};  // SIZE_MAX
  int dummy = 0;
  EXPECT_THROW(CrossNodeKvRestorePlan(kSlots, kHeads, kHeadDim, kElem,
                                      &dummy, kHuge, 2,
                                      FabricImportTransport::kHostBounce,
                                      nullptr),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
// CrossNodeKvDonatePlan -- byte-identical to the same-node NVLink path
// ---------------------------------------------------------------------------

TEST(CrossNodeDonate, DirectPathMatchesSameNode) {
  // The same local K/V is donated by (a) the same-node P2PKvDonatePlan and
  // (b) the cross-node plan over a fabric-mapped import. Both write into the
  // remote allocation at the same per-page offsets + layer offset, so the
  // remote bytes must match for every layer.
  auto local = make_local();

  auto pb_same = peer_bases_out(make_remote().data());  // placeholders
  const int* slots = unique_slots();

  for (std::size_t L = 0; L < kLayers; ++L) {
    auto remote_same = make_remote();
    auto remote_cross = make_remote();
    ASSERT_TRUE(same_bytes(remote_same, remote_cross));

    // Same-node path: peer bases into remote_same.
    auto pbs = peer_bases_out(remote_same.data());
    P2PKvDonatePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                              slots, pbs.data(), kPages, kPageSize);
    same_plan.execute(local.k.data(), local.v.data(), L * kLayerBytes);

    // Cross-node path: fabric-mapped import over remote_cross, then
    // write_back ships the mirror to the remote handle.
    FabricHandle h(1, 1, remote_cross.data(), remote_cross.size());
    FabricImport imp(FabricImportTransport::kFabricMapped, h);
    CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                                kPages, kPageSize,
                                FabricImportTransport::kFabricMapped, &imp);
    xplan.execute(local.k.data(), local.v.data(), /*remote=*/&h,
                  L * kLayerBytes);

    // The cross-node donate landed in the handle (write_back shipped the
    // imported mirror there); compare against the same-node result.
    ASSERT_TRUE(same_bytes(remote_same, handle_bytes(h)));
  }
  (void)pb_same;
}

TEST(CrossNodeDonate, SameNodePeerMatchesSameNode) {
  // kSameNodePeer: the import aliases the remote bytes, so the cross-node
  // donate writes directly into the remote allocation -- byte-identical to
  // the same-node path, and write_back is a no-op.
  auto local = make_local();
  auto remote_same = make_remote();
  auto remote_cross = make_remote();
  const int* slots = unique_slots();

  auto pbs = peer_bases_out(remote_same.data());
  P2PKvDonatePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                            slots, pbs.data(), kPages, kPageSize);
  same_plan.execute(local.k.data(), local.v.data(), 1 * kLayerBytes);

  FabricHandle h(1, 5, remote_cross.data(), remote_cross.size());
  FabricImport imp(FabricImportTransport::kSameNodePeer, h);
  CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kSameNodePeer, &imp);
  xplan.execute(local.k.data(), local.v.data(), /*remote=*/&h, 1 * kLayerBytes);
  // The kSameNodePeer donate wrote directly through the aliased peer
  // pointer into the handle (write_back is a no-op).
  ASSERT_TRUE(same_bytes(remote_same, handle_bytes(h)));
}

TEST(CrossNodeDonate, AcceptsRepeatedSlots) {
  // The donate is a gather, so source slots may repeat (the restore would
  // reject this map). The cross-node plan must accept it, same as the
  // same-node plan.
  auto local = make_local();
  auto remote_cross = make_remote();
  auto remote_same = make_remote();
  const int* slots = repeat_slots();

  FabricHandle h(1, 6, remote_cross.data(), remote_cross.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kFabricMapped, &imp);
  EXPECT_NO_THROW(xplan.execute(local.k.data(), local.v.data(), /*remote=*/&h, 0));

  auto pbs = peer_bases_out(remote_same.data());
  P2PKvDonatePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                            slots, pbs.data(), kPages, kPageSize);
  same_plan.execute(local.k.data(), local.v.data(), 0);
  ASSERT_TRUE(same_bytes(remote_same, handle_bytes(h)));
}

TEST(CrossNodeDonate, AsyncExecuteIsOneTaskAndCorrect) {
  auto local = make_local();
  auto remote_cross = make_remote();
  const int* slots = unique_slots();

  FabricHandle h(1, 7, remote_cross.data(), remote_cross.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kFabricMapped, &imp);

  Stream s;
  const std::size_t before = s.submitted();
  xplan.execute(local.k.data(), local.v.data(), /*remote=*/&h, kLayerBytes, &s);
  s.wait();
  // One task for the device write (the host write_back runs after wait).
  EXPECT_EQ(s.submitted(), before + 1u);

  auto remote_same = make_remote();
  auto pbs = peer_bases_out(remote_same.data());
  P2PKvDonatePlan same_plan(kSlots, kHeads, kHeadDim, kElem,
                            slots, pbs.data(), kPages, kPageSize);
  same_plan.execute(local.k.data(), local.v.data(), kLayerBytes);
  ASSERT_TRUE(same_bytes(remote_same, handle_bytes(h)));
}

TEST(CrossNodeDonate, HostBounceStreamIsOneTask) {
  // kHostBounce donate with a non-null stream (no graph) submits the
  // gather+send as ONE stream task (the `stream->submit(run)` path).
  auto link = make_byte_link();
  const int* slots = unique_slots();
  auto local = make_local();

  CrossNodeKvDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kHostBounce, nullptr);
  Stream s;
  const std::size_t before = s.submitted();
  donor.execute(local.k.data(), local.v.data(), /*remote=*/nullptr, 0, &s,
                /*graph=*/nullptr, link.first.get());
  EXPECT_EQ(s.submitted(), before + 1u);  // exactly one task
  s.wait();
  // The gathered layer was sent over the channel.
  std::vector<std::uint8_t> recv = link.second->recv();
  ASSERT_TRUE(recv.size() >= kLayerBytes);
}

TEST(CrossNodeDonate, ZeroPagesIsNoOp) {
  const int slots[1] = {0};
  CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots, 0,
                              kPageSize, FabricImportTransport::kHostBounce,
                              nullptr);
  EXPECT_EQ(xplan.num_pages(), 0u);
  EXPECT_EQ(xplan.total_bytes(), 0u);
  EXPECT_EQ(xplan.scratch_bytes(), 0u);
  // No remote handle needed for a no-op.
  std::uint8_t dummy = 0;
  EXPECT_NO_THROW(xplan.execute(&dummy, &dummy, nullptr, 0, nullptr, nullptr, nullptr));
}

// ---------------------------------------------------------------------------
// Host-bounce round-trip (donate on A -> restore on B over a ByteChannel)
// ---------------------------------------------------------------------------

TEST(CrossNodeHostBounce, DonateRestoreRoundTripMatchesDirect) {
  // The full cross-node round-trip over the host-bounce transport:
  //   A donates layer L (kHostBounce) -> gathers local K/V, sends over a
  //   ByteChannel.
  //   B restores layer L (kHostBounce) -> recvs the layer, scatters into
  //   local K/V slots.
  // The result on B must match the direct (fabric-mapped) round-trip:
  //   A donates layer L into B's remote allocation, B restores from it.
  auto local = make_local();
  const int* slots = unique_slots();

  auto link = make_byte_link();
  auto& chan_a = link.first;   // A sends into B's queue
  auto& chan_b = link.second;  // B sends into A's queue

  // ---- Direct reference: A donates to R, B restores from R ----
  auto remote = make_remote();
  FabricHandle hR(1, 1, remote.data(), remote.size());
  FabricImport impR(FabricImportTransport::kFabricMapped, hR);
  CrossNodeKvDonatePlan donor_direct(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize,
                                     FabricImportTransport::kFabricMapped, &impR);
  donor_direct.execute(local.k.data(), local.v.data(), /*remote=*/&hR, 0,
                       nullptr, nullptr, nullptr);

  CrossNodeKvRestorePlan restorer_direct(kSlots, kHeads, kHeadDim, kElem, slots,
                                         kPages, kPageSize,
                                         FabricImportTransport::kFabricMapped, &impR);
  auto kv_direct = make_local(0, 0);
  kv_direct.k.assign(kv_direct.k.size(), 0); kv_direct.v.assign(kv_direct.v.size(), 0);
  restorer_direct.execute(kv_direct.k.data(), kv_direct.v.data(), 0, nullptr, nullptr, nullptr);

  // ---- Host-bounce round-trip over the ByteChannel ----
  CrossNodeKvDonatePlan donor_bounce(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize,
                                     FabricImportTransport::kHostBounce, nullptr);
  CrossNodeKvRestorePlan restorer_bounce(kSlots, kHeads, kHeadDim, kElem, slots,
                                         kPages, kPageSize,
                                         FabricImportTransport::kHostBounce, nullptr);

  auto kv_bounce = make_local(0, 0);
  kv_bounce.k.assign(kv_bounce.k.size(), 0); kv_bounce.v.assign(kv_bounce.v.size(), 0);

  // Run the donor (send) on a thread; the restorer (recv) blocks on this one.
  std::thread donor_t([&] {
    donor_bounce.execute(local.k.data(), local.v.data(), /*remote=*/nullptr, 0,
                         /*stream=*/nullptr, /*graph=*/nullptr, chan_a.get());
  });
  restorer_bounce.execute(kv_bounce.k.data(), kv_bounce.v.data(), 0,
                          /*stream=*/nullptr, /*graph=*/nullptr, chan_b.get());
  donor_t.join();

  // The host-bounce round-trip produces the SAME K/V on B as the direct path.
  ASSERT_TRUE(same_bytes(kv_bounce.k, kv_direct.k));
  ASSERT_TRUE(same_bytes(kv_bounce.v, kv_direct.v));
}

TEST(CrossNodeHostBounce, NoFabricMappedPointerForBounce) {
  // A kHostBounce plan never needs a FabricImport (device_ptr unavailable).
  // Constructing without one and executing over a channel must work.
  auto local = make_local();
  const int* slots = unique_slots();
  CrossNodeKvDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kHostBounce, nullptr);
  EXPECT_FALSE(donor.is_graph_capturable());
  EXPECT_EQ(donor.bounce_bytes(), kLayerBytes);

  auto link = make_byte_link();
  CrossNodeKvRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem, slots,
                                  kPages, kPageSize,
                                  FabricImportTransport::kHostBounce, nullptr);
  EXPECT_FALSE(restorer.is_graph_capturable());

  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  std::thread donor_t([&] {
    donor.execute(local.k.data(), local.v.data(), /*remote=*/nullptr, 0,
                  nullptr, nullptr, link.first.get());
  });
  restorer.execute(out.k.data(), out.v.data(), 0, nullptr, nullptr,
                   link.second.get());
  donor_t.join();
  // The restorer recovered A's local K/V for the donated slots.
  for (std::size_t i = 0; i < kPages * kPageSize; ++i) {
    int slot = slots[i];
    for (std::size_t b = 0; b < kSlotBytes; ++b) {
      EXPECT_EQ(out.k[slot * kSlotBytes + b], local.k[slot * kSlotBytes + b]);
      EXPECT_EQ(out.v[slot * kSlotBytes + b], local.v[slot * kSlotBytes + b]);
    }
  }
}

TEST(CrossNodeHostBounce, RecvRejectsUndersizedChunk) {
  // The restorer recvs exactly one layer; a smaller chunk is a sender bug
  // surfaced as an invalid_argument (not a read past the end).
  const int* slots = unique_slots();
  CrossNodeKvRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem, slots,
                                  kPages, kPageSize,
                                  FabricImportTransport::kHostBounce, nullptr);
  auto link = make_byte_link();
  // Sender ships too few bytes.
  link.second->send(std::vector<std::uint8_t>(4, 0));
  auto out = make_local(0, 0);
  EXPECT_THROW(restorer.execute(out.k.data(), out.v.data(), 0, nullptr, nullptr,
                                link.first.get()), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Graph capture / eager-break (ties into issue #10)
// ---------------------------------------------------------------------------

TEST(CrossNodeGraph, CapturablePathRecordsOneOpNoHostProgress) {
  // kFabricMapped: execute inside a capture records ONE device op; replay
  // runs it with no host I/O (the channel is never touched).
  auto remote = make_remote();
  const int* slots = unique_slots();
  FabricHandle h(1, 1, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kFabricMapped, h);
  CrossNodeKvRestorePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               FabricImportTransport::kFabricMapped, &imp);
  ASSERT_TRUE(xplan.is_graph_capturable());

  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  GraphCapture g;
  g.begin();
  ASSERT_TRUE(g.in_capture());
  xplan.execute(out.k.data(), out.v.data(), kLayerBytes, /*stream=*/nullptr,
                &g, /*channel=*/nullptr);
  EXPECT_EQ(g.num_nodes(), 1u);  // exactly one device op
  g.end();
  EXPECT_EQ(g.num_segments(), 1u);

  // Replay runs the recorded op with no host progress and no channel.
  const std::size_t before = g.replays();
  g.replay();
  EXPECT_EQ(g.replays(), before + 1u);
  // The captured op wrote the layer; compare to a fresh direct execute.
  auto ref = make_local(0, 0); ref.k.assign(ref.k.size(), 0); ref.v.assign(ref.v.size(), 0);
  xplan.execute(ref.k.data(), ref.v.data(), kLayerBytes, nullptr, nullptr, nullptr);
  ASSERT_TRUE(same_bytes(out.k, ref.k));
  ASSERT_TRUE(same_bytes(out.v, ref.v));
}

TEST(CrossNodeGraph, HostBounceEagerBreaks) {
  // kHostBounce: execute inside a capture ENDS the current segment, runs the
  // host send/recv over the channel, then BEGINS the next -- exactly the
  // PipelineBoundaryPlan contract (#10). Replay records ZERO host-bounce ops.
  auto local = make_local();
  const int* slots = unique_slots();
  auto link = make_byte_link();

  CrossNodeKvDonatePlan donor(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kHostBounce, nullptr);
  GraphCapture g;
  g.begin();
  EXPECT_EQ(g.num_nodes(), 0u);
  // Eager-break: one segment ends, the host send runs, a new one begins.
  donor.execute(local.k.data(), local.v.data(), /*remote=*/nullptr, 0,
                /*stream=*/nullptr, &g, link.first.get());
  EXPECT_EQ(g.num_nodes(), 0u);      // nothing recorded (host-staged)
  EXPECT_EQ(g.num_segments(), 1u);   // the segment ended by the break
  EXPECT_TRUE(g.in_capture());       // a new segment was begun
  g.end();
  EXPECT_EQ(g.num_segments(), 2u);

  // Replay runs no host-bounce work (it was excluded from the graph).
  const std::size_t before = g.replays();
  g.replay();
  EXPECT_EQ(g.replays(), before + 1u);
}

TEST(CrossNodeGraph, RestoreHostBounceEagerBreaks) {
  // kHostBounce RESTORE inside a capture ENDS the current segment, runs the
  // host recv+scatter over the channel, then BEGINS the next -- the
  // PipelineBoundaryPlan eager-break contract (#10). Nothing is recorded.
  auto link = make_byte_link();
  const int* slots = unique_slots();
  // Pre-send one layer so the restorer's recv is non-blocking.
  link.first->send(std::vector<std::uint8_t>(kLayerBytes, 0x33));

  CrossNodeKvRestorePlan restorer(kSlots, kHeads, kHeadDim, kElem, slots,
                                  kPages, kPageSize,
                                  FabricImportTransport::kHostBounce, nullptr);
  auto out = make_local(0, 0); out.k.assign(out.k.size(), 0); out.v.assign(out.v.size(), 0);
  GraphCapture g;
  g.begin();
  EXPECT_EQ(g.num_nodes(), 0u);
  restorer.execute(out.k.data(), out.v.data(), 0,
                   /*stream=*/nullptr, &g, link.second.get());
  EXPECT_EQ(g.num_nodes(), 0u);     // nothing recorded (host-staged)
  EXPECT_EQ(g.num_segments(), 1u);  // the segment ended by the break
  EXPECT_TRUE(g.in_capture());      // a new segment was begun
  g.end();
  EXPECT_EQ(g.num_segments(), 2u);
}

TEST(CrossNodeGraph, SameNodePeerIsCapturable) {
  auto remote = make_remote();
  const int* slots = unique_slots();
  FabricHandle h(1, 1, remote.data(), remote.size());
  FabricImport imp(FabricImportTransport::kSameNodePeer, h);
  CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kSameNodePeer, &imp);
  ASSERT_TRUE(xplan.is_graph_capturable());
  auto out = make_remote();
  FabricHandle hout(1, 1, out.data(), out.size());
  auto local = make_local();
  GraphCapture g;
  g.begin();
  xplan.execute(local.k.data(), local.v.data(), &hout, 0,
                nullptr, &g, nullptr);
  EXPECT_EQ(g.num_nodes(), 1u);
  g.end();
}

// ---------------------------------------------------------------------------
// Contract checks (the prepared-plan invariants)
// ---------------------------------------------------------------------------

TEST(CrossNodeRestore, RejectsDuplicateSlot) {
  const int bad[kPages * kPageSize] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_THROW(CrossNodeKvRestorePlan(kSlots, kHeads, kHeadDim, kElem, bad,
                                      kPages, kPageSize,
                                      FabricImportTransport::kHostBounce, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeRestore, RejectsOutOfRangeSlot) {
  const int bad[kPages * kPageSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9999};
  EXPECT_THROW(CrossNodeKvRestorePlan(kSlots, kHeads, kHeadDim, kElem, bad,
                                      kPages, kPageSize,
                                      FabricImportTransport::kHostBounce, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeRestore, RejectsNegativeSlot) {
  const int bad[kPages * kPageSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1};
  EXPECT_THROW(CrossNodeKvRestorePlan(kSlots, kHeads, kHeadDim, kElem, bad,
                                      kPages, kPageSize,
                                      FabricImportTransport::kHostBounce, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeRestore, RejectsNonBF16) {
  const int* slots = unique_slots();
  EXPECT_THROW(CrossNodeKvRestorePlan(kSlots, kHeads, kHeadDim, /*elem=*/4,
                                      slots, kPages, kPageSize,
                                      FabricImportTransport::kHostBounce, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeRestore, RejectsCapturableWithoutImport) {
  const int* slots = unique_slots();
  EXPECT_THROW(CrossNodeKvRestorePlan(kSlots, kHeads, kHeadDim, kElem, slots,
                                      kPages, kPageSize,
                                      FabricImportTransport::kFabricMapped, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeRestore, HostBounceNeedsChannelAtExecute) {
  const int* slots = unique_slots();
  CrossNodeKvRestorePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                               kPages, kPageSize,
                               FabricImportTransport::kHostBounce, nullptr);
  auto out = make_local(0, 0);
  EXPECT_THROW(xplan.execute(out.k.data(), out.v.data(), 0, nullptr, nullptr,
                             /*channel=*/nullptr), std::invalid_argument);
}

TEST(CrossNodeDonate, RejectsOutOfRangeSlot) {
  const int bad[kPages * kPageSize] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9999};
  EXPECT_THROW(CrossNodeKvDonatePlan(kSlots, kHeads, kHeadDim, kElem, bad,
                                     kPages, kPageSize,
                                     FabricImportTransport::kHostBounce, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeDonate, RejectsNonBF16) {
  const int* slots = unique_slots();
  EXPECT_THROW(CrossNodeKvDonatePlan(kSlots, kHeads, kHeadDim, /*elem=*/1,
                                     slots, kPages, kPageSize,
                                     FabricImportTransport::kHostBounce, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeDonate, RejectsCapturableWithoutImport) {
  const int* slots = unique_slots();
  EXPECT_THROW(CrossNodeKvDonatePlan(kSlots, kHeads, kHeadDim, kElem, slots,
                                     kPages, kPageSize,
                                     FabricImportTransport::kSameNodePeer, nullptr),
               std::invalid_argument);
}

TEST(CrossNodeDonate, HostBounceNeedsChannelAtExecute) {
  const int* slots = unique_slots();
  CrossNodeKvDonatePlan xplan(kSlots, kHeads, kHeadDim, kElem, slots,
                              kPages, kPageSize,
                              FabricImportTransport::kHostBounce, nullptr);
  std::uint8_t dummy = 0;
  EXPECT_THROW(xplan.execute(&dummy, &dummy, nullptr, 0, nullptr, nullptr,
                             /*channel=*/nullptr), std::invalid_argument);
}
