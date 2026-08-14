// tests/comm/test_rccl.cpp — host tests for the HIP/RCCL transport (issue #19).
//
// The HIP/RCCL channel and graph-capturable all-reduce are device-only
// (rccl.hip); this file exercises the always-compiled host reference in
// rccl.cpp: the transport-selection cost model, the cross-node ring
// topology, OFI/CXI discovery, and the graph-capturable all-reduce plan
// API surface over the mock channel ring.
#include "minitest.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/rccl.hpp"

using vkernels::comm::build_cross_node_ring;
using vkernels::comm::cross_node_hops;
using vkernels::comm::discover_ofi_cxi;
using vkernels::comm::est_rccl_ofi_us;
using vkernels::comm::est_rccl_socket_us;
using vkernels::comm::inter_node_ring_edges;
using vkernels::comm::is_cuda_built_plugin;
using vkernels::comm::make_ring_channels;
using vkernels::comm::NodeTopology;
using vkernels::comm::OfiCxiInfo;
using vkernels::comm::prefer_slingshot_rccl;
using vkernels::comm::RcclAllreducePlan;
using vkernels::comm::RcclReduceOp;
using vkernels::comm::RcclTransport;
using vkernels::comm::RcclTransportConfig;
using vkernels::comm::RcclTransportMode;
using vkernels::comm::resolve_rccl_transport;
using vkernels::comm::resolve_transport;
using vkernels::comm::transport_name;
using vkernels::comm::RcclTransport;
using vkernels::comm::RcclTransportConfig;
using vkernels::comm::RcclTransportMode;
using vkernels::Stream;

static const std::size_t kMiB = 1024u * 1024u;

static bool vec_eq(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (std::abs(a[i] - b[i]) > 1e-4f) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Stream operators (logging / minitest failure path)
// ---------------------------------------------------------------------------

TEST(RcclStreamOps, TransportName) {
  EXPECT_EQ(std::string(transport_name(RcclTransport::kSocket)), "socket");
  EXPECT_EQ(std::string(transport_name(RcclTransport::kSlingshotOfi)),
            "slingshot-ofi");
}

TEST(RcclStreamOps, StreamsAllEnums) {
  std::ostringstream os;
  os << RcclTransport::kSlingshotOfi;
  os << RcclTransportMode::kAdaptive << RcclTransportMode::kForceSlingshot
     << RcclTransportMode::kForceSocket;
  os << RcclReduceOp::kSum << RcclReduceOp::kMax << RcclReduceOp::kMin;
  EXPECT_EQ(os.str(),
            "slingshot-ofiadaptiveforce-slingshotforce-socketsummaxmin");
}

// ---------------------------------------------------------------------------
// is_cuda_built_plugin
// ---------------------------------------------------------------------------

TEST(IsCudaBuiltPlugin, DetectsCudaBuilt) {
  EXPECT_TRUE(is_cuda_built_plugin("aws_ofi_nccl"));
  EXPECT_TRUE(is_cuda_built_plugin("libnccl-net-ofi"));
}

TEST(IsCudaBuiltPlugin, AcceptsHipAware) {
  EXPECT_FALSE(is_cuda_built_plugin("aws_ofi_rccl"));
  EXPECT_FALSE(is_cuda_built_plugin("librccl-net-ofi"));
  EXPECT_FALSE(is_cuda_built_plugin(""));
}

// ---------------------------------------------------------------------------
// resolve_rccl_transport
// ---------------------------------------------------------------------------

TEST(ResolveRcclTransport, EmptyEnvKeepsDefaults) {
  auto cfg = resolve_rccl_transport({});
  EXPECT_EQ(cfg.mode, RcclTransportMode::kAdaptive);
  EXPECT_EQ(cfg.socket_ifname, "hsn0");
  EXPECT_TRUE(cfg.ib_disabled);
  EXPECT_EQ(cfg.ofi_provider, "cxi");
  EXPECT_FALSE(cfg.plugin_is_cuda_built);
}

TEST(ResolveRcclTransport, ParsesBeverinEnv) {
  std::vector<std::pair<std::string, std::string>> env = {
      {"NCCL_NET", "aws_ofi_rccl"},
      {"NCCL_SOCKET_IFNAME", "hsn0"},
      {"NCCL_IB_DISABLE", "1"},
      {"FI_PROVIDER", "cxi"},
  };
  auto cfg = resolve_rccl_transport(env);
  EXPECT_EQ(cfg.net_plugin, "aws_ofi_rccl");
  EXPECT_EQ(cfg.socket_ifname, "hsn0");
  EXPECT_TRUE(cfg.ib_disabled);
  EXPECT_EQ(cfg.ofi_provider, "cxi");
  EXPECT_FALSE(cfg.plugin_is_cuda_built);
}

TEST(ResolveRcclTransport, RcclNetIsCaseInsensitiveAlias) {
  auto cfg = resolve_rccl_transport({{"rccl_net", "aws_ofi_nccl"}});
  EXPECT_EQ(cfg.net_plugin, "aws_ofi_nccl");
  EXPECT_TRUE(cfg.plugin_is_cuda_built);
}

TEST(ResolveRcclTransport, IbDisableVariants) {
  // Empty -> beverin default (disabled).
  EXPECT_TRUE(resolve_rccl_transport({{"NCCL_IB_DISABLE", ""}}).ib_disabled);
  // "1"/"true"/"yes"/"on" -> disabled.
  EXPECT_TRUE(resolve_rccl_transport({{"NCCL_IB_DISABLE", "1"}}).ib_disabled);
  EXPECT_TRUE(resolve_rccl_transport({{"NCCL_IB_DISABLE", "YES"}}).ib_disabled);
  EXPECT_TRUE(resolve_rccl_transport({{"NCCL_IB_DISABLE", "on"}}).ib_disabled);
  // "0"/"false"/garbage -> enabled.
  EXPECT_FALSE(resolve_rccl_transport({{"NCCL_IB_DISABLE", "0"}}).ib_disabled);
  EXPECT_FALSE(resolve_rccl_transport({{"NCCL_IB_DISABLE", "garbage"}}).ib_disabled);
}

// ---------------------------------------------------------------------------
// Cost model: est_rccl_socket_us / est_rccl_ofi_us / prefer_slingshot_rccl
// ---------------------------------------------------------------------------

TEST(EstRcclCost, ZeroBytesIsZero) {
  EXPECT_EQ(est_rccl_socket_us(0, 2), 0.0);
  EXPECT_EQ(est_rccl_ofi_us(0, 2), 0.0);
}

TEST(EstRcclCost, SocketScalesWithEdges) {
  const double no_edge = est_rccl_socket_us(kMiB, 0);
  const double two_edges = est_rccl_socket_us(kMiB, 2);
  EXPECT_GT(two_edges, no_edge + 1.0);  // +2*25us TCP penalty
}

TEST(EstRcclCost, OfiIgnoresEdges) {
  // RDMA: the inter-node edge count does not change OFI latency.
  EXPECT_NEAR(est_rccl_ofi_us(kMiB, 0), est_rccl_ofi_us(kMiB, 4), 0.0);
}

TEST(EstRcclCost, OfiFasterThanSocketAtLargeCrossNode) {
  // 48 MiB over a 2-edge cross-node ring: OFI must beat Socket (acceptance).
  EXPECT_LT(est_rccl_ofi_us(48 * kMiB, 2), est_rccl_socket_us(48 * kMiB, 2));
}

TEST(PreferSlingshot, ZeroBytesOrEdgesNeverTakesOfi) {
  RcclTransportConfig cfg;
  EXPECT_FALSE(prefer_slingshot_rccl(0, 2, cfg));
  EXPECT_FALSE(prefer_slingshot_rccl(kMiB, 0, cfg));
  EXPECT_FALSE(prefer_slingshot_rccl(0, 0, cfg));
}

TEST(PreferSlingshot, ForceModes) {
  RcclTransportConfig cfg;
  cfg.mode = RcclTransportMode::kForceSlingshot;
  EXPECT_TRUE(prefer_slingshot_rccl(512, 2, cfg));  // below floor, still forced
  cfg.mode = RcclTransportMode::kForceSocket;
  EXPECT_FALSE(prefer_slingshot_rccl(48 * kMiB, 2, cfg));  // large, still socket
}

TEST(PreferSlingshot, AdaptiveBelowFloorKeepsSocket) {
  RcclTransportConfig cfg;  // min_msg_for_ofi = 1 MiB
  EXPECT_FALSE(prefer_slingshot_rccl(512, 2, cfg));  // below floor
}

TEST(PreferSlingshot, AdaptiveAboveFloorTakesOfi) {
  RcclTransportConfig cfg;  // min_msg_for_ofi = 1 MiB
  EXPECT_TRUE(prefer_slingshot_rccl(48 * kMiB, 2, cfg));
  // Zero floor: the model decides from any cross-node payload.
  cfg.min_msg_for_ofi = 0;
  EXPECT_TRUE(prefer_slingshot_rccl(512, 2, cfg));
}

TEST(PreferSlingshot, AdaptiveVerySmallPayloadKeepsSocket) {
  // Below the 1 MiB floor the OFI bandwidth advantage does not yet cover its
  // higher launch floor, so Socket stays (model picks the cheaper one).
  RcclTransportConfig cfg;
  const std::size_t tiny = 4096;
  EXPECT_FALSE(prefer_slingshot_rccl(tiny, 2, cfg));
}

// ---------------------------------------------------------------------------
// resolve_transport
// ---------------------------------------------------------------------------

TEST(ResolveTransport, NoCrossNodeForcesSocket) {
  RcclTransportConfig cfg;
  cfg.mode = RcclTransportMode::kForceSlingshot;  // even forced
  EXPECT_EQ(resolve_transport(48 * kMiB, 0, cfg), RcclTransport::kSocket);
}

TEST(ResolveTransport, CudaBuiltPluginForcesSocket) {
  RcclTransportConfig cfg;
  cfg.net_plugin = "aws_ofi_nccl";
  cfg.plugin_is_cuda_built = true;
  EXPECT_EQ(resolve_transport(48 * kMiB, 2, cfg), RcclTransport::kSocket);
}

TEST(ResolveTransport, HipAwarePluginTakesOfi) {
  RcclTransportConfig cfg;
  cfg.net_plugin = "librccl-net-ofi";
  EXPECT_EQ(resolve_transport(48 * kMiB, 2, cfg), RcclTransport::kSlingshotOfi);
}

TEST(ResolveTransport, SmallPayloadStaysSocket) {
  RcclTransportConfig cfg;
  EXPECT_EQ(resolve_transport(4096, 2, cfg), RcclTransport::kSocket);
}

// ---------------------------------------------------------------------------
// Cross-node ring topology
// ---------------------------------------------------------------------------

TEST(CrossNodeRing, SingleNodeHasNoRemoteEdges) {
  auto topo = build_cross_node_ring({0, 0, 0, 0}, 1);
  EXPECT_EQ(topo.size(), 4u);
  EXPECT_EQ(inter_node_ring_edges(topo), 0);
  for (const auto& t : topo) EXPECT_EQ(cross_node_hops(t), 0);
}

TEST(CrossNodeRing, NodeMajorMinimisesEdges) {
  // 4 ranks on 2 nodes, node-major: ranks 0,1 on node 0; 2,3 on node 1.
  auto topo = build_cross_node_ring({0, 0, 1, 1}, 2);
  EXPECT_EQ(inter_node_ring_edges(topo), 2);  // one boundary + one wrap
  EXPECT_EQ(topo[0].rank, 0);
  EXPECT_EQ(topo[0].node, 0);
  EXPECT_EQ(topo[0].local_rank, 0);
  EXPECT_EQ(topo[0].local_size, 2);
  EXPECT_EQ(topo[0].next, 1);
  EXPECT_FALSE(topo[0].next_is_remote);
  EXPECT_EQ(topo[0].prev, 3);  // wrap: last rank of node 1
  EXPECT_TRUE(topo[0].prev_is_remote);
  // Rank 1 (last of node 0) -> rank 2 (first of node 1): remote.
  EXPECT_EQ(topo[1].next, 2);
  EXPECT_TRUE(topo[1].next_is_remote);
  EXPECT_EQ(cross_node_hops(topo[1]), 1);
  // Rank 2 (first of node 1) -> stays; prev is rank 1 (node 0): remote.
  EXPECT_EQ(topo[2].prev, 1);
  EXPECT_TRUE(topo[2].prev_is_remote);
  EXPECT_EQ(topo[2].node, 1);
  EXPECT_EQ(topo[2].local_rank, 0);
}

TEST(CrossNodeRing, InterleavedInputStillNodeMajor) {
  // Even an interleaved {0,1,0,1} layout re-sorts to node-major (ring order
  // [0,2,1,3]): same-node ranks become adjacent, so only the node boundaries
  // are inter-node hops (== number of nodes), regardless of input ordering.
  auto topo = build_cross_node_ring({0, 1, 0, 1}, 2);
  EXPECT_EQ(inter_node_ring_edges(topo), 2);
  // Ring order is node-major: rank 0 -> rank 2 (same node) -> rank 1 ->
  // rank 3 -> rank 0.
  EXPECT_EQ(topo[0].next, 2);
  EXPECT_FALSE(topo[0].next_is_remote);  // node 0 -> node 0
  EXPECT_EQ(topo[0].prev, 3);
  EXPECT_TRUE(topo[0].prev_is_remote);  // wrap: node 1 -> node 0
  EXPECT_EQ(topo[2].next, 1);
  EXPECT_TRUE(topo[2].next_is_remote);  // node 0 -> node 1
  EXPECT_EQ(topo[1].node, 1);
  EXPECT_EQ(topo[1].local_rank, 0);
  EXPECT_EQ(topo[3].node, 1);
  EXPECT_EQ(topo[3].local_rank, 1);
  for (const auto& t : topo) EXPECT_EQ(cross_node_hops(t), 1);
}

TEST(CrossNodeRing, EmptyNodeOfThrows) {
  EXPECT_THROW(build_cross_node_ring({}, 1), std::invalid_argument);
}

TEST(CrossNodeRing, NonPositiveNodesThrows) {
  EXPECT_THROW(build_cross_node_ring({0}, 0), std::invalid_argument);
  EXPECT_THROW(build_cross_node_ring({0}, -1), std::invalid_argument);
}

TEST(CrossNodeRing, NodeIdOutOfRangeThrows) {
  EXPECT_THROW(build_cross_node_ring({0, 2}, 2), std::invalid_argument);  // node 2 >= nodes
  EXPECT_THROW(build_cross_node_ring({-1, 0}, 2), std::invalid_argument);  // node -1
}

TEST(CrossNodeRing, HoleInNodesThrows) {
  // Node 1 has no rank (0 and 2 are present): a degenerate layout.
  EXPECT_THROW(build_cross_node_ring({0, 0, 2}, 3), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// discover_ofi_cxi
// ---------------------------------------------------------------------------

TEST(DiscoverOfiCxi, NoLibfabric) {
  RcclTransportConfig cfg;
  cfg.net_plugin = "librccl-net-ofi";
  auto info = discover_ofi_cxi(cfg, /*libfabric_present=*/false);
  EXPECT_FALSE(info.available);
  EXPECT_FALSE(info.reason.empty());
}

TEST(DiscoverOfiCxi, CudaBuiltPlugin) {
  RcclTransportConfig cfg;
  cfg.net_plugin = "aws_ofi_nccl";
  cfg.plugin_is_cuda_built = true;
  auto info = discover_ofi_cxi(cfg, /*libfabric_present=*/true);
  EXPECT_FALSE(info.available);
  EXPECT_TRUE(info.plugin_is_cuda_built);
  EXPECT_FALSE(info.reason.empty());
}

TEST(DiscoverOfiCxi, NoPluginConfigured) {
  RcclTransportConfig cfg;  // net_plugin empty
  auto info = discover_ofi_cxi(cfg, /*libfabric_present=*/true);
  EXPECT_FALSE(info.available);
  EXPECT_FALSE(info.reason.empty());
}

TEST(DiscoverOfiCxi, Available) {
  RcclTransportConfig cfg;
  cfg.net_plugin = "librccl-net-ofi";
  cfg.ofi_provider = "cxi";
  auto info = discover_ofi_cxi(cfg, /*libfabric_present=*/true);
  EXPECT_TRUE(info.available);
  EXPECT_EQ(info.num_devices, 1);
  EXPECT_EQ(info.provider, "cxi");
  EXPECT_TRUE(info.reason.empty());
}

// ---------------------------------------------------------------------------
// RcclAllreducePlan (host reference, over the mock channel ring)
// ---------------------------------------------------------------------------

namespace {

std::vector<float> elementwise_reduce(const std::vector<std::vector<float>>& xs,
                                      RcclReduceOp op) {
  std::vector<float> out = xs[0];
  for (std::size_t r = 1; r < xs.size(); ++r) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      if (op == RcclReduceOp::kSum) out[i] += xs[r][i];
      else if (op == RcclReduceOp::kMax) out[i] = std::max(out[i], xs[r][i]);
      else out[i] = std::min(out[i], xs[r][i]);
    }
  }
  return out;
}

// Run `world` prepared plans concurrently over a mock ring and return each
// rank's reduced buffer.
std::vector<std::vector<float>> run_plans(int world, RcclReduceOp op) {
  const std::size_t n = static_cast<std::size_t>(world) * 2;
  std::vector<std::vector<float>> bufs(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r)
    for (std::size_t i = 0; i < n; ++i)
      bufs[static_cast<std::size_t>(r)].push_back(
          static_cast<float>(r * 10 + static_cast<int>(i) + 1));
  auto channels = make_ring_channels(world);
  std::vector<std::thread> ts;
  for (int r = 0; r < world; ++r) {
    ts.emplace_back([&, r] {
      RcclAllreducePlan plan(world, r, op, n);
      plan.execute(bufs[static_cast<std::size_t>(r)].data(), n,
                   *channels[static_cast<std::size_t>(r)],
                   *channels[static_cast<std::size_t>(r)], /*stream=*/nullptr);
    });
  }
  for (auto& t : ts) t.join();
  return bufs;
}

}  // namespace

TEST(RcclAllreducePlan, ConstructorValidates) {
  EXPECT_THROW(RcclAllreducePlan(0, 0, RcclReduceOp::kSum, 4),
               std::invalid_argument);
  EXPECT_THROW(RcclAllreducePlan(2, -1, RcclReduceOp::kSum, 4),
               std::invalid_argument);
  EXPECT_THROW(RcclAllreducePlan(2, 2, RcclReduceOp::kSum, 4),
               std::invalid_argument);
  EXPECT_THROW(RcclAllreducePlan(2, 0, static_cast<RcclReduceOp>(99), 4),
               std::invalid_argument);
  EXPECT_THROW(RcclAllreducePlan(2, 0, RcclReduceOp::kSum, 0),
               std::invalid_argument);
  EXPECT_THROW(RcclAllreducePlan(2, 0, RcclReduceOp::kSum, 3),  // 3 % 2 != 0
               std::invalid_argument);
}

TEST(RcclAllreducePlan, SumMatchesElementwise) {
  auto got = run_plans(4, RcclReduceOp::kSum);
  std::vector<std::vector<float>> xs(4);
  for (int r = 0; r < 4; ++r)
    for (std::size_t i = 0; i < 8; ++i)
      xs[static_cast<std::size_t>(r)].push_back(
          static_cast<float>(r * 10 + static_cast<int>(i) + 1));
  auto ref = elementwise_reduce(xs, RcclReduceOp::kSum);
  for (const auto& g : got) EXPECT_TRUE(vec_eq(g, ref));
}

TEST(RcclAllreducePlan, MaxMatchesElementwise) {
  auto got = run_plans(3, RcclReduceOp::kMax);
  std::vector<std::vector<float>> xs(3);
  for (int r = 0; r < 3; ++r)
    for (std::size_t i = 0; i < 6; ++i)
      xs[static_cast<std::size_t>(r)].push_back(
          static_cast<float>(r * 10 + static_cast<int>(i) + 1));
  auto ref = elementwise_reduce(xs, RcclReduceOp::kMax);
  for (const auto& g : got) EXPECT_TRUE(vec_eq(g, ref));
}

TEST(RcclAllreducePlan, MinMatchesElementwise) {
  auto got = run_plans(3, RcclReduceOp::kMin);
  std::vector<std::vector<float>> xs(3);
  for (int r = 0; r < 3; ++r)
    for (std::size_t i = 0; i < 6; ++i)
      xs[static_cast<std::size_t>(r)].push_back(
          static_cast<float>(r * 10 + static_cast<int>(i) + 1));
  auto ref = elementwise_reduce(xs, RcclReduceOp::kMin);
  for (const auto& g : got) EXPECT_TRUE(vec_eq(g, ref));
}

TEST(RcclAllreducePlan, SingleRankIsNoOp) {
  auto channels = make_ring_channels(1);
  std::vector<float> buf = {1, 2, 3, 4};
  RcclAllreducePlan plan(1, 0, RcclReduceOp::kSum, 4);
  plan.execute(buf.data(), 4, *channels[0], *channels[0], /*stream=*/nullptr);
  EXPECT_TRUE(vec_eq(buf, {1, 2, 3, 4}));
}

TEST(RcclAllreducePlan, SubmitsToStream) {
  const int world = 2;
  const std::size_t n = 4;
  std::vector<std::vector<float>> bufs = {{1, 2, 3, 4}, {10, 20, 30, 40}};
  auto channels = make_ring_channels(world);
  std::vector<std::thread> ts;
  for (int r = 0; r < world; ++r) {
    ts.emplace_back([&, r] {
      Stream s;
      RcclAllreducePlan plan(world, r, RcclReduceOp::kSum, n);
      plan.execute(bufs[static_cast<std::size_t>(r)].data(), n,
                   *channels[static_cast<std::size_t>(r)],
                   *channels[static_cast<std::size_t>(r)], &s);
      s.wait();
    });
  }
  for (auto& t : ts) t.join();
  EXPECT_TRUE(vec_eq(bufs[0], {11, 22, 33, 44}));
  EXPECT_TRUE(vec_eq(bufs[1], {11, 22, 33, 44}));
}

TEST(RcclAllreducePlan, ExecuteValidates) {
  auto channels = make_ring_channels(2);
  RcclAllreducePlan plan(2, 0, RcclReduceOp::kSum, 4);
  std::vector<float> buf = {1, 2, 3, 4};
  EXPECT_THROW(plan.execute(nullptr, 4, *channels[0], *channels[0], nullptr),
               std::invalid_argument);
  EXPECT_THROW(plan.execute(buf.data(), 0, *channels[0], *channels[0], nullptr),
               std::invalid_argument);
  EXPECT_THROW(plan.execute(buf.data(), 5, *channels[0], *channels[0], nullptr),
               std::invalid_argument);  // n > capacity
  EXPECT_THROW(plan.execute(buf.data(), 3, *channels[0], *channels[0], nullptr),
               std::invalid_argument);  // 3 % 2 != 0
}
