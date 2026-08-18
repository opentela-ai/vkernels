// tests/comm/test_rccl_c.cpp
//
// Runtime tests for the `extern "C"` HIP/RCCL transport ABI (issue #19).
// The ABI (rccl_c.h / rccl_c.cpp) wraps the always-compiled host reference,
// so this is a plain C++ test (no GPU) — every entry point is exercised on
// its happy path, and the validators that throw std::invalid_argument are
// exercised on their error path so the catch-and-translate machinery
// (status code) is covered end to end.
#include "vkernels/comm/rccl_c.h"

#include "minitest.hpp"

#include <cmath>
#include <thread>
#include <vector>

#include "vkernels/comm/channel.hpp"

using vkernels::comm::Channel;
using vkernels::comm::make_ring_channels;

namespace {

// Run `world` C-ABI plans concurrently over a mock ring and return each
// rank's reduced buffer (mirrors test_rccl.cpp::run_plans over the ABI).
std::vector<std::vector<float>> run_plans_c(vkernels_rccl_reduce_op_t op,
                                            int world = 4, int n = 8) {
  std::vector<std::vector<float>> bufs(static_cast<std::size_t>(world));
  for (int r = 0; r < world; ++r)
    for (int i = 0; i < n; ++i)
      bufs[static_cast<std::size_t>(r)].push_back(
          static_cast<float>(r * 10 + i + 1));
  auto channels = make_ring_channels(world);
  std::vector<vkernels_rccl_allreduce_plan_t*> plans(
      static_cast<std::size_t>(world), nullptr);
  for (int r = 0; r < world; ++r)
    ASSERT_EQ(vkernels_rccl_allreduce_plan_create(world, r, op,
                  static_cast<uint64_t>(n), &plans[static_cast<std::size_t>(r)]),
              VKERNELS_RCCL_OK);
  std::vector<std::thread> ts;
  for (int r = 0; r < world; ++r) {
    ts.emplace_back([&, r] {
      Channel* ch = channels[static_cast<std::size_t>(r)].get();
      ASSERT_EQ(vkernels_rccl_allreduce_plan_execute(
                    plans[static_cast<std::size_t>(r)],
                    bufs[static_cast<std::size_t>(r)].data(),
                    static_cast<uint64_t>(n), ch, ch),
                VKERNELS_RCCL_OK);
    });
  }
  for (auto& t : ts) t.join();
  for (auto* p : plans) vkernels_rccl_allreduce_plan_destroy(p);
  return bufs;
}

}  // namespace

TEST(RcclCABI, ResolveTransport) {
  vkernels_rccl_env_kv_t env[] = {
      {"NCCL_NET", "aws_ofi_rccl"},
      {"NCCL_IB_DISABLE", "1"},
  };
  vkernels_rccl_config_t cfg{};
  ASSERT_EQ(vkernels_rccl_resolve_transport(env, 2, &cfg), VKERNELS_RCCL_OK);
  EXPECT_EQ(cfg.mode, VKERNELS_RCCL_MODE_ADAPTIVE);
  EXPECT_EQ(std::string(cfg.net_plugin), "aws_ofi_rccl");
  EXPECT_EQ(cfg.ib_disabled, 1);
  EXPECT_EQ(cfg.plugin_is_cuda_built, 0);

  EXPECT_EQ(vkernels_rccl_resolve_transport(nullptr, 0, &cfg), VKERNELS_RCCL_OK);
  EXPECT_EQ(vkernels_rccl_resolve_transport(env, 2, nullptr),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
}

TEST(RcclCABI, ResolveTransportFor) {
  vkernels_rccl_config_t cfg{};
  cfg.mode = VKERNELS_RCCL_MODE_ADAPTIVE;
  std::snprintf(cfg._net_plugin, sizeof(cfg._net_plugin), "librccl-net-ofi");
  cfg.net_plugin = cfg._net_plugin;
  cfg.plugin_is_cuda_built = 0;
  vkernels_rccl_transport_t t = VKERNELS_RCCL_SOCKET;
  // Large cross-node payload, HIP-aware plugin -> OFI.
  ASSERT_EQ(vkernels_rccl_resolve_transport_for(48u * 1024u * 1024u, 2, &cfg, &t),
            VKERNELS_RCCL_OK);
  EXPECT_EQ(t, VKERNELS_RCCL_SLINGSHOT_OFI);
  // No cross-node traffic -> Socket.
  ASSERT_EQ(vkernels_rccl_resolve_transport_for(48u * 1024u * 1024u, 0, &cfg, &t),
            VKERNELS_RCCL_OK);
  EXPECT_EQ(t, VKERNELS_RCCL_SOCKET);
  EXPECT_EQ(vkernels_rccl_resolve_transport_for(0, 2, nullptr, &t),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
}

TEST(RcclCABI, ResolveTransportForForce) {
  vkernels_rccl_config_t cfg{};
  std::snprintf(cfg._net_plugin, sizeof(cfg._net_plugin), "librccl-net-ofi");
  cfg.net_plugin = cfg._net_plugin;
  cfg.plugin_is_cuda_built = 0;
  vkernels_rccl_transport_t t = VKERNELS_RCCL_SOCKET;
  // Force Slingshot: OFI even below the Socket launch floor.
  cfg.mode = VKERNELS_RCCL_MODE_FORCE_SLINGSHOT;
  ASSERT_EQ(vkernels_rccl_resolve_transport_for(1u, 1, &cfg, &t),
            VKERNELS_RCCL_OK);
  EXPECT_EQ(t, VKERNELS_RCCL_SLINGSHOT_OFI);
  // Force Socket: Socket even for a huge cross-node payload.
  cfg.mode = VKERNELS_RCCL_MODE_FORCE_SOCKET;
  ASSERT_EQ(vkernels_rccl_resolve_transport_for(48u * 1024u * 1024u, 4, &cfg, &t),
            VKERNELS_RCCL_OK);
  EXPECT_EQ(t, VKERNELS_RCCL_SOCKET);
}


TEST(RcclCABI, CostModel) {
  EXPECT_EQ(vkernels_rccl_est_socket_us(0, 2), 0.0);
  EXPECT_EQ(vkernels_rccl_est_ofi_us(0, 2), 0.0);
  EXPECT_LT(vkernels_rccl_est_ofi_us(48u * 1024u * 1024u, 2),
            vkernels_rccl_est_socket_us(48u * 1024u * 1024u, 2));
}

TEST(RcclCABI, BuildCrossNodeRing) {
  int node_of[] = {0, 0, 1, 1};
  int n = 0;
  // Count query first.
  ASSERT_EQ(vkernels_rccl_build_cross_node_ring(node_of, 4, 2, nullptr, &n),
            VKERNELS_RCCL_OK);
  ASSERT_EQ(n, 4);
  std::vector<vkernels_rccl_node_topology_t> topo(static_cast<std::size_t>(n));
  ASSERT_EQ(vkernels_rccl_build_cross_node_ring(node_of, 4, 2, topo.data(), &n),
            VKERNELS_RCCL_OK);
  EXPECT_EQ(topo[0].node, 0);
  EXPECT_EQ(topo[0].local_size, 2);
  EXPECT_EQ(topo[1].next, 2);
  EXPECT_EQ(topo[1].next_is_remote, 1);
  // Too-small buffer.
  int small = 2;
  EXPECT_EQ(vkernels_rccl_build_cross_node_ring(node_of, 4, 2, topo.data(), &small),
            VKERNELS_RCCL_ERR_OUT_OF_RANGE);
  EXPECT_EQ(small, 4);
  // Null out / null node_of: argument guards.
  EXPECT_EQ(vkernels_rccl_build_cross_node_ring(node_of, 4, 2, topo.data(), nullptr),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_rccl_build_cross_node_ring(nullptr, 4, 2, topo.data(), &n),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);

  // Invalid: node id out of range.
  int bad[] = {0, 2};
  EXPECT_EQ(vkernels_rccl_build_cross_node_ring(bad, 2, 2, topo.data(), &n),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
}

TEST(RcclCABI, DiscoverOfiCxi) {
  vkernels_rccl_config_t cfg{};
  std::snprintf(cfg._net_plugin, sizeof(cfg._net_plugin), "librccl-net-ofi");
  cfg.net_plugin = cfg._net_plugin;
  vkernels_rccl_ofi_info_t info{};
  ASSERT_EQ(vkernels_rccl_discover_ofi_cxi(&cfg, /*libfabric_present=*/1, &info),
            VKERNELS_RCCL_OK);
  EXPECT_EQ(info.available, 1);
  EXPECT_EQ(info.num_devices, 1);
  EXPECT_EQ(std::string(info.provider), "cxi");

  // No libfabric -> unavailable with a reason.
  ASSERT_EQ(vkernels_rccl_discover_ofi_cxi(&cfg, 0, &info), VKERNELS_RCCL_OK);
  EXPECT_EQ(info.available, 0);
  EXPECT_NE(std::string(info.reason).size(), 0u);

  EXPECT_EQ(vkernels_rccl_discover_ofi_cxi(nullptr, 1, &info),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
}

TEST(RcclCABI, AllreducePlanSum) {
  auto bufs = run_plans_c(VKERNELS_RCCL_REDUCE_SUM);
  // Reference: element-wise sum across 4 ranks (n=8).
  std::vector<float> ref(8, 0.0f);
  for (int r = 0; r < 4; ++r)
    for (int i = 0; i < 8; ++i)
      ref[static_cast<std::size_t>(i)] += static_cast<float>(r * 10 + i + 1);
  for (const auto& g : bufs)
    for (std::size_t i = 0; i < g.size(); ++i)
      EXPECT_NEAR(g[i], ref[i], 1e-4f);
}

TEST(RcclCABI, AllreducePlanMax) {
  auto bufs = run_plans_c(VKERNELS_RCCL_REDUCE_MAX);
  // Reference: element-wise max across 4 ranks (n=8) -> rank 3's values.
  for (const auto& g : bufs)
    for (std::size_t i = 0; i < g.size(); ++i)
      EXPECT_NEAR(g[i], static_cast<float>(30 + static_cast<int>(i) + 1), 1e-4f);
}

TEST(RcclCABI, AllreducePlanMin) {
  auto bufs = run_plans_c(VKERNELS_RCCL_REDUCE_MIN);
  // Reference: element-wise min across 4 ranks (n=8) -> rank 0's values.
  for (const auto& g : bufs)
    for (std::size_t i = 0; i < g.size(); ++i)
      EXPECT_NEAR(g[i], static_cast<float>(static_cast<int>(i) + 1), 1e-4f);
}


TEST(RcclCABI, AllreducePlanCreateRejectsInvalid) {
  vkernels_rccl_allreduce_plan_t* p = nullptr;
  EXPECT_EQ(vkernels_rccl_allreduce_plan_create(0, 0, VKERNELS_RCCL_REDUCE_SUM, 4, &p),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_rccl_allreduce_plan_create(2, 0, VKERNELS_RCCL_REDUCE_SUM, 3, &p),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);  // 3 % 2 != 0
  EXPECT_EQ(vkernels_rccl_allreduce_plan_create(2, 0, VKERNELS_RCCL_REDUCE_SUM, 4, nullptr),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
}

TEST(RcclCABI, AllreducePlanExecuteRejectsInvalid) {
  vkernels_rccl_allreduce_plan_t* p = nullptr;
  ASSERT_EQ(vkernels_rccl_allreduce_plan_create(2, 0, VKERNELS_RCCL_REDUCE_SUM, 4, &p),
            VKERNELS_RCCL_OK);
  auto channels = make_ring_channels(2);
  Channel* ch = channels[0].get();
  std::vector<float> buf = {1, 2, 3, 4};
  EXPECT_EQ(vkernels_rccl_allreduce_plan_execute(nullptr, buf.data(), 4, ch, ch),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_rccl_allreduce_plan_execute(p, nullptr, 4, ch, ch),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(vkernels_rccl_allreduce_plan_execute(p, buf.data(), 0, ch, ch),
            VKERNELS_RCCL_ERR_INVALID_ARGUMENT);
  vkernels_rccl_allreduce_plan_destroy(p);
}
