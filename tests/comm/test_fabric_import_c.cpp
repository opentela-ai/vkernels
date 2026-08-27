// tests/comm/test_fabric_import_c.cpp
//
// Host tests for the always-compiled `extern "C"` fabric-import planning
// surface (issue #49): transport classification, the graph-capturable /
// eager-break decision, the transport name, the same-node roofline, and the
// per-hop cost model. The ABI (fabric_import_c.h / fabric_import_c.cpp)
// wraps the host reference with no GPU dependency, so every entry point is
// exercised on its happy path and the null-config / null-out contract on
// its error path -- the same 100% line-coverage gate the rest of the host
// CI job runs.
#include "vkernels/comm/fabric_import_c.h"

#include "minitest.hpp"

#include <string>

namespace {

// vkernels_fi_config_t is an aggregate; brace-init keeps the address stable
// for the duration of the call.
#define FI_CFG(same, gpudirect, dram_only) \
  vkernels_fi_config_t { same, gpudirect, dram_only }

}  // namespace

// ---------------------------------------------------------------------------
// classify -- precedence same_node > dram_only_libfabric > gpudirect > bounce
// ---------------------------------------------------------------------------

TEST(FabricImportCABI, ClassifySameNodePeer) {
  vkernels_fi_config_t c = FI_CFG(1, 0, 0);
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_classify(&c, &status),
            VKERNELS_FI_TRANSPORT_SAME_NODE_PEER);
  EXPECT_EQ(status, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, ClassifySameNodeWinsOverDramOnly) {
  vkernels_fi_config_t c = FI_CFG(1, 0, 1);  // same_node + dram_only
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_classify(&c, &status),
            VKERNELS_FI_TRANSPORT_SAME_NODE_PEER);
  EXPECT_EQ(status, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, ClassifyDramOnlyForcesHostBounce) {
  vkernels_fi_config_t c = FI_CFG(0, 0, 1);  // the GH200 constraint
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_classify(&c, &status),
            VKERNELS_FI_TRANSPORT_HOST_BOUNCE);
  EXPECT_EQ(status, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, ClassifyDramOnlyWinsOverGpudirect) {
  vkernels_fi_config_t c = FI_CFG(0, 1, 1);  // gpudirect + dram_only
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_classify(&c, &status),
            VKERNELS_FI_TRANSPORT_HOST_BOUNCE);
  EXPECT_EQ(status, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, ClassifyFabricMapped) {
  vkernels_fi_config_t c = FI_CFG(0, 1, 0);  // GPUDirect-RDMA
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_classify(&c, &status),
            VKERNELS_FI_TRANSPORT_FABRIC_MAPPED);
  EXPECT_EQ(status, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, ClassifyHostBounceByDefault) {
  vkernels_fi_config_t c = FI_CFG(0, 0, 0);  // no fabric path
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_classify(&c, &status),
            VKERNELS_FI_TRANSPORT_HOST_BOUNCE);
  EXPECT_EQ(status, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, ClassifyNullConfig) {
  vkernels_fi_status_t status = VKERNELS_FI_OK;
  EXPECT_EQ(vkernels_fabric_import_classify(nullptr, &status),
            VKERNELS_FI_TRANSPORT_HOST_BOUNCE);
  EXPECT_EQ(status, VKERNELS_FI_ERR_INVALID_ARGUMENT);
}

TEST(FabricImportCABI, ClassifyNullStatusOut) {
  vkernels_fi_config_t c = FI_CFG(1, 0, 0);
  EXPECT_EQ(vkernels_fabric_import_classify(&c, nullptr),
            VKERNELS_FI_TRANSPORT_SAME_NODE_PEER);
}

// ---------------------------------------------------------------------------
// eager_break -- 1 only for kHostBounce (dram_only or no-fabric)
// ---------------------------------------------------------------------------

TEST(FabricImportCABI, EagerBreakFalseForCapturable) {
  vkernels_fi_config_t a = FI_CFG(1, 0, 0);  // same-node
  vkernels_fi_status_t sa = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_eager_break(&a, &sa), 0);
  EXPECT_EQ(sa, VKERNELS_FI_OK);
  vkernels_fi_config_t b = FI_CFG(0, 1, 0);  // fabric-mapped
  vkernels_fi_status_t sb = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_eager_break(&b, &sb), 0);
  EXPECT_EQ(sb, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, EagerBreakTrueForHostBounce) {
  vkernels_fi_config_t a = FI_CFG(0, 0, 1);  // dram-only
  vkernels_fi_status_t sa = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_eager_break(&a, &sa), 1);
  EXPECT_EQ(sa, VKERNELS_FI_OK);
  vkernels_fi_config_t b = FI_CFG(0, 0, 0);  // no fabric path
  vkernels_fi_status_t sb = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_fabric_import_eager_break(&b, &sb), 1);
  EXPECT_EQ(sb, VKERNELS_FI_OK);
}

TEST(FabricImportCABI, EagerBreakNullConfig) {
  vkernels_fi_status_t status = VKERNELS_FI_OK;
  EXPECT_EQ(vkernels_fabric_import_eager_break(nullptr, &status), 0);
  EXPECT_EQ(status, VKERNELS_FI_ERR_INVALID_ARGUMENT);
}

TEST(FabricImportCABI, EagerBreakNullStatusOut) {
  vkernels_fi_config_t c = FI_CFG(0, 0, 1);
  EXPECT_EQ(vkernels_fabric_import_eager_break(&c, nullptr), 1);
}

// ---------------------------------------------------------------------------
// is_graph_capturable + transport_name + same_node_roof_gbps
// ---------------------------------------------------------------------------

TEST(FabricImportCABI, IsGraphCapturable) {
  EXPECT_EQ(vkernels_fabric_import_is_graph_capturable(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED),
            1);
  EXPECT_EQ(vkernels_fabric_import_is_graph_capturable(
                VKERNELS_FI_TRANSPORT_SAME_NODE_PEER),
            1);
  EXPECT_EQ(vkernels_fabric_import_is_graph_capturable(
                VKERNELS_FI_TRANSPORT_HOST_BOUNCE),
            0);
}

TEST(FabricImportCABI, TransportNameKnown) {
  EXPECT_EQ(std::string(vkernels_fabric_import_transport_name(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED)),
            "fabric-mapped");
  EXPECT_EQ(std::string(vkernels_fabric_import_transport_name(
                VKERNELS_FI_TRANSPORT_SAME_NODE_PEER)),
            "same-node-peer");
  EXPECT_EQ(std::string(vkernels_fabric_import_transport_name(
                VKERNELS_FI_TRANSPORT_HOST_BOUNCE)),
            "host-bounce");
}

TEST(FabricImportCABI, TransportNameUnknown) {
  EXPECT_EQ(std::string(vkernels_fabric_import_transport_name(99)), "?");
}

TEST(FabricImportCABI, SameNodeRoofByTransport) {
  EXPECT_NEAR(vkernels_fabric_import_same_node_roof_gbps(
                  VKERNELS_FI_TRANSPORT_FABRIC_MAPPED),
              88.5, 1e-9);
  EXPECT_NEAR(vkernels_fabric_import_same_node_roof_gbps(
                  VKERNELS_FI_TRANSPORT_SAME_NODE_PEER),
              88.5, 1e-9);
  EXPECT_NEAR(vkernels_fabric_import_same_node_roof_gbps(
                  VKERNELS_FI_TRANSPORT_HOST_BOUNCE),
              1.4, 1e-9);
}

// ---------------------------------------------------------------------------
// cross_node_kv_throughput -- per-hop cost vs roofline + fallback + caveat
// ---------------------------------------------------------------------------

TEST(FabricImportCABI, ThroughputFabricMappedBelowRoofline) {
  vkernels_cross_node_kv_cost_t c{};
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, 1u << 20, /*gh200=*/0,
                &c, &status),
            VKERNELS_FI_OK);
  EXPECT_EQ(status, VKERNELS_FI_OK);
  // The binding resource is min(fabric link 50, kernel roof 88.5) = 50.
  EXPECT_NEAR(c.per_hop_gbps, 50.0, 1e-9);
  EXPECT_GT(c.per_hop_us, 0.0);
  EXPECT_NEAR(c.same_node_roof_gbps, 88.5, 1e-9);
  EXPECT_LT(c.per_hop_gbps, c.same_node_roof_gbps);
  EXPECT_EQ(c.gh200_dram_only_caveat, 0);
}

TEST(FabricImportCABI, ThroughputGh200DegradesToFallback) {
  vkernels_cross_node_kv_cost_t c{};
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, 1u << 20, /*gh200=*/1,
                &c, &status),
            VKERNELS_FI_OK);
  EXPECT_NEAR(c.per_hop_gbps, c.bulk_copy_fallback_gbps, 1e-9);
  EXPECT_EQ(c.gh200_dram_only_caveat, 1);
}

TEST(FabricImportCABI, ThroughputSameNodePeerHitsRoof) {
  vkernels_cross_node_kv_cost_t c{};
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_SAME_NODE_PEER, 1u << 20, /*gh200=*/0,
                &c, &status),
            VKERNELS_FI_OK);
  EXPECT_NEAR(c.per_hop_gbps, 88.5, 1e-9);
  EXPECT_NEAR(c.same_node_roof_gbps, 88.5, 1e-9);
  EXPECT_EQ(c.gh200_dram_only_caveat, 0);
}

TEST(FabricImportCABI, ThroughputHostBounceFlagsCaveat) {
  vkernels_cross_node_kv_cost_t c{};
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_HOST_BOUNCE, 1u << 20, /*gh200=*/0,
                &c, &status),
            VKERNELS_FI_OK);
  EXPECT_NEAR(c.per_hop_gbps, c.bulk_copy_fallback_gbps, 1e-9);
  EXPECT_EQ(c.gh200_dram_only_caveat, 1);
  EXPECT_LT(c.per_hop_gbps, c.same_node_roof_gbps);
}

TEST(FabricImportCABI, ThroughputZeroBytesIsZero) {
  vkernels_cross_node_kv_cost_t c{};
  vkernels_fi_status_t status = VKERNELS_FI_ERR_INTERNAL;
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, 0, /*gh200=*/0,
                &c, &status),
            VKERNELS_FI_OK);
  EXPECT_EQ(c.per_hop_gbps, 0.0);
  EXPECT_EQ(c.per_hop_us, 0.0);
}

TEST(FabricImportCABI, ThroughputNullOut) {
  vkernels_fi_status_t status = VKERNELS_FI_OK;
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, 1u << 20, /*gh200=*/0,
                nullptr, &status),
            VKERNELS_FI_ERR_INVALID_ARGUMENT);
  EXPECT_EQ(status, VKERNELS_FI_ERR_INVALID_ARGUMENT);
}

TEST(FabricImportCABI, ThroughputNullStatusOut) {
  vkernels_cross_node_kv_cost_t c{};
  EXPECT_EQ(vkernels_cross_node_kv_throughput(
                VKERNELS_FI_TRANSPORT_FABRIC_MAPPED, 1u << 20, /*gh200=*/0,
                &c, nullptr),
            VKERNELS_FI_OK);
}
