// vkernels/comm/fabric_import_c.cpp -- thin `extern "C"` wrapper over the
// fabric import transport classification, the graph-capturable / eager-break
// decision, and the per-hop cost model (issue #49).
//
// Always compiled (no GPU, no fabric), so the host CI job and its 100%
// line-coverage gate exercise the same planning surface a non-C++ consumer
// reaches through fabric_import_c.h. The wrapped host functions
// (classify_fabric_import, is_import_graph_capturable,
// eager_break_fabric_import, fabric_import_transport_name,
// same_node_fabric_roof_gbps, cross_node_kv_throughput) are pure, so
// nothing is ever thrown across the ABI boundary.
#include "vkernels/comm/fabric_import_c.h"

#include "vkernels/comm/cross_node_kv.hpp"

#include "vkernels/comm/fabric_import.hpp"

namespace {

vkernels::comm::FabricImportTransport from_c_transport(int t) {
  switch (static_cast<vkernels_fi_transport_t>(t)) {
    case VKERNELS_FI_TRANSPORT_FABRIC_MAPPED:
      return vkernels::comm::FabricImportTransport::kFabricMapped;
    case VKERNELS_FI_TRANSPORT_SAME_NODE_PEER:
      return vkernels::comm::FabricImportTransport::kSameNodePeer;
    case VKERNELS_FI_TRANSPORT_HOST_BOUNCE:
      return vkernels::comm::FabricImportTransport::kHostBounce;
  }
  return vkernels::comm::FabricImportTransport::kHostBounce;  // LCOV_EXCL_LINE
}

int to_c_transport(vkernels::comm::FabricImportTransport t) {
  switch (t) {
    case vkernels::comm::FabricImportTransport::kFabricMapped:
      return VKERNELS_FI_TRANSPORT_FABRIC_MAPPED;
    case vkernels::comm::FabricImportTransport::kSameNodePeer:
      return VKERNELS_FI_TRANSPORT_SAME_NODE_PEER;
    case vkernels::comm::FabricImportTransport::kHostBounce:
      return VKERNELS_FI_TRANSPORT_HOST_BOUNCE;
  }
  return VKERNELS_FI_TRANSPORT_HOST_BOUNCE;  // LCOV_EXCL_LINE (exhaustive)
}

}  // namespace

extern "C" int vkernels_fabric_import_classify(
    const vkernels_fi_config_t* cfg, vkernels_fi_status_t* status_out) {
  if (cfg == nullptr) {
    if (status_out != nullptr)
      *status_out = VKERNELS_FI_ERR_INVALID_ARGUMENT;
    return VKERNELS_FI_TRANSPORT_HOST_BOUNCE;
  }
  vkernels::comm::FabricImportConfig cpp;
  cpp.same_node = cfg->same_node != 0;
  cpp.has_gpudirect_rdma = cfg->has_gpudirect_rdma != 0;
  cpp.dram_only_libfabric = cfg->dram_only_libfabric != 0;
  const int t = to_c_transport(vkernels::comm::classify_fabric_import(cpp));
  if (status_out != nullptr)
    *status_out = VKERNELS_FI_OK;
  return t;
}

extern "C" int vkernels_fabric_import_eager_break(
    const vkernels_fi_config_t* cfg, vkernels_fi_status_t* status_out) {
  if (cfg == nullptr) {
    if (status_out != nullptr)
      *status_out = VKERNELS_FI_ERR_INVALID_ARGUMENT;
    return 0;
  }
  vkernels::comm::FabricImportConfig cpp;
  cpp.same_node = cfg->same_node != 0;
  cpp.has_gpudirect_rdma = cfg->has_gpudirect_rdma != 0;
  cpp.dram_only_libfabric = cfg->dram_only_libfabric != 0;
  const int eager = vkernels::comm::eager_break_fabric_import(cpp) ? 1 : 0;
  if (status_out != nullptr)
    *status_out = VKERNELS_FI_OK;
  return eager;
}

extern "C" int vkernels_fabric_import_is_graph_capturable(int t) {
  return vkernels::comm::is_import_graph_capturable(from_c_transport(t)) ? 1 : 0;
}

extern "C" const char* vkernels_fabric_import_transport_name(int t) {
  switch (t) {
    case VKERNELS_FI_TRANSPORT_FABRIC_MAPPED:
      return vkernels::comm::fabric_import_transport_name(
          vkernels::comm::FabricImportTransport::kFabricMapped);
    case VKERNELS_FI_TRANSPORT_SAME_NODE_PEER:
      return vkernels::comm::fabric_import_transport_name(
          vkernels::comm::FabricImportTransport::kSameNodePeer);
    case VKERNELS_FI_TRANSPORT_HOST_BOUNCE:
      return vkernels::comm::fabric_import_transport_name(
          vkernels::comm::FabricImportTransport::kHostBounce);
    default:
      return "?";
  }
}

extern "C" double vkernels_fabric_import_same_node_roof_gbps(int t) {
  return vkernels::comm::same_node_fabric_roof_gbps(from_c_transport(t));
}

extern "C" vkernels_fi_status_t vkernels_cross_node_kv_throughput(
    int transport, size_t total_bytes, int gh200_dram_only,
    vkernels_cross_node_kv_cost_t* out, vkernels_fi_status_t* status_out) {
  if (out == nullptr) {
    if (status_out != nullptr)
      *status_out = VKERNELS_FI_ERR_INVALID_ARGUMENT;
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  }
  const auto c = vkernels::comm::cross_node_kv_throughput(
      from_c_transport(transport), total_bytes, gh200_dram_only != 0);
  out->transport = to_c_transport(c.transport);
  out->total_bytes = c.total_bytes;
  out->per_hop_gbps = c.per_hop_gbps;
  out->per_hop_us = c.per_hop_us;
  out->same_node_roof_gbps = c.same_node_roof_gbps;
  out->bulk_copy_fallback_gbps = c.bulk_copy_fallback_gbps;
  out->gh200_dram_only_caveat = c.gh200_dram_only_caveat ? 1 : 0;
  if (status_out != nullptr)
    *status_out = VKERNELS_FI_OK;
  return VKERNELS_FI_OK;
}

extern "C" vkernels_fi_status_t vkernels_cross_node_kv_select_route(
    const vkernels_cross_node_kv_access_t* access,
    const vkernels_fi_config_t* fabric,
    vkernels_cross_node_kv_route_t* out) {
  if (access == nullptr || fabric == nullptr || out == nullptr)
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  try {
    vkernels::comm::CrossNodeKvAccess cpp_access;
    cpp_access.world_size = access->world_size;
    cpp_access.receiver_count = access->receiver_count;
    cpp_access.evenly_sharded = access->evenly_sharded != 0;
    cpp_access.collective_available = access->collective_available != 0;
    cpp_access.collective_graph_supported =
        access->collective_graph_supported != 0;
    vkernels::comm::FabricImportConfig cpp_fabric;
    cpp_fabric.same_node = fabric->same_node != 0;
    cpp_fabric.has_gpudirect_rdma = fabric->has_gpudirect_rdma != 0;
    cpp_fabric.dram_only_libfabric = fabric->dram_only_libfabric != 0;
    const auto route =
        vkernels::comm::select_cross_node_kv_route(cpp_access, cpp_fabric);
    out->kind = static_cast<int>(route.kind);
    out->point_to_point_transport =
        static_cast<int>(route.point_to_point_transport);
    out->graph_capturable = route.graph_capturable ? 1 : 0;
    return VKERNELS_FI_OK;
  } catch (const std::invalid_argument&) {
    return VKERNELS_FI_ERR_INVALID_ARGUMENT;
  } catch (...) {  // LCOV_EXCL_LINE (pure value conversion cannot throw otherwise)
    return VKERNELS_FI_ERR_INTERNAL;  // LCOV_EXCL_LINE
  }
}
