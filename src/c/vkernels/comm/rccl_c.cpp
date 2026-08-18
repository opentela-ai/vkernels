// vkernels/comm/rccl_c.cpp — thin `extern "C"` wrapper over the HIP/RCCL
// transport host reference (issue #19).
//
// Always compiled (no GPU, no RCCL, no libfabric), so the host CI job and
// its 100% line-coverage gate exercise the same planning surface a non-C++
// consumer reaches through rccl_c.h. Every C++ exception thrown by the host
// reference (std::invalid_argument via VK_EXPECTS) is caught here and folded
// into a status code, so no exception ever crosses the ABI boundary.
#include "vkernels/comm/rccl_c.h"

#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/rccl.hpp"

#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

vkernels_rccl_status_t to_status(const std::exception& e) {
  if (dynamic_cast<const std::invalid_argument*>(&e))
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  // The host reference (rccl.cpp) only throws std::invalid_argument via
  // VK_EXPECTS; no path across the ABI throws std::out_of_range or
  // std::runtime_error, so these branches are defensively dead.
  if (dynamic_cast<const std::out_of_range*>(&e))  // LCOV_EXCL_LINE
    return VKERNELS_RCCL_ERR_OUT_OF_RANGE;  // LCOV_EXCL_LINE
  (void)e;  // LCOV_EXCL_LINE
  return VKERNELS_RCCL_ERR_INTERNAL;  // LCOV_EXCL_LINE
}

void copy_cstr(char* dst, std::size_t cap, const std::string& src) {
  if (cap == 0) return;
  const std::size_t n = src.size() < cap - 1 ? src.size() : cap - 1;
  src.copy(dst, n);
  dst[n] = '\0';
}

vkernels_rccl_mode_t to_c_mode(vkernels::comm::RcclTransportMode m) {
  switch (m) {
    case vkernels::comm::RcclTransportMode::kAdaptive: return VKERNELS_RCCL_MODE_ADAPTIVE;
    // resolve_rccl_transport only ever emits kAdaptive (no env forces
    // a mode), so these arms and the trailing post-switch return are
    // defensively dead. The switch is exhaustive over the enum.
    case vkernels::comm::RcclTransportMode::kForceSlingshot:  // LCOV_EXCL_LINE
      return VKERNELS_RCCL_MODE_FORCE_SLINGSHOT;  // LCOV_EXCL_LINE
    case vkernels::comm::RcclTransportMode::kForceSocket:  // LCOV_EXCL_LINE
      return VKERNELS_RCCL_MODE_FORCE_SOCKET;  // LCOV_EXCL_LINE
  }
  return VKERNELS_RCCL_MODE_ADAPTIVE;  // LCOV_EXCL_LINE (exhaustive switch)
}

vkernels::comm::RcclTransportMode from_c_mode(vkernels_rccl_mode_t m) {
  switch (m) {
    case VKERNELS_RCCL_MODE_FORCE_SLINGSHOT:
      return vkernels::comm::RcclTransportMode::kForceSlingshot;
    case VKERNELS_RCCL_MODE_FORCE_SOCKET:
      return vkernels::comm::RcclTransportMode::kForceSocket;
    case VKERNELS_RCCL_MODE_ADAPTIVE:
    default:
      return vkernels::comm::RcclTransportMode::kAdaptive;
  }
}

vkernels::comm::RcclReduceOp from_c_op(vkernels_rccl_reduce_op_t op) {
  switch (op) {
    case VKERNELS_RCCL_REDUCE_MAX: return vkernels::comm::RcclReduceOp::kMax;
    case VKERNELS_RCCL_REDUCE_MIN: return vkernels::comm::RcclReduceOp::kMin;
    case VKERNELS_RCCL_REDUCE_SUM:
    default:
      return vkernels::comm::RcclReduceOp::kSum;
  }
}

}  // namespace

// Opaque-handle body, visible only to this C++ translation unit. C callers
// hold an incomplete struct pointer; the lifetime is managed by the
// create/destroy pair below.
struct vkernels_rccl_allreduce_plan {
  vkernels::comm::RcclAllreducePlan* impl;
};

extern "C" {

vkernels_rccl_status_t vkernels_rccl_resolve_transport(
    const vkernels_rccl_env_kv_t* env, int n,
    vkernels_rccl_config_t* out) {
  if (out == nullptr || (n > 0 && env == nullptr))
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  try {
    std::vector<std::pair<std::string, std::string>> e;
    e.reserve(static_cast<std::size_t>(n > 0 ? n : 0));
    for (int i = 0; i < n; ++i) {
      const auto& kv = env[static_cast<std::size_t>(i)];
      e.emplace_back(kv.name ? kv.name : "", kv.value ? kv.value : "");
    }
    auto cfg = vkernels::comm::resolve_rccl_transport(e);
    out->mode = to_c_mode(cfg.mode);
    copy_cstr(out->_net_plugin, sizeof(out->_net_plugin), cfg.net_plugin);
    out->net_plugin = out->_net_plugin;
    copy_cstr(out->_socket_ifname, sizeof(out->_socket_ifname), cfg.socket_ifname);
    out->socket_ifname = out->_socket_ifname;
    out->ib_disabled = cfg.ib_disabled ? 1 : 0;
    copy_cstr(out->_ofi_provider, sizeof(out->_ofi_provider), cfg.ofi_provider);
    out->ofi_provider = out->_ofi_provider;
    out->min_msg_for_ofi = static_cast<uint64_t>(cfg.min_msg_for_ofi);
    out->plugin_is_cuda_built = cfg.plugin_is_cuda_built ? 1 : 0;
    return VKERNELS_RCCL_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

vkernels_rccl_status_t vkernels_rccl_resolve_transport_for(
    uint64_t total_bytes, int inter_node_edges,
    const vkernels_rccl_config_t* cfg, vkernels_rccl_transport_t* out) {
  if (cfg == nullptr || out == nullptr)
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  try {
    vkernels::comm::RcclTransportConfig c;
    c.mode = from_c_mode(cfg->mode);
    c.net_plugin = cfg->net_plugin ? cfg->net_plugin : "";
    c.socket_ifname = cfg->socket_ifname ? cfg->socket_ifname : "";
    c.ib_disabled = cfg->ib_disabled != 0;
    c.ofi_provider = cfg->ofi_provider ? cfg->ofi_provider : "";
    c.min_msg_for_ofi = static_cast<std::size_t>(cfg->min_msg_for_ofi);
    c.plugin_is_cuda_built = cfg->plugin_is_cuda_built != 0;
    auto t = vkernels::comm::resolve_transport(
        static_cast<std::size_t>(total_bytes), inter_node_edges, c);
    *out = (t == vkernels::comm::RcclTransport::kSlingshotOfi)
               ? VKERNELS_RCCL_SLINGSHOT_OFI
               : VKERNELS_RCCL_SOCKET;
    return VKERNELS_RCCL_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

double vkernels_rccl_est_socket_us(uint64_t total_bytes, int inter_node_edges) {
  return vkernels::comm::est_rccl_socket_us(static_cast<std::size_t>(total_bytes),
                                            inter_node_edges);
}

double vkernels_rccl_est_ofi_us(uint64_t total_bytes, int inter_node_edges) {
  return vkernels::comm::est_rccl_ofi_us(static_cast<std::size_t>(total_bytes),
                                         inter_node_edges);
}

vkernels_rccl_status_t vkernels_rccl_build_cross_node_ring(
    const int* node_of, int world, int nodes,
    vkernels_rccl_node_topology_t* out, int* inout_n) {
  if (inout_n == nullptr || (world > 0 && node_of == nullptr))
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  try {
    std::vector<int> node_of_vec(static_cast<std::size_t>(world > 0 ? world : 0));
    for (int i = 0; i < world; ++i)
      node_of_vec[static_cast<std::size_t>(i)] = node_of[i];
    auto topo = vkernels::comm::build_cross_node_ring(node_of_vec, nodes);
    const int need = static_cast<int>(topo.size());
    if (out == nullptr) {
      *inout_n = need;
      return VKERNELS_RCCL_OK;
    }
    if (*inout_n < need) {
      *inout_n = need;
      return VKERNELS_RCCL_ERR_OUT_OF_RANGE;
    }
    for (int i = 0; i < need; ++i) {
      const auto& t = topo[static_cast<std::size_t>(i)];
      auto& d = out[static_cast<std::size_t>(i)];
      d.rank = t.rank; d.world = t.world; d.node = t.node; d.nodes = t.nodes;
      d.local_rank = t.local_rank; d.local_size = t.local_size;
      d.next = t.next; d.prev = t.prev;
      d.next_is_remote = t.next_is_remote ? 1 : 0;
      d.prev_is_remote = t.prev_is_remote ? 1 : 0;
    }
    *inout_n = need;
    return VKERNELS_RCCL_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

vkernels_rccl_status_t vkernels_rccl_discover_ofi_cxi(
    const vkernels_rccl_config_t* cfg, int libfabric_present,
    vkernels_rccl_ofi_info_t* out) {
  if (cfg == nullptr || out == nullptr)
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  try {
    vkernels::comm::RcclTransportConfig c;
    c.mode = from_c_mode(cfg->mode);
    c.net_plugin = cfg->net_plugin ? cfg->net_plugin : "";
    c.ofi_provider = cfg->ofi_provider ? cfg->ofi_provider : "";
    c.plugin_is_cuda_built = cfg->plugin_is_cuda_built != 0;
    auto info = vkernels::comm::discover_ofi_cxi(c, libfabric_present != 0);
    out->available = info.available ? 1 : 0;
    out->num_devices = info.num_devices;
    copy_cstr(out->_provider, sizeof(out->_provider), info.provider);
    out->provider = out->_provider;
    copy_cstr(out->_plugin_path, sizeof(out->_plugin_path), info.plugin_path);
    out->plugin_path = out->_plugin_path;
    out->plugin_is_cuda_built = info.plugin_is_cuda_built ? 1 : 0;
    copy_cstr(out->_reason, sizeof(out->_reason), info.reason);
    out->reason = out->_reason;
    return VKERNELS_RCCL_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

vkernels_rccl_status_t vkernels_rccl_allreduce_plan_create(
    int world, int rank, vkernels_rccl_reduce_op_t op, uint64_t capacity,
    vkernels_rccl_allreduce_plan_t** out) {
  if (out == nullptr) return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  try {
    auto* p = new vkernels_rccl_allreduce_plan{};
    // Stored as the host reference via a pimpl; the opaque type is just a
    // wrapper holding the real plan.
    p->impl = new vkernels::comm::RcclAllreducePlan(
        world, rank, from_c_op(op), static_cast<std::size_t>(capacity));
    *out = p;
    return VKERNELS_RCCL_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

vkernels_rccl_status_t vkernels_rccl_allreduce_plan_execute(
    vkernels_rccl_allreduce_plan_t* plan, float* buf, uint64_t n,
    void* next, void* prev) {
  if (plan == nullptr || plan->impl == nullptr)
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  if (buf == nullptr || n == 0)
    return VKERNELS_RCCL_ERR_INVALID_ARGUMENT;
  try {
    auto* ch_next = static_cast<vkernels::comm::Channel*>(next);
    auto* ch_prev = static_cast<vkernels::comm::Channel*>(prev);
    plan->impl->execute(buf, static_cast<std::size_t>(n), *ch_next, *ch_prev,
                        /*stream=*/nullptr);
    return VKERNELS_RCCL_OK;
  } catch (const std::exception& e) {
    return to_status(e);
  }
}

void vkernels_rccl_allreduce_plan_destroy(vkernels_rccl_allreduce_plan_t* plan) {
  if (plan == nullptr) return;
  delete plan->impl;
  delete plan;
}

}  // extern "C"
