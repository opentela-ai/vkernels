// vkernels/comm/rccl.cpp — host reference for the HIP/RCCL transport (issue #19).
//
// The CPU reference is the correctness oracle for the cross-node RCCL
// transport: it carries the transport-selection cost model, the cross-node
// ring topology builder, the OFI/CXI discovery, and a graph-capturable
// all-reduce plan. It is always compiled and fully unit-tested on a machine
// with no GPU; the HIP/RCCL path (rccl.hip) performs the real rcclSend /
// rcclRecv / rcclAllReduce calls and hipGraph capture, compiled only with
// ROCm + RCCL.
#include "vkernels/comm/rccl.hpp"

#include <algorithm>
#include <cctype>
#include <ostream>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "vkernels/comm/allreduce.hpp"
#include "vkernels/util/error.hpp"

namespace vkernels::comm {
namespace {

// (Stream operators are defined out-of-line below; the helpers here are the
// always-compiled transport cost model and topology.)

// ---------------------------------------------------------------------------
// Cost-model constants (Slingshot OFI/CXI vs Socket, cross-node TP all-reduce)
// ---------------------------------------------------------------------------
//
// Fitted to the CSCS beverin (MI300A / gfx942) Slingshot-vs-Socket
// observation: cross-node TP all-reduce over RCCL Socket
// (NCCL_SOCKET_IFNAME=hsn0, NCCL_IB_DISABLE=1) is the documented bottleneck,
// and a HIP-aware OFI/CXI plugin (RDMA over the Slingshot fabric) removes the
// per-edge TCP penalty and lifts bandwidth. These mirror the p2p_gather
// cost model: a launch floor, a per-MiB bandwidth term, and (Socket only) a
// per-inter-node-edge TCP-over-Slingshot latency. Tunable per deployment via
// set_rccl_transport_config / by editing here; the bench re-fits them.
//
//   * Socket: ~50 us launch floor, ~6.0 us/MiB (TCP-over-Slingshot, well
//     below the fabric's line rate), +25 us per inter-node edge (one
//     TCP-over-Slingshot message per edge per ring step). At 48 MiB over a
//     2-edge cross-node ring: max(50, 6*48) + 2*25 = 388 us.
//   * OFI/CXI: ~20 us launch floor, ~3.0 us/MiB (RDMA, full Slingshot
//     bandwidth), no per-edge penalty. At 48 MiB: max(20, 3*48) = 164 us ->
//     ~2.4x faster than Socket, the acceptance target.
constexpr double kSocketFloorUs = 50.0;    // RCCL Socket launch floor
constexpr double kSocketPerMiBUs = 6.0;    // TCP-over-Slingshot bandwidth
constexpr double kTcpPerEdgeUs = 25.0;     // per-inter-node-edge TCP latency
constexpr double kOfiFloorUs = 20.0;       // RCCL OFI launch floor
constexpr double kOfiPerMiBUs = 3.0;       // RDMA-over-Slingshot bandwidth

// Case-insensitive ASCII string compare (the env names are case-insensitive
// in RCCL/NCCL; FI_PROVIDER is lower-case by convention).
bool iequals(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) !=
        std::tolower(static_cast<unsigned char>(b[i])))
      return false;
  }
  return true;
}

// Parse an NCCL_IB_DISABLE value: empty or "1"/"true"/"yes"/"on"
// (case-insensitive) -> IB disabled (the documented beverin default);
// "0"/"false"/... -> IB enabled.
bool ib_disabled_value(const std::string& v) {
  if (v.empty()) return true;
  return iequals(v, "1") || iequals(v, "true") || iequals(v, "yes") ||
         iequals(v, "on");
}

// Element-wise reduction of one chunk into another under `op` (sum/max/min).
// Used by the ring reduce-scatter and the all-gather copy step.
void combine_chunk(std::vector<float>& mine, const std::vector<float>& got,
                   RcclReduceOp op) {
  VK_EXPECTS(mine.size() == got.size(), "chunk sizes must match");
  switch (op) {
    case RcclReduceOp::kSum:
      for (std::size_t k = 0; k < mine.size(); ++k) mine[k] += got[k];
      break;
    case RcclReduceOp::kMax:
      for (std::size_t k = 0; k < mine.size(); ++k) mine[k] = std::max(mine[k], got[k]);
      break;
    case RcclReduceOp::kMin:
      for (std::size_t k = 0; k < mine.size(); ++k) mine[k] = std::min(mine[k], got[k]);
      break;
  }
}

}  // namespace

bool is_cuda_built_plugin(const std::string& name) {
  // The EDF aws_ofi_nccl (note "nccl", no "rccl") is CUDA-built; the
  // HIP-aware counterparts (aws_ofi_rccl, librccl-net-ofi) name "rccl".
  std::string lower;
  lower.reserve(name.size());
  for (char c : name) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  const bool has_nccl = lower.find("nccl") != std::string::npos;
  const bool has_rccl = lower.find("rccl") != std::string::npos;
  return has_nccl && !has_rccl;
}

RcclTransportConfig resolve_rccl_transport(
    const std::vector<std::pair<std::string, std::string>>& env) {
  RcclTransportConfig cfg;
  for (const auto& kv : env) {
    if (iequals(kv.first, "NCCL_NET") || iequals(kv.first, "RCCL_NET")) {
      cfg.net_plugin = kv.second;
    } else if (iequals(kv.first, "NCCL_SOCKET_IFNAME")) {
      cfg.socket_ifname = kv.second;
    } else if (iequals(kv.first, "NCCL_IB_DISABLE")) {
      cfg.ib_disabled = ib_disabled_value(kv.second);
    } else if (iequals(kv.first, "FI_PROVIDER")) {
      cfg.ofi_provider = kv.second;
    }
  }
  cfg.plugin_is_cuda_built = is_cuda_built_plugin(cfg.net_plugin);
  return cfg;
}

double est_rccl_socket_us(std::size_t total_bytes, int inter_node_edges) {
  if (total_bytes == 0) return 0.0;
  const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  const double bw = std::max(kSocketFloorUs, kSocketPerMiBUs * mib);
  const int edges = inter_node_edges > 0 ? inter_node_edges : 0;
  return bw + static_cast<double>(edges) * kTcpPerEdgeUs;
}

double est_rccl_ofi_us(std::size_t total_bytes, int inter_node_edges) {
  (void)inter_node_edges;  // RDMA: no per-edge TCP penalty
  if (total_bytes == 0) return 0.0;
  const double mib = static_cast<double>(total_bytes) / (1024.0 * 1024.0);
  return std::max(kOfiFloorUs, kOfiPerMiBUs * mib);
}

bool prefer_slingshot_rccl(std::size_t total_bytes, int inter_node_edges,
                           const RcclTransportConfig& cfg) {
  if (total_bytes == 0) return false;       // nothing to send: never pay a plugin
  if (inter_node_edges <= 0) return false;  // no cross-node traffic to optimise
  switch (cfg.mode) {
    case RcclTransportMode::kForceSlingshot: return true;
    case RcclTransportMode::kForceSocket: return false;
    case RcclTransportMode::kAdaptive: break;
  }
  if (cfg.min_msg_for_ofi > 0 && total_bytes < cfg.min_msg_for_ofi) {
    return false;  // inside the Socket launch-floor noise band
  }
  return est_rccl_ofi_us(total_bytes, inter_node_edges) <
         est_rccl_socket_us(total_bytes, inter_node_edges);
}

RcclTransport resolve_transport(std::size_t total_bytes, int inter_node_edges,
                                const RcclTransportConfig& cfg) {
  // No cross-node traffic, or the only configured plugin is CUDA-built (the
  // beverin bug: aws_ofi_nccl cannot init on ROCm): Socket is the only
  // viable transport until a HIP-aware plugin is configured.
  if (inter_node_edges <= 0) return RcclTransport::kSocket;
  if (cfg.plugin_is_cuda_built) return RcclTransport::kSocket;
  return prefer_slingshot_rccl(total_bytes, inter_node_edges, cfg)
             ? RcclTransport::kSlingshotOfi
             : RcclTransport::kSocket;
}

// ---------------------------------------------------------------------------
// Cross-node ring topology
// ---------------------------------------------------------------------------

std::vector<NodeTopology> build_cross_node_ring(const std::vector<int>& node_of,
                                                int nodes) {
  const int world = static_cast<int>(node_of.size());
  VK_EXPECTS(world > 0, "node_of must be non-empty");
  VK_EXPECTS(nodes > 0, "nodes must be positive");
  for (int n : node_of)
    VK_EXPECTS(n >= 0 && n < nodes, "node id out of range [0, nodes)");

  // Rank order: node-major. Sort rank indices by (node, rank) so consecutive
  // ranks share a node whenever possible. Stable on rank within a node.
  std::vector<int> order;
  order.reserve(static_cast<std::size_t>(world));
  for (int n = 0; n < nodes; ++n)
    for (int r = 0; r < world; ++r)
      if (node_of[r] == n) order.push_back(r);

  // Every node id in [0, nodes) must have at least one rank (no holes): a
  // node-major ring over a node with no rank is degenerate and almost
  // always a caller miscount of `nodes`.
  std::vector<char> has_rank(static_cast<std::size_t>(nodes), 0);
  for (int n : node_of) has_rank[static_cast<std::size_t>(n)] = 1;
  for (std::size_t i = 0; i < has_rank.size(); ++i)
    VK_EXPECTS(has_rank[i] == 1, "every node in [0, nodes) must have a rank");

  // Per-node local rank bookkeeping (local ranks assigned in node order).
  std::vector<int> local_size(static_cast<std::size_t>(nodes), 0);
  for (int n : node_of) local_size[static_cast<std::size_t>(n)]++;
  std::vector<int> seen(static_cast<std::size_t>(nodes), 0);
  std::vector<NodeTopology> topo(static_cast<std::size_t>(world));
  for (std::size_t pos = 0; pos < order.size(); ++pos) {
    const int r = order[pos];
    const int n = node_of[static_cast<std::size_t>(r)];
    const int prev_r = order[(pos + order.size() - 1) % order.size()];
    const int next_r = order[(pos + 1) % order.size()];
    auto& t = topo[static_cast<std::size_t>(r)];
    t.rank = r;
    t.world = world;
    t.node = n;
    t.nodes = nodes;
    t.local_rank = seen[static_cast<std::size_t>(n)]++;
    t.local_size = local_size[static_cast<std::size_t>(n)];
    t.prev = prev_r;
    t.next = next_r;
    t.prev_is_remote = node_of[static_cast<std::size_t>(prev_r)] != n;
    t.next_is_remote = node_of[static_cast<std::size_t>(next_r)] != n;
  }
  return topo;
}

int inter_node_ring_edges(const std::vector<NodeTopology>& topo) {
  int edges = 0;
  for (const auto& t : topo) edges += t.next_is_remote ? 1 : 0;
  return edges;
}

int cross_node_hops(const NodeTopology& t) {
  return (t.next_is_remote ? 1 : 0) + (t.prev_is_remote ? 1 : 0);
}

// ---------------------------------------------------------------------------
// OFI/CXI discovery
// ---------------------------------------------------------------------------

OfiCxiInfo discover_ofi_cxi(const RcclTransportConfig& cfg,
                            bool libfabric_present) {
  OfiCxiInfo info;
  info.provider = cfg.ofi_provider.empty() ? std::string("cxi") : cfg.ofi_provider;
  info.plugin_path = cfg.net_plugin;

  if (!libfabric_present) {
    info.reason = "no libfabric installation found";
    return info;
  }
  if (cfg.plugin_is_cuda_built) {
    info.reason = "configured net plugin '" + cfg.net_plugin +
                  "' is CUDA-built and cannot init on ROCm";
    info.plugin_is_cuda_built = true;
    return info;
  }
  if (cfg.net_plugin.empty()) {
    info.reason = "no net plugin configured (NCCL_NET empty): RCCL built-in "
                  "transport, Slingshot RDMA unavailable";
    return info;
  }
  // A HIP-aware plugin is configured and libfabric is present: report the
  // provider as reachable. The actual NIC count is read by the plugin at
  // runtime (fi_getinfo); the host reference reports a positive placeholder
  // so the dispatcher and the bench can proceed without a GPU.
  info.available = true;
  info.num_devices = 1;  // at least one CXI NIC; refined at runtime
  return info;
}

// ---------------------------------------------------------------------------
// Graph-capturable all-reduce plan (host reference)
// ---------------------------------------------------------------------------

RcclAllreducePlan::RcclAllreducePlan(int world, int rank, RcclReduceOp op,
                                     std::size_t capacity_elems)
    : world_(world), rank_(rank), op_(op), capacity_(capacity_elems) {
  VK_EXPECTS(world > 0, "world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world, "rank out of range");
  VK_EXPECTS(op == RcclReduceOp::kSum || op == RcclReduceOp::kMax ||
                 op == RcclReduceOp::kMin,
             "unknown reduce op");
  VK_EXPECTS(capacity_elems > 0, "capacity must be positive");
  VK_EXPECTS(capacity_elems % static_cast<std::size_t>(world) == 0,
             "capacity must be divisible by world");
}

void RcclAllreducePlan::execute(float* buf, std::size_t n, Channel& next,
                                Channel& prev, Stream* stream) {
  VK_EXPECTS(buf != nullptr, "buf must be non-null");
  VK_EXPECTS(n > 0, "n must be positive");
  VK_EXPECTS(n <= capacity_, "n exceeds capacity");
  VK_EXPECTS(n % static_cast<std::size_t>(world_) == 0,
             "n must be divisible by world");

  auto run = [this, buf, n, &next, &prev] {
    if (world_ == 1) return;  // single rank: nothing to reduce
    std::vector<float> local(buf, buf + n);
    const std::size_t chunk = n / static_cast<std::size_t>(world_);

    auto chunk_of = [&](int c) {
      auto b = local.begin() + static_cast<long>(static_cast<std::size_t>(c) * chunk);
      return std::vector<float>(b, b + static_cast<long>(chunk));
    };
    auto replace_chunk = [&](int c, std::vector<float> v) {
      std::size_t start = static_cast<std::size_t>(c) * chunk;
      std::copy(v.begin(), v.end(), local.begin() + static_cast<long>(start));
    };

    // Phase 1: reduce-scatter (world-1 steps). After step t, rank i sends
    // chunk (i-t) and combines chunk (i-t-1) under `op`. When scatter ends,
    // rank i fully owns chunk (i+1) % world reduced.
    for (int t = 0; t < world_ - 1; ++t) {
      int send_c = ((rank_ - t) % world_ + world_) % world_;
      int recv_c = ((rank_ - t - 1) % world_ + world_) % world_;
      next.send(chunk_of(send_c));
      std::vector<float> got = prev.recv();
      std::vector<float> mine = chunk_of(recv_c);
      combine_chunk(mine, got, op_);
      replace_chunk(recv_c, std::move(mine));
    }
    // Phase 2: all-gather (world-1 steps), copying (not combining) the
    // fully-reduced chunks around the ring.
    for (int t = 0; t < world_ - 1; ++t) {
      int send_c = ((rank_ - t + 1) % world_ + world_) % world_;
      int recv_c = ((rank_ - t) % world_ + world_) % world_;
      next.send(chunk_of(send_c));
      std::vector<float> got = prev.recv();
      replace_chunk(recv_c, std::move(got));
    }
    std::copy(local.begin(), local.end(), buf);
  };

  if (stream == nullptr) {
    run();
    return;
  }
  stream->submit(std::move(run));
}

// ---------------------------------------------------------------------------
// Stream operators (logging / minitest failure path)
// ---------------------------------------------------------------------------

std::ostream& operator<<(std::ostream& os, RcclTransport t) {
  return os << transport_name(t);
}

std::ostream& operator<<(std::ostream& os, RcclTransportMode m) {
  switch (m) {
    case RcclTransportMode::kAdaptive: return os << "adaptive";
    case RcclTransportMode::kForceSlingshot: return os << "force-slingshot";
    case RcclTransportMode::kForceSocket: return os << "force-socket";
  }
  return os << "?";  // LCOV_EXCL_LINE (exhaustive switch)
}

std::ostream& operator<<(std::ostream& os, RcclReduceOp op) {
  switch (op) {
    case RcclReduceOp::kSum: return os << "sum";
    case RcclReduceOp::kMax: return os << "max";
    case RcclReduceOp::kMin: return os << "min";
  }
  return os << "?";  // LCOV_EXCL_LINE (exhaustive switch)
}

}  // namespace vkernels::comm
