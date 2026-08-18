// vkernels/comm/rccl.hpp
//
// HIP/RCCL cross-node transport for vkernels comm (issue #19).
//
// On MI300A/A (gfx942) cross-node TP all-reduce runs over RCCL. Out of the
// box RCCL falls back to its Socket transport (`NCCL_SOCKET_IFNAME=hsn0`,
// `NCCL_IB_DISABLE=1`) on CSCS beverin because the EDF `aws_ofi_nccl`
// plugin is CUDA-built (`libnccl-net.so` needs `libcudart.so`) and cannot
// init on ROCm. This is the documented throughput bottleneck (cookbook
// `deployments/llm/beverin/kimi-k3-vllm/README.md`, fix 4).
//
// This module adds, per the two-implementation model:
//
//   * A HIP/RCCL channel behind the existing `Channel`/all-reduce interface
//     (`RcclChannel`, sibling to the host `MockChannel` and the CUDA
//     `NcclChannel` stub), using `rcclSend` / `rcclRecv` over the ring.
//     Declared in `rccl_hip.hpp` (HIP-only, it takes `rcclComm_t`).
//   * A HIP-aware OFI/CXI net plugin (`plugins/rccl-net-ofi/librccl-net-ofi`)
//     so RCCL selects Slingshot RDMA instead of Socket. Gated behind
//     `VKERNELS_HAS_OFI` (libfabric found).
//   * A pure transport-selection + cost model (Slingshot OFI vs Socket) and a
//     cross-node ring topology builder, both host-testable without RCCL.
//   * A graph-capturable all-reduce plan (`RcclAllreducePlan`, host reference
//     here; `RcclAllreducePlanHip` in `rccl_hip.hpp` captures `rcclAllReduce`
//     into a `hipGraph`), tying into issue #10.
//
// The host reference (rccl.cpp) carries the contract checks and the cost
// model so both are unit-testable on a machine with no GPU. The HIP path
// (rccl.hip) performs the real RCCL calls and graph capture, compiled only
// with ROCm + RCCL.
#pragma once

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/core/stream.hpp"
#include "vkernels/util/config.hpp"

namespace vkernels::comm {

// ---------------------------------------------------------------------------
// Transport selection (Slingshot OFI/CXI vs Socket)
// ---------------------------------------------------------------------------

// The two RCCL transports a cross-node collective can take. Socket is the
// current beverin fallback (TCP-over-Slingshot, no RDMA); SlingshotOfi is
// the HIP-aware OFI/CXI plugin path (RDMA over the Slingshot fabric).
enum class RcclTransport { kSocket = 0, kSlingshotOfi = 1 };

// Human-readable transport name (for logging / the bench).
inline constexpr const char* transport_name(RcclTransport t) {
  return t == RcclTransport::kSlingshotOfi ? "slingshot-ofi" : "socket";
}

// Forced-dispatch modes, for testing and A/B tuning on a target machine.
// kAdaptive applies the cost model below; the force modes bypass it.
enum class RcclTransportMode { kAdaptive = 0, kForceSlingshot = 1, kForceSocket = 2 };

// All-reduce reduction operators, mirroring RCCL's `rcclRedOp_t` so the
// host reference and the HIP path name the same operations.
enum class RcclReduceOp { kSum = 0, kMax = 1, kMin = 2 };

// Stream helpers so the dispatch decision and the minitest failure path can
// log the enums (RcclTransport/Mode/ReduceOp have no implicit int cast).
std::ostream& operator<<(std::ostream& os, RcclTransport t);
std::ostream& operator<<(std::ostream& os, RcclTransportMode m);
std::ostream& operator<<(std::ostream& os, RcclReduceOp op);

// Configuration the HIP/RCCL path resolves from the environment and the
// caller. The host reference (this header) owns the resolution and the
// dispatch decision so both are unit-testable without RCCL.
struct RcclTransportConfig {
  // Dispatch mode (kAdaptive by default; force modes for A/B tuning).
  RcclTransportMode mode = RcclTransportMode::kAdaptive;
  // Resolved `NCCL_NET`/`RCCL_NET` value (e.g. "aws_ofi_rccl" for the
  // HIP-aware vendor plugin, "librccl-net-ofi" for the one in plugins/).
  // Empty => RCCL's built-in Socket transport (the beverin fallback).
  std::string net_plugin;
  // `NCCL_SOCKET_IFNAME`, the interface Socket falls back to (hsn0 on
  // beverin). Only consulted when the transport is Socket.
  std::string socket_ifname = "hsn0";
  // `NCCL_IB_DISABLE`. Socket-over-Slingshot runs with IB disabled
  // (true on beverin); the OFI/CXI plugin ignores it.
  bool ib_disabled = true;
  // `FI_PROVIDER`, the libfabric provider the OFI plugin uses ("cxi" for
  // HPE Slingshot).
  std::string ofi_provider = "cxi";
  // Minimum cross-node payload (bytes) at which the OFI/CXI plugin is
  // eligible. Below it the Socket vs OFI margin is inside launch-floor
  // noise, so the adaptive path keeps Socket (mirrors the p2p_gather
  // min-runs floor). Defaults to 1 MiB.
  std::size_t min_msg_for_ofi = 1u << 20;
  // Set by resolve_rccl_transport when `net_plugin` names a CUDA-built
  // RCCL/NCCL net plugin (e.g. the EDF `aws_ofi_nccl`). Such a plugin
  // cannot init on ROCm, so the dispatcher must stay on Socket until a
  // HIP-aware plugin is configured — the beverin bug this module fixes.
  bool plugin_is_cuda_built = false;
};

// Heuristic: does `name` identify a CUDA-built RCCL/NCCL net plugin? The EDF
// `aws_ofi_nccl` (note "nccl", no "rccl") links `libcudart.so` and cannot
// init on ROCm; `aws_ofi_rccl` / `librccl-net-ofi` are HIP-aware. Pure, no
// dlopen, so it is unit-testable on the host.
bool is_cuda_built_plugin(const std::string& name);

// Resolve the transport configuration from a `(name, value)` environment.
// Recognised names (case-insensitive): `NCCL_NET`, `RCCL_NET`,
// `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`, `FI_PROVIDER`. Pure; reads only
// the supplied list, never the real `environ` or RCCL, so it is unit-testable.
RcclTransportConfig resolve_rccl_transport(
    const std::vector<std::pair<std::string, std::string>>& env);

// ---------------------------------------------------------------------------
// Cost model (Slingshot OFI vs Socket) — host-tested, the comm analogue of
// p2p_gather::est_*_us / prefer_gather_kernel.
// ---------------------------------------------------------------------------

// Estimated all-reduce latency (microseconds) of the RCCL Socket transport
// for a `total_bytes` cross-node payload whose ring traverses
// `inter_node_edges` inter-node links. Socket pays a fixed TCP-over-Slingshot
// latency per inter-node message (each edge in the ring is one such message
// per step); constants fitted to beverin Slingshot-vs-Socket measurements and
// tunable per deployment (see docs/comm-rccl.md). `total_bytes == 0` -> 0.
double est_rccl_socket_us(std::size_t total_bytes, int inter_node_edges);

// Estimated all-reduce latency (microseconds) of the RCCL Slingshot OFI/CXI
// transport for the same payload. RDMA removes the per-edge TCP penalty, so
// the cost is the launch floor plus a bandwidth term (independent of the
// edge count). `total_bytes == 0` -> 0.
double est_rccl_ofi_us(std::size_t total_bytes, int inter_node_edges);

// Pure dispatch decision: true -> use the OFI/CXI plugin (Slingshot RDMA),
// false -> Socket. Honours the configured mode (kForceSlingshot /
// kForceSocket bypass the model; kAdaptive compares) and the min-message
// floor. Zero bytes or zero inter-node edges never takes OFI (no cross-node
// traffic to optimise).
bool prefer_slingshot_rccl(std::size_t total_bytes, int inter_node_edges,
                           const RcclTransportConfig& cfg);

// Resolve the transport for one collective: the cost decision plus the
// deployment facts. Socket is forced when there is no cross-node traffic
// (zero edges) or the only configured plugin is CUDA-built (the beverin
// bug); otherwise prefer_slingshot_rccl decides. Pure.
RcclTransport resolve_transport(std::size_t total_bytes, int inter_node_edges,
                                const RcclTransportConfig& cfg);

// ---------------------------------------------------------------------------
// Cross-node ring topology
// ---------------------------------------------------------------------------

// A cross-node ring rank: rank/world plus the node it runs on, so the ring
// can be laid out to minimise inter-node hops (a node-major ring pays the
// slow inter-node link exactly `nodes` times — the minimum for a single ring
// touching every node). `next_is_remote` / `prev_is_remote` mark the
// inter-node edges the Socket transport pays TCP overhead on.
struct NodeTopology {
  int rank = 0;
  int world = 1;
  int node = 0;         // node id of this rank
  int nodes = 1;        // total nodes
  int local_rank = 0;   // rank within its node
  int local_size = 1;   // ranks on this node
  int next = 0;         // (rank + 1) % world
  int prev = 0;         // (rank - 1 + world) % world
  bool next_is_remote = false;  // next rank on a different node
  bool prev_is_remote = false;  // prev rank on a different node
};

// Build a cross-node ring topology. `node_of[rank]` gives the node id of
// each rank; `nodes` is the total node count. The ring is laid out in
// node-major order (all of node 0, then node 1, ...) so consecutive ranks
// share a node whenever possible — exactly `nodes` inter-node edges (one per
// node boundary, including the wrap-around). Throws std::invalid_argument
// when `node_of` is empty, a node id is out of [0, nodes), or a node has no
// rank (a hole). The per-rank hop count is `(next_is_remote?1:0) +
// (prev_is_remote?1:0)`; the ring's total inter-node edge count is
// `inter_node_ring_edges(topo)` == `nodes` (0 when single-node).
std::vector<NodeTopology> build_cross_node_ring(const std::vector<int>& node_of,
                                                int nodes);

// Total number of inter-node edges in the ring (== sum of next_is_remote).
// Equal to `nodes` for a node-major ring touching every node, 0 for a
// single-node world. This is the `inter_node_edges` the Socket transport
// pays a TCP penalty on and the input to the cost model.
int inter_node_ring_edges(const std::vector<NodeTopology>& topo);

// Per-rank inter-node hop count (0, 1, or 2): the number of this rank's two
// ring neighbours that live on a different node. Boundary ranks of each
// node have 2; interior ranks have 0; a single-rank node has both.
int cross_node_hops(const NodeTopology& t);

// ---------------------------------------------------------------------------
// OFI/CXI discovery (host-testable)
// ---------------------------------------------------------------------------

// What the HIP/RCCL path can tell about the OFI/CXI fabric without a GPU:
// whether the configured provider is reachable, how many CXI NICs it exposes,
// and which plugin .so implements it. `reason` explains an unavailable
// result so a deployment log says exactly why Socket stayed in use.
struct OfiCxiInfo {
  bool available = false;
  int num_devices = 0;        // CXI NICs the provider reports (0 if unknown)
  std::string provider;       // resolved FI_PROVIDER ("cxi" on Slingshot)
  std::string plugin_path;    // resolved NCCL_NET .so ("" => built-in Socket)
  bool plugin_is_cuda_built = false;  // the beverin bug (CUDA plugin on ROCm)
  std::string reason;             // human-readable status (empty when available)
};

// Discover the OFI/CXI fabric for `cfg`. Pure: it inspects `cfg` and, when
// `libfabric_present` is true, reports the CXI NIC count (the caller passes
// whether a libfabric installation was found so the host reference stays
// free of an actual libfabric dependency). Sets `reason` so a deployment can
// log "stayed on Socket: <reason>" — the same diagnostic the beverin
// cookbook fix #4 needs.
OfiCxiInfo discover_ofi_cxi(const RcclTransportConfig& cfg,
                            bool libfabric_present);

// ---------------------------------------------------------------------------
// Graph-capturable all-reduce plan (host reference, issue #10)
// ---------------------------------------------------------------------------
//
// ring_allreduce_rank issues 2*(world-1) host enqueues (send/recv per step).
// A prepared plan moves the validation and (on HIP) the rcclComm_t / stream
// setup to a single prepare step and execute() only enqueues ONE stream task
// (the host reference here; the real rcclAllReduce on the HIP path, captured
// into a hipGraph so a replay needs no host progress). This is the K3
// microbatch pattern: the same all-reduce runs every layer, so prepare-once
// and replay.
//
// The host reference is driven exactly like ring_allreduce_rank: execute()
// takes the rank's `next` and `prev` channels, so a test wires up a mock
// ring (make_ring_channels) and runs one plan per rank concurrently to
// verify the result matches the element-wise reduction.
class RcclAllreducePlan {
 public:
  // Validate once: `world > 0`, `rank` in [0, world), `op` a known operator,
  // `capacity_elems > 0` and divisible by `world`. Throws
  // std::invalid_argument on violation.
  RcclAllreducePlan(int world, int rank, RcclReduceOp op,
                    std::size_t capacity_elems);

  RcclAllreducePlan(const RcclAllreducePlan&) = delete;
  RcclAllreducePlan& operator=(const RcclAllreducePlan&) = delete;

  int world() const { return world_; }
  int rank() const { return rank_; }
  RcclReduceOp op() const { return op_; }
  std::size_t capacity() const { return capacity_; }

  // All-reduce `n` floats at `buf` in place across `world` ranks, using the
  // ring channels `next` (to rank+1) and `prev` (from rank-1). `n` must be
  // <= capacity() and divisible by world; a null `stream` runs to completion
  // before returning. Exactly ONE stream task regardless of world (the
  // graph-capturable contract: one task == one graph node). The host
  // reference honours `op` (sum / max / min).
  void execute(float* buf, std::size_t n, Channel& next, Channel& prev,
               Stream* stream = nullptr);

 private:
  int world_;
  int rank_;
  RcclReduceOp op_;
  std::size_t capacity_;
};

}  // namespace vkernels::comm
