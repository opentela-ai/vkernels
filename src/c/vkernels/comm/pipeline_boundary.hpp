// vkernels/comm/pipeline_boundary.hpp
//
// Graph-capturable cross-node pipeline-parallel boundary transfer
// (issue #10).
//
// When serving a PP>1 model (Kimi-K3, PP=3) with CUDA graphs, vLLM and
// sglang move the hidden state across the PP boundary with a host-side
// gloo `recv_object` / `send_object`. A captured CUDA graph replays
// without host participation, so a `recv_object` frozen inside it can
// never be serviced: every PP stage hangs (the documented gfx942
// deadlock). The only workaround today is `--enforce-eager` (no graphs),
// costing ~1.5-2x decode throughput.
//
// This module makes the boundary transfer **graph-capturable** (a pure
// device operation that replays with no host progress) where the transport
// allows, and provides an explicit **eager break-point API** mirroring
// vLLM `eager_break_during_capture` where it does not, so a host-staged
// boundary is *excluded* from the captured segment instead of silently
// frozen inside it (the second half of vllm-project/vllm#46253).
//
// Two-implementation model (mirrors rccl.{cpp,hpp} / rccl.hip):
//   * Host reference (pipeline_boundary.cpp, always compiled) — the
//     transport classification, the eager-break decision, a host model of
//     CUDA graph capture / instantiate / replay, and the boundary plan.
//     Fully unit-tested with 100% line coverage on a machine with no GPU.
//   * CUDA implementation (pipeline_boundary.cu, pipeline_boundary_cuda.hpp)
//     — the real device-side copy (peer) or ncclSend / ncclRecv (cross-node)
//     enqueued on a cudaStream_t, compiled only with a toolkit. The host
//     reference is the correctness oracle; the CUDA path mirrors its API.
//
// See docs/comm-pipeline-boundary.md for the architecture and the
// acceptance (capture + replay across two ranks, no host progress).
#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <vector>

#include "vkernels/comm/channel.hpp"

namespace vkernels::comm {

// The transport a PP-boundary token / hidden-state transfer takes.
//
// * kSameNodePeer  — same-node peer (NVLink / HBM, or ROCm IPC on
//   MI300A): a device-to-device copy, pure device, graph-capturable.
// * kCrossNodeNccl — cross-node over NCCL / RCCL whose graph-capture API
//   is supported: ncclSend / ncclRecv captured into the graph, replayed
//   with no host progress.
// * kHostStaged    — host-staged (gloo recv_object / send_object, or a
//   host-bounce transport with no graph support): NOT graph-capturable;
//   the framework must exclude it from the captured segment and run it
//   eagerly between graph launches (the eager-break path).
enum class PipelineTransport {
  kSameNodePeer = 0,
  kCrossNodeNccl = 1,
  kHostStaged = 2
};

// Human-readable transport name (for logging / the bench). Mirrors the
// rccl transport_name convention.
inline constexpr const char* pipeline_transport_name(PipelineTransport t) {
  switch (t) {
    case PipelineTransport::kSameNodePeer: return "same-node-peer";
    case PipelineTransport::kCrossNodeNccl: return "cross-node-nccl";
    case PipelineTransport::kHostStaged: return "host-staged";
  }
  return "?";  // LCOV_EXCL_LINE (exhaustive switch)
}

std::ostream& operator<<(std::ostream& os, PipelineTransport t);

// Deployment facts a PP boundary must be classified from, mirroring the
// knobs a serving runtime (vLLM, sglang) resolves before capture:
//
// * `same_node`            — the peer stage is on the same node, so
//   NVLink / HBM (or ROCm IPC on MI300A) makes the transfer a device
//   copy. Always graph-capturable.
// * `nccl_graph_supported` — the configured NCCL / RCCL exposes its
//   graph-capture API, so a cross-node ncclSend / ncclRecv can be
//   captured into a graph segment.
// * `gloo_fallback`        — a host-side gloo is wired for the boundary
//   (the vLLM / sglang default). Host-staged: NOT graph-capturable, even
//   when NCCL graphs are available — gloo is what the deadlock hangs on.
struct PipelineBoundaryConfig {
  bool same_node = false;
  bool nccl_graph_supported = false;
  bool gloo_fallback = false;
};

// Classify the boundary transport from the deployment facts. Pure.
//
//   same_node            -> kSameNodePeer   (peer device copy)
//   gloo_fallback        -> kHostStaged      (the deadlock trigger; gloo
//                                            takes precedence over NCCL)
//   nccl_graph_supported -> kCrossNodeNccl   (capturable cross-node)
//   otherwise            -> kHostStaged      (NCCL w/o graph API, or no
//                                            cross-node fabric)
PipelineTransport classify_boundary(const PipelineBoundaryConfig& cfg);

// True iff `t` is a pure device transfer that a CUDA / HIP graph segment
// can contain and replay with no host progress (same-node peer, or
// cross-node NCCL / RCCL whose graph API is supported). Host-staged is
// false: its host send / recv cannot be serviced during a replay.
bool is_graph_capturable(PipelineTransport t);

// Eager-break decision mirroring vLLM `eager_break_during_capture`: when
// the boundary transport is NOT graph-capturable, the framework must stop
// capture, run the transfer eagerly (host progress), then resume capture
// for the next segment — instead of silently freezing the host transfer
// inside the graph (the PP>1 deadlock). Pure; equivalent to
// `!is_graph_capturable(classify_boundary(cfg))`.
bool eager_break_during_capture(const PipelineBoundaryConfig& cfg);

// ---------------------------------------------------------------------------
// GraphCapture — a minimal host model of CUDA graph capture / replay
// ---------------------------------------------------------------------------
//
// A real CUDA / HIP graph is a device-side DAG: work submitted between
// cudaStreamBeginCapture and cudaStreamEndCapture is *recorded* (not run),
// instantiated once, then launched (cudaGraphLaunch) repeatedly with no
// host participation — every recorded op runs as device work, in order, on
// each launch. That is exactly the property a PP boundary must have to be
// replayable across decode iterations, and exactly what the host reference
// must demonstrate so the contract is unit-testable without a GPU.
//
// submit(op) records `op` while in_capture() (the device-work entry) and
// runs it immediately otherwise (stream semantics outside a capture
// region). replay() runs every recorded op exactly once, in capture order,
// with no new submits — the host-progress-free replay contract the tests
// assert (a captured boundary must not call back into the host ring).
//
// Multiple begin()/end() pairs model the eager-break path: a captured
// segment is ended at the boundary, the boundary runs eagerly on the host,
// and a new segment is begun; replay() runs every segment's ops in order.
class GraphCapture {
 public:
  GraphCapture() = default;
  ~GraphCapture() = default;
  GraphCapture(const GraphCapture&) = delete;
  GraphCapture& operator=(const GraphCapture&) = delete;

  // Begin a new capture segment (mirrors cudaStreamBeginCapture). Throws
  // std::logic_error if already capturing.
  void begin();

  // End the current capture segment and instantiate it for replay
  // (mirrors cudaStreamEndCapture + cudaGraphInstantiate). Throws
  // std::logic_error if not capturing.
  void end();

  // True between a successful begin() and the matching end().
  bool in_capture() const { return in_capture_; }

  // Total number of recorded ops across all completed segments. A
  // graph-capturable PP boundary is ONE op per plan; a host-staged
  // boundary records ZERO ops (it runs eagerly outside the segment).
  std::size_t num_nodes() const { return ops_.size(); }

  // Number of completed capture segments (one before the eager-break
  // path becomes necessary, more once a boundary splits the work).
  std::size_t num_segments() const { return segments_; }

  // Number of times replay() has been called (the host-progress-free
  // replay count the acceptance asserts grows across decode iterations).
  std::size_t replays() const { return replays_; }

  // Submit one op to the current segment. Recorded (not run) while
  // in_capture(); run immediately otherwise (stream semantics). Outside a
  // capture region this is the non-graph "eager" execution path.
  void submit(std::function<void()> op);

  // Replay every recorded op, once, in capture order (cudaGraphLaunch).
  // No new submits, no host-side ring I/O. Throws std::logic_error if
  // called while in_capture() (a graph must be instantiated before it
  // can be launched).
  void replay();

 private:
  bool in_capture_ = false;
  std::size_t segments_ = 0;
  std::size_t replays_ = 0;
  std::vector<std::function<void()>> ops_;
};

// Direction of one rank's participation in a directed PP boundary. A
// boundary between stage i and stage i+1 has TWO plans: stage i's send
// and stage i+1's recv. A round trip is two boundaries (forward + back).
enum class BoundaryDirection { kSend = 0, kRecv = 1 };

// Streamable for the EXPECT_EQ machinery and logging (skipped by the
// comm discovery contract, like operator<< for PipelineTransport above).
std::ostream& operator<<(std::ostream& os, BoundaryDirection d);

// ---------------------------------------------------------------------------
// PipelineBoundaryPlan — one rank's directed boundary transfer
// ---------------------------------------------------------------------------
//
// A prepared directed transfer across ONE PP boundary, bound to one
// (rank, world, payload size, transport, direction) but NOT to a buffer:
// `my_buf` is supplied at every execute() so a single plan can move one
// boundary's hidden state every decode iteration (the K3 microbatch
// pattern). The plan performs ONE operation per execute() (one graph node
// on the device path; one eager host send/recv on the eager-break path),
// exactly as RcclAllreducePlan enqueues one ncclAllReduce.
//
// Two execution modes, selected once from the boundary's transport:
//
// * Device path (is_graph_capturable(transport) == true — same-node peer,
//   or cross-node NCCL/RCCL with graph support): execute() records ONE
//   device copy into `graph` when `graph != nullptr && graph->in_capture()`
//   — send copies `my_buf -> peer_buf`; recv copies `peer_buf -> my_buf`
//   — and runs it immediately otherwise. On a real GPU this is a
//   `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice, stream)` for the
//   peer, or an `ncclSend` / `ncclRecv` (captured by the caller between
//   cudaStreamBeginCapture / cudaStreamEndCapture) for the cross-node
//   path. NO channel is touched, so a captured segment replays with no
//   host progress.
//
// * Eager-break path (transport == kHostStaged — gloo / host-bounce):
//   when `graph != nullptr && graph->in_capture()`, execute() ENDS the
//   current capture segment, runs the transfer eagerly over `next` (send)
//   or `prev` (recv) — host progress, a Channel send/recv — then BEGINS
//   the next segment. The boundary is therefore OUTSIDE the captured
//   graph: replay() replays only the device compute segments, and the
//   host re-runs the boundary between launches (the vLLM
//   `eager_break_during_capture` contract). When not capturing, execute()
//   runs the transfer eagerly (the non-graph fallback).
//
// Lifetime: `my_buf` and (for the device path) `peer_buf` must outlive
// every stream the plan is executed on. For the eager-break path the
// caller's `next` / `prev` channels must outlive the plan's stream.
class PipelineBoundaryPlan {
 public:
  // Validate once: `world > 0`, `rank` in [0, world), `payload_elems > 0`,
  // `transport` a known transport, `dir` a known direction. For the
  // device path (`is_graph_capturable(transport)`) `peer_buf` MUST be
  // non-null (the copy needs it); for the eager-break path `peer_buf` is
  // unused and may be null. Throws std::invalid_argument on violation.
  PipelineBoundaryPlan(int world, int rank, std::size_t payload_elems,
                       PipelineTransport transport, BoundaryDirection dir,
                       void* peer_buf = nullptr);

  PipelineBoundaryPlan(const PipelineBoundaryPlan&) = delete;
  PipelineBoundaryPlan& operator=(const PipelineBoundaryPlan&) = delete;

  int world() const { return world_; }
  int rank() const { return rank_; }
  std::size_t payload_elems() const { return payload_elems_; }
  std::size_t payload_bytes() const { return payload_elems_ * sizeof(float); }
  PipelineTransport transport() const { return transport_; }
  BoundaryDirection direction() const { return dir_; }
  bool is_send() const { return dir_ == BoundaryDirection::kSend; }
  void* peer_buf() const { return peer_buf_; }

  // Convenience wrapper over is_graph_capturable(transport()).
  bool is_graph_capturable() const { return vkernels::comm::is_graph_capturable(transport_); }

  // Execute one directed boundary transfer (see the class comment for the
  // device vs eager-break contract). `my_buf` is this rank's buffer of
  // `payload_elems()` floats (send: source; recv: destination). `graph`,
  // when non-null, is the capture to record into / break around. For the
  // eager-break path `next` (send) or `prev` (recv) MUST be non-null; for
  // the device path they are unused (may be null). A null `stream` is
  // accepted by the host reference (it runs synchronously); the device
  // path mirrors this with a cudaStream_t (see pipeline_boundary_cuda.hpp).
  void execute(float* my_buf, GraphCapture* graph = nullptr,
               Channel* next = nullptr, Channel* prev = nullptr) const;

 private:
  int world_;
  int rank_;
  std::size_t payload_elems_;
  PipelineTransport transport_;
  BoundaryDirection dir_;
  void* peer_buf_;
};

}  // namespace vkernels::comm
