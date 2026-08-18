// vkernels/comm/pipeline_boundary.cpp — host reference for the
// graph-capturable PP-boundary transfer (issue #10).
//
// The CPU reference is the correctness oracle for the PP-boundary
// primitive: the transport classification, the eager-break decision, a
// host model of CUDA graph capture / instantiate / replay, and the
// boundary plan over the existing ring Channel. It is always compiled
// and fully unit-tested on a machine with no GPU; the CUDA path
// (pipeline_boundary.cu) performs the real peer copy / ncclSend /
// ncclRecv on a cudaStream_t and is compiled only with a toolkit.
#include "vkernels/comm/pipeline_boundary.hpp"

#include <cstring>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::comm {

// ---------------------------------------------------------------------------
// Transport classification
// ---------------------------------------------------------------------------

PipelineTransport classify_boundary(const PipelineBoundaryConfig& cfg) {
  // Same-node peer (NVLink / HBM / ROCm IPC): a device copy, always
  // graph-capturable — the fast path the issue wants captured.
  if (cfg.same_node) return PipelineTransport::kSameNodePeer;
  // Gloo is wired for the boundary (the vLLM / sglang default): host-staged
  // and NOT graph-capturable, even when NCCL graphs are available — gloo's
  // host recv_object / send_object is exactly what the captured graph
  // cannot service and what the PP=3 deadlock hangs on. Gloo takes
  // precedence so a misconfigured runtime never silently captures it.
  if (cfg.gloo_fallback) return PipelineTransport::kHostStaged;
  // Cross-node NCCL / RCCL whose graph-capture API is supported: ncclSend /
  // ncclRecv captured into the graph segment, replayed with no host
  // progress (the second graph-capturable path).
  if (cfg.nccl_graph_supported) return PipelineTransport::kCrossNodeNccl;
  // Cross-node without a graph-capturable collective (NCCL present but its
  // graph API unavailable, or no fabric): host-bounce, same treatment as
  // gloo — eager-break so the boundary is never frozen inside the graph.
  return PipelineTransport::kHostStaged;
}

bool is_graph_capturable(PipelineTransport t) {
  return t == PipelineTransport::kSameNodePeer ||
         t == PipelineTransport::kCrossNodeNccl;
}

bool eager_break_during_capture(const PipelineBoundaryConfig& cfg) {
  return !is_graph_capturable(classify_boundary(cfg));
}

std::ostream& operator<<(std::ostream& os, PipelineTransport t) {
  return os << pipeline_transport_name(t);
}

std::ostream& operator<<(std::ostream& os, BoundaryDirection d) {
  switch (d) {
    case BoundaryDirection::kSend: return os << "send";
    case BoundaryDirection::kRecv: return os << "recv";
  }
  return os << "?";  // LCOV_EXCL_LINE (exhaustive switch)
}

// ---------------------------------------------------------------------------
// GraphCapture
// ---------------------------------------------------------------------------

void GraphCapture::begin() {
  if (in_capture_)
    throw std::logic_error("GraphCapture::begin: already capturing");
  in_capture_ = true;
}

void GraphCapture::end() {
  if (!in_capture_)
    throw std::logic_error("GraphCapture::end: not capturing");
  in_capture_ = false;
  ++segments_;
}

void GraphCapture::submit(std::function<void()> op) {
  if (in_capture_) {
    // Recorded: runs on replay() with no new host enqueues (the graph
    // contract). The op is captured by value so it is safe to run later.
    ops_.push_back(std::move(op));
    return;
  }
  // Outside a capture region: stream semantics — run now, exactly like an
  // uncaptured kernel launch.
  op();
}

void GraphCapture::replay() {
  if (in_capture_)
    throw std::logic_error("GraphCapture::replay: still capturing");
  for (auto& op : ops_) op();
  ++replays_;
}

// ---------------------------------------------------------------------------
// PipelineBoundaryPlan
// ---------------------------------------------------------------------------

PipelineBoundaryPlan::PipelineBoundaryPlan(int world, int rank,
                                           std::size_t payload_elems,
                                           PipelineTransport transport,
                                           BoundaryDirection dir,
                                           void* peer_buf)
    : world_(world),
      rank_(rank),
      payload_elems_(payload_elems),
      transport_(transport),
      dir_(dir),
      peer_buf_(peer_buf) {
  VK_EXPECTS(world > 0, "world must be positive");
  VK_EXPECTS(rank >= 0 && rank < world, "rank out of range");
  VK_EXPECTS(payload_elems > 0, "payload_elems must be positive");
  VK_EXPECTS(transport == PipelineTransport::kSameNodePeer ||
                 transport == PipelineTransport::kCrossNodeNccl ||
                 transport == PipelineTransport::kHostStaged,
             "unknown pipeline transport");
  VK_EXPECTS(dir == BoundaryDirection::kSend || dir == BoundaryDirection::kRecv,
             "unknown boundary direction");
  // The device path (same-node peer or graph-capturable cross-node) copies
  // through `peer_buf`, so it must be non-null. The eager-break path runs
  // over a Channel and never touches `peer_buf`.
  if (is_graph_capturable())
    VK_EXPECTS(peer_buf != nullptr, "device-path boundary needs a peer buffer");
}

void PipelineBoundaryPlan::execute(float* my_buf, GraphCapture* graph,
                                   Channel* next, Channel* prev) const {
  VK_EXPECTS(my_buf != nullptr, "my_buf must be non-null");
  const std::size_t n = payload_elems_;

  if (is_graph_capturable()) {
    // ---- Device path: ONE device copy, no host progress ------------
    // On a real GPU this is a peer `cudaMemcpyAsync(..., D2D, stream)`
    // (same-node) or an `ncclSend`/`ncclRecv` captured by the caller
    // (cross-node). The host model is a memcpy between `my_buf` and the
    // shared `peer_buf` (models peer UVA / the NCCL wire); it never
    // touches the ring Channel, so a captured segment replays with no
    // host participation. `peer_buf` is a constructor invariant
    // (validated there for the device path).
    void* peer = peer_buf_;
    auto copy = [my_buf, peer, n, send = is_send()]() {
      if (send)
        std::memcpy(peer, my_buf, n * sizeof(float));
      else
        std::memcpy(my_buf, peer, n * sizeof(float));
    };
    // submit() records the copy while the caller is capturing (the graph
    // replays it with no host progress) and runs it immediately otherwise
    // (stream semantics). No code path touches the ring Channel.
    if (graph != nullptr)
      graph->submit(std::move(copy));
    else
      copy();
    return;
  }

  // ---- Eager-break path (host-staged): boundary OUTSIDE the graph ----
  // When capturing, END the current segment, run the transfer eagerly over
  // the Channel (host progress), then BEGIN the next segment — exactly
  // vLLM `eager_break_during_capture`: the host send/recv is excluded from
  // the captured graph so a replay is never asked to service it (the
  // deadlock fix). When not capturing, run the transfer eagerly.
  if (is_send()) {
    VK_EXPECTS(next != nullptr, "send boundary needs a next channel");
    auto run = [my_buf, n, next]() {
      std::vector<float> chunk(my_buf, my_buf + n);
      next->send(std::move(chunk));
    };
    if (graph != nullptr && graph->in_capture()) {
      graph->end();
      run();
      graph->begin();
    } else {
      run();
    }
    return;
  }
  // kRecv
  VK_EXPECTS(prev != nullptr, "recv boundary needs a prev channel");
  auto run = [my_buf, n, prev]() {
    std::vector<float> got = prev->recv();
    // The peer boundary sends exactly `n` floats; copy them back. A
    // smaller chunk is a caller bug (mismatched payload_elems across the
    // two ranks) — surfaced here rather than read past the end.
    VK_EXPECTS(got.size() >= n,
               "pipeline boundary recv: channel chunk smaller than payload");
    std::memcpy(my_buf, got.data(), n * sizeof(float));
  };
  if (graph != nullptr && graph->in_capture()) {
    graph->end();
    run();
    graph->begin();
  } else {
    run();
  }
}

}  // namespace vkernels::comm
