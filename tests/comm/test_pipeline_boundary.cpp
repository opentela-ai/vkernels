// tests/comm/test_pipeline_boundary.cpp
//
// Host-reference tests for the graph-capturable PP-boundary transfer
// (issue #10). The CPU reference is the correctness oracle: these tests pin
// the transport classification, the eager-break decision, the GraphCapture
// model of CUDA graph capture / instantiate / replay, and the boundary plan
// over both the device path (same-node peer, cross-node NCCL — recorded
// into the graph, replayed with no host progress) and the eager-break path
// (host-staged — excluded from the graph, run eagerly between launches).
//
// The two acceptance criteria from the issue:
//   #1  A rank-pair PP token round trip can be captured and replayed N
//       times with no host progress — for the same-node peer AND the
//       cross-node NCCL path — and the host-staged path takes the
//       explicit eager-break path where it is not graph-capturable.
//   #2  A PP>1 decode holds a graph segment across the boundary without
//       deadlocking.
#include "minitest.hpp"

#include <atomic>
#include <cstddef>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/pipeline_boundary.hpp"

using vkernels::comm::BoundaryDirection;
using vkernels::comm::Channel;
using vkernels::comm::classify_boundary;
using vkernels::comm::eager_break_during_capture;
using vkernels::comm::GraphCapture;
using vkernels::comm::is_graph_capturable;
using vkernels::comm::MockChannel;
using vkernels::comm::pipeline_transport_name;
using vkernels::comm::PipelineBoundaryConfig;
using vkernels::comm::PipelineBoundaryPlan;
using vkernels::comm::PipelineTransport;
using vkernels::comm::BlockingQueue;

namespace {

// A Channel that counts every send / recv so a test can prove the device
// path touches no host ring I/O across a replay (the no-host-progress
// contract) while the eager-break path advances the counter only between
// launches. Wraps two BlockingQueues exactly like MockChannel.
class CountingChannel : public Channel {
 public:
  CountingChannel(std::shared_ptr<BlockingQueue> out,
                  std::shared_ptr<BlockingQueue> in)
      : out_(std::move(out)), in_(std::move(in)) {
      VK_EXPECTS(out_ != nullptr && in_ != nullptr, "CountingChannel needs both queues");
  }

  void send(std::vector<float> chunk) override {
    ++sends_;
    out_->push(std::move(chunk));
  }
  std::vector<float> recv() override {
    ++recvs_;
    return in_->pop();
  }
  bool closed() const override { return in_->closed(); }

  long sends() const { return sends_.load(); }
  long recvs() const { return recvs_.load(); }
  long total() const { return sends() + recvs(); }

 private:
  std::shared_ptr<BlockingQueue> out_;
  std::shared_ptr<BlockingQueue> in_;
  std::atomic<long> sends_{0};
  std::atomic<long> recvs_{0};
};

// Build a directed, in-process link: `a` sends into the queue `b` receives
// from, and `b` sends into the queue `a` receives from (two independent
// directions, like make_ring_channels(2) but with counting).
std::pair<std::unique_ptr<CountingChannel>, std::unique_ptr<CountingChannel>>
make_counting_link() {
  auto q_ab = std::make_shared<BlockingQueue>();  // a -> b
  auto q_ba = std::make_shared<BlockingQueue>();  // b -> a
  auto a = std::make_unique<CountingChannel>(q_ab, q_ba);
  auto b = std::make_unique<CountingChannel>(q_ba, q_ab);
  return {std::move(a), std::move(b)};
}

}  // namespace

// ---------------------------------------------------------------------------
// Transport classification + helpers (the planning surface)
// ---------------------------------------------------------------------------

TEST(PipelineBoundaryTransport, ClassifiesEveryPath) {
  // Same-node peer (NVLink / HBM / ROCm IPC): device copy, always
  // graph-capturable — the fast path the issue wants captured.
  {
    PipelineBoundaryConfig cfg;
    cfg.same_node = true;
    EXPECT_EQ(classify_boundary(cfg), PipelineTransport::kSameNodePeer);
  }
  // Gloo wired for the boundary (the vLLM / sglang default): host-staged,
  // NOT graph-capturable, even when NCCL graphs are available. This is the
  // branch that hangs the captured PP=3 decode and that the eager-break
  // path must exclude.
  {
    PipelineBoundaryConfig cfg;
    cfg.same_node = false;
    cfg.gloo_fallback = true;
    cfg.nccl_graph_supported = true;  // gloo still wins -> host-staged
    EXPECT_EQ(classify_boundary(cfg), PipelineTransport::kHostStaged);
  }
  // Cross-node NCCL / RCCL whose graph-capture API is supported: ncclSend /
  // ncclRecv captured into the segment (the second graph-capturable path).
  {
    PipelineBoundaryConfig cfg;
    cfg.same_node = false;
    cfg.nccl_graph_supported = true;
    EXPECT_EQ(classify_boundary(cfg), PipelineTransport::kCrossNodeNccl);
  }
  // Cross-node without a graph-capturable collective (NCCL present but its
  // graph API unavailable, or no fabric): host-bounce, eager-break.
  {
    PipelineBoundaryConfig cfg;  // all defaults false
    EXPECT_EQ(classify_boundary(cfg), PipelineTransport::kHostStaged);
  }
}

TEST(PipelineBoundaryTransport, IsGraphCapturable) {
  EXPECT_TRUE(is_graph_capturable(PipelineTransport::kSameNodePeer));
  EXPECT_TRUE(is_graph_capturable(PipelineTransport::kCrossNodeNccl));
  EXPECT_FALSE(is_graph_capturable(PipelineTransport::kHostStaged));
}

TEST(PipelineBoundaryTransport, EagerBreakDecision) {
  // Host-staged boundaries (gloo, or no graph-capturable cross-node fabric)
  // MUST eager-break: the host send/recv cannot be serviced inside a
  // replaying graph.
  {
    PipelineBoundaryConfig cfg;
    cfg.gloo_fallback = true;
    EXPECT_TRUE(eager_break_during_capture(cfg));
  }
  {
    PipelineBoundaryConfig cfg;  // no fabric, no gloo, cross-node
    EXPECT_TRUE(eager_break_during_capture(cfg));
  }
  // Graph-capturable boundaries (same-node peer, NCCL-with-graph) never
  // eager-break — they replay as pure device work.
  {
    PipelineBoundaryConfig cfg;
    cfg.same_node = true;
    EXPECT_FALSE(eager_break_during_capture(cfg));
  }
  {
    PipelineBoundaryConfig cfg;
    cfg.nccl_graph_supported = true;
    EXPECT_FALSE(eager_break_during_capture(cfg));
  }
}

TEST(PipelineBoundaryTransport, NamesAndStreams) {
  EXPECT_EQ(std::string(pipeline_transport_name(PipelineTransport::kSameNodePeer)),
            "same-node-peer");
  EXPECT_EQ(std::string(pipeline_transport_name(PipelineTransport::kCrossNodeNccl)),
            "cross-node-nccl");
  EXPECT_EQ(std::string(pipeline_transport_name(PipelineTransport::kHostStaged)),
            "host-staged");

  for (auto t : {PipelineTransport::kSameNodePeer, PipelineTransport::kCrossNodeNccl,
                 PipelineTransport::kHostStaged}) {
    std::ostringstream os;
    os << t;
    EXPECT_EQ(os.str(), std::string(pipeline_transport_name(t)));
  }

  // BoundaryDirection is streamable too (the EXPECT_EQ machinery
  // instantiates to_string_val<BoundaryDirection> even on success).
  {
    std::ostringstream os;
    os << BoundaryDirection::kSend << BoundaryDirection::kRecv;
    EXPECT_EQ(os.str(), "sendrecv");
  }
}

// ---------------------------------------------------------------------------
// GraphCapture — host model of CUDA graph capture / instantiate / replay
// ---------------------------------------------------------------------------

TEST(GraphCapture, CapturesAndReplays) {
  GraphCapture g;
  EXPECT_FALSE(g.in_capture());
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(0));
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(0));
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(0));

  int ran = 0;
  g.begin();
  EXPECT_TRUE(g.in_capture());
  g.submit([&] { ++ran; });      // recorded, not run
  EXPECT_EQ(ran, 0);
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(1));
  g.end();
  EXPECT_FALSE(g.in_capture());
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(1));

  g.replay();                    // runs the recorded op once
  EXPECT_EQ(ran, 1);
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(1));
  g.replay();                    // runs it again, host-progress-free
  EXPECT_EQ(ran, 2);
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(2));
}

TEST(GraphCapture, SubmitsEagerlyOutsideCapture) {
  // Outside a capture region submit() has stream semantics: run now, record
  // nothing (the graph node count must stay 0).
  GraphCapture g;
  int ran = 0;
  g.submit([&] { ran += 7; });
  EXPECT_EQ(ran, 7);
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(0));
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(0));
}

TEST(GraphCapture, MultipleSegmentsReplayInOrder) {
  // The eager-break path ends one segment at a boundary and begins the
  // next; replay must run every segment's ops in capture order.
  GraphCapture g;
  std::string trace;
  g.begin();
  g.submit([&] { trace += "a"; });
  g.end();
  g.begin();
  g.submit([&] { trace += "b"; });
  g.submit([&] { trace += "c"; });
  g.end();
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(2));
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(3));
  g.replay();
  EXPECT_EQ(trace, "abc");
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(1));
}

TEST(GraphCapture, RejectsBadStateTransitions) {
  GraphCapture g;
  // begin() while already capturing.
  g.begin();
  EXPECT_THROW(g.begin(), std::logic_error);
  g.end();
  // end() while not capturing.
  EXPECT_THROW(g.end(), std::logic_error);
  // replay() while still capturing (must instantiate first).
  g.begin();
  EXPECT_THROW(g.replay(), std::logic_error);
  g.end();
}

// ---------------------------------------------------------------------------
// PipelineBoundaryPlan — construction + accessors
// ---------------------------------------------------------------------------

TEST(PipelineBoundaryPlan, ConstructsAndAccessors) {
  float peer[4] = {0, 0, 0, 0};
  // A valid same-node-peer SEND plan (device path needs a peer buffer).
  PipelineBoundaryPlan send(2, 0, 4, PipelineTransport::kSameNodePeer,
                            BoundaryDirection::kSend, peer);
  EXPECT_EQ(send.world(), 2);
  EXPECT_EQ(send.rank(), 0);
  EXPECT_EQ(send.payload_elems(), static_cast<std::size_t>(4));
  EXPECT_EQ(send.payload_bytes(), static_cast<std::size_t>(16));
  EXPECT_EQ(send.transport(), PipelineTransport::kSameNodePeer);
  EXPECT_EQ(send.direction(), BoundaryDirection::kSend);
  EXPECT_TRUE(send.is_send());
  EXPECT_EQ(send.peer_buf(), static_cast<void*>(peer));
  EXPECT_TRUE(send.is_graph_capturable());

  // A valid host-staged RECV plan (eager-break path; peer_buf unused/null).
  PipelineBoundaryPlan recv(2, 1, 4, PipelineTransport::kHostStaged,
                            BoundaryDirection::kRecv, nullptr);
  EXPECT_FALSE(recv.is_send());
  EXPECT_FALSE(recv.is_graph_capturable());
  EXPECT_EQ(recv.peer_buf(), nullptr);
}

TEST(PipelineBoundaryPlan, RejectsInvalidConstruction) {
  float peer[2] = {0, 0};
  EXPECT_THROW(PipelineBoundaryPlan(0, 0, 2, PipelineTransport::kSameNodePeer,
                                    BoundaryDirection::kSend, peer),
               std::invalid_argument);  // world <= 0
  EXPECT_THROW(PipelineBoundaryPlan(2, -1, 2, PipelineTransport::kSameNodePeer,
                                    BoundaryDirection::kSend, peer),
               std::invalid_argument);  // rank < 0
  EXPECT_THROW(PipelineBoundaryPlan(2, 2, 2, PipelineTransport::kSameNodePeer,
                                    BoundaryDirection::kSend, peer),
               std::invalid_argument);  // rank >= world
  EXPECT_THROW(PipelineBoundaryPlan(2, 0, 0, PipelineTransport::kSameNodePeer,
                                    BoundaryDirection::kSend, peer),
               std::invalid_argument);  // payload_elems <= 0
  EXPECT_THROW(PipelineBoundaryPlan(2, 0, 2, static_cast<PipelineTransport>(99),
                                    BoundaryDirection::kSend, peer),
               std::invalid_argument);  // unknown transport
  EXPECT_THROW(PipelineBoundaryPlan(2, 0, 2, PipelineTransport::kSameNodePeer,
                                    static_cast<BoundaryDirection>(99), peer),
               std::invalid_argument);  // unknown direction
  // Device path requires a non-null peer buffer.
  EXPECT_THROW(PipelineBoundaryPlan(2, 0, 2, PipelineTransport::kSameNodePeer,
                                    BoundaryDirection::kSend, nullptr),
               std::invalid_argument);
  // Eager-break path accepts a null peer buffer (it never copies through it).
  EXPECT_NO_THROW(PipelineBoundaryPlan(2, 0, 2, PipelineTransport::kHostStaged,
                                       BoundaryDirection::kSend, nullptr));
}

// ---------------------------------------------------------------------------
// PipelineBoundaryPlan — device path (recorded into the graph)
// ---------------------------------------------------------------------------

TEST(PipelineBoundaryPlan, DevicePathRecordsCopyIntoGraph) {
  // Same-node peer: the send copies my_buf -> peer, the recv copies peer ->
  // my_buf. Both are recorded (NOT run) while capturing; replay runs them
  // in order, moving the data without any host ring I/O.
  float peer[3] = {0, 0, 0};
  float send_buf[3] = {1.f, 2.f, 3.f};
  float recv_buf[3] = {0, 0, 0};

  PipelineBoundaryPlan send(2, 0, 3, PipelineTransport::kSameNodePeer,
                            BoundaryDirection::kSend, peer);
  PipelineBoundaryPlan recv(2, 1, 3, PipelineTransport::kSameNodePeer,
                            BoundaryDirection::kRecv, peer);

  GraphCapture g;
  g.begin();
  send.execute(send_buf, &g);   // recorded: peer = send_buf (deferred)
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(1));
  // Not yet run: peer is still zero.
  EXPECT_NEAR(peer[0], 0.f, 0.0);
  recv.execute(recv_buf, &g);   // recorded: recv_buf = peer (deferred)
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(2));
  EXPECT_NEAR(recv_buf[0], 0.f, 0.0);  // still zero before replay
  g.end();

  g.replay();                   // now both copies run, in order
  EXPECT_NEAR(peer[0], 1.f, 0.0);
  EXPECT_NEAR(peer[1], 2.f, 0.0);
  EXPECT_NEAR(peer[2], 3.f, 0.0);
  EXPECT_NEAR(recv_buf[0], 1.f, 0.0);
  EXPECT_NEAR(recv_buf[1], 2.f, 0.0);
  EXPECT_NEAR(recv_buf[2], 3.f, 0.0);
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(1));
}

TEST(PipelineBoundaryPlan, DevicePathRunsEagerlyWithoutGraph) {
  // With graph == nullptr the device copy runs immediately (stream
  // semantics outside a capture region), in both directions.
  float peer[3] = {9.f, 9.f, 9.f};
  float send_buf[3] = {1.f, 2.f, 3.f};
  float recv_buf[3] = {0, 0, 0};

  PipelineBoundaryPlan send(2, 0, 3, PipelineTransport::kCrossNodeNccl,
                            BoundaryDirection::kSend, peer);
  PipelineBoundaryPlan recv(2, 1, 3, PipelineTransport::kCrossNodeNccl,
                            BoundaryDirection::kRecv, peer);

  send.execute(send_buf, nullptr);  // immediate: peer = send_buf
  EXPECT_NEAR(peer[0], 1.f, 0.0);
  EXPECT_NEAR(peer[2], 3.f, 0.0);
  recv.execute(recv_buf, nullptr);  // immediate: recv_buf = peer
  EXPECT_NEAR(recv_buf[0], 1.f, 0.0);
  EXPECT_NEAR(recv_buf[2], 3.f, 0.0);
}

TEST(PipelineBoundaryPlan, ExecuteRejectsNullBuffer) {
  float peer[2] = {0, 0};
  PipelineBoundaryPlan send(2, 0, 2, PipelineTransport::kSameNodePeer,
                            BoundaryDirection::kSend, peer);
  EXPECT_THROW(send.execute(nullptr, nullptr), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// PipelineBoundaryPlan — eager-break path (host-staged, outside the graph)
// ---------------------------------------------------------------------------

TEST(PipelineBoundaryPlan, EagerBreakExcludesBoundaryFromGraph) {
  // Two ranks share a directed link a -> b. While capturing, execute()
  // ENDS the current segment, runs the host send/recv eagerly, and BEGINS
  // the next segment — so the boundary is never a graph node (only the
  // surrounding compute is). When not capturing, execute() runs eagerly.
  auto link = make_counting_link();
  auto& a = link.first;   // rank 0 sends here
  auto& b = link.second;  // rank 1 recvs here

  float send_buf[2] = {4.f, 5.f};
  float recv_buf[2] = {0, 0};
  PipelineBoundaryPlan send(2, 0, 2, PipelineTransport::kHostStaged,
                            BoundaryDirection::kSend);
  PipelineBoundaryPlan recv(2, 1, 2, PipelineTransport::kHostStaged,
                            BoundaryDirection::kRecv);

  // (1) While capturing: the boundary is excluded — end, run, begin.
  GraphCapture g;
  g.begin();
  g.submit([&] {});                  // a little "compute" before the boundary
  send.execute(send_buf, &g, a.get(), nullptr);  // end seg0, send, begin seg1
  // execute() re-opens a new segment (begin), so the stream is capturing
  // again on return — the boundary sits between the two closed segments.
  EXPECT_TRUE(g.in_capture());
  // Recv while the new segment is open: end seg1, recv, begin seg2.
  recv.execute(recv_buf, &g, nullptr, b.get());
  EXPECT_TRUE(g.in_capture());       // execute re-opened a segment
  g.end();
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(1));  // only the compute
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(3));
  EXPECT_EQ(a->sends(), 1);
  EXPECT_EQ(b->recvs(), 1);
  EXPECT_NEAR(recv_buf[0], 4.f, 0.0);
  EXPECT_NEAR(recv_buf[1], 5.f, 0.0);

  // (2) Not capturing (graph == nullptr): the transfer runs eagerly, no
  // graph involvement at all.
  float send_buf2[2] = {7.f, 8.f};
  float recv_buf2[2] = {0, 0};
  send.execute(send_buf2, nullptr, a.get(), nullptr);
  recv.execute(recv_buf2, nullptr, nullptr, b.get());
  EXPECT_NEAR(recv_buf2[0], 7.f, 0.0);
  EXPECT_NEAR(recv_buf2[1], 8.f, 0.0);
  EXPECT_EQ(a->sends(), 2);
  EXPECT_EQ(b->recvs(), 2);
}

TEST(PipelineBoundaryPlan, EagerBreakRejectsMissingChannel) {
  float buf[2] = {0, 0};
  PipelineBoundaryPlan send(2, 0, 2, PipelineTransport::kHostStaged,
                            BoundaryDirection::kSend);
  PipelineBoundaryPlan recv(2, 1, 2, PipelineTransport::kHostStaged,
                            BoundaryDirection::kRecv);
  // send needs a `next` channel; recv needs a `prev` channel.
  EXPECT_THROW(send.execute(buf, nullptr, nullptr, nullptr), std::invalid_argument);
  EXPECT_THROW(recv.execute(buf, nullptr, nullptr, nullptr), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Acceptance #1: rank-pair round trip, captured + replayed N times
// ---------------------------------------------------------------------------

namespace {

// A round trip across a graph-capturable boundary (same-node peer OR
// cross-node NCCL). Two ranks, each with a forward-hidden send buffer and
// a backward-hidden recv buffer, share two peer staging buffers (one per
// direction). All four copies are recorded into ONE graph in dependency
// order, then replayed N times with a fresh hidden state each iteration.
// The CountingChannels passed as next/prev are never touched — the device
// path makes no host progress on replay.
void round_trip_device_no_host_progress(PipelineTransport transport, int n) {
  ASSERT_TRUE(is_graph_capturable(transport));

  float peer_fwd[4] = {0, 0, 0, 0};  // rank0 -> rank1
  float peer_bwd[4] = {0, 0, 0, 0};  // rank1 -> rank0
  float r0_send[4], r1_recv[4], r1_send[4], r0_recv[4];

  // Four directed plans: rank0 sends/receives, rank1 receives/sends.
  PipelineBoundaryPlan r0_fwd(2, 0, 4, transport, BoundaryDirection::kSend, peer_fwd);
  PipelineBoundaryPlan r1_fwd(2, 1, 4, transport, BoundaryDirection::kRecv, peer_fwd);
  PipelineBoundaryPlan r1_bwd(2, 1, 4, transport, BoundaryDirection::kSend, peer_bwd);
  PipelineBoundaryPlan r0_bwd(2, 0, 4, transport, BoundaryDirection::kRecv, peer_bwd);

  // Host-progress probe: passed to every execute but NEVER touched by the
  // device path, proving no host ring I/O across the N replays.
  auto probe_a = make_counting_link();
  auto probe_b = make_counting_link();
  CountingChannel* probe_fwd = probe_a.first.get();   // would-be forward wire
  CountingChannel* probe_bwd = probe_b.first.get();   // would-be backward wire

  GraphCapture g;
  g.begin();
  r0_fwd.execute(r0_send, &g, probe_fwd, nullptr);   // send: next
  r1_fwd.execute(r1_recv, &g, nullptr, probe_fwd);   // recv: prev
  r1_bwd.execute(r1_send, &g, probe_bwd, nullptr);   // send: next
  r0_bwd.execute(r0_recv, &g, nullptr, probe_bwd);   // recv: prev
  g.end();
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(4));
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(1));

  for (int it = 0; it < n; ++it) {
    const float fwd_v = static_cast<float>(100 + it);
    const float bwd_v = static_cast<float>(-it - 1);
    for (int i = 0; i < 4; ++i) {
      r0_send[i] = fwd_v + i;
      r1_send[i] = bwd_v - i;
      r1_recv[i] = 0;
      r0_recv[i] = 0;
    }
    const long before = probe_fwd->total() + probe_bwd->total();
    g.replay();  // all four copies run as device work, in order
    EXPECT_EQ(probe_fwd->total() + probe_bwd->total(), before);  // no host progress
    // Forward hidden flowed rank0 -> rank1; backward flowed rank1 -> rank0.
    for (int i = 0; i < 4; ++i) {
      EXPECT_NEAR(r1_recv[i], fwd_v + i, 0.0);
      EXPECT_NEAR(r0_recv[i], bwd_v - i, 0.0);
    }
  }
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(n));
  EXPECT_EQ(probe_fwd->total(), 0);
  EXPECT_EQ(probe_bwd->total(), 0);
}

}  // namespace

TEST(PipelineBoundaryAcceptance, RoundTripSameNodePeerNoHostProgress) {
  round_trip_device_no_host_progress(PipelineTransport::kSameNodePeer, 5);
}

TEST(PipelineBoundaryAcceptance, RoundTripCrossNodeNcclNoHostProgress) {
  // Cross-node NCCL / RCCL whose graph API is supported is graph-capturable
  // (ncclSend / ncclRecv recorded by the caller). The host model represents
  // both device paths as the same peer copy; the transport classification
  // and the documented real-GPU mapping are what distinguish them.
  round_trip_device_no_host_progress(PipelineTransport::kCrossNodeNccl, 7);
}

TEST(PipelineBoundaryAcceptance, RoundTripEagerBreakHostStaged) {
  // Host-staged boundary (gloo recv_object / send_object): NOT
  // graph-capturable, so the framework must exclude the boundary from the
  // captured segment and run it eagerly between launches (the vLLM
  // `eager_break_during_capture` contract). The captured compute segments
  // replay with NO host progress; the host re-runs the boundary each
  // iteration (host progress) — and the whole thing completes without
  // deadlocking across N iterations.
  PipelineBoundaryConfig cfg;
  cfg.same_node = false;
  cfg.gloo_fallback = true;
  EXPECT_EQ(classify_boundary(cfg), PipelineTransport::kHostStaged);
  EXPECT_TRUE(eager_break_during_capture(cfg));

  auto fwd_link = make_counting_link();  // rank0.send -> rank1.recv
  auto bwd_link = make_counting_link();  // rank1.send -> rank0.recv
  CountingChannel* r0_next = fwd_link.first.get();   // rank0 sends forward
  CountingChannel* r1_prev = fwd_link.second.get();  // rank1 recvs forward
  CountingChannel* r1_next = bwd_link.first.get();   // rank1 sends backward
  CountingChannel* r0_prev = bwd_link.second.get();  // rank0 recvs backward

  float r0_hidden[4] = {0, 0, 0, 0};  // rank0's forward hidden (send)
  float r1_hidden[4] = {0, 0, 0, 0};  // rank1's backward hidden (send)
  float r1_got[4] = {0, 0, 0, 0};     // rank1 receives rank0's forward
  float r0_got[4] = {0, 0, 0, 0};     // rank0 receives rank1's backward

  // Each rank captures a tiny "compute" (a per-replay transform) in its own
  // segment. The boundary plans are host-staged and EXCLUDED from the
  // captured graph (they run eagerly outside it).
  PipelineBoundaryPlan r0_send(2, 0, 4, PipelineTransport::kHostStaged,
                               BoundaryDirection::kSend);
  PipelineBoundaryPlan r1_recv(2, 1, 4, PipelineTransport::kHostStaged,
                               BoundaryDirection::kRecv);
  PipelineBoundaryPlan r1_send(2, 1, 4, PipelineTransport::kHostStaged,
                               BoundaryDirection::kSend);
  PipelineBoundaryPlan r0_recv(2, 0, 4, PipelineTransport::kHostStaged,
                               BoundaryDirection::kRecv);

  // Capture once: rank0's compute segment, then the forward boundary
  // (eager: end, send, begin), then rank1's compute segment, then the
  // forward recv (eager: end, recv, begin), then the backward boundary
  // (send + recv, eager). The captured graph ends up with TWO compute
  // nodes and ZERO boundary nodes — exactly "the boundary is excluded".
  GraphCapture g;
  g.begin();
  g.submit([&] { r0_hidden[0] += 10.f; });          // rank0 compute (node 0)
  r0_send.execute(r0_hidden, &g, r0_next, nullptr);  // end seg0, fwd.send, begin seg1
  g.submit([&] { r1_hidden[0] += 100.f; });         // rank1 compute (node 1)
  r1_recv.execute(r1_got, &g, nullptr, r1_prev);     // end seg1, fwd.recv, begin seg2
  r1_send.execute(r1_hidden, &g, r1_next, nullptr);  // end seg2, bwd.send, begin seg3
  r0_recv.execute(r0_got, &g, nullptr, r0_prev);     // end seg3, bwd.recv, begin seg4
  g.end();                                            // close seg4
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(2));   // only compute
  EXPECT_EQ(g.num_segments(), static_cast<std::size_t>(5));
  EXPECT_EQ(r0_next->sends(), 1);
  EXPECT_EQ(r1_prev->recvs(), 1);
  EXPECT_EQ(r1_next->sends(), 1);
  EXPECT_EQ(r0_prev->recvs(), 1);

  // Total host progress across all four boundary channels (a send lands on
  // one end of a link, the matching recv on the other; both must count).
  auto total_all = [&] {
    return r0_next->total() + r1_prev->total() +
           r1_next->total() + r0_prev->total();
  };
  EXPECT_EQ(total_all(), 4);  // every boundary ran once during capture

  const int n = 6;
  for (int it = 0; it < n; ++it) {
    const long before = total_all();

    // (a) Replay the captured compute segments with NO host progress: the
    //     boundary is excluded, so the channels are untouched here.
    g.replay();
    EXPECT_EQ(total_all(), before);  // no host progress on replay

    // (b) The host re-runs the boundary transfers eagerly between launches
    //     (execute() with a non-capturing graph runs immediately).
    r0_send.execute(r0_hidden, &g, r0_next, nullptr);   // forward: rank0 -> rank1
    r1_recv.execute(r1_got, &g, nullptr, r1_prev);
    r1_send.execute(r1_hidden, &g, r1_next, nullptr);   // backward: rank1 -> rank0
    r0_recv.execute(r0_got, &g, nullptr, r0_prev);
    EXPECT_EQ(total_all(), before + 4);  // host progress between launches

    // Data flowed across both boundaries this iteration.
    EXPECT_NEAR(r1_got[0], r0_hidden[0], 0.0);
    EXPECT_NEAR(r0_got[0], r1_hidden[0], 0.0);
  }
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(n));
  // r0_hidden += 10 each replay (capture did not run compute), r1_hidden +=100.
  EXPECT_NEAR(r0_hidden[0], 10.f * n, 0.0);
  EXPECT_NEAR(r1_hidden[0], 100.f * n, 0.0);
}

// ---------------------------------------------------------------------------
// Acceptance #2: PP>1 decode holds a graph segment across the boundary
// ---------------------------------------------------------------------------

namespace {

// A PP=3 forward pipeline (two boundaries: stage0 -> stage1 -> stage2).
// Per decode iteration the hidden state flows from stage 0 to stage 2.
// stage0 += 10, stage1 += 20, stage2 += 30 each iteration (traceable), so
// after N iterations stage2 holds 10*N + 20 + 30 = 10*N + 50.

}  // namespace

TEST(PipelineBoundaryAcceptance, PpDecodeDeviceOneGraph) {
  // Device path: ONE graph captures all three stages' compute AND both
  // boundaries. Replay moves the hidden state through all three stages in
  // one host-progress-free launch — no deadlock across N iterations.
  float peer01[4] = {0, 0, 0, 0};  // stage0 -> stage1
  float peer12[4] = {0, 0, 0, 0};  // stage1 -> stage2
  std::vector<float> s0(4, 0.f), s1(4, 0.f), s2(4, 0.f);

  PipelineBoundaryPlan p01_send(3, 0, 4, PipelineTransport::kSameNodePeer,
                                BoundaryDirection::kSend, peer01);
  PipelineBoundaryPlan p12_recv(3, 1, 4, PipelineTransport::kSameNodePeer,
                                BoundaryDirection::kRecv, peer01);
  PipelineBoundaryPlan p12_send(3, 1, 4, PipelineTransport::kSameNodePeer,
                                BoundaryDirection::kSend, peer12);
  PipelineBoundaryPlan p23_recv(3, 2, 4, PipelineTransport::kSameNodePeer,
                                BoundaryDirection::kRecv, peer12);

  GraphCapture g;
  g.begin();
  g.submit([&] { for (auto& x : s0) x += 10.f; });
  p01_send.execute(s0.data(), &g);          // stage0 -> peer01
  p12_recv.execute(s1.data(), &g);          // peer01 -> stage1
  g.submit([&] { for (auto& x : s1) x += 20.f; });
  p12_send.execute(s1.data(), &g);          // stage1 -> peer12
  p23_recv.execute(s2.data(), &g);          // peer12 -> stage2
  g.submit([&] { for (auto& x : s2) x += 30.f; });
  g.end();
  EXPECT_EQ(g.num_nodes(), static_cast<std::size_t>(7));  // 3 compute + 2 boundaries (send+recv)

  const int n = 5;
  for (int it = 0; it < n; ++it) {
    g.replay();  // entire pipeline, no host progress, no deadlock
    EXPECT_NEAR(s2[0], 10.f * (it + 1) + 50.f, 0.0);
    EXPECT_NEAR(s2[3], 10.f * (it + 1) + 50.f, 0.0);
  }
  EXPECT_EQ(g.replays(), static_cast<std::size_t>(n));
}

TEST(PipelineBoundaryAcceptance, PpDecodeEagerBreakHoldsSegmentAcrossBoundary) {
  // Eager-break path: each stage's compute is its own captured segment; the
  // two boundaries run eagerly BETWEEN segment launches (host progress,
  // excluded from the graphs). The captured segments replay with no host
  // progress, and the whole PP=3 decode completes without deadlock across
  // N iterations — "a graph segment held across the boundary".
  auto link01 = make_counting_link();  // stage0.send -> stage1.recv
  auto link12 = make_counting_link();  // stage1.send -> stage2.recv
  CountingChannel* s0_next = link01.first.get();
  CountingChannel* s1_prev = link01.second.get();
  CountingChannel* s1_next = link12.first.get();
  CountingChannel* s2_prev = link12.second.get();

  std::vector<float> s0(4, 0.f), s1(4, 0.f), s2(4, 0.f);

  PipelineBoundaryPlan p01_send(3, 0, 4, PipelineTransport::kHostStaged,
                                BoundaryDirection::kSend);
  PipelineBoundaryPlan p12_recv(3, 1, 4, PipelineTransport::kHostStaged,
                                BoundaryDirection::kRecv);
  PipelineBoundaryPlan p12_send(3, 1, 4, PipelineTransport::kHostStaged,
                                BoundaryDirection::kSend);
  PipelineBoundaryPlan p23_recv(3, 2, 4, PipelineTransport::kHostStaged,
                                BoundaryDirection::kRecv);

  // One captured segment per stage's compute (the boundary is excluded).
  GraphCapture g0, g1, g2;
  g0.begin(); g0.submit([&] { for (auto& x : s0) x += 10.f; }); g0.end();
  g1.begin(); g1.submit([&] { for (auto& x : s1) x += 20.f; }); g1.end();
  g2.begin(); g2.submit([&] { for (auto& x : s2) x += 30.f; }); g2.end();
  EXPECT_EQ(g0.num_nodes(), static_cast<std::size_t>(1));
  EXPECT_EQ(g1.num_nodes(), static_cast<std::size_t>(1));
  EXPECT_EQ(g2.num_nodes(), static_cast<std::size_t>(1));

  // Total host progress across all four boundary channels (a send lands on
  // one end of a link, the matching recv on the other).
  auto total_all = [&] {
    return s0_next->total() + s1_prev->total() +
           s1_next->total() + s2_prev->total();
  };

  const int n = 5;
  for (int it = 0; it < n; ++it) {
    const long before = total_all();

    // stage0 compute (captured, no host progress), then boundary0 -> stage1.
    const long after_g0 = total_all();
    g0.replay();
    EXPECT_EQ(total_all(), after_g0);  // no host progress during replay
    p01_send.execute(s0.data(), nullptr, s0_next, nullptr);
    p12_recv.execute(s1.data(), nullptr, nullptr, s1_prev);
    EXPECT_NEAR(s1[0], s0[0], 0.0);  // forward hidden flowed into stage1

    // stage1 compute (captured), then boundary1 -> stage2.
    const long after_g1 = total_all();
    g1.replay();
    EXPECT_EQ(total_all(), after_g1);  // no host progress during replay
    p12_send.execute(s1.data(), nullptr, s1_next, nullptr);
    p23_recv.execute(s2.data(), nullptr, nullptr, s2_prev);

    // stage2 compute (captured).
    const long after_g2 = total_all();
    g2.replay();
    EXPECT_EQ(total_all(), after_g2);  // no host progress during replay

    // Replay made no host progress; only the boundaries did.
    EXPECT_EQ(total_all(), before + 4);
    EXPECT_NEAR(s2[0], 10.f * (it + 1) + 50.f, 0.0);
    EXPECT_NEAR(s2[3], 10.f * (it + 1) + 50.f, 0.0);
  }
  EXPECT_EQ(g0.replays(), static_cast<std::size_t>(n));
  EXPECT_EQ(g1.replays(), static_cast<std::size_t>(n));
  EXPECT_EQ(g2.replays(), static_cast<std::size_t>(n));
}
