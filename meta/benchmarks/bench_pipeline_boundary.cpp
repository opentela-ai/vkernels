// meta/benchmarks/bench_pipeline_boundary.cpp
//
// Host benchmark for the graph-capturable PP-boundary transfer (issue #10).
// The host reference (pipeline_boundary.cpp) is the always-compiled,
// 100%-line-covered correctness oracle; this bench is its always-runnable
// performance analog — exactly as bench_rccl.cpp is the CI-verifiable
// surface for issue #19's HIP/RCCL path. No GPU, no NCCL: the numbers here
// are HOST-SIDE CPU coordination overhead, NOT device throughput.
//
// Two things matter for a graph-capturable boundary at decode time:
//
//   1. ONE-TIME planning cost — classify_boundary + PipelineBoundaryPlan
//      construction. Paid once per boundary; must be negligible next to a
//      layer's matmuls.
//   2. PER-DECODE-ITERATION host coordination cost — what the CPU spends
//      every iteration to keep the transfer in the graph. The device path
//      records ONE node (a pointer+size closure) and touches neither the
//      bytes nor the ring Channel, so this cost is O(1) in payload. The
//      eager-break path (host-staged) runs the transfer over the Channel
//      between segments, so its host cost scales with payload (it moves
//      the bytes on the CPU). The contrast across a payload sweep is the
//      design point: the device path keeps the host out of the critical
//      path regardless of hidden size.
//
//   ./bench_pipeline_boundary [--payloads 1024,8192,32768,131072]
//                             [--iters 1000] [--trials 20]
//
// The actual device-side win the issue cares about — recovering the
// ~1.5-2x decode throughput that `--enforce-eager` costs on beverin
// (6x MI300A, TP=8 x PP=3) by holding the boundary transfer in a
// replayed graph — is A/B'd on that multi-node setup with the same
// transport classification this bench exercises.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/pipeline_boundary.hpp"

using vkernels::comm::BoundaryDirection;
using vkernels::comm::Channel;
using vkernels::comm::GraphCapture;
using vkernels::comm::PipelineBoundaryConfig;
using vkernels::comm::PipelineBoundaryPlan;
using vkernels::comm::PipelineTransport;
using vkernels::comm::BlockingQueue;

namespace {

std::vector<int> parse_ints(const char* s) {
  std::vector<int> out;
  std::string t = s ? s : "";
  for (char& c : t)
    if (c == ',') c = ' ';
  std::istringstream is(t);
  int v;
  while (is >> v) out.push_back(v);
  return out;
}

double now_ns() {
  using namespace std::chrono;
  return duration<double, std::nano>(steady_clock::now().time_since_epoch()).count();
}

// One rank's directed boundary channel that sends into `out` and receives
// from `in` (two BlockingQueues forming a directed link, like
// make_ring_channels(2)).
class BenchChannel : public Channel {
 public:
  BenchChannel(std::shared_ptr<BlockingQueue> out,
               std::shared_ptr<BlockingQueue> in)
      : out_(std::move(out)), in_(std::move(in)) {}
  void send(std::vector<float> chunk) override { out_->push(std::move(chunk)); }
  std::vector<float> recv() override { return in_->pop(); }
  bool closed() const override { return in_->closed(); }

 private:
  std::shared_ptr<BlockingQueue> out_;
  std::shared_ptr<BlockingQueue> in_;
};

std::pair<std::unique_ptr<Channel>, std::unique_ptr<Channel>> make_link() {
  auto q_ab = std::make_shared<BlockingQueue>();  // a -> b
  auto q_ba = std::make_shared<BlockingQueue>();  // b -> a
  return {std::make_unique<BenchChannel>(q_ab, q_ba),
          std::make_unique<BenchChannel>(q_ba, q_ab)};
}

// Device path: execute() while capturing records ONE node (no copy runs,
// no Channel touched). Returns min ns/iter over `iters` (warmup 20%).
double bench_device_capture(std::size_t n, int iters) {
  std::vector<float> my(n), peer(n);
  PipelineBoundaryPlan plan(2, 0, n, PipelineTransport::kSameNodePeer,
                            BoundaryDirection::kSend, peer.data());
  const int warmup = iters / 5;
  double best = 1e30;
  for (int t = 0; t < iters; ++t) {
    GraphCapture g;
    g.begin();
    double t0 = now_ns();
    plan.execute(my.data(), &g);
    double dt = now_ns() - t0;
    g.end();
    if (t >= warmup) best = std::min(best, dt);
  }
  return best;
}

// Eager-break path: two ranks run a directed boundary (rank 0 sends,
// rank 1 recvs) concurrently over a Channel, each doing
// end()/run()/begin() per iteration (the vLLM eager_break_during_capture
// contract — the host transfer is OUTSIDE the captured segment). Returns
// min ns/iter over `trials` (warmup 20%), each trial running `iters`
// round trips.
double bench_eager_break(std::size_t n, int iters, int trials) {
  std::vector<float> buf0(n, 0.0f), buf1(n, 0.0f);
  auto [a, b] = make_link();  // a: rank0 next | b: rank1 prev
  PipelineBoundaryPlan send(2, 0, n, PipelineTransport::kHostStaged,
                            BoundaryDirection::kSend, nullptr);
  PipelineBoundaryPlan recv(2, 1, n, PipelineTransport::kHostStaged,
                            BoundaryDirection::kRecv, nullptr);
  const int tw = trials / 5;
  double best = 1e30;
  for (int tr = 0; tr < trials; ++tr) {
    double t0 = now_ns();
    std::thread ts([&] {
      GraphCapture g;
      g.begin();
      for (int i = 0; i < iters; ++i) send.execute(buf0.data(), &g, a.get(), nullptr);
      g.end();
    });
    std::thread tr1([&] {
      GraphCapture g;
      g.begin();
      for (int i = 0; i < iters; ++i) recv.execute(buf1.data(), &g, nullptr, b.get());
      g.end();
    });
    ts.join();
    tr1.join();
    const double per = (now_ns() - t0) / static_cast<double>(iters);
    if (tr >= tw) best = std::min(best, per);
  }
  return best;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> payloads = {1024, 8192, 32768, 131072};
  int iters = 1000;
  int trials = 20;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--payloads" && i + 1 < argc) payloads = parse_ints(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--trials" && i + 1 < argc) trials = std::atoi(argv[++i]);
  }
  if (iters < 1) iters = 1;
  if (trials < 1) trials = 1;

  std::printf("# vkernels comm: graph-capturable PP-boundary (issue #10)\n");
  std::printf("#   HOST-SIDE CPU coordination overhead (no GPU, no NCCL).\n");
  std::printf("#   device path = record ONE graph node/iter (no copy, no Channel);\n");
  std::printf("#   eager-break = Channel send/recv between segments (moves bytes).\n");
  std::printf("#   The device-side decode-throughput win (the ~1.5-2x that\n");
  std::printf("#   --enforce-eager costs on beverin 6x MI300A TP=8 x PP=3) is\n");
  std::printf("#   A/B'd on that multi-node setup with the same classification.\n\n");

  // --- Table 1: one-time planning overhead (min over many calls) ---
  constexpr int kPlanIters = 100000;
  {
    PipelineBoundaryConfig cfg;
    cfg.same_node = true;
    double best = 1e30;
    for (int i = 0; i < kPlanIters; ++i) {
      double t0 = now_ns();
      volatile auto t = classify_boundary(cfg);
      (void)t;
      best = std::min(best, now_ns() - t0);
    }
    std::printf("[planning] classify_boundary         : %7.1f ns/call\n", best);
  }
  {
    const std::size_t n = 32768;
    std::vector<float> peer(n);
    double best = 1e30;
    for (int i = 0; i < kPlanIters; ++i) {
      double t0 = now_ns();
      PipelineBoundaryPlan plan(2, 0, n, PipelineTransport::kSameNodePeer,
                                BoundaryDirection::kSend, peer.data());
      best = std::min(best, now_ns() - t0);
    }
    std::printf("[planning] construct device plan      : %7.1f ns (one-time)\n", best);
  }
  {
    const std::size_t n = 32768;
    double best = 1e30;
    for (int i = 0; i < kPlanIters; ++i) {
      double t0 = now_ns();
      PipelineBoundaryPlan plan(2, 0, n, PipelineTransport::kHostStaged,
                                BoundaryDirection::kSend, nullptr);
      best = std::min(best, now_ns() - t0);
    }
    std::printf("[planning] construct eager-break plan : %7.1f ns (one-time)\n\n", best);
  }

  // --- Table 2: per-iteration host coordination overhead ---
  std::printf("[per-iter]  payload    device       eager-break     ratio\n");
  std::printf("[per-iter]   elems   capture ns   round-trip ns   eager/dev\n");
  for (int p : payloads) {
    if (p < 1) continue;
    const std::size_t n = static_cast<std::size_t>(p);
    const double dev = bench_device_capture(n, iters);
    const double eager = bench_eager_break(n, iters, trials);
    const double ratio = (dev > 0.0) ? eager / dev : 0.0;
    std::printf("[per-iter] %8zu %11.1f %15.1f %10.1f\n", n, dev, eager, ratio);
  }
  std::printf("\n# device-capture is ~flat in payload (records a pointer+size);\n");
  std::printf("# eager-break grows with payload (host moves the bytes). The\n");
  std::printf("# growing ratio is the host CPU the device path avoids every\n");
  std::printf("# decode iteration — the property that makes the boundary\n");
  std::printf("# graph-capturable and removes the host from the critical path.\n");
  return 0;
}
