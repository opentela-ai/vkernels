// meta/benchmarks/bench_rccl.cpp
//
// Host benchmark for the HIP/RCCL transport selection (issue #19). The
// cost model in rccl.cpp (est_rccl_socket_us / est_rccl_ofi_us) predicts
// the Slingshot-vs-Socket speedup a cross-node TP all-reduce wins on
// beverin (gfx942); this bench sweeps the model across payload sizes and
// edge counts, then measures the host ring all-reduce (RcclAllreducePlan)
// over mock channels so the model and the reference agree on the same
// machine. No GPU, no RCCL, no libfabric — it is the acceptance surface
// the CI host job can actually run; the HIP/RCCL path (rccl.hip) is A/B'd
// on a Slingshot node with the same `--edges`/`--mib` flags.
//
//   ./bench_rccl [--edges 0,1,2,4] [--mib 1,4,16,48] [--world 8] [--iters 200]
//
// Prints two tables:
//   1. the cost model (Socket vs OFI microseconds, the chosen transport, and
//      the predicted speedup) — the acceptance criterion is OFI < Socket at
//      >= 1 MiB over >= 1 edge;
//   2. the measured host ring all-reduce over a mock ring, so a regression
//      that makes the reference slower than the model's prediction is
//      visible.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/rccl.hpp"

using vkernels::comm::RcclAllreducePlan;
using vkernels::comm::RcclReduceOp;
using vkernels::comm::RcclTransport;
using vkernels::comm::RcclTransportConfig;
using vkernels::comm::make_ring_channels;
using vkernels::comm::resolve_transport;

namespace {

std::vector<int> parse_ints(const char* s) {
  std::vector<int> out;
  std::string t = s ? s : "";
  for (char& c : t) if (c == ',') c = ' ';
  std::istringstream is(t);
  int v;
  while (is >> v) out.push_back(v);
  return out;
}

double now_us() {
  using namespace std::chrono;
  return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// Run the host ring all-reduce of `n` floats across `world` ranks once and
// return the elapsed time in microseconds. Uses the same mock-ring wiring
// as test_rccl.cpp so the reference and the test agree on a result.
double run_ring_once(int world, std::size_t n) {
  std::vector<std::vector<float>> bufs(static_cast<std::size_t>(world),
                                       std::vector<float>(n, 1.0f));
  auto channels = make_ring_channels(world);
  std::vector<std::thread> ts;
  double t0 = now_us();
  for (int r = 0; r < world; ++r) {
    ts.emplace_back([&, r] {
      RcclAllreducePlan plan(world, r, RcclReduceOp::kSum, n);
      plan.execute(bufs[static_cast<std::size_t>(r)].data(), n,
                   *channels[static_cast<std::size_t>(r)],
                   *channels[static_cast<std::size_t>(r)], nullptr);
    });
  }
  for (auto& t : ts) t.join();
  return now_us() - t0;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> edges = {0, 1, 2, 4};
  std::vector<int> mibs = {1, 4, 16, 48};
  int world = 8;
  int iters = 200;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--edges" && i + 1 < argc) edges = parse_ints(argv[++i]);
    else if (a == "--mib" && i + 1 < argc) mibs = parse_ints(argv[++i]);
    else if (a == "--world" && i + 1 < argc) world = std::atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
  }
  if (world < 1) world = 1;
  const std::size_t kMiB = 1024u * 1024u;

  std::printf("# vkernels comm: HIP/RCCL transport selection (issue #19)\n");
  std::printf("#   cost model: Socket = max(50, 6.0 us/MiB) + 25 us/edge;\n");
  std::printf("#               OFI    = max(20, 3.0 us/MiB) (RDMA, edge-free)\n");
  std::printf("#   acceptance: OFI < Socket at >= 1 MiB over >= 1 edge\n\n");

  std::printf("%-8s %-7s %12s %12s %10s %9s\n", "mib", "edges",
              "socket_us", "ofi_us", "transport", "speedup");
  RcclTransportConfig cfg;  // adaptive, HIP-aware plugin (librccl-net-ofi)
  cfg.net_plugin = "librccl-net-ofi";
  for (int m : mibs) {
    const std::size_t bytes = static_cast<std::size_t>(m) * kMiB;
    for (int e : edges) {
      const double s = vkernels::comm::est_rccl_socket_us(bytes, e);
      const double o = vkernels::comm::est_rccl_ofi_us(bytes, e);
      const RcclTransport t = resolve_transport(bytes, e, cfg);
      const double speedup = (o > 0 && o < s) ? s / o : 0.0;
      std::printf("%-8d %-7d %12.2f %12.2f %10s %9.2f\n", m, e, s, o,
                  vkernels::comm::transport_name(t), speedup);
    }
  }

  std::printf("\n# host ring all-reduce (RcclAllreducePlan, mock channels):\n");
  std::printf("#   world=%d, iters=%d (min over iters, warmup 20%%)\n\n",
              world, iters);
  std::printf("%-8s %14s %12s\n", "mib", "measured_us", "ofi_pred_us");
  const int warmup = iters / 5;
  for (int m : mibs) {
    if (m < 1) continue;
    const std::size_t n = static_cast<std::size_t>(m) * kMiB / sizeof(float);
    if (n == 0 || n % static_cast<std::size_t>(world) != 0) continue;
    double best = 1e30;
    for (int i = 0; i < iters; ++i) {
      double us = run_ring_once(world, n);
      if (i >= warmup) best = std::min(best, us);
    }
    const double pred = vkernels::comm::est_rccl_ofi_us(
        static_cast<std::size_t>(m) * kMiB, 2);
    std::printf("%-8d %14.2f %12.2f\n", m, best, pred);
  }
  return 0;
}
