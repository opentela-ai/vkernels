// meta/benchmarks/bench_cross_node_kv.cpp
//
// Host benchmark for the cross-node KV transfer (issue #49). The host
// reference (fabric_import.cpp / cross_node_kv.cpp) is the always-compiled,
// 100%-line-covered correctness oracle; this bench is its always-runnable
// performance analog — exactly as bench_pipeline_boundary.cpp is the
// CI-verifiable surface for issue #10. No GPU, no fabric: the numbers here
// are the PER-HOP COST MODEL (cross_node_kv_throughput) compared against
// the same-node roofline and the synchronous bulk-copy fallback, with the
// GH200 DRAM-only / host-bounce caveat called out — acceptance #2.
//
//   ./bench_cross_node_kv [--payloads 262144,1048576,16777216,67108864]
//                         [--iters 100000]
//
// The actual device-side win the issue cares about — the real-RDMA-fabric
// per-hop throughput on H-CLARIDEN / H-JSC, over a 400 Gb/s Slingshot pair
// instead of the UCX loopback (~0.34 GB/s) that only validated correctness
// — is the on-site measurement step the issue names explicitly ("no real
// RDMA fabric is in any container today"). This bench is the host analog:
// it prints the cost the model predicts, so a CI job can regress the
// classification, the roofline references, and the caveat.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "vkernels/comm/fabric_import.hpp"

using vkernels::comm::classify_fabric_import;
using vkernels::comm::cross_node_kv_throughput;
using vkernels::comm::FabricImportConfig;
using vkernels::comm::FabricImportTransport;
using vkernels::comm::same_node_fabric_roof_gbps;

namespace {

std::vector<std::size_t> parse_sizes(const char* s) {
  std::vector<std::size_t> out;
  std::string t = s ? s : "";
  for (char& c : t)
    if (c == ',') c = ' ';
  std::istringstream is(t);
  std::size_t v;
  while (is >> v) out.push_back(v);
  return out;
}

double now_ns() {
  using namespace std::chrono;
  return duration<double, std::nano>(steady_clock::now().time_since_epoch()).count();
}

const char* tname(FabricImportTransport t) {
  return vkernels::comm::fabric_import_transport_name(t);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::size_t> payloads = {262144, 1048576, 16777216, 67108864};
  int iters = 100000;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--payloads" && i + 1 < argc) payloads = parse_sizes(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
  }
  if (iters < 1) iters = 1;

  std::printf("# vkernels comm: cross-node KV transfer (issue #49)\n");
  std::printf("#   HOST per-hop cost model (no GPU, no fabric) — CI-verifiable\n");
  std::printf("#   acceptance #2 surface. The real-RDMA-fabric per-hop\n");
  std::printf("#   throughput on H-CLARIDEN / H-JSC is the on-site step\n");
  std::printf("#   (no real RDMA fabric is in any container today).\n\n");

  // --- Table 1: one-time planning overhead (classification) ---
  std::printf("[planning] classify_fabric_import (min over %d calls)\n", iters);
  {
    struct Case {
      const char* name;
      FabricImportConfig cfg;
    };
    const Case cases[] = {
        {"same-node",          {true, false, false}},
        {"dram-only-libfabric",{false, false, true}},
        {"gpudirect-rdma",     {false, true, false}},
        {"no-fabric",          {false, false, false}},
    };
    for (const auto& c : cases) {
      double best = 1e30;
      for (int i = 0; i < iters; ++i) {
        double t0 = now_ns();
        volatile auto r = classify_fabric_import(c.cfg);
        (void)r;
        best = std::min(best, now_ns() - t0);
      }
      std::printf("[planning]   %-20s -> %-13s : %7.1f ns/call\n",
                  c.name, tname(classify_fabric_import(c.cfg)), best);
    }
  }

  // --- Table 2: per-hop cost vs the same-node roofline + the fallback ---
  std::printf("\n[per-hop] transport       payload     hop GB/s   hop us"
              "   roof GB/s  fallback GB/s  GH200-caveat   roof/hop\n");
  for (std::size_t p : payloads) {
    if (p == 0) continue;
    for (int ti = 0; ti <= 2; ++ti) {
      const auto t = static_cast<FabricImportTransport>(ti);
      auto c = cross_node_kv_throughput(t, p);
      const double ratio = (c.per_hop_gbps > 0.0)
                               ? c.same_node_roof_gbps / c.per_hop_gbps
                               : 0.0;
      std::printf("[per-hop] %-13s %10zu %9.2f %9.3f %9.1f %12.1f %12s %10.2f\n",
                  tname(t), p, c.per_hop_gbps, c.per_hop_us,
                  c.same_node_roof_gbps, c.bulk_copy_fallback_gbps,
                  c.gh200_dram_only_caveat ? "yes" : "no", ratio);
    }
  }

  // --- Table 3: the GH200 degradation (fabric import requested, denied) ---
  std::printf("\n[gh200] kFabricMapped degrades to the bulk-copy fallback"
              " when the C2C-attached GPU is invisible to libfabric:\n");
  std::printf("[gh200]   payload     normal GB/s   gh200 GB/s   caveat(normal/cross)\n");
  for (std::size_t p : payloads) {
    if (p == 0) continue;
    auto normal = cross_node_kv_throughput(FabricImportTransport::kFabricMapped, p,
                                           /*gh200_dram_only=*/false);
    auto gh200  = cross_node_kv_throughput(FabricImportTransport::kFabricMapped, p,
                                           /*gh200_dram_only=*/true);
    std::printf("[gh200] %10zu %13.2f %12.2f   %s/%s\n", p,
                normal.per_hop_gbps, gh200.per_hop_gbps,
                normal.gh200_dram_only_caveat ? "yes" : "no",
                gh200.gh200_dram_only_caveat ? "yes" : "no");
  }

  std::printf("\n# kFabricMapped runs at min(fabric 50, kernel 88.5) = 50 GB/s\n");
  std::printf("# normally (the kernel is the binding resource; NVLink raw is\n");
  std::printf("# 220-243). On GH200 the DRAM-only libfabric constraint forces a\n");
  std::printf("# host bounce EVEN though a fabric import is requested, so the\n");
  std::printf("# per-hop rate degrades to the 1.4 GB/s bulk-copy fallback and\n");
  std::printf("# the caveat is set -- the property acceptance #2 calls out.\n");
  return 0;
}
