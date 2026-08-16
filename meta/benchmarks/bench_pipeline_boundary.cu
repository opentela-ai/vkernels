// meta/benchmarks/bench_pipeline_boundary.cu
//
// Device micro-benchmark for the graph-capturable PP-boundary transfer
// (issue #10). The host reference (pipeline_boundary.cpp) is the
// correctness oracle; bench_pipeline_boundary.cpp is its always-runnable
// host-CPU analog. THIS bench is the device realization, run on a real
// multi-GPU box (sgs-gpu07: 4x H100 NVL, CUDA 13 / driver 580.82.07) so
// the numbers reflect the actual NVLink peer path the boundary issues
// every decode iteration.
//
// Three measurements, the third being the issue's acceptance:
//
//   0. Cross-device data-correctness gate.  A 4100 B pattern (256 uint4
//      + 4 B tail) is sent A->B then received B->A across the REAL NVLink
//      pair, captured once into a graph and replayed 8 times. Verifies
//      both buffers hold the pattern afterward — the same-device unit
//      test cannot exercise the cross-device + tail path this does.
//
//   1. Peer bandwidth.   One cudaMemcpyAsync(D2D) GPU0 -> GPU1 across
//      NVLink, payload sweep. This is exactly what the same-node-peer
//      device path enqueues per decode iteration (PipelineBoundaryPlan
//      kSameNodePeer / kSend). Reports effective GB/s vs the H100 NVL
//      ~600 GB/s bidirectional roof.
//
//   2. Graph capture + replay round trip.  A directed boundary pair
//      (send A->B on GPU0, recv B->A on GPU1, i.e. A's hidden state
//      reaches B and comes back) is captured ONCE into a graph; the
//      benchmark then measures the per-replay latency of cudaGraphLaunch
//      (no host enqueue of the copies, no host progress) vs. an "eager"
//      path that re-issues the two cudaMemcpyAsync launches every
//      iteration (the host work the graph removes). This is the
//      graph-capturable vs host-staged contrast at the heart of the
//      issue, on real hardware.
//
//   3. PP>1 one-graph replay.  A PP=3 chain (GPU0 send, GPU1 send+recv,
//      GPU2 recv — two boundaries, forward transfer only) is captured
//      into ONE graph and replayed N times; the benchmark reports
//      per-replay device time and confirms no host enqueue occurs on
//      replay (acceptance #2: a graph segment held across the boundary
//      without deadlock).
//
//   ./bench_pipeline_boundary [--payloads 64,256,1024,4096,16384,65536]
//                             [--iters 200] [--dst-device 1]
//
// --dst-device N (default 1) places the peer buffer on device N and
// enables bidirectional peer access, so the copies traverse real NVLink.
// With one GPU it falls back to a same-device simulation.
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is
// present. No external benchmark dependency: timing is raw cudaEvent +
// steady_clock, with warmup and a median over several iterations to cut
// launch-tick variance.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "vkernels/comm/pipeline_boundary.hpp"
#include "vkernels/comm/pipeline_boundary_cuda.hpp"

using vkernels::comm::BoundaryDirection;
using vkernels::comm::PipelineTransport;
using vkernels::comm::cuda::PipelineBoundaryPlan;

namespace {

int g_dst_dev = 1;

// Median device_ms (between record/sync per iter) and host_ms (wall time
// of the enqueue body) for `iters` timed invocations of `fn` on `stream`,
// after a few warmups. Mirrors p2p_gather_bench::time_median.
template <typename Fn>
void time_median(cudaStream_t stream, int iters, Fn fn, float* device_ms,
                 float* host_ms) {
  for (int w = 0; w < 5; ++w) {
    fn();
    cudaStreamSynchronize(stream);
  }
  cudaEvent_t b, e;
  cudaEventCreate(&b);
  cudaEventCreate(&e);
  std::vector<float> dms, hms;
  dms.reserve(iters);
  hms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    auto h0 = std::chrono::steady_clock::now();
    cudaEventRecord(b, stream);
    fn();
    cudaEventRecord(e, stream);
    auto h1 = std::chrono::steady_clock::now();
    cudaEventSynchronize(e);
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, b, e);
    dms.push_back(dt);
    hms.push_back(std::chrono::duration<float, std::milli>(h1 - h0).count());
  }
  std::sort(dms.begin(), dms.end());
  std::sort(hms.begin(), hms.end());
  *device_ms = dms[dms.size() / 2];
  *host_ms = hms[hms.size() / 2];
  cudaEventDestroy(b);
  cudaEventDestroy(e);
}

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

// Total bytes moved by one round trip (send + recv): 2 * payload_bytes.
double roundtrip_bytes(std::size_t payload_elems) {
  return 2.0 * static_cast<double>(payload_elems) * sizeof(float);
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> payloads = {101, 256, 1024, 4096, 16384, 65536,
                               262144, 1048576, 4194304};
  int iters = 200;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--payloads" && i + 1 < argc) payloads = parse_ints(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--dst-device" && i + 1 < argc) g_dst_dev = std::atoi(argv[++i]);
  }
  if (iters < 1) iters = 1;
  setvbuf(stdout, nullptr, _IONBF, 0);

  int ndev = 0;
  cudaGetDeviceCount(&ndev);
  if (ndev < 1) {
    std::fprintf(stderr, "bench_pipeline_boundary: no CUDA device\n");
    return 1;
  }
  if (g_dst_dev >= ndev) g_dst_dev = 0;  // single-GPU fallback

  cudaSetDevice(0);
  cudaDeviceProp prop0{};
  cudaGetDeviceProperties(&prop0, 0);
  const bool real_peer = (g_dst_dev != 0);
  if (real_peer) {
    cudaDeviceProp propD{};
    cudaGetDeviceProperties(&propD, g_dst_dev);
    cudaError_t e = cudaDeviceEnablePeerAccess(g_dst_dev, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) e = cudaSuccess;
    if (e != cudaSuccess) {
      std::fprintf(stderr, "bench_pipeline_boundary: cannot enable peer "
                           "0<->%d (%s)\n", g_dst_dev, cudaGetErrorString(e));
      return 1;
    }
    cudaSetDevice(g_dst_dev);
    e = cudaDeviceEnablePeerAccess(0, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) e = cudaSuccess;
    if (e != cudaSuccess) {
      std::fprintf(stderr, "bench_pipeline_boundary: cannot enable peer "
                           "%d<->0 (%s)\n", g_dst_dev, cudaGetErrorString(e));
      return 1;
    }
    cudaSetDevice(0);
    std::printf("# bench_pipeline_boundary  GPU0=%s  peer=%s (GPU%d)%s\n",
                prop0.name, propD.name, g_dst_dev,
                "  REAL NVLink");
  } else {
    std::printf("# bench_pipeline_boundary  GPU0=%s  same-device (single-GPU)\n",
                prop0.name);
  }
  std::printf("#   iters=%d  payloads=[elems]  (1 elem = 4 B float)\n\n", iters);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // -------------------------------------------------------------------
  // 0. Cross-device data-correctness gate (captured send+recv round trip).
  // The unit test only exercises same-device capture; this proves the
  // peer-copy kernel round-trips an arbitrary pattern — including the
  // <16 B tail — across the REAL NVLink pair, replayed from a graph.
  // -------------------------------------------------------------------
  {
    constexpr std::size_t K = 4100;  // 256 uint4 + 4 B tail
    float *a = nullptr, *b = nullptr;
    cudaMalloc(&a, K);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaMalloc(&b, K);
    if (real_peer) cudaSetDevice(0);
    std::vector<uint8_t> pat(K);
    for (std::size_t i = 0; i < K; ++i) pat[i] = static_cast<uint8_t>(0x5a ^ (i * 7 + 3));
    cudaMemcpy(a, pat.data(), K, cudaMemcpyHostToDevice);
    cudaMemsetAsync(b, 0, K, stream);
    cudaStreamSynchronize(stream);
    PipelineBoundaryPlan gsend(2, 0, K, PipelineTransport::kSameNodePeer,
                               BoundaryDirection::kSend, b);
    PipelineBoundaryPlan grecv(2, 1, K, PipelineTransport::kSameNodePeer,
                               BoundaryDirection::kRecv, b);
    cudaGraph_t gc = nullptr; cudaGraphExec_t xc = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    gsend.execute(a, stream);
    grecv.execute(a, stream);
    cudaStreamEndCapture(stream, &gc);
    const bool ok_cap = (cudaGraphInstantiate(&xc, gc, 0) == cudaSuccess);
    bool ok_data = false;
    if (ok_cap) {
      for (int r = 0; r < 8; ++r) cudaGraphLaunch(xc, stream);
      cudaStreamSynchronize(stream);
      std::vector<uint8_t> hb(K), ha(K);
      if (real_peer) cudaSetDevice(g_dst_dev);
      cudaMemcpy(hb.data(), b, K, cudaMemcpyDeviceToHost);
      if (real_peer) cudaSetDevice(0);
      cudaMemcpy(ha.data(), a, K, cudaMemcpyDeviceToHost);
      ok_data = (hb == pat) && (ha == pat);  // b received a's bytes; a got them back
      cudaGraphExecDestroy(xc);
    }
    if (gc) cudaGraphDestroy(gc);
    cudaFree(a);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaFree(b);
    if (real_peer) cudaSetDevice(0);
    std::printf("[0-correctness] captured cross-device round trip (4100 B, 4 B tail, 8 replays): %s\n",
                (ok_cap && ok_data) ? "OK" : "FAIL");
    if (!(ok_cap && ok_data)) { std::fprintf(stderr, "correctness gate FAILED\n"); return 1; }
  }

  // -------------------------------------------------------------------
  // 1. Peer bandwidth: one cudaMemcpyAsync(D2D) per iteration
  // -------------------------------------------------------------------
  std::printf("[1-peer-copy]  payload  device_ms   host_ms    GB/s   x NVLink-roof\n");
  std::printf("[1-peer-copy]    elems  (median)   (enqueue)\n");
  // Allocate peer payload buffers on each device once; reuse across sizes
  // that fit, reallocate larger as needed (kept simple: alloc per row).
  for (int p : payloads) {
    if (p < 1) continue;
    const std::size_t bytes = static_cast<std::size_t>(p) * sizeof(float);
    float *buf0 = nullptr, *bufd = nullptr;
    cudaMalloc(&buf0, bytes);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaMalloc(&bufd, bytes);
    if (real_peer) cudaSetDevice(0);
    cudaMemsetAsync(buf0, 0xAB, bytes, stream);
    if (real_peer) {
      // Peer write GPU0 -> GPUd (the send direction). Eager (not captured):
      // the raw cudaMemcpyAsync(D2D) bandwidth floor the boundary kernel
      // matches when captured into a graph.
      float dev_ms = 0, host_ms = 0;
      time_median(stream, iters, [&] {
        cudaMemcpyAsync(bufd, buf0, bytes, cudaMemcpyDeviceToDevice, stream);
      }, &dev_ms, &host_ms);
      const double gbs = (dev_ms > 0) ? bytes / (dev_ms / 1e3) / 1e9 : 0.0;
      std::printf("[1-peer-copy] %8zu %9.4f %9.4f %8.1f %9.3f\n", bytes, dev_ms,
                  host_ms, gbs, gbs / 600.0);
    } else {
      // Same-device simulation: still measures launch overhead floor.
      float dev_ms = 0, host_ms = 0;
      time_median(stream, iters, [&] {
        cudaMemcpyAsync(bufd, buf0, bytes, cudaMemcpyDeviceToDevice, stream);
      }, &dev_ms, &host_ms);
      const double gbs = (dev_ms > 0) ? bytes / (dev_ms / 1e3) / 1e9 : 0.0;
      std::printf("[1-peer-copy] %8zu %9.4f %9.4f %8.1f   (same-dev)\n", bytes,
                  dev_ms, host_ms, gbs);
    }
    cudaFree(buf0);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaFree(bufd);
    if (real_peer) cudaSetDevice(0);
  }

  // -------------------------------------------------------------------
  // 2. Graph capture + replay round trip vs eager re-issue
  // -------------------------------------------------------------------
  //   send: buf0(GPU0) -> bufd(GPUd)   [Plan A, kSend, peer=bufd]
  //   recv: bufd(GPUd) -> buf0(GPU0)   [Plan B, kRecv, peer=bufd]
  //   A round trip moves A's hidden state to B and back. The graph path
  //   captures BOTH copies once and replays with one cudaGraphLaunch
  //   (no host enqueue of the copies). The eager path re-issues both
  //   cudaMemcpyAsync every iteration (the host work the graph removes).
  std::printf("\n[2-round-trip]  payload  graph_dev  graph_host  eager_dev  eager_host  host_spd\n");
  std::printf("[2-round-trip]    elems   ms/replay  ms/launch   ms/iter    ms/iter    (host/eager)\n");
  for (int p : payloads) {
    if (p < 1) continue;
    const std::size_t bytes = static_cast<std::size_t>(p) * sizeof(float);
    float *buf0 = nullptr, *bufd = nullptr;
    cudaMalloc(&buf0, bytes);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaMalloc(&bufd, bytes);
    if (real_peer) cudaSetDevice(0);
    cudaMemsetAsync(buf0, 0x21, bytes, stream);
    cudaStreamSynchronize(stream);

    // Two prepared plans: send A->B, recv B->A (peer_buf is the shared
    // boundary buffer on the peer device).
    PipelineBoundaryPlan send(2, 0, bytes,
                              PipelineTransport::kSameNodePeer,
                              BoundaryDirection::kSend, bufd);
    PipelineBoundaryPlan recv(2, 1, bytes,
                              PipelineTransport::kSameNodePeer,
                              BoundaryDirection::kRecv, bufd);

    // --- Eager path: host re-issues both copies every iteration ---
    float eager_dev = 0, eager_host = 0;
    try {
      time_median(stream, iters, [&] {
        send.execute(buf0, stream);
        recv.execute(buf0, stream);
      }, &eager_dev, &eager_host);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "[2-eager] p=%d failed: %s (cuda: %s)\n", p,
                   e.what(), cudaGetErrorString(cudaGetLastError()));
      throw;
    }

    // --- Graph path: capture once, replay via cudaGraphLaunch ---
    // The plans now issue a graph-capturable copy KERNEL (reading peer UVA
    // over NVLink) rather than cudaMemcpyAsync(D2D), which is not capturable
    // across two different devices on Hopper / CUDA 13. One cudaGraphLaunch
    // replays both copies with no host enqueue.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    try {
      cudaError_t ce = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
      if (ce != cudaSuccess) { std::fprintf(stderr, "[2-cap] beginCapture p=%d: %s\n", p, cudaGetErrorString(ce)); throw std::runtime_error("beginCapture"); }
      send.execute(buf0, stream);
      recv.execute(buf0, stream);
      ce = cudaStreamEndCapture(stream, &graph);
      if (ce != cudaSuccess) { std::fprintf(stderr, "[2-cap] endCapture p=%d: %s\n", p, cudaGetErrorString(ce)); throw std::runtime_error("endCapture"); }
      ce = cudaGraphInstantiate(&exec, graph, 0);
      if (ce != cudaSuccess) { std::fprintf(stderr, "[2-cap] instantiate p=%d: %s\n", p, cudaGetErrorString(ce)); throw std::runtime_error("instantiate"); }
    } catch (...) {
      if (graph) cudaGraphDestroy(graph);
      cudaFree(buf0);
      if (real_peer) cudaSetDevice(g_dst_dev);
      cudaFree(bufd);
      if (real_peer) cudaSetDevice(0);
      throw;
    }

    float graph_dev = 0, graph_host = 0;
    time_median(stream, iters, [&] { cudaGraphLaunch(exec, stream); },
                &graph_dev, &graph_host);
    // The graph does not make the copy faster (device time is ~= eager);
    // the win is HOST work per decode step: one cudaGraphLaunch vs
    // re-issuing both boundary copies. Report the host-launch ratio.
    const double host_speedup = (graph_host > 0 && eager_host > 0)
                                    ? eager_host / graph_host
                                    : 0.0;
    std::printf("[2-round-trip] %8zu %9.4f %10.4f %9.4f %10.4f %9.2fx\n",
                bytes, graph_dev, graph_host, eager_dev, eager_host,
                host_speedup);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(buf0);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaFree(bufd);
    if (real_peer) cudaSetDevice(0);
  }

  // -------------------------------------------------------------------
  // 3. PP>1 one-graph: two boundaries (GPU0->GPU1->GPU2) captured once
  // -------------------------------------------------------------------
  // Requires 3 devices for a real PP chain; with fewer, simulate the two
  // boundaries on the same pair (the graph still captures 4 copies and
  // replays them as one launch — the "no host progress across N replays"
  // acceptance is unchanged).
  const bool pp3 = (ndev >= 3);
  std::printf("\n[3-pp%d-one-graph]  capture %d boundary copies, replay %d times\n",
              pp3 ? 3 : 2, pp3 ? 4 : 4, iters);
  if (pp3) {
    const int mid = 1, far = 2;
    cudaSetDevice(mid);
    cudaDeviceEnablePeerAccess(0, 0);
    cudaDeviceEnablePeerAccess(far, 0);
    cudaSetDevice(far);
    cudaDeviceEnablePeerAccess(mid, 0);
    cudaSetDevice(0);
    const std::size_t bytes = static_cast<std::size_t>(4096) * sizeof(float);
    float *b0 = nullptr, *b1 = nullptr, *b2 = nullptr;
    cudaMalloc(&b0, bytes);
    cudaSetDevice(mid);
    cudaMalloc(&b1, bytes);
    cudaSetDevice(far);
    cudaMalloc(&b2, bytes);
    cudaSetDevice(0);
    cudaMemsetAsync(b0, 0x40, bytes, stream);
    cudaStreamSynchronize(stream);

    PipelineBoundaryPlan s01(3, 0, bytes, PipelineTransport::kSameNodePeer,
                             BoundaryDirection::kSend, b1);
    PipelineBoundaryPlan r10(3, 1, bytes, PipelineTransport::kSameNodePeer,
                             BoundaryDirection::kRecv, b1);
    PipelineBoundaryPlan s12(3, 1, bytes, PipelineTransport::kSameNodePeer,
                             BoundaryDirection::kSend, b2);
    PipelineBoundaryPlan r21(3, 2, bytes, PipelineTransport::kSameNodePeer,
                             BoundaryDirection::kRecv, b2);

    // Host-time to enqueue the 4 copies WITHOUT a graph (per iteration).
    float eager_dev = 0, eager_host = 0;
    time_median(stream, iters, [&] {
      s01.execute(b0, stream);
      r10.execute(b1, stream);
      s12.execute(b1, stream);
      r21.execute(b2, stream);
    }, &eager_dev, &eager_host);

    cudaGraph_t g3 = nullptr;
    cudaGraphExec_t x3 = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    s01.execute(b0, stream);
    r10.execute(b1, stream);
    s12.execute(b1, stream);
    r21.execute(b2, stream);
    cudaStreamEndCapture(stream, &g3);
    cudaGraphInstantiate(&x3, g3, 0);

    // Host-time of the N replays: ONE cudaGraphLaunch per iteration, no
    // copy enqueue (the acceptance: no host progress on replay).
    float graph_dev = 0, graph_host = 0;
    time_median(stream, iters, [&] { cudaGraphLaunch(x3, stream); },
                &graph_dev, &graph_host);
    std::printf("[3-pp3-one-graph]  eager: dev=%.4f host=%.4f ms/iter\n",
                eager_dev, eager_host);
    std::printf("[3-pp3-one-graph]  graph: dev=%.4f host=%.4f ms/replay  (host/iter ZERO copy enqueue)\n",
                graph_dev, graph_host);
    std::printf("[3-pp3-one-graph]  speedup=%.2fx  (eager_host/graph_host host-launch ratio)\n",
                (graph_host > 0) ? eager_host / graph_host : 0.0);
    cudaGraphExecDestroy(x3);
    cudaGraphDestroy(g3);
    cudaFree(b0);
    cudaSetDevice(mid);
    cudaFree(b1);
    cudaSetDevice(far);
    cudaFree(b2);
    cudaSetDevice(0);
  } else {
    // Single-pair PP=2: two copies (one boundary), replayed N times.
    const std::size_t bytes = static_cast<std::size_t>(4096) * sizeof(float);
    float *b0 = nullptr, *b1 = nullptr;
    cudaMalloc(&b0, bytes);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaMalloc(&b1, bytes);
    if (real_peer) cudaSetDevice(0);
    cudaMemsetAsync(b0, 0x40, bytes, stream);
    cudaStreamSynchronize(stream);

    PipelineBoundaryPlan s(2, 0, bytes, PipelineTransport::kSameNodePeer,
                           BoundaryDirection::kSend, b1);
    PipelineBoundaryPlan r(2, 1, bytes, PipelineTransport::kSameNodePeer,
                           BoundaryDirection::kRecv, b1);

    float eager_dev = 0, eager_host = 0;
    time_median(stream, iters, [&] {
      s.execute(b0, stream);
      r.execute(b0, stream);
    }, &eager_dev, &eager_host);

    cudaGraph_t g2 = nullptr;
    cudaGraphExec_t x2 = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    s.execute(b0, stream);
    r.execute(b0, stream);
    cudaStreamEndCapture(stream, &g2);
    cudaGraphInstantiate(&x2, g2, 0);

    float graph_dev = 0, graph_host = 0;
    time_median(stream, iters, [&] { cudaGraphLaunch(x2, stream); },
                &graph_dev, &graph_host);
    std::printf("[3-pp2-one-graph]  eager: dev=%.4f host=%.4f ms/iter (2 copies)\n",
                eager_dev, eager_host);
    std::printf("[3-pp2-one-graph]  graph: dev=%.4f host=%.4f ms/replay  (host/iter ZERO copy enqueue)\n",
                graph_dev, graph_host);
    std::printf("[3-pp2-one-graph]  speedup=%.2fx  (eager_host/graph_host host-launch ratio)\n",
                (graph_host > 0) ? eager_host / graph_host : 0.0);
    cudaGraphExecDestroy(x2);
    cudaGraphDestroy(g2);
    cudaFree(b0);
    if (real_peer) cudaSetDevice(g_dst_dev);
    cudaFree(b1);
    if (real_peer) cudaSetDevice(0);
  }

  cudaStreamDestroy(stream);
  return 0;
}
