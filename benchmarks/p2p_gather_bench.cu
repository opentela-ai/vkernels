// benchmarks/p2p_gather_bench.cu
//
// P2P run-list gather: the adaptive CUDA path vs the per-run
// cudaMemcpyPeerAsync / cudaMemcpy2DAsync loop it replaces, plus the
// prepared-plan reuse pattern. CUDA-only: the primitive issues peer reads
// from inside one kernel, so there is nothing to measure on the host.
//
// On a single-GPU box this simulates the peer source as another device
// allocation and uses cudaMemcpyDeviceToDevice for the baseline; the launch
// overhead it measures is exactly what the single-launch kernel removes. On a
// real multi-GPU system, point `src_ptrs[i]` at peer memory and enable peer
// access before running — the kernel and the measurement are unchanged.
//
// Issue #6 acceptance sweeps are built in:
//   * fixed 48 MiB payload across 1, 2, 4, 8, 16, 32, 64, 192 runs,
//   * columns for the copy-engine baseline, the forced kernel, and the
//     adaptive path (which chooses per-run copy engine below the crossover
//     and the kernel at/above it),
//   * descriptor preparation/allocation reported separately from kernel
//     execution (host-enqueue time vs device event time per iteration),
//   * a prepared-plan section that prepares ONCE and times 40 executes,
//     the KVAAS layer-reuse pattern.
//
// `--concurrent` additionally keeps a memory-bound fill kernel busy on a
// second stream while the gather is measured, approximating copy-engine vs
// SM-driven transfer under concurrent model execution (issue #6 item 4).
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is present.
// No external benchmark dependency: timing is raw cudaEvent + steady_clock,
// with warmup and a median over several iterations to cut launch-tick
// variance.
//
//   ./p2p_gather_bench [--concurrent] [--quick] [--src-device N]
//
// --src-device N (default 0) places the source buffer on device N and
// enables peer access, so the kernel and the copy-engine baseline read
// real peer memory over NVLink instead of simulating a peer on the same
// device. dst stays on the current device (0).

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/comm/p2p_gather.hpp"
#include "vkernels/comm/p2p_gather_cuda.hpp"

using vkernels::comm::Gather2DRun;
using vkernels::comm::GatherDispatchMode;
using vkernels::comm::set_gather_dispatch;
using vkernels::comm::cuda::P2PGatherPlan1D;
using vkernels::comm::cuda::P2PGatherPlan2D;
using vkernels::comm::cuda::p2p_gather_runs;
using vkernels::comm::cuda::p2p_gather_runs_2d;

namespace {

// Source device for the "peer" buffer (0 = same-device simulation).
int g_src_dev = 0;

// Memory-bound filler kernel run concurrently on a second stream in
// --concurrent mode: it competes with the gather for SMs and HBM bandwidth,
// the way attention kernels do during model execution.
__global__ void fill_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.0f;
}

struct Timings {
  float baseline_ms = 0.0f;  // per-run cudaMemcpy{Peer,2D}Async loop (device)
  float kernel_ms = 0.0f;    // forced single-launch kernel (device)
  float adaptive_ms = 0.0f;  // adaptive dispatch (device)
  float adaptive_host_ms = 0.0f;  // adaptive host-enqueue time (prep+launch)
  float kernel_host_ms = 0.0f;    // kernel-path host-enqueue time
  bool adaptive_took_kernel = false;
};

// Median elapsed time for `iters` timed invocations of `fn` on `stream`,
// after a few warmups. Returns (median_device_ms, median_host_ms): the whole
// loop sits between one record/sync pair per iteration so device_ms reflects
// per-iteration device work (H2D metadata copy + kernel/copies), and host_ms
// is the enqueue wall time (validation + descriptor construction + metadata
// allocation + H2D enqueue + launch) — the "preparation/allocation"
// contribution, reported separately from kernel execution.
template <typename Fn>
void time_median(cudaStream_t stream, int iters, Fn fn, float* device_ms,
                 float* host_ms) {
  for (int w = 0; w < 3; ++w) fn();
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

// If --concurrent, enqueue `fillers` fill kernels on `s2` writing to `buf` so
// the device is still busy with them while the timed work on `s1` runs.
// The buffer is owned by the caller (never cudaFree it here: cudaFree would
// block the host until the enqueued fills finish, serialising the very
// overlap this mode exists to create); free it after
// cudaStreamSynchronize(s2).
void maybe_start_filler(bool concurrent, float* buf, int n, cudaStream_t s2) {
  if (!concurrent) return;
  for (int i = 0; i < 20000; ++i)
    fill_kernel<<<1024, 256, 0, s2>>>(buf, n);
}

// 1-D sweep: `count` disjoint runs of `run_bytes` each, total = count*run_bytes.
// Times the per-run copy loop (baseline), the forced kernel, and the
// adaptive path; reports host-enqueue time for the adaptive path.
Timings bench_1d(cudaStream_t stream, std::size_t count, std::size_t run_bytes,
                 int iters) {
  const std::size_t total = count * run_bytes;
  std::uint8_t* dst = nullptr;
  std::uint8_t* src = nullptr;  // one contiguous "peer" region, sliced per run
  cudaMalloc(&dst, total);
  cudaSetDevice(g_src_dev);
  cudaMalloc(&src, total);
  cudaSetDevice(0);

  std::vector<const void*> src_ptrs(count);
  std::vector<std::size_t> dst_offsets(count);
  std::vector<std::size_t> lengths(count);
  for (std::size_t i = 0; i < count; ++i) {
    src_ptrs[i] = src + i * run_bytes;
    dst_offsets[i] = i * run_bytes;
    lengths[i] = run_bytes;
  }

  Timings t;
  // Baseline: one cudaMemcpyPeerAsync per run — the exact API the
  // single-launch primitive replaces. On this single-GPU node src and dst are
  // the same device (0->0); on a multi-GPU system src would be a peer device.
  // Baseline: one cudaMemcpyPeerAsync per run — the exact API the
  // single-launch primitive replaces. Same device (0->0) when --src-device
  // is unset; a real peer copy (0->N) when it is set and peer access is on.
  float h = 0.0f;
  time_median(stream, iters, [&] {
    for (std::size_t i = 0; i < count; ++i)
      cudaMemcpyPeerAsync(dst + dst_offsets[i], 0, src_ptrs[i], g_src_dev,
                          lengths[i], stream);
  }, &t.baseline_ms, &h);

  // Forced kernel: the PR #5 single-launch path, measured directly.
  set_gather_dispatch(GatherDispatchMode::kForceKernel, 1);
  time_median(stream, iters, [&] {
    p2p_gather_runs(dst, total, src_ptrs.data(), dst_offsets.data(),
                    lengths.data(), count, stream);
  }, &t.kernel_ms, &t.kernel_host_ms);

  // Adaptive: the production path — copy engine below the crossover, kernel
  // at/above it (which branch was taken is reported for verification).
  set_gather_dispatch(GatherDispatchMode::kAdaptive, 4);
  t.adaptive_took_kernel = vkernels::comm::prefer_gather_kernel(count, total);
  time_median(stream, iters, [&] {
    p2p_gather_runs(dst, total, src_ptrs.data(), dst_offsets.data(),
                    lengths.data(), count, stream);
  }, &t.adaptive_ms, &t.adaptive_host_ms);
  set_gather_dispatch();  // restore defaults

  cudaFree(dst);
  cudaFree(src);
  return t;
}

// 2-D sweep: `count` strided tiles, each `height` x `width` bytes copied from a
// row-major peer region (src stride `src_stride`) into `dst` (dst stride
// `dst_stride`). Baseline is one cudaMemcpy2DAsync per tile.
Timings bench_2d(cudaStream_t stream, std::size_t count, std::size_t width,
                 std::size_t height, std::size_t src_stride,
                 std::size_t dst_stride, int iters) {
  const std::size_t src_bytes = count * height * src_stride;
  const std::size_t dst_bytes = count * height * dst_stride;
  std::uint8_t* dst = nullptr;
  std::uint8_t* src = nullptr;
  cudaMalloc(&dst, dst_bytes);
  cudaSetDevice(g_src_dev);
  cudaMalloc(&src, src_bytes);
  cudaSetDevice(0);

  std::vector<Gather2DRun> runs(count);
  for (std::size_t i = 0; i < count; ++i)
    runs[i] = {src + i * height * src_stride, src_stride, i * height * dst_stride,
               dst_stride, width, height};

  // Baseline: one cudaMemcpy2DAsync per tile. cudaMemcpyDefault resolves
  // the devices from the pointers (same-device or peer under UVA).
  Timings t;
  float h = 0.0f;
  time_median(stream, iters, [&] {
    for (std::size_t i = 0; i < count; ++i)
      cudaMemcpy2DAsync(dst + runs[i].dst_offset, dst_stride, runs[i].src,
                        src_stride, width, height, cudaMemcpyDefault, stream);
  }, &t.baseline_ms, &h);

  set_gather_dispatch(GatherDispatchMode::kForceKernel, 1);
  time_median(stream, iters, [&] {
    p2p_gather_runs_2d(dst, dst_bytes, runs.data(), count, stream);
  }, &t.kernel_ms, &t.kernel_host_ms);

  set_gather_dispatch(GatherDispatchMode::kAdaptive, 4);
  t.adaptive_took_kernel = vkernels::comm::prefer_gather_kernel(count, count * width * height);
  time_median(stream, iters, [&] {
    p2p_gather_runs_2d(dst, dst_bytes, runs.data(), count, stream);
  }, &t.adaptive_ms, &t.adaptive_host_ms);
  set_gather_dispatch();

  cudaFree(dst);
  cudaFree(src);
  return t;
}

// Plan timing: prepare once (reporting the one-time host+device prep cost),
// then time `repeats` execute() calls — the KVAAS "one run list, 40 layers"
// pattern. Reports the median per-execute device time and the total, so the
// per-layer overhead of validation/allocation/H2D is visible as zero.
void bench_plan_1d(cudaStream_t stream, std::size_t count, std::size_t run_bytes,
                   int repeats) {
  const std::size_t total = count * run_bytes;
  std::uint8_t* dst = nullptr;
  std::uint8_t* src = nullptr;
  cudaMalloc(&dst, total);
  cudaSetDevice(g_src_dev);
  cudaMalloc(&src, total);
  cudaSetDevice(0);
  std::vector<const void*> src_ptrs(count);
  std::vector<std::size_t> dst_offsets(count), lengths(count);
  for (std::size_t i = 0; i < count; ++i) {
    src_ptrs[i] = src + i * run_bytes;
    dst_offsets[i] = i * run_bytes;
    lengths[i] = run_bytes;
  }

  // One-time prepare: validation + descriptor construction + persistent
  // device allocation + synchronous H2D upload. Host wall time only (the
  // upload is synchronous so it is included).
  auto p0 = std::chrono::steady_clock::now();
  P2PGatherPlan1D plan(dst, total, src_ptrs.data(), dst_offsets.data(),
                       lengths.data(), count);
  auto p1 = std::chrono::steady_clock::now();
  const double prepare_ms =
      std::chrono::duration<double, std::milli>(p1 - p0).count();

  float dev = 0.0f, host = 0.0f;
  time_median(stream, std::min(repeats, 25), [&] { plan.execute(stream); },
              &dev, &host);
  std::printf("  plan  %-5zu %-10zu %-12.3f %-12.2f %-12.2f\n",
              count, run_bytes, prepare_ms, dev * 1e3f, host * 1e3f);
  cudaFree(dst);
  cudaFree(src);
}

void print_header(const char* title) {
  std::printf("\n%s\n", title);
  std::printf("%-6s %-9s %-11s %-11s %-11s %-10s %-11s %-11s %s\n",
              "count", "run_bytes", "baseline_us", "kernel_us", "adaptive_us",
              "adap_kern", "adap_host_us", "kernel_host_us", "speedup");
  std::printf("%s\n",
              "--------------------------------------------------------------------------------------");
}

void print_row(std::size_t count, std::size_t run_bytes, const Timings& t) {
  const double base_us = t.baseline_ms * 1e3;
  const double kern_us = t.kernel_ms * 1e3;
  const double adap_us = t.adaptive_ms * 1e3;
  const double speedup = t.adaptive_ms > 0 ? t.baseline_ms / t.adaptive_ms : 0.0;
  std::printf("%-6zu %-9zu %-11.2f %-11.2f %-11.2f %-10s %-11.2f %-11.2f %.2fx\n",
              count, run_bytes, base_us, kern_us, adap_us,
              t.adaptive_took_kernel ? "kernel" : "copy",
              t.adaptive_host_ms * 1e3, t.kernel_host_ms * 1e3, speedup);
}

}  // namespace

int main(int argc, char** argv) {
  bool concurrent = false, quick = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--concurrent") == 0) concurrent = true;
    else if (std::strcmp(argv[i], "--quick") == 0) quick = true;
    else if (std::strcmp(argv[i], "--src-device") == 0 && i + 1 < argc)
      g_src_dev = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: %s [--concurrent] [--quick] [--src-device N]\n", argv[0]);
      return 2;
    }
  }
  const int iters = quick ? 10 : 50;
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "p2p_gather_bench: no CUDA device\n");
    return 1;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("# p2p_gather_bench  device=%s sm=%d.%d%s%s\n",
              prop.name, prop.major, prop.minor,
              concurrent ? "  concurrent-compute mode" : "",
              quick ? "  quick" : "");
  if (g_src_dev != 0) {
    // Enable bidirectional peer access so the kernel and the baseline loop
    // read src directly over NVLink. H100 NVL supports this; abort with a
    // clear message if the box cannot (e.g. peer access disabled in the
    // driver config).
    cudaDeviceProp s{};
    cudaGetDeviceProperties(&s, g_src_dev);
    cudaError_t e = cudaDeviceEnablePeerAccess(g_src_dev, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) e = cudaSuccess;
    if (e != cudaSuccess) {
      std::fprintf(stderr, "p2p_gather_bench: cannot enable peer access "
                           "0<->%d (%s)\n", g_src_dev, cudaGetErrorString(e));
      return 1;
    }
    cudaSetDevice(g_src_dev);
    e = cudaDeviceEnablePeerAccess(0, 0);
    if (e == cudaErrorPeerAccessAlreadyEnabled) e = cudaSuccess;
    if (e != cudaSuccess) {
      std::fprintf(stderr, "p2p_gather_bench: cannot enable peer access "
                           "%d<->0 (%s)\n", g_src_dev, cudaGetErrorString(e));
      return 1;
    }
    cudaSetDevice(0);
    std::printf("# src on device %d (%s): real peer reads\n", g_src_dev, s.name);
  } else {
    std::printf("# src on device 0: same-device simulation (use --src-device N)\n");
  }
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStream_t filler_stream;
  cudaStreamCreate(&filler_stream);
  // Filler buffer lives for the whole process when --concurrent; freeing it
  // before the final sync would serialise the overlap (see
  // maybe_start_filler).
  float* filler_buf = nullptr;
  const int filler_n = static_cast<int>(256u * 1024u * 1024u / sizeof(float));
  if (concurrent) cudaMalloc(&filler_buf, 256u * 1024u * 1024u);

  // Issue #6 acceptance sweep: fixed 48 MiB payload across the run counts
  // that bracket the measured 16-32 crossover (1, 2, 4, 8, 16 below it;
  // 32, 64, 192 at/above it).
  maybe_start_filler(concurrent, filler_buf, filler_n, filler_stream);
  print_header("1-D gather: 48 MiB fixed payload, issue #6 acceptance sweep");
  constexpr std::size_t k48MiB = 48u * 1024u * 1024u;
  for (std::size_t count : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 192u}) {
    const Timings t = bench_1d(stream, count, k48MiB / count, iters);
    print_row(count, k48MiB / count, t);
  }

  // Existing sweep: 4 MiB total at increasing fragmentation.
  print_header("1-D gather: 4 MiB total, sweep fragmentation");
  constexpr std::size_t k4MiB = 4u * 1024u * 1024u;
  for (std::size_t count : {1u, 8u, 32u, 128u, 512u, 2048u}) {
    const Timings t = bench_1d(stream, count, k4MiB / count, iters);
    print_row(count, k4MiB / count, t);
  }

  // Fixed page size: 4 KiB runs, sweep count.
  print_header("1-D gather: 4 KiB runs, sweep count");
  for (std::size_t count : {1u, 64u, 256u, 1024u, 2048u}) {
    const Timings t = bench_1d(stream, count, 4u * 1024u, iters);
    print_row(count, 4u * 1024u, t);
  }

  // 2-D: strided tiles, sweep count.
  print_header("2-D gather: 64x512 strided tiles, sweep count");
  for (std::size_t count : {1u, 16u, 64u, 256u}) {
    const Timings t = bench_2d(stream, count, 512u, 64u, 1024u, 1024u, iters);
    print_row(count, 0, t);
  }

  // Prepared plan: the KVAAS reuse pattern (one run list, many layer
  // launches). prepare_ms is the ONE-TIME cost; the per-execute device and
  // host times are what every layer pays after that (no allocation, no H2D).
  print_header("Prepared plan: prepare once, 40 layer executes (48 MiB total)");
  constexpr std::size_t kPlanRuns = 8;
  bench_plan_1d(stream, kPlanRuns, k48MiB / kPlanRuns, 40);

  if (concurrent) {
    cudaStreamSynchronize(filler_stream);
    cudaFree(filler_buf);
  }
  cudaStreamDestroy(filler_stream);
  cudaStreamDestroy(stream);
  return 0;
}
