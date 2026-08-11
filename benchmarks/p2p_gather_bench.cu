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

// Memory-bound filler kernels run concurrently on a second stream in
// --concurrent mode: they compete with the gather for SMs and HBM
// bandwidth, the way attention kernels do during model execution.
__global__ void fill_kernel(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.0f;
}

// Persistent filler: occupies kFillBlocks SMs for the whole bench (the
// finite burst of fill_kernel launches drained in ~56 ms, i.e. before the
// measured sections even started — no real overlap). Each thread streams
// float4 stores over the buffer in a grid-stride loop until the host sets
// *stop (a 4-byte cudaMemsetAsync on the filler stream right before the
// final sync) or a generous clock64 timeout fires. 64 blocks leaves ~half
// the H100's SMs free, so the timed gather kernels still get slots and the
// comparison measures contention rather than full starvation.
__global__ void fill_persistent_kernel(float4* p, int n4, const volatile int* stop) {
  long long t0 = clock64();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  float4 v = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  while (*stop == 0 && clock64() - t0 < 120LL * 1890000000LL) {
    p[i] = v;
    v.x += 1.0f; v.y += 1.0f; v.z += 1.0f; v.w += 1.0f;
    i += stride;
    if (i >= n4) i -= n4;  // wrap within the buffer
  }
}

// Trace helper for VK_BENCH_TRACE (debugging the concurrent filler).
double trace_ms_since_start() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
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

// If --concurrent, launch the persistent filler on `s2` writing to `buf` so
// the device stays busy with it while the timed work on `s1` runs. `stop`
// is a device int the host clears to 1 (cudaMemsetAsync on `s2`) before
// cudaStreamSynchronize(s2), releasing the SMs. The buffer is owned by the
// caller (never cudaFree it here: cudaFree would block the host until the
// enqueued fills finish, serialising the very overlap this mode exists to
// create); free it after cudaStreamSynchronize(s2).
void maybe_start_filler(bool concurrent, float* buf, int n, int* stop,
                        cudaStream_t s2) {
  if (!concurrent) return;
  // Number of blocks (SM slots) the persistent filler occupies. Tunable via
  // VK_BENCH_FILL_BLOCKS to model different concurrent-load levels; the
  // default 128 puts one 256-thread block on each H100 SM, leaving the
  // gather to contend for the remaining resident-block slots.
  int blocks = 128;
  if (const char* e = std::getenv("VK_BENCH_FILL_BLOCKS")) blocks = std::atoi(e);
  if (blocks < 1) blocks = 1;
  cudaMemsetAsync(stop, 0, sizeof(int), s2);
  fill_persistent_kernel<<<blocks, 256, 0, s2>>>(
      reinterpret_cast<float4*>(buf), n / 4, stop);
}

// 1-D sweep: `count` disjoint runs of `run_bytes` each, total = count*run_bytes.
// 1-D sweep: `count` disjoint runs of `run_bytes` each, total = count*run_bytes
// into pre-allocated scratch `dst`/`src` (owned by main: no cudaMalloc/cudaFree
// here — synchronous memory ops wait for device quiescence, which with the
// persistent filler running would block the host for the whole filler
// duration). Times the per-run copy loop (baseline), the forced kernel, and
// the adaptive path; reports host-enqueue time for the adaptive path.
Timings bench_1d(cudaStream_t stream, std::size_t count, std::size_t run_bytes,
                 int iters, std::uint8_t* dst, std::uint8_t* src) {
  const std::size_t total = count * run_bytes;

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
  if (std::getenv("VK_BENCH_TRACE"))
    std::fprintf(stderr, "[%.0f ms] bench_1d n=%zu baseline done\n", trace_ms_since_start(), count);

  // Forced kernel: the PR #5 single-launch path, measured directly.
  set_gather_dispatch(GatherDispatchMode::kForceKernel, 1);
  time_median(stream, iters, [&] {
    p2p_gather_runs(dst, total, src_ptrs.data(), dst_offsets.data(),
                    lengths.data(), count, stream);
  }, &t.kernel_ms, &t.kernel_host_ms);
  if (std::getenv("VK_BENCH_TRACE"))
    std::fprintf(stderr, "[%.0f ms] bench_1d n=%zu kernel done\n", trace_ms_since_start(), count);

  // Adaptive: the production path — copy engine below the crossover, kernel
  // at/above it (which branch was taken is reported for verification).
  set_gather_dispatch(GatherDispatchMode::kAdaptive, 4);
  t.adaptive_took_kernel = vkernels::comm::prefer_gather_kernel(count, total);
  time_median(stream, iters, [&] {
    p2p_gather_runs(dst, total, src_ptrs.data(), dst_offsets.data(),
                    lengths.data(), count, stream);
  }, &t.adaptive_ms, &t.adaptive_host_ms);
  set_gather_dispatch();  // restore defaults
  if (std::getenv("VK_BENCH_TRACE"))
    std::fprintf(stderr, "[%.0f ms] bench_1d n=%zu adaptive done\n", trace_ms_since_start(), count);
  return t;
}

// 2-D sweep: `count` strided tiles, each `height` x `width` bytes copied from a
// row-major peer region (src stride `src_stride`) into `dst` (dst stride
// `dst_stride`). Baseline is one cudaMemcpy2DAsync per tile. dst/src are the
// shared scratch buffers from main (see bench_1d).
Timings bench_2d(cudaStream_t stream, std::size_t count, std::size_t width,
                 std::size_t height, std::size_t src_stride,
                 std::size_t dst_stride, int iters, std::uint8_t* dst,
                 std::uint8_t* src) {
  const std::size_t src_bytes = count * height * src_stride;
  const std::size_t dst_bytes = count * height * dst_stride;

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
  // Report the branch the 2-D dispatch itself takes (strided model).
  t.adaptive_took_kernel =
      vkernels::comm::prefer_gather_kernel(count, count * width * height,
                                           /*strided=*/true);
  time_median(stream, iters, [&] {
    p2p_gather_runs_2d(dst, dst_bytes, runs.data(), count, stream);
  }, &t.adaptive_ms, &t.adaptive_host_ms);
  set_gather_dispatch();

  return t;
}

// Plan timing: prepare once (reporting the one-time host+device prep cost),
// then time `repeats` execute() calls — the KVAAS "one run list, 40 layers"
// pattern. Reports the median per-execute device time and the total, so the
// per-layer overhead of validation/allocation/H2D is visible as zero.
// dst/src are the shared scratch buffers from main; the plan borrows them
// (it must outlive its streams, and main frees them after the filler).
// The plan is created BEFORE the filler launches (its prepare's cudaMalloc
// and its destructor's cudaFree are synchronous and would otherwise block
// for the filler's lifetime) and destroyed AFTER the filler is released.
// Returns the one-time prepare host time; the caller prints it with the
// execute() timing.
double bench_plan_prepare(std::size_t count, std::size_t run_bytes,
                          std::uint8_t* dst, std::uint8_t* src,
                          P2PGatherPlan1D** out) {
  const std::size_t total = count * run_bytes;
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
  *out = new P2PGatherPlan1D(dst, total, src_ptrs.data(), dst_offsets.data(),
                             lengths.data(), count);
  auto p1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(p1 - p0).count();
}

// KVAAS layer-relative 2-D plan: prepare ONCE with per-page base pointers,
// then time `repeats` execute() calls with distinct layer offsets. Reports
// the one-time prepare cost and the median per-execute device/host time.
// This is the "192 pages, 40 layers" pattern from issue #8.
double bench_plan2d_kvaas(std::size_t pages, std::size_t layer_bytes,
                          std::size_t page_stride, std::uint8_t* dst,
                          std::uint8_t* src, P2PGatherPlan2D** out) {
  std::vector<Gather2DRun> runs(pages);
  for (std::size_t p = 0; p < pages; ++p)
    runs[p] = {src + p * page_stride, page_stride, p * layer_bytes, layer_bytes,
               layer_bytes, 1u};
  auto p0 = std::chrono::steady_clock::now();
  *out = new P2PGatherPlan2D(dst, pages * layer_bytes, runs.data(), pages);
  auto p1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(p1 - p0).count();
}

// Uneven-height benchmark: builds runs with a range of heights and reports
// total launched blocks versus useful (non-return) blocks. Heights are
// assigned to consecutive runs: [1, 1, ..., 2, 2, ..., max_h, max_h].
void bench_2d_uneven(cudaStream_t stream, std::size_t count,
                     std::size_t width, std::size_t max_height,
                     std::size_t src_stride, std::size_t dst_stride,
                     int iters, std::uint8_t* dst, std::uint8_t* src) {
  std::vector<Gather2DRun> runs(count);
  std::size_t total_height = 0;
  for (std::size_t i = 0; i < count; ++i) {
    // Height increases from 1 to max_height, recycling for large counts.
    const std::size_t h = 1 + (i % max_height);
    runs[i] = {src + i * max_height * src_stride, src_stride,
               i * max_height * dst_stride, dst_stride, width, h};
    total_height += h;
  }

  // Useful rows = sum of heights; useless rows = count * max_height - total_height.
  // Launched blocks (kernel): tiles(ceil(width/16)) * max_height * count
  // (vectorized path); useful row-blocks = sum over runs of tiles(ceil(width/16)) * height(r).
  const std::size_t max_units = (width + 15u) / 16u;  // vectorized units per row
  const std::size_t blocks_x = max_units > 0 ? (max_units + 255u) / 256u : 1u;
  const std::size_t launched = blocks_x * max_height * count;
  const std::size_t useful = blocks_x * total_height;

  std::printf("\nUneven-height 2-D: %zu runs, width=%zu, max_h=%zu\n",
              count, width, max_height);
  std::printf("  total rows=%zu  max_rows=%zu  useless=%zu (%.1f%%)\n",
              total_height, max_height * count,
              max_height * count - total_height,
              100.0 * (1.0 - static_cast<double>(total_height) /
                               static_cast<double>(max_height * count)));
  std::printf("  launched blocks=%zu  useful blocks=%zu  waste=%.1f%%\n",
              launched, useful,
              100.0 * (1.0 - static_cast<double>(useful) /
                               static_cast<double>(launched)));

  // Time via the one-shot 2-D API (same as the adaptive path).
  Timings t;
  // dst capacity must cover all runs: each occupies at most
  // max_height * dst_stride bytes.
  const std::size_t total = count * max_height * dst_stride;
  set_gather_dispatch(GatherDispatchMode::kForceKernel, 1);
  float h = 0.0f;
  time_median(stream, iters, [&] {
    p2p_gather_runs_2d(dst, total, runs.data(), count, stream);
  }, &t.kernel_ms, &t.kernel_host_ms);

  set_gather_dispatch(GatherDispatchMode::kAdaptive, 4);
  t.adaptive_took_kernel =
      vkernels::comm::prefer_gather_kernel(count, total,
                                           /*strided=*/true);
  time_median(stream, iters, [&] {
    p2p_gather_runs_2d(dst, total, runs.data(), count, stream);
  }, &t.adaptive_ms, &t.adaptive_host_ms);
  set_gather_dispatch();

  print_row(count, width, t);
}

void print_header(const char* title) {
  std::printf("\n%s\n", title);
  std::printf("%-6s %-9s %-11s %-11s %-11s %-10s %-11s %-11s %s\n",
              "count", "run_bytes", "baseline_us", "kernel_us", "adaptive_us",
              "adap_kern", "adap_host_us", "kernel_host_us", "speedup");
  std::printf("%s\n",
              "-----------------------------------------------------------------------------");
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
  int* filler_stop = nullptr;
  const int filler_n = static_cast<int>(256u * 1024u * 1024u / sizeof(float));
  if (concurrent) {
    cudaMalloc(&filler_buf, 256u * 1024u * 1024u);
    cudaMalloc(&filler_stop, sizeof(int));
  }

  // Shared scratch buffers, allocated ONCE before the filler and freed only
  // after it is released: synchronous cudaMalloc/cudaFree wait for device
  // quiescence, which with the persistent filler running would block the
  // host for the whole filler duration, serialising the very overlap the
  // concurrent mode exists to measure. 48 MiB covers every sweep (the 2-D
  // and plan sections need at most 16 MiB / 48 MiB).
  const std::size_t k48MiB = 48u * 1024u * 1024u;
  std::uint8_t* dst_buf = nullptr;
  std::uint8_t* src_buf = nullptr;
  cudaMalloc(&dst_buf, k48MiB);
  cudaSetDevice(g_src_dev);
  cudaMalloc(&src_buf, k48MiB);
  cudaSetDevice(0);

  // Issue #6 acceptance sweep: fixed 48 MiB payload across the run counts
  // that bracket the measured 16-32 crossover (1, 2, 4, 8, 16 below it;
  // 32, 64, 192 at/above it).
  // All prepared plans are created BEFORE the filler: synchronous
  // cudaMalloc/cudaFree wait for device quiescence, which with the
  // persistent filler running would block for the filler's lifetime.
  // Plans borrow the shared scratch buffers and must outlive the filler.
  print_header("Prepared plan: prepare once, 40 layer executes (48 MiB total)");
  constexpr std::size_t kPlanRuns = 8;
  P2PGatherPlan1D* plan = nullptr;
  const double plan_prepare_ms =
      bench_plan_prepare(kPlanRuns, k48MiB / kPlanRuns, dst_buf, src_buf,
                         &plan);

  // KVAAS 2-D layer-relative plans: prepare one per sweep count BEFORE the
  // filler, for the same reason. Stored alongside their per-layer row size.
  constexpr std::size_t kLayerRow = 256u * 1024u;  // 256 KiB
  constexpr std::size_t kKvaasStride = 10u * 1024u * 1024u;  // 10 MiB
  const std::size_t kKvaasStrideBench =
      g_src_dev == 0 ? kLayerRow : kKvaasStride;
  constexpr std::size_t kKvaasMaxPages = 192u;
  const std::size_t kvaas_src_bytes = kKvaasMaxPages * kKvaasStrideBench;
  std::uint8_t* kvaas_src = nullptr;
  std::uint8_t* kvaas_dst = nullptr;
  if (g_src_dev != 0) cudaSetDevice(g_src_dev);
  cudaMalloc(&kvaas_src, kvaas_src_bytes);
  if (g_src_dev != 0) cudaSetDevice(0);
  cudaMalloc(&kvaas_dst, kKvaasMaxPages * kLayerRow);

  constexpr std::size_t kKvaasCounts[] = {1u, 2u, 4u, 8u, 16u, 32u, 64u, 192u};
  constexpr std::size_t kNumKvaasCounts = sizeof(kKvaasCounts) / sizeof(kKvaasCounts[0]);
  P2PGatherPlan2D* kvaas_plans[kNumKvaasCounts] = {};
  double kvaas_prep_ms[kNumKvaasCounts] = {};
  for (std::size_t ci = 0; ci < kNumKvaasCounts; ++ci) {
    kvaas_prep_ms[ci] =
        bench_plan2d_kvaas(kKvaasCounts[ci], kLayerRow, kKvaasStrideBench,
                           kvaas_dst, kvaas_src, &kvaas_plans[ci]);
  }

  if (concurrent) {
    // Warm up the one-time default-pool tuning and the gather-kernel JIT
    // BEFORE the filler launches: cudaMemPoolSetAttribute on first use
    // synchronizes the device, so a first timed kernel section would block
    // until the persistent filler exits (~2 min). A tiny warmup gather on
    // the idle device makes the timed sections' pool allocations cheap
    // pool-internal bumps.
    set_gather_dispatch(GatherDispatchMode::kForceKernel, 1);
    const std::size_t kWarm = 256u * 1024u;  // fits the 64x512 2-D tile
    std::uint8_t* wsrc = nullptr;
    std::uint8_t* wdst = nullptr;
    cudaMalloc(&wsrc, kWarm);
    cudaMalloc(&wdst, kWarm);
    const void* wsrcs[1] = {wsrc};
    std::size_t woff[1] = {0};
    std::size_t wlen[1] = {4096};
    p2p_gather_runs(wdst, kWarm, wsrcs, woff, wlen, 1, stream);
    // Also warm the 2-D kernel (first launch synchronizes the device, like
    // the pool tuning): one 64x512 tile.
    Gather2DRun wrun = {wsrc, 4096, 0, 4096, 512, 64};
    p2p_gather_runs_2d(wdst, kWarm, &wrun, 1, stream);
    cudaStreamSynchronize(stream);
    cudaFree(wsrc);
    cudaFree(wdst);
    set_gather_dispatch();
  }
  maybe_start_filler(concurrent, filler_buf, filler_n, filler_stop,
                     filler_stream);
  auto wall0 = std::chrono::steady_clock::now();
  auto wall = [&](const char* tag) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - wall0)
                          .count();
    std::fprintf(stderr, "[wall %.0f ms] %s\n", ms, tag);
  };
  wall("filler launched");
  print_header("1-D gather: 48 MiB fixed payload, issue #6 acceptance sweep");
  for (std::size_t count : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 192u}) {
    const Timings t = bench_1d(stream, count, k48MiB / count, iters,
                               dst_buf, src_buf);
    print_row(count, k48MiB / count, t);
  }
  wall("48MiB sweep done");

  // Existing sweep: 4 MiB total at increasing fragmentation.
  print_header("1-D gather: 4 MiB total, sweep fragmentation");
  constexpr std::size_t k4MiB = 4u * 1024u * 1024u;
  for (std::size_t count : {1u, 8u, 32u, 128u, 512u, 2048u}) {
    const Timings t = bench_1d(stream, count, k4MiB / count, iters,
                               dst_buf, src_buf);
    print_row(count, k4MiB / count, t);
  }
  wall("4MiB sweep done");

  // Fixed page size: 4 KiB runs, sweep count.
  print_header("1-D gather: 4 KiB runs, sweep count");
  for (std::size_t count : {1u, 64u, 256u, 1024u, 2048u}) {
    const Timings t = bench_1d(stream, count, 4u * 1024u, iters,
                               dst_buf, src_buf);
    print_row(count, 4u * 1024u, t);
  }
  wall("4KiB sweep done");

  // 2-D: strided tiles, sweep count.
  print_header("2-D gather: 64x512 strided tiles, sweep count");
  for (std::size_t count : {1u, 16u, 64u, 256u}) {
    const Timings t = bench_2d(stream, count, 512u, 64u, 1024u, 1024u, iters,
                               dst_buf, src_buf);
    print_row(count, 0, t);
  }
  wall("2D sweep done");

  // Prepared plan executes: the KVAAS reuse pattern (one run list, many
  // layer launches). prepare_ms (reported above, measured on the idle
  // device before the filler) is the ONE-TIME cost; the per-execute device
  // and host times are what every layer pays after that (no allocation, no
  // H2D) — timed here under the concurrent filler when --concurrent.
  float plan_dev = 0.0f, plan_host = 0.0f;
  time_median(stream, 25, [&] { plan->execute(stream); }, &plan_dev,
              &plan_host);
  std::printf("  plan  %-5zu %-10zu %-12.3f %-12.2f %-12.2f\n",
              kPlanRuns, k48MiB / kPlanRuns, plan_prepare_ms,
              plan_dev * 1e3f, plan_host * 1e3f);
  wall("plan done");

  // -----------------------------------------------------------------------
  // -----------------------------------------------------------------------
  // KVAAS layer-relative 2-D plan: 192-page, 256-KiB-row, 10-MiB-stride
  // geometry (issue #8 acceptance). Plans were prepared above before the
  // filler; here we time their executes against the one-shot baseline.
  // -----------------------------------------------------------------------
  print_header("2-D KVAAS: 256-KiB rows, layer-relative plan vs one-shot");
  std::printf("%-6s %-12s %-12s %-12s %-12s %-8s %-12s %-12s %s\n",
              "runs", "copy_us", "oneshot_us", "plan_us",
              "plan_adap", "plan_kern", "plan_host_us", "prep_ms",
              "branch");
  std::printf("%s\n", "------------------------------------------------------------"
                    "------------------------------------------------------------");
  for (std::size_t ci = 0; ci < kNumKvaasCounts; ++ci) {
    const std::size_t count = kKvaasCounts[ci];
    // Baseline: one cudaMemcpy2DAsync per page.
    std::vector<Gather2DRun> runs(count);
    for (std::size_t p = 0; p < count; ++p)
      runs[p] = {kvaas_src + p * kKvaasStrideBench, kKvaasStrideBench,
                 p * kLayerRow, kLayerRow, kLayerRow, 1u};
    float copy_us = 0.0f, h0 = 0.0f;
    time_median(stream, iters, [&] {
      for (std::size_t i = 0; i < count; ++i)
        cudaMemcpy2DAsync(kvaas_dst + runs[i].dst_offset, kLayerRow,
                          runs[i].src, kKvaasStrideBench, kLayerRow, 1,
                          cudaMemcpyDefault, stream);
    }, &copy_us, &h0);

    // One-shot adaptive.
    Timings t = bench_2d(stream, count, kLayerRow, 1u, kKvaasStrideBench,
                         kLayerRow, iters, kvaas_dst, kvaas_src);

    // Layer-relative plan: prepared once above, timed here.
    float kv_dev = 0.0f, kv_host = 0.0f;
    time_median(stream, iters,
                [&] { kvaas_plans[ci]->execute(std::size_t(0), stream); },
                &kv_dev, &kv_host);

    bool kv_uses_kernel =
        vkernels::comm::prefer_gather_kernel(count, count * kLayerRow,
                                             /*strided=*/true);
    std::printf("%-6zu %-12.2f %-12.2f %-12.2f %-12.2f %-8s %-12.2f %-12.3f %s\n",
                count, copy_us * 1e3, t.adaptive_ms * 1e3, kv_dev * 1e3,
                t.adaptive_ms * 1e3,
                t.adaptive_took_kernel ? "kernel" : "copy",
                kv_host * 1e3, kvaas_prep_ms[ci],
                kv_uses_kernel ? "kernel" : "copy");
  }
  wall("KVAAS 2D done");

  // Clean up KVAAS plans and buffers (must happen before filler release —
  // destructors do synchronous cudaFree).
  for (std::size_t ci = 0; ci < kNumKvaasCounts; ++ci) delete kvaas_plans[ci];
  cudaFree(kvaas_src);
  cudaFree(kvaas_dst);

  if (concurrent) {
    // Release the persistent filler. The memset must go on a stream that is
    // NOT blocked behind the filler kernel (on the filler stream it would
    // queue after the kernel and never execute — the kernel polls *stop):
    // from the main stream it runs on the copy engine while the filler
    // still occupies SMs, and the filler exits on its next poll. Then wait
    // for it to drain and free the buffer.
    cudaMemsetAsync(filler_stop, 1, sizeof(int), stream);
    cudaStreamSynchronize(filler_stream);
    wall("filler released");
    cudaFree(filler_buf);
    cudaFree(filler_stop);
  }
  delete plan;  // destructor's cudaFree runs after the filler is gone
  cudaFree(dst_buf);
  cudaFree(src_buf);
  cudaStreamDestroy(filler_stream);
  cudaStreamDestroy(stream);
  return 0;
}
