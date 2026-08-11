// benchmarks/p2p_gather_bench.cu
//
// Single-launch P2P run-list gather vs the per-run cudaMemcpyPeerAsync /
// cudaMemcpy2DAsync loop it replaces. CUDA-only: the primitive issues peer
// reads from inside one kernel, so there is nothing to measure on the host.
//
// On a single-GPU box this simulates the peer source as another device
// allocation and uses cudaMemcpyDeviceToDevice for the baseline; the launch
// overhead it measures is exactly what the single-launch kernel removes. On a
// real multi-GPU system, point `src_ptrs[i]` at peer memory and enable peer
// access before running — the kernel and the measurement are unchanged.
//
// Built only when VKERNELS_BUILD_BENCHMARKS=ON and a CUDA toolkit is present.
// No external benchmark dependency: timing is raw cudaEvent, with warmup and
// a median over several iterations to cut launch-tick variance.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/comm/p2p_gather.hpp"
#include "vkernels/comm/p2p_gather_cuda.hpp"

using vkernels::comm::Gather2DRun;
using vkernels::comm::cuda::p2p_gather_runs;
using vkernels::comm::cuda::p2p_gather_runs_2d;

namespace {

struct Timings {
  float baseline_ms = 0.0f;  // per-run cudaMemcpy{Peer,2D}Async loop
  float single_ms = 0.0f;    // single-launch p2p_gather_runs[_2d]
};

// Median elapsed time (ms) for `iters` timed invocations of `fn` on `stream`,
// after a few warmups. The whole loop sits between one record/sync pair so
// the result reflects per-iteration work, not per-call host/device round-trips.
template <typename Fn>
float time_median(cudaStream_t stream, int iters, Fn fn) {
  for (int w = 0; w < 3; ++w) fn();
  cudaEvent_t b, e;
  cudaEventCreate(&b);
  cudaEventCreate(&e);
  std::vector<float> ms;
  ms.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    cudaEventRecord(b, stream);
    fn();
    cudaEventRecord(e, stream);
    cudaEventSynchronize(e);
    float dt = 0.0f;
    cudaEventElapsedTime(&dt, b, e);
    ms.push_back(dt);
  }
  std::sort(ms.begin(), ms.end());
  cudaEventDestroy(b);
  cudaEventDestroy(e);
  return ms[ms.size() / 2];
}

// 1-D sweep: `count` disjoint runs of `run_bytes` each, total = count*run_bytes.
// The destination and a single peer-equivalent source allocation are both on
// the device; each run reads a different source slice.
Timings bench_1d(cudaStream_t stream, std::size_t count, std::size_t run_bytes) {
  const std::size_t total = count * run_bytes;
  std::uint8_t* dst = nullptr;
  std::uint8_t* src = nullptr;  // one contiguous "peer" region, sliced per run
  cudaMalloc(&dst, total);
  cudaMalloc(&src, total);

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
  // The driver-call/launch overhead this loop pays is what the kernel removes.
  t.baseline_ms = time_median(stream, 50, [&] {
    for (std::size_t i = 0; i < count; ++i)
      cudaMemcpyPeerAsync(dst + dst_offsets[i], 0, src_ptrs[i], 0, lengths[i], stream);
  });
  // Single launch: the whole run list in one kernel.
  t.single_ms = time_median(stream, 50, [&] {
    p2p_gather_runs(dst, total, src_ptrs.data(), dst_offsets.data(), lengths.data(),
                    count, stream);
  });

  cudaFree(dst);
  cudaFree(src);
  return t;
}

// 2-D sweep: `count` strided tiles, each `height` x `width` bytes copied from a
// row-major peer region (src stride `src_stride`) into `dst` (dst stride
// `dst_stride`). Baseline is one cudaMemcpy2DAsync per tile.
Timings bench_2d(cudaStream_t stream, std::size_t count, std::size_t width,
                 std::size_t height, std::size_t src_stride, std::size_t dst_stride) {
  const std::size_t src_bytes = count * height * src_stride;
  const std::size_t dst_bytes = count * height * dst_stride;
  std::uint8_t* dst = nullptr;
  std::uint8_t* src = nullptr;
  cudaMalloc(&dst, dst_bytes);
  cudaMalloc(&src, src_bytes);

  std::vector<Gather2DRun> runs(count);
  for (std::size_t i = 0; i < count; ++i)
    runs[i] = {src + i * height * src_stride, src_stride, i * height * dst_stride,
               dst_stride, width, height};

  Timings t;
  t.baseline_ms = time_median(stream, 50, [&] {
    for (std::size_t i = 0; i < count; ++i)
      cudaMemcpy2DAsync(dst + runs[i].dst_offset, dst_stride, runs[i].src, src_stride,
                        width, height, cudaMemcpyDeviceToDevice, stream);
  });
  t.single_ms = time_median(stream, 50, [&] {
    p2p_gather_runs_2d(dst, dst_bytes, runs.data(), count, stream);
  });

  cudaFree(dst);
  cudaFree(src);
  return t;
}

void print_header(const char* title) {
  std::printf("\n%s\n", title);
  std::printf("%-7s %-10s %-10s %-10s %-12s %-12s %s\n",
              "count", "run_bytes", "width", "height", "baseline_us", "single_us", "speedup");
  std::printf("%s\n",
              "-------------------------------------------------------------------------------");
}

void print_row(std::size_t count, std::size_t run_bytes, std::size_t width,
               std::size_t height, const Timings& t) {
  const double base_us = t.baseline_ms * 1e3;
  const double single_us = t.single_ms * 1e3;
  const double speedup = t.single_ms > 0 ? t.baseline_ms / t.single_ms : 0.0;
  std::printf("%-7zu %-10zu %-10zu %-10zu %-12.2f %-12.2f %.2fx\n",
              count, run_bytes, width, height, base_us, single_us, speedup);
}

}  // namespace

int main() {
  if (cudaSetDevice(0) != cudaSuccess) {
    std::fprintf(stderr, "p2p_gather_bench: no CUDA device\n");
    return 1;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::printf("# p2p_gather_bench  device=%s sm=%d.%d\n",
              prop.name, prop.major, prop.minor);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // 1-D: fixed total (4 MiB), sweep run count to vary fragmentation. The
  // single-launch kernel pulls ahead as run count (and thus per-run launch
  // overhead in the baseline) grows. All counts fit the __constant__ cap
  // (2048), so these are the fast path.
  print_header("1-D gather: 4 MiB total, sweep fragmentation (__constant__ path)");
  constexpr std::size_t kTotal = 4u * 1024u * 1024u;
  for (std::size_t count : {1u, 8u, 32u, 128u, 512u, 2048u}) {
    const std::size_t run_bytes = kTotal / count;
    const Timings t = bench_1d(stream, count, run_bytes);
    print_row(count, run_bytes, 0, 0, t);
  }

  // 1-D: fixed page size (4 KiB), sweep count within the __constant__ cap.
  print_header("1-D gather: 4 KiB runs, sweep count (__constant__ path)");
  for (std::size_t count : {1u, 64u, 256u, 1024u, 2048u}) {
    const Timings t = bench_1d(stream, count, 4u * 1024u);
    print_row(count, 4u * 1024u, 0, 0, t);
  }

  // 1-D: above the __constant__ cap (2048). These fall back to a
  // stream-ordered cudaMallocAsync staging buffer — still one kernel, now
  // with a metadata allocation that very large lists justify. The single
  // launch still beats the per-run loop because N is so large.
  print_header("1-D gather: 4 KiB runs, above __constant__ cap (cudaMallocAsync fallback)");
  for (std::size_t count : {4096u, 8192u}) {
    const Timings t = bench_1d(stream, count, 4u * 1024u);
    print_row(count, 4u * 1024u, 0, 0, t);
  }

  // 2-D: strided tiles, sweep count within the __constant__ cap (1024).
  print_header("2-D gather: 64x512 strided tiles, sweep count (__constant__ path)");
  for (std::size_t count : {1u, 16u, 64u, 256u}) {
    const Timings t = bench_2d(stream, count, 512u, 64u, 1024u, 1024u);
    print_row(count, 0, 512u, 64u, t);
  }

  cudaStreamDestroy(stream);
  return 0;
}
