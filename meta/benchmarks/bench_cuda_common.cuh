// meta/benchmarks/bench_cuda_common.cuh
//
// Host-side benchmarking machinery shared verbatim by the CUDA micro-bench
// harnesses (bench_gemm_bf16.cu, bench_dsa_topk_logits.cu): the checked-CUDA
// helper, the measured roofline (bandwidth + bf16 wmma compute peak), and the
// adaptive cudaEvent timing loop. The HIP benches deliberately keep their own
// copies -- their device properties (arch/CUs) and timing fields (cv) differ
// -- so this header is CUDA-only, not a cross-backend abstraction.
//
// CUDA 13 removed clockRate / memoryClockRate from cudaDeviceProp, so both
// roofs are MEASURED here rather than read from datasheet fields:
//   * memory  -- a read-only copy kernel (copy_k) over 512 MiB;
//   * compute -- a register-resident wmma throughput kernel
//     (wmma_throughput_k), the only honest bf16 peak probe: the tiled GEMM
//     sits at AI ~= 2*bn (~32 on GB10, right at the ridge), so it can never
//     saturate the tensor cores, and the dsa kernel is memory-bound at low AI.
// Both numbers are printed by the harnesses so the roofline is grounded in
// measurement, not projection.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace vkernels_bench {

inline void check_cuda(cudaError_t err, const char* ctx) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", ctx, cudaGetErrorString(err));
    std::exit(1);
  }
}

struct GpuInfo { std::string name; int sms; double tflops, bw; };

// Read-only copy micro-kernel: every thread loads + stores a contiguous
// uint4 stride. Used to measure the practical global-memory bandwidth.
__global__ inline void copy_k(const uint4* __restrict__ src,
                              uint4* __restrict__ dst, int n4) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += gridDim.x * blockDim.x)
    dst[i] = src[i];
}

// Compute-roof micro-kernel: each warp performs `rounds` independent
// 16x16x16 bf16 wmma accumulations on register-resident fragments that are
// filled ONCE and reused, so HBM traffic is O(1) and the kernel is purely
// compute-bound. One warp per block; many blocks to load the SMs.
#include <cuda_bf16.h>
#include <mma.h>
__global__ inline void wmma_throughput_k(int rounds, float* __restrict__ sink) {
  using namespace nvcuda::wmma;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
  fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b;
  fragment<accumulator, 16, 16, 16, float> c;
  fill_fragment(a, __float2bfloat16(1.0f));
  fill_fragment(b, __float2bfloat16(1.0f));
  fill_fragment(c, 0.0f);
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) mma_sync(c, a, b, c);
  if (lane == 0) sink[warp] = c.x[0];   // defeat dead-code elimination
}

inline double measure_bw(size_t bytes) {
  const int n4 = (int)(bytes / sizeof(uint4));
  uint4 *dS, *dD;
  check_cuda(cudaMalloc(&dS, (size_t)n4 * sizeof(uint4)), "bw src");
  check_cuda(cudaMalloc(&dD, (size_t)n4 * sizeof(uint4)), "bw dst");
  check_cuda(cudaMemset(dS, 0xAB, (size_t)n4 * sizeof(uint4)), "bw fill");
  int block = 256, grid = std::min(2048, (n4 + block - 1) / block);
  copy_k<<<grid, block>>>(dS, dD, n4);
  check_cuda(cudaDeviceSynchronize(), "bw warmup");
  // Time several repeats; each repeat copies the same buffer (bandwidth, not
  // a one-shot transfer), so we divide by the number of repeats.
  const int reps = 50;
  cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for (int r = 0; r < reps; ++r) copy_k<<<grid, block>>>(dS, dD, n4);
  cudaEventRecord(e);
  cudaEventSynchronize(e);
  float ms = 0; cudaEventElapsedTime(&ms, s, e);
  cudaEventDestroy(s); cudaEventDestroy(e);
  double gb = (double)bytes * 2.0 * reps / 1e9;   // read + write
  double bw = gb / (ms / 1e3);
  cudaFree(dS); cudaFree(dD);
  return bw;
}

// Probe the bf16 compute roof with a register-resident wmma micro-kernel
// (see wmma_throughput_k above). Returns the achieved median TFLOP/s as an
// empirical peak -- independent of the memory-bound tiled kernels.
inline double measure_tflops(int sms, cudaEvent_t start, cudaEvent_t stop) {
  const int rounds = 100000;          // MMAs per warp; dominates launch oh
  const int blocks = sms * 16;        // one warp/block, load the SMs
  const int warps = blocks;           // 1 warp/block
  float* sink = nullptr;
  check_cuda(cudaMalloc(&sink, (size_t)warps * sizeof(float)), "pk sink");
  for (int i = 0; i < 3; ++i)         // warmup
    wmma_throughput_k<<<blocks, 32>>>(rounds, sink);
  check_cuda(cudaDeviceSynchronize(), "pk warmup");
  std::vector<float> ts;
  for (int i = 0; i < 30; ++i) {
    cudaEventRecord(start);
    wmma_throughput_k<<<blocks, 32>>>(rounds, sink);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0; cudaEventElapsedTime(&ms, start, stop);
    ts.push_back(ms * 1000.0f);
  }
  std::sort(ts.begin(), ts.end());
  double med = ts[ts.size() / 2];
  cudaFree(sink);
  // FLOPs = warps * rounds * 2*(16^3) for the 16x16x16 mma.
  double flops = (double)warps * rounds * 2.0 * 4096.0;
  return flops / (med / 1e6) / 1e12;
}

inline GpuInfo get_gpu_info(cudaEvent_t start, cudaEvent_t stop) {
  cudaDeviceProp p;
  check_cuda(cudaGetDeviceProperties(&p, 0), "props");
  GpuInfo i;
  i.name = p.name;
  i.sms = p.multiProcessorCount;
  i.bw = measure_bw(512ULL * 1024 * 1024);        // 512 MiB read/write
  i.tflops = measure_tflops(i.sms, start, stop);  // empirical bf16 peak
  return i;
}

struct BenchResult { double min_us, median_us, mean_us; };

// Adaptive per-launch event timing: warmup, then double the batch size until
// the per-launch stddev/mean ratio drops under 3% (or max_iters is hit) and
// report min/median/mean of the accepted batch.
inline BenchResult bench_us(const std::function<void()>& launch,
                            cudaEvent_t start, cudaEvent_t stop,
                            int warmup = 5, int max_iters = 500) {
  (void)cudaDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) launch();
  (void)cudaDeviceSynchronize();           // drain warmup before timing
  int iters = 20;
  while (iters <= max_iters) {
    std::vector<float> times;
    for (int i = 0; i < iters; ++i) {
      cudaEventRecord(start);
      launch();
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      // On GB10 the driver occasionally returns ms==0 from
      // cudaEventElapsedTime on a kernel that plainly takes >0. A zero is
      // always spurious here -- drop it before it can poison the median or
      // inflate the stddev/mean ratio (which would otherwise force the
      // adaptive loop to grow until the zeros dominate).
      if (ms > 0.0f) times.push_back(ms * 1000.0f);
    }
    // If the whole batch came back zero (a persistent driver glitch),
    // clear any sticky error and retry the batch once.
    if (times.empty()) {
      (void)cudaGetLastError();
      for (int i = 0; i < iters; ++i) {
        cudaEventRecord(start);
        launch();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if (ms > 0.0f) times.push_back(ms * 1000.0f);
      }
    }
    if (times.empty()) break;             // give up, fall back to chrono
    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / times.size();
    double sum2 = 0;
    for (double t : times) sum2 += t * t;
    double var = sum2 / times.size() - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if (stddev / mean < 0.03 || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[times.size() / 2], mean};
    }
    iters *= 2;
  }
  // Last-resort fallback (persistent event failure): one chrono-timed
  // single launch between syncs. For kernels >= ~10 us the host-side
  // overhead is small relative to the kernel, so this stays honest.
  (void)cudaDeviceSynchronize();
  (void)cudaGetLastError();
  auto t0 = std::chrono::high_resolution_clock::now();
  launch();
  (void)cudaDeviceSynchronize();
  auto t1 = std::chrono::high_resolution_clock::now();
  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  return {us, us, us};
}

// Deterministic pseudo-random in [0, 2) -- identical series on host and
// device, so harness inputs are reproducible run to run.
inline float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

}  // namespace vkernels_bench
