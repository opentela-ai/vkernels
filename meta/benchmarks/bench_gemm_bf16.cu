// meta/benchmarks/bench_gemm_bf16.cu
//
// Micro-benchmark for the CUDA wmma bf16 GEMM (issue #29 port) on NVIDIA.
// Reports, per Kimi-K3 projection shape:
//   * median / min latency (us),
//   * achieved TFLOP/s  = 2*M*N*K / t,
//   * effective GB/s    = actual HBM bytes / t  (A read once per N-tile,
//     B once per M-tile, C once; no cross-tile reuse),
//   * arithmetic intensity and the binding resource vs the GB10 roof.
//
// It also runs a per-shape tile autotuner over the compiled tiles via the
// explicit-config dispatcher cuda::gemm_bf16_with_config, so the
// gemm_bf16_config_for table in src/c can be regenerated on device.
//
// On CUDA 13 the clockRate / memoryClockRate fields were removed from
// cudaDeviceProp, so the memory roof is MEASURED here with a read-only copy
// kernel and the bf16 compute roof is MEASURED with a register-resident
// wmma throughput micro-kernel (the tiled GEMM itself is memory-bound at
// AI ~= 2*bn ~= 32 on GB10, right at the ridge, so it cannot probe the
// compute roof). Both are printed so the roofline is grounded in
// measurement, not datasheet arithmetic.
//
// Built only with VKERNELS_HAS_CUDA (and NOT VKERNELS_HAS_HIP -- on a
// dual-GPU box the .hip bench already provides gemm_bf16_bench).

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

#include "vkernels/kernels/gemm_bf16.hpp"

// Explicit-tile dispatcher (not in the public header; forward-declared
// here, same as test_gemm_bf16_correct.cu).
namespace vkernels::kernels::cuda {
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads);
}  // namespace vkernels::kernels::cuda

static void check_cuda(cudaError_t err, const char* ctx) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error (%s): %s\n", ctx, cudaGetErrorString(err));
    std::exit(1);
  }
}

struct GpuInfo { std::string name; int sms; double tflops, bw; };

// Read-only copy micro-kernel: every thread loads + stores a contiguous
// uint4 stride. Used to measure the practical global-memory bandwidth.
__global__ void copy_k(const uint4* __restrict__ src, uint4* __restrict__ dst,
                       int n4) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += gridDim.x * blockDim.x)
    dst[i] = src[i];
}

// Compute-roof micro-kernel: each warp performs `rounds` independent
// 16x16x16 bf16 wmma accumulations on register-resident fragments that are
// filled ONCE and reused, so HBM traffic is O(1) and the kernel is purely
// compute-bound. One warp per block; many blocks to load the SMs. This is
// the only honest way to measure a bf16 compute peak for wmma on this device
// -- the tiled GEMM itself is memory-bound at AI ~= 2*bn (~32 on GB10, right
// at the memory ridge), so it can never be used to probe the compute roof.
#include <cuda_bf16.h>
#include <mma.h>
__global__ void wmma_throughput_k(int rounds, float* __restrict__ sink) {
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

static double measure_bw(size_t bytes) {
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
// empirical peak -- independent of the memory-bound tiled GEMM.
static double measure_tflops(int sms, cudaEvent_t start, cudaEvent_t stop) {
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

static GpuInfo get_gpu_info(cudaEvent_t start, cudaEvent_t stop) {
  cudaDeviceProp p;
  check_cuda(cudaGetDeviceProperties(&p, 0), "props");
  GpuInfo i;
  i.name = p.name;
  i.sms = p.multiProcessorCount;
  i.bw = measure_bw(512ULL * 1024 * 1024);     // 512 MiB read/write
  i.tflops = measure_tflops(i.sms, start, stop);  // empirical bf16 peak
  return i;
}

struct BenchResult { double min_us, median_us, mean_us; };

static BenchResult bench_us(const std::function<void()>& launch,
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

static uint16_t f2bf(float v) {
  uint32_t b; std::memcpy(&b, &v, 4);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}
static float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

// ---------------------------------------------------------------------------
static void bench_shapes(const GpuInfo& info, cudaEvent_t start,
                         cudaEvent_t stop) {
  struct NK { int N, K; };
  const NK k3[] = {
      {6288, 7168}, {3584, 7168}, {896, 7168}, {2112, 7168}, {1536, 7168},
      {7168, 1536}, {7168, 768}, {7168, 3584}, {2304, 1536}, {3072, 512},
      {1536, 128},
  };
  std::printf("\n=== cuda::gemm_bf16 (config-selected) ===\n");
  std::printf("  %5s %6s %6s %9s %9s %9s %9s %5s  %s\n",
              "M", "N", "K", "us(min)", "us(med)", "TFLOP/s", "GB/s",
              "AI",
              "bound");
  for (int M : {5, 8, 16, 32, 64, 8192}) {
    for (const auto& s : k3) {
      const int N = s.N, K = s.K;
      std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N),
          C((size_t)M * N, 0);
      for (int i = 0; i < M * K; ++i) A[i] = f2bf(rnd(1, i) * 0.5f);
      for (int i = 0; i < K * N; ++i) B[i] = f2bf(rnd(2, i) * 0.5f);
      uint16_t *dA, *dB, *dC;
      check_cuda(cudaMalloc(&dA, (size_t)M * K * 2), "A");
      check_cuda(cudaMalloc(&dB, (size_t)K * N * 2), "B");
      check_cuda(cudaMalloc(&dC, (size_t)M * N * 2), "C");
      check_cuda(cudaMemcpy(dA, A.data(), (size_t)M * K * 2,
                            cudaMemcpyHostToDevice), "cpyA");
      check_cuda(cudaMemcpy(dB, B.data(), (size_t)K * N * 2,
                            cudaMemcpyHostToDevice), "cpyB");
      auto L = [&] {
        vkernels::kernels::cuda::gemm_bf16((size_t)M, (size_t)N, (size_t)K,
                                           1.0f, dA, dB, 0.0f, dC);
      };
      // Cap at max_iters=80: on GB10/CUDA-13, batches of 160+ rapid
      // re-launches drop cudaEventElapsedTime to 0 (the autotuner, also
      // capped at 80, is unaffected). Keeps per-launch event medians.
      auto r = bench_us(L, start, stop, 5, 80);
      double tflops = 2.0 * M * N * K / (r.median_us / 1e6) / 1e12;
      // Actual HBM traffic: A read once per N-tile (ceil(N/bn)), B once per
      // M-tile (ceil(M/bm)), C written once. No cross-tile reuse, so this is
      // far above the ideal 2*(M*K+K*N+M*N) once M outgrows one M-tile.
      int cbm = 0, cbn = 0, cbk = 0, cth = 0;
      vkernels::kernels::gemm_bf16_config_for((size_t)M, (size_t)N,
                                              (size_t)K, &cbm, &cbn,
                                              &cbk, &cth);
      double bytes =
          2.0 * ((double)M * K * ((N + cbn - 1) / cbn) +
                 (double)K * N * ((M + cbm - 1) / cbm) + (double)M * N);
      double gbs = bytes / (r.median_us / 1e6) / 1e9;
      double ai = (2.0 * M * N * K) / bytes;
      const char* bound = ai < (info.tflops * 1e3 / info.bw) ? "mem" : "comp";
      std::printf("  %5d %6d %6d %9.1f %9.1f %9.3f %9.1f %5.1f  %s\n", M, N, K,
                  r.min_us, r.median_us, tflops, gbs, ai, bound);
      check_cuda(cudaFree(dA), "fA"); check_cuda(cudaFree(dB), "fB");
      check_cuda(cudaFree(dC), "fC");
    }
  }
  std::printf("  Roof: %.0f TFLOP/s bf16 (empirical), %.0f GB/s (measured), "
              "ridge ~%.0f FLOP/B\n",
              info.tflops, info.bw, info.tflops * 1e3 / info.bw);
}

// ---------------------------------------------------------------------------
// Per-shape autotuner: sweep ALL compiled tiles at EVERY serving M and print
// the full matrix (us median) plus the best tile, so gemm_bf16_config_for can
// be regenerated from measured data rather than projection.
static void autotune(const GpuInfo& info, cudaEvent_t start, cudaEvent_t stop) {
  (void)info;
  struct NK { int N, K; };
  const NK k3[] = {
      {6288, 7168}, {3584, 7168}, {896, 7168}, {2112, 7168}, {1536, 7168},
      {7168, 1536}, {7168, 768}, {7168, 3584}, {2304, 1536}, {3072, 512},
      {1536, 128},
  };
  struct Cfg { int bm, bn; };
  const Cfg cfgs[] = {{16, 16}, {16, 64}, {16, 128}, {32, 64}, {32, 128},
                      {64, 64},  {64, 128}};
  constexpr int kNCfg = sizeof(cfgs) / sizeof(cfgs[0]);
  const int Ms[] = {5, 8, 16, 32, 64};
  for (int M : Ms) {
    std::printf("\n=== tile autotuner (M=%d): us(med) per tile ===\n", M);
    std::printf("  %6s %6s", "N", "K");
    for (const auto& c : cfgs) { char nm[16];
      std::snprintf(nm, sizeof(nm), "%dx%d", c.bm, c.bn);
      std::printf(" %8s", nm); }
    std::printf("  %s\n", "best");
    for (const auto& s : k3) {
      const int N = s.N, K = s.K;
      std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N),
          C((size_t)M * N, 0);
      for (int i = 0; i < M * K; ++i) A[i] = f2bf(rnd(1, i) * 0.5f);
      for (int i = 0; i < K * N; ++i) B[i] = f2bf(rnd(2, i) * 0.5f);
      uint16_t *dA, *dB, *dC;
      check_cuda(cudaMalloc(&dA, (size_t)M * K * 2), "A");
      check_cuda(cudaMalloc(&dB, (size_t)K * N * 2), "B");
      check_cuda(cudaMalloc(&dC, (size_t)M * N * 2), "C");
      check_cuda(cudaMemcpy(dA, A.data(), (size_t)M * K * 2,
                            cudaMemcpyHostToDevice), "cpyA");
      check_cuda(cudaMemcpy(dB, B.data(), (size_t)K * N * 2,
                            cudaMemcpyHostToDevice), "cpyB");
      double us[kNCfg];
      int best_i = 0; double best_us = 1e18;
      for (int i = 0; i < kNCfg; ++i) {
        const auto& c = cfgs[i];
        auto L = [&] {
          vkernels::kernels::cuda::gemm_bf16_with_config(
              (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f, dC,
              c.bm, c.bn, 64, (c.bm / 16) * (c.bn / 16) * 32);
        };
        auto r = bench_us(L, start, stop, 3, 80);
        us[i] = r.median_us;
        if (us[i] < best_us) { best_us = us[i]; best_i = i; }
      }
      std::printf("  %6d %6d", N, K);
      for (int i = 0; i < kNCfg; ++i) std::printf(" %9.1f", us[i]);
      std::printf("  %dx%d\n", cfgs[best_i].bm, cfgs[best_i].bn);
      check_cuda(cudaFree(dA), "fA"); check_cuda(cudaFree(dB), "fB");
      check_cuda(cudaFree(dC), "fC");
    }
  }
}

// ---------------------------------------------------------------------------
int main() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start); cudaEventCreate(&stop);
  auto info = get_gpu_info(start, stop);
  std::printf("GPU: %s  SMs=%d  bf16=%.0f TFLOP/s (empirical)  "
              "mem=%.0f GB/s (measured)\n",
              info.name.c_str(), info.sms, info.tflops, info.bw);
  bench_shapes(info, start, stop);
  autotune(info, start, stop);
  cudaEventDestroy(start); cudaEventDestroy(stop);
  return 0;
}
