// meta/benchmarks/bench_gemm_bf16.hip
//
// Micro-benchmark for the bf16 MFMA GEMM (issue #29) on AMD gfx942
// (MI300A). Reports, per Kimi-K3 projection shape:
//   * median / min latency (us),
//   * achieved TFLOP/s  = 2*M*N*K / t,
//   * effective GB/s    = actual HBM bytes / t  (A read once per N-tile,
//     B once per M-tile, C once; no cross-tile reuse),
//   * arithmetic intensity and the binding resource vs the MI300A roof
//     (≈1307 TFLOP/s bf16, ≈5300 GB/s HBM3, ridge ≈245 FLOP/byte).
//
// It also runs a per-shape tile autotuner over the compiled tiles via the
// explicit-config dispatcher hip::gemm_bf16_with_config, so the
// gemm_bf16_config_for table in src/c can be regenerated on device.
//
// Built only with VKERNELS_HAS_HIP + VKERNELS_BUILD_BENCHMARKS.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/gemm_bf16.hpp"

// Explicit-tile dispatcher (not in the public header; forward-declared
// here, same as test_gemm_bf16_correct.hip and the moe_aux pattern).
namespace vkernels::kernels::hip {
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads);
}  // namespace vkernels::kernels::hip

static void check_hip(hipError_t err, const char* ctx) {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", ctx, hipGetErrorString(err));
    std::exit(1);
  }
}

struct GpuInfo { std::string name; int arch; double tflops, bw; };
static GpuInfo get_gpu_info() {
  hipDeviceProp_t p;
  check_hip(hipGetDeviceProperties(&p, 0), "props");
  GpuInfo i;
  i.name = p.name;
  if (std::strstr(p.gcnArchName, "gfx942"))      { i.arch = 942; i.tflops = 1307; i.bw = 5300; }
  else if (std::strstr(p.gcnArchName, "gfx90a")) { i.arch = 90;  i.tflops = 383;  i.bw = 3200; }
  else                                            { i.arch = 0;   i.tflops = 100;  i.bw = 1000; }
  return i;
}

struct BenchResult { double min_us, median_us, mean_us; };

static BenchResult bench_us(const std::function<void()>& launch,
                            hipEvent_t start, hipEvent_t stop,
                            int warmup = 5, int max_iters = 500) {
  (void)hipDeviceSynchronize();
  for (int i = 0; i < warmup; ++i) launch();
  std::vector<float> times;
  int iters = 20;
  while (iters <= max_iters) {
    times.clear();
    for (int i = 0; i < iters; ++i) {
      hipEventRecord(start);
      launch();
      hipEventRecord(stop);
      hipEventSynchronize(stop);
      float ms = 0;
      hipEventElapsedTime(&ms, start, stop);
      times.push_back(ms * 1000.0f);
    }
    double sum = 0;
    for (double t : times) sum += t;
    double mean = sum / iters;
    double sum2 = 0;
    for (double t : times) sum2 += t * t;
    double var = sum2 / iters - mean * mean;
    double stddev = std::sqrt(std::max(0.0, var));
    if ((mean > 0 && stddev / mean < 0.03) || iters >= max_iters) {
      std::sort(times.begin(), times.end());
      return {times[0], times[iters / 2], mean};
    }
    iters *= 2;
  }
  return {};
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
static void bench_shapes(const GpuInfo& info, hipEvent_t start,
                         hipEvent_t stop) {
  struct NK { int N, K; };
  const NK k3[] = {
      {6288, 7168}, {3584, 7168}, {896, 7168}, {2112, 7168}, {1536, 7168},
      {7168, 1536}, {7168, 768}, {7168, 3584}, {2304, 1536}, {3072, 512},
      {1536, 128},
  };
  std::printf("\n=== hip::gemm_bf16 (config-selected) ===\n");
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
      check_hip(hipMalloc(&dA, (size_t)M * K * 2), "A");
      check_hip(hipMalloc(&dB, (size_t)K * N * 2), "B");
      check_hip(hipMalloc(&dC, (size_t)M * N * 2), "C");
      check_hip(hipMemcpy(dA, A.data(), (size_t)M * K * 2,
                          hipMemcpyHostToDevice), "cpyA");
      check_hip(hipMemcpy(dB, B.data(), (size_t)K * N * 2,
                          hipMemcpyHostToDevice), "cpyB");
      auto L = [&] {
        vkernels::kernels::hip::gemm_bf16((size_t)M, (size_t)N, (size_t)K,
                                          1.0f, dA, dB, 0.0f, dC);
      };
      auto r = bench_us(L, start, stop);
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
      check_hip(hipFree(dA), "fA"); check_hip(hipFree(dB), "fB");
      check_hip(hipFree(dC), "fC");
    }
  }
  std::printf("  Roof: %.0f TFLOP/s bf16, %.0f GB/s HBM, ridge ~%.0f FLOP/B\n",
              info.tflops, info.bw, info.tflops * 1e3 / info.bw);
}

// ---------------------------------------------------------------------------
// Per-shape autotuner: sweep ALL compiled tiles at EVERY serving M and print
// the full matrix (us median) plus the best tile, so gemm_bf16_config_for can
// be regenerated from measured data rather than projection.
static void autotune(const GpuInfo& info, hipEvent_t start, hipEvent_t stop) {
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
      check_hip(hipMalloc(&dA, (size_t)M * K * 2), "A");
      check_hip(hipMalloc(&dB, (size_t)K * N * 2), "B");
      check_hip(hipMalloc(&dC, (size_t)M * N * 2), "C");
      check_hip(hipMemcpy(dA, A.data(), (size_t)M * K * 2,
                          hipMemcpyHostToDevice), "cpyA");
      check_hip(hipMemcpy(dB, B.data(), (size_t)K * N * 2,
                          hipMemcpyHostToDevice), "cpyB");
      double us[kNCfg];
      int best_i = 0; double best_us = 1e18;
      for (int i = 0; i < kNCfg; ++i) {
        const auto& c = cfgs[i];
        auto L = [&] {
          vkernels::kernels::hip::gemm_bf16_with_config(
              (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f, dC,
              c.bm, c.bn, 64, (c.bm / 16) * 64);
        };
        auto r = bench_us(L, start, stop, 3, 80);
        us[i] = r.median_us;
        if (us[i] < best_us) { best_us = us[i]; best_i = i; }
      }
      std::printf("  %6d %6d", N, K);
      for (int i = 0; i < kNCfg; ++i) std::printf(" %9.1f", us[i]);
      std::printf("  %dx%d\n", cfgs[best_i].bm, cfgs[best_i].bn);
      check_hip(hipFree(dA), "fA"); check_hip(hipFree(dB), "fB");
      check_hip(hipFree(dC), "fC");
    }
  }
}

// ---------------------------------------------------------------------------
int main() {
  auto info = get_gpu_info();
  std::printf("GPU: %s  arch=%d  bf16=%.0f TFLOP/s  HBM=%.0f GB/s\n",
              info.name.c_str(), info.arch, info.tflops, info.bw);
  hipEvent_t start, stop;
  hipEventCreate(&start); hipEventCreate(&stop);
  bench_shapes(info, start, stop);
  autotune(info, start, stop);
  hipEventDestroy(start); hipEventDestroy(stop);
  return 0;
}
