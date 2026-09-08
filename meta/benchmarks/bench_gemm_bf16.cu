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
// Built whenever a CUDA toolkit is present; the _cuda target suffix keeps
// it distinct from the HIP bench, so both build side by side on a
// dual-toolkit box (see meta/benchmarks/CMakeLists.txt).

#include "vkernels/kernels/gemm_bf16.hpp"

// Roofline + timing machinery shared with bench_dsa_topk_logits.cu.
#include "bench_cuda_common.cuh"
using namespace vkernels_bench;  // leaf harness TU

// Explicit-tile dispatcher (not in the public header; forward-declared
// here, same as test_gemm_bf16_correct.cu).
namespace vkernels::kernels::cuda {
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads);
}  // namespace vkernels::kernels::cuda

// bf16 round-to-nearest-even store comes from device_numeric.cuh (the one
// definition the CPU oracle's .cpp copies and every device TU mirrors).
using vkernels::kernels::f2bf;

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
