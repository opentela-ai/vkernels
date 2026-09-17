// meta/benchmarks/test_gemm_bf16_correct.hip
//
// End-to-end correctness test for the bf16 MFMA GEMM (issue #29) on
// gfx942, against the CPU reference in vkernels::kernels::gemm_bf16_cpu.
//
// It checks two things:
//   1. The public entry hip::gemm_bf16 (config-selected tile) matches the
//      CPU oracle within bf16 tolerance for every Kimi-K3 projection
//      (N, K) shape at serving M = {8, 64} and warmup M = 8192.
//   2. Every compiled tile (16x16, 16x64, 16x128, 32x64, 32x128, 64x64,
//      64x128) -- exercised only through the explicit-tile
//      hip::gemm_bf16_with_config dispatcher -- also matches the oracle on
//      a small, awkward (M=16, N=100, K=128) shape.
//   3. (issue #77) The cross-tile B-reuse + LDS double-buffer warmup kernel
//      gemm_bf16_reuse<BM,BN,RM> (every compiled (BM,BN,RM) config, plus an
//      unaligned-N shape and a multi-K-tile/beta!=0 case), and the public
//      hip::gemm_bf16 warmup path that routes to it.
//
// Tolerance: A and C are bf16, B is bf16, accumulation is fp32. The device
// MFMA sums K in 16-element hardware blocks while the CPU oracle sums K
// sequentially; after the single round-to-nearest-even on store the two
// agree to within ~1 bf16 ULP. We report max_abs / max_rel and require
// max_rel < 2e-2 (same convention as test_moe_fused_correct.hip).

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "vkernels/kernels/gemm_bf16.hpp"

// The explicit-tile dispatcher is intentionally not in the public header
// (vkl discovery), so the harness forward-declares it here.
namespace vkernels::kernels::hip {
void gemm_bf16_with_config(std::size_t M, std::size_t N, std::size_t K,
                           float alpha, const uint16_t* A,
                           const uint16_t* B, float beta, uint16_t* C,
                           int bm, int bn, int bk, int threads);
// Cross-tile B-reuse dispatcher (issue #77; not in the public header,
// forward-declared here the same way).
void gemm_bf16_reuse_with_config(std::size_t M, std::size_t N, std::size_t K,
                                 float alpha, const uint16_t* A,
                                 const uint16_t* B, float beta, uint16_t* C,
                                 int bm, int bn, int rm);
}  // namespace vkernels::kernels::hip

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

float rnd(int seed, int i) {  // deterministic pseudo-random in [-1, 1]
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

uint16_t f2bf(float v) {
  uint32_t b; std::memcpy(&b, &v, 4);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}

// One shape: CPU oracle vs HIP, alpha/beta configurable.
struct Stats { double max_abs, max_rel; };

// Generic runner: `launch` performs the device GEMM on dA/dB -> dC.
Stats run_case_impl(int M, int N, int K, float alpha, float beta,
                    const std::vector<uint16_t>& A,
                    const std::vector<uint16_t>& B,
                    const std::vector<uint16_t>& Cinit,
                    uint16_t* dA, uint16_t* dB, uint16_t* dC,
                    const std::function<void(int, int, int, float,
                                             const uint16_t*, const uint16_t*,
                                             float, uint16_t*)>& launch) {
  std::vector<uint16_t> hC(Cinit), rC(Cinit);
  // CPU reference.
  vkernels::kernels::gemm_bf16_cpu(
      (size_t)M, (size_t)N, (size_t)K, alpha, A.data(), B.data(), beta,
      rC.data());
  // Device.
  CK(hipMemcpy(dA, A.data(), (size_t)M * K * 2, hipMemcpyHostToDevice),
     "cpyA");
  CK(hipMemcpy(dB, B.data(), (size_t)K * N * 2, hipMemcpyHostToDevice),
     "cpyB");
  CK(hipMemcpy(dC, Cinit.data(), (size_t)M * N * 2, hipMemcpyHostToDevice),
     "cpyC");
  launch(M, N, K, alpha, dA, dB, beta, dC);
  CK(hipDeviceSynchronize(), "sync");
  CK(hipMemcpy(hC.data(), dC, (size_t)M * N * 2, hipMemcpyDeviceToHost),
     "rdC");

  Stats s{0.0, 0.0};
  for (int i = 0; i < M * N; ++i) {
    float hf, rf;
    uint32_t hu = (uint32_t)hC[i] << 16, ru = (uint32_t)rC[i] << 16;
    std::memcpy(&hf, &hu, 4); std::memcpy(&rf, &ru, 4);
    double e = std::fabs((double)hf - rf);
    if (e > s.max_abs) s.max_abs = e;
    double d = std::fmax(std::fabs((double)rf), 1.0);
    if (e / d > s.max_rel) s.max_rel = e / d;
  }
  return s;
}

// Public-entry runner (config-selected tile / reuse dispatch).
Stats run_case(int M, int N, int K, float alpha, float beta,
               const std::vector<uint16_t>& A,
               const std::vector<uint16_t>& B,
               const std::vector<uint16_t>& Cinit,
               uint16_t* dA, uint16_t* dB, uint16_t* dC) {
  return run_case_impl(
      M, N, K, alpha, beta, A, B, Cinit, dA, dB, dC,
      [](int m, int n, int k, float a, const uint16_t* pA,
         const uint16_t* pB, float b, uint16_t* pC) {
        vkernels::kernels::hip::gemm_bf16((size_t)m, (size_t)n, (size_t)k, a,
                                          pA, pB, b, pC);
      });
}

void fill_AB(std::vector<uint16_t>& A, std::vector<uint16_t>& B,
             int M, int N, int K, int seed) {
  for (int i = 0; i < M * K; ++i) A[i] = f2bf(rnd(seed, i) * 0.5f);
  for (int i = 0; i < K * N; ++i) B[i] = f2bf(rnd(seed + 7, i) * 0.5f);
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  // Largest buffers we will need (M=8192, N=6288, K=7168) plus headroom.
  const int MAXM = 8192, MAXN = 6288, MAXK = 7168;
  uint16_t *dA, *dB, *dC;
  CK(hipMalloc(&dA, (size_t)MAXM * MAXK * 2), "alloc A");
  CK(hipMalloc(&dB, (size_t)MAXK * MAXN * 2), "alloc B");
  CK(hipMalloc(&dC, (size_t)MAXM * MAXN * 2), "alloc C");

  // Kimi-K3 projection (N, K) shapes from the issue.
  struct NK { int N, K; };
  const NK k3[] = {
      {6288, 7168}, {3584, 7168}, {896, 7168}, {2112, 7168}, {1536, 7168},
      {7168, 1536}, {7168, 768}, {7168, 3584}, {2304, 1536}, {3072, 512},
      {1536, 128},
  };

  int fails = 0;
  const float THRESH = 2e-2;

  // --- (1) public entry, serving M ---
  for (int M : {8, 64}) {
    std::printf("\n--- serving M=%d (public hip::gemm_bf16) ---\n", M);
    for (const auto& s : k3) {
      std::vector<uint16_t> A((size_t)M * s.K), B((size_t)s.K * s.N),
          C((size_t)M * s.N, 0);
      fill_AB(A, B, M, s.N, s.K, 1);
      // beta != 0 on a few cases; else 0 (no C read).
      float beta = ((M + s.N) % 5 == 0) ? 0.5f : 0.0f;
      for (size_t i = 0; i < C.size(); ++i) C[i] = f2bf(rnd(3, (int)i) * 0.3f);
      auto st = run_case(M, s.N, s.K, 1.0f, beta, A, B, C, dA, dB, dC);
      bool ok = st.max_rel < THRESH;
      std::printf("  M=%-4d N=%-4d K=%-4d beta=%.1f  max_abs=%.4f "
                  "max_rel=%.6f  %s\n",
                  M, s.N, s.K, beta, st.max_abs, st.max_rel,
                  ok ? "PASS" : "FAIL");
      if (!ok) ++fails;
    }
  }

  // --- (1b) public entry, warmup M=8192 (small (N,K) only; CPU is O(MNK)) ---
  {
    const int M = 8192;
    std::printf("\n--- warmup M=%d (public hip::gemm_bf16) ---\n", M);
    // {64,7168} is the small-N / large-K warmup corner; all three run the
    // M > 64 effective (128,64) tile through the (32,64,RM4) reuse kernel
    // (via the public hip::gemm_bf16).
    const NK warm[] = {{1536, 128}, {3072, 512}, {64, 7168}};
    for (const auto& s : warm) {
      std::vector<uint16_t> A((size_t)M * s.K), B((size_t)s.K * s.N),
          C((size_t)M * s.N, 0);
      fill_AB(A, B, M, s.N, s.K, 2);
      auto st = run_case(M, s.N, s.K, 1.0f, 0.0f, A, B, C, dA, dB, dC);
      bool ok = st.max_rel < THRESH;
      std::printf("  M=%-4d N=%-4d K=%-4d  max_abs=%.4f max_rel=%.6f  %s\n",
                  M, s.N, s.K, st.max_abs, st.max_rel, ok ? "PASS" : "FAIL");
      if (!ok) ++fails;
    }
  }

  // --- (2) every compiled tile via gemm_bf16_with_config ---
  // Awkward shape: M=16 (one 16-row tile, or part of a 32/64 tile),
  // N=100 (not a multiple of 16/64), K=128 (two K-tiles).
  {
    const int M = 16, N = 100, K = 128;
    std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N, 0);
    fill_AB(A, B, M, N, K, 5);
    std::vector<uint16_t> rC(C);
    vkernels::kernels::gemm_bf16_cpu((size_t)M, (size_t)N, (size_t)K, 1.0f,
                                     A.data(), B.data(), 0.0f, rC.data());
    struct Cfg { int bm, bn; };
    const Cfg cfgs[] = {{16, 16}, {16, 64},  {16, 128}, {32, 64},
                        {32, 128}, {64, 64},  {64, 128}};
    std::printf("\n--- explicit-tile sweep (M=%d N=%d K=%d, "
                "hip::gemm_bf16_with_config) ---\n",
                M, N, K);
    for (const auto& c : cfgs) {
      std::vector<uint16_t> hC(C);
      CK(hipMemcpy(dA, A.data(), (size_t)M * K * 2, hipMemcpyHostToDevice),
         "cpyA");
      CK(hipMemcpy(dB, B.data(), (size_t)K * N * 2, hipMemcpyHostToDevice),
         "cpyB");
      CK(hipMemcpy(dC, C.data(), (size_t)M * N * 2, hipMemcpyHostToDevice),
         "cpyC");
      vkernels::kernels::hip::gemm_bf16_with_config(
          (size_t)M, (size_t)N, (size_t)K, 1.0f, dA, dB, 0.0f, dC, c.bm, c.bn,
          64, (c.bm / 16) * 64);
      CK(hipDeviceSynchronize(), "sync");
      CK(hipMemcpy(hC.data(), dC, (size_t)M * N * 2, hipMemcpyDeviceToHost),
         "rdC");
      double ma = 0, mr = 0;
      for (int i = 0; i < M * N; ++i) {
        float hf, rf;
        uint32_t hu = (uint32_t)hC[i] << 16, ru = (uint32_t)rC[i] << 16;
        std::memcpy(&hf, &hu, 4); std::memcpy(&rf, &ru, 4);
        double e = std::fabs((double)hf - rf);
        if (e > ma) ma = e;
        double d = std::fmax(std::fabs((double)rf), 1.0);
        if (e / d > mr) mr = e / d;
      }
      bool ok = mr < THRESH;
      std::printf("  tile %2dx%-3d  max_abs=%.4f max_rel=%.6f  %s\n", c.bm,
                  c.bn, ma, mr, ok ? "PASS" : "FAIL");
      if (!ok) ++fails;
    }
  }

  // --- (3) cross-tile B-reuse + LDS double-buffer kernel (issue #77) -----
  // Every compiled (BM, BN, RM) config on a multi-M-tile, multi-K-tile shape
  // with beta != 0, plus an unaligned-N shape (bounds-check paths) through
  // the explicit reuse dispatcher.
  {
    struct Cfg { int bm, bn, rm; };
    const Cfg cfgs[] = {{16, 64, 4}, {32, 64, 2}, {32, 64, 4}};
    // M=128 spans more than one block footprint (BM*RM) for every config;
    // K=256 is four K-tiles, so the double-buffer ring actually wraps.
    const int M = 128, N = 128, K = 256;
    std::printf("\n--- cross-tile B-reuse sweep (M=%d N=%d K=%d, beta=0.5)"
                " ---\n", M, N, K);
    std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    fill_AB(A, B, M, N, K, 11);
    for (size_t i = 0; i < C.size(); ++i) C[i] = f2bf(rnd(4, (int)i) * 0.3f);
    for (const auto& c : cfgs) {
      auto st = run_case_impl(
          M, N, K, 1.0f, 0.5f, A, B, C, dA, dB, dC,
          [&](int m, int n, int k, float a, const uint16_t* pA,
              const uint16_t* pB, float b, uint16_t* pC) {
            vkernels::kernels::hip::gemm_bf16_reuse_with_config(
                (size_t)m, (size_t)n, (size_t)k, a, pA, pB, b, pC, c.bm,
                c.bn, c.rm);
          });
      bool ok = st.max_rel < THRESH;
      std::printf("  BM=%-2d BN=%-3d RM=%d  max_abs=%.4f max_rel=%.6f  %s\n",
                  c.bm, c.bn, c.rm, st.max_abs, st.max_rel,
                  ok ? "PASS" : "FAIL");
      if (!ok) ++fails;
    }
    // Unaligned N (100, not a multiple of 16/64): exercises the bounds-
    // checked loads / epilogue of the reuse kernel.
    const int Mu = 128, Nu = 100, Ku = 128;
    std::printf("\n--- cross-tile B-reuse, unaligned N (M=%d N=%d K=%d) ---\n",
                Mu, Nu, Ku);
    std::vector<uint16_t> Au((size_t)Mu * Ku), Bu((size_t)Ku * Nu),
        Cu((size_t)Mu * Nu, 0);
    fill_AB(Au, Bu, Mu, Nu, Ku, 12);
    auto st = run_case_impl(
        Mu, Nu, Ku, 1.0f, 0.0f, Au, Bu, Cu, dA, dB, dC,
        [](int m, int n, int k, float a, const uint16_t* pA,
           const uint16_t* pB, float b, uint16_t* pC) {
          vkernels::kernels::hip::gemm_bf16_reuse_with_config(
              (size_t)m, (size_t)n, (size_t)k, a, pA, pB, b, pC, 32, 64, 4);
        });
    bool ok = st.max_rel < THRESH;
    std::printf("  BM=32 BN=64 RM=4  max_abs=%.4f max_rel=%.6f  %s\n",
                st.max_abs, st.max_rel, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
  }

  hipFree(dA); hipFree(dB); hipFree(dC);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
