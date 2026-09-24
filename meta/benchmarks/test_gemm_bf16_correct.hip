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
// Split-K dispatcher (issue #146; same forward-declaration pattern).
void gemm_bf16_splitk_with_config(std::size_t M, std::size_t N, std::size_t K,
                                  float alpha, const uint16_t* A,
                                  const uint16_t* B, float beta, uint16_t* C,
                                  int bm, int bn, int S);
// Fused single-launch split-K (per-tile arrival counters; same pattern).
// Contract: BIT-EXACT with gemm_bf16_splitk_with_config (same fp32 partial
// planes, same fixed ascending-s combine order).
void gemm_bf16_splitk_fused_with_config(
    std::size_t M, std::size_t N, std::size_t K, float alpha,
    const uint16_t* A, const uint16_t* B, float beta, uint16_t* C,
    int bm, int bn, int S);
// Block-FP8 split-K dispatcher (issue #156; same pattern). Contract in
// gemm_bf16.hpp: W8 E4M3FNUZ [K, N], fp32 scales [ceil(K/128), ceil(N/8)].
void gemm_fp8_block_splitk_with_config(
    std::size_t M, std::size_t N, std::size_t K, float alpha,
    const uint16_t* A, const uint8_t* W8, const float* scales,
    float beta, uint16_t* C, int bm, int bn, int S);
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

// Standard E4M3FNUZ software decode (bias 8) -- the independent oracle
// decode for the block-FP8 path (the kernel uses the branchless
// fp32-bit-reinterpret trick, fnuz_value * 2^-119).
float fnuz_dec(uint8_t c) {
  const int E = (c >> 3) & 15, m = c & 7;
  float v = (E == 0) ? (m / 8.0f) * std::ldexp(1.0f, -7)
                     : (1.0f + m / 8.0f) * std::ldexp(1.0f, E - 8);
  return (c & 0x80) ? -v : v;
}

// Host fnuz encoder with round-to-nearest, |v| <= 240 (fnuz max).
uint8_t fnuz_enc(float v) {
  if (v == 0.0f) return 0x00u;
  const uint8_t sign = (v < 0) ? 0x80u : 0x00u;
  float a = std::fabs(v);
  if (a >= 240.0f) return (uint8_t)(sign | 0x7Fu);
  if (a < std::ldexp(1.0f, -7)) {
    int m = (int)std::lround(a * std::ldexp(1.0f, 10));
    if (m > 7) m = 7;
    return (uint8_t)(sign | (uint8_t)m);
  }
  int e;
  std::frexp(a, &e);
  const int E = e - 1 + 8;
  if (E > 15) return (uint8_t)(sign | 0x7Fu);
  const float man = a / std::ldexp(1.0f, E - 8) - 1.0f;
  int m = (int)std::lround(man * 8.0f);
  int Ec = E;
  if (m == 8) { m = 0; ++Ec; }
  if (Ec > 15) return (uint8_t)(sign | 0x7Fu);
  return (uint8_t)(sign | (Ec << 3) | m);
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
    // {64,7168} is the small-N / large-K warmup corner. On gfx942 all three
    // stay on the flat (64,64) tile (the issue #77 reuse sweep measured the
    // reuse schedule ~2x slower here -- see gfx942.md); the reuse kernel is
    // exercised directly in section (3) below.
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

  // --- (4) split-K kernel (issue #146) -----------------------------------
  // The public hip::gemm_bf16 now routes M <= 64 (K >= 256) through the
  // split-K path, so section (1) already exercises it end-to-end on every
  // K3 serving shape. This section exercises the EXPLICIT split dispatcher
  // on corners the auto-routing cannot reach: odd split counts (S not a
  // power of two, including S > K/BK clamping), the K = 128 minimum-split
  // corner, beta != 0 (combine reads the pre-fill C), and the unaligned-N
  // bounds-check paths.
  {
    struct Case { int M, N, K, S; float beta; const char* what; };
    const Case cases[] = {
        {8,   128,  128, 2, 0.0f, "K=128 minimum split (1 K-tile/split)"},
        {8,   128,  256, 3, 0.0f, "odd S=3, K=256 (whole-tile split)"},
        {16,  100,  192, 3, 0.5f, "unaligned N=100, odd S=3, beta!=0"},
        {64,  6288, 7168, 8, 0.5f, "QKV shape, S=8, beta!=0"},
        {8,   128,  128, 9, 0.0f, "S=9 clamped to ceil(K/BK)=2"},
        {8,   128,   64, 8, 0.0f, "S=8 clamped to ceil(K/BK)=1 (unsplit)"},
        {8,  3072,  512, 8, 0.0f, "S=8, kts=1: odd kt0 splits (ring parity)"},
        {8,  7168, 1536, 8, 0.0f, "S=8, kts=3: odd kt0 splits (ring parity)"},
        {5,  3584, 7168, 16, 0.0f, "S=16, kts=7: odd kt0 splits (ring parity)"},
        {5,  6288, 7168, 16, 0.0f, "QKV shape, S=16, kts=7 (ring parity)"},
    };
    std::printf("\n--- split-K sweep (explicit gemm_bf16_splitk_with_config)"
                " ---\n");
    for (const auto& c : cases) {
      std::vector<uint16_t> A((size_t)c.M * c.K), B((size_t)c.K * c.N),
          C((size_t)c.M * c.N, 0);
      fill_AB(A, B, c.M, c.N, c.K, 13);
      for (size_t i = 0; i < C.size(); ++i) C[i] = f2bf(rnd(5, (int)i) * 0.3f);
      auto st = run_case_impl(
          c.M, c.N, c.K, 1.0f, c.beta, A, B, C, dA, dB, dC,
          [&](int m, int n, int k, float a, const uint16_t* pA,
              const uint16_t* pB, float b, uint16_t* pC) {
            vkernels::kernels::hip::gemm_bf16_splitk_with_config(
                (size_t)m, (size_t)n, (size_t)k, a, pA, pB, b, pC, 16, 16,
                c.S);
          });
      bool ok = st.max_rel < THRESH;
      std::printf("  M=%-3d N=%-4d K=%-4d S=%d beta=%.1f  max_abs=%.4f "
                  "max_rel=%.6f  %s  (%s)\n",
                  c.M, c.N, c.K, c.S, c.beta, st.max_abs, st.max_rel,
                  ok ? "PASS" : "FAIL", c.what);
      if (!ok) ++fails;
    }
  }

  // --- (4b) FUSED split-K (single launch, per-tile arrival counters) -----
  // Same case matrix as section (4). Two gates per case:
  //   a. BIT-EXACT vs the two-kernel path: the fused kernel stores the same
  //      fp32 partial planes and folds them in the SAME fixed ascending-s
  //      order as gemm_bf16_splitk_combine_kernel, so the bf16 outputs must
  //      be identical bit patterns. Each case runs the fused path TWICE
  //      back-to-back to prove the per-tile counters self-reset between
  //      launches (the combining block resets its tile's counter).
  //   b. CPU-oracle tolerance (max_rel < 2e-2), same convention as above.
  {
    struct Case { int M, N, K, S; float beta; const char* what; };
    const Case cases[] = {
        {8,   128,  128, 2, 0.0f, "K=128 minimum split (1 K-tile/split)"},
        {8,   128,  256, 3, 0.0f, "odd S=3, K=256 (whole-tile split)"},
        {16,  100,  192, 3, 0.5f, "unaligned N=100 -> two-kernel fallback"},
        {64,  6288, 7168, 8, 0.5f, "QKV shape, S=8, beta!=0"},
        {8,   128,  128, 9, 0.0f, "S=9 clamped to ceil(K/BK)=2"},
        {8,   128,   64, 8, 0.0f, "S=8 clamped to ceil(K/BK)=1 (unsplit)"},
        {8,  3072,  512, 8, 0.0f, "S=8, kts=1: odd kt0 splits (ring parity)"},
        {8,  7168, 1536, 8, 0.0f, "S=8, kts=3: odd kt0 splits (ring parity)"},
        {5,  3584, 7168, 16, 0.0f, "S=16, kts=7: odd kt0 splits (ring parity)"},
        {5,  6288, 7168, 16, 0.0f, "QKV shape, S=16, kts=7 (ring parity)"},
    };
    std::printf("\n--- FUSED split-K (gemm_bf16_splitk_fused_with_config)"
                " ---\n");
    for (const auto& c : cases) {
      std::vector<uint16_t> A((size_t)c.M * c.K), B((size_t)c.K * c.N),
          C((size_t)c.M * c.N, 0);
      fill_AB(A, B, c.M, c.N, c.K, 13);
      for (size_t i = 0; i < C.size(); ++i) C[i] = f2bf(rnd(5, (int)i) * 0.3f);
      // CPU oracle.
      std::vector<uint16_t> rC(C);
      vkernels::kernels::gemm_bf16_cpu((size_t)c.M, (size_t)c.N, (size_t)c.K,
                                       1.0f, A.data(), B.data(), c.beta,
                                       rC.data());
      // Two-kernel reference output.
      CK(hipMemcpy(dA, A.data(), (size_t)c.M * c.K * 2, hipMemcpyHostToDevice),
         "cpyA");
      CK(hipMemcpy(dB, B.data(), (size_t)c.K * c.N * 2, hipMemcpyHostToDevice),
         "cpyB");
      CK(hipMemcpy(dC, C.data(), (size_t)c.M * c.N * 2,
                   hipMemcpyHostToDevice), "cpyC");
      vkernels::kernels::hip::gemm_bf16_splitk_with_config(
          (size_t)c.M, (size_t)c.N, (size_t)c.K, 1.0f, dA, dB, c.beta, dC,
          16, 16, c.S);
      CK(hipDeviceSynchronize(), "sync");
      std::vector<uint16_t> h2k((size_t)c.M * c.N);
      CK(hipMemcpy(h2k.data(), dC, (size_t)c.M * c.N * 2,
                   hipMemcpyDeviceToHost), "rdC2k");
      // Fused path, run TWICE (counter self-reset check on the 2nd run).
      std::vector<uint16_t> hFu[2];
      bool bit_exact = true;
      double ma = 0, mr = 0;
      for (int rep = 0; rep < 2; ++rep) {
        CK(hipMemcpy(dC, C.data(), (size_t)c.M * c.N * 2,
                     hipMemcpyHostToDevice), "cpyC2");
        vkernels::kernels::hip::gemm_bf16_splitk_fused_with_config(
            (size_t)c.M, (size_t)c.N, (size_t)c.K, 1.0f, dA, dB, c.beta, dC,
            16, 16, c.S);
        CK(hipDeviceSynchronize(), "sync");
        hFu[rep].resize((size_t)c.M * c.N);
        CK(hipMemcpy(hFu[rep].data(), dC, (size_t)c.M * c.N * 2,
                     hipMemcpyDeviceToHost), "rdCfu");
        for (size_t i = 0; i < hFu[rep].size(); ++i)
          if (hFu[rep][i] != h2k[i]) bit_exact = false;
        // Oracle tolerance on the fused output.
        for (int i = 0; i < c.M * c.N; ++i) {
          float hf, rf;
          uint32_t hu = (uint32_t)hFu[rep][i] << 16, ru = (uint32_t)rC[i] << 16;
          std::memcpy(&hf, &hu, 4); std::memcpy(&rf, &ru, 4);
          double e = std::fabs((double)hf - rf);
          if (e > ma) ma = e;
          double d = std::fmax(std::fabs((double)rf), 1.0);
          if (e / d > mr) mr = e / d;
        }
      }
      bool ok = bit_exact && mr < THRESH;
      std::printf("  M=%-3d N=%-4d K=%-4d S=%-2d beta=%.1f  bit_exact=%d "
                  "max_abs=%.4f max_rel=%.6f  %s  (%s)\n",
                  c.M, c.N, c.K, c.S, c.beta, (int)bit_exact, ma, mr,
                  ok ? "PASS" : "FAIL", c.what);
      if (!ok) ++fails;
    }
  }

  // --- (5) block-FP8 split-K kernel (issue #156) ---------------------------
  // Dequant-fused E4M3FNUZ weight stream (fnuz bias-8 contract, per
  // 128-K x 8-N fp32 block scales). Oracle: host quantizes the bf16 B,
  // dequantizes with the INDEPENDENT software fnuz decode, and runs
  // gemm_bf16_cpu (fp32 accum) on the result; the kernel must reproduce
  // that to within the same 2e-2 rel tolerance (the dequantized weights
  // are identical, so the only differences are fp32 vs MFMA accumulation
  // order). Exercises all compiled tiles, beta != 0, S clamping, and the
  // N % 8 != 0 rejection (C must be untouched).
  {
    struct Case { int M, N, K, S; float beta; int bm, bn; const char* what; };
    const Case cases[] = {
        {5,  6288, 7168, 8, 0.0f, 16, 16, "QKV shape, tile 16x16, S=8"},
        {8,  3584, 7168, 8, 0.0f, 16, 64, "gate shape, tile 16x64, S=8"},
        {16,  896, 7168, 4, 0.5f, 16, 16, "small-N shape, beta!=0, S=4"},
        {5,  6288, 7168, 9, 0.0f, 32, 64, "QKV shape, tile 32x64, S=9 clamp"},
        {5,   128,  256, 3, 0.0f, 16, 16, "tiny shape, odd S=3"},
    };
    std::printf("\n--- block-FP8 split-K (gemm_fp8_block_splitk_with_config)"
                " ---\n");
    for (const auto& c : cases) {
      const int n_s8 = (c.N + 7) / 8, k_b = (c.K + 127) / 128;
      std::vector<uint16_t> A((size_t)c.M * c.K), B((size_t)c.K * c.N),
          C((size_t)c.M * c.N, 0);
      fill_AB(A, B, c.M, c.N, c.K, 17);
      for (size_t i = 0; i < C.size(); ++i) C[i] = f2bf(rnd(6, (int)i) * 0.3f);

      // Host quantization (per-block amax scales, fnuz codes) + oracle.
      std::vector<float> scales((size_t)n_s8 * k_b);
      std::vector<uint8_t> W8((size_t)c.K * c.N);
      for (int kb = 0; kb < k_b; ++kb)
        for (int nb = 0; nb < n_s8; ++nb) {
          float amax = 0;
          for (int kk = 0; kk < 128 && kb * 128 + kk < c.K; ++kk)
            for (int nn = 0; nn < 8 && nb * 8 + nn < c.N; ++nn) {
              const size_t idx = (size_t)(kb * 128 + kk) * c.N + nb * 8 + nn;
              uint32_t u = (uint32_t)B[idx] << 16;
              float f; std::memcpy(&f, &u, 4);
              amax = std::max(amax, std::fabs(f));
            }
          const float scale = amax / 240.0f;
          scales[(size_t)kb * n_s8 + nb] = scale;
          for (int kk = 0; kk < 128 && kb * 128 + kk < c.K; ++kk)
            for (int nn = 0; nn < 8 && nb * 8 + nn < c.N; ++nn) {
              const size_t idx = (size_t)(kb * 128 + kk) * c.N + nb * 8 + nn;
              uint32_t u = (uint32_t)B[idx] << 16;
              float f; std::memcpy(&f, &u, 4);
              W8[idx] = fnuz_enc(f / scale);
            }
        }
      std::vector<uint16_t> Bdq((size_t)c.K * c.N);
      for (int k = 0; k < c.K; ++k)
        for (int n = 0; n < c.N; ++n)
          Bdq[(size_t)k * c.N + n] = f2bf(
              fnuz_dec(W8[(size_t)k * c.N + n]) *
              scales[(size_t)(k / 128) * n_s8 + (n / 8)]);
      std::vector<uint16_t> rC(C);
      vkernels::kernels::gemm_bf16_cpu((size_t)c.M, (size_t)c.N,
                                       (size_t)c.K, 1.0f, A.data(),
                                       Bdq.data(), c.beta, rC.data());

      uint8_t* dW8; float* dS;
      CK(hipMalloc(&dW8, W8.size()), "fp8 W8");
      CK(hipMalloc(&dS, scales.size() * 4), "fp8 scales");
      CK(hipMemcpy(dW8, W8.data(), W8.size(), hipMemcpyHostToDevice),
         "fp8 cpyW8");
      CK(hipMemcpy(dS, scales.data(), scales.size() * 4,
                   hipMemcpyHostToDevice), "fp8 cpyS");
      CK(hipMemcpy(dA, A.data(), (size_t)c.M * c.K * 2,
                   hipMemcpyHostToDevice), "fp8 cpyA");
      CK(hipMemcpy(dC, C.data(), (size_t)c.M * c.N * 2,
                   hipMemcpyHostToDevice), "fp8 cpyC");
      vkernels::kernels::hip::gemm_fp8_block_splitk_with_config(
          (size_t)c.M, (size_t)c.N, (size_t)c.K, 1.0f, dA, dW8, dS, c.beta,
          dC, c.bm, c.bn, c.S);
      CK(hipDeviceSynchronize(), "fp8 sync");
      std::vector<uint16_t> hC(C.size());
      CK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost),
         "fp8 rdC");
      double max_abs = 0, max_rel = 0;
      for (int i = 0; i < c.M * c.N; ++i) {
        float hf, rf;
        uint32_t hu = (uint32_t)hC[i] << 16, ru = (uint32_t)rC[i] << 16;
        std::memcpy(&hf, &hu, 4); std::memcpy(&rf, &ru, 4);
        double e = std::fabs((double)hf - rf);
        if (e > max_abs) max_abs = e;
        double d = std::fmax(std::fabs((double)rf), 1.0);
        if (e / d > max_rel) max_rel = e / d;
      }
      const bool ok = max_rel < THRESH;
      std::printf("  M=%-3d N=%-4d K=%-4d S=%d beta=%.1f %2dx%-2d"
                  "  max_abs=%.4f max_rel=%.6f  %s  (%s)\n",
                  c.M, c.N, c.K, c.S, c.beta, c.bm, c.bn, max_abs, max_rel,
                  ok ? "PASS" : "FAIL", c.what);
      if (!ok) ++fails;
      hipFree(dW8); hipFree(dS);
    }
    // Rejection path: N % 8 != 0 must print and leave C untouched.
    {
      std::vector<uint16_t> A(5 * 256), C(5 * 100, 0x3c00);
      std::vector<uint8_t> W8((size_t)256 * 100, 0);
      std::vector<float> scales(2 * 13, 1.0f);
      uint16_t *dA, *dC; uint8_t* dW8; float* dS;
      CK(hipMalloc(&dA, A.size() * 2), "rj A");
      CK(hipMalloc(&dC, C.size() * 2), "rj C");
      CK(hipMalloc(&dW8, W8.size()), "rj W8");
      CK(hipMalloc(&dS, scales.size() * 4), "rj S");
      CK(hipMemcpy(dC, C.data(), C.size() * 2, hipMemcpyHostToDevice),
         "rj cpyC");
      vkernels::kernels::hip::gemm_fp8_block_splitk_with_config(
          5, 100, 256, 1.0f, dA, dW8, dS, 0.0f, dC, 16, 16, 2);
      CK(hipDeviceSynchronize(), "rj sync");
      std::vector<uint16_t> hC(C.size());
      CK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost),
         "rj rdC");
      bool untouched = true;
      for (size_t i = 0; i < hC.size(); ++i)
        if (hC[i] != 0x3c00) { untouched = false; break; }
      std::printf("  N%%8!=0 rejection (N=100): C %s  %s\n",
                  untouched ? "untouched" : "MODIFIED",
                  untouched ? "PASS" : "FAIL");
      if (!untouched) ++fails;
      hipFree(dA); hipFree(dC); hipFree(dW8); hipFree(dS);
    }
  }

  // --- (6) decode-GEMV split-K kernels (issue #156) -----------------------
  // No-LDS-barrier GEMV route for the tiny-M decode regime: bf16 and
  // block-FP8 variants. Oracle = gemm_bf16_cpu on the dequantized weights
  // (fp8) or the raw B (bf16), same 2e-2 tolerance. Exercises tb=4/8,
  // odd/prime S values (element-granularity splits), beta != 0, and the
  // N % 8 != 0 rejection (C untouched).
  {
    struct Case { int M, N, K, S, tb, th; float beta; bool fp8; const char* what; };
    const Case cases[] = {
        {5, 6288, 7168, 32, 8, 64, 0.0f, false, "QKV shape bf16 S=32 tb=8"},
        {5, 6288, 7168, 31, 8, 64, 0.0f, false, "odd prime S=31 (element splits)"},
        {5, 6288, 7168, 32, 4, 64, 0.0f, false, "tb=4 narrow columns"},
        {5, 6288, 7168, 64, 8, 128, 0.0f, false, "th=128 blocks"},
        {5, 3584, 7168, 32, 8, 64, 0.5f, false, "beta!=0"},
        {8,  896, 7168, 64, 8, 64, 0.0f, false, "small-N shape"},
        {2,  128,  257, 32, 8, 64, 0.0f, false, "odd K=257 element splits"},
        {1,  896, 7168, 32, 8, 64, 0.0f, false, "M=1 pure GEMV"},
        {5, 6288, 7168, 32, 8, 64, 0.0f, true,  "QKV shape fp8 S=32"},
        {5, 3584, 7168, 33, 8, 64, 0.5f, true,  "fp8 odd S=33, beta!=0"},
        {5,  128,  256, 3,  8, 64, 0.0f, true,  "fp8 tiny, odd S=3"},
        {2,  896, 7168, 64, 4, 64, 0.0f, true,  "fp8 tb=4"},
    };
    std::printf("\n--- decode-GEMV split-K (gemv_decode_*_splitk) ---\n");
    for (const auto& c : cases) {
      const int n_s8 = (c.N + 7) / 8, k_b = (c.K + 127) / 128;
      std::vector<uint16_t> A((size_t)c.M * c.K), B((size_t)c.K * c.N),
          C((size_t)c.M * c.N, 0);
      fill_AB(A, B, c.M, c.N, c.K, 19);
      for (size_t i = 0; i < C.size(); ++i) C[i] = f2bf(rnd(7, (int)i) * 0.3f);
      std::vector<uint16_t> rC(C), Bdq;
      uint8_t* dW8 = nullptr; float* dS = nullptr; uint16_t* dB = nullptr;
      if (c.fp8) {
        std::vector<float> scales((size_t)n_s8 * k_b);
        std::vector<uint8_t> W8((size_t)c.K * c.N);
        for (int kb = 0; kb < k_b; ++kb)
          for (int nb = 0; nb < n_s8; ++nb) {
            float amax = 0;
            for (int kk = 0; kk < 128 && kb * 128 + kk < c.K; ++kk)
              for (int nn = 0; nn < 8 && nb * 8 + nn < c.N; ++nn) {
                const size_t idx = (size_t)(kb * 128 + kk) * c.N + nb * 8 + nn;
                uint32_t u = (uint32_t)B[idx] << 16;
                float f; std::memcpy(&f, &u, 4);
                amax = std::max(amax, std::fabs(f));
              }
            const float scale = amax / 240.0f;
            scales[(size_t)kb * n_s8 + nb] = scale;
            for (int kk = 0; kk < 128 && kb * 128 + kk < c.K; ++kk)
              for (int nn = 0; nn < 8 && nb * 8 + nn < c.N; ++nn) {
                const size_t idx = (size_t)(kb * 128 + kk) * c.N + nb * 8 + nn;
                uint32_t u = (uint32_t)B[idx] << 16;
                float f; std::memcpy(&f, &u, 4);
                W8[idx] = fnuz_enc(f / scale);
              }
          }
        Bdq.resize((size_t)c.K * c.N);
        for (int k = 0; k < c.K; ++k)
          for (int n = 0; n < c.N; ++n)
            Bdq[(size_t)k * c.N + n] = f2bf(
                fnuz_dec(W8[(size_t)k * c.N + n]) *
                scales[(size_t)(k / 128) * n_s8 + (n / 8)]);
        vkernels::kernels::gemm_bf16_cpu((size_t)c.M, (size_t)c.N,
                                         (size_t)c.K, 1.0f, A.data(),
                                         Bdq.data(), c.beta, rC.data());
        CK(hipMalloc(&dW8, W8.size()), "gemv W8");
        CK(hipMalloc(&dS, scales.size() * 4), "gemv S");
        CK(hipMemcpy(dW8, W8.data(), W8.size(), hipMemcpyHostToDevice), "gemv cW8");
        CK(hipMemcpy(dS, scales.data(), scales.size() * 4, hipMemcpyHostToDevice), "gemv cS");
      } else {
        vkernels::kernels::gemm_bf16_cpu((size_t)c.M, (size_t)c.N,
                                         (size_t)c.K, 1.0f, A.data(),
                                         B.data(), c.beta, rC.data());
        CK(hipMalloc(&dB, B.size() * 2), "gemv B");
        CK(hipMemcpy(dB, B.data(), B.size() * 2, hipMemcpyHostToDevice), "gemv cB");
      }
      CK(hipMemcpy(dA, A.data(), (size_t)c.M * c.K * 2, hipMemcpyHostToDevice), "gemv cA");
      CK(hipMemcpy(dC, C.data(), (size_t)c.M * c.N * 2, hipMemcpyHostToDevice), "gemv cC");
      if (c.fp8)
        vkernels::kernels::hip::gemv_decode_fp8_splitk(
            (size_t)c.M, (size_t)c.N, (size_t)c.K, 1.0f, dA, dW8, dS, c.beta,
            dC, c.S, c.tb, c.th);
      else
        vkernels::kernels::hip::gemv_decode_bf16_splitk(
            (size_t)c.M, (size_t)c.N, (size_t)c.K, 1.0f, dA, dB, c.beta,
            dC, c.S, c.tb, c.th);
      CK(hipDeviceSynchronize(), "gemv sync");
      std::vector<uint16_t> hC(C.size());
      CK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost), "gemv rC");
      double max_abs = 0, max_rel = 0;
      for (int i = 0; i < c.M * c.N; ++i) {
        float hf, rf;
        uint32_t hu = (uint32_t)hC[i] << 16, ru = (uint32_t)rC[i] << 16;
        std::memcpy(&hf, &hu, 4); std::memcpy(&rf, &ru, 4);
        double e = std::fabs((double)hf - rf);
        if (e > max_abs) max_abs = e;
        double d = std::fmax(std::fabs((double)rf), 1.0);
        if (e / d > max_rel) max_rel = e / d;
      }
      const bool ok = max_rel < THRESH;
      std::printf("  M=%-2d N=%-4d K=%-4d S=%-3d tb=%d th=%-3d beta=%.1f %s"
                  "  max_abs=%.4f max_rel=%.6f  %s  (%s)\n",
                  c.M, c.N, c.K, c.S, c.tb, c.th, c.beta, c.fp8 ? "fp8 " : "bf16",
                  max_abs, max_rel, ok ? "PASS" : "FAIL", c.what);
      if (!ok) ++fails;
      if (dW8) hipFree(dW8);
      if (dS) hipFree(dS);
      if (dB) hipFree(dB);
    }
    // Rejection paths: N % 8 != 0 and M > 8 must leave C untouched.
    {
      std::vector<uint16_t> A(9 * 256), C(9 * 100, 0x3c00);
      std::vector<uint16_t> B((size_t)256 * 100, 0);
      uint16_t *dA, *dB, *dC;
      CK(hipMalloc(&dA, A.size() * 2), "rj2 A");
      CK(hipMalloc(&dB, B.size() * 2), "rj2 B");
      CK(hipMalloc(&dC, C.size() * 2), "rj2 C");
      CK(hipMemcpy(dC, C.data(), C.size() * 2, hipMemcpyHostToDevice), "rj2 cpyC");
      bool untouched = true;
      vkernels::kernels::hip::gemv_decode_bf16_splitk(
          9, 100, 256, 1.0f, dA, dB, 0.0f, dC, 4, 8, 64);
      CK(hipDeviceSynchronize(), "rj2 sync");
      std::vector<uint16_t> hC(C.size());
      CK(hipMemcpy(hC.data(), dC, hC.size() * 2, hipMemcpyDeviceToHost), "rj2 rC");
      for (size_t i = 0; i < hC.size(); ++i)
        if (hC[i] != 0x3c00) { untouched = false; break; }
      std::printf("  M=9 rejection (M>8): C %s  %s\n",
                  untouched ? "untouched" : "MODIFIED", untouched ? "PASS" : "FAIL");
      if (!untouched) ++fails;
      hipFree(dA); hipFree(dB); hipFree(dC);
    }
  }

  hipFree(dA); hipFree(dB); hipFree(dC);
  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
