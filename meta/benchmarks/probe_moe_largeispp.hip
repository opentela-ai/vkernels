// meta/benchmarks/probe_moe_largeispp.hip
//
// Issue #86 stage-split probe: fused_moe_mxfp4 at ispp >= 4096 mismatches the
// CPU oracle on real weights (cpu-rel 0.165 at E=32 h7168 ispp=4096, kmajor-
// and M-independent) while every recorded *fault* shape runs clean on the
// current build, and zero-weight harness tests mask weight-indexing errors.
// This probe reproduces the exact bench recipe (same rnd()/e2m1 encoding,
// same routing, real weights) and splits the two GPU stages:
//   1. GPU act rows vs CPU oracle act rows   -> gateup_swiglu verdict
//   2. GPU out vs CPU out, per hidden column -> down_combine verdict
// and prints the per-column badness map keyed by pid_n so a boundary/stride
// defect in the ispp dimension is directly visible.
//
//   probe [E] [hidden] [ispp] [kmajor|nmajor] [M] [seed]  (defaults 32 7168 4096 kmajor 1 0)
//
// Issue #86 audit additions (fe8b5ec follow-up):
//   - [seed] shifts every rnd() stream so the mismatch can be quantified
//     across reps: if the benign-bf16-ULP hypothesis holds, the act stage
//     stays within ~1 bf16 ulp of the oracle, the down-stage-on-GPU-act
//     isolation stays small, and the GPU-vs-GPU rerun spread (atomic
//     accumulation-order noise only) stays orders of magnitude below the
//     vs-oracle residual.  A wandering/growing spread across seeds would
//     indicate a real indexing/race defect instead.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                            \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));      \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

static int g_seed = 0;
static float rnd(int seed, int i) {
  seed += 7919 * g_seed;
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;
}

static uint32_t f2bits(float v) { uint32_t b; std::memcpy(&b, &v, 4); return b; }
static float bits2f(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
static uint16_t f2bf(float v) {
  uint32_t b = f2bits(v);
  b += 0x7FFFu + ((b >> 16) & 1);
  return (uint16_t)(b >> 16);
}
static float bf2f(uint16_t v) { return bits2f((uint32_t)v << 16); }

// Pack a float -> E2M1 nibble (MXFP4: 0,+-0.5,+-1,+-1.5,+-2,+-3,+-4,+-6).
static uint8_t e2m1_nibble(float v) {
  bool neg = v < 0;
  float af = std::fabs(v);
  float vals[6] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f};
  uint8_t nibs[6] = {0, 1, 2, 3, 4, 5};
  float best_d = std::fabs(af - vals[0]);
  uint8_t best_n = nibs[0];
  for (int i = 1; i < 6; ++i) {
    float d = std::fabs(af - vals[i]);
    if (d < best_d) { best_d = d; best_n = nibs[i]; }
  }
  if (af > 4.5f) best_n = 7;  // 6.0
  return neg ? (uint8_t)(best_n | 0x8) : best_n;
}

// Same dim-swap repack the bench uses for --kmajor (value-preserving).
static void repack_kmaj(const uint8_t* src, uint8_t* dst,
                        int E, int n_fast, int k_slow) {
  for (int e = 0; e < E; ++e)
    for (int g = 0; g < 2; ++g) {
      const uint8_t* s = src + ((size_t)e * 2 + g) * n_fast * k_slow;
      uint8_t* d = dst + ((size_t)e * 2 + g) * n_fast * k_slow;
      for (int n = 0; n < n_fast; ++n)
        for (int k = 0; k < k_slow; ++k)
          d[(size_t)k * n_fast + n] = s[(size_t)n * k_slow + k];
    }
}

int main(int argc, char** argv) {
  const int E = (argc > 1) ? atoi(argv[1]) : 32;
  const int hidden = (argc > 2) ? atoi(argv[2]) : 7168;
  const int ispp = (argc > 3) ? atoi(argv[3]) : 4096;
  const bool kmajor = (argc <= 4) || std::strcmp(argv[4], "kmajor") == 0;
  const int M = (argc > 5) ? atoi(argv[5]) : 1;
  g_seed = (argc > 6) ? atoi(argv[6]) : 0;
  const int top_k = 16;
  const int BLOCK_M = 16;
  const int group_size = 32;
  const float LIMIT = 10.0f;
  const int activation = vkernels::kernels::kSiTU;
  // Match bench_moe_fused exactly (Kimi-K3 SiTU softcaps).
  const float beta = 4.0f, linear_beta = 25.0f;
  const int N = M * top_k;

  std::printf("probe: E=%d hidden=%d ispp=%d top_k=%d M=%d %s seed=%d\n", E,
              hidden, ispp, top_k, M, kmajor ? "kmajor" : "nmajor", g_seed);

  // ---- host data (identical recipe to bench_moe_fused) -------------------
  std::vector<int32_t> h_tids(N), h_sids(E * BLOCK_M), h_eids((E * BLOCK_M) / BLOCK_M);
  std::vector<float> h_tw(N);
  for (int i = 0; i < M; ++i)
    for (int s = 0; s < top_k; ++s) {
      h_tids[i * top_k + s] = (i * 17 + s * 13 + 7) % E;
      h_tw[i * top_k + s] = 0.5f + 0.5f * rnd(20, i * top_k + s);
    }
  int EM = vkernels::kernels::moe_align_block_size(
      h_tids.data(), M, top_k, BLOCK_M, E, h_sids.data(), h_eids.data());

  std::vector<uint16_t> hA(M * hidden);
  for (int i = 0; i < M * hidden; ++i) hA[i] = f2bf(rnd(10, i) * 0.1f);

  const std::size_t w13_sz = (std::size_t)E * 2 * ispp * (hidden / 2);
  const std::size_t w13s_sz = (std::size_t)E * 2 * ispp * (hidden / group_size);
  const std::size_t w2_sz = (std::size_t)E * hidden * (ispp / 2);
  const std::size_t w2s_sz = (std::size_t)E * hidden * (ispp / group_size);
  std::vector<uint8_t> hw13(w13_sz), hw13s(w13s_sz, 127), hw2(w2_sz), hw2s(w2s_sz, 127);
  for (std::size_t i = 0; i < w13_sz / 2; ++i) {
    hw13[2*i]     = e2m1_nibble(rnd(2, 2*i) * 1.5f)     | (e2m1_nibble(rnd(3, 2*i+1) * 1.5f) << 4);
    hw13[2*i + 1] = e2m1_nibble(rnd(4, i) * 1.5f)       | (e2m1_nibble(rnd(5, i) * 1.5f) << 4);
  }
  for (std::size_t i = 0; i < w2_sz / 2; ++i) {
    hw2[2*i]      = e2m1_nibble(rnd(6, 2*i) * 1.5f)     | (e2m1_nibble(rnd(7, 2*i+1) * 1.5f) << 4);
    hw2[2*i + 1]  = e2m1_nibble(rnd(8, i) * 1.5f)       | (e2m1_nibble(rnd(9, i) * 1.5f) << 4);
  }
  std::vector<float> hb13(E * 2 * ispp, 0.0f), hb2(E * hidden, 0.0f);
  std::vector<uint8_t> hw13_km, hw13s_km;
  if (kmajor) {
    hw13_km.resize(w13_sz);  hw13s_km.resize(w13s_sz);
    repack_kmaj(hw13.data(), hw13_km.data(), E, ispp, hidden / 2);
    repack_kmaj(hw13s.data(), hw13s_km.data(), E, ispp, hidden / group_size);
  }
  const uint8_t* w13_u = kmajor ? hw13_km.data() : hw13.data();
  const uint8_t* w13s_u = kmajor ? hw13s_km.data() : hw13s.data();

  // ---- CPU oracle (always N-major, value-identical to the repack) --------
  std::vector<float> h_tws(EM, 0.0f);
  for (int i = 0; i < EM; ++i)
    if (h_sids[i] < N) h_tws[i] = h_tw[h_sids[i]];
  std::vector<uint16_t> act_cpu((size_t)EM * ispp, 0);
  std::vector<float> out_cpu(M * hidden, 0.0f);
  vkernels::kernels::fused_moe_mxfp4_cpu(
      hA.data(), hw13.data(), hw13s.data(), hw2.data(), hw2s.data(),
      h_sids.data(), h_tws.data(), h_eids.data(), act_cpu.data(), out_cpu.data(),
      M, hidden, ispp, top_k, EM, group_size, LIMIT, activation, beta,
      linear_beta, hb13.data(), hb2.data());

  // ---- device buffers + GPU run ------------------------------------------
  uint8_t *dw13, *dw13s, *dw2, *dw2s;
  float *db13, *db2;
  CK(hipMalloc(&dw13, w13_sz), "w13");   CK(hipMalloc(&dw13s, w13s_sz), "w13s");
  CK(hipMalloc(&dw2, w2_sz), "w2");      CK(hipMalloc(&dw2s, w2s_sz), "w2s");
  CK(hipMalloc(&db13, E * 2 * ispp * 4), "b13");
  CK(hipMalloc(&db2, E * hidden * 4), "b2");
  CK(hipMemcpy(dw13, w13_u, w13_sz, hipMemcpyHostToDevice), "cpw13");
  CK(hipMemcpy(dw13s, w13s_u, w13s_sz, hipMemcpyHostToDevice), "cpw13s");
  CK(hipMemcpy(dw2, hw2.data(), w2_sz, hipMemcpyHostToDevice), "cpw2");
  CK(hipMemcpy(dw2s, hw2s.data(), w2s_sz, hipMemcpyHostToDevice), "cpw2s");
  CK(hipMemcpy(db13, hb13.data(), E * 2 * ispp * 4, hipMemcpyHostToDevice), "cpb13");
  CK(hipMemcpy(db2, hb2.data(), E * hidden * 4, hipMemcpyHostToDevice), "cpb2");

  uint16_t *dA, *dAct; float* dOut;
  int32_t *dTids, *dSids, *dEids;
  float* dTw;
  CK(hipMalloc(&dA, M * hidden * 2), "A");
  CK(hipMalloc(&dAct, (size_t)EM * ispp * 2), "act");
  CK(hipMalloc(&dOut, M * hidden * 4), "out");
  CK(hipMalloc(&dTw, N * 4), "tw");
  CK(hipMalloc(&dTids, N * 4), "tids");
  CK(hipMalloc(&dSids, EM * 4), "sids");
  CK(hipMalloc(&dEids, (size_t)(EM / BLOCK_M) * 4), "eids");
  CK(hipMemcpy(dA, hA.data(), M * hidden * 2, hipMemcpyHostToDevice), "cpA");
  CK(hipMemcpy(dTw, h_tw.data(), N * 4, hipMemcpyHostToDevice), "cpTw");
  CK(hipMemcpy(dTids, h_tids.data(), N * 4, hipMemcpyHostToDevice), "cpTids");
  CK(hipMemcpy(dSids, h_sids.data(), EM * 4, hipMemcpyHostToDevice), "cpSids");
  CK(hipMemcpy(dEids, h_eids.data(), (size_t)(EM / BLOCK_M) * 4, hipMemcpyHostToDevice), "cpEids");
  CK(hipMemset(dOut, 0, M * hidden * 4), "zOut");

  auto launch = [&]() {
    vkernels::kernels::hip::fused_moe_mxfp4(
        dA, dw13, dw13s, dw2, dw2s, dTids, dTw, dAct, dOut,
        M, hidden, ispp, top_k, dSids, dEids, EM, LIMIT, activation, beta,
        linear_beta, db13, db2, BLOCK_M, kmajor);
  };
  launch();
  CK(hipDeviceSynchronize(), "sync");

  // ---- compare ------------------------------------------------------------
  std::vector<uint16_t> act_gpu((size_t)EM * ispp);
  std::vector<float> out_gpu(M * hidden);
  CK(hipMemcpy(act_gpu.data(), dAct, (size_t)EM * ispp * 2, hipMemcpyDeviceToHost), "rdAct");
  CK(hipMemcpy(out_gpu.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut");

  // GPU-vs-GPU rerun spread: combine_kernel accumulates via float atomicAdd,
  // so two identical launches differ by accumulation-order noise alone.
  // This is the noise floor any "real fault" signal must exceed.
  {
    CK(hipMemset(dOut, 0, M * hidden * 4), "zOut2");
    launch();
    CK(hipDeviceSynchronize(), "sync2");
    std::vector<float> out_gpu2(M * hidden);
    CK(hipMemcpy(out_gpu2.data(), dOut, M * hidden * 4, hipMemcpyDeviceToHost), "rdOut2");
    double mx_abs = 0.0, mx_rel = 0.0;
    for (int c = 0; c < M * hidden; ++c) {
      double d = std::fabs((double)out_gpu[c] - (double)out_gpu2[c]);
      if (d > mx_abs) mx_abs = d;
      double r = d / std::fmax(std::fabs((double)out_gpu[c]), 1.0);
      if (r > mx_rel) mx_rel = r;
    }
    std::printf("[det]    rerun spread: max|delta|=%.3e worst_rel=%.6f\n",
                mx_abs, mx_rel);
  }

  // Stage 1: act rows for the real flat indices only.
  double act_worst = 0.0; int act_bad = 0;
  int act_ulp_max = 0; size_t act_ulp_gt1 = 0, act_ulp_n = 0;
  // Monotonic integer map of a bf16 pattern (classic IEEE trick) so the
  // distance between two bf16 values can be counted in ulps.
  auto bf16_ord = [](uint16_t b) -> int32_t {
    return (b & 0x8000u) ? (int32_t)(0xFFFFu - b) : (int32_t)(b | 0x8000u);
  };
  for (int i = 0; i < N; ++i) {
    int r = i;  // real sorted row for flat i (moe_align keeps order)
    double mx = 0.0;
    for (int c = 0; c < ispp; ++c) {
      uint16_t gb = act_gpu[(size_t)r * ispp + c];
      uint16_t ub = act_cpu[(size_t)r * ispp + c];
      float g = bf2f(gb), u = bf2f(ub);
      double e = std::fabs((double)g - u);
      mx = std::fmax(mx, e / std::fmax(std::fabs((double)u), 1.0));
      if (std::fabs(u) > 1e-30f) {  // skip exact zeros/subnormals for ulp count
        int d = std::abs(bf16_ord(gb) - bf16_ord(ub));
        if (d > act_ulp_max) act_ulp_max = d;
        ++act_ulp_n;
        if (d > 1) ++act_ulp_gt1;
      }
    }
    if (mx > 0.02) ++act_bad;
    act_worst = std::fmax(act_worst, mx);
  }
  std::printf("[gateup] act rows=16 real only: worst_rel=%.6f bad_rows=%d/%d | bf16-ulp max=%d >1ulp=%zu/%zu\n",
              act_worst, act_bad, N, act_ulp_max, act_ulp_gt1, act_ulp_n);

  // Stage 2: out per-column map.
  int bad_cols = 0; double out_worst = 0.0; int worst_col = -1;
  std::vector<int> bad_by_pidn(hidden / 64 + 1, 0);
  std::vector<int> bad_by_mod16(16, 0);
  for (int c = 0; c < M * hidden; ++c) {
    double e = std::fabs((double)out_gpu[c] - (double)out_cpu[c]);
    double r = e / std::fmax(std::fabs((double)out_cpu[c]), 1.0);
    out_worst = std::fmax(out_worst, r);
    if (out_worst == r) worst_col = c;
    if (r > 0.02) {
      ++bad_cols;
      ++bad_by_pidn[c / 64];
      ++bad_by_mod16[c % 16];
    }
  }
  std::printf("[down]   out cols: worst_rel=%.6f worst_col=%d bad_cols=%d/%d\n",
              out_worst, worst_col, bad_cols, hidden);
  std::printf("[down]   bad cols by pid_n (col/64): ");
  for (int b = 0; b < hidden / 64; ++b)
    if (bad_by_pidn[b]) std::printf("%d:%d ", b, bad_by_pidn[b]);
  std::printf("\n[down]   bad cols by col%%16: ");
  for (int m = 0; m < 16; ++m) if (bad_by_mod16[m]) std::printf("%d:%d ", m, bad_by_mod16[m]);
  std::printf("\n");

  // Stage-3 isolation: run the CPU down+combine on the GPU's act rows.
  // If it matches out_gpu the GPU down stage is correct and the CPU
  // oracle's down path is the deviating one (and vice versa).
  std::vector<float> partial3((size_t)EM * hidden);
  std::vector<float> out3((size_t)M * hidden, 0.0f);
  vkernels::kernels::moe_down_cpu(act_gpu.data(), hw2.data(), hw2s.data(),
      h_sids.data(), h_eids.data(), partial3.data(), M, ispp, 0, ispp,
      hidden, top_k, EM, group_size);
  vkernels::kernels::moe_combine_cpu(partial3.data(), hb2.data(), h_tws.data(),
      h_sids.data(), h_eids.data(), out3.data(), M, hidden, top_k, EM);
  {
    double r3 = 0.0, r3c = 0.0; int wc3 = -1;
    for (int c = 0; c < M * hidden; ++c) {
      double e = std::fabs((double)out3[c] - (double)out_gpu[c]);
      double r = e / std::fmax(std::fabs((double)out_gpu[c]), 1.0);
      if (r > r3) { r3 = r; wc3 = c; }
      double ec = std::fabs((double)out3[c] - (double)out_cpu[c]);
      double rc = ec / std::fmax(std::fabs((double)out_cpu[c]), 1.0);
      if (rc > r3c) r3c = rc;
    }
    std::printf("[iso] CPU-down(GPU-act): vs GPU rel=%.6f (col %d) | vs CPU rel=%.6f\n",
                r3, wc3, r3c);
  }

  CK(hipFree(dA), "fA"); CK(hipFree(dAct), "fAct"); CK(hipFree(dOut), "fOut");
  CK(hipFree(dTw), "fTw"); CK(hipFree(dTids), "fTids"); CK(hipFree(dSids), "fSids");
  CK(hipFree(dEids), "fEids");
  CK(hipFree(dw13), "fw13"); CK(hipFree(dw13s), "fw13s"); CK(hipFree(dw2), "fw2");
  CK(hipFree(dw2s), "fw2s"); CK(hipFree(db13), "fb13"); CK(hipFree(db2), "fb2");
  std::printf("Done.\n");
  return 0;
}
