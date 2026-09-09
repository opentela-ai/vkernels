// meta/benchmarks/test_kda_chunked.hip
//
// Correctness test for the chunked WY K3 forward (#70) on gfx942:
//
//   hip::kda_delta_rule_fwd_chunked_with_scratch  vs
//   kda_naive_delta_rule_fwd_cpu                  (the K3 oracle)
//
// The chunked path (kda.hip #L8) computes the SAME per-key-dim recurrence
// as the cooperative kernel but in the affine WY form (grams + explicit
// triangular inverse + U_v/W/T/Opar precompute, then an nc-step serial
// state pass) — the derivation is CPU-verified in
// tests/kernels/attn/test_kda_k3_chunked.cpp (k3_wy_chunked_fwd), which is
// where the CONTRACT below comes from.
//
// CONTRACT (stricter than the cooperative kernel's): k is L2-normalised
// (|M| <= 1 by Cauchy-Schwarz keeps the explicit inverse bounded; with
// unnormalised k it overflows at cs=64), gates in (0,1], beta <= 1,
// chunk_size == 64, S % 64 == 0, D <= 128. Inputs here follow
// test_kda_correct.hip (contractive rgate, normalised keys).
//
// Also checks the caller-owned STATE contract against the cooperative
// kernel with a NONZERO initial state (multi-turn decode shape): both
// entries must consume the same pre-filled state and write the same final
// state back.
//
// Convention (test_kda_correct.hip): max_rel < 2e-2, PASS/FAIL summary on
// the last line.

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vkernels/kernels/kda.hpp"

#define CK(e, m)                                                       \
  do {                                                                 \
    hipError_t r = (e);                                                \
    if (r != hipSuccess) {                                             \
      std::fprintf(stderr, "%s: %s\n", m, hipGetErrorString(r));       \
      std::exit(1);                                                    \
    }                                                                  \
  } while (0)

namespace {

float rnd(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return (float)((int)x % 200000) / 100000.0f;  // [-1, 1)
}

// Contractive forget gate in (0.05, 1.0) — the production regime
// (test_kda_correct.hip's rgate).
float rgate(int seed, int i) {
  unsigned x = (unsigned)(seed * 2654435761u + (unsigned)i * 40503u);
  x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
  return 0.05f + 0.95f * (float)(x % 100000u) / 100000.0f;
}

void normalize_keys(std::vector<float>& k, int B, int H, int S, int D) {
  for (int x = 0; x < B * H * S; ++x) {
    float* kk = k.data() + (size_t)x * D;
    float norm = 0.0f;
    for (int d = 0; d < D; ++d) norm += kk[d] * kk[d];
    norm = 1.0f / std::sqrt(norm + 1e-12f);
    for (int d = 0; d < D; ++d) kk[d] *= norm;
  }
}

struct Stats { double max_abs, max_rel; };

Stats cmp(const std::vector<float>& got, const std::vector<float>& ref) {
  Stats s{0.0, 0.0};
  for (size_t i = 0; i < got.size(); ++i) {
    double e = std::fabs((double)got[i] - ref[i]);
    if (e > s.max_abs) s.max_abs = e;
    double d = std::fmax(std::fabs((double)ref[i]), 1.0);
    if (e / d > s.max_rel) s.max_rel = e / d;
  }
  return s;
}

}  // namespace

int main() {
  hipDeviceProp_t p;
  CK(hipGetDeviceProperties(&p, 0), "props");
  std::printf("GPU: %s (arch=%s)\n", p.name, p.gcnArchName);

  int fails = 0, passes = 0;
  const float THRESH = 2e-2f;
  constexpr int CS = 64;

  // ---------------------------------------------------------------------
  // (1) chunked vs the CPU oracle at zero initial state, all D templates,
  //     multi-chunk and multi-(b,h) shapes.
  // ---------------------------------------------------------------------
  {
    struct Cfg { int B, H, S, D; };
    const Cfg cfgs[] = {
        {1, 1, 64, 16},     {1, 1, 64, 32},      {1, 1, 64, 64},
        {1, 1, 64, 128},    {1, 16, 64, 64},     {2, 1, 128, 128},
        {1, 1, 512, 64},    {1, 1, 512, 128},    {1, 8, 512, 128},
        {1, 128, 512, 128},
    };
    std::printf("--- kda_delta_rule_fwd_chunked vs naive oracle ---\n");
    for (const auto& c : cfgs) {
      const size_t m = (size_t)c.B * c.H * c.S * c.D;
      const size_t ng = (size_t)c.B * c.H * c.S;
      std::vector<float> q(m), k(m), v(m), g(ng * c.D), beta(ng);
      for (size_t i = 0; i < m; ++i) {
        q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i);
      }
      for (size_t i = 0; i < g.size(); ++i) g[i] = rgate(4, (int)i);
      for (size_t i = 0; i < beta.size(); ++i) beta[i] = rgate(5, (int)i);
      normalize_keys(k, c.B, c.H, c.S, c.D);

      std::vector<float> ref(m, 0.0f), host(m, 0.0f);
      vkernels::kernels::kda_naive_delta_rule_fwd_cpu(
          q.data(), k.data(), v.data(), g.data(), beta.data(), ref.data(),
          c.B, c.H, c.S, c.D);

      float *dq, *dk, *dv, *dg, *db, *dstate, *dout, *dscratch;
      CK(hipMalloc(&dq, m * 4), "aq");
      CK(hipMalloc(&dk, m * 4), "ak");
      CK(hipMalloc(&dv, m * 4), "av");
      CK(hipMalloc(&dg, g.size() * 4), "ag");
      CK(hipMalloc(&db, ng * 4), "ab");
      CK(hipMalloc(&dstate, (size_t)c.B * c.H * c.D * c.D * 4), "ast");
      CK(hipMalloc(&dout, m * 4), "ao");
      const size_t sf = vkernels::kernels::hip::kda_chunked_scratch_floats(
          c.B, c.H, c.S, c.D);
      CK(hipMalloc(&dscratch, sf * 4), "asc");
      CK(hipMemcpy(dq, q.data(), m * 4, hipMemcpyHostToDevice), "cq");
      CK(hipMemcpy(dk, k.data(), m * 4, hipMemcpyHostToDevice), "ck");
      CK(hipMemcpy(dv, v.data(), m * 4, hipMemcpyHostToDevice), "cv");
      CK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice), "cg");
      CK(hipMemcpy(db, beta.data(), ng * 4, hipMemcpyHostToDevice), "cb");
      CK(hipMemset(dstate, 0, (size_t)c.B * c.H * c.D * c.D * 4), "zst");

      vkernels::kernels::hip::kda_delta_rule_fwd_chunked_with_scratch(
          dq, dk, dv, dg, db, dstate, dout, dscratch,
          c.B, c.H, c.S, c.D, CS);
      CK(hipDeviceSynchronize(), "sync");
      CK(hipMemcpy(host.data(), dout, m * 4, hipMemcpyDeviceToHost), "co");

      auto st = cmp(host, ref);
      bool ok = st.max_rel < THRESH;
      std::printf("  B=%d H=%-3d S=%-3d D=%-3d  "
                  "max_abs=%.4f max_rel=%.6f  %s\n",
                  c.B, c.H, c.S, c.D, st.max_abs, st.max_rel,
                  ok ? "PASS" : "FAIL");
      ok ? ++passes : ++fails;
      fflush(stdout);
      hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dg); hipFree(db);
      hipFree(dstate); hipFree(dout); hipFree(dscratch);
    }
  }

  // ---------------------------------------------------------------------
  // (2) nonzero initial state: chunked vs the cooperative kernel — both
  //     consume the same pre-filled state (multi-turn decode) and must
  //     produce the same outputs AND write back the same final state.
  // ---------------------------------------------------------------------
  {
    const int B = 2, H = 3, S = 128, D = 64;
    const size_t m = (size_t)B * H * S * D;
    const size_t ng = (size_t)B * H * S;
    const size_t ns = (size_t)B * H * D * D;
    std::vector<float> q(m), k(m), v(m), g(ng * D), beta(ng),
        init(ns), outC(m, 0.f), outN(m, 0.f), stC(ns, 0.f), stN(ns, 0.f);
    for (size_t i = 0; i < m; ++i) {
      q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i);
    }
    for (size_t i = 0; i < g.size(); ++i) g[i] = rgate(4, (int)i);
    for (size_t i = 0; i < beta.size(); ++i) beta[i] = rgate(5, (int)i);
    for (size_t i = 0; i < ns; ++i) init[i] = rnd(6, (int)i) * 0.5f;
    normalize_keys(k, B, H, S, D);

    float *dq, *dk, *dv, *dg, *db, *dstC, *dstN, *doutC, *doutN, *dscratch;
    CK(hipMalloc(&dq, m * 4), "aq");
    CK(hipMalloc(&dk, m * 4), "ak");
    CK(hipMalloc(&dv, m * 4), "av");
    CK(hipMalloc(&dg, g.size() * 4), "ag");
    CK(hipMalloc(&db, ng * 4), "ab");
    CK(hipMalloc(&dstC, ns * 4), "ast1");
    CK(hipMalloc(&dstN, ns * 4), "ast2");
    CK(hipMalloc(&doutC, m * 4), "ao1");
    CK(hipMalloc(&doutN, m * 4), "ao2");
    const size_t sf = vkernels::kernels::hip::kda_chunked_scratch_floats(
        B, H, S, D);
    CK(hipMalloc(&dscratch, sf * 4), "asc");
    CK(hipMemcpy(dq, q.data(), m * 4, hipMemcpyHostToDevice), "cq");
    CK(hipMemcpy(dk, k.data(), m * 4, hipMemcpyHostToDevice), "ck");
    CK(hipMemcpy(dv, v.data(), m * 4, hipMemcpyHostToDevice), "cv");
    CK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice), "cg");
    CK(hipMemcpy(db, beta.data(), ng * 4, hipMemcpyHostToDevice), "cb");

    std::printf("--- nonzero initial state: chunked vs cooperative ---\n");
    for (int rep = 0; rep < 2; ++rep) {   // both orders, same inputs
      CK(hipMemcpy(dstC, init.data(), ns * 4, hipMemcpyHostToDevice), "cst1");
      CK(hipMemcpy(dstN, init.data(), ns * 4, hipMemcpyHostToDevice), "cst2");
      vkernels::kernels::hip::kda_delta_rule_fwd_with_scratch(
          dq, dk, dv, dg, db, dstN, doutN, B, H, S, D);
      vkernels::kernels::hip::kda_delta_rule_fwd_chunked_with_scratch(
          dq, dk, dv, dg, db, dstC, doutC, dscratch, B, H, S, D, CS);
      CK(hipDeviceSynchronize(), "sync");
      CK(hipMemcpy(outC.data(), doutC, m * 4, hipMemcpyDeviceToHost), "co1");
      CK(hipMemcpy(outN.data(), doutN, m * 4, hipMemcpyDeviceToHost), "co2");
      CK(hipMemcpy(stC.data(), dstC, ns * 4, hipMemcpyDeviceToHost), "cs1");
      CK(hipMemcpy(stN.data(), dstN, ns * 4, hipMemcpyDeviceToHost), "cs2");
      auto so = cmp(outC, outN);
      auto ss = cmp(stC, stN);
      bool ok = so.max_rel < THRESH && ss.max_rel < THRESH;
      std::printf("  rep=%d  out: max_abs=%.4f max_rel=%.6f   "
                  "final state: max_abs=%.4f max_rel=%.6f  %s\n",
                  rep, so.max_abs, so.max_rel, ss.max_abs, ss.max_rel,
                  ok ? "PASS" : "FAIL");
      ok ? ++passes : ++fails;
      fflush(stdout);
    }
    hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dg); hipFree(db);
    hipFree(dstC); hipFree(dstN); hipFree(doutC); hipFree(doutN);
    hipFree(dscratch);
  }

  std::printf("\n%s (%d/%d PASS)\n", fails ? "FAIL" : "PASS",
              passes, passes + fails);
  return fails ? 1 : 0;
}
