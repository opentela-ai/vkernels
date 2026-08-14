// meta/benchmarks/test_kda_correct.hip
//
// End-to-end correctness test for the Kimi Delta Attention kernels
// (issue #21) on gfx942, against the CPU references in vkernels::kernels:
//
//   hip::kda_layer_norm_gated  vs  kda_layer_norm_gated_cpu
//   hip::kda_gate_chunk_cumsum vs  kda_gate_chunk_cumsum_cpu
//   hip::kda_delta_rule_fwd    vs  kda_naive_delta_rule_fwd_cpu (oracle)
//   hip::kda_pack_bitmatrix    vs  kda_pack_bitmatrix_cpu
//
// The forward is the acceptance bar for the issue (a K3-shaped forward
// runs on gfx942 and matches the CPU/torch reference; K3_DISABLE_KDA no
// longer needed). The cooperative HIP kernel runs the same per-token
// recurrence as the oracle, so the only divergence is fp round-off; we
// report max_abs / max_rel and require max_rel < 2e-2 (same convention as
// test_gemm_bf16_correct.hip / test_mla_correct.hip).
//
// Shapes: KDA head dims up to 128 (kMaxD); S ∈ {64, 512} (chunk_size = S/4
// on the host reference — the device kernel streams tokens, so chunk_size
// is not load-bearing for it), a few (B, H) including H=128. Gates g, β in
// (0, 1]; k is NOT L2-normalised here (the oracle does not normalise, so
// both sides see the same k).

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
float rgate(int seed, int i) {  // (0.05, 1] — never fully forgotten
  return 0.05f + 0.95f * std::fabs(rnd(seed, i));
}

// --- forward: hip::kda_delta_rule_fwd vs kda_naive_delta_rule_fwd_cpu ---
struct Stats { double max_abs, max_rel; };

// L2-normalise each key vector in place (the contract in kda.hpp says
// "k is L2-normalised by the caller"; with normalised k the delta-rule
// state stays bounded instead of blowing up identically on both sides).
void normalize_keys(std::vector<float>& k, int B, int H, int S, int D) {
  for (int x = 0; x < B * H * S; ++x) {
    float* kk = k.data() + (size_t)x * D;
    float norm = 0.0f;
    for (int d = 0; d < D; ++d) norm += kk[d] * kk[d];
    norm = 1.0f / std::sqrt(norm + 1e-12f);
    for (int d = 0; d < D; ++d) kk[d] *= norm;
  }
}

Stats run_fwd(int B, int H, int S, int D, int chunk_size,
              const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v, const std::vector<float>& g,
              const std::vector<float>& beta,
              float* dq, float* dk, float* dv, float* dg, float* db,
              float* dout) {
  const size_t n = (size_t)B * H * S * D;
  std::vector<float> host(n, 0.0f), ref(n, 0.0f);

  vkernels::kernels::kda_naive_delta_rule_fwd_cpu(
      q.data(), k.data(), v.data(), g.data(), beta.data(), ref.data(),
      B, H, S, D);

  CK(hipMemcpy(dq, q.data(), n * 4, hipMemcpyHostToDevice), "cpyq");
  CK(hipMemcpy(dk, k.data(), n * 4, hipMemcpyHostToDevice), "cpyk");
  CK(hipMemcpy(dv, v.data(), n * 4, hipMemcpyHostToDevice), "cpyv");
  CK(hipMemcpy(dg, g.data(), (size_t)B * H * S * 4, hipMemcpyHostToDevice),
     "cpyg");
  CK(hipMemcpy(db, beta.data(), (size_t)B * H * S * 4, hipMemcpyHostToDevice),
     "cpyb");
  vkernels::kernels::hip::kda_delta_rule_fwd(
      dq, dk, dv, dg, db, dout, B, H, S, D, chunk_size);
  CK(hipDeviceSynchronize(), "sync");
  CK(hipMemcpy(host.data(), dout, n * 4, hipMemcpyDeviceToHost), "rdout");

  Stats s{0.0, 0.0};
  for (size_t i = 0; i < n; ++i) {
    double e = std::fabs((double)host[i] - ref[i]);
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

  int fails = 0;
  const float THRESH = 2e-2f;

  // ---------------------------------------------------------------------
  // (1) kda_delta_rule_fwd vs naive oracle, several (B,H,S,D) shapes.
  // ---------------------------------------------------------------------
  {
    // Peak buffers (B=1,H=128,S=512,D=128).
    const int Bm = 1, Hm = 128, Sm = 512, Dm = 128;
    const size_t n = (size_t)Bm * Hm * Sm * Dm;
    float *dq, *dk, *dv, *dg, *db, *dout;
    CK(hipMalloc(&dq, n * 4), "alloc q");
    CK(hipMalloc(&dk, n * 4), "alloc k");
    CK(hipMalloc(&dv, n * 4), "alloc v");
    CK(hipMalloc(&dg, (size_t)Bm * Hm * Sm * 4), "alloc g");
    CK(hipMalloc(&db, (size_t)Bm * Hm * Sm * 4), "alloc b");
    CK(hipMalloc(&dout, n * 4), "alloc out");

    struct Cfg { int B, H, S, D, chunk; };
    const Cfg cfgs[] = {
        {1, 1, 64, 16, 16},    {1, 1, 64, 32, 16},
        {1, 1, 64, 64, 16},    {1, 16, 64, 64, 16},
        {1, 1, 512, 64, 64},   {1, 1, 512, 128, 64},
    };
    std::printf("--- kda_delta_rule_fwd vs naive oracle ---\n");
    for (const auto& c : cfgs) {
      const size_t m = (size_t)c.B * c.H * c.S * c.D;
      std::vector<float> q(m), k(m), v(m), g((size_t)c.B * c.H * c.S),
          beta((size_t)c.B * c.H * c.S);
      for (size_t i = 0; i < m; ++i) { q[i] = rnd(1, (int)i); k[i] = rnd(2, (int)i); v[i] = rnd(3, (int)i); }
      for (size_t i = 0; i < g.size(); ++i) { g[i] = rgate(4, (int)i); beta[i] = rgate(5, (int)i); }
      normalize_keys(k, c.B, c.H, c.S, c.D);
      auto st = run_fwd(c.B, c.H, c.S, c.D, c.chunk, q, k, v, g, beta,
                        dq, dk, dv, dg, db, dout);
      bool ok = st.max_rel < THRESH;
      std::printf("  B=%d H=%-3d S=%-3d D=%-3d chunk=%-2d  "
                  "max_abs=%.4f max_rel=%.6f  %s\n",
                  c.B, c.H, c.S, c.D, c.chunk, st.max_abs, st.max_rel,
                  ok ? "PASS" : "FAIL");
      if (!ok) ++fails;
      fflush(stdout);
    }
    hipFree(dq); hipFree(dk); hipFree(dv); hipFree(dg); hipFree(db);
    hipFree(dout);
  }

  // ---------------------------------------------------------------------
  // (2) kda_layer_norm_gated vs CPU, K3-ish (N, D).
  // ---------------------------------------------------------------------
  {
    struct Cfg { int N, D; };
    const Cfg cfgs[] = {{8, 128}, {64, 512}, {8192, 128}};
    std::printf("--- kda_layer_norm_gated vs CPU ---\n");
    const float eps = 1e-6f;
    for (const auto& c : cfgs) {
      std::vector<float> x((size_t)c.N * c.D), w(c.D), gate((size_t)c.N * c.D),
          host((size_t)c.N * c.D, 0), ref((size_t)c.N * c.D, 0);
      for (size_t i = 0; i < x.size(); ++i) x[i] = rnd(1, (int)i);
      for (size_t i = 0; i < w.size(); ++i) w[i] = 0.5f + rgate(2, (int)i);
      for (size_t i = 0; i < gate.size(); ++i) gate[i] = rnd(3, (int)i);
      vkernels::kernels::kda_layer_norm_gated_cpu(
          x.data(), w.data(), gate.data(), ref.data(), c.N, c.D, eps);

      float *dx, *dw, *dg, *dout;
      CK(hipMalloc(&dx, x.size() * 4), "alloc ln x");
      CK(hipMalloc(&dw, w.size() * 4), "alloc ln w");
      CK(hipMalloc(&dg, gate.size() * 4), "alloc ln g");
      CK(hipMalloc(&dout, host.size() * 4), "alloc ln out");
      CK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice), "cx");
      CK(hipMemcpy(dw, w.data(), w.size() * 4, hipMemcpyHostToDevice), "cw");
      CK(hipMemcpy(dg, gate.data(), gate.size() * 4, hipMemcpyHostToDevice),
         "cg");
      vkernels::kernels::hip::kda_layer_norm_gated(
          dx, dw, dg, dout, c.N, c.D, eps);
      CK(hipDeviceSynchronize(), "sync");
      CK(hipMemcpy(host.data(), dout, host.size() * 4,
                   hipMemcpyDeviceToHost), "ro");
      double ma = 0, mr = 0;
      for (size_t i = 0; i < host.size(); ++i) {
        double e = std::fabs((double)host[i] - ref[i]);
        if (e > ma) ma = e;
        double d = std::fmax(std::fabs((double)ref[i]), 1.0);
        if (e / d > mr) mr = e / d;
      }
      bool ok = mr < THRESH;
      std::printf("  N=%-4d D=%-3d  max_abs=%.4f max_rel=%.6f  %s\n",
                  c.N, c.D, ma, mr, ok ? "PASS" : "FAIL");
      if (!ok) ++fails;
      hipFree(dx); hipFree(dw); hipFree(dg); hipFree(dout);
    }
  }

  // ---------------------------------------------------------------------
  // (3) kda_gate_chunk_cumsum vs CPU (B,H,n_chunks,chunk_size).
  // ---------------------------------------------------------------------
  {
    const int B = 1, H = 16, nc = 8, cs = 64;
    std::vector<float> g((size_t)B * H * nc * cs), hi((size_t)B * H * nc * cs),
        hr((size_t)B * H * nc * cs), igi((size_t)B * H * nc),
        igr((size_t)B * H * nc);
    for (size_t i = 0; i < g.size(); ++i) g[i] = rgate(1, (int)i);
    vkernels::kernels::kda_gate_chunk_cumsum_cpu(
        g.data(), hr.data(), igr.data(), B, H, nc, cs);

    float *dg, *dhi, *dig;
    CK(hipMalloc(&dg, g.size() * 4), "alloc g");
    CK(hipMalloc(&dhi, hi.size() * 4), "alloc hi");
    CK(hipMalloc(&dig, igi.size() * 4), "alloc ig");
    CK(hipMemcpy(dg, g.data(), g.size() * 4, hipMemcpyHostToDevice), "cg");
    vkernels::kernels::hip::kda_gate_chunk_cumsum(
        dg, dhi, dig, B, H, nc, cs);
    CK(hipDeviceSynchronize(), "sync");
    CK(hipMemcpy(hi.data(), dhi, hi.size() * 4, hipMemcpyDeviceToHost), "ri");
    CK(hipMemcpy(igi.data(), dig, igi.size() * 4, hipMemcpyDeviceToHost),
       "rig");
    double ma = 0, mr = 0;
    for (size_t i = 0; i < hi.size(); ++i) {
      double e = std::fabs((double)hi[i] - hr[i]);
      if (e > ma) ma = e;
      double d = std::fmax(std::fabs((double)hr[i]), 1.0);
      if (e / d > mr) mr = e / d;
    }
    for (size_t i = 0; i < igi.size(); ++i) {
      double e = std::fabs((double)igi[i] - igr[i]);
      if (e > ma) ma = e;
      double d = std::fmax(std::fabs((double)igr[i]), 1.0);
      if (e / d > mr) mr = e / d;
    }
    bool ok = mr < THRESH;
    std::printf("--- kda_gate_chunk_cumsum vs CPU ---\n");
    std::printf("  B=%d H=%d nc=%d cs=%d  max_abs=%.4f max_rel=%.6f  %s\n",
                B, H, nc, cs, ma, mr, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
    hipFree(dg); hipFree(dhi); hipFree(dig);
  }

  // ---------------------------------------------------------------------
  // (4) kda_pack_bitmatrix vs CPU (MSB-first, round trip).
  // ---------------------------------------------------------------------
  {
    const size_t n = 1000;
    std::vector<uint8_t> bits(n), hpacked((n + 7) / 8, 0),
        rpacked((n + 7) / 8, 0);
    for (size_t i = 0; i < n; ++i) bits[i] = (uint8_t)(rnd(1, (int)i) > 0.0f);
    vkernels::kernels::kda_pack_bitmatrix_cpu(
        bits.data(), rpacked.data(), n);

    uint8_t* dbits; uint8_t* dpacked;
    CK(hipMalloc(&dbits, n), "alloc bits");
    CK(hipMalloc(&dpacked, (n + 7) / 8), "alloc packed");
    CK(hipMemcpy(dbits, bits.data(), n, hipMemcpyHostToDevice), "cb");
    vkernels::kernels::hip::kda_pack_bitmatrix(dbits, dpacked, n);
    CK(hipDeviceSynchronize(), "sync");
    CK(hipMemcpy(hpacked.data(), dpacked, (n + 7) / 8,
                 hipMemcpyDeviceToHost), "rp");
    int diff = 0;
    for (size_t i = 0; i < (n + 7) / 8; ++i)
      if (hpacked[i] != rpacked[i]) ++diff;
    bool ok = diff == 0;
    std::printf("--- kda_pack_bitmatrix vs CPU ---\n");
    std::printf("  n=%zu  byte diffs=%d  %s\n", n, diff,
                ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
    hipFree(dbits); hipFree(dpacked);
  }

  std::printf("\n%s (%d failure(s))\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
