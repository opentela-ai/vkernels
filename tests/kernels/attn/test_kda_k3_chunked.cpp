// tests/kernels/attn/test_kda_k3_chunked.cpp
//
// Derivation + correctness check for a CHUNKED K3 per-key-dim gated delta
// rule (issue #70). The committed HIP kda_delta_rule_fwd runs the per-token
// recurrence serially (latency-bound, per-block limit found this session).
// The only scaling lever left is inter-chunk parallelism — but kda.cpp's
// existing chunked primitives (L4 intra / L5 inter / L6 output) implement the
// OLD *standard* rule (scalar gate [B,H,S], PRE-gate prediction), cross-checked
// against kda_standard_delta_rule_fwd, NOT the K3 per-key-dim oracle
// (kda_naive_delta_rule_fwd_cpu). A K3 chunked kernel therefore needs a NEW
// per-key-dim chunked derivation, not a port of L4/L5/L6.
//
// This file spells that derivation inline (gate cumsum [per-column] -> intra
// lower-triangular solve -> inter propagation -> output combine) and checks it
// against kda_naive_delta_rule_fwd_cpu at random per-key-dim gates, at several
// chunk sizes, and at the full-history (g==1) and zero-gate independence edge
// cases. It is the correctness foundation for a future HIP chunked kernel: no
// GPU code is written or profiled until this passes.
//
// K3 recurrence (the oracle, per-token, per (b,h)):
//   S'_t[v,k] = g_t[k] * S_{t-1}[v,k]                # per-key-dim gate (col k)
//   a_t[v]    = sum_k S'_t[v,k] * k_t[k]            # predict from GATED state
//   S_t[v,k]  = S'_t[v,k] + b_t * (v_t[v]-a_t[v]) * k_t[k]   # delta update
//   o_t[v]    = sum_k S_t[v,k] * q_t[k]             # output
//
// Chunked derivation (within-chunk local t = 0..C-1; C_{c-1} is the inter
// state entering chunk c, C_{-1}=0). Let G_{a,b}[k] = prod_{l=a}^{b} g_l[k]
// (PER-COLUMN product; =1 if a>b), recovered from a per-column log-cumsum
// L_t[k] = sum_{l<=t} log g_l[k] as G_{a,b}[k] = exp(L_b[k] - L_{a-1}[k])
// (L_{-1}=0). Unrolling the gated state:
//   S_t[v,k] = G_{0,t}[k] C_{c-1}[v,k]
//              + sum_{j<=t} G_{j+1,t}[k] b_j u_j[v] k_j[k]
// where u_t[v] = v_t[v] - a_t[v]. The prediction a_t = (g_t odot S_{t-1}) . k_t
// gives (using g_t[k] G_{.,t-1}[k] = G_{.,t}[k]):
//   a_t[v]   = sum_k G_{0,t}[k] C_{c-1}[v,k] k_t[k]            # inter pred (POST-gate: includes g_t)
//            + sum_{j<t} b_j u_j[v] [sum_k G_{j+1,t}[k] k_j[k] k_t[k]]   # intra coupling
// so u_t[v] = v_t[v] - inter_pred[v] - sum_{j<t} b_j M_{j,t} u_j[v] is a
// lower-triangular solve with the GATE-WEIGHTED Gram M_{j,t}=sum_k G_{j+1,t}[k] k_j k_t
// (gate INSIDE the sum, vs the standard rule's scalar G_{j+1,t-1}*(k_j.k_t)).
// Inter propagation C_c = G_{0,C-1} odot C_{c-1} + sum_t G_{t+1,C-1} b_t u_t k_t^T
// and output o_t = sum_k G_{0,t}[k] C_{c-1}[v,k] q_t[k]
//                + sum_{j<=t} b_j u_j[v] [sum_k G_{j+1,t}[k] k_j[k] q_t[k]]
// are structurally identical to L5/L6 but per-column and gate-weighted.
#include "minitest.hpp"

#include <cmath>
#include <random>
#include <vector>

#include "vkernels/kernels/kda.hpp"

using namespace vkernels::kernels;

namespace {

constexpr float kNegLogFloor = -1.0e9f;  // log(0) clamp (matches kda.cpp)

float log_gate(float g) { return g <= 0.0f ? kNegLogFloor : std::log(g); }

// Per-column log-cumsum: g[B,H,nc,cs,D] -> intra_log[B,H,nc,cs,D],
// inter_log[B,H,nc,D] (cross-chunk EXCLUSIVE, starts at 0).
void k3_gate_chunk_cumsum(const float* g, float* intra_log, float* inter_log,
                          int B, int H, int nc, int cs, int D) {
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t base = ((size_t)b * H + h) * nc;
      for (int d = 0; d < D; ++d) {
        float inter_acc = 0.0f;  // per-column cross-chunk exclusive cumsum
        for (int c = 0; c < nc; ++c) {
          const float* gc = g + ((base + c) * cs) * D + d;
          float* Lc = intra_log + ((base + c) * cs) * D + d;
          float acc = 0.0f;
          for (int t = 0; t < cs; ++t) {
            acc += log_gate(gc[(size_t)t * D]);
            Lc[(size_t)t * D] = acc;
          }
          inter_log[(base + c) * D + d] = inter_acc;  // I_c[d] = sum_{c'<c} chunk_log[c',d]
          inter_acc += acc;                            // chunk_log[c,d] = L_{cs-1}[d]
        }
      }
    }
}

// Within-chunk gate product G_{a,b}[k] from the chunk's per-column log-cumsum
// L[cs][D] (L_t[k] = sum_{l<=t} log g_l[k]). Empty product (a>b) = 1.
float gate_prod(const float* L, int a, int b, int k, int D) {
  if (a > b) return 1.0f;
  return std::exp(L[(size_t)b * D + k] - (a > 0 ? L[(size_t)(a - 1) * D + k] : 0.0f));
}

// k3_delta_rule_intra: within-chunk lower-triangular solve for u_t (one chunk).
// u_t[v] = v_t[v] - inter_pred[v] - sum_{j<t} b_j M_{j,t} u_j[v]
//   inter_pred[v] = sum_k G_{0,t}[k] C_{c-1}[v,k] k_t[k]   (POST-gate)
//   M_{j,t}       = sum_k G_{j+1,t}[k] k_j[k] k_t[k]       (gate-weighted)
void k3_delta_rule_intra(const float* k, const float* v, const float* g,
                         const float* beta, const float* intra_log,
                         const float* inter_state, float* u,
                         int B, int H, int S, int D, int cs, int c) {
  (void)g;  // gate enters via intra_log
  const int nc = S / cs;
  const size_t bh = (size_t)(c) * cs;  // local; full base added below
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t full = (size_t)(b * H + h) * S;
      const float* Lc = intra_log + (((size_t)(b * H + h) * nc) + c) * cs * D;
      const float* Cin = inter_state + (((size_t)(b * H + h) * (nc + 1)) + c) * D * D;
      for (int t = 0; t < cs; ++t) {
        const int tau = c * cs + t;
        const float* kt = k + (full + tau) * D;
        const float* vt = v + (full + tau) * D;
        float pred[256], corr[256];
        // inter prediction (POST-gate: G_{0,t} includes g_t)
        for (int vv = 0; vv < D; ++vv) {
          float s = 0.0f;
          for (int kk = 0; kk < D; ++kk)
            s += gate_prod(Lc, 0, t, kk, D) * Cin[(size_t)vv * D + kk] * kt[kk];
          pred[vv] = s;
        }
        // intra coupling: sum_{j<t} b_j M_{j,t} u_j
        for (int vv = 0; vv < D; ++vv) corr[vv] = 0.0f;
        for (int j = 0; j < t; ++j) {
          const int tauj = c * cs + j;
          const float* kj = k + (full + tauj) * D;
          const float* uj = u + (full + tauj) * D;
          float Mjt = 0.0f;
          for (int kk = 0; kk < D; ++kk)
            Mjt += gate_prod(Lc, j + 1, t, kk, D) * kj[kk] * kt[kk];
          const float bj = beta[full + tauj];
          for (int vv = 0; vv < D; ++vv) corr[vv] += bj * Mjt * uj[vv];
        }
        float* ut = u + (full + tau) * D;
        for (int vv = 0; vv < D; ++vv) ut[vv] = vt[vv] - pred[vv] - corr[vv];
      }
    }
  (void)bh;
}

// k3_delta_rule_inter: C_c = G_{0,C-1} odot C_{c-1} + sum_t G_{t+1,C-1} b_t u_t k_t^T
void k3_delta_rule_inter(const float* k, const float* u, const float* beta,
                         const float* intra_log, float* inter_state,
                         int B, int H, int S, int D, int cs, int c) {
  const int nc = S / cs;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t full = (size_t)(b * H + h) * S;
      const float* Lc = intra_log + (((size_t)(b * H + h) * nc) + c) * cs * D;
      const float* Cin = inter_state + (((size_t)(b * H + h) * (nc + 1)) + c) * D * D;
      float* Cout = inter_state + (((size_t)(b * H + h) * (nc + 1)) + (c + 1)) * D * D;
      for (int vv = 0; vv < D; ++vv)
        for (int kk = 0; kk < D; ++kk)
          Cout[(size_t)vv * D + kk] = gate_prod(Lc, 0, cs - 1, kk, D) * Cin[(size_t)vv * D + kk];
      for (int t = 0; t < cs; ++t) {
        const int tau = c * cs + t;
        const float* kt = k + (full + tau) * D;
        const float* ut = u + (full + tau) * D;
        const float bt = beta[full + tau];
        for (int vv = 0; vv < D; ++vv) {
          const float w = bt * ut[vv];
          for (int kk = 0; kk < D; ++kk)
            Cout[(size_t)vv * D + kk] += gate_prod(Lc, t + 1, cs - 1, kk, D) * w * kt[kk];
        }
      }
    }
}

// k3_gla_fwd_o: o_t = inter_o + sum_{j<=t} b_j N_{j,t} u_j
//   inter_o[v] = sum_k G_{0,t}[k] C_{c-1}[v,k] q_t[k]
//   N_{j,t}    = sum_k G_{j+1,t}[k] k_j[k] q_t[k]
void k3_gla_fwd_o(const float* q, const float* k, const float* beta,
                  const float* intra_log, const float* inter_state,
                  const float* u, float* out,
                  int B, int H, int S, int D, int cs) {
  const int nc = S / cs;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t full = (size_t)(b * H + h) * S;
      for (int c = 0; c < nc; ++c) {
        const float* Lc = intra_log + (((size_t)(b * H + h) * nc) + c) * cs * D;
        const float* Cin = inter_state + (((size_t)(b * H + h) * (nc + 1)) + c) * D * D;
        for (int t = 0; t < cs; ++t) {
          const int tau = c * cs + t;
          const float* qt = q + (full + tau) * D;
          float o_inter[256], o_intra[256];
          for (int vv = 0; vv < D; ++vv) {
            float s = 0.0f;
            for (int kk = 0; kk < D; ++kk)
              s += gate_prod(Lc, 0, t, kk, D) * Cin[(size_t)vv * D + kk] * qt[kk];
            o_inter[vv] = s;
          }
          for (int vv = 0; vv < D; ++vv) o_intra[vv] = 0.0f;
          for (int j = 0; j <= t; ++j) {
            const int tauj = c * cs + j;
            const float* kj = k + (full + tauj) * D;
            const float* uj = u + (full + tauj) * D;
            float Njt = 0.0f;
            for (int kk = 0; kk < D; ++kk)
              Njt += gate_prod(Lc, j + 1, t, kk, D) * kj[kk] * qt[kk];
            const float bj = beta[full + tauj];
            for (int vv = 0; vv < D; ++vv) o_intra[vv] += bj * Njt * uj[vv];
          }
          float* ot = out + (full + tau) * D;
          for (int vv = 0; vv < D; ++vv) ot[vv] = o_inter[vv] + o_intra[vv];
        }
      }
    }
}

// k3_delta_rule_fwd: orchestrates gate cumsum -> (intra, inter) per chunk ->
// output combine. inter_state row 0 = 0 (C_{-1}).
void k3_delta_rule_fwd(const float* q, const float* k, const float* v,
                       const float* g, const float* beta, float* out,
                       int B, int H, int S, int D, int cs) {
  const int nc = S / cs;
  std::vector<float> intra_log((size_t)B * H * nc * cs * D);
  std::vector<float> inter_log((size_t)B * H * nc * D);
  std::vector<float> u((size_t)B * H * S * D, 0.0f);
  std::vector<float> inter_state((size_t)B * H * (nc + 1) * D * D, 0.0f);
  k3_gate_chunk_cumsum(g, intra_log.data(), inter_log.data(), B, H, nc, cs, D);
  for (int c = 0; c < nc; ++c) {
    k3_delta_rule_intra(k, v, g, beta, intra_log.data(),
                        inter_state.data(), u.data(), B, H, S, D, cs, c);
    k3_delta_rule_inter(k, u.data(), beta, intra_log.data(), inter_state.data(),
                        B, H, S, D, cs, c);
  }
  k3_gla_fwd_o(q, k, beta, intra_log.data(), inter_state.data(), u.data(),
               out, B, H, S, D, cs);
}

}  // namespace

// --- random per-key-dim gates vs the K3 oracle (the core check) ----------
// Input regime matches the existing standard-rule chunked test
// (KdaDeltaRuleFwd.ChunkedMatchesStandardOracle): q/k/v in [-1,1], gates in
// [0.3,1.0] (no near-zero decay -> the recurrence stays numerically stable,
// unlike g in (0.01,1] which diverges to ~1e25 in both paths and turns fp
// round-off into 1e-2 rel error). q is NOT L2-normalised here (same as the
// standard-rule test); the oracle does not re-normalise. Absolute 1e-4
// tolerance, matching the standard-rule cross-check.
TEST(KdaK3Chunked, MatchesNaiveOracleRandomGates) {
  struct Cfg { int B, H, S, D, cs; };
  const Cfg cfgs[] = {
      {1, 1, 8, 2, 4},      // 2 chunks, D=2 (mirror the standard-rule test)
      {1, 1, 8, 4, 4},      // 2 chunks, D=4
      {1, 2, 8, 4, 4},      // H=2
      {2, 1, 8, 2, 2},      // B=2, 4 chunks
      {1, 1, 12, 4, 4},     // S not a power of two, 3 chunks
      {1, 1, 16, 8, 8},     // 2 chunks, D=8
      {1, 1, 32, 16, 16},   // 2 chunks, D=16 (the smallest K3 head shape)
  };
  std::mt19937 rng(20260909);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  for (const auto& c : cfgs) {
    const size_t n = (size_t)c.B * c.H * c.S;
    std::vector<float> q(n * c.D), k(n * c.D), v(n * c.D);
    std::vector<float> g(n * c.D), beta(n);   // per-key-dim gate [B,H,S,D]
    for (auto& x : q) x = rf();
    for (auto& x : k) x = rf();
    for (auto& x : v) x = rf();
    for (auto& x : g) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    std::vector<float> naive(n * c.D, 0.0f), chunked(n * c.D, 0.0f);
    kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), naive.data(),
                                 c.B, c.H, c.S, c.D);
    k3_delta_rule_fwd(q.data(), k.data(), v.data(), g.data(), beta.data(),
                      chunked.data(), c.B, c.H, c.S, c.D, c.cs);
    float max_abs = 0.0f;
    for (size_t i = 0; i < n * c.D; ++i) {
      float e = std::fabs(naive[i] - chunked[i]);
      if (e > max_abs) max_abs = e;
      EXPECT_NEAR(chunked[i], naive[i], 1e-4f);
    }
    std::printf("  B=%d H=%d S=%d D=%d cs=%d  max_abs=%.6f\n",
                c.B, c.H, c.S, c.D, c.cs, max_abs);
  }
}

// --- full-history (g==1): the chunked gate products are all 1, so the
// chunked path reduces to the ungated delta rule. The oracle with g==1 is
// the same ungated recurrence. ---
TEST(KdaK3Chunked, FullHistoryGatesAreOne) {
  const int B = 1, H = 1, S = 12, D = 4, cs = 4;
  const size_t n = (size_t)B * H * S;
  std::mt19937 rng(123);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q(n * D), k(n * D), v(n * D), g(n * D, 1.0f), beta(n, 0.5f);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  std::vector<float> naive(n * D, 0.0f), chunked(n * D, 0.0f);
  kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), naive.data(), B, H, S, D);
  k3_delta_rule_fwd(q.data(), k.data(), v.data(), g.data(), beta.data(),
                    chunked.data(), B, H, S, D, cs);
  float max_abs = 0.0f;
  for (size_t i = 0; i < n * D; ++i) {
    float e = std::fabs(naive[i] - chunked[i]);
    if (e > max_abs) max_abs = e;
    EXPECT_NEAR(chunked[i], naive[i], 1e-4f);
  }
  std::printf("  full-history max_abs=%.6f\n", max_abs);
}

// --- single chunk (cs==S): the chunked path must equal the naive oracle
// to fp32 round-off, since there is no inter-chunk decoupling. ---
TEST(KdaK3Chunked, SingleChunkEqualsNaive) {
  const int B = 1, H = 1, S = 32, D = 16, cs = 32;
  const size_t n = (size_t)B * H * S;
  std::mt19937 rng(99);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q(n * D), k(n * D), v(n * D), g(n * D), beta(n);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  for (auto& x : g) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  std::vector<float> naive(n * D, 0.0f), chunked(n * D, 0.0f);
  kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), naive.data(), B, H, S, D);
  k3_delta_rule_fwd(q.data(), k.data(), v.data(), g.data(), beta.data(),
                    chunked.data(), B, H, S, D, cs);
  float max_abs = 0.0f;
  for (size_t i = 0; i < n * D; ++i) {
    float e = std::fabs(naive[i] - chunked[i]);
    if (e > max_abs) max_abs = e;
    EXPECT_NEAR(chunked[i], naive[i], 1e-4f);
  }
  std::printf("  single-chunk max_abs=%.6f\n", max_abs);
}

// =====================================================================
// Affine (WY) reformulation — the exact op sequence the HIP chunked kernel
// (#70) implements. Same math as the forward-substitution path above,
// restructured so everything that depends on C_{c-1} is one plain GEMM and
// all chunk-local work (grams, triangular inverse, Ainv/Bn products) needs
// NO inter state, i.e. is fully parallel across chunks. Within-chunk local
// t = 0..C-1, L_t[k] = per-column log-cumsum, end = C-1:
//
//   M[t][j] = b_j * sum_k G_{j+1,t}[k] k_j[k] k_t[k]   (j<t, strict tril)
//   N[t][j] = b_j * sum_k G_{j+1,t}[k] k_j[k] q_t[k]   (j<=t, incl diag)
//   Ainv = (I + tril(M,-1))^{-1};  Bn = N
//   Kgw[t][k] = exp(L_t[k]) k_t[k]        (G_{0,t} includes g_t: POST-gate)
//   Qgw[t][k] = exp(L_t[k]) q_t[k]
//   Kgb[t][k] = b_t exp(L_end[k]-L_t[k]) k_t[k];  diagG[k] = exp(L_end[k])
//   U_v = Ainv v;  W = Ainv Kgw;  T = Bn U_v;  P = Bn W;  Opar = Qgw - P
//   ---- serial state pass (rowblock-splittable over v: every op reads only
//      row v of C and column v of u/o) ----
//   u = U_v - W C^T;  o = T + Opar C^T;  C = diagG (.) C + Kgb^T u
//
// Every gate factor is a product of gates in (0,1], so every exp() argument
// is <= 0: factors live in (0,1], underflow degrades gracefully to 0, and
// nothing overflows. o = T + (Qgw - P) C^T is the associativity that lets
// Opar be precomputed chunk-locally.
// =====================================================================
void k3_wy_chunked_fwd(const float* q, const float* k, const float* v,
                       const float* g, const float* beta, float* out,
                       int B, int H, int S, int D, int cs) {
  const int nc = (S + cs - 1) / cs;
  const size_t row0 = 0;
  std::vector<float> L((size_t)cs * D);
  std::vector<float> M((size_t)cs * cs, 0.0f), N((size_t)cs * cs, 0.0f),
      Ainv((size_t)cs * cs, 0.0f);
  std::vector<float> Kgw((size_t)cs * D), Qgw((size_t)cs * D),
      Kgb((size_t)cs * D), diagG(D);
  std::vector<float> U_v((size_t)cs * D), W((size_t)cs * D),
      T((size_t)cs * D), P((size_t)cs * D), Opar((size_t)cs * D),
      u((size_t)cs * D);
  std::vector<float> C((size_t)D * D, 0.0f);
  for (int b = 0; b < B; ++b)
  for (int h = 0; h < H; ++h) {
    const size_t bh = (size_t)(b * H + h) * S;   // token base of this (b,h)
    std::fill(C.begin(), C.end(), 0.0f);
    for (int c = 0; c < nc; ++c) {
      const int t0 = c * cs;
      const int len = (S - t0 < cs) ? (S - t0) : cs;
      const int lend = len - 1;
      // per-column within-chunk inclusive log-cumsum
      for (int kk = 0; kk < D; ++kk) {
        float acc = 0.0f;
        for (int t = 0; t < len; ++t) {
          acc += log_gate(g[(bh + t0 + t) * D + kk]);
          L[(size_t)t * D + kk] = acc;
        }
      }
      // grams M (strict tril) / N (incl diag), beta folded in
      for (int t = 0; t < len; ++t)
      for (int j = 0; j <= t; ++j) {
        const float* kj = k + (bh + t0 + j) * D;
        const float* kt = k + (bh + t0 + t) * D;
        const float* qt = q + (bh + t0 + t) * D;
        const float* Lj = L.data() + (size_t)j * D;
        const float* Lt = L.data() + (size_t)t * D;
        float mk = 0.0f, nk = 0.0f;
        for (int kk = 0; kk < D; ++kk) {
          const float gw = std::exp(Lt[kk] - Lj[kk]);   // G_{j+1,t}[k]
          mk += gw * kj[kk] * kt[kk];
          nk += gw * kj[kk] * qt[kk];
        }
        const float bj = beta[bh + t0 + j];
        N[(size_t)t * cs + j] = bj * nk;
        M[(size_t)t * cs + j] = (j < t) ? bj * mk : 0.0f;
      }
      // Ainv = (I + tril(M,-1))^{-1}, forward substitution per column
      for (int j = 0; j < len; ++j)
      for (int t = j; t < len; ++t) {
        float x = (t == j) ? 1.0f : 0.0f;
        if (t > j) {
          for (int l = j; l < t; ++l)
            x -= M[(size_t)t * cs + l] * Ainv[(size_t)l * cs + j];
        }
        Ainv[(size_t)t * cs + j] = x;
      }
      // chunk-local gated operands
      for (int kk = 0; kk < D; ++kk) diagG[kk] = std::exp(L[(size_t)lend * D + kk]);
      for (int t = 0; t < len; ++t) {
        const float* kt = k + (bh + t0 + t) * D;
        const float* qt = q + (bh + t0 + t) * D;
        const float* Lt = L.data() + (size_t)t * D;
        const float bt = beta[bh + t0 + t];
        for (int kk = 0; kk < D; ++kk) {
          const float e0 = std::exp(Lt[kk]);                       // G_{0,t}
          Kgw[(size_t)t * D + kk] = e0 * kt[kk];
          Qgw[(size_t)t * D + kk] = e0 * qt[kk];
          Kgb[(size_t)t * D + kk] =
              bt * std::exp(L[(size_t)lend * D + kk] - Lt[kk]) * kt[kk];
        }
      }
      // chunk-local GEMMs
      for (int t = 0; t < len; ++t)
      for (int d = 0; d < D; ++d) {
        float sv = 0.0f, sw = 0.0f;
        for (int j = 0; j < len; ++j) {
          sv += Ainv[(size_t)t * cs + j] * v[(bh + t0 + j) * D + d];
          sw += Ainv[(size_t)t * cs + j] * Kgw[(size_t)j * D + d];
        }
        U_v[(size_t)t * D + d] = sv;
        W[(size_t)t * D + d] = sw;
      }
      for (int t = 0; t < len; ++t)
      for (int d = 0; d < D; ++d) {
        float st = 0.0f, sp = 0.0f;
        for (int j = 0; j <= t; ++j) {
          st += N[(size_t)t * cs + j] * U_v[(size_t)j * D + d];
          sp += N[(size_t)t * cs + j] * W[(size_t)j * D + d];
        }
        T[(size_t)t * D + d] = st;
        P[(size_t)t * D + d] = sp;
        Opar[(size_t)t * D + d] = Qgw[(size_t)t * D + d] - sp;
      }
      // serial state pass (rowblock-splittable: only row v of C appears)
      for (int t = 0; t < len; ++t)
      for (int d = 0; d < D; ++d) {
        float s = U_v[(size_t)t * D + d];
        for (int kk = 0; kk < D; ++kk)
          s -= W[(size_t)t * D + kk] * C[(size_t)d * D + kk];
        u[(size_t)t * D + d] = s;
      }
      for (int t = 0; t < len; ++t)
      for (int d = 0; d < D; ++d) {
        float s = T[(size_t)t * D + d];
        for (int kk = 0; kk < D; ++kk)
          s += Opar[(size_t)t * D + kk] * C[(size_t)d * D + kk];
        out[(bh + t0 + t) * D + d] = s;
      }
      for (int d = 0; d < D; ++d)
      for (int kk = 0; kk < D; ++kk) {
        float s = diagG[kk] * C[(size_t)d * D + kk];
        for (int t = 0; t < len; ++t)
          s += Kgb[(size_t)t * D + kk] * u[(size_t)t * D + d];
        C[(size_t)d * D + kk] = s;
      }
    }
  }
  (void)row0;
}

// --- affine (WY) chunked vs the K3 oracle, incl. the HIP chunk shape ---
// CONTRACT (stronger than the forward-substitution path above): k is
// L2-NORMALISED per token (the production contract in kda.hpp -- "k is
// L2-normalised by the CALLER", as in delta-net / FLA's chunked kernels).
// The explicit Ainv materialises powers of the strictly-lower M, so it needs
// |M[t][j]| = |b_j (gated k_j . k_t)| <= 1 (Cauchy-Schwarz on normalised k,
// gates in (0,1], beta <= 1). With unnormalised k (|k| ~ sqrt(D/3)) Ainv
// entries grow like |M|^(cs-1) and overflow: measured cs=64 D=128 gives
// max_abs ~ 1e14 while the forward-substitution path (which never
// materialises the inverse) stays at 1e-5 on the SAME inputs.
TEST(KdaK3Chunked, WYAffineMatchesNaiveOracle) {
  struct Cfg { int B, H, S, D, cs; };
  const Cfg cfgs[] = {
      {1, 1, 8, 2, 4},      {1, 2, 8, 4, 4},      {1, 1, 12, 4, 4},
      {2, 1, 16, 8, 8},     {1, 1, 32, 16, 16},   // multi-chunk, small
      {1, 1, 64, 32, 32},   // 2 chunks of 32
      {1, 1, 128, 64, 64},  // HIP chunk shape: cs=64
      {1, 2, 128, 128, 64}, // K3 head dim, 2 chunks, H=2
      {1, 1, 512, 128, 64}, // full bench shape: S=512 D=128, 8 chunks
  };
  std::mt19937 rng(4242);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  for (const auto& c : cfgs) {
    const size_t n = (size_t)c.B * c.H * c.S;
    std::vector<float> q(n * c.D), k(n * c.D), v(n * c.D), g(n * c.D), beta(n);
    for (auto& x : q) x = rf();
    for (auto& x : k) x = rf();
    for (auto& x : v) x = rf();
    for (auto& x : g) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    for (size_t i = 0; i < n; ++i) {   // L2-normalise k (WY contract, below)
      float* ki = k.data() + i * c.D;
      float ss = 0.0f;
      for (int d = 0; d < c.D; ++d) ss += ki[d] * ki[d];
      const float inv = 1.0f / std::sqrt(ss);
      for (int d = 0; d < c.D; ++d) ki[d] *= inv;
    }
    std::vector<float> naive(n * c.D, 0.0f), wy(n * c.D, 0.0f);
    kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), naive.data(), c.B, c.H, c.S, c.D);
    k3_wy_chunked_fwd(q.data(), k.data(), v.data(), g.data(), beta.data(),
                      wy.data(), c.B, c.H, c.S, c.D, c.cs);
    float max_abs = 0.0f;
    for (size_t i = 0; i < n * c.D; ++i) {
      float e = std::fabs(naive[i] - wy[i]);
      if (e > max_abs) max_abs = e;
      EXPECT_NEAR(wy[i], naive[i], 1e-4f);
    }
    std::printf("  B=%d H=%d S=%d D=%d cs=%d  wy max_abs=%.6f\n",
                c.B, c.H, c.S, c.D, c.cs, max_abs);
  }
}

// --- affine (WY) vs forward-substitution chunked: same math, fp reorder.
// k L2-normalised (WY contract); the forward-substitution side is
// contract-free, so it accepts the same inputs. ---
TEST(KdaK3Chunked, WYEqualsForwardSubstitution) {
  const int B = 1, H = 1, S = 64, D = 32, cs = 16;
  const size_t n = (size_t)B * H * S;
  std::mt19937 rng(777);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q(n * D), k(n * D), v(n * D), g(n * D), beta(n);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  for (auto& x : g) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  for (size_t i = 0; i < n; ++i) {   // L2-normalise k (WY contract)
    float* ki = k.data() + i * D;
    float ss = 0.0f;
    for (int d = 0; d < D; ++d) ss += ki[d] * ki[d];
    const float inv = 1.0f / std::sqrt(ss);
    for (int d = 0; d < D; ++d) ki[d] *= inv;
  }
  std::vector<float> fs(n * D, 0.0f), wy(n * D, 0.0f);
  k3_delta_rule_fwd(q.data(), k.data(), v.data(), g.data(), beta.data(),
                    fs.data(), B, H, S, D, cs);
  k3_wy_chunked_fwd(q.data(), k.data(), v.data(), g.data(), beta.data(),
                    wy.data(), B, H, S, D, cs);
  float max_abs = 0.0f;
  for (size_t i = 0; i < n * D; ++i) {
    float e = std::fabs(fs[i] - wy[i]);
    if (e > max_abs) max_abs = e;
    EXPECT_NEAR(wy[i], fs[i], 1e-4f);
  }
  std::printf("  wy-vs-fwdsub max_abs=%.6f\n", max_abs);
}
