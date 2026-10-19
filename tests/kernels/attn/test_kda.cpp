// tests/kernels/attn/test_kda.cpp
//
// Host tests for the Kimi Delta Attention layer (issue #21, #45). The
// naive oracle (kda_naive_delta_rule_fwd_cpu) implements the K3 per-key-dim
// recurrence and is checked two ways: a hand-computed case (NaivePerKeyDim
// HandChecked) and the zero-gate independence property (NaiveZeroGateIndep
// endence). The chunked forward (kda_delta_rule_fwd_cpu, the L2..L6
// pipeline) implements the OLD standard gated delta rule (scalar gate
// [B,H,S], pre-gate prediction) -- a DIFFERENT recurrence -- and is cross-
// checked against an inline standard-rule oracle (kda_standard_delta_rule_fwd
// below) at random gates and at full-history (g==1) accumulation. The
// block-anchored grams oracle (#L8a, rung 1 of the native HIP/CK ladder) is
// cross-checked against the exact per-pair grams of the chunked WY path
// (k3_exact_grams below).
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/kda.hpp"

using namespace vkernels::kernels;

namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float dot(const float* x, const float* y, int D) {
  float s = 0.0f;
  for (int d = 0; d < D; ++d) s += x[d] * y[d];
  return s;
}

// The STANDARD gated delta rule (scalar gate [B,H,S], pre-gate prediction) --
// exactly the recurrence the chunked CPU (L4-L7) implements, spelled inline
// so the chunked path stays cross-checked after the naive oracle was
// repurposed for the K3 per-key-dim rule (issue #45). Mirrors the pre-#45
// kda_naive_delta_rule_fwd_cpu bit-for-bit.
//
//   S_0 = 0
//   for t: a_t = S_{t-1} k_t            (predict from PRE-gate state)
//          S_t = g_t S_{t-1} + b_t (v_t - a_t) k_t^T
//          o_t = S_t q_t
void kda_standard_delta_rule_fwd(const float* q, const float* k,
                                 const float* v, const float* g,
                                 const float* beta, float* out,
                                 int B, int H, int S, int D) {
  std::vector<float> state((size_t)D * D, 0.0f);
  std::vector<float> a(D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      std::fill(state.begin(), state.end(), 0.0f);
      const size_t bh = (size_t)(b * H + h) * S;
      for (int t = 0; t < S; ++t) {
        const float* kt = k + (bh + t) * D;
        const float* vt = v + (bh + t) * D;
        const float* qt = q + (bh + t) * D;
        const float gt = g[bh + t];        // scalar gate [B,H,S]
        const float bt = beta[bh + t];
        for (int d = 0; d < D; ++d)
          a[d] = dot(state.data() + (size_t)d * D, kt, D);
        for (int d = 0; d < D; ++d) {
          const float ud = vt[d] - a[d];
          float* Srow = state.data() + (size_t)d * D;
          for (int e = 0; e < D; ++e)
            Srow[e] = gt * Srow[e] + bt * ud * kt[e];
          out[(bh + t) * D + d] = dot(Srow, qt, D);
        }
      }
    }
}

}  // namespace

// --- #L1 layer_norm_gated: hand-checked identity ----------------------------
TEST(KdaLayerNormGated, IdentityWeightAndUnitGate) {
  // weight = 1, gate = large positive → silu(gate) ≈ gate; with x all ones,
  // rms = 1, so out ≈ 1 * 1 * gate (silu). Use gate=5 → silu≈4.966.
  constexpr int N = 1, D = 4;
  std::vector<float> x(D, 1.0f), w(D, 1.0f), gate(D, 5.0f), out(D, -1);
  kda_layer_norm_gated_cpu(x.data(), w.data(), gate.data(), out.data(),
                           N, D, 1e-6f);
  const float silu5 = 5.0f * sigmoid(5.0f);
  for (int d = 0; d < D; ++d) EXPECT_NEAR(out[d], silu5, 1e-5f);
}

TEST(KdaLayerNormGated, ZeroGateZeroOutput) {
  // gate = 0 → silu(0) = 0 → output is zero regardless of x/weight.
  constexpr int N = 2, D = 3;
  std::vector<float> x(N * D, 7.0f), w(D, 2.0f), gate(N * D, 0.0f), out(N * D, -1);
  kda_layer_norm_gated_cpu(x.data(), w.data(), gate.data(), out.data(),
                           N, D, 1e-6f);
  for (float v : out) EXPECT_NEAR(v, 0.0f, 1e-6f);
}

TEST(KdaLayerNormGated, RMSNormalizesThenScales) {
  // x = [3,4,0,0] -> rms = sqrt((9+16)/4) = sqrt(6.25) = 2.5.
  // weight=[1,1,1,1], gate=[10,10,10,10] -> silu(10)~9.9995.
  // out = x/2.5 * 1 * ~1 (gate≈1 since silu(10)/10 ≈ 0.99995).
  constexpr int N = 1, D = 4;
  std::vector<float> x = {3, 4, 0, 0}, w(D, 1.0f), gate(D, 10.0f), out(D, -1);
  kda_layer_norm_gated_cpu(x.data(), w.data(), gate.data(), out.data(),
                           N, D, 1e-12f);
  const float s = sigmoid(10.0f) * 10.0f;  // silu(10)
  EXPECT_NEAR(out[0], 3.0f / 2.5f * s, 1e-5f);
  EXPECT_NEAR(out[1], 4.0f / 2.5f * s, 1e-5f);
  EXPECT_NEAR(out[2], 0.0f, 1e-6f);
  EXPECT_NEAR(out[3], 0.0f, 1e-6f);
}

// --- #L2 gate_chunk_cumsum: intra + inter against independent refs ----------
TEST(KdaGateChunkCumsum, MatchesIndependentRefs) {
  constexpr int B = 1, H = 1, nc = 3, cs = 4;
  std::vector<float> g(nc * cs);
  std::mt19937 rng(11);
  for (auto& v : g) v = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;  // (0.3, 1.0]
  std::vector<float> intra(nc * cs), inter(nc);
  kda_gate_chunk_cumsum_cpu(g.data(), intra.data(), inter.data(),
                            B, H, nc, cs);
  // intra[c,t] = sum_{l<=t} log(g[c,l])
  for (int c = 0; c < nc; ++c) {
    float acc = 0;
    for (int t = 0; t < cs; ++t) {
      acc += std::log(g[c * cs + t]);
      EXPECT_NEAR(intra[c * cs + t], acc, 1e-6f);
    }
  }
  // inter[c] = sum_{c'<c} chunk_log[c'] = sum_{c'<c} intra[c', cs-1]
  for (int c = 0; c < nc; ++c) {
    float acc = 0;
    for (int cp = 0; cp < c; ++cp) acc += intra[cp * cs + (cs - 1)];
    EXPECT_NEAR(inter[c], acc, 1e-6f);
  }
}

TEST(KdaGateChunkCumsum, ZeroGateClamped) {
  // g==0 in chunk 1 → that chunk's intra drops to the floor; the chunk total
  // becomes ~kNegLogFloor, so inter[2] ≈ intra[0,cs-1] + kNegLogFloor.
  constexpr int B = 1, H = 1, nc = 2, cs = 2;
  std::vector<float> g = {1.0f, 1.0f, 0.0f, 0.0f};
  std::vector<float> intra(nc * cs), inter(nc);
  kda_gate_chunk_cumsum_cpu(g.data(), intra.data(), inter.data(),
                            B, H, nc, cs);
  EXPECT_NEAR(intra[0], 0.0f, 1e-6f);
  EXPECT_NEAR(intra[1], 0.0f, 1e-6f);
  EXPECT_LT(intra[2], -1.0e8f);  // log(0) clamped to ~-1e9
  EXPECT_LT(intra[3], -1.0e8f);
  EXPECT_NEAR(inter[0], 0.0f, 1e-6f);
  EXPECT_NEAR(inter[1], 0.0f, 1e-6f);
}

// --- #L3 naive oracle: K3 per-key-dim gated delta rule (post-gate  ---
// prediction). The hand-checked case + the zero-gate independence
// property below validate the K3 recurrence directly; the chunked CPU
// (L4-L7) implements a DIFFERENT recurrence (standard, scalar gate) and
// is cross-checked against the inline oracle further down.
TEST(KdaDeltaRuleFwd, NaivePerKeyDimHandChecked) {
  // B=1, H=1, S=2, D=2. Inputs chosen so the expected output is computable
  // by hand (see the derivation in kda.hpp).
  //   q_t0=[.6,.8] k_t0=[.8,.6] v_t0=[1,.5]  g_t0=[.9,.8] beta_t0=.5
  //   q_t1=[0,1]   k_t1=[1,0]   v_t1=[.3,.7] g_t1=[.7,.6] beta_t1=.4
  //
  // t=0: S0=0 -> a=0 -> S0=beta v⊗k -> o0 = S0 q0 = [0.48, 0.24]
  // t=1: S'=g1⊙S0 -> a=S'·k1 -> S1=S'+beta(v-a)⊗k1 -> o1=S1·q1=[0.18,0.09]
  constexpr int B = 1, H = 1, S = 2, D = 2;
  std::vector<float> q = {.6f, .8f,  0.f, 1.f};
  std::vector<float> k = {.8f, .6f,  1.f, 0.f};
  std::vector<float> v = {1.f, .5f,  .3f, .7f};
  std::vector<float> g = {.9f, .8f,  .7f, .6f};  // [B,H,S,D] per-key-dim
  std::vector<float> beta = {.5f, .4f};            // [B,H,S] scalar
  std::vector<float> out(q.size(), 0);
  kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), out.data(), B, H, S, D);
  EXPECT_NEAR(out[0], 0.48f, 1e-5f);   // o_t0[0]
  EXPECT_NEAR(out[1], 0.24f, 1e-5f);   // o_t0[1]
  EXPECT_NEAR(out[2], 0.18f, 1e-5f);   // o_t1[0]
  EXPECT_NEAR(out[3], 0.09f, 1e-5f);   // o_t1[1]
}

TEST(KdaDeltaRuleFwd, NaiveZeroGateIndependence) {
  // Per-key-dim g=0 everywhere -> S'=0 each token -> a=0 ->
  // S_t = beta_t v_t⊗k_t -> o_t = beta_t (k_t·q_t) v_t.  Each output
  // depends ONLY on its own token (no cross-token history); a strong
  // property that catches gate-shape / prediction-order regressions.
  constexpr int B = 1, H = 2, S = 8, D = 4;
  std::mt19937 rng(42);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q((size_t)B * H * S * D), k(q.size()), v(q.size());
  std::vector<float> g(q.size(), 0.0f);               // per-key-dim, all zero
  std::vector<float> beta((size_t)B * H * S);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
  std::vector<float> out(q.size(), 0);
  kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), out.data(), B, H, S, D);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
      const size_t bh = (size_t)(b * H + h) * S;
      for (int t = 0; t < S; ++t) {
        const float* kt = k.data() + (bh + t) * D;
        const float* vt = v.data() + (bh + t) * D;
        const float* qt = q.data() + (bh + t) * D;
        const float bt = beta[bh + t];
        float kq = 0;
        for (int e = 0; e < D; ++e) kq += kt[e] * qt[e];
        for (int d = 0; d < D; ++d)
          EXPECT_NEAR(out[(bh + t) * D + d], bt * kq * vt[d], 1e-4f);
      }
    }
}

// --- #L7 chunked forward vs standard-rule oracle (the core correctness check) --
// The chunked pipeline (L2..L6) implements the standard gated delta rule
// (scalar gate [B,H,S], pre-gate prediction) -- a different recurrence from
// the K3 naive oracle above. It is cross-checked against the inline
// kda_standard_delta_rule_fwd (NOT kda_naive_delta_rule_fwd_cpu), at random
// gates across configs and at full-history (g==1) accumulation.
struct ChunkConfig { int B, H, S, D, chunk; };
TEST(KdaDeltaRuleFwd, ChunkedMatchesStandardOracle) {
  const ChunkConfig configs[] = {
      {1, 1, 8, 2, 4},
      {1, 2, 8, 4, 4},
      {2, 1, 8, 2, 2},
      {1, 1, 12, 4, 4},  // S not a power of two
  };
  std::mt19937 rng(7);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  for (const auto& c : configs) {
    std::vector<float> q((size_t)c.B * c.H * c.S * c.D);
    std::vector<float> k(q.size()), v(q.size());
    std::vector<float> g((size_t)c.B * c.H * c.S);   // scalar gate [B,H,S]
    std::vector<float> beta(g.size());
    for (auto& x : q) x = rf();
    for (auto& x : k) x = rf();
    for (auto& x : v) x = rf();
    for (auto& x : g) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    std::vector<float> oracle(q.size(), 0), chunked(q.size(), 0);
    kda_standard_delta_rule_fwd(q.data(), k.data(), v.data(), g.data(),
                                beta.data(), oracle.data(),
                                c.B, c.H, c.S, c.D);
    kda_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                           beta.data(), chunked.data(),
                           c.B, c.H, c.S, c.D, c.chunk);
    for (size_t i = 0; i < oracle.size(); ++i)
      EXPECT_NEAR(chunked[i], oracle[i], 1e-4f);
  }
}

TEST(KdaDeltaRuleFwd, ZeroForgettingMatchesStandardOracle) {
  // g == 1 everywhere -> no forgetting -> the state accumulates the full
  // delta history; a strong test of inter-chunk accumulation. beta = 0.5.
  constexpr int B = 1, H = 1, S = 12, D = 4, chunk = 4;
  std::mt19937 rng(123);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q((size_t)B * H * S * D);
  std::vector<float> k(q.size()), v(q.size());
  std::vector<float> g((size_t)B * H * S, 1.0f);
  std::vector<float> beta(g.size(), 0.5f);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  std::vector<float> oracle(q.size(), 0), chunked(q.size(), 0);
  kda_standard_delta_rule_fwd(q.data(), k.data(), v.data(), g.data(),
                              beta.data(), oracle.data(), B, H, S, D);
  kda_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                         beta.data(), chunked.data(), B, H, S, D, chunk);
  for (size_t i = 0; i < oracle.size(); ++i)
    EXPECT_NEAR(chunked[i], oracle[i], 1e-3f);
}

// --- #P pack_bitmatrix: hand-checked + round-trip ---------------------------
TEST(KdaPackBitmatrix, HandChecked) {
  // 10 bits: 1,0,1,1,0,0,0,1, 0,1  -> byte0=0b10110001=0xB1, byte1=0b01000000=0x40
  std::vector<uint8_t> bits = {1,0,1,1,0,0,0,1, 0,1};
  std::vector<uint8_t> packed(2, 0);
  kda_pack_bitmatrix_cpu(bits.data(), packed.data(), bits.size());
  EXPECT_EQ(packed[0], 0xB1);
  EXPECT_EQ(packed[1], 0x40);
}

TEST(KdaPackBitmatrix, RoundTrip) {
  std::mt19937 rng(3);
  for (int trial = 0; trial < 5; ++trial) {
    std::size_t n = 1 + rng() % 200;
    std::vector<uint8_t> bits(n);
    for (auto& b : bits) b = rng() % 2;
    std::vector<uint8_t> packed((n + 7) / 8, 0);
    kda_pack_bitmatrix_cpu(bits.data(), packed.data(), n);
    for (std::size_t k = 0; k < n; ++k) {
      const std::size_t byte = k / 8;
      const std::size_t bit = 7 - (k % 8);
      EXPECT_EQ(((packed[byte] >> bit) & 1u), bits[k]);
    }
  }
}

TEST(KdaPackBitmatrix, EmptyIsNoOp) {
  std::vector<uint8_t> packed = {9};
  kda_pack_bitmatrix_cpu(nullptr, packed.data(), 0);
  EXPECT_EQ(packed[0], 9);
}

// --- #L8a block-anchored grams (rung 1 of the native HIP/CK ladder) ----
// Reference: the EXACT per-pair grams inside kda_delta_rule_fwd_state_cpu
// (M[t][j] = b_j Σ_k exp(L_t−L_j) k_j k_t, strict lower; N the same with
// q, tril-inclusive; upper cells zeroed), spelled inline. The blocked
// oracle (kda_k3_grams_blocked_cpu) must match it within the fp32
// round-off of the two-sided exp factorisation — the same class of
// agreement the trainer measured for _wy_grams vs its mask reference.

// Per-key within-chunk INCLUSIVE log-cumsum for all (b,h,c) — the layout
// the oracle consumes ([B,H,nc,cs,D], monotone decreasing in t; log(0)
// clamp exactly as kda.cpp's log_gate).
static void k3_build_L(const std::vector<float>& g, int B, int H, int nc,
                        int S, int D, int cs, std::vector<float>* L) {
  L->assign((size_t)B * H * nc * cs * D, 0.0f);
  for (int bh = 0; bh < B * H; ++bh)
    for (int c = 0; c < nc; ++c) {
      const size_t tok = (size_t)bh * S + (size_t)c * cs;
      const size_t lb = ((size_t)bh * nc + c) * cs * D;
      for (int e = 0; e < D; ++e) {
        float acc = 0.0f;
        for (int t = 0; t < cs; ++t) {
          const float gg = g[(tok + t) * D + e];
          acc += (gg <= 0.0f) ? -1.0e9f : std::log(gg);
          (*L)[lb + (size_t)t * D + e] = acc;
        }
      }
    }
}

// The exact per-pair grams (the reference the blocked form must match).
// Same signature discipline as kda_k3_grams_blocked_cpu but taking the
// normal-space gate g (it builds its own L via k3_build_L).
static void k3_exact_grams(const std::vector<float>& q,
                           const std::vector<float>& k,
                           const std::vector<float>& beta,
                           const std::vector<float>& g, int B, int H, int nc,
                           int S, int D, int cs, std::vector<float>* M,
                           std::vector<float>* N) {
  std::vector<float> L;
  k3_build_L(g, B, H, nc, S, D, cs, &L);
  M->assign((size_t)B * H * nc * cs * cs, 0.0f);
  N->assign((size_t)B * H * nc * cs * cs, 0.0f);
  for (int bh = 0; bh < B * H; ++bh)
    for (int c = 0; c < nc; ++c) {
      const size_t tok = (size_t)bh * S + (size_t)c * cs;
      const size_t lb = ((size_t)bh * nc + c) * cs * D;
      const size_t lbq = ((size_t)bh * nc + c) * cs * cs;
      for (int t = 0; t < cs; ++t)
        for (int j = 0; j <= t; ++j) {
          float mk = 0.0f, nk = 0.0f;
          for (int e = 0; e < D; ++e) {
            const float gw =
                std::exp(L[lb + (size_t)t * D + e] - L[lb + (size_t)j * D + e]);
            mk += gw * k[(tok + j) * D + e] * k[(tok + t) * D + e];
            nk += gw * k[(tok + j) * D + e] * q[(tok + t) * D + e];
          }
          const float bj = beta[tok + j];
          (*M)[lbq + (size_t)t * cs + j] = (j < t) ? bj * mk : 0.0f;
          (*N)[lbq + (size_t)t * cs + j] = bj * nk;
        }
    }
}

TEST(KdaK3GramsBlocked, MatchesExactPairwiseGrams) {
  // Random per-key gates in (0.05, 1], k L2-normalised (the #L8 contract),
  // several (cs, blk, D) combos incl. the K3 head shapes and the device
  // rung's blk=16. Tolerance: the blocked form rounds exp once per factor
  // (rel ~1e-7 each; |M|,|N| <= ~1 after beta) — 2e-4 abs is generous but
  // still 3 orders below anything a gate-shape or sign regression moves.
  struct { int cs, blk, D; } cases[] = {
      {16, 4, 16}, {16, 8, 32}, {32, 8, 128}, {64, 16, 128}, {64, 16, 64}};
  for (const auto& tc : cases) {
    const int B = 1, H = 2, nc = 3, S = nc * tc.cs;
    std::mt19937 rng(7 + tc.cs + tc.blk);
    auto rf = [&]() { return (float)(rng() % 2000) / 1000.0f - 1.0f; };
    auto rg = [&]() { return 0.05f + 0.95f * (rng() % 1000) / 1000.0f; };
    const size_t m = (size_t)B * H * S * tc.D;
    std::vector<float> q(m), k(m), g(m), beta((size_t)B * H * S);
    for (auto& x : q) x = rf();
    for (auto& x : k) x = rf();
    for (auto& x : g) x = rg();
    for (auto& x : beta) x = 0.3f + 0.7f * (rng() % 1000) / 1000.0f;
    for (int x = 0; x < B * H * S; ++x) {  // L2-normalise k (the contract)
      float* kk = k.data() + (size_t)x * tc.D;
      float n = 0.0f;
      for (int e = 0; e < tc.D; ++e) n += kk[e] * kk[e];
      n = 1.0f / std::sqrt(n + 1e-12f);
      for (int e = 0; e < tc.D; ++e) kk[e] *= n;
    }
    std::vector<float> L;
    k3_build_L(g, B, H, nc, S, tc.D, tc.cs, &L);
    std::vector<float> Mb((size_t)B * H * nc * tc.cs * tc.cs), Nb(Mb.size()),
        Me, Ne;
    kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(), L.data(),
                             Mb.data(), Nb.data(), B, H, nc, S, tc.D, tc.cs,
                             tc.blk);
    k3_exact_grams(q, k, beta, g, B, H, nc, S, tc.D, tc.cs, &Me, &Ne);
    for (size_t i = 0; i < Me.size(); ++i) {
      EXPECT_NEAR(Mb[i], Me[i], 2e-4f);
      EXPECT_NEAR(Nb[i], Ne[i], 2e-4f);
    }
  }
}

TEST(KdaK3GramsBlocked, FullHistoryEdgeAndStructure) {
  // g == 1 everywhere -> L == 0 -> the grams degenerate to the plain
  // beta-weighted Gram/dot products: M[t][j] = b_j (k_j . k_t),
  // N[t][t] = b_t (k_t . q_t) on the diagonal. Also checks the cell
  // structure the (2b) local kernel relies on: M strict lower + zero
  // diagonal, N tril-inclusive, strict-upper cells of both exactly 0.
  constexpr int B = 1, H = 1, nc = 2, cs = 32, blk = 16, D = 32;
  const int S = nc * cs;
  std::mt19937 rng(3);
  auto rf = [&]() { return (float)(rng() % 2000) / 1000.0f - 1.0f; };
  const size_t m = (size_t)B * H * S * D;
  std::vector<float> q(m), k(m), g(m, 1.0f), beta((size_t)B * H * S, 0.5f),
      L((size_t)B * H * nc * cs * D, 0.0f);  // log(1)=0 -> L stays 0
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (int x = 0; x < B * H * S; ++x) {
    float* kk = k.data() + (size_t)x * D;
    float n = 0.0f;
    for (int e = 0; e < D; ++e) n += kk[e] * kk[e];
    n = 1.0f / std::sqrt(n + 1e-12f);
    for (int e = 0; e < D; ++e) kk[e] *= n;
  }
  std::vector<float> M((size_t)B * H * nc * cs * cs), N(M.size());
  kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(), L.data(),
                           M.data(), N.data(), B, H, nc, S, D, cs, blk);
  for (int c = 0; c < nc; ++c) {
    const size_t lbq = ((size_t)c) * cs * cs;
    for (int t = 0; t < cs; ++t)
      for (int j = 0; j < cs; ++j) {
        const float mcell = M[lbq + (size_t)t * cs + j];
        const float ncell = N[lbq + (size_t)t * cs + j];
        if (j > t) {
          EXPECT_EQ(mcell, 0.0f);
          EXPECT_EQ(ncell, 0.0f);
        }
        if (j == t) {
          EXPECT_EQ(mcell, 0.0f);  // M diagonal is 0
          float kq = 0.0f;
          for (int e = 0; e < D; ++e)
            kq += k[((size_t)c * cs + t) * D + e] * q[((size_t)c * cs + t) * D + e];
          EXPECT_NEAR(ncell, 0.5f * kq, 1e-5f);  // b_t (k_t . q_t)
        }
        if (j < t) {
          float kk_ = 0.0f;
          for (int e = 0; e < D; ++e)
            kk_ += k[((size_t)c * cs + j) * D + e] * k[((size_t)c * cs + t) * D + e];
          EXPECT_NEAR(mcell, 0.5f * kk_, 1e-5f);  // b_j (k_j . k_t)
        }
      }
  }
}

TEST(KdaK3GramsBlocked, ContractThrowsAndZeroSizeNoOp) {
  const int B = 1, H = 1, nc = 1, S = 16, D = 8, cs = 16, blk = 8;
  std::vector<float> q((size_t)S * D, 0.5f), k(q), beta(S, 0.5f),
      L((size_t)nc * cs * D, -0.1f), M((size_t)nc * cs * cs), N(M.size());
  // null args (with work present) throw
  EXPECT_THROW(kda_k3_grams_blocked_cpu(nullptr, k.data(), beta.data(),
                                         L.data(), M.data(), N.data(), B, H,
                                         nc, S, D, cs, blk),
               std::invalid_argument);
  EXPECT_THROW(kda_k3_grams_blocked_cpu(q.data(), nullptr, beta.data(),
                                         L.data(), M.data(), N.data(), B, H,
                                         nc, S, D, cs, blk),
               std::invalid_argument);
  EXPECT_THROW(kda_k3_grams_blocked_cpu(q.data(), k.data(), nullptr,
                                         L.data(), M.data(), N.data(), B, H,
                                         nc, S, D, cs, blk),
               std::invalid_argument);
  EXPECT_THROW(kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(),
                                         nullptr, M.data(), N.data(), B, H,
                                         nc, S, D, cs, blk),
               std::invalid_argument);
  EXPECT_THROW(kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(),
                                         L.data(), nullptr, N.data(), B, H,
                                         nc, S, D, cs, blk),
               std::invalid_argument);
  EXPECT_THROW(kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(),
                                         L.data(), M.data(), nullptr, B, H,
                                         nc, S, D, cs, blk),
               std::invalid_argument);
  // blk must divide cs
  EXPECT_THROW(kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(),
                                         L.data(), M.data(), N.data(), B, H,
                                         nc, S, D, cs, 12),
               std::invalid_argument);
  // zero-size is a no-op (no throw, nothing written)
  std::vector<float> M0(4, -1.0f), N0(4, -1.0f);
  kda_k3_grams_blocked_cpu(q.data(), k.data(), beta.data(), L.data(),
                           M0.data(), N0.data(), 0, H, nc, S, D, cs, blk);
  EXPECT_EQ(M0[0], -1.0f);
  EXPECT_EQ(N0[3], -1.0f);
}

// --- null-arg contracts ------------------------------------------------------
TEST(KdaLayerNormGated, NullArgsThrow) {
  std::vector<float> x(4, 1), w(4, 1), g(4, 1), o(4, 0);
  EXPECT_THROW(kda_layer_norm_gated_cpu(nullptr, w.data(), g.data(), o.data(),
                                         1, 4, 1e-6f),
               std::invalid_argument);
  EXPECT_THROW(kda_layer_norm_gated_cpu(x.data(), nullptr, g.data(), o.data(),
                                         1, 4, 1e-6f),
               std::invalid_argument);
  EXPECT_THROW(kda_layer_norm_gated_cpu(x.data(), w.data(), nullptr, o.data(),
                                         1, 4, 1e-6f),
               std::invalid_argument);
  EXPECT_THROW(kda_layer_norm_gated_cpu(x.data(), w.data(), g.data(), nullptr,
                                         1, 4, 1e-6f),
               std::invalid_argument);
}

TEST(KdaDeltaRuleFwd, ChunkSizeMustDivideS) {
  std::vector<float> q(8, 1), k(8, 1), v(8, 1), g(8, 1), b(8, 1), o(8, 0);
  EXPECT_THROW(kda_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                                       b.data(), o.data(), 1, 1, 8, 1, 3),
               std::invalid_argument);  // 8 % 3 != 0
}
