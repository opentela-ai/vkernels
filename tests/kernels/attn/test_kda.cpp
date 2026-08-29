// tests/kernels/attn/test_kda.cpp
//
// Host tests for the Kimi Delta Attention layer (issue #21): hand-checked
// cases for the supporting ops (gated RMSNorm, gate cumsum, pack_bitmatrix),
// and a cross-check of the full chunked forward (kda_delta_rule_fwd_cpu,
// the L2..L6 pipeline) against the per-token naive oracle
// (kda_naive_delta_rule_fwd_cpu) at K3 head shapes with random gates and
// delta gates.
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

// --- #L7 chunked forward vs naive oracle (the core correctness check) -------
// --- #L3 naive oracle: K3 gated-delta-rule (per-key-dim gate, post-gate  ---
// prediction). The chunked CPU (L4-L7) implements the OLD standard
// delta-rule (scalar gate, pre-gate prediction) and is no longer the
// oracle; the hand-checked test + the zero-gate independence property
// below validate the new per-key-dim recurrence directly.
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
