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
TEST(KdaDeltaRuleFwd, ChunkedMatchesNaive) {
  struct Cfg { int B, H, S, D, chunk; };
  const Cfg cfgs[] = {
      {1, 1, 4, 2, 2},     // tiniest: two chunks
      {1, 1, 8, 4, 4},     // two chunks, D=4
      {1, 2, 8, 3, 4},     // D=3 (odd), H=2
      {2, 1, 12, 4, 4},    // three chunks, B=2
      {1, 1, 16, 4, 8},    // two larger chunks
      {1, 1, 64, 8, 16},   // K3-ish: D=8, 4 chunks
  };
  std::mt19937 rng(123);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  auto rg = [&]() { return 0.3f + 0.7f * (rng() % 1000) / 1000.0f; };

  for (const auto& c : cfgs) {
    std::vector<float> q((size_t)c.B * c.H * c.S * c.D);
    std::vector<float> k(q.size()), v(q.size());
    std::vector<float> g((size_t)c.B * c.H * c.S);
    std::vector<float> beta((size_t)c.B * c.H * c.S);
    for (auto& x : q) x = rf();
    for (auto& x : k) x = rf();
    for (auto& x : v) x = rf();
    for (auto& x : g) x = rg();
    for (auto& x : beta) x = rg();  // delta gate in (0.3, 1.0]

    std::vector<float> naive(q.size(), 0), chunked(q.size(), 0);
    kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                                 beta.data(), naive.data(),
                                 c.B, c.H, c.S, c.D);
    kda_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                           beta.data(), chunked.data(),
                           c.B, c.H, c.S, c.D, c.chunk);

    // The chunked recurrence accumulates the same products as the naive one,
    // so agreement is within fp32 round-off scaled by the sequence length.
    float maxd = 0.0f, maxabs = 0.0f;
    for (size_t i = 0; i < naive.size(); ++i) {
      maxd = std::max(maxd, std::fabs(naive[i] - chunked[i]));
      maxabs = std::max(maxabs, std::fabs(naive[i]));
    }
    EXPECT_NEAR(maxd, 0.0f, 1e-3f * (1.0f + maxabs));
  }
}

TEST(KdaDeltaRuleFwd, ZeroForgettingMatchesNaive) {
  // g==1 everywhere (no forgetting) stresses the inter recurrence: the state
  // accumulates across all chunks, so the chunked combine must keep the full
  // history (no decay) and still match the oracle.
  constexpr int B = 1, H = 1, S = 12, D = 4, chunk = 4;
  std::mt19937 rng(7);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  std::vector<float> q((size_t)S * D), k(q.size()), v(q.size());
  std::vector<float> g(S, 1.0f), beta(S, 0.5f);
  for (auto& x : q) x = rf();
  for (auto& x : k) x = rf();
  for (auto& x : v) x = rf();
  std::vector<float> naive(q.size(), 0), chunked(q.size(), 0);
  kda_naive_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                               beta.data(), naive.data(), B, H, S, D);
  kda_delta_rule_fwd_cpu(q.data(), k.data(), v.data(), g.data(),
                         beta.data(), chunked.data(), B, H, S, D, chunk);
  float maxd = 0, maxabs = 0;
  for (size_t i = 0; i < naive.size(); ++i) {
    maxd = std::max(maxd, std::fabs(naive[i] - chunked[i]));
    maxabs = std::max(maxabs, std::fabs(naive[i]));
  }
  EXPECT_NEAR(maxd, 0.0f, 1e-3f * (1.0f + maxabs));
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
