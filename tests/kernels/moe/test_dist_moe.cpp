// tests/kernels/moe/test_dist_moe.cpp
//
// Tests for the distributed (TP / EP / PP) fused MXFP4 MoE layer
// (vkernels/dist/dist_moe.hpp).  Validates sharding arithmetic, the
// in-process multi-rank forwards against the single-rank CPU oracle, the EP
// dispatch/all-to-all re-layout, and the PP stage-boundary interface —
// including the TP8 x PP2 Kimi-K3 layout geometry from issue #18.
#include "minitest.hpp"

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "vkernels/comm/channel.hpp"
#include "vkernels/dist/dist_moe.hpp"
#include "vkernels/kernels/moe_fused.hpp"

using vkernels::dist::fused_moe_mxfp4_ep;
using vkernels::dist::fused_moe_mxfp4_tp;
using vkernels::dist::moe_ep_dispatch;
using vkernels::dist::moe_ep_plan;
using vkernels::dist::moe_tp_plan;
using vkernels::dist::moe_tp_shard_w13;
using vkernels::dist::moe_tp_shard_w2;
using vkernels::dist::pp_boundary_recv;
using vkernels::dist::pp_boundary_send;
using vkernels::dist::round_bf16;
using vkernels::kernels::fused_moe_mxfp4_cpu;

namespace {

// --------------------------------------------------------------------------
// bf16 / fp4 / ue8m0 helpers (mirror test_moe_fused.cpp)
// --------------------------------------------------------------------------
uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}

uint8_t e2m1_nibble(float f) {
  bool neg = f < 0;
  float af = std::fabs(f);
  if (af == 0.0f || std::isnan(af)) return neg ? 0x8 : 0x0;
  if (std::isinf(af)) return neg ? 0xE : 0x6;
  float vals[5] = {0.25f, 1.0f, 1.5f, 2.0f, 3.0f};
  uint8_t nibs[5] = {1, 2, 3, 4, 5};
  float best_d = std::fabs(af - vals[0]);
  uint8_t best_n = nibs[0];
  for (int i = 1; i < 5; ++i) {
    float d = std::fabs(af - vals[i]);
    if (d < best_d) { best_d = d; best_n = nibs[i]; }
  }
  return neg ? (best_n | 0x8) : best_n;
}

uint8_t pack_e2m1_pair(float v0, float v1) {
  return e2m1_nibble(v0) | (e2m1_nibble(v1) << 4);
}

// Deterministic pseudo-random in [-1, 1].
float rnd(int seed, int i) {
  unsigned x = static_cast<unsigned>(seed * 2654435761u + static_cast<unsigned>(i) * 40503u);
  x ^= x >> 13;
  x *= 0x5bd1e995u;
  x ^= x >> 15;
  return static_cast<float>(static_cast<int>(x % 200000)) / 100000.0f;
}

// --------------------------------------------------------------------------
// A complete single-rank MoE problem: full weights + routing + oracle result.
// --------------------------------------------------------------------------
struct Problem {
  int M, hidden, ispp, top_k, E;
  std::vector<uint16_t> A;
  std::vector<uint8_t> w13, w13s, w2, w2s;
  std::vector<float> b13, b2;
  std::vector<int32_t> topk_ids;
  std::vector<float> topk_w;
};

Problem make_problem(int M, int hidden, int ispp, int top_k, int E, int seed) {
  Problem p{};
  p.M = M; p.hidden = hidden; p.ispp = ispp; p.top_k = top_k; p.E = E;
  constexpr int group = 32;

  p.A.resize(static_cast<std::size_t>(M) * hidden);
  for (int i = 0; i < M * hidden; ++i) p.A[static_cast<std::size_t>(i)] = f2bf(rnd(seed, i) * 0.2f);

  p.w13.resize(static_cast<std::size_t>(E) * 2 * ispp * (hidden / 2));
  p.w13s.resize(static_cast<std::size_t>(E) * 2 * ispp * (hidden / group));
  p.w2.resize(static_cast<std::size_t>(E) * hidden * (ispp / 2));
  p.w2s.resize(static_cast<std::size_t>(E) * hidden * (ispp / group));
  for (std::size_t i = 0; i < p.w13.size() / 2; ++i) {
    p.w13[2 * i] = pack_e2m1_pair(rnd(seed + 1, static_cast<int>(i)) * 0.5f,
                                  rnd(seed + 2, static_cast<int>(i)) * 0.5f);
    p.w13[2 * i + 1] = pack_e2m1_pair(rnd(seed + 3, static_cast<int>(i)) * 0.5f,
                                      rnd(seed + 4, static_cast<int>(i)) * 0.5f);
  }
  for (std::size_t i = 0; i < p.w2.size() / 2; ++i) {
    p.w2[2 * i] = pack_e2m1_pair(rnd(seed + 5, static_cast<int>(i)) * 0.5f,
                                 rnd(seed + 6, static_cast<int>(i)) * 0.5f);
    p.w2[2 * i + 1] = pack_e2m1_pair(rnd(seed + 7, static_cast<int>(i)) * 0.5f,
                                     rnd(seed + 8, static_cast<int>(i)) * 0.5f);
  }
  for (std::size_t i = 0; i < p.w13s.size(); ++i)
    p.w13s[i] = static_cast<uint8_t>(125 + (static_cast<int>(i) % 5));
  for (std::size_t i = 0; i < p.w2s.size(); ++i)
    p.w2s[i] = static_cast<uint8_t>(125 + (static_cast<int>(i) % 5));

  p.b13.resize(static_cast<std::size_t>(E) * 2 * ispp);
  p.b2.resize(static_cast<std::size_t>(E) * hidden);
  for (std::size_t i = 0; i < p.b13.size(); ++i)
    p.b13[i] = rnd(seed + 9, static_cast<int>(i)) * 0.3f;
  for (std::size_t i = 0; i < p.b2.size(); ++i)
    p.b2[i] = rnd(seed + 10, static_cast<int>(i)) * 0.3f;

  p.topk_ids.resize(static_cast<std::size_t>(M) * top_k);
  p.topk_w.resize(static_cast<std::size_t>(M) * top_k);
  for (int i = 0; i < M; ++i) {
    for (int s = 0; s < top_k; ++s) {
      p.topk_ids[static_cast<std::size_t>(i * top_k + s)] = (i * 3 + s * 5 + seed) % E;
      p.topk_w[static_cast<std::size_t>(i * top_k + s)] =
          0.1f + rnd(seed + 11, i * top_k + s) * 0.4f;
    }
  }
  return p;
}

// Run the single-rank CPU oracle and return (out, act).
struct OracleResult {
  std::vector<uint16_t> act;
  std::vector<float> out;
};

OracleResult run_oracle(const Problem& p, int group, float limit, int activation,
                        float beta, float linear_beta, int block_size = 16) {
  const int M = p.M, hidden = p.hidden, ispp = p.ispp, top_k = p.top_k, E = p.E;
  std::vector<int32_t> sorted(static_cast<std::size_t>(M * top_k + E * block_size + 1));
  std::vector<int32_t> eids(
      static_cast<std::size_t>(M * top_k + E * block_size + 1) / block_size + 1);
  const int EM = vkernels::kernels::moe_align_block_size(
      p.topk_ids.data(), M, top_k, block_size, E, sorted.data(), eids.data());
  sorted.resize(static_cast<std::size_t>(EM));
  eids.resize(static_cast<std::size_t>(EM / block_size));

  std::vector<float> sorted_w(static_cast<std::size_t>(EM), 0.0f);
  for (int i = 0; i < EM; ++i) {
    int f = sorted[static_cast<std::size_t>(i)];
    if (f >= 0 && f < M * top_k)
      sorted_w[static_cast<std::size_t>(i)] = p.topk_w[static_cast<std::size_t>(f)];
  }

  OracleResult r;
  r.act.assign(static_cast<std::size_t>(EM) * ispp, 0);
  r.out.assign(static_cast<std::size_t>(M) * hidden, 0.0f);
  fused_moe_mxfp4_cpu(
      p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
      sorted.data(), sorted_w.data(), eids.data(), r.act.data(), r.out.data(),
      M, hidden, ispp, top_k, EM, group, limit, activation, beta, linear_beta,
      p.b13.data(), p.b2.data());
  return r;
}

// Max relative error of `got` vs `ref` (denominator floored at 1.0).
double max_rel(const std::vector<float>& got, const std::vector<float>& ref) {
  double m = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    double e = std::fabs(static_cast<double>(got[i]) - static_cast<double>(ref[i]));
    double d = std::fmax(std::fabs(static_cast<double>(ref[i])), 1.0);
    m = std::fmax(m, e / d);
  }
  return m;
}

}  // namespace

// ======================================================================
//  TP: plan geometry (including the Kimi-K3 TP8 layout)
// ======================================================================
TEST(DistMoe, TpPlanK3Layout) {
  // Kimi-K3: E = 112 experts, D_INTER (ispp) = 3072, hidden = 7168, TP8
  // (issue #18).  Per-rank shards must stay tile-able: hidden/8 = 896,
  // ispp/8 = 384 — both multiples of 64 (BLOCK_K) and 32 (group_size).
  const vkernels::dist::MoeTPPlan p = moe_tp_plan(7168, 3072, 8);
  EXPECT_EQ(p.tp, 8);
  EXPECT_EQ(p.hidden_shard, 896);
  EXPECT_EQ(p.ispp_shard, 384);
  EXPECT_EQ(p.w13_shard_bytes, 2 * 3072 * (896 / 2));   // 2*ispp * hidden_shard/2
  EXPECT_EQ(p.w13s_shard_bytes, 2 * 3072 * (896 / 32));
  EXPECT_EQ(p.w2_shard_bytes, 7168 * (384 / 2));
  EXPECT_EQ(p.w2s_shard_bytes, 7168 * (384 / 32));

  const vkernels::dist::MoeTPPlan p2 = moe_tp_plan(256, 128, 2);
  EXPECT_EQ(p2.hidden_shard, 128);
  EXPECT_EQ(p2.ispp_shard, 64);
  EXPECT_EQ(p2.w13_shard_bytes, 2 * 128 * (128 / 2));
}

TEST(DistMoe, TpPlanRejectsBadLayouts) {
  EXPECT_THROW(moe_tp_plan(128, 64, 0), std::invalid_argument);      // tp == 0
  EXPECT_THROW(moe_tp_plan(127, 64, 2), std::invalid_argument);      // hidden % tp
  EXPECT_THROW(moe_tp_plan(128, 63, 2), std::invalid_argument);      // ispp % tp
  EXPECT_THROW(moe_tp_plan(128, 64, 3), std::invalid_argument);      // hidden % tp
  EXPECT_THROW(moe_tp_plan(96, 96, 2), std::invalid_argument);       // hidden/2 = 48 % 64
  EXPECT_THROW(moe_tp_plan(128, 64, 2), std::invalid_argument);      // ispp/2 = 32 % 64
}

// ======================================================================
//  TP: sharding layout
// ======================================================================
TEST(DistMoe, TpShardsReconstructFullWeights) {
  const int E = 4, hidden = 256, ispp = 256;
  const vkernels::dist::MoeTPPlan plan = moe_tp_plan(hidden, ispp, 4);
  Problem p = make_problem(8, hidden, ispp, 1, E, 21);

  std::vector<uint8_t> w13_shard(static_cast<std::size_t>(plan.w13_shard_bytes) * E);
  std::vector<uint8_t> w13s_shard(static_cast<std::size_t>(plan.w13s_shard_bytes) * E);
  std::vector<uint8_t> w2_shard(static_cast<std::size_t>(plan.w2_shard_bytes) * E);
  std::vector<uint8_t> w2s_shard(static_cast<std::size_t>(plan.w2s_shard_bytes) * E);

  // Concatenating the four per-rank shards must reproduce the full weight
  // byte-for-byte (column-parallel along hidden / ispp).
  std::vector<uint8_t> w13_rebuilt(p.w13.size(), 0);
  std::vector<uint8_t> w13s_rebuilt(p.w13s.size(), 0);
  std::vector<uint8_t> w2_rebuilt(p.w2.size(), 0);
  std::vector<uint8_t> w2s_rebuilt(p.w2s.size(), 0);

  for (int r = 0; r < plan.tp; ++r) {
    moe_tp_shard_w13(p.w13.data(), p.w13s.data(), E, plan, r,
                     w13_shard.data(), w13s_shard.data());
    moe_tp_shard_w2(p.w2.data(), p.w2s.data(), E, plan, r,
                    w2_shard.data(), w2s_shard.data());

    // w13: rows = E * 2*ispp, each row split into 4 contiguous slices.
    for (int row = 0; row < E * 2 * ispp; ++row) {
      std::memcpy(&w13_rebuilt[static_cast<std::size_t>(row) * (hidden / 2) +
                               static_cast<std::size_t>(r) * (plan.hidden_shard / 2)],
                  &w13_shard[static_cast<std::size_t>(row) * (plan.hidden_shard / 2)],
                  plan.hidden_shard / 2);
      std::memcpy(&w13s_rebuilt[static_cast<std::size_t>(row) * (hidden / 32) +
                                static_cast<std::size_t>(r) * (plan.hidden_shard / 32)],
                  &w13s_shard[static_cast<std::size_t>(row) * (plan.hidden_shard / 32)],
                  plan.hidden_shard / 32);
    }
    // w2: rows = E * hidden, each row split into 4 contiguous slices.
    for (int row = 0; row < E * hidden; ++row) {
      std::memcpy(&w2_rebuilt[static_cast<std::size_t>(row) * (ispp / 2) +
                              static_cast<std::size_t>(r) * (plan.ispp_shard / 2)],
                  &w2_shard[static_cast<std::size_t>(row) * (plan.ispp_shard / 2)],
                  plan.ispp_shard / 2);
      std::memcpy(&w2s_rebuilt[static_cast<std::size_t>(row) * (ispp / 32) +
                               static_cast<std::size_t>(r) * (plan.ispp_shard / 32)],
                  &w2s_shard[static_cast<std::size_t>(row) * (plan.ispp_shard / 32)],
                  plan.ispp_shard / 32);
    }
  }
  EXPECT_TRUE(w13_rebuilt == p.w13);
  EXPECT_TRUE(w13s_rebuilt == p.w13s);
  EXPECT_TRUE(w2_rebuilt == p.w2);
  EXPECT_TRUE(w2s_rebuilt == p.w2s);
}

TEST(DistMoe, TpShardRejectsBadRank) {
  const vkernels::dist::MoeTPPlan plan = moe_tp_plan(256, 128, 2);
  uint8_t b[64] = {0};
  EXPECT_THROW(moe_tp_shard_w13(b, b, 1, plan, 2, b, b), std::invalid_argument);
  EXPECT_THROW(moe_tp_shard_w2(b, b, 1, plan, -1, b, b), std::invalid_argument);
}

// ======================================================================
//  TP: distributed forward matches the CPU oracle
// ======================================================================
TEST(DistMoe, TpSingleRankMatchesOracleBitExact) {
  // tp == 1: the all-reduce is a no-op and the K-loop order is identical to
  // the oracle, so out and act must match bit-for-bit.
  constexpr int group = 32;
  Problem p = make_problem(16, 128, 64, 2, 8, 31);
  OracleResult ref = run_oracle(p, group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f);

  const auto outs = fused_moe_mxfp4_tp(
      p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
      p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
      p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
      vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 1);
  ASSERT_EQ(outs.size(), std::size_t(1));
  EXPECT_TRUE(outs[0] == ref.out);  // bit-exact
}

TEST(DistMoe, TpForwardMatchesOracleSwiGLU) {
  // hidden/ispp must keep per-rank shards at multiples of 64: hidden = 256,
  // ispp = 256 stays valid down to tp = 4 (shards 64 x 64).
  constexpr int group = 32;
  Problem p = make_problem(16, 256, 256, 2, 8, 32);
  OracleResult ref = run_oracle(p, group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f);

  for (int tp : {2, 4}) {
    const auto outs = fused_moe_mxfp4_tp(
        p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
        p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
        p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
        vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, tp);
    ASSERT_EQ(outs.size(), static_cast<std::size_t>(tp));
    // Every rank converges to the same all-reduced result ...
    for (int r = 1; r < tp; ++r) EXPECT_TRUE(outs[static_cast<std::size_t>(r)] == outs[0]);
    // ... which matches the single-rank oracle (fp32 all-reduce re-associates
    // the per-rank partial sums, so a small relative tolerance is needed).
    EXPECT_LT(max_rel(outs[0], ref.out), 1e-3);
  }
}

TEST(DistMoe, TpForwardMatchesOracleSiTU) {
  constexpr int group = 32;
  Problem p = make_problem(16, 256, 256, 2, 8, 33);
  // SiTU (Kimi-K3): beta = 4, linear_beta = 25, no clamp.
  OracleResult ref = run_oracle(p, group, 0.0f, vkernels::kernels::kSiTU, 4.0f, 25.0f);

  const auto outs = fused_moe_mxfp4_tp(
      p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
      p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
      p.M, p.hidden, p.ispp, p.top_k, p.E, group, 0.0f,
      vkernels::kernels::kSiTU, 4.0f, 25.0f, 16, 4);
  ASSERT_EQ(outs.size(), std::size_t(4));
  EXPECT_LT(max_rel(outs[0], ref.out), 1e-3);
}

TEST(DistMoe, TpNoBiasMatchesOracle) {
  constexpr int group = 32;
  Problem p = make_problem(16, 256, 128, 2, 8, 34);
  // nullptr biases exercise the skip branches of the epilogues.
  OracleResult ref = run_oracle(p, group, 2.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f);
  std::vector<float> empty;  // not used
  (void)empty;

  std::vector<int32_t> sorted(static_cast<std::size_t>(p.M * p.top_k + p.E * 16 + 1));
  std::vector<int32_t> eids(static_cast<std::size_t>(p.M * p.top_k + p.E * 16 + 1) / 16 + 1);
  const int EM = vkernels::kernels::moe_align_block_size(
      p.topk_ids.data(), p.M, p.top_k, 16, p.E, sorted.data(), eids.data());
  sorted.resize(static_cast<std::size_t>(EM));
  eids.resize(static_cast<std::size_t>(EM / 16));
  std::vector<float> sorted_w(static_cast<std::size_t>(EM), 0.0f);
  for (int i = 0; i < EM; ++i) {
    int f = sorted[static_cast<std::size_t>(i)];
    if (f >= 0 && f < p.M * p.top_k)
      sorted_w[static_cast<std::size_t>(i)] = p.topk_w[static_cast<std::size_t>(f)];
  }
  std::vector<uint16_t> act(static_cast<std::size_t>(EM) * p.ispp, 0);
  std::vector<float> out(p.M * p.hidden, 0.0f);
  fused_moe_mxfp4_cpu(
      p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
      sorted.data(), sorted_w.data(), eids.data(), act.data(), out.data(),
      p.M, p.hidden, p.ispp, p.top_k, EM, group, 2.0f,
      vkernels::kernels::kSwiGLU, 4.0f, 25.0f, nullptr, nullptr);

  const auto outs = fused_moe_mxfp4_tp(
      p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
      p.topk_ids.data(), p.topk_w.data(), nullptr, nullptr,
      p.M, p.hidden, p.ispp, p.top_k, p.E, group, 2.0f,
      vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 2);
  EXPECT_LT(max_rel(outs[0], out), 1e-3);
}

TEST(DistMoe, TpForwardRejectsBadArgs) {
  Problem p = make_problem(16, 256, 256, 2, 8, 35);
  // tp == 0
  EXPECT_THROW(fused_moe_mxfp4_tp(
                   p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
                   p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
                   p.M, p.hidden, p.ispp, p.top_k, p.E, 32, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 0),
               std::invalid_argument);
  // hidden not divisible by tp
  EXPECT_THROW(fused_moe_mxfp4_tp(
                   p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
                   p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
                   p.M, 127, p.ispp, p.top_k, p.E, 32, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 4),
               std::invalid_argument);
  // prefill (64-row) alignment is HIP-only; the CPU path rejects it before
  // any worker thread is spawned
  EXPECT_THROW(fused_moe_mxfp4_tp(
                   p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
                   p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
                   p.M, p.hidden, p.ispp, p.top_k, p.E, 32, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 64, 4),
               std::invalid_argument);
}

// Direct exercise of the per-rank channel function (tp == 1: no-op all-reduce)
// including its argument validation.
TEST(DistMoe, TpRankChannelsAndValidation) {
  constexpr int group = 32;
  Problem p = make_problem(16, 128, 64, 2, 8, 36);
  OracleResult ref = run_oracle(p, group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f);

  const vkernels::dist::MoeTPPlan plan = moe_tp_plan(p.hidden, p.ispp, 1);
  std::vector<uint8_t> w13s(static_cast<std::size_t>(plan.w13_shard_bytes) * p.E);
  std::vector<uint8_t> w13ss(static_cast<std::size_t>(plan.w13s_shard_bytes) * p.E);
  std::vector<uint8_t> w2s(static_cast<std::size_t>(plan.w2_shard_bytes) * p.E);
  std::vector<uint8_t> w2ss(static_cast<std::size_t>(plan.w2s_shard_bytes) * p.E);
  moe_tp_shard_w13(p.w13.data(), p.w13s.data(), p.E, plan, 0, w13s.data(), w13ss.data());
  moe_tp_shard_w2(p.w2.data(), p.w2s.data(), p.E, plan, 0, w2s.data(), w2ss.data());

  std::vector<int32_t> sorted(static_cast<std::size_t>(p.M * p.top_k + p.E * 16 + 1));
  std::vector<int32_t> eids(static_cast<std::size_t>(p.M * p.top_k + p.E * 16 + 1) / 16 + 1);
  const int EM = vkernels::kernels::moe_align_block_size(
      p.topk_ids.data(), p.M, p.top_k, 16, p.E, sorted.data(), eids.data());

  std::vector<uint16_t> act(static_cast<std::size_t>(EM) * p.ispp, 0);
  std::vector<float> out(p.M * p.hidden, 0.0f);
  auto ch = vkernels::comm::make_ring_channels(1);

  vkernels::dist::fused_moe_mxfp4_tp_rank(
      p.A.data(), w13s.data(), w13ss.data(), w2s.data(), w2ss.data(),
      p.topk_w.data(), p.b13.data(), p.b2.data(), act.data(), out.data(),
      p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
      vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 1, 0, EM,
      sorted.data(), eids.data(), *ch[0], *ch[0]);
  EXPECT_TRUE(out == ref.out);  // bit-exact at tp == 1

  // Argument validation (throws before any channel use).
  EXPECT_THROW(vkernels::dist::fused_moe_mxfp4_tp_rank(
                   p.A.data(), w13s.data(), w13ss.data(), w2s.data(), w2ss.data(),
                   p.topk_w.data(), p.b13.data(), p.b2.data(), act.data(), out.data(),
                   p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 1, 1, EM,
                   sorted.data(), eids.data(), *ch[0], *ch[0]),
               std::invalid_argument);  // rank out of range
  EXPECT_THROW(vkernels::dist::fused_moe_mxfp4_tp_rank(
                   p.A.data(), w13s.data(), w13ss.data(), w2s.data(), w2ss.data(),
                   p.topk_w.data(), p.b13.data(), p.b2.data(), act.data(), out.data(),
                   p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 1, 0, EM + 8,
                   sorted.data(), eids.data(), *ch[0], *ch[0]),
               std::invalid_argument);  // EM not block-aligned
  EXPECT_THROW(vkernels::dist::fused_moe_mxfp4_tp_rank(
                   p.A.data(), w13s.data(), w13ss.data(), w2s.data(), w2ss.data(),
                   p.topk_w.data(), p.b13.data(), p.b2.data(), act.data(), out.data(),
                   p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 64, 1, 0, EM,
                   sorted.data(), eids.data(), *ch[0], *ch[0]),
               std::invalid_argument);  // block_size must be 16 (decode)
}

// ======================================================================
//  EP: plan + dispatch + forward
// ======================================================================
TEST(DistMoe, EpPlanArithmetic) {
  // E=10, ep=3: blocks of 3,3,4 experts.
  auto r0 = moe_ep_plan(10, 3, 0);
  EXPECT_EQ(r0.expert_begin, 0);
  EXPECT_EQ(r0.expert_end, 4);
  EXPECT_EQ(r0.num_local, 4);
  auto r1 = moe_ep_plan(10, 3, 1);
  EXPECT_EQ(r1.expert_begin, 4);
  EXPECT_EQ(r1.expert_end, 7);
  auto r2 = moe_ep_plan(10, 3, 2);
  EXPECT_EQ(r2.expert_begin, 7);
  EXPECT_EQ(r2.expert_end, 10);
  // Even split: E=8, ep=4 → 2 experts each.
  auto r3 = moe_ep_plan(8, 4, 3);
  EXPECT_EQ(r3.expert_begin, 6);
  EXPECT_EQ(r3.expert_end, 8);

  EXPECT_THROW(moe_ep_plan(8, 0, 0), std::invalid_argument);   // ep == 0
  EXPECT_THROW(moe_ep_plan(8, 4, 4), std::invalid_argument);   // rank out of range
  EXPECT_THROW(moe_ep_plan(2, 4, 0), std::invalid_argument);   // num_experts < ep
}

TEST(DistMoe, EpDispatchLayout) {
  const int E = 8, M = 16, top_k = 2, block = 16;
  std::vector<int32_t> topk_ids(static_cast<std::size_t>(M) * top_k);
  for (int i = 0; i < M * top_k; ++i) topk_ids[static_cast<std::size_t>(i)] = i % E;

  // Rank 0 owns experts [0, 4).
  auto plan = moe_ep_plan(E, 2, 0);
  std::vector<int32_t> sorted, eids;
  const int EM = moe_ep_dispatch(topk_ids.data(), M, top_k, E, block, plan,
                                 sorted, eids);
  EXPECT_EQ(EM % block, 0);
  // Every dispatched (token, sel) maps to a locally-owned expert; every
  // block's expert id is local (< 4) or -1 (padding).
  for (int i = 0; i < EM; ++i) {
    int f = sorted[static_cast<std::size_t>(i)];
    if (f >= 0 && f < M * top_k) {
      int e = topk_ids[static_cast<std::size_t>(f)];
      EXPECT_TRUE(e >= 0 && e < 4);
    }
  }
  for (int b = 0; b < EM / block; ++b) {
    int e = eids[static_cast<std::size_t>(b)];
    if (e >= 0) EXPECT_TRUE(e < 4);  // local id
  }
  // All tokens routed to experts 4..7 must be absent from rank 0's layout.
  for (int i = 0; i < M; ++i) {
    if (topk_ids[static_cast<std::size_t>(i * top_k + 1)] >= 4) {
      // (i, 1) belongs to rank 1; its flat index must not appear here unless
      // (i, 0) also routed locally with the same flat? flat ids differ, so
      // check the specific flat id is absent.
      int absent = i * top_k + 1;
      for (int k = 0; k < EM; ++k) {
        EXPECT_TRUE(sorted[static_cast<std::size_t>(k)] != absent);
      }
    }
  }

  // block_size == 0 is rejected.
  std::vector<int32_t> s2, e2;
  EXPECT_THROW(moe_ep_dispatch(topk_ids.data(), M, top_k, E, 0, plan, s2, e2),
               std::invalid_argument);
}

TEST(DistMoe, EpForwardMatchesOracle) {
  constexpr int group = 32;
  Problem p = make_problem(16, 128, 64, 2, 8, 41);
  OracleResult ref = run_oracle(p, group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f);

  for (int ep : {1, 2, 4}) {
    const auto outs = fused_moe_mxfp4_ep(
        p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
        p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
        p.M, p.hidden, p.ispp, p.top_k, p.E, group, 4.0f,
        vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, ep);
    ASSERT_EQ(outs.size(), static_cast<std::size_t>(ep));
    // Each rank's out is the weighted sum over the (token, sel) pairs routed
    // to it; the MoE output is the sum across ranks (all-to-all back).
    std::vector<float> combined(p.M * p.hidden, 0.0f);
    for (int r = 0; r < ep; ++r) {
      for (std::size_t i = 0; i < combined.size(); ++i)
        combined[i] += outs[static_cast<std::size_t>(r)][i];
    }
    EXPECT_LT(max_rel(combined, ref.out), 1e-3);
  }
}

TEST(DistMoe, EpForwardSiTU) {
  constexpr int group = 32;
  Problem p = make_problem(16, 128, 64, 2, 9, 42);  // odd E exercises the tail
  OracleResult ref = run_oracle(p, group, 0.0f, vkernels::kernels::kSiTU, 4.0f, 25.0f);

  const auto outs = fused_moe_mxfp4_ep(
      p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
      p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
      p.M, p.hidden, p.ispp, p.top_k, p.E, group, 0.0f,
      vkernels::kernels::kSiTU, 4.0f, 25.0f, 16, 3);
  std::vector<float> combined(p.M * p.hidden, 0.0f);
  for (int r = 0; r < 3; ++r) {
    for (std::size_t i = 0; i < combined.size(); ++i)
      combined[i] += outs[static_cast<std::size_t>(r)][i];
  }
  EXPECT_LT(max_rel(combined, ref.out), 1e-3);
}

TEST(DistMoe, EpForwardRejectsBadEp) {
  Problem p = make_problem(16, 128, 64, 2, 8, 43);
  EXPECT_THROW(fused_moe_mxfp4_ep(
                   p.A.data(), p.w13.data(), p.w13s.data(), p.w2.data(), p.w2s.data(),
                   p.topk_ids.data(), p.topk_w.data(), p.b13.data(), p.b2.data(),
                   p.M, p.hidden, p.ispp, p.top_k, p.E, 32, 4.0f,
                   vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, 0),
               std::invalid_argument);
}

// ======================================================================
//  PP: stage-boundary interface + TP x PP pipeline
// ======================================================================
TEST(DistMoe, RoundBf16KnownValues) {
  std::vector<uint16_t> dst(4);
  const float src[4] = {1.0f, 0.5f, -1.0f, 1.0001f};
  round_bf16(src, dst.data(), 4);
  EXPECT_EQ(dst[0], uint16_t(0x3F80));  // 1.0
  EXPECT_EQ(dst[1], uint16_t(0x3F00));  // 0.5
  EXPECT_EQ(dst[2], uint16_t(0xBF80));  // -1.0
  EXPECT_EQ(dst[3], uint16_t(0x3F80));  // 1.0001 rounds to 1.0 (bf16 RNE)
}

TEST(DistMoe, PpBoundaryRoundTrip) {
  auto ch = vkernels::comm::make_ring_channels(1);
  std::vector<float> sent = {1.0f, 2.0f, 3.0f, 4.0f};
  pp_boundary_send(sent.data(), 2, 2, *ch[0]);
  std::vector<float> got(4, -1.0f);
  pp_boundary_recv(got.data(), 2, 2, *ch[0]);
  EXPECT_TRUE(got == sent);

  // Size mismatch on recv is rejected.
  pp_boundary_send(sent.data(), 2, 2, *ch[0]);
  EXPECT_THROW(pp_boundary_recv(got.data(), 3, 2, *ch[0]), std::invalid_argument);
}

// TP8 x PP2-style pipeline (issue #18 acceptance): two PP stages, each a TP
// MoE with its own weights, hidden state passed through the boundary.
TEST(DistMoe, PpTimesTpPipelineMatchesOracle) {
  constexpr int group = 32;
  Problem layer_a = make_problem(16, 256, 128, 2, 8, 51);  // stage 0 weights
  Problem layer_b = make_problem(16, 256, 128, 2, 8, 52);  // stage 1 weights

  // Single-rank oracle for the whole pipeline: layer A then layer B, with
  // the fp32 output re-quantised to bf16 at the stage boundary.
  OracleResult stage0 = run_oracle(layer_a, group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f);
  std::vector<uint16_t> a1(static_cast<std::size_t>(layer_a.M) * layer_a.hidden);
  round_bf16(stage0.out.data(), a1.data(), layer_a.M * layer_a.hidden);

  std::vector<int32_t> sorted(
      static_cast<std::size_t>(layer_b.M * layer_b.top_k + layer_b.E * 16 + 1));
  std::vector<int32_t> eids(
      static_cast<std::size_t>(layer_b.M * layer_b.top_k + layer_b.E * 16 + 1) / 16 + 1);
  const int EM = vkernels::kernels::moe_align_block_size(
      layer_b.topk_ids.data(), layer_b.M, layer_b.top_k, 16, layer_b.E,
      sorted.data(), eids.data());
  std::vector<float> sorted_w(static_cast<std::size_t>(EM), 0.0f);
  for (int i = 0; i < EM; ++i) {
    int f = sorted[static_cast<std::size_t>(i)];
    if (f >= 0 && f < layer_b.M * layer_b.top_k)
      sorted_w[static_cast<std::size_t>(i)] = layer_b.topk_w[static_cast<std::size_t>(f)];
  }
  std::vector<uint16_t> act2(static_cast<std::size_t>(EM) * layer_b.ispp, 0);
  std::vector<float> expected(layer_b.M * layer_b.hidden, 0.0f);
  fused_moe_mxfp4_cpu(
      a1.data(), layer_b.w13.data(), layer_b.w13s.data(),
      layer_b.w2.data(), layer_b.w2s.data(), sorted.data(), sorted_w.data(),
      eids.data(), act2.data(), expected.data(),
      layer_b.M, layer_b.hidden, layer_b.ispp, layer_b.top_k, EM, group, 4.0f,
      vkernels::kernels::kSwiGLU, 4.0f, 25.0f,
      layer_b.b13.data(), layer_b.b2.data());

  // Distributed pipeline: stage 0 with TP2, boundary transfer, stage 1 with
  // TP2.
  const int tp = 2;
  auto outs0 = fused_moe_mxfp4_tp(
      layer_a.A.data(), layer_a.w13.data(), layer_a.w13s.data(),
      layer_a.w2.data(), layer_a.w2s.data(),
      layer_a.topk_ids.data(), layer_a.topk_w.data(),
      layer_a.b13.data(), layer_a.b2.data(),
      layer_a.M, layer_a.hidden, layer_a.ispp, layer_a.top_k, layer_a.E,
      group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, tp);
  ASSERT_EQ(outs0.size(), static_cast<std::size_t>(tp));

  auto ch = vkernels::comm::make_ring_channels(1);
  pp_boundary_send(outs0[0].data(), layer_a.M, layer_a.hidden, *ch[0]);
  std::vector<float> boundary(static_cast<std::size_t>(layer_a.M) * layer_a.hidden);
  pp_boundary_recv(boundary.data(), layer_a.M, layer_a.hidden, *ch[0]);
  std::vector<uint16_t> a1_dist(static_cast<std::size_t>(layer_a.M) * layer_a.hidden);
  round_bf16(boundary.data(), a1_dist.data(), layer_a.M * layer_a.hidden);

  auto outs1 = fused_moe_mxfp4_tp(
      a1_dist.data(), layer_b.w13.data(), layer_b.w13s.data(),
      layer_b.w2.data(), layer_b.w2s.data(),
      layer_b.topk_ids.data(), layer_b.topk_w.data(),
      layer_b.b13.data(), layer_b.b2.data(),
      layer_b.M, layer_b.hidden, layer_b.ispp, layer_b.top_k, layer_b.E,
      group, 4.0f, vkernels::kernels::kSwiGLU, 4.0f, 25.0f, 16, tp);
  ASSERT_EQ(outs1.size(), static_cast<std::size_t>(tp));
  EXPECT_LT(max_rel(outs1[0], expected), 1e-2);
}
