// tests/kernels/moe/test_moe_fused.cpp
//
// Tests for fused MXFP4 MoE grouped GEMM — validates CPU reference against
// analytically-known results for small synthetic problems.
#include "minitest.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/moe.hpp"
#include "vkernels/util/config.hpp"

using vkernels::kernels::fused_moe_mxfp4_cpu;
using vkernels::kernels::moe_align_block_size;

namespace {

// bf16 → float
float bf2f(uint16_t v) {
  uint32_t b = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &b, sizeof(float));
  return f;
}

// float → bf16
uint16_t f2bf(float v) {
  uint32_t b;
  std::memcpy(&b, &v, sizeof(float));
  uint32_t lsb = (b >> 16) & 1;
  b += 0x7FFFu + lsb;
  return static_cast<uint16_t>(b >> 16);
}

// Nearest E2M1 nibble for a float value
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

// Pack packed byte from two float values (low nibble = v0, high = v1)
uint8_t pack_e2m1_pair(float v0, float v1) {
  return e2m1_nibble(v0) | (e2m1_nibble(v1) << 4);
}

}  // namespace

// ======================================================================
//  Tiny sanity: M=16, hidden=128, ispp=64, top_k=1, E=1
// ======================================================================
// All weights = 1.0 → gate = up = 128, act = silu(128)*128 ≈ 16384,
// out = act * 64 ≈ 1,048,576

TEST(FusedMoe, TinySanity) {
  constexpr int M = 16, hidden = 128, ispp = 64;
  constexpr int group_size = 32, BLOCK_M = 16;
  constexpr int EM = M;  // top_k=1, M fits BLOCK_M

  // A [M, hidden] = all 1.0 bf16
  std::vector<uint16_t> A_bf16(M * hidden);
  for (auto& v : A_bf16) v = f2bf(1.0f);

  // w13 [1, 2*ispp, hidden/2] = all 1.0 in E2M1
  uint8_t one_byte = pack_e2m1_pair(1.0f, 1.0f);
  std::vector<uint8_t> w13(2 * ispp * hidden / 2, one_byte);
  std::vector<uint8_t> w13_scale(2 * ispp * hidden / group_size, 127);

  // w2 [1, hidden, ispp/2] = all 1.0
  std::vector<uint8_t> w2(hidden * ispp / 2, one_byte);
  std::vector<uint8_t> w2_scale(hidden * ispp / group_size, 127);

  std::vector<int32_t> sorted_ids(EM);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
  std::vector<float> topk_ws(EM, 1.0f);
  std::vector<int32_t> expert_ids(EM / BLOCK_M, 0);

  std::vector<uint16_t> act_h(EM * ispp, 0);
  std::vector<float> out_h(M * hidden, 0.0f);

  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size,
      0.0f, 0, 4.0f, 25.0f, nullptr, nullptr);

  float silu128 = 128.0f / (1.0f + std::exp(-128.0f));
  float expected_act = silu128 * 128.0f;
  float expected_out = expected_act * static_cast<float>(ispp);

  float max_rel = 0.0f;
  for (int i = 0; i < M * hidden; ++i) {
    float err = std::fabs(out_h[i] - expected_out);
    if (expected_out != 0) max_rel = std::max(max_rel, err / expected_out);
  }
  EXPECT_LT(max_rel, 1e-2f);

  for (int i = 0; i < EM * ispp; ++i) {
    float err = std::fabs(bf2f(act_h[i]) - expected_act);
    EXPECT_LT(err, 0.5f);  // bf16 precision
  }
}

// ======================================================================
//  Bias-only test
// ======================================================================
TEST(FusedMoe, BiasSanity) {
  constexpr int M = 16, hidden = 128, ispp = 64;
  constexpr int group_size = 32, BLOCK_M = 16;
  constexpr int EM = M;

  // Zero weights → only bias contributes
  std::vector<uint16_t> A_bf16(M * hidden, f2bf(0.0f));
  std::vector<uint8_t> w13(2 * ispp * hidden / 2, 0);
  std::vector<uint8_t> w13_scale(2 * ispp * hidden / group_size, 127);
  std::vector<uint8_t> w2(hidden * ispp / 2, 0);
  std::vector<uint8_t> w2_scale(hidden * ispp / group_size, 127);

  std::vector<float> b13(2 * ispp);
  for (int i = 0; i < ispp; ++i) { b13[i] = 1.0f; b13[i + ispp] = 2.0f; }
  std::vector<float> b2(hidden, 1.0f);

  std::vector<int32_t> sorted_ids(EM);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
  std::vector<float> topk_ws(EM, 1.0f);
  std::vector<int32_t> expert_ids(EM / BLOCK_M, 0);

  std::vector<uint16_t> act_h(EM * ispp, 0);
  std::vector<float> out_h(M * hidden, 0.0f);

  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size,
      0.0f, 0, 4.0f, 25.0f, b13.data(), b2.data());

  float silu1 = 1.0f / (1.0f + std::exp(-1.0f));
  float expected_act = silu1 * 2.0f;

  for (int i = 0; i < EM * ispp; ++i) {
    EXPECT_NEAR(bf2f(act_h[i]), expected_act, 1e-2f);
  }
  for (int i = 0; i < M * hidden; ++i) {
    EXPECT_NEAR(out_h[i], 1.0f, 1e-4f);
  }
}

// ======================================================================
//  SwiGLU clamp test
// ======================================================================
TEST(FusedMoe, SwiGLUClamp) {
  constexpr int M = 16, hidden = 128, ispp = 64;
  constexpr int group_size = 32, BLOCK_M = 16;
  constexpr int EM = M;

  // gate=up=128 unclamped → limit=10 → gate=up=10 → act=silu(10)*10
  std::vector<uint16_t> A_bf16(M * hidden, f2bf(1.0f));
  uint8_t one_byte = pack_e2m1_pair(1.0f, 1.0f);
  std::vector<uint8_t> w13(2 * ispp * hidden / 2, one_byte);
  std::vector<uint8_t> w13_scale(2 * ispp * hidden / group_size, 127);
  std::vector<uint8_t> w2(hidden * ispp / 2, one_byte);
  std::vector<uint8_t> w2_scale(hidden * ispp / group_size, 127);

  std::vector<int32_t> sorted_ids(EM);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
  std::vector<float> topk_ws(EM, 1.0f);
  std::vector<int32_t> expert_ids(EM / BLOCK_M, 0);

  std::vector<uint16_t> act_h(EM * ispp, 0);
  std::vector<float> out_h(M * hidden, 0.0f);

  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size,
      10.0f, 0, 4.0f, 25.0f, nullptr, nullptr);

  float silu10 = 10.0f / (1.0f + std::exp(-10.0f));
  float expected_act = silu10 * 10.0f;
  float expected_out = expected_act * static_cast<float>(ispp);

  for (int i = 0; i < EM * ispp; ++i) {
    EXPECT_NEAR(bf2f(act_h[i]), expected_act, 0.2f);
  }
  for (int i = 0; i < M * hidden; ++i) {
    float err = std::fabs(out_h[i] - expected_out);
    EXPECT_LT(err / expected_out, 1e-2f);
  }
}

// ======================================================================
//  SiTU activation test (Kimi-K3 situ_and_mul)
// ======================================================================
// gate = up = 128 (A=1, w=1, hidden=128). SiTU must NOT apply the
// swiglu_limit clamp (passing limit=1.0 below is a regression guard):
//   gate_out = beta * tanh(gate/beta) * sigmoid(gate)
//   up_out   = linear_beta * tanh(up/linear_beta)
//   act      = gate_out * up_out ≈ 4.0 * 25.0 = 100
TEST(FusedMoe, SiTU) {
  constexpr int M = 16, hidden = 128, ispp = 64;
  constexpr int group_size = 32, BLOCK_M = 16;
  constexpr int EM = M;

  std::vector<uint16_t> A_bf16(M * hidden, f2bf(1.0f));
  uint8_t one_byte = pack_e2m1_pair(1.0f, 1.0f);
  std::vector<uint8_t> w13(2 * ispp * hidden / 2, one_byte);
  std::vector<uint8_t> w13_scale(2 * ispp * hidden / group_size, 127);
  std::vector<uint8_t> w2(hidden * ispp / 2, one_byte);
  std::vector<uint8_t> w2_scale(hidden * ispp / group_size, 127);

  std::vector<int32_t> sorted_ids(EM);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
  std::vector<float> topk_ws(EM, 1.0f);
  std::vector<int32_t> expert_ids(EM / BLOCK_M, 0);

  std::vector<uint16_t> act_h(EM * ispp, 0);
  std::vector<float> out_h(M * hidden, 0.0f);

  constexpr float beta = 4.0f, linear_beta = 25.0f;
  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size,
      1.0f /* clamp must be ignored */, 1 /* kSiTU */, beta, linear_beta,
      nullptr, nullptr);

  // situ_and_mul reference (vLLM): gate = up = 128, unclamped.
  float gate = 128.0f, up = 128.0f;
  float sig = 1.0f / (1.0f + std::exp(-gate));
  float expected_act = (beta * std::tanh(gate / beta) * sig)
                     * (linear_beta * std::tanh(up / linear_beta));
  float expected_out = expected_act * static_cast<float>(ispp);

  for (int i = 0; i < EM * ispp; ++i) {
    EXPECT_NEAR(bf2f(act_h[i]), expected_act, 0.5f);  // bf16 precision
  }
  for (int i = 0; i < M * hidden; ++i) {
    float err = std::fabs(out_h[i] - expected_out);
    EXPECT_LT(err / expected_out, 1e-2f);
  }

  // linear_beta <= 0: `up` passes through unmodified (no softcap) — covers
  // the passthrough branch of the SiTU epilogue.
  std::fill(out_h.begin(), out_h.end(), 0.0f);
  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size,
      1.0f, 1 /* kSiTU */, beta, 0.0f /* linear_beta <= 0 */,
      nullptr, nullptr);
  // gate_out = beta * tanh(128/beta) * sigmoid(128) = 4.0; up_out = up = 128.
  float expected_act_passthrough = (beta * std::tanh(gate / beta) * sig) * up;
  for (int i = 0; i < EM * ispp; ++i) {
    EXPECT_NEAR(bf2f(act_h[i]), expected_act_passthrough, 0.5f);
  }
}

// ======================================================================
//  Multi-expert: E=4, varied expert weights
// ======================================================================
TEST(FusedMoe, MultiExpert) {
  constexpr int M = 8, hidden = 128, ispp = 64, E = 4;
  constexpr int group_size = 32, BLOCK_M = 16;
  constexpr int EM = 16;  // padded

  std::vector<uint16_t> A_bf16(M * hidden, f2bf(1.0f));

  int w13_eb = 2 * ispp * hidden / 2;
  int w13s_eb = 2 * ispp * hidden / group_size;
  int w2_eb = hidden * ispp / 2;
  int w2s_eb = hidden * ispp / group_size;

  std::vector<uint8_t> w13(E * w13_eb);
  std::vector<uint8_t> w13_scale(E * w13s_eb, 127);
  std::vector<uint8_t> w2(E * w2_eb);
  std::vector<uint8_t> w2_scale(E * w2s_eb, 127);

  for (int e = 0; e < E; ++e) {
    float wv = 1.0f + static_cast<float>(e);  // 1,2,3,4
    uint8_t pb = pack_e2m1_pair(wv, wv);
    for (int i = 0; i < w13_eb; ++i) w13[e * w13_eb + i] = pb;
    for (int i = 0; i < w2_eb; ++i)  w2[e * w2_eb + i] = pb;
  }

  std::vector<int32_t> sorted_ids(EM);
  for (int i = 0; i < EM; ++i) sorted_ids[i] = i % M;
  std::vector<float> topk_ws(EM, 1.0f);
  std::vector<int32_t> expert_ids(EM / BLOCK_M);
  for (int i = 0; i < EM / BLOCK_M; ++i) expert_ids[i] = i % E;

  std::vector<uint16_t> act_h(EM * ispp, 0);
  std::vector<float> out_h(M * hidden, 0.0f);

  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size, 0.0f,
      0, 4.0f, 25.0f, nullptr, nullptr);

  // Just verify finite and positive
  for (int i = 0; i < M * hidden; ++i) {
    EXPECT_TRUE(std::isfinite(out_h[i]));
    EXPECT_GT(out_h[i], 0.0f);
  }
}

// ======================================================================
//  Expert filtering: expert_id = -1 → output unchanged
// ======================================================================
TEST(FusedMoe, ExpertFilter) {
  constexpr int M = 16, hidden = 128, ispp = 64;
  constexpr int group_size = 32, BLOCK_M = 16;
  constexpr int EM = M;

  std::vector<uint16_t> A_bf16(M * hidden, f2bf(1.0f));
  uint8_t one_byte = pack_e2m1_pair(1.0f, 1.0f);
  std::vector<uint8_t> w13(2 * ispp * hidden / 2, one_byte);
  std::vector<uint8_t> w13_scale(2 * ispp * hidden / group_size, 127);
  std::vector<uint8_t> w2(hidden * ispp / 2, one_byte);
  std::vector<uint8_t> w2_scale(hidden * ispp / group_size, 127);

  std::vector<int32_t> sorted_ids(EM);
  std::iota(sorted_ids.begin(), sorted_ids.end(), 0);
  std::vector<float> topk_ws(EM, 1.0f);
  std::vector<int32_t> expert_ids(EM / BLOCK_M, -1);  // all filtered

  std::vector<uint16_t> act_h(EM * ispp, 0);
  std::vector<float> out_h(M * hidden, 99.0f);

  fused_moe_mxfp4_cpu(
      A_bf16.data(), w13.data(), w13_scale.data(),
      w2.data(), w2_scale.data(),
      sorted_ids.data(), topk_ws.data(), expert_ids.data(),
      act_h.data(), out_h.data(),
      M, hidden, ispp, 1, EM, group_size, 0.0f,
      0, 4.0f, 25.0f, nullptr, nullptr);

  // Output unchanged
  for (int i = 0; i < M * hidden; ++i) {
    EXPECT_NEAR(out_h[i], 99.0f, 0.01f);
  }
}

// ======================================================================
//  moe_align_block_size test
// ======================================================================
TEST(MoeAlign, Basic8x4) {
  constexpr int M = 8, top_k = 4, E = 4, BS = 16;
  int N = M * top_k;  // 32

  // Expert 0: 5 tokens (0×4, 1×1), expert 1: 11 tokens, expert 2: 16 tokens
  std::vector<int32_t> topk_ids(N);
  for (int i = 0; i < 4; ++i) topk_ids[0*top_k + i] = 0;
  for (int i = 0; i < 4; ++i) topk_ids[1*top_k + i] = (i == 0) ? 0 : 1;
  for (int i = 0; i < 4; ++i) topk_ids[2*top_k + i] = 1;
  for (int i = 0; i < 4; ++i) topk_ids[3*top_k + i] = 1;
  for (int i = 0; i < 4; ++i) topk_ids[4*top_k + i] = 2;
  for (int i = 0; i < 4; ++i) topk_ids[5*top_k + i] = 2;
  for (int i = 0; i < 4; ++i) topk_ids[6*top_k + i] = 2;
  for (int i = 0; i < 4; ++i) topk_ids[7*top_k + i] = 2;

  int max_EM = ((N + BS - 1) / BS + E) * BS;
  std::vector<int32_t> sorted_ids(max_EM, -1);
  std::vector<int32_t> expert_ids(max_EM / BS, -99);

  int EM_padded = moe_align_block_size(
      topk_ids.data(), M, top_k, BS, E,
      sorted_ids.data(), expert_ids.data());

  // 5+11+16=32 tokens, padded to 3 blocks of 16
  EXPECT_EQ(EM_padded, 48);
  EXPECT_EQ(expert_ids[0], 0);
  EXPECT_EQ(expert_ids[1], 1);
  EXPECT_EQ(expert_ids[2], 2);

  // Expert 0 block: 5 real flat indices [0,1,2,3,4], padded with N=32
  int cnt0 = 0;
  for (int i = 0; i < BS; ++i) if (sorted_ids[i] == 0) ++cnt0;
  EXPECT_EQ(cnt0, 1);         // flat 0 (token 0 sel 0) appears once
  EXPECT_EQ(sorted_ids[4], 4);  // flat 4 = token 1 sel 0

  // Expert 2 block: flat indices 16..31 (tokens 4,5,6,7)
  int cnt16 = 0, cnt31 = 0;
  for (int i = 32; i < 48; ++i) {
    if (sorted_ids[i] == 16) ++cnt16;  // token 4 sel 0
    if (sorted_ids[i] == 31) ++cnt31;  // token 7 sel 3
  }
  EXPECT_EQ(cnt16, 1);
  EXPECT_EQ(cnt31, 1);
}
