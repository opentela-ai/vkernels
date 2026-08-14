// tests/kernels/attn/test_mla.cpp
//
// Host tests for the absorbed-form MLA forward (issue #21): a hand-checked
// two-query case, an independent softmax reference across K3-shaped
// (kv_lora_rank, qk_rope_head_dim, H) at decode and prefill, the causal
// mask via (q_start, kv_start), the per-shape config selector, and the
// null-arg / empty no-op edges.
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "vkernels/kernels/mla.hpp"

using vkernels::kernels::mla_config_for;
using vkernels::kernels::mla_fwd_cpu;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Independent two-pass softmax reference (different code path: builds the
// full score matrix, then standard softmax, then AV).
void ref(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
         int lr, int rhd, float scale,
         const std::vector<float>& q,
         const std::vector<float>& k_c,
         const std::vector<float>& k_pe,
         const std::vector<float>& v_c,
         std::vector<float>& out) {
  const int Dq = lr + rhd;
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int i = 0; i < S_q; ++i) {
        const int gqi = q_start + i;
        const float* qi = &q[((size_t)(b * H + h) * S_q + i) * Dq];
        std::vector<float> s(S_kv, kInf * -1.0f);
        float mx = -kInf;
        for (int j = 0; j < S_kv; ++j) {
          const int gkj = kv_start + j;
          if (gkj > gqi) continue;
          const float* kj = &k_c[((size_t)b * S_kv + j) * lr];
          const float* pj = &k_pe[((size_t)b * S_kv + j) * rhd];
          float d = 0;
          for (int t = 0; t < lr; ++t) d += qi[t] * kj[t];
          for (int t = 0; t < rhd; ++t) d += qi[lr + t] * pj[t];
          s[j] = scale * d;
          if (s[j] > mx) mx = s[j];
        }
        float sum = 0;
        for (int j = 0; j < S_kv; ++j)
          if (s[j] != -kInf) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float* oi = &out[((size_t)(b * H + h) * S_q + i) * lr];
        for (int t = 0; t < lr; ++t) oi[t] = 0;
        if (sum == 0) continue;
        for (int j = 0; j < S_kv; ++j) {
          if (s[j] == -kInf) continue;
          const float a = s[j] / sum;
          const float* vj = &v_c[((size_t)b * S_kv + j) * lr];
          for (int t = 0; t < lr; ++t) oi[t] += a * vj[t];
        }
      }
}

}  // namespace

// Hand-checked two-query case (see test_mla.cpp derivation in the PR).
TEST(MlaFwd, HandChecked) {
  // B=1 H=1 S_q=2 S_kv=2, lr=2 rhd=2, scale = 1/sqrt(4) = 0.5
  const float scale = 0.5f;
  std::vector<float> q = {
      1, 0, 0, 0,  // q0: nope=[1,0] rope=[0,0]
      0, 1, 1, 0,  // q1: nope=[0,1] rope=[1,0]
  };
  std::vector<float> k_c = {1, 0, 0, 1};
  std::vector<float> k_pe = {0, 1, 1, 0};
  std::vector<float> v_c = {5, 6, 7, 8};
  std::vector<float> out(2 * 2, -1);
  mla_fwd_cpu(1, 1, 2, 2, 0, 0, 2, 2, scale, q.data(), k_c.data(),
              k_pe.data(), v_c.data(), out.data());

  // q0 attends only to k0 (causal): out0 = v_c[0] = [5, 6]
  EXPECT_NEAR(out[0], 5.0f, 1e-6);
  EXPECT_NEAR(out[1], 6.0f, 1e-6);
  // q1 attends to k0,k1: s0=0, s1=1 -> w0=1/e /(1+1/e), w1=1/(1+1/e)
  // out1 = w0*[5,6] + w1*[7,8]
  const float w1 = 1.0f / (1.0f + 1.0f / std::exp(1.0f));
  const float w0 = 1.0f - w1;
  EXPECT_NEAR(out[2], w0 * 5 + w1 * 7, 1e-6);
  EXPECT_NEAR(out[3], w0 * 6 + w1 * 8, 1e-6);
}

// Cross-check against the independent reference across K3-shaped configs.
TEST(MlaFwd, MatchesReferenceK3Shapes) {
  struct Cfg { int B, H, S_q, S_kv, q0, kv0, lr, rhd; };
  const Cfg cfgs[] = {
      // decode (single query over the full KV)
      {1, 1, 1, 16, 15, 0, 8, 4},
      {1, 4, 1, 32, 31, 0, 512, 64},  // K3 decode head count + latent
      // prefill (causal square)
      {1, 1, 8, 8, 0, 0, 8, 4},
      {1, 2, 16, 16, 0, 0, 512, 64},  // K3 prefill shard
      // batched
      {2, 2, 4, 4, 0, 0, 6, 2},
  };
  std::mt19937 rng(7);
  auto rf = [&]() {
    return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;  // [-1, 1)
  };
  for (const auto& c : cfgs) {
    const int Dq = c.lr + c.rhd;
    std::vector<float> q((size_t)c.B * c.H * c.S_q * Dq);
    std::vector<float> k_c((size_t)c.B * c.S_kv * c.lr);
    std::vector<float> k_pe((size_t)c.B * c.S_kv * c.rhd);
    std::vector<float> v_c((size_t)c.B * c.S_kv * c.lr);
    std::vector<float> out((size_t)c.B * c.H * c.S_q * c.lr, 0);
    std::vector<float> exp(out.size(), 0);
    for (auto& x : q) x = rf();
    for (auto& x : k_c) x = rf();
    for (auto& x : k_pe) x = rf();
    for (auto& x : v_c) x = rf();
    const float scale = 1.0f / std::sqrt(static_cast<float>(Dq));
    mla_fwd_cpu(c.B, c.H, c.S_q, c.S_kv, c.q0, c.kv0, c.lr, c.rhd, scale,
                q.data(), k_c.data(), k_pe.data(), v_c.data(), out.data());
    ref(c.B, c.H, c.S_q, c.S_kv, c.q0, c.kv0, c.lr, c.rhd, scale,
        q, k_c, k_pe, v_c, exp);
    float maxd = 0;
    for (size_t i = 0; i < out.size(); ++i)
      maxd = std::max(maxd, std::fabs(out[i] - exp[i]));
    EXPECT_NEAR(maxd, 0.0f, 1e-5f);
  }
}

// Causal mask via (q_start, kv_start): a decode query at position P attends
// exactly the first P+1 keys.
TEST(MlaFwd, CausalMaskWithOffset) {
  // S_q=1, q_start=3, S_kv=4, kv_start=0 -> attends j in {0,1,2,3}, j=3 is the
  // "self" key (gkj=3 <= gqi=3). Make all but key 3 zero so only it should
  // contribute.
  const float scale = 1.0f;
  std::vector<float> q = {1, 1, 1, 1};  // lr=2 rhd=2
  std::vector<float> k_c(4 * 2, 0), k_pe(4 * 2, 0), v_c(4 * 2, 0);
  k_c[3 * 2 + 0] = 1; k_c[3 * 2 + 1] = 1;  // key 3 nope
  k_pe[3 * 2 + 0] = 1; k_pe[3 * 2 + 1] = 1;  // key 3 rope
  v_c[3 * 2 + 0] = 9; v_c[3 * 2 + 1] = 10;
  std::vector<float> out(2, -1);
  mla_fwd_cpu(1, 1, 1, 4, 3, 0, 2, 2, scale, q.data(), k_c.data(),
              k_pe.data(), v_c.data(), out.data());
  // Only key 3 is unmasked and nonzero; scores for j<3 are 0 (=> w=1 each),
  // j=3 is 4 -> softmax picks mostly j=3. With w_j for j<3 = exp(0-4)=e^-4,
  // w_3 = exp(4-4)=1, sum=1+3e^-4. out ≈ (1/sum)*[9,10].
  const float sum = 1.0f + 3.0f * std::exp(-4.0f);
  EXPECT_NEAR(out[0], 9.0f / sum, 1e-5f);
  EXPECT_NEAR(out[1], 10.0f / sum, 1e-5f);

  // Now shift kv_start so key 3 is masked (gkj = kv_start+3 = 4 > gqi = 3):
  // with kv_start=1, keys map to gkj in {1,2,3,4}; the last is masked.
  std::vector<float> out2(2, -1);
  mla_fwd_cpu(1, 1, 1, 4, 3, 1, 2, 2, scale, q.data(), k_c.data(),
              k_pe.data(), v_c.data(), out2.data());
  // j=0,1,2 (gkj 1,2,3) are unmasked but zero; j=3 (gkj 4) masked. Sum of
  // weights = 3 (all scores 0), out = 0.
  EXPECT_NEAR(out2[0], 0.0f, 1e-6f);
  EXPECT_NEAR(out2[1], 0.0f, 1e-6f);
}

TEST(MlaFwd, FullyMaskedQueryIsZero) {
  // kv_start=5 with q_start=0, S_q=1 -> gqi=0, every gkj=5..8 > 0, so ALL
  // keys are masked. This exercises the "sum == 0" zero-output branch
  // (mla.cpp lines 79-81): nothing is read from v_c and out stays 0.
  const float scale = 1.0f;
  std::vector<float> q = {1, 1, 1, 1};        // lr=2 rhd=2
  std::vector<float> k_c(4 * 2, 7), k_pe(4 * 2, 7), v_c(4 * 2, 9);
  std::vector<float> out(2, -1);
  mla_fwd_cpu(1, 1, 1, 4, 0, 5, 2, 2, scale, q.data(), k_c.data(),
              k_pe.data(), v_c.data(), out.data());
  EXPECT_NEAR(out[0], 0.0f, 1e-6f);
  EXPECT_NEAR(out[1], 0.0f, 1e-6f);
}

TEST(MlaConfig, DecodeVsPrefill) {
  int bq = 0, bn = 0, th = 0;
  mla_config_for(1, 512, 64, &bq, &bn, &th);
  EXPECT_EQ(bq, 1);
  EXPECT_EQ(bn, 64);
  EXPECT_EQ(th, 64);
  // S_q == 8 is still decode.
  mla_config_for(8, 512, 64, &bq, &bn, &th);
  EXPECT_EQ(bq, 1);
  EXPECT_EQ(th, 64);
  // S_q == 9 crosses into prefill.
  mla_config_for(9, 512, 64, &bq, &bn, &th);
  EXPECT_EQ(bq, 4);
  EXPECT_EQ(bn, 64);
  EXPECT_EQ(th, 256);
}

TEST(MlaFwd, NullArgsThrow) {
  std::vector<float> q(4, 1), c(4, 1);
  EXPECT_THROW(mla_fwd_cpu(1, 1, 1, 1, 0, 0, 2, 2, 1.0f, nullptr,
                            c.data(), c.data(), c.data(), c.data()),
               std::invalid_argument);
  EXPECT_THROW(mla_fwd_cpu(1, 1, 1, 1, 0, 0, 2, 2, 1.0f, q.data(),
                            nullptr, c.data(), c.data(), c.data()),
               std::invalid_argument);
  EXPECT_THROW(mla_fwd_cpu(1, 1, 1, 1, 0, 0, 2, 2, 1.0f, q.data(),
                            c.data(), nullptr, c.data(), c.data()),
               std::invalid_argument);
  EXPECT_THROW(mla_fwd_cpu(1, 1, 1, 1, 0, 0, 2, 2, 1.0f, q.data(),
                            c.data(), c.data(), nullptr, c.data()),
               std::invalid_argument);
  EXPECT_THROW(mla_fwd_cpu(1, 1, 1, 1, 0, 0, 2, 2, 1.0f, q.data(),
                            c.data(), c.data(), c.data(), nullptr),
               std::invalid_argument);
}

TEST(MlaFwd, EmptyIsNoOp) {
  std::vector<float> out = {3.0f, 4.0f};
  // B == 0 / S_q == 0 / S_kv == 0 short-circuit; nothing read or written.
  mla_fwd_cpu(0, 1, 1, 1, 0, 0, 2, 2, 1.0f, nullptr, nullptr, nullptr,
              nullptr, nullptr);
  mla_fwd_cpu(1, 1, 0, 1, 0, 0, 2, 2, 1.0f, nullptr, nullptr, nullptr,
              nullptr, out.data());
  mla_fwd_cpu(1, 1, 1, 0, 0, 0, 2, 2, 1.0f, nullptr, nullptr, nullptr,
              nullptr, out.data());
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
}
