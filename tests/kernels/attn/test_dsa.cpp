// tests/kernels/attn/test_dsa.cpp
//
// Host tests for the DeepseekSparseAttn (DSA) sparse-MLA forward
// (issue #51): hand-checked cases for BOTH tail_dim == 0 (GLM-5.3-Flash,
// the shape tilelang cannot compile) and tail_dim > 0 (DeepSeek-V3), an
// independent two-pass reference across full GLM-5.3 / DeepSeek-V3 shapes,
// the masked-index (kpool tail) behaviour, the optional LSE, the per-shape
// config selector, and the null-arg / empty no-op edges.
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "vkernels/kernels/dsa.hpp"

using vkernels::kernels::dsa_config_for;
using vkernels::kernels::dsa_sparse_fwd_cpu;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kLog2E = 1.4426950408889634f;

// Independent two-pass base-2 softmax reference (different code path:
// builds the full score vector, then standard softmax, then AV over the
// masked selected keys).
void ref(int S_q, int S_kv, int H, int dim, int tail_dim, int topk,
         int kv_group, float sm_scale, bool return_lse,
         const std::vector<float>& q, const std::vector<float>& kv,
         const std::vector<int32_t>& idx, std::vector<float>& out,
         std::vector<float>& lse) {
  const int W = dim + tail_dim;
  const int d_v = dim - tail_dim;
  for (int i = 0; i < S_q; ++i) {
    const int32_t* idx_i = idx.data() + (size_t)i * kv_group * topk;
    for (int h = 0; h < H; ++h) {
      const float* qh = &q[((size_t)i * H + h) * W];
      const float* q_main = qh;
      const float* q_tail = qh + dim;
      std::vector<float> s(topk, -kInf);
      float mx = -kInf;
      for (int k = 0; k < topk; ++k) {
        const int32_t id = idx_i[k];
        if (id < 0 || id >= S_kv) continue;
        const float* kp = &kv[(size_t)id * W];
        float d = 0;
        for (int t = 0; t < dim; ++t) d += q_main[t] * kp[t];
        for (int t = 0; t < tail_dim; ++t) d += q_tail[t] * kp[dim + t];
        s[k] = sm_scale * d;
        if (s[k] > mx) mx = s[k];
      }
      float* oh = &out[((size_t)i * H + h) * d_v];
      if (mx == -kInf) {
        for (int t = 0; t < d_v; ++t) oh[t] = 0;
        if (return_lse) lse[(size_t)i * H + h] = -kInf;
        continue;
      }
      float sum = 0;
      for (int k = 0; k < topk; ++k)
        if (s[k] != -kInf) { s[k] = std::exp2(s[k] - mx); sum += s[k]; }
      for (int t = 0; t < d_v; ++t) oh[t] = 0;
      for (int k = 0; k < topk; ++k) {
        if (s[k] == -kInf) continue;
        const int32_t id = idx_i[k];
        const float* vj = &kv[(size_t)id * W];  // v = kv[id][0:d_v]
        const float a = s[k] / sum;
        for (int t = 0; t < d_v; ++t) oh[t] += a * vj[t];
      }
      if (return_lse) lse[(size_t)i * H + h] = mx + std::log2(sum);
    }
  }
}

}  // namespace

// Hand-checked case, tail_dim == 0 (GLM-5.3-Flash): q=[1,1], two keys,
// sm_scale=1. Scores {1,1} -> weights {1,1}/2 -> out {0.5,0.5}, lse=2.
TEST(DsaFwd, HandCheckedTailDimZero) {
  const float sm_scale = 1.0f;  // base-2 fold folded into the scale by caller
  std::vector<float> q = {1, 1};               // [1,1, 1, 2]  dim=2 tail=0
  std::vector<float> kv = {1, 0,  0, 1};       // 2 keys, dim=2
  std::vector<int32_t> idx = {0, 1};
  std::vector<float> out(2, -1), lse(1, 7);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 64, 1, sm_scale, /*lse=*/true,
                     q.data(), kv.data(), idx.data(), out.data(), lse.data());
  EXPECT_NEAR(out[0], 0.5f, 1e-6);
  EXPECT_NEAR(out[1], 0.5f, 1e-6);
  EXPECT_NEAR(lse[0], 2.0f, 1e-6);  // mx=1 + log2(2)=1
}

// Hand-checked case, tail_dim > 0 (DeepSeek-V3 layout): dim=2 tail=1 d_v=1.
//   q=[1,0 | 0]   key0=[v=2,km_x=1,kt=1]  key1=[v=3,km_x=0,kt=1]
//   scores = 2, 3 -> w0=2^-1=0.5, w1=1, sum=1.5
//   out = (0.5/1.5)*2 + (1/1.5)*3 = 4/1.5 ;  lse = 3 + log2(1.5)
TEST(DsaFwd, HandCheckedTailDimPositive) {
  const float sm_scale = 1.0f;
  std::vector<float> q = {1, 0, 0};            // main=[1,0] tail=[0]
  std::vector<float> kv = {2, 1, 1,  3, 0, 1}; // 2 keys, dim+tail=3
  std::vector<int32_t> idx = {0, 1};
  std::vector<float> out(1, -1), lse(1, 7);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 1, 2, 1, 64, 1, sm_scale, true,
                     q.data(), kv.data(), idx.data(), out.data(), lse.data());
  EXPECT_NEAR(out[0], 4.0f / 1.5f, 1e-6);
  EXPECT_NEAR(lse[0], 3.0f + std::log2(1.5f), 1e-6);
}

// Masked index (kpool tail): a -1 entry contributes zero weight.
//   q=[1,1], keys [[1,0],[0,1]], indices=[0,-1] -> only key0, out=[1,0].
TEST(DsaFwd, MaskedIndexIsZero) {
  const float sm_scale = 1.0f;
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {1, 0,  0, 1};
  std::vector<int32_t> idx = {0, -1};
  std::vector<float> out(2, -1), lse(1, 7);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 64, 1, sm_scale, true,
                     q.data(), kv.data(), idx.data(), out.data(), lse.data());
  EXPECT_NEAR(out[0], 1.0f, 1e-6);
  EXPECT_NEAR(out[1], 0.0f, 1e-6);
  EXPECT_NEAR(lse[0], 1.0f, 1e-6);  // mx=1 + log2(1)=0

  // Out-of-range index (>= S_kv) is masked too.
  std::vector<int32_t> idx2 = {0, 5};
  std::vector<float> out2(2, -1);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 64, 1, sm_scale, false,
                     q.data(), kv.data(), idx2.data(), out2.data(), nullptr);
  EXPECT_NEAR(out2[0], 1.0f, 1e-6);
  EXPECT_NEAR(out2[1], 0.0f, 1e-6);
}

// All selected keys masked -> zero output and -inf LSE (the branch the
// kernel and oracle must hit when a kpool tile is entirely padding).
TEST(DsaFwd, FullyMaskedIsZero) {
  const float sm_scale = 1.0f;
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {9, 9,  9, 9};
  std::vector<int32_t> idx = {-1, -1};
  std::vector<float> out(2, -1), lse(1, 7);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 64, 1, sm_scale, true,
                     q.data(), kv.data(), idx.data(), out.data(), lse.data());
  EXPECT_NEAR(out[0], 0.0f, 1e-6);
  EXPECT_NEAR(out[1], 0.0f, 1e-6);
  EXPECT_EQ(lse[0], -kInf);
}

// return_lse == false: lse is not touched (caller may pass nullptr).
TEST(DsaFwd, LseOptional) {
  const float sm_scale = 1.0f;
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {1, 0,  0, 1};
  std::vector<int32_t> idx = {0, 1};
  std::vector<float> out(2, -1);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 64, 1, sm_scale, /*lse=*/false,
                     q.data(), kv.data(), idx.data(), out.data(), nullptr);
  EXPECT_NEAR(out[0], 0.5f, 1e-6);
  EXPECT_NEAR(out[1], 0.5f, 1e-6);
}

// Cross-check against the independent reference across BOTH the GLM-5.3
// (tail_dim == 0) and DeepSeek-V3 (tail_dim > 0) shapes, including the
// topk-padded-to-multiple-of-64 layout (trailing entries are masked), and
// the optional LSE.
TEST(DsaFwd, MatchesReferenceBothShapeFamilies) {
  struct Cfg { int S_q, S_kv, H, dim, tail_dim, topk; };
  const Cfg cfgs[] = {
      // GLM-5.3-Flash: tail_dim == 0 (the tilelang-uncompilable case)
      {1, 256, 64, 256, 0, 128},    // decode, full H, topk=128 (2 groups)
      {1, 512, 64, 256, 0, 256},    // decode, topk=256 (4 groups)
      {4, 256, 8, 256, 0, 128},     // prefill
      // DeepSeek-V3: tail_dim > 0
      {1, 256, 16, 576, 64, 256},   // decode, d_v=512
      {1, 128, 8, 576, 64, 128},    // decode, smaller
      {4, 128, 4, 576, 64, 128},    // prefill
      // generic small mixed
      {1, 16, 2, 8, 2, 16},
  };
  std::mt19937 rng(12345);
  auto rf = [&]() {
    return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;  // [-1, 1)
  };
  for (const auto& c : cfgs) {
    const int W = c.dim + c.tail_dim;
    const int d_v = c.dim - c.tail_dim;
    // Emulate kpool tails + padding by masking ~10% of entries. topk is a
    // caller convention (often a multiple of 64); the forward scores the
    // selected keys directly, so no divisibility is required.
    std::vector<float> q((size_t)c.S_q * c.H * W);
    std::vector<float> kv((size_t)c.S_kv * W);
    std::vector<int32_t> idx((size_t)c.S_q * c.topk);
    for (auto& x : q) x = rf();
    for (auto& x : kv) x = rf();
    for (int i = 0; i < c.S_q; ++i)
      for (int k = 0; k < c.topk; ++k)
        idx[(size_t)i * c.topk + k] =
            (k % 10 == 9) ? -1 : static_cast<int32_t>(rng() % c.S_kv);
    const float sm_scale =
        (1.0f / std::sqrt(static_cast<float>(W))) * kLog2E;
    std::vector<float> out((size_t)c.S_q * c.H * d_v, 0);
    std::vector<float> lse((size_t)c.S_q * c.H, 0);
    std::vector<float> rout(out.size(), 0), rlse(lse.size(), 0);
    dsa_sparse_fwd_cpu(c.S_q, c.S_kv, c.H, c.dim, c.tail_dim, c.topk, 1,
                      64, 1, sm_scale, true, q.data(), kv.data(),
                      idx.data(), out.data(), lse.data());
    ref(c.S_q, c.S_kv, c.H, c.dim, c.tail_dim, c.topk, 1, sm_scale, true,
        q, kv, idx, rout, rlse);
    float maxd = 0, maxlse = 0;
    for (size_t i = 0; i < out.size(); ++i)
      maxd = std::max(maxd, std::fabs(out[i] - rout[i]));
    for (size_t i = 0; i < lse.size(); ++i) {
      const float li = std::isfinite(lse[i]) ? lse[i] : 0.0f;
      const float rl = std::isfinite(rlse[i]) ? rlse[i] : 0.0f;
      maxlse = std::max(maxlse, std::fabs(li - rl));
    }
    EXPECT_NEAR(maxd, 0.0f, 1e-4f);
    EXPECT_NEAR(maxlse, 0.0f, 1e-4f);
  }
}

TEST(DsaConfig, DecodeVsPrefill) {
  int bq = 0, th = 0, bi = 0, ii = 0;
  dsa_config_for(1, 64, 256, 128, &bq, &th, &bi, &ii);
  EXPECT_EQ(bq, 1);
  EXPECT_EQ(th, 64);
  EXPECT_EQ(bi, 64);
  dsa_config_for(8, 64, 256, 128, &bq, &th, &bi, &ii);
  EXPECT_EQ(bq, 1);
  EXPECT_EQ(th, 64);
  dsa_config_for(16, 64, 256, 128, &bq, &th, &bi, &ii);
  EXPECT_EQ(bq, 4);
  EXPECT_EQ(th, 256);
  // inner_iter grows while it keeps topk divisible by block_I*inner_iter.
  dsa_config_for(1, 64, 256, 512, &bq, &th, &bi, &ii);
  EXPECT_EQ(bi, 64);
  EXPECT_TRUE(512 % (bi * ii) == 0 && ii >= 1);
}

TEST(DsaFwd, NullArgsThrow) {
  std::vector<float> q(4, 1), kv(4, 1), out(2, 1);
  std::vector<int32_t> idx(2, 0);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 64, 1, 1.0f, false,
                                  nullptr, kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 64, 1, 1.0f, false,
                                  q.data(), nullptr, idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 64, 1, 1.0f, false,
                                  q.data(), kv.data(), nullptr, out.data(),
                                  nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 64, 1, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), nullptr,
                                  nullptr),
               std::invalid_argument);
  // return_lse == true but lse is null.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 64, 1, 1.0f, true,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // kv_group != 1 is rejected (kernel assumes a single shared head_kv).
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 2, 64, 1, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // block_I == 0 / inner_iter == 0 are rejected (configuration hints).
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 0, 1, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
}

TEST(DsaFwd, EmptyIsNoOp) {
  std::vector<float> out = {3.0f, 4.0f};
  dsa_sparse_fwd_cpu(0, 1, 1, 2, 0, 2, 1, 64, 1, 1.0f, false, nullptr,
                     nullptr, nullptr, out.data(), nullptr);
  dsa_sparse_fwd_cpu(1, 1, 0, 2, 0, 2, 1, 64, 1, 1.0f, false, nullptr,
                     nullptr, nullptr, out.data(), nullptr);
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
}
