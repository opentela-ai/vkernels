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
using vkernels::kernels::dsa_topk_logits_cpu;
using vkernels::kernels::dsa_topk_logits_fits_lds;
using vkernels::kernels::dsa_topk_logits_fits_lds_fp8q;
using vkernels::kernels::dsa_topk_logits_fits_lds_mfma;
using vkernels::kernels::dsa_topk_logits_split_for;

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
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, sm_scale, /*lse=*/true,
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
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 1, 2, 1, 2, 1, sm_scale, true,
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
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, sm_scale, true,
                     q.data(), kv.data(), idx.data(), out.data(), lse.data());
  EXPECT_NEAR(out[0], 1.0f, 1e-6);
  EXPECT_NEAR(out[1], 0.0f, 1e-6);
  EXPECT_NEAR(lse[0], 1.0f, 1e-6);  // mx=1 + log2(1)=0

  // Out-of-range index (>= S_kv) is masked too.
  std::vector<int32_t> idx2 = {0, 5};
  std::vector<float> out2(2, -1);
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, sm_scale, false,
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
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, sm_scale, true,
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
  dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, sm_scale, /*lse=*/false,
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
    // caller convention (often a multiple of 64); divisibility IS validated
    // (issue #57), so use dsa_config_for for block_I/inner_iter.
    int bq = 0, th = 0, bi = 0, ii = 0;
    dsa_config_for(c.S_q, c.H, c.dim, c.topk, &bq, &th, &bi, &ii);
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
                      bi, ii, sm_scale, true, q.data(), kv.data(),
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
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, false,
                                  nullptr, kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, false,
                                  q.data(), nullptr, idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, false,
                                  q.data(), kv.data(), nullptr, out.data(),
                                  nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), nullptr,
                                  nullptr),
               std::invalid_argument);
  // return_lse == true but lse is null.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, true,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // kv_group != 1 is rejected (kernel assumes a single shared head_kv).
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 1, 1, 2, 0, 2, 2, 2, 1, 1.0f, false,
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
  dsa_sparse_fwd_cpu(0, 1, 1, 2, 0, 2, 1, 2, 1, 1.0f, false, nullptr,
                     nullptr, nullptr, out.data(), nullptr);
  dsa_sparse_fwd_cpu(1, 1, 0, 2, 0, 2, 1, 2, 1, 1.0f, false, nullptr,
                     nullptr, nullptr, out.data(), nullptr);
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
}

// topk not divisible by block_I*inner_iter is rejected (issue #57): the
// kernel tiles `topk` into groups of block_I*inner_iter, so a non-multiple
// is a misconfigured caller, not a silently-wrong result.
TEST(DsaFwd, DivisibilityCheckThrows) {
  std::vector<float> q(2, 1), kv(4, 1), out(2, 0);
  std::vector<int32_t> idx(2, 0);
  std::vector<float> lse(1, 0);
  // topk=2, block_I=4, inner_iter=1: 2 % (4*1) = 2 != 0.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 4, 1, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // topk=2, block_I=2, inner_iter=2: 2 % (2*2) = 2 != 0.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 2, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // topk=3, block_I=2, inner_iter=1: 3 % (2*1) = 1 != 0 (odd topk).
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 3, 1, 2, 0, 3, 1, 2, 1, 1.0f, false,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // A VALID divisible call (topk=4, block_I=2, inner_iter=2) must NOT throw.
  std::vector<float> q4(4, 1), kv4(8, 1), out4(4, 0);
  std::vector<int32_t> idx4(4, 0);
  EXPECT_NO_THROW(dsa_sparse_fwd_cpu(1, 4, 1, 2, 0, 4, 1, 2, 2, 1.0f, false,
                                     q4.data(), kv4.data(), idx4.data(),
                                     out4.data(), nullptr));
}

// `out` aliasing any input (or `lse`) is rejected (issue #57): the two-pass
// softmax writes a row into `out` then reads it back, so a shared buffer
// would mix one query's output into the next query's scores.
TEST(DsaFwd, AliasThrows) {
  std::vector<float> q(2, 1), kv(4, 1), out(2, 0);
  std::vector<int32_t> idx(2, 0);
  std::vector<float> lse(1, 0);
  // out == q (same buffer): the scores read `out` back as `q`.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, 1.0f, false,
                                  out.data(), kv.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // out == kv: the values read `out` back as `kv`.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, 1.0f, false,
                                  q.data(), out.data(), idx.data(), out.data(),
                                  nullptr),
               std::invalid_argument);
  // out == lse (when return_lse): the LSE write clobbers `out` mid-row.
  EXPECT_THROW(dsa_sparse_fwd_cpu(1, 2, 1, 2, 0, 2, 1, 2, 1, 1.0f, true,
                                  q.data(), kv.data(), idx.data(), out.data(),
                                  out.data()),
               std::invalid_argument);
}

// ---------------------------------------------------------------------------
//  DSA paged-MQA gated top-k logits (issue #51): the kpool>1 indexer path.
//  Hand-checked cases for the gated-logit formula (per-head gate weighting,
//  ReLU on the per-head dot, per-token k_scale), the seq_len truncation
//  (tokens >= seq_len left unwritten -- the caller ZEROES the output first),
//  the OOB-page mask (page_table[b,i] < 0 or >= num_blocks skipped, exactly
//  like dsa_sparse_fwd masks an index >= S_kv), the null-arg / negative-dim
//  / empty no-op edges, and an independent reference across several shapes.
// ---------------------------------------------------------------------------

// Independent reference (different decomposition: iterate (b, t) with
// i = t/block, j = t%block; truncate per-token with `continue`, not a page
// `break`; OOB page skipped with `continue`) to catch indexing / stride /
// truncation errors the hand-checked cases might miss.
static void topk_ref(int batch_size, int num_heads, int head_dim, int block,
                     int max_table_len, int num_blocks,
                     const std::vector<float>& q,    // [bs, H, D]
                     const std::vector<float>& kv,   // [num_blocks, block, D]
                     const std::vector<float>& k_scale,  // [num_blocks, block]
                     const std::vector<float>& gate,     // [bs, H]
                     const std::vector<int32_t>& seq_lens,
                     const std::vector<int32_t>& page_table,
                     std::vector<float>& out) {  // [bs, max_table_len*block], ZEROED
  const int max_t = max_table_len * block;
  for (int b = 0; b < batch_size; ++b) {
    const int seq_len = seq_lens[b];
    for (int t = 0; t < max_t; ++t) {
      if (t >= seq_len) continue;
      const int i = t / block, j = t % block;
      const int32_t page = page_table[(size_t)b * max_table_len + i];
      if (page < 0 || page >= num_blocks) continue;
      float acc = 0.0f;
      for (int h = 0; h < num_heads; ++h) {
        const float* qh = q.data() + (((size_t)b * num_heads + h) * head_dim);
        const float* kj = kv.data() + (((size_t)page * block + j) * head_dim);
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
        acc += std::fmax(dot, 0.0f) * gate[(size_t)b * num_heads + h];
      }
      out[(size_t)b * max_table_len * block + t] =
          k_scale[(size_t)page * block + j] * acc;
    }
  }
}

// Hand-checked, single head, two-token page (block=2). q=[1,1] keys
// [[1,0],[0,1]] scales [2,3] gate [1] seq_len 2:
//   t=0: dot=1, acc=1*1=1, out=2*1=2 ;  t=1: dot=1, acc=1, out=3*1=3
TEST(DsaTopk, HandCheckedSingleHead) {
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {1, 0, 0, 1};        // [1 block][2 tokens][2]
  std::vector<float> k_scale = {2, 3};          // [1 block][2 tokens]
  std::vector<float> gate = {1};
  std::vector<int32_t> sl = {2};
  std::vector<int32_t> pt = {0};
  std::vector<float> out(2, 0);
  dsa_topk_logits_cpu(1, 1, 2, 2, 1, 1, q.data(), kv.data(), k_scale.data(),
                      gate.data(), sl.data(), pt.data(), out.data());
  EXPECT_NEAR(out[0], 2.0f, 1e-6f);
  EXPECT_NEAR(out[1], 3.0f, 1e-6f);
}

// Hand-checked, two heads, one-token pages (block=1): per-head gate
// weighting [2,3] with a ReLU-masked negative dot in each head.
//   q H0=[1,0] H1=[0,1]  page0 k=[1,-1]  page1 k=[-2,2]  scales [1,1] seq 2
//   t=0: H0 dot=+1 ->2, H1 dot=-1 ->0 ; acc=2 ; out=2
//   t=1: H0 dot=-2 ->0, H1 dot=+2 ->6 ; acc=6 ; out=6
TEST(DsaTopk, HandCheckedMultiHeadGateRelu) {
  std::vector<float> q = {1, 0, 0, 1};          // [1][2 heads][2]
  std::vector<float> kv = {1, -1, -2, 2};       // [2 blocks][1 token][2]
  std::vector<float> k_scale = {1, 1};          // [2 blocks][1 token]
  std::vector<float> gate = {2, 3};
  std::vector<int32_t> sl = {2};
  std::vector<int32_t> pt = {0, 1};
  std::vector<float> out(2, 0);
  dsa_topk_logits_cpu(1, 2, 2, 1, 2, 2, q.data(), kv.data(), k_scale.data(),
                      gate.data(), sl.data(), pt.data(), out.data());
  EXPECT_NEAR(out[0], 2.0f, 1e-6f);
  EXPECT_NEAR(out[1], 6.0f, 1e-6f);
}

// seq_len truncates WITHIN a page (block=4, seq_len=2): tokens 2,3 of the
// page are past seq_len and must be LEFT UNWRITTEN (caller zeroed first).
//   q=[1,1] keys [1,1][2,2][3,3][4,4] scales [1,1,1,1] gate [1] seq_len 2
//   t=0,1: dot=2,4 -> out=2,4 ;  t=2,3: unwritten -> stay 0
TEST(DsaTopk, SeqLenTruncatesWithinPage) {
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {1, 1, 2, 2, 3, 3, 4, 4};  // [1][4 tokens][2]
  std::vector<float> k_scale = {1, 1, 1, 1};          // [1][4 tokens]
  std::vector<float> gate = {1};
  std::vector<int32_t> sl = {2};
  std::vector<int32_t> pt = {0};
  std::vector<float> out(4, 0);                         // max_seq_len = 1*4
  dsa_topk_logits_cpu(1, 1, 2, 4, 1, 1, q.data(), kv.data(), k_scale.data(),
                      gate.data(), sl.data(), pt.data(), out.data());
  EXPECT_NEAR(out[0], 2.0f, 1e-6f);
  EXPECT_NEAR(out[1], 4.0f, 1e-6f);
  EXPECT_EQ(out[2], 0.0f);   // past seq_len, left unwritten
  EXPECT_EQ(out[3], 0.0f);
}

// OOB page (page_table entry >= num_blocks) is skipped -- its tokens stay
// unwritten, exactly like dsa_sparse_fwd masks an index >= S_kv.
//   q=[1,1] 1 block (2 keys [1,1]) scales [1,1] gate [1] seq_len 4
//   page_table [0, 5] -> page 5 is OOB (num_blocks=1); t=2,3 stay 0.
TEST(DsaTopk, OutOfBoundPageIsUnwritten) {
  std::vector<float> q = {1, 1};
  std::vector<float> kv = {1, 1, 1, 1};        // [1 block][2 tokens][2]
  std::vector<float> k_scale = {1, 1};          // [1 block][2 tokens]
  std::vector<float> gate = {1};
  std::vector<int32_t> sl = {4};                // expects 4 tokens
  std::vector<int32_t> pt = {0, 5};            // page 5 is OOB
  std::vector<float> out(4, 0);                 // max_seq_len = 2*2
  dsa_topk_logits_cpu(1, 1, 2, 2, 2, 1, q.data(), kv.data(), k_scale.data(),
                      gate.data(), sl.data(), pt.data(), out.data());
  EXPECT_NEAR(out[0], 2.0f, 1e-6f);
  EXPECT_NEAR(out[1], 2.0f, 1e-6f);
  EXPECT_EQ(out[2], 0.0f);   // OOB page, left unwritten
  EXPECT_EQ(out[3], 0.0f);
}

// Cross-check the oracle against the independent reference across several
// shapes (GLM-5.3-like D=128/block=64 included, with truncated seq_lens and
// ~10% of page_table entries masked to -1 to exercise both edges).
TEST(DsaTopk, MatchesReference) {
  struct Cfg { int bs, H, D, block, max_table_len, num_blocks; };
  const Cfg cfgs[] = {
      {1, 2, 8, 4, 16, 16},     // decode-ish
      {1, 4, 16, 8, 32, 32},    // more heads
      {2, 1, 8, 4, 16, 16},     // batch > 1
      {1, 2, 128, 64, 8, 8},    // GLM-5.3-ish (D=128, block=64)
      {1, 1, 2, 2, 4, 4},       // tiny
  };
  std::mt19937 rng(98765);
  auto rf = [&]() { return static_cast<float>(rng() % 2000) / 1000.0f - 1.0f; };
  for (const auto& c : cfgs) {
    const int max_seq_len = c.max_table_len * c.block;
    std::vector<float> q((size_t)c.bs * c.H * c.D);
    std::vector<float> kv((size_t)c.num_blocks * c.block * c.D);
    std::vector<float> k_scale((size_t)c.num_blocks * c.block);
    std::vector<float> gate((size_t)c.bs * c.H);
    std::vector<int32_t> sl(c.bs);
    std::vector<int32_t> pt((size_t)c.bs * c.max_table_len);
    for (auto& x : q) x = rf();
    for (auto& x : kv) x = rf();
    for (auto& x : k_scale) x = static_cast<float>(rng() % 1000) / 1000.0f + 0.5f;
    for (auto& x : gate) x = rf();
    for (int b = 0; b < c.bs; ++b)
      sl[b] = static_cast<int32_t>(rng() % (max_seq_len + 1));  // [0, max_seq_len]
    for (size_t i = 0; i < pt.size(); ++i)
      pt[i] = (rng() % 10 == 0) ? -1 : static_cast<int32_t>(rng() % c.num_blocks);

    std::vector<float> out((size_t)c.bs * max_seq_len, 0.0f);
    std::vector<float> rout((size_t)c.bs * max_seq_len, 0.0f);
    dsa_topk_logits_cpu(c.bs, c.H, c.D, c.block, c.max_table_len, c.num_blocks,
                        q.data(), kv.data(), k_scale.data(), gate.data(),
                        sl.data(), pt.data(), out.data());
    topk_ref(c.bs, c.H, c.D, c.block, c.max_table_len, c.num_blocks,
             q, kv, k_scale, gate, sl, pt, rout);
    float maxd = 0.0f;
    for (size_t i = 0; i < out.size(); ++i)
      maxd = std::max(maxd, std::fabs(out[i] - rout[i]));
    EXPECT_NEAR(maxd, 0.0f, 1e-4f);
  }
}

// Null pointers are rejected when there is work; allowed (no-op) when
// batch_size == 0 or max_table_len == 0; negative / zero dims are rejected.
TEST(DsaTopk, NullArgsAndDimsThrow) {
  std::vector<float> q(4, 1), kv(4, 1), k_scale(2, 1), gate(2, 1), out(2, 1);
  std::vector<int32_t> sl{2}, pt{0};
  // batch_size>0, max_table_len>0 -> every pointer is required.
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, nullptr, kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, q.data(), nullptr,
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, q.data(), kv.data(),
                      nullptr, gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, q.data(), kv.data(),
                      k_scale.data(), nullptr, sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), nullptr, pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), nullptr, out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), nullptr),
               std::invalid_argument);
  // No work -> null pointers are allowed (no-op, mirrors EmptyIsNoOp).
  EXPECT_NO_THROW(dsa_topk_logits_cpu(0, 2, 2, 2, 1, 1, nullptr, nullptr,
                      nullptr, nullptr, nullptr, nullptr, nullptr));
  EXPECT_NO_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 0, 1, nullptr, nullptr,
                      nullptr, nullptr, nullptr, nullptr, nullptr));
  // Negative / zero dims are rejected.
  EXPECT_THROW(dsa_topk_logits_cpu(-1, 2, 2, 2, 1, 1, nullptr, nullptr,
                      nullptr, nullptr, nullptr, nullptr, nullptr),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 0, 2, 2, 1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 0, 2, 1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 0, 1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, -1, 1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
  EXPECT_THROW(dsa_topk_logits_cpu(1, 2, 2, 2, 1, -1, q.data(), kv.data(),
                      k_scale.data(), gate.data(), sl.data(), pt.data(), out.data()),
               std::invalid_argument);
}

// Empty (batch_size==0 or max_table_len==0) is a no-op: output untouched.
TEST(DsaTopk, EmptyIsNoOp) {
  std::vector<float> out = {3.0f, 4.0f};
  dsa_topk_logits_cpu(0, 1, 2, 2, 1, 1, nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, out.data());
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
  dsa_topk_logits_cpu(1, 1, 2, 2, 0, 1, nullptr, nullptr, nullptr, nullptr,
                      nullptr, nullptr, out.data());
  EXPECT_EQ(out[0], 3.0f);
  EXPECT_EQ(out[1], 4.0f);
}

// Whether the indexer shape fits gfx942's 64 KB non-optin dynamic-LDS cap
// under the fp32-Q kernel -- the launcher's FAST-PATH guard. The kernel
// stages Q (H*D), the per-head gate (H), one K tile (B*D) and its
// per-token scales (B), all fp32: (H*D + H + B*D + B) * 4 B. GLM-5.3
// (H=32, D=128, B=64) stages 49,536 B (verified); H=64 stages 66,048 B and
// is refused BY THIS GUARD (the launcher then falls back to the fp8-Q
// kernel, see FitsLdsFp8q below). The host oracle is shape-agnostic, so
// these only exercise the launcher's dsa_topk_logits_fits_lds guard.
TEST(DsaTopk, FitsLdsCap) {
  EXPECT_TRUE(dsa_topk_logits_fits_lds(32, 128, 64));    // GLM-5.3: 49,536 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds(2, 8, 4));       // tiny: 216 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds(63, 128, 64));   // edge: 65,532 B
  EXPECT_FALSE(dsa_topk_logits_fits_lds(64, 128, 64));  // 66,048 B (-> fp8-Q)
  EXPECT_FALSE(dsa_topk_logits_fits_lds(32, 128, 128)); // 82,560 B (larger block)
}

// Whether the indexer shape fits the fp8-Q kernel (Q staged RAW, dequantised
// on the fly in the dot loop with the SAME fp8e4m3fnuz_to_f32 helper -- so
// the dequanted Q values and the output are bit-identical to the fp32-Q
// kernel), the launcher's FALLBACK for shapes FitsLdsCap refuses:
// (H + B*D + B) * 4 + H*D B. GLM-5.3 2x (H=64, D=128, B=64) stages 41,472 B
// and is VERIFIED on a CSCS beverin node
// (meta/benchmarks/test_dsa_topk_correct.hip); the fp8-Q cap admits up to
// H=246 at D=128, B=64 (shapes fitting neither variant are refused).
TEST(DsaTopk, FitsLdsFp8q) {
  EXPECT_TRUE(dsa_topk_logits_fits_lds_fp8q(64, 128, 64));    // GLM-5.3 2x: 41,472 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds_fp8q(128, 128, 64));   // 4x: 49,920 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds_fp8q(246, 128, 64));   // edge: 65,496 B
  EXPECT_FALSE(dsa_topk_logits_fits_lds_fp8q(247, 128, 64));  // 65,628 B (just over)
  EXPECT_FALSE(dsa_topk_logits_fits_lds_fp8q(32, 128, 128));  // 70,272 B (neither fits)
  EXPECT_TRUE(dsa_topk_logits_fits_lds_fp8q(2, 8, 4));        // tiny: 168 B
}

// dsa_topk_logits_fits_lds_mfma: the bf16-MFMA variant stages Q
// transposed as bf16 sQt[D][H] ONCE, one bf16 K-tile sK[B][kBK=64], the
// gate sGate[H] and scales sKscale[B] as fp32 ->
//   (D*H + B*64)*2 + (H + B)*4   bytes
// the SMALLEST of the three variants at every GLM-5.3 width (16,768 / 25,088 /
// 41,728 B at H=32/64/128). The verified 16x16x16bf16_1k fragment needs exact
// multiples, so this is FALSE unless H%16==0, D%64==0 AND B%16==0.
TEST(DsaTopk, FitsLdsMfma) {
  EXPECT_TRUE(dsa_topk_logits_fits_lds_mfma(32, 128, 64));    // GLM-5.3: 16,768 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds_mfma(64, 128, 64));    // 2x: 25,088 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds_mfma(128, 128, 64));   // 4x: 41,728 B
  EXPECT_TRUE(dsa_topk_logits_fits_lds_mfma(208, 128, 64));   // edge H: 62,528 B
  EXPECT_FALSE(dsa_topk_logits_fits_lds_mfma(224, 128, 64));  // 66,688 B (1st mult-16 over)
  // Shape constraints -- the verified fragment needs exact multiples.
  EXPECT_FALSE(dsa_topk_logits_fits_lds_mfma(246, 128, 64));  // H not mult of 16
  EXPECT_FALSE(dsa_topk_logits_fits_lds_mfma(32, 130, 64));   // D not mult of 64
  EXPECT_FALSE(dsa_topk_logits_fits_lds_mfma(32, 128, 72));   // B not mult of 16
  // Bigger B is fine for the MFMA kernel (32,128,128 stages 25,216 B) --
  // it fits where the fp32-Q (82,560 B) and fp8-Q (70,272 B) BOTH refuse.
  EXPECT_TRUE(dsa_topk_logits_fits_lds_mfma(32, 128, 128));   // 25,216 B (fits!)
  // Same SMALLEST-footprint win at the tiny end (D must be >= 64).
  EXPECT_FALSE(dsa_topk_logits_fits_lds_mfma(2, 8, 4));       // D=8 not mult of 64
}

// dsa_topk_logits_split_for: optimal split_kv = max(1, min(ceildiv(msl,B),
// 228/bs)) -- the single source of truth the hip_capi.hpp docstring used
// to restate (with the wrong NUM_CU=256).
TEST(DsaTopk, SplitFor) {
  EXPECT_EQ(dsa_topk_logits_split_for(1, 4096, 64), 64);   // 64 pages, 228/1
  EXPECT_EQ(dsa_topk_logits_split_for(1, 1024, 64), 16);   // 16 pages, 228/1
  EXPECT_EQ(dsa_topk_logits_split_for(64, 4096, 64), 3);  // min(64, 228/64)
  EXPECT_EQ(dsa_topk_logits_split_for(228, 4096, 64), 1); // min(64, 228/228)
  EXPECT_EQ(dsa_topk_logits_split_for(0, 4096, 64), 1);   // empty -> safe 1
  EXPECT_EQ(dsa_topk_logits_split_for(1, 0, 64), 1);     // zero msl -> 1
  EXPECT_EQ(dsa_topk_logits_split_for(1, 4096, 0), 1);   // zero block -> 1
  // bs=2: each split handles ceildiv(4096,64)/sp pages across 2 batches.
  EXPECT_EQ(dsa_topk_logits_split_for(2, 4096, 64), 64);  // min(64, 228/2)
}
