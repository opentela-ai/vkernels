// tests/kernels/attn/test_mhc.cpp
//
// Host tests for the MHC multi-head hybrid-attention pre-norm kernels
// (issue #51, part 2): hand-checked cases for both
// `mhc_pre_gemm_sqrsum_cpu` (the pre-norm GEMM + squared-sum whose tilelang
// split-k stage-0 JIT-aborts on gfx942 with the "98304 exceeds device limit
// 65536" dynamic-shared error) and `mhc_post_cpu` (the post-attention
// combine), plus independent-reference sweeps and the null-arg / empty
// no-op edges.
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/mhc.hpp"

using vkernels::kernels::mhc_post_cpu;
using vkernels::kernels::mhc_pre_gemm_sqrsum_cpu;

namespace {

// Independent reference for mhc_pre_gemm_sqrsum, computed with a different
// accumulation order (inner-product helper) than the oracle.
float dot(const float* a, const float* b, int n) {
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

void ref_pre(const std::vector<float>& x, const std::vector<float>& fn,
             int num_tokens, int hc_hidden_size, int hc_mult3,
             std::vector<float>& out, std::vector<float>& sqrsum) {
  for (int n = 0; n < num_tokens; ++n) {
    const float* xn = x.data() + (size_t)n * hc_hidden_size;
    float sq = 0.0f;
    for (int h = 0; h < hc_hidden_size; ++h) sq += xn[h] * xn[h];
    sqrsum[n] = sq;
    for (int o = 0; o < hc_mult3; ++o) {
      const float* fo = fn.data() + (size_t)o * hc_hidden_size;
      out[(size_t)n * hc_mult3 + o] = dot(xn, fo, hc_hidden_size);
    }
  }
}

// Independent reference for mhc_post, computed in (h, k) order rather than
// the oracle's (k, h) order, so a transposition bug would diverge.
void ref_post(const std::vector<float>& a, const std::vector<float>& b,
              const std::vector<float>& c, const std::vector<float>& d,
              int num_tokens, int hc, int hidden, std::vector<float>& out) {
  for (int n = 0; n < num_tokens; ++n) {
    const float* an = a.data() + (size_t)n * hc * hc;    // [hc, hc]
    const float* bn = b.data() + (size_t)n * hc * hidden;  // [hc, hidden]
    const float* cn = c.data() + (size_t)n * hc;
    const float* dn = d.data() + (size_t)n * hidden;
    float* on = out.data() + (size_t)n * hc * hidden;
    for (int j = 0; j < hc; ++j) {
      const float cj = cn[j];
      for (int h = 0; h < hidden; ++h) {
        float s = cj * dn[h];
        for (int k = 0; k < hc; ++k) s += an[(size_t)k * hc + j] * bn[(size_t)k * hidden + h];
        on[(size_t)j * hidden + h] = s;
      }
    }
  }
}

}  // namespace

// ---- mhc_pre_gemm_sqrsum: hand-checked -----------------------------------

// hc_mult=2, hidden=2 -> hc_mult3 = 2*(2+2) = 8, hc_hidden_size = 4.
// fn is identity in the first 4 rows, zero in the last 4 (so hc_mult3 >
// hc_hidden_size is exercised). x = [1,2,3,4] -> out = [1,2,3,4,0,0,0,0],
// sqrsum = 1+4+9+16 = 30.
TEST(MhcPreGemmSqrsum, HandCheckedIdentityExtended) {
  const int hc_mult = 2, hidden = 2;
  const int hc_hidden = hc_mult * hidden;   // 4
  const int hc_mult3 = hc_mult * (2 + hc_mult);  // 8
  const int num_tokens = 1;
  std::vector<float> x = {1, 2, 3, 4};
  std::vector<float> fn((size_t)hc_mult3 * hc_hidden, 0.0f);
  for (int o = 0; o < hc_hidden; ++o) fn[(size_t)o * hc_hidden + o] = 1.0f;
  std::vector<float> out((size_t)num_tokens * hc_mult3, -1.0f);
  std::vector<float> sqrsum(num_tokens, -1.0f);
  mhc_pre_gemm_sqrsum_cpu(num_tokens, hc_mult, hidden, x.data(), fn.data(),
                          out.data(), sqrsum.data());
  EXPECT_EQ(out[0], 1.0f); EXPECT_EQ(out[1], 2.0f);
  EXPECT_EQ(out[2], 3.0f); EXPECT_EQ(out[3], 4.0f);
  EXPECT_EQ(out[4], 0.0f); EXPECT_EQ(out[5], 0.0f);
  EXPECT_EQ(out[6], 0.0f); EXPECT_EQ(out[7], 0.0f);
  EXPECT_EQ(sqrsum[0], 30.0f);
}

// Constant-row fn: out[n, o] = (o+1) * sum(x[n,:]).
TEST(MhcPreGemmSqrsum, HandCheckedConstantRows) {
  const int hc_mult = 3, hidden = 2;
  const int hc_hidden = hc_mult * hidden;        // 6
  const int hc_mult3 = hc_mult * (2 + hc_mult);  // 15
  const int num_tokens = 2;
  std::vector<float> x = {1, 2, 3, 4, 5, 6,   // token 0, sum = 21
                          0, -1, 2, -3, 4, 0}; // token 1, sum = 2
  std::vector<float> fn((size_t)hc_mult3 * hc_hidden, 0.0f);
  for (int o = 0; o < hc_mult3; ++o)
    for (int h = 0; h < hc_hidden; ++h) fn[(size_t)o * hc_hidden + h] = float(o + 1);
  std::vector<float> out((size_t)num_tokens * hc_mult3, -1.0f);
  std::vector<float> sqrsum(num_tokens, -1.0f);
  mhc_pre_gemm_sqrsum_cpu(num_tokens, hc_mult, hidden, x.data(), fn.data(),
                          out.data(), sqrsum.data());
  for (int o = 0; o < hc_mult3; ++o) {
    EXPECT_NEAR(out[(size_t)0 * hc_mult3 + o], float(o + 1) * 21.0f, 1e-4f);
    EXPECT_NEAR(out[(size_t)1 * hc_mult3 + o], float(o + 1) * 2.0f, 1e-4f);
  }
  // sqrsum: token0 = 1+4+9+16+25+36 = 91; token1 = 0+1+4+9+16+0 = 30.
  EXPECT_NEAR(sqrsum[0], 91.0f, 1e-4f);
  EXPECT_NEAR(sqrsum[1], 30.0f, 1e-4f);
}

// ---- mhc_post: hand-checked ----------------------------------------------

// num_tokens=1, hc=2, hidden=2. a = identity (2x2), b = [[1,2],[3,4]],
// c = [10, 20], d = [1, 1].
//   out[0,0,:] = [10*1 + 1*1+0*3, 10*1 + 1*2+0*4] = [11, 12]
//   out[0,1,:] = [20*1 + 0*1+1*3, 20*1 + 0*2+1*4] = [23, 24]
TEST(MhcPost, HandCheckedIdentityA) {
  const int num_tokens = 1, hc = 2, hidden = 2;
  std::vector<float> a = {1, 0,   // a[0,0,:]=a[:,0]
                          0, 1};  // a[0,1,:]=a[:,1]
  std::vector<float> b = {1, 2,   // b[0,0,:]
                          3, 4};  // b[0,1,:]
  std::vector<float> c = {10, 20};
  std::vector<float> d = {1, 1};
  std::vector<float> out((size_t)num_tokens * hc * hidden, -1.0f);
  mhc_post_cpu(num_tokens, hc, hidden, a.data(), b.data(), c.data(), d.data(),
               out.data());
  EXPECT_NEAR(out[0], 11.0f, 1e-6f);  // out[0,0,0]
  EXPECT_NEAR(out[1], 12.0f, 1e-6f);  // out[0,0,1]
  EXPECT_NEAR(out[2], 23.0f, 1e-6f);  // out[0,1,0]
  EXPECT_NEAR(out[3], 24.0f, 1e-6f);  // out[0,1,1]
}

// All-zero a -> pure broadcast: out[n,j,h] = c[n,j] * d[n,h].
TEST(MhcPost, ZeroABroadcastsC) {
  const int num_tokens = 1, hc = 2, hidden = 3;
  std::vector<float> a((size_t)hc * hc, 0.0f);
  std::vector<float> b((size_t)hc * hidden, 7.0f);  // ignored when a==0
  std::vector<float> c = {2, 5};
  std::vector<float> d = {10, 100, 1000};
  std::vector<float> out((size_t)num_tokens * hc * hidden, -1.0f);
  mhc_post_cpu(num_tokens, hc, hidden, a.data(), b.data(), c.data(), d.data(),
               out.data());
  // out[0,0,:] = 2 * [10,100,1000] = [20,200,2000]
  EXPECT_NEAR(out[0], 20.0f, 1e-6f); EXPECT_NEAR(out[1], 200.0f, 1e-6f);
  EXPECT_NEAR(out[2], 2000.0f, 1e-6f);
  // out[0,1,:] = 5 * [10,100,1000] = [50,500,5000]
  EXPECT_NEAR(out[3], 50.0f, 1e-6f); EXPECT_NEAR(out[4], 500.0f, 1e-6f);
  EXPECT_NEAR(out[5], 5000.0f, 1e-6f);
}

// ---- independent-reference sweeps ----------------------------------------

TEST(MhcPreGemmSqrsum, MatchesReference) {
  struct Cfg { int num_tokens, hc_mult, hidden; };
  std::vector<Cfg> cfgs = {{1, 2, 2}, {3, 2, 4}, {7, 3, 3}, {2, 4, 4}, {5, 4, 8}};
  std::mt19937 rng(12345);
  for (const auto& c : cfgs) {
    const int hc_hidden = c.hc_mult * c.hidden;
    const int hc_mult3 = c.hc_mult * (2 + c.hc_mult);
    std::vector<float> x((size_t)c.num_tokens * hc_hidden);
    std::vector<float> fn((size_t)hc_mult3 * hc_hidden);
    std::uniform_real_distribution<float> u(-1, 1);
    for (auto& v : x) v = u(rng);
    for (auto& v : fn) v = u(rng);
    std::vector<float> out((size_t)c.num_tokens * hc_mult3);
    std::vector<float> sqrsum(c.num_tokens);
    std::vector<float> rout((size_t)c.num_tokens * hc_mult3);
    std::vector<float> rsqrsum(c.num_tokens);
    mhc_pre_gemm_sqrsum_cpu(c.num_tokens, c.hc_mult, c.hidden, x.data(),
                            fn.data(), out.data(), sqrsum.data());
    ref_pre(x, fn, c.num_tokens, hc_hidden, hc_mult3, rout, rsqrsum);
    float maxd = 0, maxsq = 0;
    for (size_t i = 0; i < out.size(); ++i) maxd = std::fmax(maxd, std::fabs(out[i] - rout[i]));
    for (int i = 0; i < c.num_tokens; ++i) maxsq = std::fmax(maxsq, std::fabs(sqrsum[i] - rsqrsum[i]));
    EXPECT_NEAR(maxd, 0.0f, 1e-4f);
    EXPECT_NEAR(maxsq, 0.0f, 1e-4f);
  }
}

TEST(MhcPost, MatchesReference) {
  struct Cfg { int num_tokens, hc, hidden; };
  std::vector<Cfg> cfgs = {{1, 2, 2}, {2, 3, 4}, {4, 2, 8}, {3, 4, 5}, {2, 4, 16}};
  std::mt19937 rng(98765);
  for (const auto& c : cfgs) {
    std::vector<float> a((size_t)c.num_tokens * c.hc * c.hc);
    std::vector<float> b((size_t)c.num_tokens * c.hc * c.hidden);
    std::vector<float> co((size_t)c.num_tokens * c.hc);
    std::vector<float> d((size_t)c.num_tokens * c.hidden);
    std::uniform_real_distribution<float> u(-1, 1);
    for (auto& v : a) v = u(rng);
    for (auto& v : b) v = u(rng);
    for (auto& v : co) v = u(rng);
    for (auto& v : d) v = u(rng);
    std::vector<float> out((size_t)c.num_tokens * c.hc * c.hidden);
    std::vector<float> rout((size_t)c.num_tokens * c.hc * c.hidden);
    mhc_post_cpu(c.num_tokens, c.hc, c.hidden, a.data(), b.data(), co.data(),
                 d.data(), out.data());
    ref_post(a, b, co, d, c.num_tokens, c.hc, c.hidden, rout);
    float maxd = 0;
    for (size_t i = 0; i < out.size(); ++i) maxd = std::fmax(maxd, std::fabs(out[i] - rout[i]));
    EXPECT_NEAR(maxd, 0.0f, 1e-3f);
  }
}

// ---- edges ----------------------------------------------------------------

TEST(Mhc, EmptyIsNoOp) {
  std::vector<float> x(0), fn(4, 0), out(0), sqrsum(0);
  EXPECT_NO_THROW(mhc_pre_gemm_sqrsum_cpu(0, 2, 2, x.data(), fn.data(),
                                          out.data(), sqrsum.data()));
  std::vector<float> a(0), b(0), c(0), d(0), o(0);
  EXPECT_NO_THROW(mhc_post_cpu(0, 2, 2, a.data(), b.data(), c.data(),
                               d.data(), o.data()));
}
