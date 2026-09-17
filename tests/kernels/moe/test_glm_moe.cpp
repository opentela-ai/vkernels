// tests/kernels/moe/test_glm_moe.cpp
//
// Host tests for the GLM-5.3-Flash block-FP8 expert oracle (issue #64):
// exhaustive E4M3FN decode (normals both exponent-sign branches,
// subnormals, signed zeros, the reserved NaN encodings), a hand-checked
// multi-block GEMV that pins the scale indexing, a bit-exact cross-check
// against an independently-written fp32 reference over random codes, the
// M=2 path, and NaN propagation through the accumulation.
#include "minitest.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/glm_moe.hpp"

using vkernels::kernels::glm_e4m3_to_f32_cpu;
using vkernels::kernels::glm_fp8_block_gemv_cpu;

namespace {

// Independent bf16 helpers (RNE), not the kernel's own.
float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}
uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

// Independent E4M3FN decode written from the format contract (1 sign, 4
// exponent bits bias 7, 3 mantissa bits; e=15/m=7 is the only NaN).
float ref_e4m3(uint8_t v) {
  const bool sign = (v >> 7) != 0;
  const uint32_t e = (v >> 3) & 0xF;
  const uint32_t m = v & 0x7;
  if (e == 15 && m == 7) return std::nanf("");
  float val;
  if (e == 0) {
    val = static_cast<float>(m) / 8.0f * 0.015625f;  // 2^-6 * m/8
  } else {
    val = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, static_cast<int>(e) - 7);
  }
  return sign ? -val : val;
}

// Independent GEMV reference with the identical contract: fp32(x) *
// scale[n/128][k/128] * fp32(e4m3(w[n][k])), fp32 accumulate in k order,
// one RNE to bf16 at the end.
void ref_gemv(int M, int N, int K, const std::vector<uint16_t>& x,
              const std::vector<uint8_t>& w, const std::vector<float>& scales,
              std::vector<uint16_t>& out) {
  const int kb = K / 128;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k)
        acc += bf16_to_f32(x[static_cast<size_t>(m) * K + k]) *
               scales[static_cast<size_t>(n / 128) * kb + (k / 128)] *
               ref_e4m3(w[static_cast<size_t>(n) * K + k]);
      out[static_cast<size_t>(m) * N + n] = f32_to_bf16(acc);
    }
}

uint16_t bf16(float v) { return f32_to_bf16(v); }

}  // namespace

TEST(GlmE4m3, DecodeMatchesIndependentReferenceExhaustively) {
  // Every one of the 256 encodings except the two reserved NaN codes must
  // decode bit-identically to the independent reference (which exercises
  // both the e-10 >= 0 and e-10 < 0 branches, subnormals, and both signs).
  int bad = 0;
  for (int v = 0; v < 256; ++v) {
    const uint8_t code = static_cast<uint8_t>(v);
    if (code == 0x7F || code == 0xFF) continue;  // reserved NaN encodings
    if (glm_e4m3_to_f32_cpu(code) != ref_e4m3(code)) ++bad;
  }
  EXPECT_EQ(bad, 0);
}

TEST(GlmE4m3, HandCheckedValues) {
  // +0 / -0 (subnormal, m=0)
  EXPECT_EQ(glm_e4m3_to_f32_cpu(0x00), 0.0f);
  EXPECT_TRUE(std::signbit(glm_e4m3_to_f32_cpu(0x80)));
  // subnormals: 2^-6 * m/8
  EXPECT_NEAR(glm_e4m3_to_f32_cpu(0x01), 1.0f / 512.0f, 1e-12);
  EXPECT_NEAR(glm_e4m3_to_f32_cpu(0x07), 7.0f / 512.0f, 1e-12);
  EXPECT_NEAR(glm_e4m3_to_f32_cpu(0x81), -1.0f / 512.0f, 1e-12);
  // smallest normal (e=1): (8+0) * 2^-9
  EXPECT_NEAR(glm_e4m3_to_f32_cpu(0x08), 0.015625f, 1e-12);
  // mid normal (e=8): (8+0) * 2^1
  EXPECT_EQ(glm_e4m3_to_f32_cpu(0x40), 2.0f);
  EXPECT_EQ(glm_e4m3_to_f32_cpu(0xC0), -2.0f);
  // max finite (e=15, m=6): (8+6) * 2^5 = 448
  EXPECT_EQ(glm_e4m3_to_f32_cpu(0x7E), 448.0f);
  EXPECT_EQ(glm_e4m3_to_f32_cpu(0xFE), -448.0f);
  // reserved NaN encodings (0x7F/0xFF) decode to NaN
  EXPECT_TRUE(std::isnan(glm_e4m3_to_f32_cpu(0x7F)));
  EXPECT_TRUE(std::isnan(glm_e4m3_to_f32_cpu(0xFF)));
}

TEST(GlmFp8Gemv, HandCheckedMultiBlockScaleIndexing) {
  // N=256, K=256 (2x2 scale blocks). Nonzero taps at k=0 and k=128 only;
  // every block's scale is distinct so wrong (nb, kb) indexing shows up.
  const int M = 1, N = 256, K = 256;
  std::vector<uint16_t> x(static_cast<size_t>(M) * K, 0);
  std::vector<uint8_t> w(static_cast<size_t>(N) * K, 0x00);
  std::vector<float> scales(4);
  scales[0] = 3.0f;   // nb=0, kb=0
  scales[1] = 5.0f;   // nb=0, kb=1
  scales[2] = 7.0f;   // nb=1, kb=0
  scales[3] = 11.0f;  // nb=1, kb=1
  x[0] = bf16(1.0f);
  x[128] = bf16(2.0f);
  for (int n = 0; n < N; ++n) {
    w[static_cast<size_t>(n) * K + 0] = 0x40;    // 2.0
    w[static_cast<size_t>(n) * K + 128] = 0x40;  // 2.0
  }
  std::vector<uint16_t> out(static_cast<size_t>(M) * N, 0);
  glm_fp8_block_gemv_cpu(x.data(), w.data(), scales.data(), out.data(), M, N, K);
  // n=0 uses scales[0][0]=3 (k<128) and scales[0][1]=5 (k>=128):
  //   1.0*3*2.0 + 2.0*5*2.0 = 6 + 20 = 26
  EXPECT_EQ(bf16_to_f32(out[0]), 26.0f);
  // n=128 uses scales[1][0]=7 and scales[1][1]=11:
  //   1.0*7*2.0 + 2.0*11*2.0 = 14 + 44 = 58
  EXPECT_EQ(bf16_to_f32(out[128]), 58.0f);
}

TEST(GlmFp8Gemv, BitExactAgainstIndependentReferenceRandomCodes) {
  // Random bf16 activations, random finite E4M3FN codes (the reserved NaN
  // encodings excluded), random per-block scales; the oracle must be
  // bit-exact against the independent fp32 reference (same accumulate
  // order and precision) across contract shapes incl. M=2.
  struct Shape {
    int M, N, K;
  };
  const Shape shapes[] = {
      {1, 128, 128}, {2, 128, 128}, {1, 256, 128}, {2, 256, 256}, {1, 384, 512},
  };
  std::mt19937 rng(6424);
  for (const auto& s : shapes) {
    std::vector<uint16_t> x(static_cast<size_t>(s.M) * s.K);
    std::vector<uint8_t> w(static_cast<size_t>(s.N) * s.K);
    std::vector<float> scales(static_cast<size_t>(s.N / 128) * (s.K / 128));
    for (auto& v : x) {
      float f = static_cast<float>(rng() % 4000) / 2000.0f - 1.0f;  // [-1, 1)
      v = bf16(f);
    }
    for (auto& c : w) {
      uint8_t code = static_cast<uint8_t>(rng() % 256);
      if (code == 0x7F || code == 0xFF) code = 0x40;  // skip reserved NaNs
      c = code;
    }
    for (auto& sc : scales) sc = static_cast<float>(rng() % 1000) / 500.0f;  // [0, 2)
    std::vector<uint16_t> out(static_cast<size_t>(s.M) * s.N, 0);
    std::vector<uint16_t> expected(static_cast<size_t>(s.M) * s.N, 0);
    glm_fp8_block_gemv_cpu(x.data(), w.data(), scales.data(), out.data(), s.M, s.N, s.K);
    ref_gemv(s.M, s.N, s.K, x, w, scales, expected);
    int bad = 0;
    for (size_t i = 0; i < out.size(); ++i)
      if (out[i] != expected[i]) ++bad;
    EXPECT_EQ(bad, 0);
  }
}

TEST(GlmFp8Gemv, NanWeightPoisonsOutput) {
  // The reserved NaN encodings decode to NaN on the CPU oracle and the NaN
  // must propagate through the accumulation to the bf16 output.
  const int M = 1, N = 128, K = 128;
  std::vector<uint16_t> x(K, bf16(1.0f));
  std::vector<uint8_t> w(static_cast<size_t>(N) * K, 0x00);
  std::vector<float> scales(1, 1.0f);
  w[0] = 0x7F;   // +NaN
  w[1] = 0xFF;   // -NaN (same payload on decode)
  std::vector<uint16_t> out(N, 0);
  glm_fp8_block_gemv_cpu(x.data(), w.data(), scales.data(), out.data(), M, N, K);
  uint32_t bits;
  std::memcpy(&bits, &out[0], 4);
  EXPECT_TRUE(std::isnan(bf16_to_f32(out[0])));
  // Rows without a NaN weight stay finite even in the same launch.
  EXPECT_FALSE(std::isnan(bf16_to_f32(out[2])));
}
