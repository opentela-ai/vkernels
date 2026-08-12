// tests/kernels/moe/test_moe.cpp
//
// Tests for AMD gfx942 MXFP4 MoE low-level kernel primitives (CPU reference
// implementations and HIP-accelerated paths).
//
// References:
//   #12 — GFX942_SW_LDS_FILL
//   #13 — GFX942_SW_CVT
//   #14 — GFX942_ASYNC_OFF
//   #15 — GFX942_K16_SPLIT
#include "minitest.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "vkernels/kernels/moe.hpp"
#include "vkernels/util/config.hpp"

using vkernels::Span;
using vkernels::kernels::direct_lds_fill_bf16;
using vkernels::kernels::fp4_to_bf16_dequant;
using vkernels::kernels::mfma_f32_16x16x16bf16;
using vkernels::kernels::use_async_copy_default;

// ======================================================================
// #12 — Software direct-to-LDS fill
// ======================================================================

TEST(DirectLdsFill, BasicCopy) {
  std::vector<uint16_t> src = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint16_t> dst(8, 0);
  direct_lds_fill_bf16(dst.data(), src.data(), 8);
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }
}

TEST(DirectLdsFill, EmptyIsOk) {
  std::vector<uint16_t> src, dst;
  direct_lds_fill_bf16(dst.data(), src.data(), 0);
}

TEST(DirectLdsFill, NullWithElementsThrows) {
  EXPECT_THROW(direct_lds_fill_bf16(nullptr, nullptr, 8), std::invalid_argument);
}

TEST(DirectLdsFill, LargeCopy) {
  std::vector<uint16_t> src(1024);
  std::vector<uint16_t> dst(1024, 0);
  for (size_t i = 0; i < src.size(); ++i) src[i] = static_cast<uint16_t>(i);
  direct_lds_fill_bf16(dst.data(), src.data(), 1024);
  for (size_t i = 0; i < src.size(); ++i) {
    EXPECT_EQ(dst[i], src[i]);
  }
}

// ======================================================================
// #13 — Software fp4→bf16 dequant
// ======================================================================

namespace {

// Convert a bf16 value (uint16_t bit pattern) back to float for comparison.
float bf16_to_float(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(float));
  return f;
}

}  // namespace

TEST(Fp4ToBf16Dequant, Zero) {
  // fp4 0b0000 = +0.0
  std::vector<uint8_t> packed = {0x00};
  std::vector<uint16_t> out(2);
  fp4_to_bf16_dequant(packed, out);
  EXPECT_NEAR(bf16_to_float(out[0]), 0.0f, 1e-6);
  EXPECT_NEAR(bf16_to_float(out[1]), 0.0f, 1e-6);
}

TEST(Fp4ToBf16Dequant, NegativeZero) {
  // fp4 0b1000 = -0.0 (sign bit set on zero)
  std::vector<uint8_t> packed = {0x08};
  std::vector<uint16_t> out(2);
  fp4_to_bf16_dequant(packed, out);
  float flo = bf16_to_float(out[0]);
  EXPECT_NEAR(std::fabs(flo), 0.0f, 1e-6);
  EXPECT_TRUE(std::signbit(flo));
}

TEST(Fp4ToBf16Dequant, Subnormal) {
  // fp4 0b0001 = +0.25
  std::vector<uint8_t> packed = {0x01};
  std::vector<uint16_t> out(2);
  fp4_to_bf16_dequant(packed, out);
  EXPECT_NEAR(bf16_to_float(out[0]), 0.25f, 1e-3);
}

TEST(Fp4ToBf16Dequant, NormalValues) {
  // Test all normal representable values.  fp4 E2M1 nibble [s|e1|e0|m]:
  //   +1.0 = 0b0010 (0x2), -1.0 = 0b1010 (0xA)
  //   +1.5 = 0b0011 (0x3), -1.5 = 0b1011 (0xB)
  //   +2.0 = 0b0100 (0x4), -2.0 = 0b1100 (0xC)
  //   +3.0 = 0b0101 (0x5), -3.0 = 0b1101 (0xD)

  struct TestCase {
    uint8_t nibble;
    float expected;
  };
  TestCase cases[] = {
      {0x2, 1.0f},   // +1.0
      {0xA, -1.0f},  // -1.0
      {0x3, 1.5f},   // +1.5
      {0xB, -1.5f},  // -1.5
      {0x4, 2.0f},   // +2.0
      {0xC, -2.0f},  // -2.0
      {0x5, 3.0f},   // +3.0
      {0xD, -3.0f},  // -3.0
  };

  for (auto tc : cases) {
    // Low nibble
    std::vector<uint8_t> packed = {tc.nibble};
    std::vector<uint16_t> out(2);
    fp4_to_bf16_dequant(packed, out);
    EXPECT_NEAR(bf16_to_float(out[0]), tc.expected, 1e-3);

    // High nibble (shifted)
    packed = {static_cast<uint8_t>(tc.nibble << 4)};
    fp4_to_bf16_dequant(packed, out);
    EXPECT_NEAR(bf16_to_float(out[1]), tc.expected, 1e-3);
  }
}

TEST(Fp4ToBf16Dequant, InfAndNaN) {
  // fp4 0b0110 (0x6) = +inf (s=0, e=3, m=0)
  {
    std::vector<uint8_t> packed = {0x06};
    std::vector<uint16_t> out(2);
    fp4_to_bf16_dequant(packed, out);
    float v = bf16_to_float(out[0]);
    EXPECT_TRUE(std::isinf(v));
    EXPECT_TRUE(v > 0);
  }
  // fp4 0b1110 (0xE) = -inf (s=1, e=3, m=0)
  {
    std::vector<uint8_t> packed = {0x0E};
    std::vector<uint16_t> out(2);
    fp4_to_bf16_dequant(packed, out);
    float v = bf16_to_float(out[0]);
    EXPECT_TRUE(std::isinf(v));
    EXPECT_TRUE(v < 0);
  }
  // fp4 0b0111 (0x7) = NaN (s=0, e=3, m=1)
  {
    std::vector<uint8_t> packed = {0x07};
    std::vector<uint16_t> out(2);
    fp4_to_bf16_dequant(packed, out);
    EXPECT_TRUE(std::isnan(bf16_to_float(out[0])));
  }
  // fp4 0b1111 (0xF) = NaN (s=1, e=3, m=1)
  {
    std::vector<uint8_t> packed = {0x0F};
    std::vector<uint16_t> out(2);
    fp4_to_bf16_dequant(packed, out);
    EXPECT_TRUE(std::isnan(bf16_to_float(out[0])));
  }
}

TEST(Fp4ToBf16Dequant, WithScale) {
  // +1.0 × 2.0 = +2.0
  // fp4 +1.0 = 0b0010 = 0x2
  std::vector<uint8_t> packed = {0x02};
  std::vector<uint16_t> out(2);
  fp4_to_bf16_dequant(packed, out, 2.0f);
  EXPECT_NEAR(bf16_to_float(out[0]), 2.0f, 1e-3);
}

TEST(Fp4ToBf16Dequant, TwoValuesPerByte) {
  // low nibble = +3.0 (0101 = 0x5), high nibble = -1.0 (1010 = 0xA)
  uint8_t byte = 0xA5;  // low=5, high=A
  std::vector<uint8_t> packed = {byte};
  std::vector<uint16_t> out(2);
  fp4_to_bf16_dequant(packed, out);
  EXPECT_NEAR(bf16_to_float(out[0]), 3.0f, 1e-3);
  EXPECT_NEAR(bf16_to_float(out[1]), -1.0f, 1e-3);
}

TEST(Fp4ToBf16Dequant, MismatchedOutputThrows) {
  std::vector<uint8_t> packed(2);
  std::vector<uint16_t> out(3);  // should be 4
  EXPECT_THROW(fp4_to_bf16_dequant(packed, out), std::invalid_argument);
}

TEST(Fp4ToBf16Dequant, RoundTrip) {
  // For each representable fp4 value, dequant to bf16, then round-trip
  // back through float and verify we're close to the original float.
  // Skip NaN (no meaningful comparison) and infs.
  for (int nib = 0; nib < 16; ++nib) {
    int s = (nib >> 3) & 1;
    int e = (nib >> 1) & 3;
    int m = nib & 1;

    // Skip NaN and skip the NIbble=0x7/0xF which are NaN
    if (e == 3 && m == 1) continue;
    // Skip the other NaN encoder test

    std::vector<uint8_t> packed = {static_cast<uint8_t>(nib & 0x0F)};
    std::vector<uint16_t> out(2);
    fp4_to_bf16_dequant(packed, out);

    float got = bf16_to_float(out[0]);

    if (e == 0 && m == 0) {
      // ±0
      EXPECT_NEAR(std::fabs(got), 0.0f, 1e-6);
      if (s) EXPECT_TRUE(std::signbit(got));
    } else if (e == 0 && m == 1) {
      EXPECT_NEAR(std::fabs(got), 0.25f, 1e-3);
      if (s) EXPECT_TRUE(got < 0);
    } else if (e == 3 && m == 0) {
      EXPECT_TRUE(std::isinf(got));
      if (s) EXPECT_TRUE(got < 0);
    } else {
      // Normal
      float expected = (1.0f + m * 0.5f) * static_cast<float>(1 << (e - 1));
      if (s) expected = -expected;
      EXPECT_NEAR(got, expected, 1e-3);
    }
  }
}

// ======================================================================
// #14 — Platform async-copy gate
// ======================================================================

TEST(AsyncCopyDefault, ReturnsBool) {
  // On the CPU reference path, always returns true (unless K3_NO_ASYNC=1
  // is set in the environment). On the HIP path, it detects gfx942.
  // We can't assert a specific value without knowing the environment
  // or GPU availability, but the function must not throw.
  bool result = use_async_copy_default();
  (void)result;
  EXPECT_NO_THROW(use_async_copy_default());
}

TEST(AsyncCopyDefault, EnvVarOverride) {
  ::setenv("K3_NO_ASYNC", "1", 1);
  EXPECT_FALSE(use_async_copy_default());  // "1" forces OFF
  ::setenv("K3_NO_ASYNC", "0", 1);
  EXPECT_TRUE(use_async_copy_default());  // anything but "1" → ON
  ::unsetenv("K3_NO_ASYNC");
  EXPECT_TRUE(use_async_copy_default());  // default ON
}

// ======================================================================
// #15 — K16 bf16 MFMA
// ======================================================================

TEST(MfmaK16Bf16, ZeroAccumulate) {
  // C[4] = 0, A and B = two 1.0 bf16 values each → C[i] = 0 + 1.0*1.0 = 1.0
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // bf16(1.0) = 0x3F80 in float → bf16 is 0x3F80
  uint16_t one_bf16 = 0x3F80;
  uint32_t a[2] = {
      one_bf16 | (static_cast<uint32_t>(one_bf16) << 16),  // two 1.0 packed
      one_bf16 | (static_cast<uint32_t>(one_bf16) << 16),
  };
  uint32_t b[2] = {
      one_bf16 | (static_cast<uint32_t>(one_bf16) << 16),
      one_bf16 | (static_cast<uint32_t>(one_bf16) << 16),
  };

  mfma_f32_16x16x16bf16(c, a, b);

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(c[i], 1.0f, 1e-5);
  }
}

TEST(MfmaK16Bf16, Accumulate) {
  // Start with C = [1, -1, 2, 0], multiply by identity-like values
  float c[4] = {1.0f, -1.0f, 2.0f, 0.0f};

  // bf16(2.0) = 0x4000
  uint16_t two_bf16 = 0x4000;
  uint32_t a[2] = {
      two_bf16 | (static_cast<uint32_t>(two_bf16) << 16),
      two_bf16 | (static_cast<uint32_t>(two_bf16) << 16),
  };
  uint32_t b[2] = {
      two_bf16 | (static_cast<uint32_t>(two_bf16) << 16),
      two_bf16 | (static_cast<uint32_t>(two_bf16) << 16),
  };

  mfma_f32_16x16x16bf16(c, a, b);

  // Each C[i] += 2.0 * 2.0 = 4.0
  EXPECT_NEAR(c[0], 5.0f, 1e-5);   //  1 + 4
  EXPECT_NEAR(c[1], 3.0f, 1e-5);   // -1 + 4
  EXPECT_NEAR(c[2], 6.0f, 1e-5);   //  2 + 4
  EXPECT_NEAR(c[3], 4.0f, 1e-5);   //  0 + 4
}

TEST(MfmaK16Bf16, NullPointersThrows) {
  float c[4] = {};
  uint32_t a[2] = {};
  EXPECT_THROW(mfma_f32_16x16x16bf16(nullptr, a, a), std::invalid_argument);
  EXPECT_THROW(mfma_f32_16x16x16bf16(c, nullptr, a), std::invalid_argument);
  EXPECT_THROW(mfma_f32_16x16x16bf16(c, a, nullptr), std::invalid_argument);
}

TEST(MfmaK16Bf16, K32SplitEmulation) {
  // Emulate K32 by doing two K16 calls.
  // For a K=32 MFMA, we'd split A[K=32] into A_lo[K=0..15] and A_hi[K=16..31].
  // Here we use the same values for both halves (identity-like).
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // First K16: A_lo and B_lo (K=0..15)
  // Second K16: A_hi and B_hi (K=16..31) — same values for simplicity
  // Expected: C[i] = a[i]*b[i] + a[i]*b[i] = 2*a[i]*b[i]

  uint16_t one_bf16 = 0x3F80;
  uint32_t ab[2] = {
      one_bf16 | (static_cast<uint32_t>(one_bf16) << 16),
      one_bf16 | (static_cast<uint32_t>(one_bf16) << 16),
  };

  mfma_f32_16x16x16bf16(c, ab, ab);  // K=0..15
  mfma_f32_16x16x16bf16(c, ab, ab);  // K=16..31

  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(c[i], 2.0f, 1e-5);  // 1.0*1.0 + 1.0*1.0 = 2.0
  }
}
