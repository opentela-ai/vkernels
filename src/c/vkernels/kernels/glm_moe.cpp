// vkernels/kernels/glm_moe.cpp
//
// CPU reference (oracle) for the GLM block-FP8 expert primitives
// (issue #64). See glm_moe.hpp for the format contract.

#include "vkernels/kernels/glm_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace vkernels::kernels {

namespace {

float bf16_to_f32_cpu(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t f32_to_bf16_cpu(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  // round to nearest even
  uint32_t rounded = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
  return (uint16_t)rounded;
}

}  // namespace

float glm_e4m3_to_f32_cpu(uint8_t v) {
  const uint32_t sign = (uint32_t)(v >> 7) & 1u;
  const uint32_t exp = (uint32_t)(v >> 3) & 15u;
  const uint32_t man = (uint32_t)v & 7u;
  if (exp == 15 && man == 7) {                      // the only NaN encoding
    const uint32_t bits = (sign << 31) | 0x7FC00000u;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  float value;
  if (exp == 0) {
    // subnormal: 2^-6 * (m/8), m=0 -> +0/-0
    value = (float)man * (1.0f / 512.0f);
  } else {
    // (1 + m/8) * 2^(e-7) == (8+m) * 2^(e-10)
    value = (float)(8u + man);
    const int e = (int)exp - 10;
    value = e >= 0 ? value * (float)(1ull << e)
                   : value / (float)(1ull << (uint32_t)(-e));
  }
  return sign ? -value : value;
}

void glm_fp8_block_gemv_cpu(const uint16_t* x, const uint8_t* w,
                            const float* scales, uint16_t* out,
                            int M, int N, int K) {
  const int kb = K / 128;                        // scale columns
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      const uint8_t* wrow = w + (size_t)n * K;
      for (int k = 0; k < K; ++k)
        acc += bf16_to_f32_cpu(x[(size_t)m * K + k]) *
               scales[(size_t)(n / 128) * kb + (k / 128)] *
               glm_e4m3_to_f32_cpu(wrow[k]);
      out[(size_t)m * N + n] = f32_to_bf16_cpu(acc);
    }
}

float glm_fnuz_e4m3_to_f32_cpu(uint8_t v) {
  // e4m3fnuz (gfx942-native fp8, issue #148): bias 8, no -0 (0x80 is the
  // ONLY NaN), subnormals m * 2^-10. The hardware v_cvt_pk_f32_fp8
  // implements exactly this table.
  if (v == 0x80u) {
    const uint32_t bits = 0x7FC00000u;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
  }
  const uint32_t sign = (uint32_t)(v >> 7) & 1u;
  const uint32_t exp = (uint32_t)(v >> 3) & 15u;
  const uint32_t man = (uint32_t)v & 7u;
  float value;
  if (exp == 0) {
    value = (float)man * (1.0f / 1024.0f);  // subnormal: m * 2^-10
  } else {
    // (1 + m/8) * 2^(e-8) == (8+m) * 2^(e-11)
    value = (float)(8u + man);
    const int e = (int)exp - 11;
    value = e >= 0 ? value * (float)(1ull << e)
                   : value / (float)(1ull << (uint32_t)(-e));
  }
  return sign ? -value : value;
}

void glm_fp8_fnuz_block_gemv_cpu(const uint16_t* x, const uint8_t* w,
                                 const float* scales, uint16_t* out,
                                 int M, int N, int K) {
  // FNUZ storage (issue #148): value = fnuz(code) * s2 with s2 the DOUBLED
  // scale (payload-halved encoding, e4m3fn_to_fnuz_inplace convention).
  const int kb = K / 128;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      const uint8_t* wrow = w + (size_t)n * K;
      for (int k = 0; k < K; ++k)
        acc += bf16_to_f32_cpu(x[(size_t)m * K + k]) *
               scales[(size_t)(n / 128) * kb + (k / 128)] *
               glm_fnuz_e4m3_to_f32_cpu(wrow[k]);
      out[(size_t)m * N + n] = f32_to_bf16_cpu(acc);
    }
}

uint8_t glm_fp8_fnuz_quant_cpu(float v, float inv) {
  // fp32 product first (the hardware multiplies by the fp32 reciprocal,
  // not a fp32 divide — bit-exact match requires mul), then RNE to the
  // e4m3fnuz lattice: nearest value with ties-to-even mantissa, saturate
  // at +-240 (the fnuz max; 0x80 is the only NaN code).
  const float p = v * inv;
  const float target = std::fabs(p);
  const uint8_t sign = (p < 0.0f) ? 0x80u : 0x00u;
  if (target >= 240.0f) return sign | 0x7fu;
  uint8_t best = 0;
  float bestd = HUGE_VALF;
  for (int code = 0; code < 256; ++code) {
    if (code == 0x80) continue;
    const float vdec = std::fabs(glm_fnuz_e4m3_to_f32_cpu((uint8_t)code));
    float d = vdec - target;
    if (d < 0.0f) d = -d;
    // ties: even mantissa wins (probe-confirmed hardware RNE)
    const bool even = ((code & 0x07u) & 1u) == 0u;
    if (d < bestd || (d == bestd && even)) {
      bestd = d;
      best = (uint8_t)code;
    }
  }
  return sign | best;
}

void glm_fp8_fnuz_mfma_oracle_cpu(const uint16_t* x, const uint8_t* w,
                                 const float* scales, uint16_t* out,
                                 int M, int N, int K) {
  // Models the fused fp8 payload quantize of
  // glm_fp8_fnuz_mfma_gemv_kernel: per (token, 128-k group) amax ->
  // s2a = amax/224 (floor 1e-4), payload = RNE(x/s2a) fnuz codes, then
  // out = payload x fnuz(w) x scales, fp32 accumulation and one bf16
  // round per output — identical numeric recipe to the kernel modulo
  // dot-product ordering.
  const int kb = K / 128;
  for (int m = 0; m < M; ++m) {
    // quantized payload + its (token, group) scale, per kernel recipe
    std::vector<uint8_t> pcm(K);
    std::vector<float> s2a(K);
    for (int g = 0; g < kb; ++g) {
      float amax = 0.0f;
      for (int j = 0; j < 128; ++j) {
        const float xv =
            bf16_to_f32_cpu(x[(size_t)m * K + g * 128 + j]);
        amax = std::fmax(amax, std::fabs(xv));
      }
      const float sg = std::fmax(amax, 1e-4f) / 224.0f;
      for (int j = 0; j < 128; ++j) {
        const float xv =
            bf16_to_f32_cpu(x[(size_t)m * K + g * 128 + j]);
        pcm[g * 128 + j] = glm_fp8_fnuz_quant_cpu(xv, 1.0f / sg);
        s2a[g * 128 + j] = sg;
      }
    }
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      const uint8_t* wrow = w + (size_t)n * K;
      for (int k = 0; k < K; ++k) {
        const uint8_t pc = pcm[k];
        if (pc == 0) continue;
        acc += glm_fnuz_e4m3_to_f32_cpu(pc) * s2a[(k / 128)] *
               scales[(size_t)(n / 128) * kb + (k / 128)] *
               glm_fnuz_e4m3_to_f32_cpu(wrow[k]);
      }
      out[(size_t)m * N + n] = f32_to_bf16_cpu(acc);
    }
  }
}

}  // namespace vkernels::kernels
