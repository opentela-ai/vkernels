// vkernels/kernels/glm_moe.cpp
//
// CPU reference (oracle) for the GLM block-FP8 expert primitives
// (issue #64). See glm_moe.hpp for the format contract.

#include "vkernels/core/tuning.hpp"
#include "vkernels/kernels/glm_moe.hpp"

#include <cstring>

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

int gemv_pick_sk(int N, int K) {
  // Tuning-store override (docs/tuning-cache.md): a measured sweep on this
  // device arch (bench_glm_fp8_gemv --persist) beats the compiled-in
  // formula below. The record is validated against the with_scratch
  // contract (sk in {1,2,4,8} dividing K/128) before it is trusted.
  if (core::tuning::enabled()) {
    if (const auto* entry =
            core::tuning::find("glm_fp8_gemv_pick_sk", {N, K}))
      if (const long long* sk = entry->find("sk"))
        if (*sk >= 1 && *sk <= 8 && (K / 128) % *sk == 0)
          return static_cast<int>(*sk);
  }
  // Fill MI300A's 228 CUs with ~4 blocks each where the segment budget
  // allows, within {1,2,4,8} and divisibility of K/128.
  int sk = (912 * 32 + N - 1) / N;
  if (sk < 1) sk = 1;
  if (sk > 8) sk = 8;
  while (sk > 1 && (K / 128) % sk != 0) sk >>= 1;   // must divide K/128
  return sk;
}

int glm_fp8_gemv_pick_sk(int N, int K) { return gemv_pick_sk(N, K); }

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

}  // namespace vkernels::kernels
