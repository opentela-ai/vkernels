// vkernels/kernels/mhc.cpp -- CPU reference (oracle) implementation.
//
// MHC multi-head hybrid-attention pre-norm (issue #51, part 2). This is the
// CPU/torch reference the gfx942 HIP kernels are checked against; it is
// always compiled (no GPU toolkit) and is the correctness oracle for both the
// host unit tests and the on-device `test_mhc_correct.hip` harness.
//
// See mhc.hpp for the layouts. Two operations, mirroring the two
// tilelang kernels that fault/abort on gfx942:
//
//   mhc_pre_gemm_sqrsum_cpu : pre-norm GEMM (x @ fn^T) + per-token squared-sum
//   mhc_post_cpu            : post-attention combine
//
// The math is exactly the scalar form documented inline in sglang's
// `mhc.py` (`mhc_pre_gemm_sqrsum_tilelang`, `mhc_post_tilelang`).
#include "vkernels/kernels/mhc.hpp"

#include <cstddef>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

void mhc_pre_gemm_sqrsum_cpu(int num_tokens, int hc_mult, int hidden_size,
                             const float* x, const float* fn,
                             float* out, float* sqrsum) {
  const int hc_hidden_size = hc_mult * hidden_size;
  const int hc_mult3 = hc_mult * (2 + hc_mult);
  VK_EXPECTS(hc_mult > 0 && hidden_size > 0, "hc_mult and hidden_size must be positive");
  VK_EXPECTS(hc_mult3 <= 32, "hc_mult*(2+hc_mult) must be <= 32");
  VK_EXPECTS(num_tokens == 0 || x != nullptr, "x must not be null");
  VK_EXPECTS(num_tokens == 0 || fn != nullptr, "fn must not be null");
  VK_EXPECTS(num_tokens == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(num_tokens == 0 || sqrsum != nullptr, "sqrsum must not be null");

  for (int n = 0; n < num_tokens; ++n) {
    const float* xn = x + (size_t)n * hc_hidden_size;
    float sq = 0.0f;
    for (int h = 0; h < hc_hidden_size; ++h) sq += xn[h] * xn[h];
    sqrsum[n] = sq;
    float* on = out + (size_t)n * hc_mult3;
    for (int o = 0; o < hc_mult3; ++o) {
      const float* fo = fn + (size_t)o * hc_hidden_size;
      float acc = 0.0f;
      for (int h = 0; h < hc_hidden_size; ++h) acc += xn[h] * fo[h];
      on[o] = acc;
    }
  }
}

void mhc_post_cpu(int num_tokens, int hc, int hidden,
                  const float* a, const float* b, const float* c,
                  const float* d, float* out) {
  VK_EXPECTS(hc > 0 && hidden > 0, "hc and hidden must be positive");
  VK_EXPECTS(num_tokens == 0 || a != nullptr, "a must not be null");
  VK_EXPECTS(num_tokens == 0 || b != nullptr, "b must not be null");
  VK_EXPECTS(num_tokens == 0 || c != nullptr, "c must not be null");
  VK_EXPECTS(num_tokens == 0 || d != nullptr, "d must not be null");
  VK_EXPECTS(num_tokens == 0 || out != nullptr, "out must not be null");

  for (int n = 0; n < num_tokens; ++n) {
    const float* an = a + (size_t)n * hc * hc;        // [hc, hc]
    const float* bn = b + (size_t)n * hc * hidden;    // [hc, hidden]
    const float* cn = c + (size_t)n * hc;             // [hc]
    const float* dn = d + (size_t)n * hidden;         // [hidden]
    float* on = out + (size_t)n * hc * hidden;        // [hc, hidden]
    for (int j = 0; j < hc; ++j) {
      const float cj = cn[j];
      const float* aj = an + (size_t)j;               // a[n, :, j] (stride hc)
      float* oj = on + (size_t)j * hidden;            // out[n, j, :]
      for (int h = 0; h < hidden; ++h) {
        float acc = cj * dn[h];
        for (int k = 0; k < hc; ++k)
          acc += aj[(size_t)k * hc] * bn[(size_t)k * hidden + h];
        oj[h] = acc;
      }
    }
  }
}

}  // namespace vkernels::kernels
