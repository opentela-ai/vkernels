// vkernels/kernels/gemm_grouped.cpp — CPU reference (oracle) implementation.
#include "vkernels/kernels/gemm_grouped.hpp"

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

void m_grouped_gemm_nt_masked(std::size_t G, std::size_t M, std::size_t N,
                              std::size_t K, Span<const float> A,
                              Span<const float> B, Span<const int> masked_m,
                              Span<float> C) {
  VK_EXPECTS(A.size() == G * M * K, "A must be G*M*K");
  VK_EXPECTS(B.size() == G * N * K, "B must be G*N*K");
  VK_EXPECTS(masked_m.size() == G, "masked_m must be G");
  VK_EXPECTS(C.size() == G * M * N, "C must be G*M*N");

  for (std::size_t g = 0; g < G; ++g) {
    const int valid = masked_m[g];
    VK_EXPECTS(valid >= 0 && static_cast<std::size_t>(valid) <= M,
               "masked_m must be within [0, M]");
    for (std::size_t m = 0; m < static_cast<std::size_t>(valid); ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        float acc = 0.0f;
        for (std::size_t k = 0; k < K; ++k) {
          acc += A[(g * M + m) * K + k] * B[(g * N + n) * K + k];
        }
        C[(g * M + m) * N + n] = acc;
      }
    }
    // Rows at or beyond masked_m[g] stay untouched (graph-decode contract).
  }
}

}  // namespace vkernels::kernels
