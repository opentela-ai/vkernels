// vkernels/kernels/gemm.cpp — CPU reference (oracle) implementation.
#include "vkernels/kernels/gemm.hpp"

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

void gemm(std::size_t M, std::size_t N, std::size_t K, float alpha,
          Span<const float> A, Span<const float> B, float beta, Span<float> C) {
  VK_EXPECTS(A.size() == M * K, "A must be M*K");
  VK_EXPECTS(B.size() == K * N, "B must be K*N");
  VK_EXPECTS(C.size() == M * N, "C must be M*N");

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = alpha * acc + beta * C[i * N + j];
    }
  }
}

}  // namespace vkernels::kernels
