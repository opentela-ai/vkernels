// verifier/bmc/harness_gemm.cpp
//
// Bounded model check of the SGEMM CPU oracle
// (src/c/vkernels/kernels/gemm.cpp).  Proves that, for every nondeterministic
// input, the oracle writes C[i,j] = alpha * sum_k A[i,k]*B[k,j] + beta*C0[i,j]
// (its specified index map) and leaves A and B unmodified.  Run via
// cbmc/run.sh; see verifier/README.md.
#include <cstddef>

#include "verify_support.h"
#include "vkernels/kernels/gemm.hpp"

using vkernels::Span;
namespace vk = vkernels::kernels;

namespace {
constexpr std::size_t M = 2;
constexpr std::size_t N = 2;
constexpr std::size_t K = 2;
}

int main() {
  float a[M * K], b[K * N], c[M * N], c0[M * N];
  float a0[M * K], b0[K * N];
  for (std::size_t i = 0; i < M * K; ++i) a[i] = a0[i] = VK_NONDET_FLOAT();
  for (std::size_t i = 0; i < K * N; ++i) b[i] = b0[i] = VK_NONDET_FLOAT();
  for (std::size_t i = 0; i < M * N; ++i) {
    c[i] = VK_NONDET_FLOAT();
    c0[i] = c[i];
  }

  const float alpha = VK_NONDET_FLOAT();
  const float beta = VK_NONDET_FLOAT();
  vk::gemm(M, N, K, alpha, Span<const float>(a, M * K),
           Span<const float>(b, K * N), beta, Span<float>(c, M * N));

  // C[i,j] == alpha * sum_k A[i,k]*B[k,j] + beta * C0[i,j].
  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k) acc += a[i * K + k] * b[k * N + j];
      VK_ASSERT(vk_feq(c[i * N + j], alpha * acc + beta * c0[i * N + j]),
                "gemm: C == alpha*A@B + beta*C0");
    }
  }

  // Inputs are read-only.  Compare element-wise (the oracle takes
  // Span<const float>, so it cannot write through them, but the proof makes
  // that assumption explicit).
  for (std::size_t i = 0; i < M * K; ++i)
    VK_ASSERT(vk_feq(a[i], a0[i]), "gemm: A unmodified");
  for (std::size_t i = 0; i < K * N; ++i)
    VK_ASSERT(vk_feq(b[i], b0[i]), "gemm: B unmodified");
  return 0;
}
