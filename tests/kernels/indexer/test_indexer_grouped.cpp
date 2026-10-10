// tests/kernels/indexer/test_indexer_grouped.cpp — oracle contracts for the
// DeepGEMM-borrowed kernels: weighted-ReLU MQA logits (lightning indexer
// scoring) and the masked m-grouped GEMM (CUDA-graph MoE decode). Device
// parity runs in test_cuda_contracts.cu on toolkit builds.
#include "vkernels/kernels/gemm_grouped.hpp"
#include "vkernels/kernels/mqa_logits.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "minitest.hpp"

namespace {

std::vector<float> mk(std::size_t n, float value) { return std::vector<float>(n, value); }

}  // namespace

TEST(MqaLogits, WeightedReluOverSpan) {
  // 2 query rows, 3 KV tokens, 2 heads, head dim 2.
  const std::size_t M = 2, N = 3, H = 2, D = 2;
  std::vector<float> q{
      1, 0, 0, 1,   // m0: h0 = (1,0), h1 = (0,1)
      2, 0, 0, -2,  // m1: h0 = (2,0), h1 = (0,-2)
  };
  std::vector<float> kv{
      1, 2,   // n0
      -1, 3,  // n1
      4, 0,   // n2
  };
  std::vector<float> weights{
      0.5f, 2.0f,  // m0
      1.0f, 3.0f,  // m1
  };
  std::vector<int> ks{0, 1};
  std::vector<int> ke{2, 3};  // m0 sees n0..n1; m1 sees n1..n2
  std::vector<float> out(M * N, 0.0f);

  vkernels::kernels::mqa_logits(M, N, H, D, q, kv, weights, ks, ke, out);

  // m0, n0: relu(1)=1*0.5 + relu(2)=2*2 => 4.5
  EXPECT_NEAR(out[0], 4.5f, 1e-6);
  // m0, n1: relu(-1)=0 + relu(3)=3*2 => 6
  EXPECT_NEAR(out[1], 6.0f, 1e-6);
  // m0, n2: outside span
  EXPECT_TRUE(std::isinf(out[2]) && out[2] < 0);
  // m1, n0: outside span
  EXPECT_TRUE(std::isinf(out[3]) && out[3] < 0);
  // m1, n1: relu(-2)=0*1 + relu(-6)=0*3 => 0  (both head dots negative)
  EXPECT_NEAR(out[4], 0.0f, 1e-6);
  // m1, n2: relu(8)=8*1 + relu(0)=0 => 8
  EXPECT_NEAR(out[5], 8.0f, 1e-6);
}

TEST(MqaLogits, NegativeReluSuppresses) {
  const std::size_t M = 1, N = 1, H = 1, D = 1;
  std::vector<float> q{-3.0f}, kv{5.0f}, weights{7.0f};
  std::vector<int> ks{0}, ke{1};
  std::vector<float> out(1, 99.0f);
  vkernels::kernels::mqa_logits(M, N, H, D, q, kv, weights, ks, ke, out);
  EXPECT_NEAR(out[0], 0.0f, 1e-6);  // relu(-15) = 0, weight irrelevant
}

TEST(MqaLogits, EmptySpanIsAllMasked) {
  const std::size_t M = 1, N = 4, H = 1, D = 1;
  std::vector<float> q{1}, kv(4, 1.0f), weights{1};
  std::vector<int> ks{2}, ke{2};  // empty span
  std::vector<float> out(M * N, 0.0f);
  vkernels::kernels::mqa_logits(M, N, H, D, q, kv, weights, ks, ke, out);
  for (float v : out) EXPECT_TRUE(std::isinf(v) && v < 0);
}

TEST(MqaLogits, ShapeValidation) {
  std::vector<float> q(4), kv(4), weights(4), out(4);
  std::vector<int> ks(2), ke(2);
  EXPECT_THROW(vkernels::kernels::mqa_logits(2, 2, 2, 2, q, kv, mk(3, 0), ks, ke, out),
               std::invalid_argument);
  std::vector<int> bad_ke{0, 3};  // end beyond N=2
  EXPECT_THROW(vkernels::kernels::mqa_logits(2, 2, 1, 2, q, kv, mk(2, 0), ks, bad_ke, out),
               std::invalid_argument);
}

TEST(GroupedGemmMasked, ComputesValidRowsOnlyAndLeavesMaskedUntouched) {
  const std::size_t G = 2, M = 3, N = 2, K = 2;
  // A[g] rows: all 1s (row m of group g = m+g pattern via fill below).
  std::vector<float> A(G * M * K), B(G * N * K), C(G * M * N);
  for (std::size_t g = 0; g < G; ++g)
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t k = 0; k < K; ++k) A[(g * M + m) * K + k] = float(m + 1);
  for (std::size_t g = 0; g < G; ++g)
    for (std::size_t n = 0; n < N; ++n)
      for (std::size_t k = 0; k < K; ++k) B[(g * N + n) * K + k] = float(g + 1);
  const float poison = std::numeric_limits<float>::quiet_NaN();
  for (float& c : C) c = poison;
  std::vector<int> masked{2, 0};  // group 0: rows 0-1; group 1: none

  vkernels::kernels::m_grouped_gemm_nt_masked(G, M, N, K, A, B, masked, C);

  for (std::size_t m = 0; m < M; ++m) {
    for (std::size_t n = 0; n < N; ++n) {
      const float c0 = C[(0 * M + m) * N + n];
      const float c1 = C[(1 * M + m) * N + n];
      if (m < 2) {
        // group 0: (m+1) * (1 + 1) summed over K=2
        EXPECT_NEAR(c0, 2.0f * (m + 1), 1e-6);
      } else {
        EXPECT_TRUE(std::isnan(c0));  // masked row untouched
      }
      EXPECT_TRUE(std::isnan(c1));  // group 1 fully masked
    }
  }
}

TEST(GroupedGemmMasked, NTUsesBTransposed) {
  const std::size_t G = 1, M = 1, N = 2, K = 3;
  std::vector<float> A{1, 2, 3};
  std::vector<float> B{1, 0, 0, 0, 1, 0};  // B[0]=(1,0,0), B[1]=(0,1,0)
  std::vector<float> C(2, 0.0f);
  std::vector<int> masked{1};
  vkernels::kernels::m_grouped_gemm_nt_masked(G, M, N, K, A, B, masked, C);
  EXPECT_NEAR(C[0], 1.0f, 1e-6);  // A . B[0]
  EXPECT_NEAR(C[1], 2.0f, 1e-6);  // A . B[1]
}

TEST(GroupedGemmMasked, ShapeValidation) {
  std::vector<float> A(8), B(8), C(8);
  std::vector<int> masked{1, 5};  // 5 > M=2
  EXPECT_THROW(vkernels::kernels::m_grouped_gemm_nt_masked(2, 2, 2, 2, A, B, masked, C),
               std::invalid_argument);
}
