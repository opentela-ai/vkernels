// tests/kernels/gemm/test_gemm.cpp
#include "minitest.hpp"

#include <vector>

#include "vkernels/kernels/gemm.hpp"

using vkernels::kernels::gemm;

TEST(Gemm, IdentityAlphaOneBetaZero) {
  // 2x2 * 2x2 identity
  std::vector<float> A = {1, 2, 3, 4};          // row-major [[1,2],[3,4]]
  std::vector<float> B = {1, 0, 0, 1};          // identity
  std::vector<float> C(4, 0.0f);
  gemm(2, 2, 2, 1.0f, A, B, 0.0f, C);
  EXPECT_NEAR(C[0], 1, 1e-6);
  EXPECT_NEAR(C[1], 2, 1e-6);
  EXPECT_NEAR(C[2], 3, 1e-6);
  EXPECT_NEAR(C[3], 4, 1e-6);
}

TEST(Gemm, NonSquareAndBetaAccumulation) {
  // 2x3 * 3x2 = 2x2, with beta scaling a pre-filled C.
  std::vector<float> A = {1, 2, 3, 4, 5, 6};    // 2x3
  std::vector<float> B = {7, 8, 9, 10, 11, 12}; // 3x2
  std::vector<float> C = {1, 1, 1, 1};          // pre-filled
  // C = 1*A*B + 2*C
  gemm(2, 2, 3, 1.0f, A, B, 2.0f, C);
  // A*B = [[58,64],[139,154]]
  EXPECT_NEAR(C[0], 60, 1e-5);  // 58 + 2*1
  EXPECT_NEAR(C[1], 66, 1e-5);  // 64 + 2*1
  EXPECT_NEAR(C[2], 141, 1e-5); // 139 + 2*1
  EXPECT_NEAR(C[3], 156, 1e-5); // 154 + 2*1
}

TEST(Gemm, WrongAThrows) {
  std::vector<float> A(3, 0), B(4, 0), C(4, 0);
  EXPECT_THROW(gemm(2, 2, 2, 1.0f, A, B, 0.0f, C), std::invalid_argument);
}

TEST(Gemm, WrongBThrows) {
  std::vector<float> A(4, 0), B(3, 0), C(4, 0);
  EXPECT_THROW(gemm(2, 2, 2, 1.0f, A, B, 0.0f, C), std::invalid_argument);
}

TEST(Gemm, WrongCThrows) {
  std::vector<float> A(4, 0), B(4, 0), C(3, 0);
  EXPECT_THROW(gemm(2, 2, 2, 1.0f, A, B, 0.0f, C), std::invalid_argument);
}
