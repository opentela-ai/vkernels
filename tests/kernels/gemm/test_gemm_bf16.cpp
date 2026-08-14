// tests/kernels/gemm/test_gemm_bf16.cpp
//
// Host tests for the bf16 GEMM CPU reference (issue #29): a hand-checked
// case, alpha/beta accumulation, a bit-exact cross-check against an
// independent fp32+RNE reference across the K3 projection shapes (including
// the awkward N=6288 that is not a multiple of 64), the per-shape config
// selector, the null-arg contracts, and the M=0/N=0 no-op edges.
#include "minitest.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "vkernels/kernels/gemm_bf16.hpp"

using vkernels::kernels::gemm_bf16_config_for;
using vkernels::kernels::gemm_bf16_cpu;

namespace {

// Independent bf16 helpers (RNE), not the kernel's own.
uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(float));
  uint32_t lsb = (bits >> 16) & 1;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}
float bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, sizeof(float));
  return f;
}

// Independent fp32 reference with the identical shape/contract as
// gemm_bf16_cpu (bf16->f32, fp32 accumulate, RNE on store).
void ref(std::size_t M, std::size_t N, std::size_t K, float alpha,
         const std::vector<uint16_t>& A, const std::vector<uint16_t>& B,
         float beta, std::vector<uint16_t>& C) {
  for (std::size_t i = 0; i < M; ++i)
    for (std::size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (std::size_t k = 0; k < K; ++k)
        acc += bf16_to_f32(A[i * K + k]) * bf16_to_f32(B[k * N + j]);
      float prev = beta != 0.0f ? bf16_to_f32(C[i * N + j]) : 0.0f;
      C[i * N + j] = f32_to_bf16(alpha * acc + beta * prev);
    }
}

}  // namespace

TEST(GemmBf16, IdentityAlphaOneBetaZero) {
  std::vector<uint16_t> A = {f32_to_bf16(1), f32_to_bf16(2),  // [[1,2],[3,4]]
                             f32_to_bf16(3), f32_to_bf16(4)};
  std::vector<uint16_t> B = {f32_to_bf16(1), f32_to_bf16(0),  // identity
                             f32_to_bf16(0), f32_to_bf16(1)};
  std::vector<uint16_t> C(4, 0);
  gemm_bf16_cpu(2, 2, 2, 1.0f, A.data(), B.data(), 0.0f, C.data());
  EXPECT_NEAR(bf16_to_f32(C[0]), 1, 1e-6);
  EXPECT_NEAR(bf16_to_f32(C[1]), 2, 1e-6);
  EXPECT_NEAR(bf16_to_f32(C[2]), 3, 1e-6);
  EXPECT_NEAR(bf16_to_f32(C[3]), 4, 1e-6);
}

TEST(GemmBf16, NonSquareAndBetaAccumulation) {
  // 2x3 * 3x2 = 2x2; C = 1*A*B + 2*C (pre-filled with bf16(1)).
  std::vector<uint16_t> A = {f32_to_bf16(1), f32_to_bf16(2), f32_to_bf16(3),
                             f32_to_bf16(4), f32_to_bf16(5), f32_to_bf16(6)};
  std::vector<uint16_t> B = {f32_to_bf16(7), f32_to_bf16(8), f32_to_bf16(9),
                             f32_to_bf16(10), f32_to_bf16(11), f32_to_bf16(12)};
  std::vector<uint16_t> C = {f32_to_bf16(1), f32_to_bf16(1), f32_to_bf16(1),
                             f32_to_bf16(1)};
  std::vector<uint16_t> E = C;
  gemm_bf16_cpu(2, 2, 3, 1.0f, A.data(), B.data(), 2.0f, C.data());
  ref(2, 2, 3, 1.0f, A, B, 2.0f, E);
  EXPECT_EQ(C[0], E[0]);  // bit-exact: same math, same order
  EXPECT_EQ(C[1], E[1]);
  EXPECT_EQ(C[2], E[2]);
  EXPECT_EQ(C[3], E[3]);
}

TEST(GemmBf16, MatchesIndependentReferenceAcrossK3Shapes) {
  // K3 projection (N, K) at a small serving M, including N=6288 (not a
  // multiple of 64) and K=7168 (largest, multiple of 64).
  struct Shape {
    int M, N, K;
  };
  const Shape shapes[] = {
      {8, 128, 64},      // (1536x128)
      {8, 256, 512},     // (3072x512)
      {16, 1536, 768},   // (7168x768)
      {32, 2304, 1536},  // (2304x1536)
      {64, 3584, 7168},  // (3584x7168)
      {5, 6288, 7168},   // QKV (N=6288 not a multiple of 64)
  };
  std::mt19937 rng(12345);
  auto rand_bf16 = [&]() {
    float v = static_cast<float>(rng() % 2000) / 1000.0f - 1.0f;  // [-1, 1)
    return f32_to_bf16(v);
  };
  for (const auto& s : shapes) {
    std::vector<uint16_t> A(static_cast<size_t>(s.M) * s.K);
    std::vector<uint16_t> B(static_cast<size_t>(s.K) * s.N);
    std::vector<uint16_t> C(static_cast<size_t>(s.M) * s.N, 0);
    std::vector<uint16_t> E(static_cast<size_t>(s.M) * s.N, 0);
    for (auto& x : A) x = rand_bf16();
    for (auto& x : B) x = rand_bf16();
    gemm_bf16_cpu(s.M, s.N, s.K, 1.0f, A.data(), B.data(), 0.0f, C.data());
    ref(s.M, s.N, s.K, 1.0f, A, B, 0.0f, E);
    int bad = 0;
    for (size_t i = 0; i < C.size(); ++i)
      if (C[i] != E[i]) ++bad;
    EXPECT_EQ(bad, 0);
  }
}

TEST(GemmBf16Config, ServingShapePicks16x16) {
  int bm = 0, bn = 0, bk = 0, threads = 0;
  gemm_bf16_config_for(8, 6288, 7168, &bm, &bn, &bk, &threads);
  EXPECT_EQ(bm, 16);
  EXPECT_EQ(bn, 16);
  EXPECT_EQ(bk, 64);
  EXPECT_EQ(threads, 64);
  // The boundary M == 64 is still serving.
  gemm_bf16_config_for(64, 128, 512, &bm, &bn, &bk, &threads);
  EXPECT_EQ(bm, 16);
  EXPECT_EQ(bn, 16);
  EXPECT_EQ(threads, 64);
}

TEST(GemmBf16Config, WarmupShapePicks64x64) {
  int bm = 0, bn = 0, bk = 0, threads = 0;
  gemm_bf16_config_for(8192, 6288, 7168, &bm, &bn, &bk, &threads);
  EXPECT_EQ(bm, 64);
  EXPECT_EQ(bn, 64);
  EXPECT_EQ(bk, 64);
  EXPECT_EQ(threads, 256);
  // M == 65 crosses into the warmup / prefill tile.
  gemm_bf16_config_for(65, 128, 64, &bm, &bn, &bk, &threads);
  EXPECT_EQ(bm, 64);
  EXPECT_EQ(threads, 256);
}

TEST(GemmBf16, NullArgsThrow) {
  std::vector<uint16_t> A(4, f32_to_bf16(1)), C(4, f32_to_bf16(0));
  EXPECT_THROW(gemm_bf16_cpu(2, 2, 2, 1.0f, nullptr, C.data(), 0.0f, C.data()),
               std::invalid_argument);
  EXPECT_THROW(gemm_bf16_cpu(2, 2, 2, 1.0f, A.data(), nullptr, 0.0f, C.data()),
               std::invalid_argument);
  EXPECT_THROW(gemm_bf16_cpu(2, 2, 2, 1.0f, A.data(), A.data(), 0.0f, nullptr),
               std::invalid_argument);
}

TEST(GemmBf16, EmptyIsNoOp) {
  // M == 0 (and N == 0) short-circuit the contracts; nothing is read or written.
  std::vector<uint16_t> C = {f32_to_bf16(5), f32_to_bf16(6), f32_to_bf16(7),
                             f32_to_bf16(8)};
  std::vector<uint16_t> E = C;
  gemm_bf16_cpu(0, 2, 2, 1.0f, nullptr, nullptr, 1.0f, nullptr);  // M==0
  gemm_bf16_cpu(0, 0, 0, 1.0f, nullptr, nullptr, 1.0f, nullptr);  // M==N==0
  EXPECT_EQ(C[0], E[0]);
  EXPECT_EQ(C[1], E[1]);
  EXPECT_EQ(C[2], E[2]);
  EXPECT_EQ(C[3], E[3]);
}
