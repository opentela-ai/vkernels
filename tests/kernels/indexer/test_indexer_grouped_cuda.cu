// tests/kernels/indexer/test_indexer_grouped_cuda.cu — device parity for
// the DeepGEMM-borrowed kernels: cuda::mqa_logits and
// cuda::m_grouped_gemm_nt_masked against their CPU oracles on this device.
#include "minitest.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
#include <vector>

#include "vkernels/kernels/gemm_grouped.hpp"
#include "vkernels/kernels/mqa_logits.hpp"

namespace {

struct FBuf {
  float* ptr = nullptr;
  explicit FBuf(std::size_t n) { EXPECT_EQ(cudaMalloc(&ptr, n * sizeof(float)), cudaSuccess); }
  ~FBuf() { cudaFree(ptr); }
};

struct IBuf {
  int* ptr = nullptr;
  explicit IBuf(std::size_t n) { EXPECT_EQ(cudaMalloc(&ptr, n * sizeof(int)), cudaSuccess); }
  ~IBuf() { cudaFree(ptr); }
};

void expect_same(const std::vector<float>& got, const std::vector<float>& want, float tol) {
  EXPECT_EQ(got.size(), want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::isinf(want[i])) {
      EXPECT_TRUE(std::isinf(got[i]) && got[i] < 0);
    } else {
      EXPECT_NEAR(got[i], want[i], tol);
    }
  }
}

}  // namespace

TEST(IndexerGroupedCuda, MqaLogitsMatchesOracle) {
  const std::size_t M = 5, N = 37, H = 3, D = 8;
  std::vector<float> q(M * H * D), kv(N * D), weights(M * H);
  std::vector<int> ks(M), ke(M);
  unsigned seed = 12345;
  auto rnd = [&seed]() {
    seed = seed * 1103515245u + 12345u;
    return static_cast<float>(static_cast<int>(seed >> 16 & 0x7fff) % 200 - 100) / 10.0f;
  };
  for (float& v : q) v = rnd();
  for (float& v : kv) v = rnd();
  for (float& v : weights) v = rnd();
  for (std::size_t m = 0; m < M; ++m) {
    ks[m] = static_cast<int>(m % 7);
    ke[m] = static_cast<int>(N - (m % 11));
  }

  std::vector<float> oracle(M * N), device(M * N);
  vkernels::kernels::mqa_logits(
      M, N, H, D, {q.data(), q.size()}, {kv.data(), kv.size()},
      {weights.data(), weights.size()}, {ks.data(), ks.size()},
      {ke.data(), ke.size()}, {oracle.data(), oracle.size()});

  FBuf dq(q.size()), dkv(kv.size()), dw(weights.size()), dout(device.size());
  IBuf dks(ks.size()), dke(ke.size());
  cudaMemcpy(dq.ptr, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dkv.ptr, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dw.ptr, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dks.ptr, ks.data(), ks.size() * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dke.ptr, ke.data(), ke.size() * sizeof(int), cudaMemcpyHostToDevice);
  vkernels::kernels::cuda::mqa_logits(M, N, H, D, {dq.ptr, q.size()}, {dkv.ptr, kv.size()},
                                      {dw.ptr, weights.size()}, {dks.ptr, ks.size()},
                                      {dke.ptr, ke.size()}, {dout.ptr, device.size()});
  EXPECT_EQ(cudaMemcpy(device.data(), dout.ptr, device.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
            cudaSuccess);
  expect_same(device, oracle, 1e-4);
}

TEST(IndexerGroupedCuda, MaskedGroupedGemmMatchesOracleAndKeepsMaskedRows) {
  const std::size_t G = 3, M = 9, N = 7, K = 16;
  std::vector<float> A(G * M * K), B(G * N * K), C(G * M * N, 0.0f);
  std::vector<int> masked{4, 0, 9};
  unsigned seed = 999;
  auto rnd = [&seed]() {
    seed = seed * 1103515245u + 12345u;
    return static_cast<float>(static_cast<int>(seed >> 16 & 0x7fff) % 50 - 25) / 5.0f;
  };
  for (float& v : A) v = rnd();
  for (float& v : B) v = rnd();

  std::vector<float> oracle(G * M * N, 0.0f);
  vkernels::kernels::m_grouped_gemm_nt_masked(
      G, M, N, K, {A.data(), A.size()}, {B.data(), B.size()}, {masked.data(), masked.size()},
      {oracle.data(), oracle.size()});

  FBuf da(A.size()), db(B.size()), dc(C.size());
  IBuf dm(G);
  cudaMemcpy(da.ptr, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(db.ptr, B.data(), B.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dm.ptr, masked.data(), G * sizeof(int), cudaMemcpyHostToDevice);
  vkernels::kernels::cuda::m_grouped_gemm_nt_masked(G, M, N, K, {da.ptr, A.size()},
                                                    {db.ptr, B.size()}, {dm.ptr, G},
                                                    {dc.ptr, C.size()});
  EXPECT_EQ(cudaMemcpy(C.data(), dc.ptr, C.size() * sizeof(float), cudaMemcpyDeviceToHost),
            cudaSuccess);
  expect_same(C, oracle, 1e-3);
}
