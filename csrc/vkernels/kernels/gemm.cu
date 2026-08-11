// vkernels/kernels/gemm.cu — CUDA tiled SGEMM (compiled with a toolkit).
#include "vkernels/kernels/gemm.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/util/error.hpp"

namespace vkernels::kernels {

constexpr int kTile = 16;

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M,
                            int N, int K, float alpha, float beta) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  __shared__ float sA[kTile][kTile];
  __shared__ float sB[kTile][kTile];

  float acc = 0.0f;
  for (int t = 0; t < (K + kTile - 1) / kTile; ++t) {
    sA[threadIdx.y][threadIdx.x] =
        (t * kTile + threadIdx.x < K && row < M) ? A[row * K + t * kTile + threadIdx.x] : 0.0f;
    sB[threadIdx.y][threadIdx.x] =
        (t * kTile + threadIdx.y < K && col < N) ? B[(t * kTile + threadIdx.y) * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < kTile; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }
  C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

void gemm(std::size_t M, std::size_t N, std::size_t K, float alpha,
          Span<const float> A, Span<const float> B, float beta, Span<float> C) {
  VK_EXPECTS(A.size() == M * K, "A must be M*K");
  VK_EXPECTS(B.size() == K * N, "B must be K*N");
  VK_EXPECTS(C.size() == M * N, "C must be M*N");

  dim3 block(kTile, kTile);
  dim3 grid(static_cast<int>((N + kTile - 1) / kTile), static_cast<int>((M + kTile - 1) / kTile));
  gemm_kernel<<<grid, block>>>(A.data(), B.data(), C.data(),
                               static_cast<int>(M), static_cast<int>(N),
                               static_cast<int>(K), alpha, beta);
  cudaError_t err = cudaGetLastError();
  VK_ENSURES(err == cudaSuccess, "cuda gemm launch failed");
}

}  // namespace vkernels::kernels

#endif  // VKERNELS_HAS_CUDA
