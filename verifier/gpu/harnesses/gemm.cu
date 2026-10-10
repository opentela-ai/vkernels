// verifier/gpu/harnesses/gemm.cu
//
// GPUVerify-dialect mirror of the tiled SGEMM kernel in
// src/c/vkernels/kernels/gemm.cu.  This is the interesting one for the shared
// memory / barrier properties: two shared tiles, a __syncthreads() between the
// load and the accumulate, and a second before the next iteration overwrites
// the tiles.
//
// The size guard is not in the production kernel.  GPUVerify explores the
// index arithmetic for *all* M/N/K; with 32-bit `int` indices a huge N makes
// `row*N + col` overflow and alias, producing a spurious race.  Restricting
// M/N/K to a range that cannot overflow is the model of the launcher's domain
// (the launcher caps sizes to fit the grid), and is what makes the proof
// meaningful.  Keep the kernel body in sync with the production source.
//
// Invocation:
//   verifier/gpu/gpuverify.sh --blockDim=16,16,1 --gridDim=2,2,1 \
//       verifier/gpu/harnesses/gemm.cu
#include "cuda.h"

#define TILE 16

__global__ void gemm_kernel(const float* A, const float* B, float* C, int M,
                            int N, int K, float alpha, float beta) {
  if (M < 0 || N < 0 || K < 0 || M > 64 || N > 64 || K > 64) return;

  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float sA[TILE][TILE];
  __shared__ float sB[TILE][TILE];

  float acc = 0.0f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    sA[threadIdx.y][threadIdx.x] =
        (t * TILE + threadIdx.x < K && row < M) ? A[row * K + t * TILE + threadIdx.x] : 0.0f;
    sB[threadIdx.y][threadIdx.x] =
        (t * TILE + threadIdx.y < K && col < N) ? B[(t * TILE + threadIdx.y) * N + col] : 0.0f;
    __syncthreads();

    for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N)
    C[row * N + col] = alpha * acc + (beta == 0.0f ? 0.0f : beta * C[row * N + col]);
}
