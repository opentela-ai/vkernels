// vkernels/kernels/gemm_grouped.cu — CUDA implementation (compiled with a
// toolkit). Masked m-grouped GEMM for CUDA-graph MoE decode, borrowed
// from DeepGEMM's m_grouped_*_gemm_nt_masked contract: the per-group
// valid row count lives in device memory, so a captured graph computes
// exactly the live rows and leaves masked rows untouched. First cut maps
// one thread per output element (oracle parity is the deliverable; the
// tile/pipeline ladder lands with profiling).
#include "vkernels/kernels/gemm_grouped.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include "vkernels/kernels/common/launch.hpp"
#  include "vkernels/util/error.hpp"

namespace vkernels::kernels {

namespace {

__global__ void m_grouped_gemm_nt_masked_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                const int* __restrict__ masked_m,
                                                float* __restrict__ C, int G, int M,
                                                int N, int K) {
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(G) * M * N;
  if (linear >= total) return;
  const std::size_t n = linear % N;
  const std::size_t m = (linear / N) % M;
  const std::size_t g = linear / (static_cast<std::size_t>(M) * N);
  if (m >= static_cast<std::size_t>(masked_m[g])) return;  // row stays untouched

  const float* a = A + (g * M + m) * K;
  const float* b = B + (g * N + n) * K;
  float acc = 0.0f;
  for (std::size_t k = 0; k < K; ++k) acc += a[k] * b[k];
  C[linear] = acc;
}

}  // namespace

namespace cuda {

void m_grouped_gemm_nt_masked(std::size_t G, std::size_t M, std::size_t N,
                              std::size_t K, Span<const float> A,
                              Span<const float> B, Span<const int> masked_m,
                              Span<float> C) {
  VK_EXPECTS(A.size() == G * M * K, "A must be G*M*K");
  VK_EXPECTS(B.size() == G * N * K, "B must be G*N*K");
  VK_EXPECTS(masked_m.size() == G, "masked_m must be G");
  VK_EXPECTS(C.size() == G * M * N, "C must be G*M*N");
  const std::size_t total = G * M * N;
  if (total == 0) return;

  const int threads = static_cast<int>(common::default_block_size());
  const dim3 grid(static_cast<unsigned>(common::ceil_div(total, static_cast<std::size_t>(threads))));
  common::launch(m_grouped_gemm_nt_masked_kernel, grid,
                 dim3(static_cast<unsigned>(threads)), "cuda m_grouped_gemm_nt_masked",
                 A.data(), B.data(), masked_m.data(), C.data(),
                 static_cast<int>(G), static_cast<int>(M), static_cast<int>(N),
                 static_cast<int>(K));
}

}  // namespace cuda
}  // namespace vkernels::kernels

#endif  // VKERNELS_HAS_CUDA
