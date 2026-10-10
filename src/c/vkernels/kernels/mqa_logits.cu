// vkernels/kernels/mqa_logits.cu — CUDA implementation (compiled with a
// toolkit). Kernel design borrowed from DeepGEMM's sm90_fp8_mqa_logits
// family: one program per (query row, KV tile), the weighted-ReLU MQA
// scoring loop over the row's span. This first cut keeps the mapping
// simple (one thread per output element) — the oracle-parity contract is
// the deliverable; the tuned tile ladder lands with profiling.
#include "vkernels/kernels/mqa_logits.hpp"

#if VKERNELS_HAS_CUDA
#  include <cuda_runtime.h>

#  include <limits>

#  include "vkernels/kernels/common/launch.hpp"
#  include "vkernels/util/error.hpp"

namespace vkernels::kernels {

namespace {

__global__ void mqa_logits_kernel(const float* __restrict__ q,
                                  const float* __restrict__ kv,
                                  const float* __restrict__ weights,
                                  const int* __restrict__ cu_ks,
                                  const int* __restrict__ cu_ke,
                                  float* __restrict__ out, int M, int N, int H,
                                  int D) {
  const int m = blockIdx.y;
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M || n >= N) return;
  const float neg_inf = -std::numeric_limits<float>::infinity();
  if (n < cu_ks[m] || n >= cu_ke[m]) {
    out[m * N + n] = neg_inf;
    return;
  }
  float acc = 0.0f;
  for (int h = 0; h < H; ++h) {
    float dot = 0.0f;
    const float* q_row = q + (static_cast<std::size_t>(m) * H + h) * D;
    const float* k_row = kv + static_cast<std::size_t>(n) * D;
    for (int d = 0; d < D; ++d) dot += q_row[d] * k_row[d];
    if (dot > 0.0f) acc += weights[m * H + h] * dot;
  }
  out[m * N + n] = acc;
}

}  // namespace

namespace cuda {

void mqa_logits(std::size_t M, std::size_t N, std::size_t H, std::size_t D,
                Span<const float> q, Span<const float> kv,
                Span<const float> weights, Span<const int> cu_seq_len_k_start,
                Span<const int> cu_seq_len_k_end, Span<float> out) {
  VK_EXPECTS(q.size() == M * H * D, "q must be M*H*D");
  VK_EXPECTS(kv.size() == N * D, "kv must be N*D");
  VK_EXPECTS(weights.size() == M * H, "weights must be M*H");
  VK_EXPECTS(cu_seq_len_k_start.size() == M, "cu_seq_len_k_start must be M");
  VK_EXPECTS(cu_seq_len_k_end.size() == M, "cu_seq_len_k_end must be M");
  VK_EXPECTS(out.size() == M * N, "out must be M*N");
  if (M == 0 || N == 0) return;

  const int threads = static_cast<int>(common::default_block_size());
  const dim3 grid(static_cast<unsigned>(common::ceil_div(N, static_cast<std::size_t>(threads))),
                  static_cast<unsigned>(M));
  common::launch(mqa_logits_kernel, grid, dim3(static_cast<unsigned>(threads)),
                 "cuda mqa_logits", q.data(), kv.data(), weights.data(),
                 cu_seq_len_k_start.data(), cu_seq_len_k_end.data(), out.data(),
                 static_cast<int>(M), static_cast<int>(N), static_cast<int>(H),
                 static_cast<int>(D));
}

}  // namespace cuda
}  // namespace vkernels::kernels

#endif  // VKERNELS_HAS_CUDA
