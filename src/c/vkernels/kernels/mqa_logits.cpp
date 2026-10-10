// vkernels/kernels/mqa_logits.cpp — CPU reference (oracle) implementation.
#include "vkernels/kernels/mqa_logits.hpp"

#include <limits>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

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

  const float neg_inf = -std::numeric_limits<float>::infinity();
  for (std::size_t m = 0; m < M; ++m) {
    const int k_start = cu_seq_len_k_start[m];
    const int k_end = cu_seq_len_k_end[m];
    VK_EXPECTS(k_start >= 0 && static_cast<std::size_t>(k_start) <= N,
               "cu_seq_len_k_start must be within [0, N]");
    VK_EXPECTS(k_end >= k_start && static_cast<std::size_t>(k_end) <= N,
               "cu_seq_len_k_end must be within [start, N]");
    for (std::size_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      if (static_cast<int>(n) >= k_start && static_cast<int>(n) < k_end) {
        for (std::size_t h = 0; h < H; ++h) {
          float dot = 0.0f;
          for (std::size_t d = 0; d < D; ++d) {
            dot += q[(m * H + h) * D + d] * kv[n * D + d];
          }
          if (dot > 0.0f) acc += weights[m * H + h] * dot;
        }
      } else {
        acc = neg_inf;
      }
      out[m * N + n] = acc;
    }
  }
}

}  // namespace vkernels::kernels
