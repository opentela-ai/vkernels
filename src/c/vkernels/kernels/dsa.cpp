// vkernels/kernels/dsa.cpp -- CPU reference (oracle) implementation.
//
// DeepseekSparseAttn (DSA) sparse-MLA forward (issue #51). This is the
// CPU/torch reference the gfx942 HIP kernel is checked against; it is always
// compiled (no GPU toolkit) and is the correctness oracle for both the host
// unit tests and the on-device `test_dsa_correct.hip` harness.
//
// See dsa.hpp for the layout, the absorbed-value / decoupled-rope contract,
// and the base-2 softmax. The combined output is grouping-independent, so the
// oracle computes a single two-pass softmax over all `topk` selected keys per
// (query, head); the `block_I`/`inner_iter` group tiling is validated but not
// used to form the result.
#include "vkernels/kernels/dsa.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

void dsa_sparse_fwd_cpu(int S_q, int S_kv, int H, int dim, int tail_dim,
                        int topk, int kv_group, int block_I, int inner_iter,
                        float sm_scale, bool return_lse,
                        const float* q, const float* kv,
                        const int32_t* indices, float* out, float* lse) {
  const int d_v = dim - tail_dim;
  VK_EXPECTS(dim > 0 && tail_dim >= 0 && d_v > 0, "dim > tail_dim >= 0 required");
  VK_EXPECTS(topk > 0, "topk must be positive");
  VK_EXPECTS(kv_group == 1, "kv_group must be 1 (single shared head_kv)");
  // block_I / inner_iter are the kernel's group-tiling CONFIGURATION hints
  // (the indexer pads `topk` to a multiple of block_I in the sglang layout).
  // The forward scores every selected key directly, so divisibility is a
  // caller convention, not a math requirement (mirrors MLA's bn_kv).
  VK_EXPECTS(block_I > 0 && inner_iter > 0, "block_I and inner_iter must be positive");
  VK_EXPECTS(S_q == 0 || H == 0 || q != nullptr, "q must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || S_kv == 0 || kv != nullptr,
             "kv must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || topk == 0 || indices != nullptr,
             "indices must not be null");
  VK_EXPECTS(S_q == 0 || H == 0 || out != nullptr, "out must not be null");
  VK_EXPECTS(!return_lse || lse != nullptr, "lse must not be null");
  if (S_q == 0 || H == 0 || topk == 0) return;  // nothing to do
  // S_kv == 0: every selected index is out of range -> all rows zero.
  const int W = dim + tail_dim;
  const float kNegInf = -std::numeric_limits<float>::infinity();
  std::vector<float> w(topk);

  for (int i = 0; i < S_q; ++i) {
    const int32_t* idx_i = indices + (size_t)i * kv_group * topk;  // kv_group==1
    for (int h = 0; h < H; ++h) {
      const float* qh = q + (((size_t)i * H + h) * W);
      const float* q_main = qh;
      const float* q_tail = qh + dim;
      float* oh = out + (((size_t)i * H + h) * d_v);
      float* lh = return_lse ? (lse + ((size_t)i * H + h)) : nullptr;

      // --- pass 1: scores + running max (numerically stable) ---
      float max_s = kNegInf;
      for (int k = 0; k < topk; ++k) {
        const int32_t idx = idx_i[k];
        if (idx < 0 || idx >= S_kv) { w[k] = kNegInf; continue; }
        const float* kv_p = kv + (size_t)idx * W;  // kv_group==1 -> j*W
        float dot = 0.0f;
        for (int d = 0; d < dim; ++d) dot += q_main[d] * kv_p[d];
        for (int d = 0; d < tail_dim; ++d)  // skipped when tail_dim == 0
          dot += q_tail[d] * kv_p[dim + d];
        const float s = sm_scale * dot;
        w[k] = s;
        if (s > max_s) max_s = s;
      }

      // --- pass 2: base-2 softmax denominator + weighted accumulation ---
      if (max_s == kNegInf) {  // every selected key masked (e.g. all -1)
        for (int d = 0; d < d_v; ++d) oh[d] = 0.0f;
        if (lh) *lh = kNegInf;
        continue;
      }
      float sum = 0.0f;
      for (int k = 0; k < topk; ++k) {
        if (w[k] == kNegInf) { w[k] = 0.0f; continue; }
        w[k] = std::exp2(w[k] - max_s);
        sum += w[k];
      }
      const float inv = 1.0f / sum;
      for (int d = 0; d < d_v; ++d) oh[d] = 0.0f;
      for (int k = 0; k < topk; ++k) {
        if (w[k] == 0.0f) continue;  // masked key
        const int32_t idx = idx_i[k];
        const float* vj = kv + (size_t)idx * W;  // v = kv[idx][0:d_v]
        const float a = w[k] * inv;
        for (int d = 0; d < d_v; ++d) oh[d] += a * vj[d];
      }
      if (lh) *lh = max_s + std::log2(sum);
    }
  }
}

void dsa_config_for(int S_q, int H, int dim, int topk, int* bq, int* threads,
                    int* block_I, int* inner_iter) {
  (void)H;
  (void)dim;
  // Decode: a single query streams its `topk` selected keys; one wavefront
  // keeps the per-row online state in registers and saturates the CUs.
  // Prefill: BQ query rows per block shares the key reads across rows.
  if (S_q <= 8) {
    *bq = 1;
    *threads = 64;  // one wavefront
  } else {
    *bq = 4;
    *threads = 256;  // four wavefronts
  }
  *block_I = 64;
  // inner_iter so that block_I * inner_iter divides topk (power-of-two topk
  // is the common case; fall back to 1).
  int ii = 1;
  while (ii < 8 && topk % (*block_I * (ii * 2)) == 0) ii *= 2;
  *inner_iter = ii;
}

}  // namespace vkernels::kernels
