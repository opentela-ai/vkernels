// vkernels/kernels/mla.cpp -- CPU reference (oracle) implementation.
//
// Absorbed-form Multi-head Latent Attention (issue #21), fp32 throughout with
// a numerically-stable two-pass softmax. This is the CPU/torch reference the
// gfx942 HIP kernel is checked against; it is always compiled (no GPU
// toolkit) and is the correctness oracle for both the host unit tests and the
// on-device `test_mla_correct.hip` harness.
//
// See mla.hpp for the layout, the absorbed-query / decoupled-RoPE contract,
// and the causal mask (q_start, kv_start).
#include "vkernels/kernels/mla.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {

void mla_fwd_cpu(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
                 int kv_lora_rank, int qk_rope_head_dim, float scale,
                 const float* q, const float* k_c, const float* k_pe,
                 const float* v_c, float* out) {
  const int D_q = kv_lora_rank + qk_rope_head_dim;
  VK_EXPECTS(B == 0 || H == 0 || S_q == 0 || S_kv == 0 || q != nullptr,
             "q must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S_q == 0 || S_kv == 0 || k_c != nullptr,
             "k_c must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S_q == 0 || S_kv == 0 || k_pe != nullptr,
             "k_pe must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S_q == 0 || S_kv == 0 || v_c != nullptr,
             "v_c must not be null");
  VK_EXPECTS(B == 0 || H == 0 || S_q == 0 || S_kv == 0 || out != nullptr,
             "out must not be null");
  VK_EXPECTS(kv_lora_rank > 0 && qk_rope_head_dim > 0, "ranks must be positive");
  if (B == 0 || H == 0 || S_q == 0 || S_kv == 0) return;  // nothing to do

  // Reusable softmax-weight scratch for one query row. Masked keys are stored
  // as -inf (exp() yields 0; the accumulation test `w[j] < 0.0f` skips them
  // since a real weight is always >= 0).
  std::vector<float> w(S_kv);
  const float kNegInf = -std::numeric_limits<float>::infinity();

  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      const float* qh = q + (((size_t)b * H + h) * S_q) * D_q;
      float* oh = out + (((size_t)b * H + h) * S_q) * kv_lora_rank;
      for (int i = 0; i < S_q; ++i) {
        const float* qi = qh + (size_t)i * D_q;
        const float* q_nope = qi;
        const float* q_rope = qi + kv_lora_rank;
        const int gqi = q_start + i;

        // --- pass 1: scores + running max (numerically stable) ---
        float max_s = kNegInf;
        for (int j = 0; j < S_kv; ++j) {
          const int gkj = kv_start + j;
          if (gkj > gqi) { w[j] = kNegInf; continue; }
          const float* kj = k_c + ((size_t)b * S_kv + j) * kv_lora_rank;
          const float* pj = k_pe + ((size_t)b * S_kv + j) * qk_rope_head_dim;
          float dot = 0.0f;
          for (int d = 0; d < kv_lora_rank; ++d) dot += q_nope[d] * kj[d];
          for (int d = 0; d < qk_rope_head_dim; ++d) dot += q_rope[d] * pj[d];
          const float s = scale * dot;
          w[j] = s;
          if (s > max_s) max_s = s;
        }

        // --- pass 2: softmax denominator (skip masked, exp() >= 0) ---
        float sum = 0.0f;
        for (int j = 0; j < S_kv; ++j) {
          if (w[j] == kNegInf) continue;  // masked key
          w[j] = std::exp(w[j] - max_s);
          sum += w[j];
        }
        float* oi = oh + (size_t)i * kv_lora_rank;
        if (sum == 0.0f) {  // every key masked (e.g. gqi < kv_start)
          for (int d = 0; d < kv_lora_rank; ++d) oi[d] = 0.0f;
          continue;
        }
        const float inv = 1.0f / sum;

        // --- weighted accumulation over v_c ---
        for (int d = 0; d < kv_lora_rank; ++d) oi[d] = 0.0f;
        for (int j = 0; j < S_kv; ++j) {
          if (w[j] < 0.0f) continue;  // masked key (exp() is always >= 0)
          const float a = w[j] * inv;
          const float* vj = v_c + ((size_t)b * S_kv + j) * kv_lora_rank;
          for (int d = 0; d < kv_lora_rank; ++d) oi[d] += a * vj[d];
        }
      }
    }
  }
}

void mla_config_for(int S_q, int kv_lora_rank, int qk_rope_head_dim,
                    int* bq, int* bn_kv, int* threads) {
  (void)kv_lora_rank;
  (void)qk_rope_head_dim;
  // Decode: tiny S_q, memory-bound on the per-head K/V latent reads. One
  // query row per block lets every block stream its own key window and
  // saturate the 228 CUs; BN_kv keeps the register/LDS pressure modest so
  // the kernel fits at the K3 latent rank (512).
  if (S_q <= 8) {
    *bq = 1;
    *bn_kv = 64;
    *threads = 64;  // one wavefront
  } else {
    // Prefill: tiled (bq x bn_kv) causal attention. 4 query rows per block
    // keeps the score tile small enough for online softmax in registers
    // while BN_kv=64 reuses each loaded key window across the rows.
    *bq = 4;
    *bn_kv = 64;
    *threads = 256;  // four wavefronts
  }
}

}  // namespace vkernels::kernels
