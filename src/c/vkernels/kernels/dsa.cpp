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

// ---------------------------------------------------------------------------
//  DSA paged-MQA gated top-k logits — CPU reference (oracle), issue #51.
//  See dsa.hpp for the layout and the gated-logit formula. Pure fp32, single
//  pass (no softmax — the topk IS the softmax-free scoring stage). The HIP
//  kernel (dsa.hip) is cross-checked against this by
//  meta/benchmarks/test_dsa_topk_correct.hip: it dequants the fp8 e4m3fnuz
//  source on the host (via fp8e4m3fnuz_to_f32, the same helper the kernel
//  uses on load) and feeds the recovered fp32 to this oracle; the Python
//  caller dequants via torch.float8_e4m3fnuz. The host tests
//  (tests/kernels/attn/test_dsa.cpp, DsaTopk*) cover the oracle directly.
//
//  Tokens t >= seq_len[b] are LEFT UNWRITTEN — the caller ZEROES the output
//  first (strictly safer than the tilelang wrapper's `new_empty`, harmless
//  under sglang's `topk_from_pooled_history_logits` masking).
// ---------------------------------------------------------------------------
void dsa_topk_logits_cpu(int batch_size, int num_heads, int head_dim, int block,
                         int max_table_len, int num_blocks,
                         const float* q, const float* kv,
                         const float* k_scale, const float* gate,
                         const int32_t* seq_lens, const int32_t* page_table,
                         float* out) {
  VK_EXPECTS(batch_size >= 0 && num_heads > 0 && head_dim > 0 && block > 0,
             "batch_size >= 0 and num_heads/head_dim/block > 0 required");
  VK_EXPECTS(max_table_len >= 0 && num_blocks >= 0,
             "max_table_len >= 0 and num_blocks >= 0 required");
  // With work to do, every pointer is required (mirrors dsa_sparse_fwd_cpu).
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || q != nullptr,
             "q must not be null");
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || kv != nullptr,
             "kv must not be null");
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || k_scale != nullptr,
             "k_scale must not be null");
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || gate != nullptr,
             "gate must not be null");
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || seq_lens != nullptr,
             "seq_lens must not be null");
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || page_table != nullptr,
             "page_table must not be null");
  VK_EXPECTS(batch_size == 0 || max_table_len == 0 || out != nullptr,
             "out must not be null");
  if (batch_size == 0 || max_table_len == 0) return;  // nothing to write
  // num_blocks == 0: every page is out of range -> output stays zeroed
  // (the OOB-page guard below mirrors dsa_sparse_fwd_cpu's idx<S_kv mask).

  for (int b = 0; b < batch_size; ++b) {
    const int seq_len = seq_lens[b];
    const int np_total = (seq_len + block - 1) / block;  // pages with content
    const int32_t* pt_b = page_table + (size_t)b * max_table_len;
    const float* qb = q + (size_t)b * num_heads * head_dim;
    const float* gb = gate + (size_t)b * num_heads;
    float* ob = out + (size_t)b * (size_t)max_table_len * block;
    for (int i = 0; i < np_total; ++i) {
      const int32_t page = pt_b[i];
      if (page < 0 || page >= num_blocks) continue;  // OOB page -> unwritten
      const float* kbase = kv + (size_t)page * block * head_dim;
      const float* sbase = k_scale + (size_t)page * block;
      for (int j = 0; j < block; ++j) {
        const int t = i * block + j;
        if (t >= seq_len) break;  // rest of this page is past seq_len
        const float* kj = kbase + (size_t)j * head_dim;
        float acc = 0.0f;
        for (int h = 0; h < num_heads; ++h) {
          const float* qh = qb + (size_t)h * head_dim;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) dot += qh[d] * kj[d];
          acc += std::fmax(dot, 0.0f) * gb[h];
        }
        ob[t] = sbase[j] * acc;
      }
    }
  }
}

// Whether the indexer shape fits gfx942's 64 KB non-optin dynamic-LDS cap
// under the fp32-Q kernel (the launcher's fast path). Pure arithmetic
// mirroring the HIP kernel's shared-memory request; see dsa.hpp for the
// staged bytes, the verified GLM-5.3 config and the cap rationale (gfx942
// has NO hipFuncSetAttribute opt-in past 64 KB -- see the KB note
// mi300a-dynamic-lds-no-optin; the fp8-Q variant below is the path for
// larger H). All-int so the host build (WARNINGS_AS_ERROR) stays clean.
bool dsa_topk_logits_fits_lds(int num_heads, int head_dim, int block) {
  // gfx942 (MI300A) non-optin dynamic-LDS cap; EQUALS the opt-in ceiling
  // (hipFuncSetAttribute(MaxDynamicSharedMemorySize, N>65536) is a silent
  // no-op, verified on a CSCS beverin node). Larger H falls back to the
  // fp8-Q variant instead of raising this cap. See dsa.hpp and the KB note
  // mi300a-dynamic-lds-no-optin.
  static constexpr int kDsaTopkLDSNonOptinCap = 64 * 1024;
  // The kernel stages Q (H*D), the gate (H), one K tile (B*D) and its
  // per-token scales (B), all fp32 -> 4 bytes/element (see dsa.hpp).
  const int bytes =
      (num_heads * head_dim + num_heads + block * head_dim + block) * 4;
  return bytes > 0 && bytes <= kDsaTopkLDSNonOptinCap;
}

// Whether the indexer shape fits gfx942's 64 KB non-optin dynamic-LDS cap
// under the fp8-Q variant (dsa_topk_logits_kernel_fp8q) -- the launcher's
// FALLBACK for shapes dsa_topk_logits_fits_lds refuses. The fp8-Q kernel
// stages Q as RAW fp8 (1 byte/element; dequantised on the fly in the dot
// loop with the SAME helper, so the output is bit-identical to the fp32-Q
// kernel), and the gate (H), one K tile (B*D) and its per-token scales (B)
// as fp32 -> (H + B*D + B) * 4 + H*D bytes. At the GLM-5.3 2x indexer
// (H=64, D=128, B=64) that is 41,472 B < the fp32-Q kernel's 66,048 B.
// All-int, same cap constant as above. See dsa.hpp and the host unit test
// tests/kernels/attn/test_dsa.cpp::DsaTopk::FitsLdsFp8q.
bool dsa_topk_logits_fits_lds_fp8q(int num_heads, int head_dim, int block) {
  static constexpr int kDsaTopkLDSNonOptinCap = 64 * 1024;
  const int bytes =
      (num_heads + block * head_dim + block) * 4 + num_heads * head_dim;
  return bytes > 0 && bytes <= kDsaTopkLDSNonOptinCap;
}

// Whether the indexer shape fits gfx942's 64 KB non-optin dynamic-LDS cap
// under the MFMA kernel (dsa_topk_logits_kernel_mfma) -- the launcher's
// FAST path for shapes the fp32-Q kernel refuses (H>=64 at D=128, B=64).
// The kernel stages Q transposed as bf16 sQt[D][H] ONCE per block (reused
// across every page in the split -- fp8->bf16 is lossless), one bf16 K-tile
// sK[B][BK=64] (reloaded per K-tile per page), the gate sGate[H] and the
// per-token scales sKscale[B] as fp32:
//
//   bytes = (head_dim*num_heads + block*kBK)*2 + (num_heads + block)*4
//
// the SMALLEST of the three variants at every GLM-5.3 width (16,768 B at
// H=32; 25,088 B at H=64; 41,728 B at H=128). The verified 16x16x16bf16_1k
// fragment needs exact multiples, so this is FALSE unless num_heads % 16,
// head_dim % 64 and block % 16 are all zero (e.g. H=246 -- not a multiple
// of 16 -- is refused, exactly as under the other two variants). All-int,
// same cap constant as above. See dsa.hpp and the host unit test
// tests/kernels/attn/test_dsa.cpp::DsaTopk::FitsLdsMfma.
bool dsa_topk_logits_fits_lds_mfma(int num_heads, int head_dim, int block) {
  static constexpr int kDsaTopkLDSNonOptinCap = 64 * 1024;
  static constexpr int kBK = 64;   // MFMA K-tile (fixed; see dsa.hip)
  if (num_heads <= 0 || head_dim <= 0 || block <= 0) return false;
  if (num_heads % 16 != 0 || head_dim % kBK != 0 || block % 16 != 0)
    return false;   // 16x16x16bf16_1k fragment needs exact multiples
  const int bytes =
      (head_dim * num_heads + block * kBK) * 2 + (num_heads + block) * 4;
  return bytes <= kDsaTopkLDSNonOptinCap;
}

int dsa_topk_logits_split_for(int batch_size, int max_seq_len, int block) {
  // gfx942 (MI300A) CU count (hipDeviceProp_t::multiProcessorCount,
  // verified on a CSCS beverin node). Raise alongside the fp8-Q variant's
  // cap (dsa_topk_logits_fits_lds_fp8q) + tiled-key/MFMA for devices with
  // a different CU count.
  static constexpr int kDsaTopkNumCU = 228;
  if (batch_size <= 0 || max_seq_len <= 0 || block <= 0) return 1;
  const int pages = (max_seq_len + block - 1) / block;   // ceildiv
  const int by_cu = kDsaTopkNumCU / batch_size;
  int s = pages < by_cu ? pages : by_cu;
  return s < 1 ? 1 : s;
}

}  // namespace vkernels::kernels
