// vkernels/kernels/dsa_kpool.cpp -- host oracle for the DSA kpool-cache
// compress/write kernels (issue #60). See dsa_kpool.hpp for the computation
// and the bf16-storage / no-fp8e4nv design.
#include "vkernels/kernels/dsa_kpool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "vkernels/util/error.hpp"

namespace vkernels::kernels {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// Normalized 128-point Hadamard (1/sqrt(128)), the exact stage decomposition
// sglang's kpool_fp8_index.py::_hadamard128 uses (involutory up to fp32
// rounding -- verified offline). Applied to a single 128-vector (the per-pool
// weighted mean), so this is the 1-D form mirroring the Triton kernel's
// reshape(64,2,1)->...->reshape(128) ladder. ``head_dim`` MUST be 128 (the
// GLM-5.3 / DeepSeek-V3 ``index_head_dim``); the loop is fully unrolled and
// bounded so a larger head_dim is a logic error caught up front.
inline void hadamard128(float* x) {
  struct Stage { int groups; int stride; };
  constexpr Stage kStages[] = {{64, 1}, {32, 2}, {16, 4}, {8, 8},
                               {4, 16}, {2, 32}, {1, 64}};
  for (const Stage st : kStages) {
    const int G = st.groups;
    const int S = st.stride;
    const int two_s = 2 * S;
    float tmp[128];
    for (int g = 0; g < G; ++g) {
      const int base = g * two_s;
      for (int k = 0; k < 2; ++k)
        for (int s = 0; s < S; ++s) {
          const float a = x[base + s];
          const float b = x[base + S + s];
          tmp[base + k * S + s] = (k == 0) ? (a + b) : (a - b);
        }
    }
    for (int i = 0; i < 128; ++i) x[i] = tmp[i];
  }
  constexpr float kScale = 0.08838834764831845f;  // 1/sqrt(128)
  for (int i = 0; i < 128; ++i) x[i] *= kScale;
}

// Online (FlashAttention-style) softmax: maintain running max, weight-sum
// denominator and weighted accumulator across ``pool_size`` slots, then
// write the mean (acc/denom) + its Hadamard transform. ``scores``/``keys``
// are [pool_size, head_dim] fp32.
inline void compress_pool(const float* scores, const float* keys,
                          int pool_size, int head_dim, float* out_vec) {
  // head_dim is 128; accumulate the full vector to bound the register traffic
  // on the host (the device kernel lanes it across a wavefront).
  std::vector<float> m(head_dim, kNegInf);
  std::vector<float> acc(head_dim, 0.0f);
  std::vector<float> denom(head_dim, 0.0f);
  for (int slot = 0; slot < pool_size; ++slot) {
    const float* s = scores + slot * head_dim;
    const float* k = keys + slot * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      const float score = s[d];
      const float new_m = std::max(m[d], score);
      const float rescale = std::exp(m[d] - new_m);
      const float prob = std::exp(score - new_m);
      denom[d] = denom[d] * rescale + prob;
      acc[d] = acc[d] * rescale + k[d] * prob;
      m[d] = new_m;
    }
  }
  for (int d = 0; d < head_dim; ++d) out_vec[d] = acc[d] / denom[d];
  if (head_dim == 128) hadamard128(out_vec);
}

}  // namespace

bool dsa_kpool_group_topk_supported(int32_t group_topk) {
  switch (group_topk) {
  case 128:
  case 160:
  case 192:
  case 224:
  case 256:
  case 512:
    return true;
  default:
    return false;
  }
}

int32_t dsa_kpool_max_closed_pools(int32_t num_draft_tokens, int32_t pool_size) {
  if (num_draft_tokens <= 0 || pool_size <= 0) return 0;
  return (num_draft_tokens + pool_size - 1) / pool_size;
}

void dsa_kpool_assemble_cpu(int n_pools, int pool_size, int head_dim,
                            int tail_size, int slots_per_page, int num_pages,
                            int num_chunks, int n_reqs,
                            const float* chunk_k, const float* chunk_score,
                            const float* tail_k, const float* tail_score,
                            const float* ape,
                            const int32_t* req_pool_idx,
                            const int32_t* n_from_tail,
                            const int32_t* chunk_src_start,
                            const int32_t* tail_logical_base,
                            const int32_t* loc, const int32_t* write_mask,
                            float* out) {
  if (n_pools <= 0) return;
  VK_EXPECTS(pool_size > 0 && head_dim == 128 && tail_size > 0 &&
                 slots_per_page > 0 && num_pages > 0 && num_chunks >= 0 &&
                 n_reqs >= 0,
             "dsa_kpool_assemble: positive pool_size/tail_size/"
             "slots_per_page/num_pages, head_dim==128, non-negative counts");
  VK_EXPECTS((num_chunks == 0 || (chunk_k && chunk_score)) &&
                 (n_reqs == 0 || (tail_k && tail_score)) &&
                 ape && req_pool_idx && n_from_tail && chunk_src_start &&
                 tail_logical_base && loc && out,
             "dsa_kpool_assemble: non-null tensors when n_pools>0");
  const int total_slots = num_pages * slots_per_page;
  for (int r = 0; r < n_pools; ++r) {
    if (write_mask && write_mask[r] == 0) continue;  // early return row
    const int req = static_cast<int>(req_pool_idx[r]);
    const int n_tail = static_cast<int>(n_from_tail[r]);
    const int chunk_src = static_cast<int>(chunk_src_start[r]);
    const int tail_base = static_cast<int>(tail_logical_base[r]);
    const int lr = static_cast<int>(loc[r]);
    VK_EXPECTS(req >= 0 && req < n_reqs,
               "dsa_kpool_assemble: req_pool_idx out of range");
    VK_EXPECTS(n_tail >= 0 && n_tail <= pool_size,
               "dsa_kpool_assemble: n_from_tail out of range [0, pool_size]");
    const int n_chunk = pool_size - n_tail;
    if (n_chunk > 0)
      VK_EXPECTS(chunk_src >= 0 && chunk_src + n_chunk - 1 < num_chunks,
                 "dsa_kpool_assemble: chunk_src_start out of range");
    VK_EXPECTS(lr >= 0 && lr < total_slots,
               "dsa_kpool_assemble: loc out of range [0, num_pages*slots_per_page)");

    std::vector<float> scores(pool_size * head_dim);
    std::vector<float> keys(pool_size * head_dim);
    for (int slot = 0; slot < pool_size; ++slot) {
      const float* src_score;
      const float* src_key;
      if (slot < n_tail) {
        const int phys = ((tail_base + slot) % tail_size + tail_size) % tail_size;
        src_score = tail_score + (size_t)req * tail_size * head_dim
                                + (size_t)phys * head_dim;
        src_key = tail_k + (size_t)req * tail_size * head_dim
                          + (size_t)phys * head_dim;
      } else {
        const int ci = chunk_src + (slot - n_tail);
        src_score = chunk_score + (size_t)ci * head_dim;
        src_key = chunk_k + (size_t)ci * head_dim;
      }
      const float* a = ape + slot * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        scores[slot * head_dim + d] = src_score[d] + a[d];
        keys[slot * head_dim + d] = src_key[d];
      }
    }
    float out_vec[128];
    compress_pool(scores.data(), keys.data(), pool_size, head_dim, out_vec);
    const int page = lr / slots_per_page;
    const int slot_in_page = lr % slots_per_page;
    float* dst = out + (size_t)page * slots_per_page * head_dim
                       + (size_t)slot_in_page * head_dim;
    for (int d = 0; d < head_dim; ++d) dst[d] = out_vec[d];
  }
}

void dsa_kpool_decode_update_cpu(int batch, int pool_size, int head_dim,
                                 int tail_size, int slots_per_page,
                                 int block_table_cols, int n_reqs,
                                 int num_pages,
                                 const float* key, const float* slot_score,
                                 float* tail_k, float* tail_score,
                                 const float* ape,
                                 const int32_t* block_tables,
                                 const int32_t* req_pool_indices,
                                 const int32_t* positions,
                                 const int32_t* seq_lens,
                                 const int32_t* out_cache_loc, float* out) {
  if (batch <= 0) return;
  VK_EXPECTS(pool_size > 0 && head_dim == 128 && tail_size > 0 &&
                 slots_per_page > 0 && block_table_cols > 0 && n_reqs >= 0 &&
                 num_pages >= 0,
             "dsa_kpool_decode_update: positive pool_size/tail_size/"
             "slots_per_page/block_table_cols, head_dim==128, "
             "non-negative counts");
  VK_EXPECTS(key && slot_score && tail_k && tail_score && ape && block_tables &&
                 req_pool_indices && positions && seq_lens &&
                 out_cache_loc && out,
             "dsa_kpool_decode_update: non-null tensors when batch>0");
  for (int r = 0; r < batch; ++r) {
    const int req_raw = static_cast<int>(req_pool_indices[r]);
    const bool req_valid = req_raw >= 0 && req_raw < n_reqs;
    const int req = req_valid ? req_raw
                              : std::min(std::max(req_raw, 0), n_reqs - 1);
    const int pos_raw = static_cast<int>(positions[r]);
    const int safe_pos = std::max(pos_raw, 0);
    const int seq_len = static_cast<int>(seq_lens[r]);
    const int cache_loc = static_cast<int>(out_cache_loc[r]);
    const bool pos_valid = req_valid && cache_loc != 0 && pos_raw >= 0 &&
                           pos_raw < seq_len;
    const int slot = safe_pos % pool_size;
    const int phys_slot = ((safe_pos % tail_size) + tail_size) % tail_size;

    const float* cur_k = key + (size_t)r * head_dim;
    const float* cur_s = slot_score + (size_t)r * head_dim;

    // Compress the COMPLETE pool (slot == pool_size-1 and pos_valid).
    if (pos_valid && slot == pool_size - 1) {
      const int pool_logical_start = safe_pos - slot;
      std::vector<float> scores(pool_size * head_dim);
      std::vector<float> keys(pool_size * head_dim);
      // Pass 1: running max (mirrors the Triton kernel exactly).
      std::vector<float> max_score(head_dim, kNegInf);
      for (int p = 0; p < pool_size; ++p) {
        const bool is_current = (p == slot);
        const int phys =
            ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
        const float* s_buf = tail_score + (size_t)req * tail_size * head_dim
                                          + (size_t)phys * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          const float s = (is_current ? cur_s[d] : s_buf[d]) + ape[p * head_dim + d];
          max_score[d] = std::max(max_score[d], s);
        }
      }
      // Pass 2: weighted sum under the fixed max.
      for (int p = 0; p < pool_size; ++p) {
        const bool is_current = (p == slot);
        const int phys =
            ((pool_logical_start + p) % tail_size + tail_size) % tail_size;
        const float* s_buf = tail_score + (size_t)req * tail_size * head_dim
                                          + (size_t)phys * head_dim;
        const float* k_buf = tail_k + (size_t)req * tail_size * head_dim
                                      + (size_t)phys * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          const float s = (is_current ? cur_s[d] : s_buf[d]) + ape[p * head_dim + d];
          const float prob = std::exp(s - max_score[d]);
          scores[p * head_dim + d] = prob;
          keys[p * head_dim + d] = (is_current ? cur_k[d] : k_buf[d]) * prob;
        }
      }
      // Recompute mean directly from the fixed-max probs (online would land
      // at the same value; the two-pass form is what the kernel does).
      std::vector<float> acc(head_dim, 0.0f), denom(head_dim, 0.0f);
      for (int p = 0; p < pool_size; ++p)
        for (int d = 0; d < head_dim; ++d) {
          denom[d] += scores[p * head_dim + d];
          acc[d] += keys[p * head_dim + d];
        }
      float out_vec[128];
      for (int d = 0; d < head_dim; ++d) out_vec[d] = acc[d] / denom[d];
      if (head_dim == 128) hadamard128(out_vec);

      const int pool_id = safe_pos / pool_size;
      const int pool_page_group = pool_id / slots_per_page;
      int token_page_row = pool_page_group * pool_size;
      token_page_row = std::min(std::max(token_page_row, 0), block_table_cols - 1);
      const int packed_page =
          static_cast<int>(block_tables[(size_t)r * block_table_cols +
                                        token_page_row]);
      const int loc_slot = pool_id % slots_per_page;
      VK_EXPECTS(packed_page >= 0 && packed_page < num_pages,
                 "dsa_kpool_decode_update: block_tables entry out of range "
                 "[0, num_pages)");
      float* dst = out + (size_t)packed_page * slots_per_page * head_dim
                         + (size_t)loc_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) dst[d] = out_vec[d];
    }

    // Unconditional live-tail write (masked to a no-op when !pos_valid).
    if (pos_valid) {
      float* tk_dst = tail_k + (size_t)req * tail_size * head_dim
                              + (size_t)phys_slot * head_dim;
      float* ts_dst = tail_score + (size_t)req * tail_size * head_dim
                                  + (size_t)phys_slot * head_dim;
      for (int d = 0; d < head_dim; ++d) {
        tk_dst[d] = cur_k[d];
        ts_dst[d] = cur_s[d];
      }
    }
  }
}

}  // namespace vkernels::kernels
