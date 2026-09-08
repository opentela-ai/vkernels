// vkernels/capi/capi_attn.cpp — attention-family entry points of the C ABI
// declared in capi.hpp: MLA, DSA (sparse fwd / top-k transform / paged-MQA
// top-k logits / kpool cache), MHC and KDA. Error plumbing and the other
// domain TUs: see capi_internal.hpp.
#include <cstddef>
#include <cstdint>

#include "vkernels/capi/capi.hpp"
#include "vkernels/capi/capi_internal.hpp"

#include "vkernels/kernels/mla.hpp"
#include "vkernels/kernels/dsa.hpp"
#include "vkernels/kernels/dsa_topk.hpp"
#include "vkernels/kernels/dsa_kpool.hpp"
#include "vkernels/kernels/mhc.hpp"
#include "vkernels/kernels/kda.hpp"

extern "C" {

/* ------------------------------------------------------------------ */
/* kernels: MLA — absorbed-form Multi-head Latent Attention (mla.hpp)   */
/* ------------------------------------------------------------------ */

int32_t vk_mla_fwd(int B, int H, int S_q, int S_kv, int q_start,
                   int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                   float scale, const float* q, const float* k_c,
                   const float* k_pe, const float* v_c, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::mla_fwd_cpu(B, H, S_q, S_kv, q_start, kv_start,
                                 kv_lora_rank, qk_rope_head_dim, scale, q,
                                 k_c, k_pe, v_c, out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

void vk_mla_config(int S_q, int kv_lora_rank, int qk_rope_head_dim,
                   int* bq, int* bn, int* threads) {
  vkernels::kernels::mla_config_for(S_q, kv_lora_rank, qk_rope_head_dim, bq,
                                    bn, threads);
}

/* ------------------------------------------------------------------ */
/* kernels: DSA — DeepseekSparseAttn sparse-MLA forward (dsa.hpp)   */
/* ------------------------------------------------------------------ */

int32_t vk_dsa_sparse_fwd(int S_q, int S_kv, int H, int dim, int tail_dim,
                          int topk, int kv_group, int block_I,
                          int inner_iter, float sm_scale, int return_lse,
                          const float* q, const float* kv,
                          const int32_t* indices, float* out, float* lse) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_sparse_fwd_cpu(
      S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I, inner_iter,
      sm_scale, return_lse != 0, q, kv, indices, out, lse);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_dsa_topk_transform(
    int32_t batch_size, const float* score, const int32_t* lengths,
    int32_t* dst_token_indices, int64_t score_stride, int32_t pool_size,
    int32_t token_topk, int32_t out_cols, const int32_t* page_table,
    int64_t page_table_stride, const int32_t* page_table_row_index,
    const int32_t* topk_indices_offset, const int32_t* row_starts,
    const int32_t* seq_lens) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_topk_transform_cpu(
      batch_size, score, lengths, dst_token_indices, score_stride, pool_size,
      token_topk, out_cols, page_table, page_table_stride, page_table_row_index,
      topk_indices_offset, row_starts, seq_lens);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int vk_dsa_topk_group_topk_supported(int32_t group_topk) {
  return vkernels::kernels::dsa_topk_transform_group_topk_supported(group_topk)
             ? 1
             : 0;
}

void vk_dsa_config(int S_q, int H, int dim, int topk, int* bq,
                   int* threads, int* block_I, int* inner_iter) {
  vkernels::kernels::dsa_config_for(S_q, H, dim, topk, bq, threads, block_I,
                                    inner_iter);
}

/* ------------------------------------------------------------------ */
/* kernels: DSA — paged-MQA gated top-k logits (dsa.hpp, #51)         */
/* ------------------------------------------------------------------ */

int32_t vk_dsa_topk_logits(int batch_size, int num_heads, int head_dim,
                           int block, int max_table_len, int num_blocks,
                           const float* q, const float* kv,
                           const float* k_scale, const float* gate,
                           const int32_t* seq_lens,
                           const int32_t* page_table, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_topk_logits_cpu(batch_size, num_heads, head_dim,
                                         block, max_table_len, num_blocks,
                                         q, kv, k_scale, gate, seq_lens,
                                         page_table, out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_dsa_topk_logits_split_for(int batch_size, int max_seq_len,
                                     int block, int* split_kv) {
  VK_CAPI_TRY
  if (split_kv == nullptr) return VK_ERROR_INVALID_ARGUMENT;
  *split_kv = vkernels::kernels::dsa_topk_logits_split_for(batch_size,
                                                           max_seq_len, block);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* kernels: DSA — kpool-cache compress/write (dsa_kpool.hpp, #60)    */
/* ------------------------------------------------------------------ */

int vk_dsa_kpool_group_topk_supported(int32_t group_topk) {
  return vkernels::kernels::dsa_kpool_group_topk_supported(group_topk) ? 1 : 0;
}

int32_t vk_dsa_kpool_max_closed_pools(int32_t num_draft_tokens,
                                       int32_t pool_size) {
  return vkernels::kernels::dsa_kpool_max_closed_pools(num_draft_tokens,
                                                       pool_size);
}

int32_t vk_dsa_kpool_assemble(int n_pools, int pool_size, int head_dim,
                              int tail_size, int slots_per_page,
                              int num_pages, int num_chunks, int n_reqs,
                              const float* chunk_k, const float* chunk_score,
                              const float* tail_k, const float* tail_score,
                              const float* ape, const int32_t* req_pool_idx,
                              const int32_t* n_from_tail,
                              const int32_t* chunk_src_start,
                              const int32_t* tail_logical_base,
                              const int32_t* loc, const int32_t* write_mask,
                              float* out) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_kpool_assemble_cpu(
      n_pools, pool_size, head_dim, tail_size, slots_per_page, num_pages,
      num_chunks, n_reqs, chunk_k, chunk_score, tail_k, tail_score, ape,
      req_pool_idx, n_from_tail, chunk_src_start, tail_logical_base, loc,
      write_mask, out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_dsa_kpool_decode_update(int batch, int pool_size, int head_dim,
                                   int tail_size, int slots_per_page,
                                   int block_table_cols, int n_reqs,
                                   int num_pages, const float* key,
                                   const float* slot_score, float* tail_k,
                                   float* tail_score, const float* ape,
                                   const int32_t* block_tables,
                                   const int32_t* req_pool_indices,
                                   const int32_t* positions,
                                   const int32_t* seq_lens,
                                   const int32_t* out_cache_loc, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_kpool_decode_update_cpu(
      batch, pool_size, head_dim, tail_size, slots_per_page, block_table_cols,
      n_reqs, num_pages, key, slot_score, tail_k, tail_score, ape,
      block_tables, req_pool_indices, positions, seq_lens, out_cache_loc, out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_dsa_kpool_assemble_fp8(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const float* chunk_k, const float* chunk_score, const float* tail_k,
    const float* tail_score, const float* ape, const int32_t* req_pool_idx,
    const int32_t* n_from_tail, const int32_t* chunk_src_start,
    const int32_t* tail_logical_base, const int32_t* loc,
    const int32_t* write_mask, uint8_t* cache_u8,
    float* round_scale_or_null) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_kpool_assemble_fp8_cpu(
      n_pools, pool_size, head_dim, tail_size, slots_per_page, num_pages,
      num_chunks, n_reqs, chunk_k, chunk_score, tail_k, tail_score, ape,
      req_pool_idx, n_from_tail, chunk_src_start, tail_logical_base, loc,
      write_mask, cache_u8, round_scale_or_null);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_dsa_kpool_decode_update_fp8(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const float* key, const float* slot_score, float* tail_k,
    float* tail_score, const float* ape, const int32_t* block_tables,
    const int32_t* req_pool_indices, const int32_t* positions,
    const int32_t* seq_lens, const int32_t* out_cache_loc, uint8_t* cache_u8,
    float* round_scale_or_null) {
  VK_CAPI_TRY
  vkernels::kernels::dsa_kpool_decode_update_fp8_cpu(
      batch, pool_size, head_dim, tail_size, slots_per_page, block_table_cols,
      n_reqs, num_pages, key, slot_score, tail_k, tail_score, ape,
      block_tables, req_pool_indices, positions, seq_lens, out_cache_loc,
      cache_u8, round_scale_or_null);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* kernels: MHC — multi-head hybrid-attention pre-norm (mhc.hpp, #51)  */
/* ------------------------------------------------------------------ */

int32_t vk_mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult, int hidden_size,
                               const float* x, const float* fn,
                               float* out, float* sqrsum) {
  VK_CAPI_TRY
  vkernels::kernels::mhc_pre_gemm_sqrsum_cpu(num_tokens, hc_mult, hidden_size,
                                             x, fn, out, sqrsum);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_mhc_post(int num_tokens, int hc, int hidden,
                    const float* a, const float* b, const float* c,
                    const float* d, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::mhc_post_cpu(num_tokens, hc, hidden, a, b, c, d, out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* kernels: KDA — Kimi Delta Attention (kda.hpp)                        */
/* ------------------------------------------------------------------ */

int32_t vk_kda_layer_norm_gated(const float* x, const float* weight,
                                const float* gate, float* out, int N,
                                int D, float eps) {
  VK_CAPI_TRY
  vkernels::kernels::kda_layer_norm_gated_cpu(x, weight, gate, out, N, D,
                                              eps);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_gate_chunk_cumsum(const float* g, float* intra_log,
                                 float* inter_log, int B, int H,
                                 int n_chunks, int chunk_size) {
  VK_CAPI_TRY
  vkernels::kernels::kda_gate_chunk_cumsum_cpu(g, intra_log, inter_log, B,
                                               H, n_chunks, chunk_size);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_naive_delta_rule_fwd(const float* q, const float* k,
                                    const float* v, const float* g,
                                    const float* beta, float* out, int B,
                                    int H, int S, int D) {
  VK_CAPI_TRY
  vkernels::kernels::kda_naive_delta_rule_fwd_cpu(q, k, v, g, beta, out, B,
                                                  H, S, D);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_delta_rule_fwd(const float* q, const float* k,
                              const float* v, const float* g,
                              const float* beta, float* out, int B, int H,
                              int S, int D, int chunk_size) {
  VK_CAPI_TRY
  vkernels::kernels::kda_delta_rule_fwd_cpu(q, k, v, g, beta, out, B, H,
                                            S, D, chunk_size);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_delta_rule_intra(const float* q, const float* k,
                                const float* v, const float* g,
                                const float* beta, const float* intra_log,
                                const float* inter_state, float* u, int B,
                                int H, int S, int D, int chunk_size,
                                int chunk_idx) {
  VK_CAPI_TRY
  vkernels::kernels::kda_delta_rule_intra_cpu(q, k, v, g, beta, intra_log,
                                              inter_state, u, B, H, S, D,
                                              chunk_size, chunk_idx);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_delta_rule_inter(const float* k, const float* v,
                                const float* g, const float* beta,
                                const float* intra_log, const float* u,
                                float* inter_state, int B, int H, int S,
                                int D, int chunk_size, int chunk_idx) {
  VK_CAPI_TRY
  vkernels::kernels::kda_delta_rule_inter_cpu(k, v, g, beta, intra_log, u,
                                              inter_state, B, H, S, D,
                                              chunk_size, chunk_idx);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_gla_fwd_o(const float* q, const float* k, const float* g,
                         const float* beta, const float* intra_log,
                         const float* inter_state, const float* u,
                         float* out, int B, int H, int S, int D,
                         int chunk_size) {
  VK_CAPI_TRY
  vkernels::kernels::kda_gla_fwd_o_cpu(q, k, g, beta, intra_log,
                                       inter_state, u, out, B, H, S, D,
                                       chunk_size);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_kda_pack_bitmatrix(const uint8_t* bits, uint8_t* packed,
                              size_t n_bits) {
  VK_CAPI_TRY
  vkernels::kernels::kda_pack_bitmatrix_cpu(bits, packed, n_bits);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

}  // extern "C"
