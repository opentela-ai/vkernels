// vkernels/capi/capi.cpp — implementation of the C ABI declared in capi.hpp.
//
// Every exported function wraps the C++ API in a try/catch: exceptions (the
// VK_EXPECTS / VK_ENSURES contract checks, std::bad_alloc) cannot cross the
// C ABI, so they are translated into a status code plus a thread-local
// message readable via vk_last_error() / vk_last_error_code(). See capi.hpp
// for the full contract.
#include "vkernels/capi/capi.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "vkernels/comm/allreduce.hpp"
#include "vkernels/comm/channel.hpp"
#include "vkernels/comm/overlap.hpp"
#include "vkernels/comm/p2p_gather.hpp"
#include "vkernels/comm/topology.hpp"
#include "vkernels/core/device.hpp"
#include "vkernels/core/stream.hpp"
#include "vkernels/kernels/elementwise.hpp"
#include "vkernels/kernels/gemm.hpp"
#include "vkernels/kernels/gemm_bf16.hpp"
#include "vkernels/kernels/kda.hpp"
#include "vkernels/kernels/mla.hpp"
#include "vkernels/kernels/dsa.hpp"
#include "vkernels/kernels/dsa_topk.hpp"
#include "vkernels/kernels/mhc.hpp"
#include "vkernels/kernels/moe.hpp"
#include "vkernels/kernels/moe_aux.hpp"
#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/reduce.hpp"
#include "vkernels/util/version.hpp"

namespace {

// Most recent error message on the calling thread; valid until the next
// failing call on the same thread.
thread_local std::string g_last_error;

// Status code of the most recent failing call (VK_OK when none).
thread_local int g_last_error_code = VK_OK;

// The C++ library's contract checks throw std::invalid_argument (VK_EXPECTS)
// and std::runtime_error (VK_ENSURES); allocations can throw std::bad_alloc.
// Translate each to the status codes declared in capi.hpp. Both macros set
// the thread-local code/message so vk_last_error_code() is always accurate.
#define VK_CAPI_TRY try {
#define VK_CAPI_CATCH_RETURN_CODE()                                    \
  }                                                                    \
  catch (const std::invalid_argument& e) {                             \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return VK_ERROR_INVALID_ARGUMENT;                                  \
  }                                                                    \
  catch (const std::out_of_range& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_OUT_OF_RANGE;                         \
    return VK_ERROR_OUT_OF_RANGE;                                      \
  }                                                                    \
  catch (const std::length_error& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return VK_ERROR_INVALID_ARGUMENT;                                  \
  }                                                                    \
  catch (const std::runtime_error& e) {                                \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (const std::bad_alloc&) {                                      \
    g_last_error = "out of memory";                                    \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (const std::exception& e) {                                    \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }                                                                    \
  catch (...) {                                                        \
    g_last_error = "unknown C++ exception";                            \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return VK_ERROR_INTERNAL;                                          \
  }

// Catch variant for handle-returning functions: report the error and return
// nullptr instead of a status code.
#define VK_CAPI_CATCH_RETURN_NULL()                                    \
  }                                                                    \
  catch (const std::invalid_argument& e) {                             \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::out_of_range& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_OUT_OF_RANGE;                         \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::length_error& e) {                                 \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INVALID_ARGUMENT;                     \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::runtime_error& e) {                                \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::bad_alloc&) {                                      \
    g_last_error = "out of memory";                                    \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }                                                                    \
  catch (const std::exception& e) {                                    \
    g_last_error = e.what();                                           \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }                                                                    \
  catch (...) {                                                        \
    g_last_error = "unknown C++ exception";                            \
    g_last_error_code = VK_ERROR_INTERNAL;                             \
    return nullptr;                                                    \
  }

// Opaque handle types: heap-allocated wrappers around the C++ objects.
struct vk_device_impl {
  vkernels::Device d;
};
struct vk_stream_impl {
  vkernels::Stream s;
};
struct vk_queue_impl {
  std::shared_ptr<vkernels::comm::BlockingQueue> q;
};
struct vk_channel_impl {
  std::shared_ptr<vkernels::comm::Channel> ch;
};
struct vk_overlap_impl {
  vkernels::comm::OverlapExecutor ex;
};

// Copy a vector into a malloc'd buffer of the same element count; the caller
// owns the result (release with vk_free). Returns nullptr on failure only
// when the vector is non-empty (g_last_error is set in that case).
template <typename T>
void* malloc_copy(const std::vector<T>& v) {
  void* p = std::malloc(v.size() * sizeof(T));
  if (p == nullptr && !v.empty()) {
    g_last_error = "malloc failed";         // LCOV_EXCL_LINE
    g_last_error_code = VK_ERROR_INTERNAL;  // LCOV_EXCL_LINE
    return nullptr;                         // LCOV_EXCL_LINE
  }
  if (!v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  return p;
}

}  // namespace

extern "C" {

/* ------------------------------------------------------------------ */
/* Version / config                                                    */
/* ------------------------------------------------------------------ */

const char* vk_version(void) { return VKERNELS_VERSION_STRING; }

int vk_has_cuda(void) { return VKERNELS_HAS_CUDA; }

const char* vk_last_error(void) { return g_last_error.c_str(); }

int vk_last_error_code(void) { return g_last_error_code; }

void vk_free(void* p) { std::free(p); }

/* ------------------------------------------------------------------ */
/* kernels                                                             */
/* ------------------------------------------------------------------ */

int32_t vk_add(const float* a, size_t a_len, const float* b, size_t b_len,
               float* out, size_t out_len) {
  VK_CAPI_TRY
  vkernels::kernels::add(vkernels::Span<const float>(a, a_len),
                         vkernels::Span<const float>(b, b_len),
                         vkernels::Span<float>(out, out_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_scale(const float* x, size_t x_len, float alpha, float* out,
                 size_t out_len) {
  VK_CAPI_TRY
  vkernels::kernels::scale(vkernels::Span<const float>(x, x_len), alpha,
                           vkernels::Span<float>(out, out_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_relu(const float* x, size_t x_len, float* out, size_t out_len) {
  VK_CAPI_TRY
  vkernels::kernels::relu(vkernels::Span<const float>(x, x_len),
                          vkernels::Span<float>(out, out_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_sum(const float* x, size_t x_len, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::sum(vkernels::Span<const float>(x, x_len), *out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_max(const float* x, size_t x_len, float* out) {
  VK_CAPI_TRY
  vkernels::kernels::max(vkernels::Span<const float>(x, x_len), *out);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_gemm(size_t M, size_t N, size_t K, float alpha, const float* A,
                size_t A_len, const float* B, size_t B_len, float beta,
                float* C, size_t C_len) {
  VK_CAPI_TRY
  vkernels::kernels::gemm(M, N, K, alpha, vkernels::Span<const float>(A, A_len),
                          vkernels::Span<const float>(B, B_len), beta,
                          vkernels::Span<float>(C, C_len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* kernels: gfx942 primitives (moe.hpp)                                 */
/* ------------------------------------------------------------------ */

int32_t vk_direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                                size_t elements) {
  VK_CAPI_TRY
  vkernels::kernels::direct_lds_fill_bf16(lds_dst, global_src, elements);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_fp4_to_bf16_dequant(const uint8_t* packed, size_t packed_len,
                               uint16_t* out, size_t out_len, float scale) {
  VK_CAPI_TRY
  vkernels::kernels::fp4_to_bf16_dequant(
      vkernels::Span<const std::uint8_t>(packed, packed_len),
      vkernels::Span<std::uint16_t>(out, out_len), scale);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int vk_use_async_copy_default(void) {
  return vkernels::kernels::use_async_copy_default() ? 1 : 0;
}

int32_t vk_mfma_f32_16x16x16bf16(float* c, const uint32_t* a,
                                 const uint32_t* b, int cbsz, int abid,
                                 int blgp) {
  VK_CAPI_TRY
  vkernels::kernels::mfma_f32_16x16x16bf16(c, a, b, cbsz, abid, blgp);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* kernels: bf16 GEMM (gemm_bf16.hpp, issue #29)                        */
/* ------------------------------------------------------------------ */

int32_t vk_gemm_bf16(size_t M, size_t N, size_t K, float alpha,
                     const uint16_t* A, const uint16_t* B, float beta,
                     uint16_t* C) {
  VK_CAPI_TRY
  vkernels::kernels::gemm_bf16_cpu(M, N, K, alpha, A, B, beta, C);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

void vk_gemm_bf16_config(size_t M, size_t N, size_t K, int* bm, int* bn,
                         int* bk, int* threads) {
  vkernels::kernels::gemm_bf16_config_for(M, N, K, bm, bn, bk, threads);
}

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

/* ------------------------------------------------------------------ */
/* kernels: MoE orchestration (moe_aux.hpp, issue #22)                  */
/* ------------------------------------------------------------------ */

int32_t vk_mxfp4_moe_quant(const uint16_t* A, uint8_t* packed,
                           uint8_t* scales, int M, int hidden,
                           int group_size) {
  VK_CAPI_TRY
  vkernels::kernels::mxfp4_moe_quant(A, packed, scales, M, hidden,
                                     group_size);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                          uint16_t* A_sorted, int M, int hidden, int top_k,
                          int EM) {
  VK_CAPI_TRY
  vkernels::kernels::mxfp4_moe_sort(A, sorted_ids, A_sorted, M, hidden,
                                    top_k, EM);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_mxfp4_moe_sort_scales(const uint8_t* scales,
                                 const int32_t* sorted_ids,
                                 uint8_t* scales_sorted, int M,
                                 int n_groups, int top_k, int EM) {
  VK_CAPI_TRY
  vkernels::kernels::mxfp4_moe_sort_scales(scales, sorted_ids, scales_sorted,
                                           M, n_groups, top_k, EM);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_mxfp4_moe_scatter_reduce(const float* partial,
                                    const float* topk_w,
                                    const int32_t* sorted_ids, float* out,
                                    int M, int width, int top_k, int EM) {
  VK_CAPI_TRY
  vkernels::kernels::mxfp4_moe_scatter_reduce(partial, topk_w, sorted_ids,
                                              out, M, width, top_k, EM);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                      const uint8_t* partial_s,
                                      const float* topk_w,
                                      const int32_t* sorted_ids, float* out,
                                      int M, int width, int top_k, int EM,
                                      int group_size) {
  VK_CAPI_TRY
  vkernels::kernels::mxfp4_moe_scatter_reduce_q(partial_q, partial_s,
                                                topk_w, sorted_ids, out, M,
                                                width, top_k, EM, group_size);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* kernels: fused MXFP4 MoE grouped GEMM (moe_fused.hpp)               */
/* ------------------------------------------------------------------ */

int32_t vk_fused_moe_mxfp4(const uint16_t* A, const uint8_t* w13,
                           const uint8_t* w13_scale, const uint8_t* w2,
                           const uint8_t* w2_scale, const int32_t* sorted_ids,
                           const float* topk_w_sorted,
                           const int32_t* expert_ids, uint16_t* act_scratch,
                           float* out, int M, int hidden, int ispp, int top_k,
                           int EM, int group_size, float swiglu_limit,
                           int activation, float beta, float linear_beta,
                           const float* b13, const float* b2) {
  VK_CAPI_TRY
  // The C++ reference (fused_moe_mxfp4_cpu) does no length checks of its
  // own (BLAS-style); the safe bindings validate shapes before calling.
  // Guard the primary buffers here so a C consumer cannot dereference null.
  if (M > 0 && EM > 0) {
    if (A == nullptr || w13 == nullptr || w13_scale == nullptr ||
        w2 == nullptr || w2_scale == nullptr || sorted_ids == nullptr ||
        topk_w_sorted == nullptr || expert_ids == nullptr ||
        act_scratch == nullptr || out == nullptr) {
      throw std::invalid_argument("fused_moe_mxfp4: buffers must not be null");
    }
  }
  vkernels::kernels::fused_moe_mxfp4_cpu(
      A, w13, w13_scale, w2, w2_scale, sorted_ids, topk_w_sorted, expert_ids,
      act_scratch, out, M, hidden, ispp, top_k, EM, group_size, swiglu_limit,
      activation, beta, linear_beta, b13, b2);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

size_t vk_moe_align_block_size_max_em(int32_t M, int32_t top_k,
                                      int32_t block_size,
                                      int32_t num_experts) {
  if (block_size <= 0) return 0;
  int32_t N = M * top_k;
  int32_t max_em = ((N + block_size - 1) / block_size + num_experts) * block_size;
  return max_em > 0 ? static_cast<size_t>(max_em) : 0;
}

int32_t vk_moe_align_block_size(const int32_t* topk_ids, int32_t M,
                                int32_t top_k, int32_t block_size,
                                int32_t num_experts, int32_t* sorted_ids,
                                int32_t* expert_ids, int32_t* out_EM) {
  VK_CAPI_TRY
  // The C++ (moe_align_block_size) does no contract checks of its own; the
  // safe bindings validate shapes before calling. Guard the buffers here so
  // a C consumer cannot dereference null. (M == 0 is a valid no-op that
  // writes one all-padding block when num_experts > 0.)
  if (block_size <= 0) throw std::invalid_argument("block_size must be positive");
  if (num_experts < 0) throw std::invalid_argument("num_experts must be non-negative");
  if (M < 0 || top_k < 0) throw std::invalid_argument("M and top_k must be non-negative");
  if (out_EM == nullptr) throw std::invalid_argument("out_EM must not be null");
  if (M > 0 && (topk_ids == nullptr || sorted_ids == nullptr || expert_ids == nullptr)) {
    throw std::invalid_argument("topk_ids/sorted_ids/expert_ids must not be null");
  }
  const int32_t EM = vkernels::kernels::moe_align_block_size(
      topk_ids, M, top_k, block_size, num_experts, sorted_ids, expert_ids);
  *out_EM = EM;
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* core: device + stream                                               */
/* ------------------------------------------------------------------ */

vk_device* vk_device_new(int index) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_device*>(new vk_device_impl{vkernels::Device(index)});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_device_delete(vk_device* d) { delete reinterpret_cast<vk_device_impl*>(d); }

int vk_device_index(const vk_device* d) { return reinterpret_cast<const vk_device_impl*>(d)->d.index(); }

int vk_device_supports_peer(const vk_device* d, const vk_device* other) {
  return reinterpret_cast<const vk_device_impl*>(d)->d.supports_peer(
             reinterpret_cast<const vk_device_impl*>(other)->d)
             ? 1
             : 0;
}

int vk_device_eq(const vk_device* a, const vk_device* b) {
  return reinterpret_cast<const vk_device_impl*>(a)->d ==
                 reinterpret_cast<const vk_device_impl*>(b)->d
             ? 1
             : 0;
}

int32_t vk_device_set_current(vk_device* d) {
  VK_CAPI_TRY
  reinterpret_cast<vk_device_impl*>(d)->d.set_current();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_device_sync(vk_device* d) {
  VK_CAPI_TRY
  reinterpret_cast<vk_device_impl*>(d)->d.sync();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

vk_stream* vk_stream_new(void) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_stream*>(new vk_stream_impl{vkernels::Stream()});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_stream_delete(vk_stream* s) { delete reinterpret_cast<vk_stream_impl*>(s); }

int32_t vk_stream_submit(vk_stream* s, void (*fn)(void*), void* ctx) {
  VK_CAPI_TRY
  reinterpret_cast<vk_stream_impl*>(s)->s.submit([fn, ctx]() { fn(ctx); });
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

void vk_stream_wait(vk_stream* s) { reinterpret_cast<vk_stream_impl*>(s)->s.wait(); }

size_t vk_stream_submitted(const vk_stream* s) { return reinterpret_cast<const vk_stream_impl*>(s)->s.submitted(); }

/* ------------------------------------------------------------------ */
/* comm: topology                                                      */
/* ------------------------------------------------------------------ */

int32_t vk_ring_rank(int32_t rank, int32_t world, vk_topology* out) {
  VK_CAPI_TRY
  const vkernels::comm::Topology t = vkernels::comm::ring_rank(rank, world);
  out->rank = t.rank;
  out->world = t.world;
  out->next = t.next;
  out->prev = t.prev;
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_build_ring_topology(int32_t world, vk_topology** out,
                               size_t* out_count) {
  VK_CAPI_TRY
  const std::vector<vkernels::comm::Topology> ts =
      vkernels::comm::build_ring_topology(world);
  vk_topology* arr = static_cast<vk_topology*>(malloc_copy(ts));
  if (arr == nullptr && !ts.empty()) return VK_ERROR_INTERNAL;
  *out = arr;
  *out_count = ts.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: channels                                                      */
/* ------------------------------------------------------------------ */

vk_queue* vk_queue_new(void) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_queue*>(new vk_queue_impl{std::make_shared<vkernels::comm::BlockingQueue>()});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_queue_delete(vk_queue* q) { delete reinterpret_cast<vk_queue_impl*>(q); }

int32_t vk_queue_push(vk_queue* q, const float* data, size_t len) {
  VK_CAPI_TRY
  reinterpret_cast<vk_queue_impl*>(q)->q->push(std::vector<float>(data, data + len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_queue_pop(vk_queue* q, float** out_data, size_t* out_len) {
  VK_CAPI_TRY
  const std::vector<float> v = reinterpret_cast<vk_queue_impl*>(q)->q->pop();
  float* p = static_cast<float*>(malloc_copy(v));
  if (p == nullptr && !v.empty()) return VK_ERROR_INTERNAL;
  *out_data = p;
  *out_len = v.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

void vk_queue_close(vk_queue* q) { reinterpret_cast<vk_queue_impl*>(q)->q->close(); }

int vk_queue_closed(const vk_queue* q) { return reinterpret_cast<const vk_queue_impl*>(q)->q->closed() ? 1 : 0; }

vk_channel* vk_channel_new(vk_queue* out, vk_queue* in) {
  VK_CAPI_TRY
  if (out == nullptr || in == nullptr) {
    throw std::invalid_argument("MockChannel needs both queues");
  }
  return reinterpret_cast<vk_channel*>(new vk_channel_impl{
      std::make_shared<vkernels::comm::MockChannel>(
          reinterpret_cast<vk_queue_impl*>(out)->q,
          reinterpret_cast<vk_queue_impl*>(in)->q)});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_channel_delete(vk_channel* c) { delete reinterpret_cast<vk_channel_impl*>(c); }

int32_t vk_channel_send(vk_channel* c, const float* data, size_t len) {
  VK_CAPI_TRY
  reinterpret_cast<vk_channel_impl*>(c)->ch->send(std::vector<float>(data, data + len));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_channel_recv(vk_channel* c, float** out_data, size_t* out_len) {
  VK_CAPI_TRY
  const std::vector<float> v = reinterpret_cast<vk_channel_impl*>(c)->ch->recv();
  float* p = static_cast<float*>(malloc_copy(v));
  if (p == nullptr && !v.empty()) return VK_ERROR_INTERNAL;
  *out_data = p;
  *out_len = v.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int vk_channel_closed(const vk_channel* c) { return reinterpret_cast<const vk_channel_impl*>(c)->ch->closed() ? 1 : 0; }

int32_t vk_make_ring_channels(int32_t world, vk_channel*** out,
                              size_t* out_count) {
  VK_CAPI_TRY
  std::vector<std::unique_ptr<vkernels::comm::Channel>> channels =
      vkernels::comm::make_ring_channels(world);
  vk_channel** arr = static_cast<vk_channel**>(
      std::malloc(channels.size() * sizeof(vk_channel*)));
  if (arr == nullptr) {
    g_last_error = "malloc failed";         // LCOV_EXCL_LINE
    g_last_error_code = VK_ERROR_INTERNAL;  // LCOV_EXCL_LINE
    return VK_ERROR_INTERNAL;               // LCOV_EXCL_LINE
  }
  for (std::size_t i = 0; i < channels.size(); ++i) {
    vk_channel_impl* c = new vk_channel_impl{};
    c->ch = std::shared_ptr<vkernels::comm::Channel>(std::move(channels[i]));
    arr[i] = reinterpret_cast<vk_channel*>(c);
  }
  *out = arr;
  *out_count = channels.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: ring all-reduce                                               */
/* ------------------------------------------------------------------ */

int32_t vk_ring_allreduce_rank(float* local, size_t local_len, int32_t rank,
                               int32_t world, vk_channel* next,
                               vk_channel* prev) {
  VK_CAPI_TRY
  // ring_allreduce_rank takes ownership of a std::vector<float>, so copy the
  // caller's buffer in and out; on error the caller's buffer is untouched.
  std::vector<float> buf(local, local + local_len);
  vkernels::comm::ring_allreduce_rank(
      buf, rank, world, *reinterpret_cast<vk_channel_impl*>(next)->ch,
      *reinterpret_cast<vk_channel_impl*>(prev)->ch);
  std::memcpy(local, buf.data(), buf.size() * sizeof(float));
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: overlap                                                       */
/* ------------------------------------------------------------------ */

vk_overlap* vk_overlap_new(void) {
  VK_CAPI_TRY
  return reinterpret_cast<vk_overlap*>(new vk_overlap_impl{});
  VK_CAPI_CATCH_RETURN_NULL()
}

void vk_overlap_delete(vk_overlap* ex) { delete reinterpret_cast<vk_overlap_impl*>(ex); }

int vk_overlap_uses_two_streams(const vk_overlap* ex) {
  return reinterpret_cast<const vk_overlap_impl*>(ex)->ex.uses_two_streams() ? 1 : 0;
}

int32_t vk_overlap_run(vk_overlap* ex, size_t iters,
                       int (*compute)(size_t, void*), void* compute_ctx,
                       void (*comm)(size_t, int, void*), void* comm_ctx,
                       size_t* out_compute_count, size_t* out_comm_count) {
  VK_CAPI_TRY
  const vkernels::comm::OverlapExecutor::Result res =
      reinterpret_cast<vk_overlap_impl*>(ex)->ex.run(
      iters,
      [compute, compute_ctx](std::size_t i) { return compute(i, compute_ctx); },
      [comm, comm_ctx](std::size_t i, int v) { comm(i, v, comm_ctx); });
  *out_compute_count = res.compute_count;
  *out_comm_count = res.comm_count;
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

/* ------------------------------------------------------------------ */
/* comm: p2p run-list gather                                           */
/* ------------------------------------------------------------------ */

int32_t vk_stage_runs_1d(const uint8_t* dst, size_t dst_capacity,
                         const void* const* src_ptrs,
                         const size_t* dst_offsets, const size_t* lengths,
                         size_t num_runs, vk_staged_run_1d** out,
                         size_t* out_count) {
  VK_CAPI_TRY
  const std::vector<vkernels::comm::StagedRun1D> runs =
      vkernels::comm::stage_runs_1d(dst, dst_capacity, src_ptrs, dst_offsets,
                                    lengths, num_runs);
  vk_staged_run_1d* arr = static_cast<vk_staged_run_1d*>(malloc_copy(runs));
  if (arr == nullptr && !runs.empty()) return VK_ERROR_INTERNAL;
  *out = arr;
  *out_count = runs.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_stage_runs_2d(const uint8_t* dst, size_t dst_capacity,
                         const vk_gather_2d* runs, size_t num_runs,
                         vk_staged_run_2d** out, size_t* out_count) {
  VK_CAPI_TRY
  std::vector<vkernels::comm::Gather2DRun> input;
  input.reserve(num_runs);
  for (std::size_t i = 0; i < num_runs; ++i) {
    input.push_back({runs[i].src, runs[i].src_stride, runs[i].dst_offset,
                     runs[i].dst_stride, runs[i].width, runs[i].height});
  }
  const std::vector<vkernels::comm::StagedRun2D> staged =
      vkernels::comm::stage_runs_2d(dst, dst_capacity, input.data(),
                                    input.size());
  vk_staged_run_2d* arr = static_cast<vk_staged_run_2d*>(malloc_copy(staged));
  if (arr == nullptr && !staged.empty()) return VK_ERROR_INTERNAL;
  *out = arr;
  *out_count = staged.size();
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_p2p_gather_runs(uint8_t* dst, size_t dst_capacity,
                           const void* const* src_ptrs,
                           const size_t* dst_offsets, const size_t* lengths,
                           size_t num_runs, vk_stream* stream) {
  VK_CAPI_TRY
  vkernels::comm::p2p_gather_runs(
      vkernels::Span<std::uint8_t>(dst, dst_capacity), src_ptrs, dst_offsets,
      lengths, num_runs, stream == nullptr ? nullptr : &reinterpret_cast<vk_stream_impl*>(stream)->s);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_p2p_gather_runs_2d(uint8_t* dst, size_t dst_capacity,
                              const vk_gather_2d* runs, size_t num_runs,
                              vk_stream* stream) {
  VK_CAPI_TRY
  std::vector<vkernels::comm::Gather2DRun> input;
  input.reserve(num_runs);
  for (std::size_t i = 0; i < num_runs; ++i) {
    input.push_back({runs[i].src, runs[i].src_stride, runs[i].dst_offset,
                     runs[i].dst_stride, runs[i].width, runs[i].height});
  }
  vkernels::comm::p2p_gather_runs_2d(
      vkernels::Span<std::uint8_t>(dst, dst_capacity), input,
      stream == nullptr ? nullptr : &reinterpret_cast<vk_stream_impl*>(stream)->s);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

int32_t vk_memcpy_peer_batch_async(uint8_t* dst, size_t dst_capacity,
                                   const void* const* src_ptrs,
                                   const size_t* dst_offsets,
                                   const size_t* lengths, size_t num_runs,
                                   vk_stream* stream) {
  VK_CAPI_TRY
  vkernels::comm::memcpy_peer_batch_async(
      vkernels::Span<std::uint8_t>(dst, dst_capacity), src_ptrs, dst_offsets,
      lengths, num_runs, stream == nullptr ? nullptr : &reinterpret_cast<vk_stream_impl*>(stream)->s);
  return VK_OK;
  VK_CAPI_CATCH_RETURN_CODE()
}

}  // extern "C"
