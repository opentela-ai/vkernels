// vkernels/capi/hip_capi.cpp — C ABI over the gfx942 HIP compute kernels.
//
// Thin `extern "C"` wrappers around the host launchers in
// `vkernels::kernels::hip::*`. The wrapper is the sole source of the
// `vkernels_hip` shared library; it links the static `vkernels` for the
// C++ implementation (the device kernels are compiled there under
// `VKERNELS_HAS_HIP`).
//
// Built as a plain shared library — no pybind11, no `-fvisibility=hidden`,
// no LTO — so the entry points are not pruned. (pybind11 v3.1.0
// auto-enables `-flto=auto` via pybind11Common.cmake:379-389, and
// HIP_SEPARABLE_COMPILATION propagates the same flag; both strip
// `extern "C"` symbols from the final `.so`.) See issue #43.
//
// Names are `vk_hip_*` (not `vk_*`) to avoid colliding with the CPU
// reference wrappers in capi.hpp/cpp, which already own the bare names.
// Integration surface for vLLM and other Python runtimes: load with
// `ctypes.CDLL` or `torch.ops.load_library`, pass device pointers from
// torch tensors via `.data_ptr()`.

#include "vkernels/capi/hip_capi.hpp"

#include "vkernels/capi/capi.hpp"  // VK_OK / VK_ERROR_* status codes
#include <stdexcept>   // std::invalid_argument / std::exception (issue #57 catch)
#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/mla.hpp"
#include "vkernels/kernels/dsa.hpp"
#include "vkernels/kernels/dsa_topk.hpp"
#include "vkernels/kernels/kda.hpp"
#include "vkernels/kernels/mhc.hpp"

// --- fused MXFP4 MoE (Kimi-K3 SiTU via activation=1) ---
extern "C" void vk_hip_fused_moe_mxfp4(
    const uint16_t* A, const uint8_t* w13, const uint8_t* w13_scale,
    const uint8_t* w2, const uint8_t* w2_scale,
    const int32_t* topk_ids, const float* topk_w,
    uint16_t* act_scratch, float* out,
    int M, int hidden, int ispp, int top_k,
    const int32_t* sorted_ids, const int32_t* expert_ids,
    int EM, float swiglu_limit,
    int activation, float beta, float linear_beta,
    const float* b13, const float* b2, int block_size, void* stream) {
  vkernels::kernels::hip::fused_moe_mxfp4(
      A, w13, w13_scale, w2, w2_scale,
      topk_ids, topk_w, act_scratch, out,
      M, hidden, ispp, top_k,
      sorted_ids, expert_ids, EM, swiglu_limit,
      activation, beta, linear_beta, b13, b2, block_size, /*kmajor=*/false,
      stream);
}

// --- GPU moe_align_block_size (decode; issue #46 follow-up) ---
// Reads topk_ids on-device on `stream` (ordered after the expert-dispatch
// all-to-all) and writes the block-aligned sorted_ids / expert_ids consumed
// by vk_hip_fused_moe_mxfp4, WITHOUT the topk_ids.cpu() host round-trip.
// Returns a status code (not void) so the ctypes caller can pick the
// CPU fallback when M*top_k > 1024. Host-side validation only; the kernel
// is enqueued and the wrapper returns immediately (no sync).
extern "C" int vk_hip_moe_align_block_size(
    const int32_t* topk_ids, const int32_t* expert_map,
    int32_t M, int32_t top_k, int32_t block_size,
    int32_t num_experts, int32_t local_n, int32_t max_EM,
    int32_t* sorted_ids, int32_t* expert_ids, int32_t* out_EM,
    void* stream) {
  if (block_size <= 0)
    return VK_ERROR_INVALID_ARGUMENT;
  if (M < 0 || top_k < 0 || num_experts < 0 || local_n < 0 || max_EM < 0)
    return VK_ERROR_INVALID_ARGUMENT;
  if (max_EM == 0 || (max_EM % block_size) != 0)
    return VK_ERROR_INVALID_ARGUMENT;
  if (local_n > num_experts)  // local shard never larger than the global set
    return VK_ERROR_INVALID_ARGUMENT;
  // out_EM is always written (even for M==0 the kernel emits one padding
  // block), and the routing buffers are required for any non-trivial batch.
  if (out_EM == nullptr)
    return VK_ERROR_INVALID_ARGUMENT;
  if (M > 0 && (topk_ids == nullptr || sorted_ids == nullptr ||
                expert_ids == nullptr)) {
    return VK_ERROR_INVALID_ARGUMENT;
  }
  const long Nl = (long)M * (long)top_k;
  if (Nl > 1024)  // single-block shared memory; caller falls back to CPU
    return VK_ERROR_UNSUPPORTED;
  vkernels::kernels::hip::moe_align_block_size_hip(
      topk_ids, expert_map, M, top_k, block_size, num_experts, local_n,
      max_EM, sorted_ids, expert_ids, out_EM, stream);
  return VK_OK;
}
extern "C" void vk_hip_mla_fwd(
    int B, int H, int S_q, int S_kv, int q_start, int kv_start,
    int kv_lora_rank, int qk_rope_head_dim, float scale,
    const float* q, const float* k_c, const float* k_pe,
    const float* v_c, float* out) {
  vkernels::kernels::hip::mla_fwd(
      B, H, S_q, S_kv, q_start, kv_start,
      kv_lora_rank, qk_rope_head_dim, scale,
      q, k_c, k_pe, v_c, out);
}

// --- DSA sparse-MLA forward (GLM-5.3-Flash / DeepSeek-V3) ---
extern "C" int vk_hip_dsa_sparse_fwd(
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk, int kv_group,
    int block_I, int inner_iter, float sm_scale, int return_lse,
    const void* q, const void* kv, const void* indices, void* out,
    void* lse) {
  // Host-side preconditions (mirror vk_dsa_sparse_fwd, issue #57): a
  // misconfigured caller gets a named error instead of a silent hang or
  // an out-of-bounds launch. Validated here (before the launch, no sync)
  // AND in dsa_sparse_fwd with VK_EXPECTS (defensive for direct C++ callers).
  const int d_v = dim - tail_dim;
  if (!(dim > 0 && tail_dim >= 0 && d_v > 0)) {
    vk_set_last_error(VK_ERROR_INVALID_ARGUMENT,
                      "dim > tail_dim >= 0 required");
    return VK_ERROR_INVALID_ARGUMENT;
  }
  if (topk <= 0) {
    vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, "topk must be positive");
    return VK_ERROR_INVALID_ARGUMENT;
  }
  if (kv_group != 1) {
    vk_set_last_error(VK_ERROR_INVALID_ARGUMENT,
                      "kv_group must be 1 (single shared head_kv)");
    return VK_ERROR_INVALID_ARGUMENT;
  }
  if (block_I <= 0 || inner_iter <= 0) {
    vk_set_last_error(VK_ERROR_INVALID_ARGUMENT,
                      "block_I and inner_iter must be positive");
    return VK_ERROR_INVALID_ARGUMENT;
  }
  if (topk % (block_I * inner_iter) != 0) {
    vk_set_last_error(VK_ERROR_INVALID_ARGUMENT,
                      "topk must be a multiple of block_I*inner_iter");
    return VK_ERROR_INVALID_ARGUMENT;
  }
  if (S_q && H && topk) {
    if (q == nullptr) { vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, "q must not be null"); return VK_ERROR_INVALID_ARGUMENT; }
    if (kv == nullptr) { vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, "kv must not be null"); return VK_ERROR_INVALID_ARGUMENT; }
    if (indices == nullptr) { vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, "indices must not be null"); return VK_ERROR_INVALID_ARGUMENT; }
    if (out == nullptr) { vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, "out must not be null"); return VK_ERROR_INVALID_ARGUMENT; }
    if (return_lse && lse == nullptr) { vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, "lse must not be null"); return VK_ERROR_INVALID_ARGUMENT; }
    if (out == q || out == kv || out == indices ||
        (return_lse && out == lse)) {
      vk_set_last_error(VK_ERROR_INVALID_ARGUMENT,
                        "out must not alias q, kv, indices, or lse");
      return VK_ERROR_INVALID_ARGUMENT;
    }
  }
  try {
    vkernels::kernels::hip::dsa_sparse_fwd(
        S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I, inner_iter,
        sm_scale, return_lse != 0, q, kv, indices, out, lse);
  } catch (const std::invalid_argument& e) {
    vk_set_last_error(VK_ERROR_INVALID_ARGUMENT, e.what());
    return VK_ERROR_INVALID_ARGUMENT;
  } catch (const std::exception& e) {
    vk_set_last_error(VK_ERROR_INTERNAL, e.what());
    return VK_ERROR_INTERNAL;
  }
  return VK_OK;
}

// --- DSA pool-level radix top-k transform ---
extern "C" void vk_hip_dsa_topk_transform(
    int32_t batch_size, const float* score, const int32_t* lengths,
    int32_t* dst_token_indices, int64_t score_stride, int32_t pool_size,
    int32_t token_topk, int32_t out_cols, const int32_t* page_table,
    int64_t page_table_stride, const int32_t* page_table_row_index,
    const int32_t* topk_indices_offset, const int32_t* row_starts,
    const int32_t* seq_lens) {
  vkernels::kernels::hip::dsa_topk_transform(
      batch_size, score, lengths, dst_token_indices, score_stride, pool_size,
      token_topk, out_cols, page_table, page_table_stride,
      page_table_row_index, topk_indices_offset, row_starts, seq_lens);
}

// --- DSA per-shape tile selector ---
extern "C" void vk_hip_dsa_config(int S_q, int H, int dim, int topk, int* bq,
                                 int* threads, int* block_I, int* inner_iter) {
  vkernels::kernels::dsa_config_for(S_q, H, dim, topk, bq, threads, block_I,
                                    inner_iter);
}

// --- DSA paged-MQA gated top-k logits (GLM-5.3-Flash indexer, #51) ---
extern "C" void vk_hip_dsa_topk_logits(
    int batch_size, int num_heads, int head_dim, int block,
    int max_table_len, int max_seq_len, int split_kv,
    const void* q_fp8, const void* kvcache_u8,
    const void* weight, const void* seq_lens,
    const void* page_table, void* out) {
  vkernels::kernels::hip::dsa_topk_logits(
      batch_size, num_heads, head_dim, block,
      max_table_len, max_seq_len, split_kv,
      q_fp8, kvcache_u8, weight, seq_lens, page_table, out);
}

// --- MHC multi-head hybrid-attention pre-norm (GLM-5.3-Flash) ---
extern "C" void vk_hip_mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult3,
                                        int hc_hidden_size,
                                        const void* x, const void* fn,
                                        void* out, void* sqrsum) {
  vkernels::kernels::hip::mhc_pre_gemm_sqrsum(num_tokens, hc_mult3,
                                              hc_hidden_size, x, fn, out,
                                              sqrsum);
}

extern "C" void vk_hip_mhc_post(int num_tokens, int hc, int hidden,
                               const void* a, const void* b, const void* c,
                               const void* d, void* out) {
  vkernels::kernels::hip::mhc_post(num_tokens, hc, hidden, a, b, c, d, out);
}

// --- KDA delta-rule forward ---
extern "C" void vk_hip_kda_delta_rule_fwd(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* out,
    int B, int H, int S, int D, int chunk_size) {
  vkernels::kernels::hip::kda_delta_rule_fwd(
      q, k, v, g, beta, out, B, H, S, D, chunk_size);
}

// --- KDA delta-rule forward (caller-owned state scratch) ---
extern "C" void vk_hip_kda_delta_rule_fwd_with_scratch(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* state,
    float* out, int B, int H, int S, int D) {
  vkernels::kernels::hip::kda_delta_rule_fwd_with_scratch(
      q, k, v, g, beta, state, out, B, H, S, D);
}
