// vkernels/capi/capi_moe.cpp — MoE entry points of the C ABI declared in
// capi.hpp: MXFP4 quantize/sort/scatter orchestration (moe_aux.hpp), the
// fused grouped GEMM (moe_fused.hpp) and moe_align_block_size. Error
// plumbing and the other domain TUs: see capi_internal.hpp.
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "vkernels/capi/capi.hpp"
#include "vkernels/capi/capi_internal.hpp"

#include "vkernels/kernels/moe_aux.hpp"
#include "vkernels/kernels/moe_fused.hpp"

extern "C" {

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

}  // extern "C"
