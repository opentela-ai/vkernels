// vkernels/capi/capi_kernels.cpp — elementwise / reduce / GEMM and the
// gfx942 primitive entry points of the C ABI declared in capi.hpp.
// Error plumbing and the other domain TUs: see capi_internal.hpp.
#include <cstddef>
#include <cstdint>

#include "vkernels/capi/capi.hpp"
#include "vkernels/capi/capi_internal.hpp"

#include "vkernels/kernels/elementwise.hpp"
#include "vkernels/kernels/gemm.hpp"
#include "vkernels/kernels/gemm_bf16.hpp"
#include "vkernels/kernels/moe.hpp"
#include "vkernels/kernels/reduce.hpp"

extern "C" {

/* ------------------------------------------------------------------ */
/* kernels: elementwise / reduce / GEMM                                */
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

}  // extern "C"
