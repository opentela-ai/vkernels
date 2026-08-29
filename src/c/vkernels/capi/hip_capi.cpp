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

#include "vkernels/kernels/moe_fused.hpp"
#include "vkernels/kernels/mla.hpp"
#include "vkernels/kernels/kda.hpp"

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

// --- MLA forward (absorbed form) ---
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
