// vkernels/kernels/moe_fused.hpp
//
// MXFP4 fused-MoE grouped GEMM — end-to-end MoE layer that mirrors the
// xkernels `fused_moe_mxfp4` interface.
//
// Architecture (from xkernels/ops/moe/triton/moe_mxfp4_kernel.py):
//
//   Stage 0 — gate_up + SwiGLU:
//     act [EM, ispp] = silu(clamp(A_sorted @ w13_gate + b13_gate, L)) *
//                      clamp(A_sorted @ w13_up   + b13_up,   L)
//
//   Stage 1 — down + routed combine:
//     out [M, hidden] += act @ w2^T * topk_w_sorted
//
// Dequant (E2M1 + ue8m0) is done inline during the K loop — no full bf16
// materialization of the 138 GB expert weight buffer.
//
// The caller is responsible for expert alignment (moe_align_block_size).
//
// Follows the vkernels two-implementation model:
//   moe_fused.cpp — CPU reference, always compiled, in vkernels::kernels
//   moe_fused.hip — HIP kernel, compiled with VKERNELS_HAS_HIP, in
//                   vkernels::kernels::hip
#pragma once

#include <cstdint>

namespace vkernels::kernels {

// ---------------------------------------------------------------------------
// Epilogue activation for the fused MoE gate/up stage
// ---------------------------------------------------------------------------
//
//   kSwiGLU — silu(clamp(gate)) * clamp(up)  (the default; DeepSeek-V3/V4).
//   kSiTU   — Kimi-K3 "SoftCap-GLU", matching vLLM's `situ_and_mul` exactly:
//                gate' = beta * tanh(gate / beta) * sigmoid(gate)
//                up'   = linear_beta * tanh(up / linear_beta)   (linear_beta > 0;
//                        otherwise up is passed through unmodified)
//                act   = gate' * up'
//             No swiglu_limit clamp is applied on the SiTU path (the tanh
//             softcaps bound the output instead).
//
//   The weight dequant (E2M1 + ue8m0 `s<<23`) is identical for both
//   activations: Kimi-K3's "SiTU scale format" referenced by the cookbooks is
//   the `activation_clamp == 0.03125` epilogue sentinel in DeepGEMM's
//   sm100_fp8_fp4_mega_moe.cuh, not a different ue8m0 decode.
enum MoEActivation : int {
  kSwiGLU = 0,
  kSiTU = 1,
};

// ---------------------------------------------------------------------------
// fused_moe_mxfp4 — CPU reference (oracle)
// ---------------------------------------------------------------------------
//
// Parameters (matching xkernels):
//   A             [M, hidden]  bf16 activations (uint16_t)
//   w13           [E, 2*ispp, hidden//2] uint8 packed E2M1
//   w13_scale     [E, 2*ispp, hidden//group_size] uint8 ue8m0
//   w2            [E, hidden, ispp//2] uint8 packed E2M1
//   w2_scale      [E, hidden, ispp//group_size] uint8 ue8m0
//   sorted_ids    [EM] int32  flat topk index (token*top_k + sel) per row
//   topk_w_sorted [EM] float  routing weights (sorted to match sorted_ids)
//   expert_ids    [EM/BLOCK_M] int32  expert per M-block (-1 = filtered)
//   act_scratch   [EM, ispp] bf16  intermediate, indexed by SORTED ROW
//   out           [M, hidden] fp32  output
//   M, hidden, ispp, top_k, EM  dimensions
//   group_size                ue8m0 group size (typically 32)
//   swiglu_limit              SwiGLU clamp limit (>= 0; 0 = no limit; ignored
//                             for the SiTU activation)
//   activation                epilogue tag: kSwiGLU (0) or kSiTU (1)
//   beta                      SiTU gate softcap (e.g. 4.0 for Kimi-K3)
//   linear_beta               SiTU up softcap (e.g. 25.0 for Kimi-K3; <= 0
//                             passes `up` through unmodified)
//   b13           [E, 2*ispp]  gate_up bias (nullptr → skip)
//   b2            [E, hidden]  down bias (nullptr → skip)
//
// Constraints:
//   BLOCK_M = 16, BLOCK_N = 64, BLOCK_K = 64 (decode regime)
//   hidden % 64 == 0, ispp % 64 == 0, EM % 16 == 0
//   group_size == 32 (ue8m0 scale shared across 32 consecutive K elements)

void fused_moe_mxfp4_cpu(
    const uint16_t* A,
    const uint8_t*  w13,
    const uint8_t*  w13_scale,
    const uint8_t*  w2,
    const uint8_t*  w2_scale,
    const int32_t*  sorted_ids,
    const float*    topk_w_sorted,
    const int32_t*  expert_ids,
    uint16_t*       act_scratch,
    float*          out,
    int M, int hidden, int ispp, int top_k, int EM,
    int group_size,
    float swiglu_limit,
    int activation = kSwiGLU,
    float beta = 4.0f,
    float linear_beta = 25.0f,
    const float* b13 = nullptr,
    const float* b2 = nullptr);

// ---------------------------------------------------------------------------
// moe_align_block_size — map topk_ids → sorted_ids + expert_ids
// ---------------------------------------------------------------------------
//
// Converts the [M, top_k] token→expert routing table into the
// block-aligned sorted layout required by the grouped GEMM kernel.
//
//   topk_ids   [M][top_k]  int32  — which expert each (token, sel) maps to
//   block_size             e.g. 16 (BLOCK_M)
//   num_experts            total experts (e.g. 256)
//
// Outputs:
//   sorted_ids [EM_padded]    int32  — FLAT index (token*top_k + sel),
//                                      sorted by expert; padded with M*top_k
//   expert_ids [EM_padded/B]  int32  — expert id per block (-1 = pad)
//   → returns EM_padded (multiple of block_size)

int moe_align_block_size(
    const int32_t* topk_ids,
    int M, int top_k,
    int block_size, int num_experts,
    int32_t* sorted_ids,
    int32_t* expert_ids);

}  // namespace vkernels::kernels


// HIP implementation (only declared when VKERNELS_HAS_HIP is set).
#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// Same interface as the CPU reference, but the topk_w here is the RAW
// [M, top_k] matrix (the HIP host launcher gathers into sorted order
// internally, matching the xkernels contract).
//
// block_size selects the tile config:
//   16  — decode regime (16x64 tiles, 64 threads); sorted_ids/expert_ids
//         must be aligned with block_size=16.
//   64  — prefill regime (64x64 tiles, 256 threads); the caller must align
//         with block_size=64 (EM % 64 == 0, expert_ids per 64-row block).

void fused_moe_mxfp4(
    const uint16_t* A,
    const uint8_t*  w13,
    const uint8_t*  w13_scale,
    const uint8_t*  w2,
    const uint8_t*  w2_scale,
    const int32_t*  topk_ids,
    const float*    topk_w,
    uint16_t*       act_scratch,
    float*          out,
    int M, int hidden, int ispp, int top_k,
    const int32_t* sorted_ids,
    const int32_t* expert_ids,
    int EM,
    float swiglu_limit,
    int activation = kSwiGLU,
    float beta = 4.0f,
    float linear_beta = 25.0f,
    const float* b13 = nullptr,
    const float* b2 = nullptr,
    int block_size = 16);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
