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
// Distributed (TP) stage split of the fused MoE
// ---------------------------------------------------------------------------
//
// `fused_moe_mxfp4_cpu` fuses the activation into stage 0, which blocks
// tensor-parallel execution: a TP rank owns a K-slice of `hidden`, so the
// gate/up GEMM of each rank must be summed (all-reduced) BEFORE the
// nonlinear activation.  The fused call is therefore split into four
// composable stages with an explicit boundary between the linear GEMMs and
// the elementwise epilogues:
//
//   Stage 0a  moe_gateup_cpu       gate_up[EM, 2*ispp]  = A_slice @ w13^T
//            (caller: all-reduce gate_up across TP ranks)
//   Stage 0b  moe_act_epilogue_cpu act[EM, ispp] = act(gate_up + b13)
//   Stage 1a  moe_down_cpu         partial[EM, hidden] = act_slice @ w2^T
//            (caller: all-reduce partial across TP ranks)
//   Stage 1b  moe_combine_cpu      out[M, hidden] += (partial + b2) * topk_w
//
// `fused_moe_mxfp4_cpu` is exactly this composition with k_base = 0 and the
// full K dimensions — byte-identical results, exposed for the TP rank whose
// weights cover `k_base .. k_base + k_dim`.  The all-reduce points are the
// hooks where the caller (e.g. vLLM on gfx942) runs its own collective; see
// `dist_moe.hpp` for the distributed orchestrators.
//
// Stage 0a — gate/up GEMM (linear, pre-activation, pre-bias).
//   A       [M, a_stride] bf16   input activations (full hidden rows on a TP
//                                rank; a_stride == full hidden)
//   k_base  column in each A row where this rank's weight shard starts
//           (0 for the single-rank call; r*hidden_shard for TP rank r)
//   hidden_k  K dimension of this GEMM == width of this rank's w13 shard
//           (full hidden for single rank, hidden/tp for a TP rank)
//   gate_up [EM, 2*ispp] fp32  raw gate (cols 0..ispp-1) and up
//           (cols ispp..2*ispp-1) accumulators; rows of padding blocks
//           (expert_ids[mb] < 0) are left untouched
void moe_gateup_cpu(
    const uint16_t* A,
    const uint8_t*  w13,
    const uint8_t*  w13_scale,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    float*          gate_up,
    int M, int a_stride, int k_base, int hidden_k, int ispp, int top_k,
    int EM, int group_size);

// Stage 0b — bias + activation epilogue → act [EM, ispp] bf16.
//   gate_up   [EM, 2*ispp] fp32 (all-reduced when TP > 1)
//   b13       [E, 2*ispp] fp32 gate/up bias (nullptr → skip)
//   act       [EM, ispp] bf16  written for real rows only (padding rows of
//             the caller's buffer are left untouched, as in the fused call)
void moe_act_epilogue_cpu(
    const float*    gate_up,
    const float*    b13,
    uint16_t*       act,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    int M, int top_k, int EM, int ispp,
    float swiglu_limit,
    int activation = kSwiGLU,
    float beta = 4.0f,
    float linear_beta = 25.0f);

// Stage 1a — down GEMM (linear, pre-bias, pre-combine).
//   act       [EM, a_stride] bf16  stage-0 output (full ispp rows on a TP
//                                   rank; a_stride == full ispp)
//   k_base    column in each act row where this rank's w2 shard starts
//   ispp_k    K dimension of this GEMM == width of this rank's w2 shard
//   partial   [EM, hidden] fp32 raw down accumulators (full output width)
void moe_down_cpu(
    const uint16_t* act,
    const uint8_t*  w2,
    const uint8_t*  w2_scale,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    float*          partial,
    int M, int a_stride, int k_base, int ispp_k, int hidden, int top_k,
    int EM, int group_size);

// Stage 1b — routed combine epilogue: out[M, hidden] += (partial + b2) * topk_w.
//   partial  [EM, hidden] fp32 (all-reduced when TP > 1)
//   b2       [E, hidden] fp32 down bias (nullptr → skip)
//   out      [M, hidden] fp32  accumulates in place (caller zero-initialises)
void moe_combine_cpu(
    const float*    partial,
    const float*    b2,
    const float*    topk_w_sorted,
    const int32_t*  sorted_ids,
    const int32_t*  expert_ids,
    float*          out,
    int M, int hidden, int top_k, int EM);

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


// ---------------------------------------------------------------------------
// HIP stage kernels (TP mode), declared only when VKERNELS_HAS_HIP is set.
// ---------------------------------------------------------------------------
#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// Distributed (TP) variant of the fused MoE, split so the caller can
// all-reduce between the linear stages (see dist_moe.hpp).  Each function
// consumes a per-rank weight shard: w13/w13_scale are [E, 2*ispp, hidden/tp]
// slices of the full weight, w2/w2_scale are [E, hidden, ispp/tp] slices,
// b13/b2 are shared (added post-all-reduce).  `k_base` is the column offset
// into A / act where this rank's shard starts (r*hidden/tp, r*ispp/tp).
//
//   gate_up [EM, 2*ispp] fp32   ← stage 0a (then caller all-reduces)
//   act     [EM, ispp]    bf16  ← stage 0b
//   partial [EM, hidden]  fp32  ← stage 1a (then caller all-reduces)
//   out     [M, hidden]   fp32  ← stage 1b (accumulates; zero-init first)
//
// The call sequence for one TP rank:
//   moe_gateup_preact(...)   → all-reduce gate_up → moe_act_epilogue(...)
//     → moe_down_preact(...) → all-reduce partial → moe_combine(...)
// These are the GPU counterparts of the CPU stage functions above.  Decode
// regime only (block_size == 16, 64-thread tiles); prefill follows the same
// pattern with the 256-thread templates.

void moe_gateup_preact(
    const uint16_t* A, const uint8_t* w13, const uint8_t* w13_scale,
    const int32_t* sorted_ids, const int32_t* expert_ids, float* gate_up,
    int M, int hidden, int ispp, int top_k, int EM,
    int k_base, int hidden_k);

void moe_act_epilogue(
    const float* gate_up, const float* b13, uint16_t* act,
    const int32_t* sorted_ids, const int32_t* expert_ids,
    int M, int top_k, int EM, int ispp, float swiglu_limit,
    int activation, float beta, float linear_beta);

void moe_down_preact(
    const uint16_t* act, const uint8_t* w2, const uint8_t* w2_scale,
    const int32_t* sorted_ids, const int32_t* expert_ids, float* partial,
    int M, int ispp, int hidden, int top_k, int EM,
    int k_base, int ispp_k);

void moe_combine(
    const float* partial, const float* b2, const float* topk_w_sorted,
    const int32_t* sorted_ids, const int32_t* expert_ids, float* out,
    int M, int hidden, int top_k, int EM);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP


// HIP implementation (only declared when VKERNELS_HAS_HIP is set).
#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// Same interface as the CPU reference, but the topk_w here is the RAW
// [M, top_k] matrix: the HIP host launcher gathers into sorted order
// internally via `gather_weights_kernel`, writing the sorted routing
// weights into `sorted_w_scratch`.
//
// `act_scratch` [EM, ispp] bf16 and `sorted_w_scratch` [EM] float are BOTH
// caller-provided, persistent scratch buffers (issue #41, item 1).  The
// launcher performs no device allocation of its own: a backend serving a
// 61-layer model allocates these once and reuses them across every forward
// pass, instead of paying 122 allocator round-trips per generated token.
// `sorted_w_scratch` must hold at least `EM` float32 elements.
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
    float*          sorted_w_scratch,
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
