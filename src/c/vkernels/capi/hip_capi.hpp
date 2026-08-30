// vkernels/capi/hip_capi.hpp — C ABI over the HIP compute kernels.
//
// vkernels ships two C ABI surfaces:
//
//   * `capi.hpp` / `vkernels_c` — host-callable CUDA primitives, the
//     RCCL / P2P / KV-donation kernels, and CPU reference wrappers
//     (`vk_fused_moe_mxfp4` etc. that call `fused_moe_mxfp4_cpu`).
//     Present whenever a CUDA toolkit is available (`VKERNELS_HAS_CUDA`).
//
//   * `hip_capi.hpp` / `vkernels_hip` (this file) — the gfx942 HIP
//     *device* kernels (issues #11, #21, #29): the fused MXFP4 MoE, the
//     absorbed MLA forward, and the KDA delta-rule forward. Present
//     whenever ROCm is available (`VKERNELS_HAS_HIP`).
//
// The HIP wrappers are deliberately a **separate** shared library with
// **separate names** (`vk_hip_*`, not `vk_*`). The CPU reference wrappers
// in `capi.hpp` already own the `vk_*` names and have different
// signatures (they return `int32_t` status codes, take pre-sorted
// routing, and carry a `group_size`). Linking both into one process —
// e.g. a Rust crate that pulls in `vkernels_c` for unit tests and
// `vkernels_hip` for GPU serving — must not collide, so the device entry
// points are namespaced.
//
// Conventions
// -----------
//  * The HIP compute launchers return **void**, faithfully wrapping the
//    C++ `vkernels::kernels::hip::*` functions which are themselves void.
//    Those launchers enqueue device work and return; the caller is
//    responsible for synchronising (`hipDeviceSynchronize` /
//    `hipStreamSynchronize`) and checking `hipGetLastError()` afterwards,
//    exactly as the in-tree C++ tests do.
//  * Parameters are raw pointers + integer dimensions. Buffers are device
//    memory allocated by the caller (e.g. via `hipMalloc` or a
//    `torch.Tensor` on CUDA/HIP). No C++ type crosses the boundary.
//  * The `MoEActivation` tag (`kSwiGLU = 0`, `kSiTU = 1`) is passed as a
//    plain `int`, matching the C++ enum.
//
// Build: `vkernels_hip` is a plain `extern "C"` shared library compiled
// without LTO (HIP_SEPARABLE_COMPILATION would otherwise propagate
// `-flto=auto` and prune the entry points). Link `vkernels::vkernels_hip`.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* fused MXFP4 MoE (src/c/vkernels/kernels/moe_fused.hip, issue #11)    */
/* ------------------------------------------------------------------ */
/*
 * End-to-end MXFP4 fused-MoE grouped GEMM with an inline E2M1 + ue8m0
 * dequant, the gate+up GEMM, a SwiGLU or SiTU epilogue, the down GEMM,
 * and a routed scatter-add combine — in two kernel launches (plus a tiny
 * routing-weight gather).
 *
 * This is the *device* kernel; the CPU reference (`fused_moe_mxfp4_cpu`,
 * exposed as `vk_fused_moe_mxfp4` in `capi.hpp`) takes pre-sorted routing
 * and a `group_size`. The HIP kernel instead takes the raw `topk_ids` +
 * unsorted `topk_w` and computes `sorted_ids`/`expert_ids` via
 * `vk_moe_align_block_size` (CPU-only) before launch.
 *
 *   A           [M, hidden]            bf16 activations (uint16_t)
 *   w13         [E, 2*ispp, hidden/2]  packed E2M1 gate|up weights
 *   w13_scale   [E, 2*ispp, hidden/32] ue8m0 per-group scales
 *   w2          [E, hidden, ispp/2]    packed E2M1 down weights
 *   w2_scale    [E, hidden, ispp/32]   ue8m0 per-group scales
 *   topk_ids    [M, top_k]             raw token->expert routing table
 *   topk_w      [M, top_k]             raw routing weights (gathered into
 *                                     sorted order internally)
 *   act_scratch [EM, ispp]             bf16 intermediate (caller-allocated)
 *   out         [M, hidden]            fp32 output (caller zero-initialised)
 *   sorted_ids  [EM]                   flat topk index (token*top_k + sel),
 *                                     block-aligned (moe_align_block_size)
 *   expert_ids  [EM/block_size]        expert per block (-1 = padding)
 *   EM          padded sorted-row count (multiple of block_size)
 *   swiglu_limit  SwiGLU clamp (>= 0; 0 = none; ignored for SiTU)
 *   activation    kSwiGLU (0) or kSiTU (1)
 *   beta          SiTU gate softcap (e.g. 4.0 for Kimi-K3)
 *   linear_beta   SiTU up softcap (e.g. 25.0 for Kimi-K3; <= 0 passes up)
 *   b13         [E, 2*ispp]            gate|up bias (NULL = skip)
 *   b2          [E, hidden]            down bias (NULL = skip)
 *   block_size    tile config: 16 = decode, 64 = prefill
 */
// The device stream to launch the MoE kernels on. Pass the caller's
// current stream (e.g. PyTorch's torch.cuda.current_stream().cuda_stream,
// or hipStreamGetCurrent()) so the device kernels are ordered with respect
// to the buffers the caller allocated / will read on that same stream —
// which removes the device-wide synchronize the caller previously needed.
// NULL (default) launches on the HIP default stream (legacy behaviour).
void vk_hip_fused_moe_mxfp4(
    const uint16_t* A, const uint8_t* w13, const uint8_t* w13_scale,
    const uint8_t* w2, const uint8_t* w2_scale,
    const int32_t* topk_ids, const float* topk_w,
    uint16_t* act_scratch, float* out,
    int M, int hidden, int ispp, int top_k,
    const int32_t* sorted_ids, const int32_t* expert_ids,
    int EM, float swiglu_limit,
    int activation, float beta, float linear_beta,
    const float* b13, const float* b2, int block_size, void* stream);

/* ------------------------------------------------------------------ */
/* MLA forward — absorbed form (src/c/vkernels/kernels/mla.hip, #21)   */
/* ------------------------------------------------------------------ */
/*
 * Multi-head Latent Attention forward: a causal latent-attention match of
 * vLLM's TRITON_MLA / AITER *absorbed* semantics. `q` carries its own
 * RoPE slice; `k_c`/`v_c` are the compressed latent (kv_lora_rank);
 * `k_pe` is the decoupled RoPE part.
 *
 *   q           [B, S_q, H, kv_lora_rank + qk_rope_head_dim]  float32
 *   k_c         [B, S_kv, kv_lora_rank]                       float32
 *   k_pe        [B, S_kv, qk_rope_head_dim]                   float32
 *   v_c         [B, S_kv, kv_lora_rank]                       float32
 *   out         [B, S_q, H, kv_lora_rank]                     float32
 *   q_start/kv_start  row offsets into the (B, S) batch
 *   scale             1/sqrt(kv_lora_rank + qk_rope_head_dim)
 */
void vk_hip_mla_fwd(
    int B, int H, int S_q, int S_kv, int q_start, int kv_start,
    int kv_lora_rank, int qk_rope_head_dim, float scale,
    const float* q, const float* k_c, const float* k_pe,
    const float* v_c, float* out);

/* ------------------------------------------------------------------ */
/* DSA sparse-MLA forward (src/c/vkernels/kernels/dsa.hip, #51)      */
/* ------------------------------------------------------------------ */
/*
 * DeepseekSparseAttn (GLM-5.3-Flash / DeepSeek-V3) sparse-MLA forward.
 * The indexer has already selected, per query token, the top-`topk` KV
 * tiles (`indices`); this kernel scores each query against those keys and
 * produces the combined attention output.
 *
 *   q       [1, S_q,  H,  dim + tail_dim]      bf16
 *   kv      [1, S_kv, kv_group, dim + tail_dim] bf16  (kv_group == 1)
 *   indices [1, S_q, kv_group, topk]           int32 (<0/>=S_kv masked)
 *   out     [1, S_q, H, dim - tail_dim]        bf16
 *   lse     [1, S_q, H]                         fp32 (nullable; set when
 *                                                    return_lse != 0)
 *   sm_scale = (1/sqrt(dim + tail_dim)) * log2(e)   (folds log2(e))
 *
 * tail_dim == 0 (GLM-5.3) is handled by skipping the rope-tail dot at
 * runtime -- no zero-size GEMM, the exact case the tilelang code path
 * cannot compile. Online base-2 softmax, fp32 accumulation.
 */
void vk_hip_dsa_sparse_fwd(
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk, int kv_group,
    int block_I, int inner_iter, float sm_scale, int return_lse,
    const void* q, const void* kv, const void* indices, void* out, void* lse);

/* Per-shape (bq, threads, block_I, inner_iter) tile selector for the HIP
 * DSA kernel (mirrors vk_dsa_config). Never throws. */
void vk_hip_dsa_config(int S_q, int H, int dim, int topk, int* bq,
                       int* threads, int* block_I, int* inner_iter);

/* ------------------------------------------------------------------ */
/* MHC multi-head hybrid-attention pre-norm (src/c/vkernels/kernels/    */
/* mhc.hip, #51 part 2)                                                 */
/* ------------------------------------------------------------------ */
/*
 * Two gfx942 HIP kernels re-implementing the tilelang MHC kernels that
 * fault/abort on MI300A (dynamic-shared > 64 KB non-optin cap). This is
 * the "vkernels HIP MHC path" that runs WITHOUT the hidden_block 256->128
 * workaround: static shared memory, no pipelining, no hipFuncSetAttribute
 * opt-in.
 *
 *   mhc_pre_gemm_sqrsum:
 *     x      [num_tokens, hc_hidden_size]  bf16   (hc_hidden=hc_mult*hidden)
 *     fn     [hc_mult3,    hc_hidden_size] fp32   (hc_mult3=hc_mult*(2+hc_mult))
 *     out    [num_tokens, hc_mult3]        fp32   (= x @ fn^T)
 *     sqrsum [num_tokens]                  fp32   (= sum_h x[n,h]^2)
 *     hc_mult3 <= 32; hc_hidden_size <= 28672.
 *
 *   mhc_post:
 *     a (comb_res_mix)[num_tokens, hc, hc]    fp32
 *     b (residual)   [num_tokens, hc, hidden] bf16
 *     c (post_layer_mix)[num_tokens, hc]      fp32
 *     d (x)          [num_tokens, hidden]     bf16
 *     out[n,j,h] = c[n,j]*d[n,h] + sum_k a[n,k,j]*b[n,k,h]   (bf16)
 *     hc <= 63.
 */
void vk_hip_mhc_pre_gemm_sqrsum(int num_tokens, int hc_mult3, int hc_hidden_size,
                                const void* x, const void* fn,
                                void* out, void* sqrsum);

void vk_hip_mhc_post(int num_tokens, int hc, int hidden,
                     const void* a, const void* b, const void* c,
                     const void* d, void* out);

/* ------------------------------------------------------------------ */
/* KDA delta-rule forward (src/c/vkernels/kernels/kda.hip, #21, #45)   */
/* ------------------------------------------------------------------ */
/*
 * Kimi Delta Attention: a gated delta-rule linear-attention layer. A
 * per-head state matrix S_t (D x D) is decayed each token by a
 * per-key-dim forget gate g_t[k] and then updated by a delta correction
 * beta_t (v_t - a_t) k_t^T (prediction a_t from the GATED state).
 * This is the portable gfx942 replacement for the AITER / Triton
 * chunked kernels, which GPU-fault on gfx942.
 *
 *   q, k, v     [B, H, S, D]  float32 queries/keys/values
 *                                     (k L2-normalised, q L2-norm +
 *                                      scaled by D^{-1/2} by the caller)
 *   g           [B, H, S, D]  float32 per-key-dim forget gate
 *                                     (normal space, (0, 1])
 *   beta        [B, H, S]     float32 scalar delta rate per (token, head)
 *   out         [B, H, S, D]  float32 output
 *   chunk_size  tiles the S dimension (e.g. 64)
 */
void vk_hip_kda_delta_rule_fwd(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* out,
    int B, int H, int S, int D, int chunk_size);

/* Same as vk_hip_kda_delta_rule_fwd but the caller owns the state
 * scratch [B, H, D, D] float32 -- pre-fill with the gathered initial
 * state (zeros for first-turn prefill); the final state S_S is written
 * back into the SAME buffer (read it after the call for multi-turn
 * decode). No internal allocation; the caller reuses one buffer. */
void vk_hip_kda_delta_rule_fwd_with_scratch(
    const float* q, const float* k, const float* v,
    const float* g, const float* beta, float* state,
    float* out, int B, int H, int S, int D);

#ifdef __cplusplus
}
#endif
