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

// The VK_OK / VK_ERROR_* status codes are shared with the CPU ABI
// (vk_moe_align_block_size). Including capi.hpp here makes them part of
// the HIP ABI surface for every consumer (ctypes uses the numeric values;
// C/Rust FFI consumers need the enum). The CPU vk_* declarations it also
// brings in are extern "C" and disjoint from the vk_hip_* names below.
#include "vkernels/capi/capi.hpp"

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
/* moe_align_block_size — device, decode (issue #46 follow-up)         */
/* ------------------------------------------------------------------ */
/*
 * GPU counterpart of `vk_moe_align_block_size` (CPU, capi.hpp). Reads
 * `topk_ids` on-device on `stream` — ordered after the expert-dispatch
 * all-to-all — and writes the block-aligned `sorted_ids` / `expert_ids`
 * consumed by `vk_hip_fused_moe_mxfp4`, WITHOUT the `topk_ids.cpu()` host
 * round-trip that dominated PP0's MoE (97-100% of its per-call
 * `moe:vkernel_apply`; see docs/performance/moe-fused/gfx942-pp-pipeline.md).
 *
 * `expert_map` [num_experts] (global->local, -1 = skip) or NULL (1:1).
 * `max_EM` is the host-computed high-water bound (a constant for a given
 * batch shape, so the GEMM grid `max_EM/block_size` is capture-safe and
 * the kernels early-out padding blocks/rows). `*out_EM` receives the
 * actual padded EM — a device write; the capture-safe fast path does NOT
 * read it (it launches the GEMM with `max_EM/block_size` blocks).
 *
 * Limit: `M*top_k` must be `<= 1024` (single-block shared memory, decode).
 * Larger N returns VK_ERROR_UNSUPPORTED; the caller falls back to the CPU
 * `vk_moe_align_block_size`.
 *
 * Returns VK_OK, or VK_ERROR_INVALID_ARGUMENT on null/bad params,
 * VK_ERROR_UNSUPPORTED if M*top_k > 1024. Note: unlike the void HIP
 * compute launchers, this wrapper returns a status code (matches the CPU
 * ABI and lets the ctypes caller pick the fallback path) — it performs
 * only host-side validation and returns immediately after launch.
 *
 *   topk_ids    [N]            int32  global expert ids (M*top_k)
 *   expert_map  [num_experts]  int32  global->local (-1 = skip), or NULL
 *   block_size                16 (decode) or 64 (prefill)
 *   local_n                   number of local experts on this rank
 *   max_EM                    host high-water bound on EM
 *   sorted_ids  [max_EM]            int32  flat indices, padded with N
 *   expert_ids  [max_EM/block_size] int32  local expert per block, -1 pad
 *   out_EM      [1]                 int32  actual padded EM (device write)
 *   stream                     caller's current stream, or NULL (default)
 */
int vk_hip_moe_align_block_size(
    const int32_t* topk_ids, const int32_t* expert_map,
    int32_t M, int32_t top_k, int32_t block_size,
    int32_t num_experts, int32_t local_n, int32_t max_EM,
    int32_t* sorted_ids, int32_t* expert_ids, int32_t* out_EM,
    void* stream);

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
int vk_hip_dsa_sparse_fwd(
    int S_q, int S_kv, int H, int dim, int tail_dim, int topk, int kv_group,
    int block_I, int inner_iter, float sm_scale, int return_lse,
    const void* q, const void* kv, const void* indices, void* out,
    void* lse);

/* Pool-level radix top-k transform for the DSA kpool indexer.
 *
 *   score             [batch_size, score_stride] strided fp32
 *   lengths           [batch_size] int32 valid group counts
 *   dst_token_indices [batch_size, out_cols] contiguous int32
 *   page_table        [batch_size, page_table_stride] int32 (optional)
 *
 * token_topk / pool_size must be one of 128, 160, 192, 224, 256, or
 * 512. page_table and topk_indices_offset are mutually exclusive. When
 * page_table_row_index is null, score row i uses page-table row i; otherwise
 * every entry must be in [0, batch_size). When
 * seq_lens is non-null, out_cols must equal token_topk + pool_size - 1 and
 * the seq_len % pool_size trailing tokens are appended after history.
 */
void vk_hip_dsa_topk_transform(
    int32_t batch_size, const float* score, const int32_t* lengths,
    int32_t* dst_token_indices, int64_t score_stride, int32_t pool_size,
    int32_t token_topk, int32_t out_cols, const int32_t* page_table,
    int64_t page_table_stride, const int32_t* page_table_row_index,
    const int32_t* topk_indices_offset, const int32_t* row_starts,
    const int32_t* seq_lens);

/* Per-shape (bq, threads, block_I, inner_iter) tile selector for the HIP
 * DSA kernel (mirrors vk_dsa_config). Never throws. */
void vk_hip_dsa_config(int S_q, int H, int dim, int topk, int* bq,
                       int* threads, int* block_I, int* inner_iter);

/* ------------------------------------------------------------------ */
/* DSA paged-MQA gated top-k logits (src/c/vkernels/kernels/dsa.hip,    */
/* #51 -- the kpool>1 indexer path)                                     */
/* ------------------------------------------------------------------ */
/*
 * The INDEXER scores each query against its paged KV tiles; the top-k
 * selects over the results. See dsa.hpp for the gated-logit formula.
 *
 *   q_fp8       [batch_size, num_heads, head_dim]        fp8 e4m3fnuz
 *   kvcache_u8  [num_blocks, block*(head_dim+4)]         uint8 (block*head_dim
 *                 fp8 keys, then block fp32 per-token scales)
 *   weight      [batch_size, num_heads]                  fp32 (the gate)
 *   seq_lens    [batch_size]                             int32 (POOLED valid)
 *   page_table  [batch_size, max_table_len]              int32
 *   out         [batch_size, max_seq_len]                fp32 (ZERO first;
 *                 max_seq_len = max_table_len*block. Tokens >= seq_len[b]
 *                 are left unwritten -- sglang masks invalid positions)
 *   split_kv    perf only: max(1, min(max_seq_len//block, NUM_CU//batch_size))
 *                 with NUM_CU = 228 (MI300A / gfx942;
 *                 hipDeviceProp_t::multiProcessorCount, verified on a CSCS
 *                 beverin node -- the cap was wrongly stated as 256 here
 *                 before dsa.hpp::dsa_topk_logits_split_for). Grouping-
 *                 independent (any positive split_kv yields the same top-k).
 *                 The host C ABI vk_dsa_topk_logits_split_for is the single
 *                 source of truth for this formula; call it instead of
 *                 recomputing.
 *
 * Dispatches on the staged-bytes cap (gfx942's 64 KB NON-OPTIN dynamic-LDS
 * limit; NO hipFuncSetAttribute opt-in past it -- see the KB note
 * mi300a-dynamic-lds-no-optin). The AUTO dispatcher (dsa_topk_logits in
 * dsa.hip, via dsa_topk_logits_with_variant(0)) picks the SMALLEST-
 * footprint variant that fits the shape's cap:
 *   1. bf16-MFMA (dsa_topk_logits_kernel_mfma) -- Q transposed as bf16
 *      sQt[D][H] staged once, one bf16 K-tile reloaded, gated via Matrix
 *      Cores. Needs H%16==0, head_dim%64==0, block%16==0 and fits
 *      dsa_topk_logits_fits_lds_mfma. At the GLM-5.3 indexer (H=32, D=128,
 *      B=64) that is 16,768 B and runs ~10x faster than the fp32-Q kernel;
 *      it also fits H=64 (25,088 B) and H=128 (41,728 B), so those now
 *      take the MFMA fast path instead of the fp8-Q fallback.
 *   2. fp32-Q (dsa_topk_logits_kernel) -- Q dequanted once into shared.
 *      Fallback for shapes the MFMA kernel refuses but that fit
 *      dsa_topk_logits_fits_lds (e.g. H not a multiple of 16).
 *   3. fp8-Q (dsa_topk_logits_kernel_fp8q) -- Q staged raw, dequantised
 *      on the fly -- bit-identical output. Fallback for shapes fitting
 *      neither of the above but dsa_topk_logits_fits_lds_fp8q.
 * Shapes fitting NONE are REFUSED -- a stderr diagnostic + no-op, leaving
 * `out` as the caller provided.
 */
void vk_hip_dsa_topk_logits(
    int batch_size, int num_heads, int head_dim, int block,
    int max_table_len, int max_seq_len, int split_kv,
    const void* q_fp8, const void* kvcache_u8,
    const void* weight, const void* seq_lens,
    const void* page_table, void* out);

/* ------------------------------------------------------------------ */
/* DSA kpool-cache compress/write (src/c/vkernels/kernels/dsa_kpool.hip,*/
/* #60 -- the kpool>1 INDEXER cache path; never declares fp8e4nv)       */
/* ------------------------------------------------------------------ */
/*
 * Native gfx942 HIP kernels for sglang's kpool_assemble_softmax_rotate-
 * _write_cache (prefill) and kpool_decode_update_and_maybe_write_cache
 * (decode) in kpool_fp8_index.py -- both declare tl.float8e4nv and JIT-fail
 * on SM80 (A100) on the first forward (job 82822). These reimplement them
 * with bf16 STORAGE + fp32 ACCUMULATION and NO fp8e4nv (the bf16 range is
 * ample, so the per-vector scale is dropped and the cache collapses to
 * flat [num_pages, ssp, head_dim] bf16). See dsa_kpool.hpp for the full
 * computation and vk_dsa_kpool_* for the host fp32 reference these are
 * checked against (meta/benchmarks/test_dsa_kpool_correct.hip).
 *
 * All non-ape device pointers are bf16 (raw uint16); ape, block_tables,
 * req_pool_indices, positions, seq_lens, out_cache_loc are fp32/int32
 * device. `out` is a bf16 device pointer (ZERO first). `tail_k`/
 * `tail_score` are IN-PLACE bf16 device. Never throws on valid input;
 * bad dims are a no-op (stderr diagnostic), mirroring vk_hip_dsa_topk_.*
 */
void vk_hip_dsa_kpool_assemble(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const void* chunk_k, const void* chunk_score, const void* tail_k,
    const void* tail_score, const void* ape, const void* req_pool_idx,
    const void* n_from_tail, const void* chunk_src_start,
    const void* tail_logical_base, const void* loc, const void* write_mask,
    void* out);

void vk_hip_dsa_kpool_decode_update(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const void* key, const void* slot_score, void* tail_k, void* tail_score,
    const void* ape, const void* block_tables, const void* req_pool_indices,
    const void* positions, const void* seq_lens, const void* out_cache_loc,
    void* out);

/* Issue #61: graph-capturable overloads and fp8+scale store variants.
 * Same contracts as vk_hip_dsa_kpool_assemble / vk_hip_dsa_kpool_decode_update
 * above, except:
 *   - *_on_stream takes an explicit stream (NULL -> default stream).
 *   - *_fp8 writes the legacy uint8 cache [num_pages, ssp*(head_dim+4)]
 *     with head_dim fp8e4m3fn bytes + one fp32 scale per vector. cache_u8
 *     is a device uint8 pointer (zeroed first). tail_k/tail_score are still
 *     IN-PLACE bf16 device (decode live tail). key/slot_score are bf16 device.
 *   - round_scale_or_null is a device int pointer (or NULL); *it > 0 selects
 *     power-of-two scale rounding, otherwise raw absmax/448.
 */
void vk_hip_dsa_kpool_assemble_on_stream(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const void* chunk_k, const void* chunk_score, const void* tail_k,
    const void* tail_score, const void* ape, const void* req_pool_idx,
    const void* n_from_tail, const void* chunk_src_start,
    const void* tail_logical_base, const void* loc, const void* write_mask,
    void* out, void* stream);
void vk_hip_dsa_kpool_decode_update_on_stream(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const void* key, const void* slot_score, void* tail_k, void* tail_score,
    const void* ape, const void* block_tables, const void* req_pool_indices,
    const void* positions, const void* seq_lens, const void* out_cache_loc,
    void* out, void* stream);
void vk_hip_dsa_kpool_assemble_fp8(
    int n_pools, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int num_pages, int num_chunks, int n_reqs,
    const void* chunk_k, const void* chunk_score, const void* tail_k,
    const void* tail_score, const void* ape, const void* req_pool_idx,
    const void* n_from_tail, const void* chunk_src_start,
    const void* tail_logical_base, const void* loc, const void* write_mask,
    void* cache_u8, const void* round_scale_or_null, void* stream);
void vk_hip_dsa_kpool_decode_update_fp8(
    int batch, int pool_size, int head_dim, int tail_size,
    int slots_per_page, int block_table_cols, int n_reqs, int num_pages,
    const void* key, const void* slot_score, void* tail_k, void* tail_score,
    const void* ape, const void* block_tables, const void* req_pool_indices,
    const void* positions, const void* seq_lens, const void* out_cache_loc,
    void* cache_u8, const void* round_scale_or_null, void* stream);

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
