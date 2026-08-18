// vkernels/capi/capi.hpp — C ABI for the vkernels library.
//
// The C++ API under src/c/vkernels/ is consumed directly by C++ (tests,
// benchmarks) and through a thin pybind11 layer (src/vkernels/_core.cpp).
// This header exposes the same surface as a plain C ABI so that bindings in
// languages without a C++ toolchain — first of all the Rust crate under
// src/rust/ — can link the library without a C++ compiler. The functions are
// implemented in capi.cpp and are compiled into the vkernels static library
// itself, so there is nothing extra to link.
//
// Conventions
// -----------
//  * Buffers are raw pointers + lengths. Every function that can fail
//    (contract checks, allocations) returns an int32_t status code: VK_OK on
//    success, one of the VK_ERROR_* codes otherwise. Pure getters that cannot
//    throw return their value directly.
//  * The C++ contract checks (VK_EXPECTS / VK_ENSURES) throw exceptions,
//    which cannot cross the ABI. capi.cpp catches every exception and stores
//    the message in a thread-local buffer; call vk_last_error() to read it.
//    The message is valid until the next failing call on the same thread.
//  * Functions that allocate results (vk_queue_pop, vk_channel_recv,
//    vk_stage_runs_1d/2d, vk_make_ring_channels) return memory owned by the
//    caller; release it with vk_free() (and, for the channel arrays,
//    vk_channel_delete() on each element first).
//  * Opaque handle types (vk_device, vk_stream, vk_queue, vk_channel,
//    vk_overlap) are heap-allocated wrappers around the C++ objects; create
//    with vk_*_new and destroy with vk_*_delete. Handles are not thread-safe
//    to destroy while another thread uses them.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Status codes (mirror vkernels::Code)                                */
/* ------------------------------------------------------------------ */
enum {
  VK_OK = 0,
  VK_ERROR_INVALID_ARGUMENT = 1,
  VK_ERROR_OUT_OF_RANGE = 2,
  VK_ERROR_UNSUPPORTED = 3,
  VK_ERROR_INTERNAL = 4
};

/* ------------------------------------------------------------------ */
/* Version / config                                                    */
/* ------------------------------------------------------------------ */

/* Static version string of the C++ library ("0.1.0"). */
const char* vk_version(void);

/* 1 when the library was compiled with CUDA enabled, else 0. */
int vk_has_cuda(void);

/* Thread-local message of the most recent failing call ("" if none). */
const char* vk_last_error(void);

/* Status code of the most recent failing call (VK_OK if none). */
int vk_last_error_code(void);

/* Free memory returned by vk_* functions that allocate results. */
void vk_free(void* p);

/* ------------------------------------------------------------------ */
/* kernels: element-wise, reduce, gemm (src/c/vkernels/kernels)         */
/* ------------------------------------------------------------------ */

/* out = a + b. Lengths must match; raises VK_ERROR_INVALID_ARGUMENT. */
int32_t vk_add(const float* a, size_t a_len, const float* b, size_t b_len,
               float* out, size_t out_len);

/* out = alpha * x. Lengths must match. */
int32_t vk_scale(const float* x, size_t x_len, float alpha, float* out,
                 size_t out_len);

/* out = max(x, 0). Lengths must match. */
int32_t vk_relu(const float* x, size_t x_len, float* out, size_t out_len);

/* *out = sum(x). x must be non-empty. */
int32_t vk_sum(const float* x, size_t x_len, float* out);

/* *out = max(x). x must be non-empty. */
int32_t vk_max(const float* x, size_t x_len, float* out);

/* C = alpha * A @ B + beta * C. A is M*K, B is K*N, C is M*N elements. */
int32_t vk_gemm(size_t M, size_t N, size_t K, float alpha, const float* A,
                size_t A_len, const float* B, size_t B_len, float beta,
                float* C, size_t C_len);

/* ------------------------------------------------------------------ */
/* kernels: gfx942 primitives (src/c/vkernels/kernels/moe.hpp)          */
/*                                                                       */
/* Software fallbacks for the CDNA4-only instructions on gfx942/MI300A.  */
/* direct_lds_fill_bf16 and fp4_to_bf16_dequant take raw pointers; the  */
/* mfma entry takes fixed-size arrays (c: 4 floats, a/b: 2 uint32 each). */
/* ------------------------------------------------------------------ */

/* Copy `elements` bf16 values from global memory to LDS (host: memcpy). */
int32_t vk_direct_lds_fill_bf16(void* lds_dst, const void* global_src,
                                size_t elements);

/* Convert packed fp4 (E2M1, two per byte, low nibble first) to bf16.
 * `out` must have exactly 2*packed_len elements (raises
 * VK_ERROR_INVALID_ARGUMENT otherwise). */
int32_t vk_fp4_to_bf16_dequant(const uint8_t* packed, size_t packed_len,
                               uint16_t* out, size_t out_len, float scale);

/* 1 if async copy should be used by default (gfx942: OFF; host: ON).
 * The K3_NO_ASYNC env var overrides: "0"=ON, "1"=OFF. Never throws. */
int vk_use_async_copy_default(void);

/* K16 bf16 MFMA: c[0..3] += a[0..1] x b[0..1] (16x16x16 bf16, acc fp32).
 * `c` is 4 floats updated in-place; `a` and `b` are 2 packed bf16 uint32_t
 * each. cbsz/abid/blgp are MFMA control flags (ignored on host). */
int32_t vk_mfma_f32_16x16x16bf16(float* c, const uint32_t* a,
                                 const uint32_t* b, int cbsz, int abid,
                                 int blgp);

/* ------------------------------------------------------------------ */
/* kernels: bf16 GEMM (src/c/vkernels/kernels/gemm_bf16.hpp, issue #29)  */
/*                                                                       */
/* C = alpha * A @ B + beta * C, bf16 (uint16) in/out, fp32 accumulate,  */
/* single round-to-nearest-even on store. A is [M,K], B is [K,N],       */
/* C is [M,N] uint16 bit patterns (no length checks in C++; validate in  */
/* the safe bindings, like the Python kernels.py layer).                 */
/* ------------------------------------------------------------------ */

int32_t vk_gemm_bf16(size_t M, size_t N, size_t K, float alpha,
                     const uint16_t* A, const uint16_t* B, float beta,
                     uint16_t* C);

/* Per-shape tuned (bm, bn, bk, threads) MFMA tile the HIP bf16 GEMM
 * should launch for (M, N, K). BK is fixed at 64 for the K3 shapes.
 * Never throws. */
void vk_gemm_bf16_config(size_t M, size_t N, size_t K, int* bm, int* bn,
                         int* bk, int* threads);

/* ------------------------------------------------------------------ */
/* kernels: MLA — absorbed-form Multi-head Latent Attention              */
/* (src/c/vkernels/kernels/mla.hpp, issue #21)                           */
/*                                                                       */
/* q  : [B, H, S_q, kv_lora_rank + qk_rope_head_dim]  fp32              */
/* k_c: [B, S_kv, kv_lora_rank]                       fp32              */
/* k_pe:[B, S_kv, qk_rope_head_dim]                   fp32              */
/* v_c: [B, S_kv, kv_lora_rank]                       fp32              */
/* out: [B, H, S_q, kv_lora_rank]                     fp32              */
/* Two-pass stable softmax, causal mask via (q_start, kv_start).          */
/* ------------------------------------------------------------------ */

int32_t vk_mla_fwd(int B, int H, int S_q, int S_kv, int q_start,
                   int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                   float scale, const float* q, const float* k_c,
                   const float* k_pe, const float* v_c, float* out);

/* Per-shape (bq, bn_kv, threads) tile selector for the HIP MLA kernel.
 * decode (S_q <= 8) -> (1, 64, 64); prefill -> (4, 64, 256). Never throws. */
void vk_mla_config(int S_q, int kv_lora_rank, int qk_rope_head_dim,
                    int* bq, int* bn, int* threads);

/* ------------------------------------------------------------------ */
/* kernels: KDA — Kimi Delta Attention                                    */
/* (src/c/vkernels/kernels/kda.hpp, issue #21)                           */
/*                                                                       */
/* q,k,v : [B, H, S, D] float; g,beta : [B, H, S] float (k L2-normalised  */
/* by the caller). chunk_size must divide S. All fp32 CPU references.     */
/* ------------------------------------------------------------------ */

/* Gated RMSNorm: out[n,d] = (x[n]/rms_n) * weight[d] * silu(gate[n,d]).
 * x/gate [N,D], weight [D], out [N,D]. */
int32_t vk_kda_layer_norm_gated(const float* x, const float* weight,
                                const float* gate, float* out, int N,
                                int D, float eps);

/* Log-gate cumsum: g [B,H,n_chunks,chunk_size] -> (intra_log
 * [B,H,n_chunks,chunk_size] within-chunk INCLUSIVE, inter_log [B,H,n_chunks]
 * cross-chunk EXCLUSIVE). */
int32_t vk_kda_gate_chunk_cumsum(const float* g, float* intra_log,
                                 float* inter_log, int B, int H,
                                 int n_chunks, int chunk_size);

/* Per-token delta-rule oracle (O(S*D^2), the correctness reference). */
int32_t vk_kda_naive_delta_rule_fwd(const float* q, const float* k,
                                    const float* v, const float* g,
                                    const float* beta, float* out, int B,
                                    int H, int S, int D);

/* Chunked delta-rule forward (gate cumsum -> intra -> inter -> output).
 * chunk_size must divide S. */
int32_t vk_kda_delta_rule_fwd(const float* q, const float* k,
                              const float* v, const float* g,
                              const float* beta, float* out, int B, int H,
                              int S, int D, int chunk_size);

/* Within-chunk delta-corrected value solve u_t for chunk `chunk_idx`.
 * inter_state[..,chunk_idx] is C_{c-1} (read; row 0 must be zero). */
int32_t vk_kda_delta_rule_intra(const float* q, const float* k,
                                const float* v, const float* g,
                                const float* beta, const float* intra_log,
                                const float* inter_state, float* u, int B,
                                int H, int S, int D, int chunk_size,
                                int chunk_idx);

/* Cross-chunk state propagation for chunk `chunk_idx`: fills
 * inter_state[..,chunk_idx+1] = C_c. */
int32_t vk_kda_delta_rule_inter(const float* k, const float* v,
                                const float* g, const float* beta,
                                const float* intra_log, const float* u,
                                float* inter_state, int B, int H, int S,
                                int D, int chunk_size, int chunk_idx);

/* Output (intra + inter) combine: o_t = G_{0,t}(C_{c-1} q_t) +
 * sum_{j<=t} G_{j+1,t} beta_j (k_j . q_t) u_j. */
int32_t vk_kda_gla_fwd_o(const float* q, const float* k, const float* g,
                         const float* beta, const float* intra_log,
                         const float* inter_state, const float* u,
                         float* out, int B, int H, int S, int D,
                         int chunk_size);

/* Pack a binary [n_bits] uint8 (each 0 or 1) array into ceil(n/8) bytes,
 * MSB first (bit k -> byte k/8, bit 7 - k%8). */
int32_t vk_kda_pack_bitmatrix(const uint8_t* bits, uint8_t* packed,
                              size_t n_bits);

/* ------------------------------------------------------------------ */
/* kernels: MoE orchestration (mxfp4 quant / sort / scatter-reduce)      */
/* (src/c/vkernels/kernels/moe_aux.hpp, issue #22)                       */
/* ------------------------------------------------------------------ */

/* MXFP4 activation quant: bf16 A [M, hidden] -> (packed [M, hidden/2]
 * uint8 E2M1, scales [M, hidden/group_size] uint8 ue8m0). hidden must be
 * even and divisible by group_size (>= 1). */
int32_t vk_mxfp4_moe_quant(const uint16_t* A, uint8_t* packed,
                           uint8_t* scales, int M, int hidden,
                           int group_size);

/* Gather bf16 A [M, hidden] into sorted row order [EM, hidden] by
 * sorted_ids (>= M*top_k = padding row, zeroed). */
int32_t vk_mxfp4_moe_sort(const uint16_t* A, const int32_t* sorted_ids,
                          uint16_t* A_sorted, int M, int hidden, int top_k,
                          int EM);

/* Gather per-token ue8m0 scales [M, n_groups] into sorted row order
 * [EM, n_groups] by sorted_ids (padding rows zeroed). */
int32_t vk_mxfp4_moe_sort_scales(const uint8_t* scales,
                                 const int32_t* sorted_ids,
                                 uint8_t* scales_sorted, int M,
                                 int n_groups, int top_k, int EM);

/* Routed combine of float32 partials [EM, width] -> out [M, width]
 * (zero-initialised by caller): out[token] += partial[r] * topk_w[r]. */
int32_t vk_mxfp4_moe_scatter_reduce(const float* partial,
                                    const float* topk_w,
                                    const int32_t* sorted_ids, float* out,
                                    int M, int width, int top_k, int EM);

/* Routed combine of a quantized partial [EM, width/2] uint8 E2M1 +
 * [EM, width/group_size] uint8 ue8m0 -> out [M, width] float32 (zero-init),
 * dequantizing inline. width must be divisible by group_size and by 2. */
int32_t vk_mxfp4_moe_scatter_reduce_q(const uint8_t* partial_q,
                                      const uint8_t* partial_s,
                                      const float* topk_w,
                                      const int32_t* sorted_ids, float* out,
                                      int M, int width, int top_k, int EM,
                                      int group_size);

/* ------------------------------------------------------------------ */
/* kernels: fused MXFP4 MoE grouped GEMM                                */
/* (src/c/vkernels/kernels/moe_fused.hpp, issues #22/#23/#28)            */
/*                                                                       */
/* act_scratch [EM, ispp] bf16 and out [M, hidden] fp32 (caller zero-     */
/* inits out, which accumulates). b13/b2 are nullable (nullptr = skip    */
/* bias). activation: 0 = kSwiGLU, 1 = kSiTU. The C++ reference does no  */
/* length checks of its own (BLAS-style); validate shapes in the safe    */
/* bindings (as Python's kernels.py does).                               */
/* ------------------------------------------------------------------ */

int32_t vk_fused_moe_mxfp4(const uint16_t* A, const uint8_t* w13,
                           const uint8_t* w13_scale, const uint8_t* w2,
                           const uint8_t* w2_scale, const int32_t* sorted_ids,
                           const float* topk_w_sorted,
                           const int32_t* expert_ids, uint16_t* act_scratch,
                           float* out, int M, int hidden, int ispp, int top_k,
                           int EM, int group_size, float swiglu_limit,
                           int activation, float beta, float linear_beta,
                           const float* b13, const float* b2);

/* Map the [M, top_k] token->expert routing table to the block-aligned
 * sorted layout. `sorted_ids` must hold max_EM entries and `expert_ids`
 * max_EM/block_size entries (see vk_moe_align_block_size_max_em). On
 * success *out_EM is the padded row count (a multiple of block_size);
 * padding entries in sorted_ids are M*top_k and in expert_ids are -1. */
int32_t vk_moe_align_block_size(const int32_t* topk_ids, int32_t M,
                                int32_t top_k, int32_t block_size,
                                int32_t num_experts, int32_t* sorted_ids,
                                int32_t* expert_ids, int32_t* out_EM);

/* Upper bound on EM for (M, top_k, block_size, num_experts) — the size to
 * allocate `sorted_ids` (and EM/block_size for `expert_ids`). Never throws. */
size_t vk_moe_align_block_size_max_em(int32_t M, int32_t top_k,
                                      int32_t block_size,
                                      int32_t num_experts);

/* ------------------------------------------------------------------ */
/* core: device + stream (src/c/vkernels/core)                          */
/* ------------------------------------------------------------------ */

typedef struct vk_device vk_device;

vk_device* vk_device_new(int index); /* index == -1 selects the default. */
void vk_device_delete(vk_device* d);

int vk_device_index(const vk_device* d);
/* 1 when d can access `other` directly (always 0 on a host build). */
int vk_device_supports_peer(const vk_device* d, const vk_device* other);
int vk_device_eq(const vk_device* a, const vk_device* b);

/* Make this device current / block until its work finishes. On the host
 * build both are no-ops; under CUDA they can raise VK_ERROR_INTERNAL. */
int32_t vk_device_set_current(vk_device* d);
int32_t vk_device_sync(vk_device* d);

typedef struct vk_stream vk_stream;

vk_stream* vk_stream_new(void);
void vk_stream_delete(vk_stream* s);

/* Enqueue `fn(ctx)` to run on the stream, in submission order. The
 * callback runs on the stream's worker thread; it must not call back into
 * the stream. */
int32_t vk_stream_submit(vk_stream* s, void (*fn)(void*), void* ctx);
/* Block the calling thread until every submitted task has run. */
void vk_stream_wait(vk_stream* s);
/* Number of tasks submitted so far (completed + queued). */
size_t vk_stream_submitted(const vk_stream* s);

/* ------------------------------------------------------------------ */
/* comm: topology (src/c/vkernels/comm/topology.hpp)                    */
/* ------------------------------------------------------------------ */

typedef struct vk_topology {
  int32_t rank;
  int32_t world;
  int32_t next; /* (rank + 1) % world */
  int32_t prev; /* (rank - 1 + world) % world */
} vk_topology;

/* *out = ring slot for `rank` of a ring of `world` ranks. */
int32_t vk_ring_rank(int32_t rank, int32_t world, vk_topology* out);

/* Fills `out` with `world` entries, one per rank; the array is malloc'd and
 * owned by the caller (release with vk_free). */
int32_t vk_build_ring_topology(int32_t world, vk_topology** out,
                               size_t* out_count);

/* ------------------------------------------------------------------ */
/* comm: channels (src/c/vkernels/comm/channel.hpp)                     */
/* ------------------------------------------------------------------ */

typedef struct vk_queue vk_queue;   /* BlockingQueue of float32 chunks */
typedef struct vk_channel vk_channel; /* MockChannel (send/recv) */

vk_queue* vk_queue_new(void);
void vk_queue_delete(vk_queue* q);

/* Append one float32 chunk (copied). */
int32_t vk_queue_push(vk_queue* q, const float* data, size_t len);
/* Blocking pop; returns a malloc'd copy of the chunk (vk_free it). */
int32_t vk_queue_pop(vk_queue* q, float** out_data, size_t* out_len);
void vk_queue_close(vk_queue* q);
int vk_queue_closed(const vk_queue* q);

/* Channel that sends into `out` and receives from `in`. */
vk_channel* vk_channel_new(vk_queue* out, vk_queue* in);
void vk_channel_delete(vk_channel* c);

/* Blocking send of one float32 chunk to the peer (copied). */
int32_t vk_channel_send(vk_channel* c, const float* data, size_t len);
/* Blocking receive; returns a malloc'd copy (vk_free it). */
int32_t vk_channel_recv(vk_channel* c, float** out_data, size_t* out_len);
int vk_channel_closed(const vk_channel* c);

/* Build `world` mock channels in a ring: channel[r].send() reaches
 * channel[(r+1) % world].recv(). On success *out is a malloc'd array of
 * `world` channel handles; delete each with vk_channel_delete, then
 * vk_free the array. */
int32_t vk_make_ring_channels(int32_t world, vk_channel*** out,
                              size_t* out_count);

/* ------------------------------------------------------------------ */
/* comm: ring all-reduce (src/c/vkernels/comm/allreduce.hpp)            */
/* ------------------------------------------------------------------ */

/* Run rank `rank` of a ring all-reduce; on success `local` holds the
 * element-wise sum across every rank. `local` is unchanged on error. */
int32_t vk_ring_allreduce_rank(float* local, size_t local_len, int32_t rank,
                               int32_t world, vk_channel* next,
                               vk_channel* prev);

/* ------------------------------------------------------------------ */
/* comm: overlap (src/c/vkernels/comm/overlap.hpp)                      */
/* ------------------------------------------------------------------ */

typedef struct vk_overlap vk_overlap;

vk_overlap* vk_overlap_new(void);
void vk_overlap_delete(vk_overlap* ex);
int vk_overlap_uses_two_streams(const vk_overlap* ex);

/* Run `iters` iterations: compute(i) -> int on stream A, comm(i, value) on
 * stream B; the data dependency is honoured via a per-iteration future. The
 * callbacks run on the executor's worker threads; both must be non-null.
 * On success *out_compute_count and *out_comm_count are both `iters`. */
int32_t vk_overlap_run(vk_overlap* ex, size_t iters,
                       int (*compute)(size_t i, void* ctx), void* compute_ctx,
                       void (*comm)(size_t i, int value, void* ctx),
                       void* comm_ctx, size_t* out_compute_count,
                       size_t* out_comm_count);

/* ------------------------------------------------------------------ */
/* comm: p2p run-list gather (src/c/vkernels/comm/p2p_gather.hpp)       */
/* ------------------------------------------------------------------ */

/* 1-D copy run (mirrors comm::StagedRun1D). */
typedef struct vk_staged_run_1d {
  const void* src;
  size_t dst_offset;
  size_t length;
} vk_staged_run_1d;

/* Strided 2-D tile input (mirrors comm::Gather2DRun). */
typedef struct vk_gather_2d {
  const void* src;
  size_t src_stride;
  size_t dst_offset;
  size_t dst_stride;
  size_t width;
  size_t height;
} vk_gather_2d;

/* Staged 2-D tile (mirrors comm::StagedRun2D). */
typedef struct vk_staged_run_2d {
  const void* src;
  size_t dst_offset;
  size_t src_stride;
  size_t dst_stride;
  size_t width;
  size_t height;
} vk_staged_run_2d;

/* Validate a 1-D run list against `dst` and return the staged runs as a
 * malloc'd array of `*out_count` vk_staged_run_1d (vk_free it). Raises
 * VK_ERROR_INVALID_ARGUMENT on contract violations. */
int32_t vk_stage_runs_1d(const uint8_t* dst, size_t dst_capacity,
                         const void* const* src_ptrs,
                         const size_t* dst_offsets, const size_t* lengths,
                         size_t num_runs, vk_staged_run_1d** out,
                         size_t* out_count);

/* Validate a 2-D run list against `dst` and return the staged tiles as a
 * malloc'd array of `*out_count` vk_staged_run_2d (vk_free it). */
int32_t vk_stage_runs_2d(const uint8_t* dst, size_t dst_capacity,
                         const vk_gather_2d* runs, size_t num_runs,
                         vk_staged_run_2d** out, size_t* out_count);

/* Copy every 1-D run into dst in a single operation (one stream task when
 * `stream` is non-null; synchronous when null). `dst` must outlive the
 * stream. */
int32_t vk_p2p_gather_runs(uint8_t* dst, size_t dst_capacity,
                           const void* const* src_ptrs,
                           const size_t* dst_offsets, const size_t* lengths,
                           size_t num_runs, vk_stream* stream);

/* Copy every 2-D tile into dst in a single operation. */
int32_t vk_p2p_gather_runs_2d(uint8_t* dst, size_t dst_capacity,
                              const vk_gather_2d* runs, size_t num_runs,
                              vk_stream* stream);

/* Legacy seam: one copy per run (one stream task per run, so
 * vk_stream_submitted grows by the run count instead of 1). */
int32_t vk_memcpy_peer_batch_async(uint8_t* dst, size_t dst_capacity,
                                   const void* const* src_ptrs,
                                   const size_t* dst_offsets,
                                   const size_t* lengths, size_t num_runs,
                                   vk_stream* stream);

#ifdef __cplusplus
}  // extern "C"
#endif
