// vkernels/kernels/mla.hpp
//
// Multi-head Latent Attention (MLA) — the attention path of Kimi-K3 (issue
// #21), matching vLLM's TRITON_MLA / AITER *absorbed* semantics.
//
// MLA (DeepSeek-V2/-V3, Kimi-K1/K2/K3) projects the KV cache into a single
// low-rank `kv_lora_rank` latent shared across all `H` heads, plus a small
// *decoupled* RoPE key of `qk_rope_head_dim` that is NOT compressed (so the
// rotary positions survive the projection). At inference time the query is
// *absorbed* — the up-projections W_UQ (query) and W_UK (key) are folded into
// q so the attention scores factor over the latent directly, and only W_UV
// remains to up-project the output off-device (or in a follow-up GEMM).
//
//   q  : [B, H, S_q, D_q]   D_q = kv_lora_rank + qk_rope_head_dim
//                              q[..., 0 : kv_lora_rank]    = q_nope (absorbed)
//                              q[..., kv_lora_rank : D_q]  = q_rope (post-RoPE)
//   k_c : [B, S_kv, kv_lora_rank]   compressed KV latent (1 head, shared)
//   k_pe: [B, S_kv, qk_rope_head_dim] decoupled RoPE key (post-RoPE)
//   v_c : [B, S_kv, kv_lora_rank]   value latent (1 head, shared)
//   out : [B, H, S_q, kv_lora_rank] latent output (→ up-projection W_UV)
//
//   score[b,h,i,j] = scale · ( q_nope[b,h,i] · k_c[b,j]
//                            + q_rope[b,h,i] · k_pe[b,j] )
//   attn  = softmax_causal(score)
//   out[b,h,i] = Σ_j attn[b,h,i,j] · v_c[b,j]
//
// `scale` is the MLA pre-softmax scale, conventionally
//   1 / sqrt(kv_lora_rank + qk_rope_head_dim).
//
// Causality. `q_start` is the global index of query row 0; a key at global
// index `kv_start + j` attends query `q_start + i` iff `kv_start + j <=
// q_start + i` (and `kv_start + j >= 0`). Prefill sets `q_start == kv_start`
// and `S_q == S_kv`; decode sets `S_q == 1`, `q_start == past_len`,
// `S_kv == past_len + 1`.
//
// Two-implementation model:
//   mla.cpp  -- CPU reference (oracle), always compiled, in vkernels::kernels.
//               Two-pass numerically-stable softmax, fp32 throughout.
//   mla.hip  -- HIP kernel (gfx942), compiled with VKERNELS_HAS_HIP, in
//               vkernels::kernels::hip. Online softmax, fp32 accumulation,
//               bf16 storage. Matches the oracle within a relative tolerance.
#include <cstddef>
#include <cstdint>

namespace vkernels::kernels {

// CPU reference (oracle). Computes the absorbed-form MLA forward in fp32 with
// a numerically-stable two-pass softmax and the causal mask above.
//
//   D_q       = kv_lora_rank + qk_rope_head_dim
//   q  : [B, H, S_q, D_q]
//   k_c : [B, S_kv, kv_lora_rank]
//   k_pe: [B, S_kv, qk_rope_head_dim]
//   v_c : [B, S_kv, kv_lora_rank]
//   out: [B, H, S_q, kv_lora_rank]   (may alias v_c only when S_q==S_kv and
//                                     H==1; otherwise must not overlap)
void mla_fwd_cpu(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
                 int kv_lora_rank, int qk_rope_head_dim, float scale,
                 const float* q, const float* k_c, const float* k_pe,
                 const float* v_c, float* out);

// Per-shape (decode vs prefill) launch tile selector. Writes the tile the
// HIP kernel should use:
//   decode  (S_q <= 8) : one query row per block, BN_kv keys per pass,
//                        one wavefront (64 threads)
//   prefill (S_q >  8) : BQ query rows × BN_kv key columns per block, 256
//                        threads (4 wavefronts); bn_kv selects the
//                        double-buffered LDS key tile of the fused-head
//                        prefill kernel (4 or 8; issue #151)
void mla_config_for(int S_q, int kv_lora_rank, int qk_rope_head_dim,
                    int* bq, int* bn_kv, int* threads);

// MI300A compute-unit count (gfx942) — the occupancy cap the split-K helper
// fills toward (same convention as the dsa_topk split rule), and the minimum
// number of keys a split's slice must keep so the latent reads still
// coalesce. Both feed mla_fwd_split_for and the HIP split-kernel sizing.
static constexpr int kMlaCus = 228;
static constexpr int kMlaMinSplitKeys = 32;

// Recommended split-K for the HIP decode path (issue #82). A decode grid with
// tiny `B*H*S_q` (e.g. H=1, S_q=1, S_kv=8192: a single 64-thread block on
// MI300A's 228 CUs) cannot fill the machine, so the kernel splits the S_kv key
// window across blocks -- each writing a partial-softmax output + per-split
// lse, combined in fixed order (the scheme proven on dsa_sparse_fwd_split).
// Returns the split count for `mla_fwd`; 1 means the plain single-block path
// (prefill shapes, or grids that already fill the CUs). Host-pure; unit-tested
// in tests/kernels/attn/test_mla.cpp (MlaSplitFor).
// Formula (occupancy rule, as in dsa_topk_logits_split_for): fill the CUs the
// plain grid leaves idle, capped so every split keeps >= kMlaMinSplitKeys
// keys (coalescing floor).
//
// Issue #151 extends the rule to CHUNKED / SHORT PREFILL (S_q > 8): the
// fused-head prefill grid (BQ=4 rows per block, mla_prefill_heads_per_block
// heads per block) leaves CUs idle when tiles·head_groups·B << kMlaCus
// (e.g. H=1, S_q=64, S_kv=8192: 16 blocks on 228 CUs, 2% of HBM). Such
// grids split their S_kv window too, via the BQ=4 split variant + the #82
// combine kernel. Prefill grids that already fill the CUs return 1 (the
// fused kernel handles their KV reuse).
int mla_fwd_split_for(int B, int H, int S_q, int S_kv);

// Heads per block for the fused-head prefill kernel (issue #151). k_c/k_pe/
// v_c are shared across heads, so batching `bh` heads per block divides the
// per-query-tile KV re-reads by bh. The heuristic caps bh at H and at
// 16/bq (1024-thread workgroup limit), rounded down to a power of two so
// the (row, head) -> wavefront mapping divides evenly. Host-pure.
int mla_prefill_heads_per_block(int bq, int H);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// HIP MLA forward (gfx942). Online softmax in fp32, bf16-tolerant against the
// CPU oracle. Same contract as mla_fwd_cpu. q/k_pe/k_c/v_c are float* on the
// host side (the kernel converts to its working precision internally); device
// pointers must reside in device or host-pinned memory.
//
// stream (issue #69/#45 convention, matching kda_delta_rule_fwd_with_scratch):
// pass a hipStream_t (as void*) to enqueue on a caller stream WITHOUT any
// internal synchronisation or blocking allocation — required for graph
// capture; the caller owns ordering/sync. nullptr (default/legacy) keeps the
// default-stream-0 behaviour. The split-K decode path (issue #82) enqueues on
// the caller stream too; if its grow-only workspace would need a resize
// (hipMalloc/hipFree — device-synchronising, capture-illegal) while `stream`
// is capturing, it falls back to the capture-safe non-split kernel instead.
void mla_fwd(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
             int kv_lora_rank, int qk_rope_head_dim, float scale,
             const float* q, const float* k_c, const float* k_pe,
             const float* v_c, float* out, void* stream = nullptr);

// `mla_fwd` with an explicit split-K count (perf/tuning harness entry,
// same relationship to mla_fwd as dsa_sparse_fwd_split has to dsa_sparse_fwd):
// split <= 1 or a failed split launch (capture-unsafe workspace resize)
// falls back to the plain single-block path. Same stream contract.
void mla_fwd_with_split(int B, int H, int S_q, int S_kv, int q_start,
                        int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                        float scale, const float* q, const float* k_c,
                        const float* k_pe, const float* v_c, float* out,
                        int split, void* stream = nullptr);

// Explicit-tile entry point (offline autotuner hook). Dispatches the
// concrete (bq, bn_kv) tile; threads is derived as max(bq,1)*64 capped at 256.
// Same stream contract as mla_fwd.
void mla_fwd_with_tile(int B, int H, int S_q, int S_kv, int q_start,
                       int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                       float scale, const float* q, const float* k_c,
                       const float* k_pe, const float* v_c, float* out,
                       int bq, int bn_kv, void* stream = nullptr);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
