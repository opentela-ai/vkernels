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
//                        threads (4 wavefronts)
void mla_config_for(int S_q, int kv_lora_rank, int qk_rope_head_dim,
                    int* bq, int* bn_kv, int* threads);

}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {

// HIP MLA forward (gfx942). Online softmax in fp32, bf16-tolerant against the
// CPU oracle. Same contract as mla_fwd_cpu. q/k_pe/k_c/v_c are float* on the
// host side (the kernel converts to its working precision internally); device
// pointers must reside in device or host-pinned memory.
void mla_fwd(int B, int H, int S_q, int S_kv, int q_start, int kv_start,
             int kv_lora_rank, int qk_rope_head_dim, float scale,
             const float* q, const float* k_c, const float* k_pe,
             const float* v_c, float* out);

// Explicit-tile entry point (offline autotuner hook). Dispatches the
// concrete (bq, bn_kv) tile; threads is derived as max(bq,1)*64 capped at 256.
void mla_fwd_with_tile(int B, int H, int S_q, int S_kv, int q_start,
                       int kv_start, int kv_lora_rank, int qk_rope_head_dim,
                       float scale, const float* q, const float* k_c,
                       const float* k_pe, const float* v_c, float* out,
                       int bq, int bn_kv);

}  // namespace vkernels::kernels::hip
#endif  // VKERNELS_HAS_HIP
