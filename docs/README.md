# vkernels — Supported Kernels & Primitives

This document lists every kernel and communication primitive in vkernels,
along with the mathematical computation each performs. Every operation follows
the **two-implementation model**: a CPU reference (oracle, always compiled)
and a GPU-accelerated path (CUDA or HIP, compiled when the toolkit is present).

All kernels operate on **float32** unless noted. Inputs are C-contiguous
arrays. Contract violations raise exceptions (C++), `ValueError` (Python),
or `Error::InvalidArgument` (Rust).

---

## Element-wise kernels

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `add(a, b)` | `out[i] = a[i] + b[i]` | float32 | CUDA |
| `scale(x, α)` | `out[i] = α · x[i]` | float32 | CUDA |
| `relu(x)` | `out[i] = max(x[i], 0)` | float32 | CUDA |

- **File**: `src/c/vkernels/kernels/elementwise.cpp` (CPU), `.cu` (CUDA)
- **Python**: `vkernels.kernels.add / scale / relu`
- **Rust**: `vkernels::kernels::add / scale / relu`

---

## Reduction kernels

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `sum(x)` | `Σ x[i]` (float32-accumulated) | float32 → float32 | CUDA (two-stage) |
| `max(x)` | `max x[i]` | float32 → float32 | CUDA |

- **File**: `src/c/vkernels/kernels/reduce.cpp` (CPU), `.cu` (CUDA)
- **Python**: `vkernels.kernels.sum / max`
- **Rust**: `vkernels::kernels::sum / max`
- Empty input raises an error on all backends.

---

## GEMM (SGEMM)

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `gemm(M, N, K, α, A, B, β, C)` | `C = α · A @ B + β · C` | float32 | CUDA (tiled 16×16) |

- **File**: `src/c/vkernels/kernels/gemm.cpp` (CPU), `.cu` (CUDA)
- **Python**: `vkernels.kernels.gemm(A, B, alpha=1.0, beta=0.0)` (shapes inferred)
- **Rust**: `vkernels::kernels::gemm(M, N, K, alpha, A, B, beta, C)` (explicit dimensions)
- Row-major layout. `A` is M×K, `B` is K×N, `C` is M×N.
- CUDA kernel uses shared-memory tiling with 16×16 thread blocks.

---

## GEMM (bf16 MFMA, gfx942)

A tiled bf16 dense matrix multiply built on the AMD K16 bf16 MFMA
(`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`) for the **Kimi-K3 projection
shapes** — the ones that today fall back to AITER's untuned
"torch solution:0" because `bf16_tuned_gemm.csv` has no gfx942 entries.

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `gemm_bf16_cpu(M, N, K, α, A, B, β, C)` | `C = α·A@B + β·C` (per-output fp32 dot, single RNE bf16 store) | bf16 (uint16_t) | CPU (oracle) |
| `gemm_bf16_config_for(M, N, K, &bm, &bn, &bk, &threads)` | serving `M≤64`→`(16,16,64)` (measured; `BN=16` saturates the 228 CUs, 1.4–2.9× over `BN=64`), warmup `M>64`→`(64,64,256)`, `BK=64` | — | CPU |
| `hip::gemm_bf16(M, N, K, α, A, B, β, C)` | tiled K16-MFMA GEMM, cooperative `uint2` loads, M/N/K bounds-checked | bf16 (uint16_t) | HIP |

- **Layout**: `A` is M×K, `B` is K×N (the **transposed** projection
  weight `W[N,K].T`), `C` is M×N. All K3 `K` are multiples of 64, so
  `BK=64` never triggers its (defensive) K bounds-check; `N` is a
  multiple of 16 (e.g. 6288 is 393×16, not a multiple of 64), handled by
  a column bounds-check.
- **MFMA fragment layout** mirrors the empirically verified `mfma_k64_pf`
  helper in `moe_fused.hip` (A `m=lane%16`, B `n=lane%16`, C
  `col=lane%16, row=(lane/16)*4+i`), generalised to arbitrary `(BM, BN)`.
- **Files**: `src/c/vkernels/kernels/gemm_bf16.{hpp,cpp,hip}`;
  `hip::gemm_bf16_with_config` (the explicit-tile dispatcher used by the
  autotuner) is forward-declared by the harnesses, not in the public
  header (keeps discovery at three entries).
- **Tests**: `tests/kernels/gemm/test_gemm_bf16.cpp` (host),
  `meta/benchmarks/test_gemm_bf16_correct.hip` (device vs CPU).
- **Docs**: [kernels/gemm_bf16.md](kernels/gemm_bf16.md),
  [performance/gemm-bf16/gfx942.md](performance/gemm-bf16/gfx942.md)

---

## Attention kernels — Kimi-K3 / GLM-5.3 (gfx942 / MI300A)

These replace the AITER / tilelang / Triton attention kernels that
GPU-fault or JIT-abort on gfx942 (CDNA3), so Kimi-K3 and GLM-5.3-Flash
serving no longer needs a vendor-specific fallback. Every kernel follows
the **two-implementation model**: a CPU reference (oracle, always
compiled) and a HIP path (`VKERNELS_HAS_HIP`), validated against the
oracle on gfx942.

### MLA — Multi-head Latent Attention (absorbed form, issue #21)

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `mla_fwd_cpu(B,H,S_q,S_kv,q_s,kv_s,lr,rhd,scale,q,k_c,k_pe,v_c,out)` | `out[i] = Σ_j softmax_causal(scale·(q_nope[i]·k_c[j] + q_rope[i]·k_pe[j]))[i,j] · v_c[j]` | float32 | HIP |
| `hip::mla_fwd(...)` | online softmax, fp32 accum, bf16 storage | bf16 (device) / fp32 (oracle) | HIP |
| `mla_config_for(S_q, lr, rhd, &bq, &bn_kv, &th)` | decode `S_q≤8`→1 row·1 wf; prefill→BQ rows·4 wf | — | — |

- `q` is `[B,H,S_q, kv_lora_rank+qk_rope_head_dim]` (`[q_nope | q_rope]`);
  `k_c`/`v_c` are `[B,S_kv,kv_lora_rank]`; `k_pe` is `[B,S_kv,qk_rope_head_dim]`.
  Causality is `kv_start + j <= q_start + i` (chunked prefill masked correctly).
- **Files**: `src/c/vkernels/kernels/mla.{hpp,cpp,hip}`
- **Python**: `vkernels.kernels.mla_fwd / mla_config`
- **Rust**: `vkernels::kernels::mla_fwd / mla_config`
- **Docs**: [kernels/mla.md](kernels/mla.md)

### KDA — Kimi Delta Attention (gated delta-rule, issues #21, #45)

A per-key-dim gated delta-rule linear attention: state `S_t` (`D×D` per
head) is decayed by a **per-key-dim** forget gate `g_t[k]`, then updated by
the delta correction `β_t (v_t − a_t) ⊗ k_t`, with output `o_t = S_t · q_t`.

| Function | Computation | Notes |
|---|---|---|
| `kda_naive_delta_rule_fwd_cpu(q,k,v,g,beta,out,B,H,S,D)` | per-token oracle (`O(S·D²)`), **per-key-dim** gate `[B,H,S,D]`, post-gate prediction | the K3 / HIP oracle |
| `kda_delta_rule_fwd_cpu(q,k,v,g,beta,out,B,H,S,D,chunk)` | chunked forward (gate cumsum → intra → inter → output) | the **standard** (scalar-gate) rule; cross-checked against its own inline oracle |
| `hip::kda_delta_rule_fwd(...)` | cooperative per-token recurrence, `D×D` state in LDS | K3 oracle; matches `kda_naive_delta_rule_fwd_cpu` |
| `hip::kda_delta_rule_fwd_with_scratch(...,state,...)` | caller-owned `[B,H,D,D]` state scratch (multi-turn decode: pre-fill `S_0`, read back `S_S`) | K3 serving path (issue #45) |
| `kda_layer_norm_gated_cpu / hip::kda_layer_norm_gated` | gated RMSNorm × `silu(gate)` pre-attention normaliser | |
| `kda_gate_chunk_cumsum_cpu / hip::kda_gate_chunk_cumsum` | intra inclusive + inter exclusive log-gate cumsum | |
| `kda_delta_rule_intra_cpu` / `kda_delta_rule_inter_cpu` / `kda_gla_fwd_o_cpu` | the chunked pieces (standard rule), standalone | |
| `kda_pack_bitmatrix_cpu / hip::kda_pack_bitmatrix` | MSB-first bit packing of binary gate/routing matrices | |

- `k` is L2-normalised by the caller; `g` is `[B,H,S,D]` in normal space
  `(0,1]`; `β` is `[B,H,S]` scalar per (token, head).
- **Files**: `src/c/vkernels/kernels/kda.{hpp,cpp,hip}`
- **Python**: `vkernels.kernels.{kda_layer_norm_gated, kda_gate_chunk_cumsum,
  kda_naive_delta_rule_fwd, kda_delta_rule_intra, kda_delta_rule_inter,
  kda_gla_fwd_o, kda_delta_rule_fwd, kda_pack_bitmatrix}`
- **Rust**: `vkernels::kernels::{kda_layer_norm_gated, kda_gate_chunk_cumsum,
  kda_naive_delta_rule_fwd, kda_delta_rule_fwd, kda_delta_rule_intra,
  kda_delta_rule_inter, kda_gla_fwd_o, kda_pack_bitmatrix}`
- **Docs**: [kernels/kda.md](kernels/kda.md)

### DSA — DeepseekSparseAttn sparse-MLA forward (issue #51)

A sparse-MLA forward (GLM-5.3-Flash / DeepSeek-V3): an external *indexer*
has already picked, per query token, the `topk` most relevant KV tiles
(`indices`); the kernel scores each query against exactly those selected
keys and produces the combined output. The `tail_dim == 0` shape
(GLM-5.3-Flash) that tilelang *cannot compile* is a first-class,
hand-checked case (the rope-tail dot is skipped at runtime).

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `dsa_sparse_fwd_cpu(...)` | two-pass base-2 softmax over the `topk` selected keys; masked keys → weight 0 | float32 | — |
| `hip::dsa_sparse_fwd(...)` | online softmax, fp32 accum, bf16 storage | bf16 (device) / fp32 (oracle) | HIP |
| `dsa_config_for(S_q,H,dim,topk,&bq,&th,&block_I,&inner_iter)` | decode `S_q≤8`→1 row·1 wf; prefill→BQ rows·4 wf | — | — |
| `dsa_topk_logits_cpu(...)` / `hip::dsa_topk_logits(...)` | paged-MQA gated top-k **logits** — the FIRST stage (kpool logits) that feeds this forward (kpool>1 indexer path, issue #51). FP8 e4m3fnuz Q/K dequantised to fp32 on load; tokens ≥ `seq_len[b]` left unwritten (zero the output first) | fp8 e4m3fnuz (device) / fp32 (oracle) | HIP |

- `q` is `[1,S_q,H, dim+tail_dim]`; `kv` is `[1,S_kv,kv_group, dim+tail_dim]`
  with `d_v = dim − tail_dim`; `kv_group == 1`. `sm_scale = (1/√(dim+tail_dim))·log2(e)`.
- **C ABI**: `vk_dsa_config`, `vk_dsa_sparse_fwd` (host),
  `vk_hip_dsa_sparse_fwd` / `vk_hip_dsa_topk_logits` (device) in `src/c/vkernels/capi/`
- **Files**: `src/c/vkernels/kernels/dsa.{hpp,cpp,hip}`
- **Python**: `vkernels.kernels.dsa_sparse_fwd / dsa_config`
- **Docs**: [kernels/dsa.md](kernels/dsa.md)

### MHC — Multi-head hybrid-attention pre/post (issue #51, part 2)

Two of the tilelang MHC kernels from `sglang/kernels/ops/layernorm/mhc.py`
that JIT-abort on gfx942 (dynamic shared memory > the 64 KB non-optin cap):
the split-k `mhc_pre_gemm_sqrsum` stage-0 kernel and the `mhc_post`
combine. The HIP kernels use *static* shared memory within MI300A's
non-optin cap (no `hipFuncSetAttribute` opt-in).

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `mhc_pre_gemm_sqrsum_cpu(...)` / `hip::mhc_pre_gemm_sqrsum(...)` | `out[n,o] = Σ_h x[n,h]·fn[o,h]` (GEMM `x@fnᵀ`, `hc_mult3 = hc_mult·(2+hc_mult)` cols) **plus** `sqrsum[n] = Σ_h x[n,h]²` | bf16 in (device) / fp32 (oracle) → fp32 out | HIP |
| `mhc_post_cpu(...)` / `hip::mhc_post(...)` | `out[n,j,h] = c[n,j]·d[n,h] + Σ_k a[n,k,j]·b[n,k,h]` (post-attention combine) | bf16 `b`/`d`/`out` (device) / fp32 (oracle) | HIP |

- **C ABI**: `vk_mhc_pre_gemm_sqrsum`, `vk_mhc_post` (host),
  `vk_hip_mhc_pre_gemm_sqrsum`, `vk_hip_mhc_post` (device) in `src/c/vkernels/capi/`
- **Files**: `src/c/vkernels/kernels/mhc.{hpp,cpp,hip}`
- **Python**: `vkernels.kernels.mhc_pre_gemm_sqrsum / mhc_post`
- **Docs**: [kernels/mhc.md](kernels/mhc.md)

---

## MoE (Mixture of Experts) — AMD gfx942 / CDNA3 low-level primitives

These fill gaps where CDNA4-only (gfx950) instructions used in the AITER flydsl
MXFP4 fused-MoE path do not lower on gfx942.

### #12 — Software direct-to-LDS fill

| Function | Computation | Data type | Backend |
|---|---|---|---|
| `direct_lds_fill_bf16(lds_dst, global_src, elements)` | Copy `elements` bf16 values from global → LDS | bf16 (uint16_t) | HIP (vectorised loads), CPU (memcpy) |

Replaces CDNA4's `rocdl.raw_ptr_buffer_load_lds` with vectorised global loads
(`global_load_dwordx4`) into VGPRs followed by LDS stores at lane-major offsets.

- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)

### #13 — Software fp4→bf16 dequant

| Function | Computation | Data type | Backend |
|---|---|---|---|
| `fp4_to_bf16_dequant(packed, scale)` | `out[2i], out[2i+1] = fp4_to_bf16(packed[i] & 0xF, (packed[i] >> 4) & 0xF) · scale` | fp4 (packed uint8) → bf16 (uint16_t) | HIP, CPU |

Decodes E2M1 microscaling format (sign|2-bit-exponent|1-bit-mantissa, two
values per byte, low nibble first).

- **Representable fp4 values**: 0, ±0.25, ±1.0, ±1.5, ±2.0, ±3.0, ±∞, NaN
- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.fp4_to_bf16_dequant(packed, scale=1.0)`

### #14 — Platform async-copy gate

| Function | Computation |
|---|---|
| `use_async_copy_default()` | Returns `true` if async copy should be enabled; `false` on gfx942 |

- Defaults to OFF on gfx942 (CDNA3, MI300X/A) where the async-copy path
  misbehaves. ON everywhere else.
- Override with env var `K3_NO_ASYNC=0` (force ON) or `=1` (force OFF).

- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)

### #15 — K16 bf16 MFMA

| Function | Computation | Data type |
|---|---|---|
| `mfma_f32_16x16x16bf16(c[4], a[2], b[2])` | `C₀₋₃ += A₀₋₁ ⊗ B₀₋₁` (16×16×16 bf16, fp32 accum) | bf16 packed → fp32 |

A single `v_mfma_f32_16x16x16bf16_1k` instruction on gfx942. K32 bf16
MFMA (CDNA4-only) is emulated by calling this K16 function twice:
once for K=0..15 (low halves of A/B), once for K=16..31 (high halves).

The standalone primitive issues the instruction via inline asm with pinned
VGPRs; the fused-MoE kernel instead calls the clang builtin
`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` directly, which keeps register
allocation correct where the pinned-VGPR form corrupted accumulators and
crashed for >64-thread blocks on gfx90a.

**Fragment layout** (one warp, lane `0..63`):
- A operand: `m = lane % 16`, `a[i]` packs rows `m` for K `k0+i` (`k0 = (lane/16)*4`)
- B operand: `n = lane % 16`, `b[i]` packs column `n` for K `k0+i`
- C accumulator: `col = lane % 16`, `row = (lane/16)*4 + i` — four **consecutive rows** per thread (not transposed)

- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.mfma_f32_16x16x16bf16(c, a, b)`
- **Docs**: [kernels/moe.md](kernels/moe.md)

---

## MoE Aux — MXFP4 orchestration (quant, sort, scatter-reduce)

The five per-block data-movement primitives from issue #22 that bracket a
grouped MXFP4 GEMM: per-token / per-group activation quantization,
token→expert gather (activation **and** scales), and the routed
scatter-reduce combine (fp32 partials, plus a bandwidth-reduced MXFP4 form
that dequantizes inline). On gfx950 these are AITER's
`module_moe_mxfp4_aux` (its ~82 KB LDS exceeds the gfx942 / MI300A 64 KB
limit), so vkernels re-implements them as portable host references + HIP
kernels.

| Function | Computation |
|---|---|
| `mxfp4_moe_quant(A, group_size)` | bf16 → packed E2M1 + ue8m0 per group (low nibble = even K) |
| `mxfp4_moe_sort(A, sorted_ids, top_k)` | gather `A` into expert-grouped, block-aligned `[EM, hidden]` (pad rows zeroed) |
| `mxfp4_moe_sort_scales(scales, sorted_ids, top_k)` | same gather for the per-token ue8m0 scales |
| `mxfp4_moe_scatter_reduce(partial, topk_w, sorted_ids, M, width, top_k)` | bias-free weighted scatter-add of fp32 partials → `out[M, hidden]` |
| `mxfp4_moe_scatter_reduce_q(partial_q, partial_s, topk_w, …, group_size)` | same combine with the partial in MXFP4, dequantized inline |

- `moe_align_block_size` (above) produces `sorted_ids`; the pipeline is
  `align → sort → quant → sort_scales → fused_moe_mxfp4 → scatter_reduce[_q]`.
- Scale bytes are clamped to `[1, 254]` (never `0`); `0xFF` is the explicit
  zero-group flag. Largest finite E2M1 is `FP4_MAX = 3.0`.
- **Files**: `src/c/vkernels/kernels/moe_aux.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.mxfp4_moe_{quant,sort,sort_scales,scatter_reduce,scatter_reduce_q}`
- **Docs**: [kernels/moe_aux.md](kernels/moe_aux.md)

---

## MoE Fused — End-to-end MXFP4 grouped GEMM

Implements the full xkernels `fused_moe_mxfp4` interface by wiring together
the low-level primitives above. Two HIP kernels (plus a routing-weight gather):

### Kernel 0 — gate_up + SwiGLU (`gateup_swiglu_kernel`)
```
act[EM, ispp] = silu(clamp(A_sorted @ w13_gate + b13_gate, L))
              · clamp(A_sorted @ w13_up   + b13_up,   L)
```
Gate and up are accumulated in the same kernel and the SwiGLU product is
rounded to bf16 **once**, matching the xkernels oracle exactly (a split
3-kernel pipeline introduced a spurious intermediate bf16 rounding). The
gate value is clamped to `L` *before* the sigmoid (SwiGLU clamp).

With `activation="situ"` (Kimi-K3, matches vLLM's `situ_and_mul`) the
epilogue is a soft-capped gate × up instead of SwiGLU:
`gate' = beta·tanh(gate/beta)·sigmoid(gate)`,
`up' = linear_beta·tanh(up/linear_beta)` (no `swiglu_limit` clamp).
`activation` (`"swiglu"` | `"situ"`), `beta` and `linear_beta` are the
MoE-level knobs.

### Kernel 1 — down + routed combine (`down_combine_kernel`)
```
out[M, hidden] += act @ w2^T · topk_w_sorted + b2
```
(scatter-add by token row via `atomicAdd`, weighted by the routing weight)

| Function | Tile constants | Data types | Backend |
|---|---|---|---|
| `fused_moe_mxfp4(..., block_size=16)` | decode: BM=16, BN=64, BK=64, 64 threads/block; `activation`∈{`swiglu`,`situ`} (K3) | bf16 activations, fp4 (E2M1) weights with ue8m0 scales, fp32 output | HIP, CPU |
| `fused_moe_mxfp4(..., block_size=64)` | prefill: BM=64, BN=64, BK=64, 256 threads/block; `activation`∈{`swiglu`,`situ`} | same | HIP |

`block_size` selects the tile config: `16` = decode (16×64, 64 threads),
`64` = prefill (64×64, 256 threads). The caller aligns with the matching
`block_size` (so `expert_ids` is indexed per 16- or 64-row block).

- Dequantization (E2M1 + ue8m0) is done inline during the K-loop — no full
  bf16 materialization of the 138 GB expert weight buffer.
- MFMA via `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`; A-tiles staged with
  vectorised `uint2` global loads; N-dimension is split across `blockIdx.z`
  (fixed 64-thread blocks, avoiding gfx90a pinned-VGPR crashes).
- `act_scratch` is indexed by **sorted row** (`EM`), not token (`M`) — this
  is required for `top_k > 1`, where the same token appears in multiple
  experts and would otherwise race.
- The caller must zero-initialise `out` (down-combine accumulates into it).

### Distributed (TP / EP / PP) — issue #18

The fused kernel is single-device; `vkernels.dist` (C++ `dist/dist_moe.hpp`,
Python `vkernels.dist`) shards the weights so per-rank shards are consumed
verbatim by the fused kernel's stage functions, and provides the
orchestration around them:

- **TP** — gate/up weights split along `hidden`, down weights along `ispp`;
  the linear stages are separated (`moe_gateup_cpu` / `moe_down_cpu`) so
  rank partials can be all-reduced *before* the nonlinear epilogues.  The
  multi-rank forward matches the CPU oracle.
- **EP** — experts partitioned across ranks; `moe_ep_dispatch` produces the
  all-to-all / sort re-layout with local expert ids.
- **PP** — `pp_boundary_send`/`recv` fix the stage-boundary transfer
  interface (graph-capturable primitive, issue #10) and `round_bf16`
  re-quantises the bf16 stage input.
- **Files**: `src/c/vkernels/dist/dist_moe.cpp` (+ `.hpp`), stage split in
  `src/c/vkernels/kernels/moe_fused.{cpp,hip}`, Python `src/python/vkernels/dist.py`
- **Tests**: `tests/kernels/moe/test_dist_moe.cpp`,
  `tests/python/test_dist.py`, `meta/benchmarks/test_moe_fused_dist_correct.hip`
  (GPU vs CPU oracle)
- **Docs**: [kernels/moe_dist.md](kernels/moe_dist.md)

### Expert alignment helper

| Function | Computation |
|---|---|
| `moe_align_block_size(topk_ids, M, top_k, block_size, num_experts)` | Maps `[M, top_k]` token→expert routing into block-aligned `sorted_ids` and `expert_ids` |

- `sorted_ids` stores the **flat topk index** (`token*top_k + sel`), padded
  per expert with `M*top_k`. This preserves the selection index so the
  routing-weight gather (`sw[i] = tw[sorted_ids[i]]`) is correct for
  `top_k > 1`. Consumers derive `token = flat / top_k`.
- **Files**: `src/c/vkernels/kernels/moe_fused.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.moe_align_block_size / fused_moe_mxfp4`
  (CPU-reference backed; see [python-bindings.md](python-bindings.md))
- **Docs**: [kernels/moe_fused.md](kernels/moe_fused.md)

### Verified performance (vs xkernels torch-loop)

> **Full benchmark records** (reproduce commands, per-M tables, primitives,
> caveats, journal): [`docs/performance/moe-fused/gfx90a.md`](performance/moe-fused/gfx90a.md)
> (MI250X) and [`docs/performance/moe-fused/gfx942.md`](performance/moe-fused/gfx942.md)
> (MI300A).

E=256, hidden=4096, ispp=512, top_k=6. Latency in ms (lower is better);
TFLOP/s is arithmetic on the *padded* EM rows, so the decode regime is
heavily padding-dominated.

| M | vkernels HIP (MI250X) | vkernels HIP (MI300A) | xkernels torch (MI250X) | xkernels torch (MI300A) |
|---|---:|---:|---:|---:|
| 1  | 0.66 | 0.45 | 6.0  | 4.4  |
| 8  | 3.4  | 0.89 | ~30  | ~15  |
| 16 | 4.1  | 2.2  | ~40  | ~30  |
| 48 | 10.4 | 4.0  | 162.2 | 127.2 |

GPU results are exact matches against the CPU reference
(`max_rel < 0.00001` decode, `< 0.02` prefill) on both gfx90a and gfx942.

The prefill config (E=8, top_k=2, denser routing) wins once each expert
fills its 64-row block: ~1.2× on gfx90a (M ≥ 512), ~1.3–1.5× on gfx942
(M ≥ 1024). With sparse routing (few tokens/expert) the 64-row padding
dominates and decode stays faster — see [kernels/moe_fused.md](kernels/moe_fused.md)
for full numbers.

---

## Communication primitives

### Ring all-reduce

| Function | Computation |
|---|---|
| `ring_allreduce_rank(local, rank, world, next, prev)` | Sum-reduces `local` in-place across `world` ranks via ring topology |
| `ring_allreduce(locals)` | Simulates all ranks in one process: every rank's `local` becomes `Σ locals[0..world-1]` |

- **File**: `src/c/vkernels/comm/allreduce.cpp`
- **Python**: `vkernels.comm.ring_allreduce(locals)`
- **Rust**: `vkernels::comm::ring_allreduce(&[a, b])`

### P2P run-list gather

Single-launch gather of many disjoint byte-runs from peer UVA into a local
scratch buffer, replacing per-run `cudaMemcpyPeerAsync` loops.

| Function | Computation |
|---|---|
| `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, N)` | For each `i`: `dst[dst_offsets[i]:...] = peer[src_ptrs[i]:...]` (1-D) |
| `p2p_gather_runs_2d(dst, runs)` | Strided 2-D tiles: `height × width` bytes per run, with independent src/dst strides |

- Adaptive dispatch: copy engine below the crossover run count, single kernel
  launch above it.
- Plan API (`P2PGatherPlan1D/2D`) for reuse across layer launches.
- Vectorized 16-byte (`uint4`) path for aligned runs.

- **File**: `src/c/vkernels/comm/p2p_gather.cpp` (CPU), `.cu` (CUDA)
- **Docs**: [kernels/p2p-gather.md](kernels/p2p-gather.md)
- **Performance**: [performance/p2p-gather/](performance/p2p-gather/)

### P2P KV restore (fused)

Fuses peer gather + indexed scatter into one kernel:

| Function | Computation |
|---|---|
| `p2p_kv_restore(k_dst, v_dst, slot_ids, peer_src_ptrs, ...)` | Reads KV data directly from peer UVA and writes into indexed K/V slot destinations |
| `kv_scatter(k_dst, v_dst, scratch, slot_ids, ...)` | Indexed scatter of an already-gathered contiguous scratch buffer (the second stage, for the PR #9 gather+scatter baseline) |

- Eliminates the intermediate scratch buffer and separate scatter launch.
- **File**: `src/c/vkernels/comm/p2p_kv_restore.cpp` (CPU), `.cu` (CUDA)

**Prepared plan (issue #27)** — `P2PKvRestorePlan` (host + CUDA) and the C
ABI `vkernels_p2p_kv_restore_plan_t` validate the slot map and upload the
page descriptors ONCE; `execute(k_dst, v_dst, source_layer_offset_bytes,
stream)` then launches a single page-by-token-group kernel. The destination
is taken per call because KVAAS/SGLang keep a distinct K/V pair per model
layer — one plan fans one run list out across all 40 layers with no
per-layer allocation or H2D copy. Three creation modes:

- Host `slot_ids` (`const int*`): validated at create, owned copy uploaded.
- `from_device_slots` (`const int*`, e.g. SGLang's `device_indices`):
  borrows the device pointer, no D2H sync, no content validation.
- `from_device_slots_int64` (`const int64_t*`): SGLang's `torch.int64`
  indices; converted at create (device-side on CUDA, no sync) into an owned
  int32 buffer so the caller may free the int64 buffer immediately.

The C ABI mirrors all three (`vkernels_p2p_kv_restore_plan_create`,
`..._create_device_slots`, `..._create_device_slots_int64`) plus
`vkernels_p2p_kv_restore_plan_execute_offset(plan, k_dst, v_dst, offset,
stream)`.

### P2P KV donate (fused) — issue #36

The donation-side mirror of `p2p_kv_restore` — the data flow is reversed:
fuses an indexed gather of local paged-KV slots with the peer store into
one kernel. KVAAS today materializes a full all-layer packed scratch
tensor before peer DMA (`pack_pages`: a PyTorch advanced-index gather of
K/V per layer into `[pages, layers, page_size, 2, heads, dim]`, then
scratch-to-peer copies, with the scratch pinned until the completion ACK).
The donate skips the scratch entirely: it reads arbitrary local K/V slots
and writes directly into the layer-major peer-page destination through
peer-accessible UVA pointers, eliminating the scratch allocation, the
extra local-HBM read/write pass, and the separate peer copy.

| Function | Computation |
|---|---|
| `p2p_kv_donate_layer(k_src, v_src, slot_ids, peer_dst_ptrs, dst_page_offsets, ...)` | Reads indexed local K/V slots and writes KV data directly into peer UVA page destinations (adaptive: fused kernel or two-stage) |
| `p2p_kv_donate_layer_twostage(...)` | Gather into a per-call scratch + one `cudaMemcpyAsync` per page (the copy-engine reference, byte-identical output) |
| `kv_gather(scratch, k_src, v_src, slot_ids, ...)` | Indexed gather of local slots into a contiguous scratch — the first stage, exposed so the plan fallback can gather once |

- Unlike the restore (a scatter, requiring UNIQUE destination slots), the
  donate is a gather: source slots may repeat and be non-monotonic.
- CUDA one-shot dispatch is ADAPTIVE — `prefer_direct_store(...)` picks the
  direct SM store or the two-stage fallback from a cost model fitted on
  H100 NVL peer-write measurements (the restore's mirror path); tunable at
  runtime via `set_donate_dispatch(mode, min_pages_for_direct)` (C++ only,
  not exposed through the C ABI). The host one-shot is always fused.
- **File**: `src/c/vkernels/comm/p2p_kv_donate.cpp` (CPU), `.cu` (CUDA)
- **Bench**: `meta/benchmarks/p2p_kv_donate_bench.cu`
- **Performance**: [performance/p2p-kv-donate/](performance/p2p-kv-donate/)

**Prepared plan (issue #36)** — `P2PKvDonatePlan` (host + CUDA) and the C
ABI `vkernels_p2p_kv_donate_plan_t` validate the slot map and upload the
page descriptors ONCE; `execute(k_src, v_src,
destination_layer_offset_bytes, stream)` then launches ONE
page-by-token-group kernel that adds the scalar offset to every peer page
base before writing. The per-layer SOURCE pair is taken per call because
KVAAS keeps a distinct K/V buffer per model layer — one prepared run list
fans out across all 40 layers with no per-layer allocation, D2H sync, or
H2D descriptor upload. `execute_via_scratch(...)` is the documented
fallback for systems where direct peer stores are unsupported or lose to
the copy engine: one kv_gather into a caller-owned scratch plus per-page
`cudaMemcpyAsync`, no per-call allocation, byte-identical output. The
three creation modes mirror the restore — host `slot_ids` (validated at
create: non-negative and `< num_slots`; repeats allowed),
`from_device_slots` (borrows the device pointer, no D2H sync, no content
validation), and `from_device_slots_int64` (converted device-side at
create into an owned int32 buffer so the caller may free the int64 buffer
immediately).

The C ABI mirrors all three (`vkernels_p2p_kv_donate_plan_create`,
`..._create_device_slots`, `..._create_device_slots_int64`) plus
`vkernels_p2p_kv_donate_plan_execute_offset(plan, k_src, v_src, offset,
stream)` and `vkernels_p2p_kv_donate_plan_execute_via_scratch(plan, k_src,
v_src, scratch, offset, stream)`.

### Fused indexed K/V layer gather — issue #2

The donation-side building block KVAAS performs today as two separate
advanced-index gathers (one for K, one for V) into the packed
`[num_pages, page_size, 2, num_kv_heads, head_dim]` destination. This
primitive fuses both gathers (and the K/V interleave) into a SINGLE launch
— the reverse of `p2p_kv_restore` (which scatters) and the gather the
donate plan's two-stage fallback exposes as `kv_gather`.

| Function | Computation |
|---|---|
| `kv_gather_layer(dst, k_src, v_src, slot_ids, slot_ids_int64, num_slots, num_pages, page_size, num_kv_heads, head_dim, elem_size, stream)` | `dst[:, :, 0] = k_src[slot_ids]` and `dst[:, :, 1] = v_src[slot_ids]` in one operation |
| `kv_gather_layer_device_slots(...)` | CUDA-only: caller-owned DEVICE `slot_ids` (no D2H sync, no content validation) |

- **Contract**: `k_src`/`v_src` are `[num_slots, num_kv_heads, head_dim]`,
  BF16 or FP16 (`elem_size == 2`); `slot_ids` is `[num_pages, page_size]`,
  int32 or int64, arbitrary non-monotonic source slots that MAY REPEAT
  (gather semantics, unlike the restore's unique-destination scatter).
  Only the range `[0, num_slots)` is enforced; `num_pages == 0` is a valid
  no-op.
- **Lifetime**: for the host-input `kv_gather_layer` the slot map is copied
  into owned storage before the function returns (the caller may free it
  immediately); `k_src`, `v_src` and `dst` must outlive `stream`. Async on
  the caller's stream with NO device-wide synchronization.
- **File**: `src/c/vkernels/comm/kv_gather.cpp` (CPU oracle), `.cu` (CUDA
  kernel), `kv_gather_c.{h,cu}` (C ABI)
- **Bench**: `meta/benchmarks/kv_gather_bench.cu` — fused vs the two-gather
  PyTorch reference, 64 through 8,192 tokens.
- **Python**: `vkernels.kv_gather_layer(k_src, v_src, slot_ids, dst, *,
  stream=None)` (also `vkernels.comm.kv_gather_layer`); non-default strides
  are rejected explicitly (a silent copy would write into a throwaway
  buffer). The pure-Python reference in `_fallback.py` is byte-exact for
  both BF16 (passed as a `uint16` view) and FP16.

### Fused indexed K/V layer scatter — issue #1

The restore-side reverse of `kv_gather` (issue #2): KVAAS scatters a
contiguous, already-gathered per-layer scratch buffer back into the paged
KV pool. The destination token slots are arbitrary and non-contiguous, so
a memcpy cannot place the data directly. The current fallback performs two
PyTorch advanced-index writes (one for K, one for V); this kernel fuses
both writes (and the K/V split) into a single launch that reads each slot
id once and writes K and V together.

| Function | Computation |
|---|---|
| `kv_scatter_layer(k_dst, v_dst, slot_ids, src, *, stream=None)` | `k_dst[slot_ids] = src[:,:,0]` and `v_dst[slot_ids] = src[:,:,1]` in one operation |
| `kv_scatter_layer_device_slots(...)` | CUDA-only: caller-owned DEVICE `slot_ids` (no D2H sync, no content validation) |

- **Contract**: `k_dst`/`v_dst` are `[num_slots, num_kv_heads, head_dim]`,
  BF16 or FP16 (`elem_size == 2`); `slot_ids` is `[num_pages, page_size]`,
  int32 or int64, with **UNIQUE** destination slots in `[0, num_slots)`
  (scatter writes disjoint destinations, unlike the gather's repeatable
  sources); `src` is packed `[num_pages, page_size, 2, num_kv_heads, head_dim]`.
  `num_pages == 0` is a valid no-op.
- **Lifetime**: for the host-input `kv_scatter_layer` the slot map is copied
  into owned storage before the function returns; `k_dst`, `v_dst`, `src`
  must outlive `stream`. Async on the caller's stream with NO device-wide sync.
- **File**: `src/c/vkernels/comm/kv_scatter.{hpp,cpp,cu}`, `kv_scatter_cuda.hpp`
  (CUDA entry points), `kv_scatter_c.{h,cu}` (C ABI)
- **Python**: `vkernels.kv_scatter_layer(k_dst, v_dst, slot_ids, src, *,
  stream=None)` (also `vkernels.comm.kv_scatter_layer`); non-default strides
  are rejected explicitly. The pure-Python reference in `_fallback.py` is
  byte-exact for both BF16 and FP16.

### Compute/communication overlap

| Class / Function | Computation |
|---|---|
| `OverlapExecutor.run(iters, compute_fn, comm_fn)` | Runs `compute_fn` on stream A and `comm_fn` on stream B in lockstep, returning a `Result(compute_count, comm_count)` |

- In-order execution within a stream, concurrency across streams.
- **File**: `src/c/vkernels/comm/overlap.cpp`
- **Python**: `vkernels.comm.OverlapExecutor()`
- **Rust**: `vkernels::comm::OverlapExecutor::new()`

### HIP/RCCL transport + OFI/CXI net plugin — issue #19

A second HIP/RCCL channel behind the existing `Channel` / all-reduce
interface, plus a HIP-aware OFI/CXI net plugin for Slingshot RDMA instead
of Socket, and graph-capturable all-reduce variants. Built on ROCm only.

| Surface | Role |
|---|---|
| `RcclChannel` | `Channel` over RCCL send/recv (host→device→host) |
| `RcclAllreducePlan` | Graph-capturable host plan: single `rcclAllReduce`, no host allocation after construction |
| `RcclAllreducePlanHip` | HIP path: one `rcclAllReduce` between begin/end graph capture |
| `resolve_transport(bytes, edges, cfg)` | Adaptive Socket↔Slingshot selection from a cost model |
| `est_rccl_socket_us` / `est_rccl_ofi_us` | Socket = `max(50, 6.0 us/MiB) + 25 us/edge`; OFI = `max(20, 3.0 us/MiB)` (RDMA, edge-free) |
| `discover_ofi_cxi` | Detects the `cxi` libfabric provider for the net plugin |
| `vkernels_rccl_*` | C ABI wrapping the host reference (always compiled) |

- **Host reference**: `src/c/vkernels/comm/rccl.{cpp,hpp}` (always compiled)
- **HIP/RCCL path**: `src/c/vkernels/comm/rccl.hip`, `rccl_hip.hpp` (`VKERNELS_HAS_RCCL`)
- **C ABI**: `src/c/vkernels/comm/rccl_c.{h,cpp}`
- **OFI/CXI net plugin**: `plugins/rccl-net-ofi/` (`librccl-net-ofi.so`, `VKERNELS_HAS_OFI`)
- **Build discovery**: `meta/cmake/RcclSupport.cmake`
- **Bench**: `meta/benchmarks/bench_rccl.cpp` (`rccl_bench`)
- **Docs**: [comm-rccl.md](comm-rccl.md)

### Pipeline-parallel boundary transfer — issue #10

A graph-capturable primitive for the hidden-state transfer at a PP
(pipeline-parallel) boundary, plus an eager-break path that mirrors vLLM's
`eager_break_during_capture`. The host reference is the correctness oracle;
the CUDA path is the device realization.

| Surface | Role |
|---|---|
| `PipelineBoundaryConfig` | Deployment facts: `same_node`, `nccl_graph_supported`, `gloo_fallback` |
| `classify_boundary(cfg)` | `same_node`→peer copy, `gloo_fallback`→host-staged, `nccl_graph_supported`→cross-node NCCL, else host-staged |
| `is_graph_capturable(t)` | A boundary is capturable on a device path (same-node peer, cross-node NCCL) |
| `eager_break_during_capture(cfg)` | 1 when the host send/recv must be excluded from the captured segment (host-staged), 0 otherwise |
| `PipelineBoundaryPlan` (host) | Directed boundary transfer over the ring `Channel`; device path enqueues a `memcpy` (no host progress on replay), eager-break path runs the `Channel` between `GraphCapture::end`/`begin` |
| `GraphCapture` | RAII begin/end capture + submit/replay, with eager (no-graph) and multi-segment support |
| `PipelineBoundaryPlan` (CUDA) | `vkernels::comm::cuda::PipelineBoundaryPlan` — one `cudaMemcpyAsync` (peer) or `ncclSend`/`ncclRecv` (cross-node) on a `cudaStream_t` |
| `vkernels_pp_*` | C ABI: classification (always compiled) + device plan (CUDA-only) |

- **Host reference**: `src/c/vkernels/comm/pipeline_boundary.{cpp,hpp}` (always compiled, 100% line-covered CI gate)
- **CUDA device path**: `src/c/vkernels/comm/pipeline_boundary.cu`, `pipeline_boundary_cuda.hpp` (`VKERNELS_HAS_CUDA`)
- **C ABI**: `src/c/vkernels/comm/pipeline_boundary_c.{h,cpp}` (always compiled) + `pipeline_boundary_c.cu` (CUDA-only)
- **Docs**: [comm-pipeline-boundary.md](comm-pipeline-boundary.md)

### Cross-node KV transfer + fabric import — issue #49

A vkernels-side mechanism that makes the existing **prepared fused KV
restore / donate kernels** (issues #27, #36 — same-node NVLink peer only)
operate **across nodes**, instead of being replaced by a separate,
synchronous host-staged bulk copy (kvaas NIXL/libfabric `transfer()`).
The prepared kernels are transport-agnostic at the pointer level:
`*_execute_offset` dereferences `peer_src_ptrs`/`peer_dst_ptrs` that must
*already be device-addressable on the local fabric*. `fabric_import`
yields that directly device-addressable handle (`CU_MEM_HANDLE_TYPE_FABRIC`
/ IMEX) so the existing kernels run UNCHANGED over cross-node memory.

| Surface | Role |
|---|---|
| `FabricImportConfig` | `same_node` \| `has_gpudirect_rdma` \| `dram_only_libfabric` |
| `classify_fabric_import(cfg)` | precedence `same_node` > `dram_only` > `gpudirect` > bounce: `kSameNodePeer` / `kFabricMapped` / `kHostBounce` |
| `is_import_graph_capturable(t)` | `true` for the two device transports, `false` for `kHostBounce` |
| `eager_break_fabric_import(cfg)` | 1 when a host-bounce must be excluded from the captured segment (ties into #10) |
| `FabricImport` | performs the import ONCE; `device_ptr()` (nullptr on `kHostBounce`) + `write_back(remote)` |
| `CrossNodeKvRestorePlan` / `CrossNodeKvDonatePlan` | reuse `P2PKvRestorePlan`/`P2PKvDonatePlan` over the imported pointer; graph-capturable or eager-break host bounce (`kv_gather`/`kv_scatter` over a `ByteChannel`) |
| `cross_node_kv_throughput(transport, total_bytes, gh200_dram_only)` | per-hop cost model vs the same-node roofline (88.5 GB/s) and bulk-copy fallback (1.4) |

- On GH200 the libfabric plugin only carries DRAM↔DRAM (hwloc-PCIe
  discovery cannot see the C2C-attached H100), so cross-node VRAM is
  reported as `kHostBounce` regardless of the RDMA bit — the caveat called
  out explicitly. Measured on JSC InfiniBand HDR, the cross-node hop caps
  at **one HDR-200 port (~24 GB/s, ~75× slower than the 3.3 TB/s same-node
  HBM ceiling)** regardless of operation type.
- **Host reference**: `src/c/vkernels/comm/fabric_import.{hpp,cpp}`, `cross_node_kv.{hpp,cpp}` (always compiled, 100% line-covered CI gate)
- **CUDA device path**: `fabric_import.cu`, `fabric_import_cuda.hpp`, `cross_node_kv.cu`, `cross_node_kv_cuda.hpp` (`VKERNELS_HAS_CUDA`)
- **C ABI**: `fabric_import_c.{h,cpp,cu}`, `cross_node_kv_c.{h,cu}`
- **Bench**: `meta/benchmarks/bench_cross_node_kv.cpp` (host cost model), `bench_cross_node_nccl.cu` (JSC HDR measurement, `VKERNELS_HAS_NCCL`)
- **Docs**: [comm-cross-node-kv.md](comm-cross-node-kv.md)

### Cross-node KV all-gather — issue #49 (draft)

The cross-node hop caps at one HDR-200 port because a 2-rank
point-to-point donate/restore cannot stripe across the ≥3 edges a ring
provides. An **all-gather** plan is the primitive that *does* use the
N−1 edges a ≥3-node ring provides (climbing toward the 4-port aggregate a
single donate cannot reach), for the case where every node needs the full
KV. A prepared plan plus access-pattern routing (each rank publishes its
shard; every rank gathers the full set over the fabric).

- **File**: `src/c/vkernels/comm/cross_node_kv_allgather.cu`,
  `cross_node_kv_allgather_cuda.hpp` (communicator + prepared plan),
  `cross_node_kv_allgather_c.{h,cu}` (stable serving-runtime C ABI)
- **Docs**: [comm-cross-node-kv-allgather-draft.md](comm-cross-node-kv-allgather-draft.md)
  (draft only; the multi-port result is unmeasured)

---

## Core infrastructure

| Component | Description |
|---|---|
| `Device(index)` | GPU device selection, synchronization, peer-access queries |
| `Stream()` | In-order task queue; one worker thread per stream |
| `Span<T>` | Non-owning view of contiguous memory (C++ only) |

---

## File layout

```
src/c/vkernels/
├── kernels/
│   ├── elementwise.{cpp,cu,hpp}  # add, scale, relu
│   ├── reduce.{cpp,cu,hpp}       # sum, max
│   ├── gemm.{cpp,cu,hpp}         # tiled SGEMM
│   ├── gemm_bf16.{cpp,hip,hpp}   # bf16 K16-MFMA GEMM (gfx942, #29)
│   ├── mla.{cpp,hip,hpp}         # Multi-head Latent Attention, absorbed form (#21)
│   ├── kda.{cpp,hip,hpp}         # Kimi Delta Attention — gated delta-rule (#21, #45)
│   ├── dsa.{cpp,hip,hpp}         # DeepseekSparseAttn sparse-MLA + paged-MQA top-k logits (#51)
│   ├── mhc.{cpp,hip,hpp}         # MHC pre-norm GEMM+sqrsum / post combine (#51)
│   ├── moe.{cpp,hip,hpp}         # gfx942 primitives (#12–#15)
│   ├── moe_device.hip            # shared bf16↔f32 helpers used by the MoE kernels
│   ├── moe_aux.{cpp,hip,hpp}     # MXFP4 MoE orchestration: quant, sort, scatter-reduce (#22)
│   └── moe_fused.{cpp,hip,hpp}   # fused MXFP4 MoE grouped GEMM
├── dist/
│   └── dist_moe.{cpp,hpp}        # distributed MoE: TP/EP/PP sharding (#18)
├── comm/
│   ├── allreduce.{cpp,cu,hpp}    # ring all-reduce
│   ├── overlap.{cpp,hpp}         # compute/comm overlap executor
│   ├── channel.{cpp,hpp}         # blocking queue & mock channel
│   ├── topology.hpp              # ring topology helpers
│   ├── p2p_gather.{cpp,cu,hpp}   # single-launch peer gather
│   ├── p2p_gather_cuda.hpp       #   CUDA entry points
│   ├── p2p_gather_c.{h,cu}       #   C ABI for the peer gather
│   ├── p2p_kv_restore.{cpp,cu,hpp} # prepared fused KV restore (#27)
│   ├── p2p_kv_restore_cuda.hpp   #   CUDA plan declarations
│   ├── p2p_kv_restore_c.{h,cu}   #   C ABI
│   ├── p2p_kv_donate.{cpp,cu,hpp} # prepared fused KV donate (#36)
│   ├── p2p_kv_donate_cuda.hpp    #   CUDA plan declarations
│   ├── p2p_kv_donate_c.{h,cu}    #   C ABI
│   ├── kv_gather.{hpp,cpp,cu}    # fused indexed K/V layer gather (#2)
│   ├── kv_gather_cuda.hpp        #   CUDA entry points
│   ├── kv_gather_c.{h,cu}        #   C ABI for the fused K/V gather (#2)
│   ├── kv_scatter.{hpp,cpp,cu}   # fused indexed K/V layer scatter (#1)
│   ├── kv_scatter_cuda.hpp       #   CUDA entry points
│   ├── kv_scatter_c.{h,cu}       #   C ABI for the fused K/V scatter (#1)
│   ├── cross_node_kv.{cpp,cu,hpp} # cross-node KV restore/donate plans (#49)
│   ├── cross_node_kv_cuda.hpp    #   CUDA plan declarations
│   ├── cross_node_kv_c.{h,cu}    #   C ABI for the cross-node KV transfer (#49)
│   ├── cross_node_kv_allgather.cu # equal-shard NCCL KV all-gather (#49)
│   ├── cross_node_kv_allgather_cuda.hpp # communicator + prepared plan
│   ├── cross_node_kv_allgather_c.{h,cu} # stable serving-runtime C ABI
│   ├── fabric_import.{cpp,cu,hpp} # fabric / VMM import host reference (#49)
│   ├── fabric_import_cuda.hpp    #   CUDA fabric-import declarations
│   ├── fabric_import_c.{h,cpp,cu} # C ABI for the fabric import (#49)
│   ├── slot_map.hpp              # shared slot-map validators (#1, #2, #27, #36)
│   ├── pipeline_boundary.{cpp,cu,hpp} # graph-capturable PP boundary (#10)
│   ├── pipeline_boundary_cuda.hpp #   CUDA plan declarations
│   ├── pipeline_boundary_c.{h,cpp,cu} # C ABI for the PP boundary (#10)
│   ├── rccl.{cpp,hpp}            # HIP/RCCL transport host reference (#19)
│   ├── rccl.hip                  # HIP/RCCL all-reduce (VKERNELS_HAS_RCCL)
│   ├── rccl_hip.hpp              # RcclChannel / plan declarations
│   └── rccl_c.{h,cpp}            # C ABI for the RCCL transport
├── capi/
│   ├── capi.{cpp,hpp}            # C ABI (host path), exceptions → status codes
│   ├── hip_capi.{cpp,hpp}        # C ABI over the gfx942 HIP compute kernels (#44)
│   └── serving_c.{h,cpp}         # serving-runtime CUDA ABI
├── core/
│   ├── device.{cpp,hpp}          # Device abstraction
│   ├── stream.{cpp,cu,hpp}       # Stream (async task queue)
│   └── allocator.hpp             # CUDA memory pool tuning
└── util/
    ├── annotations.hpp           # [[nodiscard]] / VK_EXPORT helpers
    ├── config.hpp                # VKERNELS_HAS_CUDA / HIP / RCCL / OFI macros
    ├── error.hpp                 # VK_EXPECTS / VK_ENSURES status codes
    ├── logging.hpp               # compile-time logging
    ├── span.hpp                  # non-owning view (C++ only)
    └── version.hpp               # version constants
```

## Language bindings

| Language | Module | Docs |
|---|---|---|
| Python | `vkernels.kernels`, `vkernels.comm`, `vkernels.core`, `vkernels.dist` | [python-bindings.md](python-bindings.md) |
| Rust | `vkernels::kernels`, `vkernels::comm`, `vkernels::core` | [rust-bindings.md](rust-bindings.md) |
| C | `vkernels_c_*` / `vk_hip_*` (C ABI via `capi.hpp`) | — |
