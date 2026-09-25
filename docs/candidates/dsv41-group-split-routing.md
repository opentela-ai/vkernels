# Candidate: DSV41 group-split expert routing (fp4_tc / fp4_tcw) for `moe_fused`

> Provenance: reference repo (shi3z/deepseekv4.1-A100-custom) carries **no license file** as of
> this writing — analysis only; don't copy code verbatim without resolving licensing.


**Status**: analysis + port plan (no code yet)
**Reference**: `/tmp/dsv41/dsv41/cuda/fp4_tc.cu` (small groups, ≤ 8/16 tokens,
weights-as-B), `/tmp/dsv41/dsv41/cuda/fp4_tcw.cu` (operand-flipped, ≤ 64 tokens,
weights-as-A), launcher `cukern.fp4_gemm_tc` + callers `decode.py:322–345`
**Ours**: `src/c/vkernels/kernels/moe_fused.{hpp,cpp,hip}` (decode = block_size 16,
prefill = block_size 64), CPU oracle in `moe_fused.cpp`

---

## 1. What the reference does

Both kernels compute the same grouped expert GEMM over pairs `(tok[p], expert g)`
with MXFP4 weights (E2M1 + ue8m0), exact register decode to bf16, fp32
accumulate. They differ only in **who is the mma A operand** and **how many
tokens share one weight read**:

| | `fp4_tc.cu` (`tc8`/`tc16`) | `fp4_tcw.cu` (operand-flipped) |
|---|---|---|
| mma A operand | activations x (16 rows) | **weights** W (16 n-rows) |
| mma B operand | weights (8 cols) | activations x (8·MT token cols) |
| weight re-read | once **per ≤ 16-token group** | once **per ≤ 64-token group** |
| group cap (`min_tok..max_tok`) | 1..8 (`tc8`) or 9..16 (`tc16`) | 9..64 (paired with `tc8` by the launcher) |
| x staging | registers, direct gmem | LDS, `cp.async` 3-deep ring, XOR-swizzled |
| tile | 16×8 per `mma.m16n8k16` | 16×(8·MT), MT ∈ {2,4,8} → 16/32/64 tokens |
| EP | `shard_start/shard_n` skip + `zero_out` | same |

The launcher (`cukern.fp4_gemm_tc`) always runs the small kernel over groups
≤ 8 (or ≤ 16 when the flipped layout is disabled) and, when
`max_tokens > 8`, additionally launches `fp4_gemm_tcw` for groups in
`9..max_tokens`, splitting groups > 64 at `FP4_W_MAX = 64`. This is the
"adaptive small/large group routing" under review.

## 2. Would it help our decode shapes? — **No, quantified**

**Metric**: expert re-read amplification
`amp(C) = Σ_e ceil(g_e / C) / |{e : g_e > 0}|` — how many times the average
hit expert's weights are streamed from HBM per layer, for a group cap C
(our decode regime = cap 16 via BLOCK_M=16; flipped = cap 64).

Monte-Carlo (uniform routing, 4000 trials × 5 EP shards, shard = ⌈384/5⌉ = 77,
top_k = 8, E = 384; λ = M·top_k/E per expert — **EP sharding does not change
per-expert λ**, it only reduces the number of hit experts per GPU):

| M (batch) | pairs total | hit experts/GPU | amp, cap 16 (ours / `tc`) | amp, cap 64 (`tcw`) | max group seen |
|---:|---:|---:|---:|---:|---:|
| 1   | 8    | 1.9  | 1.0000 | 1.0000 | 3  |
| 8   | 64   | 11.8 | 1.0000 | 1.0000 | 5  |
| 32  | 256  | 37.4 | 1.0000 | 1.0000 | 8  |
| 64  | 512  | 56.6 | 1.0000 | 1.0000 | 10 |
| 128 | 1024 | 71.5 | 1.0000 | 1.0000 | 15 |
| 256 | 2048 | 76.4 | 1.0000 | 1.0000 | 20 |
| 512 | 4096 | 76.8 | **1.0445** | 1.0000 | 31 |

**Interpretation.** At decode batch sizes the pair population (M·8 ≤ 512) is
smaller than the expert count (384), so per-expert load λ ≤ 1.33 and
P(group > 16) ≈ O(λ¹⁷/17!) — groups essentially never exceed the 16-row
block, and **both layouts read each hit expert's weights exactly once**:
amplification is 1.000 either way. Concretely (DeepSeek-V3-like dims,
hidden 7168 / ispp 2048 → 22.3 MiB per expert per layer), at M = 64 the
per-GPU weight traffic is ~1.24 GiB/layer in *both* schemes; the flipped
layout buys zero bytes. The decode step is weight-bandwidth-bound, so this
is the metric that matters; the extra compute the 16-row padding wastes is hidden under
the weight stream (at M = 64 tokens: 512 pairs over ~282 hit experts → ~1.8 pairs per
expert → **~11 % real rows per 16-row tile**, measured 11.3 % by Monte-Carlo).

**Crossover.** cap-64 starts paying off only when groups overflow 16 rows:
aggregate amp(cap16) ≈ 1.000 at M = 300, 1.005 at M = 400, 1.037 at M = 500,
1.13 at M = 600, 1.50 at M = 800, 1.84 at M = 1000 (top_k = 8; exact Poisson,
λ = M·top_k/E per expert, verified by simulation — an earlier draft of this
doc carried an inflated series). I.e. the
adaptive split is a **large-batch / prefill / chunked-prefill** feature, not a
decode feature. Our prefill regime (block_size = 64, 64×64 tiles) *already*
has cap-64 semantics, so the reference's `tcw` value-add over what vkernels
ships today is confined to:

1. **Mixed bursts**: decode-shaped steps where routing is skewed/hot and a few
   experts receive > 16 tokens (MTP draft batches, bursting sessions). Worst
   case all P = M·8/5 pairs on one expert: cap-16 streams ⌈P/16⌉ weight copies,
   cap-64 ⌈P/64⌉ — a hard **4×** bound (e.g. M = 64 fully skewed: 0.15 →
   0.04 GiB). Cheap insurance; today's block_size-16 path has no such bound
   until a switch to the prefill regime.
2. **The mid regime (17–64 tokens/group) on the decode kernel path** —
   currently impossible without going to 256-thread prefill tiles.

**Verdict**: do **not** expect measurable decode gains on the uniform-routing
shapes (M ≤ 128, E = 384, top_k 8 — or our bench shape E = 256, top_k 6, where
λ is even smaller). Port the flipped layout for the skew bound and mid-regime
coverage, gated by a dispatch on observed max group size.

## 3. How the operand-flip schedule maps to CDNA MFMA (gfx90a/gfx942)

The repo already ships an **empirically verified** fragment layout for
`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` (`gemm_bf16.hip`, confirmed on
gfx90a via `moe_fused.hip`):

- scope: one **wavefront (64 lanes)** per MFMA (vs 32-lane warp on CUDA);
  a 128-thread block = 2 wavefronts ≈ the reference's 4×32-thread block;
- A: `m = lane%16`, `k0 = (lane/16)*4`, `a[i] = A[m][k0+i]` (4 bf16/lane);
- B: `n = lane%16`, `k0 = (lane/16)*4`, `b[i] = B[k0+i][n]`;
- C/D: `col = lane%16`, `row = (lane/16)*4 + i`, `c[i] = C[row][col]` (v4f).

Mapping the `tcw` schedule (W = A, x = B) onto these fragments:

| Reference (CUDA) | CDNA port |
|---|---|
| `mma.m16n8k16` (16×8 out, K=16) | `mfma_f32_16x16x16bf16` (16×16 out, K=16); token-tile granularity doubles: caps 16/32/48/64 = MT ∈ {1,2,3,4} 16-token tiles |
| lane (g=lane/4, t=lane%4) owns weight row n0+g, 32 k | lane owns weight row `n0 + lane%16`, the 4 consecutive k at `(lane/16)*4` **per 16-k step** → 2 packed bytes + one scale byte per lane; the 4-nibble decode is a trimmed `e2m1x8_to_bf16`, same `2^(s-127+126)` bf16x2 scale fold |
| x in the "8-k permuted" layout (`cukern.permute_x`: 0,4,2,6,1,5,3,7) | **identity** — the CDNA B fragment wants 4 k-*consecutive* bf16 per lane, i.e. x rows in natural order. The permute pass (and the DSV41 layout coupling) disappears |
| x-tile in smem via `cp.async`, 3 stages, `xc0 ^ (xr0&7)` swizzle | no `cp.async` on CDNA3: cooperative `buffer_load` → LDS double-buffer with `vmcnt` waits (the `gemm_bf16.hip` pattern); same XOR swizzle on `ds_write` for bank conflicts; 48 KiB LDS budget fits gfx942's 64 KB/workgroup |
| weights prefetched 2 iterations ahead in registers | unchanged (register ring, `__ldg` → non-temporal `buffer_load`) |
| C store: token cols `8·mt + 2t` | C fragment: token col `lane%16`, rows `(lane/16)*4 + i` — same routed-scatter epilogue with `M` bounds checks |
| group dispatch `M ≤ 16 / ≤ 32 / else` (MT 2/4/8) | `M ≤ 16 / ≤ 32 / ≤ 48 / else` (MT 1/2/3/4), or the 32×32×16 bf16 MFMA (one instruction = 32 n-rows × 32 tokens, 16 VGPR accumulators/lane) as an autotuned cap-64 variant |

Numerics: the E2M1 → bf16 decode is exact and the scale fold is exact, so
results are bit-identical to the reference **up to K-order inside the dot**;
`mfma 16x16x16` may sum the 16 products in a different fixed order than
`mma.m16n8k16`, so expect last-bit divergence vs the CUDA reference and the
usual tolerance vs the `moe_fused.cpp` oracle (same class of difference the
existing MFMA kernels already carry).

Forward-looking: gfx950 (CDNA4) has native block-scaled E2M1 MFMA, which would
collapse the whole decode loop into one scaled-MFMA per tile; the group-split
routing and epilogues port unchanged. Not in scope here (target is gfx942).

## 4. Port plan for vkernels

**Phase 0 — dispatch instrumentation (no kernel work).**
Extend `moe_align_block_size_hip` / the host align to also report (device-side)
the max real rows per expert group. Add a kill-switched
(`VK_MOE_SPLIT_ROUTING`, default off) dispatch in the `fused_moe_mxfp4` HIP
launcher: max group ≤ 16 → existing decode path (unchanged); 17–64 → new
flipped path. Log the chosen bucket per call for the A/B record.

**Phase 1 — flipped kernel (new `moe_fused_hip` decode-adjacent kernel).**
Port `fp4_gemm_tcw_body` per §3: weights-as-A, 16-token B tiles (MT 1..4),
LDS x-staging (double-buffered, swizzled), routed epilogue with `zero_out`
semantics mapped to the existing `expert_map = -1` skip. Keep the CUDA
reference as the schedule spec; reuse `moe_device.hip` fp4 helpers.
Accept: hidden, ispp multiples of 64 (decode contract unchanged).

**Phase 2 — validation.**
GPU-vs-oracle (`meta/benchmarks/test_moe_fused_correct.hip` pattern) across
group-size histograms {1..64}, EP skip/zero-out cases, bias + SiTU/SwiGLU
epilogues; bit-comparison against the existing decode path on groups ≤ 16
(must match to fp32 accumulate tolerance); new regression
`test_group_split_routing.hip` covering the skew worst case (all pairs → one
expert) and the cap-64 group splitting.

**Phase 3 — benchmark & adopt.**
A/B on gfx942: uniform decode sweep M ∈ {1..128} (expect parity — this
*validates* §2), skewed/hot-expert decode, MTP burst, and the M ≈ 500–1000
mixed regime vs forcing the prefill regime. Adopt the dispatch only if the
skewed/mid buckets show the predicted ≥ 4× worst-case traffic cut without
regressing uniform decode.

## 5. Risks / open questions

- Exact A/B fragment bit-layouts for the *scaled* path and for `32x32x16` need
  on-device confirmation (the verified table above covers `16x16x16` only).
- LDS-swizzled B reads with 4-k column access may need the `gemm_bf16`
  conflict-avoidance treatment; measure before trusting.
- Two-launch dispatch (small + flipped) mirrors the reference; if launch
  overhead dominates at M = 1, keep the single-kernel decode path for tiny
  batches and dispatch the flipped kernel only when a > 16 group exists
  (device-side flag already produced in Phase 0).
- The reference's `FP4_W_MAX = 64` group split changes pair ordering vs our
  sorted_ids layout; the combine epilogue is order-independent, but the
  bit-exactness tests against CPU align must pin the split rule.
