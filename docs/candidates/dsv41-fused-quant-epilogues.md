# Candidate 6 — fused SwiGLU/RMSNorm + FP8-quant epilogues (dsv41 port)

> Provenance: reference repo (shi3z/deepseekv4.1-A100-custom) carries **no license file** as of
> this writing — analysis only; don't copy code verbatim without resolving licensing.


Reference: `/tmp/dsv41/dsv41/fused.py` (`swiglu_quant`, `_round_e4m3`,
`_ceil_log2`, `_pow2`, `_fq8`-equivalent per-32 fake quant) and
`/tmp/dsv41/dsv41/fused2.py` (`norm_quant`, `hc_pre_norm_quant2`,
`hc_pre_norm_quant`, two-pass multi-CTA variants).
Current ops: `src/python/vkernels/torch_ops/elementwise.py`
(`swiglu_limit` / `silu_mul`, no quant epilogue),
`src/python/vkernels/torch_ops/aiter_ops.py`
(`fp8_blockscale_experts` — STAGE SPLIT with the floe clamp-swiglu applied
in torch BETWEEN the two CK GEMM stages, because aiter's in-kernel Swiglu
is the gpt-oss formula and its CK codegen has no activation-free stage1),
`src/python/vkernels/torch_ops/per_group_quant_fp8` (same file) for the
re-quant. Status: **analysis + plan, no code yet**. Follows the
NOTES-155 format; sibling precedents NOTES-156/157 (moe-aux fused
sorted-quant) and docs/candidates/dsv41-fp4-gemv-pairs.md.

## Baseline — what the stage-split pays today (traffic model)

Shapes (K3-class decode, GLM MoE): `hidden H = 7168`, `ispp I = 2048`
(typical; the moe_aux K3 fixture uses ispp = 3072 — scale I-terms by 1.5),
`top_k = 16`, `M = 112` → `EM = 2048` padded sorted rows (1792 real).
bf16 = 2 B, fp8 = 1 B, per-128 fp32 scale = `I/128·4 = I/32` B.
All counts below are **per EM row** (the act/gate|up intermediates are
`[EM, ·]`, token-major scattered), then × EM per MoE layer.

Stage-split chain per layer (aiter_ops flow, swiglu in torch):

| step | kernel(s) | bytes / EM row | I=2048 |
|---|---|---:|---:|
| stage-1 out (gate\|up) write | ck_moe_stage1 | 2I·2 = 4I | 8,192 B |
| clamp(g) + clamp(u) + silu + mul | torch eager (4 launches) | 18I | 36,864 B |
| requant read+write+scale | per_group_quant_hip | 3I + I/32 | 6,208 B |
| **total (eager swiglu)** | 6 launches | **25.0I** | **51.3 KB** |
| total with `elementwise.swiglu_limit` (1 launch) | 3 launches | 13.0I | 26.7 KB |

Per layer at EM = 2048: **~105 MB** (torch-eager swiglu) or **~54.7 MB**
(Triton swiglu) of intermediate traffic before stage-2 even starts.

Fused act+quant epilogue (ONE kernel: read stage-1's bf16 gate|up, clamp,
silu·mul, per-group amax → pow2/e4m3 scale, quantize, write fp8 + scales):

| variant | bytes / EM row | I=2048 | per layer (EM=2048) | vs split |
|---|---:|---:|---:|---|
| fused standalone epilogue | 4I + I + I/32 = 5.03I | 10.3 KB | 21.1 MB | **−61%** (−80% vs eager) |
| fused into stage-1 GEMM epilogue | I + I/32 = 1.03I | 2.1 KB | 4.3 MB | **−92%** (−96%) |

Also in scope (same rounding-preserving port):

* **`norm_quant`** (rmsnorm + per-group fp8 quant): split = norm write 2H +
  quant read 2H + write H = 5H; fused = read 2H + write H = 3H → saves
  `2H = 14.3 KB/token/layer` (−40%), 1.6 MB/layer at T = 112.
* **`hc_pre_norm_quant2`** (sinkhorn + hc_pre + rmsnorm + quant,
  two multi-CTA passes): replaces a 5–15-launch torch chain; port the
  fused2 two-pass multi-CTA variant (~3 µs each), NOT the single-CTA
  fused.py kernel (~13 µs/row — wrong shape for B > 1 decode).
* **W4A4 (MXFP4) path**: the `[EM, ispp]` bf16 `act_scratch` round-trip
  (write 2I + stage-1 read 2I = 4I/row ≈ 33.6 MB/layer at EM = 2048)
  becomes packed-E2M1 + ue8m0 (write 0.53I + read 0.53I ≈ 1.06I) → **−74%**
  on the act round-trip; same shape of win as NOTES-157's `A_sorted`
  elimination (−24% measured on GB10 for the sort→quant chain).

Bandwidth realization caveat (be honest in the A/B): at K3 decode the
intermediates (≤ 105 MB) fit MI300A's ~256 MB L2, so warm iterations run at
the **L2 roof** (3,219 GB/s measured, moe_aux.md), not HBM (2,868–2,938).
Nominal −61% traffic ⇒ roughly −7 µs/layer at the L2 roof for the
Triton-swiglu baseline; the full HBM-roof saving (~11–12 µs/layer, ~35 µs
at eager) applies at prefill / large-batch where the intermediates spill.
The launch-count win (6 → 1, or 3 → 1) is independent of residency.

## Change (plan)

1. **torch_ops activation-quant lane** (new module, elementwise.py
   conventions: lazy kernel set, per-call `OpNotEligible`, eager
   `*_reference` oracles, `enable_fp_fusion=False`):
   port `_round_e4m3`, `_ceil_log2`, `_pow2` verbatim (exact IEEE
   bit-pattern ceil-log2; `libdevice.div_rn` everywhere — Triton's `/` is
   approximate and would break scale bit-exactness) and
   `swiglu_quant(gate, up | interleaved gu, weights | None, limit,
   permute=False)` → fp8 (fnuz on gfx942, via the existing dtype
   conversion helper) `[rows, I]` + `[rows, I/128]` fp32 scales, plus a
   bf16 fake-quant mode matching the reference chain. Env kill-switches on
   every call site (`VK_FUSED_ACT_QUANT=0` etc., VK_DSA_DECODE_SPLIT /
   VK_MOE_AUX_FUSED_QUANT naming precedent).
2. **Port `norm_quant` + `hc_pre_norm_quant2`** with the same conventions
   (persistent-buffer `out=` dict form included — the static-graph runtime
   needs it, per fused2).
3. **Wire into `aiter_ops.fp8_blockscale_experts`**: replace the
   `swiglu(...)` + `per_group_quant_fp8(act)` pair (steps 4–5 of the flow)
   with one epilogue launch that emits fnuz fp8 + the per_1x128 scale
   layout directly — this also removes the fnuz dtype-conversion pass the
   current two-step path implies. `swiglu_limit` (elementwise) stays the
   fallback path, untouched.
4. **W4A4 follow-on (separate lane)**: emit packed-E2M1 + ue8m0 from the
   same epilogue (reuse the NOTES-156/157 `quant_row` conventions
   including the padding-row `0xFF` encoding) so `act_scratch` goes
   quantized straight into `fused_moe_mxfp4`.

### Triton vs HIP — decision: **Triton, in torch_ops**

* These epilogues are memory-bound elementwise + per-group row reductions —
  no MFMA/tensor-core content; the win is *traffic elimination*, not
  bandwidth engineering, so HIP buys nothing measurable for the standalone
  epilogue. The proven moe_aux HIP kernels are themselves partially
  compute-bound at these footprints (quant 3.6% L2, moe_aux.md).
* Portability: the dsv41 sources are Triton and were validated on CUDA
  (GB10 shim path); MI300A container has triton 3.4.0. HIP would fork the
  kernel per backend for zero expected gain.
* The only place HIP pays is a stage-1 GEMM **epilogue fusion** (the −92%
  row), and that is blocked on the CK path anyway: aiter's codegen enum is
  `ActivationType { Silu, Gelu }` — no identity (aiter_ops.py, BLOCKED
  2026-09-20 note). A Triton grouped GEMM could take the epilogue later;
  do not build HIP glue for it now.
* Revisit HIP only if the A/B shows the Triton epilogue missing the L2
  roof (kill-switch covers rollback in the meantime).

## Numerics — does one-pass fused quant change results?

**No, by construction, with two pinned caveats.**

1. **fp8 e4m3 + pow2 scale (W8A8)**: the per-group scale
   `s = pow2(ceil_log2(amax/448))` (amax floored at 1e-4) is a
   deterministic function of the group's bf16-rounded values. The fused
   kernel reproduces every reference rounding point: `h` is rounded
   `.to(bf16).to(fp32)` before the amax (fused.py does this explicitly);
   ceil-log2 is the exact bit-pattern form; divisions are `div_rn`;
   rounding is `rint` (ties-to-even); clamps before quant. Result is
   **bit-identical** to the two-pass materialize-then-quantize chain
   *provided the group partitioning matches* (per-32 fake-quant reference
   vs per-128 CK format — the port must expose both group sizes; the
   scale *layout* changes with the CK contract, the values do not).
   **Caveat A (must gate)**: fused.py's `_swiglu_quant_kernel` computes
   `gate·sigmoid(gate)·up` fully in fp32 with a **single** bf16 round
   before quant, while the serving-path `elementwise.swiglu_limit` rounds
   `silu(g)` to bf16 *before* the up-multiply (and rounds the product) —
   the HF-eager contract. The two differ by ≤1 ulp bf16 on some elements,
   which can flip one quant code. The port must pick the contract of the
   reference it replaces (GLM MoE floe reference = fused.py single-round)
   and A/B bit-check against the serving path before flipping any knob.
2. **ue8m0 scales computed in sorted-row order (W4A4)**: scale =
   f(group values); sorted row `r` maps to token `sorted_ids[r]//top_k`, so
   computing scales on sorted rows is **mathematically identical** to
   token-order scales + `sort_scales` — for real rows. The only divergence
   is **padding rows** (`0xFF` + zero nibbles for sort→quant vs literal
   `0` bytes for `quant → sort_scales`) — exactly the NOTES-156 incident;
   the fused op follows the sort→quant convention it replaces and tests
   pin both encodings. Group boundaries never straddle rows
   (`ispp % 64 == 0` constraint), so no cross-row group hazard exists.

## Jobs / validation plan

1. Unit parity (GB10 first, then gfx942 srun): fused epilogue **bit-exact**
   vs the eager two-pass chain (torch swiglu + per-group quant) on a shape
   matrix (I ∈ {512, 2048, 3072}, group ∈ {32, 128}, limit ∈ {10, +inf},
   with/without routing weights, permute on/off, NaN/±inf/zero/limit-edge
   groups); norm_quant bit-exact vs `rms_norm_reference` + quant.
2. A/B perf (same-job baseline, NOTES-155 discipline): K3 decode
   M ∈ {8, 32, 112}, ispp ∈ {2048, 3072}, E = 64, top_k = 16; prefill
   M = 1024 row; gate on/off in the same binary; record sclk/mclk.
   Target: epilogue chain time ≤ 0.5× of the split pair at the K3 shape,
   no regression at prefill; report GB/s against the L2 and HBM copy roofs.
3. End-to-end: parity of the full `fp8_blockscale_experts` output
   (rel gate as in aiter_ops, plus max-ulp histogram on the act) before any
   knob flips; serving-path `swiglu_limit` consumers unchanged.

## Doc deliverables

- docs/kernels/elementwise.md (or a new activation-quant section):
  contract, rounding points, scale layouts, kill-switches.
- docs/performance/moe-fused/gfx942.md: new evidence rows + journal entry.
- Raw A/B log under docs/performance/moe-fused/.

## Risks / open questions

- Rounding-point contract (Caveat A) is the one real numerics fork —
  decide against the serving reference, not against fused.py blindly.
- Per-128 CK format vs per-32 reference quant: confirm the CK kernels'
  `per_1x128` group size is the quantization group (it is, per
  aiter_ops docs) so the epilogue emits the right grouping directly.
- L2 residency flatters decode A/Bs — always report the traffic model
  alongside latency; prefill row is the honest HBM-roof test.
- aiter `fmoe_fp8_blockscale_g1u1` remains unusable (gpt-oss Swiglu
  formula, no clamp) — this epilogue is the correct way to keep the
  stage-split *and* kill its cost; the CK-codegen GEMM fusion stays
  blocked upstream.
