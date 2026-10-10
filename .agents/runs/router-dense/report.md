# LANE 28 — node-cut D: the router GEMV fold (`fused_router_dense`)

**Repo**: `/local/home/xiayao/Documents/code/vkernels` (main @ 6f6de90). **Scope**: local
code + GB10 only, no rig. All lane files stay uncommitted (`src/python/vkernels/torch_ops/glm_router_dense.py`,
`tests/python/test_glm_router_dense.py`, `meta/benchmarks/bench_router_dense.py`); other
lanes' untracked/modified files untouched.

## 1. The fold

Per-MoE-layer decode chain (node-cuts.md step D):

    x.float() cast      [T, 4096] bf16 -> fp32 materialization   (1 kernel)
    F.linear(x.f, w)    cuBLAS fp32 GEMV over [288, 4096]        (1 kernel + splitK)
    glm_router._kernel  sigmoid/bias/top-8/norm/scale epilogue   (1 kernel)

`fused_router_dense` replaces this with **TWO launches**:

1. a Triton GEMV (`_router_dense_gemv`) that reads bf16 `x` directly, widens
   in-kernel (exact: bf16→fp32 is a mantissa bit-append), fp32-accumulates over the
   fp32 `[E, H]` router weight, and writes the fp32 `[T, E]` logits;
2. the incumbent router kernel — **imported from `glm_router.py` and launched
   verbatim** (same wrapper launch params: `BLOCK_E=next_pow2(E)`, `num_warps=4`,
   `enable_fp_fusion=False`). The routing core is deliberately NOT duplicated: a
   numerics or tie-break fix in the incumbent propagates to the fold, and the
   epilogue is bit-identical to the incumbent's for identical logits by construction.

The cast materialization (a `[T, 4096]` fp32 tensor per layer) and one launch per
layer disappear. Both entry points exist: `fused_router_dense` (int32 `[T, K]`, the
`fused_router` ABI) and `fused_router_dense_shared` (int64 `[T, K+1]` with the shared
slot appended in the incumbent `_router_shared` kernel — lane 12's ABI, also launched
verbatim).

## 2. GB10 measurement — the finding that shaped the design

First-cut design (one program per token row, whole-`[E, H]` weight in one kernel with
the routing math inline) was a **regression at every T** (T=256: 662 us vs the chain's
104 us): the 512×64 fp32 weight tile spilled registers, and one program per row is a
latency-bound single SM at T=1. Device facts that constrain everything here:

* **GB10: 48 SMs, 25 MB L2, ~273 GB/s DRAM, ~15 TF fp32.** It is a small desktop
  superchip, not a datacenter part.
* cuBLAS's fp32 GEMV at T=1024 runs at ~11 TF = **75% of the fp32 peak —
  compute-bound** (register-tiled, each weight element FMA'd across many rows).
* A per-row Triton fp32-FMA GEMV cannot match that: each weight element is used once
  per row-program, so there is no reuse to amortize FMA issue slots.

**Variant matrix measured** (production shape [288, 4096], wall us back-to-back):

| variant | T=1 | T=8 | T=64 | T=256 | T=1024 |
|---|---|---|---|---|---|
| eager chain (cast+gemv+router) | 22.8 | 29.1 | 46.9 | 104 | 325 |
| v1 whole-E one program/row | 76 | ~ | ~ | 631 | 2539 |
| v2 per-(row, 16-expert slice), chunk tree | 10.3 | 25.3 | 159 | 626 | 2511 |
| v2b same, deferred [E-slice, H] tree | **8.3** | 25.6 | 134 | 425 | 1489 |
| v2c es-major grid order | 8.3 | 27.3 | 167 | 632 | 2486 |
| v2d + `tl.range(num_stages)` | 8.0 | 28.0 | 192 | 734 | 2842 |
| v3 `tl.dot(input_precision="ieee")`, 16-row tiles | 67 | 66 | 129 | 371 | 1467 |

v2 (shipped) wins the decode regime and loses prefill to cuBLAS by 3–7x. The ieee-dot
path was rejected: no large-T win *and* a wider logit band (7.6e-4 — its internal
K-reduction reassociates differently). The deferred-tree variant v2b is marginally
better at T=1 (8.3 vs 10.3) but ships as v2 because the chunk-sequential order was
already fully characterized by the test suite; noted as a possible follow-up tweak.

## 3. The perf contract: decode-regime eligibility cap

The op raises `OpNotEligible` for `T > 8` (measured crossover, wall us):

| T | chain | fold | ratio |
|---|---|---|---|
| 1 | 23.0 | 23.3 | 0.99x |
| 2 | 25.5 | 23.2 | 1.10x |
| 4 | 27.9 | 23.4 | 1.18x |
| 8 | 31.4 | 28.7 | 1.09x |
| 16 | 36.6 | — | falls back |
| 32 | 33.2 | — | falls back |

Profiler device time per call (in-graph truth; production shape):

    T=1: chain cast 1.8 + gemvx 5.5 + router 2.8 = 10.1 us / 3 nodes
         fold  gemv 9.2           + router 2.8 = 12.0 us / 2 nodes
    T=4: chain 1.8 + 15.1 + 2.8                 = 19.7 us / 3 nodes
         fold 13.4              + 2.8           = 16.2 us / 2 nodes
    T=8: chain 2.5 + 19.9 + 3.2                 = 25.6 us / 3 nodes
         fold 21.8              + 3.2           = 24.9 us / 2 nodes

**Net for the decode step**: −1 node per MoE layer (−42/step at GLM-5's 42 layers,
plus the cast's [T, 4096] fp32 traffic), device-time parity to +18% at T ≤ 8, and the
op self-falls-back above the crossover so prefill keeps cuBLAS. The original −84
nodes/step estimate assumed the fold also serves prefill; on GB10's 48-SM fp32 budget
it cannot (cuBLAS is compute-bound-optimal there), so the honest cut is −42 nodes +
one cast's traffic per step.

## 4. Numerics contract

* **Widening exact**: bf16/fp16 x/weight widen to fp32 losslessly in-kernel — the
  multiplied VALUES are bit-identical to the eager `x.float()`/`w.float()` casts.
* **Deterministic GEMV order**: each expert's dot is computed wholly inside one
  program (expert slices are disjoint — no atomics, no split-K, no cross-CTA
  reduction): ascending `BLOCK_H` chunks, pairwise tree within a chunk, sequential
  accumulation across chunks, `enable_fp_fusion=False` (no FMA contraction). The
  production `(288, 4096)` config is pinned in `_CFG` so the order is stable.
* **Logit band vs cuBLAS**: max ~1.8e-4 absolute at [288, 4096] over a seed sweep —
  fp32 reassociation only (relative error is meaningless near-zero logits; early
  "7e-3 rel" readings were near-zero logits with cancellation). Test pins < 5e-4.
* **Selection**: sub-noise boundary ties can legitimately flip (the HF-router-warned
  ulp class). Pinned 0 flips vs eager on 320 fixed-seed production rows; the
  adversarial boundary test proves flips can ONLY cross sub-noise (<1e-5) reference
  boundaries.
* **Epilogue rounding points = the incumbent's** (the epilogue IS the incumbent
  kernel).

## 5. Tests (`tests/python/test_glm_router_dense.py`, 10 green)

CPU contract tests + GPU: eager-reference parity (shapes incl. E-tail E=100,
H-tail H=96/100, K=1, T=1), the logit band, 320-row fixed-seed selection parity,
adversarial boundary pairs (eps=1e-1 pinned 0 flips; eps=2^-22 flips all
sub-noise-legitimate), all-tie bit-exactness vs the incumbent (dense + shared, both
bias regimes), K==E all-tie with the epsilon-floor caveat, dtype superset
(fp32/bf16/fp16 x × fp32/bf16/fp16 w), deterministic repeat, and edge envelopes
(empty T, group-config rejection, non-contiguous rejection, **the T > 8 cap firing**
with T=8 at the edge still eligible). Sibling suites (`test_glm_router.py`,
`test_glm_router_shared.py`) still green: 26/26 total.

ty diagnostics on the op are the house lazy-import false positives (identical class
in `glm_router.py`); the bench file is ty-clean.

## 6. Wiring notes (for the floe side, not done here)

Call the op under the policy knob with the usual `except OpNotEligible` fallback; the
cap makes prefill fall back automatically. `return_logits=True` returns the fold's own
fp32 `[T, E]` logits (already materialized for the epilogue launch — no extra node)
for callers keeping the `(logits, weights, indices)` signature.
