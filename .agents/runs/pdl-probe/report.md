# PDL probe — report

**Lane:** 17 (PDL / launch-gap compression) · **Date:** 2026-02-14 · **Rig:** none (local GB10)
**Verdict: NO-GO for a serve arm on GB10. GO for a code-only rig probe (harness is done and portable).**

---

## 1. Installed Triton PDL surface & GB10 support

| item | finding |
|---|---|
| Triton | **3.8.0** installed in `.venv` (context said 3.7.1+; API family is the same) |
| API | `triton.language.extra.cuda.gdc_wait()` / `gdc_launch_dependents()` (`gdc.py`, inline `griddepcontrol.*` PTX); `launch_pdl: bool` is a `CUDAOptions` field accepted **as a per-launch kwarg** (`kernel[grid](..., launch_pdl=True)`) |
| plumbing | `launch_pdl` → compile metadata → `backends/nvidia/driver.c` sets `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 1` on the launch. **Attribute is consumer-side only** — the producer needs no attribute and no annotation |
| GPU | NVIDIA GB10, **compute capability 12.1 (sm_121)**, 48 SMs, 24 MB L2, driver 580.95.05, torch 2.14.0+cu130 |
| HW support | PDL is documented sm_90+ (Hopper and beyond) → **GB10 qualifies; the probe is NOT code-only** — the microbench ran locally |
| empirical | the attribute is *honored*: a consumer's 74 µs producer-independent prologue hides completely under a 167 µs producer (functional probe below), eagerly **and** under CUDA-graph capture+replay |

Smoke test: `.agents/runs/pdl-probe/smoke_pdl.py` — eager PDL launch, graph capture of 10 chained PDL pairs, replay, all bit-correct.

## 2. Harness

`.agents/runs/pdl-probe/bench_pdl.py` (self-contained; production kernels untouched — all PDL variants are **copies** in the bench file):

- **Functional probe** (P1): 167 µs spin producer + consumer with a ~74 µs producer-independent prologue before its single dependent load. If PDL works, chain time drops by ≈ the prologue.
- **Minimal elementwise pair** (silu_mul_clamp's shape class): 100 × (fill → scale), per-pair buffers (WAR-free, so an un-waited PDL consumer is memory-safe), 200 launches per graph.
- **mHC chain** (real fused serving cycle, per site): `pre_gemv` (streams→mix logits) → `big_fuse` (gates+Sinkhorn+collapse+RMSNorm) → `compose` (writeback); site *i*'s compose output feeds site *i+1* — every boundary is a true producer→consumer dependency. 40 sites = 120 launches; copied verbatim baseline bodies validated **bit-identical** against the production wrappers (`mhc_pre_gemv` / `mhc_pre_big_fuse` / `mhc_compose`) before timing.
- **Variants per chain:** `baseline` (verbatim launches) / `pdl_early` (`gdc_launch_dependents()` at kernel top) / `pdl_late` (trigger after the last store — donor-style) / `pdl_consumer_only` (untouched producer, PDL consumer — the only option when the producer isn't ours, e.g. silu_mul_clamp after a grouped GEMM). All consumers `gdc_wait()` before their first dependent load.
- **Timing:** CUDA-graph capture + replay (the serving regime), **A/B interleaved** rounds across variants (8 rounds × min-of-8 replays) to cancel GB10 DVFS drift (±3-4% run-to-run otherwise). Exec estimates for the minimal pair via 64-pass loop kernels (L2-resident).
- **Parity after graph replay** checked for every PDL variant: bit-identical to the baseline chain on all outputs.

## 3. Measurements

### P1 — is PDL honored? YES (twice, A/B interleaved)

| pair (spin 166.8 µs + consumer 84.7 µs) | graph min µs |
|---|---|
| baseline | 249.3 |
| **PDL** | **176.0** |

Overlap = **73.3 µs ≈ the entire prologue**. PDL works on GB10/sm_121 under graph replay and eager alike. The mechanism is available; what fails is the *payoff* on tiny kernels ↓.

### P2 — chained-launch gap compression (per-launch cost, µs; two independent runs)

| config | baseline | pdl_early | pdl_late | consumer_only |
|---|---|---|---|---|
| minimal pair (200 launches) | 1.594 / 1.535 | 1.568 / 1.503 | 1.595 / 1.556 | **1.733 / 1.682 (+9%)** |
| mHC chain b=1 (120 launches) | 3.106 / 3.110 | 2.977 (−4%) / 2.976 | **2.897 (−7%) / 2.896** | 3.108 / 3.113 (0) |
| mHC chain b=4 (120 launches) | 3.674 / 3.668 | **4.051 (+10%) / 4.075 (+11%)** | 3.639 / 3.614 (−1%) | 3.691 / 3.672 (0) |

Readings:

- **Best case ≈ 0.21 µs/launch saved** (mHC b=1, late trigger) ≈ **0.3-0.45 µs per mHC→mHC boundary** — i.e. **~30-45% of the donor's 0.74→0.06 µs law** (0.68 µs recoverable/boundary claimed; we realize ≤0.3 µs).
- The minimal elementwise pair compresses **nothing** beyond noise (implied baseline gap ≈ **1.17-1.26 µs** on GB10 — the 0.74 µs baseline itself does not reproduce on this part; boundaries are *bigger* and mostly PDL-immutable).
- **consumer_only is a consistent ~9% regression** on the elementwise pair (3/3 runs) and neutral on mHC. The gdc_wait early-launch-poll path costs more than plain same-stream serialization for tiny kernels. This kills silu_mul_clamp (its producer is a grouped fp8 GEMM we don't own → consumer_only is its only wiring).
- **Early trigger is actively harmful at b=4** (+10-11%, reproduced): `pre_gemv` at 96 CTAs × 16 warps already fills the machine; early-launched consumer CTAs polling in `gdc_wait` steal SM slots from the producer. Trigger placement is a first-order design variable, not a detail.
- Latency structure at b=1: pre_gemv (~2.3-2.7 µs solo) leaves room for the dependent's launch to hide → that's where the 0.21 µs comes from; at b=4 the machine is full and there is no slack.

### P3 — parity

All PDL variants bit-identical to the baseline chain, eagerly and **after graph capture+replay**, on both token buckets (`pdl_chain_parity: true`, `copies_match_production: true` in both result files).

## 4. Serving extrapolation (conservative, using measured numbers only)

Assumptions from the lane context: mHC chain = 270 nodes/step; under the `mhc_big_fuse` knob that is 3 mHC launches per block × 90 blocks (45 layers × attn+ffn). Eligible boundaries = adjacent same-stream pairs whose consumer we control:

- 90 × (`pre_gemv` → `big_fuse`) + 90 × (`compose` → next block's `pre_gemv`) = **180 eligible mHC→mHC boundaries**;
- 90 × (sublayer GEMV → `compose`) are consumer-only → measured **≤ 0, likely negative** → counted as 0.

At bs=1 with the best measured config (late trigger): 180 × ~0.2-0.3 µs ≈ **36-55 µs/step against ~840 µs of mHC-chain graph time → ~4-7% of the chain, low-single-digit % of a full decode step**. At bs=4: **~0** (and −10% if trigger placement is naive). Every non-mHC boundary (silu_mul_clamp, MoE aux, router) is consumer-only → measured negative. The donor's extrapolation (0.68 µs × boundaries ≈ 120-180 µs/step) does **not** transfer to GB10.

## 5. GO/NO-GO

**NO-GO for a serve arm on GB10:**

1. The win is ~0.2-0.3 µs/boundary best case (bs=1 only) vs the 0.74 µs law — a few % of step time, within the risk envelope of the reproduced regressions (−9% consumer-only, −10% early-trigger at b=4).
2. GB10's boundary dead time (~1.2-1.3 µs elementwise) is PDL-immutable for wait-first tiny kernels: the cost is completion→flush→next-grid issue, not launch-API latency. The donor's 0.06 µs endpoint is not reachable here.
3. Wiring cost is real: kernel-body edits (or a TRIG constexpr) in production kernels + `launch_pdl` threading through the model seam + a per-boundary producer/consumer audit across 270 nodes + capture/parity tests ≈ 3-5 days, for a bs=1-only few-% return.

**GO for the follow-up rig probe (code-only cost ≈ 0):** the harness is CUDA-portable as-is (`PYTHONPATH=src/python python .agents/runs/pdl-probe/bench_pdl.py --out ...`). On the donor hardware class (sm_90+, H100/GB200 where the 0.74 µs law was measured) the same A/B run decides in minutes; if the law reproduces there, the serve-arm math changes to ~120-180 µs/step upper bound (180-270 eligible boundaries × 0.68 µs) and the wiring effort above becomes justified.

**Capture-replay risk (verified, for the record):** PDL inside CUDA graphs is legal and bit-exact on torch 2.14.0+cu130 / driver 580.95.05 — verified in the smoke test and in all bench variants, including parity checked *after* replay. Wiring rules that keep it safe: (a) `gdc_wait()` before every load of producer-written data, (b) **no stores before the wait** (replay reuses capture-pool buffers), (c) trigger after the last store (never at kernel top when the producer saturates the SMs), (d) per-launch `launch_pdl=True` is captured into the graph node — no replay-side re-launch needed.

## Files (all uncommitted, this lane only)

- `.agents/runs/pdl-probe/smoke_pdl.py` — API + graph-capture smoke test
- `.agents/runs/pdl-probe/bench_pdl.py` — the harness (copied kernel variants; production untouched)
- `.agents/runs/pdl-probe/bench_results.json` — probe + minimal + mHC (run 1)
- `.agents/runs/pdl-probe/bench_results_rerun.json` — minimal + mHC (stability rerun)

No production files modified; no pushes; no rig used.
