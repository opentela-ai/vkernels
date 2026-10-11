# Remaining vkernels branch reconciliation — 2026-10-11

Base: `48b8517dae603497cfcab2d0456a4cbfffd58b62` (user consolidation, pushed to origin/main).

This review compares branch-modified paths against main, including blob equality and semantic differences. Commit ancestry alone overcounts missing work after squash merges. The accompanying inventory records merge bases, complete source heads, touched paths, exact matches and differing files.

## Registry resolution

Keep main’s canonical registry. Its priority bands, `FormatSignature` gates, declared weight preprocessors and lazy imports extend the earlier registry concept. In particular, the old native priority `100` is incompatible with the current `[0,20)` contract. Replaying add/add resolutions from the old experiment branches would remove these features. Recovered dense and expert experiments remain explicit APIs and do not register themselves.

Keep main’s zero-route support and backend-qualified geometry. Shape/warp changes alter reduction order; the glm53 siblings are alternatives, not additive optimizations. Main’s arbitrary-leading-dimension forget-gate reference also replaces the older variant’s flattened shape assumption.

## Disposition of every branch

No original branches are deleted or force-updated. Each original head is preserved remotely as `archive/remaining-20261011/<original-branch>` (a lightweight tag). The draft Issue 148 reconciliation is reviewed separately from these recovered CUDA experiments.

| Branch / source head | Disposition | Rationale |
| --- | --- | --- |
| `beverin/fnuz-grouped` / `ab3fe45` | Covered by main | FNUZ storage support retained; main also has BF16 resident kernels, backend-specific geometry, and safer optional imports. |
| `consolidate/serving-performance-20261006` / `2c86f32` | Covered by main | Latent gather, BF16 route boundaries, fixed-order combine and zero-route work are present; retain later registry/capture fixes. |
| `consolidate/serving-stack-20261010` / `22cb8d1` | Covered by main | 153 of 156 touched files identical; remaining differences are AMD eligibility documentation and later KDA rounding support. |
| `consolidate/serving-stack-20261011` / `826545b` | Recovered here | Carry forward the final combined validation documentation from 826545b. |
| `fix/issue-148` / `2268cf0` | Separate draft PR | Rebase additive FNUZ/MFMA experiments onto main, preserve tuning/registry/zero-route support, fix quantizer sign/zero handling, and require gfx942 qualification. |
| `import/sgl-push-20261009` / `bd9abf6` | Covered by main | 20 of 21 files identical, including donor code/provenance; remaining test difference is optional Torch import handling. |
| `lvr/allreduce` / `e0e74dd` | Recovered and repaired here | Recover benchmark; require torchrun TP4, bind device before NCCL initialization, honor payload threshold, restore source each iteration, check exact integer sums, propagate failures. |
| `optimize/clariden-ep-gemv-zero-route-skip` / `fe7dfb6` | Covered by main | Native zero-route early return and changed-input registry regression already present. |
| `optimize/clariden-ep-zero-route-skip` / `887ca4c` | Covered by main | SGL tiny-MoE zero-route support already present alongside later weighted-down support. |
| `optimize/clariden-expert-geometry` / `ce2f09a` | Alternative archived | Preserve current qualified geometry and registry integration; branch is an older launch variant and would remove zero-route support. |
| `perf/decode-row-invariance-20261004` / `215a72b` | Recovered here | Add opt-in independent_rows for BF16/FP8 dense GEMV; retain canonical registry and default batched dispatch. |
| `perf/dsa-latent-gather-20261006` / `a2172c6` | Covered by main | Latent-gather implementation is byte-identical; retain newer optional-import fixes and canonical registry. |
| `perf/gemv-mi300` / `201a321` | WIP archived, not merged | 109 of 118 touched files identical to main. All nine differences are older zero-route/KDA rounding, source-path, or test fixes; no distinct missing file. Whole WIP replay would regress them. |
| `perf/glm53-clusters-20261003` / `5e23c31` | Alternative archived | Common fixed-order route kernel exists in main; keep newer registry, bounded dispatch, and KDA precision support. |
| `perf/glm53-decode-campaign-20261003` / `bb59ed6` | Alternative archived | Keep main backend-specific tuning policy; sibling instead rejects all non-4-row/4-warp grouped configurations. That stricter numerical policy remains available in the archive for future target qualification. |
| `perf/glm53-kernels-20261004` / `de1d8e8` | Tests recovered here; alternative archived | Recover native-vs-portable bitwise and graph regression; retain main launch geometry and routing support. |
| `perf/glm53-native-tile-20261004` / `33aaf9e` | Tests recovered here; alternative archived | Same recovered bitwise regression; row-8 launch variant requires independent device/performance/quality qualification before promotion. |
| `perf/glm53-native-tile8w-20261004` / `a0b6dc1` | Tests recovered here; alternative archived | Same recovered bitwise regression; row-8/warp-8 defaults cannot be combined with competing launch defaults. |
| `perf/glm53-qualified-20261004` / `4048aa5` | Alternative archived | Fixed-order route accumulation already retained; keep later shared registry and precision/capture fixes. |
| `perf/long-decode-20261005` / `438fbeb` | Covered by main | Serving reduction layout and long-decode cap support retained; older registry copies not restored. |
| `perf/moe-bf16-combine-20261004` / `d1b9571` | Covered by main | BF16 route boundary and deterministic combine already retained; preserve newer bounded dispatch and tests. |
| `perf/moe-bf16-tensorcore-20261006` / `fa20915` | Experimental API recovered here | Recover isolated expert_fp8_tensorcore and tests. No automatic selection; BF16 tensor-core accumulation order needs full-model quality qualification. |
| `perf/small-batch-dense-20261006` / `f160594` | Experimental API recovered here | Recover isolated split-K BF16 tensor-core dense operator and tests. Historical SM90 table remains explicit experimental API, outside default dispatch. |
| `snapshot/local-vkernels-20261010` / `7efb396` | Covered by main | 122 of 133 touched files identical; other files have later eligibility, numerical, optional-import and test fixes. |
| `t5-soup` / `c3de2f5` | Code covered; follow-ups archived here | Reviewed 66-commit history. Main retains source features including router shared-slot output and row resolver. Preserve missing 418-line append under docs/branches/t5-soup-followups.md; retain later reference-rank test fixes. |

## Validation and limits

Local NVIDIA GB10 (SM121), Torch 2.13.0+cu130, Triton 3.7.1: 201 passed, 2 skipped across the recovered experiments plus dense/expert GEMV, registry, grouped MoE, combine, latent-gather and forget-gate suites. The skips are backend/architecture eligibility cases. New independent-row tests compare exact output bits with separate M=1 calls at batch widths 1/3/4/8, then replay with changed activations, weights and scales. Expert native/portable graph tests also change expert IDs.

Ruff passes on all edited Python modules; collective benchmark CLI and syntax checked. The collective needs four colocated GPUs and has not been executed locally. SM90 public small-batch selection, GH200/MI300A timing and full-model NLL are not established by SM121 operator tests. The two tensor-core experiments remain outside automatic dispatch. No new serving speedup or quality pass is claimed; the existing +0.01 nats/token budget remains required before serving promotion.

Historical `t5-soup` timings include corrections and confounds and are archived as such. The MI300 WIP is preserved without reapplying code already superseded by main. The separate Issue 148 draft requires ROCm/gfx942 compile, operator parity, graph/stream checks and model NLL before promotion.

CI follow-up: optional Torch imports were guarded in three consolidated MoE
test modules and the CPU mHC oracle test. A clean NumPy/pytest environment
runs **735 passed, 403 skipped, 499 subtests passed**. The three MoE suites
also pass **44 tests** on GB10. The hosted benchmark comparison reproduced
failures on add/scale/relu/GEMM; this PR changes none of their native or Python
benchmark execution code. That check remains unresolved; its threshold was
not relaxed and its results are not evidence of a serving regression or win.
