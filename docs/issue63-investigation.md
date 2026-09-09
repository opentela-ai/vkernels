# Issue #63 subissue investigation — GLM-5.3-Flash / floe gfx942 roadmap

Investigation of **all seven** subissues under the tracking roadmap
[#63 — complete GLM-5.3-Flash / floe kernel adoption and close measured
gfx942 gaps](https://github.com/opentela-ai/vkernels/issues/63):

| # | Title | Primary area |
|---|---|---|
| #64 | graph-safe GLM block-FP8 expert backend (prefill + decode) | floe `fp8_experts`/`glm5_fp8_gemv` (absent) + reusable `moe_fused` |
| #65 | upstream GLM device-native Torch API + complete floe adoption | `torch_ops/` + tests + benchmarks (uncommitted) |
| #66 | specialize GLM batch-one DSA input projections | `q_a_proj [1536,4096]`, `kv_a_proj_with_mqa [512,4096]` |
| #67 | persist GLM kernel tuning; separate reproducibility from quality | frozen TunableOp CSV lifecycle |
| #68 | optimize GLM batch-one KDA gate projections (occupancy-aware GEMV) | `[128,4096]`, `[64,4096]` |
| #69 | make GLM DSA/mHC/KDA HIP bindings stream-safe + explicit backend | `src/python/vkernels/hip_dsa_mhc.py` |
| #70 | accelerate GLM per-key-gated chunked prefill (FP32 state) | floe `_kda_chunk` (absent) + reusable `kda_delta_rule_fwd` |

Every subissue was exercised on CSCS **beverin** `mi300` compute nodes
(AMD Instinct MI300A / gfx942). Evidence manifests live in
`work/issue63/native/` (gitignored) and the cited Slurm jobs.

## Methodology

Three reusable beverin drivers, all sbatch-safe on `mi300`:

* **`meta/scripts/run_issue63_torchops_mi300.sh`** — the torch_ops test
  suite + QKV/MHC projection benchmarks + TunableOp preflight + roofline +
  counter probes, one gated section each (a failure in one section no
  longer aborts the rest).
* **`meta/scripts/run_dsa_mhc_bench_mi300.sh`** (#69) and the sibling
  `run_{dsa_topk,dsa_kpool,moe_fused}_bench_mi300.sh` — build the native
  HIP C ABI (`libvkernels_hip.so`) + correctness tests + perf benches.
* **`meta/scripts/run_kda_bench_mi300.sh`** (#70, new this session) and
  **`meta/scripts/smoke_hip_dsa_mhc_70.sh`** (Python wrapper + KDA perf).

A shared gotcha is documented in
[`docs/torch-ops-mi300.md`](torch-ops-mi300.md): beverin `mi300` compute
nodes ship no `python3.11-devel`, so `/usr/include/python3.11/Python.h`
is absent and Triton's AMD backend cannot JIT — the harness bootstraps the
header from a python-build-standalone tarball and exports `CPATH`.

## Cross-cutting findings (new this session)

1. **Native-HIP harness mislanded on `mi200` (gfx90a).**
   `run_{dsa_mhc,dsa_topk,dsa_kpool}_bench_mi300.sh` had **no `#SBATCH`
   partition directive**, so a plain `sbatch` inherited beverin's default
   `mi200` partition (MI250X / gfx90a) and ran a **gfx942-built** binary on
   the wrong GPU — every DSA shape reported `r=1.000000` (garbage), 17/17
   "failures" (job **629193**, nid002538). **Fixed** by adding
   `#SBATCH --partition=mi300 --gres=gpu:1` to all three; a plain `sbatch`
   now lands on `mi300` (verified job **629263**, nid002484). On the
   correct node (job **629196**, nid002348) the identical DSA + MHC
   correctness is clean (see #69). This is exactly #69's "do not equate
   imported with selected, graph-safe, and production-qualified."
2. **#67 frozen-TunableOp validator bug (fixed, verified).** The loader in
   `torch_ops/qkv_tuned_blas.py` required a validator named `ROCM_VERSION`,
   but PyTorch's `torch.cuda.tunable.write_file` emits `HIP_VERSION`
   (ROCm `major*100+minor`) — **no `ROCM_VERSION` validator exists**, so
   the loader rejected *every* genuine artifact (job 628932, section 5).
   Fixed `ROCM_VERSION → HIP_VERSION` and added a regression
   (`test_accepts_genuine_pytorch_validator_set`, job-628932-shaped
   artifact) plus a `missing_hip_version` rejection case. Verified on
   beverin: `test_qkv_tuned_blas` 16/16 green.
3. **#67 reproducibility-vs-quality gap (design-level, documented).** The
   loader intentionally rejects `Default` winners
   (`and row[2] != "Default"`). But the autotuner is **non-deterministic
   across runs**: job 628932 selected `Gemm_Rocblas_…` for M=1, while a
   re-run (this session, `qkv_projection.tunable0.csv`) selected `Default`
   for M=1 and `Gemm_Rocblas_…` for M=2. So the frozen feature is
   **unconfigurable whenever the autotuner picks `Default` for M=1**,
   conflating reproducibility (record the choice) with quality (must beat
   default) — the very separation #63 calls for.
4. **#70 reusable `kda_delta_rule_fwd` is incorrect for small chunked
   shapes.** `test_kda_correct` (from a single commit, issue #21, whose
   docstring claims all-PASS at `max_rel < 2e-2` with no known-skip
   markers) reports **5 failures** on gfx942 (job **629195**, nid002482):
   `B=1 H=1 S=64 D∈{16,32,64} chunk=16`, `B=1 H=16 S=64 D=64 chunk=16`
   (`max_abs=inf`), and `B=1 H=1 S=512 D=64 chunk=64`
   (`max_abs≈1.2e33`). Larger shapes pass perfectly
   (`N=64 D=512`, `N=8192 D=128`, `max_rel=0.000000`). The huge `max_abs`
   indicates uninitialized/out-of-bounds memory growing through the
   recurrence — a real regression or latent defect in the primitive #70
   must reuse.

---

## #64 — graph-safe GLM block-FP8 expert backend (prefill + decode)

**Requirement.** GLM: 42 MoE layers, 288 routed experts, top-k=8, H=4096,
intermediate=2048; E4M3FN weights with FP32 scales per 128×128 block;
gate/up `[E,4096,4096]`, down `[E,4096,2048]`. Keep weights compressed,
bounded caller-owned scratch, graph-safe decode. Compare vs CPU/Torch on
balanced/skewed routing, M=1/2/64/128/1024; preserve BF16 rounding before
GEMM.

**Current code.** The GLM-specific kernels (`floe/…/fp8_experts.py`,
`glm5_fp8_gemv.py`) are **not in this checkout** (absent on beverin). The
reusable K3-MXFP4 MoE primitive (#41, which #64 says to reuse but "does
not cover this format/shape/scale contract") *is* present:
`meta/benchmarks/{moe_fused,bench_moe_fused,bench_dequant_ab,
moe_fused_oracle}{.hip,.cpp}` + `test_capi_moe.hip` +
`test_moe_fused_correct.hip`, driven by `run_moe_fused_bench_mi300.sh`.

**beverin evidence (job 629194, nid002348, gfx942).** All three criteria:
* GPU vs CPU oracle (K3 routing, E=256 hidden=7168 ispp=512 top_k=16):
  M=1/2/4 all `ok`, cpu-rel 0.0015–0.029, evt 1106–1813 µs, 0.020–0.049 TFLOP/s.
* Default harness single decode (E=256 hidden=4096 ispp=512 top_k=6):
  M=1/2/4/8 all `ok`, cpu-rel ≈0.00057, evt 397–820 µs.
* Host dequant A/B: **2.82× speedup** (LUT 5.40 µs/tile vs ref 15.22 µs/tile),
  **bit-exact PASS** (7.34 M bf16 across 1792 tiles; 4.19 M across 1024 tiles).
* Host oracle (full fused-MoE CPU reference) for hidden=512/7168.

**Findings.** (1) The reusable K3-MXFP4 MoE primitive is healthy on gfx942
(correct + 2.82× dequant, bit-exact) — the infrastructure #64 builds on is
sound. (2) The **GLM block-FP8 (E4M3FN + per-128×128 FP32 scales,
`42×[E,4096,4096]`/`[E,4096,2048]`) path is not in vkernels** — it lives in
floe and is the #65/#64 integration gap. (3)
`run_moe_aux_correct_mi300.sh` requires the gone 20.8 GB
`vkernels-full-v2.tar` container → broken harness. (4) No graph-safe-decode
or quality evidence for the GLM block-FP8 path itself.

**Status.** Reusable primitive verified on beverin; GLM-specific E4M3FN +
per-block-scale expert backend + graph-safe decode = open (kernels absent
from this checkout).

---

## #65 — upstream the GLM device-native Torch API + complete floe adoption

**Requirement.** Land the existing prototypes (don't recreate). Define
public dtype/layout/device/state/scratch contracts. Move remaining GLM
kernels into vkernels with thin floe adapters. Publish/version artifacts.
Record selected backend, source/version, fallback reason. Acceptance:
clean checkout/install without an unstaged sibling tree; GPU tests cover
parity, non-current-device restoration, graph replay with changed inputs,
missing-backend fallback; floe dispatch tests prove which implementation
runs; opt-in for unqualified variants.

**Current code.** `src/python/vkernels/torch_ops/{mhc_projection,
qkv_projection,qkv_tuned_blas}.py` + `hip_dsa_mhc.py` + their tests
(`test_{mhc_projection,qkv_projection,qkv_tuned_blas}.py`) + benchmarks —
**all untracked locally and absent from GitHub main** (the core gap). Floe's
beverin launcher stages sibling `src/python/vkernels` into each source
snapshot — source-snapshot integration, not a released dependency.

**beverin evidence (job 628932 + re-run).** `torch_ops` tests —
`test_mhc_projection` + `test_qkv_projection` + `test_qkv_tuned_blas` —
**34 passed** on MI300A (gfx942). MHC + QKV projection benchmarks ran
(JSON). After the #67 fix, `test_qkv_tuned_blas` is **16/16 green**
(15 original + the new regression) on beverin.

**Findings.** (1) The package + tests + benchmarks pass on beverin but are
**uncommitted** — the #65 "upstream/versioned" deliverable is not met.
(2) Clean-install without an unstaged sibling tree is **not verified**
(floe uses source-snapshot staging). (3) The GPU acceptance criteria
(graph replay with changed inputs, missing-backend fallback dispatch,
non-current-device restoration) are partially covered by the numpy/torch
CPU unit tests (importability, lazy imports, capture-refusal) but the floe
**dispatch tests that prove which implementation executes** are not present
in this checkout.

**Status.** Implementation validated on beverin; upstream/release +
dispatch-test acceptance = open (uncommitted + floe adapters absent).

---

## #66 — specialize GLM batch-one DSA input projections

**Requirement.** Across 11 DSA layers, `q_a_proj [1536,4096]` 2.061
ms/token and `kv_a_proj_with_mqa [512,4096]` 1.644 ms/token = 3.705
ms/token of attributed GPU work. Compare existing/tuned BLAS and
device-native GEMV for BF16 M=1/2; retain efficient BLAS for larger
prefills. Preserve dtype/rounding; validate numerical error, graph replay,
device/stream; matched microbenchmarks + full-model A/B. Do not claim the
~0.062 ms/token launch reference is an attainable speedup.

**Current code.** `torch_ops/mhc_projection` (weights `[16384,4096]`) and
`torch_ops/qkv_projection` (weights `[8192,4096]` ×3) are specialized
JIT-by-shape Triton GEMV kernels — **not** the DSA input shapes
`[1536,4096]`/`[512,4096]`.

**beverin evidence (job 628932).** The projection *methodology* (default
BLAS vs tuned BLAS vs fused Triton GEMV, M=1/2, parity, graph capture) is
demonstrated on QKV `[8192,4096]` and MHC `[16384,4096]`. No benchmark
exercises the exact #66 shapes.

**Findings.** (1) **No DSA-input-projection specialization exists** in
`torch_ops` — `q_a_proj` and `kv_a_proj_with_mqa` are a distinct
implementation effort that would reuse the established batch-one GEMV
methodology but on different shapes. (2) The existing QKV/MHC benchmarks
prove the methodology is sound but do not directly measure #66's
opportunity. (3) The measured 3.705 ms/token gap (job 628645) is
unvalidated against the current HIP wiring (per #63).

**Status.** Methodology validated on beverin; DSA-input specialization +
full-model A/B = open.

---

## #67 — persist GLM kernel tuning; separate reproducibility from quality

**Requirement.** Persist tuning with shape/dtype/layout, device arch/CU,
software versions, kernel-source fingerprint, chosen algorithm; reject
stale/mismatched artifacts. Finish compilation/tuning before capture. Test
cold/warm capture and invalidation. Define separate reproducibility,
quality/non-inferiority, performance-only modes; held-out evaluation and
acceptance thresholds before promotion. Record actual argmax IDs, not
topk tie order. Resolve process-global TunableOp concurrency semantics.

**Current code.** `torch_ops/qkv_tuned_blas.py` — frozen TunableOp CSV
loader + wrapper; toggles process-global settings; serialized-inference-
only; refuses capture. The preflight script is
`meta/benchmarks/check_qkv_tuned_blas.py`.

**beverin evidence.** Two concrete defects surfaced by running the harness:
* **Finding A — validator-name bug (fixed, verified).** The loader required
  `ROCM_VERSION`; PyTorch emits `HIP_VERSION`. Fixed + regression test; 16/16
  on beverin (see cross-cutting #2).
* **Finding B — `Default`-winner reproducibility gap (documented).** The
  loader rejects `Default`, but the autotuner picks `Default` for M=1
  non-deterministically across runs (see cross-cutting #3). The genuine
  CSV (`work/issue63/native/qkv_projection.tunable0.csv`) shows
  `…,tn_8192_1_…,Default,0.0203689` and `…,tn_8192_2_…,Gemm_Rocblas_…,0.0209353`.

**Findings.** (1) The frozen artifact loader had a hard portability bug
that rejected every real artifact — now fixed. (2) The "reject `Default`"
rule conflates reproducibility with quality and must be separated per
#67's explicit scope (record the autotuner's choice, even `Default`, and
gate promotion separately). (3) Process-global TunableOp concurrency and
cold/warm capture/invalidation are not yet tested.

**Status.** One real bug fixed + verified on beverin; reproducibility/
quality separation, concurrency, and capture-invalidation acceptance =
open.

---

## #68 — optimize GLM batch-one KDA gate projections (occupancy-aware GEMV)

**Requirement.** Three KDA input projections per layer sharing hidden
states: `forget_gate.f_a_proj [128,4096]` 2.483, `g_a_proj [128,4096]`
2.297, `b_proj [64,4096]` 1.765 ms/token = 6.545 ms/token. Counter job
628657 found one workgroup on 228 CUs, ~32× padded FLOPs. Compare tuned
BLAS with occupancy-aware GEMV/split-K; optionally fuse shared-input
projections after individual improvements. CPU oracle + MI300A tests for
M=1/2 and unsupported-shape fallback, graph replay, device/stream,
read-only weights. The ~0.210 ms/token launch-adjusted roofline is
**optimistic**, not a requirement.

**Current code.** `torch_ops/qkv_projection` (default/tuned BLAS vs
fused Triton, M=1/2) and the counter probe `meta/benchmarks/bench_roofline_counters.py case=gate128`
(weights `[128,4096]`, `F.linear`, rocprof `Cijk_` filter).

**beverin evidence (job 628932, section 7).** `bench_roofline_counters.py gate128`
ran **rc=0 on gfx942** — weights `[128,4096]`, `F.linear`, exactly #68's
representative shape, instrumented to find the single-workgroup dispatch
(discard 1 warmup, 8 cold). `bench_qkv_projection` (default/tuned BLAS +
fused Triton) also ran on gfx942.

**Findings.** (1) The counter probe **directly reproduces #68's diagnosis**
(`[128,4096]` default BLAS = single workgroup on 228 CUs, ~32× padded
FLOPs) on gfx942. (2) The `qkv_projection` methodology is the template for
#68's occupancy-aware GEMV/split-K, but #68's `[128,4096]`/`[64,4096]`
shapes are not yet specialized (same shape of gap as #66). (3) The
split-K/fusion of the three shared-input projections is open.

**Status.** Diagnosis reproduced on beverin; occupancy-aware GEMV/split-K
specialization + fused shared-input projections + full-model A/B = open.

---

## #69 — make GLM DSA/mHC/KDA HIP bindings stream-safe + explicit backend

**Requirement.** Extend the existing ABI with an explicit stream and
caller-owned scratch; do not relaunch on stream 0 or allocate/free per
decode. Establish explicit capability/backend/precision selection before
capture; log fallback reasons. Define and test normalization, rounding,
state-update, output-dtype per backend. Repair the `_fn_f32` cache
invalidation (device changes only, not weight replacement/in-place-update).
Test eager + captured execution on each MI300A, non-default streams,
non-current devices, changed-input replay, reset/reuse, missing-library.
Never recover from a partial launch/capture by silently continuing.

**Current code.** `src/python/vkernels/hip_dsa_mhc.py` wraps the HIP C ABI
(`dsa_sparse_fwd`, `mhc_pre_gemm_sqrsum`, `mhc_post`, `kda_delta_rule_fwd`,
`kda_delta_rule_fwd_with_scratch`) and **explicitly refuses graph capture**
(`_check_device` raises if `is_current_stream_capturing()`): the kernels
launch on the legacy default stream, which cannot be captured. Floe tries
the HIP mHC path before the `GLM53_MHC_PROJECTION` Triton selector. The
HIP pre-GEMM path computes raw + rescales after; the measured Triton path
normalizes and rounds BF16 before projection.

**beverin evidence (job 629196, nid002348, gfx942 + smoke nid002484).**
* Native `test_dsa_correct`: **PASS (0/16)**, all shapes `max_rel < 0.004`.
* Native `test_mhc_correct`: **PASS (0/11)** — `mhc_pre_gemm_sqrsum`
  (abs ≈1e-1, rel ≈2e-5), `mhc_post` (abs ≈3e-2, rel ≈3.9e-3).
* DSA sparse-MLA forward perf: 9 shapes, us(med) 204–1841 µs, 0.013–2.625
  TFLOP/s, 7.1–1473 GB/s, AI 1.8–2.0 — all **mem** bound vs roof 1307
  TFLOP/s, 5300 GB/s, ridge ~247 FLOP/B. Autotuner BQ sweep 3126–3191 µs.
* MHC pre/post + DSA indexer top-k logits perf (bs/H/ms_len sweep,
  266–926 µs, AI 57.2–60.2, all mem bound, 228 CUs).
* Python wrapper smoke: `hip_dsa_mhc.available() = True` (lib loads via
  `$VKERNELS_LIB`), **capture refused OK** (stream-0 refusal fires inside
  an active graph), **eager `dsa_sparse_fwd` OK** (out `(1,1,1,4)`,
  lse `(1,1,1)`).

**Findings.** (1) The stream-0 capture refusal is **by design and verified
to fire**; the #69 "extend the ABI with an explicit stream parameter"
remains open — the wrapper still launches on legacy stream 0, so it is
correct-but-incapable-of-capture, exactly the gap the issue names.
(2) The HIP-vs-Triton **precision-policy distinction** (raw + rescale vs
normalize + round-before-projection) and the `_fn_f32` cache invalidation
gap are **not yet tested** per-backend. (3) The earlier `r=1.000000`
"failures" (job 629193) were the `mi200` mislanding, not a kernel defect
(see cross-cutting #1) — now fixed by the SBATCH directive.

**Status.** Correctness + perf + Python-wrapper capture-refusal verified on
beverin; explicit-stream ABI, per-backend precision/cache tests,
missing-library + changed-input + reset/reuse acceptance = open.

---

## #70 — accelerate GLM per-key-gated chunked prefill (FP32 state)

**Requirement.** Decode recurrence is fused (~0.292 ms/token); prefill is
dominated by floe's Torch `_kda_chunk` (1.125 s / 22.98% of 1024-token
GPU kernel time, profile 628645). Establish which existing HIP primitives
can be reused and benchmark vs the current Torch chunk path. Preserve
GLM per-key gates `[B,H,S,D]`, normalization epsilon, FP32 recurrent
state/accumulation, causal ordering, BF16 projection/output boundaries.
Support nonzero initial state, arbitrary chunk/tail lengths, masks,
reset/reuse, chunked-prefill-to-decode handoff. CPU/Torch oracles +
chunked-vs-recurrent tests (fresh and continued sequences). Report
isolated prefill timings + full-model TTFT/NLL/continuation.

**Current code.** The reusable native primitive is
`meta/benchmarks/{test_kda_correct,bench_kda}.hip` (issue #21); `hip_dsa_mhc.py`
exposes `kda_delta_rule_fwd`/`kda_delta_rule_fwd_with_scratch` (the
per-key-dim gated forward `[B,H,S,D]` GLM needs). floe's `_kda_chunk` (the
cost this primitive would replace) is **not in this checkout**.
**No `run_kda_bench_mi300.sh` existed** — a documented-runner gap, filled
this session.

**beverin evidence (job 629195, nid002482, gfx942).**
* `test_kda_correct`: `kda_gate_chunk_cumsum` PASS, `kda_pack_bitmatrix`
  PASS, but `kda_delta_rule_fwd` **5 failures** on small chunked shapes
  (B=1, chunk=16, D≤128; one `inf`; one S=512 → 1.2e33). Larger shapes
  pass with `max_rel=0.000000`. (See cross-cutting #4.)
* `bench_kda.sh` (perf, job 629195 smoke, A_rc=0): `kda_delta_rule_fwd` 6
  shapes, us(med) 63–3930 µs, 0.002–0.153 TFLOP/s, 4.4–355 GB/s,
  AI 0.41–0.43 — all **mem** bound vs roof 1307 TFLOP/s, 5300 GB/s,
  ridge ~247 FLOP/B. `kda_layer_norm_gated N=8192 D=128` 175.9 µs / 71.5 GB/s.

**Findings.** (1) The reusable `kda_delta_rule_fwd` is **HBM-bound at
~1–7% of peak BW** (AI 0.41–0.43, far below ridge ~247), confirming the
chunked-state traffic (~16 KB/token/head) is the dominant cost — the
#70 optimization target. (2) The same primitive is **incorrect for small
chunked shapes** (cross-cutting #4), which must be fixed before #70 can
reuse it for "arbitrary chunk/tail lengths." (3) floe's `_kda_chunk` (the
1.125 s cost) is absent — the integration gap. (4) `run_kda_bench_mi300.sh`
was missing (now added).

**Status.** Reusable primitive benchmarked on beverin (mem-bound, matches
the diagnosis); small-shape correctness defect + GLM chunked-prefill
integration + TTFT/NLL/continuation acceptance = open.

---

## Evidence manifest

All artifacts under `work/issue63/native/` (gitignored; sanitized job logs
only — no weights or private paths).

| Job | Node | Partition | Issue | Artifact |
|---|---|---|---|---|
| 628932 | nid002344 | mi300 | #65 #66 #67 #68 | torch_ops tests + QKV/MHC bench + roofline + counter probes |
| 628645 | (prior) | mi300 | #63 baseline | full GLM-5.3-Flash profile (referenced in issues) |
| 629193 | nid002538 | **mi200** ✗ | #69 | DSA garbage on gfx90a (the mislanding, pre-fix) |
| 629194 | nid002348 | mi300 | #64 | `moe_fused_bench.629194.out` — oracle + dequant 2.82× + host oracle |
| 629195 | nid002482 | mi300 | #70 | `kda_bench.629195.out` — 5 delta_fwd failures + gate/pack PASS |
| 629196 | nid002348 | mi300 | #69 | `dsa_mhc_bench.629196.out` — DSA+MHC correct + 3 perf tables |
| 629263 | nid002484 | mi300 | #69 | SBATCH-fix verification (plain `sbatch` lands on mi300) |
| (smoke) | nid002484 | mi300 | #69 #70 | `hip_dsa_mhc.py` available + capture-refusal + eager + KDA perf |

## Changes made this session

* **`src/python/vkernels/torch_ops/qkv_tuned_blas.py`** — fixed the
  frozen-artifact validator (`ROCM_VERSION → HIP_VERSION`) with a comment
  documenting PyTorch's actual validator schema (cross-cutting #2).
* **`tests/python/test_qkv_tuned_blas.py`** — aligned the fake CSV with the
  corrected schema; added `test_accepts_genuine_pytorch_validator_set`
  (job-628932-shaped regression) and a `missing_hip_version` rejection case.
* **`meta/scripts/run_{dsa_mhc,dsa_topk,dsa_kpool}_bench_mi300.sh`** — added
  `#SBATCH --partition=mi300 --gres=gpu:1` (+ job-name/time) so a plain
  `sbatch` lands on MI300A, not the default `mi200` (cross-cutting #1).
* **`meta/scripts/run_kda_bench_mi300.sh`** (new) — documented MI300A
  runner for the KDA primitive (#70), building `test_kda_correct` +
  `kda_bench` and driving `bench_kda.sh`.
* **`meta/scripts/smoke_hip_dsa_mhc_70.sh`** (new) — one-shot `hip_dsa_mhc.py`
  wrapper smoke (available + capture-refusal + eager) + KDA perf.
* **`docs/torch-ops-mi300.md`** (prior) + this `docs/issue63-investigation.md`.

The torch_ops package/tests/benchmarks and `hip_dsa_mhc.py` remain
uncommitted local work pending the #65 upstream decision.
