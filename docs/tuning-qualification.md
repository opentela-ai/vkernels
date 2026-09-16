# GLM kernel tuning persistence and qualification (#67)

Frozen tuning artifacts are deployment inputs, not benchmark exhaust. This
document defines the artifact format, the three qualification modes, the
process-global TunableOp concurrency rules, and the reproducible commands.
The implemented slice covers the QKV projection artifact
(`GemmTunableOp_BFloat16_TN`); the same machinery generalizes to future
kernel-specific artifacts.

## Artifact = TunableOp CSV + sidecar manifest

A tuning run produces two files:

* `qkv_projection.tunable0.csv` — PyTorch TunableOp's own artifact
  (`tunable.write_file`, device ordinal inserted). Its validator block
  (`PT_VERSION`, `HIP_VERSION`, `HIPBLASLT_VERSION`, `GCN_ARCH_NAME`,
  `ROCBLAS_VERSION` — note PyTorch emits `HIP_VERSION` = ROCm
  `major*100+minor`, never `ROCM_VERSION`) plus one result row per shape.
* `<csv>.manifest.json` — schema `vk-tuning-manifest/1`, written by
  `vkernels.torch_ops.tuning_manifest`. It records:

  | Field | Meaning |
  | --- | --- |
  | `csv`, `csv_sha256` | the exact CSV bytes described (an edited CSV invalidates the manifest) |
  | `validators` | the CSV validator block, cross-checked on load |
  | `shapes` | per-signature M/K/N/ld, dtype, layout |
  | `algos` | chosen algorithm per shape, **including `Default`** |
  | `device`, `software` | arch + CU count; torch + HIP version code |
  | `producer.fingerprints` | sha256 of the producing/consuming sources |
  | `quality_gates` | thresholds **declared before evaluation** |
  | `job`, `created`, `notes` | provenance |

`Default` winners are recorded as chosen. Recording a choice is
reproducibility; gating it is quality. The loader configures a `Default`
winner (it reproduces the autotuner's decision exactly) and reports it;
`configure_qkv_tuned_blas(..., accept_default=False)` rejects it for
promotions that demand a tuned implementation. This resolves the Finding-B
conflation from `docs/issue63-investigation.md`: the autotuner picks
`Default` for M=1 non-deterministically across runs, so a loader that
rejected `Default` made the frozen feature unconfigurable whenever that
choice won.

## Staleness and mismatch rejection

`configure_qkv_tuned_blas(path)` requires the sibling manifest by default
(`require_manifest=False` is an interactive-debugging escape hatch) and
rejects, with the offending field named:

* missing manifest, unsupported schema, missing keys;
* CSV bytes changed since the manifest was written (sha256);
* manifest validators or algorithms disagreeing with the CSV rows;
* device architecture, CU count, torch or HIP version differing from the
  recording host (skipped only on CUDA-less hosts, where they cannot be
  checked);
* any fingerprinted source absent from the checkout or changed since
  production (producer = `bench_qkv_projection.py` +
  `torch_ops/qkv_projection.py`);
* a PyTorch-side validator rejection or a loaded-results mismatch
  (`read_file` enforces version equality; the loader then re-checks the
  exact winners against `tunable.get_results()` so a preexisting global
  result cannot stand in).

Failure is loud: a stale artifact raises, it is never downgraded to a
warning.

## Qualification modes

`meta/benchmarks/qualify_qkv_tuned_blas.py <csv> --mode {repro,quality,perf}`
writes a JSON report and exits nonzero naming every failed gate. There is no
waiver flag; thresholds live in the manifest, declared before evaluation.

* **repro** — cold and warm artifact load, eager/BLAS parity, graph capture
  with bit-identical replay, changed-input replay, warm reconfiguration
  equality. `Default` winners are legal and reported.
* **quality** — per-configuration numerics against the **FP64-direct BF16
  oracle** (`torch_ops/bf16_oracle.py`), argmax evidence as **actual argmax
  IDs** with top-1/top-2 margins and tie flags (never top-k tie order),
  baseline determinism floor, and the model-level NLL-regression gate. The
  NLL gate **fails closed** until a floe-side held-out measurement is passed
  via `--nll-report {"baseline": …, "candidate": …}`.
* **perf** — graph timings only. Its report is stamped
  `"not claimed - performance-only measurement (issue #67)"`; it is not
  promotion approval.

Declared QKV gates (recorded in every manifest at production time):
`max_abs_vs_fp64_oracle ≤ 0.008` (two bf16 ulps at output scale ~1 — the
candidate and the oracle each sit within one ulp of the exact value),
`max_rel_vs_fp64_oracle ≤ 0.008` (denominator clamped at 0.5), `argmax
agreement ≥ 0.95`, `reference repeat identical`, `model NLL regression
≤ 0.005`. The earlier diagnostic evidence (224 forced-token positions,
217/224 agreement; mean NLL baseline 1.428966 / Triton 1.433701 / tuned BLAS
1.427932) is consistent with these gates but does not by itself constitute a
promotion.

### Why the FP64-direct oracle exists

Torch's `float64 -> float32 -> bfloat16` chain double-rounds near midpoints:
a value just below a bfloat16 midpoint can round up to the midpoint in
float32 and then round up again, while direct round-to-nearest-even from
float64 rounds down (`bf16_oracle.bf16_round_fp64` makes the rounding
decision once, on the FP64 bit pattern; the test suite proves the two paths
disagree on constructed witnesses). Quality comparisons must round from FP64
directly.

## Process-global TunableOp concurrency semantics

TunableOp state is process-global and thread-unsafe:

* `tunable.enable()` / `tunable.tuning_enable()` flags are **not**
  thread-local; the wrapper's context manager restores them but cannot
  isolate concurrent threads.
* Loaded results persist in the process; nothing unloads them, and an
  already captured graph keeps its baked choices forever.
* `configure_qkv_tuned_blas` is serialized by a module lock and refuses to
  run during stream capture (tested).

Serving rule (enforced by convention, documented here as the resolution of
#67's concurrency item): configure **once at startup, single-threaded,
before any capture**, and serialize forward execution against all other BLAS
work in the process. `qkv_tuned_blas_state()` reports the configured winners
and manifest provenance for deployment/benchmark reports. This wrapper
remains serialized-inference-only; concurrent-serving adoption needs the
floe-side scheduler to own these invariants.

## Reproduce

Host (CPU, no GPU required for the mock + oracle suites):

```bash
PYTHONPATH=src/python python -m pytest tests/python/test_qkv_tuned_blas.py tests/python/test_bf16_oracle.py -q
```

Produce + qualify on a beverin `mi300` node (sections 4 → 5b/5c; the tuning
bench writes the manifest automatically):

```bash
VK63_SECTIONS='2. torch_ops* 4. BENCH QKV* 5.*' sbatch meta/scripts/run_issue63_torchops_mi300.sh
# then, against a specific artifact:
PYTHONPATH=src/python python meta/benchmarks/qualify_qkv_tuned_blas.py \
    "$OUT/qkv_projection.tunable0.csv" --mode quality --nll-report nll.json
```

A promotion (flip an opt-in to the frozen path in floe) requires **all**
quality gates green on the promotion artifact, with the NLL gate measured on
the full 45-layer model per #67's evidence plan. The 13.71 tok/s number in
#67 remains performance-only.

## Checked-in example

`meta/benchmarks/artifacts/qkv-tunable-example/` holds the sanitized genuine
job-628932 CSV with a post-hoc generated manifest. It validates against
today's sources and is rejected the moment a fingerprinted source changes —
demonstrating the staleness rule on a real artifact.
