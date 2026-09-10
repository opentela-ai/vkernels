# Device-native torch_ops on beverin MI300A — harness & Python.h fix

`vkernels.torch_ops` (the optional Torch/Triton inference operators that
back the GLM-5.3-Flash / floe roadmap, GitHub **#63** and subissues
**#64–#70**) is validated and benchmarked on CSCS **beverin** `mi300`
(MI300A / gfx942) compute nodes by a single self-contained driver:

```bash
# from a beverin login node, with the repo checked out at $SRC
sbatch meta/scripts/run_issue63_torchops_mi300.sh
# or interactive on one MI300A (≈2 min for the test suite alone):
VK63_SECTIONS='1. ENV* 2. torch_ops*' \
  srun --partition=mi300 -N1 -G1 --time=00:12:00 \
  bash meta/scripts/run_issue63_torchops_mi300.sh
```

The script is self-bootstrapping: it needs no Slurm module, no container,
and no `python3.11-devel` package. It writes one JSON/log artifact per
section into `work/issue63/` and prints a per-section `rc=` summary so a
single failed benchmark cannot hide the rest of a run.

## The gotcha that makes this non-obvious: `Python.h`

The beverin `mi300` compute nodes ship **Python 3.11.13** and a working
**torch 2.9.1+rocm6.3 / triton 3.5.1** stack in `~/.local`, but **no
`python3.11-devel` package** — so `/usr/include/python3.11/Python.h` is
absent. Triton's AMD backend JIT-compiles a host helper
(`hip_utils.c`, see `triton/runtime/backends/amd/driver.py`) with
`gcc -I/usr/include/python3.11`, which then fails:

```
/tmp/.../hip_utils.c:1:10: fatal error: Python.h: No such file or directory
 #include <Python.h>
          ^~~~~~~~~~
```

That one missing header kills **every** GPU test and Triton benchmark
(the CPU contract tests still pass). The historical torch_ops runs cited
in `docs/mhc-projection.md` / `docs/qkv-projection.md` (Torch 2.9.0a0,
ROCm 7.0, Triton 3.4.0) were done inside a container that has since been
removed from `/capstor`, so this harness targets the bare-node stack
directly.

### Fix (in `_bootstrap_pyheaders`)

1. Fetch a [python-build-standalone](https://github.com/indygreg/python-build-standalone)
   CPython 3.11.x x86_64 glibc tarball (`install_only`).
2. Extract its `python/include/python3.11/` next to the checkout
   (`.pybuild/python/include/python3.11`).
3. `export CPATH="${SRC}/.pybuild/python/include/python3.11${CPATH:+:${CPATH}}"`.

`CPATH` is a gcc-standard search path that gcc consults **after** any
explicit `-I` dirs, so the missing `/usr/include/python3.11` is filled in
transparently — **no patching Triton, no root packages, no container**.
The standalone 3.11.16 headers are ABI-compatible with the system 3.11.13
runtime (verified: a `PyModule_Create` helper `.so` built against them
loads and runs under the system interpreter). `.pybuild/` is git-ignored.

If a future node has `python3.11-devel` installed, `_bootstrap_pyheaders`
still runs harmlessly (an extra `CPATH` entry that duplicates the system
one) and the section `1. ENV` fingerprint records the stack so a run is
self-describing.

## What each section produces, and which subissue it backs

| # | Section (label) | Produces | Subissue |
|---|---|---|---|
| 1 | `ENV fingerprint` | torch/triton/ROCm/device props in the log | all (reproducibility) |
| 2 | `torch_ops pytest …` | pass/skip counts in the log | **#65** (upstream API), **#69** (graph/capture/precision contracts) |
| 3 | `BENCH mHC projection` | `work/issue63/mhc_projection.json` + `.tunable0.csv` | **#66** (batch-one DSA input projections — here the small-output mHC GEMM) |
| 4 | `BENCH QKV projection` | `work/issue63/qkv_projection.json` + `.tunable0.csv` | **#68** (KDA gate GEMV — the analogous BF16 M=1/2 projection), **#67** (TunableOp lifecycle) |
| 5 | `CHECK frozen TunableOp QKV preflight` | per-device graph/parity/context log | **#67** (persist tuning; separate reproducibility from quality) |
| 6 | `BENCH roofline` | `work/issue63/roofline.json` (logical BW, BF16 GEMM, launch floor) | **#66 / #68** (the ~0.21 ms/token and ~3.5 TB/s references) |
| 7 | `COUNTER probe qkv` / `… gate128` | JSON summaries (instrumented only — **not** perf numbers) | **#66 / #68** (the "one workgroup / 32× padded FLOPs" diagnosis) |

### Running a subset

`VK63_SECTIONS` is a space-separated list of glob patterns matched
against the section label (prefix match is enough), so you can run a
single benchmark or a fast smoke test without editing the script:

```bash
# just the roofline benchmark
VK63_SECTIONS='6. BENCH roofline*' sbatch meta/scripts/run_issue63_torchops_mi300.sh
# fast correctness-only smoke (no benchmarks)
VK63_SECTIONS='2. torch_ops*' \
  srun --partition=mi300 -N1 -G1 --time=00:12:00 \
  bash meta/scripts/run_issue63_torchops_mi300.sh
```

## Notes & caveats

- This harness measures the **device-native torch_ops** kernels
  (Triton GEMV/split-K projections, TunableOp BLAS). The HIP C-ABI
  DSA/MHC/KDA kernels in `src/python/vkernels/hip_dsa_mhc.py` (issue #69)
  are exercised separately by `meta/scripts/run_dsa_mhc_bench_mi300.sh`
  and the KDA/MoE scripts; they deliberately **refuse graph capture**
  (legacy default stream) — see the [#69 issue body](https://github.com/opentela-ai/vkernels/issues/69)
  and [`docs/performance/{mhc,dsa}/gfx942.md`](performance/mhc/gfx942.md).
- `bench_roofline.py` reports **logical** bytes/GPU-event time, not
  physical HBM counters (the 64 MiB case may be cache-resident). The
  counter probes (section 7) are explicitly instrumented and are **not**
  performance measurements — see their own docstrings.
- Tuning is in-process (cached by device + row count), **not** a portable
  deployment artifact; the frozen TunableOp CSV written beside each bench
  is validated for version/device but retunes in a fresh process. This is
  the gap issue **#67** calls out and is intentionally surfaced, not
  hidden, by the harness.

## Postscript: the #58 grouped-GEMM NaN (found, fixed, A/B-validated)

The real-checkpoint fp8 A/B (#64) produced **NaN NLL on random-token
prompts** with `GLM53_FP8_GROUPED=1` while natural prompts stayed
finite, non-deterministically across runs. Diagnostics
(`floe/docker/beverin/glm5-smoke/diag_fp8_nan{3,4,5,6}.py`, one model
load to capture a bundle, then bundle-only iterations) localized it:

- The NaN was born in the **gate_up grouped GEMM** with verified-clean
  fnuz operands (pure-torch oracle on the same tensors: finite).
- The same inputs through the same kernel in a **fresh process were
  clean** -> the "NaN" was the `torch.empty()` garbage underneath
  **silently-unwritten output rows**, not computed NaN. Allocator
  poison (2 GiB of NaN freed before the call) made 231k NaN values
  survive the kernel: the smoking gun.
- Root cause: the tile map repeated each expert's **segment start and
  full count** across all of its row tiles, so every tile of a
  count > BM (hot) expert rewrote the expert's first 64 sorted slots
  and slots beyond the first 64 were written by no one. Uniform
  synthetic routing never produces a multi-tile expert at BM=64
  (mean ~14 slots/expert even at T=1024), which is why the #64
  synthetic parity benches never caught it; the real model's skewed
  routing at T>=256 created the first ones.

Fix (73bb30d): per-tile `r0 = seg_start + local*BM`, `me = clamped
remainder`. Post-fix verification on the real L3 bundle: **0/16.7M
mismatches** vs the oracle, 5/5 deterministic, poison-clean. Full A/B
re-run: NLL random 13.3988 vs bf16 13.4000 (was NaN), NLL natural
+0.0116, argmax agreement 218/256, prefill **1.76x** (287.9 vs 163.8
tok/s). Regression test:
`tests/python/test_glm_fp8_grouped_multitile.py` (skewed routing,
hot experts, fnuz oracle with the runner's silu-form swiglu — note
`glm_moe_grouped_gemm`'s sigmoid-form wrapper is NOT a valid
reference for the native path).

Follow-up resolution: the fp8 decode gap was protocol artifact — the
A/B pinned GLM53_FUSED_GEMV=0, forcing the fp8 arm through the shared
gather+dequant path. In-process decode A/B (bench_fp8_decode_ab.py):
fused GEMV 60.11 ms/tok vs gather-dequant 82.31 (1.369x), identical
first token (69e3bdf fixes the HIP dispatch gate that crashed the
fused path on MI300A). Remaining known follow-ups:
(b) fnuz conversion cache costs ~9.7 GiB/layer permanent (both fp8
originals and fnuz copies resident), capping how many layers group
under 128 GiB (~5/9 here); (c) the per-layer conversion transient is
~19 GiB fp32 (chunkable).
