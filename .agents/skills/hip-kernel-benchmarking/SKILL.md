---
name: hip-kernel-benchmarking
description: Measure an AMD GPU (or host CPU-reference) kernel reproducibly and judge the number against a roof. Use when asking how fast a HIP kernel is, whether a number is good, setting a baseline, comparing two versions of a kernel, computing FLOP/s or GB/s or effective bandwidth, picking warmup and iteration counts, cutting measurement variance, locking clocks, or deciding whether an optimisation step actually helped the binding resource.
---

# HIP Kernel Benchmarking

A timing number with no roof is noise: 800 TFLOP/s means one thing on an MI300X
that peaks at ≈1.3 PFLOP/s and another on an MI210 that peaks at ≈181 TFLOP/s.
Benchmarking is the discipline that turns a wall-clock sample into a
**reproducible percentage of a named roof**, so a later optimisation step can be
accepted or rejected on evidence rather than feel. Profiling (load
[hip-kernel-profiling](../hip-kernel-profiling/SKILL.md)) *explains* the gap;
this skill establishes the number and guards every version change.

Load [hip-efficient-kernels](../hip-efficient-kernels/SKILL.md) first: its
roofline classification tells you *which* metric to measure here.

## Step 1 — Name the metric and the roof

Pick the metric the kernel's expected regime dictates, and the memory level
the roof is measured at:

- **Compute-bound** (large GEMM, attention) → FLOP/s vs the peak Matrix-Core
  (or Vector-ALU) throughput. Compute work `2·M·N·K` for GEMM, `2·FLOPs` per
  fused multiply-add.
- **Memory-bound** (elementwise, reduction, most comm) → GB/s vs the HBM
  bandwidth (or L2 / LDS if that is the level you stage through). Count the
  **bytes actually moved**, including C writes and any `beta≠0` C read.
- **Communication** (ring allreduce, overlap executor) → bandwidth per link
  and/or latency; for a ring, the asymptotic bandwidth is `(payload) /
  (2·(world−1)·hop_latency + serialisation)`.
- **Latency-bound** (tiny launches, a single allreduce of a few bytes) →
  end-to-end latency, and report separately from any bandwidth run.

State the device's **peak** for that metric and the **ridge point**
(`peak_compute / bandwidth`). Completion: you can write the metric formula,
the peak number, and which side of the ridge the kernel should sit on, *before
measuring*. See [references/metrics-and-roofs.md](references/metrics-and-roofs.md)
for per-kernel target numbers and AMD device reference points.

## Step 2 — Build a reproducible harness

A benchmark that swings 20% between runs optimises nothing. The harness
exists to make the sample a stable estimate of steady-state performance.

- **Warmup, then time.** Run a throwaway batch first so caches, page tables,
  driver paths, and the kernel's persistent state are hot. Never include the
  first launch in the timed window.
- **Iteration count driven by variance, not a fixed N.** Time enough timed
  iterations that the per-call runtime is large relative to the timer's
  resolution (≥100× the timer granularity is a safe floor) and that the
  **coefficient of variation is small** — keep doubling until `std/mean`
  settles (commonly <1–2%). A fixed 100 iterations is a guess, not a method.
- **Use the right clock.** For GPU kernels, time with **HIP events**
  (`hipEventRecord`/`hipEventElapsedTime` around the timed window, with a
  `hipDeviceSynchronize` before reading) — host `std::chrono::steady_clock`
  around a launch measures launch + sync overhead, not the kernel. For host
  CPU-reference kernels and for host-side overlap timing, use
  `std::chrono::steady_clock`.
- **Lock the environment.** AMD GPU: fix clocks via `rocm-smi --setperflevel`
  or `rocm-smi --setsclk`/`--setmclk` to cut thermal swing; CPU: pin
  frequency, disable turbo, set the governor. Run isolated from other
  workloads.
- **Respect data lifetime.** For correctness comparisons the host CPU
  reference and the HIP kernel must run on the same inputs; do not let an
  earlier timed iteration mutate the input unless the kernel is intentionally
  in-place.
- **AMD-specific clock tools:**
  - `rocm-smi --showclocks` — view current GPU clocks
  - `rocm-smi --setsclk <level>` — set SCLK (core clock)
  - `rocm-smi --setmclk <level>` — set MCLK (memory clock)
  - `rocm-smi --setfan <speed>` — fix fan speed
  - `rocm-smi --setperflevel high` — lock to high performance

Full mechanics are in [references/harness.md](references/harness.md).

## Step 3 — Record the baseline

Capture not one number but a distribution: **min, median, mean, std** (min for
"best achievable", median for "typical", std for reproducibility). Express the
result as a **percentage of the roof** (`achieved / peak`). Save the baseline
(number + harness config + device + problem size) so every later step compares
against the same reference.

Completion: the coefficient of variation is small, the result is bounded by a
stated roof, and the baseline is recorded in a form a future run can diff.

## Step 4 — Compare every version under the identical harness

The GEMM ladder in [hip-gemm-tuning](../hip-gemm-tuning/SKILL.md) is built
so each step keeps the earlier kernel passing as a reference; the **only**
way to know a step helped is to measure both under the same harness, same
device, same problem size, same environment.

- A step is **accepted** if it measurably improves the binding resource
  (higher % of roof, or the same % at a larger problem where the prior
  version fell off).
- A step is **rejected** if it regresses the binding resource, *even if* it
  looks cleaner. (Some steps, like wavefront specialisation, may temporarily
  regress before a later step exploits their structure — measure across the
  step group, not in isolation, and only if the ladder's design predicts it.)
- A step that improves a **non-binding** resource is not progress: per the
  roofline skill, optimising a resource that is not the bottleneck does not
  raise the achievable number.

For this project, add one benchmark target per kernel that links the host
library (always built) and, when `VKERNELS_HAS_HIP`, the HIP path; drive it
with the same `--preset hip` / `--preset host` split the tests use, so a
result is always reported against the right roof for the build.

## Final completion criterion

Every reported number is reproducible (low CV), bounded by a stated roof, and
expressed as a percentage of it; every version change is justified by a
measured delta on the **binding** resource under an identical harness. You
measured the verdict, not the profile.
