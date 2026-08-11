---
name: kernel-profiling
description: Find the single binding resource and the instruction or region causing it, using Nsight Systems for the timeline and Nsight Compute for a single kernel's stalls and utilization. Use when a kernel is slower than its roof and you need to know why, when you are reading Nsight Compute metrics (memory throughput, L2 hit, Tensor Core %, warp stall reasons) or an Nsight Systems timeline, when warp stalls dominate and you need to name the stall, when CPU and GPU are not overlapping, or when profiling each particular kernel (elementwise, reduction, GEMM, ring allreduce, overlap executor) for its characteristic bottleneck.
---

# Kernel Profiling

Profiling **explains the gap** between a measured number and its roof; it does
not produce the number. Open a profiler only after [kernel-benchmarking](../kernel-benchmarking/SKILL.md)
has a reproducible result and [cuda-efficient-kernels](../cuda-efficient-kernels/SKILL.md)
has named which resource *should* be binding. A profile taken blind generates
a wall of metrics and tempts you to "fix" a resource that is not the
bottleneck.

## Step 1 — Confirm the regime before opening a tool

State the hypothesis the profile will confirm or refute: which resource
(Tensor-Core compute, HBM/L2/SMEM bandwidth, latency, launch/overlap) should
limit this kernel at this problem size, and why. If you cannot, go back to the
roofline skill — profiling will not tell you.

Completion: a one-sentence hypothesis ("this `64×64` GEMM is memory-bound at
this size; the profile should show HBM near 100% and low Tensor-Core
utilization") that the next step tests.

## Step 2 — Choose the tool by scope

- **Nsight Systems (`nsys`)** — the whole-program **timeline**. Use it first
  when the problem is *between* kernels or across the CPU/GPU boundary: launch
  gaps, serial host work, missing overlap between a compute stream and a comm
  stream, kernel A's tail idling while kernel B launches. It tells you *which*
  kernel or host region to drill into.
- **Nsight Compute (`ncu`)** — **one kernel's** resource utilization and stall
  reasons. Use it after `nsys` has pointed at a kernel, or immediately when
  you already know a single kernel is the issue. Do not use `ncu` to diagnose
  a launch-tail or CPU-overhead problem — it profiles in isolation and hides
  exactly the cross-kernel behaviour `nsys` shows.

Commands and report structure are in [references/tools.md](references/tools.md).

## Step 3 — Read the binding metric

Read the metric that maps to your roofline hypothesis, not the first headline
on the dashboard:

- **Memory-bound** → `gpc__cycles_elapsed.avg` × memory throughput to derive
  achieved GB/s; **L2 hit rate** (a low hit rate with high HBM demand
  contradicts a tiling claim); **replay / coalescing** (high replay means
  accesses aren't coalescing — cross-check against
  [gpu-memory-layout](../gpu-memory-layout/SKILL.md)).
- **Compute-bound** → **Tensor Core utilization** (`sm__pipe_tensor_cycles_active`),
  pipe utilization, and the **warp stall reasons** sorted by weight. The
  dominant stall names the resource the warp is actually waiting on.
- **Latency / overlap** → `nsys` timeline gaps, kernel-to-kernel distance, and
  whether two streams actually overlap or run serially.

Interpret each as a trend and a percentage of roof, not a pass/fail. The full
catalog (throughput, hit rate, coalescing, Tensor Core %, every warp stall
reason and what it implies) is in [references/metrics-and-stalls.md](references/metrics-and-stalls.md).

## Step 4 — Localize to an instruction or region

A metric that says "memory-bound" without a location is a lead, not a fix.
Use `ncu --set full --import-source yes` to attribute to source lines, then:

- Tie the dominant stall or replay to a **specific line or access pattern**
  — a non-coalesced loop, a column read of a row-major tile, a barrier waited
  at the wrong phase.
- Cross-reference the access pattern against the layout you *intended*
  (gpu-memory-layout) and the synchronisation you wrote
  (async-kernel-coordination). The most common "slow but correct" bug is an
  intended layout/sync that the generated code does not actually implement.
- For each kernel type in this project, confirm the **characteristic**
  bottleneck before hunting an exotic one — see
  [references/per-kernel.md](references/per-kernel.md).

Completion: a named cause — one source line, one access pattern, or one
synchronisation point — that explains the gap. Not "memory is slow."

## Step 5 — Act on one cause, then re-benchmark (not re-profile)

Fix **exactly one** thing the profile named. Then return to the
**benchmarking** harness for the verdict: the profiler is for diagnosis, the
benchmark is for acceptance. Trusting a profiler's before/after is a frequent
failure mode, because `ncu` overhead distorts timing and a single fix can
move the bottleneck to a different resource.

If the re-benchmark shows the binding resource improved, keep the step and
re-profile to find the *next* binding resource (the bottleneck moves — that is
expected, not a contradiction). If it did not improve, the profile's cause was
wrong or incomplete; re-state the hypothesis before profiling again.

## Final completion criterion

Every profile reading is tied to a specific instruction or region and to the
roofline hypothesis it confirms or refutes; you fixed exactly one named cause
and let the benchmark, not the profiler, decide whether it helped.
