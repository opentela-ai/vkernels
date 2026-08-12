---
name: hip-kernel-profiling
description: Find the single binding resource and the instruction or region causing it, using omnitrace for the timeline and omniperf/rocprof for a single kernel's stalls and utilization. Use when a HIP kernel is slower than its roof and you need to know why, when you are reading AMD profiling metrics (memory throughput, L2 hit, Matrix Core %, wavefront stall reasons) or a timeline, when wavefront stalls dominate and you need to name the stall, when CPU and GPU are not overlapping, or when profiling each particular kernel (elementwise, reduction, GEMM, ring allreduce, overlap executor) for its characteristic bottleneck.
---

# HIP Kernel Profiling

Profiling **explains the gap** between a measured number and its roof; it does
not produce the number. Open a profiler only after
[hip-kernel-benchmarking](../hip-kernel-benchmarking/SKILL.md) has a
reproducible result and [hip-efficient-kernels](../hip-efficient-kernels/SKILL.md)
has named which resource *should* be binding. A profile taken blind generates
a wall of metrics and tempts you to "fix" a resource that is not the
bottleneck.

## AMD profiling tool landscape

ROCm provides a layered profiling stack. Choose by scope:

| Tool | Scope | Best for | NVIDIA equivalent |
|---|---|---|---|
| **omnitrace** | Whole-program timeline | Host/device overlap, launch gaps, multi-stream | Nsight Systems (`nsys`) |
| **omniperf** | Single-kernel analysis | Compute/memory utilization, stall reasons | Nsight Compute (`ncu`) |
| **rocprof** | Single-kernel raw counters | Low-level hardware counter access | CUPTI raw counters |
| **rocprofv3** | Latest single-kernel | Next-gen profiling (ROCm 6+) | — |
| **ROCProfiler** | Programmatic API | Custom profiling integration | CUPTI |

> **Important:** `rocprof` v1/v2 is the legacy tool; `omniperf` is the
> recommended high-level analysis tool for MI200/MI300 series. `omnitrace`
> provides the timeline view. For quick sanity checks, `rocprof --stats`
> gives basic kernel timing and counter info.

## Step 1 — Confirm the regime before opening a tool

State the hypothesis the profile will confirm or refute: which resource
(Matrix-Core compute, HBM/L2/LDS bandwidth, latency, launch/overlap) should
limit this kernel at this problem size, and why. If you cannot, go back to the
roofline skill — profiling will not tell you.

Completion: a one-sentence hypothesis ("this `64×64` GEMM is memory-bound at
this size; the profile should show HBM near 100% and low Matrix-Core
utilization") that the next step tests.

## Step 2 — Choose the tool by scope

- **omnitrace** — the whole-program **timeline**. Use it first when the problem
  is *between* kernels or across the CPU/GPU boundary: launch gaps, serial host
  work, missing overlap between a compute stream and a comm stream, kernel A's
  tail idling while kernel B launches. It tells you *which* kernel or host
  region to drill into.

  ```
  omnitrace -- omnitrace-instrument -- ./my_app
  ```

- **omniperf** — **one kernel's** resource utilization and stall reasons. Use
  it after `omnitrace` has pointed at a kernel, or immediately when you already
  know a single kernel is the issue. Do not use `omniperf` to diagnose a
  launch-tail or CPU-overhead problem — it profiles in isolation and hides
  exactly the cross-kernel behaviour `omnitrace` shows.

  ```
  omniperf profile -n my_kernel -- ./my_app
  omniperf analyze -p workloads/my_kernel/ &> analysis.txt
  ```

- **rocprof** — raw hardware counter dumps for custom analysis or when
  `omniperf` is not available. Less user-friendly but gives full counter access.

  ```
  rocprof --stats ./my_app          # basic timing
  rocprof --hip-trace ./my_app      # HIP API trace
  rocprof -i counters.txt ./my_app  # custom counter set
  ```

Commands and report structure are in [references/tools.md](references/tools.md).

## Step 3 — Read the binding metric

Read the metric that maps to your roofline hypothesis, not the first headline
on the dashboard:

- **Memory-bound** → `SQ_INSTS_VALU` × memory throughput to derive achieved
  GB/s; **L2 hit rate** (`TCP_TCP_TA_TOTAL_ACCESS`, `TCP_TCP_TA_TOTAL_MISS`);
  **coalescing quality** (look at `TCP_TCP_TA_TOTAL_ACCESS` vs coalesced
  segment size — high `TA_TOTAL_ACCESS` for few bytes means poor coalescing).
- **Compute-bound** → **Matrix Core utilization** (`GRBM_COUNT` vs active
  cycles); **Vector ALU utilization** (`SQ_INSTS_VALU`); and the **wavefront
  stall reasons** sorted by weight. The dominant stall names the resource the
  wavefront is actually waiting on.
- **Latency / overlap** → `omnitrace` timeline gaps, kernel-to-kernel distance,
  and whether two streams actually overlap or run serially.

> **AMD counter naming:** AMD hardware counters are grouped by block — `TCP`
> (L1/TCP cache), `SQ` (sequencer/wavefront dispatch), `GRBM` (graphics
> resource block manager), `CP` (command processor), `TA`/`TD` (texture
> address/data). `omniperf` abstracts these into human-readable derived
> metrics.

Interpret each as a trend and a percentage of roof, not a pass/fail. The full
catalog (throughput, hit rate, coalescing, Matrix Core %, every wavefront stall
reason and what it implies) is in
[references/metrics-and-stalls.md](references/metrics-and-stalls.md).

## Step 4 — Localize to an instruction or region

A metric that says "memory-bound" without a location is a lead, not a fix.
Use `omniperf analyze` with source correlation, then:

- Tie the dominant stall or replay to a **specific line or access pattern**
  — a non-coalesced loop, a column read of a row-major tile, a barrier waited
  at the wrong point.
- Cross-reference the access pattern against the layout you *intended*
  (hip-gpu-memory-layout) and the synchronisation you wrote
  (hip-async-coordination). The most common "slow but correct" bug is an
  intended layout/sync that the generated code does not actually implement.
- For each kernel type in this project, confirm the **characteristic**
  bottleneck before hunting an exotic one — see
  [references/per-kernel.md](references/per-kernel.md).

### Key AMD-specific stall reasons

| omniperf metric | Meaning | Typical cause |
|---|---|---|
| `MemUnitStalled` | Waiting on memory | Poor coalescing, high L2 miss |
| `MemUnitBusy` | Memory unit active | High bandwidth — good if near roof |
| `VALUInsts` | Vector ALU active | Compute work — good if compute-bound |
| `MFMAInsts` | Matrix Core active | Matrix work — the target for GEMM |
| `LDSBankConflict` | LDS bank conflict | Unswizzled column access from LDS |
| `VALUUtilization` | % VALU active | Target > 80% for compute-bound |
| `VALUBusy` | VALU doing work | % of VALU cycles with work |
| `FetchSize` | Bytes fetched per request | Low means poor coalescing |
| `WriteSize` | Bytes written per request | Low means poor coalescing |
| `SQ_WAVE_STALLS` | Wavefront stall | Check sub-reason for detail |

Completion: a named cause — one source line, one access pattern, or one
synchronisation point — that explains the gap. Not "memory is slow."

## Step 5 — Act on one cause, then re-benchmark (not re-profile)

Fix **exactly one** thing the profile named. Then return to the
**benchmarking** harness for the verdict: the profiler is for diagnosis, the
benchmark is for acceptance. Trusting a profiler's before/after is a frequent
failure mode, because profiling overhead distorts timing and a single fix can
move the bottleneck to a different resource.

If the re-benchmark shows the binding resource improved, keep the step and
re-profile to find the *next* binding resource (the bottleneck moves — that is
expected, not a contradiction). If it did not improve, the profile's cause was
wrong or incomplete; re-state the hypothesis before profiling again.

## Final completion criterion

Every profile reading is tied to a specific instruction or region and to the
roofline hypothesis it confirms or refutes; you fixed exactly one named cause
and let the benchmark, not the profiler, decide whether it helped.
