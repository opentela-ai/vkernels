# Profiling Tools: Nsight Systems vs Nsight Compute

Choose by **scope**, not familiarity. The most common mistake is using the
single-kernel profiler to diagnose a between-kernels problem (and vice versa).

## Nsight Systems (`nsys`) — the whole-program timeline

Use **first** when the problem is not obviously inside one kernel, or when it
crosses the CPU/GPU boundary:

- **Launch gaps** — time between kernel end and the next kernel start that no
  host work explains (driver launch latency, a missing stream, an early sync).
- **CPU/GPU overlap** — host code running during GPU execution (good) or
  blocking on it serially (bad). A `cudaDeviceSynchronize` on the hot path
  shows up as a long host wait with the GPU idle afterward.
- **Stream overlap** — the project's `OverlapExecutor` and any multi-stream
  comm should show compute and comm kernels overlapping on separate rows. If
  they are on one row or never overlap, the dependency or the stream choice
  is wrong.
- **Which kernel to drill into** — the longest, or the one whose tail does not
  overlap the next launch.

```bash
nsys profile --stats=true --gpu-metrics-device=all \
  --trace=cuda,nvtx,osrt --output=baseline ./bench
# then: nsys-ui baseline.nsys-rep   (or `nsys stats` for a text summary)
```

Wrap a region in **NVTX** (`nvtxRangePushA/Pop`) so the timeline shows exactly
the timed window and you can correlate host and GPU rows without guessing.

## Nsight Compute (`ncu`) — one kernel's resources and stalls

Use **after** `nsys` points at a kernel, or immediately when you already know
a single kernel is the issue. `ncu` replays the kernel in isolation with
hardware counters, so it shows resource utilization and stall reasons in
detail but **hides cross-kernel behaviour** — do not use it for launch-tail,
overlap, or host-overhead problems.

```bash
# Full collection, source attribution, on the one kernel you care about:
ncu --set full --import-source yes \
  --kernel-name regex:my_kernel \
  --launch-skip 5 --launch-count 20 \
  --target-processes all -o prof ./bench
# then: ncu-ui prof.ncu-rep
```

- `--launch-skip` past warmup so you profile steady state, not the first
  launch.
- `--kernel-name` (regex) profiles one kernel; profiling every kernel in a
  bench run is slow and noisy.
- `--set full` collects everything (slow, one kernel at a time);
  `--set basic`/`--set roofline` are lighter and often enough first.

### The three report sections that matter

1. **GPU Speed Of Light** — throughput as % of peak for compute and memory,
   and the **memory chart** that shows which level (HBM/L2/SMEM) is saturated.
   This is your roofline hypothesis, measured.
2. **Warp State Statistics** — the **stall reasons** sorted by weight. The
   top reason is what warps actually wait on; see
   [metrics-and-stalls.md](metrics-and-stalls.md).
3. **Source** — per-line metrics (memory throughput, branch, divergence,
  instruction issue) attributed to the line that generated them. This is
  where a stall becomes a location.

## What each tool cannot do

- `ncu` **distorts timing** (replays + counter collection) — never use its
  times as a benchmark. Use [kernel-benchmarking](../../kernel-benchmarking/SKILL.md)
  for the verdict.
- `nsys` shows *that* two kernels do not overlap; it does not show *why* a
  single kernel is slow. Hand off from `nsys` to `ncu` when the bottleneck is
  inside one kernel.
- Neither tool reads your intent. A profile that says "memory-bound" with a
  low L2 hit rate contradicts a tiling claim only if you remember you made
  the claim — always cross-reference against the layout and sync skills.
