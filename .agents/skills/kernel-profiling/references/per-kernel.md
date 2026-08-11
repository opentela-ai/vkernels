# Profiling Each Particular Kernel

For each kernel in this project: the regime the roofline skill predicts, the
single metric that confirms or refutes it, the tool to reach for, and the
common cause. Profile the characteristic bottleneck first; an exotic cause
that does not move the binding resource is a distraction.

## Elementwise — `add`, `scale`, `relu`

- **Regime:** memory-bound, always (read 1–2, write 1, ~1 op/element).
- **Confirming metric:** `dram__bytes_*` per second as % of the HBM roof. A
  naive launch is 50–70%; expect 80–90%+ after vectorizing/coalescing.
- **Tool:** `ncu --set basic --kernel-name regex:add|scale|relu`. A kernel
  this short needs no `nsys` pass unless you suspect launch overhead at small
  problem sizes.
- **Characteristic stalls:** `long_scoreboard` (loads not coming back fast
  enough). If it dominates with sub-roof bandwidth, the cause is **coalescing
  or too few bytes in flight**, not compute.
- **Common causes, in order:**
  1. Scalar (1-byte/4-byte) loop instead of vectorized loads — `ncu` shows
     high `lsu` instruction count per byte.
  2. Strided/non-contiguous access (wrong leading dimension) → high replay,
     low L2 hit.
  3. Too few iterations in flight (one launch, small problem) — bandwidth
     never saturates; this is a benchmark-harness issue, not a kernel one.
  4. `relu` divergent branch on mixed sign — a predicated `setp`/`selp` beats
     a branch; divergence shows as `stall_drain`/`not_selected` spikes.

## Reduction — `sum`, `max`

- **Regime:** memory-bound; the tree adds latency, not bandwidth.
- **Confirming metric:** effective GB/s = `bytes_in / time` (one scalar write).
  A single-pass reduction should approach the HBM roof.
- **Tool:** `ncu --set basic --kernel-name regex:sum|max`.
- **Characteristic stalls:** `long_scoreboard` early (filling the partial
  array), then `not_selected`/`short_scoreboard` in the tree as lanes run out
  of work.
- **Common causes:**
  1. **Two-pass** (write partials to GMEM, read them back) → ~2× off the roof;
     fold the first reduction step into shared memory with a warp-level
     `redux.sync`/`shfl`.
  2. **Bank conflicts** in the shared-memory reduction →
     `l1tex__data_bank_conflicts_*`; sequence the writes or pad.
  3. **Divergence** in the tree (unmasked halves) — use predicates or
     `shfl.down` so only the surviving lanes issue.
  4. For `max`, ensure the identity is `-FLT_MAX`, not `0` — a wrong identity
     is a correctness bug the CPU reference catches first (it is **not** a
     profiling finding).

## GEMM — `gemm` (fp32 here; the full ladder in gemm-kernel-tuning)

- **Regime:** compute-bound for large `M,N,K` (AI ≈ N/3 at fp16); memory-bound
  when tiles are small. For the **fp32** path the roof is the **CUDA-core**
  peak, not the Tensor-Core roof — grade accordingly.
- **Confirming metric:** `sm__pipe_tensor_cycles_active…pct` (Tensor Core) for
  a `.cu` path; `sm__inst_executed…pct` for the fp32 path. Cross-check
  `dram__bytes_*` to see whether memory or compute is actually binding.
- **Tool:** `ncu --set full --kernel-name regex:gemm` (you need source
  attribution and the full stall set). For a launch-tail or "later CTAs run
  alone" symptom, use `nsys` first — the kernel is fine, the scheduler isn't.
- **Characteristic profile by ladder step** (see
  [gemm-kernel-tuning](../../gemm-kernel-tuning/SKILL.md)):
  - **Steps 1–3** (thread-copy): memory-bound, low Tensor-Core %, high
    `long_scoreboard`. Expected and not a bug — they are baselines.
  - **Step 4 (TMA)**: the **first large jump**. If Tensor-Core % does not rise
    here, TMA is not overlapping — profile the barrier (the wait is
    immediately after the load, not deferred).
  - **Steps 5–6** (pipeline, persistent): `stall_wait` should fall as load
    overlaps compute. If it does not, a `full`/`empty` phase is wrong (load
    [async-kernel-coordination](../../async-kernel-coordination/SKILL.md)).
  - **Steps 7–9** (warp spec, clusters): `sm__warps_active` **falls** (by
    design — latency is hidden within few resident CTAs, not by stacking
    warps). Low occupancy here is correct; judge by Tensor-Core %, not
    occupancy.
- **Common causes:** non-coalesced operand copy (pre-TMA); a barrier waited at
  the wrong phase; accumulator read-back before the `tcgen05` fence; a
  scheduler that loses L2 locality (re-fetching tiles the last CTA had).

## Ring allreduce — `comm::ring_allreduce`

- **Regime:** bandwidth-bound asymptotically (2·(world−1) transfers/element),
  latency-bound for tiny payloads. Never report one number across sizes.
- **Confirming metric:** `payload / wall_time` (bandwidth) for large sizes;
  end-to-end latency for small. The theoretical ring bandwidth is
  `payload / (2·(world−1)·hop_latency + serialisation)`.
- **Tool:** `nsys` first (multi-thread host launches + the per-rank channels
  overlap or don't), then `ncu` on the per-rank kernel if a single rank is
  the outlier.
- **Characteristic profile:** host threads should show all ranks in flight
  during the scatter and again during the gather. If ranks run one-at-a-time,
  the channel (blocking queue) is serializing — the `MockChannel`/queue
  depth, not the arithmetic, is the cause.
- **Common causes:**
  1. **Channel serialization** — a blocking queue with depth 1 forces a full
     round-trip per step; raise depth or use a true async channel.
  2. **Copy, not overlap** — each rank copies its chunk then waits; the
     gather/scatter should overlap across ranks. If `nsys` shows no overlap,
     the dependency is too eager (waiting before issuing the next send).
  3. **Stride not divisible by world** is a correctness check (the tests cover
     it), not a profiling finding.

## Overlap executor — `comm::OverlapExecutor`

- **Regime:** neither compute- nor memory-bound — its metric is **overlap
  efficiency** = `(T_compute + T_comm − T_wall) / T_wall` (0 = serial, toward
  1 = perfect). Per-stream timing via the executor's two `Stream`s.
- **Confirming metric:** overlap fraction per iteration and across the run.
- **Tool:** `nsys` — the two streams should occupy distinct rows with the
  compute(i+1) region overlapping comm(i). `ncu` is only useful if one stream
  is clearly the outlier and you need its resource utilization.
- **Characteristic profile:**
  - overlap ≈ 1 when `compute` and `comm` costs are similar and the only
    ordering is the per-iteration future (comm needs compute's output).
  - overlap ≈ 0 when the host awaits the future **before** issuing the next
    compute — the dependency is honoured too early and everything serializes.
- **Common causes:**
  1. **Early await** — `comm(i)`'s result is awaited before `compute(i+1)`
     starts. Issue compute(i+1) first, await only what comm(i) actually needs.
  2. **Not two distinct streams** — `uses_two_streams()` is the contract;
     if both land on one stream, there is no overlap to be had.
  3. **Compute far shorter than comm** (or vice versa) — overlap is bounded
     by the longer; report the ratio alongside the overlap fraction so a
     reader is not misled.

## General rule across all kernels

Profile the **characteristic** bottleneck first (coalescing for elementwise,
single-pass for reduction, Tensor-Core % + scheduling for GEMM, channel
overlap for comm). Only when the characteristic cause is ruled out — the
matching metric is healthy and the kernel is still off its roof — hunt the
exotic one. A profile that names a non-binding resource is the most common
way to "fix" something that does not matter.
