---
name: gemm-kernel-tuning
description: Build a high-performance dense GEMM on NVIDIA Hopper/Blackwell one mechanism at a time, and apply the same ladder to FlashAttention. Use when writing or tuning a matrix-multiply kernel, choosing GEMM tile sizes and accumulator placement, replacing thread copies with TMA, adding a multi-stage software pipeline, warp specialising producers and consumers, forming a two-CTA cooperative cluster, or building an attention kernel from two MMAs and online softmax.
---

# GEMM Kernel Tuning

GEMM is the canonical high-AI kernel: each loaded A/B element participates in
many MACs, so it can reach the Tensor-Core compute roof — but only after the
right instructions, layouts, staging, synchronisation, and scheduling are in
place. The method below evolves a kernel **one mechanism at a time**, keeping
every earlier version passing as a correctness reference. The same ladder
builds FlashAttention (two MMAs with online softmax between them).

Load [cuda-efficient-kernels](../cuda-efficient-kernels/SKILL.md) for the
roofline classification that decides whether a step is worth it, and
[async-kernel-coordination](../async-kernel-coordination/SKILL.md) for the
TMA/mbarrier/cluster APIs each step uses. Accept or reject each step on
[measurement](../kernel-benchmarking/SKILL.md) (the first large jump at Step 4,
L2-locality gains at Step 6, overlap gains at 5/7/8/9), and when a step does
not move as expected, [profile](../kernel-profiling/SKILL.md) it before
blaming the next.

## The ladder

Tile sizes below are the book's running example (`B_M = B_N = 128`, `B_K = 64`,
fp16/bf16 on B200). The point is the mechanism each step adds, not the numbers.

**1. Sequential single-tile.** One CTA computes one `128×128` output tile,
`K = 64` (a single MMA, no K-loop). It fixes the data path for every later
version: GMEM → SMEM by a **thread-driven** copy → `T.cuda.cta_sync()` →
`tcgen05.mma` writes a TMEM accumulator (`accum=False`) → epilogue reads TMEM
back to registers, converts, stores to GMEM. This is the correctness baseline.

**2. K-loop accumulation.** Lift the `K = 64` restriction: iterate K in `B_K`
chunks, reusing **one** SMEM tile pair and **one** TMEM accumulator slot
(`accum=True` after the first). Update the MMA barrier's wait state per
iteration. No new storage; operands stream through fixed buffers.

**3. Spatial tiling (multi-CTA).** Partition the `M×N` output into `128×128`
tiles and launch one CTA per tile on a 2-D grid; `(bx, by)` identifies the
output tile, and each CTA runs the Step 2 K-loop internally. Same per-CTA
path; the change is purely how the grid covers the output.

**4. TMA async load.** Replace the thread-driven GMEM → SMEM copy with
`cp.async.bulk` (TMA) descriptor-driven copies. This is the **first large
measured jump**: a single thread submits each tile, the engine fills SMEM
in the swizzled layout the MMA expects, and the threads the copy used to
occupy are free. Detail in [references/gemm-steps.md](references/gemm-steps.md).

**5. Software pipeline (`PIPE_DEPTH ≥ 2`).** ≥2 SMEM stages with `full`/`empty`
mbarriers each; while the MMA computes tile k, TMA fills tile k+1. Phase
parity per barrier distinguishes consecutive uses. Hides load latency behind
compute — the gap-closer once the kernel is Tensor-Core-bound.

**6. Persistent kernel + tile scheduler.** Launch a fixed pool of long-lived
CTAs; each computes several output tiles in a loop, cutting launch and setup
overhead. The next tile comes from a scheduler (a static formula here; CLC in
a later step).

**7. Warp specialisation.** Dedicate **producer** warps (TMA loads) and
**consumer** warps (MMA) instead of one warp doing both. It may raise resource
use and lower occupancy before it pays, but it creates the structure for the
deep overlap the later steps need.

**8. Two-CTA cluster.** Two CTAs cooperate on one larger MMA via
`cta_group::2` + DSMEM: each CTA holds part of A/B and reads the peer's slice,
so the effective MMA tile doubles. Requires `.reqnctapercluster`.

**9. Multi-consumer warp specialisation.** Multiple consumer warps share a
producer's output for fuller utilisation, reaching near the compute roof
end-to-end.

Steps 1–4 raise arithmetic intensity / cut copy cost; steps 5–9 reduce waiting
among load, compute, and store. Some steps do not improve performance
immediately — warp specialisation may first raise resource use — but they
provide the structure the later steps exploit.

## Why bigger tiles raise AI

Keeping `M, N, K` fixed and changing only the CTA tile, an A/B K-stage moves
`2·(B_M·B_K + B_K·B_N)` bytes for `2·B_M·B_N·B_K` FLOPs, so

```
AI ≈ 2·B_M·B_N / (s·(B_M + B_N))      # s = bytes/element
```

and at `B_M = B_N = B` this is `B/s`. A `16×16` fp16 tile (`s = 2`) gives
AI ≈ 8; a `64×64` tile gives ≈ 32. Each loaded element serves more MACs, so
less HBM traffic buys the same work. The trade is register/TMEM/SMEM pressure
cutting occupancy — acceptable when the pipeline keeps the units busy.

## Accumulator placement

Pre-Blackwell MMA accumulators lived in registers, and as tiles grew they ate
a large share of the register file. Blackwell `tcgen05` writes accumulators to
**Tensor Memory** instead, cutting register pressure; the epilogue must
**explicitly** read them back, with the four warps of a warpgroup each loading
its own 32-lane TMEM window.

## FlashAttention

The ladder's flagship application: rather than materialising the `QK^T` score
matrix to HBM (which tanks AI), keep it on-chip and interleave two MMAs per
outer step with **online softmax** between them.

- `S = QK^T` in TMEM; **online softmax** maintains a per-row running max `m`
  and running sum — when a new tile's max exceeds `m`, rescale the running
  `P·V` accumulator by `exp(m_old − m_new)` before accumulating.
- `O = P·V` accumulated in TMEM/registers; final writeback converts to the
  output dtype (often via SMEM + a TMA store).
- **Causal masking** masks future positions within the `S` tile before softmax
  (predicated or masked MMA).
- **GQA**: Q has more heads than K/V — tile so each Q-head group shares the
  same K/V, loading K/V once per group.

Producer warps load Q/K/V via TMA; consumer warps run the two MMAs with the
softmax between; barriers carry the `QK → softmax → PV` handoffs at the right
phases. Full structure is in
[references/flash-attention.md](references/flash-attention.md).

## Completion criterion

You reach the step the problem actually needs, every earlier version still
passes, and you can name which resource each step addresses: steps 1–4 raise
AI / cut copy cost; steps 5–9 cut idle time on TMA, the Tensor Cores, or the
store path. You did not add a mechanism whose benefit is unmeasured.
