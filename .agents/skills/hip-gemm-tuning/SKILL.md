---
name: hip-gemm-tuning
description: Build a high-performance dense GEMM on AMD CDNA one mechanism at a time, and apply the same ladder to FlashAttention. Use when writing or tuning a matrix-multiply kernel on AMD GPUs, choosing GEMM tile sizes and VGPR accumulator placement, replacing thread copies with vectorized LDS transfers, adding a multi-stage LDS pipeline, wavefront-specialising producers and consumers, or building an attention kernel from two MFMAs and online softmax.
---

# HIP GEMM Kernel Tuning

GEMM is the canonical high-AI kernel: each loaded A/B element participates in
many MACs, so it can reach the Matrix-Core compute roof — but only after the
right instructions, layouts, staging, synchronisation, and scheduling are in
place. The method below evolves a kernel **one mechanism at a time**, keeping
every earlier version passing as a correctness reference. The same ladder
builds FlashAttention (two MFMAs with online softmax between them).

Load [hip-efficient-kernels](../hip-efficient-kernels/SKILL.md) for the
roofline classification that decides whether a step is worth it, and
[hip-async-coordination](../hip-async-coordination/SKILL.md) for the LDS
double-buffering and `s_waitcnt` / wavefront specialisation APIs each step
uses. Accept or reject each step on
[measurement](../hip-kernel-benchmarking/SKILL.md) (the first large jump at
Step 4, L2-locality gains at Step 6, overlap gains at 5/7/8), and when a step
does not move as expected, [profile](../hip-kernel-profiling/SKILL.md) it
before blaming the next.

## The ladder

Tile sizes below are running examples (`B_M = B_N = 128`, `B_K = 64`, fp16 on
MI300X). The point is the mechanism each step adds, not the numbers.

**1. Sequential single-tile.** One workgroup computes one `128×128` output
tile, `K = 64` (a single MFMA, no K-loop). It fixes the data path for every
later version: HBM → VGPRs by a **thread-driven** scalar copy → LDS via
`ds_write` → `__syncthreads()` → `v_mfma` writes VGPR accumulators → epilogue
reads accumulators, converts, stores to HBM. This is the correctness baseline.

**2. K-loop accumulation.** Lift the `K = 64` restriction: iterate K in `B_K`
chunks, reusing **one** LDS tile pair and **one** set of VGPR accumulators.
Zero accumulators before the first iteration; subsequent `v_mfma` calls
accumulate automatically. No new storage; operands stream through fixed
buffers.

**3. Spatial tiling (multi-workgroup).** Partition the `M×N` output into
`128×128` tiles and launch one workgroup per tile on a 2-D grid; `(bx, by)`
identifies the output tile, and each workgroup runs the Step 2 K-loop
internally. Same per-workgroup path; the change is purely how the grid covers
the output.

**4. Vectorized LDS copy.** Replace the thread-driven scalar HBM → VGPR copy
with wide vector loads (`global_load_dwordx4` for 16 bytes/thread) followed by
wide LDS writes (`ds_write_b128`). This is the **first large measured jump**: a
single thread moves 16 bytes instead of 4, cutting the copy cost by ~4×. The
threads that did the copy are free sooner.

**5. LDS double-buffering.** ≥2 LDS stages per tile. While the MFMA computes
tile k (consuming LDS stage 0), the copy engine loads tile k+1 into LDS stage
1. Synchronise with `__syncthreads()` per stage, or with fine-grained
`s_waitcnt` for more precise control. Hides load latency behind compute — the
gap-closer once the kernel is Matrix-Core-bound.

**6. Persistent kernel + tile scheduler.** Launch a fixed pool of long-lived
workgroups; each computes several output tiles in a loop, cutting launch and
setup overhead. The next tile comes from a scheduler (grid-stride loop or
work-queue).

**7. Wavefront specialisation.** Dedicate **producer** wavefronts (HBM → LDS
copies) and **consumer** wavefronts (MFMA) instead of one wavefront doing both.
It may raise VGPR/LDS use and lower occupancy before it pays, but it creates
the structure for the deep overlap the later steps need.

**8. Prefetch scheduling.** Issue loads for tile k+2 while computing k and
storing k−1. On AMD this uses `s_waitcnt` with non-zero counts to allow the
next batch of loads to start before the current compute finishes. See
[hip-async-coordination](../hip-async-coordination/SKILL.md) for the detailed
handoff.

Steps 1–4 raise arithmetic intensity / cut copy cost; steps 5–8 reduce waiting
among load, compute, and store. Some steps do not improve performance
immediately — wavefront specialisation may first raise resource use — but they
provide the structure the later steps exploit.

> **AMD vs NVIDIA GEMM ladder:** The AMD ladder has no TMA step (no Tensor
> Memory Accelerator on AMD). Instead, vectorized LDS copy (Step 4) is the
> first big jump. AMD has no `tcgen05`/TMEM — accumulators stay in VGPRs.
> AMD has no warpgroup (4-warp MMA) concept — MFMA occupies a full 64-thread
> wavefront. AMD has no Cluster Launch Control.

## Why bigger tiles raise AI

Keeping `M, N, K` fixed and changing only the workgroup tile, an A/B K-stage
moves `2·(B_M·B_K + B_K·B_N)` bytes for `2·B_M·B_N·B_K` FLOPs, so

```
AI ≈ 2·B_M·B_N / (s·(B_M + B_N))      # s = bytes/element
```

and at `B_M = B_N = B` this is `B/s`. A `16×16` fp16 tile (`s = 2`) gives
AI ≈ 8; a `64×64` tile gives ≈ 32. Each loaded element serves more MACs, so
less HBM traffic buys the same work. The trade is VGPR/LDS pressure cutting
occupancy — acceptable when the pipeline keeps the units busy.

## Accumulator placement (AMD-specific)

AMD MFMA accumulators live in VGPRs. A `v_mfma_f32_16x16x16f16` writes a
`16×16` fp32 accumulator tile into 4 VGPRs. As tiles grow, the accumulator
eats a large share of the VGPR file, cutting occupancy.

- On CDNA2/3 (MI200/MI300): 256 VGPRs available per thread, but 256 VGPRs ×
  64 threads = 16K VGPRs total; the hardware limit per CU is 1536 VGPRs
  (MI300X). A large tile's accumulators + operands + scratch can exhaust this.
- **Strategy:** keep the accumulation in VGPRs during the K-loop; after the
  loop, the epilogue reads, converts, and writes back. Use `rocWMMA` to
  manage accumulator packing automatically.
- **Accumulator zeroing:** unlike NVIDIA's `accum=False`, AMD MFMA always
  accumulates. Zero the VGPRs explicitly before the first K iteration.

## rocWMMA vs raw MFMA vs composable_kernel

Three levels of GEMM implementation on AMD:

| Level | API | Control | Portability | Use when |
|---|---|---|---|---|
| **rocBLAS** | `rocblas_gemm_ex` | Zero | Any ROCm | Deploying, baseline |
| **composable_kernel** | CK C++ templates | Tile/loop config | CDNA GPUs | Production fused ops |
| **rocWMMA** | `rocwmma::fragment` | Register layout, instruction | CDNA2/3 | Research kernels, MFMA access |
| **raw MFMA** | `__asm__ v_mfma_*` | Full control | One target | Studying MFMA, extreme tuning |

This ladder targets the **rocWMMA** level. The composable_kernel library
already implements Steps 1–8 for most configurations; use CK for production
and rocWMMA for education/research.

## FlashAttention on AMD

The ladder's flagship application: rather than materialising the `QK^T` score
matrix to HBM (which tanks AI), keep it on-chip and interleave two MFMAs per
outer step with **online softmax** between them.

- `S = QK^T` in VGPRs/LDS; **online softmax** maintains a per-row running max
  `m` and running sum — when a new tile's max exceeds `m`, rescale the running
  `P·V` accumulator by `exp(m_old − m_new)` before accumulating.
- `O = P·V` accumulated in VGPRs; final writeback converts to the output dtype.
- **Causal masking** masks future positions within the `S` tile before softmax
  (predicated or masked MFMA).
- **GQA**: Q has more heads than K/V — tile so each Q-head group shares the
  same K/V, loading K/V once per group.

Producer wavefronts load Q/K/V into LDS; consumer wavefronts run the two MFMAs
with the softmax between; `__syncthreads()` or fine-grained `s_waitcnt` carry
the `QK → softmax → PV` handoffs.

> **AMD attention note:** AMD's `composable_kernel` library contains a
> production FlashAttention implementation (`ck_tile::fmha`) that serves as
> both a baseline and a reference for the patterns above.

## Completion criterion

You reach the step the problem actually needs, every earlier version still
passes, and you can name which resource each step addresses: steps 1–4 raise
AI / cut copy cost; steps 5–8 cut idle time on LDS, the Matrix Cores, or the
store path. You did not add a mechanism whose benefit is unmeasured.
