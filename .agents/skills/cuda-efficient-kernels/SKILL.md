---
name: cuda-efficient-kernels
description: Optimise CUDA/PTX kernels with a roofline-guided method — classify a kernel by arithmetic intensity against the ridge point, then attack the single binding resource (HBM traffic when memory-bound, idle time on the compute path when compute-bound). Use when writing or tuning a GPU kernel, deciding whether it is memory- or compute-bound, picking tile sizes, fusing operators, raising arithmetic intensity, overlapping compute with data movement, or reasoning about occupancy versus overlap.
---

# Efficient CUDA Kernels

A kernel is only fast relative to a **roof**: an upper bound set by either memory
bandwidth or compute throughput. Optimisation that does not name which roof is
binding is guesswork. This skill fixes the *process*: classify, then attack the
binding resource, then evolve the kernel one mechanism at a time.

## Step 1 — Classify against the roofline

Compute the kernel's **arithmetic intensity** (AI) in FLOP/byte, at a chosen
memory level (default HBM):

```
AI = compute work / data moved        (FLOP / byte)
attainable <= min(peak compute, bandwidth * AI)
```

- One FP add or mul = 1 FLOP; one fused multiply-add `a*b+c` = 2 FLOPs.
- For square GEMM `C=A@B` with `M=N=K`, ideal AI ≈ `N/3` (A,B read once, C
  written once, `beta=0`). AI grows with the tile because each loaded element
  is reused for many MACs.
- Elementwise ops (GELU) and reductions (RMSNorm) read/write large tensors with
  little work per element → low AI → almost always **memory-bound**.
- Attention sits between: standard attention writes `QK^T` to HBM and reads it
  back, tanking its AI; FlashAttention keeps it on-chip, raising AI.

Find the **ridge point** = `peak compute / bandwidth`. Below it the kernel is
memory-bound; above it, compute-bound. On B200 round numbers (≈2 PFLOP/s
fp16/bf16 Tensor Core, ≈8 TB/s HBM3e) the ridge is ≈250 FLOP/byte: a kernel
needs ~250 FLOPs per HBM byte before the compute roof can bind.

**Completion criterion:** you can state, with numbers, which side of the ridge
point the kernel is on and which resource (HBM bandwidth vs compute throughput)
currently limits it. No further step is justified until this is known.

## Step 2 — Attack the binding resource

### Memory-bound: cut HBM traffic, then push bandwidth

1. **Fuse** the producer of an intermediate with its consumer so the value
   stays in registers / SMEM / TMEM instead of round-tripping HBM. Fuse GEMM
   with an elementwise epilogue; fuse normalization into an adjacent op; compute
   attention without materialising the score matrix.
2. **Tile** so on-chip data is reused many times. A `16×16` tile at `B_K=64`
   fp16 gives AI ≈ 8 FLOP/byte; a `64×64` tile gives ≈ 32. Each A/B byte loaded
   serves more MACs.
3. **Use a smaller dtype** (fp32→fp16/bf16→fp8→fp4). Block-scaled fp8/fp4 needs
   scale factors and conversions, so the real gain is less than the size ratio,
   but it is the most direct way to raise AI.

Once traffic cannot fall further, push **effective bandwidth** to the roof:
move each byte once (no redundant reads); coalesce/vectorise accesses; use TMA
for regular bulk tiles; keep enough requests in flight that the memory pipeline
never idles. Past the memory roof, the only lever is to change the algorithm so
it moves fewer bytes — more compute instructions do nothing.

### Compute-bound: keep the Tensor Cores busy

A compute-bound-by-roofline kernel is *not* automatically at the compute roof.
The remaining gap is **idle time** when one of load / compute / store is waiting
on another. The cure is **overlap**, not more arithmetic:

- A naive kernel runs `load k → wait → compute k → wait → store k` serially,
  leaving every unit idle in turn.
- A pipelined kernel runs `load k+1 · compute k · store k-1` together. TMA
  moves the next tile while the Tensor Core computes the current one while the
  epilogue writes the previous.
- `mbarrier` coordinates the safe handoffs between these stages (see the
  [async-kernel-coordination](../async-kernel-coordination/SKILL.md) skill).

Some changes (e.g. warp specialisation) raise resource use and *lower*
occupancy before they pay off. That is the deliberate trade explored below.

## Step 3 — Occupancy is not a quality metric

SM occupancy is how much work can reside on one SM at once; it hides latency by
running another ready warp when one stalls. It is limited by registers, SMEM,
warp slots, and CTA slots. Modern Tensor Core kernels deliberately spend these:
multi-stage pipelines eat SMEM, large fragments eat registers, TMEM allocations
eat Tensor Memory, warp specialisation reserves whole warps.

These kernels hide latency **within** a few resident CTAs by explicitly
overlapping stages, not by stacking many warps. A low-occupancy kernel is fine
*if the pipeline keeps TMA, the Tensor Cores, and the store path active*. The
real question is always: are the critical hardware units staying busy?

## The ladder

Once the binding resource is named, evolve the kernel **one mechanism at a
time**, keeping every earlier version passing as a correctness reference. The
canonical ladder is GEMM (load the [gemm-kernel-tuning](../gemm-kernel-tuning/SKILL.md)
skill for the concrete steps):

1. thread-copy tiled path → 2. K-loop accumulation → 3. spatial tiling →
4. **TMA async load** (first large measured jump — delegates regular tile
   movement to hardware) → 5. software pipeline → 6. persistent kernel + tile
   scheduler → 7. warp specialisation → 8. 2-CTA cluster → 9. multi-consumer.

Steps 1–4 raise arithmetic intensity / cut copy cost; steps 5–9 reduce waiting
among load, compute, and store. The same ladder builds FlashAttention (two MMAs
with online softmax between them — see gemm-kernel-tuning's
[flash-attention](../gemm-kernel-tuning/references/flash-attention.md) reference).

## When to load a sibling skill

- Authoring or reading PTX — instructions, special registers, directives, memory
  consistency: load [ptx-isa](../ptx-isa/SKILL.md).
- Arranging data in memory — spaces (GMEM/SMEM/TMEM/RF/DSMEM), shape-stride
  layouts, named axes, fixing bank conflicts with swizzling: load
  [gpu-memory-layout](../gpu-memory-layout/SKILL.md).
- The async primitives — TMA, mbarrier, software pipelines, warp
  specialisation, clusters, Cluster Launch Control: load
  [async-kernel-coordination](../async-kernel-coordination/SKILL.md).
- Building or tuning a dense GEMM / matmul or an attention kernel concretely:
  load [gemm-kernel-tuning](../gemm-kernel-tuning/SKILL.md).
- Measuring a kernel reproducibly and judging it against a roof, or comparing
  two versions of one: load [kernel-benchmarking](../kernel-benchmarking/SKILL.md).
- Explaining the gap between a measured number and its roof — choosing
  Nsight Systems vs Nsight Compute, reading utilization and warp stall
  reasons, localizing a cause for each particular kernel: load
  [kernel-profiling](../kernel-profiling/SKILL.md).

## Final completion criterion

Every change you make is justified against the binding roof: for a
memory-bound kernel you can show it moves fewer bytes (or pushes bandwidth
closer to the roof); for a compute-bound kernel you can show it cuts idle time
on a specific unit. You measured rather than assumed, and you did not touch a
resource that is not the bottleneck.
