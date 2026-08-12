---
name: hip-efficient-kernels
description: Optimise HIP/AMDGPU kernels with a roofline-guided method — classify a kernel by arithmetic intensity against the ridge point, then attack the single binding resource (HBM traffic when memory-bound, idle time on the compute path when compute-bound). Use when writing or tuning an AMD GPU kernel, deciding whether it is memory- or compute-bound, picking tile sizes, fusing operators, raising arithmetic intensity, overlapping compute with data movement, or reasoning about occupancy versus overlap on AMD CDNA/GFX architectures.
---

# Efficient HIP Kernels

A kernel is only fast relative to a **roof**: an upper bound set by either memory
bandwidth or compute throughput. Optimisation that does not name which roof is
binding is guesswork. This skill fixes the *process*: classify, then attack the
binding resource, then evolve the kernel one mechanism at a time — adapted for
AMD CDNA architectures (MI300X, MI250X, MI210) and the HIP programming model.

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
- Attention sits between: standard attention writes `QK^T` to HBM and reads
  it back, tanking its AI; FlashAttention keeps it on-chip, raising AI.

Find the **ridge point** = `peak compute / bandwidth`. Below it the kernel is
memory-bound; above it, compute-bound. On MI300X round numbers (≈1.3 PFLOP/s
fp16 Matrix Core, ≈5.3 TB/s HBM3) the ridge is ≈245 FLOP/byte; on MI250X
(≈383 TFLOP/s fp16, ≈3.2 TB/s HBM2e) the ridge is ≈120 FLOP/byte.

### AMD CDNA device reference points (approximate, sanity only)

| Device | fp16 Matrix Core | fp32 Vector | HBM | Ridge (FLOP/B) |
|---|---|---|---|---|
| MI300X | ≈1300 TFLOP/s | ≈81 TFLOP/s | ≈5.3 TB/s | ≈245 |
| MI250X | ≈383 TFLOP/s | ≈48 TFLOP/s | ≈3.2 TB/s | ≈120 |
| MI210 | ≈181 TFLOP/s | ≈45 TFLOP/s | ≈1.6 TB/s | ≈113 |

> **AMD vs NVIDIA terminology:** AMD "Matrix Cores" = NVIDIA "Tensor Cores."
> AMD "Vector ALUs" ≈ NVIDIA "CUDA Cores." AMD "Compute Unit (CU)" ≈ NVIDIA
> "Streaming Multiprocessor (SM)." AMD "wavefront" (64 threads) ≈ NVIDIA
> "warp" (32 threads). An AMD CU runs 4 wavefronts simultaneously.

**Completion criterion:** you can state, with numbers, which side of the ridge
point the kernel is on and which resource (HBM bandwidth vs compute throughput)
currently limits it. No further step is justified until this is known.

## Step 2 — Attack the binding resource

### Memory-bound: cut HBM traffic, then push bandwidth

1. **Fuse** the producer of an intermediate with its consumer so the value
   stays in registers / LDS (Local Data Share = shared memory) instead of
   round-tripping HBM. Fuse GEMM with an elementwise epilogue; fuse
   normalization into an adjacent op; compute attention without materialising
   the score matrix.
2. **Tile** so on-chip data is reused many times. A `16×16` tile at `B_K=64`
   fp16 gives AI ≈ 8 FLOP/byte; a `64×64` tile gives ≈ 32. Each A/B byte
   loaded serves more MACs.
3. **Use a smaller dtype** (fp32→fp16/bf16→fp8). AMD MI300X supports fp8 with
   block scaling via the rocWMMA library and the `mfma` instructions for older
   types. Block-scaled fp8 needs scale factors and conversions, so the real
   gain is less than the size ratio, but it is the most direct way to raise AI.

Once traffic cannot fall further, push **effective bandwidth** to the roof:
move each byte once (no redundant reads); coalesce/vectorise accesses (AMD
wavefronts are 64 threads, so 256-byte aligned accesses are ideal); keep
enough requests in flight that the memory pipeline never idles. Past the
memory roof, the only lever is to change the algorithm so it moves fewer bytes
— more compute instructions do nothing.

### Compute-bound: keep the Matrix Cores busy

A compute-bound-by-roofline kernel is *not* automatically at the compute roof.
The remaining gap is **idle time** when one of load / compute / store is waiting
on another. The cure is **overlap**, not more arithmetic:

- A naive kernel runs `load k → wait → compute k → wait → store k` serially,
  leaving every unit idle in turn.
- A pipelined kernel runs `load k+1 · compute k · store k-1` together. On
  AMD, explicit async copy engines and LDS double-buffering provide the
  mechanism.
- Use `__builtin_amdgcn_s_waitcnt` and LDS `__syncwarp`-equivalent barriers
  (or `__threadfence_block`) to coordinate handoffs between stages.

Some changes (e.g. LDS double-buffering) raise resource use and *lower*
occupancy before they pay off. That is the deliberate trade explored below.

## Step 3 — Occupancy is not a quality metric

CU occupancy is how many wavefronts can reside on one CU at once; it hides
latency by running another ready wavefront when one stalls. It is limited by
VGPRs (vector registers), LDS, wavefront slots, and workgroup slots.

AMD CUs are designed for high occupancy — MI300X supports 32 wavefronts per CU
(8 workgroups of 4 wavefronts each). But high-performance Matrix-Core kernels
deliberately spend these resources: multi-stage LDS pipelines eat shared memory,
large fragments eat VGPRs.

These kernels hide latency **within** a few resident workgroups by explicitly
overlapping stages, not by stacking many wavefronts. A low-occupancy kernel is
fine *if the pipeline keeps the Matrix Cores and the memory path active*. The
real question is always: are the critical hardware units staying busy?

## The ladder

Once the binding resource is named, evolve the kernel **one mechanism at a
time**, keeping every earlier version passing as a correctness reference:

1. thread-copy tiled path → 2. K-loop accumulation → 3. spatial tiling →
4. **Vectorised LDS copy** (use `ds_write_b128`/`ds_read_b128` — first large
   measured jump — delegates regular tile movement to hardware cache) →
5. software pipeline (LDS double-buffering) → 6. persistent kernel + tile
   scheduler → 7. wavefront specialisation → 8. pre-fetch with async copy.

Steps 1–4 raise arithmetic intensity / cut copy cost; steps 5–8 reduce waiting
among load, compute, and store. The same ladder builds FlashAttention (two MMAs
with online softmax between them).

> **Key AMD difference from NVIDIA:** AMD has no TMA (Tensor Memory
> Accelerator). Instead, use LDS as a manual scratchpad with wide vector
> loads/stores. AMD's matrix instructions (`mfma` on CDNA2, `mfma` on CDNA3)
> operate on VGPR accumulators, not a separate TMEM space. On MI300X, the
> `rocWMMA` library provides a higher-level abstraction over these instructions,
> and `composable_kernel` (CK) provides fused collectives.

## When to load a sibling skill

- Authoring or reading AMDGPU assembly — instructions (`s_*`, `v_*`, `ds_*`,
  `flat_*`, `global_*`), special registers (`vcc`, `scc`, `exec`), wait
  counters: load [amdgpu-isa](../amdgpu-isa/SKILL.md).
- Arranging data in LDS — shape-stride layouts, fixing bank conflicts with
  swizzling on AMD's 32-bank LDS: load
  [hip-gpu-memory-layout](../hip-gpu-memory-layout/SKILL.md).
- The async primitives — LDS double-buffering, `__builtin_amdgcn_s_waitcnt`,
  wavefront specialisation, persistent kernels: load
  [hip-async-coordination](../hip-async-coordination/SKILL.md).
- Building or tuning a dense GEMM / matmul on AMD with rocWMMA / composable_kernel:
  load [hip-gemm-tuning](../hip-gemm-tuning/SKILL.md).
- Measuring a kernel reproducibly on AMD hardware and judging it against a
  roof: load [hip-kernel-benchmarking](../hip-kernel-benchmarking/SKILL.md).
- Explaining the gap between a measured number and its roof — choosing
  omniperf vs omnitrace vs rocprof, reading utilization and wavefront stall
  reasons: load [hip-kernel-profiling](../hip-kernel-profiling/SKILL.md).

## Final completion criterion

Every change you make is justified against the binding roof: for a
memory-bound kernel you can show it moves fewer bytes (or pushes bandwidth
closer to the roof); for a compute-bound kernel you can show it cuts idle time
on a specific unit. You measured rather than assumed, and you did not touch a
resource that is not the bottleneck.
