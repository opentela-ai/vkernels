# moe-fused — MXFP4 fused-MoE grouped GEMM (AMD gfx90a, MI250X)

An end-to-end Mixture-of-Experts layer that wires vkernels' four low-level
MXFP4 primitives into the xkernels `fused_moe_mxfp4` interface: fp4→bf16
dequantization, the gate+up grouped GEMM, SwiGLU activation, and the down
grouped GEMM with a routed scatter-add combine — fused into **two kernel
launches** (plus a tiny routing-weight gather). Weights stay in packed
MXFP4 E2M1 (2 values/byte) with ue8m0 per-group scales; no full bf16
materialization of the ~138 GB expert weight buffer.

Two tile configs are compiled in, selected by `block_size` (16 = decode,
64 = prefill). The decode config targets small-batch autoregressive decode
(BM=16, BN=64, 64 threads); the prefill config targets prompt processing
(BM=64, BN=64, 256 threads) and reuses each dequantized B tile across 64
rows instead of 16.

## Kernel (pseudocode)

```cpp
// Stage 0 — gate_up + SwiGLU (one bf16 rounding, matches xkernels oracle)
//   act[EM, ispp] = silu(clamp(A_sorted @ w13_gate + b_gate, L))
//                 · clamp(A_sorted @ w13_up   + b_up,   L)
// Stage 1 — down + routed combine (caller zero-initialises out)
//   out[M, hidden] += act @ w2^T · topk_w_sorted + b2     (atomicAdd per token)
//
// Dequant is inline in the K-loop; MFMA via the clang builtin
// __builtin_amdgcn_mfma_f32_16x16x16bf16_1k (16x16x16 tiles, K=64 per loop).
// Decode:   grid=(EM/16, N/64, 4), 64 threads; blockIdx.z splits N into 4×16.
// Prefill:  grid=(EM/64, N/64),  256 threads (4 wavefronts × 4 col fragments).
```

Full source: `src/c/vkernels/kernels/moe_fused{.hpp,.cpp,.hip}`,
`moe_device.hip` (E2M1/ue8m0 decode, bf16 rounding). The correctness oracle
is `fused_moe_mxfp4_cpu` (block-for-block mirror of the GPU path).

## What was implemented

- **Two fused kernels** (`gateup_swiglu_kernel[_prefill]`,
  `down_combine_kernel[_prefill]`): gate and up accumulate in fp32 in the
  same kernel and the SwiGLU product is rounded to bf16 once, matching the
  xkernels 2-stage oracle exactly (a split 3-kernel pipeline introduced a
  spurious intermediate bf16 rounding). SwiGLU gate is clamped to `L`
  *before* the sigmoid; up is clamped symmetrically to ±L.
- **Inline MXFP4 dequant**: E2M1 nibble → fp32 (LUT in `__constant__`),
  × ue8m0 scale (`s<<23` bit trick), → bf16, written straight to LDS. The
  gate and up weights share one LDS buffer in the prefill kernel (dequant
  gate → MFMA → dequant up → MFMA) to cap LDS at 16 KB.
- **Flat-index `sorted_ids`**: `moe_align_block_size` stores
  `token*top_k + sel` (padded with `M*top_k`), so the routing-weight gather
  keeps the selection index for `top_k > 1`; `act_scratch` is indexed by
  sorted row (not token), so a token routed to multiple experts never races.
- **MFMA fragment layout** (verified empirically with one-hot matrices):
  A `m=lane%16`, B `n=lane%16`, C `col=lane%16, row=(lane/16)*4+i`
  (4 consecutive rows × 1 column per lane).
- **Prefill tile config**: BM=64/BN=64/BK=64, 256 threads, 4 column
  fragments per wavefront. Added in a follow-up for the M>32 regime; the
  BN was later narrowed 128→64 to lift occupancy (see Caveats).
- **Double-buffering experiment** (documented in Caveats): 2-stage LDS was
  implemented, measured, and **reverted** — it halved decode occupancy and
  cost ~50%.

## Workload

- **Device**: AMD Instinct **MI250X** (2 GCDs; benchmarks run on device 0 =
  one GCD, ~110 CU). gfx90a, ROCm 6.2.4. Measured on the beverin `mi200`
  partition inside the `vkernels:hip-full` container (image 20.8 GB, loaded
  per job from `/capstor/scratch/cscs/xyao/vkernels-full-v2.tar`).
- **Config**: E=256 experts, hidden=4096, ispp=512, top_k=6, group_size=32,
  SwiGLU clamp L=10. Random routing. Weights are random MXFP4 (E2M1) with
  unit (127) ue8m0 scales, zero biases.
- **Baseline**: xkernels' reference torch loop
  (`bench_xkernels_torch.py`) — per-expert `dequant_mxfp4_weight` →
  `A[t] @ w13e.T` → SiLU → `act @ w2e.T` → `index_add_`, in torch (no
  Triton; the Triton backend segfaults under ROCm 6.2.4 on both gfx90a and
  gfx942). Exact-match-verified against the vkernels CPU oracle.
- **Correctness**: GPU output matches the CPU oracle to
  `max_rel < 0.00001` (decode) and `< 0.02` (prefill).

**Reproduce** (on a gfx90a host with the image loaded):

```bash
podman run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
  -v /users/xyao/vkernels-prefill:/workspace/vkernels vkernels:hip-full \
  /workspace/vkernels/build-hip-prefill/meta/benchmarks/moe_fused_bench
# and: .../moe_fused_prefill_bench , .../moe_bench , .../test_moe_fused_correct
```

## Results — decode regime (E=256, hidden=4096, ispp=512, top_k=6)

Latency is the median of 10 iterations (3 warmups), HIP-event timed. `EM`
is the padded sorted-row count. **GFLOP/s = 6·M·ispp·hidden·top_k / t**
(2·MAC × the gate+up+down GEMMs × the top_k routed experts — the *useful*
arithmetic; padded EM rows are extra work not counted).

| M | EM | vkernels (ms) | xkernels torch (ms) | speedup | GFLOP/s | note |
|---:|---:|---:|---:|---:|---:|---|
| 1  | 96   | 0.648 | 6.0 | 9.3× | 117 | 96 padded rows vs 6 real |
| 2  | 192  | 0.821 | — | — | 184 | |
| 4  | 384  | 1.189 | — | — | 254 | |
| 8  | 768  | 3.446 | ~30 | ~8.7× | 175 | |
| 16 | 1408 | 3.983 | ~40 | ~10× | 303 | |
| 32 | 2432 | 9.029 | — | — | 268 | |
| 48 | 3456 | 10.326 | 162.2 | 15.7× | 351 | 3456 padded vs 288 real (12× waste) |

The decode regime is padding-dominated: per-expert 16-row blocks mean EM
scales with the number of *routed experts*, not tokens, so effective
throughput stays at ~0.1% of the ~191 TFLOP/s per-GCD bf16 roof.

## Results — prefill vs decode (E=8, hidden=4096, ispp=512, top_k=2)

Dense routing (E=8, so each 64-row block fills once M ≥ 512). GFLOP/s =
6·M·ispp·hidden·top_k / t.

| M | decode (ms) | decode GFLOP/s | prefill (ms) | prefill GFLOP/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 128  | 0.958 | 3364 | 1.175 | 2740 | 0.81× |
| 256  | 1.356 | 4752 | 1.201 | 5365 | 1.13× |
| 512  | 2.697 | 4777 | 1.945 | 6625 | 1.39× |
| 1024 | 5.151 | 5003 | 2.993 | 8609 | 1.72× |
| 2048 | 10.236 | 5035 | 5.781 | 8916 | 1.77× |

The prefill config (BN=64) wins from M ≥ 256; below that the 64-row padding
waste dominates and decode is faster. Peak prefill throughput is ~8.9
TFLOP/s ≈ 4.7% of the ~191 TFLOP/s per-GCD roof — the kernel is
occupancy-bound (see Caveats).

## Results — primitives (microbenchmarks)

Single-wavefront microbenchmarks of the four low-level building blocks
(`bench_moe.hip`); the MFMA number is a **dependent-chain floor** (one
accumulator, `__asm__ __volatile__`), not a throughput ceiling.

| Primitive | Result | Roof / note |
|---|---|---|
| K16 MFMA (16×16×16bf16) | 0.431 TFLOP/s | dependent chain, 1 wavefront; not the device roof |
| K32 emulation (2×K16) | 0.363 TFLOP/s | 84% of K16 (amortizes the loop) |
| fp4→bf16 dequant | 279 GElem/s | 537M elems, scalar LUT path |
| LDS fill (global_load_dwordx4) | 4898 GB/s | 64 threads × uint4, exceeds HBM roof (LDS-bound) |

## Caveats and notes

- **One GCD, not the full package.** MI250X exposes 2 GCDs; every number
  above is device 0 alone (one GCD ≈ 110 CU ≈ 191 TFLOP/s bf16). The bench's
  printed "383 TFLOP/s" roof is the whole 2-GCD package. A real deployment
  would shard experts across both GCDs (~2×).
- **Occupancy-bound, not barrier-bound.** LDS double-buffering (2-stage
  sA/sB) was implemented and reverted: it doubled decode LDS 6→12 KB,
  halving blocks/CU, and slowed decode ~50% (e.g. 1.43→2.11 ms at M=256).
  The redundant barrier between the A-tile load and the weight dequant was
  removed (they write different LDS buffers) — a free, small win. Acting on
  this, the prefill BN was narrowed 128→64 (24→16 KB LDS, 64→32 accumulator
  VGPRs), lifting occupancy 2→3 blocks/CU for a measured ~1.5× (M=2048:
  8.75→5.78 ms). The lever is *more* occupancy, not deeper pipelining.
- **~1.5% of peak, and it's the dequant, not the MFMA.** Each block
  dequantizes every fp4 weight inline (LUT + scale + bf16 convert), which is
  ALU-heavy relative to the 16×16×16 MFMAs it feeds. Materializing weights
  once (xkernels Triton's approach) trades 138 GB of HBM traffic for this
  ALU; the fused kernel wins by never touching that buffer.
- **Decode padding waste.** `moe_align_block_size` pads every routed expert
  to a 16-row block, so small-M latency is ~(experts routed)·16 rows, not
  M tokens. A padding-aware launch (skip empty block tails) would help M<32.
- **gfx90a constraints.** >64-thread blocks crash with hand-pinned VGPRs;
  `__volatile__` on MFMA asm breaks accumulation; `reinterpret_cast` on
  `void*` is rejected in device code (C-style casts required). The fused
  kernel sidesteps all three (builtin MFMA, 64-thread decode blocks, C
  casts).
- **Baseline is torch, not Triton.** xkernels' Triton backend segfaults
  under ROCm 6.2.4 on both gfx90a and gfx942 (SIGSEGV inside Triton 3.3.1),
  so the comparison is against the reference torch loop.

## Journal

**What was done.** Wired the four MXFP4 primitives (issues #12–15) into a
2-launch fused MoE matching xkernels' `fused_moe_mxfp4` interface, with a
CPU oracle, Python bindings, and a prefill (BM=64/BN=64) tile config.
Benchmarked on MI250X and recorded under `docs/performance/`.

**Challenges.**
1. *Fragment layout.* The MFMA C fragment is 4 consecutive rows × 1 column
   per lane (not a 4×4 sub-tile) — pinned down empirically with one-hot
   matrices before the kernel could be correct.
2. *Single bf16 rounding.* A 3-kernel split (gate, up, combine) rounds the
   activation twice and mismatches the oracle; gate+up must share one kernel
   so `silu(g)·u` is rounded once.
3. *Register pinning vs. reliability.* Hand-pinned VGPR asm crashed for
   >64-thread blocks and broke accumulation with `__volatile__`; the fused
   kernel uses the clang builtin, which keeps register allocation correct
   across ROCm versions.
4. *Double-buffering didn't pay.* The occupancy loss (6→12 KB LDS) outweighed
   the load→MFMA overlap; reverted after measurement (see Caveats).
5. *Occupancy was the real lever.* Narrowing the prefill BN 128→64 (same
   64-row M-tile, half the columns) traded B-tile reuse for occupancy
   (2→3 blocks/CU) and won ~1.5× across the board — the dequant ALU is the
   bottleneck, and more resident warps hide it better than wider tiles reuse it.

**Future work.**
- Shard across both MI250X GCDs and measure the ~2× scaling.
- Raise occupancy further: a BM=32 prefill tile (or `__launch_bounds__`
  nudging) could cut VGPRs below 64 and reach 4 blocks/CU; split gate/up to
  reduce live registers is another path.
- Wavefront-specialised producer/consumer dequant once occupancy saturates.
- Revisit the Triton baseline when ROCm 6.3 / a fixed Triton ships.
