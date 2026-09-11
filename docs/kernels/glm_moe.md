# glm_moe — GLM-5.3 block-FP8 expert GEMV (gfx942 / MI300A)

A dequant-fused block-FP8 GEMV for the GLM-5.3-Flash expert weights
(issue #64). The MoE weights are E4M3FN (`uint8`, `[N, K]`, row-major,
row = output) with one FP32 scale per `128 × 128` block:

```
w      : [N, K]          uint8, E4M3FN codes, row-major (row = output)
scales : [N/128, K/128]  float32, scale[n/128][k/128] multiplies the block
x      : [M, K]          bfloat16 activations (decode: M = 1 or 2)
out    : [M, N]          bfloat16

out[m][n] = bf16( Σ_k  fp32(x[m][k]) · scale[n/128][k/128] · e4m3(w[n][k]) )
```

The kernel decodes the FP8 weights *inside* the dot — nothing BF16/FP32 is
ever materialised for the expert buffer. At decode (`M ∈ {1,2}`) the op is
memory-bound on the weight bytes (arithmetic intensity ≈ 2 FLOP/byte), so
fusing the dequant removes both the materialised-buffer HBM traffic and its
allocation. This is the standalone **native** GEMV; the torch_ops GLM FP8
path (`glm_expert_gemv.py`, `glm_fp8_blockwise_gemm.py`) is a separate
Triton/CuTe implementation adopted from floe (issue #65) — see
[../glm53-decode-kernels.md](../glm53-decode-kernels.md) and
[../torch-ops-mi300.md](../torch-ops-mi300.md).

## E4M3FN (OCP FP8, finite-only)

1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits; no infinities. The
encoding `e=15, m=7` (`0x7F`/`0xFF`) is the only NaN; everything else is
finite with max magnitude 448. Decode:

```
e > 0 : (-1)^s · 2^(e-7) · (1 + m/8)      (normals)
e = 0 : (-1)^s · 2^(-6)   · (m/8)         (subnormals, m=0 → +0)
```

- **Source (CPU)**: `src/c/vkernels/kernels/glm_moe.cpp` (oracle, always
  compiled). The reserved NaN encodings (`0x7F`/`0xFF`) decode to NaN.
- **Source (HIP)**: `src/c/vkernels/kernels/glm_moe.hip` (`VKERNELS_HAS_HIP`).
  The branchless E4M3 decode places `(e<<23 | m<<20)` into an f32 and scales
  by `2^120`, mapping normals and subnormals in one multiply — but the
  reserved NaN encodings map to `±480` there (the oracle yields NaN). Weight
  tensors never carry them, so this is a documented divergence, not a bug.

## Two-implementation model

| Operation | CPU (`glm_moe.cpp`) | HIP (`glm_moe.hip`) |
|---|---|---|
| dequant-fused block-FP8 GEMV | `glm_fp8_block_gemv_cpu` | `glm_fp8_block_gemv` (split-K) |
| E4M3FN decode | `glm_e4m3_to_f32_cpu` | `e4m3_f32` (branchless, device-only) |
| scratch size / occupancy | — | `glm_fp8_block_gemv_with_scratch` (autotune hook), `glm_fp8_gemv_pick_sk` |

The HIP kernel is split-K: a partial kernel (`glm_fp8_gemv_partial_kernel`,
one block per `[row0, row0+32)` output stripe, 256 threads/block, `k`
segmented into `K/128` blocks) produces fp32 partials, then a fixed-order
`glm_fp8_gemv_reduce_kernel` sums them. `SEGS > 0` is a template constant
so the segment loop fully unrolls and per-segment loads batch ahead of
their consumers; `SEGS == 0` is the runtime-bound fallback for odd segment
counts (e.g. `K = 1152 → 9`). `sk ∈ {1,2,4,8}` divides `K/128`;
`glm_fp8_gemv_pick_sk(N, K)` returns the occupancy heuristic the plain
wrapper uses (≥ ~2 blocks per CU on the 228-CU MI300A).

## Contract

- `M ∈ {1, 2}`, `N % 128 == 0`, `K % 128 == 0`, `K ≤ 4096`.
- `x` is `[M, K]` bf16 (passed as a `uint16` view); `out` is `[M, N]` bf16.
- Accumulation in fp32; exactly one round-to-nearest-even to bf16 at the end.
- `part` (the `with_scratch` overload) is `M · N · sk` float32s, clobbered
  each call — allocate once and reuse.
- `out` must not alias any input.

## Tests & benchmark

- **Device correctness**: `meta/benchmarks/test_glm_fp8_gemv.hip` —
  `hip::glm_fp8_block_gemv` vs `glm_fp8_block_gemv_cpu` at the GLM decode
  expert shapes and small edge shapes, bf16-tolerant (`max_rel < 2e-2`,
  the `test_kda_correct.hip` convention). The NaN decode itself is checked
  separately for CPU/GPU bit agreement via the shared code path.
- **torch_ops parity**: `tests/python/test_glm_fp8_blockwise_gemm.py`,
  `tests/python/test_glm_expert_gemv.py`.
- **Benchmark**: `meta/benchmarks/bench_glm_fp8_gemv.hip` — per `(M, N, K)`
  compares the fused path (streams FP8 + scales; nothing materialised)
  against a BF16 mat-GEMV (the steady-state cost of the dequantize-once
  path floe uses today) and a one-time FP8→BF16 dequant, against the
  MI300A roof (~5300 GB/s). A decode token pays gate/up + down per selected
  expert; the per-matrix medians compose to that.

## Not exposed through the public API

`glm_moe` has **no C ABI** and is **not** in `vkernels.kernels`. It is a
standalone native GEMV with its own device test + micro-benchmark; the
serving path reaches a separate Triton/CuTe GLM FP8 GEMV through the
`vkernels.torch_ops` package (issue #65, adopted from floe).
