# Performance Roofs and Roofline Reference

Formulas for computing FLOP/s, GB/s, arithmetic intensity, and ridge points
across common kernel types. For methodology, load
[hip-efficient-kernels](../skills/hip-efficient-kernels/SKILL.md).

## Key formulas

### Arithmetic intensity (AI)

```
AI = compute_work_FLOPs / data_moved_bytes
```

One fused multiply-add (FMA) `a*b + c` = 2 FLOPs.
One multiply = 1 FLOP. One add = 1 FLOP.

### Roofline bound

```
attainable_performance = min(peak_compute, peak_bandwidth × AI)
ridge_point = peak_compute / peak_bandwidth   (FLOP/byte)
```

Below the ridge → memory-bound. Above → compute-bound.

### Memory bandwidth utilization

```
achieved_GBps = bytes_moved / kernel_time_seconds
efficiency = achieved_GBps / peak_bandwidth
```

### Compute utilization

```
achieved_FLOPs = compute_work_FLOPs / kernel_time_seconds
efficiency = achieved_FLOPs / peak_compute   (use Matrix Core or Vector ALU peak)
```

## Per-kernel formulas

### GEMM: C = α·A·B + β·C

```
FLOPs  = 2 × M × N × K   (plus β·C if nonzero)
bytes  = (M×K + K×N) × sizeof(input) + M×N × sizeof(output)
         + (M×N × sizeof(C)) if beta ≠ 0 (C is read)
AI     = 2MNK / bytes

; Square (M=N=K), fp16 (sizeof=2), beta=0:
bytes  = 2M² × 2 + M² × 2 = 6M²   (read A,B, write C)
FLOPs  = 2M³
AI     = 2M³ / 6M² = M/3
```

### Elementwise: out = op(in)

```
FLOPs  = N × flops_per_element   (1 for add/mul, ~4 for GELU)
bytes  = N × (sizeof(in) + sizeof(out))   (one read, one write)
AI     = flops_per_element / (sizeof(in) + sizeof(out))

; fp32 elementwise add (1 FLOP):
AI = 1 / 8 = 0.125   — deeply memory-bound
```

### Reduction: sum(in)

```
FLOPs  = N-1 (or ~N)
bytes  = N × sizeof(in) + sizeof(out) ≈ N × sizeof(in)
AI     = ~1  FLOP/byte   — memory-bound

; fp32 reduction:
AI = 1 / 4 = 0.25   — deeply memory-bound
```

### FlashAttention (forward)

```
; Standard attention: S = QK^T written to HBM → read back for softmax → PV
; bytes = O(N²), tanks AI

; FlashAttention: keeps S on-chip, passes only O = softmax(QK^T)V to HBM
; Much higher AI — potentially compute-bound for large sequence length
```

## Ridge points for common GPUs

| Device | Peak fp16 (TFLOP/s) | Peak BW (TB/s) | Ridge (FLOP/byte) |
|---|---|---|---|
| AMD MI300X | ~1307 | ~5.3 | ~245 |
| AMD MI250X | ~383 | ~3.2 | ~120 |
| AMD MI210 | ~181 | ~1.6 | ~113 |
| NVIDIA B200 | ~2000 | ~8.0 | ~250 |
| NVIDIA H100 SXM | ~990 | ~3.35 | ~295 |
| NVIDIA H100 PCIe | ~756 | ~2.04 | ~371 |
| NVIDIA H200 | ~990 | ~4.8 | ~206 |
| NVIDIA A100 | ~312 | ~1.55 | ~200 |

## Using the ridge point

For a given kernel with AI = X FLOP/byte:

```
If X < ridge_point:
    → memory-bound: optimize data movement
    → target: approach peak bandwidth
    → fuse ops, use smaller dtypes, tile for reuse

If X > ridge_point:
    → compute-bound: optimize compute utilization
    → target: approach peak FLOP/s
    → pipeline, overlap, keep Matrix Cores busy
```

## Tiling's effect on AI

For GEMM with square tiles of size B:

```
AI = B / sizeof(element)

; fp16 (sizeof=2):
B=16  → AI=8    (memory-bound on all GPUs)
B=32  → AI=16   (memory-bound on all GPUs)
B=64  → AI=32   (memory-bound on most GPUs)
B=128 → AI=64   (memory-bound on H100, borderline on MI300X/B200)
B=256 → AI=128  (compute-bound on MI250X/MI210, borderline on MI300X)
```

## dtype scaling effects

| dtype | sizeof | AI multiplier (relative to fp32) | Notes |
|---|---|---|---|
| fp32 | 4 bytes | 1× | Baseline |
| fp16/bf16 | 2 bytes | 2× | Standard Matrix Core type |
| fp8/bf8 | 1 byte | 4× | Block-scaled (scale factors add overhead) |
| fp4 (NVIDIA B200) | 0.5 byte | 8× | Block-scaled, B200 only |
| int8 | 1 byte | 4× | Integer matrix multiply |

Real gains are less than the size ratio because:
- Scale factors for block-scaled types add memory traffic
- Some kernels have a significant non-memory component (e.g., epilogue)
- Alignment and granularity constraints waste some bytes

## Sanity-checking a measurement

Before diving into optimization, verify the number makes physical sense:

```
FLOP/s check:  achieved_TFLOPs <= peak_TFLOPs
GB/s check:    achieved_TBps   <= peak_TBps
AI check:      AI × achieved_TBps <= peak_TFLOPs
```

If any of these fail, your FLOP or byte count is wrong, or your timing is
broken. Fix the measurement before interpreting results.
