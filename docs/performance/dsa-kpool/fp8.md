# dsa-kpool — fp8e4m3fn+scale cache path (issue #61)

Latency + roofline characterization of the **fp8+scale** storage
overloads `dsa_kpool_{assemble,decode_update}_fp8` in
`src/c/vkernels/kernels/dsa_kpool.hip`, benchmarked alongside the
bf16 kernels from issue #60 on **both** target accelerators:

  * **AMD Instinct MI300A** (gfx942, hipcc/ROCm 6.3.0) — beverin
  * **NVIDIA A100-SXM4** (sm_80, nvcc) — bristen, via the HIP→CUDA shim

Both kernels are one warp-group (64 threads) per pool/request; the
fp8 overloads store the compressed mean as `H` fp8e4m3fn bytes **plus
one fp32 scale per vector** (round-trip `fp32→fp8→fp32` RNE when
`round_scale=1`, the heavier path measured here), instead of `2*H`
bf16 bytes. The reads are identical (keys/scores stay bf16), so
rows are directly comparable to the bf16 sections. Raw logs:
[`bristen-dsa-kpool-bench-A100-fp8.log`](bristen-dsa-kpool-bench-A100-fp8.log),
[`beverin-dsa-kpool-bench-fp8.log`](beverin-dsa-kpool-bench-fp8.log).

## Binding

Every fp8 row — like the bf16 rows — is **launch/occupancy bound**:
effective GB/s is a small fraction of the HBM roof (MI300A 5300 GB/s,
A100 2039 GB/s) and arithmetic intensity is 1.3–1.6 FLOP/B, far below
the ridge (~247 FLOP/B MI300A, ~10 FLOP/B A100). These are sub-µs
per-warp bookkeeping kernels; latency is dominated by dispatch and
single-warp occupancy, not memory or compute.

## Headline: fp8 vs bf16 (largest serving shape, ps=8)

| | A100 (bf16 → fp8) | MI300A (bf16 → fp8) |
|---|---|---|
| **PREFILL** n_pools=256 | 10.37 → 12.36 µs (**1.19x** slower) | 12.89 → 14.48 µs (**1.12x** slower) |
| **DECODE** batch=512 | 16.32 → 18.50 µs (**1.13x** slower) | 16.76 → 18.52 µs (**1.10x** slower) |

The ~10–20% per-launch cost is the **extra block-wide shared-memory
reductions** the fp8 store epilogue needs — an OR over "any element
finite" (skip gate) and a max of |mean| (the single per-vector scale)
— plus the round-trip RNE quantize. All pure compute/occupancy
overhead at one-warp-per-pool shapes; the *write* itself is half the
bf16 bytes, but the kernel was never memory-bound so the smaller store
does not buy latency. The fp8 path's value is therefore **half the
persistent cache footprint** (one byte vs two per K element, plus a
fp32 scale per vector) and **graph-capturability** (the `round_scale`
flag is a device int a prior graph node can produce on the same stream,
read on-device — never host-dereferenced).

## PREFILL — `dsa_kpool_assemble_fp8` (round_scale=1)

```
=== A100 (roof 20 TFLOP/s fp32, 2039 GB/s HBM) ===
npool   ps  tail  ssp  npg  warps  us(min)  us(med)  TFLOP/s    GB/s    AI  bound
    1    4    64    8   16      1     8.93     8.93   0.0007     0.48   1.6  launch
    4    4    64    8   16      4     9.31     9.32   0.0029     1.83   1.6  launch
   16    4    64    8   16     16     9.83     9.84   0.0108     6.92   1.6  launch
   64    4    64    8   16     64     9.84     9.84   0.0433    27.65   1.6  launch
  128    4    64    8   32    128     9.73     9.75   0.0874    55.84   1.6  launch
  256    4    64    8   32    256    10.30    10.31   0.1653   105.59   1.6  launch
    1    8   128   16   16      1    10.06    10.06   0.0011     0.83   1.3  launch
   16    8   128   16   32     16    11.37    11.38   0.0158    11.74   1.3  launch
   64    8   128   16   32     64    11.57    11.58   0.0622    46.12   1.3  launch
  128    8   128   16   64    128    11.59    11.60   0.1243    92.15   1.3  launch
  256    8   128   16   64    256    12.35    12.36   0.2333   172.92   1.3  mem

=== MI300A (roof 1307 TFLOP/s bf16, 5300 GB/s HBM) ===
npool   ps  tail  ssp  npg  warps  us(min)  us(med)  TFLOP/s    GB/s    AI  bound
    1    4    64    8   16      1    12.92    12.93   0.0005     0.33   1.6  launch
    4    4    64    8   16      4    12.92    12.92   0.0021     1.32   1.6  launch
   16    4    64    8   16     16    12.95    12.95   0.0082     5.25   1.6  launch
   64    4    64    8   16     64    12.95    12.95   0.0329    21.02   1.6  launch
  128    4    64    8   32    128    12.96    12.97   0.0657    41.96   1.6  launch
  256    4    64    8   32    256    13.00    13.01   0.1310    83.69   1.6  launch
    1    8   128   16   16      1    14.30    14.31   0.0008     0.58   1.3  launch
   16    8   128   16   32     16    14.36    14.36   0.0125     9.30   1.3  launch
   64    8   128   16   32     64    14.37    14.37   0.0502    37.17   1.3  launch
  128    8   128   16   64    128    14.43    14.43   0.0999    74.04   1.3  launch
  256    8   128   16   64    256    14.48    14.48   0.1992   147.60   1.3  launch
```

## DECODE — `dsa_kpool_decode_update_fp8` (round_scale=1)

```
=== A100 (roof 20 TFLOP/s fp32, 2039 GB/s HBM) ===
batch   ps  tail  ssp  npg  warps  us(min)  us(med)  TFLOP/s    GB/s    AI  bound
    1    4    64    8   16      1    12.22    12.23   0.0005     0.39   1.4  launch
    8    4    64    8   16      8    12.48    12.48   0.0043     3.05   1.4  launch
   32    4    64    8   32     32    12.68    12.68   0.0168    12.00   1.4  launch
  128    4    64    8   64    128    12.80    12.80   0.0666    47.56   1.4  launch
  256    4    64    8  128    256    13.26    13.26   0.1285    91.81   1.4  launch
  512    4    64    8  256    512    14.74    14.75   0.2310   165.09   1.4  launch
    1    8   128   16   16      1    15.51    15.52   0.0007     0.57   1.3  launch
    8    8   128   16   16      8    15.97    15.97   0.0056     4.43   1.3  launch
   32    8   128   16   32     32    15.97    15.98   0.0226    17.73   1.3  launch
  128    8   128   16   64    128    16.21    16.21   0.0889    69.89   1.3  launch
  256    8   128   16  128    256    16.83    16.84   0.1712   134.58   1.3  launch
  512    8   128   16  256    512    18.50    18.50   0.3117   244.96   1.3  mem

=== MI300A (roof 1307 TFLOP/s bf16, 5300 GB/s HBM) ===
batch   ps  tail  ssp  npg  warps  us(min)  us(med)  TFLOP/s    GB/s    AI  bound
    1    4    64    8   16      1    13.85    13.86   0.0005     0.34   1.4  launch
    8    4    64    8   16      8    13.97    13.98   0.0038     2.72   1.4  launch
   32    4    64    8   32     32    13.93    13.93   0.0153    10.93   1.4  launch
  128    4    64    8   64    128    14.01    14.01   0.0608    43.45   1.4  launch
  256    4    64    8  128    256    14.03    14.03   0.1214    86.78   1.4  launch
  512    4    64    8  256    512    16.55    16.55   0.2059   147.14   1.4  launch
    1    8   128   16   16      1    15.78    15.79   0.0007     0.56   1.3  launch
    8    8   128   16   16      8    15.93    15.93   0.0057     4.45   1.3  launch
   32    8   128   16   32     32    15.91    15.92   0.0226    17.80   1.3  launch
  128    8   128   16   64    128    15.97    15.97   0.0903    70.93   1.3  launch
  256    8   128   16  128    256    16.16    16.16   0.1785   140.24   1.3  launch
  512    8   128   16  256    512    18.51    18.52   0.3115   244.77   1.3  launch
```

## Method

`bench_dsa_kpool.hip` (built with `VKERNELS_BUILD_BENCHMARKS=ON`)
runs 2000 launches inside one timed `hipEvent` region, divided by
2000, over up to 40 samples (CV<5%), to beat MI300A's ~0.5 µs event
resolution and dynamic-sclk clock-domain corruption — see the bf16
[`gfx942.md`](gfx942.md) / [`A100.md`](A100.md) notes. A100 ran under
`uenv run prgenv-gnu/24.11:v1`; MI300A under system ROCm 6.3.0.

## Journal

Measured 2026-09-04 on both clusters right after landing the two A100
correctness fixes (the host-deref-of-device-pointer SIGSEGV and the
`__shfl_xor`→shared-memory reduction). Before the fixes the fp8 bench
could not run on A100 at all (segfault at first prefill launch); it now
completes clean (rc=0) on both, with the bf16 sections reproducing the
issue-60 baselines within noise. The fp8 kernels are 10–20% slower per
launch than bf16, entirely in the launch/occupancy regime — expected,
since the only added work is two shared-memory reductions and the
RNE quantize, and the kernel was never memory-bound. The trade is
deliberate: half the persistent cache footprint and a stream-correct
`round_scale` for CUDA-graph capture.
