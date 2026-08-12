# NVIDIA GPU Hardware Specifications

Quick-reference numbers for NVIDIA Hopper/Blackwell accelerators. Always verify
against `nvidia-smi -q` and the datasheet for your exact SKU.

## B200 (Blackwell) — Flagship AI accelerator

| Parameter | Value |
|---|---|
| Architecture | Blackwell |
| Streaming Multiprocessors (SMs) | 160 |
| Peak fp16/bf16 (Tensor Core, dense) | ~2000 TFLOP/s (2 PFLOP/s) |
| Peak fp8/bf8 (Tensor Core) | ~4000 TFLOP/s (4 PFLOP/s) |
| Peak fp4 (Tensor Core) | ~8000 TFLOP/s (8 PFLOP/s) |
| Peak fp32 (CUDA Core) | ~80 TFLOP/s |
| Peak fp64 (Tensor Core) | ~40 TFLOP/s |
| Peak int8 (Tensor Core) | ~4000 TOP/s |
| HBM type / capacity | HBM3e / 192 GB |
| HBM bandwidth | ~8 TB/s |
| L2 cache | 128 MB |
| Shared memory (SMEM) per SM | 228 KB |
| Tensor Memory (TMEM) per SM | 512 KB (128 lanes × 512 cols × 4B) |
| Max registers per thread | 255 |
| Max warps per SM | 64 |
| Max threads per SM | 2048 |
| Max threads per block | 1024 |
| Max blocks per SM | 32 |
| Max cluster size | 16 SMs (with DSMEM) |
| TDP | 1000 W |
| Ridge point (fp16, FLOP/byte) | ~250 |
| Key features | TMA, mbarrier, tcgen05, CLC, DSMEM, fp4 |

## H100 (Hopper) — Current-gen AI accelerator (SXM)

| Parameter | Value |
|---|---|
| Architecture | Hopper |
| SMs | 132 |
| Peak fp16/bf16 (Tensor Core) | ~990 TFLOP/s |
| Peak fp8/bf8 (Tensor Core) | ~1980 TFLOP/s |
| Peak fp32 (CUDA Core) | ~67 TFLOP/s |
| Peak fp64 (Tensor Core) | ~67 TFLOP/s |
| Peak int8 (Tensor Core) | ~1980 TOP/s |
| HBM type / capacity | HBM3 / 80 GB |
| HBM bandwidth | ~3.35 TB/s |
| L2 cache | 50 MB |
| SMEM per SM | 228 KB |
| Max registers per thread | 255 |
| TDP | 700 W |
| Ridge point (fp16) | ~295 |
| Key features | TMA, mbarrier, wgmma, DSMEM, fp8, DPX |

## H100 (PCIe) — Lower-power variant

| Parameter | Value |
|---|---|
| Peak fp16/bf16 (Tensor Core) | ~756 TFLOP/s |
| Peak fp8/bf8 (Tensor Core) | ~1513 TFLOP/s |
| Peak fp32 (CUDA Core) | ~51 TFLOP/s |
| HBM bandwidth | ~2.04 TB/s |
| TDP | 350 W |

## H200 (Hopper) — High-memory variant

| Parameter | Value |
|---|---|
| Architecture | Hopper (same SM as H100) |
| HBM type / capacity | HBM3e / 141 GB |
| HBM bandwidth | ~4.8 TB/s |
| Peak fp16 (Tensor Core) | ~990 TFLOP/s (same as H100 SXM) |
| Ridge point (fp16) | ~206 |

## A100 (Ampere) — Prior-gen baseline

| Parameter | Value |
|---|---|
| Architecture | Ampere |
| SMs | 108 |
| Peak fp16/bf16 (Tensor Core) | ~312 TFLOP/s |
| Peak fp32 (CUDA Core) | ~19.5 TFLOP/s |
| Peak fp64 (Tensor Core) | ~19.5 TFLOP/s |
| Peak int8 (Tensor Core) | ~624 TOP/s |
| HBM type / capacity | HBM2e / 80 GB (or 40 GB) |
| HBM bandwidth | ~1.55 TB/s |
| L2 cache | 40 MB |
| SMEM per SM | 164 KB |
| Max registers per thread | 255 |
| TDP | 400 W |
| Ridge point (fp16) | ~200 |
| Key features | Async copy (cp.async), no TMA, no mbarrier |

## SM internals (per SM, Blackwell)

| Unit | Count | Purpose |
|---|---|---|
| Tensor Cores (5th gen) | 4 | tcgen05 MMA (fp16, bf16, fp8, fp4, int8) |
| CUDA Cores (fp32) | 128 | Scalar FP/int ops |
| CUDA Cores (fp64) | 64 | Double-precision FP |
| Special Function Units | 4 | Transcendentals |
| Tensor Memory (TMEM) | 512 KB | MMA accumulator (new in Blackwell) |
| Shared Memory (SMEM) | 228 KB | L1/SMEM configurable |
| Warp Scheduler | 4 | 1 warp issue per scheduler per cycle |
| Register File | 65536 × 32-bit | 255 per thread max |

## Warp and block limits

| Parameter | Blackwell | Hopper | Ampere |
|---|---|---|---|
| Warp width | 32 threads | 32 threads | 32 threads |
| Max warps/SM | 64 | 64 | 64 |
| Max threads/block | 1024 | 1024 | 1024 |
| Max blocks/SM | 32 | 32 | 32 |
| Max grid dim X | 2^31-1 | 2^31-1 | 2^31-1 |
| Max grid dim Y/Z | 65535 | 65535 | 65535 |
| Max cluster size | 16 SMs | 16 SMs | N/A |

## Memory latency (approximate cycles, H100)

| Operation | Approximate cycles |
|---|---|
| SMEM read (hit) | ~20-30 |
| L1 cache hit | ~30-50 |
| L2 cache hit | ~200-300 |
| HBM access | ~400-600 |
| Register read | 0 |
| TMA load (descriptor setup) | ~20 + SMEM fill |
