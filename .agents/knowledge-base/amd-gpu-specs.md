# AMD GPU Hardware Specifications

Quick-reference numbers for AMD CDNA accelerators. Always verify against
`rocm-smi --showproductname` and the datasheet for your exact SKU.

## MI300X (CDNA3, gfx942) — Flagship HPC/AI accelerator

| Parameter | Value |
|---|---|
| Architecture | CDNA3 (gfx942) |
| Compute Units (CUs) | 304 |
| Peak fp16/bf16 (Matrix Core) | ~1307 TFLOP/s |
| Peak fp32 (Matrix Core) | ~653 TFLOP/s |
| Peak fp64 (Matrix Core) | ~81 TFLOP/s |
| Peak fp32 (Vector ALU) | ~81 TFLOP/s |
| Peak fp64 (Vector ALU) | ~5.1 TFLOP/s |
| Peak int8 (Matrix Core) | ~2614 TOP/s |
| Peak fp8/bf8 (Matrix Core) | ~2614 TFLOP/s |
| HBM type / capacity | HBM3 / 192 GB |
| HBM bandwidth | ~5.3 TB/s |
| HBM bus width | 8192-bit |
| L2 cache | 256 MB |
| LDS per CU | 64 KB |
| Max VGPRs per thread | 256 |
| Max SGPRs per wavefront | 104 |
| Max wavefronts per CU | 32 |
| Max threads per CU | 2048 |
| Max workgroup size | 1024 |
| GCDs (Graphics Compute Dies) | 8 |
| CUs per GCD | 38 |
| TDP | 750 W |
| Ridge point (fp16, FLOP/byte) | ~245 |
| Process node | TSMC 5nm/6nm |

## MI250X (CDNA2, gfx90a) — Dual-die HPC accelerator

| Parameter | Value |
|---|---|
| Architecture | CDNA2 (gfx90a) |
| Compute Units (CUs) | 220 (2×110) |
| Peak fp16/bf16 (Matrix Core) | ~383 TFLOP/s |
| Peak fp32 (Matrix Core) | ~191 TFLOP/s |
| Peak fp64 (Matrix Core) | ~95.7 TFLOP/s |
| Peak fp32 (Vector ALU) | ~47.9 TFLOP/s |
| Peak fp64 (Vector ALU) | ~47.9 TFLOP/s (full rate!) |
| Peak int8 (Matrix Core) | ~383 TOP/s |
| HBM type / capacity | HBM2e / 128 GB (2×64 GB) |
| HBM bandwidth | ~3.2 TB/s |
| HBM bus width | 8192-bit (2×4096) |
| L2 cache | 16 MB (2×8 MB) |
| LDS per CU | 64 KB |
| Max VGPRs per thread | 256 |
| Max wavefronts per CU | 32 |
| GCDs (Graphics Compute Dies) | 2 |
| TDP | 560 W |
| Ridge point (fp16, FLOP/byte) | ~120 |
| Notes | Full-rate fp64 Vector ALU (unique among GPUs) |

## MI210 (CDNA2, gfx90a) — Single-die accelerator

| Parameter | Value |
|---|---|
| Architecture | CDNA2 (gfx90a) |
| Compute Units (CUs) | 104 |
| Peak fp16/bf16 (Matrix Core) | ~181 TFLOP/s |
| Peak fp32 (Matrix Core) | ~90.5 TFLOP/s |
| Peak fp64 (Matrix Core) | ~45.2 TFLOP/s |
| Peak fp32 (Vector ALU) | ~45.2 TFLOP/s |
| Peak fp64 (Vector ALU) | ~45.2 TFLOP/s (full rate!) |
| HBM type / capacity | HBM2e / 64 GB |
| HBM bandwidth | ~1.6 TB/s |
| HBM bus width | 4096-bit |
| L2 cache | 8 MB |
| LDS per CU | 64 KB |
| TDP | 300 W |
| Ridge point (fp16) | ~113 |

## MI100 (CDNA, gfx908) — First-gen CDNA

| Parameter | Value |
|---|---|
| Architecture | CDNA (gfx908) |
| Compute Units (CUs) | 120 |
| Peak fp16 (Matrix Core) | ~184 TFLOP/s |
| Peak fp32 (Matrix Core) | ~46.1 TFLOP/s |
| Peak fp64 (Matrix Core) | ~11.5 TFLOP/s |
| Peak fp32 (Vector ALU) | ~23.1 TFLOP/s |
| Peak fp64 (Vector ALU) | ~11.5 TFLOP/s |
| HBM type / capacity | HBM2 / 32 GB |
| HBM bandwidth | ~1.2 TB/s |
| Ridge point (fp16) | ~153 |

## Compute unit internals (CDNA3, per CU)

| Unit | Count | Purpose |
|---|---|---|
| Matrix Cores | 4 | MFMA instructions (fp16, bf16, fp32, fp64, int8, fp8) |
| Vector ALUs (SIMD) | 4×16 = 64 lanes | Scalar FP/integer ops |
| Scalar ALU | 1 | Uniform control/branch/address |
| LDS | 64 KB | Shared memory, 32 banks |
| Scheduler | 1 | Issues 1 wavefront per cycle |
| VGPR file | ~1536 total | 256 per thread max |
| SGPR file | ~128 total | 104 per wavefront max |

## Wavefront and workgroup limits

| Parameter | CDNA3 (MI300X) | CDNA2 (MI250X/MI210) |
|---|---|---|
| Wavefront width | 64 threads | 64 threads |
| Max wavefronts/CU | 32 | 32 |
| Max workgroup size | 1024 (16 wf) | 1024 (16 wf) |
| Max workgroups/CU | 8 | 8 |
| Max grid dim | 2^32 - 1 | 2^32 - 1 |
| Max threads total | 2^32 - 1 | 2^32 - 1 |

## Memory latency (approximate cycles, CDNA3)

| Operation | Approximate cycles |
|---|---|
| LDS read (hit) | ~20 |
| L2 cache hit | ~100-150 |
| HBM access | ~290-350 |
| VGPR read | 0 (register) |
| SGPR read | 0 (register) |
| `global_load` issue to data | ~300+ (HBM dependent) |
