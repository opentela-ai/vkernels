# AMD vs NVIDIA: Cross-Vendor Reference

Side-by-side terminology, concepts, and tool mappings for working across both
GPU ecosystems. Use this when translating a technique or kernel between
platforms.

## Terminology map

| Concept | AMD | NVIDIA |
|---|---|---|
| GPU kernel language | HIP C++ | CUDA C++ |
| Device compiler | `hipcc` (clang-based) | `nvcc` |
| Compute unit | Compute Unit (CU) | Streaming Multiprocessor (SM) |
| Thread grouping (hardware) | Wavefront (64 threads) | Warp (32 threads) |
| Thread grouping (software) | Workgroup / thread block | Thread block / CTA |
| Matrix multiply unit | Matrix Core | Tensor Core |
| Scalar FP unit | Vector ALU | CUDA Core |
| Shared memory | LDS (Local Data Share) | SMEM (Shared Memory) |
| Matrix accumulator | VGPRs (acc VGPRs) | TMEM (Blackwell) / Registers (Hopper-) |
| Async bulk copy | N/A (use vectorized loads) | TMA (cp.async.bulk) |
| Synchronization primitive | `s_waitcnt` + `__syncthreads` | mbarrier |
| Memory fence | `__threadfence_block` / `system` | `__threadfence_block` / `system` |
| L1/LDS split | Fixed 64 KB LDS | Configurable SMEM/L1 |
| Multi-GPU in package | GCDs (Graphics Compute Dies) | GPC partitions within the die |
| Cross-die memory | `hipDeviceEnablePeerAccess` | NVLink / NVSwitch |
| Profiler (timeline) | omnitrace | Nsight Systems (nsys) |
| Profiler (kernel) | omniperf, rocprof | Nsight Compute (ncu) |
| GEMM library | rocBLAS | cuBLAS |
| Matrix fragment library | rocWMMA | CUTLASS / nv_wmma |
| Fused kernel library | composable_kernel (CK) | CUTLASS |
| Deep learning | MIOpen | cuDNN |
| Communication | RCCL | NCCL |
| System management | rocm-smi | nvidia-smi |
| Kernel launch config | `<<<grid, block, LDS, stream>>>` | `<<<grid, block, SMEM, stream>>>` |
| Device sync | `hipDeviceSynchronize()` | `cudaDeviceSynchronize()` |
| Stream | `hipStream_t` | `cudaStream_t` |
| Event timing | `hipEventRecord/hipEventElapsedTime` | `cudaEventRecord/cudaEventElapsedTime` |
| Memory allocation | `hipMalloc` | `cudaMalloc` |
| Host memory | `hipHostMalloc` | `cudaHostAlloc` |
| Managed memory | `hipMallocManaged` | `cudaMallocManaged` |

## Register file comparison

| Aspect | AMD (CDNA3) | NVIDIA (Blackwell) |
|---|---|---|
| Thread register type | VGPR (vector) + SGPR (scalar) | GP register (unified) |
| VGPRs per thread | 256 | 255 |
| SGPRs per wavefront | 104 | N/A |
| Special registers | vcc, scc, exec, m0, mode, status | %laneid, %warpid, %smid, %clock |
| Accumulator location | VGPRs (acc0-acc255) | TMEM (Blackwell) or registers |
| Spill destination | Scratch (VGPR spill to HBM) | Local memory (LMEM, L1/L2 cached) |

## Memory hierarchy comparison

| Level | AMD (MI300X) | NVIDIA (B200) |
|---|---|---|
| VGPR / Register | 256 per thread × 64 threads = 16K | 255 per thread × 32 threads = 8K |
| LDS / SMEM | 64 KB per CU, 32 banks, 4B granularity | 228 KB per SM, 32 banks, 4B granularity |
| Matrix accumulator | VGPRs (inline) | TMEM 512 KB (separate space) |
| L1 cache | 32 KB (read-only vector L1) | Configurable, shared with SMEM |
| L2 cache | 256 MB (MI300X total) | 128 MB (B200) |
| HBM | 192 GB HBM3, ~5.3 TB/s | 192 GB HBM3e, ~8 TB/s |

## Synchronization comparison

| Operation | AMD | NVIDIA |
|---|---|---|
| Workgroup barrier | `__syncthreads()` | `__syncthreads()` / `__syncwarp()` |
| Memory fence (block) | `__threadfence_block()` | `__threadfence_block()` |
| Memory fence (device) | `__threadfence()` | `__threadfence()` |
| Memory fence (system) | `__threadfence_system()` | `__threadfence_system()` |
| Wait for global loads | `__builtin_amdgcn_s_waitcnt(0, 0, 0)` | `__syncwarp()` + barrier |
| Wait for LDS writes | `__builtin_amdgcn_s_waitcnt(0, 0x7F, 0)` | `__syncthreads()` or mbarrier |
| Async bulk copy | N/A (manual `global_load` → LDS) | `cp.async.bulk` (TMA) |
| Async copy completion | `s_waitcnt vmcnt(0)` | mbarrier `expect_tx` + `try_wait` |
| Pipeline stages | LDS double-buffering | SMEM multi-stage + mbarrier phases |

## Instruction set comparison

| Operation | AMDGPU ISA | PTX |
|---|---|---|
| Thread ID | `v_mbcnt` or `__lane_id()` | `%laneid` (special register) |
| Workgroup barrier | `s_barrier` | `bar.sync` |
| LDS/SMEM read | `ds_read_b32`/`b128` | `ld.shared` |
| LDS/SMEM write | `ds_write_b32`/`b128` | `st.shared` |
| Global load | `global_load_dword`/`dwordx4` | `ld.global` |
| Global store | `global_store_dword`/`dwordx4` | `st.global` |
| Matrix multiply-add | `v_mfma_f32_16x16x16f16` | `mma.sync` / `wgmma` / `tcgen05` |
| Wait counter drain | `s_waitcnt vmcnt(0)` | `bar.sync` / mbarrier `wait` |
| Lane shuffle | `ds_swizzle_b32` / `v_permlanex16` | `shfl.sync` |
| Predicate | `v_cndmask_b32` on `vcc` | `@%p` predicate registers |

## Key architectural differences

1. **Wavefront width:** AMD wavefront = 64 threads; NVIDIA warp = 32 threads.
   This affects coalescing granularity (256 bytes vs 128 bytes).

2. **No TMA on AMD:** AMD relies on explicit `global_load` + `ds_write`
   sequences for tile staging. The closest optimization is using wide loads
   (`global_load_dwordx4`, 16 bytes/thread) and LDS write combining.

3. **No mbarrier on AMD:** AMD's `s_waitcnt` is a simpler counter-drain model
   compared to NVIDIA's phase-based mbarriers. For pipelines, AMD uses LDS
   double-buffering with explicit `lgkmcnt(0)` waits.

4. **No TMEM on AMD:** MFMA accumulators are VGPR-based. The register pressure
   from large accumulator tiles cuts occupancy directly. NVIDIA Blackwell's
   TMEM offloads this pressure to a dedicated memory space.

5. **No warpgroup MMA on AMD:** Each MFMA occupies a single 64-thread
   wavefront. There is no cooperative multi-warp MMA (like `wgmma`).

6. **SGPR vs unified registers:** AMD dedicates a scalar register file for
   uniform operations, which saves VGPRs when a value is the same across all
   64 threads.

7. **Multi-die topology differs:** AMD's MI300X has 8 GCDs as separate
   devices (peer access for cross-GCD). NVIDIA's B200 is a monolithic die.

8. **Full-rate fp64 on CDNA2:** MI250X/MI210 have full-rate fp64 on Vector
   ALUs, which is unique among GPUs and useful for HPC workloads.
