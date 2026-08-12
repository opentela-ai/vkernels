---
name: amdgpu-isa
description: Write and read AMDGPU assembly, the virtual ISA layered between HIP C++ and machine code on AMD GPUs. Use when authoring inline assembly in HIP kernels, lowering a kernel to AMDGPU ISA, choosing Matrix-Core instructions (mfma, v_mfma), reading or using special registers (vcc, scc, exec, m0), using memory instructions (global_load, ds_read, ds_write), or reasoning about the AMDGPU state spaces and memory model.
---

# AMDGPU ISA

AMDGPU is AMD's GPU instruction set, targeting the GFX9/GFX90a (CDNA2) and
GFX940/GFX942 (CDNA3) architectures on MI200 and MI300 series accelerators.
Unlike NVIDIA's PTX, AMDGPU directly targets hardware — there is no
intermediate virtual layer. The assembly you write in `__asm__` blocks is
close to what executes. This skill covers the instruction classes, register
kinds, memory spaces, and the wait-counter model that replaces explicit
barriers.

## Architecture targets

| GPU | Architecture | GFX target | Key instructions |
|---|---|---|---|
| MI300X | CDNA3 | gfx942 | `v_mfma_f32_16x16x32_f8f6f4`, packed fp8 |
| MI250X | CDNA2 | gfx90a | `v_mfma_f32_16x16x16f16`, full-rate fp64 |
| MI210 | CDNA2 | gfx90a | `v_mfma_f32_16x16x16f16` |
| MI100 | CDNA | gfx908 | `v_mfma_f32_16x16x16f16` (first Matrix Core) |
| Consumer | RDNA3 | gfx1100 | `v_wmma`, wave matrix ops |

> **Compile with:** `hipcc -x hip --offload-arch=gfx942` (MI300X) or
> `gfx90a` (MI250X). Use `rocminfo` or `hipconfig --platform` to see your
> target.

## Register file

AMDGPU has three distinct register types — more segregated than NVIDIA's
unified register file:

### Scalar registers (SGPRs) — `s0`–`s103`
- One per wavefront, shared by all 64 threads.
- Hold addresses, loop counters, constants, and control state.
- Written by scalar ALU (`s_add_u32`, `s_mov_b32`, `s_cmp_*`).
- **Limited:** 104 SGPRs on CDNA3; excess spills to VGPRs.

### Vector registers (VGPRs) — `v0`–`v255`
- One per thread, holds the wavefront's SIMD data.
- Written by vector ALU (`v_add_f32`, `v_mul_f32`, `v_mfma_*`).
- **Architected:** 256 VGPRs on CDNA3; excess spills to LDS or HBM.
- AGFPRs (accumulator VGPRs, `acc0`–`acc255`) hold double-precision or wide
  MMA accumulators on CDNA architectures.

### Special registers
| Register | Width | Purpose |
|---|---|---|
| `vcc` | 64-bit (two SGPRs) | Vector condition code — per-thread compare result |
| `scc` | 1-bit | Scalar condition code — uniform branch condition |
| `exec` | 64-bit | Execution mask — which threads in the wavefront are active |
| `m0` | 32-bit SGPR | LDS addressing parameter; must be set before LDS ops |
| `flat_scratch` | 64-bit SGPR pair | Base address for scratch (VGPR spill) memory |
| `mode` | 32-bit | Rounding mode, denorm control |
| `status` | 32-bit | Wavefront status — includes `halt` bit |

> **AMD vs NVIDIA:** `exec` is the AMD equivalent of NVIDIA's predicate mask
> but applies to *every* vector instruction. `vcc` is a 64-bit per-thread
> compare flag, unlike NVIDIA's single-thread condition codes. There is no
> AMD equivalent of NVIDIA's `%laneid` special register — compute lane ID from
> `v_mbcnt` or use `__lane_id()` in HIP.

## Memory hierarchy and instructions

AMDGPU ISA exposes the full memory hierarchy through five instruction
classes — all are explicit, unlike CUDA's implicit global/shared resolution:

| Space | Instructions | Latency | Typical use |
|---|---|---|---|
| Scalar memory | `s_load_dword`, `s_buffer_load` | ~20 cycles | Kernel args, constants via scalar pipe |
| Vector global | `global_load_dword`, `global_store_dword` | ~300+ cycles | HBM access |
| LDS (shared) | `ds_read_b32`, `ds_write_b32`, `ds_read_b128` | ~20 cycles | Workgroup scratchpad |
| Flat | `flat_load_dword`, `flat_store_dword` | Variable | Generic pointer (resolves to global/LDS/scratch) |
| Scratch | `buffer_load_dword`, `buffer_store_dword` | ~300+ cycles | VGPR spill/fill |

### Vector memory instructions (global → VGPR)

```
global_load_dword  v[data], v[addr], s[base+offset]
global_store_dword v[addr], v[data], s[base+offset]
```

- Address is 64-bit: a VGPR pair `v[addr:addr+1]` for the lower/upper 32 bits,
  plus an SGPR base `s[base:base+1]` for the resource descriptor.
- Coalescing: 64 threads × 4 bytes = 256 bytes is the ideal access width.
  Mismatch causes multiple transactions.

### LDS instructions (shared memory)

```
ds_write_b32  v[addr], v[data]        ; 4 bytes per thread
ds_read_b128  v[data:data+3], v[addr] ; 16 bytes per thread
ds_write_b128 v[addr], v[data:data+3] ; 16 bytes per thread
```

- `b128` (128-bit = 16 bytes) is the widest practical LDS access; it writes
  4 consecutive VGPRs to 4 consecutive LDS dwords.
- Bank conflict rules: 32 banks, 4-byte granularity. `bank = (addr // 4) % 32`.
  A wavefront (64 threads) doing `ds_read_b32` accesses 32 banks × 2 cycles;
  two threads hitting the same bank in the same access cycle serialize.

### Buffer instructions (scratch)

```
buffer_load_dword  v[data], v[offset], s[rsrc]
buffer_store_dword v[data], v[offset], s[rsrc]
```

- Resource descriptor `s[rsrc:rsrc+3]` encodes base address, stride, and
  bounds. Used for VGPR spills and formal parameter access.

## Wait counters — the AMDGPU memory model

AMD GPU memory operations are **not automatically ordered**. Explicit
`S_WAITCNT` instructions drain specific in-flight counters before the consumer
reads results. This replaces the mbarrier model of NVIDIA.

```
s_waitcnt lgkmcnt(0)    ; wait for all LDS/GPR/KMEM operations to complete
s_waitcnt vmcnt(0)      ; wait for all vector-memory (global/scratch) ops
s_waitcnt expcnt(0)     ; wait for all export (pixel/interpolation) ops
s_waitcnt vmcnt(0) & lgkmcnt(0)  ; wait for both
```

### Three counters

| Counter | Tracks | Incremented by | Waited by |
|---|---|---|---|
| `vmcnt` | Vector memory (global, flat, buffer, image) | Load/store to HBM | Consumer of loaded data |
| `lgkmcnt` | LDS, GPR (scalar mem), constants | LDS read/write, `s_load` | Consumer of LDS or scalar loads |
| `expcnt` | Export operations | Pixel/compute export | Graphics interop |

### The dance: copy then consume

```
; Producer: copy tile from HBM to LDS
global_load_dword v[0:63], v[addr], s[desc]  ; 64 dwords, sets vmcnt
; ... more loads ...
s_waitcnt vmcnt(0)                            ; HBM → VGPRs done
ds_write_b128 v[lds_addr], v[0:3]             ; VGPR → LDS, sets lgkmcnt
ds_write_b128 v[lds_addr+4], v[4:7]
s_waitcnt lgkmcnt(0)                          ; LDS writes done
; Consumer: reads LDS safely now
ds_read_b128 v[8:11], v[lds_addr]
```

> **Key insight:** `s_waitcnt vmcnt(N)` waits until `vmcnt ≤ N`. You often
> wait until `vmcnt(0)` (all done) for correctness, but pipelining uses
> `vmcnt(N)` with N>0 to allow some in-flight ops while ensuring a specific
> batch is done.

## Matrix-Core instructions (MFMA)

Matrix Fused Multiply-Add operates on VGPR fragments, not a separate TMEM
space (unlike NVIDIA `tcgen05`):

```
; CDNA2 (gfx90a): fp16 input, fp32 accumulate
v_mfma_f32_16x16x16f16  v[acc0:acc3], v[a0:a1], v[b0:b1], v[acc0:acc3]

; CDNA3 (gfx942): fp8/bf8 input
v_mfma_f32_16x16x32_f8f6f4  v[acc], v[a], v[b], v[acc]
```

- Format: `v_mfma_<acc_type>_<M>x<N>x<K>_<input_type>`
- `M×N` = output tile size, `K` = reduction dimension
- `mfma` occupies the full wavefront for several cycles; the destination
  VGPRs are locked during execution
- On CDNA3, fp8 MFMA with block scaling: scale factors packed into a separate
  VGPR argument

> **AMD vs NVIDIA:** No `accum=False/True` parameter. The first MFMA in a
> sequence writes the accumulator; subsequent MFMAs on the same accumulators
> add to them automatically (they always accumulate). Zero the accumulators
> explicitly before a new K-loop.

### rocWMMA — the higher-level path

Most HIP code should use `rocwmma::fragment` instead of raw `__asm__` MFMA:

```cpp
#include <rocwmma/rocwmma.hpp>
rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, half, rocwmma::row_major> a_frag;
rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag;
rocwmma::fill_fragment(c_frag, 0.0f);
rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

rocWMMA abstracts register packing, swizzle modes, and instruction selection
across CDNA2/CDNA3.

## HIP inline assembly

```cpp
__global__ void my_kernel(float* out, const float* in) {
    float val;
    // Raw inline AMDGPU assembly
    __asm__ volatile(
        "global_load_dword %0, %1, %2\n\t"
        "s_waitcnt vmcnt(0)"
        : "=v"(val)             // output: VGPR
        : "v"(in), "s"(desc)    // inputs: VGPR addr, SGPR descriptor
    );
    *out = val;
}
```

### Constraints
- `"v"` — VGPR operand (per-thread vector data)
- `"s"` — SGPR operand (uniform scalar data)
- `"=v"` — VGPR output
- `"=s"` — SGPR output
- `"vcc"`, `"scc"` — special register constraints

## Performance directives

AMDGPU assembly supports directives that influence resource allocation:

| Directive | Purpose |
|---|---|
| `.vgpr_max_limit N` | Max VGPRs used (0–255, 0=default) |
| `.sgpr_max_limit N` | Max SGPRs used |
| `.workgroup_size X, Y, Z` | Expected workgroup dimensions |
| `.workgroup_num_vgprs N` | Pre-allocate N VGPRs per thread |
| `.workgroup_num_sgprs N` | Pre-allocate N SGPRs |
| `.occupancy N` | Target occupancy (wavefronts/CU) |
| `.amdgpu_target "gfx942"` | Target ISA variant |

The compiler uses these to guide register allocation and occupancy heuristics.

## Commonly-used instructions cheat sheet

### Scalar ALU (uniform ops)
```
s_add_u32   s[dst], s[src0], s[src1]     ; 32-bit unsigned add
s_mov_b32   s[dst], s[src]               ; move
s_cmp_eq_u32 s[src0], s[src1]            ; compare (sets scc)
s_cbranch_scc1 label                     ; branch if scc==1
s_getpc_b64 s[dst]                       ; get program counter
s_and_b32   s[dst], s[src0], s[src1]     ; bitwise AND
```

### Vector ALU (SIMD ops)
```
v_add_f32   v[dst], v[src0], v[src1]     ; fp32 add
v_mul_f32   v[dst], v[src0], v[src1]     ; fp32 mul
v_fma_f32   v[dst], v[a], v[b], v[c]     ; fma: dst = a*b + c
v_mov_b32   v[dst], v[src]               ; move
v_cmp_gt_f32 vcc, v[src0], v[src1]       ; compare, sets vcc
v_cndmask_b32 v[dst], v[else], v[then], vcc  ; select by vcc
v_and_b32   v[dst], v[src0], v[src1]     ; bitwise AND
v_lshlrev_b32 v[dst], v[shift], v[src]   ; left shift
v_mbcnt_lo_u32_b32 v[dst], -1, 0         ; thread id in wavefront
```

### LDS (shared memory)
```
ds_write_b32  v[addr], v[data]           ; write 4B
ds_read_b32   v[dst], v[addr]            ; read 4B
ds_write_b128 v[addr], v[d0:d3]          ; write 16B (4 VGPRs)
ds_read_b128  v[d0:d3], v[addr]          ; read 16B
ds_swizzle_b32 v[dst], v[src], pattern   ; swizzle (permute)
ds_permute_b32 v[dst], v[addr], v[src]   ; permute across lanes
```

### Global memory
```
global_load_dword  v[dst], v[addr], s[rsrc]
global_store_dword v[addr], v[data], s[rsrc]
global_load_dwordx4 v[d0:d3], v[addr], s[rsrc]  ; 16B vectorized
global_store_dwordx4 v[addr], v[d0:d3], s[rsrc]
```

## Completion criterion

You can read and write AMDGPU inline assembly for common kernel patterns, name
the register type (SGPR/VGPR/special) of each operand, apply the correct
`s_waitcnt` before consuming asynchronously-loaded data, and choose between
raw MFMA instructions and rocWMMA for Matrix-Core operations.
