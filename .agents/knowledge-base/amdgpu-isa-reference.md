# AMDGPU ISA Quick Reference

Core instructions for writing AMD GPU inline assembly in HIP kernels.
Targets gfx942 (CDNA3/MI300X) and gfx90a (CDNA2). For full detail, load the
[amdgpu-isa skill](../skills/amdgpu-isa/SKILL.md).

## Register types

| Type | Syntax | Width | Shared by | Use |
|---|---|---|---|---|
| SGPR | `s0`–`s103` | 32-bit | Wavefront (64 threads) | Addresses, constants, control |
| VGPR | `v0`–`v255` | 32-bit | Per-thread | SIMD data, MFMA operands/accumulators |
| Acc VGPR | `acc0`–`acc255` | 32-bit | Per-thread | MFMA accumulators (CDNA) |
| vcc | `vcc` | 64-bit SGPR pair | Per-thread | Vector condition code |
| scc | `scc` | 1-bit | Wavefront | Scalar condition code |
| exec | `exec` | 64-bit | Wavefront | Execution mask |
| m0 | `m0` | 32-bit SGPR | Wavefront | LDS addressing control |

## Wait counters

```
s_waitcnt vmcnt(N)       ; wait for vector memory counter ≤ N
s_waitcnt lgkmcnt(N)     ; wait for LDS/GPR/constant counter ≤ N
s_waitcnt expcnt(N)      ; wait for export counter ≤ N
s_waitcnt vmcnt(0) & lgkmcnt(0)   ; wait for both
```

**HIP intrinsic:** `__builtin_amdgcn_s_waitcnt(vmcnt, lgkmcnt, expcnt)`
- `__builtin_amdgcn_s_waitcnt(0, 0, 0)` — wait for all in-flight ops
- `__builtin_amdgcn_s_waitcnt(0, 0x7F, 0)` — drain LDS only
- `__builtin_amdgcn_s_waitcnt(1, 0, 0)` — allow 1 in-flight vector op

## Memory instructions

### Global memory (HBM)

```
; Scalar loads (32-bit)
global_load_dword    v[dst], v[addr], s[rsrc]         ; 4 bytes
global_load_dwordx2  v[dst:dst+1], v[addr], s[rsrc]   ; 8 bytes
global_load_dwordx4  v[dst:dst+3], v[addr], s[rsrc]   ; 16 bytes

; Scalar stores
global_store_dword   v[addr], v[data], s[rsrc]
global_store_dwordx2 v[addr], v[data:data+1], s[rsrc]
global_store_dwordx4 v[addr], v[data:data+3], s[rsrc]
```

### LDS (shared memory)

```
; 32-bit
ds_read_b32   v[dst], v[addr]
ds_write_b32  v[addr], v[data]

; 64-bit
ds_read_b64   v[dst:dst+1], v[addr]
ds_write_b64  v[addr], v[data:data+1]

; 128-bit (widest practical)
ds_read_b128  v[dst:dst+3], v[addr]
ds_write_b128 v[addr], v[data:data+3]

; Swizzle
ds_swizzle_b32 v[dst], v[src], pattern
```

### Buffer (scratch)

```
buffer_load_dword   v[dst], v[offset], s[rsrc:rsrc+3]
buffer_store_dword  v[data], v[offset], s[rsrc:rsrc+3]
```

### Flat (generic pointer)

```
flat_load_dword   v[dst], v[addr]
flat_store_dword  v[addr], v[data]
```

## Scalar ALU (uniform ops)

```
s_add_u32      s[dst], s[src0], s[src1]     ; add
s_sub_u32      s[dst], s[src0], s[src1]     ; subtract
s_mul_i32      s[dst], s[src0], s[src1]     ; multiply (32-bit)
s_mov_b32      s[dst], s[src]               ; move
s_and_b32      s[dst], s[src0], s[src1]     ; bitwise AND
s_or_b32       s[dst], s[src0], s[src1]     ; bitwise OR
s_xor_b32      s[dst], s[src0], s[src1]     ; bitwise XOR
s_lshl_b32     s[dst], s[src0], s[src1]     ; left shift
s_lshr_b32     s[dst], s[src0], s[src1]     ; logical right shift
s_ashr_i32     s[dst], s[src0], s[src1]     ; arithmetic right shift
s_cmp_eq_u32   s[src0], s[src1]             ; compare equal (sets scc)
s_cmp_gt_i32   s[src0], s[src1]             ; compare greater (sets scc)
s_cmp_lt_u32   s[src0], s[src1]             ; compare less (sets scc)
s_cbranch_scc0 label                        ; branch if scc==0
s_cbranch_scc1 label                        ; branch if scc==1
s_getpc_b64    s[dst:dst+1]                 ; program counter
s_abs_i32      s[dst], s[src]               ; absolute value
s_min_u32      s[dst], s[src0], s[src1]     ; unsigned min
s_max_i32      s[dst], s[src0], s[src1]     ; signed max
```

## Vector ALU (SIMD, 64-wide)

```
v_add_f32      v[dst], v[src0], v[src1]     ; fp32 add
v_sub_f32      v[dst], v[src0], v[src1]     ; fp32 subtract
v_mul_f32      v[dst], v[src0], v[src1]     ; fp32 multiply
v_fma_f32      v[dst], v[a], v[b], v[c]     ; fp32 fma (a*b + c)
v_mov_b32      v[dst], v[src]               ; move
v_and_b32      v[dst], v[src0], v[src1]     ; bitwise AND
v_or_b32       v[dst], v[src0], v[src1]     ; bitwise OR
v_xor_b32      v[dst], v[src0], v[src1]     ; bitwise XOR
v_lshlrev_b32  v[dst], v[shift], v[src]     ; left shift
v_lshrrev_b32  v[dst], v[shift], v[src]     ; right shift
v_cmp_gt_f32   vcc, v[src0], v[src1]        ; compare fp32 > (sets vcc)
v_cmp_eq_u32   vcc, v[src0], v[src1]        ; compare u32 == (sets vcc)
v_cndmask_b32  v[dst], v[else], v[then], vcc  ; select by vcc
v_max_f32      v[dst], v[src0], v[src1]     ; fp32 maximum
v_min_f32      v[dst], v[src0], v[src1]     ; fp32 minimum
v_exp_f32      v[dst], v[src]               ; 2^src (base-2 exponent)
v_log_f32      v[dst], v[src]               ; log2(src)
v_rcp_f32      v[dst], v[src]               ; reciprocal
v_sqrt_f32     v[dst], v[src]               ; square root
v_rsq_f32      v[dst], v[src]               ; reciprocal sqrt
```

## MFMA instructions (Matrix Cores)

### CDNA2 (gfx90a)

```
; fp16 input, fp32 accumulate — tile: 16×16 output, K=16 reduction
v_mfma_f32_16x16x16f16   v[acc:acc+3], v[a:a+1], v[b:b+1], v[acc:acc+3]

; bf16 input, fp32 accumulate
v_mfma_f32_16x16x16bf16  v[acc:acc+3], v[a:a+1], v[b:b+1], v[acc:acc+3]

; fp32 input, fp32 accumulate (32×32 output, K=2)
v_mfma_f32_32x32x2f32    v[acc:acc+15], v[a:a+1], v[b:b+1], v[acc:acc+15]

; fp64 input, fp64 accumulate (16×16 output, K=4)
v_mfma_f64_16x16x4f64    v[acc:acc+7], v[a:a+3], v[b:b+3], v[acc:acc+7]
```

### CDNA3 (gfx942)

```
; fp16, same instruction as CDNA2
v_mfma_f32_16x16x16f16   v[acc:acc+3], v[a:a+1], v[b:b+1], v[acc:acc+3]

; fp8 input, fp32 accumulate (16×16 output, K=32)
v_mfma_f32_16x16x32_f8f6f4  v[acc], v[a], v[b], v[acc]

; bf8 input, fp32 accumulate
v_mfma_f32_16x16x32_bf8f6f4  v[acc], v[a], v[b], v[acc]
```

### MFMA naming convention

```
v_mfma_<acc_type>_<M>x<N>x<K>_<input_type>
       ^^^^^^^^  ^^^^^^^^^^^  ^^^^^^^^^^
       fp32/fp64  output tile  fp16/bf16/fp32/fp64/f8/bf8
                  M rows × N cols, K reduction
```

## Lane ID and thread indexing

```cpp
// In HIP C++:
int lane_id = __lane_id();  // 0-63 within wavefront
int wf_id   = __lane_id() / 64;  // wavefront index within workgroup

// In assembly:
v_mbcnt_lo_u32_b32 v[lane_lo], -1, 0   ; low 32 lanes
v_mbcnt_hi_u32_b32 v[lane_hi], -1, v[lane_lo]  ; high 32 lanes
```

## Control flow

```
s_branch label          ; unconditional branch
s_cbranch_scc0 label    ; branch if scc == 0
s_cbranch_scc1 label    ; branch if scc == 1
s_cbranch_vccz label    ; branch if all vcc bits == 0
s_cbranch_vccnz label   ; branch if any vcc bit != 0
s_cbranch_execnz label  ; branch if any exec bit != 0
s_setpc_b64 s[addr]     ; indirect jump (function return)
```

## Performance directives

```
.vgpr_max_limit 128          ; max VGPRs per thread
.sgpr_max_limit 80           ; max SGPRs per wavefront
.workgroup_size 256, 1, 1    ; expected workgroup dimensions
.workgroup_num_vgprs 128     ; pre-allocate VGPRs
.workgroup_num_sgprs 80      ; pre-allocate SGPRs
.occupancy 4                  ; target occupancy (wavefronts/CU)
```

## HIP inline assembly template

```cpp
float val;
const float* in_ptr = ...;
__asm__ volatile(
    "global_load_dword %0, %1, %2\n\t"
    "s_waitcnt vmcnt(0)"
    : "=v"(val)                        // outputs: VGPR
    : "v"(in_ptr), "s"(desc)           // inputs: VGPR, SGPR
    : "memory"                         // clobbers
);
```

### Constraints

| Constraint | Register type |
|---|---|
| `"v"` | VGPR (32-bit) |
| `"s"` | SGPR (32-bit) |
| `"=v"` | VGPR output |
| `"=s"` | SGPR output |
| `"vcc"` | Vector condition code |
| `"scc"` | Scalar condition code |
