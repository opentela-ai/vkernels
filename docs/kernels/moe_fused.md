# MoE Fused — End-to-end MXFP4 Fused-MoE Grouped GEMM

Wires together the four low-level gfx942 primitives (`#12`–`#15` from
`moe.hip`) into an end-to-end Mixture-of-Experts layer that matches the
xkernels `fused_moe_mxfp4` interface. The kernel fuses dequantization,
two grouped GEMMs, SwiGLU activation, and a third grouped GEMM with
scatter-add combine — all in two kernel launches (stage 0 + stage 1).

- **Source (CPU)**: `src/c/vkernels/kernels/moe_fused.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/moe_fused.hip`
- **Header**: `src/c/vkernels/kernels/moe_fused.hpp`
- **Tests**: `tests/kernels/moe/test_moe_fused.cpp`

---

## Architecture

The kernel mirrors the Triton `moe_mxfp4_kernel.py` from xkernels exactly.
It runs in two stages:

```
Stage 0 — gate_up + SwiGLU:
  act[EM, ispp] = silu(clamp(A_sorted @ w13_gate + b13_gate, L))
                · clamp(A_sorted @ w13_up   + b13_up,   L)

Stage 1 — down + routed combine:
  out[M, hidden] += act @ w2^T · topk_w_sorted + b2
```

- **Weights** are stored in MXFP4 E2M1 format with ue8m0 per-group scales.
  Dequantization happens **inline during the K-loop** — no full bf16
  materialization of the weight buffer.
- **Activations** are bf16. The intermediate activation (stage 0 output)
  is written to a scratch buffer in bf16.
- **Output** is float32, accumulated with `atomicAdd` at the end of stage 1
  to handle multiple experts routing to the same token.

---

## Data formats

### Weights — MXFP4 E2M1 + ue8m0

| Weight | Shape | Element size | Packing |
|---|---|---|---|
| `w13` | `[E, 2·ispp, hidden/2]` | uint8 | 2 fp4 per byte |
| `w13_scale` | `[E, 2·ispp, hidden/group_size]` | uint8 | ue8m0 per group |
| `w2` | `[E, hidden, ispp/2]` | uint8 | 2 fp4 per byte |
| `w2_scale` | `[E, hidden, ispp/group_size]` | uint8 | ue8m0 per group |

`w13` holds both the gate and up projections interleaved: the first `ispp`
rows are the gate weight, the next `ispp` rows are the up weight.

### ue8m0 scale format

Unsigned 8-bit exponent, no mantissa. The decoded float value is:

```
if s == 0xFF: 0.0
else: 2^(s - 127)
```

Representable values: `2^-127` through `2^127`, plus zero. The ue8m0
scale is shared across `group_size` consecutive K elements (typically 32).

---

## Tile constants

| Constant | Value | Meaning |
|---|---|---|
| BLOCK_M (`kBM`) | 16 | Rows per output tile (expert-aligned tokens) |
| BLOCK_N (`kBN`) | 64 | Columns per output tile |
| BLOCK_K (`kBK`) | 64 | Inner dimension per K-tile |
| GROUP_SIZE (`kGroupSize`) | 32 | ue8m0 scale shared across 32 K elements |
| MFMA_K (`kMfmaK`) | 16 | K dimension of one MFMA instruction |
| WARPS (`kWarps`) | 4 | Warps per thread block |
| THREADS (`kWhreads`) | 256 | 4 warps × 64 lanes |

### Constraints

```
hidden % 64 == 0
ispp % 64 == 0
EM % 16 == 0
group_size == 32
```

---

## Stage 0 — gate_up + SwiGLU

### Grid

```
grid(EM/BLOCK_M, ispp/BLOCK_N)    # 2-D: M-blocks × N-blocks
```

### Per-block computation

For each K-block (`kBK=64`):

1. **Load A tile**: 16 rows × 64 columns of bf16 activations from the
   sorted token indices into shared memory `sA[16][64]`. The 256 threads
   cooperate: thread `tid` loads 4 consecutive bf16 values from its
   assigned token row (16 rows, 16 threads per row → each thread covers
   4 columns × 2 bytes = 8 bytes = `uint2` load).

2. **Dequant gate half of w13**: call `dequant_b_tile_device` to
   cooperatively dequantize `64×64` packed fp4 bytes with ue8m0 scales
   into shared memory `sB[64][64]` bf16. Each thread handles 8 packed
   bytes across 8 iterations (256 threads × 8 bytes = 2048 packed bytes
   → 4096 bf16 values).

3. **MFMA accumulation (gate)**: `__syncthreads()`, then 4 MFMA
   instructions (64÷16=4 K-steps). Each warp computes a 16×16 tile of the
   64-column output; the 4 warps together cover `warp×16` = columns `[0,16)`,
   `[16,32)`, `[32,48)`, `[48,64)`.

4. **Dequant up half of w13**: reload `sB` with the up-projection weights
   (these are at offset `ispp` rows in `w13`). Same dequant pattern.

5. **MFMA accumulation (up)**: `__syncthreads()`, then 4 more MFMA
   instructions.

After all K-blocks, the **SwiGLU epilogue** runs:

```
for each lane's 4 accumulators:
    g = gate_acc[i] + b13_gate  (if bias present)
    u = up_acc[i]   + b13_up    (if bias present)
    clamp(g, ±swiglu_limit); clamp(u, ±swiglu_limit)
    act = silu(g) · u
    write act (rounded to bf16) to act_scratch[token][col]
```

### Warp-to-column mapping within BLOCK_N

```
warp 0 → columns [0,  16)
warp 1 → columns [16, 32)
warp 2 → columns [32, 48)
warp 3 → columns [48, 64)
```

Within each 16-column warp tile, each lane writes to a 4×4 sub-tile:

```
Lane row = lane / 4         (0..3 within the warp's 16 rows)
Lane col base = (lane % 4) * 4   (0, 4, 8, or 12 within the 16 columns)
```

The 4 accumulators `C[0..3]` are stored at column offsets `+0, +1, +2, +3`
from the lane's base column.

---

## Stage 1 — down + combine

### Grid

```
grid(EM/BLOCK_M, hidden/BLOCK_N)
```

### Per-block computation

Identical K-tile loop structure to stage 0, but:

- **A** is the activation scratch `[EM, ispp]` in bf16 (stage 0 output).
- **B** is `w2` `[E, hidden, ispp]` — the down-projection weights.
- **Output** `[M, hidden]` is `float32`.

After all K-blocks, the **combine epilogue**:

```
for each lane's 4 accumulators:
    v = acc[i] + b2[expert][col]  (if bias present)
    v *= topk_w_sorted[token]
    atomicAdd(&out[token][col], v)
```

The `atomicAdd` is mandatory because multiple experts can route to the
same token row (when `top_k > 1`). The `expert_ids[block] == -1` guard
skips blocks that are pure padding.

---

## Expert alignment: `moe_align_block_size`

Before the fused kernel runs, the `[M, top_k]` token→expert routing table
must be converted into the block-aligned sorted layout.

### Computation

```
Input:  topk_ids [M][top_k]    — which expert each (token, selection) maps to
Output: sorted_ids [EM_padded] — token indices grouped by expert, padded to
         block_size multiples
        expert_ids [EM_padded/block_size] — expert id per block (-1 = padding)
Returns: EM_padded
```

### Algorithm

1. **Group**: collect token indices per expert.
2. **Pad**: each expert's token list is padded to a multiple of `block_size`
   (typically BLOCK_M = 16).
3. **Concatenate**: write all experts' token lists (including padding with
   token 0) into `sorted_ids`.
4. **Tag**: fill `expert_ids` with one entry per block: the expert id if
   that block contains at least one real token, `-1` if it's pure padding.

Empty experts contribute zero blocks. The total number of output blocks
is `EM_padded / block_size`.

### Example

```
M=8, top_k=4, E=4, block_size=16

Expert 0: tokens 0×4 + token 1×1 = 5 tokens → padded to 16
Expert 1: 11 tokens → padded to 16
Expert 2: 16 tokens → exactly 16 (no padding needed)
Expert 3: 0 tokens → skipped

sorted_ids  = [0,0,0,0,1, 0,0,...,0,  1,1,...,1,  4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7]
              |-- expert 0 16 --|  |expert 1 16|  |-- expert 2 16 --|

expert_ids  = [0, 1, 2]   (one per occupied block)
EM_padded   = 48
```

---

## HIP implementation details

### Shared-memory usage

| Stage | Buffer | Size |
|---|---|---|
| Stage 0 | `sA[16][64]` bf16 | 2048 bytes |
| Stage 0 | `sB[64][64]` bf16 | 8192 bytes |
| Stage 0 | **Total** | **10240 bytes** |
| Stage 1 | `sA[16][64]` bf16 | 2048 bytes |
| Stage 1 | `sB[64][64]` bf16 | 8192 bytes |
| Stage 1 | **Total** | **10240 bytes** |

At 64 KB LDS per CU on gfx942, the 10 KB footprint allows ~6 concurrent
wavefronts, which is well above the occupancy limit imposed by VGPR
usage.

### Inline dequantization

The `dequant_b_tile_device` function decodes E2M1 + ue8m0 to bf16
cooperatively:

- 256 threads cooperatively dequantize 2048 packed bytes → 4096 bf16
  values.
- Each thread handles 8 packed bytes across 8 iterations.
- The mapping: `tid = n + kp_base · 64`, so consecutive threads in a
  warp handle consecutive `n` values (coalesced writes to shared memory).

### A-tile load (stage 0)

Each thread loads a `uint2` (8 bytes = 4 bf16 values) from global memory:

```
Thread tid:  row = tid % 16,  col_off = (tid / 16) * 4
Loads A[token][k_start + col_off .. col_off+3] as uint2
→ writes to sA[row][col_off .. col_off+3] as two uint32_t
```

All threads write to different LDS rows → zero bank conflicts on the store.

### A-tile load (stage 1)

Stage 1 uses a simpler linear mapping:

```
el = tid * 4          (linear element index into the 16×64 tile)
for each of 4 elements: m = (el+e)/64, k = (el+e)%64
```

This is less optimized than stage 0's pattern but sufficient for the
baseline.

### MFMA invocation

Each K-step issues one `v_mfma_f32_16x16x16bf16_1k` per warp. With 4
K-steps per K-tile (64÷16) and 2 halves (gate + up), stage 0 issues
**8 MFMA instructions per warp per K-tile**. Stage 1 issues **4 MFMA
instructions per warp per K-tile** (only one weight matrix).

---

## CPU reference

The CPU reference (`fused_moe_mxfp4_cpu`) mirrors the HIP kernel block by
block:

1. Three nested loops over M-blocks, N-blocks, K-blocks.
2. Inside each K-block: dequantize a `[BLOCK_K, BLOCK_N]` tile of weights
   to bf16, load a `[BLOCK_M, BLOCK_K]` tile of activations, and compute
   the sub-GEMM with three nested loops.
3. After the K-loop: apply SwiGLU (stage 0) or bias+scale+scatter (stage 1).

The CPU reference is ~500 lines of straight-line C++ and serves as the
golden oracle for testing the HIP kernel.

---

## Contract

| Condition | Error / Behavior |
|---|---|
| `hidden % 64 != 0` or `ispp % 64 != 0` | Undefined (constraint assumed by caller) |
| `EM % 16 != 0` | Undefined |
| `group_size != 32` | Undefined (may produce incorrect results) |
| `expert_ids[i] == -1` | Block skipped (no outputs written for that block) |
| `b13 == nullptr` or `b2 == nullptr` | Bias skipped for that stage |
| `swiglu_limit <= 0` | No clamping applied |
| `topk_w_sorted[i] == 0.0` | Token contributes zero to output |

---

## Performance characteristics

- **Weights**: ~138 GB in bf16 for a typical 256-expert, 8192-hidden,
  2048-ispp configuration, but stored in MXFP4 + ue8m0 (~35 GB).
- **Arithmetic intensity**: The kernel is **compute-bound** — the inner
  loops are dominated by MFMA instructions (2 FLOP per multiply-add per
  bf16 element per K-step).
- **LDS bandwidth**: Each K-tile dequant writes 4096 bf16 values to shared
  memory (8 KB), read twice (gate + up in stage 0). The dequant step is
  purely ALU + LDS, no global-memory traffic for weights after the initial
  packed load.
- **No LDS double-buffering yet**: compute stalls waiting for dequant and
  `__syncthreads()`. Double-buffering `sB` would hide dequant latency
  behind the previous tile's MFMA.
- **No wavefront specialization**: all wavefronts do both dequant and MMA.
  Splitting into producer warps (dequant) and consumer warps (MFMA) would
  hide dequant latency entirely.

---

## File layout

```
src/c/vkernels/kernels/
├── moe_fused.hpp       # public API (fused_moe_mxfp4_cpu, moe_align_block_size)
├── moe_fused.cpp       # CPU reference (always compiled, ~500 lines)
└── moe_fused.hip       # HIP kernel + hip::fused_moe_mxfp4 launcher (VKERNELS_HAS_HIP)
```
