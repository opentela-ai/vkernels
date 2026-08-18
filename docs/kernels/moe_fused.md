# MoE Fused — End-to-end MXFP4 Fused-MoE Grouped GEMM

Wires together the four low-level gfx942 primitives (`#12`–`#15` from
`moe.hip`) into an end-to-end Mixture-of-Experts layer that matches the
xkernels `fused_moe_mxfp4` interface. The kernel fuses dequantization, the
gate+up grouped GEMM, SwiGLU activation, and the down grouped GEMM with a
routed scatter-add combine — all in **two kernel launches** (plus a tiny
weight-gather kernel).

- **Source (CPU)**: `src/c/vkernels/kernels/moe_fused.cpp`
- **Source (HIP)**: `src/c/vkernels/kernels/moe_fused.hip`
- **Header**: `src/c/vkernels/kernels/moe_fused.hpp`
- **Tests**: `tests/kernels/moe/test_moe_fused.cpp` (host),
  `meta/benchmarks/test_moe_fused_correct.hip` (GPU vs CPU oracle, decode),
  `meta/benchmarks/test_moe_fused_prefill_correct.hip` (GPU vs CPU oracle, prefill)

---

## Architecture

The kernel mirrors the xkernels 2-stage Triton `moe_mxfp4_kernel.py`
exactly (including the single bf16 rounding of the intermediate):

```
Kernel 0 — gate_up + SwiGLU  (gateup_swiglu_kernel):
  act[EM, ispp] = silu(clamp(A_sorted @ w13_gate + b13_gate, L))
                · clamp(A_sorted @ w13_up   + b13_up,   L)

Kernel 1 — down + routed combine  (down_combine_kernel):
  out[M, hidden] += act @ w2^T · topk_w_sorted + b2
```

- **Weights** are MXFP4 E2M1 with ue8m0 per-group scales. Dequantization
  happens **inline during the K-loop** — no full bf16 materialization of the
  ~138 GB expert weight buffer.
- **Gate and up are fused in one kernel**: both fp32 accumulators are held
  across the K-loop, and the SwiGLU product is rounded to bf16 **once** in
  the epilogue. Splitting into separate kernels introduced a spurious
  intermediate bf16 rounding that no longer matches the oracle.
- **The gate is clamped to `L` before the sigmoid** (SwiGLU clamp); up is
  clamped symmetrically to `±L`.
- **Activations** are bf16. The stage-0 output lives in a `[EM, ispp]`
  bf16 scratch buffer.
- **Output** is float32. Stage 1 uses `atomicAdd` so multiple experts
  routing to the same token (`top_k > 1`) accumulate correctly. The caller
  must zero-initialise `out`.

---

## Data formats

### Weights — MXFP4 E2M1 + ue8m0

| Weight | Shape | Element size | Packing |
|---|---|---|---|
| `w13` | `[E, 2·ispp, hidden/2]` | uint8 | 2 fp4 per byte |
| `w13_scale` | `[E, 2·ispp, hidden/group_size]` | uint8 | ue8m0 per group |
| `w2` | `[E, hidden, ispp/2]` | uint8 | 2 fp4 per byte |
| `w2_scale` | `[E, hidden, ispp/group_size]` | uint8 | ue8m0 per group |

`w13` holds gate and up interleaved: the first `ispp` rows are the gate
weight, the next `ispp` rows are the up weight.

E2M1 is `sign | 2-bit-exponent | 1-bit-mantissa`, two values per byte, low
nibble first. ue8m0 is an unsigned 8-bit exponent with no mantissa:
`2^(s-127)` (or `0.0` when `s == 0xFF`), shared across `group_size`
consecutive K elements.

---

## Tile constants

Two tile configs are compiled in, selected by the `block_size` argument to
`hip::fused_moe_mxfp4` (16 = decode, 64 = prefill):

| Constant | Decode | Prefill | Meaning |
|---|---:|---:|---|
| BLOCK_M | 16 (`kBM`) | 64 (`kBM_pf`) | Rows per output tile (expert-aligned sorted rows) |
| BLOCK_N | 64 (`kBN`) | 64 (`kBN_pf`) | Columns per output tile |
| BLOCK_K | 64 (`kBK`) | 64 (`kBK_pf`) | Inner dimension per K-tile |
| GROUP_SIZE (`kGroupSize`) | 32 | 32 | ue8m0 scale shared across 32 K elements |
| MFMA_K (`kMfmaK`) | 16 | 16 | K dimension of one MFMA instruction |
| THREADS | 64 (`kWhreads`) | 256 (`kThreads_pf`) | Threads per block |

### Constraints

```
decode:  hidden % 64 == 0,  ispp % 64 == 0,  EM % 16 == 0
prefill: hidden % 64 == 0,  ispp % 64 == 0,  EM % 64 == 0
group_size == 32
```

---

## Thread-block decomposition

### Decode (64 threads, 16×64 tile)

Each block is **64 threads (one wavefront)**. The N dimension (64 columns)
is split across `blockIdx.z ∈ {0,1,2,3}`; each z-slice owns a 16-column
slice. This fixed-64-thread shape is deliberate: hand-pinned VGPRs and
>64-thread blocks crash on gfx90a ("Memory access fault"), so the
multi-warp N-split is moved to the grid instead of the block.

```
grid  = (EM / BLOCK_M,  N / BLOCK_N,  4)
block = 64 threads
n_global = blockIdx.y * 64 + blockIdx.z * 16     (16-column slice)
```

The warp-to-column mapping is therefore `warp_n = blockIdx.z`, and each
lane within a wavefront covers one 16×16 MFMA tile:

```
col       = lane % 16
row_base  = (lane / 16) * 4          (rows row_base .. row_base+3)
```

### Prefill (256 threads, 64×64 tile)

Four wavefronts per block; wavefront `w` owns the 16-row fragment
`rows w*16 .. w*16+15` and loops over **4 column fragments** (64 columns),
accumulating `acc[4]` (one `v4f` per column fragment). The gate and up
weights share one `sB[64][64]` buffer (dequant gate → MFMA → dequant up →
MFMA) to keep LDS at 16 KB. The caller must align with `block_size=64`, so
`expert_ids` is indexed per 64-row block and each block is a single expert.

BN is 64, not 128: BN=128 costs 24 KB LDS and 64 accumulator VGPRs, capping
the kernel at 2 blocks/CU; BN=64 (16 KB LDS, 32 accumulator VGPRs) lifts it
to 3 blocks/CU, which measures ~1.5× faster despite halving the B-tile
reuse (the kernel is occupancy-bound, not reuse-bound).

```
grid  = (EM / 64,  N / 64)        (N = ispp for stage 0, hidden for stage 1)
block = 256 threads = 4 wavefronts
warp  = tid / 64,  lane = tid % 64
```

---

## MFMA fragment layout

The kernel issues `v_mfma_f32_16x16x16bf16_1k` through the clang builtin
`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` (not inline asm — the builtin
keeps register allocation correct across ROCm versions, where the pinned-VGPR
inline-asm form corrupted accumulators and crashed for >64 threads).

```
A: m = lane % 16,      a[i] = A[m][k0+i]      (k0 = (lane/16)*4)
B: n = lane % 16,      b[i] = B[k0+i][n]      (k0 = (lane/16)*4)
C: col = lane % 16,    row = (lane/16)*4 + i,  c[i] = C[row][col]
```

The C layout is **4 consecutive rows × 1 column** per lane (verified
empirically with one-hot matrices on gfx90a) — not a 4×4 sub-tile.

---

## Stage 0 — `gateup_swiglu_kernel`

### Per-block computation

For each K-block (`kBK=64`):

1. **Load A tile**: 64 threads load the 16×64 bf16 activation tile into
   shared memory `sA[16][64]`. Each thread handles one row (`lane % 16`)
   and loads 16 bf16 = 4 × `uint2` reads from columns `(lane/16)*16 + {0,4,8,12}`.
   Padding rows (flat index `>= M*top_k`) are zero-filled — never skipped,
   to avoid stale `sA` data.
2. **Dequant gate half of w13** into `sB_gate[64][16]`, and **up half** into
   `sB_up[64][16]` (both are 16-column slices, one per lane). `dequant_b_16cols`
   decodes E2M1 + ue8m0 inline; each thread dequantizes 8 packed bytes.
3. **MFMA accumulate**: after `__syncthreads()`, 4 K-steps of K=16 each,
   accumulating both `acc_g` and `acc_u` (two `v4f` fp32 accumulators).

### Epilogue

```
for each lane's 4 accumulator rows i:
    flat = sorted_ids[tk_base + row_base + i]
    if flat < M*top_k:                       (real row, not padding)
        g = acc_g[i] + b13_gate              (if bias)
        g = clamp(g, -L, +L) before sigmoid  (SwiGLU clamp)
        u = acc_u[i] + b13_up                (if bias)
        u = clamp(u, -L, +L)
        act[tk_base + row_base + i][n_global + col] = bf16(silu(g) * u)
```

The act write is indexed by **sorted row** (`tk_base + row`), not token —
each expert's intermediate occupies its own row, so a token routed to
multiple experts never races on the scratch buffer.

---

## Stage 1 — `down_combine_kernel`

Identical K-tile loop structure, but:

- **A** is the act scratch `[EM, ispp]` (bf16), loaded by sorted row.
- **B** is `w2` `[E, hidden, ispp/2]` — the down-projection weights.
- **Output** `[M, hidden]` is float32.

### Epilogue

```
for each lane's 4 accumulator rows i:
    flat = sorted_ids[tk_base + row_base + i]
    if flat < M*top_k:
        token = flat / top_k               (recover the token from the flat index)
        v = acc[i] + b2[expert][n_global + col]   (if bias)
        v *= topk_w_sorted[tk_base + row_base + i]  (routing weight)
        atomicAdd(&out[token][n_global + col], v)
```

`atomicAdd` is required because multiple experts route to the same token
row. `expert_ids[block] == -1` (pure-padding blocks) return immediately.

---

## Weight gather

The xkernels host API takes the raw `[M, top_k]` routing weight matrix. A
tiny `gather_weights_kernel` reorders it into `sorted_w[EM]` matching
`sorted_ids` before stage 1:

```
sorted_w[i] = (sorted_ids[i] in [0, M*top_k)) ? topk_w[sorted_ids[i]] : 0
```

`sorted_w` lives in a **caller-provided scratch buffer** (`sorted_w_scratch`,
`EM` floats), not a per-call `hipMalloc`. A backend serving a 61-layer model
allocates `act_scratch` and `sorted_w_scratch` once and reuses them across
every forward pass, eliminating the 122 allocator round-trips per generated
token that a per-call allocation cost (issue #41, item 1). The
`hip::fused_moe_mxfp4` launcher therefore performs no device allocation of
its own.

---

## Expert alignment: `moe_align_block_size`

Maps the `[M, top_k]` token→expert routing table into the block-aligned
sorted layout the grouped GEMM consumes.

```
Input:  topk_ids [M][top_k]        — which expert each (token, sel) maps to
Output: sorted_ids [EM_padded]     — FLAT topk indices (token*top_k + sel),
                                      grouped by expert, per-expert padded
        expert_ids [EM_padded/B]   — expert id per block (-1 = padding)
Returns: EM_padded
```

The flat index (not the token) is stored so that the weight gather keeps
the selection index for `top_k > 1`. Consumers recover `token = flat / top_k`
and treat `flat >= M*top_k` as padding.

### Example

```
M=8, top_k=4, E=4, block_size=16
N = M*top_k = 32   (the padding sentinel)

Expert 0: flat [0,1,2,3,4] (token 0 ×4 sels, token 1 sel 0) → padded to 16
Expert 1: flat [5..15]     (11 entries)                     → padded to 16
Expert 2: flat [16..31]    (16 entries, no padding)
Expert 3: empty → skipped

sorted_ids = [0,1,2,3,4, 32×11,  5..15, 32×5,  16..31]
expert_ids = [0, 1, 2]
EM_padded  = 48
```

---

## CPU reference

`fused_moe_mxfp4_cpu` mirrors the HIP kernel block by block (M-blocks ×
N-blocks × K-blocks, dequant → sub-GEMM → epilogue) and is the golden oracle
for the GPU path. It takes `top_k` explicitly and `sorted_ids` in flat-index
form.

---

## Contract

| Condition | Behavior |
|---|---|
| decode: `hidden % 64 != 0` or `ispp % 64 != 0` | Undefined (caller constraint) |
| prefill: `hidden % 128 != 0` or `ispp % 128 != 0` | Undefined |
| `EM % block_size != 0` | Undefined |
| `group_size != 32` | Undefined (incorrect results) |
| `expert_ids[i] == -1` | Block skipped |
| `b13 == nullptr` / `b2 == nullptr` | Bias skipped for that stage |
| `swiglu_limit <= 0` | No clamping |
| `out` not zero-initialised | Undefined (stage 1 accumulates into it) |
| `act_scratch`/`sorted_w_scratch` not caller-owned | Undefined — the launcher does no per-call allocation (issue #41, item 1); both scratch buffers are owned and reused by the caller |

---

## Verified performance

> **Full benchmark records** (reproduce commands, per-M tables, primitives,
> caveats, journal): [`docs/performance/moe-fused/gfx90a.md`](../performance/moe-fused/gfx90a.md)
> (MI250X) and [`docs/performance/moe-fused/gfx942.md`](../performance/moe-fused/gfx942.md)
> (MI300A).

E=256, hidden=4096, ispp=512, top_k=6. Latency in ms (lower is better);
TFLOP/s is arithmetic on the **padded** EM rows, so the decode regime is
heavily padding-dominated (e.g. M=1 has 6 real rows but 96 padded rows).

| M | vkernels HIP (MI250X) | vkernels HIP (MI300A) | xkernels torch (MI250X) | xkernels torch (MI300A) |
|---|---:|---:|---:|---:|
| 1  | 0.66 | 0.45 | 6.0  | 4.4  |
| 8  | 3.4  | 0.89 | ~30  | ~15  |
| 16 | 4.1  | 2.2  | ~40  | ~30  |
| 48 | 10.4 | 4.0  | 162.2 | 127.2 |

GPU results are exact matches against the CPU reference
(`max_rel < 0.00001` decode, `< 0.02` prefill) on both gfx90a and gfx942.

### Prefill vs decode (E=8, hidden=4096, ispp=512, top_k=2)

Effective TFLOP/s on the **padded** EM rows.

MI250X (gfx90a):

| M | decode (µs) | decode TFLOP/s | prefill (µs) | prefill TFLOP/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 128  | 958  | 1.68 | 1175 | 1.37 | 0.81× |
| 256  | 1356 | 2.38 | 1201 | 2.68 | 1.13× |
| 512  | 2697 | 2.39 | 1945 | 3.31 | 1.39× |
| 1024 | 5151 | 2.50 | 2993 | 4.31 | 1.72× |
| 2048 | 10236 | 2.52 | 5781 | 4.46 | 1.77× |

MI300A (gfx942):

| M | decode (µs) | decode TFLOP/s | prefill (µs) | prefill TFLOP/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 128  | 557  | 2.89 | 724  | 2.22 | 0.77× |
| 256  | 722  | 4.46 | 737  | 4.37 | 0.98× |
| 512  | 1209 | 5.33 | 804  | 8.02 | 1.50× |
| 1024 | 2352 | 5.48 | 1303 | 9.89 | 1.81× |
| 2048 | 4982 | 5.17 | 1981 | 13.01 | 2.51× |

The prefill config wins once each expert fills its 64-row block (M ≥ 256 on
gfx90a, M ≥ 512 on gfx942 here); below that, 64-row padding waste dominates
and the decode config is faster. Both configs sit far below the MFMA roof
because dequant and MFMA are serialised by `__syncthreads()` and the kernel
is occupancy-bound (prefill: ~3 blocks/CU at 16 KB LDS + 32 accumulator
VGPRs; the earlier BN=128 prefill was ~2 blocks/CU at 24 KB LDS + 64
accumulator VGPRs and ~1.5× slower).

---

## Current limitations / future work

- **Persistent scratch buffer (issue #41, item 1 — done)**: `act_scratch`
  and `sorted_w_scratch` are now caller-provided; the `hip::fused_moe_mxfp4`
  launcher performs no `hipMalloc`/`hipFree` of its own, so a 61-layer model
  no longer pays 122 allocator round-trips per generated token. Items 2–5 of
  issue #41 (warp-level segmented combine replacing `atomicAdd`, an
  occupancy-first tile sweep with `__launch_bounds__`, a K-major weight
  layout study, and a wavefront-specialized producer/consumer prefill)
  remain open — all are GPU-kernel changes that require gfx942 (MI300A)
  hardware to implement and verify, which is unavailable in this tree's
  CPU-only build.
- **Occupancy-bound, not barrier-bound** (measured on gfx90a): LDS
  double-buffering was implemented and **reverted** — doubling the decode
  LDS (6→12 KB) halved blocks/CU and slowed decode by ~50%, because the
  extra concurrent warps provide more latency hiding than the load→MFMA
  overlap does. The redundant barrier between the A-tile load and the
  weight dequant *was* removed (they write different LDS buffers); that is
  a free, small win. Acting on this finding, the prefill BN was reduced
  128→64 (24→16 KB LDS, 64→32 accumulator VGPRs), lifting occupancy from 2
  to 3 blocks/CU and measuring ~1.5× faster — confirming that the lever is
  **more occupancy** (fewer VGPRs / less LDS per block), not deeper
  software pipelining.
- **No wavefront specialization**: every wavefront both dequantizes and
  issues MFMA. A producer/consumer split could overlap them without
  doubling LDS, but costs VGPRs and is only worth it once occupancy is
  already saturated.
- **Prefill is now a solid win** (BN=64): from M ≥ 256 on gfx90a (1.13×)
  and M ≥ 512 on gfx942 (1.50×), rising to 1.77× and 2.51× at M=2048.
  The 64-row block still wastes compute when routing is sparse
  (tokens/expert < 64); a smaller prefill tile (e.g. BM=32) or a
  padding-aware launch heuristic could close the remaining small-M gap.
- **Padding overhead in decode**: per-expert block padding (16 rows) makes
  small-M latency scale with the number of routed experts, not the number
  of tokens.

---

## File layout

```
src/c/vkernels/kernels/
├── moe_fused.hpp       # public API (fused_moe_mxfp4_cpu, moe_align_block_size)
├── moe_fused.cpp       # CPU reference (always compiled)
├── moe_fused.hip       # 2 HIP kernels + gather + hip::fused_moe_mxfp4 launcher
└── moe_device.hip      # shared device helpers (E2M1/ue8m0 decode, bf16 rounding)
```
