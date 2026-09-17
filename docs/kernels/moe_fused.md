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

### Prefill variants (issue #76)

The prefill path is selected by `VK_MOE_PF_VARIANT` (read once into a
function-local `static`).  **The default is 5.**

| variant | gateup | down | what it is |
|---|---|---|---|
| 0 | `gateup_swiglu_kernel_prefill` | `down_combine_kernel_prefill` | baseline: 64×64 tile, A staged in LDS, gate/up share one `sB` |
| 1 | `_prefill_pipe<64>` | `_prefill_pipe<64>` | register-staged A (never through LDS on the load path) + 2-halfword-padded `sA` rows; one barrier pair per K-block instead of two |
| 2 | `_prefill_pipe<32>` | `_prefill_pipe<64>` | variant 1 with a 32-wide gateup N tile |
| 3 | `_prefill_ws<32>` | `_prefill_pipe<64>` | producer/consumer wavefront specialisation: 8 warps, 0-3 MFMA-only, 4-7 stage+dequant only |
| 4 | `_prefill_pipe<16>` | `_prefill_pipe<64>` | variant 2 with a 16-wide gateup N tile |
| 5 | `_prefill_pipe<16>` | `_prefill_pipe<32>` | variant 4 plus a 32-wide down N tile |

**Why the N tile, and not the pipeline, is the lever.** At the prefill shape
(E=8, hidden=4096, ispp=512, M=2048) the baseline launches
`(EM/64) × (ispp/64) = 64 × 8 = 512` blocks over the 228 CUs of MI300A
(`rocminfo`: `Compute Unit: 228`) — only ~2.2
workgroups per CU of total work, and the baseline's static footprint (107
VGPR, 16.25 KB LDS) admits at most 2 resident blocks/CU — rocprof measures
1.8, i.e. under 2 wavefronts per SIMD (see the performance doc) — so the
kernel is **latency-bound, not throughput-bound**. Total dequant and MFMA
work is *independent of the N tile* (the same fp4 bytes are decoded and the
same MFMAs issued either way), but the grid size is proportional to
`ispp / kBN`. The 16-wide gateup tile therefore quadruples the grid
(512 → 2048 blocks) and cuts the live accumulator footprint from 32 to 4
VGPRs for free. That is what produces the win; the register-staged pipeline
(variant 1) is real but secondary, and the narrow tile multiplies it.

The pipeline changes in variants 1-5 are:

- **A is staged from registers.** `prefetch_a_pf` reads the global A tile
  into VGPRs; `store_a_pf_from_regs` writes LDS only after the loads have
  landed. The baseline instead issues a global→LDS copy and then waits, so
  the load latency sits on the critical path of every K-block.
- **One barrier pair per K-block.** The baseline's `load A → sync →
  dequant gate → sync → MFMA gate → dequant up → sync → MFMA up` chain is
  replaced by one stage/prefetch sync and one consume sync. `prefetch_a_pf`
  / `prefetch_b_pfT<kBN>` / `dequant_b_pf_from_regsT<kBN>` are templated on
  the N tile so the same movers serve every variant.
- **`sA` rows padded by 2 halfwords** (`kSA_P_pf = kBK_pf + 2`). The MFMA
  A-fragment read makes all 16 lanes of a row group touch the same bank when
  the row stride is a multiple of 32 banks; the padding removes the
  conflict. (The decode kernels already used `kSA_P`; the prefill kernel had
  not inherited it.)

**Variant 3 is a measured dead end and is kept only as evidence.** The
split is the mechanism the issue asks for, and it is correct (the oracle
passes), but it cannot help here: the dequant ALU demand is ~30× the MFMA
issue demand (counted off the gfx942 ISA with `meta/scripts/isa_inst_mix.py`
— 495 non-MFMA VALU warp-instructions against 16 MFMA per kb loop body in
the specialised kernel, 241 against 8 in the shipped `_prefill_pipe<16>`;
~4 lane-ALU ops per fp4 element).
Moving the MFMA onto its own wavefronts therefore parks half the resident
wavefronts while the producer half still issues all the ALU on the same
four SIMDs, and the result is a slowdown (see the performance doc). It
would only pay off after the dequant ALU cost is reduced to the point
where the critical resource moves — e.g. to a hardware fp4→bf16 convert.

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

## Routing-weight gather (eliminated)

The xkernels host API takes the **raw `[M, top_k]` routing-weight
matrix**. The down-combine epilogue reads `topk_w[flat]` directly, where
`flat = sorted_ids[...]` is already guarded by `flat < M*top_k`:

```
v *= topk_w[flat];          // in the (flat < M*top_k) branch
```

Because `moe_align_block_size` pads `sorted_ids` with the out-of-bounds
flat index `N = M*top_k` (never negative), `topk_w[flat]` is provably
in bounds inside the guard. This removes the former
`gather_weights_kernel` and the per-call `sorted_w[EM]` buffer entirely:
the launcher performs no device allocation of its own and launches one
fewer kernel per forward (issue #41, items 1 + 2).

`act_scratch` `[EM, ispp]` bf16 remains a **caller-provided scratch
buffer**: a backend serving a 61-layer model allocates it once and reuses
it across every forward pass, instead of paying 122 allocator round-trips
per generated token (issue #41, item 1).

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
| `act_scratch` not caller-owned | Undefined — the launcher does no per-call allocation (issue #41, item 1); the scratch buffer is owned and reused by the caller. No `sorted_w` scratch is required (the combine reads `topk_w` directly, items 1+2). |

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
| 1  | 0.66 | 0.39 | 6.0  | 4.4  |
| 8  | 3.4  | 0.81 | ~30  | ~15  |
| 16 | 4.1  | 2.03 | ~40  | ~30  |
| 48 | 10.4 | 3.96 | 162.2 | 127.2 |

The MI300A `vkernels HIP` column above is the post-issue-41 figure
(no per-call allocation + gather removed, items 1-2), measured as
batched steady-state latency (100 launches, one `hipDeviceSynchronize`,
divided).  It is ~8-12% lower than the pre-item-1 baseline at small M
(was 0.45 / 0.89 / 2.2 / 4.0 ms); the MI250X and xkernels columns are
unchanged from the original cross-check.

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
is occupancy-bound (prefill: rocprof measures ~1.8 resident blocks/CU for
the gateup tile and ~2.6 for the down tile at 16.5 KB LDS + 108/76 VGPRs;
the earlier BN=128 prefill was ~2 blocks/CU at 24 KB LDS + 64 accumulator
VGPRs and ~1.5× slower).

### Prefill variants, issue #76 (MI300A, E=8 hidden=4096 ispp=512 top_k=2)

Same harness as above, `VK_MOE_PF_VARIANT` swept on one build.  Variant 5
is the shipped default.

| M | v0 baseline µs | v1 pipe µs | v2 pipe/BN32 µs | v3 ws µs | v4 pipe/BN16 µs | v5 (default) µs | v5 vs v0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128  | 602  | 595  | 362  | 348  | 238  | **227** | **2.65×** |
| 256  | 613  | 596  | 370  | 355  | 249  | **238** | **2.57×** |
| 512  | 684  | 638  | 421  | 681  | 391  | **380** | **1.80×** |
| 1024 | 1145 | 822  | 694  | 1049 | 628  | **627** | **1.83×** |
| 2048 | 1941 | 1497 | **1144** | 1819 | 1168 | 1172 | **1.66×** |

(job 640344, md5 85dfaee9 — the script prints this table itself).
Prefill effective TFLOP/s at M=2048 rises 13.28 → 21.98.  The decode column
is bit-for-bit unchanged across variants (`VK_MOE_PF_VARIANT` only selects
prefill kernels), and the prefill correctness oracle
(`test_moe_fused_prefill_correct`) PASSes on the default path and on
baseline.  The 32-wide gateup tile (v2) is ~1.1% faster than the 16-wide one
at M=2048 and much slower below it; the two are within run-to-run variance
at the largest M, so 16-wide is the single better default.  See the
*Prefill variants (issue #76)* section above for why the N tile, not the
pipeline, is the lever, and why variant 3 is a dead end.

The acceptance run on the frozen final source (MI300A nid003030, default
path, job 640343, md5 85dfaee9) gives **1.65× at M=2048** (1939 → 1176 µs,
≈26.6 → 43.8 TFLOP/s in the issue's scale) and 2.65× at M=128; six further
runs put M=2048 in 1.65–1.69× and M=128 in 2.61–2.66×.  The decode oracle
(`max_rel=0.000004`), the prefill oracle on clean *and* padded routing
(`max_rel=0.010937`), and all 11 big-shape cases PASS; and the decode
kernels show no regression (3178.2 → 3183.1 µs at M=2048, 404.0 → 404.0 µs
at M=128).  Raw logs
and the full acceptance table are in
[`docs/performance/moe-fused/gfx942.md`](../performance/moe-fused/gfx942.md#issue-76--prefill-variants-mi300a--gfx942)
and `docs/performance/moe-fused/beverin-issue76-prefill-variants.txt`.

---

## Current limitations / future work

- **No per-call allocation + gather eliminated (issue #41, items 1+2 —
  done)**: `act_scratch` is caller-provided; the `hip::fused_moe_mxfp4`
  launcher performs no `hipMalloc`/`hipFree` of its own, so a 61-layer model
  no longer pays 122 allocator round-trips per generated token. Item 2 also
  removed the per-forward `gather_weights_kernel` and the `sorted_w[EM]`
  buffer: the down-combine epilogue reads `topk_w[flat]` directly (the
  `flat < M*top_k` guard already makes `topk_w[flat]` in bounds), so the
  atomic-add scatter is the only remaining combine cost. Items 3-4 of
  issue #41 are also done (see below); item 5 (wavefront-specialized
  producer/consumer prefill) was picked up by issue #76 and is now
  answered: implemented as variant 3, measured, and rejected — the win
  came from the small N tile instead. See *Prefill variants (issue #76)*.
- **`__launch_bounds__` for decode + prefill (issue #41, item 3 - done)**:
  The two baseline prefill kernels carry `__launch_bounds__(256, 4)` and both
  decode kernels carry `__launch_bounds__(64, 10/16)`.  On gfx942 the
  baseline prefill kernels are **LDS-limited at 4 blocks/CU** (16 KB x 4 =
  64 KB, the full CU scratch); `__launch_bounds__` reduced gateup-prefill
  VGPRs 112->107
  and down-prefill 81->75 but did not change occupancy because VGPRs were
  never the binding resource.  The decode kernels are **grid-limited**, not
  occupancy-limited: at M=1 the gateup grid is only 192 blocks for 228 CUs
  (0.84/CU), so no amount of register reduction can raise concurrency.
- **K-major weight layout study (issue #41, item 4 - done)**: an
  alternative gate/up weight packing `[E,2,hidden/2,ispp]` (K-major, N
  contiguous) was implemented behind a `kmajor` launcher flag and the
  bench's `--kmajor` repack.  It makes the 16 dequant columns owned by each
  decode block read **one coalesced cache line** per K-step instead of 16
  scattered lines (N-major stride `hidden/2`).  Measured at the harness
  shapes (E=256, hidden=4096, ispp=512, top_k=6, gfx942): K-major is
  **slower at M=1-4** (e.g. 479 vs 395 µs at M=1) because fewer outstanding
  memory requests reduce memory-level parallelism when the grid is below
  1 block/CU, **even at M=8** (813 vs 813 µs), and **faster at
  M>=16** (1688 vs 2034 µs at M=16) once bandwidth dominates.  The
  crossover is ~M=8.  Conclusion: K-major is a throughput optimization for
  larger M, not a decode-latency optimization; the default remains N-major.
  (Numbers are batched steady-state latency — 100 kernel launches with
  no per-launch sync, then a single `hipDeviceSynchronize`, divided by
  100 — which the per-iteration `hipEvent` timing of this bench agrees
  with to within ~1% at every M, confirming both are faithful.  An
  earlier session reported a spurious ~4.5 µs at M=1 that did not
  reproduce under this methodology; the current numbers supersede it.)
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
- **Prefill variants (issue #76 — done, default is variant 5)**: the
  prefill path now takes `VK_MOE_PF_VARIANT`; the default is 5 (16-wide
  gateup N tile + 32-wide down N tile + register-staged A + padded `sA`
  rows), which is 1.67× at M=2048 and 2.66× at M=128 on gfx942 over the
  v0 baseline, with decode bit-identical and the oracle passing. The
  mechanism is grid parallelism, not deeper pipelining: see *Prefill
  variants (issue #76)*.

  The narrow-tile pipeline kernels relax `__launch_bounds__` to `(256, 2)`
  — the v0 baseline kernels ask for `(256, 4)`: the depth-2 register ring
  costs VGPRs, and forcing 4 blocks/CU would mean fitting the kernel in
  ~64 VGPRs per thread, which the pipeline kernels miss while LDS would
  still fit.  `(256, 2)` is a floor, not a cap, so the compiler is free to
  land higher, and the **measured** residency is ~4 blocks/CU for *both*
  narrow tiles — 84 allocated VGPR / 12.5 KB LDS for the 16-wide gateup
  tile, 64 / 12.5 KB for the 32-wide down tile — against ~1.8 and ~2.6
  blocks/CU for the 64-wide baseline (108 VGPR / 16.5 KB and 76 / 16.5 KB);
  measured occupancy 22.7% → 51.0% and 32.1% → 57.2% of the 32 wavefront
  slots per CU.  The variant-3 ws kernel carries
  `__launch_bounds__(512)` (8 warps, no min-blocks hint).  Full counters
  and the method are in
  [gfx942.md](../performance/moe-fused/gfx942.md#hardware-counters-job-640346-mi300a-md5-85dfaee9).
- **Wavefront specialization (issue #76, variant 3 — measured and
  rejected)**: implemented and correct, but slower than the pipelined
  variants (1815 µs vs 1140 µs at M=2048 on gfx942). The dequant ALU
  demand is ~31× the MFMA issue demand (measured: 495 vs 16 non-MFMA
  VALU/MFMA warp instructions per kb loop body, ~4 lane-ALU ops per fp4
  element), so
  splitting them onto separate
  wavefronts parks half the resident wavefronts without shortening the
  critical resource. It stays in the tree as variant 3, selectable for
  the record, and would only pay off once a hardware fp4→bf16 convert
  collapses the dequant VALU cost (none is exposed for gfx942 by ROCm
  6.3/6.4; checked).
- **Prefill is now a solid win**: from M ≥ 256 on gfx90a (1.13×) and
  M ≥ 512 on gfx942 (1.50×), rising to 1.77× and 2.51× at M=2048 before
  issue #76; with the issue #76 variants the gfx942 numbers become
  2.66× at M=128 and 1.69× at M=2048 against the same baseline. The
  64-row block still wastes compute when routing is sparse
  (tokens/expert < 64); a smaller prefill tile (e.g. BM=32) or a
  padding-aware launch heuristic could close the remaining small-M gap,
  and is now the larger of the two remaining levers since the N-tile
  knob is spent.
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

## Caller contract: persistent, capture-safe scratch

Issue #41, item 1 requires the **caller** to own and reuse the scratch
buffers (`act_scratch`, `out`, `sorted_ids`, `expert_ids`) — the launcher
performs no device allocation of its own. A naive ctypes wrapper that
does `torch.empty()` per call **faults under CUDA-graph capture/replay**:
the fresh caching-allocator storage is not replay-stable, so when the
captured graph replays the C kernel runs against memory the allocator has
recycled (`Memory access fault by GPU node-X`).

The validated, reusable fix lives in
[`vkernels.vllm_experts`](../python-bindings.md#vllm-integration-optional-vkernelsvllm_experts):
`CaptureSafeScratch` sizes each `(device, key)` buffer ONCE (on the eager
profile/warmup run, before capture) and slices into it forever after,
refusing to grow while a capture session is active — so any ctypes caller
(whether or not it uses vLLM) should reuse it instead of allocating per
call. The accompanying `VkernelFusedExperts` is the drop-in vLLM
`FusedMoE` expert layer (gfx942 / Kimi-K3) that backs all C-ABI scratch
this way and launches on the caller's stream.
