# Candidate 1 — DSV41 sm_80 FP8/FP4 tensor-core GEMM family → vkernels port plan

Source tree: `/tmp/dsv41/dsv41` (DeepSeek-V4.1 serving stack, NVIDIA sm_80/A100,
driver-API cubins launched via `cukern.py`).

Files studied:

| Reference file | What it is |
|---|---|
| `cuda/fp8_tc.cu` | Skinny FP8-weight GEMM, M ≤ 16, weights per-lane 16-B loads, split-K fused epilogue |
| `cuda/fp8_tcw.cu` | Second layout (M ≤ 64): weights are the mma **A** operand, x staged in smem (cp.async), weights read once per token tile; 9 template instantiations (MT/STAGES/MW) |
| `cuda/fp8_tcg.cu` | CUTLASS-style tile GEMM (M ≥ 32, BM 64 × BN 128 × BK 128, 2-stage cp.async, `ldmatrix`, XOR swizzles) |
| `cuda/fp4_tc.cu` | Grouped-expert E2M1 GEMM, ≤ 16 tokens/expert, x as mma A operand |
| `cuda/fp4_tcw.cu` | Grouped-expert E2M1 GEMM, ≤ 64 tokens/expert, weights as mma A operand, MT picked per group at launch |
| `w8.py` | `W8` container, `PERM_K` byte permutation, `linear_w` dispatch ladder, `oproj_a` (block-diagonal) |
| `cukern.py` | Host launchers: split-factor heuristics, fused-epilogue counter buffer, `permute_x`, EP shard args |

Our side: `src/c/vkernels/kernels/gemm_bf16.hpp` (two-implementation model, HIP MFMA +
CUDA wmma + fp8-block split-K + fused split-K combine), `moe_fused.hpp/.hip` (MXFP4 fused
MoE with inline E2M1+ue8m0 dequant), `src/python/vkernels/torch_ops/v41_fp8_gemm.py` /
`v41_mxfp4_gemv.py` (Triton + torch-oracle paths, V4.1 UE8M0 scheme),
`docs/kernels-reference.md` (gap-to-SOL methodology).

---

## 1. The four techniques, extracted

### 1.1 prmt-based e4m3 → bf16 register decode (exact, 3 ops per bf16 pair)

`fp8_tc.cu::e4m3x4_to_bf16` — one `uint32_t` holding 4 e4m3 bytes becomes two bf16x2 words
of *unscaled placed bits*, with no fp16 hardware convert:

```
t0 = prmt(w, 0, 0x1404);  t1 = prmt(w, 0, 0x3424);   // byte b0 -> bits 15:8, b1 -> bits 31:24
lo = ((t0 >> 4) & 0x07F007F0) | (t0 & 0x80008000);   // s eeee mmm -> bf16 s 0000eeee mmm0000
```

Exactness argument (why this is not an approximation): byte `[s eeee mmm]` becomes the bf16
`s 0000eeee mmm0000`. Normals decode to `2^(e-7)(1+m/8)` — the e4m3 value. The bf16 value
is subnormal for `e == 0`, and there bf16 arithmetic yields `m * 2^-9`, which is exactly the
e4m3 subnormal. NaN (0x7F/0xFF) is payload-dependent and clamped upstream. The **E8M0
block scale is folded into the same multiply**: the per-32-k scale byte `S` builds
`f2 = bf16x2(2^(S-7))` via `(S + 120) << 7` as the exponent field (S=0 → f2=0, zeroing the
block — a free E8M0 zero handling), and one `fma.rn.bf16x2` with addend 0 applies it
(sm_80 has no `mul.bf16x2`; `fma` with 0 is exact). Products fed to
`mma.m16n8k16.row.col.f32.bf16.bf16.f32` are therefore the **exactly dequantized weights**,
accumulation is fp32 — bit-identical to the bf16 cuBLAS path it replaces at half the bytes.

FP4 variant (`fp4_tc.cu::e2m1x8_to_bf16`) is the same idea with pure shift/mask/and (no
prmt): nibbles `[s e1 e0 m]` → bf16 `s<<15 | e<<7 | m<<6`, × `2^(S-1)` (factor `2^126`),
decode is exact including the E2M1 subnormal (`64m·2^-133 → m/2`). 8 nibbles → 4 bf16x2
words in the order (0,4),(2,6),(1,5),(3,7) — which is why **x must be pre-permuted**
(`cukern.permute_x`, 8-k order `0,4,2,6,1,5,3,7`): a dot product is order-invariant, so the
nibble extraction order defines the required x layout, and no x gather is ever needed.

**Port value on AMD:** `v_perm_b32` is the exact gfx942 analog of `prmt`; the placed-bits
masks are plain VALU; gfx942 has packed bf16 `v_pk_fma_f32` (CDNA3) for the scale multiply
— with an fp32-decode fallback if the packed path underperforms. Every step has a
1:1 AMD equivalent; nothing here is sm_80-only.

### 1.2 PERM_K — the weight byte layout that makes (ld)matrix fragments free

`w8.py::PERM_K = [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]`: within every 16-k group, byte
`4t+j` holds `k = 2t+j (j<2)` or `2t+8+j-2`. One permutation is shared by three kernels:

* `fp8_tcg` (ldmatrix): the 32-bit word `ldmatrix` hands lane t of a row = bytes `4t..4t+3`
  = logical k `(2t, 2t+1, 2t+8, 2t+9)` — precisely the mma A fragment. No fragment
  reconstruction, no extra register shuffles, no bank-conflict re-read.
* `fp8_tc` / `fp8_tcw` (per-lane 16-B loads): decoded words `2s, 2s+1` pair with x words
  `s` and `s+4`, so `mma` step `s` is `{xav[s], xbv[s], xav[s+4], xbv[s+4]}` — the x
  indexing is a compile-time constant, weights stream straight from L2/registers into mma.
* `W8.bf16()` un-permutes for prefill (dequant + cuBLAS), so **one storage layout serves
  decode (tensor cores) and prefill (cuBLAS)** — conversion cost paid once at load time.

**Port note (gfx942):** MFMA fragment ownership differs from `mma.m16n8k16`; `ldmatrix`
does not exist (LDS-sourced operands with a documented lane mapping instead). The *idea*
transfers ("permute bytes at quantize time so the register word IS the operand"), but the
permutation must be re-derived for `mfma_f32_16x16x16bf16` — call it `PERM_K_GFX942`.
Quantize-time cost is identical; the win is the same.

### 1.3 Split-K with the fused last-block epilogue (atomic arrival counters)

All three FP8 GEMMs write fp32 partials `[splits, M, N]` and, when the caller wants bf16,
fuse the reduction into the same launch: `__threadfence(); atomicInc(counters[tile],
splits-1)`; the **last-arriving block** for the tile `__threadfence()`s, then sums the
`S` planes **in fixed ascending-split order** and does the single RNE bf16 store.
Counters are a per-device buffer that self-resets via the `atomicInc` wrap-around
(`_tile_counters` grows one zeroed int32 buffer per device, reused across launches) —
no pre-launch memset, CUDA-graph capturable, no host sync.

This is exactly the design already proven in our repo: `gemm_bf16.hpp` documents
`gemm_bf16_splitk_fused_with_config` ("per-output-tile arrival counters … last-arriving
block sums all S planes in the SAME fixed ascending-s order … BIT-EXACT with the
two-kernel path"), currently A/B-gated behind `VK_GEMM_SPLITK_FUSED`. The DSV41 family is
independent confirmation at A100 scale that the fused epilogue is where the split-K
serving gap closes (our 3.1 record: 56% of HBM with the round-trip still un-fused).
Tile granularity differs per kernel (8 columns in `fp8_tc`, 16 in `fp8_tcw`, one
(m,n)-block in `fp8_tcg`) — pick per kernel so the epilogue's serial sum stays off the
critical path.

### 1.4 `group_cols` — block-diagonal o-projection inside the GEMM

`group_cols > 0` changes only the x-row mapping: output row `b`, column `n` reads
`x[b * (N/group_cols) + n / group_cols]` (in `fp8_tcg`: the x-tile slice is selected per
n-block at load time; in `fp8_tc/tcw` the A-fragment rows pick `g * xgroups + n0/group_cols`),
the output is kept on row 0 of the tile, and the partials/epilogue are unchanged.
`w8.oproj_a` uses it to run the block-diagonal MLA-style o-projection
(`o[b,s,g,d] × wo_a`, rows of `wo_a` touching only group g) **as one GEMM with zero
gather/scatter**: no einsum, no per-group kernel, no intermediate. Shapes constrained so
`group_cols % 8 == 0` (tcw: % 64; tcg: % 128) — i.e. the block structure must be tile-
compatible, otherwise it falls back to the einsum.

---

## 2. What the family achieves at the source (evidence it is worth porting)

* `w8.py` docstring: 1.1–1.2 TB/s effective on A100 vs 1.4 TB/s bf16 cuBLAS ⇒ **~1.6× per
  decode GEMV at half the weight bytes, exactly equal numbers** (exact dequant, fp32 acc).
* `fp4_tc.cu` header: same numbers as the FP4 GEMV, "a quarter of the instructions per
  weight", and all tokens routed to an expert share one weight read (the grouped-GEMM win
  the `glm_expert_gemv` family cannot get at M > 1).
* `fp8_tcw` header records the two measured failure modes it fixed: weights re-read per
  16 rows ⇒ batched decode cost ~linear in rows; and 2-deep weight prefetch insufficient
  (latency-bound ~1.2 µs/iter; 3 blocks/SM only ~600 GB/s) ⇒ **two register sets + smem
  x-staging**. Both are directly reusable tuning lessons for any port.

---

## 3. Port plan for vkernels

### 3.1 New kernel family (two-implementation model)

**New: `src/c/vkernels/kernels/gemm_w8.{hpp,cpp,hip,cu}`** — weight-only block-FP8 GEMM,
mirroring the `gemm_bf16` file quartet:

* `gemm_w8.hpp` — contracts:
  * `gemm_w8_cpu(...)` CPU oracle (see 3.3).
  * `hip::gemm_w8_splitk_with_config(...)` / `_fused(...)` — MI300A MFMA port of
    §1.1 + §1.3, using the repo's existing split-K scaffolding (`gemm_fp8_block_splitk_with_config`
    staging, `gemm_bf16_splitk_fused_with_config` counter-combine) with **UE8M0 [32×32]
    scales** instead of fp32 [128×8] scales, and weights in a PERM_K-style layout
    (§1.2, `PERM_K_GFX942` re-derived for the MFMA fragment).
  * `cuda::gemm_w8_...` — direct transcription of the sm_80 kernels (wmma shim already
    exists in `gemm_bf16.cu`; the inline-ptx bodies port as-is to `gemm_w8.cu`).
  * `group_cols` parameter for the block-diagonal o-projection (§1.4) on both backends.
* `moe` side — extend `src/c/vkernels/kernels/moe_fused.{hpp,hip}` (or new
  `moe_grouped_w4.{hpp,cpp,hip}`): grouped E2M1 GEMM entry with
  `grp_expert / grp_start / pair_tok / shard_start / shard_n / zero_out` (§ fp4_tc)
  and the small/large token-count kernel pairing of `fp4_tc.cu` + `fp4_tcw.cu`
  (≤ 8/16 tokens: x-as-A; ≤ 64: weights-as-A with per-group MT selection). The inline
  E2M1+ue8m0 decode already exists in `moe_fused.hip`; the new part is the fragment
  scheduling + the grouped indexing, not the decode.

### 3.2 Backend map (what changes per architecture)

| DSV41 primitive | MI300A (gfx942) | NVIDIA (A100/GB10, existing shim) |
|---|---|---|
| `prmt` byte shuffle | `v_perm_b32` | `prmt.b32` (as-is) |
| `fma.rn.bf16x2` scale mul | `v_pk_fma_f32` (packed bf16, CDNA3) or fp32 fallback | as-is |
| `mma.m16n8k16` bf16 | `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` (repo standard, gemm_bf16) | as-is |
| `ldmatrix` + XOR-swizzled LDS | LDS buffer loads with the documented lane mapping; swizzle re-derived | as-is |
| `cp.async` x-staging | double-buffered global→LDS (repo precedent: GB10 cp.async ring; MI300A stays synchronous per #77 finding) | as-is |
| e4m3fn (OCP) payloads | **fnuz vs fn bias trap — see 4.1** | as-is |
| fp32 partials + counters | same; reuse `_tile_counters`-equivalent buffer in C++ | same |

**MI300A design option worth an A/B before the bit-twiddle port:** gfx942 has native FP8
MFMA (`__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8`, already used in `dsa.hip:2028`).
Since UE8M0 scales are per-32-k block, one can run MFMA **directly on raw e4m3 payloads**
in 32-k chunks, multiplying the accumulator by `2^(S-127)` (plus the fn↔fnuz factor-2)
between chunks — the §1.1 decode disappears entirely on the A-operand side. Cost: an ACC
scale every 32 k (cheap VALU) vs decode VALU per weight. This is the same "fold the scale
into fewer, larger ops" direction the dsa-topk mfma-fp8 variant took. Keep the bf16-decode
port as the baseline and correctness cross-check.

### 3.3 CPU oracle requirements (non-negotiable, per the two-implementation model)

* `gemm_w8.cpp`: exact e4m3fn decode from bits (table or arithmetic — mirroring
  `_e4m3_to_f32` in `v41_fp8_gemm.py`: normals `2^(e-7)(1+m/8)`, subnormals `m·2^-9`,
  0x7F/0xFF NaN), E8M0 scale = `2^(S-127)` with `S == 0 ⇒ block contributes 0`, fp32
  accumulation, **one** RNE bf16 round on store; split-K combine sums in fixed
  ascending-split order so host and fused device agree to the bit (same contract as
  `gemm_bf16_cpu`).
* `moe_grouped_w4.cpp`: gather-dequant + fp32 dot oracle (mirror of
  `mxfp4_expert_gemv_reference` / `FP4_VALUES`), including the `pair_tok` mapping,
  EP shard skipping (`zero_out`), and the min/max token-count partition contract so the
  two-kernel split is invisible to the caller.
* `group_cols` oracle: the block-diagonal x-row mapping spelled out (output row b,
  column n uses x row `b·(N/group_cols) + n/group_cols`) — it is the one part of the
  contract where a silent indexing bug is cheap to write and expensive to notice.
* Oracles must be encoding-explicit (fn vs fnuz asserted, never inferred) — see 4.1.

### 3.4 Python / ops layer

* `src/python/vkernels/torch_ops/`: extend `v41_fp8_gemm.py` with a
  `gemm_w8` backend entry (x bf16, w8 uint8 + UE8M0 scales, optional `group_cols`),
  dispatched through `_dispatch`/`fast_path` like `v41_mxfp4_gemv.py`, keeping the
  existing Triton and torch-oracle paths as fallbacks and the
  `VKERNELS_V41_FP8_GEMM_BACKEND` env semantics. Add quantize-time helpers:
  `permute_k`/`unpermute_k` (the `PERM_K`/`PERM_K_GFX942` reorders) and, for the fp4
  grouped path, `permute_x` (8-k order `0,4,2,6,1,5,3,7`).
* Launcher policy (port from `cukern.fp8_gemm_tc`): shape gate on M
  (≤ 16 per-lane-load kernel → ≤ 64 weights-as-A → tiled beyond), split-factor heuristic
  "≥ ~1024 warps stream the weights, partials ≤ half the weight bytes", and the
  fused-epilogue-only-when-bf16-output rule (`out_dtype` float ⇒ caller reduces).

### 3.5 Tests & records

* `tests/test_gemm_w8.cpp` (oracle parity, bit-exact fused-vs-two-kernel split-K,
  group_cols indexing, S=0-scale zero blocks), `tests/test_moe_grouped_w4.cpp`
  (grouped parity vs oracle, EP shard skip/zero_out, token-count partition boundary).
* Perf records: `docs/performance/gemm-w8/gfx942.md` (+ `/gb10` if run) following
  `docs/performance/README.md`; then one master-table row in `docs/kernels-reference.md`
  with the binding roof (HBM for decode shapes — the family's whole point is bytes,
  so "GB/s effective vs the bf16-stream baseline" is the honest headline, not TFLOP/s).

## 4. Risks

1. **fp8 encoding mismatch (top risk).** DSV41 weights are OCP e4m3fn (bias 7, NaN at
   0x7F/0xFF); our in-repo fp8 kernels (`gemm_fp8_block_splitk_with_config`, glm paths,
   native MFMA fp8) are E4M3FNUZ (bias 8). fn value = 2 × fnuz value for the same bits, so
   a scale-fold of one extra exponent absorbs it — but the conversion must be explicit at
   quantize time and asserted in the oracle, or results differ by 2× silently.
2. **PERM_K does not transfer literally.** It is tuned to `mma.m16n8k16` + `ldmatrix`
   fragment ownership; MFMA 16×16×16 has a different lane→element map. Re-derivation is
   required (and a unit test that the stored word equals the expected fragment at each
   lane), otherwise the "free fragment" becomes a per-step shuffle tax.
3. **No `ldmatrix`/`cp.async` on gfx942.** The tcg-style tiled kernel loses its two main
   levers; the honest MI300A expectation is the per-lane-load `fp8_tc/tcw` shape, with LDS
   double-buffering only if the #77 occupancy lesson (≥1536 threads/CU while pipelining)
   permits. The native-fp8-MFMA option (3.2) sidesteps much of this.
4. **Fused split-K counters + CUDA graphs.** Self-resetting counters assume every launch
   touches the same tile set; a shape change between replays (or a tile skipped by an
   early-exit such as the EP shard filter) leaves a stale count. The DSV41 launchers
   always launch the full grid and early-exit *inside* — the port must preserve that
   invariant, and the fused path stays behind the `VK_GEMM_SPLITK_FUSED`-style gate until
   the A/B (house precedent) plus a graph-replay test.
5. **Workspace footprint.** fp32 partials `[splits, M, N]` at large N is the reason the
   DSV41 launcher keeps "partials ≤ half the weight bytes"; on the 6×MI300A serving shape
   (N up to 7168) the same bound must be enforced by the split heuristic, not assumed.
6. **Grouped fp4 launcher dynamic behavior.** The per-group MT selection and the
   small/large token-count partition are host-side branches; graph-capturable only when
   group counts are stable per step (true for our serving MoE decode, false for
   ragged prefill) — document and gate accordingly.

## 5. Order of work

1. `gemm_w8` CPU oracle + CUDA transcription of `fp8_tc.cu` (smallest kernel, whole
   technique stack in 157 lines) with bit-exactness tests on A100/GB10.
2. HIP port: fnuz conversion + `PERM_K_GFX942` derivation + MFMA body; A/B the
   bf16-decode port against the native-fp8-MFMA accumulator-scaling variant (3.2).
3. Fused split-K epilogue (reuse the `gemm_bf16_splitk_fused` pattern) + lift the
   existing `VK_GEMM_SPLITK_FUSED` gate if the A/B wins — closes the 3.1 record's
   56%-of-HBM serving gap with the same mechanism DSV41 measured at A100.
4. `group_cols` support + torch_ops dispatch + `v41_fp8_gemm` backend wiring.
5. Grouped fp4 family (`moe_grouped_w4`), pairing small/large kernels, then the perf
   record and master-table row.
