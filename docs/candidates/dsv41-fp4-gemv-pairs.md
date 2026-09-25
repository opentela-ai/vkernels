# Candidate analysis: `fp4_gemv_pairs` (CUDA → HIP port) for V4.1 MXFP4 MoE decode

Reference: `/tmp/dsv41/dsv41/cuda/fp4_gemv.cu` + wrapper `cukern.py::fp4_gemv_pairs`.
Current op: `src/python/vkernels/torch_ops/v41_mxfp4_gemv.py` (Triton per-row GEMV).
Tuning-store precedent: `tuner.py` `Tunable("glm_fp8_gemv_pick_sk", toolkit="hip", bench="glm_fp8_gemv_bench")` + `bench_glm_fp8_gemv --persist`.

## 1. Exact launch geometry and footprint

Compute: `out[pair, n] = wt[pair] * Σ_k x[row_in[pair], k] * W[expert[pair], n, k]`,
W packed E2M1 (two codes/byte, low nibble = even k), one E8M0 scale per (n, 32 k).

| Item | Value |
|---|---|
| Grid | `(ceil(N / 32), n_pairs, 1)` — x over output rows, y over routed pairs |
| Block | `(256, 1, 1)` = 8 warps (`WARPS=8`) |
| Rows per block | `WARPS * ROWS_PER_WARP = 8 * 4 = 32` output rows `n`; each warp owns 4 consecutive `n` (`n0 = (blockIdx.x*8 + warp)*4`) |
| Shared memory (static) | `float xs[32 * MAX_CHUNKS]` = `32 * 160 * 4` = **20,480 B**, plus `float lut[16]` = 64 B → **20,544 B ≈ 20.1 KB** (`MAX_CHUNKS = 5120/32 = 160`) |
| Loads | 16-byte `uint4` per warp instruction = 32 contiguous bytes = 64 codes (32 per nibble-half... 512 contiguous weight bytes per warp instruction row stream); lane `l` owns k-chunks `c = l, l+32, …` |
| Reduction | fp32 accumulation; full-warp `__shfl_xor` butterfly (16→1); `lane==0` writes `wt[pair] * acc` — **no atomics, one writer per (pair, n)** |
| Smem activation staging | Transposed: `xs[j*chunks + c] = x[32c + j]`; after decode, lanes read `xs[j*chunks + c]` for fixed `j`, `c = lane` → consecutive addresses, conflict-free |
| E2M1 decode | 16-entry shared fp32 LUT built once per block (`mags = {0,.5,1,1.5,2,3,4,6}`, sign from bit 3) |
| Scale | `sb == 0 ? 0 : __int_as_float(sb << 23)` (i.e. 2^(sb−127)) — no `decode_scale` preprocessing pass needed |
| Constraints | `x` bf16 contiguous rows, `K % 32 == 0`, `K ≤ 5120` (smem table bound), int32 `row_in`/`expert`, fp32 `wt`, fp32 out `[pairs, N]` |

## 2. What the pair-list + transposed-smem + no-atomics schedule buys vs our Triton `v41_mxfp4_gemv`

1. **Vectorized weight streaming.** Triton kernel does byte-granular loads with `col//2` and per-element `%2`/`>>4` nibble extraction. The CUDA kernel streams 16-byte `uint4` words and unpacks 32 codes per load with unrolled bit ops — a large instruction-count reduction on the dominant traffic.
2. **Activation loaded once per pair, not per output row.** Triton launches `(t*k, cdiv(O,4))` programs; every program re-loads its `x` row through global memory. Here one block stages the row into LDS once and all 32 output rows in the block reuse it — global traffic on `x` drops by the rows-per-block factor.
3. **Conflict-free transposed staging.** The transposed layout (`xs[j*chunks + c]`) makes the 32-lane decode read (`xc[j*chunks]`, `j = 0..31`) fully coalesced/bank-conflict-free, so the per-chunk 32-term `fmaf` chain stays throughput-bound, not smem-bound.
4. **No global LUT.** Triton loads `FP4_VALUES` from a global tensor per element; here the LUT lives in LDS (and the `(sb==0→0)` pow-2 scale trick removes the separate `decode_scale` tensor we materialize per call).
5. **Dispatch fused into the kernel (pair-list).** `row_in`/`expert`/`wt` are consumed on device; no `expand` of indices, no host-side gather. One launch covers the whole `[pairs, N]` tile instead of `t*k` programs — far fewer kernel/program launches at decode batch sizes.
6. **Deterministic, atomics-free reduction.** Butterfly shuffle within one warp; exactly one writer per output — bit-stable across runs, unlike any split-K/atomic variant. The 5120-K cap plus fp32 accumulation also means **no split-K** is needed at these shapes (K ≤ 5120 in one pass).
7. **Numerics are *better* than our oracle, not just equal.** The Triton path rounds `w*scale` to bf16 before the fp32 dot (to match the gather-dequant oracle). The CUDA kernel keeps decoded E2M1 magnitudes in fp32 throughout — strictly less rounding. Output is fp32 `[pairs, N]` (router weight fused), vs our bf16 `[T, K, O]` (weight applied downstream).
8. **Fewer Python-side ops per call.** Today's wrapper does `contiguous()`, `decode_scale`, `indices.to(int64)`, a LUT tensor allocation, and a Triton JIT dispatch. The pair kernel needs pre-packed int32 pair arrays (which our router can emit directly) and nothing else.

## 3. Cost / risks for a HIP port

- **No atomics, no wmma, no callbacks** — the kernel is pure LDS + FMA + shuffle, i.e. squarely in the "portable" class; `__shfl_xor_sync` maps to full-wavefront shuffle on CDNA. 20.5 KB static LDS is trivial vs 64 KB.
- **`K ≤ 5120` static bound.** Our V4.1 MoE FFN has three matrices per expert; confirm hidden/intermediate dims fit, or bump `MAX_K` (smem grows linearly: K=7168 → 224 chunks → 28.7 KB, still fine). Make `MAX_K` a template/compile-time knob, not a runtime assert surprise.
- **16-byte loads need 16-byte alignment** of each weight row (`stride_wn` must preserve it); assert in the wrapper.
- **Pair-array format**: we must produce int32 `row_in`, `expert` and fp32 `wt` (router weight). Our router currently emits int64 `indices [T, K]` and applies weights after the GEMV — fusing `wt` changes where the multiply happens (need parity test against the post-hoc path; fp32-vs-bf16 weight ordering differs).
- **Numeric delta vs current CI expectations**: fp32-decoded weights will not bit-match `mxfp4_expert_gemv_reference` (bf16-rounded weights). Reuse the `tests/python/test_mxfp4_serving_parity.py` tolerance framework rather than bit-exact asserts.

## 4. Integration path (follow the `glm_fp8_gemv_pick_sk` pattern)

1. HIP-ify `fp4_gemv.cu` (bf16 typedef, keep `__restrict__`), load via the existing native/ctypes launcher path (`_dispatch`/HIP module loader, mirror `cukern.py::get_function`/`launch`).
2. New torch op `v41_mxfp4_gemv_pairs(x, weights, scales, row_in, expert, wt, n_pairs)` in `torch_ops/`, falling back to `mxfp4_expert_gemv` when pair arrays aren't available.
3. Benchmark harness `bench_v41_mxfp4_gemv_pairs` sweeping (N, K, PAIRS, DEVICE) — Triton current vs pair-kernel — and a `Tunable("v41_mxfp4_gemv_pairs", "native", toolkit="hip", bench=…, persists=True)` entry so the winner is persisted per shape key, exactly like `glm_fp8_gemv_pick_sk`.
4. Router emits pair triplets (`row_in = flat pair // K`, `expert`, `wt = router weight`); keep the old op for prefill/batched-M shapes (this kernel is decode-only, M=1-per-pair).
5. Parity tests vs `mxfp4_expert_gemv_reference` (tolerance) and end-to-end `test_moe` decode checks.

## 5. Verdict

**Worth porting.** For decode (pairs ≈ tokens × top-k, M=1 per pair) the current Triton kernel's per-program x reloads, scalar nibble unpacking, global LUT, and `t*k × cdiv(O,4)` program count are all eliminated by this schedule. The kernel is atomics-free, tensor-core-free, LDS-light, and shuffle-based — the lowest-risk HIP port class — and it fuses the router weight, removing a downstream pass. Main engineering tasks: pair-triplet emission from the router, `MAX_K` knob, and tolerance-based parity tests.

---
### 10-line summary

1. Kernel: `out[pair,n] = wt[pair]·Σ_k x[row_in[pair],k]·W[expert[pair],n,k]`, E2M1 + E8M0/32-k, K ≤ 5120.
2. Launch: grid `(ceil(N/32), n_pairs)`, block 256 (8 warps), 32 output rows/block, each warp owns 4 rows.
3. Smem: transposed activation `xs[j*chunks+c]` (20,480 B) + 16-entry LUT (64 B) = 20,544 B static, bank-conflict-free.
4. Decode: 16-byte `uint4` weight loads (32 codes/load), shared-LUT nibbles, fp32 FMA, butterfly shuffle, no atomics.
5. Gains vs Triton op: no per-program x reloads (LDS reuse across 32 rows), no scalar nibble div/mod, no global LUT/`decode_scale`, one launch for all pairs, fused router weight, fp32 output.
6. Deterministic (single writer per output), and more accurate than our bf16-rounded oracle — needs tolerance, not bit-exact, parity tests.
7. Risks: `K ≤ 5120` static bound (make `MAX_K` a knob; smem stays < 32 KB for K=7168), 16 B row-alignment assert, int32 pair-triplet emission from the router.
8. Kernel is HIP-portable by class: LDS + FMA + shuffle only, no wmma/atomics/callbacks; ~20 KB LDS vs 64 KB available.
9. Integration: new `v41_mxfp4_gemv_pairs` torch op + `bench_v41_mxfp4_gemv_pairs` + `Tunable(…, toolkit="hip", persists=True)` following the `glm_fp8_gemv_pick_sk` store pattern.
10. Verdict: port it for decode; keep the Triton op as fallback and for batched/prefill shapes.
