# GLM decode kernels: block-FP8 expert GEMV (#64) and small-output projections (#66/#68)

Two new vkernels entries for the GLM-5.3-Flash decode path, validated on
beverin MI300A (gfx942, jobs 629753/629763). Reproduce everything from a
clean checkout:

```bash
SRC=$SCRATCH/vkernels sbatch meta/scripts/bench_glm53_issues_mi300.sh
```

## #64 — `glm_fp8_block_gemv`: decode-fused E4M3FN block-scaled GEMV

Native HIP (`glm_moe.{hpp,cpp,hip}`) with a CPU oracle. Contract: weights
`[N,K]` uint8 E4M3FN row-major, scales `[N/128, K/128]` fp32 per 128×128
block, x `[M,K]` bf16 (M ≤ 2), out `[M,N]` bf16; FP32 accumulation, one
round-to-nearest-even at the end. GLM shapes: gate/up `[4096,4096]`, down
`[4096,2048]`. The NaN encodings (0x7F/0xFF) decode to NaN in the oracle
and to ±480 through the GPU's branchless path (documented; never present
in weights).

Structure: split-K partial kernel — grid `(N/32 row-blocks, sk)`, 8 warps
× 4 rows, the block's K-slice staged in LDS as fp32, one uint32 = 4 fp8
codes per lane per 128-wide scale segment (coalesced), branchless decode
(place `(e<<23|m<<20)` in an f32 and multiply by `2^120` — covers normals
AND subnormals in one op), warp-shuffle reduction, one fp32 partial per
row-split; then a fixed-order reduce + bf16 round. `sk` ∈ {1,2,4,8}
dividing K/128, picked for ≥ ~2 blocks/CU (`glm_fp8_gemv_pick_sk`).

Correctness: 8/8 configs vs the CPU oracle (max_rel ≤ 7.5e-3 = bf16
rounding level), including the SEGS-template instantiations and the
generic odd-segment path (K=1152).

Measured (beverin, median, vs a structure-matched materialized-BF16 GEMV
plus the one-time dequant; best sk swept over {1,2,4,8}; job 629781):

| shape | M | fused | mat-GEMV | dequant (once) | steady win |
|---|---|---:|---:|---:|---:|
| gate/up [4096,4096] | 1 | 22.4 µs (751 GB/s) | 42.0 µs | 47.0 µs | **1.88×** |
| gate/up [4096,4096] | 2 | 29.5 µs | 62.2 µs | — | **2.11×** |
| down [4096,2048] | 1 | 16.4 µs | 17.4 µs | 24.4 µs | 1.06× |
| down [4096,2048] | 2 | 18.1 µs | 26.6 µs | — | 1.47× |

Per decode token (top-8 experts × gate/up + down): ~324 µs fused vs
~529 µs materialized steady-state, plus the dequant buffer and its
traffic eliminated entirely.

**Measured lessons (three structural rounds):**
1. Row-parallel-only (128 blocks on 228 CUs, ~14% occupancy) ran at
   175 GB/s — split-K partials bought 1.5×.
2. The branchy per-value decode serialised SIMT lanes; the branchless
   `2^120` decode plus a single weight pass for M=2 bought 2.7×.
3. A runtime-bounded segment loop prevented unrolling/pipelining (warp
   issue serialised load→use per segment); a SEGS template with full
   unroll plus an sk sweep to 8 blocks/CU bought ~1.2× and made M=2
   consistently 2.1× vs materialized.
All three kernels (fused, toy bf16 baseline, dequant) plateau at
~500–800 GB/s regardless of structure after round 3 — the op is now
instruction-issue-bound, not bandwidth-bound; further gains need a
different design (hardware fp8→fp16 converts, MFMA, or fusing the
top-k gather), not tuning.

## #66/#68 — `vkernels.torch_ops.glm_projection`: batch-one small-output projection

Triton operator generalising the proven mHC/QKV methodology to arbitrary
`N` (the GLM shapes: `[64,4096]`, `[128,4096]`, `[512,4096]`,
`[1536,4096]`), autotuned over ROWS × SPLITK × warps, deterministic
FP32 split-K partials + fixed reduce, one bf16 round. Contract: K a
positive multiple of 4096 (every SPLITK ∈ {1..16} divides 4096, so no
runtime config pruning), M ∈ {1,2}, contiguous bf16, eager warmup before
capture (same graph contract as the sibling operators).

Tests: 16/16 PASS on beverin (contract, per-shape parity vs an FP32
reference, graph replay with changed inputs, read-only inputs, known
values, cold-capture refusal, non-current-device restoration).

**Benchmark result (honest negative):** at these isolated shapes, default
BLAS is already near-optimal — the Triton operator does not beat it:

| M=1 | default BLAS | TunableOp | Triton |
|---|---:|---:|---:|
| N=64 | 5.81 µs | 5.58 µs | 5.85 µs |
| N=128 | 5.92 µs | 5.40 µs | 6.12 µs |
| N=512 | 7.15 µs | 6.19 µs | 7.04 µs |
| N=1536 | 11.68 µs | 7.97 µs | 9.74 µs |

(M=2 similar; full JSON in `work/glm53/glm_projection.json` on beverin.)
The ~2.3–2.5 ms/token attributed to the `[128,4096]` gate projections in
the #68 profile therefore does **not** reproduce as an isolated-GEMM
algorithm problem: a fresh contiguous `[128,4096]` BF16 GEMV is ~6 µs on
the same GPU, and ×34 layers ×2 sites is ~400 µs/token, not 6.5 ms. The
in-situ cost must come from the real dispatch path (strides/layout,
eager launch gaps, or counter attribution) — chasing it requires the floe
reproduction, not a better isolated GEMM kernel. TunableOp already
recovers 8–32% at N ≥ 512 and is the cheap in-situ lever. The operator
stays in-tree: it is correct, graph-safe, and is the natural vehicle if
the floe-side investigation finds a shape where BLAS regresses.

Two Triton pitfalls found by testing (both encoded in the module):
1. **Autotune trial pollution** — trials with a larger SPLITK write
   partial slots that the winning smaller-SPLITK config never rewrites;
   the reduce then sums stale trial data. Fix: run the tuning pass, zero
   the partial buffer, then launch the cached config.
2. **Cross-program zeroing races** — a kernel must never zero partial
   slots owned by concurrent programs (other splits of the same row);
   only the launcher's pre-launch memset is safe.
