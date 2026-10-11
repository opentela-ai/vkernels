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

### Fused-epilogue GEMV+silu (torch_ops companion, `glm_expert_gemv_fused`)

The first "different design" rung, delivered additively:
`vkernels.torch_ops.glm_expert_gemv_fused.expert_gemv_silu` runs the same
block-FP8 GEMV with the SwiGLU applied in-kernel, writing only
`act[T,K,IA]` instead of the `[T,K,2*IA]` gate/up halves plus a
`silu_mul` launch. The numerics contract is **bit-exact equality with the
two-launch chain on the same device** (the gate/up dots are rounded to
bf16 in registers exactly where the chain stores/reloads; the silu
expression is verbatim `elementwise._swiglu_limit`), asserted by
`tests/python/test_glm_expert_gemv_fused.py` on both storages (e4m3fn and
issue-#71 fnuz) and under CUDA-graph capture. Old path stays default
(`GLM53_MOE_SILU_FUSED=1` opts in).

**Honest roofline (GB10 indicative, MI300A pending):** the fusion saves
`k·(4·O + 2·IA)` activation bytes per call against `k·O·I` weight bytes —
0.10% at the GLM serving shape (IA=2048, I=4096, top-8). Measured on GB10
(4×1000-launch event pairs, median): 559.7 → 555.0 µs/call at t=1 (0.8%)
and 1.1–2.2% at the half shape, with BOTH arms already at ~240–298 GB/s
model bandwidth ≈ GB10's LPDDR roof — i.e. where the op is
bandwidth-bound, the fused epilogue wins ≈ its byte saving, nothing more.
On MI300A (instruction-issue-bound at 500–800 GB/s) the expected win is
the removed kernel boundary (~2–5 µs/call of dispatch floor + silu
launch) against ~324 µs/token — a low-single-digit percent, worth having
at M≤2 decode under CUDA graphs, not a bandwidth story. The bandwidth
argument for going further (fusing silu into the down-GEMV read, or a
single persistent kernel) is **near zero for the same reason: weights
dominate** (~16 MiB per expert GEMV vs KB-scale activations); the real
remaining levers are launch count and the router-gather plumbing — see
`NOTES-fusion-glm-decode-fused.md` for the full stage-0→stage-1 design
analysis and the MI300A A/B plan.

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

## #148 — hardware-fp8 decode on the fnuz storage: measured, acceptance NOT met

Issue #148 proposed replacing the manual E4M3FN bit-decode (~8–10 VALU ops
per weight byte, instruction-issue bound at 751 GB/s) with hardware-fp8
paths once #71's `e4m3fn_to_fnuz_inplace` made fnuz storage available:
(1) an MFMA (`v_mfma_f32_16x16x32_fp8`) decode GEMV, and (2) a Triton
`float8e4b8` bitcast decode in `expert_gemv`. The kernels, oracles, probes
and bench landed via #166 (`glm_fp8_fnuz_block_gemv`,
`glm_fp8_fnuz_mfma_block_gemv` in `glm_moe.{hpp,cpp,hip}`,
`glm_decode_fp8_gemm`/`_triton_decode_kernel` in
`glm_fp8_blockwise_gemm.py`, `GLM53_GEMV_FNUZ_NATIVE` in
`glm_expert_gemv.py`); this section records the gfx942 qualification
measurements (the step #166 explicitly did not claim).

Correctness (job 676509, nid003018): `test_glm_fp8_gemv` **24/24 PASS** —
manual, fnuz-cvt and MFMA HIP kernels vs CPU oracles over 8 configs each;
the Triton decode GEMM parity gates pass at rel ≤ 4.4e-3 vs the torch
blockwise oracle on the same quantized operands (fnuz-rounded class, same
policy as the prefill native path); `expert_gemv` FNUZ_NATIVE output is
**bit-identical** to the manual decode in all four shape/M cells.

Same-node HIP timing (job 676509, median of 200 per-iteration event pairs,
unpinned clocks — ratios are the signal; this node's manual baseline ran
~27 µs vs the 22.4 µs record, i.e. ~0.8× clock state):

| shape | M | manual | fnuz-cvt | MFMA | MFMA vs manual |
|---|---:|---:|---:|---:|---:|
| gate/up [4096,4096] | 1 | 27.3 µs / 616 GB/s | 27.5 µs (0.99×) | **23.3 µs / 722 GB/s** | **1.17×** |
| gate/up [4096,4096] | 2 | 31.6 µs / 531 GB/s | 27.3 µs (1.16×) | 26.5 µs / 635 GB/s | 1.19× |
| down [4096,2048] | 1 | 14.8 µs / 566 GB/s | 14.6 µs (1.01×) | 16.8 µs / 500 GB/s | 0.88× |
| down [4096,2048] | 2 | 16.4 µs / 512 GB/s | 16.0 µs (1.03×) | 18.4 µs / 456 GB/s | 0.89× |

Triton MFMA decode sweep (job 676512, nid003028, `--sweep` over BN∈{64,128}
× stages∈{2,3} × sk∈{2,4,8}, BM=16): best gate/up M=1 **363 GB/s** (BN=64,
s=2, sk=2), down M=1 **183 GB/s** — ~2× behind the HIP manual kernel and
~2× behind the HIP MFMA kernel; end-to-end `glm_decode_fp8_gemm` is ~74–76
µs because the fused activation quantize alone costs ~36 µs. The
`torch_blockwise` default backend (BM=128 tiles) is 2758/1400 µs at these
shapes — the M≤16 tile is what makes the decode path viable at all.

**Verdict.** The ≥2× acceptance (≥1502 GB/s on gate/up M=1) is not met by
any variant. Best measured is the HIP MFMA kernel at 722 GB/s — 1.17× the
same-node manual decode and still below the 751 GB/s record in absolute
GB/s. Removing the per-byte decode did NOT make the kernels bandwidth-bound:
at M=1/BM=16 the op mix is dominated by scale handling + LDS staging +
the padded-tile MAC overhead, and on the down shape (K=2048) MFMA actively
regresses. The manual-decode plateau therefore stands as the decode
record; `glm_fp8_fnuz_mfma_block_gemv` stays in-tree as a validated
experiment (correct, oracle-parity) but is not adopted for serving, and
the serving dispatch remains unchanged. The secondary FNUZ_NATIVE bitcast
is numerics-identical and perf-neutral (±3%, four cells) — kept merged, no
win. The full-model quality gate (+0.01 nats/token vs SGLang) from
`docs/issue148-reconciliation.md` item 4 is moot for adoption at this
speedup and remains unexecuted.

Reproduce: `SRC=/capstor/scratch/cscs/xyao/issue148 sbatch --exclusive -A
a-infra01 meta/scripts/issue148_decode_mfma_mi300.sh` (jobs 676509,
676512; JSON at `work/glm53/issue148_decode.json` on beverin).
