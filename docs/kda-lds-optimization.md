# kda_delta_rule_fwd: D×D state cache in LDS (2.0× on S=512 D=128)

**Issue:** #70 (efficient GLM-compatible chunked prefill), reusing the #21 KDA
primitive. Root cause of the #63 "5/16 failures" proven to be a **test bug**
(see [issue-63-investigation](#issue-63-investigation)), not a kernel defect —
so the optimization below is safe on a kernel already correct vs the CPU oracle.

## The binding resource (before)

`kda_delta_rule_fwd` kept the per-`(b,h)` D×D recurrent state in a **global
scratch buffer** and round-tripped it from HBM ~6× per token — GATE
read-modify-write, PREDICT read, UPDATE read-modify-write, OUTPUT read. The
roofline diagnosis was unambiguous: arithmetic intensity ~0.43 (legacy gmem
model) ≪ MI300A ridge ~247 → **strongly memory-bound on the state traffic**.
For D=128 that is ~96 KB/token/head of HBM that scales with both S and H.

## The change

Cache each z-block's `Db` state rows in **LDS** (`__shared__ float srow[Db*D]`):
load once at chunk start, run all four phases (GATE → PREDICT → UPDATE → OUTPUT)
on the on-chip copy, store once at chunk end. Because row-parallelism means each
z-block owns **disjoint** state rows, the cache needs no cross-block
coordination; the gmem state is only the between-chunk handoff. The math is
unchanged (same fp ops, just on-chip), so results are **bit-identical** to
`kda_naive_delta_rule_fwd_cpu` — `test_kda_correct` still 11/11 PASS.

LDS footprint: `Db*D` floats = 16 KB at D=128, Db=32 (MI300A has 256 KB/CU →
full occupancy by wavefront slots, not LDS).

## A/B on beverin MI300A (same fixed bench; `meta/scripts/ab_kda_lds_vs_gmem.sh`)

Fixed shapes (H S D), single config per short-lived process (MI300A watchdog):

| H S D   | GMEM us(med) | LDS us(med) | speedup |
|---------|-------------|-------------|---------|
| 1 64 32 |   137.1      |   109.0     | 1.26×   |
| 1 64 64 |   181.9      |   144.6     | 1.26×   |
| 16 64 64|   184.2      |   145.0     | 1.27×   |
| 1 512 64|  1336.3      |  1028.8     | 1.30×   |
| 1 512 128| 3849.7      |  1897.3     | **2.03×** |

`1 64 16` reads 0 µs and ~60 µs for both kernels — sub-hipEvent-resolution timer
artifact, not a regression.

## A/B at realistic GLM occupancy (`meta/scripts/ab_kda_scale.sh`)

The fixed shapes use H≤16; the GLM-relevant case is H=128. Sweeping H at
S=512 D=128 (grid = `H × D/Db` blocks; MI300A has 304 CUs):

| H   | GMEM us | LDS us | speedup | GMEM tf | LDS tf |
|-----|---------|--------|---------|---------|--------|
| 1   | 3777    | 1885   | 2.00×   | 3861    | 1897   |
| 8   | 3864    | 1924   | 2.01×   | 3926    | 1930   |
| 16  | 4788    | 2925   | 1.64×   | 4835    | 2939   |
| 32  | 4834    | 2955   | 1.64×   | 4868    | 3008   |
| 64  | 4889    | 2956   | 1.65×   | 4941    | 2975   |
| 128 | 5625    | 3189   | 1.76×   | 5655    | 3202   |

At H=128 S=512 D=128 the **GMEM kernel achieves ~3065 GB/s = 58% of the
5300 GB/s HBM roof** — solidly memory-bound; its per-block time rises with H
as blocks contend for HBM bandwidth and then plateaus at the roof.

## The new binding resource (after)

The LDS kernel drops to **~13 GB/s actual** (q/k/v/g only) and **~0.24 TFLOP/s**
(0.3% of the ~81 TFLOP/s fp32 vector roof): **neither** memory-bound **nor**
compute-bound. It is now **latency/occupancy-bound** on the serial per-token
recurrence — four barrier-separated phases per token, with PREDICT and OUTPUT
using only `Db=32` of `kTh=256` threads (12.5%) for the length-`D` dot products,
leaving most of each wavefront idle at every barrier.

## A negative result: 4→3 barriers is a RACE (do not retry)

Removing the post-OUTPUT `__syncthreads` looks redundant by a naive
same-thread argument — GATE-(t+1) is a read-modify-write of the exact
`srow[idx]` that UPDATE-(t) wrote on the *same* thread, and the cross-thread
UPDATE-(t)→PREDICT-(t+1) dependency is published by the post-GATE-(t+1)
barrier. **That argument is wrong and was disproven by experiment**
(`meta/scripts/verify_kda_3barrier.sh` + `ab_kda_barriers.sh`, beverin
MI300A): the 3-barrier kernel fails **5 of 6** delta_rule_fwd configs with
clean, systematic max_rel ≈ 1.4–2.0, while the 4-barrier kernel stays
11/11 PASS. The post-GATE barrier is *after* GATE, so it cannot stop a
race *during* GATE: GATE-(t+1) on thread `tid` **writes** `srow[idx]`
(element `idx%D` of row `idx/D`) while OUTPUT-(t) on a *different* thread
(`tid2 = idx/D`) is still reading that same row (`srow[tid2*D + e]` for all
`e`, including `idx%D`) for its dot product. The post-OUTPUT barrier is
therefore a **true dependency**, not dead synchronisation. (The race is
worst at low block counts — grid ≤ 4 — and is masked by serialization only
at very high H.)

## Negative result: PREDICT/OUTPUT compute is non-binding (do not parallelise the dots)

Hypothesis: with only `Db=32` of `kTh=256` threads doing the length-`D`
(=128) dots in PREDICT and OUTPUT (12.5% utilisation), each thread runs a
~128-deep dependent add chain — so parallelising each row's dot across
`r = kTh/Db = 8` threads (strided over `e`, `__shfl_xor` butterfly reduce)
should cut the per-block critical path from D to D/r dependent adds and
speed the kernel up at every H.

**Disproven by experiment** (`meta/benchmarks/kda_pardot.hip` +
`meta/scripts/ab_kda_pardot{,_run}.sh`, beverin MI300A, S=512 D=128):

| H | 4-barrier (µs) | parallel-dot (µs) | speedup |
|---|---|---|---|
| 1 | 1888 | 2140 | **0.88× (−12%)** |
| 8 | 1897 | 2161 | **0.89× (−11%)** |
| 16 | 2892 | 2707 | 1.07× |
| 32 | 2920 | 2722 | 1.07× |
| 64 | 2935 | 2733 | 1.07× |
| 128 | 3189 | 3055 | 1.04× |

Stable across 3 runs. The low-H regression is decisive: **if per-block
PREDICT/OUTPUT compute were the binding resource, parallelising the
length-D dots would speed up every H, especially low H** (where there is
no occupancy to fall back on — only `D/Db = 4` blocks exist). It did not:
at H=1 the parallel-dot is 12% *slower* because the per-token critical
path is **not** compute-limited, so the extra `__shfl_xor` reduction is
pure overhead on a path that was already barrier/serial-recurrence bound.
The 7% win at H≥16 is an **occupancy-mediated** effect (more blocks fill
the GPU), not a per-block latency breakthrough. The reordered summation
still passes `test_kda_correct` 11/11 (max_rel 1–2e-6 ≪ 2e-2 threshold —
the contractive recurrence keeps the reorder cost negligible), so this is
a clean speed question, not a correctness one.

## Conclusion: the kernel is at its practical per-block latency limit

Two negative results now localise the bottleneck precisely:
- **All four per-token barriers are required** (the 4→3 experiment is a
  cross-thread GATE-vs-OUTPUT race).
- **PREDICT/OUTPUT per-block compute is non-binding** (parallelising the
  length-D dots regresses low-H and gives only occupancy-mediated gains).

The kernel is therefore **latency-bound on the serial per-token recurrence
+ the four unavoidable per-token barrier latencies** — exactly the regime
the LDS cache left it in. No intra-block optimisation remains: barriers
can't be removed (race) and the per-phase compute can't be sped up
regresses the latency-critical case). The only lever that scales past one
block's serial token loop is **inter-chunk parallelism** below.

## Inter-chunk parallelism (#70): TAKEN — chunked WY kernel, 1.1–4.0x

**Status (2026-09-09, beverin MI300A): implemented, oracle-validated
(12/12 GPU configs, max_rel ≤ 6e-6), and benchmarked.** The chunked WY
forward (`kda_delta_rule_fwd_chunked[_with_scratch]`, `kda.hip` #L8) beats
the committed cooperative kernel at EVERY measured shape; the long-context
prefill shapes that motivated #70 are ~3.3–3.6x.

| Shape (B=1) | coop (µs) | chunked (µs) | speedup |
| --- | ---: | ---: | ---: |
| H=1 S=512 D=128 | 1897 | 718 | 2.64x |
| H=8 S=512 D=128 | 1963 | 727 | 2.70x |
| H=16 S=512 D=128 | 2887 | 729 | 3.96x |
| H=32 S=512 D=128 | 2966 | 1107 | 2.68x |
| H=64 S=512 D=128 | 2929 | 1744 | 1.68x |
| H=128 S=512 D=128 | 3191 | 2941 | 1.08x |
| H=32 S=1024 D=128 | 5978 | 1825 | 3.28x |
| H=32 S=2048 D=128 | 11758 | 3313 | 3.55x |

Median graph latency per forward, identical events/inputs; the state is
re-zeroed per sample. Reproduce from a clean checkout:
`SRC=$SCRATCH/vkernels sbatch meta/scripts/ab_kda_chunked_mi300.sh`
(copies live under `$SCRATCH/vkernels` on beverin; home is no longer used).

### Derivation (CPU-verified first)

The chunked primitives in `kda.cpp` (L4/L5/L6) implement the OLD
*standard* rule (scalar gate, pre-gate prediction) — NOT the K3 per-key-dim
oracle. The K3 chunked derivation is spelled out and CPU-verified in
`tests/kernels/attn/test_kda_k3_chunked.cpp`: forward-substitution form
(`k3_delta_rule_fwd`, ≤1e-6 vs oracle at 7 configs + full-history +
single-chunk) and the **affine WY form** (`k3_wy_chunked_fwd`, what the HIP
kernel implements op-for-op, ≤1e-6 incl. S=512 D=128 cs=64):

  M[t][j] = b_j Σ_k G_{j+1,t}[k] k_j k_t (strict tril);  Ainv=(I+tril M,−1)⁻¹
  N[t][j] = b_j Σ_k G_{j+1,t}[k] k_j q_t (incl diag)
  Kgw=exp(L_t)k_t; Qgw=exp(L_t)q_t; Kgb=b_t·exp(L_end−L_t)k_t; diagG=exp(L_end)
  U_v=Ainv v; W=Ainv Kgw; T=N U_v; Opar=Qgw−N W
  serial pass (rowblock-splittable): u=U_v−W Cᵀ; o=T+Opar Cᵀ; C=diagG⊙C+Kgbᵀu

All exp() arguments are ≤0 (products of gates in (0,1]): no overflow,
graceful underflow. **CONTRACT (stronger than the coop kernel): k must be
L2-normalised** (production contract) — |M|≤1 by Cauchy–Schwarz keeps the
explicit Ainv bounded; with unnormalised k, Ainv entries grow like |M|^(cs−1)
and overflow at cs=64 (measured 1e14; the forward-substitution form is
immune). Encoded in both the CPU test and `test_kda_chunked.hip`.

### HIP structure (kda.hip #L8) and what actually mattered

Four launches: (1) per-key log-cumsum; (2a) grams M/N — k and L staged at
full D in LDS (64 KB, the whole per-workgroup budget), q streamed, one
warp per (t,j) pair; (2b) Ainv (warp-per-16×16-diagonal-block forward
substitution + 3 blocked levels) and the U_v/W/T/Opar GEMMs; (3) the
nc-step serial state pass, split over D-row blocks like the coop kernel
(each state row is independent). Per-phase micro-tuning of the coop
kernel (barrier layout, dot parallelism) remains exhausted — do not
revisit; this section supersedes it.

Measured lessons (all on beverin, `kda_chunked_phase_times`):
- The obvious port (grams streaming from gmem, thread-per-pair) was
  **L2-latency-bound**: ~950 µs/block even at 512 threads. Staging k/L in
  LDS (→ all gram operands but q are LDS hits, 32 lanes × 1 float4 each)
  plus 512-thread blocks (LDS-capped at 1 block/CU → 16 warps) took the
  gram phase 950→268 µs.
- **LDS bank conflicts from D-strided rows**: sC rows at stride D (multiple
  of 32) put all 32 lanes in one bank on every state read — the whole
  serial pass ran ~20x slow. Stride D+4 fixes banks AND keeps rows 16B-
  aligned for float4.
- **float4 everything** (grams, GEMM streams, state pass): 4x fewer load
  instructions on latency-bound phases; another ~1.15–1.3x end-to-end.
- Splitting grams (2a) from inverse+GEMMs (2b) via a gmem M/N round-trip
  was neutral-to-positive and keeps each kernel's LDS small.

Remaining headroom (not taken): the gram phase is still the largest
single block cost (268 µs of 748 at H=1; ~25x above its SFU/exp floor) and
the H=128 throughput point sits at 1.08x — a deeper gram restructure
(e.g. safe two-factor gate products with per-16-token renormalisation,
FLA-style) or fp16 grams could push both, but the math risk grows.

These are diminishing returns against a serial recurrence; the LDS cache was
the single high-value win (it removed the dominant, H-scaling HBM traffic).
