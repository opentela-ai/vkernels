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

## Remaining lever (not yet taken)

1. **Token-level pipelining / parallel scan** for inter-chunk parallelism
   (the #70 chunked-prefill work): FLA's `C_{c-1}`-decoupled solve lets
   chunks within a (b,h) run concurrently once the inter-chunk log-cumsum
   is known, breaking out of the one-block-per-(b,h) serial loop. This is
   the only remaining scaling lever but a much larger architectural change
   (the cooperative kernel would split into intra-chunk + inter-chunk
   passes, as `kda.cpp` already sketches). Per-phase micro-tuning
   (barrier layout, dot parallelism) is exhausted — do not revisit.

These are diminishing returns against a serial recurrence; the LDS cache was
the single high-value win (it removed the dominant, H-scaling HBM traffic).
