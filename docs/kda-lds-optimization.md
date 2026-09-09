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

## Next levers (not yet taken)

1. **Profile to localise the latency** — `omniperf` single-kernel metrics
   (wavefront stall reasons, VALU utilization) on the S=512 D=128 H=128 case to
   confirm whether the barrier cost or the PREDICT/OUTPUT thread starvation
   dominates, per hip-kernel-profiling.
2. **Parallelise the dot products** — give each of the `Db` rows a small group
   of threads (length-D dot product split across r threads + `__shfl`/LDS
   reduce) so PREDICT/OUTPUT use more of the 256-thread block.
3. **Fewer barriers per token** — GATE→PREDICT and UPDATE→OUTPUT are true data
   dependencies (3 barriers needed); the 4th (post-OUTPUT) folds into the next
   token's GATE.
4. **Token-level pipelining / parallel scan** for inter-chunk parallelism (the
   #70 chunked-prefill work) — the only lever that scales past one block's
   serial token loop, but a much larger change.

These are diminishing returns against a serial recurrence; the LDS cache was
the single high-value win (it removed the dominant, H-scaling HBM traffic).
