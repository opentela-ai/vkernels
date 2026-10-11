# Issue #156 — MoE decode expert-GEMV bandwidth (MI300A) — work log (v2)

Lane branch: `issue-156/gemv-bw`. Prior delivery: `7bf4773` on main (v1
`gemv_decode_{bf16,fp8_block}_splitk`, ceiling ~1.6-1.7 TB/s at tiny-M,
instruction/latency-bound — see the issue comment and the NOTES in that
commit). This lane implements the structural follow-up sketched there.

## v2 design (commit 9a9555b)

`gemv_decode_{bf16,fp8}_splitk_v2` in `src/c/vkernels/kernels/gemm_bf16.{hip,hpp}`
(**measured and REJECTED — see Results; kept in-tree as a documented
negative result, v1 remains the recommended path**):

1. **TB fixed at 8 (exact uint4).** The v1 default config tb=4 issued a
   16 B uint4 load per thread per K row but consumed only 8 B (4 columns),
   so every B byte was requested by FOUR threads; wavefronts overlapped
   4x on the same lines. v2 gives each thread 8 columns; a 64-thread
   wavefront covers 1 KiB of B contiguously with zero overlap.
2. **Multi-K register blocking (KK ∈ {2,4,8}).** KK independent in-flight
   uint4 B loads per thread per iteration, KK·8·M FMAs amortise the loop.
3. **Optional `a_gmem` A path.** A[m][k] read straight from global memory
   (uniform across threads; the split's slice is M·kl·2 B, L1-resident) —
   removes the LDS stage and its `__syncthreads`.
4. **Deferred-scale fp8 — REJECTED, replaced by the v1 contract.** The
   original v2 fp8 variant accumulated raw fnuz codes UNSCALED (uniform
   value·2^-119 domain of the branchless bit decode) and applied each
   (128-K × 8-N) block scale ONCE at the block boundary. Measured on
   MI300A (job 676545) it ran ~5-6x SLOWER than the bf16 v2 kernel
   (17-89 GB/s vs 250 GB/s at N=896): the second `M×8` accumulator bank
   (`blk[M][8]`) pushes VGPR pressure past the spill threshold. It also
   introduced a numerics contract that diverges from the #146 dequant
   oracle (no per-element `f2bf` rounding) and fails the rounded-oracle
   cross-check at small-magnitude outputs (max_rel ~3 on |C|≈0.03
   elements). Replaced by the v1/MFMA per-element-rounded contract
   (`w = bf16(f2bf(sc·0x1p119·decode(b)))`, bit-exact vs the dequant
   oracle) inside the same v2 structure — single accumulator bank,
   VGPR profile identical to bf16 v2.

Same fp32 partial planes + fixed ascending-S combine as v1 ⇒ combine-side
numerics contract unchanged; prefill V5 path untouched.

## Probe / tests

- `probe_issue156_batched.cpp` phase `v2`: sweep (S, th, kk, a_gmem) on
  N=6288/3584/896 K=7168 M=5, bf16 + fp8; batched 1000-launch event
  pairs × 4 batches + checksums + per-row CPU-oracle gates (2e-2 rel;
  fp8 vs the bf16-rounded dequant oracle). Probe stdout is now
  line-buffered (an earlier run lost buffered output sections).
- `test_gemm_bf16_correct.hip` section (6b): 14 v2 cases (odd/prime S,
  odd K=257 tails, β≠0, a_gmem, fp8 incl. splits spanning scale blocks
  and unaligned split starts) + M=9 / kk=3 rejection (C untouched).

## Jobs 676545 / 676552 / 676555 (beverin mi300, --exclusive -N1)

- 676545 (deferred-fp8 v2): gates PASS 0 failures, but the deferred fp8
  variant ran 17-89 GB/s (spills) and showed S=96 max_rel ~4-8 vs the
  exact oracle — two independent reasons to reject it.
- 676552 (rounded-contract v2): fp8 v2 now 331-500 GB/s, but a REAL bug
  at S=96 (max_rel up to 4.9): `kts_e` is element-granular, so a split's
  `k0` need not be 128-aligned and a scale-block boundary can fall
  MID-KK-group (kts=75: boundary at k=128 lies inside group [127,131)
  for kk=4); the per-group boundary check then applied the previous
  block's scale to the group tail.
- Fixed by iterating in SCALE-ALIGNED segments (fast KK loop only runs
  while it cannot cross a boundary; one scale reload per segment).
  Commit 91f48ec + fp8 S=96 unaligned-k0 kk=4 regression case.
- 676555: gates PASS 0 failures (all 6b v2 cases green incl. the new
  regression case); prefill V5 and moe_aux unchanged; full sweep in
  `logs/issue-156-gemv-v2-run.txt`.

## Job 676559 — v1-vs-v2 back-to-back A/B (same job, same clock state)

Cross-job clock variance is large (the v1 record run read 53.5 µs at
N=6288 bf16; the same config in this job read 65.8 µs), so the final
comparison runs BOTH kernel families in one exclusive job:

| shape (M=5, K=7168) | v1 best | v2 best | v2/v1 |
|---|---:|---:|---:|
| bf16 N=6288 | 65.80 µs | 72.34 µs | 1.10x slower |
| bf16 N=3584 | 52.90 µs | 59.28 µs | 1.12x slower |
| bf16 N=896  | 43.47 µs | 49.50 µs | 1.14x slower |
| fp8 N=6288  | 73.95 µs | 89.78 µs | 1.21x slower |
| fp8 N=3584  | 63.77 µs | 67.60 µs | 1.06x slower |
| fp8 N=896   | 51.77 µs | 62.36 µs | 1.20x slower |

(best config per family: v1 S=32/64 tb=4 th=128/256; v2 uniformly
S=64 th=64 kk=4 gm=0. Raw log: `logs/issue-156-gemv-v2-ab.txt`.)

**Why v2 loses.** (1) `acc[M][8]` doubles the accumulator registers vs
v1's `acc[M][4]` (40 vs 20 VGPRs just for accumulators, ~80-90 total vs
~40-50), halving occupancy on a latency-bound kernel. (2) TB=8 halves
the thread count covering N at equal `th`, reducing TLP; raising `th`
or `S` to compensate pays more in combine-plane traffic (M·N·S·4 B
round trip) than it wins in concurrency. (3) The exact-uint4 B
consumption does remove the v1 2x line overlap, but L2 already dedups
that, so the traffic win never materialises while the occupancy cost
is paid in full.

## Decision

- **v1 `gemv_decode_{bf16,fp8_block}_splitk` remains the recommended
  (and only routed) decode-GEMV path**; its records (1.6-1.7 TB/s bf16
  at N=6288, instruction/latency-bound) stand.
- v2 kernels stay in-tree as a tested, documented negative result
  (public entry points `gemv_decode_{bf16,fp8}_splitk_v2`, header
  marked experimental/not-recommended) so the A/B is reproducible.
- The deferred-scale fp8 contract is rejected outright (spills +
  contract divergence); any future fp8 GEMV work should keep the
  #146 per-element-rounded contract.
- Reusable finding: scale-block boundaries must be treated as
  segment boundaries in any split-K fp8 kernel with multi-K unrolling
  when split starts are not 128-aligned (regression case added).
