# p2p-gather — adaptive P2P run-list gather (NVIDIA H100 NVL, sm_90)

Issue #6: the single-launch kernel (PR #5) is fast for heavily fragmented
run lists but slower than the per-run `cudaMemcpy2DAsync` loop for the
common few-run case. This doc records the H100 NVL measurements that drive
the adaptive dispatch, and the acceptance sweep to re-run on the target.

Environment (from issue #6):

- sgs-gpu07, physical CUDA devices 2 and 3, peer access enabled
- H100 NVL, CUDA 13, sm90
- Qwen3-14B KV geometry: 64-token pages, 40 layers, 8 KV heads, head
  dimension 128, BF16
- 192 pages per layer, ~48 MiB copied per measured launch
- PR #5 head tested: `74f47bce6a0c8ea4ce4643960d85a222e8d2070d`

## Baseline: PR #5 kernel vs per-run cudaMemcpy2DAsync loop (issue #6 table)

| Coalesced runs | `cudaMemcpy2DAsync` loop | PR gather | Gather speedup |
|---:|---:|---:|---:|
| 1 | 195.49 us | 252.64 us | 0.77x |
| 2 | 198.85 us | 252.58 us | 0.79x |
| 4 | 206.02 us | 252.64 us | 0.82x |
| 8 | 220.16 us | 252.93 us | 0.87x |
| 16 | 247.46 us | 253.25 us | 0.98x |
| 32 | 292.54 us | 254.34 us | 1.15x |
| 64 | 382.37 us | 256.74 us | 1.49x |
| 192 | 784.29 us | 269.22 us | 2.91x |

Crossover between 16 and 32 runs. Model fit used by
`prefer_gather_kernel` (see `csrc/vkernels/comm/p2p_gather.cpp`):

- copy engine ≈ 4.07 µs/MiB (48 MiB / 195.5 µs) + 3.08 µs per extra run
  (slope (784.29 − 195.49) / 191)
- gather kernel ≈ 5.27 µs/MiB (253 µs / 48 MiB), flat in run count
- crossover at ~19 runs; dispatch floor 24 keeps the 1-16 run range on the
  copy engine (0% regression vs the baseline by construction).

## Implementation (current)

- Adaptive dispatch in the CUDA path and the prepared plans
  (`kAdaptive` default, `kForceKernel` / `kForceCopyEngine` for A/B).
- Vectorized 16-byte `uint4` copy path with scalar tail for aligned runs;
  byte-per-thread fallback for unaligned runs (flag computed once on host).
- Prepared plan API (`P2PGatherPlan1D`/`2D`): validate + upload metadata
  once, execute() enqueues only — the KVAAS one-run-list-40-layers pattern
  with no per-layer allocation or H2D copy.

## Reproduce (run on the H100 NVL target)

```sh
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda
# idle GPU:
./build/cuda/benchmarks/p2p_gather_bench
# concurrent-compute (filler kernel on a second stream):
./build/cuda/benchmarks/p2p_gather_bench --concurrent
```

The bench prints, for each run count in {1, 2, 4, 8, 16, 32, 64, 192} at
48 MiB: baseline (copy loop), forced kernel, adaptive path + branch taken,
host enqueue (prep) time, and the prepared-plan prepare-once/execute-40
timing. Acceptance criteria to verify:

- adaptive ≤5% slower than baseline for 1-16 runs (expect ~0%: same path);
- adaptive ≥ PR #5 wins at 32/64/192 runs;
- plan: 40 executes with no per-layer allocation/H2D (prepare_ms reported
  once, per-execute host time ~0).

## Journal

2026-08-11 — Issue #6 kick-off: implemented adaptive dispatch with a fitted
cost model and 24-run floor, vectorized the copy kernel (16-byte chunks +
scalar tail), added the prepared-plan API (host + CUDA + C ABI), extended
the benchmark to the acceptance sweep with prep/allocation reported
separately and a concurrent-compute mode, and added host tests for the
dispatch policy and plan semantics (100% line coverage on p2p_gather.cpp)
plus CUDA runtime tests for both dispatch branches, vectorized tails and
plan reuse. Host-only verification so far (no GPU on the dev box); the
table above is to be refreshed with the adaptive column and the plan
timing on sgs-gpu07.
