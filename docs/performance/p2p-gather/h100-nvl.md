# p2p-gather — adaptive P2P run-list gather (NVIDIA H100 NVL, sm_90)

Issue #6: the single-launch kernel (PR #5) is fast for heavily fragmented
run lists but slower than the per-run `cudaMemcpy2DAsync` loop for the
common few-run case. This doc records the H100 NVL measurements that drive
the adaptive dispatch and the acceptance verification.

## Environment

- sgs-gpu07: 4x NVIDIA H100 NVL (95,830 MiB), driver 580.82.07,
  CUDA 13.0, Ubuntu 24.04
- topology: GPU0<->GPU1 over 12 NVLinks (NV12); GPU0/1<->GPU2/3 over PCIe
- benchmark: `p2p_gather_bench --src-device 1` — dst on device 0, src on
  device 1 (real NVLink peer reads, bidirectional peer access enabled by
  the bench), 50-iteration medians, idle GPU unless noted
- commit: `8876974` (implementation) + retune commits on top

The issue #6 table was measured with the non-vectorized PR #5 kernel
(252-269 µs at 48 MiB). The current kernel copies 16 bytes per thread
(`uint4`), which measured 210 µs — a ~17% kernel win on top of the
dispatch fix.

## Acceptance sweep: 48 MiB fixed payload, real NVLink peer (idle)

| Runs | baseline us | kernel us | adaptive us | branch | speedup |
|---:|---:|---:|---:|---|---:|
| 1  | 200.38 | 209.50 | 200.93 | copy   | 1.00x |
| 2  | 205.15 | 209.54 | 205.12 | copy   | 1.00x |
| 4  | 221.38 | 209.57 | 209.60 | kernel | 1.06x |
| 8  | 252.48 | 209.66 | 209.60 | kernel | 1.20x |
| 16 | 308.35 | 209.70 | 209.73 | kernel | 1.47x |
| 32 | 439.20 | 210.05 | 210.08 | kernel | 2.09x |
| 64 | 728.29 | 210.75 | 210.88 | kernel | 3.45x |
| 192| 1823.90| 213.63 | 213.60 | kernel | 8.54x |

Same sweep under concurrent compute (memory-bound fill kernel on a second
stream):

| Runs | baseline us | kernel us | adaptive us | speedup |
|---:|---:|---:|---:|---:|
| 1  | 200.26 | 209.57 | 200.83 | 1.00x |
| 2  | 204.93 | 209.60 | 204.83 | 1.00x |
| 4  | 220.42 | 209.73 | 209.50 | 1.05x |
| 8  | 252.26 | 209.70 | 209.73 | 1.20x |
| 16 | 308.35 | 209.76 | 209.70 | 1.47x |
| 32 | 439.14 | 210.08 | 210.18 | 2.09x |
| 64 | 729.18 | 210.88 | 210.88 | 3.46x |
| 192| 1832.06| 213.89 | 213.76 | 8.57x |

The kernel path is immune to the concurrent filler (it waits on NVLink
reads, not SMs); the copy-engine loop degrades ~0-2%. The acceptance
criteria hold in both modes:

- **adaptive ≤5% slower than baseline for 1-16 runs**: adaptive is never
  slower (1.00x/1.00x/1.06x/1.20x/1.47x; 1-2 runs take the identical copy
  path, 4-16 take the kernel which is faster);
- **wins preserved at 32+**: 2.09x/3.45x/8.54x (idle) vs the issue #6
  targets of 1.15x/1.49x/2.91x — better in both modes;
- **48 MiB sweep across 1/2/4/8/16/32/64/192 runs**: above, idle and
  concurrent.

## Fragmentation / shape behaviour (measured, same setup)

- 4 MiB total: 1.00x at 1 run (copy, floor), 3.3x at 8 runs, 10.9x at 32,
  184x at 2048 runs (per-run copy overhead dominates the loop).
- 4 KiB runs: 1.75x even at 1 run (the copy engine's ~16-20 µs per-call
  floor dominates 4 KiB; the model's <1 MiB payload rule sends it to the
  kernel), 45x at 64 runs, 152x at 2048.
- 2-D 64x512 tiles: 1.00x at 1 tile (single-tile stays on the copy engine
  — the 2-D strided model), 11.8x at 16 tiles, 28x at 64, 43.6x at 256.

## Prepared plan (KVAAS reuse pattern, 48 MiB, 8 runs, 40 executes)

- prepare once: **0.158 ms** (validation + descriptor build + persistent
  `cudaMalloc` + synchronous H2D upload)
- per-execute device time: **206.5 µs** (single kernel launch, stable
  across all 40)
- per-execute host enqueue: **4.2 µs** — no validation, no allocation, no
  H2D copy per layer; the one-shot path pays ~7 µs host per launch and the
  copy-engine path ~10-70 µs depending on run count.

## Model fit

Measured 48 MiB line: copy loop 200.4 µs @1 run + ~7.3-8.5 µs per extra
run (slope grows at high fragmentation); kernel flat ~209.7-213.6 µs.
Fitted constants used by `prefer_gather_kernel` (1-D / strided-2-D):

- copy engine: `max(20.0, 4.20*MiB) + 7.37*(runs-1)` /
  `max(10.75, 4.20*MiB) + 7.30*(runs-1)`
- gather kernel: `max(8.6, 4.20*MiB)` / `14.0 + 0.13*(runs-1) + 4.20*MiB`
- dispatch floor: 4 runs, applied only at ≥1 MiB 1-D payloads (the 1-2-run
  margins are ~1%, inside noise; below 1 MiB the engine never wins)

Crossover at 48 MiB is ~3 runs (kernel wins from 4). Retune for another
machine/driver via `set_gather_dispatch(mode, min_runs)` or by editing the
constants in `p2p_gather.cpp` (host-tested).

## Reproduce

```sh
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda
# idle GPU, real NVLink peer (src GPU1 -> dst GPU0):
./build/cuda/benchmarks/p2p_gather_bench --src-device 1
# concurrent-compute (fill kernel on a second stream):
./build/cuda/benchmarks/p2p_gather_bench --src-device 1 --concurrent
# same-device reference (D2D over HBM, ~6x faster than NVLink peer):
./build/cuda/benchmarks/p2p_gather_bench
```

The bench prints baseline (copy loop), forced kernel, adaptive path +
branch taken, host-enqueue (prep) time, and the prepared-plan
prepare-once/execute-40 timing. `--quick` drops to 10 iterations.

## Journal

2026-08-11 — Issue #6 kick-off: adaptive dispatch with a fitted cost model,
vectorized kernel (16-byte chunks + scalar tail), prepared-plan API
(host + CUDA + C ABI), benchmark acceptance sweeps with prep reported
separately and a concurrent-compute mode, host tests (100% coverage on
p2p_gather.cpp) and CUDA runtime tests. Host-only verification at the
time (no GPU on the dev box).

2026-08-11 — First real run on sgs-gpu07: the issue-table fit was off for
this machine (kernel 210 µs not 253; copy 201.7 µs @1 run + ~7.4 µs/run
not 195.5 + 3.08), so the model was retuned to the measured constants and
the floor dropped 24 -> 4 with a 1 MiB payload guard (below it the engine
never wins — the 4 KiB single-run case is 1.75x faster on the kernel).
The 2-D single-tile case exposed a shape-blind model (0.78x regression);
the model gained a strided variant with 2-D floors, restoring 1.00x and
matching the whole sweep. Full acceptance sweep verified idle and
concurrent (tables above); plan prepare 0.158 ms once, 206.5 µs per
execute, 4.2 µs host enqueue.
