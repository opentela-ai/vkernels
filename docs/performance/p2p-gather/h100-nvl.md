# p2p-gather — adaptive P2P run-list gather (NVIDIA H100 NVL, sm_90)

Issue #6: the single-launch kernel (PR #5) is fast for heavily fragmented
run lists but slower than the per-run `cudaMemcpy2DAsync` loop for the
common few-run case. This doc records the H100 NVL measurements that drive
the adaptive dispatch, and the acceptance sweep.

## Environment

Measured 2026-08-11 on sgs-gpu07 (4x H100 NVL, CUDA 13.0 / driver
580.82.07, sm_90, peer access enabled). The issue #6 table used physical
devices 2 and 3; here the same NV12 NVLink pair is reproduced with the
benchmark's `--src-device 1` (dst on GPU 0, source on NVLink peer GPU 1).
Payloads follow the Qwen3-14B KV geometry: 64-token pages, 40 layers,
8 KV heads, head dim 128, BF16, 192 pages per layer, ~48 MiB per launch.
Medians of 50 iterations per point; the machine shows a few % run-to-run
variance (the copy loop's high-run numbers swing most).

## Before (PR #5 kernel, issue #6 table)

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

The PR #5 kernel was flat (~253 us at 48 MiB) and crossed the copy loop
between 16 and 32 runs. The vectorized uint4 kernel measured here is
~43 us faster (~210 us at 48 MiB), so the crossover moved down to ~3 runs.

## After: adaptive dispatch, vectorized kernel, prepared plan

Idle GPU, real NVLink peer reads (`p2p_gather_bench --src-device 1`,
48 MiB fixed payload, issue #6 acceptance sweep):

| Runs | Copy loop baseline | Gather kernel | Adaptive | Branch | Adaptive vs baseline |
|---:|---:|---:|---:|---:|---:|
| 1 | 201.70 us | 210.05 us | 202.24 us | copy | 1.00x |
| 2 | 206.75 us | 209.98 us | 207.17 us | copy | 1.00x |
| 4 | 224.19 us | 210.18 us | 209.92 us | kernel | 1.07x |
| 8 | 258.02 us | 210.02 us | 210.21 us | kernel | 1.23x |
| 16 | 313.34 us | 210.40 us | 210.18 us | kernel | 1.49x |
| 32 | 455.46 us | 210.53 us | 210.59 us | kernel | 2.16x |
| 64 | 759.94 us | 211.36 us | 211.33 us | kernel | 3.60x |
| 192 | 1790.91 us | 214.34 us | 214.30 us | kernel | 8.36x |

Concurrent-compute (a 256 MiB memory-bound fill kernel on a second stream)
does not change the picture on this machine — the gather is a few % of SM
and NVLink capacity either way: 1.00x / 1.00x / 1.07x / 1.23x / 1.48x /
2.16x / 3.60x / 8.41x over the same run counts. Same-device D2D (no
`--src-device`) shows the same crossover shape with a much cheaper copy
loop: adaptive 1.00x / 1.02x / 1.15x / 1.45x / 2.02x / 3.15x / 5.30x
(idle; concurrent is within a few %).

Acceptance criteria (issue #6):

- adaptive within 5% of the copy-loop baseline for 1-16 runs: yes — 1-3
  runs take the copy path (1.00x), 4+ take the kernel (faster);
- wins at 32/64/192 preserved/improved: yes — 2.16x / 3.60x / 8.36x
  (PR #5 measured 1.15x / 1.49x / 2.91x; the vectorized kernel also wins
  from 4 runs, which PR #5's 0.77-0.98x never did);
- 48 MiB swept at 1, 2, 4, 8, 16, 32, 64, 192 runs, idle and concurrent:
  tables above;
- descriptor preparation/allocation reported separately: `adap_host_us` /
  `kernel_host_us` columns — the kernel path pays ~7.7 us host time
  (validation + descriptor construction + cudaMallocAsync + H2D + launch)
  flat across run counts, vs the copy path's per-run enqueue cost (up to
  ~70 us at 16 runs);
- prepared plan: below.

Smaller payloads only strengthen the kernel case (the copy loop has a
~20 us per-call floor, the kernel ~8.6 us): at 4 KiB the kernel wins even
for a single run (1.86x) and the 4 MiB sweep shows 3.4x at 8 runs up to
185x at 2048 runs. The dispatch model reflects this: below 1 MiB total the
run-count floor does not apply and the model decides from one run.

The 2-D (strided) path is modelled separately — cudaMemcpy2DAsync has a
lower per-call floor (~10.8 us) and the 2-D kernel pays one block per row,
so the model keeps a single small tile on the copy engine (1.00x vs
baseline) and switches to the kernel at ~16+ tiles (11x at 16 tiles,
26x at 64, 41x at 256 in the 64x512-tile sweep above).

## Dispatch model

Fitted to the idle peer table above (`csrc/vkernels/comm/p2p_gather.cpp`):

- copy engine ≈ max(20 µs, 4.20 µs/MiB) + 7.37 µs per extra run;
- gather kernel ≈ max(8.6 µs, 4.38 µs/MiB), flat in run count;
- floor: at ≥1 MiB payload the kernel is not eligible below 4 runs (the
  1-2 run margins are ~1%, inside noise); below 1 MiB the model decides
  from one run. Tunable at runtime via `set_gather_dispatch(mode, min_runs)`
  with `kForceKernel` / `kForceCopyEngine` for A/B.

## Prepared plan (KVAAS 40-layer reuse)

One run list prepared once, executed 40 times (8 runs, 48 MiB total —
above the crossover, so each execute is one kernel launch):

- prepare (validate + descriptor build + persistent device upload): 0.18 ms
  one-time;
- per-execute device time: 206.6 us (kernel path);
- per-execute host enqueue: 5.0 us — no validation, no allocation, no H2D
  metadata copy per layer (the one-shot kernel path pays ~7.7 us host
  including staging, allocation and H2D).

## Reproduce

```sh
cmake --preset cuda -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset cuda
# idle GPU, same-device simulation:
./build/cuda/benchmarks/p2p_gather_bench
# idle GPU, real NVLink peer reads (GPU 0 <- GPU 1):
./build/cuda/benchmarks/p2p_gather_bench --src-device 1
# concurrent-compute (filler kernel on a second stream):
./build/cuda/benchmarks/p2p_gather_bench --concurrent --src-device 1
```

## Journal

2026-08-11 — Issue #6 implemented and measured on sgs-gpu07: adaptive
dispatch (fitted model + floor), vectorized uint4 kernel with scalar tail,
prepared-plan API (host + CUDA + C ABI), acceptance benchmark with
prep/allocation separated and concurrent mode. The uint4 vectorization cut
the kernel's flat 48 MiB cost from ~253 us (PR #5) to ~210 us, moving the
crossover from ~19 runs down to ~3; the model was re-fitted accordingly.
All 15 host+CUDA tests pass on the target (three CUDA tests were fixed on
first GPU run: one used an overlapping run list, two assumed zeroed
cudaMalloc memory).
