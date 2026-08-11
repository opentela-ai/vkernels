# The Benchmark Harness

A benchmark that swings between runs optimises nothing. The harness exists so
one timed sample is a stable estimate of steady-state performance. Every
choice below serves that one goal.

## Warmup, then time

Run a throwaway batch **before** the timed window. Warmup fills caches,
establishes page tables, runs driver one-time paths, and — on GPU — lets any
persistent kernel state reach steady state. Never include the first launch in
the timed numbers; the first launch can be 10–100× the steady-state time.

A practical split: warmup for ~10% of the total runtime budget, then time the
remainder. If the kernel is sub-microsecond, warmup is still needed — but the
timed window must contain many iterations so the timer can resolve it.

## Iteration count driven by variance

Pick the timed iteration count so that (a) the window is far above the timer's
resolution and (b) the **coefficient of variation** (CV = std/mean) is small.

- Timer resolution: a CUDA event timer resolves ~0.5 µs; `steady_clock`
  resolves ~1 µs on Linux but jitter is larger. Make the timed window at
  least ~100× the resolution, ideally milliseconds, so one timer read is a
  small fraction of the total. If the kernel is 2 µs, run ≥5000 iterations so
  the window is ≥10 ms.
- Convergence: start with a guess, then keep doubling the iteration count
  until CV settles (commonly <1–2%). If CV does not settle, the kernel is
  non-deterministic in a way worth understanding (a race? thermal? a stream
  not actually synced?) before trusting any number.
- Report the **per-call** time as `window / iterations`, and also keep the
  window time itself so a reader can judge resolution.

A fixed "100 iterations" is a guess. The count is correct when CV is small,
not when it is round.

## Use the right clock

- **GPU kernel timing → CUDA events.** `cudaEventRecord(start, stream)`
  before the timed loop, `cudaEventRecord(stop, stream)` after, then
  `cudaEventSynchronize(stop)` and `cudaEventElapsedTime(&ms, start, stop)`.
  Host wall-clock around a launch measures launch + sync, not the kernel, and
  adds host scheduling jitter. Put the events on the **same stream** the
  kernel uses.
- **Host CPU-reference kernel → `std::chrono::steady_clock`.** `steady_clock`
  is monotonic and not subject to wall-clock adjustments; `system_clock` is
  not. Time a loop of iterations, divide for per-call.
- **Host-side overlap / comm timing → `steady_clock`** for the host-arranged
  path (the project's `OverlapExecutor` and host `ring_allreduce` run on host
  threads). When comparing a CUDA comm path, time it on its stream with
  events, matching how the compute path is timed.
- **Do not mix clocks in one comparison.** A host-reference number (steady
  clock) and a CUDA number (events) measure different things; report them as
  *correctness* agreement plus a *separate* GPU-vs-roof number, never as a
  naive "GPU is Nx faster than CPU" from mismatched timers.

## Lock the environment

GPU variance is dominated by **clock and thermal fluctuation** (10–20% is
normal unattended). Before a serious run:

```bash
# discover supported clocks, then lock to a fixed point and cap power
nvidia-smi -q -d SUPPORTED_CLOCKS
nvidia-smi -lgc <minGpuClockMHz>,<maxGpuClockMHz>   # lock GPU clocks
nvidia-smi -ac <memClockMHz>,<gpuClockMHz>          # (older chips) app clocks
nvidia-smi -pl <watts>                              # power limit
# afterward, restore defaults so the machine is left clean
nvidia-smi -rgc
```

Prefer a fixed clock point (not a range) and disable boost, so a run that
heats up cannot downclock mid-benchmark. On the CPU side, pin frequency,
disable turbo, and set the performance governor (`cpupower frequency-set
-g performance`) for the duration, then restore.

Keep one GPU process on the device — concurrent contexts contend for the same
resources you are trying to measure.

## Data lifetime and correctness

For a kernel with a CPU reference (every kernel in this project has one), the
**fastest** path to a wrong conclusion is a benchmark that diverges from the
test. Keep the harness honest:

- The host reference and the CUDA kernel consume **the same input**, and
  neither mutates it unless the kernel is intentionally in-place (then give
  each its own copy).
- Time the kernel, not the setup. Allocate and fill inputs outside the timed
  window; `cudaDeviceSynchronize()` (or the stream's sync) **before** you
  start the event window, and again after, so only kernel execution is timed.
- Include the **C read** when `beta ≠ 0` and the **C write** always, in both
  the byte count and the timed work. Forgetting the C traffic is the most
  common way to make a "memory-bound" kernel look artificially close to the
  roof.

## Capturing the distribution

Report `min, median, mean, std` (and the CV). `min` is the best achievable —
the number to compare against the roof, because it strips scheduling jitter.
`median` is the typical experience. `std` (via CV) is the reproducibility
contract: if CV is large, no conclusion drawn from one run is sound, and the
harness itself is suspect before the kernel is.

Save the baseline as the tuple `(kernel, problem size, device, peak, achieved,
CV, harness config)` so a future run can diff against it under the same terms.
