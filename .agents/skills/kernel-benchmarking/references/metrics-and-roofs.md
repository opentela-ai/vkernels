# Metrics and Roofs

The metric you measure is dictated by the regime the roofline skill predicts.
Below: how to derive each metric, the theoretical roof for common devices, and
the per-kernel target for this project. Numbers are approximate and meant for
sanity — always read the **actual** peak from the device you are on
(`nvidia-smi -q`, the datasheet, or `deviceQuery`).

## Compute: FLOP/s

```
achieved_FLOPs = (compute work in FLOPs) / (kernel time in seconds)
utilization   = achieved_FLOPs / peak_FLOPs
```

- One FP add or mul = 1 FLOP; one fused multiply-add `a*b+c` = **2** FLOPs.
- GEMM `C=A@B` (M×K by K×N) → `2·M·N·K` FLOPs.
- For a "real" GEMM count, include `beta≠0` adds only if your kernel does
  them; the Tensor-Core path typically does not.

Peak Tensor-Core throughput is dtype- and generation-specific (B200 ≈ 2 PFLOP/s
dense fp16/bf16, H100 ≈ 990 TFLOP/s fp16). A scalar/CUDA-core kernel's roof is
the CUDA-core peak instead (B200 ≈ 80 TFLOP/s fp32) — do not grade a
CUDA-core elementwise kernel against the Tensor-Core roof.

## Memory: GB/s and arithmetic intensity

```
achieved_GBps = (bytes moved) / (kernel time)
AI            = (FLOPs) / (bytes moved)        # FLOP/byte
ridge_point   = peak_FLOPs / peak_BW           # FLOP/byte; below = memory-bound
```

- **Bytes moved** are read at the level the roof is measured at: HBM bytes for
  the HBM roof, L2 bytes for the L2 roof, SMEM bytes for an on-chip roof. The
  default roof in this project is HBM.
- For GEMM with A read once, B read once, C written once, `beta=0`:
  `bytes ≈ 2·(M·K + K·N)·s + M·N·s_out`, ideal `AI ≈ 2MNK / bytes`. For
  square `M=N=K`, `AI ≈ N/3` at fp16 (`s=2`).
- Effective bandwidth for a pure copy/reduction = `bytes / time`; compare to
  the device's HBM peak (B200 ≈ 8 TB/s HBM3e, H100 ≈ 3.35 TB/s HBM3).

## Common device reference points (approximate, sanity only)

| Device | fp16/bf16 Tensor Core | fp32 CUDA core | HBM |
|---|---|---|---|
| B200 | ≈ 2000 TFLOP/s | ≈ 80 TFLOP/s | ≈ 8 TB/s (ridge ≈ 250 FLOP/B) |
| H100 | ≈ 990 TFLOP/s | ≈ 67 TFLOP/s | ≈ 3.35 TB/s (ridge ≈ 295 FLOP/B) |
| A100 | ≈ 312 TFLOP/s | ≈ 19.5 TFLOP/s | ≈ 1.55 TB/s (ridge ≈ 200 FLOP/B) |

## Per-kernel targets in this project

### Elementwise (`add`, `scale`, `relu`)
- **Regime:** memory-bound, always. Read 1–2 input tensors, write 1 output,
  ~1 op/element → AI ≪ ridge.
- **Metric:** effective HBM GB/s = `(bytes_in + bytes_out) / time`.
- **Target:** as close to the HBM roof as possible. A naive launch reaches
  50–70%; coalesced/vectorized reaches 80–90%+. **If it does not approach the
  roof, the bug is coalescing or too few bytes in flight — not compute.**

### Reduction (`sum`, `max`)
- **Regime:** memory-bound; the final reduction tree adds latency but not
  much bandwidth.
- **Metric:** effective GB/s = `(bytes_in) / time` (the write is one scalar).
- **Target:** close to the HBM roof for a single-pass reduction. Watch for
  **divergence** in the tree phase (lanes masked off) and for a partial
  reduction that round-trips through memory. A two-pass (write partials, read
  them back) kernel is ~2× off the roof and usually fixable.

### GEMM (`gemm`, fp32 here; the ladder in gemm-kernel-tuning)
- **Regime:** compute-bound for large `M,N,K`; memory-bound when tiles are
  small. Square `K` is the knob — `AI ≈ N/3` (fp16).
- **Metric:** FLOP/s = `2MNK / time`, reported as % of the Tensor-Core (or, for
  the fp32 CPU reference and any fp32 CUDA path, CUDA-core) peak.
- **Target:** a tuned dense GEMM reaches 80–95% of the Tensor-Core roof. The
  ladder's steps are measured at fixed `(M,N,K)`: step 1 (thread-copy) is
  far off; the TMA step is the first large jump; steps 5–9 close the
  remaining gap. A result that does not move at the TMA step means TMA is
  not actually overlapping — profile it.

### Ring allreduce (`comm::ring_allreduce`)
- **Regime:** bandwidth-bound asymptotically (2·(world−1) data transfers per
  element), latency-bound for tiny payloads.
- **Metric:** achieved allreduce bandwidth = `(payload bytes) / (wall time)`,
  and the per-hop latency for the latency-bound regime.
- **Target:** the theoretical ring bandwidth is `payload /
  (2·(world−1)·hop_latency + serialisation)`. Small payloads fall far below
  the bandwidth roof (latency floor); large ones approach it. Report **both**
  regimes, never a single number across sizes.

### Overlap executor (`comm::OverlapExecutor`)
- **Regime:** neither compute- nor memory-bound intrinsically — its metric is
  **overlap efficiency**, the fraction of wall time during which the compute
  stream and comm stream run simultaneously.
- **Metric:** `overlap = (T_compute + T_comm − T_wall) / T_wall` (so 0 = fully
  serial, up toward 1 = perfect overlap), measured via per-stream timing.
- **Target:** when the per-iteration `compute` and `comm` costs are similar
  and the data dependency (comm needs compute's output) is the only ordering,
  overlap should approach 1 for many iterations. If it stays near 0, the
  dependency is forcing serialization (the future is awaited too early) or
  the two streams are not distinct — profile with Nsight Systems.
