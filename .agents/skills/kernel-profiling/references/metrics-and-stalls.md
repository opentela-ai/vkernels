# Metrics and Warp Stalls

Read the metric that maps to your roofline hypothesis. Each one below is
paired with what it indicates about the binding resource, so a number becomes
a cause.

## Memory metrics

- **`dram__bytes_read.sum` / `dram__bytes_write.sum` per second** → achieved
  HBM bandwidth. Compare to the device's HBM peak (the memory roof). Below the
  roof with high demand ⇒ something between HBM and the SM is limiting.
- **`lts__t_sector_hit_rate.pct`** — **L2 hit rate**. High demand on HBM with
  a *low* hit rate contradicts a tiling claim: your tiles are not being reused
  in L2 (wrong tile size, no L2-friendly launch order, or a persistent
  scheduler that does not preserve locality).
- **`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / replay** — global load
  volume and replays. **High replay ⇒ accesses are not coalescing.** Cross-check
  against [gpu-memory-layout](../../gpu-memory-layout/SKILL.md): a non-coalesced
  loop or a column read of a row-major tile will show here before anywhere.
- **`l1tex__data_bank_conflicts_pipe_lsu_mem_global_op_st.sum`** — shared
  memory bank conflicts (reported per access). Non-zero on the hot path means
  the SMEM layout is not swizzled or the access pattern does not match the
  swizzle mode you intended.
- **`sm__inst_executed_pipe_lsu.sum`** vs bytes — load/store instruction count
  per byte moved. High means you are issuing many small loads instead of
  vectorized/TMA bulk transfers.

## Compute metrics

- **`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`** —
  **Tensor Core utilization**, the headline for a compute-bound GEMM. Low with
  high HBM demand ⇒ memory-bound despite the label. Low with *low* demand ⇒
  the Tensor Core is stalling, not starving — read the stall reasons.
- **`sm__inst_executed.avg.pct_of_peak_sustained_active`** — overall
  instruction issue. Low ⇒ warps are not resident or are stalled.
- **`sm__warps_active.avg.pct_of_peak_sustained_active`** — achieved occupancy
  in flight. Remember occupancy is **not** a quality metric (per
  [cuda-efficient-kernels](../../cuda-efficient-kernels/SKILL.md)): a
  low-occupancy warp-specialized kernel is expected. Use this to confirm a
  stall is *not* because nothing is resident to hide it.

## Warp stall reasons (what the warps actually wait on)

`ncu` reports these sorted by weight. The top reason names the resource
behind the gap:

| Stall reason | Means | First place to look |
|---|---|---|
| **stall_long_scoreboard** | waiting on a global/SMEM memory op | coalescing, L2 hit, bytes in flight; load [gpu-memory-layout](../../gpu-memory-layout/SKILL.md) |
| **stall_short_scoreboard** | waiting on a local/short-latency op (often register->TMEM, shared) | excess register pressure, a dependent load too close |
| **stall_wait** | waiting on a sync/barrier/fence | wrong barrier phase, a wait that should have been overlapped; load [async-kernel-coordination](../../async-kernel-coordination/SKILL.md) |
| **stall_membar** | waiting on a memory fence | an unnecessary fence, or the wrong proxy fence |
| **stall_not_selected** | warp ready but not scheduled | occupancy/scheduling — usually benign unless it dominates |
| **stall_no_instruction** | instruction fetch | code size, I-cache pressure, divergence |
| **stall_drain** | waiting for outstanding ops on an exit path | a divergent exit that drains many warps |
| **stall_misc_trap / sync** | a trap or sync | correctness issue, not performance — investigate before optimising |

A stall is a *symptom of a resource*, not the resource itself. `long_scoreboard`
dominating a memory-bound kernel says "the loads you issued are not coming
back fast enough" — the fix is fewer bytes or more in flight, *not* "reduce
scoreboard stalls".

## Putting a number next to a roofline claim

Before accepting any reading, ask the one question the roofline skill makes
cheap: **does this metric move the binding resource?**

- Memory-bound kernel shows 60% of HBM roof and `long_scoreboard` dominant →
  the binding resource *is* memory; the gap is coalescing/in-flight depth, and
  no compute change will help.
- Compute-bound GEMM shows 70% of Tensor-Core roof with `wait` dominant and
  high memory throughput → the binding resource is *scheduling* (a barrier
  serializing load and compute); the fix is overlap, not more arithmetic.
- A reading that does not move the binding resource is a distraction. Name
  the resource first (roofline), read the matching metric, and ignore the rest
  until the bottleneck moves.
