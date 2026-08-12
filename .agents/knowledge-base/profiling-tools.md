# Profiling Tools Quick Reference

Tool selection, common commands, and metric interpretation for both AMD and
NVIDIA profiling stacks. For methodology, load the profiling skills:
[hip-kernel-profiling](../skills/hip-kernel-profiling/SKILL.md) or
[kernel-profiling](../skills/kernel-profiling/SKILL.md).

## Tool selection matrix

| Need | AMD tool | NVIDIA tool |
|---|---|---|
| Whole-program timeline (CPU+GPU) | `omnitrace` | `nsys` (Nsight Systems) |
| Single-kernel analysis | `omniperf` | `ncu` (Nsight Compute) |
| Raw counter dump | `rocprof` | CUPTI / `ncu --raw` |
| HIP/CUDA API trace | `rocprof --hip-trace` | `nsys --trace=cuda` |
| Memory/bandwidth only | `rocprof --metrics` | `ncu --metrics` |
| Source-line attribution | `omniperf --source` | `ncu --import-source yes` |
| Roofline model | `omniperf analyze --roof` | Nsight Compute Roofline tab |
| PC sampling / statistical | `rocprof --pc-sampling` | `nsys --sample=cpu` |

## omnitrace (AMD timeline)

```bash
# Basic capture
omnitrace -- omnitrace-instrument -- ./my_app

# With HIP API tracing
omnitrace --trace -- ./my_app

# Generate report
omnitrace-avail -r omnitrace-output/
```

## omniperf (AMD kernel analysis)

```bash
# Profile a specific kernel
omniperf profile -n my_kernel_name -- ./my_app

# List available profiles
omniperf analyze -p workloads/my_kernel_name/ --list-metrics

# Generate full analysis
omniperf analyze -p workloads/my_kernel_name/ --roof &

# Source-level attribution (requires -g -lineinfo at compile)
omniperf analyze -p workloads/my_kernel_name/ --source &
```

## rocprof (AMD raw counters)

```bash
# Basic timing + stats
rocprof --stats ./my_app

# HIP API trace
rocprof --hip-trace ./my_app

# Custom counter set
echo "pmc: SQ_WAVES,VALUInsts,MemUnitStalled" > counters.txt
rocprof -i counters.txt ./my_app

# Kernel occupancy
rocprof --stats -o results.csv ./my_app
```

## Nsight Systems (nsys) — NVIDIA timeline

```bash
# Basic capture
nsys profile -o report ./my_app

# With CUDA API + kernel trace
nsys profile --trace=cuda,nvtx,osrt -o report ./my_app

# Stats only (no timeline)
nsys stats report.nsys-rep
```

## Nsight Compute (ncu) — NVIDIA kernel analysis

```bash
# Full analysis
ncu --set full -o report ./my_app

# Specific kernel only
ncu --kernel-name my_kernel -o report ./my_app

# Source-level with line attribution
ncu --set full --import-source yes -o report ./my_app

# Single metric query
ncu --metrics sm__pipe_tensor_cycles_active ./my_app

# Roofline analysis
ncu --set roofline -o report ./my_app

# Launch count with no profiling overhead (just count)
ncu --launch-count 100 --launch-skip 10 ./my_app
```

## Key metrics to read first

### AMD (omniperf)

| Category | Metric | What it tells you |
|---|---|---|
| Compute | `VALUUtilization` | % of Vector ALU cycles active |
| Matrix | `MFMAInsts` | Matrix Core instructions issued |
| Memory | `MemUnitStalled` | Wavefronts waiting on memory |
| Memory | `MemUnitBusy` | Memory unit actively transferring |
| LDS | `LDSBankConflict` | Bank conflicts in LDS |
| Cache | `TCP_TCP_TA_TOTAL_ACCESS` | Total L1 cache accesses |
| Cache | `TCP_TCP_TA_TOTAL_MISS` | L1 cache misses |
| Occupancy | `SQ_WAVES` | Active wavefronts per CU |
| Stalls | `SQ_WAVE_STALLS` | Wavefront stall cycles (check sub-reason) |

### NVIDIA (ncu)

| Category | Metric | What it tells you |
|---|---|---|
| Compute | `sm__pipe_tensor_cycles_active` | Tensor Core utilization |
| Compute | `sm__inst_executed.avg` | Instructions per cycle |
| Memory | `dram__bytes.sum` | Bytes to/from HBM |
| Memory | `lts__t_bytes.sum` | Bytes through L2 |
| Memory | `l1tex__t_bytes.sum` | Bytes through L1 |
| Stalls | `smsp__warp_cycles_per_issue_stalled` | Cycles warps are stalled |
| Stalls | `smsp__average_warps_issue_stalled_barrier` | Waiting on barrier |
| Stalls | `smsp__average_warps_issue_stalled_long_scoreboard` | Waiting on memory |
| Stalls | `smsp__average_warps_issue_stalled_math_pipe_throttle` | Math pipe busy |
| Occupancy | `sm__occupancy_achieved` | % of theoretical occupancy |

## The diagnostic flow

1. **Is the kernel running at all?** → Check `omnitrace`/`nsys` for launch.
2. **Is it the right kernel that's slow?** → Timeline identifies the offender.
3. **Memory or compute bound?** → Check `MemUnitStalled` vs `VALUUtilization`
   (AMD) or `sm__pipe_tensor_cycles_active` vs DRAM bytes (NVIDIA).
4. **If memory-bound:** look at coalescing (FetchSize/WriteSize on AMD, sector
   misalignment on NVIDIA), L2 hit rate, and bank conflicts.
5. **If compute-bound:** look at Matrix Core % and **which stall reason**
   dominates. The stall reason names the missing resource.
6. **If overlap-limited:** timeline shows gaps between kernels or no overlap
   between streams.

## Common profiling mistakes

- **Profiling a debug build** — compile with `-O2` or `-O3`, keep `-g -lineinfo`
  for source attribution but not `-O0`.
- **Including the first launch** — always warm up before profiling.
- **Comparing profiled vs unprofiled numbers** — profiling adds overhead; the
  profiler diagnoses, the benchmark verifies.
- **Reading every metric** — read only the metric tied to your roofline
  hypothesis.
- **Calling `hipDeviceSynchronize`/`cudaDeviceSynchronize` inside the profiled
  region** — this serializes and distorts the timeline.
