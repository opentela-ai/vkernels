# moe-fused — K3 PP=3 pipeline per-kernel profile (AMD gfx942, MI300A)

A per-kernel profile of the **eager vs. breakable piecewise-cudagraph**
MoE path served through `src/python/vkernels/vllm_experts.py` on a
6-node MI300A (gfx942) deployment of Kimi-K3 (dummy weights), pipeline
parallel size 3 (93 hidden layers -> **31 layers/PP**). The goal is to
locate the throughput-limiting kernel on the pipeline bottleneck stage
and frame the next optimization.

The `moe:vkernel_apply` / `moe:apply.{cpu_copy,cpu_align,gpu_copy,
launch}` `record_function` regions emitted by `vllm_experts.py` are
parsed by `meta/benchmarks/moe_profile.py`, which reports **per-call
means** (reliable regardless of cross-thread overlap) rather than
aggregate sums (a chrome trace spans many overlapping threads and
overcounts aggregates).

## Workload

Two jobs, same code (`origin/main` @ 1f14d61), only
`K3_BREAKABLE_PIECEWISE=1` (-> `VLLM_USE_BREAKABLE_CUDAGRAPH=1`)
toggled. `LOAD_FORMAT=dummy`, `GEN_CORRECTNESS_SMOKE=1`,
`STEP_PROFILE=1` (torch.profiler, ~20 s steady-state capture, ranks
0/8/16). PP=3, TP=8, EP. `out_tok=256`, `in_tok~50`, `C=8`, `N=16`.

| job | path | wall | agg tok/s | per-req | lat p50 |
|:---:|:---:|---:|---:|---:|---:|
| 603394 | eager | 162.2 s | 25.30 | 3.60 | 92.0 s |
| 603395 | breakable | 116.4 s | **35.20** | **4.70** | **62.5 s** |
| | **delta** | **1.39x** | **1.39x** | **1.31x** | **1.47x lower** |

Per-step on the PP0 gate (31 layers/step; wall/step = 8/agg):

| component | eager (603394) | breakable (603395) | delta |
|:---|---:|---:|---:|
| wall / step | 316.8 ms | 227.3 ms | **1.39x** |
| MoE / step (31 x `moe:vkernel_apply` mean) | 130.2 ms (41%) | 122.6 ms (54%) | 0.94x (unchanged) |
| non-MoE / step (wall - MoE) | 186.6 ms (59%) | 104.7 ms (46%) | **1.78x (captured graph)** |

**The entire 1.4x throughput win is the non-MoE captured graph** (the
attention + MLP + sampling majority runs as a cudagraph replay with no
per-op Python dispatch, and under breakable it is partly hidden under
PP0's MoE sync waits). **The MoE region is unchanged (0.94x)** -- it runs
the same `VkernelFusedExperts.apply` eager-break body in both paths.

## Per-call MoE means (us, dominant compute thread per rank)

| set | rank | n_apply | vk_apply | cpu_cp | cpu_al | gpu_cp | launch | cpu_cp% |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager | 0 (PP0) | 3420 | 4199.4 | 4185.1 | 83.5 | 174.1 | 238.7 | **100%** |
| eager | 8 (PP1) | 3596 |  242.6 |  214.5 | 85.5 | 175.6 | 237.8 | 88% |
| eager | 16 (PP2) | 3596 |  234.4 |  189.5 | 86.9 | 175.4 | 254.7 | 81% |
| breakable | 0 (PP0) | 5160 | 3953.3 | 3844.4 | 110.5 | 153.0 | 301.6 | **97%** |
| breakable | 8 (PP1) | 5332 |  758.3 |  660.3 | 113.4 | 155.2 | 281.1 | 87% |
| breakable | 16 (PP2) | 5270 |  802.0 |  727.2 | 98.5 | 148.8 | 281.0 | 91% |

Reproduce:

```bash
python3 meta/benchmarks/moe_profile.py \
  --label eager   run-603394/step_profiles/step_profile_rank0.json \
                 run-603394/step_profiles/step_profile_rank8.json \
                 run-603394/step_profiles/step_profile_rank16.json \
  --label breakable run-603395/step_profiles/step_profile_rank0.json \
                 run-603395/step_profiles/step_profile_rank8.json \
                 run-603395/step_profiles/step_profile_rank16.json \
  --head-to-head
```

(Traces live under `$B=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin`
on beverin; the request-level throughput / latency table is in the
cookbook's `deployments/llm/beverin/kimi-k3-vllm/BENCHMARK.md`.)

## Findings

1. **PP0 is the pipeline gate.** Its `moe:vkernel_apply` mean is
   **17.3x (eager) / 5.2x (breakable)** slower than PP1/PP2. PP0 computes
   the expert routing (`topk_ids`) on its own GPU and pays the full
   GPU->host sync; PP1/PP2 receive the routing via `recv_object` (already
   on host) and only do a ~190-760 us memcpy.
2. **`moe:apply.cpu_copy` dominates PP0's MoE (97-100%).** It is the
   `topk_ids.contiguous().view(-1).cpu().numpy()` (and `expert_map.cpu()`)
   host round-trip at the top of `VkernelFusedExperts.apply`. Its ~4 ms
   duration is dominated by **waiting for the expert-dispatch all-to-all
   that produced `topk_ids`** -- the `.cpu()` is a stream synchronization
   point. `gpu_copy` / `launch` / `cpu_align` are each <1-8% of MoE.
3. **The MoE region is unchanged between paths** (4199 -> 3953 us, 0.94x),
   so the 1.4x throughput gain is entirely the non-MoE captured graph.
4. **PP1/PP2 regressed ~3x under breakable** (234 -> 758/802 us) --
   currently hidden under PP0's gate, so it becomes the new floor once
   PP0 is fixed and must be addressed in the same change.

## Caveats

- Aggregate `record_function` sums across a rank overcount (overlapping
  vLLM V1 compute + async threads); only the per-call means and the
  per-thread spans reported here are trustworthy. The request-level
  numbers in BENCHMARK.md (agg tok/s, latency p50) are independent of
  the trace and fully reliable.
- Dummy weights (`LOAD_FORMAT=dummy`) are used to avoid the A_log
  fault and isolate the serving path; per-kernel timings are
  representative of the real-weight serving path (MoE compute is
  weight-independent; confirmed real-vs-dummy parity in BENCHMARK.md).
- `max_num_seqs=8` serializes the C=32/C=64 client pools beyond the
  per-request timeout -- a harness artifact affecting both jobs equally,
  not a path defect.

## Journal

Profiled the PP0 gate to locate the remaining throughput limit after the
capture-safe MoE integration (PR #46) and the stream-aware-launch change
(launch `fused_moe_mxfp4` on `torch.cuda.current_stream()`, removing the
device-wide `torch.cuda.synchronize()`) both landed on `main`. The
stream-aware change removed the *explicit* device sync, but
`moe:apply.cpu_copy` -- the `topk_ids.cpu()` needed to feed the
CPU-only `moe_align_block_size` -- is now the dominant residual sync on
PP0 (97-100% of its MoE). The breakable win is real (1.4x) and entirely
in the captured non-MoE graph; the MoE eager-break body is unchanged
because the alignment host round-trip still runs in both paths.

## Future work — on-device `moe_align_block_size`

The single highest-value follow-up is to **eliminate the PP0
`topk_ids.cpu()` host round-trip** by moving `moe_align_block_size` onto
the GPU and keeping `sids` / `eids` / `topk_ids` on-device for the
duration of `apply`:

- Add a HIP `moe_align_block_size` kernel (histogram + prefix scan +
  block-aligned padding, matching the CPU reference in
  `src/c/vkernels/kernels/moe_fused.cpp` and the
  `vk_moe_align_block_size` C ABI which is currently CPU-only) that
  writes `sids` / `eids` into the persistent device buffers already used
  by `_scratch.get(("sids", ...), ...)` and `d_eids`.
- Size the `gateup_swiglu` / `down_combine` launch grids by the **max**
  aligned token-expert count (`ceil(M * top_k / block_size)`) and have
  each block early-out on a padding sentinel written by the align
  kernel, so the host no longer needs the data-dependent scalar `EM` to
  size the grid. This is what makes the whole MoE region
  **stream-ordered**: the align kernel + the MoE kernels all issue on
  `current_stream()` with no host wait for the dispatch all-to-all.
- The PP1/PP2 breakable regression (~3x, finding 4) must be fixed in the
  same change -- once PP0 is no longer the gate, the ~760-800 us/layer
  on PP1/PP2 (~24 ms/step) becomes the new floor. Investigate before
  claiming the end-to-end win.

Acceptance: per-step MoE on PP0 drops from ~130 ms toward the
~7-9 ms of pure `launch` GPU work (the alignment sync is the only thing
between them today), with **no per-request latency regression** and the
CPU-oracle equality preserved (`vk_moe_align_block_size` parity tests in
`tests/` + `meta/benchmarks/test_capi_moe.hip`).
