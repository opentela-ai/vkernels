# MoE Distributed — TP / EP / PP orchestration for the fused MXFP4 MoE

Closes the single-device gap of the fused kernel (`fused_moe_mxfp4` /
`fused_moe_mxfp4_cpu` are single-rank: grid `(EM/BLOCK_M, N/BLOCK_N[, 4])`,
no rank/world/shard concept).  Kimi-K3 does not fit on one GPU — it needs
≥4 nodes × 4 MI300A = 16 GPUs (TP8 × PP2, ~1557 GiB) or 6 nodes (TP8 × PP3)
— so the distributed layer shards the weights such that **each rank's shard
is consumed verbatim by the fused kernel's stage functions**, and provides
the collectives / re-layouts the caller runs around them (issue #18).

- **Source (host reference)**: `src/c/vkernels/dist/dist_moe.cpp` (+ `.hpp`)
- **Stage split (CPU)**: `src/c/vkernels/kernels/moe_fused.cpp`
- **Stage kernels (HIP)**: `src/c/vkernels/kernels/moe_fused.hip`
- **Python host reference**: `src/python/vkernels/dist.py`
- **Tests**: `tests/kernels/moe/test_dist_moe.cpp` (host),
  `tests/python/test_dist.py`,
  `meta/benchmarks/test_moe_fused_dist_correct.hip` (GPU vs CPU oracle, TP)

---

## Why the fused call must be split

`fused_moe_mxfp4` fuses the nonlinear activation into the gate/up GEMM's
epilogue.  Tensor parallelism splits `hidden` across ranks, so rank *r*
computes only a **partial sum** of the gate/up GEMM over its K-slice; the
true value is the sum across ranks.  The activation is nonlinear, so the
all-reduce must happen **between the GEMM and the activation**.  The fused
call is therefore decomposed into four composable stages with the linear
outputs exposed:

```
Stage 0a  moe_gateup_cpu / moe_gateup_preact
          gate_up[EM, 2*ispp] fp32  = A_slice @ w13^T      (per rank)
          ↓  caller's TP all-reduce (ring / NCCL / RCCL)
Stage 0b  moe_act_epilogue_cpu / moe_act_epilogue
          act[EM, ispp] bf16 = act(gate_up + b13)
Stage 1a  moe_down_cpu / moe_down_preact
          partial[EM, hidden] fp32 = act_slice @ w2^T      (per rank)
          ↓  caller's TP all-reduce
Stage 1b  moe_combine_cpu / moe_combine
          out[M, hidden] fp32 += (partial + b2) · topk_w
```

`fused_moe_mxfp4_cpu` is exactly this composition with `k_base = 0` and the
full K dimensions — byte-identical results (validated by the existing
`moe_fused` tests).  The four CPU stages are in `moe_fused.cpp`; the HIP
stage kernels mirror them in `moe_fused.hip` (decode regime, 16×64 tiles).

## TP — tensor parallel

### Sharding (weight-layout-aware)

`dist::moe_tp_plan(hidden, ispp, tp)` validates the split against the kernel
constraints and returns the shard geometry; `moe_tp_shard_w13` /
`moe_tp_shard_w2` copy each rank's contiguous slice:

| Weight | Full layout | Rank `r` shard | Sharded dim |
|---|---|---|---|
| `w13` | `[E, 2·ispp, hidden/2]` | `[E, 2·ispp, hidden_shard/2]` | hidden |
| `w13_scale` | `[E, 2·ispp, hidden/32]` | `[E, 2·ispp, hidden_shard/32]` | hidden |
| `w2` | `[E, hidden, ispp/2]` | `[E, hidden, ispp_shard/2]` | ispp |
| `w2_scale` | `[E, hidden, ispp/32]` | `[E, hidden, ispp_shard/32]` | ispp |

Biases `b13`/`b2` are **not** sharded — they are added after the all-reduce
and every rank keeps a full copy.  The shards keep the fused kernel's layout
with the sharded dimension simply narrower, so the stage functions consume
them with `k_base = r · shard` and the shard width as the GEMM K.

Kimi-K3 (E = 112, D_INTER = 3072, hidden = 7168) with TP8: per-rank shards
896 × 384 — both multiples of BLOCK_K 64 and the ue8m0 group 32, so the
tiles stay intact (`moe_tp_plan` asserts this).

### Forward

```
per rank:  shard = moe_tp_shard_*(full, plan, r)
           gate_up_r = moe_gateup_cpu(A, shard, k_base = r·hidden_shard)
allreduce: gate_up = Σ_r gate_up_r                       ← caller's TP collective
           act     = moe_act_epilogue_cpu(gate_up, b13)
per rank:  partial_r = moe_down_cpu(act, shard, k_base = r·ispp_shard)
allreduce: partial = Σ_r partial_r                       ← caller's TP collective
           out     = moe_combine_cpu(partial, b2, topk_w)
```

Every rank ends with the same `out[M, hidden]` (the all-reduced result).
`dist::fused_moe_mxfp4_tp` runs the whole TP forward in-process over `tp`
simulated ranks (threads + ring channels) and matches the CPU oracle;
`fused_moe_mxfp4_tp_rank` is the per-rank step over a pair of
`comm::Channel`s for a real multi-process deployment.

## EP — expert parallel

Experts are partitioned across ranks (contiguous blocks;
`dist::moe_ep_plan(num_experts, ep, rank)`).  `moe_ep_dispatch` builds the
local block-aligned sorted layout: the `(token, sel)` pairs whose expert is
locally owned, masked and aligned exactly like `moe_align_block_size`, with
**local** expert ids (`0 .. num_local-1`) so a compact per-rank weight buffer
(starting at this rank's first expert) is indexed without a global offset.

```
per rank:  dispatch → local sorted_ids / expert_ids / EM      (all-to-all #1:
           tokens whose experts live here)                      activation rows)
           out_r = fused MoE on local experts (full hidden, no all-reduce)
           ↓ (all-to-all #2: weighted partial outputs back to token owners)
combine:   out = Σ_r out_r
```

`dist::fused_moe_mxfp4_ep` runs the in-process EP forward and returns the
per-rank partial outs; their element-wise sum is the MoE output (validated
against the oracle in tests).

## PP — pipeline parallel

The MoE layer is stage-local: each PP stage runs its transformer layers'
MoE with that stage's weights.  The stage boundary is a hidden-state
transfer, fixed here as the interface the graph-capturable primitive of
issue #10 must satisfy:

- `dist::pp_boundary_send` / `pp_boundary_recv` — fp32 `[M, hidden]` transfer
  over a `comm::Channel` (host reference; on gfx942 a deployment replaces the
  channel with a stream-ordered device copy / peer transfer captured into the
  graph segment so a replay needs no host progress).
- `dist::round_bf16` — re-quantises the fp32 stage output to the bf16
  activation format the next stage's MoE consumes (RNE).

The TP8 × PP2 acceptance pipeline (two stages, each TP) is exercised by
`DistMoe.PpTimesTpPipelineMatchesOracle` in C++ and
`PpBoundaryTest.test_pp_tp_pipeline_matches_oracle` in Python.

## Python host reference

`vkernels.dist` mirrors the C++ layer in pure numpy on top of the public
`vkernels.kernels` / `vkernels.comm` APIs, so it runs with or without the
compiled extension:

```python
from vkernels import dist

# TP8 x PP2 Kimi-K3 layout checks out
plan = dist.tp_plan(hidden=7168, ispp=3072, tp=8)   # shards 896 x 384

# Multi-rank TP forward vs the single-rank oracle
outs = dist.dist_moe_tp(A, w13, w13_scale, w2, w2_scale, topk_ids, topk_w,
                        b13=b13, b2=b2, top_k=8, tp=8)   # 8 equal outputs

# Expert-parallel forward; sum(outs) is the MoE output
outs = dist.dist_moe_ep(A, w13, w13_scale, w2, w2_scale, topk_ids, topk_w,
                        b13=b13, b2=b2, top_k=8, ep=4)
```

## GPU verification

`meta/benchmarks/test_moe_fused_dist_correct.hip` runs the four HIP stage
kernels over `tp` simulated ranks on gfx942 (with host-assisted all-reduces,
since vkernels' ring all-reduce is host-side) and compares every rank's
result against `fused_moe_mxfp4_cpu` — the acceptance criterion of #18.
A vLLM-style deployment substitutes NCCL/RCCL on device for the two
all-reduce points.

```
cmake --preset hip -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build --preset hip --target test_moe_fused_dist_correct
./build/hip/meta/benchmarks/test_moe_fused_dist_correct [situ]
```
