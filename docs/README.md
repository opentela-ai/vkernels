# vkernels — Supported Kernels & Primitives

This document lists every kernel and communication primitive in vkernels,
along with the mathematical computation each performs. Every operation follows
the **two-implementation model**: a CPU reference (oracle, always compiled)
and a GPU-accelerated path (CUDA or HIP, compiled when the toolkit is present).

All kernels operate on **float32** unless noted. Inputs are C-contiguous
arrays. Contract violations raise exceptions (C++), `ValueError` (Python),
or `Error::InvalidArgument` (Rust).

---

## Element-wise kernels

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `add(a, b)` | `out[i] = a[i] + b[i]` | float32 | CUDA |
| `scale(x, α)` | `out[i] = α · x[i]` | float32 | CUDA |
| `relu(x)` | `out[i] = max(x[i], 0)` | float32 | CUDA |

- **File**: `src/c/vkernels/kernels/elementwise.cpp` (CPU), `.cu` (CUDA)
- **Python**: `vkernels.kernels.add / scale / relu`
- **Rust**: `vkernels::kernels::add / scale / relu`

---

## Reduction kernels

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `sum(x)` | `Σ x[i]` (float32-accumulated) | float32 → float32 | CUDA (two-stage) |
| `max(x)` | `max x[i]` | float32 → float32 | CUDA |

- **File**: `src/c/vkernels/kernels/reduce.cpp` (CPU), `.cu` (CUDA)
- **Python**: `vkernels.kernels.sum / max`
- **Rust**: `vkernels::kernels::sum / max`
- Empty input raises an error on all backends.

---

## GEMM (SGEMM)

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `gemm(M, N, K, α, A, B, β, C)` | `C = α · A @ B + β · C` | float32 | CUDA (tiled 16×16) |

- **File**: `src/c/vkernels/kernels/gemm.cpp` (CPU), `.cu` (CUDA)
- **Python**: `vkernels.kernels.gemm(A, B, alpha=1.0, beta=0.0)` (shapes inferred)
- **Rust**: `vkernels::kernels::gemm(M, N, K, alpha, A, B, beta, C)` (explicit dimensions)
- Row-major layout. `A` is M×K, `B` is K×N, `C` is M×N.
- CUDA kernel uses shared-memory tiling with 16×16 thread blocks.

---

## GEMM (bf16 MFMA, gfx942)

A tiled bf16 dense matrix multiply built on the AMD K16 bf16 MFMA
(`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`) for the **Kimi-K3 projection
shapes** — the ones that today fall back to AITER's untuned
"torch solution:0" because `bf16_tuned_gemm.csv` has no gfx942 entries.

| Function | Computation | Data type | GPU backend |
|---|---|---|---|
| `gemm_bf16_cpu(M, N, K, α, A, B, β, C)` | `C = α·A@B + β·C` (per-output fp32 dot, single RNE bf16 store) | bf16 (uint16_t) | CPU (oracle) |
| `gemm_bf16_config_for(M, N, K, &bm, &bn, &bk, &threads)` | serving `M≤64`→`(16,16,64)` (measured; `BN=16` saturates the 228 CUs, 1.4–2.9× over `BN=64`), warmup `M>64`→`(64,64,256)`, `BK=64` | — | CPU |
| `hip::gemm_bf16(M, N, K, α, A, B, β, C)` | tiled K16-MFMA GEMM, cooperative `uint2` loads, M/N/K bounds-checked | bf16 (uint16_t) | HIP |

- **Layout**: `A` is M×K, `B` is K×N (the **transposed** projection
  weight `W[N,K].T`), `C` is M×N. All K3 `K` are multiples of 64, so
  `BK=64` never triggers its (defensive) K bounds-check; `N` is a
  multiple of 16 (e.g. 6288 is 393×16, not a multiple of 64), handled by
  a column bounds-check.
- **MFMA fragment layout** mirrors the empirically verified `mfma_k64_pf`
  helper in `moe_fused.hip` (A `m=lane%16`, B `n=lane%16`, C
  `col=lane%16, row=(lane/16)*4+i`), generalised to arbitrary `(BM, BN)`.
- **Files**: `src/c/vkernels/kernels/gemm_bf16.{hpp,cpp,hip}`;
  `hip::gemm_bf16_with_config` (the explicit-tile dispatcher used by the
  autotuner) is forward-declared by the harnesses, not in the public
  header (keeps discovery at three entries).
- **Tests**: `tests/kernels/gemm/test_gemm_bf16.cpp` (host),
  `meta/benchmarks/test_gemm_bf16_correct.hip` (device vs CPU).
- **Docs**: [kernels/gemm_bf16.md](kernels/gemm_bf16.md),
  [performance/gemm-bf16/gfx942.md](performance/gemm-bf16/gfx942.md)

---

## MoE (Mixture of Experts) — AMD gfx942 / CDNA3 low-level primitives

These fill gaps where CDNA4-only (gfx950) instructions used in the AITER flydsl
MXFP4 fused-MoE path do not lower on gfx942.

### #12 — Software direct-to-LDS fill

| Function | Computation | Data type | Backend |
|---|---|---|---|
| `direct_lds_fill_bf16(lds_dst, global_src, elements)` | Copy `elements` bf16 values from global → LDS | bf16 (uint16_t) | HIP (vectorised loads), CPU (memcpy) |

Replaces CDNA4's `rocdl.raw_ptr_buffer_load_lds` with vectorised global loads
(`global_load_dwordx4`) into VGPRs followed by LDS stores at lane-major offsets.

- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)

### #13 — Software fp4→bf16 dequant

| Function | Computation | Data type | Backend |
|---|---|---|---|
| `fp4_to_bf16_dequant(packed, scale)` | `out[2i], out[2i+1] = fp4_to_bf16(packed[i] & 0xF, (packed[i] >> 4) & 0xF) · scale` | fp4 (packed uint8) → bf16 (uint16_t) | HIP, CPU |

Decodes E2M1 microscaling format (sign|2-bit-exponent|1-bit-mantissa, two
values per byte, low nibble first).

- **Representable fp4 values**: 0, ±0.25, ±1.0, ±1.5, ±2.0, ±3.0, ±∞, NaN
- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.fp4_to_bf16_dequant(packed, scale=1.0)`

### #14 — Platform async-copy gate

| Function | Computation |
|---|---|
| `use_async_copy_default()` | Returns `true` if async copy should be enabled; `false` on gfx942 |

- Defaults to OFF on gfx942 (CDNA3, MI300X/A) where the async-copy path
  misbehaves. ON everywhere else.
- Override with env var `K3_NO_ASYNC=0` (force ON) or `=1` (force OFF).

- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)

### #15 — K16 bf16 MFMA

| Function | Computation | Data type |
|---|---|---|
| `mfma_f32_16x16x16bf16(c[4], a[2], b[2])` | `C₀₋₃ += A₀₋₁ ⊗ B₀₋₁` (16×16×16 bf16, fp32 accum) | bf16 packed → fp32 |

A single `v_mfma_f32_16x16x16bf16_1k` instruction on gfx942. K32 bf16
MFMA (CDNA4-only) is emulated by calling this K16 function twice:
once for K=0..15 (low halves of A/B), once for K=16..31 (high halves).

The standalone primitive issues the instruction via inline asm with pinned
VGPRs; the fused-MoE kernel instead calls the clang builtin
`__builtin_amdgcn_mfma_f32_16x16x16bf16_1k` directly, which keeps register
allocation correct where the pinned-VGPR form corrupted accumulators and
crashed for >64-thread blocks on gfx90a.

**Fragment layout** (one warp, lane `0..63`):
- A operand: `m = lane % 16`, `a[i]` packs rows `m` for K `k0+i` (`k0 = (lane/16)*4`)
- B operand: `n = lane % 16`, `b[i]` packs column `n` for K `k0+i`
- C accumulator: `col = lane % 16`, `row = (lane/16)*4 + i` — four **consecutive rows** per thread (not transposed)

- **File**: `src/c/vkernels/kernels/moe.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.mfma_f32_16x16x16bf16(c, a, b)`
- **Docs**: [kernels/moe.md](kernels/moe.md)

---

## MoE Aux — MXFP4 orchestration (quant, sort, scatter-reduce)

The five per-block data-movement primitives from issue #22 that bracket a
grouped MXFP4 GEMM: per-token / per-group activation quantization,
token→expert gather (activation **and** scales), and the routed
scatter-reduce combine (fp32 partials, plus a bandwidth-reduced MXFP4 form
that dequantizes inline). On gfx950 these are AITER's
`module_moe_mxfp4_aux` (its ~82 KB LDS exceeds the gfx942 / MI300A 64 KB
limit), so vkernels re-implements them as portable host references + HIP
kernels.

| Function | Computation |
|---|---|
| `mxfp4_moe_quant(A, group_size)` | bf16 → packed E2M1 + ue8m0 per group (low nibble = even K) |
| `mxfp4_moe_sort(A, sorted_ids, top_k)` | gather `A` into expert-grouped, block-aligned `[EM, hidden]` (pad rows zeroed) |
| `mxfp4_moe_sort_scales(scales, sorted_ids, top_k)` | same gather for the per-token ue8m0 scales |
| `mxfp4_moe_scatter_reduce(partial, topk_w, sorted_ids, M, width, top_k)` | bias-free weighted scatter-add of fp32 partials → `out[M, hidden]` |
| `mxfp4_moe_scatter_reduce_q(partial_q, partial_s, topk_w, …, group_size)` | same combine with the partial in MXFP4, dequantized inline |

- `moe_align_block_size` (above) produces `sorted_ids`; the pipeline is
  `align → sort → quant → sort_scales → fused_moe_mxfp4 → scatter_reduce[_q]`.
- Scale bytes are clamped to `[1, 254]` (never `0`); `0xFF` is the explicit
  zero-group flag. Largest finite E2M1 is `FP4_MAX = 3.0`.
- **Files**: `src/c/vkernels/kernels/moe_aux.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.mxfp4_moe_{quant,sort,sort_scales,scatter_reduce,scatter_reduce_q}`
- **Docs**: [kernels/moe_aux.md](kernels/moe_aux.md)

---

## MoE Fused — End-to-end MXFP4 grouped GEMM

Implements the full xkernels `fused_moe_mxfp4` interface by wiring together
the low-level primitives above. Two HIP kernels (plus a routing-weight gather):

### Kernel 0 — gate_up + SwiGLU (`gateup_swiglu_kernel`)
```
act[EM, ispp] = silu(clamp(A_sorted @ w13_gate + b13_gate, L))
              · clamp(A_sorted @ w13_up   + b13_up,   L)
```
Gate and up are accumulated in the same kernel and the SwiGLU product is
rounded to bf16 **once**, matching the xkernels oracle exactly (a split
3-kernel pipeline introduced a spurious intermediate bf16 rounding). The
gate value is clamped to `L` *before* the sigmoid (SwiGLU clamp).

### Kernel 1 — down + routed combine (`down_combine_kernel`)
```
out[M, hidden] += act @ w2^T · topk_w_sorted + b2
```
(scatter-add by token row via `atomicAdd`, weighted by the routing weight)

| Function | Tile constants | Data types | Backend |
|---|---|---|---|
| `fused_moe_mxfp4(..., block_size=16)` | decode: BM=16, BN=64, BK=64, 64 threads/block | bf16 activations, fp4 (E2M1) weights with ue8m0 scales, fp32 output | HIP, CPU |
| `fused_moe_mxfp4(..., block_size=64)` | prefill: BM=64, BN=64, BK=64, 256 threads/block | same | HIP |

`block_size` selects the tile config: `16` = decode (16×64, 64 threads),
`64` = prefill (64×64, 256 threads). The caller aligns with the matching
`block_size` (so `expert_ids` is indexed per 16- or 64-row block).

- Dequantization (E2M1 + ue8m0) is done inline during the K-loop — no full
  bf16 materialization of the 138 GB expert weight buffer.
- MFMA via `__builtin_amdgcn_mfma_f32_16x16x16bf16_1k`; A-tiles staged with
  vectorised `uint2` global loads; N-dimension is split across `blockIdx.z`
  (fixed 64-thread blocks, avoiding gfx90a pinned-VGPR crashes).
- `act_scratch` is indexed by **sorted row** (`EM`), not token (`M`) — this
  is required for `top_k > 1`, where the same token appears in multiple
  experts and would otherwise race.
- The caller must zero-initialise `out` (down-combine accumulates into it).

### Distributed (TP / EP / PP) — issue #18

The fused kernel is single-device; `vkernels.dist` (C++ `dist/dist_moe.hpp`,
Python `vkernels.dist`) shards the weights so per-rank shards are consumed
verbatim by the fused kernel's stage functions, and provides the
orchestration around them:

- **TP** — gate/up weights split along `hidden`, down weights along `ispp`;
  the linear stages are separated (`moe_gateup_cpu` / `moe_down_cpu`) so
  rank partials can be all-reduced *before* the nonlinear epilogues.  The
  multi-rank forward matches the CPU oracle.
- **EP** — experts partitioned across ranks; `moe_ep_dispatch` produces the
  all-to-all / sort re-layout with local expert ids.
- **PP** — `pp_boundary_send`/`recv` fix the stage-boundary transfer
  interface (graph-capturable primitive, issue #10) and `round_bf16`
  re-quantises the bf16 stage input.
- **Files**: `src/c/vkernels/dist/dist_moe.cpp` (+ `.hpp`), stage split in
  `src/c/vkernels/kernels/moe_fused.{cpp,hip}`, Python `src/python/vkernels/dist.py`
- **Tests**: `tests/kernels/moe/test_dist_moe.cpp`,
  `tests/python/test_dist.py`, `meta/benchmarks/test_moe_fused_dist_correct.hip`
  (GPU vs CPU oracle)
- **Docs**: [kernels/moe_dist.md](kernels/moe_dist.md)

### Expert alignment helper

| Function | Computation |
|---|---|
| `moe_align_block_size(topk_ids, M, top_k, block_size, num_experts)` | Maps `[M, top_k]` token→expert routing into block-aligned `sorted_ids` and `expert_ids` |

- `sorted_ids` stores the **flat topk index** (`token*top_k + sel`), padded
  per expert with `M*top_k`. This preserves the selection index so the
  routing-weight gather (`sw[i] = tw[sorted_ids[i]]`) is correct for
  `top_k > 1`. Consumers derive `token = flat / top_k`.
- **Files**: `src/c/vkernels/kernels/moe_fused.cpp` (CPU), `.hip` (HIP)
- **Python**: `vkernels.kernels.moe_align_block_size / fused_moe_mxfp4`
  (CPU-reference backed; see [python-bindings.md](python-bindings.md))
- **Docs**: [kernels/moe_fused.md](kernels/moe_fused.md)

### Verified performance (vs xkernels torch-loop)

> **Full benchmark records** (reproduce commands, per-M tables, primitives,
> caveats, journal): [`docs/performance/moe-fused/gfx90a.md`](performance/moe-fused/gfx90a.md)
> (MI250X) and [`docs/performance/moe-fused/gfx942.md`](performance/moe-fused/gfx942.md)
> (MI300A).

E=256, hidden=4096, ispp=512, top_k=6. Latency in ms (lower is better);
TFLOP/s is arithmetic on the *padded* EM rows, so the decode regime is
heavily padding-dominated.

| M | vkernels HIP (MI250X) | vkernels HIP (MI300A) | xkernels torch (MI250X) | xkernels torch (MI300A) |
|---|---:|---:|---:|---:|
| 1  | 0.66 | 0.45 | 6.0  | 4.4  |
| 8  | 3.4  | 0.89 | ~30  | ~15  |
| 16 | 4.1  | 2.2  | ~40  | ~30  |
| 48 | 10.4 | 4.0  | 162.2 | 127.2 |

GPU results are exact matches against the CPU reference
(`max_rel < 0.00001` decode, `< 0.02` prefill) on both gfx90a and gfx942.

The prefill config (E=8, top_k=2, denser routing) wins once each expert
fills its 64-row block: ~1.2× on gfx90a (M ≥ 512), ~1.3–1.5× on gfx942
(M ≥ 1024). With sparse routing (few tokens/expert) the 64-row padding
dominates and decode stays faster — see [kernels/moe_fused.md](kernels/moe_fused.md)
for full numbers.

---

## Communication primitives

### Ring all-reduce

| Function | Computation |
|---|---|
| `ring_allreduce_rank(local, rank, world, next, prev)` | Sum-reduces `local` in-place across `world` ranks via ring topology |
| `ring_allreduce(locals)` | Simulates all ranks in one process: every rank's `local` becomes `Σ locals[0..world-1]` |

- **File**: `src/c/vkernels/comm/allreduce.cpp`
- **Python**: `vkernels.comm.ring_allreduce(locals)`
- **Rust**: `vkernels::comm::ring_allreduce(&[a, b])`

### P2P run-list gather

Single-launch gather of many disjoint byte-runs from peer UVA into a local
scratch buffer, replacing per-run `cudaMemcpyPeerAsync` loops.

| Function | Computation |
|---|---|
| `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, N)` | For each `i`: `dst[dst_offsets[i]:...] = peer[src_ptrs[i]:...]` (1-D) |
| `p2p_gather_runs_2d(dst, runs)` | Strided 2-D tiles: `height × width` bytes per run, with independent src/dst strides |

- Adaptive dispatch: copy engine below the crossover run count, single kernel
  launch above it.
- Plan API (`P2PGatherPlan1D/2D`) for reuse across layer launches.
- Vectorized 16-byte (`uint4`) path for aligned runs.

- **File**: `src/c/vkernels/comm/p2p_gather.cpp` (CPU), `.cu` (CUDA)
- **Docs**: [p2p-gather.md](p2p-gather.md)
- **Performance**: [performance/p2p-gather/](performance/p2p-gather/)

### P2P KV restore (fused)

Fuses peer gather + indexed scatter into one kernel:

| Function | Computation |
|---|---|
| `p2p_kv_restore(k_dst, v_dst, slot_ids, peer_src_ptrs, ...)` | Reads KV data directly from peer UVA and writes into indexed K/V slot destinations |
| `kv_scatter(k_dst, v_dst, scratch, slot_ids, ...)` | Indexed scatter of an already-gathered contiguous scratch buffer (the second stage, for the PR #9 gather+scatter baseline) |

- Eliminates the intermediate scratch buffer and separate scatter launch.
- **File**: `src/c/vkernels/comm/p2p_kv_restore.cpp` (CPU), `.cu` (CUDA)

**Prepared plan (issue #27)** — `P2PKvRestorePlan` (host + CUDA) and the C
ABI `vkernels_p2p_kv_restore_plan_t` validate the slot map and upload the
page descriptors ONCE; `execute(k_dst, v_dst, source_layer_offset_bytes,
stream)` then launches a single page-by-token-group kernel. The destination
is taken per call because KVAAS/SGLang keep a distinct K/V pair per model
layer — one plan fans one run list out across all 40 layers with no
per-layer allocation or H2D copy. Three creation modes:

- Host `slot_ids` (`const int*`): validated at create, owned copy uploaded.
- `from_device_slots` (`const int*`, e.g. SGLang's `device_indices`):
  borrows the device pointer, no D2H sync, no content validation.
- `from_device_slots_int64` (`const int64_t*`): SGLang's `torch.int64`
  indices; converted at create (device-side on CUDA, no sync) into an owned
  int32 buffer so the caller may free the int64 buffer immediately.

The C ABI mirrors all three (`vkernels_p2p_kv_restore_plan_create`,
`..._create_device_slots`, `..._create_device_slots_int64`) plus
`vkernels_p2p_kv_restore_plan_execute_offset(plan, k_dst, v_dst, offset,
stream)`.

### Compute/communication overlap

| Class / Function | Computation |
|---|---|
| `OverlapExecutor.run(iters, compute_fn, comm_fn)` | Runs `compute_fn` on stream A and `comm_fn` on stream B in lockstep, returning a `Result(compute_count, comm_count)` |

- In-order execution within a stream, concurrency across streams.
- **File**: `src/c/vkernels/comm/overlap.cpp`
- **Python**: `vkernels.comm.OverlapExecutor()`
- **Rust**: `vkernels::comm::OverlapExecutor::new()`

### HIP/RCCL transport + OFI/CXI net plugin — issue #19

A second HIP/RCCL channel behind the existing `Channel` / all-reduce
interface, plus a HIP-aware OFI/CXI net plugin for Slingshot RDMA instead
of Socket, and graph-capturable all-reduce variants. Built on ROCm only.

| Surface | Role |
|---|---|
| `RcclChannel` | `Channel` over RCCL send/recv (host→device→host) |
| `RcclAllreducePlan` | Graph-capturable host plan: single `rcclAllReduce`, no host allocation after construction |
| `RcclAllreducePlanHip` | HIP path: one `rcclAllReduce` between begin/end graph capture |
| `resolve_transport(bytes, edges, cfg)` | Adaptive Socket↔Slingshot selection from a cost model |
| `est_rccl_socket_us` / `est_rccl_ofi_us` | Socket = `max(50, 6.0 us/MiB) + 25 us/edge`; OFI = `max(20, 3.0 us/MiB)` (RDMA, edge-free) |
| `discover_ofi_cxi` | Detects the `cxi` libfabric provider for the net plugin |
| `vkernels_rccl_*` | C ABI wrapping the host reference (always compiled) |

- **Host reference**: `src/c/vkernels/comm/rccl.{cpp,hpp}` (always compiled)
- **HIP/RCCL path**: `src/c/vkernels/comm/rccl.hip`, `rccl_hip.hpp` (`VKERNELS_HAS_RCCL`)
- **C ABI**: `src/c/vkernels/comm/rccl_c.{h,cpp}`
- **OFI/CXI net plugin**: `plugins/rccl-net-ofi/` (`librccl-net-ofi.so`, `VKERNELS_HAS_OFI`)
- **Build discovery**: `meta/cmake/RcclSupport.cmake`
- **Bench**: `meta/benchmarks/bench_rccl.cpp` (`rccl_bench`)
- **Docs**: [comm-rccl.md](comm-rccl.md)

---

## Core infrastructure

| Component | Description |
|---|---|
| `Device(index)` | GPU device selection, synchronization, peer-access queries |
| `Stream()` | In-order task queue; one worker thread per stream |
| `Span<T>` | Non-owning view of contiguous memory (C++ only) |

---

## File layout

```
src/c/vkernels/
├── kernels/
│   ├── elementwise.{cpp,cu}     # add, scale, relu
│   ├── reduce.{cpp,cu}          # sum, max
│   ├── gemm.{cpp,cu}            # tiled SGEMM
│   ├── gemm_bf16.{cpp,hip}      # bf16 K16-MFMA GEMM (gfx942, #29)
│   ├── moe.{cpp,hip}            # gfx942 primitives (#12–#15)
│   ├── moe_aux.{cpp,hip}        # MXFP4 MoE orchestration: quant, sort, scatter-reduce (#22)
│   └── moe_fused.{cpp,hip}      # fused MXFP4 MoE grouped GEMM
├── comm/
│   ├── allreduce.{cpp,cu}       # ring all-reduce
│   ├── p2p_gather.{cpp,cu}      # single-launch peer gather
│   ├── p2p_kv_restore.{cpp,cu}  # fused KV restore
│   ├── overlap.cpp              # compute/comm overlap executor
│   ├── channel.cpp              # blocking queue & mock channel
│   ├── topology.{cpp,hpp}       # ring topology helpers
│   ├── rccl.{cpp,hpp}           # HIP/RCCL transport host reference (#19)
│   ├── rccl.hip                 # HIP/RCCL all-reduce (VKERNELS_HAS_RCCL)
│   ├── rccl_hip.hpp             # RcclChannel / plan declarations
│   └── rccl_c.{h,cpp}           # C ABI for the RCCL transport
└── core/
    ├── device.cpp               # Device abstraction
    ├── stream.{cpp,cu}          # Stream (async task queue)
    └── allocator.cpp            # CUDA memory pool tuning
```

## Language bindings

| Language | Module | Docs |
|---|---|---|
| Python | `vkernels.kernels`, `vkernels.comm`, `vkernels.core` | [python-bindings.md](python-bindings.md) |
| Rust | `vkernels::kernels`, `vkernels::comm`, `vkernels::core` | [rust-bindings.md](rust-bindings.md) |
| C | `vkernels_c_*` (C ABI via `capi.hpp`) | — |
