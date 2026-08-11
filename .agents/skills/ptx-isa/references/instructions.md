# PTX Instruction Reference

The families a high-performance kernel is built from, with the concrete shapes
that matter. Tensor-Core and warp-coordination instructions change most across
generations; the rest are stable.

## Tensor Core MMA — the dominant FLOP source

A Tensor Core computes `D = A·B + C` at tile granularity in one instruction,
~10× the FLOP/s of a CUDA core. Three generations, increasing scope and
async-ness:

### `mma.sync` (Volta, Turing, Ampere)
- **Scope:** one warp (32 threads). **Synchronous** — returns when the tile is
  in registers.
- Operands and accumulators are **register fragments** distributed across the
  32 lanes (see the [gpu-memory-layout](../../gpu-memory-layout/SKILL.md)
  skill for the `@laneid`/`@reg` fragment layout).
- Shapes such as `m8n8k4`, `m16n8k8`, `m16n8k16`; mixed precision
  (`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`).
- Load/store fragments with `wmma.load`/`wmma.store` using matrix descriptors,
  or with manual `ldmatrix` (`ldmatrix.sync.aligned.m8n8.x4.shared.b16`).

### `wgmma.mma_async` (Hopper)
- **Scope:** a warpgroup (4 warps / 128 threads). **Asynchronous** — issue,
  then `wgmma.commit_group` + `wgmma.wait_group` to complete.
- Accumulator in **registers** (or shared for some shapes). A comes from
  registers or SMEM (via a descriptor); B comes from SMEM.
- Ordering: `wgmma.fence` before the issue (so prior shared writes are
  visible), `wgmma.commit_group` to bundle in-flight ops, `wgmma.wait_group N`
  to wait until ≤N groups remain.
- Descriptor encodes the SMEM operand's base, leading dim, stride, and swizzle
  — it must match the layout TMA wrote.

### `tcgen05.mma` (Blackwell, 5th generation)
- **Scope:** warpgroup; accumulator lives in **Tensor Memory**, not registers,
  cutting register pressure for large tiles.
- **One designated thread** issues the MMA and commits completion with
  `tcgen05.commit...mbarrier::arrive`; the epilogue waits that barrier and
  applies the required **tcgen05 fence** before reading the accumulator back
  through the four 32-lane TMEM windows.
- Block-scaled variants (MXFP8, NVFP4) read per-block scale factors (SFA/SFB)
  from TMEM; PTX requires SFA and SFB duplicated across all four partitions
  (`.warpx4`).

## Warp-level exchange and reduction

- **`shfl.sync.{bfly,up,down,idx}.b32 r, a, lane, mask`** — exchange a value
  between lanes of a warp in registers, no SMEM. `bfly` for reductions and
  nearest-neighbour; `idx` for arbitrary gather (broadcast with
  `srcLane` constant). Optional `.b32/.f32/...` type and predicate.
- **`redux.sync.{add,min,max}.u32`** — one-instruction integer warp reduction;
  faster than a `shfl` tree for the supported ops.

## Asynchronous copy

- **`cp.async.cg.shared.global`** (Ampere) — per-thread async global → shared
  load. Commit with `cp.async.commit_group`; wait with
  `cp.async.wait_group N` (≤N groups outstanding) or `cp.async.wait_all`.
  The non-bulk predecessor of TMA.
- **`cp.async.bulk.shared.global`** — TMA bulk copy from a tensor-map
  descriptor (see [../../async-kernel-coordination/references/tma.md](../../async-kernel-coordination/references/tma.md)).
  Load completion via mbarrier + `expect_tx`; store completion via
  `cp.async.bulk.commit_group` + `cp.async.bulk.wait_group`.
- **`cp.async.bulk.tensor.{1d..5d}.shared::cluster.global`** — multi-dimensional
  TMA from a descriptor.

## Atomics, fences, and the proxy model

- **`atom{.scope}.{relaxed/acquire/release/acq_rel}.global/shared.b32 dst,[addr],v`**
  — `scope` is `cta`, `cluster`, `gpu`, or `system`. Pair release with
  acquire across the right scope to form handoff patterns.
- **`fence.sc.sync`** — sequentially-consistent fence; combined scope arg.
- **Proxies.** Async operations (TMA, CLC) write through the **async proxy**;
  ordinary thread code uses the **generic proxy**. PTX requires explicit
  `fence.proxy.async` fences (before submitting a new async request and after
  reading its response) to order accesses across proxies — otherwise a later
  async write can race a value still being read.
- **Reductions do not form acquire patterns** (PTX §8.11.1): an atomic
  reduction is not a synchronization; pair it with a separate release/acquire
  if you need ordering.

## Control flow and predication

- Most instructions accept a guard: `@p insn ...` or `@!p insn ...`.
- `setp.{eq,ne,lt,le,gt,ge}.{u32,f32} p, a, b` builds a predicate; `selp r, a,
  b, p` selects. `and`/`or`/`xor` combine predicates.
- `bra label`, `call fn`, `bar.sync` (CTA barrier, alias `bar.sync 0`),
  `bar.warp.sync mask` (warp barrier), `barrier.cluster.arrive`/
  `barrier.cluster.wait` (cluster scope).
- **Divergence cost:** a warp following N taken paths pays N passes; prefer
  warp-uniform control flow, mask with predicates, or use `shfl`/`redux` to
  make decisions uniform.
