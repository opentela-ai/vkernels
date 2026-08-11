# The GEMM Ladder, Step by Step

The running example is fp16/bf16 GEMM on Blackwell with `B_M = B_N = 128`,
`B_K = 64`. Each step keeps the algorithm and changes how one tile is computed
or scheduled; keep every earlier version passing as a correctness reference.

## Step 1 — Sequential single-tile

One CTA computes one `128×128` output tile with `K = 64` (a single MMA, no
K-loop). It fixes the data path for everything after it:

```
GMEM --(thread-driven copy)--> SMEM --cta_sync()--> tcgen05.mma --> TMEM accumulator
TMEM --read back--> registers --convert--> GMEM store
```

`accum=False` starts a fresh accumulator (no earlier partial sum exists). This
is the baseline: it uses synchronous GMEM→SMEM copies rather than TMA, and
load, compute, and store run strictly in turn.

## Step 2 — K-loop accumulation

Lift the `K = 64` restriction by iterating K in `B_K` chunks. Reuse **one**
SMEM tile pair and **one** TMEM accumulator slot; `accum=True` from the second
iteration onward reads the previous partial sum from TMEM. Update the MMA
barrier's wait state each iteration. No new storage — operands stream through
fixed buffers and the accumulator stays in one TMEM slot.

## Step 3 — Spatial tiling (multi-CTA)

Partition the `M×N` output into `128×128` tiles and launch one CTA per tile on
a 2-D grid; `(bx, by)` identifies the output tile and each CTA runs the Step 2
K-loop internally. The per-CTA data path is unchanged; only the grid coverage
of the output changes.

## Step 4 — TMA async load

Replace the thread-driven `Tx.cta.copy` with TMA. **One thread** issues each
copy from a descriptor; the engine generates addresses and transfers the tile
asynchronously and can swizzle on write. Completion is tracked by an
**mbarrier with `expect_tx`**:

```python
T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)   # 1 arrival + bytes
# engine applies complete_tx as each transfer lands
T.ptx.mbarrier.try_wait(tma_bar, phase)                # wait before MMA reads SMEM
```

`byte_count = (B_M·B_K + B_N·B_K) · 2` for the two fp16 operand tiles. Step 4
still waits immediately after each TMA load, so load and compute do not yet
overlap — the win is that address generation and tile movement move off the
CTA threads. **This is the first large measured jump.** Full role-level
overlap arrives in Step 7.

Stores use a bulk async group on the writeback path: `cp_async.bulk.commit_group()`
then `cp_async.bulk.wait_group(0)` before `Dsmem` can be overwritten.

## Step 5 — Software pipeline (`PIPE_DEPTH = 2`)

Turn the A and B buffers into a **double-buffered SMEM ring**. While MMA reads
stage `k`, TMA prefetches stage `k+1`; stages exchange roles each iteration.
Each stage gets a `full` (TMA filled it) and `empty` (consumer done) barrier;
track `phase_tma` and `phase_mma` parities and flip each after its wait
succeeds:

```python
T.ptx.mbarrier.try_wait(mma_bar[stage], phase_mma); phase_mma ^= 1
```

A deeper pipeline (PIPE_DEPTH > 2) hides more memory latency but allocates
another `Asmem`/`Bsmem` pair per stage.

## Step 6 — Persistent kernel + tile scheduler

Launch a fixed pool of long-lived CTAs; each computes several output tiles in
a loop via a `ClusterPersistentScheduler2D`, cutting launch and repeated setup
overhead and improving **L2 locality** (nearby tiles reuse data sitting in L2).
The scheduler supplies the next tile coordinate; the mainloop and epilogue
need not know whether it came from a static formula or, later, CLC.

## Step 7 — Warp specialisation

Split the sequential load → MMA → writeback path in one warpgroup into three
concurrent roles connected by **four barriers**:

- **TMA producer** (one warpgroup) prefetches the next tile while the MMA
  consumer computes the current one.
- **MMA consumer** (one warpgroup) runs `tcgen05.mma`, committing completion
  with `tcgen05.commit(...mbarrier::arrive)`.
- **Writeback** (the consumer warpgroup's threads, or a dedicated path) reads
  the TMEM accumulator into registers, writes `Dsmem`, waits the whole tile is
  present (`warpgroup_sync(wg_id + 10)` — `cta_sync()` would deadlock because
  the producer warpgroup never reaches it), then one thread issues the TMA
  store.

Completion-signal type depends on the producer: TMA loads use `TMABar` (byte
counting), MMA uses `TCGen05Bar`, TMA stores use a commit/wait group. In Step 7
each signal updates a barrier in the current CTA only (`cta_mask=0`).
`PIPE_DEPTH = 2` is the minimum depth needed to overlap load and compute.

## Step 8 — Two-CTA cluster

Form a **two-CTA cluster** (`cta_group::2`) and run a cooperative MMA so two
CTAs share one larger output tile. Each CTA holds part of the SMEM operands
and reads the peer's slice through DSMEM; barrier arrivals now use
`cta_mask=3` (binary `11`) so they update the corresponding barriers in
**both** CTAs. Allocate TMEM with `cta_group=1` (per-CTA) and dispatch the MMA
with `cta_group=1`.

## Step 9 — Multi-consumer warp specialisation

Add a **second MMA consumer** so two groups of A blocks share the same staged
B tile, and a second writeback warpgroup (`warpgroup_sync(wg_id + 10)` and
`+ 11` — different barrier-slot IDs so their arrivals are not counted
together). Under the book's benchmark conditions the final kernel matches the
cuBLAS reference.

## Debugging a stalled pipeline

If the kernel waits indefinitely, inspect each barrier in turn: which role
waits, which role arrives, and whether the initialized arrival count matches
the number of notifications. If it finishes but produces the wrong result,
check whether a consumer reads data before the required wait **and fence**
have completed.
