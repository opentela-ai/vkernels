---
name: async-kernel-coordination
description: Overlap a GPU kernel's stages — load, compute, store — with asynchronous hardware and coordinate the handoffs so no unit waits on another. Use when moving tiles with TMA (cp.async.bulk), synchronising with mbarriers (arrival, phase, expect_tx), building a software pipeline of shared-memory stages, splitting producer and consumer warps (warp specialisation), cooperating across CTAs with distributed shared memory and multicast, or running a persistent kernel with Cluster Launch Control work stealing.
---

# Async Kernel Coordination

Issuing a TMA copy or a Tensor-Core MMA only *starts* the operation; the
hardware runs it independently. Program order (you issued it before the read)
proves the transfer *started*, not that it *finished* — the consumer may read
incomplete data. The cure is an explicit completion signal: the producer
**arrives**, the consumer **waits**. This skill is the machinery for that, and
for using it to keep load, compute, and store busy at the same time.

## mbarrier

An `mbarrier` (memory barrier) is a hardware synchronization object in shared
memory. Track three fields: an **arrival count** (arrivals still missing this
round), a **phase** (the current round; you track only **phase parity**, 0/1),
and — for TMA — a **tx-count** (bytes still in flight).

- **init** sets the expected arrival count; the barrier starts phase 0.
- **arrive** reduces the count. Three arrival paths:
  - `mbarrier.arrive.expect_tx(bytes)` — one thread arrival **plus** it
    registers the bytes a TMA engine must still transfer. The phase completes
    only when arrival count == 0 **and** tx-count == 0.
  - `tcgen05.commit...mbarrier::arrive` — associates one arrival with the
    previously issued async `tcgen05` ops; hardware reports it only after they
    complete.
  - `mbarrier.arrive` — a plain thread arrival (e.g. a consumer signalling a
    buffer is now reusable).
- **wait** is the consumer side. Wait the phase for the current iteration and
  read/reuse only after it completes. Raw `mbarrier.try_wait.parity` may
  return `false` early and must be retried; wrap it in a blocking wait.

Because `arrive` and `wait` are separate, a producer can report and continue
with other work; the consumer waits only when it actually needs the result.
The same barrier is reused across rounds — **phase parity** alternates so a
consumer cannot mistake the previous round's completion for the current data
being ready. Full detail is in
[references/mbarrier.md](references/mbarrier.md).

## TMA

TMA (Tensor Memory Accelerator) asynchronously moves a tile between global and
shared memory. **One thread** of a warp issues the copy and supplies two
things (the rest of the warp is masked off until the request is submitted):

1. A **tensor-map descriptor** — the global tensor's element type, per-dimension
   shape and strides, the tile shape for one copy, and the swizzle mode to
   apply when writing SMEM. Reusable across many copies. ("How is this tensor
   organized?")
2. **Per-copy arguments** — the tile's starting coordinates in the global tensor
   and its destination SMEM address. ("Where does this copy begin, and where
   should it land?")

The engine performs address calculation and transfer asynchronously and can
apply the swizzle on write, so the tile arrives in the layout the later MMA
expects. The descriptor, the SMEM tile layout, and the consumer MMA must all
describe the **same** physical arrangement — a mismatch means the bytes are
present but read as the wrong elements.

Completion differs by direction:
- **Load** — consumer waits on the mbarrier that tracked `expect_tx` bytes.
- **Store** — producer waits for source-buffer reuse via `cp.async.bulk.commit_group`
  (bundle the uncommitted stores) then `cp.async.bulk.wait_group 0` (until all
  committed groups are done) before overwriting the source SMEM.

Detail (3-D TMA for multiple swizzle atoms, the 128-byte row-stride trap, the
byte-tracking example) is in [references/tma.md](references/tma.md).

## Software pipeline

Overlap is the payoff once a kernel is compute-bound. With ≥2 SMEM stages:

```
time t:   MMA reads stage 0   |  TMA fills stage 1
time t+1: MMA reads stage 1   |  TMA fills stage 0
```

While the Tensor Core reads stage k, TMA writes stage k+1 and the epilogue
processes k−1. Each stage gets a **pair** of barriers:

- `full[stage]` — TMA filled it; producer → consumer ("data ready").
- `empty[stage]` — consumer finished; consumer → producer ("buffer reusable").

Track `full` and `empty` phase parities **separately**. Before MMA reads a
stage it waits its `full` barrier; before TMA overwrites a stage it waits the
matching `empty` barrier. This hides load latency behind compute.

## Three common handoffs

1. **Threads → async hardware.** If threads write SMEM and a later TMA store
   or MMA reads it, establish the required synchronization and ordering first;
   otherwise the async op may begin reading before the threads finish writing.
2. **TMA → MMA.** The producer registers bytes via `expect_tx`; the MMA
   consumer waits the current barrier phase (plus any instruction ordering)
   before reading the tile.
3. **MMA → epilogue.** `tcgen05.commit` on an mbarrier signals the accumulator
   is complete; the epilogue waits that barrier and applies the required
   tcgen05 fence before reading TMEM.

## Warp specialisation

Instead of one warp doing load *and* compute, dedicate whole warps to roles —
**producer** warps issue TMA loads, **consumer** warps run the MMA. This may
temporarily increase resource use and lower occupancy, but it is the structure
that enables the deep multi-stage overlap that pays off later. mbarriers
coordinate the cross-warp handoffs.

## Clusters

A **cluster** (Hopper+) is a group of co-scheduled CTAs, possibly on different
SMs, that synchronize at cluster scope and access one another's SMEM through
**DSMEM** — avoiding GMEM round trips. Clusters enable two GEMM-critical moves
(full detail in [references/clusters-and-clc.md](references/clusters-and-clc.md)):

- **2-CTA cooperative MMA** (`cta_group::2`) — two CTAs each hold part of the
  SMEM operands and read the peer's slice through DSMEM, producing a larger
  output tile.
- **TMA multicast** — one GMEM load delivers the same tile to several CTAs,
  avoiding redundant traffic when CTAs share data.

## Cluster Launch Control

A **persistent kernel** launches a fixed pool of long-lived CTAs (or clusters)
that each compute several output tiles in a loop, cutting launch and setup
overhead. The scheduling question is: where does the next tile come from?

- **Static scheduling** derives the next tile from the worker id and iteration
  (grid stride). Near-zero overhead; fine when SM availability is stable and
  tile costs are similar. It fails on launch tails — some workers idle while
  one finishes the remaining tiles — and on unequal tile costs.
- **Cluster Launch Control (CLC)** (Blackwell) launches a grid that still
  covers every output tile, but a running worker can **cancel a pending launch
  and inherit its coordinate** (dynamic work stealing). One thread issues
  `clusterlaunchcontrol.try_cancel.async` (16-byte response via the async
  proxy, tracked by an mbarrier with `expect_tx(16)`), waits the barrier after
  the current tile, then `query_cancel.is_canceled` → on true
  `get_first_ctaid` → decode that coordinate → compute it; on false, leave the
  loop. **Issue `try_cancel` before computing the current tile** to overlap
  scheduling latency behind compute, just as TMA hides transfer latency. Use
  CLC when SM availability or tile costs are hard to predict.

## Completion criterion

Every asynchronous read is preceded by the matching barrier **wait** at the
correct phase parity; every reused buffer waits on its `empty` barrier (or
`commit_group`/`wait_group` for stores); the TMA descriptor, the SMEM layout,
and the consumer agree on swizzle; and the pipeline keeps TMA, the Tensor
Cores, and the store path active rather than serialising
`load → wait → compute → wait → store`.
