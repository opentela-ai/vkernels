# Clusters, Warp Specialisation, and Cluster Launch Control

## Clusters (Hopper+)

A **cluster** is a group of co-scheduled CTAs, possibly on different SMs, that
synchronize at cluster scope and access one another's SMEM through
**distributed shared memory (DSMEM)** — avoiding GMEM round trips when one CTA
needs another's data. Each CTA still owns its own SMEM; DSMEM means peers can
read it directly rather than forcing the owner to write it back to GMEM first.

Two GEMM-critical capabilities rely on clusters:

- **2-CTA cooperative MMA (`cta_group::2`).** Two CTAs form a CTA pair and
  execute a cooperative MMA: each holds part of the SMEM operands (slices of
  A and B) and reads the peer's slice through DSMEM. The output tile is larger
  than either CTA could produce alone. The allocations are **not merged** —
  each CTA keeps its own SMEM; only the cross-SM read is enabled.
- **TMA multicast.** One GMEM load delivers the same tile to several CTAs,
  avoiding redundant traffic when multiple CTAs need the same data (e.g. a
  shared B tile across a column of output tiles).

Cluster CTAs share a cluster id; a CTA's local index within the cluster is
`%cluster_ctaid` and its rank `%cluster_ctarank`. Declare the cluster with
`.reqnctapercluster` (and `.maxclusterrank`). Cluster-wide synchronization uses
`barrier.cluster.arrive`/`barrier.cluster.wait`.

## Warp specialisation

Instead of one warp doing load **and** compute, dedicate whole warps to roles:

- **Producer warps** issue TMA loads and fill SMEM stages.
- **Consumer warps** run the Tensor-Core MMA (and, later, the epilogue).

mbarriers coordinate the cross-warp handoffs (producer arrives "full",
consumer arrives "empty"). This may **temporarily increase resource use and
lower occupancy** — whole warps are reserved for one role — but it is the
structure that enables the deep multi-stage overlap the later optimisation
steps exploit. A step that does not improve performance immediately can still
be worth taking because of the structure it provides.

## Persistent kernels

A persistent kernel launches a **fixed pool of long-lived CTAs (or clusters)**
that each compute several output tiles in a loop, rather than one CTA per tile
that exits when done. This cuts launch overhead and repeated setup. The
scheduling question it introduces is: after a worker finishes its current
tile, where does the next one come from?

### Static scheduling

The next tile is derived from the worker id and the iteration count (grid
stride):

```
worker 0: tile 0, 4, 8
worker 1: tile 1, 5, 9
worker 2: tile 2, 6, 10
worker 3: tile 3, 7, 11
```

Almost no task-acquisition overhead. It works when SM availability is stable
and tile costs are similar. It fails on **launch tails** — if worker 3 is
delayed, workers 0–2 finish and exit, leaving one worker to run the remaining
tiles — and on unequal tile costs (boundary handling, masks, sparse compute),
which a static formula cannot redistribute.

### Cluster Launch Control (Blackwell)

CLC launches a grid that **still covers every output tile**, but a running
worker can **cancel a pending CTA or cluster launch and inherit its
coordinate** — dynamic work stealing. No register or execution state moves;
the canceled CTA never started, so hardware hands the worker only the
`blockIdx` it would have used.

One thread submits the request:

```
clusterlaunchcontrol.try_cancel.async [resp], [bar]
```

The response is a 16-byte record written to SMEM through the **async proxy**;
track it with an mbarrier and `expect_tx(16)`. The worker then:

1. Submit `try_cancel` **before** computing the current tile (overlap
   scheduling latency behind compute, the same idea as TMA).
2. Compute the current tile while the request is in flight.
3. Wait the CLC request's mbarrier.
4. `clusterlaunchcontrol.query_cancel.is_canceled` → predicate.
5. On **true**: `clusterlaunchcontrol.query_cancel.get_first_ctaid` →
   `(x,y,z)` of the canceled CTA (or the first CTA of the canceled cluster),
   decode it to an output tile, continue. On **false**: leave the request loop
   — the queue is empty or a higher-priority kernel is coming.

### CLC rules that catch people out

- **No sentinel.** The coordinate from `get_first_ctaid` is valid **only**
  when `is_canceled` returns true; querying it after a failed request is
  undefined behavior.
- After a failed request the worker **must** leave the loop; a subsequent
  cancellation request is undefined.
- If several threads issue `try_cancel`, each gets a separate response
  location and all must be accounted for in the barrier's arrival count and
  tx-count.
- A CTA pair (cluster) is taken over as a unit: `get_first_ctaid` returns the
  first CTA's coordinate; each CTA combines it with its local block index to
  recover its own grid coordinate.
- Issue the required **proxy fences** before submitting a new request and after
  reading the response, plus the CTA-/cluster-wide thread synchronization.

### When to use CLC

Static scheduling and CLC can share the *exact same* tile computation; they
differ only in how the next coordinate is obtained. CLC is worth it when
available resources or tile costs are hard to predict — workers that finish
early claim pending coordinates instead of idling, shortening the launch tail.
When SM availability is stable and tiles are uniform, a static formula is
simpler and nearly free.
