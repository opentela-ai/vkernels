# Cross-node KV all-gather (draft)

> **Status: equal-shard implementation available; multi-node performance not
> yet measured.** The communicator, reusable plan, C ABI, KVAAS Rust bindings,
> and an `nccl-allgthr` benchmark row now exist. Ragged shards remain rejected.

## 1. The problem the 2-node result exposed

PR #50 measured the real per-hop number acceptance #2 names, on JSC over
InfiniBand HDR with 2× GH200 (see `comm-cross-node-kv.md` §8). The
finding was blunt:

- **All three cross-node rows — `nccl-xfer` (send‖recv), `nccl-pipe`
  (N back-to-back), `nccl-allred` (a collective) — cap at ~24 GB/s =
  ~96% of one HDR-200 port**, regardless of operation type.
- A GH200 node has **4 mlx5 HCAs**; NCCL *can* stripe across all four
  (the aggregate is ~100 GB/s). It did not, with 2 ranks.

The reason is topological, not algorithmic: **2 ranks have one logical
edge.** NCCL assigns channels (→ HCAs) to edges; with one edge there is
nothing to stripe across, so the relationship saturates a single port
no matter how the work is shaped on it. Adding nodes C, D, E does not
make the A→B edge faster — but it *adds edges*, and an operation that
uses all of them can climb toward the 4-port aggregate.

The PR's `CrossNodeKvRestorePlan` / `CrossNodeKvDonatePlan` are
**point-to-point by construction**: each carries a single peer base
(`peer_src_ptrs` / `peer_dst_ptrs` for one remote node) and ships one
layer down one edge. That is correct for the kvaas/NIXL offload pattern
(donate a specific layer to a specific node), and a single donation
stays one-port no matter how many nodes exist. It is the wrong primitive
when **every node needs the full KV**.

## 2. When all-gather is the right primitive

The cache is **sharded** across N nodes (each holds 1/N), and **every
node needs the reassembled whole**. Two concrete serving patterns hit
this:

| pattern | why every node needs the full KV |
|---|---|
| **Sequence / context parallel prefill** | a long prompt is split across N nodes; each computes its KV shard, then every node needs the full sequence's KV for the next attention layer (or for the decode handoff). |
| **Broadcast the cache to all decode replicas** | every replica runs attention over the same KV; one canonical copy is sharded across the replica group and must be reassembled on each. |

It is **wasteful** when only one node needs a specific shard: the
point-to-point donate ships `1×` a layer, all-gather ships `(N−1)×`
everything. The primitive choice follows the access pattern — all-gather
is not "all KV transfer, but better," it is the correct and efficient
choice for the all-to-all-receive case only.

## 3. Why a ring all-gather unlocks the multi-port bandwidth

A ring all-gather of the full KV across N nodes, each holding 1/N,
proceeds in **N−1 steps**. At every step each node sends its current
shard to rank+1 and receives a new shard from rank−1, so **N edges are
active simultaneously**:

```
total bytes moved   = (N − 1) × total           (same as N×(N-1) donates)
active edges/step   = N                          (vs 1 for a single donate)
per-step aggregate  → 4-port  (~100 GB/s)        as channels→HCAs fill
total time          = (N − 1)/N × total / agg    → total/agg as N grows
```

The decisive comparison is not "all-gather moves less" — it doesn't;
the total fabric bytes are **identical** to N×(N−1) point-to-point
donates. All-gather wins by **overlapping the same bytes across N edges
in parallel** instead of serializing each donate on a single port. As
`N` grows, `(N−1)/N → 1` and the per-step aggregate → the 4-port
ceiling, so the all-gather approaches `total / 100 GB/s` — roughly **4×
the single-port ceiling (~24 GB/s)** the 2-node donate hit. That is the
number to go measure; the 2-node run *suggests* it (the `nccl-allred`
capped at one port with 2 ranks) but cannot prove it.

| N | edges | total bytes | if agg = 100 GB/s | vs 2-node donate (~24 GB/s) |
|---|---|---|---|---|
| 2 | 1 | 1× total | capped at one port (measured) | 1.0× |
| 3 | 3 | 2× total | 2/3 × total / 100 ≈ 0.67× total/100 | ~2.8× |
| 4 | 6 (ring) | 3× total | 3/4 × total / 100 ≈ 0.75× total/100 | ~3.1× |

(The "4×" is the asymptote as `N → ∞`; the first few nodes capture most
of it.)

## 4. What `CrossNodeKvAllGatherPlan` would look like

A third cross-node plan, built once over (geometry, **global** slot map,
per-node shard, transport) and reused per layer — mirroring the two that
shipped:

```
kv_gather_layer(local_slots -> sendbuf)      # pack THIS node's 1/N shard
ncclAllGather(sendbuf -> recvbuf, comm, ...) # N-rank ring, CAPTURABLE
kv_scatter_layer(recvbuf -> local_slots)     # unpack the reassembled full KV
```

- **Graph-capturable** for `kFabricMapped` / `kSameNodePeer`:
  `ncclAllGather` is a capturable collective (same as the `ncclAllReduce`
  the bench already uses), so it records **one** device op and replays
  with no host progress — the existing `PipelineBoundaryPlan` contract
  (#10).
- **Eager-break** for `kHostBounce`: `graph->end()`, all-gather over a
  byte channel, `graph->begin()` — exactly the two existing plans'
  bounce contract.
- **Reuses `kv_gather_layer` / `kv_scatter_layer` unchanged**: they
  already take a slot map + page geometry. The only new state is the
  per-node shard descriptor and the NCCL communicator.

### Cost-model entry

`cross_node_kv_throughput` would gain an all-gather transport whose
`per_hop_gbps ≈ (N−1)/N × min(fabric_agg, kernel)`, with the same GH200
DRAM-only degradation the existing entries carry. This is where the
one-port-vs-aggregate distinction becomes explicit in the model instead
of only in the prose.

## 5. Open questions (why this is a draft, not a plan)

1. **Even vs ragged shards.** Plain `ncclAllGather` assumes each rank's
   sendbuf is the same size. Paged KV isn't necessarily evenly
   distributed across nodes. Options: pad to even (waste), or use
   NCCL's variable-sendcount all-to-all for the ragged case. Which
   depends on how the cache is actually sharded — needs a concrete
   serving layout before designing the plan's input contract.
2. **Is the access pattern actually all-to-all-receive?** The math in §3
   only pays off when every node needs the full KV. If the serving
   workload is one-to-one offload (the pattern PR #50 targets),
   all-gather is overhead and the point-to-point plan is already
   correct. The decision forks on the workload, not on "collectives are
   generally good."
3. **The multi-port result is a hypothesis, not measured.** The 2-node
   run shows one port caps; it does not show ≥3 reaches the aggregate.
   The honest next step is empirical: extend `bench_cross_node_nccl.cu`
   with an `nccl-allgather` row and submit `-N3 -n3` then `-N4 -n4` jobs
   on JSC. If the all-gather climbs past ~24 GB/s toward 100 GB/s as N
   grows, that both justifies the primitive and calibrates the
   cost-model entry. If it doesn't, the section above is wrong and the
   doc should say so.
4. **Placement.** A third `CrossNodeKv*Plan` fits the existing module
   (`cross_node_kv.{hpp,cpp,cu}`), but a separate `cross_node_kv_allgather.*`
   may be cleaner given the N-rank communicator lifecycle is different
   from a single borrowed `FabricImport`. Defer until the measurement
   says go.

## 6. Implemented interface

The implementation is deliberately separate from the single-peer plans:

- `cuda::NcclCommunicator` owns one rank-local `ncclComm_t`. Rank 0 creates a
  unique id and distributes it out of band; every rank creates its communicator
  on the intended current CUDA device.
- `cuda::CrossNodeKvAllGatherPlan` borrows that communicator and owns device
  copies of the local/global slot maps plus reusable send and receive buffers.
- `execute(k_src, v_src, k_dst, v_dst, stream)` enqueues
  `kv_gather_layer_device_slots -> ncclAllGather ->
  kv_scatter_layer_device_slots` on one stream. Source and destination K/V may
  alias because stream order protects the packed input before scatter begins.
- The stable C ABI is `cross_node_kv_allgather_c.h`. It exposes NCCL capability
  and graph-capture probes, unique-id bootstrap, communicator create/poll/
  finalize/destroy/abort, and prepared-plan create/execute/query/destroy.
- KVAAS binds the same surface as `NcclCommunicator` and
  `PreparedCrossNodeKvAllGatherPlan<'comm>`; the Rust lifetime prevents a
  communicator from being destroyed while a plan still borrows it.

The first supported layout is page-aligned and equal: `num_pages % world == 0`
and `global_slot_ids` is rank-major. The slot map must be unique and in range.
This rejects ragged input rather than silently padding cache pages. The raw
benchmark pads byte shards explicitly because it measures fabric behavior, not
the prepared plan's page contract.

Graph capture is reported separately from NCCL availability. NCCL collectives
are capturable only with NCCL >= 2.9 and CUDA >= 11.3, and capture/launch must
be uniform across every participating rank. While waiting for completion,
callers can poll asynchronous communicator errors and must abort/recreate the
communicator after a fatal result.

## 7. Concrete next step

Submit the implemented `nccl-allgthr` row in
`meta/benchmarks/bench_cross_node_nccl.cu` as a 3-node then 4-node job on JSC.
The row's `%p100` (wire bytes / 4-port
aggregate) is the single number that decides whether the plan is
performance-justified: if it climbs toward 100% as N grows, all-gather
delivers the multi-port bandwidth the 2-node run could not reach. If it stays
pinned at one port, keep the P2P path as the serving default and treat this
collective as a correctness-capable but non-preferred option.
