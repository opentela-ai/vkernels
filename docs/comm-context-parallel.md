# Context-parallel data path — assessment against existing vkernels comm primitives

> **Scope.** Sequence-striped context parallelism (CP): each rank owns a
> contiguous shard of the sequence; every rank needs (a) the KV of the whole
> sequence for its attention, and (b) the linear-attention recurrent state
> produced by the *previous* shard. This document assesses what the existing
> comm primitives already cover and what is missing. Kernel side, the KDA
> state-handoff contract is landed (see
> [kernels/kda.md](kernels/kda.md) § "Context-parallel state handoff").

## 1. The two data paths of sequence-striped CP

```
   rank r-1                      rank r                        rank r+1
   shard [t0,t1)                 shard [t1,t2)                 shard [t2,t3)
   ┌──────────────┐   state      ┌──────────────┐   state      ┌──────────────┐
   │ KDA layer L  │ ───────────► │ KDA layer L  │ ───────────► │ KDA layer L  │
   │  S_out[L] ───┘  B·H·D·D     │  S_in[L]     │ ───S_out[L]─►│  S_in[L]     │
   │              │              │              │              │              │
   │ attention(Q_r│ ◄─────────── │              │              │              │
   │  full KV)    │  per-layer   │              │              │              │
   └──────────────┘  KV all-     └──────────────┘              └──────────────┘
                     gather / peer gather        (DSA prefill: selected-block
                                                  gather instead of all-gather)
```

Two distinct wires per layer per step:

1. **KV path (attention)** — rank r's Q shard must attend over the full
   sequence's K/V. Either all-gather (dense MLA) or selected-block gather
   (DSA sparse prefill).
2. **State path (KDA layers)** — the `B·H·D·D` recurrent state rotates
   around the ring, one hop per layer, opposite the shard order. At
   K3 shapes (H=16, D=128, fp32) this is **1 MB per layer per rank** —
   three orders of magnitude below the KV path, but on the critical path
   (shard r cannot start layer L until r−1 finished layer L there).

## 2. What the existing primitives already cover

| primitive | CP role | status |
|---|---|---|
| `cross_node_kv_allgather` (communicator + prepared plan + C ABI) | **KV path, dense prefill**: equal-shard per-layer KV all-gather across nodes is exactly pattern ("cache sharded across N nodes, every node needs the reassembled whole"). Caches the prepared plan; rejects ragged shards. | **Available** (multi-node performance unmeasured; see `comm-cross-node-kv-allgather-draft.md` §1 for the per-port ~24 GB/s finding that the all-gather's multi-edge design addresses) |
| `kv_gather` / `kv_scatter` (+ p2p_kv_restore/donate over `FabricImport`) | **KV path, DSA sparse prefill**: instead of all-gathering everything, gather only the selected top-k blocks from peer pool pages — `kv_gather` already gathers arbitrary (repeating) source slots into a contiguous per-layer page buffer, and the cross-node restore/donate plans make peer pages device-addressable (mapped) or bounce through pinned scratch. | **Mostly available**: local gather is done; the missing piece is a *remote* gather plan — slot ids resolved against peer pools and executed through a `FabricImport`/host-bounce transport (compose `kv_gather` with `cross_node_kv`'s transport classification; no new kernel needed) |
| `pipeline_boundary` (graph-capturable transfer + eager-break API) | **State path transport candidate**: the KDA state handoff is a point-to-point transfer inside a captured decode graph — precisely the problem pipeline_boundary solves for PP (frozen host `recv_object` ⇒ replay deadlock). The mapped-fabric path records one device op; else explicit eager break. | **Reusable in principle**: semantics are PP stage→stage (one producer, one consumer, one direction). A CP ring needs the same transfer executed *per layer with rotating peer ranks*; the plan would need per-layer peer rebinding, or nccl-style send/recv capture (below) |
| `rccl` (`RcclChannel` send/recv, `RcclAllreducePlan` graph capture, OFI/CXI plugin) | **State path transport, simplest correct form**: a ring rotation of a 1 MB buffer per layer is `rcclSend`/`rcclRecv` to rank+1 — the channel abstraction (`make_ring_channels`) already models exactly this topology, and the graph-capture plan shows how to record collectives. | **Available as building block**: no ready-made "ring rotate a buffer" plan with capture support for send/recv (the captured plan today is all-reduce only) |
| `allreduce` (ring, host + CUDA) | Not the CP primitive (nothing to sum across ranks in the state path), but its in-process multi-rank harness pattern is the template for testing a state-rotation collective without a cluster. | n/a (test pattern only) |
| `OverlapExecutor` (compute stream ‖ comm stream, per-iteration future) | **Overlap reuse**: hide the state hop behind the previous layer's compute — while rank r computes layer L−1 on shard r, the state for layer L from r−1 is in flight. The executor's compute/comm split maps 1:1. | **Available at host level**: value/future based, host-synchronising — not directly usable inside a captured graph; pairs with pipeline_boundary's eager-break for the capture story |
| KDA state API (`kda_delta_rule_fwd_state_cpu` / HIP `*_with_scratch`) | Kernel-side contract: seed `S_in`, export `S_out`, aliasing ring buffer allowed. | **Landed** (this change; host oracle + composition tests) |

## 3. What is missing

1. **A state-rotation collective (the "cp_ring_pass").** The one genuinely
   new primitive: per KDA layer, rotate the `B·H·D·D` state buffer one hop
   around the ring — send-recv to/from adjacent ranks with the in/out
   aliasing contract the KDA API expects. Host reference over
   `make_ring_channels` (testable in-process, allreduce-style), HIP path as
   either captured `rcclSend`/`rcclRecv` pairs or a
   `pipeline_boundary`-style device op where the fabric allows. Metadata is
   one buffer descriptor (dtype, B·H·D·D, layer id) — no slot maps needed.

2. **Shard/chunk metadata.** A small `CpPlan`: world size, per-rank token
   range, and the constraint set that makes the ring well-defined — shard
   length a multiple of the KDA chunk (`cs=64`, the HIP chunked kernel's
   `S % 64 == 0` contract), identical per-rank shard lengths (the
   all-gather already rejects ragged shards; keep CP equal-shard too), and
   per-layer state buffer geometry (B, H, D per KDA layer). Nothing here
   exists; it is host-only bookkeeping.

3. **Remote selected-block gather (DSA prefill CP).** Compose, don't build:
   indexer top-k runs on the local shard's pool, but the selected blocks may
   live on peer ranks — a plan that runs `kv_gather`'s contract against
   `FabricImport`-mapped peer pages (mapped path) or staged through pinned
   bounce (host-bounce path), reusing `cross_node_kv`'s transport
   classification unchanged. The fused gather kernel itself needs no change.

4. **Capture-safe overlap wiring.** `OverlapExecutor` is host-future based;
   under CUDA graphs the state hop must be either captured (device op /
   captured RCCL send-recv) or explicitly eager-broken
   (`pipeline_boundary`'s API). The missing piece is the *decision rule*
   (mapped fabric ⇒ capture; else eager break) applied per layer to the
   state rotation — a thin composition of existing pieces.

5. **(Kernel-side, for completeness) position-offset attention.** With
   sequence-striped CP, each rank's Q covers positions `[t0,t1)` while K/V
   span the whole sequence — the MLA/DSA forwards need a position-offset
   (and for DSA, block-index) convention for cross-shard K/V. That is a
   kernel contract item, not a comm primitive; flagging it because it gates
   end-to-end CP independent of the transport.

## 4. Suggested order of work

1. `cp_ring_pass` host reference over `make_ring_channels` + in-process
   multi-rank test (allreduce test pattern) — no GPU needed.
2. `CpPlan` metadata + validation (shard % cs == 0, equal shards).
3. HIP path: captured send/recv rotation or pipeline_boundary reuse, behind
   the same mapped/host-bounce classification `cross_node_kv` already uses.
4. Remote block gather plan (kv_gather × FabricImport).
5. Multi-node measurement of the KV all-gather path (the draft's open item).
