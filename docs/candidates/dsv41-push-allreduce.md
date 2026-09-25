# Porting candidate: push-style one-shot all-reduce for small decode vectors

> Provenance: reference repo (shi3z/deepseekv4.1-A100-custom) carries **no license file** as of
> this writing — analysis only; don't copy code verbatim without resolving licensing.


Source: `dsv41/cuda/allreduce.cu` (+ its P2P substrate `dsv41/cuda/p2p.cu`),
a single-process, no-NCCL all-reduce over P2P-connected GPUs.

**Question this doc answers:** our existing all-reduce paths are
bandwidth-optimal but latency-poor for decode-sized messages — the ring
(`ring_allreduce_rank`) pays `2*(world-1)` blocking `Channel` send/recv
round-trips and the RCCL plan pays one `rcclAllReduce` whose algorithm
degrades at small sizes. The push-style candidate complements both with a
**latency-optimal** same-node path: three kernel launches per round
(push, signal, reduce), one shot of data movement, fully CUDA-graph
capturable — the shape a TP=… decode step wants for a `hidden=7168` bf16
vector (14 KB, i.e. ~896 `uint4`).

---

## 1. The reference algorithm

Every GPU `d` owns three device allocations, prepared once:

* `buf[W][n]` — row `p` of GPU `d`'s buffer is where GPU `p` pushes its
  vector (`peer_rows[p]` on GPU `d` = `&buf_d[me*n]`, i.e. GPU `d`'s row
  inside GPU `p`'s buffer, pre-offset, self included);
* `flags[W]` — `flags_d[p]` is set to `seq` by GPU `p` once its row landed;
* a device `seq` counter, bumped once per round.

Three kernels per round per GPU, all pure device work:

| Kernel | What it does | Notes |
|---|---|---|
| `ar_push` | grid-stride copy of `x` (`uint4`, vectorized) into every peer's `buf[me]` row **and** own `buf[me]` | direct kernel stores into peer UVA over NVLink — same mechanism `pipeline_boundary` documents as the *only* stream-capturable cross-GPU copy (`cudaMemcpyPeerAsync` is not) |
| `ar_signal` | single thread: `__threadfence_system()`, store `seq` to every peer's `flags[me]`, fence, spin until own `flags[p] >= seq` for all `p`, bump `seq` | the ordering point; `ar_signal2` generalizes set/wait counts and the bump so a two-phase round can share one counter |
| `ar_reduce` | `out = Σ_p buf[p]`, bf16x2 math via `__bfloat1622float2` | reads only local memory; trivially parallel |

Why it is fast for small messages: the ring moves each chunk `2*(W-1)`
times through `W-1` latency-bearing hops (host round-trips in our mock,
NCCL channel latency on device). The push variant moves the whole vector
**once** (W parallel pushes) and synchronizes with one flag round — cost is
~3 launch floors + one NVLink traversal, independent of a ring's hop count.

---

## 2. Where it slots into the existing abstractions

New module `src/c/vkernels/comm/push_allreduce.{hpp,cpp,cu}`
(`push_allreduce_cuda.hpp` for the device decls), sibling to
`rccl` / `pipeline_boundary`:

```
classify_boundary(cfg) ──┬─ kSameNodePeer  ─▶ PushAllreducePlan (this candidate)
                         ├─ kCrossNodeNccl ─▶ RcclAllreducePlan (existing)
                         └─ kHostStaged    ─▶ eager-break / ring (existing)
```

* **Topology.** Push all-reduce is one-shot over a *fully-connected
  same-node peer set*, not a ring: the plan takes the peer list
  (`world`, `rank`, and per-peer pre-offset row/flag pointers) rather than
  `next`/`prev`. The host reference derives the peer set from
  `build_ring_topology`'s sibling — a `Peers` helper over `Topology`
  (`all ranks except me on the same node`) — so the same
  `ring_rank`/`build_ring_topology` facts feed both paths.
* **Channel.** The device path bypasses `Channel` entirely (pure device
  ops, exactly like `PipelineBoundaryPlan`'s device path: "NO channel is
  touched"). The host reference implements the *same semantics* over
  `MockChannel`: rank `p`'s push is a `send` of its vector into every
  peer's inbox and the signal/wait is a rendezvous on shared atomic
  flags — so the existing `make_ring_channels`-style in-process harness
  validates the contract without a GPU.
* **Selection.** A small pure dispatch —
  `prefer_push_allreduce(total_bytes, world, cfg)` — sits next to
  `prefer_slingshot_rccl`: push for small same-node payloads, RCCL plan for
  large or cross-node, host ring/mock otherwise. Force modes
  (`VK_PUSH_ALLREDUCE=0/1` style env, mirroring `VK_MOE_AUX_FUSED_QUANT`)
  keep an A/B escape hatch and a bit-identical fallback to
  `RcclAllreducePlan`.

API sketch (mirrors `RcclAllreducePlan`: validate once at prepare, execute
enqueues a fixed task count, read-only thereafter):

```cpp
class PushAllreducePlan {          // host reference; cuda::PushAllreducePlan mirrors it
 public:
  PushAllreducePlan(int world, int rank, std::size_t capacity_elems, DType dt);
  // Device path: peer_rows[p] / peer_flags[p] pre-offset pointers into every
  // peer's (and own) buf / flags allocations, peer access enabled by caller.
  void execute(float* x /*bf16 on device path*/, GraphCapture* graph = nullptr);
  // Host mock: execute(x, next /* fan-out */, prev /* fan-in */) over Channels.
};
```

One plan serves every decode iteration (prepare-once, K3 microbatch
pattern); `my_buf` is supplied per `execute()`.

---

## 3. Two-implementation model

| Layer | File | Compiled when | Coverage |
|---|---|---|---|
| Host reference (peer-set derivation, dispatch decision, channel-mock of push/signal/reduce semantics, `GraphCapture` integration) | `push_allreduce.{hpp,cpp}` | **always** | 100% line (CI gate), no GPU |
| CUDA device plan (`ar_push` / `ar_signal` / `ar_reduce` kernels over `cudaStream_t`, bf16 + `uint4` vectorization) | `push_allreduce.cu`, `push_allreduce_cuda.hpp` | `VKERNELS_HAS_CUDA` | on-device |
| C ABI (create / execute / destroy) | `push_allreduce_c.{h,cpp,cu}` | mirrors `pipeline_boundary_c` | host / on-device |

The host reference is the correctness oracle: it simulates `buf`, `flags`
(`std::atomic<int>`), and `seq` in-process, runs all ranks on threads (the
`ring_allreduce` harness pattern), and asserts the element-wise sum matches
a sequential reduction — so the CUDA path's *contract* is testable on a
GPU-less CI box, and the device translation unit stays well-formed and
compiled only where a toolkit exists (the `allreduce.cu` stub slot this
candidate fills out).

---

## 4. CUDA-graph capture rules

Same rules the `pipeline_boundary` doc establishes, applied to three nodes
instead of one:

1. **Capture verbatim.** `execute()` issues `ar_push → ar_signal →
   ar_reduce` on the caller's stream between the caller's
   `cudaStreamBeginCapture` / `cudaStreamEndCapture`. All three are pure
   device kernels — no host send/recv, no allocation, no validation after
   construction — so the captured segment replays with **no host
   progress**. Contract test (host model): submit the three ops into a
   `GraphCapture`, `replay()` N times, assert no `Channel` touched and the
   result matches after every replay.
2. **Peer copies are kernel stores.** Cross-GPU movement stays inside
   `ar_push` as UVA kernel stores; `cudaMemcpyAsync` /
   `cudaMemcpyPeerAsync` are **not** used (not stream-capturable across
   devices on Hopper/CUDA 13 — "operation not permitted while stream is
   capturing"), per the documented `pipeline_boundary` finding.
3. **Launch-error policy.** `cudaGetLastError()` is *not* checked after
   each launch (stale status can abort an otherwise-valid capture);
   errors surface at `cudaStreamEndCapture` (graph path) or next sync.
4. **Spin-wait safety = lockstep replay.** `ar_signal` spins until every
   peer's flag reaches `seq`; this is deadlock-free only if every rank's
   graph replays the same round cadence (push-then-signal ordering inside
   each stream guarantees a rank never waits on a peer's *future* round).
   This is the same lockstep assumption the PP-boundary graph replay
   makes, and the host reference's threaded oracle reproduces the ordering
   with atomics so a broken schedule fails on CI, not on a node.
5. **Node count.** Unlike `RcclAllreducePlanHip` (one graph node), one
   round records **three** nodes; `GraphCapture::num_nodes()` observable
   asserts exactly 3 per segment in the host tests.
6. **Vectorization contract.** `ar_push`'s `uint4` path requires the
   payload 16-byte aligned and `n*sizeof(T) % 16 == 0` (7168 bf16 = 14 KB
   ✓); the host reference validates this once at prepare and falls back to
   an element path otherwise — checked by the oracle, trusted by the
   kernel.

---

## 5. HIP / RCCL extension (MI300A)

Two same-node transports exist on gfx942, and the module keeps both behind
the dispatch:

* **Hand-rolled xGMI push** (`push_allreduce.hip`, `VKERNELS_HAS_HIP`):
  the kernels port near-verbatim — `hipEnablePeerAccess` (or ROCm IPC
  handles) up front, kernel stores traverse xGMI, `__threadfence_system()`
  maps to the same acquire/release point. This is the latency-optimal
  decode path and, on MI300A (APU, CPU/GPU share HBM), peer mappings are
  cheap to establish once per process.
* **RCCL plan** (`RcclAllreducePlanHip`, existing): one `rcclAllReduce`,
  the bandwidth-optimal large-message / cross-node path; cross-node traffic
  still goes through the Slingshot OFI plugin selection
  (`resolve_transport`, `plugins/rccl-net-ofi`).

**Cost model** (host-tested, extends the `rccl` model):

```
est_push_us    ≈ 3 * launch_floor + bytes / xgmi_bw + flag_roundtrip
est_rccl_us    ≈ ring latency model (existing est_rccl_socket_us / est_rccl_ofi_us
                 with inter_node_edges == 0 for same-node)
prefer_push    = same_node && est_push_us < est_rccl_us   (force modes bypass)
```

At 14 KB the push model wins by construction (RCCL's ring pays `2*(W-1)`
channel latencies before any byte moves); the crossover with RCCL's
bandwidth-optimal large-message algorithm is measured by the bench and
recorded in `docs/comm-rccl.md`'s table, tuning `prefer_push`'s threshold.

---

## 6. Tests and bench

* `tests/comm/test_push_allreduce.cpp` — host oracle: correctness across
  `world ∈ {1..8}`, `n` ∈ {16-B-misaligned, 7168, large}; dispatch decision
  matrix; `GraphCapture` capture → `replay()` × N with **zero host
  progress** (no `Channel` touched, 3 nodes/segment); eager fallback when
  capture is inactive; force-env escape hatch.
* `tests/comm/test_push_allreduce_cuda_c.cu` — on-device: two-GPU peer
  capture + replay, result vs `RcclAllreducePlan` / host sum.
* `bench/` — push vs RCCL-plan vs ring sweep over message size
  (1 KB … 64 MB) and `world`, per the `p2p_gather` bench pattern; outputs
  the crossover point the dispatch threshold is tuned from.

## 7. Risks

* **Deadlock on non-lockstep replay** — mitigated by the lockstep contract
  (§4.4) and the threaded oracle.
* **Flag/seq reuse across segments** — `seq` is device state; a re-capture
  must not reset it mid-round (the `ar_signal2` bump convention covers
  two-phase rounds).
* **bf16 vs host `float` channels** — the host oracle runs in `float`; the
  device path is bf16, so parity tests compare against a bf16 reference
  sum, not the float oracle (same discipline as the quant fusion oracles).

## 8. Acceptance

1. Host oracle (GPU-less CI, 100% line coverage) reproduces the
   element-wise sum for all ranks via the channel mock.
2. Captured round replays N times with no host progress and stays correct
   (host model, then two-GPU device test).
3. Dispatch prefers push at 14 KB same-node and RCCL at large/cross-node;
   force-env overrides both directions.
4. Bench records the push/RCCL crossover on the target part
   (GB10 / MI300A) and the doc table is updated.
