# Candidate 5 — device-side EP token dispatch (indexed row scatter + flag wait)

> Provenance: reference repo (shi3z/deepseekv4.1-A100-custom) carries **no license file** as of
> this writing — analysis only; don't copy code verbatim without resolving licensing.


Porting plan for the expert-parallel decode scheme of the dsv41 reference
(`/tmp/dsv41/dsv41/ep.py` + `/tmp/dsv41/dsv41/cuda/p2p.cu`) onto the
`src/c/vkernels/comm/` abstractions. The reference runs **one CUDA graph per
GPU for the whole token with zero host round trips**: the layer owner pushes
the quantized activation + routing into every peer's inbox with P2P stores
and raises a device flag; each peer computes the experts it holds, sums them
into a partial, pushes it into the owner's inbox with another flag; the owner
waits device-side, adds the shared expert, and continues. All synchronization
is flag kernels spinning on a per-token sequence number, so the whole decode
step is a handful of graph launches and one host sync for the logits.

> **Question this plan answers:** we already cover *indexed scatter for the KV
> path* (`kv_scatter`, `p2p_kv_restore`). Do we have the EP equivalent —
> per-token routed row scatter into **peer-owned** receive buffers plus
> device-side flag wait — and a device-side sum-combine for the routed-expert
> partials? **Short answer: no.** Three of the five primitives the reference
> needs are missing; the classification / capture / DMA-dispatch scaffolding
> they hang on already exists. Details below.

---

## 1. Mapping the reference onto our abstractions

| dsv41 symbol | What it does | Our equivalent today | Status |
|---|---|---|---|
| `p2p_copy` | flat uint4 copy into a peer inbox (kernel stores over peer access) | `p2p_gather_runs` / `memcpy_peer_batch_async` (single-launch peer reads, adaptive DMA/kernel dispatch) | **exists** (generalized: many→one; the 1→N direction is a trivial sibling) |
| `p2p_copy_row` | per-row indexed scatter: `dst_base[seq[b]*bstride16 + row_idx[b]*row16 + j] = src[b]` into a **peer** buffer | `kv_scatter_layer` — indexed scatter, but destinations are **local** pool slots only | **new** (peer-destination indexed row scatter) |
| `p2p_sum_rows` | grouped row sum-combine: `dst[g*dst_stride+i] = Σ_r src[(g*rows+r)*n+i]` (the routed-expert partial combine, fp32 accumulate, optional one rounding to bf16) | `dist::moe_ep_*` combines on the **host** over `Channel`; no device-side combine in `comm/` | **new** |
| `p2p_signal` / `p2p_wait` / `p2p_seq_bump` | device flag publish / spin-wait on a per-token sequence number, `threadfence_system` ordering | nothing — every sync we have is stream-order or host-side (`Stream::wait`, `GraphCapture` segments) | **new** |
| `p2p_multicast` | one launch: copy `n16` uint4 into up to 8 peer inboxes; last block (atomicInc counter) signals their flags | `p2p_gather` has the *dispatch* machinery (copy-engine vs kernel crossover, `est_*_us`, `prefer_gather_kernel`); no fused copy+signal kernel | **new kernel**, reuses existing dispatch model |
| `p2p_stamp` | `%globaltimer` timeline stamps for per-layer tracing | no device clock primitive | optional, low priority |
| inbox/outbox message layout (`[xqp bf16 \| eid i32 \| wt f32]` in one 16B-padded buffer) | routing packet geometry | none (`dist_moe` plans expert sharding, not message bytes) | **new** (a plan struct, trivial) |
| `EP_DMA auto` mode switch (kernel stores < 32 KiB, copy engines above) | transport choice per message size | `prefer_gather_kernel(num_runs, bytes)` — same decision, fitted constants | **exists**, reuse the model |
| NVLink-pair relay (`EP_RELAY`, partner = `d^1`, relay-of-owner) | 4-GPU pair topology: route via NVLink inside a pair, one PCIe hop across pairs | `topology.hpp` only models a ring | **new** (pair/relay topology helper) |
| flag/seq buffers, capture discipline (`dry` pass, per-device non-blocking streams, `capture_error_mode="thread_local"`) | the CUDA-graph rules that make 40 layers one graph per GPU | `pipeline_boundary.hpp`: `is_graph_capturable`, `eager_break_during_capture`, `GraphCapture` host model | **exists** (extend with the EP-specific capture checklist) |
| `classify` of same-node peer vs cross-node | transport classification | `classify_boundary` / `classify_fabric_import` (`kSameNodePeer` is exactly the reference's "P2P access between all GPUs" assumption) | **exists** |

### What genuinely does not exist (the gaps)

1. **Peer-destination indexed row scatter** (`ep_scatter_rows`): our indexed
   scatters all write *local* memory. The EP dispatch writes rows into
   `nd-1` peer inboxes at data-dependent offsets — the owner only knows the
   destination *base* pointers, the routing metadata supplies the row indices.
2. **Device-side flag mailbox** (`ep_signal` / `ep_wait` / `ep_seq_bump`):
   the entire communication protocol in `comm/` is stream-ordered or
   host-driven. A graph replayed with no host progress cannot use either.
3. **Device-side sum-combine** (`ep_sum_rows`): the routed-expert partials
   are combined on the host in `dist::fused_moe_mxfp4_ep` over `Channel`
   objects — unusable inside a graph, and not device-resident anyway.

Everything else — peer-access plumbing, adaptive DMA-vs-kernel dispatch,
transport classification, graph-capture modeling, C-ABI conventions — is in
place and is the *reason* this port is a kernel+plan job rather than a
framework job.

---

## 2. Proposed module

Follows the repo's two-implementation model exactly (the
`p2p_kv_restore` / `pipeline_boundary` / `fabric_import` pattern):

| Layer | Files | Compiled when | Tested |
|---|---|---|---|
| Host reference (mailbox geometry, row-scatter/sum-combine oracles, flag-mailbox **host model**, capture checklist, relay topology) | `src/c/vkernels/comm/ep_dispatch.{hpp,cpp}` | **always** | `tests/comm/test_ep_dispatch.cpp`, 100% line coverage on a GPU-less CI box |
| CUDA device path (5 kernels over `cudaStream_t`, peer-access assumed pre-established) | `src/c/vkernels/comm/ep_dispatch_cuda.{hpp,cu}` | `VKERNELS_HAS_CUDA` | on-device test `tests/comm/test_ep_dispatch_cuda_c.cu` |
| C ABI (create mailbox / signal / wait / execute plan / destroy) | `src/c/vkernels/comm/ep_dispatch_c.{h,cpp,cu}` | mirrors `p2p_kv_*_c` | `tests/comm/test_ep_dispatch_c.cpp` / `_cuda_c.cu` |
| ROCm | same `.cu` compiles as HIP (the `rccl.hip` precedent); MI300A notes in §6 | `VKERNELS_HAS_HIP` | on-device |

CMake: add the `.cpp` to the always-built list next to
`vkernels/comm/fabric_import.cpp` (src/c/CMakeLists.txt:42), the `.cu` to the
`VKERNELS_HAS_CUDA` block (:68–77), the C-ABI TUs to the `VKERNELS_C_HAS_CUDA`
group (:135–147).

### 2.1 Mailbox plan (host, always compiled)

```cpp
namespace vkernels::comm {

// One routing packet, byte-identical to the reference's outbox views():
// [activation bf16 [B, dim] | expert ids int32 [B, topk] | weights fp32 [B, topk]]
// padded to a 16-byte multiple (uint4 copies). Pure geometry + validation.
struct EpMessagePlan {
  int batch, dim, topk;
  std::size_t activation_bytes() const;   // B*dim*2
  std::size_t routing_bytes() const;      // 2*B*topk*4
  std::size_t packet_bytes()  const;      // ceil to 16
};

// Pair topology for the NVLink relay: partner = rank ^ 1 when the world is
// a power of two of pairs; relay_of(owner) = lowest rank of the far pair.
// Pure; throws when the world is not pair-structured.
struct EpPairTopology { int world; int partner_of(int r) const; int relay_of(int o) const; };
std::vector<EpPairTopology> build_pair_topology(int world);
}
```

`build_pair_topology` generalizes `ring_rank` (`topology.hpp`) the same way
the reference hard-codes `partner = d^1`; the host oracle unit-tests the relay
route (owner → partner + one far-pair relay → leaf) for world = 2 and 4.

### 2.2 `ep_scatter_rows` — the EP token dispatch (host oracle + CUDA kernel)

```
for every row b in [0, nb):
  dst_peer[seq[b] * bstride16 + row_idx[b] * row16 + j] = src[b * row16 + j],  j < row16
```

* Generalizes `p2p_copy_row`: `dst_base` is a **caller-established
  peer-accessible UVA pointer** (the peer's inbox), exactly the lifetime rule
  `p2p_gather_runs` already documents ("peer access and the IPC mapping must
  be established by the caller BEFORE the launch"). One `dst_base` per peer;
  the multi-peer fan-out is N launches of this kernel (or one
  `ep_multicast`, §2.4) — matching the reference, which loops targets for
  DMA mode and passes a pointer table for kernel-store mode.
* **Host oracle** (`ep_dispatch.cpp`): copies into a plain host byte buffer
  using the identical index math and validates the full contract — positive
  `row16`/`nb`, `row16 % 4 == 0` (uint4 granularity), `row_idx[b]` in range
  of the inbox row count, destinations **disjoint** across `b` within one
  launch (analogous to `validate_unique_slots` in `slot_map.hpp`), and
  `seq[b]*bstride16 + row_idx[b]*row16 + row16` within the inbox capacity.
  *Why disjointness:* `kv_scatter` rejects duplicate destination slots
  because two threads race one destination; here two rows land at
  `(seq, row_idx)` coordinates of one inbox, so the same uniqueness
  argument applies per `(seq[b], row_idx[b])` pair.
* **CUDA kernel** (`ep_dispatch.cu`): one block-tile per `(b, j-chunk)`,
  uint4 stores into peer memory, **check-free** — identical policy to
  `kv_scatter_layer_device_slots` (reading device memory to validate would
  force a D2H sync; the hot path trusts metadata the plan validated). The
  host-input variant stages the index arrays into owned device buffers once
  (plan-style), the device-input variant takes raw device pointers.
* **Sum-combine sibling** (`ep_sum_rows`): `dst[g*dst_stride + i] =
  Σ_{r<rows} src[(g*rows + r)*n + i]` for `g < groups`, fp32 accumulate,
  sequential row order for deterministic rounding. Host oracle trivially
  mirrors the kernel; the bf16-rounded variant (`EP_BF16_PART`) is a
  separate explicit rounding step so the oracle can assert the
  "fp32 sum, rounded once" contract instead of hiding conversion inside
  the accumulate loop.
* **Plan reuse:** mirror `P2PGatherPlan1D` — validate once, upload the
  pointer/index tables once, `execute()` enqueues with no per-call
  validation, allocation, or H2D metadata traffic. This is what makes 40
  layers × N peers of dispatch affordable inside a captured graph.

### 2.3 `DeviceFlagMailbox` — device-side flags (host model + CUDA kernels)

The heart of the port; nothing like it exists in `comm/` today.

```cpp
// Host model (ep_dispatch.cpp) — testable with no GPU:
//
//   flags_[rank]  : std::atomic<int> per (layer, sender-slot), monotonically
//                   increasing with the per-token sequence number
//   seq_[rank]    : the token counter, bumped once per token_end()
//   signal(rank, slots, seq)  : store `seq` to every slot (release)
//   wait(rank, slots, seq)    : block until every slot >= seq (acquire)
//
// The wait BLOCKS THE CALLING STREAM'S WORKER THREAD — with the host
// Stream model (one worker thread per stream, stream.hpp) each rank's
// stream progresses independently, so a cross-rank flag handshake makes
// progress exactly when the device scheme does: sender and waiter are on
// different streams/devices. A wait on the same stream that should signal
// it is a self-deadlock and is asserted in tests.
class DeviceFlagMailbox {
 public:
  DeviceFlagMailbox(int world, int layers, int slots_per_layer);
  void signal(int rank, Span<const int> slots, int seq);
  void wait(int rank, Span<const int> slots, int seq);   // blocking (condvar, not spin)
  void seq_bump(int rank);
  int  seq(int rank) const;
  bool has_reached(int rank, int slot, int seq) const;   // non-blocking probe (tests)
};
```

CUDA path (`ep_dispatch.cu`), mirroring the reference kernels:

* `ep_signal(flags_table, n, seq_ptr)` — `__threadfence_system()` before and
  after publishing `*seq_ptr` to up to 8 peer flag addresses (pointer table
  staged by the plan; same table shape as the reference's `sig_route` /
  `sig_part` / `sig_hop` tensors).
* `ep_wait(flags, n, seq_ptr)` — spin until every flag reaches the value;
  `threadfence_system` on exit. Requires the **dedicated non-blocking
  compute stream** rule (§3) — a wait enqueued on a legacy default stream
  would synchronize with the peer's default stream and deadlock, which is
  exactly the footgun `ep.py` documents for its `torch.cuda.Stream(d)`.
* `ep_seq_bump(seq_ptr)` — one atomicAdd, last op of a token.
* Host-input vs device-input `seq_ptr`: same two-variant split as
  `kv_scatter` (host value staged once per token; device counter owned by
  the mailbox so graph replays need no H2D).

**Watch-dog contract (new, worth adding beyond the reference):** the spin
kernel takes a max-iteration abort that writes a poisoned flag value so a
lost signal fails loudly instead of hanging the replay. The reference has no
timeout; our device tests use it, production can disable it.

### 2.4 `ep_multicast` — fused copy + last-block signal

One kernel: threads copy `n16` uint4 from the local outbox into up to 8 peer
inboxes; a device-scope `atomicInc` counter elects the last block, which
publishes the seq to every peer flag (the reference's `p2p_multicast`). The
host oracle asserts the two properties a plan must guarantee:

1. **one stream task per multicast** (count via `Stream::submitted()`, the
   same observable `p2p_gather` uses to prove "no per-run API calls"), and
2. **signal exactly once per execution** (the host model runs the
   last-block election as a deterministic tail step).

Dispatch note: the reference switches route packets between kernel stores
and copy engines at ~32 KiB (kernel stores win small: ~10 µs vs ~41 µs;
engines win large: 553 → 76 µs at B=16). Our `p2p_gather` already carries a
fitted cost model with the same shape (`est_copy_engine_us` /
`est_gather_kernel_us`, ~20 µs per-call floor vs ~8.6 µs kernel floor). The
`EpDispatchPlan` reuses `prefer_gather_kernel(bytes)` with an EP-specific
threshold knob, defaulting to the reference's 32 KiB; forced modes
(`kForceKernel` / `kForceCopyEngine`) mirror `GatherDispatchMode` for A/B.

### 2.5 Relation to `dist_moe` (EP semantics unchanged)

`dist::moe_ep_plan` / `moe_ep_dispatch` / the combine in
`fused_moe_mxfp4_ep` define *which* experts go where and how partials merge
— that layer is untouched. This candidate replaces its **transport**: the
host `Channel` all-to-all of activations and outputs becomes the device
mailbox (scatter-rows out, sum-combine back, flags instead of blocking
queue). `MockChannel` remains the GPU-less test double for the *collective*
tests; the new host model of `DeviceFlagMailbox` is the GPU-less test double
for the *mailbox* tests. The two are never mixed: `Channel` is a blocking
host queue (eager path only), the mailbox is stream-ordered device work
(capturable path only).

---

## 3. CUDA-graph capture rules

The existing `pipeline_boundary` machinery answers "is this capturable";
the EP scheme needs the operational checklist the reference
`EPRuntime::capture` discovered, stated as plan-level invariants:

1. **Everything in the captured segment is pure device work.** Scatter,
   sum-combine, signal, wait, seq-bump, multicast — all kernels, no host
   call. `classify` = `kSameNodePeer` ⇒ `is_graph_capturable` = true ⇒ no
   eager break inside a token. A deployment whose transport degrades to
   `kHostStaged` / `kHostBounce` (cross-node, no GPUDirect) must eager-break
   per token exactly as `eager_break_during_capture` prescribes — the
   `DeviceFlagMailbox` host model is what the eager path runs.
2. **Dry pass before capture.** Every kernel must be loaded/compiled on
   every device *before* any wait spins (a module load while a device spins
   in a wait kernel deadlocks the host — the reference's `self.dry` pass).
   Plan API consequence: `EpDispatchPlan::prepare()` executes a no-flag
   (`dry = true`) enqueue of all five kernels once per device before
   capture is allowed; `prepare()` refuses to capture if skipped.
3. **Non-blocking per-device streams.** Each rank's graph is captured and
   replayed on its own non-blocking stream; the plan asserts this (legacy
   default stream cross-synchronizes with peers ⇒ deadlock against flag
   waits). Host model analog: the Stream worker threads are per-rank, and
   tests exercise two ranks' streams interleaved per layer.
4. **Flag reset discipline.** Flags are monotonic (seq increases per token)
   and never zeroed between replays; only an explicit
   `reset_sync_state()`-style call (error recovery, as in the reference)
   zeroes flags *and* counters together, with all devices synchronized.
   Captured replays must never depend on a host-side reset between tokens —
   the seq bump is the last captured op (reference `token_end`).
5. **Static shapes and pre-resolved pointers.** All inbox/outbox/flag
   addresses are baked into the plan at prepare time; any buffer the token
   mutates per position (candidate buffers, rope tables) must be
   preallocated to the *graph* limit before capture, not grown lazily
   afterwards (the reference's exact-prealloc + `graph_token_limit` clamp
   exists precisely because lazy growth after capture invalidates captured
   pointers and 1M-token logical limits explode capture workspace).
6. **No allocation inside the segment.** execute() enqueues only — the
   `P2PGatherPlan` discipline, extended to all five kernels.

Acceptance (mirrors the pipeline-boundary acceptance style): N=2 and N=4
ranks, one token captured as a graph per rank, replayed M times with **no
host progress** between replays; partial results bit-identical to the
eager token; a killed-signal test hits the watch-dog instead of hanging.

---

## 4. Message flow per layer (what a captured token looks like)

```
owner device o (layer L)                     peer device p
─────────────────────────────                ─────────────────────────────
rmsnorm/swiglu_quant → outbox packet
gate_topk → eid, wt (in packet)
[ep_multicast outbox → inboxes, signal route_flag]  ──▶  ep_wait(route_flag)
ep_wait(part_flag[o][L])                     compute owned experts (grouped GEMM)
  ▲                                          ep_sum_rows → partial (fp32)
  ├── own partial signal                     [relay hop: partner/leaf variant]
compute own shard ‖ shared expert (side stream)
combine part_in rows + shared → residual
signal hop_flag / p2p_copy residual+bookkeeping ──▶ next owner waits hop_flag
ep_seq_bump (token end, every device)
```

The relay variant (`world == 4`, NVLink pairs, cross-pair links are PCIe):
route packet → partner (NVLink) + one far-pair relay (PCIe); relay forwards
to its partner over NVLink and raises the owner's partial slots for both far
GPUs; the far leaf sends its partial to the relay partner, which adds it
into its own partial — one PCIe transfer instead of two per direction. The
host oracle tests the route table (`build_pair_topology` + signal-slot
tables) for worlds 2 and 4; the device test only worlds it can address.

---

## 5. Testing plan

| Suite | Kind | Covers |
|---|---|---|
| `tests/comm/test_ep_dispatch.cpp` | host, no GPU, 100% line gate | `EpMessagePlan` geometry; `ep_scatter_rows` oracle vs hand-built expected bytes (incl. non-monotonic `seq`, duplicate-row rejection, capacity overflow); `ep_sum_rows` fp32 exactness + one-rounding bf16 variant; `DeviceFlagMailbox` handshake across two host `Stream`s (interleaved per-layer like `_eager_token`), self-wait deadlock assertion, monotonic flags, `reset_sync_state`; pair-topology relay routes; one-task-per-multicast counting |
| `tests/comm/test_ep_dispatch_cuda_c.cu` | on-device | peer-access round trip of scatter/multicast into a peer inbox, `ep_signal`/`ep_wait` across two devices, capture → replay ×N with no host progress, eager-vs-graph bit equality, watch-dog abort |
| `tests/comm/test_ep_dispatch_c.cpp` | host | C ABI surface: create/execute/destroy, error codes mapped from `std::invalid_argument` (the `c_abi_catch.hpp` convention) |
| bench | on-device | route-packet and partial-payload sweeps crossing the DMA threshold; per-layer trace stamps (`p2p_stamp` analog) reported like `trace_report` |

---

## 6. MI300A / ROCm analog

On MI300A the whole node is one coherent xGMI domain in a single process:
every partition's memory is already device-addressable from every other
partition, peer access is implicit, and — critically — GPU memory is
coherent with the CPU. Consequences for this plan:

* Transport classification: `same_node = true` always ⇒ `kSameNodePeer` ⇒
  capturable; `fabric_import` / `kHostBounce` machinery is dead weight on
  this target (relevant only for cross-node MI300X/CGRA deployments).
* The five kernels compile as HIP nearly unchanged (`rccl.hip` precedent):
  `__threadfence_system()` exists in HIP and, on MI300A, orders against the
  CPU as well (the APU shares one coherence domain), so the flag protocol is
  if anything *safer* than on discrete-GPU PCIe peers.
* No copy engines between partitions (one die, NPS-partitioned): the DMA
  branch of the adaptive dispatch is always the loser on MI300A — the
  threshold resolves to kernel-stores-only; keep the model but pin
  `min_runs`/threshold off for the AMD build.
* Spin-wait caveat: a spinning wait kernel occupies SMs; on MI300A the
  partitions share L2/interconnect backpressure with compute, so the
  watch-dog iteration bound (§2.3) matters more than on H100. Same guideline
  the repo already applies on gfx942: bounded waits, no unbounded resident
  spinners next to latency-critical GEMMs.
* `%globaltimer` → `wall_clock64`/`globaltimer` in HIP inline asm for the
  stamp kernel; cross-partition timestamps share one clock on the APU, so
  the trace timeline is directly comparable across ranks (better than
  multi-GPU discrete, where the reference's "same clock on every GPU of the
  node" claim needs NVLink SYSMeas to be true).

---

## 7. Deliverables checklist

1. `ep_dispatch.hpp/.cpp` — host reference: `EpMessagePlan`,
   `build_pair_topology`, `ep_scatter_rows`/`ep_sum_rows` oracles,
   `DeviceFlagMailbox`, capture checklist predicates, dispatch threshold.
2. `ep_dispatch_cuda.hpp/.cu` — the five kernels + plan execute paths
   (`VKERNELS_HAS_CUDA`).
3. `ep_dispatch_c.{h,cpp,cu}` — C ABI mirroring the `p2p_kv_*_c` pattern.
4. Tests per §5, wired into CMake (host list :29–43, CUDA list :68–77,
   C-ABI list :135–147 of `src/c/CMakeLists.txt`).
5. `docs/comm-ep-dispatch.md` — architecture doc in the
   `comm-pipeline-boundary.md` format (this plan is its seed).
6. Bench + trace-report port for the A/B evidence file.
