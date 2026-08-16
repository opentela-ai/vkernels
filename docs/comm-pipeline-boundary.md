# Graph-capturable PP-boundary transfer (issue #10)

A primitive for the hidden-state transfer at a **pipeline-parallel (PP)
boundary** that can be captured into a CUDA graph and replayed every decode
iteration with **no host progress** — the property that fixes the
graph-replay deadlock vLLM serving Kimi-K3 hits at the PP boundary, where a
captured `recv` waits forever for a host `send` that the graph never runs.

The boundary is classified into one of three transports. The **device path**
(same-node peer, cross-node NCCL/RCCL) is graph-capturable and never touches
the host ring; the **eager-break path** mirrors vLLM's
`eager_break_during_capture` and excludes the host `send`/`recv` from the
captured segment, running it between `GraphCapture::end` and the next
`GraphCapture::begin`.

> **Acceptance (from the issue):** a rank-pair round trip captured once and
> replayed N times with no host progress, and a PP>1 decode holding a graph
> segment across the boundary without deadlock. Both are exercised by
> `tests/comm/test_pipeline_boundary.cpp`; the CUDA device path is exercised
> by `tests/comm/test_pipeline_boundary_cuda_c.cu`.

---

## 1. Architecture

```
   vLLM / serving runtime
        │
        ▼  classify_boundary(cfg)
   ┌────────────────────────────────────────────────┐
   │ PipelineBoundaryConfig                          │
   │   same_node │ nccl_graph_supported │ gloo_fallback │
   └────────────────────────────────────────────────┘
        │
        ├─ kSameNodePeer  ─▶ device path: one peer-copy kernel     ─┐
        ├─ kCrossNodeNccl ─▶ device path: one ncclSend / ncclRecv     ├─ capturable
        └─ kHostStaged    ─▶ eager-break path: Channel send/recv      ┘   run between
                                                                              graph segments
   C++ caller ───────▶ PipelineBoundaryPlan (host, pipeline_boundary.cpp)
                         execute()  ├─ device: graph->submit_or_replay(memcpy)
                                    └─ eager:   graph->end(); channel.send/recv; graph->begin()
   Non-C++ caller ───▶ vkernels_pp_* (pipeline_boundary_c.{h,cpp,cu})
```

### Two-implementation model

| Layer | File | Compiled when | Coverage |
|---|---|---|---|
| Host reference (classification, eager-break decision, `GraphCapture`, host plan over `Channel`) | `src/c/vkernels/comm/pipeline_boundary.{cpp,hpp}` | **always** | 100% line (CI gate) |
| CUDA device plan (one peer-copy kernel / `ncclSend`-`ncclRecv` over `cudaStream_t`) | `src/c/vkernels/comm/pipeline_boundary.cu`, `pipeline_boundary_cuda.hpp` | `VKERNELS_HAS_CUDA` | on-device |
| C ABI — classification + eager-break | `src/c/vkernels/comm/pipeline_boundary_c.{h,cpp}` | **always** | 100% line (CI gate) |
| C ABI — device plan (create / execute / destroy over `cudaStream_t`) | `src/c/vkernels/comm/pipeline_boundary_c.cu` | `VKERNELS_C_HAS_CUDA` | on-device |

The host reference is the tested correctness oracle and the only artifact
the host CI job (and its 100% line-coverage gate) compiles. The CUDA path
mirrors its API exactly but takes a raw `cudaStream_t` and is built only
where a toolkit is present — exactly the `rccl` (host `.cpp` + HIP `.hip`)
and `p2p_kv_restore` (host `.cpp` + CUDA `.cu`) pattern.

---

## 2. Transport classification

`classify_boundary(cfg)` picks the transport with a fixed precedence:

| Config | Transport | Capturable? |
|---|---|---|
| `same_node` | `kSameNodePeer` — one peer-copy kernel over NVLink/HBM | yes |
| `gloo_fallback` (and not same-node) | `kHostStaged` — gloo `recv_object`/`send_object` or a host bounce | **no** |
| `nccl_graph_supported` (and not same-node, not gloo) | `kCrossNodeNccl` — `ncclSend`/`ncclRecv` whose graph-capture API is available | yes |
| otherwise | `kHostStaged` — NCCL without the graph API, or no NCCL | **no** |

`gloo_fallback` wins over `nccl_graph_supported`: a host-side gloo is the
deadlock trigger (a captured `recv` waiting on a host `send` the graph never
runs), so it is reported as host-staged regardless of NCCL availability.
`is_graph_capturable(t)` is `true` for the two device transports and `false`
for `kHostStaged`.

---

## 3. The device path (graph-capturable)

`PipelineBoundaryPlan` (host) over a `Channel` enqueues **one** `std::memcpy`
(`send`: `my_buf → peer_buf`; `recv`: `peer_buf → my_buf`) into the
`GraphCapture` supplied to `execute`. The CUDA `PipelineBoundaryPlan`
(`vkernels::comm::cuda`) is the device realization. For the same-node peer
it issues a small **copy kernel** (`peer_copy_kernel`) that reads peer UVA
over NVLink/HBM and writes the destination — *not* `cudaMemcpyAsync(...,
cudaMemcpyDeviceToWorld)`, which is **not** stream-capturable across two
*different* GPUs on Hopper / CUDA 13 ("operation not permitted when stream
is capturing" / "legacy stream depend on a capturing blocking stream");
`cudaMemcpyPeerAsync` fails identically. A kernel reading peer UVA IS
capturable in both the same-device and cross-device cases (peer access
enabled by the caller, exactly as `p2p_gather_bench` and `p2p_kv_restore`
document), which is why the boundary uses it. For cross-node the plan issues
one `ncclSend`/`ncclRecv` on `stream`. Either is captured verbatim between
`cudaStreamBeginCapture` / `cudaStreamEndCapture` and replayed every
iteration with **no host participation** — no `Channel` touch, no
validation, no allocation after construction. (Launch errors surface at
`cudaStreamEndCapture` on the graph path and at the next sync on the eager
path; `cudaGetLastError()` is intentionally *not* checked after the launch,
since during capture it can report a stale/warning status that aborts a
capture `endCapture` would otherwise accept.)

`PipelineBoundaryPlan::execute` is read-only after construction, so one plan
may be executed concurrently on several streams; the caller guarantees
`my_buf` (and, on the device path, `peer_buf`) outlive every stream the plan
is executed on.

---

## 4. The eager-break path

When the boundary is **not** graph-capturable (`kHostStaged`), the host
`send`/`recv` must be excluded from the captured segment and run between
launches — exactly vLLM's `eager_break_during_capture`. While a
`GraphCapture` is active, `execute` does:

```
graph->end();          // close the captured segment BEFORE the boundary
channel.send/recv();   // host progress, never captured
graph->begin();        // open the next segment AFTER the boundary
```

While **not** capturing, `execute` runs the `Channel` transfer immediately.
The boundary thus splits a decoded sequence of graph segments at every
host-staged hop, so a `recv` on the captured side never blocks on a `send`
the graph does not run.

---

## 5. `GraphCapture`

RAII wrapper over `cudaStreamBeginCapture` / `cudaStreamEndCapture` plus
`cudaGraphInstantiate` / `cudaGraphLaunch` (modelled on the host). The host
model records submitted work into a vector of captured segments and exposes:

- `begin()` / `end()` — open/close a segment (state-transition checked).
- `submit(work)` — record `work` into the open segment, or run it eagerly
  when no graph is being captured.
- `submit()` (no arg) — accept an already-arrived producer segment
  (multi-segment capture).
- `replay(n)` — replay the captured segments `n` times with no host work,
  returning the number of segments replayed.

Every state transition (`begin`/`end`/`submit`/`replay`) throws
`std::logic_error` on misuse (e.g. `end` without `begin`, `replay` while
capturing); the boundary plan never relies on these on the happy path.

---

## 6. Acceptance methodology

The host reference `tests/comm/test_pipeline_boundary.cpp` asserts both
acceptance criteria directly:

1. **Rank-pair round trip, captured and replayed N times with no host
   progress.** A same-node-peer round trip (send A→B, recv B→A) is captured
   into one `GraphCapture`, replayed `N` times, and the host `Channel` is
   asserted untouched after each replay (zero host progress). The same is
   asserted for a cross-node-NCCL round trip.
2. **PP>1 decode holding a graph segment across the boundary.** A PP=3
   pipeline (3 compute stages, 2 boundaries) runs the device path at every
   boundary inside a single `GraphCapture`; the captured segment is held
   across both boundaries and replayed with no deadlock. A PP=3 eager-break
   variant runs each host-staged boundary between `end`/`begin` and asserts
   the expected per-stage host progress.

The CUDA C ABI test `tests/comm/test_pipeline_boundary_cuda_c.cu` exercises
the device realization on a single GPU (same-device buffers as a stand-in
for peer memory): a captured send+recv round trip replayed `N` times via
`cudaGraphLaunch`, the create/execute/destroy status paths, and the
cross-node create path (execute refuses `VKERNELS_PP_ERR_UNSUPPORTED` when
NCCL is not linked — a real multi-node + NCCL harness exercises the
`ncclSend`/`ncclRecv` happy path).

---

## 7. C ABI

Non-C++ consumers reach the boundary through `pipeline_boundary_c.h`:

| Entry point | Compiled when | Role |
|---|---|---|
| `vkernels_pp_classify(cfg, status)` | **always** | `vkernels::comm::classify_boundary` over the C `vkernels_pp_config_t` |
| `vkernels_pp_eager_break(cfg, status)` | **always** | `vkernels::comm::eager_break_during_capture` (0/1) |
| `vkernels_pp_transport_name(t)` | **always** | human-readable transport name |
| `vkernels_pp_boundary_plan_create / _execute / _destroy` | `VKERNELS_C_HAS_CUDA` | prepared directed device plan over a `cudaStream_t` |

Every C++ exception thrown by the wrapped reference is caught and folded
into a `vkernels_pp_status_t`; nothing is ever thrown across the ABI
boundary. The always-compiled classification surface is 100% line-covered
by the host CI job (`tests/comm/test_pipeline_boundary_c.cpp`).

---

## 8. Device micro-benchmark (sgs-gpu07, 4× H100 NVL, CUDA 13)

`meta/benchmarks/bench_pipeline_boundary.cu` (built only with a CUDA
toolkit) is the device realization. It runs on a real NVLink pair
(CUDA_VISIBLE_DEVICES=2,3, 12× NVLink, ~600 GB/s bidirectional roof) and
reports four sections:

- **[0-correctness]** a 4100 B pattern (256 `uint4` + 4 B tail) sent
  A→B then received B→A, captured once into a graph and replayed 8×; both
  buffers are asserted to hold the pattern afterward. This exercises the
  cross-device + `<16 B` tail path the same-device unit test cannot.
- **[1-peer-copy]** the raw `cudaMemcpyAsync(D2D)` bandwidth floor the
  boundary kernel matches: NVLink saturates at ~265 GB/s unidirectional
  (44 % of the 600 GB/s roof), launch-bound below ~1 MiB.
- **[2-round-trip]** a directed boundary pair (send A→B, recv B→A)
  captured once vs. re-issued eagerly every iteration. The graph replays
  in a single `cudaGraphLaunch` (~3.3 µs, **payload-independent**) while
  the eager path re-enqueues both copies (~5.5 µs) — a **1.6–1.7× host-
  launch reduction** per boundary round trip. Device time is ≈ eager
  (the graph removes host coordination, not copy time — the honest
  result; the copy itself is NVLink-bound at ~265 GB/s).
- **[3-pp>1-one-graph]** PP=2 (one boundary, 2 copies) and PP=3 (two
  boundaries, 4 copies) captured into **one** graph and replayed `N`
  times with no host enqueue of the boundary copies on replay — the
  issue's acceptance #2 (a graph segment held across the boundary
  without deadlock).

Sample run (2025-05, sgs-gpu07, `--iters 300 --dst-device 1`):

```
[0-correctness] captured cross-device round trip (4100 B, 4 B tail, 8 replays): OK
[1-peer-copy]    1048576  0.0090  3.3   117.0 GB/s  0.195
[1-peer-copy]    67108864 0.2576  3.7   260.5 GB/s  0.434
[2-round-trip]    1048576 0.0156  0.0033  0.0171  0.0055  1.67x
[2-round-trip]    67108864 0.5416  0.0037  0.5436  0.0062  1.66x
[3-pp2-one-graph] eager: dev=0.0082 host=0.0055 ms/iter (2 copies)
[3-pp2-one-graph] graph: dev=0.0065 host=0.0033 ms/replay  (host/iter ZERO copy enqueue)
[3-pp2-one-graph] speedup=1.68x  (eager_host/graph_host host-launch ratio)
```

The host-side `bench_pipeline_boundary.cpp` is the always-runnable CPU
analog (no GPU needed): it measures the one-time planning overhead
(`classify_boundary` ~19 ns, plan construction ~29 ns) and the
per-decode host coordination cost of the device path (flat ~60 ns,
graph-captured) vs. the eager-break path (grows linearly in payload,
~1 µs→47 µs across 1 K→128 K elements) — a 19×→618× ratio, the
coordination work the graph removes.
