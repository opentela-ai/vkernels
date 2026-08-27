# Cross-node transfer for the prepared fused KV restore/donate kernels (issue #49)

A vkernels-side mechanism that makes the existing **prepared fused KV
restore / donate kernels** (issues #27, #36 — same-node NVLink peer only)
operate **across nodes**, instead of being replaced by a separate,
synchronous host-staged bulk copy (kvaas NIXL/libfabric `transfer()`).

The prepared kernels are transport-agnostic at the pointer level:
`*_execute_offset` dereferences `peer_src_ptrs` / `peer_dst_ptrs` that
must *already be device-addressable on the local fabric*. vkernels
provided nothing that made **remote VRAM directly addressable as a device
pointer** (`CU_MEM_HANDLE_TYPE_FABRIC` / IMEX), so cross-node KV reuse
today can only use the per-layer bulk copy — which bypasses the fused
kernels. On GH200 that path is further restricted: the libfabric plugin
only carries **DRAM↔DRAM** (hwloc-PCIe accelerator discovery cannot see
the C2C-attached H100, so VRAM is denied) and cross-node VRAM must bounce
through a pinned host buffer.

This module is the remaining slice.

> **Acceptance (from the issue):**
> 1. A cross-node prepared fused **restore** round trip (peer pages on
>    host B → local K/V layers on host A) and a cross-node **donate**
>    round trip (local K/V on A → peer pages on B) produce **byte-
>    identical results to the same-node NVLink path**, exercised over a
>    real RDMA fabric (not UCX loopback).
> 2. Measured cross-node throughput is reported per hop and compared
>    against the same-node roofline (~220–243 GB/s NVLink, ~88.5 GB/s
>    fused restore) and the current synchronous bulk-copy fallback, with
>    the GH200 DRAM-only / host-bounce caveat called out explicitly.
> 3. The host-bounce fallback validates correctly where a fabric-mapped
>    device pointer is unavailable.

Acceptance #1 and #2 are, by the issue's own statement, **hardware
exercises** on H-CLARIDEN / H-JSC — *"no real RDMA fabric is in any
container today."* The host reference (always compiled, 100% line-covered)
is the **byte-exact oracle** the CUDA path mirrors: it runs the existing
`*_execute_offset` kernels UNCHANGED over a `FabricImport` that mirrors
the remote bytes, so the host's `memcpy` *is* the same `memcpy` the
same-node path issues; the host-bounce path runs the existing
`kv_gather` / `kv_scatter` primitives, which the two-stage oracles
already prove byte-identical. The per-hop cost model (always compiled)
is the CI-verifiable surface for acceptance #2.

---

## 1. Architecture

```
   vLLM / serving runtime (peer pages on B, local K/V on A)
        │
        ▼  classify_fabric_import(cfg)
   ┌──────────────────────────────────────────────────────┐
   │ FabricImportConfig                                   │
   │   same_node │ has_gpudirect_rdma │ dram_only_libfabric │
   └──────────────────────────────────────────────────────┘
        │  precedence: same_node > dram_only > gpudirect > bounce
        ├─ kSameNodePeer  ─▶ device path: existing peer kernel UNCHANGED ┐
        ├─ kFabricMapped  ─▶ device path: VMM import -> existing kernel  ├─ capturable
        └─ kHostBounce    ─▶ eager-break path: kv_gather / ByteChannel      ┘   run between
                              send/recv / kv_scatter                              graph segments
   C++ caller ───▶ CrossNodeKvRestorePlan / CrossNodeKvDonatePlan (host,
                   cross_node_kv.cpp)        ├─ device: import.device_ptr()
                                               as peer base; ONE graph op
                                             └─ eager:   graph->end(); channel
                                               send/recv; graph->begin()
```

### Two-implementation model

| Layer | File | Compiled when | Coverage |
|---|---|---|---|
| Host reference (classification, eager-break decision, `FabricImport` mirror, cross-node plans, `GraphCapture` integration, cost model) | `src/c/vkernels/comm/fabric_import.{hpp,cpp}`, `src/c/vkernels/comm/cross_node_kv.{hpp,cpp}` | **always** | 100% line (CI gate) |
| CUDA device path (`CU_MEM_HANDLE_TYPE_FABRIC` import + the existing kernels over the imported pointer; pinned-host `cudaMallocHost` bounce) | `src/c/vkernels/comm/fabric_import.cu`, `src/c/vkernels/comm/cross_node_kv.cu`, `fabric_import_cuda.hpp` | `VKERNELS_HAS_CUDA` | on-device |

The host reference is the tested correctness oracle and the only artifact
the host CI job (and its 100% line-coverage gate) compiles. The CUDA path
mirrors its API but takes a raw `cudaStream_t` and the real **imported
device pointer** (the result of `cuda::fabric_import_device_ptr`, which the
caller owns — the host `FabricImport` class owns a host mirror a GPU
cannot dereference), and is built only where a toolkit is present —
exactly the `p2p_kv_restore` (host `.cpp` + CUDA `.cu`) and
`pipeline_boundary` (host `.cpp` + CUDA `.cu`) pattern.

---

## 2. Fabric import classification

`classify_fabric_import(cfg)` picks the transport with a fixed
precedence (mirrors `pipeline_boundary.hpp`'s `classify_boundary`):

| Config | Transport | Capturable? |
|---|---|---|
| `same_node` | `kSameNodePeer` — the existing NVLink/HBM/ROCm-IPC peer path, no import needed | yes |
| `dram_only_libfabric` (not same-node) | `kHostBounce` — the GH200 constraint: even with GPUDirect-RDMA on paper the C2C-attached GPU is invisible to the fabric plugin, so VRAM must bounce through pinned host | **no** |
| `has_gpudirect_rdma` (not same-node, not dram-only) | `kFabricMapped` — `CU_MEM_HANDLE_TYPE_FABRIC` / IMEX import yields a directly device-addressable pointer | yes |
| otherwise | `kHostBounce` — no fabric path at all | **no** |

`dram_only_libfabric` wins over `has_gpudirect_rdma`: on GH200 a
fabric-mapped device pointer is *unavailable in practice* (the import is
denied), so cross-node VRAM is reported as host-bounce regardless of
the RDMA capability bit — the caveat the issue asks to call out
explicitly. `is_import_graph_capturable(t)` is `true` for the two device
transports and `false` for `kHostBounce`.

---

## 3. The device path (graph-capturable)

A `FabricImport` yields a directly device-addressable pointer to the
remote VRAM:

- **kFabricMapped** (host reference) — an **owned local mirror** of the
  remote bytes, pre-populated from the published `FabricHandle` at
  construction. The existing `*_execute_offset` kernel reads the
  pre-populated mirror (restore) or writes it and `write_back()` ships
  it to the remote `FabricHandle` (donate). On the device this is the
  real `CU_MEM_HANDLE_TYPE_FABRIC` import: `cuMemAddressReserve` →
  `cuMemImportFromShareableHandle` → `cuMemMap` → `cuMemSetAccess`, yielding a
  pointer the kernels dereference UNCHANGED.
- **kSameNodePeer** (host reference) — the imported pointer **aliases**
  the published `FabricHandle` bytes (no copy), exactly as the
  same-node NVLink path aliases a peer pointer; `write_back()` is a
  no-op (the donate wrote directly through the alias). On the device
  this is `cudaIpcOpenMemHandle` / ROCm IPC / NVLink C2C: the peer
  pointer already names the remote physical memory.

`CrossNodeKvRestorePlan` / `CrossNodeKvDonatePlan` lay their per-page
peer bases out exactly as the same-node path
(`import.device_ptr() + p * page_layer_bytes`) and reuse the existing
`P2PKvRestorePlan` / `P2PKvDonatePlan`. `execute()` adds the per-layer
offset exactly as on the same node, so **the host reference's `memcpy`
is the same `memcpy` the same-node path issues** — the byte-identical
property acceptance #1 requires. While a `GraphCapture` is active,
`execute()` records **one** device op into the open segment and replays
it with no host progress (ties into #10).

---

## 4. The eager-break path (host bounce)

When the import is **not** graph-capturable (`kHostBounce`),
`FabricImport::device_ptr()` is `nullptr` and the plan takes the
host-bounce fallback:

```
donate:   kv_gather(local K/V -> pinned scratch)   channel.send(scratch)
restore:  scratch = channel.recv()                 kv_scatter(scratch -> local K/V)
```

The gather/scatter are the existing primitives the two-stage oracles
already prove byte-identical to the fused kernels, so the host-bounce
path produces the **same bytes** as the direct-store kernel. While a
`GraphCapture` is active, `execute()` does:

```
graph->end();          // close the captured segment BEFORE the bounce
channel.send/recv();   // host progress, never captured
graph->begin();        // open the next segment AFTER the bounce
```

exactly vLLM's `eager_break_during_capture` (ties into #10), so a
host-staged cross-node transfer is excluded from the captured segment
instead of freezing inside it. `fabric_import.hpp`'s
`eager_break_fabric_import(cfg)` is the pure decision the framework
uses; `CrossNodeKv*Plan::execute(channel=nullptr)` throws
`std::invalid_argument` — the acceptance-#3 validation that a
fabric-mapped device pointer is unavailable.

---

## 5. `FabricHandle` / `FabricImport`

A `FabricHandle` is the opaque descriptor a peer node publishes for a
range of its VRAM (or, on the DRAM-only path, its pinned host memory):
`{remote_node, token, bytes[], size}`. On CUDA it carries a
`CUmemGenericAllocationHandle` + the fabric-relevant fields; on ROCm the
IMEX descriptor equivalent. The host reference carries the published
bytes inline (a copy the local node reads through its imported mirror),
which is what makes the existing kernels produce byte-identical results
when they dereference the imported pointer.

`FabricImport` performs the import ONCE (per scope #1) and owns the
resulting pointer (or `nullptr` on `kHostBounce`). For `kFabricMapped`
the pointer points into an owned mirror and `write_back(remote)` ships
it to the remote `FabricHandle`; a move re-bases the pointer onto the
new mirror. For `kSameNodePeer` the pointer aliases a borrowed
`FabricHandle` (no mirror; the handle must outlive the import), so a
move keeps the alias (the handle does not move with the import) and
`write_back()` is a no-op.

---

## 6. Per-hop cost model

`cross_node_kv_throughput(transport, total_bytes, gh200_dram_only=)`
reports the estimated per-hop throughput (GB/s) and time (µs), compared
against the same-node roofline and the synchronous bulk-copy fallback:

| `CrossNodeHopCost` field | Source |
|---|---|
| `per_hop_gbps` | `kFabricMapped`: `min(fabric 50, kernel 88.5)` GB/s (HDR 400 Gb/s ÷ 8; the kernel is the binding resource) — **degrades to the bulk-copy fallback when `gh200_dram_only`**. `kSameNodePeer`: the fused-restore roof 88.5. `kHostBounce`: the bulk-copy fallback. |
| `same_node_roof_gbps` | 88.5 (the fused Stage-3 peer restore on sgs-gpu07; NVLink raw is 220–243) |
| `bulk_copy_fallback_gbps` | 1.4 (the Slingshot TCP fallback on the DRAM-only libfabric transport; UCX has no CXI provider) |
| `gh200_dram_only_caveat` | `true` for `kHostBounce`, and for `kFabricMapped` when `gh200_dram_only` is set |

Roofline references are measured on the sgs-gpu07 devbox (see the issue
and `.agents/docs/operations`):
~220–243 GB/s on NVLink pairs, ~54–55 GB/s cross-pair PCIe, ~88.5 GB/s
fused restore, ~1.4 GB/s synchronous bulk-copy fallback, ~0.34 GB/s
NIXL/UCX loopback (context only — the issue requires a real RDMA
fabric, not a loopback). The real-RDMA-fabric per-hop number is a
hardware exercise on H-CLARIDEN / H-JSC.

The host bench `meta/benchmarks/bench_cross_node_kv.cpp` (no GPU needed)
is the CI-verifiable surface for acceptance #2: it prints the per-hop
cost table across a payload sweep, comparing each transport against the
same-node roofline and the bulk-copy fallback, with the GH200
DRAM-only / host-bounce caveat called out.

---

## 7. Acceptance methodology

The host reference `tests/comm/test_cross_node_kv.cpp` (53 tests) asserts
all three acceptance criteria directly on the byte-exact oracle:

1. **Byte-identical to the same-node NVLink path.**
   - `CrossNodeRestore.DirectPathMatchesSameNode` /
     `SameNodePeerMatchesSameNode` / `AsyncExecuteIsOneTaskAndCorrect`:
     the cross-node restore produces the same K/V as the same-node
     `P2PKvRestorePlan` for every layer.
   - `CrossNodeDonate.DirectPathMatchesSameNode` /
     `SameNodePeerMatchesSameNode` / `AcceptsRepeatedSlots` /
     `AsyncExecuteIsOneTaskAndCorrect`: the cross-node donate lands the
     same bytes in the remote `FabricHandle` as the same-node
     `P2PKvDonatePlan`.
   - `CrossNodeHostBounce.DonateRestoreRoundTripMatchesDirect`: a full
     donate-on-A → restore-on-B round trip over a `ByteChannel` matches
     the direct (fabric-mapped) round trip.
2. **Per-hop throughput vs roofline + bulk-copy, GH200 caveat.**
   `CrossNodeCost.*` pins `cross_node_kv_throughput` for every transport
   (fabric-mapped below the 88.5 roof and at 50 GB/s; same-node at roof;
   host-bounce at the 1.4 fallback *with the caveat set*; fabric-mapped
   degrading to the fallback when `gh200_dram_only`).
3. **Host-bounce validates where device ptr is unavailable.**
   `FabricImport.HostBounceHasNoDevicePtr` / `DefaultIsEmptyBouncePlaceholder`,
   `CrossNodeRestore.HostBounceNeedsChannelAtExecute`,
   `CrossNodeDonate.HostBounceNeedsChannelAtExecute`,
   `CrossNodeHostBounce.NoFabricMappedPointerForBounce` /
   `RecvRejectsUndersizedChunk`, plus the contract tests
   (`RejectsCapturableWithoutImport`, `RejectsDuplicateSlot`,
   `RejectsOutOfRangeSlot`, `RejectsNonBF16`).

The graph-capturable integration (ties into #10) is asserted by
`CrossNodeGraph.CapturablePathRecordsOneOpNoHostProgress` (one device
op, replayed with no host progress / no channel) and
`CrossNodeGraph.HostBounceEagerBreaks` (the bounce ends the segment,
runs over the channel, begins the next — the
`PipelineBoundaryPlan` eager-break contract).

The on-device realization (fabric import + the existing kernels over the
imported pointer, pinned-host bounce) is exercised on H-CLARIDEN /
H-JSC over a real RDMA fabric — the measurement step the issue's
acceptance #1/#2 names explicitly and that no container can run today.

Where a CUDA toolkit is present, the gated C-ABI tests
`tests/comm/test_cross_node_kv_c.cu` and `tests/comm/test_fabric_import_cuda_c.cu`
mirror the byte-exact oracle through the `vkernels_cross_node_kv_*` /
`vkernels_fabric_bounce_*` `extern "C"` surface: a same-device pointer
stands in for the imported remote VRAM (kFabricMapped) and a
`cudaMallocHost` buffer for the network payload (kHostBounce), sufficient
to exercise the validators, the kernel launches, and the status-code
return path for all three acceptance criteria — the same
host-`.cpp`-+-CUDA-`.cu` split every other CUDA C ABI in the module uses.

---

## 8. References

- Same-node prepared fused kernels: `src/c/vkernels/comm/p2p_kv_restore.{hpp,cpp}`, `src/c/vkernels/comm/p2p_kv_donate.{hpp,cpp}` (issues #27, #36).
- Graph-capturable boundary (the eager-break contract this ties into): `src/c/vkernels/comm/pipeline_boundary.{hpp,cpp}`, `docs/comm-pipeline-boundary.md` (issue #10).
- kvaas FFI inventory: `src/crates/kvaas-py/src/vkernels_ffi.rs` ("Kernel inventory" section).
- kvaas Python transport: `src/python/kvaas_sglang/{peer_transport,nixl_transport}.py`.
- kvaas design note: `.agents/docs/design/kvaas-owned-device-kv-pool.md` (cross-node VMM = Fabric-handle/IMEX phase).
- Same-node roofline: `.agents/docs/operations/codebase-drift.md` §9.2; `.agents/docs/experiments/2026-08-14-bristen-stage3-bandwidth-64k-restore.md`.
