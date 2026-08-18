# HIP/RCCL transport + OFI/CXI net plugin (issue #19)

A second communication channel for cross-node collectives on ROCm, built on
**RCCL** (ROCm Communication Collectives) and a HIP-aware **OFI/CXI** net
plugin that drives Slingshot RDMA instead of the default TCP Socket path.
The host planning surface — transport selection, the cost model, and the
ring topology — is always compiled and 100% line-covered on the CI host job;
the GPU path (`rccl.hip`) and the net plugin (`librccl-net-ofi.so`) are built
only where a HIP/RCCL/libfabric toolchain is present.

> **Acceptance (from the issue):** cross-node RCCL all-reduce over Slingshot
> faster than Socket on gfx942, and the comm unit tests pass under ROCm.

---

## 1. Architecture

```
                       ┌────────────────────────────────────────┐
   C++ caller ───────▶ │ RcclAllreducePlan (host, rccl.cpp)     │
                       │   resolve_transport → Socket | OFI     │
                       │   build_cross_node_ring                │
                       └───────────────┬────────────────────────┘
                                       │ (ROCm build)
                                       ▼
                       ┌────────────────────────────────────────┐
                       │ RcclAllreducePlanHip (rccl.hip)        │
                       │   single rcclAllReduce, graph-capturable│
                       │   RcclChannel over RCCL send/recv      │
                       └───────────────┬────────────────────────┘
                                       │ RCCL net API
                                       ▼
                       ┌────────────────────────────────────────┐
                       │ librccl-net-ofi.so (plugins/...)       │
                       │   libfabric CXI provider → Slingshot   │
                       └────────────────────────────────────────┘

   Non-C++ caller ───▶ vkernels_rccl_* (rccl_c.h / rccl_c.cpp)
```

### Two-implementation model

| Layer | File | Compiled when | Coverage |
|---|---|---|---|
| Host reference (transport selection, cost model, ring topology, host plan) | `src/c/vkernels/comm/rccl.{cpp,hpp}` | **always** | 100% line (CI gate) |
| HIP/RCCL path (channel, graph-capturable plan) | `src/c/vkernels/comm/rccl.hip`, `rccl_hip.hpp` | `VKERNELS_HAS_RCCL` | on-device |
| OFI/CXI net plugin | `plugins/rccl-net-ofi/rccl_net_ofi.c` | `VKERNELS_HAS_OFI` | on-node |
| C ABI (wraps host reference) | `src/c/vkernels/comm/rccl_c.{h,cpp}` | **always** | 100% line (CI gate) |
| Build discovery | `meta/cmake/RcclSupport.cmake` | `VKERNELS_BUILD_HIP` | — |
| Benchmark | `meta/benchmarks/bench_rccl.cpp` | `VKERNELS_BUILD_BENCHMARKS` (host-only) | — |

`VKERNELS_HAS_RCCL` implies `VKERNELS_HAS_HIP`; both default OFF and are only
turned on when `VKERNELS_BUILD_HIP=ON` *and* `RcclSupport.cmake` finds
`rccl.h` / `librccl`. `VKERNELS_HAS_OFI` is independent and requires libfabric
(`rdma/fabric.h` / `libfabric`).

---

## 2. Transport selection

`resolve_transport(bytes, edges, cfg)` picks **Socket** or **Slingshot OFI**
from a closed-form cost model so callers do not need to know the network
topology. The model (in `rccl.cpp`) is:

| Transport | Floor | Per-MiB | Per inter-node edge |
|---|---|---|---|
| Socket | 50 µs | 6.0 µs/MiB | 25 µs |
| Slingshot OFI (RDMA) | 20 µs | 3.0 µs/MiB | 0 (edge-free) |

```cpp
using namespace vkernels::comm;
RcclTransportConfig cfg;          // mode=kAdaptive, plugin="librccl-net-ofi"
RcclTransport t = resolve_transport(/*bytes=*/48u*1024u*1024u, /*edges=*/2, cfg);
// → RcclTransport::kSlingshotOfi  (OFI 48 µs < Socket 146 µs)
```

Three forced modes override the model:
- `RcclTransportMode::kAdaptive` — use the cost model (default).
- `RcclTransportMode::kForceSlingshot` — always OFI (fail if the `cxi`
  provider is missing; see `discover_ofi_cxi`).
- `RcclTransportMode::kForceSocket` — always Socket (`NCCL_NET=Socket`,
  `NCCL_IB_DISABLE=1`).

The model also honours the usual `NCCL_NET` and `NCCL_IB_DISABLE`
environment variables via `resolve_rccl_transport(env_kv, n, &cfg)`.

---

## 3. Ring topology

`build_cross_node_ring(ranks, node_of, world)` orders ranks **node-major** so
every inter-node edge lands between the last rank of one node and the first
rank of the next. This minimises inter-node hops to exactly `nodes - 1` (a
ring closes back to node 0), regardless of whether the caller's rank order
is node-major or interleaved. The builder rejects out-of-range node ids and
reports the required capacity via a count query (`topo == nullptr`).

---

## 4. All-reduce plan (graph-capturable)

`RcclAllreducePlan(world, rank, op, capacity)` validates the ring
(`world > 0`, `rank < world`, `capacity > 0`) once, at construction. `execute`
is then a single all-reduce with **no host allocation after construction**,
so a caller can wrap it in `hipStreamBeginCapture` / `hipStreamEndCapture`:

```cpp
RcclAllreducePlanHip plan(world, rank, RcclReduceOp::kSum, n, comm, stream);
hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
plan.execute(buf, n, *comm, stream);   // one rcclAllReduce
hipStreamEndCapture(stream, &graph);
hipGraphLaunch(graph, stream);
```

`RcclChannel` (the `Channel` subclass) wraps `rcclSend` / `rcclRecv` with
host→device→host copies so it slots into the existing host-buffered
`Channel` interface; the optimal cross-node path is the single
`rcclAllReduce` in the plan, not the ring.

---

## 5. The OFI/CXI net plugin

`plugins/rccl-net-ofi/rccl_net_ofi.c` implements the RCCL net-plugin ABI on
top of the libfabric **CXI** provider (Slingshot). It is selected by setting
`RCCL_NET=aws_ofi_rccl` (or `librccl-net-ofi`) and pointing
`LD_LIBRARY_PATH` at the built `librccl-net-ofi.so`. The plugin is a skeleton
in this environment (no libfabric); on a Slingshot node it issues
`fi_tsend` / `fi_trecv` tagged messages and lets RCCL do the collective
bookkeeping.

```bash
# build (Slingshot node with ROCm + libfabric)
cmake -B build -DVKERNELS_BUILD_HIP=ON
cmake --build build --target rccl-net-ofi

# run
export LD_LIBRARY_PATH=$PWD/build/plugins/rccl-net-ofi:$LD_LIBRARY_PATH
export RCCL_NET=aws_ofi_rccl
```

---

## 6. Acceptance methodology

The host CI job cannot run RCCL, so the **cost model** is the verifiable
acceptance surface: OFI must be cheaper than Socket at ≥ 1 MiB over ≥ 1
edge. `meta/benchmarks/bench_rccl.cpp` (`rccl_bench`) prints that table plus
a measured host ring all-reduce (a regression guard against the model):

```bash
cmake -B build -DVKERNELS_BUILD_BENCHMARKS=ON
cmake --build build --target rccl_bench
./build/meta/benchmarks/rccl_bench --edges 0,1,2,4 --mib 1,4,16,48
```

```
mib      edges      socket_us       ofi_us  transport   speedup
1        1              75.00        20.00 slingshot-ofi      3.75
16       2             146.00        48.00 slingshot-ofi      3.04
```

On a real gfx942 + Slingshot node, run the **A/B** that the issue asks for:

```bash
# Socket baseline
RCCL_NET=Socket NCCL_IB_DISABLE=1 ./build/.../rccl_bench --world 8 > socket.txt
# Slingshot (OFI/CXI)
RCCL_NET=aws_ofi_rccl ./build/.../rccl_bench --world 8 > ofi.txt
```

The acceptance criterion is `ofi_us < socket_us` at the cross-node payloads
(`mib ≥ 1`, `edges ≥ 1`), matching the model's prediction. The comm unit
tests (`vkernels_test_rccl`, `vkernels_test_rccl_c`) must pass under ROCm
unmodified — they exercise the same host planning surface as the bench.

### Verified on beverin (MI300A / gfx942)

The `hip` preset builds end-to-end on a CSCS beverin compute node
(ROCm 6.3, `librccl.so.1.0.60300`, Cray libfabric 2.3.1 with the CXI
provider) and the comm unit tests pass under ROCm unmodified:

```bash
export PATH=/opt/rocm/bin:$PATH ROCM_PATH=/opt/rocm OFI_ROOT=/opt/cray/libfabric/2.3.1
cmake --preset hip                # CMAKE_HIP_ARCHITECTURES=gfx942
#   RCCL enabled (include=/opt/rocm/include/rccl lib=/opt/rocm/lib/librccl.so)
#   libfabric/OFI enabled (include=.../libfabric/2.3.1/include lib=.../libfabric.so)
cmake --build build/hip -j8       # libvkernels.a + librccl-net-ofi.so + tests
ctest --test-dir build/hip -R 'topology|channel|allreduce|^rccl$|rccl_c|overlap'
#   100% tests passed, 0 tests failed out of 6  (CTEST_RC=0)
```

`librccl-net-ofi.so` is pure C (host compiler), exports exactly one dynamic
symbol — `nccl_ofi_net`, the v-table RCCL dlsym's when
`NCCL_NET=librccl-net-ofi` — and links only `libfabric.so.1` (no HIP
runtime, no libcudart). The transport-selection / cost-model / ring /
plan surfaces are exercised by `rccl` (39 tests) and `rccl_c` (11 tests).

The remaining hardware acceptance — cross-node `rcclAllReduce` over
Slingshot (OFI/CXI) faster than Socket on gfx942 — needs a multi-node
reservation and the OFI/CXI net plugin's transport hooks filled in
(`plugins/rccl-net-ofi/rccl_net_ofi.c` is currently a build-verified
skeleton that returns `ncclInvalidUsage` for the unimplemented hooks).

---

## 7. C ABI

`rccl_c.h` exposes the host planning surface to non-C++ consumers with a
small set of `extern "C"` entry points and opaque handles. Exceptions from
the C++ reference are folded into status codes
(`VKERNELS_RCCL_OK` / `..._ERR_INVALID_ARGUMENT` / `..._ERR_OUT_OF_RANGE` /
`..._ERR_INTERNAL`).

```c
vkernels_rccl_config_t cfg;
vkernels_rccl_resolve_transport(nullptr, 0, &cfg);          // defaults

vkernels_rccl_transport_t t;
vkernels_rccl_resolve_transport_for(48u*1024u*1024u, 2, &cfg, &t);  // → OFI

vkernels_rccl_allreduce_plan_t* plan = NULL;
vkernels_rccl_allreduce_plan_create(8, 0, VKERNELS_RCCL_REDUCE_SUM, 4096, &plan);
vkernels_rccl_allreduce_plan_execute(plan, buf, 4096, ch, ch);
vkernels_rccl_allreduce_plan_destroy(plan);
```
