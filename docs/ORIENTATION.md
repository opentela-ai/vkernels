# Orientation — `vkernels`

> You just cloned this repo. This doc tells you what it is, why it exists,
> how it's organized, and where to go next. It is the conceptual entry
> point; the per-feature docs under `docs/kernels/` and `docs/` are the
> references it links out to.

---

## 1. What `vkernels` is, in one paragraph

`vkernels` is the **kernel layer for [opentela](https://github.com/opentela-ai)** —
a library of hand-tuned GPU kernels (attention, MoE, GEMM) and the GPU
**communication primitives** (collectives, peer transfers, pipeline
boundaries) that sit beside them, written in C++ with a CPU reference and a
GPU path for every operation. Its reason for being is concrete and slightly
unusual: the production attention and MoE layers of modern open-weight
models — **Kimi-K3** (MLA + gated-delta attention), **GLM-5.3-Flash /
DeepSeek-V3** (sparse MLA, hybrid attention), and a **MXFP4 fused-MoE** —
are served today by vendor frameworks (AMD AITER, tilelang) that **GPU-fault
or JIT-abort on gfx942** (MI300A/MI300X, CDNA3), because they depend on
CDNA4-only (gfx950) instructions. `vkernels` re-implements those layers as
portable, hand-tuned HIP kernels that run correctly on gfx942 *and* match a
cross-checked CPU oracle bit-for-bit, so the models serve without custom
overlays or feature-disable flags.

It lives inside the `opentela-ai/serving-stack` workspace, alongside
sibling projects `kvaas` (KV-cache-as-a-service), `diffcache`, `floe`, and
`benchmark`. `vkernels` is the layer those services call down into for the
actual on-device math and data movement.

---

## 2. The one idea that explains the whole codebase: the two-implementation model

Every kernel and every collective in this repo is provided **twice**:

| Implementation | File suffix | Compiled when | Role |
|---|---|---|---|
| **CPU reference** | `*.cpp` | **always** (no GPU needed) | The correctness oracle. Fully unit-tested. The only artifact the host CI job (and its **100% line-coverage gate**) compiles. |
| **GPU implementation** | `*.cu` (NVIDIA) or `*.hip` (AMD) | only when a toolkit is found (`VKERNELS_HAS_CUDA` / `VKERNELS_HAS_HIP`) | The fast path. GPU tests compare its output against the CPU reference. |

This single decision drives almost everything you'll see:

- You can **develop, review, and reach 100% coverage on any laptop** with no
  GPU. The GPU path is exercised only on machines that have a toolkit and
  device (self-hosted CUDA runners; gfx942 boxes).
- The CPU reference is **not** a toy — for the real models it is a faithful,
  numerically-stable, often-bit-exact emulation of the GPU kernel (same
  online-softmax, same bf16 rounding points, same stage decomposition). The
  GPU kernel is "correct" precisely because it matches this reference.
- Headers under `src/c/vkernels/**` carry a comment block stating this split
  explicitly (e.g. "The CPU reference implementations below are always
  compiled and treated as the correctness oracle; the CUDA launchers live in
  `elementwise.cu`").

When you open any kernel directory, expect this trio: `name.hpp` (public
API), `name.cpp` (CPU oracle), `name.cu` or `name.hip` (device kernel).

---

## 3. How the repository is laid out

```
vkernels/
├── src/c/                 # the C++/CUDA/HIP kernel library (the core)
│   └── vkernels/
│       ├── util/          #   Span, error/status, logging, config flags, version
│       ├── core/          #   Device, Stream (worker-thread CUDA-stream model), allocator
│       ├── kernels/       #   elementwise, reduce, gemm, gemm_bf16, moe*,
│       │                  #   mla, kda, dsa, mhc  (attention + MoE)
│       ├── comm/          #   allreduce, overlap, topology, channel,
│       │                  #   p2p_gather, p2p_kv_restore/donate, kv_gather/scatter,
│       │                  #   cross_node_kv_allgather, pipeline_boundary, rccl (+OFI)
│       ├── dist/          #   distributed MoE: TP / EP / PP orchestration
│       └── capi/          #   extern "C" ABI shim (+ exception translation) for non-C++ consumers
├── src/python/            # `vkl` CLI + Python bindings (compiled pybind11 OR pure-Python fallback)
├── src/rust/              # Rust bindings: `vkernels-sys` (FFI) + safe `vkernels` crate
├── tests/                 # C++ unit tests mirroring src/c/ + tests/python/ + tests/capi/
├── meta/                  # build support: cmake helpers, scripts, docker, benchmarks
├── docs/                  # architecture, per-kernel and per-comm docs, performance records
├── plugins/rccl-net-ofi/  # HIP-aware OFI/CXI net plugin for Slingshot RDMA (built with HIP)
└── .github/workflows/     # host (coverage) + CUDA CI
```

Three things to notice up front:

- **`src/c/` is the source of truth.** The Python and Rust packages are
  thin bindings over it; the `vkl` CLI is a *scanner* of `src/c/` headers,
  not a separate manifest (see §6).
- **`meta/` is not docs** — it's the build/test/benchmark toolchain: CMake
  modules (`CudaSupport`, `HipSupport`, `NcclSupport`, `RcclSupport`,
  `Coverage`, `Sanitizers`, `Testing`), the 100%-coverage gate script,
  Dockerfiles (host + NV profiling image), and the `.hip`/`.cu`
  micro-benchmarks and GPU correctness harnesses.
- **`docs/` is dense and worth reading.** Each kernel has a focused doc
  giving the math, the CPU oracle, the device-kernel strategy, the
  contract, and a measured roofline benchmark. Many also keep
  per-GPU performance journals under `docs/performance/`.

---

## 4. What's actually implemented — and why each thing exists

The kernels split into two tiers. Understanding both is the fastest way to
"get" the project.

### Tier 1 — the pedagogical / infrastructure kernels (CUDA path)

Small, fully portable kernels that establish the pattern and the build/test
machinery. Documented in [`docs/README.md`](README.md) and per-kernel docs.

| Kernel | What it does | GPU backend |
|---|---|---|
| `add`, `scale`, `relu` | element-wise | CUDA |
| `sum`, `max` | block/grid reduction (two-stage) | CUDA |
| `gemm` | tiled 16×16 SGEMM, `C = α·A@B + β·C` | CUDA |

These exist so the two-implementation model, the C ABI, the Python and
Rust bindings, and the 100%-coverage gate all have something concrete to
exercise on every platform.

### Tier 2 — the production attention & MoE layers (HIP / gfx942)

This is the heart of the project. **Each one exists because a vendor path
broke on gfx942** (MI300A/MI300X, CDNA3), and `vkernels` replaces it with a
correct, hand-tuned HIP kernel validated against a CPU oracle.

| Component | Model / role | Why it's here | Doc |
|---|---|---|---|
| **MLA** | Kimi-K3 Multi-head Latent Attention forward (absorbed form, online softmax) | vLLM TRITON_MLA / AITER needs no vendor fallback on gfx942 | [`kernels/mla.md`](kernels/mla.md) |
| **KDA** | Kimi-K3 gated delta-rule linear attention (7 kernels: gated RMSNorm, gate cumsum, delta-rule intra/inter/output, bit-matrix pack) | AITER/Triton chunked kernels GPU-fault on gfx942 (job 586165); K3 serving had to set `K3_DISABLE_KDA=1` | [`kernels/kda.md`](kernels/kda.md) |
| **DSA** | GLM-5.3-Flash / DeepSeek-V3 sparse-MLA forward (top-k indexed KV) | tilelang `sparse_mla_*` decode/prefill kernels GPU-fault on gfx942; the `tail_dim == 0` GLM-5.3-Flash shape **can't even compile** in tilelang | [`kernels/dsa.md`](kernels/dsa.md) |
| **MHC** | GLM-5.3-Flash hybrid-attention pre/post kernels | tilelang forms JIT-abort on gfx942 (dynamic shared memory > 64 KB non-optin cap) | [`kernels/mhc.md`](kernels/mhc.md) |
| **MoE primitives** (#12–15) | direct-to-LDS fill, fp4→bf16 dequant, async-copy gate, K16 bf16 MFMA | fill gaps where CDNA4-only (gfx950) AITER instructions don't lower on CDNA3 | [`kernels/moe.md`](kernels/moe.md) |
| **MoE aux** (#22) | MXFP4 quant, expert sort, scatter-reduce | AITER `module_moe_mxfp4_aux` ~82 KB LDS exceeds the gfx942 64 KB limit | [`kernels/moe_aux.md`](kernels/moe_aux.md) |
| **MoE fused** | end-to-end MXFP4 fused-MoE grouped GEMM (2 kernel launches) | wires the primitives above into the `xkernels fused_moe_mxfp4` interface | [`kernels/moe_fused.md`](kernels/moe_fused.md) |
| **bf16 GEMM** | K16 bf16 MFMA GEMM for the Kimi-K3 projection shapes | AITER falls back to an untuned torch path (no gfx942 entries in `bf16_tuned_gemm.csv`) | [`kernels/gemm_bf16.md`](kernels/gemm_bf16.md) |
| **Distributed MoE** | TP / EP / PP orchestration around the fused kernel | Kimi-K3 needs ≥4 nodes × 4 MI300A (TP8×PP2); the activation is nonlinear so the TP all-reduce must land *between* the GEMM and the SwiGLU, which forces the fused call to be split into composable stages | [`kernels/moe_dist.md`](kernels/moe_dist.md) |

The recurring motif: **a CPU reference that is the cross-checked oracle,
plus a correctness-first HIP kernel** (often a cooperative per-token
recurrence or an online-softmax path that matches the oracle to within fp32
round-off), followed by documented, measured performance and a list of the
specific tuning levers that helped (and the ones — like LDS double-buffering
— that were tried and *reverted* because the kernel was occupancy-bound, not
bandwidth-bound).

### Communication primitives

Built with the same two-implementation model. These matter because the
serving stack (`kvaas`, vLLM-style deployments) moves KV-cache and
hidden-state across GPUs and nodes, and graph-capture correctness is the
live concern.

| Primitive | Role | Doc |
|---|---|---|
| **Ring all-reduce** | reduce-across-ranks via ring topology (CPU + CUDA) | [`README.md`](README.md#communication-primitives) |
| **P2P run-list gather** | single-launch gather of many disjoint byte-runs from peer UVA, replacing per-run `cudaMemcpyPeerAsync` loops (adaptive copy-engine vs kernel) | [`kernels/p2p-gather.md`](kernels/p2p-gather.md) |
| **P2P KV restore / donate** | fused indexed gather + scatter between local paged-KV and peer UVA (the KV-cache-as-a-service data path) | [`README.md`](README.md#communication-primitives) |
| **KV gather / scatter** | fused indexed K/V layer gather (and its scatter) | [`README.md`](README.md#communication-primitives) |
| **Cross-node KV all-gather** | equal-shard NCCL KV all-gather with a prepared, stable serving-runtime C ABI | [`comm-cross-node-kv.md`](comm-cross-node-kv.md) |
| **Overlap executor** | runs `compute_fn` / `comm_fn` on two streams in lockstep (testable on host via the worker-thread stream model) | [`README.md`](README.md#communication-primitives) |
| **RCCL + OFI/CXI net plugin** | a second HIP/RCCL transport behind the `Channel`/all-reduce interface, plus a Slingshot RDMA net plugin (AMD only) | [`comm-rccl.md`](comm-rccl.md) |
| **Pipeline-parallel boundary** | a graph-capturable hidden-state transfer that **fixes the graph-replay deadlock** vLLM serving Kimi-K3 hits at a PP boundary (captured `recv` waiting forever for a host `send` the graph never runs) | [`comm-pipeline-boundary.md`](comm-pipeline-boundary.md) |

A pattern you'll see across the comm primitives: a **prepared plan** (host +
GPU) that validates the slot map and uploads page descriptors *once*, then
`execute(...)` launches a single kernel per layer with no per-call allocation
or H2D copy — because serving runtimes reuse the same run list across all
model layers and must stay capture-safe. The `PipelineBoundaryPlan`,
`P2PKvRestorePlan`, and `P2PKvDonatePlan` all follow this shape.

---

## 5. The three language bindings (and the C ABI they share)

`vkernels` is a C++ library first, but it's meant to be called from
non-C++ runtimes. The layering is deliberate:

```
                       src/c/vkernels/   (C++ kernel library)
                              │
            ┌─────────────────┼─────────────────────┐
            ▼                 ▼                     ▼
   src/c/vkernels/capi/   src/python/           src/rust/
   (extern "C" ABI;       (pybind11 _core OR    (vkernels-sys FFI
    exception translation)  pure-Python fallback)  + safe vkernels crate)
```

- **C ABI** (`src/c/vkernels/capi/`): the stable boundary for non-C++
  consumers and for the Rust FFI. Every C++ exception is caught and folded
  into a `vk_*_status_t` + thread-local last-error string; nothing is ever
  thrown across the ABI. Several kernels (DSA, MoE, pipeline boundary, the
  KV primitives) expose both an always-compiled host entry and a
  device-only entry.
- **Python** (`src/python/vkernels/`): the public API
  (`vkernels.kernels`, `vkernels.comm`, `vkernels.core`, `vkernels.dist`)
  is backend-agnostic. It dispatches to a compiled pybind11 extension
  (`vkernels._core`, built when `VKERNELS_BUILD_PYTHON=ON`) and otherwise
  to a pure-NumPy reference (`_fallback.py`) that mirrors the C++ CPU
  references — so the package is importable and tested with **no build at
  all**. The tests cross-check the two backends bit-for-bit. There's also a
  `vkernels.vllm_experts` drop-in (`CaptureSafeScratch`) for vLLM
  integration on gfx942. See [`python-bindings.md`](python-bindings.md).
- **Rust** (`src/rust/`): `vkernels-sys` links the C++ library through the
  C ABI (built by the `cmake` crate in `build.rs`); the safe `vkernels`
  crate mirrors the Python modules with `Result<T, Error>`. Host-only by
  default; `VKERNELS_RUST_CUDA=ON` for the CUDA path. See
  [`rust-bindings.md`](rust-bindings.md).

---

## 6. `vkl` — "what is implemented in this repo?"

`vkl` is a small Python CLI (the `vkl` console script from `src/python/`)
that answers the newcomer's first question by **scanning the public
headers** under `src/c/vkernels/kernels/` and `src/c/vkernels/comm/` and
pairing each declaration with its `.cpp`/`.cu`/`.hip` implementations. The
list always reflects the sources — no build step, no separate manifest
(`tests/python/test_discovery.py` pins the contract).

```bash
uv run vkl list                 # kernels + comm primitives
uv run vkl list --kernels       # kernels only
uv run vkl info gemm            # details for one entry
```

This is the fastest way to confirm "is X implemented, and where?".

---

## 7. Build, test, and the 100% coverage gate

Tooling is pinned with [`mise`](https://github.com/jdx/mise) (see
`.mise.toml`); Python is managed with `uv`. CUDA and HIP toolkits are
expected at the system level and are **not** pinned.

```bash
mise install                          # one-time: python, uv, cmake, rust, clang-format…
cmake --preset host && cmake --build --preset host && ctest --preset host
```

The `Makefile` is just a shortener over the CMake presets (`make build`,
`make test`, `make coverage`, `make vkl ARGS=list`, `make py-test`,
`make rust-test`).

**The coverage gate is load-bearing.** Host CI
(`.github/workflows/ci.yml`) compiles with `VKERNELS_ENABLE_COVERAGE=ON`,
runs `ctest --preset coverage`, then enforces **100% line coverage on
`src/c/`** via `meta/scripts/coverage.py --min 100`. The CUDA job
(self-hosted, currently opted off with `if: false`) builds the GPU path
and a wheel. This is the mechanism that makes "develop and review on a
laptop with no GPU" real: the host job alone is a meaningful correctness
gate because the CPU reference *is* the oracle.

GPU hosts without dev packages use the `meta/docker/Dockerfile.nv` image
(CUDA + pinned NCCL runtime); profiling uses `Dockerfile.prof` + the
`meta/docker/run-profile.sh` helper (see the `.agents/skills/container-profiling`
skill if you have it).

---

## 8. How to read this codebase (suggested path)

1. **This doc**, then the top-level [`README.md`](../README.md) for the
   build/CLI mechanics and the current status.
2. [`docs/README.md`](README.md) — the full kernel + comm index with the
   math each one performs.
3. Pick **one** Tier-1 kernel and read the trio top to bottom:
   [`src/c/vkernels/kernels/elementwise.{hpp,cpp,cu}`](../src/c/vkernels/kernels/elementwise.hpp)
   to feel the two-implementation model, then
   [`tests/kernels/elementwise/test_elementwise.cpp`](../tests/kernels/elementwise/test_elementwise.cpp).
4. Pick **one** Tier-2 kernel end-to-end. `moe_fused` is the best single
   example because it ties the low-level primitives (`moe.hip` #12–15),
   the aux orchestration (`moe_aux`), the fused kernel itself, *and* the
   distributed split (`dist/dist_moe.cpp`) together. Read
   [`docs/kernels/moe_fused.md`](kernels/moe_fused.md) →
   `src/c/vkernels/kernels/moe_fused.{hpp,cpp,hip}` →
   `tests/kernels/moe/test_moe_fused.cpp`.
5. For the communication story, read
   [`docs/comm-pipeline-boundary.md`](comm-pipeline-boundary.md) — it
   crystallizes the "graph-capturable, no host progress on replay" design
   goal that pervades the whole `comm/` layer.
6. For how a non-C++ runtime consumes this, skim
   [`src/python/vkernels/_backend.py`](../src/python/vkernels/_backend.py)
   (compiled vs fallback) and the safe Rust crate
   [`src/rust/vkernels/src/lib.rs`](../src/rust/vkernels/src/lib.rs).

---

## 9. Status, at a glance

- **Infrastructure**: bootstrapped — build, CUDA/HIP gating, testing, the
  100%-coverage gate, Python and Rust bindings, Docker profiling, CI are
  all in place and passing.
- **Kimi-K3** (issues #21, #29): MLA + KDA layers run on gfx942 and match
  the CPU/torch reference; `K3_DISABLE_KDA=1` is no longer required.
  Includes a bf16 K16-MFMA GEMM for the projection shapes.
- **GLM-5.3-Flash / DeepSeek-V3** (issue #51): DSA (sparse-MLA) and MHC
  (hybrid-attention) kernels serve on gfx942 with `--dsa-prefill-backend
  vkernels --dsa-decode-backend vkernels` and no custom tilelang overlay.
- **MoE**: the low-level gfx942 primitives (#12–15), the MXFP4 aux
  orchestration (#22), the fused grouped GEMM, and the TP/EP/PP
  distributed split (#18) are implemented and validated against the CPU
  oracle, with measured speedups over the xkernels torch-loop.
- **Communication**: ring all-reduce, P2P gather / KV restore / KV donate,
  fused KV gather, cross-node NCCL KV all-gather, RCCL + OFI/CXI, and the
  graph-capturable pipeline boundary (#10) are all present with prepared
  plans and stable C ABIs.

Everything in the "Status" sections of the top-level README maps to a
specific GitHub issue number referenced throughout the docs — when a doc
says "#21" or "#41", that's the design issue the feature closes.

---

## 10. Mental model to take away

> `vkernels` is a **portable, oracle-driven** GPU kernel library. For every
> operation there is a CPU reference you can read, test, and cover to 100%
> on a laptop, and a GPU kernel that is "correct" precisely because it
> matches that reference. The production work is **replacing vendor
> attention/MoE paths that fail on AMD gfx942 (MI300A/MI300X)** with
> hand-tuned HIP kernels, plus the **graph-capturable communication
> primitives** a vLLM-style serving runtime needs to move KV-cache and
> pipeline state without deadlocking. When in doubt about *why* something
> exists, read its doc under `docs/kernels/` or `docs/` — the rationale is
> always written down there.
