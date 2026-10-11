# vkernels — low-level CUDA kernels and communication primitives

`vkernels` is the kernel layer for [opentela](https://github.com/opentela-ai): hand-tuned
CUDA kernels and the communication primitives (collectives, overlap, topology) that sit
beside them. Everything ships with a **CPU reference implementation** so it can be unit-tested
and measured for coverage on machines **without a GPU**, and a CUDA implementation that is
compiled automatically when a CUDA toolkit is present.

> **Documentation in depth:** the [project wiki](https://github.com/opentela-ai/vkernels/wiki)
> distills the engineering knowledge — the roofline method, the kernel optimization ladder,
> occupancy/latency findings, cross-architecture lessons (MI300A vs GB10 vs MI250X), numerics
> and oracle contracts, LDS layout, CUDA-graph capture rules, debugging case files, and the
> per-kernel catalog with gaps to speed of light.

## Layout

```
vkernels/
├── pyproject.toml            # Python project (src/python/vkernels), managed with uv
├── uv.lock                   #   `uv sync` installs the `vkl` CLI + deps
├── src/c/                     # C++/CUDA sources (the kernel library)
│   └── vkernels/
│       ├── util/             # Span, error/status, logging, build annotations
│       ├── core/             # device, stream, allocator abstractions
│       ├── kernels/          # low-level CUDA kernels, each with a CPU oracle
│       │   ├── elementwise/  #   add, scale, relu, …
│       │   ├── reduce/       #   block/grid reduction
│       │   └── gemm/         #   tiled SGEMM
│       └── comm/             # communication optimisation
│           ├── topology.{hpp,cpp}   #   rank/world/peer discovery
│           ├── channel.{hpp,cpp}    #   transport abstraction + mock backend
│           ├── allreduce.{hpp,cpp,cu}#  ring all-reduce
│           ├── overlap.{hpp,cpp}    #   compute/communication overlap
│           └── pipeline_boundary.{hpp,cpp,cu} # graph-capturable PP boundary (#10)
├── tests/                    # C++ unit tests mirroring src/c/ (100% line coverage target)
│   ├── python/               #   Python tests (vkl CLI + bindings)
│   └── third_party/minitest.hpp     # dependency-free test harness
├── meta/                     # build support: cmake, scripts, docker, benchmarks
│   ├── cmake/                #   CMake helpers (CUDA, coverage, sanitizers, testing)
│   ├── scripts/              #   build / test / coverage / format helpers
│   ├── docker/               #   Dockerfiles and profiling toolchain
│   └── benchmarks/           #   micro-benchmarks
├── src/                      # Language bindings
│   ├── python/               #   Python package: `vkl` CLI + kernel bindings
│   │   └── vkernels/
│   │       ├── cli/          #     argparse CLI (`list`, `info`)
│   │       ├── torch_ops/    #     device-op kernels adopted from floe (see below)
│   │       ├── kernels.py    #     add/scale/relu/sum/max/gemm (public API)
│   │       ├── comm.py       #     topology/channels/allreduce/overlap/p2p
│   │       ├── core.py       #     Device/Stream
│   │       ├── _core.cpp     #     optional pybind11 backend (built by CMake)
│   │       └── _fallback.py  #     pure-Python reference (no build needed)
│   └── rust/                 #   Rust bindings (workspace: `vkernels-sys` FFI + `vkernels` safe API)
│       ├── vkernels-sys/     #     unsafe FFI; build.rs links the C++ library via CMake
│       └── vkernels/         #     safe `kernels`/`comm`/`core` API mirroring the Python one
├── src/c/vkernels/capi/       # C ABI shim (extern "C" + exception translation) for non-C++ consumers
├── docs/                     # architecture, kernels, communication, testing
└── .github/workflows/        # host (coverage) + CUDA CI
```

## Two-implementation model

Every kernel and collective in this repo is provided **twice**:

1. **CPU reference** (`*.cpp`) — always compiled, fully unit-tested, and treated as the
   correctness oracle. This is the code that is measured for 100% line coverage on host CI.
2. **CUDA implementation** (`*.cu`) — compiled only when `VKERNELS_BUILD_CUDA=ON` and a
   CUDA toolkit is found. GPU tests compare the CUDA path against the CPU reference.

This means you can develop, review, and get to 100% coverage on any laptop, and run the
GPU path only on machines that have a toolkit and device.

## Kernel catalog

One row per kernel family (details, per-function contracts and benchmarks in
[`docs/README.md`](docs/README.md)). **Verified?** means:

- **formal** — mechanically proved (GPUVerify/ESBMC; see the next section)
- **CI** — CPU/eager oracle + GPU parity test, run in CI (host + cuda jobs)
- **MI300A** — oracle + on-device parity validated on MI300A (the `hip` CI job
  has no runner; records under `docs/performance/`)
- **oracle** — oracle exists, GPU qualification open (experimental, out of dispatch)
- **draft** — design only, unmeasured

| Type | Name | Precision | Supported hardware | Verified? | Notes |
|---|---|---|---|---|---|
| act | `elementwise` (C++) | fp32 | CUDA + CPU oracle | formal | add/scale/relu; race-free all schedules |
| act | `elementwise` (torch_ops) | bf16/fp32 | CUDA (Triton) | CI | HF RMSNorm/RoPE/SiLU fusions, paged KV store |
| act | `silu_mul_clamp` | bf16 | CUDA (Triton) | CI | clamp+clamp+SiLU+mul, one launch |
| comp | `reduce` (sum/max) | fp32 | CUDA + CPU oracle | formal | two-stage grid reduce |
| comp | `gemm` (SGEMM) | fp32 | CUDA + CPU oracle | formal | 16×16 tiled; z3 index-map proof |
| comp | `gemm_bf16` | bf16 (fp32 acc) | HIP gfx942; MI300A/GB10 tile tables | MI300A | K16 MFMA, Kimi-K3 projection shapes |
| comp | `gemm_grouped` | fp8 (masked m-grouped) | CUDA + CPU oracle | CI | DeepSeek `m_grouped_fp8_gemm_nt_masked` contract |
| comp | `mqa_logits` | fp8-e4m3 → fp32 | CUDA + CPU oracle | CI | paged MQA gated top-k logits (kpool stage) |
| comp | `glm_gemv` + projection GEMVs | bf16 | CUDA (Triton) | CI | pinned decode GEMVs; `independent_rows` bitwise row-invariance |
| comp | `glm_dense_fp8_gemv` | bf16 act, fp8-e4m3 w, fp32 scales | CUDA (Triton) | CI | dense block-fp8 decode GEMV, small M |
| comp | `glm_expert_gemv` (+`_fused`) | fp8-e4m3 w, fp32 scales → bf16 | CUDA (Triton) | CI | selected-expert decode GEMV (#64/#65) |
| comp | `glm_fp8_blockwise_gemm` (+`_glm_fp8_sm90_gemm`) | fp8 blockwise | SM90 (CuTe DSL) + Triton | CI | GLM-5.3 grouped/MoE GEMM |
| comp | `fp8_linear` | fp8 act, block scales | CUDA (Triton) | CI | per-token-group quant dense projections |
| comp | `deepgemm_linear` / `deepgemm_moe` | fp8 W8A8 | SM90 (pinned DeepGEMM) | CI | block-128 dense; masked grouped MoE decode |
| comp | `v41_fp8_gemm` / `v41_mxfp4_gemv` | fp8 UE8M0 / MXFP4 | CUDA (Triton) | CI | DeepSeek-V4.1 stacks |
| comp | `dense_fp8_small_batch` | fp8 w → bf16 | CUDA sm ≥ 9 (Triton) | oracle | small-M tensor-core GEMM, out of dispatch |
| comp | `expert_fp8_tensorcore` | fp8 w → bf16 | CUDA sm ≥ 9 (Triton) | oracle | expert projection, out of dispatch |
| comp | `qkv_tuned_blas` | bf16 | CUDA (BLAS) | CI | opt-in pre-tuned serialized-inference QKV |
| attn | `mla` (C++) | bf16 (fp32 acc) | HIP gfx942 | MI300A | absorbed MLA, online softmax, chunked prefill masks |
| attn | `triton_attn` | bf16 (fp32 acc) | CUDA (Triton) | CI | paged single-token decode |
| attn | `tokenspeed_mla` | bf16 | CUDA (Triton, vendored) | CI | MLA prefill/decode portable path |
| attn | `tokenspeed_dsa` / `vllm_sparse_mla` / `sgl_sparse_mla` | bf16 | CUDA (vendored) | CI | sparse-MLA donors (TokenSpeed / vLLM / SGLang, SM90) |
| attn | `v41_sparse_attention` | bf16 | CUDA (Triton) | CI | DSA forward + attention sink |
| attn | `dsa` (C++) | bf16; fp8-e4m3fnuz logits | HIP gfx942 | MI300A | sparse-MLA fwd + top-k transform + kpool cache (#51, #58, #60) |
| attn | DSA Triton tier (`dsa_kpool_*`, `dsa_resolve_rows`, `dsa_tail_scatter`, `dsa_latent_gather`, `dsa_qkva_stacked`) | bf16/fp8 | CUDA (Triton) | CI | indexer metadata, row resolve, gathers/scatters |
| attn | `glm_indexer_topk` / `v41_dsa_indexer` | fp32 | CUDA (Triton) | CI | exact deterministic top-k; V4.1 Lightning scores |
| attn | tokenspeed_amd MHA/MLA/RMHA | bf16 | AMD Gluon CDNA3/4 | MI300A | vendored TokenSpeed-Kernel-AMD (MIT); CDNA3 ports validated on-device |
| ssm | `kda` (C++) | bf16 | HIP gfx942 | MI300A | per-key-dim delta rule; WY chunked (#70); CP state handoff |
| ssm | `glm_kda_decode` / `_fused` / `_packed` | fp32/bf16 | CUDA | CI | decode variants; packed ≈2× the Triton path |
| ssm | `glm_kda_conv_update` / `glm_forget_gate` | bf16 | CUDA (Triton) | CI | conv+SiLU+state roll; fused 2nd projection + gate |
| ssm | `vllm_kda` / `glm_kda_chunk` | bf16 | CUDA (Triton, vendored fla) | CI | chunked prefill |
| ssm | `vllm_kda_amd` | bf16 | HIP gfx942 (vendored vLLM) | MI300A | the GLM-5.3 ROCm path |
| ssm | `gdn_commit` | bf16 | CUDA (Triton) | CI | deferred GDN SSM commit (DFlash2 verify) |
| mhc | `mhc` (C++) | bf16 in → fp32 out | HIP gfx942 | MI300A | pre GEMM+sqrsum and post combine, static LDS (#51) |
| mhc | `glm_mhc_mix` / `mhc_compose` / `mhc_chain` / `glm_mhc_big_fuse` | bf16 | CUDA (Triton) | CI | gate/Sinkhorn/collapse chain fusions |
| mhc | `aiter_mhc_fused` | bf16 | gfx942 (AITER) | vendor | `mhc_fused_post_pre` bridge |
| moe | `glm_moe` (C++ `glm_fp8_block_gemv`) | fp8-e4m3 w, fp32 scales | HIP gfx942 | MI300A | split-K dequant-fused expert GEMV (#64) |
| moe | `moe` primitives #12–#15 | bf16 / fp4 | HIP gfx942 + CPU | MI300A | direct-LDS fill, fp4→bf16, async gate, K16 MFMA |
| moe | `moe_aux` | bf16 → MXFP4 | HIP gfx942 + CPU | MI300A | quant / sort / scatter-reduce orchestration (#22) |
| moe | `moe_fused` | bf16 act, fp4 (E2M1+UE8M0) w | HIP gfx90a + gfx942 | MI300A + MI250X | end-to-end fused MXFP4 MoE; exact vs oracle |
| moe | `dist` (TP/EP/PP) | — | host + HIP stages | CI | MoE weight sharding + orchestration (#18) |
| moe | `glm_moe_grouped` / `glm_moe_decode` | fp8 w, bf16 act | CUDA (Triton) | CI | 2-/3-launch grouped MoE decode |
| moe | `sgl_moe` | fp8_w8a8, block [128,128] | CUDA (Triton, vendored SGLang) | CI | fused MoE + per-token-group quant; per-layout tol gate |
| moe | `moe_combine` / `moe_deterministic_combine` / `moe_ep_remap` | bf16/fp32 | CUDA (Triton) | CI | weighted top-k combine; EP route translation |
| moe | `glm_router` / `glm_router_dense` | bf16/fp32 | CUDA (Triton) | CI | sigmoid top-k router + scoring-GEMV fold |
| quant | `glm_expert_gather_dequant` | e4m3 → bf16/fp16 | CUDA (Triton) | CI | gather + block-dequant |
| quant | `v41_mxfp4_dequant` | E2M1+E8M0 → bf16/fp16 | CUDA (Triton) | CI | microscale dequant |
| spec | `dflash` | bf16 | CUDA (Triton) | CI | block accept, draft prepare, selector walk |
| mem | `state_transfer` | bit-exact (any) | CUDA (Triton) | CI | row pool ↔ fixed-address stage tensors |
| comm | ring `allreduce` | fp32 | CUDA + mock channel/CPU | formal | GPUVerify-checked fused stub; channel transport |
| comm | `overlap` executor | — | CUDA | CI | dual-stream lockstep compute/comm overlap |
| comm | `pipeline_boundary` | bytes | CUDA / NCCL | formal | graph-capturable PP transfer + eager-break (#10) |
| comm | `kv_gather` / `kv_scatter` | bf16/fp16 | CUDA | formal | fused indexed K/V layer gather/scatter (#1, #2) |
| comm | `p2p_gather` (run-list) | bytes | CUDA (UVA) | CI | 1-D/2-D disjoint runs, plan API, adaptive dispatch |
| comm | `p2p_kv_restore` (+plan) | bf16/fp16 | CUDA (UVA) | CI | fused peer gather + indexed scatter; per-layer reuse (#27) |
| comm | `p2p_kv_donate` (+plan) | bf16/fp16 | CUDA (UVA) | CI | fused indexed gather + peer store; adaptive (#36) |
| comm | `cross_node_kv` + `fabric_import` | bf16/fp16 | CUDA / NCCL / fabric (IMEX) | CI | prepared plans over cross-node memory; JSC-HDR measured (#49) |
| comm | `cross_node_kv_allgather` | bf16/fp16 | CUDA / fabric | draft | N−1-edge all-gather plan, unmeasured (#49) |
| comm | RCCL transport + OFI/CXI plugin | bytes | HIP/RCCL (+Slingshot) | CI | graph-capturable RCCL plans; host oracle always built (#19) |
| comm | `sgl_push` | bf16 (in-place SUM) | SM90, TP4 | CI | pinned SGLang 0.5.20 one-shot push |

## Formally verified kernels

Structural properties (data-race freedom for **all** thread/block schedules) are proved
mechanically with GPUVerify over dialect mirrors of the CUDA kernels; the CPU oracle
kernels are proved against their specifications with ESBMC. Full details, harnesses,
and install instructions live in [`verifier/README.md`](verifier/README.md).

| Kernels | Production source | Proved property |
|---|---|---|
| `add_kernel`, `scale_kernel`, `relu_kernel` | `src/c/vkernels/kernels/elementwise.cu` | no data races, all schedules |
| `reduce<false>`, `reduce<true>` (`sum`/`max`) | `src/c/vkernels/kernels/reduce.cu` | no data races, no barrier divergence |
| `gemm_kernel` | `src/c/vkernels/kernels/gemm.cu` | no data races, bounded sizes (M,N,K ≤ 64) |
| `fused_reduce_stub` | `src/c/vkernels/comm/allreduce.cu` | no data races, all schedules |
| `peer_copy_kernel` | `src/c/vkernels/comm/pipeline_boundary.cu` | no data races, all schedules |
| `kv_gather_kernel` (`SlotT` = i32/i64) | `src/c/vkernels/comm/kv_gather.cu` | no data races on the pinned page/slot domain |
| `kv_scatter_kernel` (`SlotT` = i32/i64) | `src/c/vkernels/comm/kv_scatter.cu` | same, via slot-uniqueness contract |

That is 9 of the 20 CUDA kernels in `src/c/vkernels` (12 proof entries — reduce and
gather/scatter split per template instantiation). Also proved: the CPU oracles match
their specs (ESBMC: elementwise, reduce, gemm), the tiled-GEMM index map is equivalent
to the naive traversal (z3), and the fp32 sequential-sum error bound `|fl(Σx) − Σx| ≤
5·2⁻²⁴` (Gappa). Functional correctness of the kernels themselves is covered by the
CTest CUDA-vs-CPU parity suite; the remaining kernels and the HIP families are not yet
formally verified (see the coverage note in `verifier/README.md`).

## Torch operator kernels (`torch_ops`)

`src/python/vkernels/torch_ops/` hosts the device-op kernels that serve
floe's runner paths (GLM-5.3-Flash et al.). They are **adopted from floe**
(the #64/#65 thin-adapter model: vkernels owns the kernel, floe imports it
back through a re-export adapter in `floe/engine/runner/kernels/`),
following the same two-implementation discipline as the C++ kernels — each
module carries a `*_reference` eager oracle, loads torch/triton lazily
(importing the package pulls in neither), and ships contract + GPU-parity
tests under `tests/python/`:

| module | kernel |
|---|---|
| `glm_expert_gemv` | selected-expert E4M3FN block-FP8 decode GEMV |
| `glm_expert_gather_dequant` | gather + explicit E4M3FN block-dequant |
| `glm_fp8_blockwise_gemm` (+ `_glm_fp8_sm90_gemm`) | blockwise-scaled FP8 GEMM / MoE grouped GEMM (Triton + CuTe DSL sm90) |
| `glm_mhc_mix` | mHC gate + Sinkhorn fusion |
| `glm_moe_grouped` | two-launch grouped MoE decode over stacked fp8 experts (gate/up GEMV + clamped swiglu, down GEMV + weighted combine) |
| `sgl_moe` (+ `configs/`) | vendored SGLang Triton fused-MoE (fp8_w8a8, block [128,128]) + per-token-group quant + tuned tile sidecars |
| `moe_combine` | weighted top-k MoE combine (one launch; SGLang `_moe_sum_reduce` design) |
| `silu_mul_clamp` | grouped-MoE activation: clamp+clamp+silu+mul in one launch (`swiglu_fn` seam callable) |
| `mhc_chain` | mHC chain-fuse: previous site's compose tail + pre head in ONE launch (decode rows) |
| `glm_kda_conv_update` | KDA decode conv + SiLU + in-place state roll (SGLang `causal_conv1d_update` design) |
| `glm_kda_decode` | single-token per-dimension-gated KDA decode |
| `glm_projection`, `mhc_projection`, `qkv_projection`, `qkv_tuned_blas` | fused projection GEMVs |
| `gdn_commit` | deferred GDN SSM commit (DFlash2 spec decoding) |
| `triton_attn` | paged single-token decode attention |
| `elementwise` | HF RMSNorm/RoPE/SiLU fusions + paged KV store |

## Development environment (mise)

All language toolchains and build tools are pinned in [`.mise.toml`](.mise.toml) and
managed by [mise](https://github.com/jdx/mise). After installing mise, trust the
config and install the tools:

```bash
mise trust    # one-time: trust the repo-level .mise.toml
mise install  # fetch python 3.12, uv, cmake, rust, clang-format, …
```

mise ensures every developer (and CI) uses the same versions of the tools listed below.
It also provides tasks that mirror the Makefile — run `mise tasks` to see them.

| Tool | Pinned version | Why |
|---|---|---|
| `python` | 3.12 | pyproject.toml requires `>=3.9`; uv manages `.venv/` from this python |
| `uv` | latest | Python package manager (`uv sync`, `uv run`, …) |
| `cmake` | latest | Build system; CMakeLists.txt requires `>=3.18` |
| `rust` | latest | Optional Rust bindings under `src/rust/` (edition 2021) |
| `clang-format` | latest | Code formatter; also provides `clang-tidy` (shipped from LLVM) |
| `ccache` | latest | *Optional* — faster C++ recompilation |
| `ninja` | latest | *Optional* — alternative CMake generator |

CUDA and HIP toolkits are expected at the system level and are **not** pinned in
`.mise.toml`.

## Quick start

```bash
# One-time: install pinned tools
mise install

# Configure + build + test on the host (no GPU required)
cmake --preset host
cmake --build --preset host
ctest --preset host          # or: cmake --build --preset host --target test

# Line coverage (enforces 100% on code under src/c/)
# Or simply: mise run coverage
cmake --preset coverage
cmake --build --preset coverage
ctest --preset coverage
python3 meta/scripts/coverage.py --build-dir build/coverage --source-dir src/c --min 100
```

When a CUDA toolkit is available:

```bash
cmake --preset cuda          # enables VKERNELS_HAS_CUDA, compiles the .cu files
cmake --build --preset cuda
ctest --preset cuda
```

For GPU hosts that should not carry CUDA/NCCL development packages, build the
NVIDIA image instead. It contains the CUDA toolkit plus pinned NCCL runtime and
headers; the host only needs a compatible NVIDIA driver and NVIDIA Container
Toolkit:

```bash
docker build -t vkernels:cuda -f meta/docker/Dockerfile.nv .
docker run --rm --gpus all --ipc=host vkernels:cuda
```

Override `NCCL_PACKAGE_VERSION` at image build time when coordinating a CUDA or
NCCL upgrade. The image build fails if CMake does not detect NCCL.
To validate only the reusable CUDA/NCCL toolchain layer, build with
`--target toolchain`.

## vkl — listing the implemented kernels

`vkl` (Python, under `src/python/`) answers "what is implemented in this
repository?”. It scans the public headers under `src/c/vkernels/kernels/` and
`src/c/vkernels/comm/` and pairs each declaration with its implementations —
a `.cpp` CPU reference and/or a `.cu` CUDA file — so the list always
reflects the sources. No build step is required. The Python project is
managed with [uv](https://docs.astral.sh/uv/) from the root `pyproject.toml`:

```bash
uv sync                      # first time: create .venv, install deps + `vkl`
uv run vkl list              # kernels + comm primitives
uv run vkl list --kernels    # kernels only
uv run vkl list --comm --json   # machine-readable
uv run vkl info gemm         # details for one entry
uv run python -m vkernels.cli list   # equivalent (no console-script needed)
```

Or use the Makefile shortcuts (`make vkl ARGS=list`, `make py-test`,
`make py-sync`). Installing the package puts a `vkl` command on your PATH
(`uv sync` does this in editable mode; plain `pip install -e .` works too).
The repository root is auto-detected and can be overridden with `--root DIR`
or the `VKERNELS_ROOT` environment variable; `vkl --version` reports the
version from `src/c/vkernels/util/version.hpp`. See
`tests/python/test_discovery.py` for the exact discovery contract.

## Python bindings (opt-in)

Beyond the CLI, `src/python/vkernels/` exposes a documented Python interface to the
kernels and communication primitives themselves — `vkernels.kernels`
(add/scale/relu/sum/max/gemm), `vkernels.comm` (topology, channels, ring
all-reduce, compute/communication overlap, the p2p run-list gather) and
`vkernels.core` (Device, Stream). The interface dispatches to a compiled
pybind11 backend (`vkernels._core`, built from `src/python/vkernels/_core.cpp`
when `VKERNELS_BUILD_PYTHON=ON`) that calls the C++ library under `src/c/`,
and falls back to pure-Python reference implementations otherwise — so it
works with or without a build, and the two backends are cross-checked
bit-for-bit by the tests.

```bash
cmake --preset python        # host build + compiled backend + Python tests
cmake --build --preset python
ctest --preset python
uv run pytest                # or: make py-test

uv run python -c "from vkernels import kernels; print(kernels.add([1, 2], [3, 4]))"
```

See [`docs/python-bindings.md`](docs/python-bindings.md) for the full API.

See [`docs/README.md`](docs/README.md) for the full kernel and communication
catalogue, and [`docs/ORIENTATION.md`](docs/ORIENTATION.md) for the design,
build, and testing conventions. See [runtime and compiler contracts](docs/code-quality-remediation.md)
for workspace ownership, backend selection, tuning migration, and device execution.

## Rust bindings (opt-in)

`src/rust/` is the Rust counterpart of the Python bindings: `vkernels-sys` links
the C++ library (built by CMake through the `cmake` crate; host-only by
default, `VKERNELS_RUST_CUDA=ON` for CUDA) via the C ABI in
`src/c/vkernels/capi/`, and `vkernels` wraps it in a safe, `Result`-based API
(`kernels`, `comm`, `core`). Build and test on any machine:

```bash
cargo test --manifest-path src/rust/Cargo.toml   # or: make rust-test
# or via the CMake preset (registers `rust_bindings` with CTest):
cmake --preset rust && cmake --build --preset rust && ctest --preset rust
```

See [`docs/rust-bindings.md`](docs/rust-bindings.md) for the full API.

## Status

Bootstrapped. The infrastructure (build, CUDA gating, testing, coverage, CI) is in place
and the reference implementations are wired up with passing, fully-covered tests.

### Kimi-K3 hybrid attention (issues #21, #29)

- **MLA** — Multi-head Latent Attention forward (absorbed form, online
  softmax), gfx942. See [`docs/kernels/mla.md`](docs/kernels/mla.md).
- **KDA** — the seven Kimi Delta Attention kernels (gated RMSNorm,
  gate chunk-cumsum, the gated delta-rule forward and its intra/inter/
  output sub-kernels, and bit-matrix packing), gfx942. The host reference
  is the cross-checked oracle; the device forward is a correctness-first
  cooperative recurrence. See [`docs/kernels/kda.md`](docs/kernels/kda.md).
- **bf16 GEMM** — tiled K16 bf16 MFMA GEMM for the K3 projection shapes.
  See [`docs/kernels/gemm_bf16.md`](docs/kernels/gemm_bf16.md).

A K3-shaped forward (MLA + KDA layers) runs on gfx942 and matches the
CPU/torch reference; `K3_DISABLE_KDA=1` is no longer required.

### GLM-5.3-Flash / DeepSeek-V3 sparse attention (issue #51)

- **DSA** — DeepseekSparseAttn sparse-MLA forward (absorbed form, online
  softmax, bf16-tolerant vs the CPU oracle). Replaces the tilelang
  `sparse_mla_fwd_decode_partial` (+`_combine`) decode kernels and the
  `_v1`/`_v2` prefill kernels, which GPU-fault on gfx942; the
  GLM-5.3-Flash `tail_dim == 0` shape tilelang cannot compile is a
  first-class, hand-checked case (the rope-tail dot is skipped at runtime).
  See [`docs/kernels/dsa.md`](docs/kernels/dsa.md).
- **MHC** — the two GLM-5.3-Flash / DeepSeek-V3 multi-head hybrid-attention
  pre/post kernels (`mhc_pre_gemm_sqrsum`, `mhc_post`) whose tilelang
  forms JIT-abort on gfx942 (dynamic shared > 64 KB non-optin cap). The
  HIP kernels use static shared memory sized to fit MI300A's non-optin
  cap (no `hipFuncSetAttribute` opt-in, no `hidden_block 256 -> 128`
  patch), bf16-tolerant vs the CPU oracle. See
  [`docs/kernels/mhc.md`](docs/kernels/mhc.md).

GLM-5.3-Flash serves on gfx942 with `--dsa-prefill-backend vkernels
--dsa-decode-backend vkernels` and no custom tilelang overlay.
Measured MI300A performance for the stack (indexer → kpool → sparse
forward, MHC pre/post): [`docs/performance/dsa/gfx942.md`](docs/performance/dsa/gfx942.md),
[`docs/performance/mhc/gfx942.md`](docs/performance/mhc/gfx942.md).
The sparse forward's decode uses the split-key path
(`dsa_sparse_fwd_split` + `dsa_sparse_fwd_split_for`): 3.12 → 0.156 ms
per layer at full topk (20.0x), 0.40 → 0.059 ms at serving topk=256
(6.7x), cross-checked against the fp32 oracle including uneven splits.
