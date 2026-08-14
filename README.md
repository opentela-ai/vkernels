# vkernels — low-level CUDA kernels and communication primitives

`vkernels` is the kernel layer for [opentela](https://github.com/opentela-ai): hand-tuned
CUDA kernels and the communication primitives (collectives, overlap, topology) that sit
beside them. Everything ships with a **CPU reference implementation** so it can be unit-tested
and measured for coverage on machines **without a GPU**, and a CUDA implementation that is
compiled automatically when a CUDA toolkit is present.

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
│           └── overlap.{hpp,cpp}    #   compute/communication overlap
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

See [`docs/architecture.md`](docs/architecture.md), [`docs/testing.md`](docs/testing.md),
and [`docs/communication.md`](docs/communication.md) for the design and conventions.

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
