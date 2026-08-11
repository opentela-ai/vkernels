# vkernels — low-level CUDA kernels and communication primitives

`vkernels` is the kernel layer for [opentela](https://github.com/opentela-ai): hand-tuned
CUDA kernels and the communication primitives (collectives, overlap, topology) that sit
beside them. Everything ships with a **CPU reference implementation** so it can be unit-tested
and measured for coverage on machines **without a GPU**, and a CUDA implementation that is
compiled automatically when a CUDA toolkit is present.

## Layout

```
vkernels/
├── pyproject.toml            # Python project (src/vkernels), managed with uv
├── uv.lock                   #   `uv sync` installs the `vkl` CLI + deps
├── csrc/                     # C++/CUDA sources (the kernel library)
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
├── tests/                    # C++ unit tests mirroring csrc/ (100% line coverage target)
│   ├── python/               #   Python tests (vkl CLI + bindings)
│   └── third_party/minitest.hpp     # dependency-free test harness
├── cmake/                    # CMake helpers (CUDA, coverage, sanitizers, testing)
├── scripts/                  # build / test / coverage / format helpers
├── src/                      # Python package: `vkl` CLI + kernel bindings
│   └── vkernels/
│       ├── cli/              #   argparse CLI (`list`, `info`)
│       ├── kernels.py        #   add/scale/relu/sum/max/gemm (public API)
│       ├── comm.py           #   topology/channels/allreduce/overlap/p2p
│       ├── core.py           #   Device/Stream
│       ├── _core.cpp         #   optional pybind11 backend (built by CMake)
│       └── _fallback.py      #   pure-Python reference (no build needed)
├── rust/                     # Rust bindings (workspace: `vkernels-sys` FFI + `vkernels` safe API)
│   ├── vkernels-sys/         #   unsafe FFI; build.rs links the C++ library via CMake
│   └── vkernels/             #   safe `kernels`/`comm`/`core` API mirroring the Python one
├── csrc/vkernels/capi/       # C ABI shim (extern "C" + exception translation) for non-C++ consumers
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

## Quick start

```bash
# Configure + build + test on the host (no GPU required)
cmake --preset host
cmake --build --preset host
ctest --preset host          # or: cmake --build --preset host --target test

# Line coverage (enforces 100% on code under csrc/)
cmake --preset coverage
cmake --build --preset coverage
ctest --preset coverage
python3 scripts/coverage.py --build-dir build/coverage --source-dir csrc --min 100
```

When a CUDA toolkit is available:

```bash
cmake --preset cuda          # enables VKERNELS_HAS_CUDA, compiles the .cu files
cmake --build --preset cuda
ctest --preset cuda
```

## vkl — listing the implemented kernels

`vkl` (Python, under `src/`) answers “what is implemented in this
repository?”. It scans the public headers under `csrc/vkernels/kernels/` and
`csrc/vkernels/comm/` and pairs each declaration with its implementations —
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
version from `csrc/vkernels/util/version.hpp`. See
`tests/python/test_discovery.py` for the exact discovery contract.

## Python bindings (opt-in)

Beyond the CLI, `src/vkernels/` exposes a documented Python interface to the
kernels and communication primitives themselves — `vkernels.kernels`
(add/scale/relu/sum/max/gemm), `vkernels.comm` (topology, channels, ring
all-reduce, compute/communication overlap, the p2p run-list gather) and
`vkernels.core` (Device, Stream). The interface dispatches to a compiled
pybind11 backend (`vkernels._core`, built from `src/vkernels/_core.cpp`
when `VKERNELS_BUILD_PYTHON=ON`) that calls the C++ library under `csrc/`,
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

`rust/` is the Rust counterpart of the Python bindings: `vkernels-sys` links
the C++ library (built by CMake through the `cmake` crate; host-only by
default, `VKERNELS_RUST_CUDA=ON` for CUDA) via the C ABI in
`csrc/vkernels/capi/`, and `vkernels` wraps it in a safe, `Result`-based API
(`kernels`, `comm`, `core`). Build and test on any machine:

```bash
cargo test --manifest-path rust/Cargo.toml   # or: make rust-test
# or via the CMake preset (registers `rust_bindings` with CTest):
cmake --preset rust && cmake --build --preset rust && ctest --preset rust
```

See [`docs/rust-bindings.md`](docs/rust-bindings.md) for the full API.

## Status

Bootstrapped. The infrastructure (build, CUDA gating, testing, coverage, CI) is in place
and the reference implementations are wired up with passing, fully-covered tests.
