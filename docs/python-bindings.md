# Python bindings

The Python package under [`src/python/vkernels/`](../src/python/vkernels/) is the documented
way to drive the kernels and communication primitives in
[`src/c/vkernels/`](../src/c/vkernels/) from Python. It has two layers:

* **Discovery / CLI** — [`vkernels.discovery`](../src/python/vkernels/discovery.py)
  scans the C++/CUDA sources and [`vkernels.cli`](../src/python/vkernels/cli/) backs
  the `vkl` command (`python3 -m vkernels.cli list`). No third-party
  dependencies.
* **Bindings** — [`vkernels.kernels`](../src/python/vkernels/kernels.py),
  [`vkernels.comm`](../src/python/vkernels/comm.py) and
  [`vkernels.core`](../src/python/vkernels/core.py) implement the callable
  interface to the kernels. These require numpy.

This document covers the bindings.

## Two backends, one interface

The public modules dispatch to one of two interchangeable backends:

| Backend | Implementation | When used |
|---|---|---|
| **compiled** | [`vkernels._core`](../src/python/vkernels/_core.cpp) — a pybind11 extension that calls the C++ library directly | whenever the extension can be loaded (see below) |
| **fallback** | [`vkernels._fallback`](../src/python/vkernels/_fallback.py) — pure-Python (numpy) reference mirroring the C++ CPU oracles | when the extension is not built / not loadable |

The active backend is exposed as `vkernels.backend` (`"compiled"` or
`"fallback"`). Both backends are bit-identical for element-wise ops, GEMM
and the reductions (the fallback replicates the C++ float32 operation order),
so tests and user code do not need to care which one is loaded. The compiled
backend is the fast path and the one that exercises the real kernels; the
fallback keeps the package importable and testable on any laptop — the same
"CPU reference always works" philosophy as the C++ side.

The loader ([`vkernels/_backend.py`](../src/python/vkernels/_backend.py)) resolves
the extension in this order:

1. an importable `vkernels._core` (installed copy, or `PYTHONPATH` pointing
   at a build tree), then
2. a freshly built `build/*/python/vkernels/_core*.so` under the repository
   root (the default CMake output location), newest first, then
3. the pure-Python fallback.

## Building the compiled backend

The extension is opt-in, matching the `VKERNELS_BUILD_PYTHON` CMake option:

```sh
cmake --preset python        # host build + bindings
cmake --build --preset python
ctest --preset python        # includes the Python test suite
```

This compiles [`src/python/vkernels/_core.cpp`](../src/python/vkernels/_core.cpp) into
`build/python/python/vkernels/_core.cpython-*.so`. pybind11 is found via
`find_package` or fetched at configure time (pinned release); a CUDA build
(`--preset cuda -DVKERNELS_BUILD_PYTHON=ON`) links the same extension against
the CUDA-enabled library, so the Python API automatically becomes the entry
point to the GPU kernels as well.

The Python test suite runs under CTest as `python_bindings`; point it at an
interpreter that has numpy if the system Python does not:

```sh
cmake --preset python -DVKERNELS_PYTHON_EXECUTABLE=/path/to/venv/bin/python
```

Without a build, everything still works through the fallback:

```sh
python3 -c "import numpy, sys; sys.path.insert(0, 'src/python'); import vkernels; print(vkernels.backend)"  # fallback
```

## Usage

```python
import numpy as np
import vkernels
from vkernels import kernels, comm, core

print("backend:", vkernels.backend)          # "compiled" or "fallback"

# Element-wise kernels -----------------------------------------------------
kernels.add([1.0, 2.0], [3.0, 4.0])          # array([4., 6.], dtype=float32)
kernels.scale([1.0, 2.0], 2.0)               # array([2., 4.], dtype=float32)
kernels.relu([-1.0, 2.0])                    # array([0., 2.], dtype=float32)

# Reductions ----------------------------------------------------------------
kernels.sum([1.0, 2.0, 3.0])                 # 6.0
kernels.max([1.0, 5.0, 3.0])                 # 5.0

# SGEMM: C = alpha * A @ B + beta * C --------------------------------------
A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
B = np.array([[1.0], [1.0]], dtype=np.float32)
kernels.gemm(A, B)                           # array([[3.], [7.]], dtype=float32)

# Ring all-reduce (host simulation) ----------------------------------------
a = np.array([1.0, 2.0], dtype=np.float32)
b = np.array([3.0, 4.0], dtype=np.float32)
comm.ring_allreduce([a, b])                  # [array([4., 6.]), array([4., 6.])]

# Compute/communication overlap ---------------------------------------------
ex = comm.OverlapExecutor()
res = ex.run(4, lambda i: i * 2, lambda i, v: None)   # Result(4, 4)

# P2P run-list gather -------------------------------------------------------
src = np.arange(6, dtype=np.uint8)
dst = np.zeros(6, dtype=np.uint8)
addr = src.__array_interface__["data"][0]
comm.p2p_gather_runs(dst, [addr], [2], [4])  # src[0:4] -> dst[2:6]: [0, 0, 0, 1, 2, 3]

# Streams --------------------------------------------------------------------
s = core.Stream()
s.submit(lambda: None)
s.wait()
```

## API reference

### `vkernels.kernels`

All kernels accept anything convertible to a C-contiguous `float32` numpy
array as input (a copy is made only when the dtype/layout differs) and write
into a caller-provided `out` or a freshly allocated array.

| Function | Semantics | Notes |
|---|---|---|
| `add(a, b, out=None)` | `out = a + b` | `a.size == b.size` |
| `scale(x, alpha, out=None)` | `out = alpha * x` | `alpha` converted to float32 |
| `relu(x, out=None)` | `out = max(x, 0)` | |
| `sum(x) -> float` | float32-accumulated sum | raises on empty input |
| `max(x) -> float` | maximum | raises on empty input |
| `gemm(A, B, alpha=1.0, beta=0.0, out=None)` | `C = alpha*A@B + beta*C` | shapes `(M,K)`, `(K,N)`, `(M,N)` |

Contract violations raise `ValueError`; a badly-typed `out` raises
`TypeError`. `out` must be writable, C-contiguous `float32` and exactly the
right length — never passed by value expecting a silent copy.

### `vkernels.comm`

| Function / class | Semantics |
|---|---|
| `ring_rank(rank, world) -> Topology` | `(rank, world, next, prev)` ring slot |
| `build_ring_topology(world) -> list[Topology]` | one entry per rank |
| `BlockingQueue` | thread-safe queue of float32 chunks (`push/pop/close/closed`) |
| `MockChannel(out, in_)` | in-process channel (`send/recv/closed`) |
| `make_ring_channels(world)` | `world` channels in a ring (`r` sends to `r+1`) |
| `ring_allreduce_rank(local, rank, world, next, prev)` | one rank's all-reduce, `local` summed in place |
| `ring_allreduce(locals) -> list[array]` | all ranks simulated in one process |
| `OverlapExecutor.run(iters, compute, comm) -> Result` | compute on stream A, comm on stream B, per-iteration future |
| `stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)` | validate + stage 1-D runs (`StagedRun1D`) |
| `stage_runs_2d(dst, runs)` | validate + stage 2-D tiles (`StagedRun2D`) |
| `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, *, stream=None)` | single-launch 1-D gather |
| `p2p_gather_runs_2d(dst, runs, *, stream=None)` | single-launch strided-tile gather |
| `memcpy_peer_batch_async(dst, src_ptrs, dst_offsets, lengths, *, stream=None)` | legacy per-run seam (benchmarks) |

`Gather2DRun` is a dataclass with fields `(src, src_stride, dst_offset,
dst_stride, width, height)`; 6-tuples are accepted wherever a run list is
expected. `src_ptrs` are raw byte addresses — e.g.
`arr.__array_interface__["data"][0]` or `ctypes.addressof(...)` — of
peer-accessible memory under CUDA or simply readable memory on the host.
The p2p functions validate the run list up front (capacity, disjoint output
runs, src/dst non-overlap) and raise `ValueError` on violation; a
`num_runs == 0` list is a valid no-op.

When ``stream=None`` the p2p work runs to completion before returning; with a
:class:`~vkernels.core.Stream` the work is enqueued and the caller owns
ordering and completion via `stream.wait()`. The run-metadata arrays are read
before returning, but the *source buffers* must stay alive until the stream
completes.

### `vkernels.core`

* `Device(index=-1)` — `index()`, `set_current()`, `sync()`,
  `supports_peer(other)` (all no-ops on a host build; real device semantics
  under CUDA), equality. `default_device()` returns `Device(-1)`.
* `Stream()` — `submit(task)`, `wait()`, `submitted()`; one worker thread
  per stream, in-order execution within a stream, concurrency across
  streams. A destroyed stream first drains its queue.

## Testing

```sh
uv sync                       # one-time: .venv/ + editable install + deps
uv run pytest                 # full suite (discovery + bindings)
uv run python -m unittest discover -s tests/python -v   # same via unittest
# or, from the CMake build:
ctest --preset python         # vkl_smoke + python_bindings
```

The Python tests live in ``tests/python/`` (``test_discovery.py`` for the
CLI, plus ``test_kernels.py``, ``test_comm.py``, ``test_core.py`` and
``test_backend.py`` for the bindings). The binding tests run against
whichever backend is loaded and, when both are available, cross-check them
bit-for-bit on random float32 data. Under CTest the interpreter can be
overridden with ``-DVKERNELS_PYTHON_EXECUTABLE`` (e.g. a venv that has
numpy when the system Python does not).

## Layout

```
pyproject.toml                # uv-managed Python project (src-layout, numpy dep)
├── src/
│   ├── python/
│   │   ├── CMakeLists.txt    # extension build + Python tests under CTest
│   │   └── vkernels/
│   │       ├── __init__.py   # version, backend flag, lazy submodules
│   │       ├── _backend.py   # extension loader (compiled vs fallback)
│   │       ├── _core.cpp     # pybind11 bindings to src/c (the caller)
│   │       ├── _fallback.py  # pure-Python reference (numpy)
│   │       ├── _types.py     # shared dataclasses (Topology, runs, ...)
│   │       ├── core.py       # public Device/Stream API
│   │       ├── kernels.py    # public elementwise/reduce/gemm API
│   │       ├── comm.py       # public collectives/overlap/p2p API
│   │       ├── discovery.py  # C++/CUDA source scanner (no deps)
│   │       └── cli/          # the `vkl` command
│   └── rust/                 # Rust bindings (workspace: vkernels-sys FFI + vkernels)
└── tests/python/             # unittest suites (CTest, uv, make py-test)
```
