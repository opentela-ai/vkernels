# Rust bindings

The Rust workspace under [`src/rust/`](../src/rust/) is the Rust counterpart of the
Python bindings under [`src/python/vkernels/`](../src/python/vkernels/): it drives the same
kernels and communication primitives from [`src/c/vkernels/`](../src/c/vkernels/)
with an idiomatic, safe API.

It has two layers:

* **`vkernels-sys`** ([`src/rust/vkernels-sys/`](../src/rust/vkernels-sys)) — the
  unsafe FFI layer. Its `build.rs` configures the repository's own CMake
  build (the single source of truth for the library) with the `cmake` crate
  and links the resulting static library. The C ABI it binds to lives in
  [`src/c/vkernels/capi/`](../src/c/vkernels/capi/) — `extern "C"` wrappers
  around the C++ API that fold every C++ exception into a status code plus a
  thread-local message (exceptions cannot cross the ABI). The C shim is
  compiled into the `vkernels` static library itself, so there is nothing
  extra to link.
* **`vkernels`** ([`src/rust/vkernels/`](../src/rust/vkernels)) — the safe,
  idiomatic API built on `vkernels-sys`, mirroring the Python modules
  `vkernels.kernels`, `vkernels.comm` and `vkernels.core` function for
  function. Contract violations (length mismatches, empty inputs, invalid
  topology, out-of-capacity runs, …) surface as
  [`Error`](https://docs.rs/vkernels/latest/vkernels/enum.Error.html)
  (`Result`), exactly like the `ValueError`s of the Python bindings and the
  `std::invalid_argument`s of the C++ library.

## Building and testing

The crate is host-buildable on any machine with a Rust toolchain — the
library is compiled by CMake in the host (CPU-reference) configuration by
default, so no CUDA toolkit is required:

```sh
cargo test --manifest-path src/rust/Cargo.toml    # from the repository root
# or: make rust-test
# or, via the CMake presets (registers `rust_bindings` with CTest):
cmake --preset rust
cmake --build --preset rust
ctest --preset rust
```

With a CUDA toolkit, set `VKERNELS_RUST_CUDA=ON` when building to compile
the CUDA kernels too (the same entry points then drive the GPU path):

```sh
VKERNELS_RUST_CUDA=ON cargo test --manifest-path src/rust/Cargo.toml
```

`vkernels::has_cuda()` reports how the linked library was built.

## Usage

```rust
use vkernels::kernels;
use vkernels::comm;
use vkernels::core::{Device, Stream};

// Element-wise -----------------------------------------------------------
let a = [1.0_f32, 2.0];
let b = [3.0_f32, 4.0];
let mut out = [0.0_f32; 2];
kernels::add(&a, &b, &mut out)?;                 // out == [4.0, 6.0]
kernels::scale(&a, 2.0, &mut out)?;              // out == [2.0, 4.0]
kernels::relu(&[-1.0, 2.0], &mut out)?;          // out == [0.0, 2.0]

// Reductions -------------------------------------------------------------
kernels::sum(&[1.0, 2.0, 3.0])?;                 // 6.0
kernels::max(&[1.0, 5.0, 3.0])?;                 // 5.0

// SGEMM: C = alpha * A @ B + beta * C -------------------------------------
let a = [1.0, 2.0, 3.0, 4.0];                    // 2x2 row-major
let b = [1.0, 1.0];                              // 2x1
let mut c = [0.0_f32; 2];
kernels::gemm(2, 1, 2, 1.0, &a, &b, 0.0, &mut c)?;  // c == [3.0, 7.0]

// Ring all-reduce (host simulation) --------------------------------------
let a = vec![1.0_f32, 2.0];
let b = vec![3.0_f32, 4.0];
let out = comm::ring_allreduce(&[a, b])?;        // two ranks, each [4.0, 6.0]

// Compute/communication overlap -------------------------------------------
let ex = comm::OverlapExecutor::new();
let res = ex.run(4, |i| i as i32 * 2, |_, _| {})?;   // compute_count == comm_count == 4

// P2P run-list gather -----------------------------------------------------
let src: Vec<u8> = (0..6).collect();
let mut dst = vec![0u8; 6];
comm::p2p_gather_runs(&mut dst, &[src.as_ptr() as usize], &[2], &[4], None)?;
assert_eq!(dst, vec![0, 0, 0, 1, 2, 3]);

// Streams ------------------------------------------------------------------
let s = Stream::new();
s.submit(move || println!("on the stream"))?;
s.wait();

# Ok::<(), vkernels::Error>(())
```

## API reference

### `vkernels::kernels`

| Function | Semantics | Notes |
|---|---|---|
| `add(a, b, out) -> Result<()>` | `out = a + b` | `a.len() == b.len() == out.len()` |
| `scale(x, alpha, out) -> Result<()>` | `out = alpha * x` | |
| `relu(x, out) -> Result<()>` | `out = max(x, 0)` | |
| `sum(x) -> Result<f32>` | float32-accumulated sum | `Err` on empty input |
| `max(x) -> Result<f32>` | maximum | `Err` on empty input |
| `gemm(M, N, K, alpha, A, B, beta, C) -> Result<()>` | `C = alpha*A@B + beta*C` | `A` is `M*K`, `B` is `K*N`, `C` is `M*N` |

Buffers are `&[f32]` / `&mut [f32]`; `out` is always written in place (never
silently copied). Contract violations return
`Err(Error::InvalidArgument)`.

### `vkernels::comm`

| Function / type | Semantics |
|---|---|
| `Topology` | `rank`/`world`/`next`/`prev` ring slot |
| `ring_rank(rank, world) -> Result<Topology>` | one ring slot |
| `build_ring_topology(world) -> Result<Vec<Topology>>` | one entry per rank |
| `BlockingQueue` | thread-safe queue of float32 chunks (`push`/`pop`/`close`/`closed`) |
| `MockChannel::new(out, in)` | in-process channel (`send`/`recv`/`closed`) |
| `make_ring_channels(world)` | `world` channels in a ring (`r` sends to `r+1`) |
| `ring_allreduce_rank(local, rank, world, next, prev)` | one rank's all-reduce, `local` summed in place |
| `ring_allreduce(locals) -> Result<Vec<Vec<f32>>>` | all ranks simulated in one process |
| `OverlapExecutor::run(iters, compute, comm) -> Result<OverlapResult>` | compute on stream A, comm on stream B, per-iteration future |
| `stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)` | validate + stage 1-D runs (`StagedRun1D`) |
| `stage_runs_2d(dst, runs)` | validate + stage 2-D tiles (`StagedRun2D`) |
| `p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, stream)` | single-launch 1-D gather |
| `p2p_gather_runs_2d(dst, runs, stream)` | single-launch strided-tile gather |
| `memcpy_peer_batch_async(dst, src_ptrs, dst_offsets, lengths, stream)` | legacy per-run seam (benchmarks) |

`Gather2DRun` / `StagedRun1D` / `StagedRun2D` mirror the C++ structs; run
fields are `usize` byte addresses / offsets, matching the raw-address API of
the Python bindings (e.g. `arr.as_ptr() as usize`). The p2p functions
validate the run list up front (capacity, disjoint output runs, src/dst
non-overlap) and return `Err(Error::InvalidArgument)` on violation; a
`num_runs == 0` list is a valid no-op.

When `stream` is `None` the p2p work runs to completion before returning;
with a `core::Stream` the work is enqueued and the caller owns ordering and
completion via `stream.wait()`.

**Lifetime contract:** with a `stream`, `dst` and every source must stay
alive until `stream.wait()` completes the enqueued copy — the borrow checker
cannot see the asynchronous use, so this is on the caller, exactly as in the
C++ and Python APIs.

### `vkernels::core`

* `Device::new(index)` — `index()`, `set_current()`, `sync()`,
  `supports_peer(&other)` (all no-ops on a host build; real device semantics
  under CUDA), `PartialEq`. `default_device()` returns `Device::new(-1)`.
* `Stream::new()` — `submit(task)`, `wait()`, `submitted()`; one worker
  thread per stream, in-order execution within a stream, concurrency across
  streams. Tasks are `FnOnce() + Send + 'static`; a panicking task is caught
  and does not abort the process. Dropping a stream first runs every
  still-queued task.

## Error model

`vkernels::Error` has four variants mirroring `vkernels::Code`:
`InvalidArgument` (the C++ `VK_EXPECTS` contract checks, like Python's
`ValueError`), `OutOfRange`, `Unsupported` and `Internal` (the `VK_ENSURES`
invariant checks and allocation failures). Every fallible call returns
`Result`; only handle *construction* (`Device::new`, `Stream::new`, ...)
panics, and only on an out-of-memory failure of the C++ side.

## Testing

```sh
cargo test --manifest-path src/rust/Cargo.toml   # unit + integration tests
ctest --preset rust                          # same, via CTest (rust_bindings)
```

The tests live inline in the crate modules (`core.rs`, `kernels.rs`,
`comm.rs`) and mirror `tests/python/` scenario for scenario: contract
violations, cross-thread ring all-reduce, overlap ordering, stream
semantics and the p2p validation rules. The crate builds and tests on any
machine — the CPU-reference path always works, the same philosophy as the
rest of the repository.

## Layout

```
src/rust/                     # Rust workspace
├── Cargo.toml                #   members: vkernels-sys, vkernels
├── vkernels-sys/             #   unsafe FFI (build.rs drives CMake, links src/c)
│   ├── build.rs              #     cmake crate + link lines (CUDA opt-in)
│   └── src/lib.rs            #     extern "C" declarations of the C ABI
└── vkernels/                 #   safe API
    ├── src/{lib,core,kernels,comm}.rs
    └── tests/                #   integration tests (mirror tests/python)
src/c/vkernels/capi/           # C ABI shim compiled into the vkernels library
├── capi.hpp                  #   the C interface (also usable from plain C)
└── capi.cpp                  #   exception -> status-code translation
```
