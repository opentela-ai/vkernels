# vkernels — low-level CUDA kernels and communication primitives

`vkernels` is the kernel layer for [opentela](https://github.com/opentela-ai): hand-tuned
CUDA kernels and the communication primitives (collectives, overlap, topology) that sit
beside them. Everything ships with a **CPU reference implementation** so it can be unit-tested
and measured for coverage on machines **without a GPU**, and a CUDA implementation that is
compiled automatically when a CUDA toolkit is present.

## Layout

```
vkernels/
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
├── tests/                    # unit tests mirroring csrc/ (100% line coverage target)
│   └── third_party/minitest.hpp     # dependency-free test harness
├── cmake/                    # CMake helpers (CUDA, coverage, sanitizers, testing)
├── scripts/                  # build / test / coverage / format helpers
├── src/                      # Python bindings (opt-in, off by default)
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

See [`docs/architecture.md`](docs/architecture.md), [`docs/testing.md`](docs/testing.md),
and [`docs/communication.md`](docs/communication.md) for the design and conventions.

## Status

Bootstrapped. The infrastructure (build, CUDA gating, testing, coverage, CI) is in place
and the reference implementations are wired up with passing, fully-covered tests.
