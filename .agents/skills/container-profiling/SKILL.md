---
name: container-profiling
description: Build and use the vkernels Docker/Podman containers for benchmarking and profiling on NVIDIA and AMD GPUs. Use when setting up a profiling session, when you need the exact podman/docker incantation for nsys/ncu/omnitrace/omniperf, when translating between bare-metal and container profiling workflows, or when running CI-compatible profiling in a container.
---

# Container Profiling

The vkernels repo ships two Dockerfiles that produce self-contained build +
profile containers. They bundle the kernel libraries, benchmarks, PyTorch, and
the platform profiling tools so you can go from source checkout to a roofline
annotation in a single command — no host toolchain setup needed.

Load this skill **before** opening `nsys`/`ncu`/`omnitrace`/`omniperf`;
the container already pins the tool versions and driver-compatibility
assumptions, so you spend zero time on install.

## When to use a container vs bare metal

| Situation | Container? |
|---|---|
| Reproducible CI/CD profiling | Yes — pinned image, no host drift |
| Quick one-off profile on a new machine | Yes — one `podman build` |
| You already have ROCm/CUDA + tools installed | No — use the bare-metal skills |
| You need the absolute latest nightly profiler | No — containers pin stable releases |
| Multi-node profiling (MPI + NCCL/RCCL) | Partial — containers help with tooling, networking is host |

## Step 1 — Build the right container

Pick the container matching your GPU vendor:

### NVIDIA (`Dockerfile.nv`)

```bash
# From the vkernels repo root:
podman build -t vkernels:cuda -f meta/docker/Dockerfile.nv .

# Override CUDA architectures for your target GPU:
podman build --build-arg CUDA_ARCHS="80;90" -t vkernels:cuda \
    -f meta/docker/Dockerfile.nv .

# Override profiling-tool versions (check developer.nvidia.com for latest):
podman build \
    --build-arg NSIGHT_SYSTEMS_VER=2025.6 \
    --build-arg NSIGHT_COMPUTE_VER=2025.4 \
    --build-arg CUDA_ARCHS="native" \
    -t vkernels:cuda -f meta/docker/Dockerfile.nv .
```

| `--build-arg` | Default | Purpose |
|---|---|---|
| `CUDA_ARCHS` | `native` | Target SM architectures (`70;80;86;90` etc.) |
| `NSIGHT_SYSTEMS_VER` | `2025.6` | Nsight Systems `.deb` version |
| `NSIGHT_COMPUTE_VER` | `2025.4` | Nsight Compute `.deb` version |

### AMD (`Dockerfile`)

```bash
# From the vkernels repo root:
podman build -t vkernels:hip -f meta/docker/Dockerfile .
```

The AMD container is pinned to ROCm 6.3 (`rocm/dev-ubuntu-22.04:6.3`) and
targets `gfx90a;gfx942` by default. No extra `--build-arg` needed for basic
use.

## Step 2 — Run interactively (no profiling)

No special caps needed for builds, tests, or benchmarks:

```bash
# NVIDIA
podman run --rm --gpus all -it vkernels:cuda bash
# Inside: nvidia-smi, ./build-cuda/benchmarks/moe_bench, python3, etc.

# AMD
podman run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined -it vkernels:hip bash
# Inside: rocminfo, ./build-hip/benchmarks/moe_bench, python3, etc.
```

## Step 3 — Profile with a timeline tool

Both profilers need access to hardware performance counters. Add
`--cap-add SYS_ADMIN` (or `--privileged` if that's the only option in your
environment).

### NVIDIA — Nsight Systems (`nsys`)

```bash
# Capture a timeline for a specific benchmark:
podman run --rm --gpus all --cap-add SYS_ADMIN \
    vkernels:cuda \
    nsys profile --trace=cuda,nvtx,osrt -o /tmp/nsys-report \
    /workspace/vkernels/build-cuda/benchmarks/moe_bench

# Then copy the report out of a fresh (non-running) container:
podman run --rm --gpus all -v $(pwd):/out \
    vkernels:cuda cp /tmp/nsys-report.nsys-rep /out/

# Open on the host:
nsys-ui /tmp/nsys-report.nsys-rep   # or scp to a machine with a GUI

# Stats-only (no GUI needed):
podman run --rm --gpus all --cap-add SYS_ADMIN \
    vkernels:cuda \
    bash -c "nsys profile --stats=true /workspace/vkernels/build-cuda/benchmarks/moe_bench"
```

### AMD — omnitrace

```bash
# Capture a timeline:
podman run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    vkernels:hip \
    omnitrace --trace -- \
    /workspace/vkernels/build-hip/benchmarks/moe_bench

# View on the host:
# Copy omnitrace-output/ out of the container, then:
omnitrace-avail -r omnitrace-output/
```

## Step 4 — Profile a single kernel

### NVIDIA — Nsight Compute (`ncu`)

```bash
# Full analysis of a specific kernel:
podman run --rm --gpus all --cap-add SYS_ADMIN \
    vkernels:cuda \
    ncu --set full --kernel-name regex:my_kernel \
    -o /tmp/ncu-report \
    /workspace/vkernels/build-cuda/benchmarks/moe_bench

# Roofline analysis:
podman run --rm --gpus all --cap-add SYS_ADMIN \
    vkernels:cuda \
    ncu --set roofline -o /tmp/ncu-report \
    /workspace/vkernels/build-cuda/benchmarks/moe_bench

# Single metric query (fast, low overhead):
podman run --rm --gpus all --cap-add SYS_ADMIN \
    vkernels:cuda \
    ncu --metrics sm__pipe_tensor_cycles_active,dram__bytes.sum \
    /workspace/vkernels/build-cuda/benchmarks/moe_bench

# Launch count only (no profiling, just how many times):
podman run --rm --gpus all --cap-add SYS_ADMIN \
    vkernels:cuda \
    ncu --launch-count 100 --launch-skip 10 \
    /workspace/vkernels/build-cuda/benchmarks/moe_bench

# Copy report to host:
podman run --rm --gpus all -v $(pwd):/out \
    vkernels:cuda cp /tmp/ncu-report.ncu-rep /out/
```

### AMD — omniperf

```bash
# Profile a specific kernel:
podman run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    vkernels:hip \
    omniperf profile -n my_kernel_name -- \
    /workspace/vkernels/build-hip/benchmarks/moe_bench

# Generate analysis with roofline:
# (run omniperf analyze in a second interactive session after the profile)
podman run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    -v $(pwd)/workloads:/workloads \
    vkernels:hip \
    omniperf analyze -p workloads/my_kernel_name/ --roof &

# Quick stats via rocprof:
podman run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    vkernels:hip \
    rocprof --stats /workspace/vkernels/build-hip/benchmarks/moe_bench
```

## Step 5 — Mount source for iterative profiling

When you are changing kernel code and want to re-profile without rebuilding the
full image each time, mount the repo as a volume:

```bash
# NVIDIA — bind-mount the vkernels source tree:
podman run --rm --gpus all --cap-add SYS_ADMIN \
    -v $(pwd):/workspace/vkernels:ro \
    -v $(pwd)/build-cuda:/workspace/vkernels/build-cuda \
    vkernels:cuda \
    ncu --set full -o /tmp/ncu-report \
    /workspace/vkernels/build-cuda/benchmarks/moe_bench

# AMD — same pattern:
podman run --rm --device=/dev/kfd --device=/dev/dri \
    --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
    -v $(pwd):/workspace/vkernels \
    vkernels:hip \
    omniperf profile -n target_kernel -- \
    /workspace/vkernels/build-hip/benchmarks/moe_bench
```

> **Note:** The `build-cuda/` (or `build-hip/`) directory is host-built,
> mounted read-write so the container sees the latest binaries. The source
> tree can be `:ro` if you only re-run binaries; make it `:rw` if you want
> to `cmake --build` inside the container.

## Quick-reference: cap requirements

| Operation | Minimum flags |
|---|---|
| Build image | None (build-time only) |
| `nvidia-smi` / `rocminfo` | `--gpus all` or `--device=/dev/kfd --device=/dev/dri` |
| Run benchmarks/tests | GPU devices only |
| `nsys` timeline | `--cap-add SYS_ADMIN` + GPU devices |
| `ncu` kernel analysis | `--cap-add SYS_ADMIN` + GPU devices |
| `omnitrace` timeline | `--cap-add SYS_ADMIN` + `--device=/dev/kfd --device=/dev/dri` + `--security-opt seccomp=unconfined` |
| `omniperf` kernel | `--cap-add SYS_ADMIN` + `--device=/dev/kfd --device=/dev/dri` + `--security-opt seccomp=unconfined` |
| `rocprof` counters | `--cap-add SYS_ADMIN` + `--device=/dev/kfd --device=/dev/dri` + `--security-opt seccomp=unconfined` |

## Integration with the profiling skills

This skill handles the **container machinery**. For the profiling
**methodology** — which metric to read first, how to interpret stall reasons,
what to do with a roofline — load the platform-specific profiling skill:

| Platform | Profiling skill | Benchmarking skill |
|---|---|---|
| NVIDIA | [kernel-profiling](../kernel-profiling/SKILL.md) | [kernel-benchmarking](../kernel-benchmarking/SKILL.md) |
| AMD | [hip-kernel-profiling](../hip-kernel-profiling/SKILL.md) | [hip-kernel-benchmarking](../hip-kernel-benchmarking/SKILL.md) |

Typical workflow:

1. Build the container (**this skill, Step 1**).
2. Run the benchmark with no profiling to get a baseline number (load the
   benchmarking skill, apply its methodology).
3. Classify the kernel as memory- or compute-bound (load the
   efficient-kernels skill).
4. Profile the binding resource (**this skill, Steps 3–4** for the tool
   invocation, then the profiling skill for metric interpretation).
5. Fix, rebuild (inside or outside the container), and re-measure.

## Completion criterion

You can build the correct container for your GPU, launch an interactive shell,
run a benchmark, and capture a profile with the right tool — all with the
`podman` commands above — without installing any profiling tools on the host.
