# vkernels CUDA profiling toolchain (aarch64 / GB10 Grace-Blackwell)
#
# Self-contained image with:
#   * nvcc   13.0.48  (from the base image)         — compile sm_121 kernels
#   * ncu    Nsight Compute  (from the base image)   — per-kernel profiling
#   * nsys   Nsight Systems (installed below)        — whole-program timeline
#   * cmake + ninja + g++                             — host-side builds
#
# The host driver (580.x / CUDA 13.0) and the userspace CUDA libs in this image
# share major version 13, so the container's libcuda.so.13 talks to the host
# GPU directly when the device nodes are injected — no `--gpus` needed.
#
# NOTE on `--gpus all`: on this host the NVIDIA Container Toolkit config has
# `no-cgroups = true`, which breaks NVML initialisation inside `--gpus`
# containers ("Failed to initialize NVML: Unknown Error").  We therefore
# inject the device nodes with plain `--device` flags (runc manages those
# device-cgroup entries itself).  See docker/run-profile.sh.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake \
      ninja-build \
      nsight-systems-2025.1.3 \
      python3 \
      pciutils \
    && rm -rf /var/lib/apt/lists/*

# nvcc / ncu already live in /usr/local/cuda/bin in the base image.
ENV PATH=/usr/local/cuda/bin:${PATH}

WORKDIR /work

# Smoke-check the toolchain at build time (no GPU required to list versions).
RUN nvcc --version | tail -1 \
 && ncu  --version | tail -1 \
 && nsys --version | head -1
