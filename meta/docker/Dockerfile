# meta/docker/Dockerfile
#
# Build container for vkernels + xkernels with ROCm HIP support.
# Includes PyTorch for running xkernels Triton benchmarks side-by-side.
#
# Build (from vkernels repo root):
#   podman build -t vkernels:hip -f meta/docker/Dockerfile .
#
# Run tests + both benchmarks:
#   podman run --rm --device=/dev/kfd --device=/dev/dri \
#       --security-opt seccomp=unconfined vkernels:hip
#
# Run interactively:
#   podman run --device=/dev/kfd --device=/dev/dri \
#       --security-opt seccomp=unconfined -it vkernels:hip bash

FROM rocm/dev-ubuntu-22.04:6.3

# ------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build python3.11 python3.11-venv python3.11-dev \
    python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up python3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install PyTorch for ROCm
RUN python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/rocm6.2.4

# ------------------------------------------------------------------
# vkernels: build library + benchmarks + tests
# ------------------------------------------------------------------
WORKDIR /workspace/vkernels
COPY . .

RUN rm -rf build-hip && \
    cmake -B build-hip -G Ninja \
    -DCMAKE_HIP_ARCHITECTURES="gfx90a;gfx942" \
    -DVKERNELS_BUILD_HIP=ON \
    -DVKERNELS_BUILD_TESTS=ON \
    -DVKERNELS_BUILD_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build-hip -j$(nproc)

# ------------------------------------------------------------------
# xkernels: copy source and install
# ------------------------------------------------------------------
COPY xkernels-src /workspace/xkernels
RUN python3 -m pip install --no-cache-dir /workspace/xkernels

# Verify both stacks
RUN python3 -c "import torch; print('torch', torch.__version__, 'rocm=', torch.version.hip)" \
    && python3 -c "import xkernels; print('xkernels OK')"

CMD ["bash", "-c", "\
echo '=== vkernels tests ===' && \
cd /workspace/vkernels/build-hip && ctest --output-on-failure && \
echo && echo '=== vkernels moe benchmark ===' && \
./benchmarks/moe_bench && \
echo && echo '=== xkernels moe benchmark ===' && \
cd /workspace/xkernels && \
python3 benchmarks/bench_moe_mxfp4.py --regime decode \
"]
