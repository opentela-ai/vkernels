#!/bin/bash
# ============================================================================
# Issue #60 — stage + build the dsa_kpool tier-1 (C-ABI device) path for the
# bristen A100 sglang serving image.
#
# Two phases; both idempotent.
#
# Phase 1 "stage" — run from a vkernels checkout (any machine with ssh access
# to bristen.cscs.ch). Exports the current checkout (git archive), refreshes
# the deployed vkernels python package, installs the cookbook tier-1
# kpool_fp8_index.py into patches_full, and inserts the VKERNELS_LIB export
# into the deployed sbatch:
#
#   meta/scripts/issue60_kpool_a100_bringup.sh stage \
#       [/path/to/cookbook checkout]
#
# Phase 2 "build" — run INSIDE the serving container on a bristen GPU node
# (same image the serving job uses, so the .so ABI matches the runtime):
#
#   srun -A infra02 -N1 --gpus-per-node=1 --time=0:40:00 \
#       --container-image="$DEPLOY_DIR/cache/enroot/sglang-dev-cu13.sqsh" \
#       --container-mounts=/capstor:/capstor --container-name=vk60build \
#       bash meta/scripts/issue60_kpool_a100_bringup.sh build
#
# After phase 2 the serving sbatch finds $VKERNELS_LIB and the kpool bridge
# dispatch order becomes: (1) C-ABI device kernels, (2) torch-on-GPU port,
# (3) numpy reference. A failed device call falls back to tier 2 per call.
# ============================================================================
set -euo pipefail

DEPLOY_DIR="${DEPLOY_DIR:-/capstor/scratch/cscs/xyao/glm-53-flash-bristen}"
SRC="${SRC:-$DEPLOY_DIR/vkernels-src}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRISTEN="${BRISTEN:-bristen.cscs.ch}"
COOKBOOK_RAW="https://raw.githubusercontent.com/opentela-ai/cookbook/main/deployments/llm/bristen/glm-53-flash"

phase_stage() {  # $1 = optional local cookbook checkout
    local cookbook="${1:-}"
    echo "== [stage] exporting checkout -> $BRISTEN:$SRC"
    ssh "$BRISTEN" "mkdir -p '$SRC'"
    git -C "$REPO_ROOT" archive HEAD | ssh "$BRISTEN" "tar -x -C '$SRC'"

    echo "== [stage] refreshing $DEPLOY_DIR/vkernels-python/vkernels"
    ssh "$BRISTEN" "rm -rf '$DEPLOY_DIR/vkernels-python/vkernels/__pycache__'"
    rsync -a --delete "$REPO_ROOT/src/python/vkernels/" \
        "$BRISTEN:$DEPLOY_DIR/vkernels-python/vkernels/"

    echo "== [stage] installing tier-1 kpool_fp8_index.py into patches_full"
    local patched="$DEPLOY_DIR/patches_full/sglang/srt/layers/attention/dsa/kpool_fp8_index.py"
    local tmp; tmp="$(mktemp)"
    if [ -n "$cookbook" ]; then
        cp "$cookbook/deployments/llm/bristen/glm-53-flash/patched_sources/sglang/srt/layers/attention/dsa/kpool_fp8_index.py" "$tmp"
    else
        curl -fsSL "$COOKBOOK_RAW/patched_sources/sglang/srt/layers/attention/dsa/kpool_fp8_index.py" -o "$tmp"
    fi
    ssh "$BRISTEN" "[ -f '$patched.pre-tier1' ] || cp '$patched' '$patched.pre-tier1'"
    scp -q "$tmp" "$BRISTEN:$patched"
    rm -f "$tmp"

    echo "== [stage] inserting VKERNELS_LIB export into the deployed sbatch"
    ssh "$BRISTEN" bash -s "$DEPLOY_DIR" <<'REMOTE'
    set -euo pipefail
    sb="$1/serve_glm_53_flash_sglang.sbatch"
    if grep -q 'VKERNELS_LIB' "$sb"; then
        echo "   sbatch already carries the VKERNELS_LIB block"
    else
        cp "$sb" "$sb.pre-tier1"
        python3 - "$sb" <<'PY'
import sys
path = sys.argv[1]
block = '''
# Issue #60 tier-1: point the kpool bridge at the C-ABI device kernels.
# Absent library -> the sbatch unsets VKERNELS_LIB and the bridge serves the
# torch-on-GPU tier (the pre-tier1 behavior).
export VKERNELS_LIB="${VKERNELS_LIB:-$VKERNELS_PYTHON/lib/libvkernels_c.so}"
if [ -f "$VKERNELS_LIB" ]; then
  echo "[kpool] device tier: $VKERNELS_LIB"
else
  echo "[kpool] device library not found, serving the torch tier"
  unset VKERNELS_LIB
fi
'''
src = open(path).read()
anchor = 'export PYTHONPATH="${SGLANG_PATCH_DIR}'
i = src.index(anchor)
open(path, "w").write(src[:i] + block.lstrip("\n") + "\n" + src[i:])
print("   VKERNELS_LIB block inserted")
PY
    fi
REMOTE

    echo "== [stage] done"
}

phase_build() {  # runs inside the serving container on a GPU node
    echo "== [build] nvcc / cmake"
    nvcc --version | tail -1
    cmake --version | head -1

    echo "== [build] configure (CUDA-only container)"
    cmake -S "$SRC" -B "$SRC/build-cu13" -DCMAKE_BUILD_TYPE=Release \
        -DVKERNELS_BUILD_HIP=OFF
    echo "== [build] compile vkernels_c + device-vs-oracle gate"
    cmake --build "$SRC/build-cu13" --target vkernels_c test_dsa_kpool_correct \
        -j "$(nproc)"

    echo "== [gate] device-vs-oracle (expect 14/14 PASS, sm_80)"
    "$SRC/build-cu13/tests/test_dsa_kpool_correct"

    echo "== [install] libvkernels_c.so -> $DEPLOY_DIR/vkernels-python/lib/"
    mkdir -p "$DEPLOY_DIR/vkernels-python/lib"
    cp "$SRC/build-cu13/src/c/libvkernels_c.so" "$DEPLOY_DIR/vkernels-python/lib/"
    echo "== [install] exported kpool entry points:"
    nm -D "$DEPLOY_DIR/vkernels-python/lib/libvkernels_c.so" \
        | grep vk_hip_dsa_kpool

    echo "== [probe] ctypes load inside THIS container (serving-runtime ABI)"
    python3 - "$DEPLOY_DIR/vkernels-python/lib/libvkernels_c.so" <<'PY'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
for sym in ("vk_hip_dsa_kpool_assemble_fp8", "vk_hip_dsa_kpool_decode_update_fp8",
            "vk_hip_dsa_kpool_assemble", "vk_hip_dsa_kpool_decode_update"):
    getattr(lib, sym)  # AttributeError if missing
print("ctypes load + 4 entry points OK:", sys.argv[1])
PY
    echo "===== BUILD PHASE DONE"
}

case "${1:-}" in
    stage)  shift; phase_stage "$@" ;;
    build)  phase_build ;;
    *) echo "usage: $0 stage [cookbook-checkout] | build" >&2; exit 2 ;;
esac
