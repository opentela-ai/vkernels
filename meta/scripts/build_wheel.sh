#!/usr/bin/env bash
# build_wheel.sh — build a platform wheel for the vkernels compiled extension.
#
# Usage:
#   ./meta/scripts/build_wheel.sh                    # host-only, current python
#   ./meta/scripts/build_wheel.sh --cuda             # with CUDA kernels
#   ./meta/scripts/build_wheel.sh --hip              # with AMD ROCm/HIP kernels (gfx942)
#   VKERNELS_PYTHON=python3.11 ./meta/scripts/build_wheel.sh
#   ./meta/scripts/build_wheel.sh --output-dir /tmp/wheels
#
# What it does:
#   1. Builds the pybind11 extension via CMake (preset "python", "cuda-python", or "hip-python")
#   2. Copies the .so into the source tree temporarily
#   3. Builds a wheel with `python -m build`
#   4. Removes the temp .so
#   5. The wheel in dist/ contains the compiled extension
#
# The wheel filename encodes the platform + Python version (e.g.
# vkernels_cli-0.1.0-cp312-cp312-linux_x86_64.whl).  pip install
# picks the right one; the fallback backend is used when none matches.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ------------------------------------------------------------------
# Options
# ------------------------------------------------------------------
WITH_CUDA=0
WITH_HIP=0
OUTPUT_DIR=""
PYTHON="${VKERNELS_PYTHON:-python3}"
EXTRA_ARGS=""    # extra cmake flags (e.g. -DVKERNELS_BUILD_TESTS=OFF)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)   WITH_CUDA=1 ;;
    --hip)    WITH_HIP=1 ;;
    --output-dir) OUTPUT_DIR="$2"; shift ;;
    -o)       OUTPUT_DIR="$2"; shift ;;
    --python) PYTHON="$2"; shift ;;
    *)        echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

if [[ "$WITH_CUDA" -eq 1 && "$WITH_HIP" -eq 1 ]]; then
  echo "ERROR: --cuda and --hip are mutually exclusive (pick one toolkit)."
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/dist}"

# ------------------------------------------------------------------
# Check prerequisites
# ------------------------------------------------------------------
command -v cmake >/dev/null 2>&1 || { echo "cmake is required"; exit 1; }
$PYTHON -c "import numpy" 2>/dev/null || { echo "numpy is required: pip install numpy"; exit 1; }

# Ensure build tool is available (uses uv if present, else pip).
# We use `--no-isolation` so setuptools sees the .so we just copied into
# the source tree; without it, `build` creates an isolated venv and
# copies the source, missing our temporary .so.
if command -v uv >/dev/null 2>&1; then
  # Install build + setuptools into the venv (--no-isolation needs them).
  uv sync --only-group dev 2>/dev/null || uv sync
  uv pip install build setuptools 2>/dev/null || true
  BUILD_CMD="uv run python -m build --no-isolation"
else
  BUILD_CMD="$PYTHON -m build --no-isolation"
  $PYTHON -m pip install -q build setuptools 2>/dev/null || true
fi

# ------------------------------------------------------------------
# 1. Build the compiled extension via CMake
# ------------------------------------------------------------------
if [[ "$WITH_CUDA" -eq 1 ]]; then
  PRESET="cuda-python"
  PRESET_ARG="--preset cuda-python"
elif [[ "$WITH_HIP" -eq 1 ]]; then
  PRESET="hip-python"
  PRESET_ARG="--preset hip-python"
  # The hip-python preset requires ROCm (hipcc) on PATH.
  if ! command -v hipcc >/dev/null 2>&1; then
    echo "ERROR: --hip requires hipcc (ROCm HIP compiler) on PATH."
    echo "Install ROCm or use the Dockerfile: meta/docker/Dockerfile"
    exit 1
  fi
else
  PRESET="python"
  PRESET_ARG="--preset python"
  # The python preset has VKERNELS_BUILD_TESTS=ON by default.  Override to
  # OFF for wheel builds — tests are not shipped and may pull in extra
  # compilation units that aren't needed.
  EXTRA_ARGS="-DVKERNELS_BUILD_TESTS=OFF"
fi

echo "=== Building extension (preset=$PRESET) ==="
if [[ -n "$PRESET_ARG" ]]; then
  cmake $PRESET_ARG $EXTRA_ARGS
  cmake --build --preset "$PRESET"
else
  cmake --build "build/$PRESET"
fi

# Locate the built .so.
SO_FILE=$(find "build/$PRESET/python/vkernels" -maxdepth 1 -name '_core*.so' 2>/dev/null | head -1)
if [[ -z "$SO_FILE" ]]; then
  echo "ERROR: could not find _core*.so in build/$PRESET/python/vkernels/"
  echo "Make sure the CMake build succeeded and VKERNELS_BUILD_PYTHON=ON."
  exit 1
fi
echo "  built: $SO_FILE"

# ------------------------------------------------------------------
# 2. Copy .so into the source package temporarily
# ------------------------------------------------------------------
PKG_DIR="$REPO_ROOT/src/python/vkernels"
SO_BASENAME="$(basename "$SO_FILE")"
TEMP_SO="$PKG_DIR/$SO_BASENAME"

# Remove any stale .so left from a previous aborted build.
rm -f "$PKG_DIR"/_core*.so
cp "$SO_FILE" "$TEMP_SO"
echo "  copied to: $TEMP_SO"

# ------------------------------------------------------------------
# 3. Build the wheel
# ------------------------------------------------------------------
echo "=== Building wheel ==="
mkdir -p "$OUTPUT_DIR"

cleanup() {
  echo "=== Cleaning up temp .so ==="
  rm -f "$PKG_DIR"/_core*.so
}
trap cleanup EXIT

$BUILD_CMD --wheel --outdir "$OUTPUT_DIR"

# ------------------------------------------------------------------
# 4. Fix platform tag — the wheel contains a native .so but setuptools
#    doesn't know about it (the extension is pre-built by CMake and
#    included as package data).  Without this, the wheel stays
#    `py3-none-any` and pip may install it on incompatible platforms.
# ------------------------------------------------------------------
WHEEL_FILE=$(ls -1t "$OUTPUT_DIR"/vkernels_cli-*.whl 2>/dev/null | head -1)
if [[ -n "$WHEEL_FILE" ]]; then
  # Derive the correct platform tag.
  PY_VER="$($PYTHON -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
  # Map uname + toolkit to platform tag.
  ARCH=$(uname -m)
  if [[ "$ARCH" == "x86_64" ]]; then
    if [[ "$WITH_HIP" -eq 1 ]]; then
      PLAT_TAG="manylinux_2_17_x86_64.manylinux2014_x86_64_rocm"
    else
      PLAT_TAG="manylinux_2_17_x86_64.manylinux2014_x86_64"
    fi
  elif [[ "$ARCH" == "aarch64" ]]; then
    PLAT_TAG="manylinux_2_17_aarch64.manylinux2014_aarch64"
  else
    PLAT_TAG="linux_$ARCH"
  fi

  # Rename: py3-none-any → {py_ver}-none-{plat}
  NEW_FILE=$(echo "$WHEEL_FILE" | sed "s/py3-none-any/${PY_VER}-none-${PLAT_TAG}/")
  mv "$WHEEL_FILE" "$NEW_FILE"

  # Also patch the WHEEL metadata inside the zip so the Tag field matches.
  $PYTHON -c "
import zipfile, os, sys
wheel = '$NEW_FILE'
tmp = wheel + '.tmp'
with zipfile.ZipFile(wheel, 'r') as zin:
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith('.dist-info/WHEEL'):
                text = data.decode().replace('Root-Is-Purelib: true', 'Root-Is-Purelib: false')
                text = text.replace('Tag: py3-none-any', 'Tag: ${PY_VER}-none-${PLAT_TAG}')
                data = text.encode()
            zout.writestr(item, data)
os.replace(tmp, wheel)
print('  updated WHEEL metadata')
"
  echo "  platform-tagged: $(basename "$NEW_FILE")"
fi

echo ""
echo "=== Done ==="
ls -lh "$OUTPUT_DIR"/vkernels_cli-*.whl 2>/dev/null || ls -lh "$OUTPUT_DIR"
