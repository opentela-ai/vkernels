#!/usr/bin/env bash
# meta/scripts/hip_syntax_check.sh — host-side front-end check of every HIP TU.
#
# The `hip` CI job only runs where a ROCm runner exists, and the HIP-on-CUDA
# nvcc build covers a handful of .hip files (gfx942 builtins do not compile
# under nvcc). This check runs clang's HIP front end in -fsyntax-only mode for
# the host AND the gfx942 device side of every library .hip translation unit,
# so redefinitions, missing declarations and type errors fail CI without a GPU
# or a device library (-nogpulib).
#
# Requirements: a clang++ with HIP support (LLVM >= 17) and the HIP runtime
# headers (<hip/hip_runtime.h>). Resolution order:
#   HIP_CLANG     clang++ to use       (default: $ROCM_PATH/llvm/bin/clang++,
#                                       then clang++ on PATH)
#   HIP_PATH      HIP install prefix   (default: $ROCM_PATH or /opt/rocm);
#                 passed as --hip-path, must contain include/hip/hip_runtime.h
#   HIP_ARCH      offload arch         (default: gfx942)
#
# Usage: meta/scripts/hip_syntax_check.sh [file.hip ...]
#   With no arguments, checks every git-tracked .hip TU (library, tests and
#   meta/benchmarks harnesses) except the files in `skip_re` below.
#   Exit status: 0 all clean, 1 at least one TU failed, 77 toolchain absent.
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

rocm="${ROCM_PATH:-/opt/rocm}"
hip_path="${HIP_PATH:-$rocm}"
arch="${HIP_ARCH:-gfx942}"
clang="${HIP_CLANG:-}"
if [ -z "$clang" ]; then
  if [ -x "$rocm/llvm/bin/clang++" ]; then
    clang="$rocm/llvm/bin/clang++"
  else
    clang="$(command -v clang++ || true)"
  fi
fi
if [ -z "$clang" ] || [ ! -f "$hip_path/include/hip/hip_runtime.h" ]; then
  echo "hip_syntax_check: SKIP (clang++='${clang:-none}', HIP headers under '$hip_path' not found)" >&2
  exit 77
fi

if [ "$#" -gt 0 ]; then
  files=("$@")
else
  # Not standalone TUs:
  #   moe_device.hip            helper header #included by other TUs
  #   kda_4barrier/kda_pardot   A/B drop-in replacements for kda.hip
  #                             (meta/scripts/ab_kda_pardot.sh swaps them in);
  #                             they collide with kda.hpp when compiled alone
  skip_re='/moe_device\.hip$|/kda_4barrier\.hip$|/kda_pardot\.hip$'
  mapfile -t files < <(git ls-files '*.hip' | grep -Ev "$skip_re")
fi

# rccl.hip needs <rccl/rccl.h>; check it only when RCCL headers are present.
rccl_inc=""
for d in "$rocm/include" "$hip_path/include" /usr/include; do
  if [ -f "$d/rccl/rccl.h" ]; then rccl_inc="$d"; break; fi
done

host_quirks=()
if [ "$(uname -m)" = aarch64 ]; then
  # On AArch64 hosts the device pass sees glibc's SVE vector-math typedefs
  # (bits/math-vector.h), which the amdgcn target does not know. Neutralise
  # them for the device side only; x86-64 hosts (CI) need nothing here.
  host_quirks=(-Xarch_device -D__SVFloat32_t=float -Xarch_device -D__SVFloat64_t=double
               -Xarch_device -D__SVBool_t=int)
fi

check_one() {  # $1 = file, $2 = label, rest = extra flags
  local f="$1" label="$2"; shift 2
  if "$clang" -x hip -std=c++17 -fsyntax-only --offload-arch="$arch" -nogpulib \
       --hip-path="$hip_path" -Wno-unused-command-line-argument \
       "${host_quirks[@]}" -DVKERNELS_HAS_HIP=1 -Isrc/c "$@" "$f"; then
    echo "ok    $f$label"
  else
    echo "FAIL  $f$label"
    fail=1
  fi
}

fail=0
for f in "${files[@]}"; do
  case "$f" in
    */comm/rccl.hip)
      if [ -z "$rccl_inc" ]; then echo "skip  $f (no rccl/rccl.h)"; continue; fi
      extra=(-DVKERNELS_HAS_RCCL=1 -isystem "$rccl_inc") ;;
    *) extra=() ;;
  esac
  # Benchmark builds (VKERNELS_BUILD_BENCHMARKS) compile the library with
  # VKERNELS_PROFILING_HOOKS=1: check harnesses in that mode, and library
  # TUs that carry hooks in both modes.
  case "$f" in
    meta/*) check_one "$f" "" -DVKERNELS_PROFILING_HOOKS=1 "${extra[@]}" ;;
    *)
      check_one "$f" "" "${extra[@]}"
      if grep -q VKERNELS_PROFILING_HOOKS "$f"; then
        check_one "$f" " [profiling hooks]" -DVKERNELS_PROFILING_HOOKS=1 "${extra[@]}"
      fi ;;
  esac
done
# Host-only TUs compiled solely in HIP builds (plain C++ against the HIP
# headers): the HIP C ABI shim.
# Adding -isystem /usr/include reorders it ahead of libstdc++'s directory and
# breaks <cstdlib>'s #include_next, so skip it when HIP lives in /usr.
hip_inc=()
[ "$(cd "$hip_path/include" && pwd -P)" != /usr/include ] && hip_inc=(-isystem "$hip_path/include")
if [ "$#" -eq 0 ]; then
  for f in src/c/vkernels/capi/hip_capi.cpp; do
    if "$clang" -x c++ -std=c++17 -fsyntax-only -D__HIP_PLATFORM_AMD__=1 \
         "${hip_inc[@]}" -DVKERNELS_HAS_HIP=1 -Isrc/c "$f"; then
      echo "ok    $f"
    else
      echo "FAIL  $f"
      fail=1
    fi
  done
fi
exit "$fail"
