#!/usr/bin/env bash
# verifier/gpu/run.sh — GPU-side verification hooks.
#
# The kernels in src/c/vkernels/kernels/*.{cu,hip} are concurrent shared-memory
# programs.  The properties worth proving there are *no data races* and *no
# barrier divergence* -- historically the failure mode behind the gfx942
# GPU-fault case files (docs/kernels/kda.md, docs/kernels/dsa.md).  Two tools
# cover this:
#
#   * GPUVerify  -- static race/barrier analysis of CUDA/OpenCL.  The
#     authoritative check; run it on each kernel entry.  NB: GPUVerify is
#     distributed for x86-64 only and is no longer maintained, so on this
#     aarch64 host it is run under amd64 emulation via gpuverify.sh.  The
#     harnesses under harnesses/ restate the kernel bodies in GPUVerify's
#     dialect (see harnesses/*.cu for why the production TUs cannot be fed
#     directly).
#   * compute-sanitizer (NVIDIA) / rocprof racecheck (AMD) -- dynamic race
#     detectors, used as the sampled complement on a real device.
#
# This runner executes whichever is present and otherwise skips (exit 77).
# It intentionally does not fabricate a pass: with no tool and no GPU, the
# honest result is "not verified here".
#
# compute-sanitizer instruments the CUDA *runtime* through the dynamic
# libcudart, so the test it is pointed at must (a) actually launch device
# kernels and (b) link libcudart shared.  Build one with:
#   PATH=/usr/local/cuda/bin:$PATH cmake --preset cuda \
#     -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -B build/cuda-racecheck -S .
#   cmake --build build/cuda-racecheck --target vkernels_test_cuda_contracts
# then either place it where the search below finds it or set
# VK_CUDA_TEST_BIN to its path.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

skip() {
  echo "SKIP: $1"
  exit 77
}

# --- static: GPUVerify -----------------------------------------------------
# Prefer a native gpuverify (x86-64 hosts); otherwise use the amd64-emulated
# release via gpuverify.sh.  Each harness is run at the block/grid dims of its
# production launcher.  Exit 77 means the tool is genuinely unavailable, in
# which case we fall through to the dynamic check.
gv=""
if command -v gpuverify >/dev/null 2>&1; then
  gv="gpuverify"
elif [[ -x "$here/gpuverify.sh" ]]; then
  gv="$here/gpuverify.sh"
fi

if [[ -n "$gv" ]]; then
  echo "==> GPUVerify: static race/barrier-divergence analysis"
  rc=0
  "$gv" --blockDim=32 --gridDim=4 "$here/harnesses/elementwise.cu" || rc=$?
  if [[ $rc -ne 77 ]]; then
    "$gv" --blockDim=256 --gridDim=1 "$here/harnesses/reduce.cu" || rc=$?
    [[ $rc -ne 0 ]] || { "$gv" --blockDim=16,16,1 --gridDim=2,2,1 \
        "$here/harnesses/gemm.cu" || rc=$?; }
    [[ $rc -ne 0 ]] || { "$gv" --blockDim=256 --gridDim=2 \
        "$here/harnesses/comm_allreduce.cu" || rc=$?; }
    [[ $rc -ne 0 ]] || { "$gv" --blockDim=256 --gridDim=2 \
        "$here/harnesses/comm_peer_copy.cu" || rc=$?; }
    # The kv gather/scatter harnesses carry two kernels each; their Boogie
    # queries are larger, so they get a generous timeout (see the headers
    # for why the grid-stride loops are collapsed to guards).
    [[ $rc -ne 0 ]] || { "$gv" --blockDim=256 --gridDim=1,2 --timeout=800 \
        "$here/harnesses/comm_kv_gather.cu" || rc=$?; }
    [[ $rc -ne 0 ]] || { "$gv" --blockDim=256 --gridDim=1,2 --timeout=800 \
        "$here/harnesses/comm_kv_scatter.cu" || rc=$?; }
    exit $rc
  fi
fi

# --- dynamic: compute-sanitizer (NVIDIA) -----------------------------------
if command -v compute-sanitizer >/dev/null 2>&1; then
  candidates=()
  [[ -n "${VK_CUDA_TEST_BIN:-}" ]] && candidates+=("$VK_CUDA_TEST_BIN")
  for d in "$root"/build/*/tests "$root"/build/tests; do
    candidates+=("$d/vkernels_test_cuda_contracts")
  done

  bin=""
  for c in "${candidates[@]}"; do
    [[ -x "$c" ]] || continue
    # racecheck needs the dynamic runtime; a static-cudart build is silent.
    if ldd "$c" 2>/dev/null | grep -q 'libcudart\.so'; then
      bin="$c"
      break
    fi
  done

  if [[ -z "$bin" ]]; then
    echo "SKIP: compute-sanitizer present, but no shared-libcudart CUDA test found."
    echo "      racecheck instruments the CUDA runtime, so the test must link"
    echo "      libcudart shared and actually launch device kernels:"
    echo "        PATH=/usr/local/cuda/bin:\$PATH cmake --preset cuda \\"
    echo "          -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -B build/cuda-racecheck -S ."
    echo "        cmake --build build/cuda-racecheck --target vkernels_test_cuda_contracts"
    echo "      or set VK_CUDA_TEST_BIN to an already-built test."
    exit 77
  fi

  echo "==> compute-sanitizer racecheck: ${bin#"$root"/}"
  compute-sanitizer --tool racecheck "$bin"
  exit $?
fi

# --- dynamic: rocprof racecheck (AMD) --------------------------------------
if command -v rocprof >/dev/null 2>&1; then
  echo "SKIP: rocprof present but no HIP test binary configured here."
  echo "      Run it against the built HIP tests, e.g. for a HIP build dir:"
  echo "        rocprof --tool racecheck -- ./build/hip/tests/vkernels_test_cuda_contracts"
  exit 77
fi

skip "no GPUVerify (native or emulated via gpuverify.sh) / compute-sanitizer / rocprof found (see verifier/gpu/README.md)."
