#!/bin/bash
#SBATCH --job-name=vk63
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
# =============================================================================
# run_issue63_torchops_mi300.sh
#   Reusable MI300A (gfx942) harness for the GLM-5.3-Flash vkernels roadmap
#   (GitHub #63 and its subissues #64-#70): the device-native torch_ops test
#   suite plus the projection / roofline / TunableOp benchmarks that back
#   #65, #66, #67, #68 and provide baseline evidence for #69 / #70.
#
# Design goals
#   * Self-bootstrapping on a CSCS beverin mi300 node: no extra modules,
#     no container, no manual setup. `sbatch meta/scripts/run_issue63_*.sh`
#     just works.
#   * Robust: each section is independent. One benchmark failing is recorded
#     and reported; it does NOT abort the rest of the run.
#   * Reproducible: prints the exact torch/triton/ROCm/device fingerprint and
#     writes one JSON/log artifact per section into work/issue63/.
#
# What it runs (each gated and reported separately)
#   1. Environment fingerprint (torch/triton/hip/device).
#   2. torch_ops pytest suite (CPU contract + GPU parity/graph/capture) —
#      tests/python/test_{mhc_projection,qkv_projection,qkv_tuned_blas}.py.
#   3. bench_mhc_projection.py — default BLAS vs tuned BLAS vs Triton mHC
#      projection (issue #66 baseline).
#   4. bench_qkv_projection.py — default/tuned BLAS vs fused Triton QKV
#      (issue #68 baseline + #67 TunableOp lifecycle).
#   5. check_qkv_tuned_blas.py — frozen TunableOp QKV preflight
#      (graph/parity/context) — issue #67 reproducibility gate.
#   6. bench_roofline.py — logical streaming BW, BF16 GEMM, launch floor
#      (the ~0.21 ms/token and ~3.5 TB/s references in #66/#68).
#   7. bench_roofline_counters.py --case {qkv,gate128} — cold dispatch
#      counter probes (instrumented only; NOT performance numbers).
#
# ----------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS — the "Python.h" gotcha (beverin mi300 nodes)
#   The compute nodes ship Python 3.11.13 and a working torch 2.9.1+rocm6.3 /
#   triton 3.5.1 stack in ~/.local, but NO `python3.11-devel` package, so
#   /usr/include/python3.11/Python.h is absent. Triton's AMD backend
#   JIT-compiles a host helper (`hip_utils.c`, see
#   triton/runtime/backends/amd/driver.py) with `gcc -I/usr/include/python3.11`,
#   which therefore fails with `fatal error: Python.h: No such file or
#   directory` and kills every GPU test + Triton benchmark.
#
#   Fix (_bootstrap_pyheaders, below): fetch a python-build-standalone
#   cpython-3.11 x86_64 glibc tarball, extract its include/python3.11/ dir,
#   and export CPATH at it. `CPATH` is a gcc-standard search path consulted
#   AFTER explicit -I dirs, so the missing /usr/include/python3.11 is filled
#   in transparently — no patching Triton, no root packages. The standalone
#   3.11.16 headers are ABI-compatible with the system 3.11.13 runtime
#   (verified: a helper .so built against them loads and the CPython C API
#   works). Reusable on any beverin node with the same /usr/bin/python3.11.
#
# Override knobs (env vars, all optional):
#   SRC              checkout containing src/python + tests + meta (default
#                    ${HOME}/vkernels-issue63)
#   VK63_PYTHON      interpreter (default /usr/bin/python3.11)
#   VK63_TIME        SBATCH --time override, e.g. `sbatch --time=02:00:00 ...`
#   VK63_SECTIONS    space-separated SECTION LABELS to run instead of all
#                    (see the run_section calls at the bottom). Useful for a
#                    fast smoke test, e.g. VK63_SECTIONS='2. torch_ops pytest*'
#                    — any label prefix match (glob) counts.
# =============================================================================

# Catch unset-variable bugs early (job 628925 died on an unbound
# LD_LIBRARY_PATH), but let run_section swallow command failures so a bad
# benchmark cannot abort the whole run.
set -uo pipefail

: "${SRC:=${HOME}/vkernels-issue63}"                 # checkout with src/python/tests/meta
: "${VK63_PYTHON:=/usr/bin/python3.11}"              # system python (torch+triton in ~/.local)
PY="${VK63_PYTHON}"
OUT="${SRC}/work/issue63"                            # artifacts land here
PYINC="${SRC}/.pybuild/python/include/python3.11"   # bootstrap target for Python.h
mkdir -p "${OUT}" "${SRC}/.pybuild"

# --- environment required before anything runs --------------------------------
# ROCm user-space libs for torch + triton host helpers; `:-` keeps an unset
# var from being a hard error under `set -u`.
export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${SRC}/src/python:${PYTHONPATH:-}"
# Triton host-helper (hip_utils.c) needs Python.h; see the gotcha note above.
export CPATH="${PYINC}${CPATH:+:${CPATH}}"
export TMPDIR="${TMPDIR:-/tmp}"

# --- helpers ------------------------------------------------------------------
banner() { printf '\n############### %s ###############\n' "$*"; }

_bootstrap_pyheaders() {
  # Make Python.h available for Triton's AMD host helper. Idempotent.
  if [ -f "${PYINC}/Python.h" ]; then
    echo "pyheaders present: ${PYINC}/Python.h"; return 0
  fi
  local url="https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.11.16%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"
  echo "bootstrapping Python 3.11 headers (python-build-standalone)..."
  local tmp; tmp="$(mktemp -d)"
  curl -fsSL "${url}" -o "${tmp}/py311.tar.gz" || { echo "ERROR: download failed"; rm -rf "${tmp}"; return 1; }
  tar -C "${SRC}/.pybuild" -xzf "${tmp}/py311.tar.gz" || { echo "ERROR: extract failed"; rm -rf "${tmp}"; return 1; }
  rm -rf "${tmp}"
  test -f "${PYINC}/Python.h" || { echo "ERROR: Python.h missing after extract"; return 1; }
  echo "bootstrapped ${PYINC}/Python.h"
}

s_env() {
  echo "node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
  rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -4
  "${PY}" - <<'EOF'
import torch, triton, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "git", torch.version.git_version)
print("hip", torch.version.hip)
print("triton", triton.__version__)
p = torch.cuda.get_device_properties(0)
print("dev0", p.name, "mp", p.multi_processor_count,
      "arch", getattr(p, "gcnArchName", "?"), "mem_MiB", p.total_memory // 1024 // 1024)
assert torch.cuda.is_available()
EOF
}

s_pytest() {
  "${PY}" -m pytest tests/python/test_mhc_projection.py \
                    tests/python/test_qkv_projection.py \
                    tests/python/test_qkv_tuned_blas.py -q -rN 2>&1 | tail -45
}

s_mhc() {
  "${PY}" meta/benchmarks/bench_mhc_projection.py --output "${OUT}/mhc_projection.json" 2>&1 | tail -30
}

s_qkv() {
  "${PY}" meta/benchmarks/bench_qkv_projection.py --output "${OUT}/qkv_projection.json" 2>&1 | tail -30
}

s_tuned() {
  # bench_qkv_projection writes qkv_projection.tunable0.csv (insert_device_ordinal).
  local csv; csv="$(ls -1t "${OUT}"/qkv_projection.tunable*.csv 2>/dev/null | head -1)"
  if [ -z "${csv}" ]; then
    echo "(no qkv_projection.tunable*.csv in ${OUT} — run section 4 first); skipping"
    return 0
  fi
  "${PY}" meta/benchmarks/check_qkv_tuned_blas.py "${csv}" 2>&1 | tail -20
}

s_roofline() {
  "${PY}" meta/benchmarks/bench_roofline.py --output "${OUT}/roofline.json" \
      --sizes-mib 64 512 2048 --gemm-size 4096 --samples 15 2>&1 | tail -20
}

s_counters_qkv()   { "${PY}" meta/benchmarks/bench_roofline_counters.py --case qkv 2>&1 | tail -8; }
s_counters_gate128(){ "${PY}" meta/benchmarks/bench_roofline_counters.py --case gate128 2>&1 | tail -8; }

# run_section NAME FUNC -- run a section in a subshell, record rc, never abort.
# A section runs iff no VK63_SECTIONS filter is set, or one of the
# space-separated glob patterns matches NAME (prefix match is enough).
declare -A SECTION_RC
_wanted() {  # _wanted NAME -> 0 if NAME should run
  local name="$1" pat
  [ -z "${VK63_SECTIONS:-}" ] && return 0
  for pat in ${VK63_SECTIONS}; do case "${name}" in ${pat}*) return 0;; esac; done
  return 1
}
run_section() {
  local name="$1" func="$2"
  if ! _wanted "${name}"; then
    SECTION_RC["${name}"]="SKIP"; echo "SECTION ${name}: rc=SKIP"; return 0
  fi
  banner "${name}"
  if ( "${func}" ); then
    SECTION_RC["${name}"]=0
  else
    SECTION_RC["${name}"]=$?
    echo "!!! section '${name}' failed with rc=${SECTION_RC[${name}]} (continuing)"
  fi
  echo "SECTION ${name}: rc=${SECTION_RC[${name}]}"
}

# --- bootstrap is the one hard gate ------------------------------------------
banner "0. BOOTSTRAP Python 3.11 headers for Triton host helper"
if ! _bootstrap_pyheaders; then
  echo "FATAL: cannot obtain Python.h; aborting (no GPU tests/benchmarks possible)"
  exit 1
fi

cd "${SRC}" || { echo "FATAL: cannot cd to ${SRC}"; exit 1; }

run_section "1. ENV fingerprint"                                s_env
run_section "2. torch_ops pytest (mhc + qkv + tuned_blas)"      s_pytest
run_section "3. BENCH mHC projection (default/tuned BLAS + Triton)" s_mhc
run_section "4. BENCH QKV projection (default/tuned BLAS + fused Triton)" s_qkv
run_section "5. CHECK frozen TunableOp QKV preflight (#67)"     s_tuned
run_section "6. BENCH roofline (BW + BF16 GEMM + launch floor)" s_roofline
run_section "7. COUNTER probe qkv (cold dispatch)"             s_counters_qkv
run_section "7. COUNTER probe gate128 (cold dispatch)"         s_counters_gate128

# --- summary -----------------------------------------------------------------
banner "ARTIFACTS"
ls -la "${OUT}"
banner "SUMMARY"
for k in "1. ENV fingerprint" "2. torch_ops pytest (mhc + qkv + tuned_blas)" \
         "3. BENCH mHC projection (default/tuned BLAS + Triton)" \
         "4. BENCH QKV projection (default/tuned BLAS + fused Triton)" \
         "5. CHECK frozen TunableOp QKV preflight (#67)" \
         "6. BENCH roofline (BW + BF16 GEMM + launch floor)" \
         "7. COUNTER probe qkv (cold dispatch)" \
         "7. COUNTER probe gate128 (cold dispatch)"; do
  printf '  %-58s rc=%s\n' "$k" "${SECTION_RC[${k}]:-NA}"
done
echo "############### DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ###############"
