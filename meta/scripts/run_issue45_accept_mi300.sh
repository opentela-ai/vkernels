#!/bin/bash
# Issue #45 -- K3/gfx942 attention acceptance RUNBOOK (tooling only).
#
# Executes each acceptance item of vkernels#45 with machine-readable
# evidence.  This script NEVER launches the multi-node K3 acceptance
# campaign itself (that needs 6 nodes, real 96-shard weights and >1h --
# partition limit is 1h; it belongs to the owner, see
# meta/benchmarks/artifacts/k3-attn-accept/README.md for the campaign
# sequence).  It runs the per-item tools against an ALREADY-RUNNING serve
# (started by the owner's serve_kimi_k3_otela_beverin.sbatch fork) and
# against local rocprof captures / bench JSONs.
#
# Subcommands:
#   selftest   -- offline validation of every tool (no GPU, no serve) plus,
#                 on a mi300 node, a REAL rocprof round-trip: a tiny HIP
#                 kernel is profiled and the capture is asserted clean, then
#                 poisoned with a fake aiter::mla kernel and asserted to
#                 FAIL.  ~2 min, 1 GPU.  THE on-cluster smoke test.
#   recall     -- AC2: run KDA_RECALL_PROBE against a live serve.
#   bench      -- AC4: compare measured bench JSON vs the 638514 baseline.
#   ac3        -- assert a rocprof capture has zero AITER/Triton attn kernels.
#   gen        -- inspect gen_correctness report(s): coherent vs smoke-only.
#   campaign   -- print the owner-only campaign sequence; refuses to run.
#
# Environment (serve subcommands): SERVE_URL (default http://127.0.0.1:8080),
# MODEL (default SwissAI-Research/moonshot/kimi-k3-rocm), OUT_DIR.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ART="$HERE/../benchmarks/artifacts/k3-attn-accept"
SERVE_URL="${SERVE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-SwissAI-Research/moonshot/kimi-k3-rocm}"
OUT_DIR="${OUT_DIR:-${ART}/evidence-\$(date -u +%Y%m%dT%H%M%SZ)}"
BASELINE_BENCH="$ART/benchmark_638514.json"
BASELINE_GEN="$ART/gen_correctness_638514.json"

PY=python3
rc=0

case "${1:-selftest}" in
  selftest)
    echo "=== [1/2] offline selftests (stdlib only) ==="
    for t in issue45_kda_recall_probe issue45_rocprof_attention_assert \
             issue45_bench_compare issue45_gen_inspect; do
      "$PY" "$HERE/$t.py" --selftest || rc=1
    done
    echo "=== [2/2] on-cluster rocprof round-trip (1 GPU, ~2 min) ==="
    if ! command -v hipcc >/dev/null 2>&1; then
      echo "hipcc not on PATH -> skipping the real-capture round-trip"
    elif ! command -v rocprof >/dev/null 2>&1; then
      echo "rocprof not on PATH -> skipping the real-capture round-trip"
    else
      # keep artifacts next to the checkout for post-mortem; rocprof v2/v3
      # naming differs, so print everything it produces
      T="i45-smoke-artifacts-$$"
      rm -rf "$T"; mkdir -p "$T"
      cat > "$T/tiny.hip" <<'EOF'
#include <hip/hip_runtime.h>
__global__ void vk_smoke_probe_kernel(float* x) { x[threadIdx.x] = threadIdx.x * 2.0f; }
int main() {
  float* d = nullptr;
  if (hipMalloc(&d, 256) != hipSuccess) return 2;
  hipLaunchKernelGGL(vk_smoke_probe_kernel, dim3(1), dim3(64), 0, 0, d);
  if (hipDeviceSynchronize() != hipSuccess) return 3;
  float h[64];
  if (hipMemcpy(h, d, 256, hipMemcpyDeviceToHost) != hipSuccess) return 4;
  hipFree(d);
  return (h[3] == 6.0f) ? 0 : 5;
}
EOF
      hipcc -O2 -o "$T/tiny" "$T/tiny.hip" 2>&1 | tail -2
      "$T/tiny"; echo "tiny-kernel rc=$?"
      rocprof --version 2>&1 | head -2
      # rocprof exits non-zero even on successful collection
      rocprof --stats -o "$T/rp_clean" "$T/tiny" >"$T/rocprof.log" 2>&1 || true
      tail -5 "$T/rocprof.log"
      echo "--- artifacts produced:"; ls -la "$T" | grep -v tiny.hip
      CSV=$(ls "$T"/rp_clean*.csv "$T"/*.csv 2>/dev/null | head -1)
      if [ -n "${CSV:-}" ]; then
        echo "--- clean capture: $CSV"
        "$PY" "$HERE/issue45_rocprof_attention_assert.py" "$CSV" --json "$T/ac3_clean.json" \
          && echo "AC3 smoke: clean capture PASS" || rc=1
        # poison it: the historical gfx950-only offender must FAIL
        { head -1 "$CSV"; echo '1,"aiter::mla_decode_mla_gluon",100'; } > "$T/rp_poison.csv"
        if "$PY" "$HERE/issue45_rocprof_attention_assert.py" "$T/rp_poison.csv" >/dev/null 2>&1; then
          echo "AC3 smoke: POISONED CAPTURE WRONGLY PASSED"; rc=1
        else
          echo "AC3 smoke: poisoned capture correctly FAILS"
        fi
      else
        echo "no rocprof CSV produced -- check rocprof version"; rc=1
      fi
    fi
    ;;

  recall)  # AC2 item
    mkdir -p "$OUT_DIR"
    exec "$PY" "$HERE/issue45_kda_recall_probe.py" "$SERVE_URL" "$MODEL" \
      "$OUT_DIR/kda_recall_probe.json"
    ;;

  gen)  # AC1/AC2 inspection of one or more gen_correctness reports
    shift
    mkdir -p "$OUT_DIR"
    exec "$PY" "$HERE/issue45_gen_inspect.py" "$@" --json "$OUT_DIR/gen_inspect.json"
    ;;

  ac3)  # AC3 on provided rocprof capture(s)
    shift
    mkdir -p "$OUT_DIR"
    exec "$PY" "$HERE/issue45_rocprof_attention_assert.py" "$@" \
      --expect-vk --json "$OUT_DIR/ac3_assert.json"
    ;;

  bench)  # AC4
    if [ $# -lt 2 ]; then
      echo "usage: $0 bench BASELINE.json MEASURED.json [--threshold 1.05]" >&2
      exit 2
    fi
    shift
    mkdir -p "$OUT_DIR"
    exec "$PY" "$HERE/issue45_bench_compare.py" "$@" --json "$OUT_DIR/ac4_compare.json"
    ;;

  campaign)
    echo "The 6-node K3 acceptance campaign is OWNER-ONLY:" >&2
    echo "multi-node (TP8xPP3), real 96-shard weights (~77 min Lustre load)," >&2
    echo ">1h wall -- exceeds the mi300 partition limit.  Sequence:" >&2
    grep -n -A40 "Owner acceptance campaign runbook" "$ART/README.md" 2>/dev/null || true
    echo "Full text: $ART/README.md" >&2
    exit 2
    ;;

  *)
    echo "usage: $0 {selftest|recall|gen|ac3|bench|campaign}" >&2
    exit 2
    ;;
esac
exit "$rc"
