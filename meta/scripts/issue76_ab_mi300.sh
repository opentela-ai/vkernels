#!/bin/bash
#SBATCH --job-name=vk76ab
#SBATCH --partition=mi300
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# Issue #76 prefill variant sweep (gfx942 / MI300A).
#
#   VK_MOE_PF_VARIANT: 0 baseline, 1 register-staged pipeline, 2 = 1 + 32-wide
#   gateup N tile, 3 wavefront-specialised gateup, 4 = 2 + 16-wide gateup N
#   tile, 5 = 4 + 32-wide down N tile (the shipped default).
#
# All six variants are built once and swept in one job, so the speedup column
# of the table in docs/kernels/moe_fused.md and
# docs/performance/moe-fused/gfx942.md comes from a single consistent run
# rather than a cross-job comparison.  The bench prints one row per M and the
# script re-emits them as a markdown table at the end.
#
#   SRC=/iopsstor/scratch/cscs/xyao/vk76/src sbatch -A <acct> \
#       meta/scripts/issue76_ab_mi300.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk76ab-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target moe_fused_prefill_bench test_moe_fused_prefill_correct \
  -j 64 2>&1 | grep -E "error|Error|Built target" | tail -8
PF="$B/meta/benchmarks/moe_fused_prefill_bench"
PFC="$B/meta/benchmarks/test_moe_fused_prefill_correct"

for V in 0 1 2 3 4 5; do
  echo; echo "############ VK_MOE_PF_VARIANT=$V ############"
  VK_MOE_PF_VARIANT=$V "$PF" situ 2>&1 | tee "$TMPDIR/pf$V.txt" \
    || echo "(bench exit $?)"
done

# Correctness on the shipped default, and on the eliminated variants so the
# "kept as evidence" claim stays verifiable.
for V in 0 5 1 3; do
  echo; echo "############ prefill correctness, VARIANT=$V ############"
  VK_MOE_PF_VARIANT=$V "$PFC" || echo "(correct exit $?)"
done

echo; echo "############ markdown table (prefill µs per M) ############"
echo "| M | v0 | v1 | v2 | v3 | v4 | v5 | v5 vs v0 |"
echo "|---|---|---|---|---|---|---|---|"
python3 - "$TMPDIR" <<'PY'
import os, re, sys
d = sys.argv[1]
rows = {}
for v in range(6):
    p = os.path.join(d, "pf%d.txt" % v)
    if not os.path.exists(p):
        continue
    for line in open(p):
        # "   128      512 |      403.7      3.990 |      602.5      2.673 |   0.67x"
        m = re.match(r"\s*(\d+)\s+\d+\s*\|\s*([\d.]+)\s+[\d.]+\s*\|\s*([\d.]+)\s", line)
        if m:
            rows.setdefault(int(m.group(1)), {})[v] = float(m.group(3))
for M in sorted(rows):
    r = rows[M]
    def g(v):
        return "%.0f" % r[v] if v in r else "-"
    sp = "**%.2f×**" % (r[0] / r[5]) if 0 in r and 5 in r and r[5] else "-"
    print("| %d | %s | %s | %s | %s | %s | %s | %s |"
          % (M, g(0), g(1), g(2), g(3), g(4), g(5), sp))
PY
echo; echo "===== ALL DONE ====="
