#!/bin/bash
#SBATCH --job-name=vk76ab90
#SBATCH --partition=mi200
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
# Issue #76 prefill variant sweep (gfx90a / MI250X) — the gfx90a half of the
# "the win is not gfx942-specific" claim in
# docs/performance/moe-fused/gfx90a.md.
#
# Sweeps only the two variants the gfx90a claim needs -- baseline (0) and the
# shipped default (5) -- on one build, plus the prefill correctness oracle on
# those two and on the retained-as-evidence variant 3.  The full six-variant
# sweep lives in issue76_ab_mi300.sh.
#
#   SRC=/iopsstor/scratch/cscs/xyao/vk76/src sbatch -A <acct> \
#       meta/scripts/issue76_ab_mi200.sh
set -uo pipefail
: "${SRC:=$HOME/vkernels}"
export TMPDIR=/tmp/vk76ab90-$$
mkdir -p "$TMPDIR"; B="$TMPDIR/build"
trap 'rm -rf "$TMPDIR"' EXIT

echo "=== node: $(hostname) job=$SLURM_JOB_ID ==="
echo "src hip md5: $(md5sum "$SRC/src/c/vkernels/kernels/moe_fused.hip" | cut -c1-8)"
cmake -S "$SRC" -B "$B" -DVKERNELS_BUILD_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a -DCMAKE_BUILD_TYPE=Release \
  -DVKERNELS_BUILD_BENCHMARKS=ON >/dev/null 2>&1
cmake --build "$B" --target moe_fused_prefill_bench test_moe_fused_prefill_correct \
  -j 64 2>&1 | grep -E "error|Error|Built target" | tail -8
PF="$B/meta/benchmarks/moe_fused_prefill_bench"
PFC="$B/meta/benchmarks/test_moe_fused_prefill_correct"

for V in 0 5; do
  echo; echo "############ VK_MOE_PF_VARIANT=$V ############"
  VK_MOE_PF_VARIANT=$V "$PF" situ 2>&1 | tee "$TMPDIR/pf$V.txt" \
    || echo "(bench exit $?)"
done

for V in 0 5 3; do
  echo; echo "############ prefill correctness, VARIANT=$V ############"
  VK_MOE_PF_VARIANT=$V "$PFC" || echo "(correct exit $?)"
done

echo; echo "############ markdown table (prefill µs per M, gfx90a) ############"
echo "| M | v0 | v5 | v5 vs v0 |"
echo "|---|---|---|---|"
python3 - "$TMPDIR" <<'PY'
import os, re, sys
d = sys.argv[1]
rows = {}
for v in (0, 5):
    p = os.path.join(d, "pf%d.txt" % v)
    if not os.path.exists(p):
        continue
    for line in open(p):
        m = re.match(r"\s*(\d+)\s+\d+\s*\|\s*([\d.]+)\s+[\d.]+\s*\|\s*([\d.]+)\s", line)
        if m:
            rows.setdefault(int(m.group(1)), {})[v] = float(m.group(3))
for M in sorted(rows):
    r = rows[M]
    sp = "**%.2f×**" % (r[0] / r[5]) if 0 in r and 5 in r and r[5] else "-"
    print("| %d | %.0f | %.0f | %s |"
          % (M, r.get(0, 0), r.get(5, 0), sp))
PY
echo; echo "===== ALL DONE ====="
