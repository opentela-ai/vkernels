#!/bin/bash
#SBATCH --job-name=i42-gpu-tests
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels-i42/logs/%x-%j.out   # override via --output
set -uo pipefail
export EDF_PATH="/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/vkernels:${EDF_PATH:-}"
export OTELA_EDF_NAME="vkernels-rocm"
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels-i42}"
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-192}" --gpus-per-task=4 \
     --environment="$OTELA_EDF_NAME" bash -c "
set -uo pipefail
cd \"$VKERNELS_DIR/build/hip\"
echo \"[nm] libvkernels.so oracle check (in-container nm)\"
nm -D src/c/libvkernels.so | grep -E \" T vk_mla_fwd| T vk_kda_naive_delta_rule_fwd\" || echo MISSING
echo \"[tests] GPU correctness suite (issue #42/#45 serving ABI)\"
PASS=0; FAIL=0
for t in tests/vkernels_test_mla tests/vkernels_test_kda tests/vkernels_test_kda_k3_chunked tests/vkernels_test_kda_chunked_fused_gpu tests/vkernels_test_capi; do
  echo \"  --- \$t ---\"
  if timeout 900 ./\$t > /tmp/vk_i42_\$(basename \$t).log 2>&1; then
    echo \"  PASS \$(basename \$t)\"; PASS=\$((PASS+1))
  else
    echo \"  FAIL \$(basename \$t) (tail:)\"; tail -20 /tmp/vk_i42_\$(basename \$t).log; FAIL=\$((FAIL+1))
  fi
done
echo \"[tests] \$PASS passed, \$FAIL failed\"
"
echo "[$(date -Is)] srun exited code=$?"
