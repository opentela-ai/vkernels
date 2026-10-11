#!/bin/bash
#SBATCH --job-name=i42-abi-verify
#SBATCH --account=a-infra02
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/capstor/scratch/cscs/xyao/vkernels-i42/logs/%x-%j.out   # override via --output
#
# Issue #42 PART A: verify the serving C ABI at main HEAD (issue-42/vllm-attn-accept).
# Build the HIP preset on a MI300A compute node, then prove with nm -D that
# libvkernels_hip.so exports vk_hip_mla_fwd and vk_hip_kda_delta_rule_fwd
# (+ stream/scratch variants) and libvkernels.so exports the CPU oracles,
# then run the GPU correctness suite incl. the capture-safety test.
set -uo pipefail
export EDF_PATH="/capstor/scratch/cscs/xyao/opentela-cookbook/deployments/llm/beverin/vkernels:${EDF_PATH:-}"
export OTELA_EDF_NAME="vkernels-rocm"
export VKERNELS_DIR="${VKERNELS_DIR:-/capstor/scratch/cscs/xyao/vkernels-i42}"
export BUILD_DIR="$VKERNELS_DIR/build/hip"
mkdir -p "$VKERNELS_DIR/logs"
echo "[$(date -Is)] i42 ABI verify on $SLURM_JOB_NODELIST"
cd "$VKERNELS_DIR" && git log --oneline -1
srun --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_ON_NODE:-192}" --gpus-per-task=4 \
     --environment="$OTELA_EDF_NAME" bash -c "
set -euo pipefail
cd \"$VKERNELS_DIR\"
echo \"[build] cmake configure (hip preset)\"
cmake --preset hip 2>&1 | tail -5
echo \"[build] cmake build\"
cmake --build --preset hip -j\${SLURM_CPUS_ON_NODE:-192} 2>&1 | tail -5
B=\"$BUILD_DIR\"
HIP_SO=\$(find \"\$B\" -name libvkernels_hip.so | head -1)
CPU_SO=\$(find \"\$B\" -name libvkernels.so | head -1)
echo \"[abi] HIP_SO=\$HIP_SO\"
echo \"[abi] CPU_SO=\$CPU_SO\"
echo \"[abi] --- nm -D libvkernels_hip.so (required serving symbols) ---\"
for s in vk_hip_mla_fwd vk_hip_mla_fwd_stream vk_hip_kda_delta_rule_fwd \
         vk_hip_kda_delta_rule_fwd_with_scratch vk_hip_kda_delta_rule_fwd_chunked_with_scratch \
         vk_hip_kda_delta_rule_fwd_with_scratch_stream vk_hip_kda_delta_rule_fwd_chunked_with_scratch_stream; do
  if nm -D \"\$HIP_SO\" | grep -q \" T \$s\$\"; then echo \"  EXPORTED  \$s\"; else echo \"  MISSING   \$s\"; fi
done
echo \"[abi] --- nm -D libvkernels.so (CPU oracle symbols) ---\"
for s in vk_mla_fwd vk_kda_naive_delta_rule_fwd; do
  if nm -D \"\$CPU_SO\" | grep -q \" T \$s\$\"; then echo \"  EXPORTED  \$s\"; else echo \"  MISSING   \$s\"; fi
done
echo \"[tests] GPU correctness suite\"
PASS=0; FAIL=0
for t in test_mla_correct test_kda_correct test_mla_capture test_kda_chunked_fused_gpu; do
  bin=\$(find \"\$B\" -name \"\$t\" -type f -executable | head -1)
  if [ -z \"\$bin\" ]; then echo \"  SKIP \$t (not built)\"; continue; fi
  echo \"  --- \$t ---\"
  if timeout 600 \"\$bin\" > /tmp/\$t.log 2>&1; then echo \"  PASS \$t\"; PASS=\$((PASS+1));
  else echo \"  FAIL \$t (tail:)\"; tail -15 /tmp/\$t.log; FAIL=\$((FAIL+1)); fi
done
echo \"[tests] \$PASS passed, \$FAIL failed\"
[ \"\$FAIL\" -eq 0 ]
"
RC=$?
echo "[$(date -Is)] srun exited code=$RC"
exit $RC
