#!/bin/bash
# Issue #78 — submit the GLM-5.3-Flash/K3 serving deployment with the
# vkernels on-device moe_align_block_size shim (VkernelFusedExperts).
#
# Reconstructed after run 640362 (job FAILED: the shim's libvkernels_hip.so
# is linked against soname libamdhip64.so.6 [login-node ROCm 6.3], while the
# serving container's torch stack runs HIP 7.2 — /opt/rocm-7.2.3; the .so.6
# soname did not resolve inside the container -> PP1_TP0 died at the first
# MoE layer of profile_run).
#
# Fix strategy (this script):
#   1. fork the deploy_v2 5-file bundle into the issue scratch (shared
#      deployment untouched);
#   2. append a sitecustomize block that registers VkernelFusedExperts for
#      the AITER_MXFP4_BF16 backend (the stock bundle lost this block when
#      the previous worker's edits were wiped);
#   3. create a compat symlink libamdhip64.so.6 -> /opt/rocm-7.2.3/.../so.7
#      (the loader reuses the object torch already loaded — one HIP runtime);
#   4. submit with VKERNELS_DIR pointing at the vk-issue-78 build + the
#      compat dir prepended to LD_LIBRARY_PATH.
#
# Usage (from beverin):
#   bash serve_issue78_k3.sh           # submit + print job id
set -euo pipefail

SCRATCH=/capstor/scratch/cscs/xyao/vk-issue-78
DEPLOY=/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin
BUNDLE="$DEPLOY/deploy_v2"
FORK="$SCRATCH/deploy_i78"

mkdir -p "$FORK" "$SCRATCH/compat"
# 1. bundle fork (shared deployment stays stock)
for f in serve_kimi_k3_otela_beverin.sbatch kimi-k3-vllm.toml k3_patch.py \
         sitecustomize.py gen_correctness.py; do
  cp -u "$BUNDLE/$f" "$FORK/$f"
done

# 2. compat symlink: .so.6 soname resolves to the container's HIP 7.2 lib
ln -sfn /opt/rocm-7.2.3/lib/libamdhip64.so.7 "$SCRATCH/compat/libamdhip64.so.6"

# 3. registration block (idempotent)
if ! grep -q "VkernelFusedExperts registered" "$FORK/sitecustomize.py"; then
  cat >> "$FORK/sitecustomize.py" <<'VKREG'

# --- issue-78: vkernels on-device MoE shim (VkernelFusedExperts) ---
import os as _os
if _os.environ.get("VKERNELS_MXFP4_BF16", "1") != "0":
    try:
        from vllm.model_executor.layers.quantization import mxfp4 as _k3mxfp4
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            Mxfp4MoeBackend as _Backend,
        )
        from vkernels_experts import VkernelFusedExperts as _VKE

        _orig_b2kc = _k3mxfp4.backend_to_kernel_cls

        def _b2kc_with_vke(backend, *_a, **_kw):
            if backend == _Backend.AITER_MXFP4_BF16:
                return [_VKE]
            return _orig_b2kc(backend, *_a, **_kw)

        _k3mxfp4.backend_to_kernel_cls = _b2kc_with_vke
        print("[sitecustomize] VkernelFusedExperts registered (VKERNELS_MXFP4_BF16)", flush=True)
    except Exception as _e:
        print(f"[sitecustomize] vkernels MoE shim not registered: {_e!r}", flush=True)
VKREG
fi

# 4. submit (env set inside the sbatch copy — submit-line --export does not
#    survive into the container)
cd "$FORK"
sbatch serve_kimi_k3_otela_beverin.sbatch
echo "monitor: tail -f $DEPLOY/logs/k3-vllm-beverin-<JOBID>.out"
echo "markers: 'VkernelFusedExperts registered' (shim active),"
echo "         'OSError.*libamdhip64' (compat failure), gen_probe results at end"
