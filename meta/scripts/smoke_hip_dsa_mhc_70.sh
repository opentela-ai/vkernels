#!/bin/bash
# One-shot beverin MI300A smoke for the remaining #69/#70 evidence that the
# build-once harnesses (#69 dsa_mhc, #70 kda) left uncollected:
#   A. #70 KDA perf (bench_kda.sh) against the kda_bench built by job 629195,
#      bypassing run_kda_bench_mi300.sh's `set -e` abort on the 5 small-shape
#      correctness failures (those are reported separately, not a perf blocker).
#   B. #69 hip_dsa_mhc.py Python wrapper: lib loads via $VKERNELS_LIB, the
#      stream-0 capture refusal fires, and an eager DSA forward runs. The
#      native DSA/MHC correctness was already PASS on this node (job 629196).
set -uo pipefail
cd /users/xyao/vkernels-issue63 || exit 1

echo "=== node: $(hostname)  date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
rocm-smi --showproductname 2>/dev/null | grep -A1 "GPU\[0\]" | head -3
ulimit -c 0

echo; echo "########## A. #70 KDA perf (bench_kda.sh, each (H,S,D) own process) ##########"
bash meta/benchmarks/bench_kda.sh build_kda/meta/benchmarks/kda_bench 2>&1 | head -40
echo "A_rc=${PIPESTATUS[0]}"

echo; echo "########## B. #69 hip_dsa_mhc.py Python wrapper smoke ##########"
export VKERNELS_LIB=/users/xyao/vkernels-issue63/build_dsamhc/src/c/libvkernels_hip.so
export PYTHONPATH=/users/xyao/vkernels-issue63/src/python
export LD_LIBRARY_PATH=/users/xyao/vkernels-issue63/build_dsamhc/src/c:${LD_LIBRARY_PATH:-}
/usr/bin/python3.11 - <<'PYEOF' 2>&1 | tail -25
import torch
import vkernels.hip_dsa_mhc as h

print("torch:", torch.__version__, "dev:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("hip_dsa_mhc.available():", h.available())          # libvkernels_hip.so loads on this node
print("NUM_CU_GFX942:", h.NUM_CU_GFX942)

# (1) Capture refusal (#69): the C ABI launches on legacy stream 0 and must
#     not be captured. _check_device raises inside an active capture region
#     before any kernel launches -- safe, no DSA args needed.
q = torch.ones(1, 1, 4, device="cuda", dtype=torch.bfloat16)
refused = False
try:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        h._check_device(q, "q")                            # MUST raise RuntimeError
except RuntimeError as e:
    refused = True
    print("capture_refused: OK (", str(e).splitlines()[0][:70], ")")
print("capture_refused_ok:", refused)

# (2) Eager DSA forward (#69): a tiny GLM-5.3 shape (dim=kv_lora_rank,
#     tail_dim=0, H=1, S_q=1, S_kv=8, topk=8) vs the CPU-equivalent check
#     that the wrapper launches and returns rc=0 on a non-default stream.
if h.available():
    # Wrapper contract: q (1,S_q,H,dim+tail), kv (1,S_kv,1,dim+tail),
    # indices (1,S_q,1,topk), all bf16/int32 device tensors.
    H, S_q, S_kv, dim, tail_dim, topk = 1, 1, 8, 4, 0, 8
    q  = torch.randn(1, S_q, H, dim + tail_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, S_kv, 1, dim + tail_dim, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(1, S_q, 1, topk, device="cuda", dtype=torch.int32)
    try:
        out, lse = h.dsa_sparse_fwd(q, kv, idx, dim=dim, tail_dim=tail_dim,
                                    topk=topk, return_lse=True)
        torch.cuda.synchronize()
        print("eager_dsa_sparse_fwd: OK  out", tuple(out.shape), out.dtype,
              "lse", tuple(lse.shape), "rc path healthy")
    except Exception as e:
        print("eager_dsa_sparse_fwd: ERROR (", type(e).__name__, str(e).splitlines()[0][:80], ")")
else:
    print("eager_dsa_sparse_fwd: SKIPPED (lib not available on this node)")
print("SMOKE_DONE")
PYEOF
echo "B_rc=$?"

echo; echo "===== SMOKE DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
