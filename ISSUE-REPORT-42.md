# ISSUE-REPORT — lane issue-42/vllm-attn-accept (issues #42 + #45)

Date: 2026-10-11 · Worktree: `/local/home/xiayao/Documents/code/serving-sys/vk-wt/issue-42` · Branch: `issue-42/vllm-attn-accept`

## What was done

### PART A — C ABI verification (issue #42 scope item 4): COMPLETE, no code needed

Reconciliation with main showed the ABI work already merged before this lane
started: `0472a06` (initial `vk_hip_*` C ABI), `3594f8c` (KDA chunked WY over
C ABI), `7e6b5a4` (stream-safe ABI), `806acb4` (capture-safe
`vk_hip_mla_fwd_stream`). Both required symbols are declared in
`src/c/vkernels/capi/hip_capi.hpp` and defined in `hip_capi.cpp`. The lane
therefore *verified* rather than implemented:

- **Local gates:** `cmake --preset host && cmake --build --preset host &&
  ctest --preset host` → **47/47 pass** (2 GPU tests disabled by design on host).
- **On-cluster build + export proof** (beverin mi300, MI300A/gfx942, fresh
  clone of this branch at main HEAD `100398f`, `vkernels-rocm` EDF / ROCm 7.2):
  - **Job 676514** (nid003032): `cmake --preset hip` builds clean; `nm -D
    libvkernels_hip.so` shows all seven serving entries EXPORTED:
    `vk_hip_mla_fwd`, `vk_hip_mla_fwd_stream`, `vk_hip_kda_delta_rule_fwd`,
    `vk_hip_kda_delta_rule_fwd_with_scratch`,
    `vk_hip_kda_delta_rule_fwd_chunked_with_scratch`,
    `vk_hip_kda_delta_rule_fwd_with_scratch_stream`,
    `vk_hip_kda_delta_rule_fwd_chunked_with_scratch_stream`.
  - **Job 676516**: in-container `nm -D libvkernels.so` confirms the CPU
    oracles `vk_mla_fwd` (T 0x24000) and `vk_kda_naive_delta_rule_fwd`
    (T 0x24440); GPU correctness suite **5/5 PASS** — `vkernels_test_mla`,
    `vkernels_test_kda`, `vkernels_test_kda_k3_chunked`,
    `vkernels_test_kda_chunked_fused_gpu`, `vkernels_test_capi`.
  - Repro scripts checked in: `meta/scripts/issue42_abi_verify_beverin.sbatch`,
    `meta/scripts/issue42_gpu_tests_beverin.sbatch`.

### PART B — #45 on-cluster acceptance: staged, not executed (see "Why blocked")

The full VKERNELS real-weight acceptance serve was **not** run by this lane.
What this lane did instead:

- Verified the exact launch path from the cluster artifacts: the recipe
  `serve_kimi_k3_otela_beverin.sbatch` (beverin cookbook deployment at
  `e1e8aad`) propagates `--export` vars into `engine.sh`
  (`export VKERNELS_MLA="${VKERNELS_MLA:-0}"` etc.); `VKERNELS_*_VALIDATE`
  defaults to ON inside `vkernels_attn.py`; `VKERNELS_MOE=0` is required to
  bypass the NaN-generating `VKERNELS_MXFP4_BF16` MoE backend (sitecustomize
  gate, cookbook `e8c3163`); real-weight serves need
  `DISTRIBUTED_TIMEOUT_SECONDS=21600` (638605 post-mortem).
- Recorded the launch command + AC1–AC4 tooling pointers
  (`meta/scripts/run_issue45_accept_mi300.sh gen|recall|ac3|bench`) in the
  status addendum of `meta/benchmarks/artifacts/k3-attn-accept/README.md`.

## Evidence

| Item | Value |
|---|---|
| Lane commits | `1ec932c` docs(serving-abi) on `issue-42/vllm-attn-accept` (pushed) |
| Slurm jobs (this lane) | 676514 (build+nm), 676516 (oracle nm + GPU suite), both COMPLETED on mi300 |
| GPU suite result | 5/5 PASS (MLA, KDA, K3-chunked, chunked-fused, C API) |
| PR | https://github.com/opentela-ai/vkernels/pull/170 (draft, base main) |
| Doc updates | `docs/serving-c-abi.md` (new "On-cluster export verification" section), `meta/benchmarks/artifacts/k3-attn-accept/README.md` (status addendum) |

## Why PART B was not executed by this lane

1. The repo's own owner runbook (`k3-attn-accept/README.md`) marks the
   6-node real-weight campaign **owner-only** ("beyond the mi300 partition
   limit / >1 h wall — not launched by this tooling").
2. Budget: ~3 h lane total vs a serve that needs ~77 min weight load + boot +
   probe/bench, with 2-of-2 prior attempts dying on infra (639740 boot-phase
   NCCL RECV timeout at ~85% load; 639143/639260 NaN logits from the MoE
   backend) — a repeat failure would have produced no lane deliverables.
3. The beverin cookbook checkout has uncommitted modifications to the K3
   recipe (working tree reverted to an older recipe revision) that appear to
   belong to another session; launching from that state risks another lane's
   WIP.

## What remains (for the owner, exact commands)

```bash
ssh beverin
cd /capstor/scratch/cscs/xyao/opentela-cookbook
sbatch --export=ALL,VKERNELS_MLA=1,VKERNELS_MLA_FORCE=1,VKERNELS_MLA_VALIDATE=1,VKERNELS_KDA=1,VKERNELS_KDA_VALIDATE=1,VKERNELS_MOE=0,ENFORCE_EAGER=1,DISTRIBUTED_TIMEOUT_SECONDS=21600 \
  deployments/llm/beverin/kimi-k3-vllm/serve_kimi_k3_otela_beverin.sbatch
# then, per artifacts/k3-attn-accept/README.md:
#   AC1: GEN_PROBE=1 (recipe runs it) -> run_issue45_accept_mi300.sh gen <report>
#   AC2: run_issue45_accept_mi300.sh recall  (PASS w/ KDA on; FAIL on 638514 baseline)
#   AC3: rocprof --stats capture -> run_issue45_accept_mi300.sh ac3 results.csv --expect-vk
#   AC4: benchmark.py both arms -> run_issue45_accept_mi300.sh bench <base> <measured>
#   AC5: flip recipe default only after 1-4 pass
```

AC2's KDA-contrast anchor (baseline FAIL, `recall_baseline_638514.json`) and
the AC4 baseline (`benchmark_638514.json`: C=1 p50 66.7 s) are already checked in.
