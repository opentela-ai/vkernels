# Issue #45 — on-cluster acceptance runbook: VKERNELS MLA + KDA on K3/gfx942

Operational runbook for the remaining on-cluster acceptance of
[opentela-ai/vkernels#45](https://github.com/opentela-ai/vkernels/issues/45)
(parent: #42). Everything runnable off-cluster has been run and is recorded
under "Off-cluster state" below; every on-cluster item below is **one command
away** given the pre-flight state.

Written 2026-10-17 from a clean `origin/main` worktree (1cdb507) and a fresh
clone of `opentela-ai/cookbook` (main = 5d3f316; issue-pinned wiring commit
006cd58 verified present). Baseline-arm evidence already in-tree:
`meta/benchmarks/artifacts/k3-attn-accept/` (job 638514, commit 98c2860).

**Do not close #42 or #45 from this document.** Acceptance requires the
measured runs on Beverin MI300A recorded in the issue.

---

## 0. Off-cluster state (verified, no gfx942 needed)

Local hardware is one NVIDIA GB10 — no gfx942, no second node. All GPU work
below was necessarily skipped; nothing on-cluster is claimed as passed.

### vkernels repo (origin/main, 1cdb507)

- **C ABI contracts present and defined**
  - `vk_hip_mla_fwd` — `src/c/vkernels/capi/hip_capi.hpp:178`,
    `hip_capi.cpp:96` (legacy, default stream, caller owns sync).
  - `vk_hip_mla_fwd_stream` — `hip_capi.hpp:200` (capture-safe, #69
    convention; landed for this issue, on-cluster 641087 all-PASS).
  - `vk_hip_kda_delta_rule_fwd` (+`_with_scratch`, `_chunked_with_scratch`,
    stream variants) — `hip_capi.hpp:472`, `hip_capi.cpp:370`.
  - CPU oracles `vk_mla_fwd`, `vk_kda_naive_delta_rule_fwd` —
    `capi/capi_attn.cpp:24,255`; exported for serving validators as
    `libvkernels.so` by the `vkernels_hostc` target (`src/c/CMakeLists.txt:256`,
    HIP builds only, by design).
- **Host build + tests**: `cmake --preset host && cmake --build --preset host`
  → OK; `ctest --preset host` → **47/47 passed**,
  `kda_chunked_fused_gpu` + `dsa_topk_fused_gpu` disabled (gfx942-only).
  Oracle symbols confirmed in `build/host/src/c/libvkernels.a`.
- **Python suite**: `uv run pytest tests/python -q` → **765 passed,
  7 failed, 384 skipped**. The 7 failures are pre-existing host-tolerance
  failures of compiled-vs-fallback consistency at `rtol=1e-5`
  (gemm / moe-aux / kda backend-consistency / numerics[gemm]) on this GB10
  host — the worktree is pristine `origin/main`, no local change involved.
  Skips are torch/vllm/triton/GPU-gated suites (torch not installed here).

### cookbook repo (cloned read-only into `work/cookbook` for verification)

- **test_hip_bindings.py dual-name resolution** —
  `deployments/llm/beverin/vkernels/test_hip_bindings.py:134,174,200`
  resolves `vk_hip_fused_moe_mxfp4` / `vk_hip_mla_fwd` /
  `vk_hip_kda_delta_rule_fwd` with the `vk_hip_*`-primary + bare-fallback
  pattern, identical at 006cd58 and main. `py_compile` OK. Execution needs
  gfx942 (`torch.device("cuda")` + `libvkernels_hip.so`) — not runnable here.
- **sitecustomize.py wiring** — `VKERNELS_MLA`/`VKERNELS_KDA` opt-in (default
  `0`); `K3_DISABLE_KDA` precedence: `VKERNELS_KDA=1` is checked **first** and
  wins (sitecustomize.py:781, same at 006cd58:777); `on_gfx942()` guard
  (vkernels_attn.py:273, ROCm + `gcnArchName` contains `gfx942`).
- **Validate-vs-CPU-oracle path** — first-use cross-check:
  `[VkernelMLA] validate vs CPU oracle: max_rel=…` (gate `max_rel < 1e-2`,
  vkernels_attn.py:343) and `[VkernelKDA] validate vs CPU oracle: max_rel=…
  scale_rel=…` (**hard gate `scale_rel < 1e-2`**, `max_rel` logged for context,
  vkernels_attn.py:372). Mismatch → `RuntimeError` → selector falls back to
  `TRITON_MLA` / patch disabled. Note: the issue text says "confirm
  `max_rel < 1e-2`" — the KDA gate actually enforced is `scale_rel`; record
  both numbers.
- **KDA patch scope (current wiring, supersedes the issue body)** — the patch
  replaces the **prefill leaf** `chunk_kda_with_fused_gate` with
  `vk_hip_kda_delta_rule_fwd_with_scratch`. The **decode leaves**
  (`fused_recurrent_kda[_packed_decode]`, Triton) are deliberately left in
  place because they do not fault on gfx942 (job 586165). Consequence for
  item 3 below: the serve still launches Triton KDA *decode* kernels by
  design. See the AC3 note.
- **benchmark.py** — CLI verified: `benchmark.py <base_url> <model> <out_json>
  <spec> [out_tok] [in_tok] [base_url_kda_off]` with the 7th arg /
  `BENCH_BASE_KDA_OFF` baseline compare printing `lat_ratio = KDA-on /
  KDA-off` per level.

---

## 1. Pre-flight (one-time, beverin login node)

```bash
COOK=/capstor/scratch/cscs/xyao/opentela-cookbook   # deployed cookbook tree
VK=$VKERNELS_DIR                                     # deployed vkernels tree

# (a) Pin + record the refs actually deployed (the campaign ran on
#     006cd58 + 3b53d6e-era files; main has since evolved).
git -C "$COOK" rev-parse HEAD
git -C "$VK"  rev-parse HEAD

# (b) Recall-probe availability check (AC2 depends on it). The
#     KDA_RECALL_PROBE machinery exists at 006cd58 but was REMOVED from
#     cookbook main by the 967523e cleanup (shared meta/common/gen_correctness.py
#     has no recall support; README knob table is stale).
grep -c KDA_RECALL_PROBE "$COOK"/meta/common/gen_correctness.py \
      "$COOK"/deployments/llm/beverin/kimi-k3-vllm/gen_correctness.py 2>/dev/null || true
# If both are 0/missing, recover the probe (self-contained script):
git -C "$COOK" show 006cd58:deployments/llm/beverin/kimi-k3-vllm/gen_correctness.py \
  > "$COOK/deployments/llm/beverin/kimi-k3-vllm/gen_correctness_recall.py"
#    and use gen_correctness_recall.py for the AC2 recall commands.

# (c) Build the HIP + host-oracle libs on one MI300A node (one command;
#     produces build/hip/.../libvkernels_hip.so AND libvkernels.so):
cd "$COOK" && rcc --profile beverin run --cwd "$PWD" -- \
  sbatch deployments/llm/beverin/vkernels/build_test_vkernels.sbatch

# (d) ABI smoke on the node (optional, 1 GPU, ~2 min): copy
#     test_hip_bindings.py + vkernels_attn.py into an isolated PYTHONPATH dir
#     per the issue-45 comment-0 recipe; expect 4/4 PASS with non-trivial,
#     finite outputs.
```

## 2. Launch the two arms

Both arms: `ENFORCE_EAGER=1` (the MoE bridge is not capture-safe, #69 scope)
and `VKERNELS_MOE=0` (erratum control: routes MoE to the known-good
`TRITON_UNFUSED` so the attention A/B is not confounded by the separate
MXFP4 MoE defect). Submit from the recipe dir
(`$COOK/deployments/llm/beverin/kimi-k3-vllm`).

```bash
# Arm B (VKERNELS, real weights, port 8080) — correctness + acceptance arm:
VKERNELS_MLA=1 VKERNELS_MLA_FORCE=1 VKERNELS_MLA_VALIDATE=1 \
VKERNELS_KDA=1 VKERNELS_KDA_VALIDATE=1 VKERNELS_MOE=0 ENFORCE_EAGER=1 \
K3_DISABLE_KDA=0 DISTRIBUTED_TIMEOUT_SECONDS=21600 PYTHONFAULTHANDLER=1 \
sbatch --export=ALL serve_kimi_k3_otela_beverin.sbatch
#   - VKERNELS_MLA_FORCE=1 is REQUIRED for real traffic (vLLM calls
#     forward_mqa bare); without it VkernelMLAImpl raises and falls back.
#   - K3_DISABLE_KDA=0 is explicit belt-and-braces (sitecustomize already
#     gives VKERNELS_KDA=1 precedence over the sbatch default of 1).
#   - Real weights: ~77 min Lustre load; DISTRIBUTED_TIMEOUT_SECONDS=21600
#     avoids the gloo 1800 s init hang (post-mortem of job 638605).

# Arm A (baseline, port 8081) — TRITON_MLA + K3_DISABLE_KDA=1, dummy weights
# (all-MLA cannot load the real checkpoint: KeyError '…self_attn.A_log'):
SERVE_PORT=8081 K3_DISABLE_KDA=1 LOAD_FORMAT=dummy ENFORCE_EAGER=1 \
VKERNELS_MOE=0 BENCH=1 GEN_CORRECTNESS_SMOKE=1 OTELA_BIN=/bin/true \
sbatch --export=ALL serve_kimi_k3_otela_beverin.sbatch
```

Arm A may instead reuse the in-tree baseline evidence (job 638514: C=1 p50
66.7 s / 3.8 tok/s per-req) — but note 638514 ran the MXFP4 MoE backend
(before the `VKERNELS_MOE` control existed), so a fresh
`VKERNELS_MOE=0` A/B is the cleaner AC4 evidence.

---

## 3. Acceptance items — one command each

Run from `$COOK/deployments/llm/beverin/kimi-k3-vllm` (or use
`$COOK/meta/common/gen_correctness.py` directly if the recipe-dir copy is
absent at your cookbook ref). `$RUN` is the arm's `run-<jobid>` dir.

### Item 1 — 6/6 factual probes (real weights, arm B)

```bash
GEN_PROBE=1 python3 gen_correctness.py http://127.0.0.1:8080 \
    SwissAI-Research/moonshot/kimi-k3-rocm out.json
```

**Pass:** `6/6` — capital→Paris, sky→Rayleigh, primes, train→40 km/h,
fibonacci, entropy. Crisp prompts (Paris/primes/40 km/h) must all pass
(recipe default `GEN_CORRECTNESS_MIN_PASS=5` tolerates one soft fluke).
Dummy weights are the negative control (job 589456: garbage) — real weights
are required for a meaningful PASS. The report must not carry `"smoke": true`.

### Item 2 — KDA active + oracle match + recall contrast (arm B vs arm A)

```bash
# (a) First-use validate logs (fire the first completion, then grep):
grep -hE "\[Vkernel(MLA|KDA)\] validate vs CPU oracle" "$RUN"/*.log "$RUN"/slurm*.out 2>/dev/null
# Pass: [VkernelKDA] validate vs CPU oracle: max_rel=<m> scale_rel=<s> (gate scale_rel<1e-2)
#       with s < 1e-2 (record m; near-zero v entries can inflate it — the
#       enforced gate is scale_rel), and the analogous MLA line with
#       max_rel < 1e-2. Absence of the lines = validate flags not reaching
#       the engine → investigate before proceeding.

# (b) Recall probe, KDA ON (arm B, real weights) — must PASS:
KDA_RECALL_PROBE=1 KDA_RECALL_REQUIRED=1 GEN_PROBE=1 \
  python3 gen_correctness.py http://127.0.0.1:8080 \
    SwissAI-Research/moonshot/kimi-k3-rocm kda_on.json

# (c) Recall probe, KDA OFF (arm A) — contrast, expected FAIL:
KDA_RECALL_PROBE=1 KDA_RECALL_REQUIRED=1 GEN_PROBE=1 \
  python3 gen_correctness.py http://127.0.0.1:8081 \
    SwissAI-Research/moonshot/kimi-k3-rocm kda_off.json
```

**Pass:** (b) retrieves the buried marker verbatim (`Q9XZ-7K2P` at 006cd58;
`KDA_RECALL_CTX` default 2048). **Contrast caveat (honest form):** arm A can
only run dummy weights (all-MLA cannot load real weights), and dummy weights
fail every probe — the executable contrast is "real+KDA-on PASS" vs
"dummy+KDA-off FAIL" (the latter already recorded in-tree,
`recall_baseline_638514.json`). A real-weight KDA-off arm does not exist on
this hardware (the Triton KDA path GPU-faults). If your cookbook ref is at
main, use the recovered `gen_correctness_recall.py` (pre-flight step b).

### Item 3 — zero AITER/Triton attention kernels

Primary evidence (selection + routing logs, one command):

```bash
grep -hE "VkernelMLA registered|patched chunk_kda_with_fused_gate|\
validate vs CPU oracle|Using '.*' Mxfp4 MoE backend|Attention backend" \
  "$RUN"/*.log "$RUN"/slurm*.out 2>/dev/null | sort | uniq -c
```

**Pass:** `VkernelMLA registered (… selected ahead of TRITON_MLA on gfx942)`
on every rank, `[VkernelKDA] patched chunk_kda_with_fused_gate`, no
`TRITON_MLA` selection on MLA layers, MoE on `TRITON_UNFUSED` (expected,
not attention).

**rocprof cross-check (kernel-level, optional but the issue's letter):**
attach rocprof stats to a probe against arm B (or a single-rank
`--load-format dummy` reproduce inside the same container), then:

```bash
rocprof --stats -d "$RUN/rocprof" <engine/probe command>
grep -icE "triton|aiter" "$RUN"/rocprof/*.csv   # per-kernel table
```

**Documented deviation:** the KDA **decode** leaves
(`fused_recurrent_kda[_packed_decode]`) are Triton by design in the current
wiring (they do not fault; only the chunked prefill leaf did). "Zero
AITER/Triton attention kernels" is therefore satisfied as: zero Triton
*MLA* kernels + zero AITER kernels + zero faulting KDA prefill kernels;
Triton KDA-decode kernels remain. If the issue's literal wording must hold,
routing the decode leaf is additional work (not in scope of the current
wiring) — flag it in the issue rather than reinterpreting silently.

### Item 4 — latency ≤ 1.05x vs baseline at C=1

```bash
python3 benchmark.py http://127.0.0.1:8080 \
    SwissAI-Research/moonshot/kimi-k3-rocm bench.json \
    "1:16 8:32 32:64 64:64" 256 32 http://127.0.0.1:8081
#   (equivalently: BENCH_BASE_KDA_OFF=http://127.0.0.1:8081 python3 benchmark.py …)
```

**Pass:** the printed `lat_ratio = KDA-on / KDA-off` per level satisfies
`lat_ratio <= 1.05` at **C=1** (the AC target level; C=32/64 exceeded the
600 s per-request timeout on the PP3-eager baseline — measure with a longer
`BENCH_PER_REQ_TIMEOUT` if those levels are wanted). Reference point from
the in-tree baseline arm: C=1 p50 66.7 s / 3.8 tok/s per-req (job 638514,
dummy weights, MXFP4 MoE — prefer the fresh same-config A/B above).

### Item 5 — AC4 sbatch default flip (only after 1–4 pass)

```bash
# In the deployed cookbook tree (base: main 5d3f316; the patch file lives in
# the vkernels checkout that carries this runbook, as
# docs/issue45-ac4-cookbook-defaults.patch):
PATCH="$VK/docs/issue45-ac4-cookbook-defaults.patch"
git -C "$COOK" apply --check "$PATCH"   # must print nothing (clean)
git -C "$COOK" apply "$PATCH"           # or: git am "$PATCH" (keeps the message)
bash -n "$COOK"/deployments/llm/beverin/kimi-k3-vllm/serve_kimi_k3_otela_beverin.sbatch
```

The patch flips `VKERNELS_KDA` default `0 → 1` and `K3_DISABLE_KDA` default
`1 → 0` in the `engine.sh` heredoc (the double-quoted heredoc is where EDF
picks up outer-shell defaults), keeps `K3_DISABLE_KDA=1` working as the
`--load-format dummy` fallback / emergency kill-switch, and updates the two
stale comment blocks. It is a **draft only**: do not merge/deploy before
items 1–4 have measured PASS evidence on MI300A. After flipping, redeploy
and re-run the item-1 probe as a smoke check; update the cookbook README
knob-table row for `VKERNELS_KDA`/`K3_DISABLE_KDA` (still describes the
pre-flip default) as a follow-up.

---

## 4. Gaps / risks recorded up front

1. **Recall probe removed at cookbook main** (967523e cleanup; README
   knob table stale). Recovery path in pre-flight (b). The on-cluster
   deployed tree at 006cd58+3b53d6e still has it.
2. **AC3 wording vs wiring** — Triton KDA-decode kernels remain by design
   (item-3 note). Needs an explicit issue-note or decode-leaf routing work.
3. **KDA validate gate is `scale_rel < 1e-2`**, not the issue's literal
   `max_rel < 1e-2` (both logged; record both).
4. **MoE confound** — all acceptance arms must set `VKERNELS_MOE=0`
   (erratum: `VKERNELS_MXFP4_BF16` has never passed a correctness gate).
   The MoE defect is tracked separately and must not gate attention
   acceptance.
5. **Real-weight boots need `DISTRIBUTED_TIMEOUT_SECONDS=21600`** (gloo
   1800 s default hangs after the ~77 min load; 638605 post-mortem) and hit
   a boot-phase NCCL RECV timeout once (639740, pre-probe) — budget one
   re-run.
6. **All-MLA + real checkpoint cannot load** (`KeyError … A_log`) — the
   baseline arm is dummy-weights-only; see the item-2 contrast caveat.
7. Host-side (this machine): 7 pre-existing pytest tolerance failures on
   GB10 documented above; 2 ctests disabled (gfx942-only). Not caused by,
   and not blocking, this runbook.
