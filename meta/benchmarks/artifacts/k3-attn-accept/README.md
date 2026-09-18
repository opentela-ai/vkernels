# K3 attention acceptance — baseline arm (vkernels#45)

Beverin (6×MI300A, gfx942), job **638514** (2026-09-15).

Baseline = the pre-#42 safe serving state: `K3_DISABLE_KDA=1` (all-MLA
architecture — every layer `KimiMLAAttention`/TRITON_MLA, delta-rule layers
absent), `LOAD_FORMAT=dummy` (random weights — perf-only arm; the real
checkpoint cannot load into the all-MLA architecture, see the KeyError note
below), `ENFORCE_EAGER=1` (proven no-cudagraph path), `K3_PREFIX_CACHE=0`,
TP8×PP3 over 24 ranks, RCCL Socket transport.

Files:
- `benchmark_638514.json` — bench sweep `1:16 8:32 32:64 64:64`, 256 out /
  32 in tok, greedy `ignore_eos`. Measured: **C=1 p50 66.7 s / 3.8 tok/s
  per-req; C=8 29.2 tok/s agg / p50 70.5 s**. C=32/64 exceeded benchmark.py's
  600 s per-request timeout (PP3-eager decode at high concurrency) — measure
  those with a longer per-req timeout if needed; the AC4 target is C=1.
- `gen_correctness_638514.json` — smoke gate (dummy weights): 6/6 prompts
  returned non-empty continuations; pipeline OK.
- `recall_baseline_638514.json` — `KDA_RECALL_PROBE=1` long-context
  associative recall: **FAIL** (0/6 factual + recall miss — expected on
  random weights; the meaningful contrast is that the same probe with real
  weights must PASS when the delta-rule layer is served by
  `vk_hip_kda_delta_rule_fwd`).

## Defects recorded on the way

1. **All-MLA + real checkpoint cannot load.** The checkpoint carries
   delta-rule weights (`layers.<i>.self_attn.A_log`, …) for every KDA layer;
   with `K3_DISABLE_KDA=1` the model builds all-MLA and the loader's fallback
   raises `KeyError: 'layers.0.self_attn.A_log'` (job 638175). The recipe
   comment already says the all-MLA state is "ONLY valid with --load-format
   dummy" — so real-weight serving REQUIRES the KDA layers to exist, i.e.
   `VKERNELS_KDA=1` (the faulting Triton path is the only alternative).
2. **vkernels_experts MoE bridge is not capture-safe** (vkernels#69 scope):
   `vkernels_experts.py:538` does `topk_ids…cpu()` inside apply; under CUDA
   graph capture (non-eager run, job 638508) this raises "Cannot copy between
   CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is
   pinned". ENFORCE_EAGER=1 avoids it; capture-safety still needs the #69
   treatment for the K3 MoE bridge.

## ERRATUM — the vkernels MoE backend has never passed a correctness gate

Discovered 2026-09-16 while triaging why the acceptance serve returned `!`
for every prompt (token id 0 = the argmax of all-NaN logits). A scan of every
`gen_correctness_*.json` under
`/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/run-*/` shows the raw
continuations, independent of any verdict field:

| jobs | MoE backend | weights | raw `capital -> Paris` continuation | real? |
|---|---|---|---|---|
| 588856, 589458 | `TRITON_UNFUSED` | `auto` (real) | `' Paris.",\n+  "The Eiffel Tower is locate'` | **coherent** |
| 589456 | `TRITON_UNFUSED` | `dummy` | `'.dartampionship.dartampionship…'` | negative control |
| 597880–603711 | `VKERNELS_MXFP4_BF16` | `auto` (real) | `'.dartampionship…'` (597880: `'!!!!…'`) | **degenerate** |
| 639143, 639260 | `VKERNELS_MXFP4_BF16` | `auto` (real) | `'!!!!…'` (NaN logits) | **degenerate** |

Decisive points:

- **The `VkernelFusedExperts` "PASS"es were smoke-only.** Jobs 597880 and
  603711 ran with `GEN_CORRECTNESS_SMOKE=1` (real weights, `load_format=auto`,
  `[SMOKE] probing` in the log), whose matcher is `_nonempty` — *non-empty*,
  not correct. `.dartampionship…` and `!!!!…` are both non-empty, so the gate
  passed while the model emitted garbage. This is why the earlier "597880
  passed 6/6 with real weights" reading was wrong. `gen_correctness.py` now
  records `"smoke": true/false` in the report so this cannot be misread again.
- **589456 is the negative control**: same `TRITON_UNFUSED` backend, same
  garbage, but `load_format=dummy` — so random weights, as expected. 589458
  is the same backend with real weights and is coherent. The MoE backend, not
  the pipeline, is the variable.
- Therefore the acceptance runs must use the **known-good MoE backend**:
  `VKERNELS_MOE=0` removes `VKERNELS_MXFP4_BF16` from
  `_get_priority_backends`, so selection falls through to `TRITON_UNFUSED`.
  This is orthogonal to #45 (attention) and is the correct control.
- Job **639740** (`VKERNELS_KDA=1 VKERNELS_MOE=0`): the log confirms
  `Using 'TRITON_UNFUSED' Mxfp4 MoE backend`, weights loaded 96/96, and the
  KDA leaf patch applied — then the engine died on a pipeline-parallel
  **NCCL RECV timeout** (`[PG ID 5 Rank 1] … remote process exited`, first
  error 04:59:14) during the post-load memory-profiling forward, before the
  probe could run. That is a boot-phase hang unrelated to the kernels; it
  needs a re-run after the 2026-09-16 07:00 maintenance.

The MoE correctness defect is tracked separately; it is NOT an attention
(#45) defect and must not block the attention acceptance once the run
completes with `VKERNELS_MOE=0`.

## Acceptance runbook (issue #45 tooling — in-repo, machine-readable)

The per-item tools live in `meta/scripts/` (stdlib-only python, no torch:
run bare-env on login nodes, inside the kimi-k3-vllm container, or in CI).
Each writes a JSON verdict and exits 0/1.

| AC | tool | what it asserts |
|---|---|---|
| 1 | `issue45_gen_inspect.py` | gen_correctness report is a **coherent** pass — not a smoke-only (`_nonempty`) pass, not degenerate. Uses the report's `smoke` field when present; the deployed gen_correctness (deploy_v2 / cookbook `f3a12b04`) does **not yet write it**, so smoke mode is inferred from the only reachable signature (`PASS` + `min_pass==6` + `crisp_pass==0`; a non-smoke PASS requires all 3 crisp). Raw continuations are always run through the degeneracy detector (repetition loops / symbol garbage) — this is what re-labels 638514's smoke `PASS` as `degenerate` (`'.dartampionship' x33`). |
| 2 | `issue45_kda_recall_probe.py` | self-contained `KDA_RECALL_PROBE` runner: secret code (`Q9XZ-7K2P`, `--code`) at a controlled depth (`--needle-pos start|middle|end`) inside a long filler (`--filler-words`, default 3000 ≈ 4k tok); PASS = code found **and** continuation not degenerate. `--merge-into <gen report>` attaches the result under the `kda_recall` key used by the 638514/639143/639740 artifacts. |
| 3 | `issue45_rocprof_attention_assert.py` | rocprof capture (`--stats` CSV with a `Kernel Name` column, or raw log) contains **zero AITER/Triton attention kernels**. Allowed: AITER/Triton MoE GEMMs; required (with `--expect-vk`): ≥1 `vk_hip_mla*`/`vk_hip_kda*` kernel in the capture. |
| 4 | `issue45_bench_compare.py` | measured bench JSON vs the baseline `benchmark_638514.json`: p50 latency ratio at C=1 `<=1.05` (`--threshold`); a measured arm that timed out at C=1 FAILs rather than silently passing. |
| — | `run_issue45_accept_mi300.sh selftest` | offline selftest of every tool (fixture level) + a **real rocprof round-trip on 1 GPU**: compile a tiny HIP kernel, `rocprof --stats` it, assert the capture clean, then poison it with `aiter::mla_decode_mla_gluon` and assert FAIL. ~2 min, fits one mi300 job. |

AC5 (default flip in `serve_kimi_k3_otela_beverin.sbatch`) is a cookbook
change made only after 1–4 pass on hardware — not covered here.

### Commands (run from a checkout with `meta/` on beverin)

```bash
# 0. tooling smoke (no serve needed; ~2 min, 1 GPU):
sbatch -p mi300 -N1 --gres=gpu:1 --time=00:15:00 -J i45-tooling \
    --wrap 'bash meta/scripts/run_issue45_accept_mi300.sh selftest'

# 1. AC1 inspection (after the serve's GEN_PROBE wrote gen_correctness_<job>.json):
meta/scripts/run_issue45_accept_mi300.sh gen gen_correctness_<job>.json

# 2. AC2 recall (serve must already be healthy; merge into the gen report):
SERVE_URL=http://127.0.0.1:8080 OUT_DIR=$RUNDIR/evidence \
    meta/scripts/run_issue45_accept_mi300.sh recall
# expect PASS with VKERNELS_KDA=1 and FAIL on the K3_DISABLE_KDA=1 baseline
# (contrast anchor: recall_baseline_638514.json, kda_recall ok=false)

# 3. AC3 (capture first: rocprof --stats -o rp.csv <a request-driving proc>;
#    on the serving head, or attach to the engine with rocr/rocprofv2):
meta/scripts/run_issue45_accept_mi300.sh ac3 rp.csv

# 4. AC4 (measured = benchmark.py output of the VKERNELS arm):
meta/scripts/run_issue45_accept_mi300.sh bench \
    meta/benchmarks/artifacts/k3-attn-accept/benchmark_638514.json measured.json
```

Verified locally (default python3, no torch): all four `--selftest` suites
pass, including the checked-in artifact anchors — 638514 gen report →
`degenerate`, 638514 recall baseline → `kda_recall` FAIL, 638514 bench vs
itself → ratio 1.000 PASS.

On-cluster smoke (beverin, 1 GPU, mi300): the `selftest` mode was run as a
15-min single-GPU Slurm job (`bash meta/scripts/run_issue45_accept_mi300.sh
selftest`). Final green run: **job 641176** (2026-09-18) — all four offline
selftests pass under the container/login python (3.6-compatible), and the
real rocprof round-trip succeeds against **rocprof 6.3.0** on a gfx942 MI300A:
a tiny HIP kernel is profiled (`results.csv` / `results.stats.csv`,
`KernelName` column), the capture asserts **PASS** (zero AITER/Triton
attention kernels), and the capture poisoned with
`aiter::mla_decode_mla_gluon` correctly asserts **FAIL**. Debug iterations on
the way (also cited as evidence of the fixes): 641127 (py3.6
`__future__.annotations` incompat → fixed), 641163 (offline part green,
rocprof 6.3 rejects `-o` → fixed), 641169/641174 (artifact-dir redirect →
fixed). Artifacts kept in
`/capstor/scratch/cscs/xyao/vkernels-i45-accept/i45-smoke-artifacts-*/`.
NOTE for AC3 on the real serve: capture with `rocprof --stats` run from the
desired output dir (no `-o`), then pass the `results*.csv` files to
`issue45_rocprof_attention_assert.py`.

### Owner acceptance campaign runbook (OWNER-ONLY — not launched by this tooling)

The full campaign needs 6 nodes (TP8×PP3, 24 ranks), the real 96-shard
checkpoint (~77 min Lustre load + raised
`--cpu-distributed-timeout-seconds 21600`, see the 638605 post-mortem above)
and >1h wall — beyond the mi300 partition limit, so it is left to the owner:

1. Fork the deploy bundle (as `serve_issue78_k3.sh` does), with
   `VKERNELS_MLA=1 VKERNELS_MLA_FORCE=1 VKERNELS_MLA_VALIDATE=1
   VKERNELS_KDA=1 VKERNELS_KDA_VALIDATE=1 VKERNELS_MOE=0` (real weights;
   keep the baseline arm on dummy weights + `K3_DISABLE_KDA=1`).
2. After /health: AC1 `GEN_PROBE=1` probe → inspect with
   `issue45_gen_inspect.py` (must classify `coherent-pass`).
3. AC2: `issue45_kda_recall_probe.py` on the VKERNELS arm (expect PASS) and
   on the baseline arm (expect FAIL); check the serving log's
   `[VkernelKDA] validate vs CPU oracle: max_rel=… < 1e-2` line.
4. AC3: capture a rocprof window while driving requests, then
   `issue45_rocprof_attention_assert.py rp.csv --expect-vk`.
5. AC4: `benchmark.py` on both arms → `issue45_bench_compare.py` (C=1 gate).
6. AC5 only after 1–4: flip the recipe default (cookbook change).

No acceptance claim is made by this runbook; it only executes the checks and
writes machine-readable evidence.
