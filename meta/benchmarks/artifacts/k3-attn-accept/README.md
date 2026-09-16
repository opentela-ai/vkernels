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
