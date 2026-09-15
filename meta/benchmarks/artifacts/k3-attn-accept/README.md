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
