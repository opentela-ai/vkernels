# Archived t5-soup follow-ups

Source: `t5-soup@c3de2f5`; historical claims and corrections, not fresh validation.
The preceding report is already in main as `ROWS-REPORT.md`.


## T2R-5 lever-1 (graph prefetch) — FALSIFIED on device time; the captured-step baseline is 36ms device-bound

Same-node A/B (nid002706, GLM5_GRAPH_STEP_TIMING=1 both sides):
- OFF (658193): captured step wall 36-37ms, DEVICE 36ms, run-host 3-4.7ms; p50 12.95.
- ON  (658225): captured step wall 47-48ms, DEVICE 46ms, run-host 3.5-5ms; p50 12.03.

The prefetch ADDS ~10ms of device time. Mechanism: the host-side
planner (plan_prefetch_pages) sees only the CPU tables -- not the
device-resident tag state -- so it re-copies already-hot rows; and the
fills execute as per-layer strided page-slice copies (94 layers x
~1-2 pages) whose device cost (~10ms) duplicates rather than replaces
the resolve's in-kernel fill path. Prediction accuracy cannot fix
this: the planner is structurally tag-blind.

Also resolved this turn: the node confound (2706 vs 2346/2350 legs
differ by ~10ms/step -- ALL cross-node p50 comparisons are void; only
same-node A/Bs count) and the parallel-legs $SNAP/repo-prof race
(658110: a sibling job's rebuild yanked the tree mid-serve).

THE REAL BASELINE: the captured width-1 decode step = 36ms, ~100%
device-bound (host 3-5ms fully pipelined). The single-stream levers
that matter are all device-side:
- lever 2 (dense fleet): 8.6ms at 250-580 GB/s -> 3-6x kernel headroom;
- lever 3 (resolve_rows): ~9ms incl. its in-kernel fill path;
- the remaining ~18ms: attention/MoE kernels + collectives -- a kineto
  trace of the CAPTURED step (not the eager fallback) would split it.

## T2R-6 SGFUSE A/B (658285/658286, same node 2346, GST=1 both)

- The "node confound" is DEAD: OFF@2346 captured step = 36-37ms device =
  OFF@2706. The earlier 36-vs-46ms delta was PURELY the lever-1 prefetch
  regression (node-independent).
- SGFUSE (skinny_gemv + fused_norms, both engaged per dispatch knobs):
  width-1 captured step UNCHANGED (36.8-37.5ms, composition identical in
  the kineto re-harvest: Cijk 5.4ms/84 calls, resolve 5.0ms, glue ~9ms,
  MoE 5.5ms, RCCL 3.7ms).
  BUT conc wall 30.4s -> 17.4s (agg 16.9 -> 29.4, all_paris) — the best
  conc ever (fused_norms at width-4).
- WHY width-1 is unmoved: the trace has ZERO GEMV-fleet kernels — the
  dense fp8 GEMV fleet never engages in-serve. The aten::linear
  neighborhoods show the t/transpose/as_strided decomposition; the
  microbench (contiguous textbook w8 [O,I]) accepted all shapes, so the
  prime suspect = the serve's loader produces non-contiguous/transposed
  weight storage -> dense_gemv_fp8's contiguity check declines ->
  dequant-on-use F.linear = the Cijk block. NEXT: inspect the staged-fp8
  loader's weight layout vs the op's eligibility contract; fix = store
  [O,I] contiguous or accept strided reads in the kernel (Triton strides
  are free).
- p50 today on 2346 = environmentally cold (6.8/8.56 with 36ms captured
  steps; runs improve within-leg). The captured-step device-ms is the
  stable gating metric; client p50 only for same-hour comparisons.

Priority queue (device-side, width-1, 37ms budget):
1. the Cijk block 5.4-10.7ms -> GEMV fleet engagement (loader layout).
2. aten glue ~9ms -> the remaining unfused elementwise/reduce soup.
3. resolve_rows 5.0ms + MoE 5.5ms (lever 2/3 kernel work).
4. RCCL 3.7ms (small; TP restructuring later).

## T2R-6 follow-up: the Cijk block IDENTIFIED (658387 + trace re-analysis)

The gemv-decline diagnostic leg: ZERO decode-M declines (only prefill
M=21-23 chunks) -- the fp8 dense GEMV fleet ENGAGES at decode all
along. The 658286 trace recount: _gemv_fp8 = 63.7 calls/step at only
361us/step TOTAL (~5.7us/call; _pre_gemv another 13ms/trace-window).
The "fleet never engages" read was a name-grep artifact.

The Cijk block (~147 calls, 5.4-10.7ms/step) = the gated-delta-rule
ATTENTION CONTRACTIONS (forward.py ~2836-2849: attn @ v_beta, attn @
(k_beta * g.exp()), the per-layer state updates + core_attn_out
assembly) -- ~3 tiny batched GEMMs x 45 layers, each ~40-70us of
rocBLAS small-tile time for trivial FLOPs. Not projection weights:
attention math outside every fleet.

Fix class: extend the vkernels fused attention family (_big_fuse /
_compose / the aiter_mhc path, currently OFF) to absorb these
contractions -- real kernel work (the lever-2/3 class), not config.

Revised width-1 device budget (37ms): delta-rule matmuls 5.4-10.7 |
aten glue ~9.2 | resolve 5.0 | MoE 5.5 | GEMV fleet 0.36 | RCCL 3.7 |
attention/misc remainder.

## T2R-7 KDAFUSE staircase (658395/658396, node 2346, sequential)

The decode S==1 path already routes to the TRITON fused-recurrent
kda_decode under the never-legged `fused_kda` opt-in knob (the CUDA
packed/fused donors are nvcc-compiled = MI300-blocked; the Triton
glm_kda_decode is ROCm-ready and audited bit-identical to the fp32
reference with out_fp32=True).

Staircase (captured-step device-ms, the stable metric):
- OFF (658285):        37.0   conc 30.4s
- +SGFUSE (658286):    36.8   conc 17.4s
- +KDAFUSE (658395):   32.7   conc 16.6s/30.8 agg (best yet)
- +FCU (658396):       33.5   -> DROP FCU (neutral-to-negative)

Net: -4.3ms/step (37.0 -> 32.7), zero regressions, all_paris green
throughout. The environmental p50 swings (12.98 and 6.7 on the same
node within the hour) reconfirm: device-ms is the only gate.

Remaining width-1 budget at 32.7ms: glue ~9 | resolve 5.0 | MoE 5.5 |
RCCL 3.7 | reduced Cijk remainder | misc. Next biggest block = the
elementwise/reduce glue (~1200 calls) — kernel work beyond
fused_norms; then resolve/MoE.

## T2R-8 MHCFUSE — REGRESSION, dropped (658446)

fused_mhc_compose (the Triton mHC stream kernels): 33.6-34.2ms
(+1.1 vs KDAFUSE-only) AND conc 27.0s (vs 16.6) — worse on both.
Dropped. The persistent Cijk_MT128x32x32 block (~80 calls, ~5ms) is
NOT the mHC compose (or the fusion's own cost exceeds the eager
matmuls at these shapes).

FINAL RECIPE (zero code, dispatch flags): SGFUSE + KDAFUSE
= skinny_gemv + fused_norms + fused_kda:
- width-1 captured step 37.0 -> 32.7ms (-12%)
- conc 30.4 -> 16.6s (agg 16.9 -> 30.8, +83%), all_paris green

Staircase summary (node 2346, sequential, device-ms gate):
OFF 37.0 | +SGFUSE 36.8 | +KDAFUSE 32.7* | +FCU 33.5x | +MHCFUSE 33.8x

Remaining width-1 blocks at 32.7ms: glue ~5.2 | resolve 4.9 | MoE 5.3 |
Cijk-persistent ~5.0 | rccl 2.9 | gemv 0.36. All kernel-work class.

## T2R-9 router_gemv — the fp32 router scoring GEMM off BLAS (658473/477/479/480/482)

The router logits GEMM [1,4096]x[288,4096] fp32 ran as rocBLAS small-
tile Cijk (~2.7ms/step across 42 MoE layers). New vkernels op
router_gemv (fp32 GEMV, the HF rounding class preserved: fp32
accumulate, one round at the store):
- v1 ROWS=16/BK=1024: 18 CTAs = 8% occupancy -> 52us/call, no win
  (658473: engaged but step unchanged).
- MI300A tiling sweep (658479): ROWS=1/BK=4096 = 3.06us/call (1.56
  TB/s, 288 CTAs) vs cuBLAS fp32 18.77us = 6x.
- v2 (658480): width-1 captured step 32.7 -> 31.95-32.36ms (-0.5, the
  router block collapses) BUT conc 41.3s -- full-width BK at M=4
  spilled registers (x tile [4,4096] fp32 per CTA).
- v3 (658482, MP*BK<=8192 budget): width-1 32.25-32.86 (win holds),
  conc 25.65s vs the 16.6 KDAFUSE baseline -- today's environment
  swings (p50 6.7-13 within hours) make the conc gate unreliable;
  worst case at M=4 = neutral after the fix. If a calm-window leg
  still shows regression: route only M<=2 through the op.

Branches: vkernels perf/router-gemv (worktree ../vkernels-routergemv,
WIP in the main tree untouched) @ 6a8ffaa; floe perf/router-gemv @
9b39e20 (OPT-IN router_gemv + vkl_ops wiring). Beverin: wired + legged.

## T2R-10 MoE grouped config sweep (658491) + the session's arc close

VK_MOE_GROUPED_CFG sweep at the serving shapes (E73/IA512/I4096/K8/T1,
in-graph, all configs bit-identical): best r1=2/w1=8/r2=4 = 66.2us
(760 GB/s) vs the default 75.0us -> default retuned (05460c8), ~-0.2ms/
step, below leg noise; rides the next leg.

The sweep ALSO bounds the MoE headroom honestly: the kernel tops out
at ~760 GB/s vs the 3.4TB/s wall -- the fp8 decode chain (software
e4m3fn + the parity-pinned double bf16 round trips) is instruction-
bound. The remaining big MoE win requires decode-chain surgery with a
parity decision (drop the intermediate bf16 rounds or a native-fp8 dot
path) -- a heavier arc than this session's config wins.

SESSION OPTIMIZATION LEDGER (width-1 device-ms, same-node gates):
37.0 (baseline) -> 36.8 (SGFUSE) -> 32.7 (KDAFUSE) -> 32.3 (RGEMV)
-> ~32.1 (MoE retune pending next leg) = -13% with zero regressions;
conc 30.4s -> 16.6s (SGFUSE+KDAFUSE recipe, RGEMV pending calm-window
verification). Remaining blocks all kernel-surgery class: MoE decode
chain (5.1), resolve (4.9), glue (4.6), residual Cijk, RCCL (2.1-2.9).

## T2R-11 moe_bf16_decode — implemented, MEMORY-BLOCKED on this config (658522/658530)

The 658521 bound: a bf16-resident read of the same weight values runs
3672 GB/s (L2-assisted) vs the tuned fp8 chain's 760 GB/s — 2.4x
despite 2x bytes; the decode chain is instruction-bound. Implemented
end-to-end (vkernels _kernels16 bit-exact by construction + the floe
OPT-IN moe_bf16_decode with lazy twins; 7715f32/20208f3/2e3f374):

- 658522: OOM — the full-stack _dq fp32 transient (~5GB at E=289).
- 658530 (chunked, ~270MB transient): STILL oom_killed — the
  PERSISTENT twins (+3.6GB/rank, +14.4GB node-wide) exceed the config's
  headroom (the fp8 stacks exist FOR memory; kvaasd's pool shares the
  node). The lazy-at-first-decode timing lands after the KV/daemon
  pools take their share.

Disposition: OPT-IN stays off; the code is correct and ready. The 2.4x
lands where the memory math works: EP-sharded experts (1/4 the twin
cost/rank), smaller-IA variants, or evictable/hot-set twins (a
top-working-set bf16 cache with the fp8 path as the miss handler).

## T2R-12 consolidation — the merged all-in tree WORKS, the 6-knob stack REGRESSES (658566, 2026-10-02)

The consolidation lane: git 3-way (base/l6/rows/soup snapshots) produced
run-full-clone = l6 all-in + rows wiring (sparse_runtime hand-fused with
T2R-5 selected_log) + soup caches + dsa_resolve_rows.py in vkernels +
the 419-line parity test. sbatch gained composable ROWS=1/SOUP=1 flags
(the RGEMV/KDAFUSE/SGFUSE pattern). All 21 knobs verified True in-serve
(the knobs dict prints non-None only — it is the delivery oracle that
caught 658539 silently running bare: RGEMV is env-flag gated, and the
submission didn't pass it).

Verdict vs the SAME-snapshot bare baseline (658539, t1r arm, no env
knobs, correctness green):

  B=1 p50:      16.57 vs 18.03 tok/s   (-8%)
  conc-4 warm:  32.08 vs 39.39 tok/s   (-19%)
  E2-PREW c4:   24.07 vs 25.84         (-7%)
  TTFT:         +5-15% on all 4 probes
  all_paris:    True everywhere; 0 fallbacks; 0 capture-fails

The opt-in knobs LOSE to the free baseline. Context: 658520 -> 658539
(14.26 -> 18.03 B=1, 23.69 -> 39.39 conc-4) was delivered by the T2R
kernel-default landings alone (MoE retune default, tiling bounds) — the
bar the config knobs must now beat, and they don't. Suspects: (a)
tp_control_stream+commit-defer (tuned on the pre-T2R base), (b) fp32
soup caches doubling hot-tier rows, (c) fused_kda/SGFUSE double-applying
over the new defaults. Bisect in flight: 658578 (RGEMV+KDAFUSE+SGFUSE
only) vs 658579 (ROWS+SOUP only), both on run-full-clone, concurrent.

Two infrastructure findings:
1. kineto phase-2 dies on the post-T2R snapshot state (2/2: 658539,
   658566) — "External init callback must run in same thread as
   registerClient" kills the serve during the traced bench; 658520/658530
   completed fine. Attribution microscope broken exactly when needed.
2. run-full-clone needs $SNAP/{repo,kvaas-src,apply_kineto_patch_bev.py}
   — the kvaas client stages from $SNAP/kvaas-src (else silent fallthrough
   to the stale EDF kvaas-fix = the ManagedResidencySessionLost ImportError).

## T2R-12b — merge-regression bisect + trace attribution (658578/658579/658752/658759)

**Bisect verdict (either subset alone reproduces the full regression — non-additive, knobs exonerated):**

| leg | tree | knobs | B=1 p50 | conc-4 warm |
|---|---|---|---|---|
| 658539 | old (run-wZTCex7s) | bare+t1r | 18.03 | 39.39 |
| 658566 | merged | all 6 | 16.57 | 32.08 |
| 658578 | merged | RGEMV+KDAFUSE+SGFUSE | 16.56 | 32.27 |
| 658579 | merged | ROWS+SOUP | 16.38 | 33.37 |
| 658752 | merged | **zero** | **15.75** | **29.9** |

The knobs are exonerated (each subset ≈ the full stack, zero-knob worst). **The merge
base itself costs -12.6% B=1 / -24% conc-4** vs the old bare cell. Knobs only mask it.

**Trace attribution (kineto decode-w1, 656984 old vs 658578 merged; correlation-mapped):**

- Merged **graph replays** (hipGraphLaunch, T=1) run `_moe_gate_up` at **171.7 µs vs
  44.1 µs** in the old tree — 3.9x, same kernel name, same T=1, 1134 launches/window
  (+4.6 ms/step). `_moe_down` 84.9 vs 27.0 µs (3.1x, +2.1 ms/step). This is the single
  biggest clean device-time adder.
- Merged **eager** launches are healthy per token: 83.4 µs @ T=2, 163.9 @ T=4
  (~41 µs/token ≈ old 44). The T2R-10 config (rows1 2/warps1 8) is fine eager but
  catastrophic in the captured replay → the T2R-10 sweep validated the eager microbench
  only. The 4x replay tax is either (a) the baked config or (b) capture-path memory
  placement. **DIAG-A leg 658759 discriminates**: merged + `_DEF` reverted to
  rows1 4/warps1 4/rows2 8/warps2 4.
- Window composition: ~27 replay steps + 4 eager steps (capture-fail re-decode at
  T=2/3/4 — the grid[0]=18/27/36 signature). Capture-fail count is equal old vs new
  (12/serve) → not the differentiator.
- `aten::linear` → hipBLASLt `Cijk MT128x32x32`: 48 mm/step @ ~70 µs in merged
  (+2.5 ms/step net vs old Cijk) — the skinny/linear ops are NOT on the custom GEMV
  fleet in the merged tree (old ran 351 `_gemv_bf16`/step @ 4.8 µs — the router).
- `_grouped_gemm` appears in merged decode (11.2 launches/step, +2.3 ms/step) — new
  shared-expert/attention route, absent in old decode.
- `resolve_rows` calls DOUBLED in the ROWS-off leg (11.7→23.8/step); ROWS=1 recovers
  the count (12.5/step) but tok/s is unchanged → not the driver.
- Host syncs HALVED in merged (105/step, 516 ms vs 120/step, 1129 ms) — merged better.
- rccl wait is node-noise-dominated (509 vs 777 µs/launch between identical-code legs)
  — conc-4 attribution from single legs is unreliable.

**Kineto tooling fixes (keep):** (1) every chunker since kin_sum2 dropped the
`else: buf.append(s)` branch → silently empty chunks, all-zero counts; kin_sum2 was the
only correct one. (2) beverin front-ends are round-robin with a SHARED stale /tmp —
push+execute must be atomic (scp the script, then run). (3) The correlation→launch-ts→
cpu_op parent-chain method works (1528/5184 Cijk mapped → 1440 aten::mm ← matmul ←
linear). (4) Phase-2 death is now 5/5 on merged states but E1-E3 traces DO land before
the phase-B kill (BrokenPipeError post-734MB-export) — harvestable.

**Discriminator legs (all merged tree, arm=stack-moe-t1r unless noted):**

| leg | variant | B=1 p50 | conc-4 | trace replay gate_up |
|---|---|---|---|---|
| 658752 | bare merged | 15.75 | 29.9 | 171.7 µs |
| 658761 | `_DEF` reverted (rows1 4/w1 4/r2 8/w2 4) | 15.09 | 29.56 | **149.1 µs** |
| 659025 | kvaasd `--device-pool` removed | 15.6 | 33.3 | 155.6 µs |
| 659089 | arm=stack-moe-l5-debug (host step tables) | pending | — | — |

Read-outs so far:
- The T2R-10 moe config moves the REPLAY time only ~13% (171.7→149.1) and the WALL
  not at all → config exonerated for the wall.
- The kvaas device pool (6 GiB/GPU) moves neither the wall nor the replay → pool
  exonerated. (conc-4 33.3 vs 29.9 is within the leg cluster 29.9-33.4.)
- The old tree's trace shows a graph CAN replay `_moe_gate_up` at 44.1 µs — but
  cross-tree per-launch comparisons are polluted: the merged per-launch durs are
  2-4x across ALL families (moe 3.8x, elementwise 2.9x, Cijk 1.8x) while the WALL
  is only 1.09x — most of the cross-tree per-launch delta is profiling overhead of
  the merged stack's kineto/roctracer build, not device time. Within-merged
  comparisons (eager vs replay) remain valid: the replay tax is real but its wall
  share is bounded by the replay step fraction.
- Recipe deltas ruled out: tp_control_stream (658566/658579 have it, no gain),
  commit-defer markers (same legs), KEYTRACE logging (132 event-driven lines/serve).
- The `_DEF` revert was restored; the nopool sbatch variant is
  glm5-tp4-bench-nopool.sbatch on beverin scratch.

**BASELINE CORRECTION (supersedes the T2R-12b table's snapshot labels):**

Job 658539 — the 18.03 B=1 anchor — ran on **run-l6-clone**, NOT run-wZTCex7s.
The eras: run-wZTCex7s = the PRE-T2R base (_DEF rows1:4/w1:4, no sparse_shape_warm
in knobs; 13.0 B=1 re-measured today on the same flags); run-l6-clone = the l6-era
floe + post-T2R vkernels (same _DEF as the merge). Corrected pairing, both
measured within hours of each other, bare recipes:

- run-l6-clone: **17.9 B=1** (659094, today; 18.03 on 658539's day) / conc-4 31.68
- run-full-clone: **15.75 B=1** (658752) — **merge delta -11.8% B=1**; conc-4 deltas
  are day-noise-dominated (31.68 vs 29.9-35.97 across legs) — do not quote -19%.

The earlier "old tree" trace comparison (656984, run-ctrl-clone) is also cross-era;
its per-launch numbers were profiling-polluted anyway (see above). The merge delta
surface is SMALL: diff vs run-l6-clone = sparse_runtime.py (rows lane, latched off
under t1r), forward.py (opt-in soup knobs + dispatch-contextvar fix), dispatch.py,
knobs.py, NEW glm53flash/knobs.py (registry consolidation) + NEW engine/forward.py
(the pure-PyTorch port moved to engine/). kvaas-src IDENTICAL (daemon theory dead:
both trees run the same kvaasd + device_pool=on; pool on/off changes nothing).
vkernels identical except dsa_resolve_rows.py (off under t1r). All six floe diffs
are ~inert plumbing — yet the -11.8% reproduces across nodes/days. Next microscope:
GST per-stage step timing (GLM5_GRAPH_STEP_TIMING), which needed a marker-file
fallback patch in both snapshots' knobs.py (the step-time env reads hit the EDF
os.environ-replacement trap; the import-time alias survives but the per-step gates
don't). Legs 659095 (full-clone) / 659096 (l6-clone) = GST head-to-head.

**Control failure worth remembering:** adding `--gpu-timing` (CUDA events per
forward) + GST to BOTH trees moved the OLD snapshot's B=1 from 18.03 to **13.0**
(659092) while the merged tree was unmoved (15.56) — the l6-era serve pays heavily
for the instrumentation the merged one shrugs off. Never trust cross-flag wall
comparisons; and today's per-node numbers vary enough that every claim above is a
same-flag, same-day pairing.

**STALE-MARKER CONFOUND FOUND (T2R-12d):** the universal marker block in the
sbatch is ctrl-arm-only; t1r legs never rewrite the step-time sparse markers, so
run-full-clone carried **stale FLOE_SPARSE_BATCH_TELEMETRY=1 +
HOST_STEP_TABLES=1/DEBUG=1** (from an old diagnostic leg) into EVERY recent
merged leg, while run-l6-clone's dir held 0s — a systematic recipe inequality
between the trees that the scrubber-proof `_env_flag` faithfully honored at step
time. BATCH_TELEMETRY=1 is the *batched-sync* fast path (one reduce+sync for the
resolver fleet vs 100+ per-resolver `.item()` reads — kineto 656346), so its
effect direction is unclear; zeroed on run-full-clone (repo + repo-prof) for the
equalized rerun. HOST_STEP_TABLES has no consumer in the merged model code.

**Instrumentation asymmetry (control failure #2):** GST=1 costs l6-clone ~2.5
tok/s B=1 (17.9 -> 15.45, 659098) while the merged tree is unmoved (15.56, 659091
with GPU-timing too). Cross-tree GST/GPU-timing deltas are therefore not
evidence; only bare same-flag pairings count. The GST per-stage structure at
matched context bands did show merged dev/step +2-4ms, but given the asymmetry
that number is suggestive only.

**New anomaly:** the GST=1 merged leg (659099) collapsed within the leg — B=1
runs degraded 15 -> 6.6 tok/s, conc 11.9 — progressive in-run degradation never
seen on bare legs. Suspect the per-step CUDA Event churn (prepare_inputs creates
2 timing events per step under GST) or an accumulating kvaas state; untested.

Equalized bare pair running: 659100 (merged, markers zeroed) / 659101 (l6).
Verdict rule: merged ~15.7 -> regression real, not markers; merged ~17 ->
the "merge regression" was the stale markers and T2R-12b bisect conclusions
need a full re-audit against marker-equalized baselines.

**ROOT CAUSE FOUND (T2R-12e): the merge duplicated the kvaas resolve — every
DSA layer resolves TWICE per decode step in every merged-tree leg.**

The rows-lane hunk in the merged sparse_runtime.py swap_in was added WITHOUT
consuming the original unconditional resolve it was supposed to replace:

    r = self._resolvers[g]
    rows = None
    if self._rows_lane_init():
        rows = self._resolve_rows_fast(r, self._selected)
    if rows is None:
        rows = r.resolve(self._selected)  # [1, width, W]
    rows = self._resolvers[g].resolve(self._selected)  # [1, width, W]  ← LEFTOVER

so even with the rows lane latched off (t1r), swap_in pays r.resolve(...) AND
self._resolvers[g].resolve(...) — two identical kvaas resolves per layer per
step. Trace proof: kin-659091 (merged) shows **715 resolve_rows/window
(594 replays ≈ 22/step)** vs the pre-merge era's 352/window (~11/step, one per
DSA layer) — exactly 2x — while every other kernel family's per-step replay
count is unchanged (moe gate_up/down 42/step both trees; fp8_grouped=True in
both dispatches; bucket widths identical). resolve_rows was the T2R motivation
target (829 µs x 11/step under kineto; the doubling adds the same again).
Fixed by deleting the leftover line in run-full-clone repo + repo-prof;
verification leg 659115 (bare t1r, markers zeroed, good node). Also note:
under the ROWS arm the same bug nullified the vectorized resolver (it ran AND
the kvaas oracle still resolved), so any rows-lane B=1 numbers to date are
upper bounds on the true win.

Node hygiene: nid002696 systematically degrades B=1 (13.0-13.5 bare, collapses
to 5-6.6 within-leg on GST legs) — exclude it from all comparisons
(--exclude=nid002696). Same-node control on nid002674: l6 18.03 (658539) vs
merged 14.92 (659105) — the regression is node-controlled, not placement noise.
Marker-equalized control (659100/659105, BATCH_TELEMETRY zeroed): merged still
~15 — the stale-marker confound is real but NOT the regression.

**T2R-12e VERIFIED (659115, nid002698, bare t1r, markers equalized):**
merged tree + one-line fix = **18.24 B=1 p50 (17.23/18.24/18.28) / 39.53 conc-4**
vs the same snapshot pre-fix 14.92-15.85 B=1 / 29.9-32.5 conc — **+2.4 tok/s B=1
(+16%), +9.6 tok/s conc (+32%)** — and now level with run-l6-clone on the same
node (17.87/36.2, 659106). The double resolve also suppressed conc
disproportionately (each of 4 concurrent requests re-resolves the full DSA
fleet). The merge regression chapter closes: no node effect, no kvaas effect,
no config effect, no capture-path mystery — one leftover line.

**Follow-ups opened:**
1. Re-run the rows arm (dsa_resolve_rows_fast=true) on the FIXED tree — every
   rows-lane number to date was measured with the bug (vectorized resolver ran
   AND the kvaas oracle still resolved), so the true win is unmeasured; the
   9.1ms/step kineto anatomy that motivated the lane was itself the DOUBLED
   cost, so the expected steady-state win is roughly half the original estimate.
2. The source repo (~/Documents/code/floe @ 2d373fe) now carries the corrected
   rows-lane + T2R-5 prefetch port; snapshot patch scripts that re-apply these
   hunks should be retired in favor of snapshotting from the source.
3. Re-audit the T2R-12b bisect table: its per-cell walls were all measured on
   the buggy tree; relative orderings likely survive but absolute numbers shift
   ~+2.4 B=1.
