# NOTES — fusion candidate: GLM decode expert seam, fused-epilogue GEMV+silu (rank 5)

Branch: `fusion/glm-decode-fused` (from main 8e2bc1b). Slug: `glm-decode-fused`.
Scope: `src/python/vkernels/torch_ops/glm_expert_gemv_fused.py` (new),
`tests/python/test_glm_expert_gemv_fused.py` (new),
`bench/bench_glm_expert_gemv_fused.py` (new), `docs/torch-ops.md` (one table row),
`docs/glm53-decode-kernels.md` (one section). No shared-file edits; no C++ touched.

## Baseline (from the docs, not re-measured)

- `glm_expert_gemv` (torch_ops, issues #64/#65, adopted from floe
  `glm5_fp8_gemv.py`): block-FP8 selected-expert GEMV, one program per
  (token, expert-slot), writes BF16 `[T,K,O]` (stacked gate|up for the
  gate/up projection, `O=2*IA`). Decode-validated on MI300A at ~1.0 TB/s
  effective (docs/glm53-decode-kernels.md header).
- Native HIP `glm_fp8_block_gemv` numbers (same doc): gate/up [4096,4096]
  22.4 µs (751 GB/s) at M=1, 29.5 µs at M=2; **~324 µs per decode token**
  for top-8 gate/up + down fused vs ~529 µs materialized. After three
  structural rounds the op is **instruction-issue-bound at 500–800 GB/s
  (9–15% of HBM)** — "further gains need a different design … or fusing
  the top-k gather", not tuning. That named design gap is this lane.
- The activation today is a SECOND launch: `elementwise.silu_mul` over
  the `[T,K,2*IA]` halves — one extra dispatch + a full gate/up
  store→reload round trip between the two GEMV stages.

## What changed

1. `glm_expert_gemv_fused.expert_gemv_silu(x, w, s, idx, storage=...)` —
   new additive module; reuses `_t_cap`, the validation order, the byte
   decode, scale gather, weight-bf16 rounding, dot shape (`ROWS=4`,
   `COLS=next_pow2(I)`, `num_warps=4`, `enable_fp_fusion=False`) and the
   HIP/NVIDIA backend gate verbatim from `glm_expert_gemv`. Differences:
   - each program loads the gate rows AND the matching up rows (offset
     `+IA`), dots both against x, and applies the activation;
   - writes only `act[T,K,IA]`; grid halves to `(t*k, cdiv(IA,4))`;
   - shapes: `weights[E, 2*IA, I]` with IA (act width) and I (input
     width) INDEPENDENT — the GLM serving shape is `[E,4096,4096]`
     (IA=2048, I=4096); an earlier draft that collapsed IA==I would have
     rejected the serving shape (caught in review of the shape contract,
     before any MI300A run);
   - both `e4m3fn` (native `float8e4nv` bitcast on NVIDIA, manual decode
     on HIP) and `e4m3fnuz` (manual decode both backends, doubled scales)
     storages, mirroring the sibling.
2. **Numerics contract = bit-exact equality with the unfused chain on the
   same device**, not a tolerance: the gate/up fp32 dots are rounded to
   bf16 in registers exactly where the chain stores/reloads them, silu is
   the verbatim `elementwise._swiglu_limit` expression (fp32, rounded to
   bf16 at silu_mul's storage boundary), product stored bf16. NaN
   propagation identical. Because the contract is device-local (tested on
   the device it runs on), triton version/backend differences in
   reduction layout or `exp` ULPs cannot silently break it.
3. Opt-in knob `silu_fused_enabled()` / env `GLM53_MOE_SILU_FUSED`
   (default `0` — the proven `expert_gemv` + `silu_mul` path is
   untouched and remains the default; kill-switch precedent
   `VK_DSA_DECODE_SPLIT=0`).
4. Tests (`tests/python/test_glm_expert_gemv_fused.py`, 14 cases, all
   passing on GB10): laziness, knob default-off, CPU contract envelope
   (`OpNotEligible`: cap, dtypes, scales, stacked-layout, storage),
   **bit-exact vs the two-launch chain** across (t,k) ∈ {1,2}×{1,8},
   broadcast and per-slot x, asymmetric shapes (IA=128/192/256 with
   I=256/512 — IA=192 deliberately straddles a 128 scale-row boundary),
   fnuz storage (incl. losslessness vs pristine e4m3fn bytes), parity vs
   `expert_gemv_silu_reference` at the repo's 3e-3 norm gate (secondary —
   the composed silu can locally amplify the GEMV's bf16-level dot
   difference near zero activations), CUDA-graph capture/replay with
   changed inputs.

## GB10 results (indicative ONLY — LPDDR5x unified memory, clocks could
## NOT be locked: `nvidia-smi -lgc` permission denied on this box)

`bench/bench_glm_expert_gemv_fused.py`, 25 warmup discarded + 4 batches ×
1000 launches, CUDA event pairs, median (us/call):

| shape (E=32, top-8) | unfused (2 launches) | fused (1 launch) | delta | bytes saved |
|---|---:|---:|---:|---:|
| IA=2048 I=4096 t=1 | 559.71 | 555.00 | **0.8%** | 0.10% |
| IA=2048 I=4096 t=2 | 958.42 | 952.75 | 0.6% | 0.10% |
| IA=1024 I=2048 t=1 | 134.50 | 131.52 | **2.2%** | 0.19% |
| IA=1024 I=2048 t=2 | 227.85 | 225.35 | 1.1% | 0.19% |

Both arms run at ~240–298 GB/s model bandwidth — **at GB10's LPDDR roof**
— so the fused win equals its byte saving plus the removed kernel
boundary, exactly as the roofline predicts. Direction is consistent at
every shape; spread ≤ 13.5 µs across batches (worst batch on the first
fused row; the other seven batches spread ≤ 3.9).

## Design section: full stage-0 → stage-1 chaining (the honest roofline)

Stage model: stage-0 = gate/up GEMV (+ silu) producing `act[T,K,IA]`;
stage-1 = down GEMV `act → hidden[T,K,I]` (`weights_down[E,I,IA]`, GLM:
[4096,2048] per expert).

**Byte model per decode token (top-8, GLM serving shape).** Weights:
stage-0 reads `k·2·IA·I = 8·4096·4096 = 134.2 MiB` fp8; stage-1 reads
`k·I·IA = 67.1 MiB`. Activations, ALL of them together (router logits
aside): gate/up output 8·4096·2 B = 64 KiB, act 8·2048·2 B = 32 KiB,
down output 8·4096·2 B = 64 KiB — ~160 KiB against ~201 MiB of weight
traffic, i.e. **activations are ~0.08% of the bytes**. Any fusion that
removes activation round trips (silu_mul's 96 KiB/token here; a further
act store→load into the down GEMV, 96 KiB more) removes at most ~0.1% of
the traffic. The bandwidth win from fusing the activation is **near
zero** — on GB10 this is not a prediction, it is the measurement (0.6–2.2%
wall, tracking the 0.10–0.19% byte delta plus boundary effects). This is
the same shape of honest negative as the `glm_projection` entry
(kernels-reference §3.5): the operator is correct, cheap, and NOT a
bandwidth story.

**What actually pays, and when:**

1. **Launch count at M ≤ 2 decode (this deliverable).** The chain is
   2 launches → 1; the measured GB10 delta (~3–5.7 µs/call) is the silu
   kernel + the boundary. On MI300A the dispatch floor is ~2.8 µs
   (kernels-reference row 23) and the silu launch is a few µs more;
   against ~324 µs/token that is a low-single-digit percent per MoE
   block, multiplied by every MoE layer under CUDA graphs where launch
   gaps are serialized dependencies. Pays at decode; irrelevant at
   prefill M ≥ 64 where the GEMMs amortize everything.
2. **Router-gather plumbing (named in glm53-decode-kernels.md, NOT
   delivered here).** Fusing the top-k ids/scale production into the GEMV
   launch removes another launch and the `[T,E]` logits round trip. This
   is plumbing with a fixed µs payoff, independent of the bandwidth
   argument — the actual next lever if decode launch overhead dominates.
3. **Fusing silu into the down-GEMV read (stage-1 read-side).** The down
   GEMV's x-operand load could read gate/up GEMV outputs and apply silu
   on the fly (each act element is re-read once per down row-block, so
   the silu would be recomputed `I/BN` times unless the act row is staged
   once in LDS per program — LDS-resident, 4 KiB/slot, feasible).
   Removes the act write+read (32+32 KiB/token ≈ 0.03% of bytes) and one
   launch. Verdict: worth it ONLY as a launch-count play bundled with a
   down-GEMV rewrite; the bytes say do not bother for bandwidth.
4. **Single persistent kernel (stage-0+stage-1 in one launch).** The
   dependency is real: the down GEMV for slot (t,k) needs the FULL act
   row (IA=2048 dots of length 4096) before its first output tile. A
   persistent kernel must either (a) recompute the act row per
   down-row-block program — re-reading that expert's 16 MiB gate/up
   stack `I/BN ≈ 32` times, a 32× bandwidth explosion — or (b) compute
   the whole slot in ONE program (24 MiB weights/slot, grid = t·k = 8
   programs at decode) — the `mhc_pre_gemm_sqrsum` failure mode
   (kernels-reference row 26: 1 block on 228 CUs, 6.4× regression) — or
   (c) cooperative grid sync, which forfeits ordinary graph-capture
   composability and adds a full-device barrier between stages whose
   weight streams could otherwise overlap. **Does not pay at decode M≤2
   on either roof; revisit only if a serving profile ever shows the two
   GEMV stages' weight streams underlapping the memory system** (they do
   not today: 500–800 GB/s issue-bound on MI300A, at-roof on GB10).

**One-line verdict:** fuse the epilogue (done, opt-in), keep the two-GEMV
staging, chase launches and the router seam — not bytes.

## MI300A A/B plan (UNVALIDATED on gfx942 — nothing here has run on the
## serving target; the .hip path cannot even compile on this box)

1. Build (unchanged C++ tree, but gate the baseline libs anyway):
   ```bash
   cmake --preset hip && cmake --build --preset hip && ctest --preset hip
   ```
2. Correctness on gfx942 (the bit-exact chain contract is device-local,
   so it re-validates itself there; triton 3.5.1 on beverin vs 3.8 here):
   ```bash
   VK63_SECTIONS='2. torch_ops*' srun --partition=mi300 -N1 -G1 --time=00:12:00 \
     bash meta/scripts/run_issue63_torchops_mi300.sh
   # plus directly:
   srun --partition=mi300 -N1 -G1 --time=00:10:00 \
     .venv/bin/python -m pytest tests/python/test_glm_expert_gemv_fused.py -q
   ```
   (with the torch-ops-mi300 `Python.h` CPATH bootstrap if the node still
   lacks python3.11-devel).
3. Bench target (same harness, both arms in one job for the honest ratio):
   ```bash
   srun --partition=mi300 -N1 -G1 --time=00:30:00 \
     .venv/bin/python bench/bench_glm_expert_gemv_fused.py --t 1 2
   ```
   plus, if floe-side reproduction is available, an end-to-end decode A/B
   with `GLM53_MOE_SILU_FUSED=0/1` (the knob exists for exactly this).
   Record into `docs/performance/` per repo convention after the run.

**Expected win model on MI300A:** byte saving 0.10% of a ~201 MiB/token
weight stream ≈ 0.2 µs at 1 TB/s — noise. Launch/boundary saving ~3–6 µs
per MoE block per token (dispatch floor ~2.8 µs + silu kernel ~2–4 µs on
160–320 KiB) against ~324 µs/token ≈ **1–2% per MoE layer**, i.e. real
but second-order; the decisive number is whether gfx942's triton shows
any register-pressure regression from the doubled per-program weight
footprint (see risks). If the fused arm regresses at IA=2048/I=4096, the
fallback tuning is `ROWS=2` (restores the unfused kernel's per-program
footprint) — one-constant change, both kernels already parameterized.

## Risks / open questions

- gfx942 register allocation for the two-tile program body is UNVALIDATED
  (SGPR/VGPR budget differs from GB10 SM spill behavior); mitigate with
  the `ROWS=2` fallback above.
- Bit-exactness is asserted per-device by the test; a future triton
  upgrade that changes reduction layout selection can only break it
  loudly (test failure), never silently — that is the contract working
  as intended, but it means CI on BOTH backends before a default flip.
- The wrapper duplicates ~40 lines of validation from `glm_expert_gemv`
  (additive-by-design: the sibling is owned by the issue-#65 contract and
  this lane must not edit it). If a third variant ever appears, factor a
  shared validator — flagged as a likely future refactor point, not done
  now to keep the diff disjoint from siblings.
- Open: does floe call `elementwise.silu_mul` or its own silu on this
  seam? (Determines where `silu_fused_enabled()` gates — the op-level
  contract is identical either way.) Open: MoE layer count in
  GLM-5.3-Flash for a per-token extrapolation of the 1–2% figure. Open:
  whether the router-gather plumbing (lever 2) should live in this module
  or `glm_router.py` — left to the issue, out of scope here.

## Jobs

- GB10 (this box): fused tests 14/14 PASS (incl. bit-exact chain equality,
  fnuz, CUDA-graph replay); sibling `test_glm_expert_gemv.py` still 9/9
  PASS. Bench table above (unlocked clocks — treat absolute values as
  ±10%, ratios as robust).
- MI300A: NONE. Everything above labeled indicative-only; the A/B plan is
  the contract for flipping `GLM53_MOE_SILU_FUSED` defaults, if ever.

## Incidents (so the same footguns aren't re-stepped on)

- Draft 1 collapsed the act width into the input width (`o == 2*i`): fine
  for square test shapes, WRONG for the GLM serving stack ([E,4096,4096]
  ⇒ IA=2048 ≠ I=4096). Fixed to the two-dim contract with asymmetric-shape
  tests before any perf run.
- Draft 2 scale indexing used `expert*2` for the per-expert scale stride;
  correct is `expert*(O//128)` (scales are `[E, O/128, I/128]`). Caught by
  the bit-exact gate (87% mismatch → 0 after fix) — the strong contract
  paid for itself on the first bug.
- `e4m3fn_to_fnuz_inplace` mutates weights AND doubles scales in place;
  "pristine e4m3fn" comparisons must clone BEFORE the conversion (test
  initially cloned after → 2×-scale garbage that looked like a decode
  bug).
- CUDA-graph replay reads the captured STATIC buffers: feeding `x2` as a
  new tensor does nothing; the test must `x.copy_(x2)` before replay.
