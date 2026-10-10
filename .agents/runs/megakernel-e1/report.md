# Megakernel E1 — the KDA-spine slice: compiler model + CPU parity + GB10 span pricing

Status: **landed (uncommitted)**. Repo state: `vkernels @ 6f6de90` + this lane's
edits. Hardware: this laptop IS the GB10 (sm_121, 48 SMs, unified LPDDR) — parts
(1)–(3) all ran locally; no rig, no serving stack, no floe changes.

Readiness doc: `.agents/runs/gap-levers/megakernel-readiness.md` (E1 = Lever E
step 1). Numerics oracle: the uncommitted lane-3
`src/python/vkernels/torch_ops/glm_kda_fused_decode.py` (CUDA kernel + eager
fp32 reference) — its docstrings ARE the contract the megakernel's KDA tasks
now pin.

---

## 1. What landed

### Compiler model (the E1 slice in `src/python/vkernels/compiler/`)

* **New op `kda_fused_decode`** (`capture_recurrent.py`, `contracts.py`,
  `lowerings/recurrent.py`, `reference_recurrent.py`): the KDA decode block as
  ONE task family — conv update + element-wise-decay delta rule +
  sigmoid-gated per-head RMSNorm — capturing the **fused-decode contract** of
  the CUDA oracle verbatim in the op's numerical contract:
  * **RAW dot ABI**: consumes the RAW pre-conv fused q|k|v row and the RAW
    `f_b`/`b_proj`/`g_b` dots; they are rounded to the **bf16 grid at task
    entry**, and every gate nonlinearity (beta, o-norm gate) applies
    **after** the round (sigmoid-after-bf16-round);
  * conv: fp32 **time-major taps `[Kt, Cc]`** over the bf16-valued window;
    SiLU output **not rounded** before the recurrence; pool shift = pure
    bf16 moves; floe's conv1d bias-free;
  * **state-pool access pattern**: ssm pool **V-MAJOR `[slots, H, V, K]` fp32**
    (the transpose of the decomposed `kda_delta` op's `[B, H, K, V]` — layout
    only), conv pool **w-major `[slots, Kw, Cc]`**; both **slot-indirected**
    through an external i32 `[B]` table (`Region.indirect`, the #94 pattern)
    with the fused kernel's own slot contract: `-1` = padded CUDA-graph slot
    (zero output row, pools untouched), duplicate live slots = UB;
  * one task per (batch row, head) — the CUDA kernel's CTA decomposition;
    task-level pool regions are the conservative whole-pool slab (sound under
    phase order; exact under slot disjointness), precise stripes on the row
    tensors.
* **`Glm53KdaSpineArgs` + `build_glm53_kda_spine_forward`** (`model_glm53.py`):
  the E1 spine body — per KDA layer
  `mhc_pre → ln1 → qkv/f_a/f_b/b/g_a/g_b GEMVs → kda_fused_decode → o_proj →
  mhc_post` (11 ops = 11 phases/layer; a 34-layer spine = 374 phases = 374
  in-kernel barriers, one launch). Existing whole-step frontend untouched.
* **fp64 fused-contract mirror** (`glm53_arch.py`):
  `kda_fused_decode_reference` (one layer's fused op, explicit bf16 rounds)
  + `glm53_reference_kda_spine_step` (the chained spine). The existing
  decomposed mirror (`glm53_reference_decode_step`) untouched.
* **`bf16_round`** (`reference_types.py`): numpy round-to-nearest-even to the
  bf16 grid — verified **bit-exact vs torch `.to(torch.bfloat16)`**; this is
  the ABI-round primitive both the executor body and the mirror use.
* **Triton device template `_t_kda_fused`** (`triton_recurrent.py`): the fused
  contract as a device task body (same ABI/rounds; V-major pool; padded
  slots), GB10-attested below.

### Tests (`tests/python/test_megakernel_glm53_e1.py`, 10 new, torch-optional)

Structure (11 ops/layer, disjoint slot pools, RMW effects, taps ABI) ·
recurrence (layer i's `mhc_post` RAW-feeds layer i+1's `mhc_pre` — the
hazard edge the phase barrier gates; phases op-ordered; 1 simulated launch
covers the spine) · parity vs the fused-contract mirror across
workers {1,3} × batch {1,3} and **two chained decode steps** (pool RMW
across launches, TOL 1e-6) · padded-slot contract · **cross-check vs
`glm_kda_fused_decode_reference`** (torch, CPU) · fused-vs-decomposed
divergence band. The pre-existing 11-test `test_megakernel_glm53.py` and the
adjacent megakernel suites still pass (159 passed / 9 skipped in the family
sweep).

### GB10 bench (`bench/glm53_e1_spine_bench.py`)

A hand-assembled **spine megakernel** (`glm53_kda_spine_megakernel`, the
`device_triton_hybrid.py` pattern at floe's REAL per-rank TP4 dims: L=34,
C=4096, H=16×128, hc=4, Sinkhorn 20) reusing the compiler's attested task
bodies (`_t_mhc_pre/_t_mhc_post/_t_rms2d/_t_gemv`) + `_t_kda_fused`, with
independent GEMV phases coalesced (11 op-phases → 7 barriers/layer = 238 for
34 layers). Gates before timing; launch counts via torch.profiler (§15.2);
spans via CUDA events.

## 2. Parity status

| check | result |
|---|---|
| compiled spine (CPU executor) vs fp64 fused-contract mirror | **1.0e-7** max abs (fp32 pool-store class), workers {1,3} × batch {1,3}, 2 chained steps |
| `kda_fused_decode_reference` vs **`glm_kda_fused_decode_reference`** (torch fp32 oracle) | **bit-exact** on the bf16-grid out AND conv pool; ssm 3.3e-7 (fp32-ULP) — with a `-1` padded slot in the fixture |
| compiled op alone (capture→lower→schedule→execute) vs the torch oracle | out on-grid (≤2% elements off by one bf16 step from fp32-vs-fp64 tie flips), conv bit-exact, ssm fp32-ULP |
| GB10 megakernel vs eager decomposed chain (bench gate) | streams rel **1.8e-2**, ssm abs 2.8e-2, chained step-2 rel 2.2e-2 — inside the documented band |
| barrier soak | 502 steps, monotonic counter exactly tracks (no drift) |

**Documented intentional divergence** (fused ABI vs the round-free decomposed
chain, asserted ≤5e-2 in-test and measured ~2e-2): the bf16 entry rounds on
the four RAW dot streams (sigmoid-after-round for beta/gate), and the out-row
bf16 store. The conv-output-unrounded choice is *shared* by both compiler
paths (fp32 workspace). Non-divergences pinned: V-major vs K-major pool is a
layout transpose; o_norm `[V]`-shared-across-heads matches both floe and K3.
The CUDA kernel's fast-math intrinsics (`__expf`/`log1pf`) are NOT modeled —
the docstring's ~1e-2 cross-path class covers them.

## 3. The span comparison (GB10, B=1, random weights, real per-rank TP4 dims)

| path | min µs/step | median | launches/step |
|---|---:|---:|---:|
| A  eager decomposed chain (floe's class) | 41 809 | 42 238 | 6 868 |
| A' CUDA-graph replay of A | 21 856 | 24 016 | 6 868 |
| B  fused-CUDA chain (lane-3 kernel) | 59 215 | 60 084 | 5 475 |
| B' CUDA-graph replay of B | 20 219 | 20 239 | 5 475 |
| **M  spine megakernel (P=48, 4 warps)** | **14 995** | **15 025** | **1** |

**E1 decision rule: PASSED on GB10** — M beats the graph-of-kernels baseline
A' by **+6.86 ms/step** (and B' by +5.22 ms), far above the +0.3 ms funding
threshold, with parity gates green.

Honest decomposition of the win:

* Launch tax on this box: A−A' = 19.95 ms over 6 868 launches ≈ **2.9 µs per
  launch** (GB10's weak host CPU; the H100 census number was 0.74 µs). The
  graph already removes that — M's win is **not** the launch tax.
* Device efficiency: M's tiled GEMV tasks beat cublas bf16 `F.linear` at
  these skinny shapes on GB10 (qkv GEMV: torch 300 µs vs Triton tasks ~70 µs
  L2-hot / TILE-dependent). The decisive tuning: **TILE=64 bf16 tiles** (a
  full 128B transaction per weight row-segment). TILE=16 = 32B segments =
  2× read amplification: measured 410 → 1041 GB/s effective on the qkv shape;
  that single change took M from 27.0 → 15.0 ms/step. The spine is
  bandwidth-bound on GB10 (~2.65 GB/step of weights ≈ 10 ms at LPDDR rate);
  M sits ~1.5× above that floor, A' ~2.2×.
* Barriers are NOT the cost: 0.8 µs each at every P (240-round milestone-0
  soak), 238/step ≈ 190 µs total. B=1 mHC phases (single-task) cost ~11
  µs/layer — also minor.
* B (lane-3 fused chain) is slow *eager* (59 ms) purely from per-call host
  work (taps slicing/contiguous, dispatch); its graph B' is strong.

## 4. Caveats + risks (what this does NOT show)

1. **GB10 ≠ the serving target.** Single rank, TP1-shape dims, B=1, random
   weights, no ARs, no FFN/MoE blocks (this slice prices the KDA attention
   sub-blocks only — the readiness E1 scope). On H100 TP4 the arithmetic
   differs both ways: launch tax is 4× smaller, but HBM is 12× faster — the
   GEMV-task efficiency win should matter *more*, the launch-tax win *less*.
   Re-run before any E2 claim transfers.
2. **Triton 8-warp mis-execution found and dodged, not root-caused:**
   `_t_kda_fused`'s `[V, K]`-tile reductions are bit-exact vs the fp64 mirror
   at **num_warps=4** but WRONG at 8 on this triton/GB10 stack (out err
   6.9e-1). The bench launches the persistent kernel at 4 warps (commented).
   Root-cause (or restructure the tile math) before sharing the template
   across warp configs.
3. **The megakernel assembly is bench-local**, not a compiler backend:
   `compile.py` still has no GLM device path. The compiler's op-level schedule
   emits 11 phases/layer; the bench hand-coalesces independent GEMV ops into
   shared phases (7/layer). Phase coalescing of independent ops is a
   compiler-side follow-up (schedule-level, §10 territory), as is making
   TILE=64 the dense-gemv lowering default for skinny bf16 GEMVs.
4. **Cross-step recurrence is across launches** (one launch per decode step,
   pools persist in device memory, barrier base advances on host). The
   multi-step-in-one-launch variant (step loop inside the kernel, wrap-around
   stream dependency) is untested.
5. Conv-pool values ride fp32 storage with bf16-grid values (the compiled-pool
   convention); the lane-3 kernel wants a native-bf16 pool — a boundary
   conversion or a pool-dtype decision is needed at the serving seam.

## 5. Next-slice decision

E1's rule passed on the only GPU available → per the readiness plan,
**fund E2 (the floe opt-in lane) and start E3 (persistent fp8 MoE research)**.
Recommended E2 shape, given this slice's findings:

* **E2a — per-layer segments first** (`FLOE_GLM5_KDA_MEGA=1`, the
  `dispatch.py:_opt` pattern): one megakernel launch per KDA layer's
  attention sub-block (7 barriers), replacing ~200 torch launches/layer of
  that block. Under TP4 the segment naturally **ends at the attn-block
  all-reduce boundary** (the readiness §6 fence constraint); the FFN sub-block
  stays on the existing path until E3. This needs: moving the bench assembly
  into a `device_triton_glm53.py` backend (the `qwen35_hybrid_megakernel`
  slot), the GLM pool layout at the kvaas seam (readiness gap #4), and the
  floe greedy-parity gates (argmax over ≥100 prompts) before any default-on.
* **Not E2-whole-spine yet**: the spine chains attention blocks directly —
  serving can't skip the interleaved FFN blocks, so the 34-layer single
  launch only exists inside E3's whole-step kernel.
* **In parallel**: root-cause the 8-warp Triton issue; port TILE=64 + phase
  coalescing into the compiler lowerings/scheduler so the compiled schedule
  matches what the bench hand-built; re-run this bench on an H100 when
  available before quoting serving deltas.

## 6. Files

Compiler: `capture_recurrent.py` (op), `contracts.py`, `lowerings/recurrent.py`
+ `lowerings/__init__.py` (task family), `reference_recurrent.py` (executor
body), `glm53_arch.py` (fp64 fused mirror + spine mirror), `model_glm53.py`
(spine args + builder), `reference_types.py` (`bf16_round`),
`triton_recurrent.py` (`_t_kda_fused` device template).
Tests: `tests/python/test_megakernel_glm53_e1.py` (10).
Bench: `bench/glm53_e1_spine_bench.py`.
All uncommitted; other lanes' untracked files untouched; the pre-existing
`test_megakernel_glm53.py` 11/11 still green.
