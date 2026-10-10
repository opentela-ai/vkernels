# LANE 26 — elementwise rungs 2-3 (floe-side eliminations L1 + L7-g26)

Lane 26 of the elementwise program. Lane 21 (elementwise-fleet) priced the
L1–L11 ladder; lane 21 itself landed rung 1 (L2, `dsa_tail_scatter`). This
lane took the next two rungs by (impact × certainty)/effort and implemented
them as **floe-side wirings** — both chosen rungs are *eliminations*, and
the ladder's own prescription for each is "floe" (no new kernel; the
delivery is deleted launches, bit-identical outputs, and wiring notes for
the consolidation pass).

Repo state: floe @ b72a1de (working tree shared with in-flight lanes —
my edits are targeted hunks in two tracked files; nothing committed, no
pushes). vkernels source is untouched by this lane. New files are disjoint
from every other lane's (ledger at the end).

## 1. Rung selection

| candidate | nodes/step | class | effort | verdict |
|---|---|---|---|---|
| **L1** constant-true KDA mask-mul drop | −68 | bit-identical (x·1.0 ≡ x, all bf16 bits) | floe, ~10 lines + tests | **taken (rung 2)** — biggest certainty-weighted win on the ladder; ladder's own sequencing: "L1 first, cheapest big win" |
| **L7 (g26 half)** dead MoE tail `zeros_like` | −42 | bit-identical (dead-code motion: value never read on taken paths) | floe, 3 lines | **taken (rung 3)** — "fill-free zero-init" hint; pure dead work on every served shape |
| L3 router int64 emit | −42 | bit-identical (integers) | vkernels+floe | skipped: `glm_router.py` is modified by the in-flight router-shared lane, and node-cuts lever-A's companion rewrites this exact emit — spoken for |
| L6 kda out-cast into kernel | −34 | bit-identical (double-round no-op) | vkernels+floe | skipped: lives in the kda decode call-site cluster the kda-fused-decode lane owns (their untracked `glm_kda_fused_decode.py` + kda wiring tests are in flight) |
| mHC compose/mix/collapse merges (G0) | — | — | — | skipped: node-cuts marks G0 banked/in-flight, "not this lane's work" |
| k_norm LayerNorm chain (node-cuts further-headroom) | −11 measured, not ~40 | — | new kernel | skipped and **re-audited**: the census shows exactly ONE `vectorized_layer_norm_kernel<BFloat16>` per DSA layer = 11 nodes/step (32.6 µs), not the "~−40" the note sketched. A gemv+LN fused op is a real but small future rung; correcting the estimate here so a later lane doesn't chase 4× the headroom |

## 2. Verified pricing (code anchor + census + GB10 bench)

Census: `.agents/runs/gap-probe-data/floe-b1-summary.json` (508-step window,
whole-window totals ÷ 508; the fleet report's per-graph attribution differs
somewhat on the copy row — node counts are the solid currency).

### L1 — KDA constant-true mask mul (g26 census classes #7 + #8)

Code anchor: `floe/engine/runner/models/glm53flash/forward.py`
`_apply_mask_to_padding_states` (fwd.py:1448), called once per KDA layer
forward (fwd.py, in `Glm53LinearAttention.forward`) → 34 layers/step, each

```python
return x * mask.unsqueeze(-1).to(x.dtype)      # bool→bf16 cast + broadcast mul
```

Census rows reconciled:
- mul: `elementwise_kernel<128,4, gpu_kernel_impl_nocast<BinaryFunctor<BFloat16,…` — 34.0/step, 75.23 µs/step (matches the ladder's #7 = 75.2 exactly)
- cast: `direct_copy_kernel_cuda` (instance 3) — 34.0/step, 60.28 µs/step whole-window (ladder's per-graph attribution said 30.6; same 34 nodes)

Every decode producer of that mask is a capture-static ALL-TRUE constant,
verified by reading all five producer sites:
- `graphs.py:1687` + `graphs.py:1962` — merged graphs' `self.mask` (never written in place: no `mask.fill_/copy_/setitem` anywhere in graphs.py)
- `graphs.py:377/490` — stage masks via `_hoisted_full(..., True)` (never-evicted `_STATIC_CONSTS` buffers)
- `forward.py` `Glm53TextModel.forward` — default `torch.ones` mask (all-true by construction)

Correctness argument: `x * 1.0` is the exact identity for every bf16/fp32
bit pattern (NaN payloads, ±inf, −0.0, denormals keep their bits), and the
mask contributes no pad-zeroing today (it is all-true even for padded
bucket rows — pads are handled elsewhere), so deleting both launches is
bit-identical for every row. The indexer/glue consumers still receive the
real mask object (untouched code).

GB10 microbench (`meta/benchmarks/bench_elementwise_rungs2_3.py`, graph
replay, bs=1·S=1·H=4096 bf16): the pair costs 2.08 µs/layer device-side;
×34 ≈ 70.7 µs/step + 68 × 0.74 µs node-gap ≈ **~0.12 ms/step all-in** —
consistent with the ladder's −0.106 ms (H100 rig).

### L7 (g26 half) — dead MoE tail fill (g26 census class #1, g26 share)

Code anchor: `Glm53Experts.forward` materialized `final = torch.zeros_like(x)`
at ENTRY, but the only reader is the per-expert loop fallback
(`final.index_add_`, fwd.py:2224-2230 old numbering). Every decode/prefill
fast path returns before the loop: fp8_grouped (:2094 return), aiter,
batched_experts, sgl_fused_moe, grouped_moe, expert_gemv, and the einsum
gather block — all `return` without touching `final`. The deployed decode
recipe (grouped_moe/expert_gemv) therefore paid one dead
`FillFunctor<BFloat16>` per MoE layer per step:

- census: 64.0 fills/step, 67.26 µs/step total; this site's share = 42 nodes ≈ 44 µs/step (+ a graph-pool allocation per layer under capture)

Correctness argument: none needed beyond scoping — the moved constructor
is value-identical (`zeros_like(x)` at the same dtype/device, `x`
unmodified inside the loop branch) and every previously-taken return path
never observed it. The loop branch keeps its exact incumbent math.

GB10 microbench: 1.38 µs/fill deleted ×42 ≈ 57.8 µs/step + 42 × 0.74 µs
gap ≈ **~0.09 ms/step all-in**.

## 3. What changed (all uncommitted)

**floe/engine/runner/models/glm53flash/forward.py**
- `_hoisted_full` (:1499): all-True fills tag their never-evicted buffer
  `buf._floe_const_true = True` (:1512) — the sentinel producer.
- `_apply_mask_to_padding_states` (:1448): skips the mul for sentinel
  buffers. NOTE: a concurrent lane wrapped this in the OPT-IN
  `kda_mask_elide` dispatch knob (dispatch.py:195-203, default off) as its
  A/B gate — see §5; the tagging and the skip body are this lane's, the
  gate is theirs.
- `Glm53TextModel.forward` (:~5003): the internally-constructed default
  mask (all-true by construction) is tagged. Caller-supplied masks are
  never tagged.
- `Glm53Experts.forward`: entry-side `final = torch.zeros_like(x)` moved
  into the per-expert loop branch (its only consumer), with the census
  annotation (:2224-2230).

**floe/engine/runner/models/glm53flash/graphs.py**
- `:107` import `_hoisted_full`; `:1687` and `:1962`: the merged graphs'
  `self.mask = torch.ones((Bb,1))` → `_hoisted_full((Bb, 1), dev,
  torch.bool, True)` (same values, same never-evicted lifetime, now
  sentinel-tagged). The `debug_dump` validation chains (:2495/:2545) keep
  fresh untagged `torch.ones` — eager-only, zero replay value, and they
  serve as live incumbent oracles.

## 4. Tests (GB10, floe/.venv)

New (this lane, disjoint names):
- `tests/test_glm53_const_mask_skip.py` — 13 tests: producer tagging
  (`_hoisted_full` tags True-fills only, False/float fills untagged);
  helper contract (sentinel → aliased input; None pass-through; untagged
  masks keep the incumbent formula bit-for-bit incl. NaN-at-pad mul
  semantics and the non-bool cast branch); knob-off gate (default
  dispatch keeps the incumbent mul even for tagged masks); alias-safety
  through the real KDA layer on CPU and CUDA bf16 (the captured graphs
  re-read the static window next step — an in-place downstream write
  would corrupt it; pinned clean); KDA layer tagged-vs-untagged outputs
  bit-identical (CPU fp32 + CUDA bf16, S=1 and S=3); full tiny-model
  token-by-token decode with the tagged default mask vs explicit
  untagged mask bit-identical; merged-graph producer source pin.
- `tests/test_glm53_moe_tail_fill.py` — 4 tests (CPU tiny config,
  counting `torch.zeros_like` wrapper): T=1 and T=2 (decode cap) fast
  paths make ZERO zeros_like calls and are bit-identical to the
  pure-torch gather oracle; T=3 loop path still constructs its
  accumulator and matches a dense fp64 recompute to fp32 round-off; the
  CUDA-capture guard still raises BEFORE the fill is constructed.

```
.venv/bin/python -m pytest tests/test_glm53_const_mask_skip.py \
    tests/test_glm53_moe_tail_fill.py -q     # 17 passed
```

Regression (tracked suites over the touched code, all green):
- `test_glm5_arch.py` — 14 passed (CPU full-model HF cross-check + prefill/decode self-consistency: pins both rungs against the transformers reference)
- `test_glm53_shared_expert_fusion.py` + `test_glm53_stream_overlap.py` — 34 passed (MoE tail/fork paths)
- `test_glm5_decode_graphs.py` — 3 passed (GPU captured-graph parity with the tagged hoisted mask)
- `test_glm53_graph_bank_bucket8.py` + `test_cuda_graph_dispatch.py` — 32 passed

Microbench: `vkernels/meta/benchmarks/bench_elementwise_rungs2_3.py`
(meta/benchmarks conventions: argparse `--output`, parity gate before
timing, CUDA-graph replay medians, JSON artifact banked at
`vkernels/.agents/runs/elementwise-rungs2-3/bench_rungs2_3.json`).
Device: NVIDIA GB10, torch 2.13.0+cu130.

## 5. Floe wiring notes for the consolidation pass

- **L1 deployment gate**: the skip is behind `--model-opt kda_mask_elide=true`
  (`Glm53DispatchConfig.kda_mask_elide`, default off — added by a
  concurrent lane as the A/B gate; the field participates in
  `fingerprint()`, so warm state is namespaced per knob value and
  capture/replay can't straddle a flip). The sentinel TAGGING is
  unconditional, so knob flips are pure dispatch, no re-capture hazards
  beyond the fingerprinted namespace. Default-on after the rig A/B banks
  −68 nodes/step with zero numerics exposure (bit-identity is argued and
  tested, not measured-tolerance).
- **L7 needs no knob**: the dead-fill move is unconditional and
  knob-independent; nothing to flip, nothing to A/B (the served shapes
  never read `final`).
- Call sites touched (for the consolidation diff): fwd.py:1448 (skip),
  :1499/:1512 (tag), :~5003 (model default tag), :2224 (fill), graphs.py:107,
  :1687, :1962. Remaining untagged all-true producers: `debug_dump`'s two
  `torch.ones` masks (graphs.py:2495/:2545) — deliberate (eager validation
  oracles).
- Adjacent census classes left on the table (next rungs by the same
  ranking): L4 hoisted-arange at the ragged site (−11, one-liner), L8
  glue nan-guard (−11, kernel edit in `glm53_indexer_glue`), L5
  flat-index selects (−33), L3/L6 after their owning lanes land, and the
  k_norm gemv+LN fusion (−11 measured — see the corrected estimate in §1).
- The routed+shared add (census #5, CUDAFunctor_add 42/step, the other
  half of ladder L7) folds only via lever A (fused_shared_expert) or a
  grouped-kernel epilogue — NOT by an in-place add (same node count);
  it belongs to the router-shared lane's landing.

## 6. Disjointness ledger (this lane's files only)

New: `floe/tests/test_glm53_const_mask_skip.py`,
`floe/tests/test_glm53_moe_tail_fill.py`,
`vkernels/meta/benchmarks/bench_elementwise_rungs2_3.py`,
`vkernels/.agents/runs/elementwise-rungs2-3/` (this report + bench JSON).
Modified (tracked, targeted hunks): the two floe files above. No vkernels
source changes; no other lanes' untracked files touched
(dsa_tail_scatter / glm_indexer_topk / glm_kda_fused_decode /
dsa_kpool_compress / sgl_moe untouched). Nothing committed or pushed.
