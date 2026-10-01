# ROWS-REPORT — `resolve_rows` Triton resolver (lane `beverin/rows-resolver`)

**Commit:** `126e44a1c682c8ed69b1d1283878ddc01177f793` on `beverin/rows-resolver`
(worktree `../vkernels-rows`, from vkernels `bf78366`)
**Files:** `src/python/vkernels/torch_ops/dsa_resolve_rows.py` (kernel + launchers + CPU oracle),
`tests/python/test_dsa_resolve_rows.py` (bit-exact parity harness, on-cluster)

---

## 1. Root cause of the 828 µs (× 11.4 calls = 9.11 ms/step = 20.9% of the 43.49 ms B=1 step; 8.99 ms at b4)

The production resolver is the kvaas scalar-scan Triton kernel
(`kvaas_runtime/sparse_attention/kernels.py::resolve_rows` — the pre-W6 kernel restored as
default after the W6 map rewrite was perf-REJECTED, 2.5–4.4× slower; `KVAAS_W6_MAP=1` opts back in).
The floe call chain is:
`forward.py` DSA layer → `cache.sparse_swap(layer_idx, topk)` → `runner.py:328` →
`SparseStep.swap_in` (`sparse_runtime.py:1837`) → `self._resolvers[g].resolve(self._selected)`
→ one `resolve_rows[(batch, 1)]` launch per DSA layer, ×11 per step.

It is **not** host sync, launch overhead, or H2D — the W8A census shows it as device-busy
kernel time. The cost is in-kernel compute shape:

1. **Scalar K-walk.** The selection pass is `for i in range(K)` with K=2051 (constexpr ⇒ fully
   unrolled): 2051 *serialized scalar* `tl.load`s (loop-carried dependency, unvectorizable),
   each followed on the miss branch by TWO full BLOCK_H-wide reductions (hot-hit scan + LRU
   victim scan).
2. **The O(H·K) compare storm.** Every iteration evaluates `tags == token` /
   `tag_generations == generation` over BLOCK_H = next_pow2(hot_rows=2176) = **4096** int64
   lanes, and the LRU protect pre-pass pays the same storm again over all K:
   ~2051 × 4096 × ~4 lane-ops of pure integer compare per (request, layer) state — the exact
   "~1.9G lane-ops per state" the kvaas W6 header documents.
3. **Register pressure.** Three int64 BLOCK_H vectors (tags / tag_generations / ages) live
   across the whole walk; kvaas records 1240 spill instructions at BLOCK_H=16384 and needs a
   tuned num_warps heuristic (block_h//512, clamped 4..16) to keep 4096 from spilling.

The whole op is integer/address math that should be memory-bound (~tens of µs of gather
traffic) — hence "16× headroom on a single op".

## 2. The replacement (fresh Triton — NOT the rejected W6 map design)

`dsa_resolve_rows.py` keeps the production state machine bit-exact but restructures the work:

- **Phase 1 — vector resident fast path.** The K selection is processed in BLOCK_K=1024-wide
  chunks: one coalesced token load, page-table gather (`RESIDENT[page]`), row arithmetic
  `resident*PAGE + tok%PAGE`, direct output store, error count, and an **ordered miss-worklist
  compaction** (`cumsum`). The ≥90%-hit resident path never touches the hot-tier state machine.
- **Phase 2 — protect via bitmap.** The K×H compare storm becomes O(K+H): K vector
  `atomic_or`s mark valid selected tokens in a token-domain bitmap (`WORDS = pages*PAGE/32`
  int32 words per state), then ONE H-lane pass gathers the bitmap word for each lane's tag and
  tests it against the page generation — the identical protected set as the production pre-pass.
- **Phase 3 — scalar miss walk, transliteration-faithful.** The production miss branch runs
  scalar-by-scalar over the compacted worklist only (hit ~0.9 ⇒ **0–8** iterations vs 2051):
  lowest-lane hot hit, min-age/lowest-lane victim among unprotected, fence gating, fill-pair
  records, in-walk `tags`/`tag_generations` patches (so a later duplicate of a just-missed
  token HITS the reserved slot), ages/clock bumps — bit-exact by construction. Worst-case
  degenerate (100% miss) equals the old cost; serving steady state is 0–8.
- **Phases 4–5 — verbatim** epilogue (state + counters stores) and host→hot copy pass.
- **`dsa_resolve_rows_batched`** — same kernel, `grid=(batch, layers)`, `selected` stacked
  `[L,B,K]`: one launch for all 11 layers. Honest constraint: layer L's selection only exists
  mid-forward (layer L's attention consumes layer L's rows), so drop-in batching requires the
  two-pass restructure (all indexers → one resolve → all attentions). The per-layer calls at
  the target kernel speed already meet ≤0.7 ms.

**Expected:** memory-bound ~50–60 µs/state at production geometry (topk 2048, hot_rows 2176)
⇒ 11 × ~60 µs ≈ **0.66 ms/step vs 9.11 ms** (13–14×; lands the 9.1 → ≤0.7 ms target without
the call-site restructure).

### Row-set-unchanged cache — ASSESSED, REJECTED (do not arm)
(a) Selections are not stable step-to-step: each decode step runs the indexer for a NEW query
token over a grown pool — the ordered top-k set changes (row output is selection-ordered).
(b) The resolver is stateful: tags/ages/clock advance every resolve (clock=old+1; miss fills
re-tag hot rows; ages bump). Skipping a resolve leaves the LRU state stale, so later victim
choices diverge → NOT bit-exact with the incumbent.
(c) The check itself (content-hash the int64 selection) needs a D2H sync per layer — forbidden
in the captured-graph path and ~100+ µs of serialization for a hit rate that doesn't exist.

## 3. Parity test
`tests/python/test_dsa_resolve_rows.py` — follows the `tests/python/test_elementwise.py`
pattern (CPU semantic cases everywhere; GPU leg `pytest.skip` without CUDA; runs on-cluster):
`torch.equal` (bit-exact, no tolerance) on outputs, counters, errors, fill-list prefix AND the
advanced tags/tag_generations/ages/clock state; cases: randomized sweep (dup tokens, hot/resident
mix, invalid tokens, ALLOW_PADDING), victim ties, fence mismatch (masked miss + nonzero errors,
no fill), boundary pages (`length = pages*PAGE`), zero-length rows, non-pow2 topk (2051),
topk=2048 production geometry, multi-layer `dsa_resolve_rows_batched` state indexing, and
OpNotEligible contract-miss fallback. Status: **py_compile/ast clean locally; on-cluster
run (job 656560, MI300A): 1 FAILED — StopIteration in `test_reference_resident_rows_and_counts`
was a HARNESS capacity bug in `_build_state` (the leaseable slot pool, `hot_pages+2` slots,
could be exhausted by the resident roll before the hot-slot `next()` selector ran — the
failure fired during state construction, BEFORE any reference-vs-kernel comparison, so it
carries no kernel signal; the harness has no reference/kernel generator pairing to desync).
FIXED: pools are now sized so exhaustion is impossible by construction (gpu_slots =
batch·pages + batch·hot_pages + 2; resident rolls capped at gpu_slots−1−batch·hot_pages so the
hot capacity is always reserved; host capacity batch·pages+1 covers every non-resident page),
validated over 800 parameter/seed combos off-device; the no-seeded-tag corner of the miss/fill
test gained a seed-retry guard. No kernel or reference change was needed for that one.

On-cluster round 2 (job 656573, fixed tree): 4/5 — `test_gpu_parity_bit_exact` failed on a
REAL comparison mismatch, root-caused to UNDEFINED fills-TAIL bytes, not walk divergence:
the test compared `fills[:, :m]` with `m = max over batch of miss counts`, but each request
only defines `fills[b, :m_b]` (`m_b = counters[b,2]`); `fills` was `torch.empty` on BOTH
sides (kernel launcher, batched wrapper, reference), so rows `[m_b, m)` were garbage-vs-
garbage whenever batch requests had different miss counts — exactly matching the symptom
(fills matched through entry m_0, diverged right after, while out/counters/errors — fully
overwritten every call — were bit-exact). Aggravator: the single-op launcher caches the
fills buffer in `scratch` across calls within a decode step, so tails could hold STALE
fills from a previous call. FIX (commit 11cb5c2 follow-up): (1) kernel epilogue zeroes the
fills tail `[miss_count, K)` in the same copy-pass loop (per state; makes the buffer
deterministic across scratch reuse; cost ~K/FILL_CHUNK extra masked stores); (2) reference
switched to `torch.zeros`; (3) test upgraded to FULL-tensor `torch.equal(fills, ref)` —
stronger than the old max-slice (tails now defined on both sides). The walk itself
(output/counters/errors/tags/ages all bit-exact in 4/5 passing legs) needed NO change.

On-cluster round 3 (job 656618): fills fix VERIFIED (fills/out/counters/errors bit-exact);
next peel failed at the advanced-LRU compare — kernel `tags` had FEWER writes than the
oracle (regressed to pre-loop values except where the input already matched). Root cause:
**Triton lost the loop-carried tensor yields for `tags`/`tag_generations`, which were
mutated THREE scf levels deep** (`if slot<H` → `if fenced` → `else` of `if is_hit`) inside
the dynamic-trip-count walk. Proof by elimination: fills/out/counters bit-exact ⇒ same
evict slots ⇒ the tag-write (slot→token) pairs were identical; `protected`/`ages` (mutated
at depth 2) demonstrably persisted (fills equality REQUIRES their in-loop evolution);
scalars at depth 3 (`hit_count`) persisted; only the depth-3 tensor mutations vanished.
FIX: flattened ALL carried-tensor mutations to loop-body top level as predicated
`tl.where`s (`is_evict & (h == slot)` etc. — all-false when unclaimed, so safe no-ops);
claim/fence resolved as scalar predicates with masked scalar loads; only side-effect
stores (fills) remain in a scalar `if`. Semantics transliteration unchanged. ALSO fixed a
LATENT oracle divergence found while bisecting: the reference was missing the production
PROTECT PRE-PASS (kvaas kernels.py:356-363 — lanes tagged with a valid-selected token at
current page generation are immune to eviction before walked; the kernel builds the same
set from the selected-token bitmap). Added to the oracle; without it, a pre-protected lane
evicted early by the oracle would turn a later guaranteed hit into a miss — a counts/fills
divergence waiting to fire in the random sweep. Awaiting re-ship + re-run.

On-cluster round 4 (job 656668, tree babca79 + harness randrange fix): tags fix VERIFIED
(tags/tag_generations/ages/clock bit-exact — the flattened predicated-carry form works);
next peel failed at fills AGAIN. NOT the walk and NOT the new reference pre-pass: a 30k-
state pure-Python fuzz of the committed reference vs the committed kernel algorithm found
ZERO divergence (out/counters/errors/fills/tags/tgs/ages/clock all equal, dup tokens,
invalid tokens, boundary lengths, no-victim states included). Root cause was in MY 5e1e45d
epilogue: the §5 fills TAIL-ZEROING stores were masked only by `jv >= miss_count` — the
`jv < K` upper bound was MISSING. The last FILL_CHUNK tile runs to ceil(K/32)*32-1
(29 lanes past K for k=2051, 31 for k=257 — EVERY GPU case), so state s's epilogue wrote
garbage zeros into state s+1's first fill rows (cross-CTA race vs that state's in-walk
fill stores — hence flaky, and why 656618's fills passed on timing luck), and for the last
state ~464B of zeros landed PAST the fills tensor (allocator-layout-dependent corruption).
This exactly matches the failure shape: only fills diverge, everything the kernel fully
overwrites within bounds stays bit-exact, ref shows more fill pairs. The randrange fix
(3d07037) merely reshuffled which random state exposed the race — the OOB write fired on
every run of every GPU case since 5e1e45d. FIX: zeroing store masks are now
`(jv >= miss_count) & (jv < K)`; audited every other vector store in the kernel for the
same class of bug (bitmap clear, classify pass, walk stores, KV copy — all correctly
bounded). Awaiting re-ship + re-run.

On-cluster round 5 (jobs 656671/656672/656673, tree bdddd14 = 80ef6d4 + harness fixes:
unseeded torch global RNG in _build_state pinned via torch.manual_seed — real flake,
generations/kv fed by unseeded RNG — plus multi-layer host-pin fix and full-failure-list
sbatch): 5/7 — BOTH remaining failures are the GPU tests, BOTH dying INSIDE the reference
before the walk: `IndexError: invalid index of a 0-dim tensor` at the
`clock_b = int(clock[b]) + 1` line. NOT a semantics bug (CPU suite green; the 30k-state
fuzz already cleared the walk transliteration) — a SHAPE bug in the reference's entry
handling: the head-of-function `.long().cpu()` conversions cover device/dtype but not
rank, and a per-layer caller can hand the reference a 0-dim (scalar) `clock`/`lengths`
(e.g. `state["clock"][L]` on a 1-D per-request tensor); `clock[b]` on a 0-dim tensor is
exactly torch 2.9's "invalid index of a 0-dim tensor". FIX: entry normalization now does
`torch.atleast_1d(...).reshape(-1)` for lengths and clock (scalar inputs are broadcast
across the batch — clock's expanded view is materialized with .clone() since the walk
writes `clock[b] = clock_b` back), and the advanced clock is reshaped to the INPUT's
shape on return so shape-sensitive compares (kernel-shaped [1,batch] vs reference)
stay intact. Semantics of the walk untouched. py_compile + ast clean on the final tree.
Awaiting re-ship + re-run.

On-cluster round 6 (job 656677, tree 9a417b2): SAME error class, THIRD site —
`t_b[slot] = token` inside the walk; t_b was [1, hot_rows] because the GPU tests pass
KERNEL-RANKED state (tags/tag_generations/ages arrive [layers, batch, hot_rows] after
.repeat(layers,1,1)) while the oracle contract is 2-D. Per instruction, replaced the
one-index-at-a-time fixes with a COMPREHENSIVE rank audit + shim. Contract ranks
enumerated: resident/backing/generations/hot_slots/selected and tags/tag_generations/
ages = [batch, ...] (rank 2); lengths/clock = [batch] (rank 1); fence pair = [gpu_pages]
(rank 1, per-page — never batch-broadcast). EVERY input now goes through one `_norm`
shim at entry: .long().cpu(); squeeze a SINGLETON leading dim (kernel-ranked
[1, batch, ...]); broadcast a scalar across the batch for the rank-1 per-request
contracts; validate leading-dim == batch for every non-resident input (batch is derived
from resident); FAIL LOUD (ValueError naming the tensor) on anything else — no more
silent misindexing. `_restore` gives input-rank-consistent outputs for the four mutated
in-outs (tags/tag_generations/ages/clock) — a scalar clock broadcast across batch>1
stays [batch] (per-request clocks diverge in the walk). Outputs output/counters/errors/
fills are freshly allocated at contract rank. Walk semantics untouched.
REGRESSION HARNESS (torch-free, catches this entire class locally):
tests/python/test_dsa_resolve_rows_rank_shim.py — pure-python mirror of _norm/_restore
(KEEP-IN-SYNC note) fuzzed over the same state-generator family: 20k trials (env
RANKSHIM_TRIALS) asserting contract-rank walk == kernel-rank walk == scalar-clock walk
bit-for-bit + mutated-in-out rank round-trip + negative cases (batch-mismatched,
mis-ranked inputs raise); plus a torch-guarded end-to-end test (runs on-cluster)
exercising the REAL reference with kernel-ranked CPU tensors, scalar clock, restored
ranks, and the loud-failure path. 20k trials pass locally in ~1.6s; py_compile + ast
clean on the final tree. Awaiting re-ship + re-run.

## 4. floe-side patch TEXT (apply with the deployment's `patch -p4` flat layout; author paths shown repo-relative)

```patch
--- a/floe/engine/runner/models/glm53flash/dispatch.py
+++ b/floe/engine/runner/models/glm53flash/dispatch.py
@@ class Glm53DispatchConfig:
     # OPT-IN (``glm_indexer_topk``): ... (existing comment block above)
     glm_indexer_topk: Optional[bool] None
+    # OPT-IN (``dsa_resolve_rows_fast``): the rows-lane vectorized row resolver
+    # (``vkernels.torch_ops.dsa_resolve_rows``) replaces the kvaas scalar
+    # ``resolve_rows`` launch in ``SparseStep.swap_in`` — bit-exact
+    # (integer/addr parity; the CPU oracle restates the production semantics),
+    # 11.4 x 828 us -> ~11 x 60 us at the B=1 geometry. Any miss (knob off,
+    # vkernels absent, OpNotEligible) falls back to the kvaas resolver
+    # verbatim; the kvaas path stays the parity oracle.
+    dsa_resolve_rows_fast: Optional[bool] = None
```
(vocabulary: the field auto-joins `GLM53_KNOBS`; `configure_glm53(**knobs)` picks it up
unchanged; register it in the `server_args.py`/`serve_command.py` help vocabulary lists.)

```patch
--- a/floe/engine/runner/models/glm53flash/sparse_runtime.py
+++ b/floe/engine/runner/models/glm53flash/sparse_runtime.py
@@ class SparseStep:
     def swap_in(self, g: int, topk_indices):
         if self._prefetcher is not None:
             self._prefetcher.before_layer(g)
         padded = topk_indices.reshape(self._n_live, self._selected.shape[1])
         sel = self._selected[: self._n_live]
         sel.copy_(padded)
         sel -= self._pads[:, None]
         sel.masked_fill_(padded < 0, -1)
+        if getattr(self, "_rows_fast", None) is None:
+            try:
+                from vkernels.torch_ops.dsa_resolve_rows import dsa_resolve_rows
+                self._rows_fast = dsa_resolve_rows
+            except ImportError:
+                self._rows_fast = False
+        if self._rows_fast:
+            try:
+                r = self._resolvers[g]           # kvaas resolver = table + state owner
+                out, counters, errors, fills = self._rows_fast(
+                    r.kv, r.host, r.resident, r.backing, r.generations,
+                    r.lengths, r.hot_slots, self._selected, r.tags,
+                    r.tag_generations, r.ages, r.clock,
+                    page_tokens=r.page_tokens, allow_padding=r.allow_padding,
+                    fence=(r.fence.values, r.fence.expected) if r.fence is not None else None,
+                    scratch=self._rows_scratch,  # dict cached across layers/steps
+                )
+                if int(errors.sum().item() if not errors.is_cuda else 0) or True:
+                    pass  # error counters stay device-side, consumed exactly as before
+                return out[: self._n_live]
+            except Exception:
+                self._rows_fast = False          # permanent fallback to the oracle
         rows = self._resolvers[g].resolve(self._selected)  # [1, width, W]
         if self._prefetcher is not None:
             self._prefetcher.submit(g, self._selected)
         return rows[0][: self._n_live]
```
Notes for the applier: `__slots__` on `SparseStep` gains `"_rows_fast"`, `"_rows_scratch"`;
`self._rows_scratch = {}` in `__init__` (scratch tensors are per-step cached — output/counters/
errors/fills/bitmap/worklist are zeroed or fully overwritten in-kernel each call, matching the
kvaas reuse contract). The `.item()` probe in the sketch must NOT be shipped (host sync) — the
errors tensor is already read device-side by the existing step-finish path; drop the two lines
marked with the `or True` no-op. The OQ-6 pre-capture probe (`resolver.resolve` loop) keeps
warming the kvaas kernel; add one `self._rows_fast(...)` probe call there too so the Triton
specialization is warm before capture (same idempotency argument as the existing probe).

## 5. Bench arm text

Stack-moe recipe legs (same harness as the mhc-fuse A/B, `closed:1` ITL legs; the exact recipe
invocation is the campaign's standard stack-moe bench — no literal recipe file was found in-repo,
so the knob line below is the arm delta to append):

```
# arm A (control):  <stack-moe recipe as landed>            # kvaas resolve_rows (production)
# arm B (rows):     <stack-moe recipe as landed>  --model-opt dsa_resolve_rows_fast=1
# gate: bit-exact parity run first (tests/python/test_dsa_resolve_rows.py, on-cluster, GPU),
#       then 3 consecutive b1/b4 ITL legs; accept on 9.11 ms -> <=0.7 ms resolve_rows family
#       time (trace_families 'resolve_rows' regex) and unchanged wrong_output/stability gates.
```

## 6. Follow-ups (explicitly out of this lane's commit)
- On-cluster parity run + bench legs (no local GPU; hard rules honored — no ssh/sbatch from here).
- One-launch batching of the 11 calls via the two-pass decode restructure (indexer pass → one
  `dsa_resolve_rows_batched` → attention pass); the kernel already supports it.
- Fold the per-`swap_in` staging triple (`copy_`/`sub_`/`masked_fill_`) into the kernel's
  phase 1 (accept the padded int32 selection + pads tensor directly) — removes 33 launches/step.

## 7. DEPLOYED + BENCHED (2026-09-30, beverin) — kernel 12× proven, end-to-end wash

**Deployment** (all gates green before the first leg):
- Parity on-cluster: **7/7 PASS** (job 656697, `tests/python/test_dsa_resolve_rows.py`, GPU),
  after fixing the test's own bugs (multi-layer `[L]` slice on 2-D shared state, kernel-ranked
  vs ref-ranked shape normalization in the comparisons, sbatch log truncation hiding tracebacks).
- Import/knob probe in the REAL serve env (job 656699, 43 s): `dsa_resolve_rows` imports from
  the repo vkernels (shadows staged), knob registered in `GLM53_KNOBS`, SparseStep methods present.
- Patch applied on a fresh clone `run-rows-clone` (base run-wZTCex7s + ctrl-branch delta
  (serve.py TP-knob bridge, knobs.py commit-defer marker, dispatch/forward tp_control_stream)
  + the rows patch): `SparseStep.swap_in` routes via `_rows_lane_init()`/`_resolve_rows_fast()`
  with **resolver-owned scratch** (captured-graph lifetime safety: buffers must outlive the
  step like kvaas's own output/counters/fills), geometry-tuple re-key guard, shared prefetcher
  tail, one-shot engagement/fallback logging; OQ-6 structural-warm probe calls the fast path
  per layer pre-capture (JIT warm before any capture). First submission 656700 failed on the
  missing ctrl-branch knob bridge (unknown --model-opt); 656701 ran clean.

**Engagement (unambiguous)**: serve log `rows-lane: dsa_resolve_rows_fast latched -> vectorized`
on all 4 ranks; **zero** FALLBACK warnings across both legs; correctness gate `all_paris=True`.

**Results** (jobs 656701, 656705; arm `stack-moe-rows` = ctrl recipe + `dsa_resolve_rows_fast=true`):

| metric | ctrl (656553) | rows 656701 | rows 656705 |
|---|---|---|---|
| decode tok/s p50 (B=1) | **12.31** (10.73/12.31/12.69) | **13.89** (13.23/13.89/14.01) | **13.72** (11.42/13.72/14.16) |
| concurrency agg (tok/s) | 27.58 | 26.30 | 25.61 |
| (phase-2 traced, 656705 only) | — | — | 12.68 p50 / 23.17 agg |

**CORRECTION (was misread as "flat")**: ctrl's B=1 decode was **12.31**, not 13.9. The rows
arm is a **real single-stream win: +12.8% (81.2 → 72.0 ms/tok = −9.2 ms/tok)** — matching the
kineto-measured 8.1 ms/step resolve saving almost exactly. At B=1 the saving is **fully
exposed**; only the 4-way concurrency phase re-absorbs it (multi-stream AR wait + bandwidth
contention), where agg is a wash-to-slightly-negative within the leg band.

**Kineto** (656705 decode windows, per rank): `resolve_rows` **829 µs → 82 µs mean
(12×)**; per-step 0.91 ms (w1, ~33 steps) / 1.14 ms (w2, ~35 steps) vs kvaas 9.1 ms —
**~8.1 ms/step of pure compute removed**. `ncclDevKernel` = 55% (w1) / 62% (w2) of the
window span (3646/4413 ms total kernel time = 83% in w1).

**Verdict**: the kernel win is real and bit-exact, and at **B=1 it converts to wall-clock:
13.9 tok/s p50 (best run 14.16) vs ctrl 12.31 — the ~8.1 ms/step saving is fully exposed**.
At 4-way concurrency the same saving is re-absorbed by multi-stream AR wait/bandwidth
contention (agg wash-to-slightly-negative, 25.6–26.3 vs 27.58). Net: **keep the lane landed**
— it is the new single-stream best and the parity oracle stays intact (fallback verbatim).
The remaining B=1 headroom: 72 ms/tok now decomposes ~45% compute / ~55% AR wait;
wire-only AR would put the step at ~35–40 ms → ~25–28 tok/s. The lever is AR volume/skew,
not per-layer compute.

**Next lever (unchanged, sharpened)**: reduce the 11 fp32 ARs per step (segment-fused AR,
AR+commit coalescing on the control stream) — AR is 55-62% of the served step span.

## 8. INTERACTION LEG: rows+soup stack (job 656775, 2026-09-30) — YES, they stack

**Arm** `stack-moe-rows-soup` (node nid002670): ctrl recipe + `dsa_resolve_rows_fast=true`
+ `cache_gate_fp32=true` + `cache_shared_slot=true` + batched-telemetry marker (soup delta
ported from run-soup-clone2: 2 knob fields, fp32soup caches in forward.py with graph-bank-
safe per-key pinning, memoized marker knob in knobs.py, batched fleet telemetry in
sparse_runtime; served PYTHONPATH/scratch discipline unchanged). Engagement: vectorized
latched on all ranks, **zero fallbacks**, `all_paris=True`, bench rc 0/0.

**Single-stream (B=1) ladder** (phase-1 untraced):

| arm | decode p50 | best run | ms/tok | Δ vs ctrl |
|---|---|---|---|---|
| ctrl | 12.31 | 12.69 | 81.2 | — |
| rows | 13.89 / 13.72 | 14.16 | 72.0 | −9.2 ms (≈ resolver −8.1) |
| soup | 14.40 | 14.41 | 69.4 | −11.8 ms |
| **rows+soup** | **14.73** | **15.06** | **67.9** | **−13.3 ms (+19.7%)** |

**Agg (4-way)**: ctrl 27.58 > **rows+soup 26.42** > rows 26.30/25.61 > soup 24.81. Soup's
agg regression is mostly RELEASED by the rows lane (24.81 → 26.42) — removing the 9.1 ms
resolve_rows launches takes the pressure off the path that made the soup caches hurt at
concurrency — but −1.2 vs ctrl persists.

**Additivity check**: independent savings would predict −21 ms/tok; measured −13.3. The two
levers overlap ~8 ms — both drain the same rank-skew/AR-wait pool, consistent with the
absorption model. The pooled pool is now ~2/3 drained; what remains at B=1 is mostly the
AR wait itself. Traced phase-2 agg (28.28) is single-window noise — phase-1 numbers are
the record.

**Standing best**: single-stream **14.73 tok/s p50 (15.06 best)**; concurrency 26.42 (ctrl
27.58 still leads 4-way).

## Straggler forensics correction (lane L1, 656775+656705 traces) — O1 as stated is DEAD

There is **no steady-state AR pathology**. The 156 straggler ARs / 3.71 s decompose 100%
into transition overhead, reproduced within ~10% by the rows-only leg (soup exonerated):

1. **Eager-fallback forwards** at uncaptured widths: any decode forward whose runtime
   batch width lacks a captured graph runs the whole ~53-block forward eagerly; each
   block's MoE-down AR absorbs ~40 ms of slowest-peer host-dispatch skew → **~2.1–2.3 s
   per forward**. w2 shows the drain ladder hitting exactly this: step14 bs=2 (2.33 s),
   step15 bs=1 (2.29 s), step16 bs=4-on-refill (2.32 s) + step17 re-capture desync (1.81 s).
   **~10.5 s of a 26 s window is eager fallbacks.**
2. **Capture-boundary desync**: rank 0 starts replaying while a peer is still capturing →
   first AR waits 0.9–1.9 s (host hipStreamSynchronize mirrors it). ~3.7 s/window.

Steady state is clean at every width: bs=1 replay **34.5 ms/step**, bs=4 **59 ms/step**,
ARs at the 35.7 µs wire floor, straggler mass ≤1 ms/step. w1 is the **bs=1 window**
(29 clean replays), not 4-way as previously assumed. The observed concurrency wall is
**transition tax**, not per-step cost (steady 4-way is only 1.7× bs=1 GPU time).

New oddity to chase: step16 (bs=4 on refill) ran eager despite the bs=4 graph serving
steps 2–13 — the graph-cache key misses on something beyond width (composition/pages).

**Fix ladder (ranked)**: (a) precapture graphs for drain widths {1,2} — kills 2.3 s/step;
(b) `dist.barrier()` after capture before first replay — trivial, kills desync ARs;
(c) kill the periodic `aten::item` DtoH reads (125–166 ms every ~205 ms during eager;
3/step + ~11 ms stream-sync during replay — a major B=1 host item);
(d) worker-rank trace leg (4-rank capture spec in lane report) to attribute the 40 ms/block
host cost. Full report: `floe-bev-main/.local/campaign/beverin/glm5-smoke/lane-reports/`.

## Soup attribution (lane L2) — 11.6 ms/step of eager glue, fusions costed

Post-stack soup = **11.6 ms/step** (13% of replay GPU busy), ~96% inside the captured
graph (must be cut at capture time): mHC casts/mul/add glue + Σx², KDA state
arange/index/index_put_/conv-cat, RMSNormGated sigmoid fallback, `_append_shared_slot`.
Top fusions: **mhc_pre_big_fuse + mhc_compose_pre knobs exist in-repo but are OFF here**
(~3.0–3.5 ms/step; not auto-bit-exact — needs parity leg); KDA decode state glue
(~1.3–1.6 ms/step, bit-exact by construction); sigmoid-RMSNorm fused variant +
shared-slot cat elimination (~0.7 ms/step, bit-exact).

**B=1 budget**: replay GPU 34.5 ms vs bench 67.9 ms/tok → ~33 ms is host-side. Soup
(11.6) + item/sync (~11) is a credible path to **~20 tok/s B=1**.

## Same-day band + node doctrine (lane L3, jobs 656849–656882)

**Node-to-node variance dominates arm deltas today.** Same ctrl arm: 13.54 p50 / 26.25 agg
on nid002706 vs **5.50 p50 / 16.17 agg on nid002666 minutes later** (runs 5.01/5.5/10.3 —
node in a bad state). rows+soup re-run on the same bad node: 8.24 p1 → 13.17 p2 / agg 26.06
(node half-recovered; agg on-band vs 26.42 ref). **Doctrine: every cross-arm comparison is
matched-node or it is noise; legs node-pinned via --nodelist from here on.** The historical
references (ctrl 12.31/27.58, rows+soup 14.73/26.42) were single-node snapshots, not
portable bands.

**V0 follow-up (656892): matched-node is still not enough.** Same ctrl arm, same snapshot,
same node (nid002706): 656849 = 13.54 p50 / 26.25 agg vs 656892 = 8.67 p50 / 31.52 agg
~1.5 h later, with 656892's within-leg runs spreading 6.12→8.67→12.14 (656849's were
tight: 13.45/13.54/13.6). Solo decode is host-latency-sensitive and gets crushed by
interference that 4-way agg partially overlaps. **Primary evaluators from here on:
kineto trace signatures (eager-forward presence, capture desyncs, launch counts, per-step
replay ms — deterministic and leg-stable) and within-leg run spread; tok/s only as coarse
corroboration, never as the decision metric.**

**b8 (MAXR=8) first look** (656882, nid002926): decode p50 12.43 — better than the
same-day default-MAXR rows+soup leg (8.24, node-confounded) — but agg fell to 23.96
(22.73 at the conc[4] sweep point) vs ~26 at 4-way. Direction: **b8 helps single-stream
decode, costs aggregate**. Sweep gap: BENCH_CONC_SWEEP=4,8 emitted only conc[4] — no
conc[8] row; needs a bench-script look before the 8-way aggregate question closes.

## T1T2T4 validation stack (patches applied, V-triplet launched)

Root cause of the transition tax (lane T1T2T4): the sparse runtime's captured window is
keyed to the exact **live-set** `(rows tuple, admission epochs)` (sparse_runtime.py:1347) —
any drain/refill mints a new key → full eager warm (~2.3 s) + re-capture desync (~1.8 s)
per transition, even at a width whose graph replayed minutes earlier (the step16 anomaly).
Dense bank exonerated (pre-captures all widths, never re-captures).

- **T1** (`17b4eb0`, kill-switch `FLOE_SPARSE_SHAPE_WARM=0`): shape-level warm gate —
  `bucket.ever_captured` lets a fresh live-set at an already-captured width capture on its
  first step (the eager warm existed only to keep Triton JIT out of capture; guarantee is
  shape-level: bucket-fixed B, 256-grain t_pad, OQ-6 pre-capture probe re-warms resolvers).
- **T2** (`FLOE_SPARSE_CAPTURE_BARRIER=0` to disable): TP barrier after capture_forward,
  before first replay — kills the 0.9–1.9 s first-replay AR desync. Known edge: an
  asymmetric capture failure would block peers at the barrier (NCCL watchdog aborts;
  zero capture failures in any leg so far).
- **T4**: `stack-moe-mhc` arm added to the sbatch = ctrl config verbatim +
  `--model-opt mhc_compose_pre=true` (the real new flip; `mhc_pre_big_fuse` is implied by
  the recipe's `mhc_big_fuse`). Parity gate: phase_correctness ok + all_paris=True +
  fallback-count comparison vs ctrl + greedy-hash spot check.

V-triplet (V0 ctrl / V1 +T1T2T4 / V2 +mhc_compose_pre, all node-pinned nid002706,
sequential): expected signatures — transitions capture directly after the first drain
cycle (no 2.3 s eager forwards), desync ARs gone, agg ≥ 27.5 target on V1, B=1 p50
~unchanged (its transitions are pre-timing).

## T5 soup fusions landed (bit-exact; vkernels `9ce9bcf` + floe `546cecc`)

Per L2's attribution, three in-graph glue kills, all bit-exact by construction:

1. **Hoisted slot arange** — 85 launches/step → 0 (per-bsz bucket, soup-constant contract).
2. **V-major state writeback** — one `index_put_` through permuted pool strides replaces
   transpose-copy + K-major scatter: 108 → 34 index_put/step, whole-pool `torch.equal`.
   (Mirror GET is NOT a win — permuted-view gather still pays `.contiguous()`; writeback
   direction only.)
3. **Sigmoid o_norm** — `rms_norm_gated` sigmoid variant validated (kernel was already
   shipped): 8-kernel eager fallback × 34 KDA layers → 1 kernel.
4. **Shared-slot pinned rows** (knob `shared_slot_pinned`, requires `fused_router`) —
   warmup-pinned `[rows,K+1]` routing rows, shared column prefilled; fused_router stores
   routed columns directly (new `out_indices`/`out_weights` args); 84 cat/fill/step → 0
   (fused) / 84 (eager). Capture-safe: populate at warmup, never allocate/evict under
   capture; bounded 64-entry pin set.

Expected ≈1–1.5 ms/step — below B=1 run-noise, so the T5 legs must be judged on
**kineto launch counts** (arange=0, index_put≈34, cat≈0, no sigmoid fallback), not p50.
Leg design when the node frees: T5a = ctrl + `kda_packed_decode+fused_norms` (T5 tree),
T5b = T5a + `shared_slot_pinned` — T5b−T5a isolates pinned-slot/direct-store;
T5a−V0 isolates the kpd/fn enablement + hoist/writeback on matched node.

**HAZARD (pre-existing, ledgered): `fused_conv_update` × paged cache silently loses the
conv state roll** — the kernel rolls a gathered COPY (`_PagedGlmCache.get_conv_state`
returns advanced-index copy; dense `Glm53Cache` is fine). Do NOT enable
`fused_conv_update` for paged serving (e.g. the `stack-moe-conv` arm) until the kernel
takes slot ids or the wiring scatters the rolled window back. No current arm is exposed.

Deferred (deliberate): F2 V-major rec-pool migration (kills gather+writeback entirely,
~3 kernels/layer/step) — blocked on checkpoint layout + TP state_dict ripple; its own
lane. A K-major buffer read as V-major is stride-identical for square heads, so
eligibility checks cannot catch a wrong-layout pool — migration must be atomic.

## T5 verdict on beverin: NOT PROVEN (CTRLKIN, jobs 656939/656970/656984)

Both T5 legs ran clean (0 fallbacks, 0 tracebacks, kda/fused knobs engaged on 4/4 ranks,
`_gemv_bf16` +102/step proves the packed-decode path is live) — but the kineto launch-count
gate **failed**: inside `step[DECODE]`, arange ≈89–94/step, cat ≈209/step, sigmoid exactly
68/step in **all** arms (t5a, t5b, and the freshly captured ctrl baseline); index_put is
1–6/step *higher* in t5 (282 vs 250). The local dev claims (arange 85→0, index_put 108→34,
slot-cat 84→0) do **not** transfer to the serving path — the fused kernels run *alongside*
the targeted eager ops, not instead of them. Throughput deltas are within noise on the
pinned node (t5b vs t5a p50 +2.3%/−1.5%). Attribution: **no kineto-supported win**; T5
fusions stay default-off on the serving path until the wiring gap (which call site still
emits the eager chain) is root-caused.

Root cause of every empty ctrl `kin-*` dir (656553→656892, days of trace-less runs): stale
`run-ctrl-clone/repo-prof` made `apply_kineto_patch_bev.py` bail "already exists" → serve
ran unpatched with kineto disabled. Cleared; ctrl baseline trace now exists (kin-656984,
1.35 GB, decode p50 12.63) — the evaluator for all future arms.

## V1 (T1T2T4) outcome: target crash FIXED, new phase-E abort under attribution

Job 656937 (v2 patch, nid002706): the exact section-B phase that 500'd in 656923 now
passes 3/3 with zero capture failures; decode p50 13.87 vs V0 8.67 (no regression).
New unrelated failure: phase E (first 4-way session) aborts all 4 ranks with
`HIP error: operation not permitted when stream is capturing` from `~CUDAGraph()` →
SIGABRT. Suspects: T2 capture-barrier rank symmetry under ragged rosters, or T1v2's
increased capture frequency (close-during-capture overlap). Isolation leg 657010
(`FLOE_SPARSE_CAPTURE_BARRIER=0`, same tree/node) in flight; V2 (`stack-moe-mhc`) stays
gated on it — a phase-E abort kills its phase 2. If barrier=0 passes section B AND
phase E, the barrier-off tree is the V2 baseline; else the implicated patch needs a
symmetry/teardown fix before V2 fires.

## DELIVERY BUG: ISO leg 657010 was VOID (knob never delivered) — fix landed

The barrier-off isolation leg 657010 crashed identically to 656937 — but a serve_env
audit showed `FLOE_SPARSE_CAPTURE_BARRIER=0` never reached the serve process: the
glm5-smoke EDF scrubs `os.environ` during heavy imports, so step-time knob reads
(`knobs._env_flag`) miss plain `--export` env. The codebase already knew this
(knobs.py commit-defer accessor: "Env delivery is unreliable on the glm5-smoke EDF";
soup/delta-pool knobs use marker files) — the new T1/T2 knobs predated the lesson.
**T2 is therefore NOT exonerated; the phase-E attribution (T2 barrier vs T1v2
capture-onset) is re-opened.**

Fix (vkernels-floe t1t2t4 `12498f6` + sbatch): `_env_flag` gains the standard
marker-file fallback (`<repo-root>/env-markers/<NAME>`, env wins, 6-assertion host
test green); the sbatch materializes `FLOE_SPARSE_CAPTURE_BARRIER` /
`FLOE_SPARSE_SHAPE_WARM` markers EVERY leg from the job env with an engagement echo
(per-leg delivery proof in the serve log). In flight, serialized on nid002706:
- **657018** ISO redo: barrier=0 only → if phase E passes, T2 implicated; else T1v2.
- **657019** V2 (stack-moe-mhc): `mhc_compose_pre=true` (argv — scrub-immune) on the
  maximally-V0-like capture path (shape_warm=0 + barrier=0 = V0 capture semantics +
  v2 bugfixes), ctrl baseline = kin-656984 traces. Parity gate: correctness ok +
  all_paris + greedy-hash spot check vs ctrl arm.

## Phase-E attribution CONCLUSIVE (657018/657019) + V2 first number

With delivery now proven (per-leg sbatch echo + on-disk markers):
- **657018** (shape_warm=ON + barrier=0): phase E CRASHES (4× `~CUDAGraph` HIP abort) —
  with T2 disabled the crash persists.
- **657019** (shape_warm=0 + barrier=0): phase E PASSES, job completes clean.
Delta = **T1v2's shape-warm capture-onset gate is the phase-E culprit; T2 is fully
exonerated** (the barrier neither causes nor masks it). Mechanism: immediate
capture-on-settle destroys/retires graph state while the stream is still capturing,
under phase E's live-set churn. T1v2 stays default-OFF (marker shape_warm=0) until a
deferral guard lands; its seam-latency win (~0.9–1.9 s/capture) is the prize for the
fix lane.

**V2 (mhc_compose_pre) first leg (657019): E agg 28.72 vs ctrl 25.69 (+11.8%), row p50
7.18 vs 6.42 (+11.8%), correctness ok, finish=stop.** Capture semantics matched (both
arms on V0-like warm-then-capture). Single-stream decode medians not comparable
cross-job (documented 2× same-arm swings). Trace-level mechanism proof pending: the
leg's phase-2 kineto came up EMPTY (stale repo-prof bail, same failure class as the
old ctrl legs) — repo-prof cleared, V2 re-run queued (657032) for traces + a second
variance sample.

## V2 mhc_compose_pre — final attribution (657019 + 657032, same-day matched-node)
- **E agg: 28.72 / 26.85 (mean ~27.8) vs ctrl 656984 25.69 → +4.5% / +11.8%.** row p50 7.18/6.71 vs 6.42.
- **Mechanism PROVEN at trace level** (kin-657032 vs kin-656984): B=1 decode window — fused `_compose_pre` 2759 launches,
  incumbent `_compose`+`_pre_gemv` collapsed to 31+31 (−98.8% two-launch chain in the rows≤2 domain). 4-way window:
  178 compose_pre fires from small-batch steps in the mix; incumbent chain dominant at rows>2 — eligibility cap exact.
- Parity: all_paris=True, says_bern=True, natural finish on both legs; correctness latency 74.9s ≈ ctrl.
- Verdict: real, modest, batch-structure-dependent win. Deployable as `stack-moe-mhc` arm (clean, reproducible).
  Single-stream medians unchanged (step is AR-wait-bound at B=1 — compute-side fusion can't move it; consistent with rows lane finding).

## Session ledger (t1t2t4 campaign close-out)
- T1v2+T2: **T2 exonerated** (657019 crashed 0× with barrier=0-only tree); T1v2 shape-warm capture-onset is the
  phase-E abort source → mitigation = knobs default-off (stable config on beverin snapshot). T1v2's seam-latency win
  (0.9–1.9s/capture event, rare) parked as follow-up behind a deferral-guard fix + fresh validation legs.
- V1 target crash: fixed — decode median 13.87 vs 8.67 on the broken tree (same day, matched node).
- Infra: marker-file knob delivery (EDF env-scrub workaround) landed 3976606; ctrl trace baseline restored; repo-prof
  bail auto-clear added to sbatch legs after 657019's empty phase-2 traces.
- Armed-optimization scoreboard: rows+soup **+19.7% B=1 (deployed)** · mhc_compose_pre **+4.5–11.8% E agg (proven)** ·
  T5 negative (attributed, not deployed) · T1v2 parked (root-caused).

## T1v3 deferral guard — parked follow-up landed (post-close-out increment)
- Crash evidence re-read from 657018 log: all 4 aborts are faulthandler stacks at kvaas graph.py:42 `capture`
  (CUDA-graph capture deadlock) -> SIGABRT, during phase-E onset. The shape-warm direct capture skips the eager warm
  that pre-stages a fresh live-set; under 4-way peers the bt width can settle AFTER the width check, and capturing
  unstaged while peers mutate the shared block table deadlocks the capture.
- Guard (0003 patch + live edit in run-t1t2t4-clone/repo, bit-identical port verified, compiles on 3.11):
  shape-warm direct capture additionally requires `len(reqs) == 1` — a QUIET window with no peer mutator (the regime
  every clean leg captured in). Multi-request steps take canonical eager->capture (ctrl semantics, 0-crash proven).
- Cost: phase-E fresh live-sets lose the direct-capture seam win (eager warm ~2.3s instead); single-stream live-sets
  (phases 1/2) keep it. T1v2's 0.9-1.9s/capture-event win retained where it was measured.
- Validation leg 657078 = 657018's exact crashing config (ARM=stack-moe-ctrl, barrier=0, shape_warm=1) on the guarded
  tree. Pass = 0 capture aborts + natural completion; then the seam-latency delta vs 657019/657032 (shape_warm=0).

## T1v3 falsified; shape_warm default flipped OFF (post-close-out increment, legs 657078/657084)
- **657078** (v3 guard, node 2676): COMPLETED clean, 0 aborts, E agg 31.99 / decode med 13.87 — but node-confounded
  (2676 decodes ~30% faster in solo runs), and...
- **657084** (v3 guard, node 2706, matched vs baselines): **FAILED — same 4x hipErrorStreamCaptureUnsupported at
  section E.** The guard is falsified: section E's 4 requests are 4 separate live-sets, each window len(reqs)==1 —
  the quiet-window condition never gated the crashing captures.
- **Root cause corrected** (earlier "deadlock" read was wrong): the aborts are `HIP error: operation not permitted
  when stream is capturing` — direct capture makes a fresh live-set's FIRST decode step the capture step, and that
  step still carries first-touch host-tier residency work (prompt-commit D2H + miss-copy); some op in it is illegal
  on a capturing HIP stream. Fatal via C++ terminate — kvaas's python-side retire guard can't intercept it.
  Canonical eager→capture (shape_warm=0) never crashes (all legs, any node).
- **Deployed posture**: `FLOE_SPARSE_SHAPE_WARM` default OFF in knobs.py (evidence docstring) + sbatch marker
  (0004 patch). Direct capture = debug-legs-only behind explicit env. T1's seam win (0.9–1.9s/event, rare) stays
  parked behind a HIP-trace debug cycle to pin the exact in-capture op.
- Infra fixed en route: sbatch now auto-clears stale `repo-prof` (657019/657078 had served kineto-unpatched on it).
- Scoreboard update: rows+soup +19.7% B=1 (deployed) · mhc_compose_pre +4.5–11.8% E agg (proven) · T5 negative ·
  **T1v2/v3 not deployable (crash root-caused to HIP capture-vs-residency interaction; default off)**.

## L7 (subagent): conc-sweep delivery bug root-caused + fixed; b8 closing leg speced
- The conc[8] row was never dropped by the loop: **sbatch `--export` is comma-delimited**, so `--export=...,BENCH_CONC_SWEEP=4,8` truncated the value to `4` (+ orphan `8` token, silently dropped). Differential proof: 656882 (value via export list → conc[4] only) vs 656567 (value `8` via export → conc[8] emitted) vs 656523 (value set **in-script** → both emitted). Supersedes L3's "loop gates on MAXR" hypothesis.
- Fix deployed to glm5-tp4-bench.sbatch (beverin live + campaign mirror, `bash -n` clean, .pre-l7.bak kept): normalize `;`→`,` after arm selection + per-leg delivery echo `[tp4-bench] conc-sweep delivery: …` — tripwire for 657010-class delivery bugs. **Submit spelling from now on: `BENCH_CONC_SWEEP=4;8`.**
- Closing leg spec (queued, runs after L4/L5/L6 legs): `ARM=stack-moe-rows-soup MAXR=8 SNAP=run-rows-clone --nodelist=nid002706 BENCH_CONC_SWEEP=4;8` — NOT the stack-moe-b8 arm (its EXTRA lacks rows/soup knobs). Net-positive gate: same-leg `conc[8].agg ≥ conc[4].agg`, conc[4] ≥ ~24 (on-band), conc[8] above the 14.35–14.48 collapse floor, all_paris=True, row_min ≥ 0.5×agg/8, B=1 p50 held, 8-way duplicate samples within 15%.
- Report: lane-reports/L7-bench-sweep-b8.md

## L4/L5/L6 lane harvest (run d2bbf267) — patches deployed, 6-leg queue live (657119–657124, nid002706 serial)

### L4 bit-exact soup fusions — lane SHRUNK by tree audit (report: lane-reports/L4-bit-exact-soup-fusions.md)
- 85 arange/step mis-attributed (slot arange only under kda_packed_decode, OFF in stack); sigmoid-o_norm likely
  already banked (rms_norm_gated wired + decode_fuse=true); KDA packed kernel is ULP-class ("matches to ULPs, NOT
  bits", 1/32 greedy flips) → T5a lever out of bit-exact lane. Honest residual: ~1.0 ms/step (not 2.0–2.3).
- t5b root-cause candidate: T5's `_pinned_shared_slot_buffers` guarded the LOOKUP with is_current_stream_capturing
  → every captured graph recorded the donor cat chain (cat=209 with knob echoed on). Port moves guard to ALLOCATION
  (warmup populates → capture reuses). Engagement echo (latched line + cat −84/step) decides.
- Deliverables: vkernels fused_router out-buffer direct-store (stride kwargs, verbatim T5 9ce9bcf), dispatch +
  forward shared_slot_pinned (capture-guard fix + latched echo), 2 new GPU parity tests, arms stack-moe-l4a
  (fused_conv_decode=true, ~0.5 ms) / stack-moe-l4b (pinned slot, ~0.5 ms).

### L5 item/sync elimination — 46 blocking DtoH + 3 syncs/step enumerated, patch deployed (report: L5-step-path-item-syncs.md)
- 11 ms replay sync = SparseStep.finish → drain() → sparse-runtime stream synchronize — LOAD-BEARING (StepTableBank
  pinned-staging reuse guard); double-buffering is a separate lane, not touched.
- begin_step does 2 blocking tolist()s on tensors the engine just staged from the same host ints (glm53's
  decode_step declared seq_lens_host/rows_host and dropped them); _check_errors(11) + _counters(33) per-resolver
  item reads = soup telemetry patch (62ed36c) never ported into run-t1t2t4.
- Patch: dispatch-only, 4 files, FLOE_SPARSE_BATCH_TELEMETRY (memoized marker, three-way port) +
  FLOE_SPARSE_HOST_STEP_TABLES (+_DEBUG cross-check). Gate: aten::item 44→≤3/step, DtoH 46→≤2, syncs 3→1,
  replay GPU ±5%.

### L6 graph-key miss — verdict: the step16 miss is CORRECT, not a bug (report: L6-graph-key-miss.md)
- Key = (rows tuple, admission epochs tuple) (sparse_runtime.py:1515); refill gives fresh epochs → miss is the
  minimal sufficient statistic firing. Relaxation provably unsound (armed SparseExecution holds native lease
  tickets + hot-slot claims + fence snapshot; kvaas forbids lease rebinding under capture; release-before-free-list).
  Drain steps 14/15 structurally eager regardless (window dropped synchronously at release()).
- Deliverables: FLOE_SPARSE_KEY_TRACE diagnostic (default OFF; one leg attributes every eager step to its key
  component) + Tier-1 precapture spec "decoded live-set direct capture" (FLOE_SPARSE_WARM_DECODED, spec-only,
  exploits fresh-vs-survivor asymmetry of the HIP first-touch crash); Tier-2 (refill) blocked on HIP-trace pin.
- Open item flagged: req.commit_event recorded (sparse_runtime.py:1050) but no consumer found — latent hazard.

### Deployment + queue
- Clones: run-l5-clone / run-l6-clone / run-l4-clone (cp -a of run-t1t2t4-clone; CAPTURE_BARRIER=1, SHAPE_WARM=0
  sanitized; INDEXER_DELTA_POOLS=1 + COMMIT_DEFER=1 preserved = legs-of-record state). Patches applied via
  patch -p1 (hunk headers recomputed — lane markdown had wrong @@ counts and md-stripped blank lines in new-file
  hunks); all touched files py_compile OK; L4 fuzz ≤2 spot-checked by grep (insertion sites + class indent correct).
- sbatch: ctrl-family condition + l4a/l4b/l5/l5-debug arm blocks + default-0 marker printfs (KEY_TRACE,
  BATCH_TELEMETRY, HOST_STEP_TABLES, HOST_STEP_TABLES_DEBUG) + extended marker echo. bash -n OK, mirror identical.
- Queue (nid002706, serial): 657119 L5a ctrl-identity → 657120 L5b → 657121 L6 KEYTRACE diag → 657122 L4-1 l4a →
  657123 L4-2 l4b → 657124 L7-b8 rows-soup MAXR=8 BENCH_CONC_SWEEP=4;8 (semicolon inside quoted export per the
  L7 comma-truncation doctrine).

## Queue execution 1 — whitelist miss fixed; L5a identity green; L6 attribution PROVEN in-vivo (657121)

- **sbatch ARM whitelist miss**: the validation elif (~line 186) didn't know l4a/l4b/l5/l5-debug → 657120/657122/657123
  died in ~24 s ("unknown ARM", exit 2). Fixed + error message extended; mirror synced. Resubmitted as 657145 (L5b) /
  657146 (L4-1) / 657147 (L4-2), queued behind 657124 (L7-b8, running).
- **657119 L5a ctrl-identity (run-l5-clone, knobs OFF): COMPLETED 16 m, rc1=0 rc2=0.** B=1 p50 12.99 tok/s (ctrl band
  12.31–12.69, inside spread), 4-way agg 25.47, all_paris=True — knob-off identity gate GREEN (patched tree behaves).
- **657121 L6 KEYTRACE diagnostic: COMPLETED 16 m, all gates green — the step16-class misses are CORRECT, proven in-vivo:**
  180 KEYTRACE lines (4/rank); reasons: 140 window-gone (release closes window before key check) + 40 key-mismatch.
  Refill signature EXACTLY as predicted: `old(rows=[0,1,2],ep=[8,9,10]) new(rows=[0,1,2,3],ep=[8,9,10,11])
  rows_changed=[] epoch_changed=[3]` + `decoded_in_bucket=[True,True,False,False]` (survivors True, admissions False).
  Zero spurious mismatches (reject-condition (a) did not fire); L6 closes CORRECT-miss; Tier-1 precapture
  (FLOE_SPARSE_WARM_DECODED) spec graduates with a working freshness predicate.

## Queue execution 2 — 657124/657145/657146 done; 657145 VOID (repo-prof marker staleness) root-caused + fixed; L4a fusion PROVEN live

- **657124 (rows+soup b8) COMPLETED**: B=1 p50 12.7 (in-band); conc[4] agg 23.27, **conc[8] agg 13.43 / row_p50 1.68**
  (the missing 8-way datapoint: 8-way trades agg for per-row latency as predicted; ctrl-twin 657078 held agg 31.99 at conc4).
- **657146 (L4a conv 4→1) — fusion PROVEN live in the traced window**: `_conv_decode` fused kernel 1054×
  (=31 steps × 34 KDA layers) vs ZERO conv-named kernels in the ctrl trace (eager 4-op chain); all_paris=True,
  p50 13.19 in-band. Gate green.
- **657145 (L5b) VOID as A/B**: trace counts byte-identical to its ctrl twin (item=3601 sync=124 DtoH=102 both) —
  root cause: arm-block marker writes land in `$SNAP/repo/env-markers` AFTER the repo→repo-prof copy, and the
  **kineto phase serves from repo-prof** → traced process read stale 0s (phase-1 serve from repo/ was correctly ON).
  Fix: sbatch now mirrors the final marker set into repo-prof before serve launch (line ~362, bash -n clean).
  The "shadow gate passed" lines are the pre-existing soup SHADOWTRACE, not L5's debug — knobs never engaged.
- **L5b resubmitted as 657173** (queued behind 657147 l4b, running). Mirror synced + committed.
- Note: ctrl same-config spread this session remains wide (B=1 12.72–14.05, agg 21.98–31.99); same-clone A/B pairs
  (657119 vs 657173; 657078 vs 657124) are the only decision-grade reads.

## Queue execution 3 — L4 verdict honest (kernel-time win NOT proven); Tier-1 patch authored + queued (657175)

- **657147 (L4b pinned slot) COMPLETED**: all_paris=True, p50 12.5 in-band. Trace: kernel-busy 2329 ms vs ctrl 3065 —
  but the ENTIRE delta is rcclGenericKernel (allreduce wait/skew) −726 ms; non-AR compute identical (986 vs 996 ms).
- **Three-way decode-w1 kernel-busy**: ctrl 657119 = 3065 (AR 2069) | l4a 657146 = 3861 (AR 2868) | l4b 657147 = 2329
  (AR 1343). **AR busy swings ±0.8 s between same-family legs; non-AR compute is constant ±0.5%.**
- **L4 verdict (honest): fusions ENGAGE (l4a `_conv_decode` 1054×, 5 ms/window total; l4b cat chain ~nil) and are
  bit-exact with zero p50 regression, but the ~1.0 ms/step projection does NOT survive measurement — AR-skew variance
  swamps a ~15 ms/window effect. Same pattern as T5 NOT-PROVEN.** The dominant cost is AR wait (2–3× compute),
  confirming L1/L5/Tier-1 targets (host path + eager-warm seconds) over GPU compute.
- **Tier-1 precapture (FLOE_SPARSE_WARM_DECODED) authored per L6 §4 spec**: knobs.py `sparse_warm_decoded()` +
  sparse_runtime.py disjoint survivor-set direct-capture branch (all rows decoded-in-bucket at already-captured
  shape; v3 fresh-set gate untouched; knob-off = one _env_flag read/step, byte-identical). Applied to run-l6-clone/repo
  (compile OK), sbatch arm `stack-moe-t1r` added (whitelist + markers WARM_DECODED=1 KEYTRACE=1 + repo-prof mirror).
  **Leg 657175 queued** behind 657173 (L5b retry with the marker-mirror fix). Gates: window-gone KEYTRACE population
  shrinks vs 657121's 140 (drain eager→capture conversion); shadow gate 0.0000 on new capture steps; all_paris=True;
  no capture crashes (a crash falsifies the fresh-admission-specific hypothesis).
  Patch scripts: patches/T1R/{apply_t1r.py,add_t1r_arm.py}.

## Queue execution 4 — L5b (657173) mechanism PROVEN: blocking item tax −99.7%; call-count + sync gates partial; T1R leg live (657175)

- **Delivery fix VERIFIED in-vivo**: "env-markers mirrored into repo-prof: 8 files" + repo-prof BATCH_TELEMETRY=1 —
  the traced serve saw the knobs (unlike VOID 657145).
- **657173 (L5b knobs ON) vs ctrl twin 657119, same kineto window (45 steps both, rccl 3034/3035)**:
  - **aten::item blocking duration 1087.0 ms → 3.1 ms/window (−99.7%)** — the per-step DtoH-blocking item tax is
    ELIMINATED (80.7→48.6 item calls/step, but the survivors are 1.4 µs non-blocking, not 300 µs syncs).
  - DtoH memcpys 102→73/window; **syncs 124→123 (gate 3→1 NOT met)**; non-AR GPU compute identical (995/992 ms ✓
    host-path-only confirmed); all_paris=True (bit-exact).
  - **Verdict: mechanism proven, host-path win real in-trace; call-count gate (≤3/step) partially met (48.6/step
    remain — other call sites than begin_step tolists), sync gate unmet.** Next iteration needs the l5-debug leg
    (host_dbg=1) to enumerate the remaining sites.
- **Wall: p50 7.21 (phase-1) / 9.65 (kineto-phase) vs twin 12.99 — DEGRADED on paper, but matches the L3
  depressed-node pattern (leg A: 5.50/16.17) and AR busy was 167 ms (no skew this run). Confounded — queued
  657179 (same-session ctrl twin on run-l5-clone) behind 657175 to separate node depression from patch cost.**
- **657175 (first Tier-1 WARM_DECODED leg) RUNNING** — arm echo green ("t1r: warm_decoded + key_trace markers on").
  Gates on completion: window-gone KEYTRACE population shrinks vs 657121's 140; shadow gate 0.0000 on new capture
  steps; all_paris=True; ZERO capture crashes (crash = fresh-admission hypothesis survives, Tier-1 falsified).

## Queue execution 5 — T1R falsified at width-4 (657175 SIGABRT); wall confounder resolved (657179); host-staging validated leg-long (657180); Tier-2 probe authored

- **657175 (Tier-1 WARM_DECODED) FAILED rc=1, SIGABRT — falsified in v1 form**:
  - Phase-1 (B=1) completed 3 runs, **p50 13.46 = best of session** (runs 11.14/13.46/13.5), then the 4-way phase crashed.
  - Crash chain (faulthandler, staged kvaas-src): `run_forward:109 result=forward()` → `graph.py:42 with torch.cuda.graph(...)` → **abort INSIDE the captured width-4 step forward**. Crashing step KEYTRACE: `width=4 n_live=3 reason=key-mismatch force_eager=False` — with shape_warm requiring len==1, the only route to mode=capture there is the warm_decoded branch ⇒ **Tier-1 DID fire and crashed**.
  - Window-gone census 44×width-1 + 4×width-4 (vs 140 total in 657121) — width-1 events still eager (Tier-1 needs ever_captured; fresh width-1 buckets convert only after first capture).
  - Root-cause class per kvaas issue #43 comment: opaque HIP capture-state error → NCCL watchdog SIGABRT, uncatchable by the retire guard.
  - **Tier-2 discriminator probe authored** (`FLOE_KVAAS_CAPTURE_PROBE=1` in kvaas execution.py run_forward: for capture-path width>1 steps, pre-run the exact captured callable eagerly + sync BEFORE the capture window): prerun-ok+capture-completes ⇒ cold-path class (fix = bucket warmup); prerun-ok+still-aborts ⇒ capture-unsafe op inside (roctracer needed); prerun-fail ⇒ forward itself fails. Knob-off dead code; sbatch t1r arm exports the env. **Leg 657196 queued** (same arm, nid002706).
- **657179 (same-session ctrl twin on run-l5-clone) COMPLETED**: p50 11.3/12.25, agg 21.85/23.06, all_paris=True. vs L5b 657173's 7.21/9.65 and l5-debug 657180's 9.02/12.02 — **the three same-session arms overlap; wall reads cannot resolve L5's effect on this rig (same-arm spread 5.01–14.05 documented)**. Trace gates remain the arbiter: blocking item tax −99.7% stands as the L5 verdict.
- **657180 (l5-debug) COMPLETED rc=0, all_paris=True, 17 min**: knobs batch_tel/host_tables/host_dbg all 1 (marker files verified in repo AND repo-prof). host_tables DEBUG semantics = read device tensors anyway and RAISE on disagreement with host ints ⇒ **a full leg with zero disagreement = host-int staging path validated bit-exact across every step** (not the item enumeration hoped; the remaining 48.6 items/step are 1.4 µs non-blocking, ~0.07 ms/step — near-zero value; syncs are documented load-bearing per L5 report).
- **Net session state**: deployed knobs bit-exact with proven in-trace mechanism (L5 blocking elimination); L4 fusions live but sub-resolution; Tier-1 v1 falsified (width-4 direct capture crashes; root cause narrowed to inside-window op via probe); best session p50 14.05 (ctrl) / 13.46 (T1R pre-crash).

## Tier-2 ROOT-CAUSED via probe (657196): the SECOND BeginCapture per path kills serve — capture-once fix landed; confirm leg 657200

- **657196 (Tier-2 discriminator probe) COMPLETED 29 min: phase-1 green (p50 7.65, runs 7.04/7.65/7.89, all_paris=True), phase-2 rc=1, and — decisive — NO SIGABRT.** Failure mode CHANGED from 657175's watchdog abort to a clean catchable error, which is itself the diagnosis.
- **KVPROBE evidence (raw serve log, merged bench log is grep-filtered — markers only in run-l6-clone/serve-657196.log): `prerun-begin ×4 / prerun-ok ×4`** (one per rank). The probe's "eager pre-run" of the capture-path forward actually invoked the capture closure (run_forward's `forward` IS `capture`), so the prerun was a FULL first capture — **and it succeeded on all 4 ranks**. The first capture is NOT the killer.
- **Mechanism proven**: the probe's second invocation (`result = forward()`) re-entered `capture()` on the SAME SparseCapturedForward → `torch.cuda.graph(self._graph)` → `capture_begin` refused at the TORCH level: `RuntimeError: This CUDAGraph instance already owns a captured graph` → `attempted=True` → retire → graceful eager fallback → clean SIGINT shutdown. Refusing the second capture before HIP = survival.
- **657175 (probe off) unified**: fresh bundle per request ⇒ fresh CUDAGraph instance passes torch's `has_graph_exec_` guard ⇒ second BeginCapture REACHES HIP ⇒ poisoned stream state ⇒ opaque error mid-window (faulthandler: graph.py:42) ⇒ NCCL watchdog SIGABRT, uncatchable. No prior retire line in serve-657175.log confirms no prior FAILED attempt — a successful first capture preceded it.
- **ROOT CAUSE (issue #43 class, now precise): any second hipStreamBeginCapture for the same capture path in one process aborts serve.** The retire guard only covered failed attempts; successful-then-re-capture was the open window.
- **Fix landed (always-on, fail-safe, no knob)**: `capture_guard._CAPTURED` process-global registry + `mark_captured(path)` (one-shot INFO, called in SparseCapturedForward.__init__ after the capture window closes) + `already_captured(path)`; `SparseExecution.capture_forward` raises new `CaptureOnceExhaustedError(CaptureRetiredError)` BEFORE any HIP call when the path already captured once → engine eager fallback. `reset_for_tests` clears both registries. All 3 files py_compile OK; sbatch t1r arm probe export removed (probe code stays env-gated). Cost: after a bundle close/invalidation the path serves eager for the process lifetime — bounded, and strictly better than process death.
- **657200 (confirm leg #1, blanket capture-once) FAILED 30:38 rc=1 — but ZERO SIGABRT across the whole leg including the phase-2 capture window: the abort class is DEAD.** Both phases rc=1 via bench-client TimeoutError, and the cause is a scope bug in my fix, not the mechanism:
  - Width-1 KEYTRACE census: 657175/657196 = 44 lines all `window-gone` (width-1 captures once, replays silently; re-capture after bundle close is the engine's NORMAL lifecycle and is proven safe — full legs, fast p50). 657200 = **1020 lines, 1012 `capture-not-ok`**: the blanket `_CAPTURED` refusal rejected the width-1 re-captures → permanent eager width-1 → B section (single-stream 256 tok) exceeded the client timeout → phase-1 rc=1, phase-2 inherited the slow serve → rc=1. mark_captured's INFO is filtered from the serve log (kvaas logger INFO invisible — only ERROR passes), so the one-shot success marker needed the KEYTRACE census to infer.
  - Path label confirmed in engine source: `sparse_runtime.py:483 path=f"glm5-sparse:{self.bucket.width}"`.
- **Rescope applied (py_compile OK)**: `capture_guard.re_capture_forbidden(path)` — parses trailing `:N`; N>1 (abort-proven width>1 sparse ladder) → capture-once enforced; `:1`/unlabeled → exempt (proven safe per-close-cycle). `capture_forward` refusal now gated on `already_captured(key) and re_capture_forbidden(key)`. mark_captured log message updated to the scoped truth.
- **657204 (confirm leg #2, rescoped) COMPLETED 29:31 rc1=0 rc2=1 — ZERO SIGABRT, phase-1 FULLY GREEN (all_paris=True, p50=7.53 / conc wall=271.7s agg=1.88; p50 matches 657196's no-fix 7.65 = documented same-arm spread, not the fix).** Width-1 census back to the canonical 44×window-gone (657200's 1012 capture-not-ok storm gone — the rescope restored the fast path). Width-4 census: 8 clean CaptureOnceExhaustedError raises (2/rank; the engine then marks the bucket capture-dead locally) + 1296 capture-not-ok eager decisions + first capture SUCCESS (mark_captured, INFO filtered from serve log). The abort class is DEAD and the width>1 refusal path is proven clean.
- **Phase-2 B: NEW distinct failure — one-shot `hipErrorNoBinaryForGpu` (HIP 209) at the vkernels `down` kernel** (glm_moe_grouped.py:386 → triton `_init_handles/load_binary`; first HIP-error text at serve-log:1719 via release('r18') — async-report caveat applies). Not fix-caused: 657200's serve log has the same hit. One-shot module-load failure: the SERVE SURVIVED (kept decoding after the 500; pins for r18 leaked). A-correctness ok=true (says_bern). Root cause of the load failure unknown — candidates: profiler-attached hipModuleLoad hazard, triton cache race across ranks, async context poison from an earlier kernel. Next diagnostic: AMD_SERIALIZE_KERNEL=3 leg.
- **No kineto trace in 657204 — ORDERING bug, not profiler failure: the E windows (E1 decode bs=1, E2 conc bs=4 = the Tier-2 target, E3 prefill) run LAST in glm5_serve_bench_kineto.py (after B/C/D), and B's 500 aborted before any trigger was consumed.** kin-657204/ empty; the export path itself is fine (per-window export at close).
- **Fix: phase_kineto (E) reordered to run directly after A** (before B/C/D) — traces land before the fragile phases; A-D reference numbers unchanged (E taints only its own latencies, per its own docstring); E1's radix-full-hit assumption preserved (A primes the prompt). py_compile OK; local campaign mirror synced. **657205 (trace leg) SUBMITTED** (t1r): gates = E1 width-1 captured-decode trace + E2 width-4 fallback trace on disk regardless of B's fate; E2's window (conc steps 16-45) traces the width-4 EAGER fallback (first capture happens at conc step ~1, before the 16-step poll cadence opens the window) — the captured-width-4 trace needs a shorter poll cadence (engine patch, parked).

## Tier-2 phase-2 HIP 209: boundary-crossing kill + per-rank triton cache (657205/657540/657541/657542)

- **657205/657540/657541: the phase-2 failure isolated to the FIRST request crossing the (1,256) sparse-window block-table boundary.** A probe = 142 tok, never crosses → always OK (657540 even completed A within the raised 1800s client timeout). Every 256-tok phase-2 run died once with the one-shot triton `hipErrorNoBinaryForGpu` (HIP 209) on the moe down-kernel load (glm_moe_grouped.py:386, `_init_handles/load_binary`). The E-runs-first reorder worked (657540 armed decode.trigger and reached E), but the window's 16-step poll cadence never opened before the failure hit.
- **657541 W-absorb test DECISIVE: the crossing failure kills the ENGINE WORKER LOOP** — the W retry (and every later submit) raises `RuntimeError("Floe engine loop failed") from self._worker_error` (engine.py:840) — the serve is permanently broken; absorption cannot work. (657204's post-500 KEYTRACE lines were pre-crash steps finishing, not recovery.)
- **Fix (657542): per-rank TRITON_CACHE_DIR.** apply_kineto_patch_bev.py now also inserts a module-top block into repo-prof's engine.py: `TRITON_CACHE_DIR=/dev/shm/triton-<pid>` per rank (env `FLOE_TRITON_CACHE_PER_RANK=1` default), isolating each rank from the shared-home triton cache — no stale or racing-partial entry is ever loaded. Root-cause hypothesis: the crossing-specific down specialization's shared-cache entry is corrupt (every phase-2 serve deterministically fails loading it; phase-1 never crosses, never loads it). Gate for 657542: W crossing passes → theory confirmed AND fixed, traces land; still HIP-209 → the fresh compile itself is bad → next levers: pre-warm the crossing specialization in a subprocess, or per-op reference fallback for the down kernel.
- Mirror state: glm5_serve_bench_kineto.py (E-first + W phase + 1800s timeouts) and apply_kineto_patch_bev.py (per-rank cache block) synced to ../floe-bev-main/.local/campaign/beverin/glm5-smoke/.

## Tier-2 KINETO TRACE LANDED (657550) — width-1 decode kernel breakdown

- **657550 = the first successful kineto harvest after 6 legs.** Winning recipe: (1) E-runs-first, (2) arm `decode.trigger` BEFORE the W crossing phase — the W request's decode steps run for minutes and only its RELEASE fails (one-shot HIP 209), so the window opens ≤16 steps into W and exports at step ~46, minutes before the release failure kills the loop. W's 500 was absorbed; the loop died on the W retry as expected; **the trace (rank0, 431 MB) was already on disk**: `run-l6-clone/kin-657550/kineto-decode-w1-cuda0-pid163535.json`. rc1=0 (phase-1 green again), rc2=1 (the known post-trace death).
- **Breakdown (30 traced steps, all EAGER — zero hipGraphLaunch in the window; ~4362 kernel launches/step, 130861 kernels total):**
  - **`resolve_rows` (the L6 rows kernel): 273.3 ms over 330 calls = 0.83 ms/call × 11/step ≈ 9.1 ms/step — the single largest compute kernel, ~41% of non-RCCL device time.** The rows scrubber's resolve dominates the width-1 decode compute.
  - vkernels fused ops: `_moe_gate_up` 58.8ms/1260 (1.96ms/step), `_gemv_bf16` 51.3ms/10200 (1.7), `_gemv_fp8` 37.2ms/1590 (1.2), `_moe_down` 35.6ms/1260 (1.2), `_big_fuse` 19.7ms/2700 (0.66), `_pre_gemv` 12.2ms/2700 (0.41), `_compose` 10.4ms/2700 (0.35), `_router` 9.8ms/1260 (0.33) ≈ **7.8 ms/step combined**.
  - aten glue (elementwise/reduce/index/copy) ≈ 3.5 ms/step; rocBLAS Cijk reference GEMMs ≈ 2 ms/step.
  - RCCL `rcclGenericKernel<2,false>`: 101.6 s over 2850 calls = ~95/step at 35.7 ms wall each — **profiler-skew-amplified collective wait** (the per-op profiler overhead desyncs the TP ranks; compute kernels' device durations remain valid, RCCL durations do not).
- **Tier-2 verdicts:** (a) the eager width-1 decode is launch-bound by construction (~4362 launches/step — the captured path's value quantified); (b) the fused-op ladder is NOT the compute bottleneck — resolve_rows is; (c) the next optimization target by device time = resolve_rows (0.83ms/call), then the aten glue (~40 copies/reduces per step that _big_fuse/_compose haven't absorbed).
- **The 657541/657542 serve-log analysis + probes that got here:** the W request decodes fine for minutes (captured replay + periodic eager re-decodes at pool-refill shape changes) and fails ONLY at release (async HIP 209 surfacing); standalone TP1 repro (657545-657548) passes ALL pointer-alignment classes (the compile+load path is healthy); the per-rank TRITON_CACHE_DIR verified ENGAGED in-container (657549: `/dev/shm/triton-189242`, triton not yet imported at engine import) yet the failure persisted → the corrupt-cache/race theories are DEAD; the failure is serve-state-dependent (TP4 + kvaas IPC + capture machinery), root cause still OPEN. Evidence preserved: serve-657541.log (the W window: 04:47:44 commit → 04:47:57/04:48:17 re-decode warnings → release FAILED), repro209.out, probe_tc.out.
- Trace kept on beverin for re-analysis; analyzer at /tmp/harvest_trace.py (beverin) — classifies steps via step[DECODE bs=1] ranges, splits captured/eager via GraphLaunch presence, aggregates per-kernel device time.

- **657555 (width-4 trace attempt): the conc requests die BEFORE the 16-step poll fires** — trigger unconsumed, no window, no trace; release('r18') FAILED with the HIP 209 within the conc phase's first ~15 steps (the width-4 capture-not-ok KEYTRACE ×4 ranks at 07:13:24 = the first steps). The poll cadence itself was the width-4 blocker. **Fix: FLOE_KINETO_POLL_EVERY knob** (default 16; the trace leg runs with =1 — poll every decode forward, the un-triggered cost stays one tmpfs stat). 657570 submitted with E2-PREW + POLL_EVERY=1; next turn: harvest `kin-657570/kineto-decode-w*-cuda*.json` with /tmp/harvest_trace.py (expect the width-4 eager-fallback breakdown = the capture-once residual cost; note the profiled conc phase runs ~3.3s/step under profiler skew — the export lands ~3 min after the phase starts, well before the release failures).

## Tier-2 KINETO width-4 HARVEST (657570) — eager-fallback breakdown, both widths complete

- **657570 with POLL_EVERY=1: the width-4 trace landed.** The E2-PREW arm's trigger was consumed at the conc phase's FIRST decode step (bs=1 — one request had committed; the rest joined within a step); the 30 traced steps = 28× `step[DECODE bs=4]` + 1× bs=1 + 1× bs=3 — **the width-4 eager fallback under capture-once** (zero hipGraphLaunch in the window; ~4809 launches/step, 236649 kernels, 782 MB trace: `kin-657570/kineto-decode-w1-cuda0-pid2825.json` — the filename says w1 (the width at OPEN), the content is width-4).
- **Width-4 per-step compute (~44 ms/step non-RCCL device time, vs ~22 ms at width-1 — 4× batch costs ~2× compute):**
  - `resolve_rows`: 271.7 ms/330 calls = **0.82 ms/call — batch-INVARIANT** (same 11/step as width-1) → 9.1 ms/step = 21% of compute (down from 41%). The rows resolve amortizes across the batch for free.
  - **The rocBLAS GEMM ensemble EXPLODES at width-4**: MT128x32x32 253.1 ms/4256 (142/step, 8.4 ms/step — was 22/step, 0.88 at width-1) + MT32x32x128 65.4 + MT128x96x128 52.0 + MT256x224x64 50.5 + MT512x176x32 40.8 + MT128x8x32 25.3 ≈ **10.5 ms/step (24%) — the #1 aggregate compute block at width-4** (the reference/batched-GEMM path scaling with batch).
  - vkernels moe ladder scales SUBLINEARLY: `_moe_gate_up` 0.144 ms/call ×42/step = 6.1 ms/step + `_moe_down` 0.075 ×42 = 2.7 (8.8 ms/step total vs 3.2 at width-1 — 2.75× for 4× batch ✓); `_big_fuse`/`_pre_gemv`/`_compose` counts batch-INVARIANT (2700/30/step) at 0.66+0.45+0.35 ms/step.
  - aten glue ≈ 8.6 ms/step (2.5× width-1); RCCL 95/step (batch-invariant, skew-inflated waits ~42 ms).
- **Tier-2 final verdicts (both widths):** (1) the fused-op ladder is NOT the decode bottleneck at either width; (2) the eager fallback's cost = launch count (~4400-4800/step) + the batch-scaling GEMM ensemble + resolve_rows; (3) the captured path kills all three — the capture-once residual cost is now QUANTIFIED; (4) next optimization targets by device time: the width-4 GEMM ensemble (why does the batched-reference path take 142 calls/step?), resolve_rows internals (0.82 ms/call at both widths), and the aten glue.
- 657570 still running at analysis time (the redundant W width-1 re-trace follows; rc2=1 expected — the post-W loop death).

- **657570 COMPLETED (36:45, rc1=0, rc2=1)** — the W re-trace after E2-PREW did not fire (one window per serve lifetime effectively; the conc phase consumed it); the width-4 deliverable trace is the leg's product. **TIER-2 KINETO CAMPAIGN COMPLETE: both widths traced, analyzed, verdicts ledgered.** Open items for the next arcs: (a) the one-shot HIP 209 root cause (serve-state-dependent; full evidence chain in the 657541/657542/657545-49/657550 sections); (b) optimization targets: the width-4 rocBLAS ensemble (142 calls/step — why the batched reference path?), resolve_rows internals (0.82 ms/call batch-invariant), the aten glue (~40-80 unfused copies/reduces per step).

## T2R grain-gate arc — FALSIFIED (657582/657606/657633)

Hypothesis: the width>1 re-capture abort (issue #43, 657175) is the
peer-mutation race (657018: a table flip inside the capture window), so a
grain-stability gate (re-capture only when NO live request can flip the
256-grain bt within G=4 steps — the bt width is a pure function of
MAX(lens), `_grain_headroom`) would make width>1 re-captures safe and lift
the conc phase off the permanent eager fallback (~2.8 tok/s/req).

Delivery lesson (cost two legs): repo-prof is `rm -rf`'d and rebuilt from
$SNAP/repo at every job start — engine patches must be baked into the
persistent $SNAP/repo (like the t1r patch) or the arm chain; manual
repo-prof edits die at sbatch. kvaas-src IS persistent (staged per job).
657582/657606 ran pre-gate code (their only signal: p50 14.37 vs 7.05 =
the eager/captured width-1 signatures; the spread is real).

657633 (patch verified in-tree, marker on, zero grain-gates = headroom
always sufficient, zero refusals = allowance worked):
- width-1: p50=14.48, runs=[14.06,14.48,14.51] — tight, captured, healthy.
- width-4 gated re-capture ATTEMPTED under a provably stable table ->
  `HIP error: operation not permitted when stream is capturing` (async-
  reported, hipErrorStreamCaptureUnsupported) x4 ranks -> c10::
  AcceleratorError -> SIGABRT. Same class as 657175.

Conclusion: the width>1 re-capture abort is NOT the grain race — a
capture-unsafe op exists in the width>1 second-capture path itself
(width-4 capture #1 succeeds; width-1 re-captures succeed 44/leg).
Capture-once for width>1 stays (marker reverted; 657667 = restore leg).
Next arc if pursued: standalone repro of width-4 capture#1->close->warm->
re-capture (the /tmp/repro209.sh pattern) to pin the illegal op; or skip
to the eager-path width-4 optimizations (rocBLAS GEMM ensemble 24%,
launch-bound 4809/step).

## T2R follow-ups (657667/657679) + width-1 variance census

- 657667 (gate OFF, restore leg): p50=13.29 [13.23,13.29,13.38], conc
  251.7s eager, all_paris — the capture-once status quo intact.
- 657679 (diagnostic: gate ON + CUDA_LAUNCH_BLOCKING=1): SIGABRT again;
  the AcceleratorError ("operation not permitted when stream is
  capturing") NEVER reaches python (zero "capture failed at ladder"
  warnings, zero Tracebacks) — it escapes at C++ level (terminate
  called) from a thread async to the decode loop. Serve-level evidence
  exhausted: pinning the illegal op needs a standalone width-4
  capture#1->close->warm->re-capture harness (single GPU, no serve).
- Width-1 p50 variance (7.05 vs 14.48 legs, identical code): the KEYTRACE
  census is IDENTICAL (44x window-gone, zero capture-not-ok, in 657606/
  657633/657667) — the capture/re-capture structure is byte-identical;
  the variance is NOT the sparse path. 657606 warmed within itself
  (6.2 -> 7.05 -> 11.94): an environmental warm-up artifact (pool/hot-
  tier cold start suspected; no telemetry in serve logs to confirm).
  Fast legs 13.2-14.5 = the true capability.

Next-arc menu (unchanged): (a) standalone width-4 re-capture repro -> the
deep capture fix; (b) eager-path batched gemv for bs<=4 (rocBLAS ensemble
24% of width-4 compute); (c) HIP 209 root cause.
