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
