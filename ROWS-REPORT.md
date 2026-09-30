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
Awaiting the re-ship + re-run for the N/N confirmation; CPU semantic cases run anywhere,
GPU legs skip without CUDA.

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
