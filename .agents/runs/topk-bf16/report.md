# LANE 16 — glm_indexer_topk bf16-widen follow-up

**Repo**: `/local/home/xiayao/Documents/code/vkernels` (main @ 6f6de90, lane 5's op still
uncommitted). **Scope**: local code + GB10 only, no rig. All lane files stay uncommitted
(`src/python/vkernels/torch_ops/glm_indexer_topk.py`, `tests/python/test_glm_indexer_topk.py`,
`meta/benchmarks/bench_glm_indexer_topk.py`); other lanes' untracked/modified files untouched.

## 1. The change: accept bf16 (and fp16) by widening IN-KERNEL

Lane 5's op gated on fp32 scores, so the deployed recipe
(`--model-opt gemm_dtype=bfloat16`), which makes floe's `index_scores` bf16, raised
`OpNotEligible` and the op never fired. The op now accepts **fp32 / bf16 / fp16**
(`OpNotEligible` otherwise), widening sub-fp32 inputs to fp32 **inside the kernel** —
`x.to(tl.float32)` immediately after each of the two SCORES loads — and narrowing the
emitted values back to the input dtype on store (`VALUES.dtype.element_ty`, the house
store-cast pattern). `values` now come back **in the input dtype** (torch.topk's contract,
drop-in for `index_scores.topk(k)`); indices stay int32; other dtypes (e.g. fp64) are
rejected.

**Why in-kernel and not at the wrapper boundary** (documented in the module docstring's
new DTYPE paragraph):

1. **Exactness is preserved by construction.** bf16→fp32 (and fp16→fp32) widening is
   exact — a mantissa bit-append; every bf16/fp16 value *is* an fp32 value — and
   monotone. So the fp32 radix key of the widened value is the ordered key of the
   *input* value: bf16 quantization-grid tie classes widen to identical fp32 bit
   patterns, the radix histogram still counts them exactly, and the lowest-index
   tie-break applies unchanged. Selection on bf16 is by construction identical to
   selection on `scores.float()` — no radix-histogram exactness argument is lost.
   A wrapper-boundary `.float()` would be equivalent semantically but strictly worse
   operationally: an extra elementwise cast launch + an fp32 `[T, N]` materialization
   per indexer layer (a real fraction of the 10-12 us/layer decode win), whereas the
   in-kernel widen keeps one launch and *halves* score load traffic (2 vs 4 B/elem).
2. **The fp32 path is bit-identical.** Triton's semantic layer elides identity casts
   (`cast` returns the input when src == dst scalar type — verified against the
   installed Triton 3.8.0), so the fp32 specialization compiles the same IR as before;
   all 10 pre-existing tests pass unchanged (one eligibility assertion in
   `test_contract` updated: bf16 is now accepted, fp64 carries the rejection case).

One documented cosmetic: the fp32→bf16 narrowing store canonicalizes returned NaN
*value* bits (PTX `cvt` NaN canonicalization — injected `0xFFC0` came back `0x7FFF`), but
NaN *order* keeps the sign-carrying key (positive NaN above +inf, negative NaN below
-inf; the widen itself preserves the sign — proven by the pinned ranking). floe never
feeds NaN (`nan_to_num` on the pool logits upstream; invalid pools `masked_fill` to
`finfo.min`), so this cannot fire in the wired path.

## 2. Tests (10 pre-existing green + 6 new, all passing on GB10)

`tests/python/test_glm_indexer_topk.py` — new GPU tests:

- `test_gpu_bf16_sorted_parity_vs_reference_and_torch` — full SHAPES sweep (decode/
  prefill envelopes + k edges): values+indices **exactly** the stable oracle on the
  widened fp32; values exactly torch.topk's bf16 values; selection sets exactly
  torch.topk's (bf16 randn rows carry exact ties → torch's tie order is unspecified, so
  only the set is pinned against torch, the order against the stable reference — this is
  the comparison that matters on quantization grids).
- `test_gpu_bf16_adversarial_ties` — all-equal row, quarter-step coarse grid,
  small-magnitude (~1e-3, coarse relative mantissa), the floe short-context shape
  (relu'd scores + `finfo(bf16).min`-masked tail with fewer valid pools than k), and
  ±0.0 mixes. All exact vs the stable reference.
- `test_gpu_bf16_k_edges` — k=1, k==N on an all-equal row (the m == K−a tie-fill
  invariant), k==N−1, non-power-of-two N.
- `test_gpu_bf16_nan_documented` — pinned sign-bit NaN ranking (positive NaN above
  +inf, negative NaN below −inf, −0.0 folded to the zero class), deterministic replay,
  payload canonicalization documented.
- `test_gpu_fp16_defensive_parity` — same sweep in fp16 (defensive acceptance; same
  lossless-widen argument).
- `test_gpu_bf16_unsorted_selection_set` — `sorted=False` on bf16: torch-equal
  selection set, distinct in-range indices, values pair via `gather`.

**Found while testing** (documented, not fixed — it is the pre-existing fp32 contract):
when the k-boundary lands inside the ±0.0 tie class, set parity vs torch.topk is
*impossible* — the kernel folds −0.0 onto +0.0 (one class, lowest-index break) while
torch's radix ranks +0.0 strictly above −0.0 by bits. The fp32 test suite never trips
this only by seed luck (its zeros row keeps ≥64 strictly-positive values, so the cut
misses the zero class). The bf16 test asserts the reference parity exactly and confines
any torch set-difference to the zero class; floe's relu'd scores never produce −0.0, so
the divergence cannot fire in the wired path.

## 3. GB10 microbench — the perf class MOVED at 3 shapes

`meta/benchmarks/bench_glm_indexer_topk.py` now sweeps both score dtypes (parity checked
before timing, CUDA-graph medians, k=512; full JSON:
`bench_glm_indexer_topk.json` alongside this report). torch 2.14.0+cu130, GB10.

| shape | dtype | torch_sorted | torch_unsorted | vk_sorted | vk_unsorted | gate |
|---|---|---:|---:|---:|---:|---|
| T=1 N=4096 | fp32 | 30.0 | 18.2 | 11.7 | 10.9 | PASS |
| T=1 N=4096 | bf16 | 23.3 | 15.2 | **11.5** | **9.8** | PASS |
| T=1 N=16384 | fp32 | 61.1 | 52.0 | 27.3 | 25.4 | PASS |
| T=1 N=16384 | bf16 | 44.8 | 41.9 | **26.0** | **25.2** | PASS |
| T=8 N=32768 | fp32 | 52.0 | 40.7 | 50.7 | 49.2 | FAIL (known limit) |
| T=8 N=32768 | bf16 | 38.2 | 30.3 | 46.3 | 42.7 | FAIL (known limit) |
| T=32 N=32768 | fp32 | 69.0 | 58.0 | 52.8 | 49.0 | PASS |
| T=32 N=32768 | bf16 | 50.5 | 42.6 | 46.0 | 44.3 | **FAIL (moved)** |
| T=64 N=8192 | fp32 | 57.2 | 46.4 | 31.5 | 27.5 | PASS |
| T=64 N=8192 | bf16 | 42.2 | 34.4 | **30.3** | **26.2** | PASS |
| T=512 N=8192 | fp32 | 210.7 | 183.9 | 154.5 | 134.0 | PASS |
| T=512 N=8192 | bf16 | 120.3 | 102.0 | 148.5 | 130.3 | **FAIL (moved)** |
| T=2048 N=32768 | fp32 | 5954.7 | 5808.3 | 2215.8 | 2106.8 | PASS |
| T=2048 N=32768 | bf16 | 2143.1 | 2063.0 | **2007.4** | **1894.9** | PASS |

**Verdict: the perf class moved — fp32 gate 6/7 PASS, bf16 gate 4/7 PASS.** The win
*holds and widens* at bs=1 decode (both context lengths), bs=64 decode, and 2048-row
prefill. It *regresses* at bf16-only **T=32 N=32768** (44.3 vs 42.6 us — a 4% loss to the
free knob) and **prefill 512×8192** (130.3 vs 102.0 us); T=8 N=32768 fails in both
dtypes (the already-documented one-CTA-per-row latency limit). Cause: torch's multi-pass
radix select is **bandwidth-bound** and roughly halves under bf16, while this kernel's
per-row histogram/emit chain is **latency/compute-bound** (the widen adds per-element
convert work while only the two score scans get cheaper) — so bf16 narrows the margin as
row count grows. A num_warps sweep {4,8,16,32} on the two moved shapes does not close the
gap (best still loses: 44.3/123.7 vs 44.1/100.6), so it is structural, not a tuning knob.
The module docstring's PERF GATE paragraph now carries the bf16 numbers and the advice
that wiring be per-shape with fallback to the incumbent where the gate fails.

## 4. Floe wiring note (one paragraph)

The two seam sites — `floe/engine/runner/models/glm53flash/forward.py:3431`
(`_decode_glue_rows`, decode glue path) and `:3477` (`_index_rows`, prefill/chunked
path) — both build `index_scores` as `torch.matmul(_g(weights).unsqueeze(-2),
scores)` where `scores = relu(matmul(_g(q), _g(pool_keys)) * softmax_scale)`, i.e. the
scores dtype is **exactly the `_gemm_dtype()` policy**: with the deployed
`--model-opt gemm_dtype=bfloat16`, `_g()` casts both matmul operands to bf16 and
`index_scores` is bf16 end-to-end (Python-scalar `softmax_scale` and `relu` preserve
dtype, `_tp_scores_reduce` is a dtype-preserving all-reduce, and the invalid-pool
`masked_fill` already adapts with `finfo(index_scores.dtype).min`); nothing upstream
forces fp32 — fp32 appears *only* under the exact-gemm policy (`gemm_dtype=None`, where
`_g()` upcasts, the HF-cross-check reference). So with this lane's bf16 acceptance the
op is eligible under **both** recipes, and since bf16→fp32 widening is lossless the
selection is identical to what the fp32 policy would produce on the same (unquantized)
scores modulo the bf16 quantization that already happened in the GEMM — i.e. parity with
the incumbent `index_scores.topk(select_k)` is by construction (ties still break
lowest-index; torch's tie order is unspecified but stable in practice). Wiring at those
sites needs only: `values, idx = glm_indexer_topk(index_scores, select_k, sorted=not
_opt("indexer_topk_unsorted"))` → `selected = idx.long()` (the op returns int32; the
sites' `valid_candidates.gather(-1, selected)` wants int64 — one `.long()` on a
`[T, 512]` tensor), with `OpNotEligible` falling back to the incumbent topk; per the
bench above, gate the wiring per shape (engage bs=1–64 decode and large prefill; let
T=8/N=32k and the 512×8192 bf16 prefill class stay on torch).

## Files

- `src/python/vkernels/torch_ops/glm_indexer_topk.py` — in-kernel widen + input-dtype
  values + eligibility; DTYPE and bf16-perf docstring paragraphs (lane file, uncommitted).
- `tests/python/test_glm_indexer_topk.py` — 6 new GPU tests + contract update
  (uncommitted).
- `meta/benchmarks/bench_glm_indexer_topk.py` — dtype sweep (uncommitted).
- `.agents/runs/topk-bf16/{report.md,bench_glm_indexer_topk.json}` — this report + data.

Validation: `.venv/bin/python -m pytest tests/python/test_glm_indexer_topk.py -q` →
**16 passed** (10 pre-existing fp32 tests unchanged and green; fp32 semantics
bit-identical). Bench: parity-checked before timing, both dtypes.
