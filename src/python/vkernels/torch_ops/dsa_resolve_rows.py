"""DSA sparse-attention row resolver — fresh vectorized Triton kernel (rows lane).

Resolves DSA indexer top-k token positions to paged-KV hot/resident row ids,
transliterating the kvaas production ``resolve_rows`` scalar kernel
(``kvaas_runtime/sparse_attention/kernels.py``) bit-exactly while removing
its cost structure. That kernel is the beverin #1-latency step's #2 sink:
11.4 calls x 828 us = 9.11 ms of a 43.49 ms B=1 step (20.9%).

ROOT CAUSE of the 828 us (why index/addr math costs 16x its memory budget):

* **Scalar K-walk.** The selection pass is ``for i in range(K)`` with K=2051
  *scalar* loop iterations. Each iteration is a serialized scalar
  ``tl.load`` of one selected token (loop-carried dependency — the
  compiler cannot vectorize or batch them) plus, on the miss branch, two
  full BLOCK_H-wide reductions (hit scan + LRU victim scan).
* **The compare storm.** Every one of the K iterations pays
  ``tags == token`` / ``tag_generations == generation`` compare vectors
  over BLOCK_H=4096 lanes (hot_rows 2176 -> next pow2), and the LRU
  protect pre-pass pays it AGAIN over all K. ~2051 x 4096 x ~4 lane-ops
  of pure integer compare per (request, layer) state — the exact
  "O(H*K) compare storm" the kvaas W6 header documents (~1.9G lane-ops).
* **Register pressure.** Three int64 BLOCK_H vectors (tags,
  tag_generations, ages) must stay live across the whole K walk; the kvaas
  header records 1240 spill instructions at BLOCK_H=16384 and a tuned
  num_warps heuristic just to keep BLOCK_H=4096 from spilling.
* **NOT host sync / launch overhead / H2D.** The launch is a single plain
  Triton launch with no synchronization (the W8A census shows the 828 us
  as device-busy kernel time); the cost is entirely in-kernel compute
  shape. The W6 map-based rewrite attacked the same storm and was
  perf-REJECTED (2.5-4.4x slower — structural barrier costs at serving
  geometry); this module is a fresh design, NOT that one.

THE FIX (this module): keep the scalar walk ONLY for the small miss set
(production hit rate ~0.9 => ~0-8 misses/state), vectorize everything else:

  1. Phase 1 vectorizes the K selection in BLOCK_K-wide chunks: one
     coalesced token load, page-table gather, resident-row arithmetic,
     output store, error count, and an ordered miss-worklist compaction
     (cumsum) — the resident fast path (>=90% of selections) never
     touches the hot-tier state machine.
  2. The K x H protect compare storm becomes a selected-token BITMAP
     (K vector atomic_or into token-domain words, then one H-lane
     gather-and-test). O(K + H) instead of O(K*H).
  3. Phase 3 transliterates the original miss branch scalar-by-scalar
     over the compacted worklist only (dup tokens, victim ties, fence
     mismatch, tag/age patches, fill records — bit-exact by
     construction), so state evolution matches the production kernel
     exactly while its cost drops from K to m scalar iterations.
  4. Epilogue + host->hot copy pass are verbatim.

Expected: memory-bound ~50-60 us/state at the production geometry
(topk=2048, hot_rows=2176, 11 DSA layers) => 11 x ~60 us ~= 0.66 ms/step,
vs 9.11 ms. A batched one-launch entry point
(:func:`dsa_resolve_rows_batched`) is provided, but note the honest
limit: the per-layer selections only exist mid-forward (layer L's
attention consumes layer L's rows), so drop-in batching requires the
two-pass restructure (all indexers -> one resolve -> all attentions);
the drop-in per-layer calls already meet the <=0.7 ms target.

Bit-exactness contract: integer/address math only — every output row,
counter, error, fill pair, and the resulting tags/tag_generations/ages/
clock state are required EQUAL to the production kernel, no tolerance.
:func:`resolve_rows_reference` restates the production semantics as the
CPU oracle; ``tests/python/test_dsa_resolve_rows.py`` asserts
``torch.equal`` on randomized + targeted cases (duplicates, victim ties,
fence mismatch, boundary pages, zero-length rows, non-pow2 topk,
ALLOW_PADDING) and runs on-cluster.

Stdlib+torch at import; triton JIT-compiles on first CUDA call (the
per-layer calls are warmed eagerly before capture — the OQ-6 probe — so
a cold compile never lands under capture).
"""

from __future__ import annotations

import torch

from ._dispatch import OpNotEligible

__all__ = ["dsa_resolve_rows", "dsa_resolve_rows_batched", "resolve_rows_reference"]


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

def _kernel():
    """JIT-scoped kernel definition (import triton lazily, moe_combine style)."""
    import triton
    import triton.language as tl

    _INT64_MAX = tl.constexpr(0x7FFFFFFFFFFFFFFF)

    @triton.jit
    def _resolve_rows_fast(
        KV, HOST, FILL, RESIDENT, BACKING, GENERATIONS, LENGTHS, HOT_SLOTS,
        SELECTED, TAGS, TAG_GENERATIONS, AGES, CLOCK,
        OUTPUT, COUNTERS, ERRORS, FENCE, EXPECTED,
        BITMAP, WORKLIST,
        PAGE: tl.constexpr, WIDTH: tl.constexpr, GPU_ROWS: tl.constexpr,
        HOST_ROWS: tl.constexpr, PAGES: tl.constexpr, BATCH: tl.constexpr,
        K: tl.constexpr, H: tl.constexpr, HOT_PAGES: tl.constexpr,
        WORDS: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr, FILL_CHUNK: tl.constexpr,
        ALLOW_PADDING: tl.constexpr, HAS_FENCE: tl.constexpr,
    ):
        request = tl.program_id(0)
        layer = tl.program_id(1)
        state = layer * BATCH + request
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, BLOCK_D)
        hm = h < H
        length = tl.load(LENGTHS + request)

        # ---- 0. clear this state's selected-token bitmap ----
        w = tl.arange(0, 1024)
        for w0 in range(0, WORDS, 1024):
            tl.store(BITMAP + state * WORDS + w0 + w, 0, mask=w0 + w < WORDS)
        tl.debug_barrier()

        # ---- 1. vector classify + bitmap build + ordered miss compaction ----
        resident_count = tl.zeros((), tl.int64)
        err_count = tl.zeros((), tl.int64)
        workcnt = tl.zeros((), tl.int32)
        n_chunks = (K + BLOCK_K - 1) // BLOCK_K
        for c0 in range(0, n_chunks):
            jv = c0 * BLOCK_K + tl.arange(0, BLOCK_K)
            jm = jv < K
            tok = tl.load(SELECTED + request * K + jv, mask=jm, other=-1)
            valid = jm & (tok >= 0) & (tok < length)
            # bitmap mark (valid tokens only — mirrors the protect pre-pass
            # gating on `valid`): O(K) atomics replace O(K*H) compares.
            bit = (1 << (tok & 31)).to(tl.int32)
            tl.atomic_or(BITMAP + state * WORDS + (tok >> 5), bit, mask=valid)
            page = tok // PAGE
            r = tl.load(RESIDENT + request * PAGES + page, mask=valid, other=-1)
            is_res = valid & (r >= 0)
            row = r * PAGE + (tok % PAGE)
            tl.store(OUTPUT + state * K + jv, row, mask=is_res)
            resident_count += tl.sum(is_res.to(tl.int64))
            miss = valid & ~is_res
            pref = tl.cumsum(miss.to(tl.int32), 0)
            tl.store(WORKLIST + state * K + workcnt + pref - 1, jv, mask=miss)
            workcnt += tl.sum(miss.to(tl.int32))
            errv = jm & ~valid
            if ALLOW_PADDING:
                errv = errv & (tok != -1)
            err_count += tl.sum(errv.to(tl.int64))
            tl.store(OUTPUT + state * K + jv, -1, mask=jm & ~valid)
        tl.debug_barrier()

        # ---- 2. hot-tier state as BLOCK_H vectors + protect from bitmap ----
        tags = tl.load(TAGS + state * H + h, hm, -1)
        tag_generations = tl.load(TAG_GENERATIONS + state * H + h, hm, -1)
        ages = tl.load(AGES + state * H + h, hm, 0)
        clock = tl.load(CLOCK + state) + 1
        gen = tl.load(GENERATIONS + request * PAGES + tags // PAGE,
                      mask=hm & (tags >= 0), other=-1)
        word = tl.load(BITMAP + state * WORDS + (tags >> 5),
                       mask=hm & (tags >= 0), other=0)
        selbit = ((word >> (tags & 31).to(tl.int32)) & 1) != 0
        # identical protected set as the production pre-pass:
        # valid-selected AND tag at its page's current generation.
        protected = hm & (tags >= 0) & (gen == tag_generations) & selbit

        # ---- 3. scalar walk over the (small) ordered miss set ----
        # Transliteration of the production miss branch, iteration for
        # iteration: dup tokens hit slots reserved by earlier misses,
        # victims take min-age/lowest-lane among unprotected lanes, the
        # fence gates every claim, fills record (src host row, dst hot row).
        hit_count = tl.zeros((), tl.int64)
        miss_count = tl.zeros((), tl.int64)
        for i in range(0, workcnt):
            mi = tl.load(WORKLIST + state * K + i)
            token = tl.load(SELECTED + request * K + mi)
            page = token // PAGE
            offset = token % PAGE
            generation = tl.load(GENERATIONS + request * PAGES + page)
            hit = hm & (tags == token) & (tag_generations == generation)
            slot = tl.min(tl.where(hit, h, BLOCK_H), 0)
            is_hit = slot < H
            if not is_hit:
                scores = tl.where(hm & ~protected, ages, _INT64_MAX)
                oldest = tl.min(scores, 0)
                slot = tl.min(
                    tl.where(hm & ~protected & (ages == oldest), h, BLOCK_H), 0)
                # H >= K guarantees a victim after protecting selected hits.
            # claim/fence resolved as scalar predicates, NOT nested mutation:
            # job-656618 — loop-carried TENSOR updates mutated three scf
            # levels deep (if slot<H → if fenced → else of if is_hit) lost
            # their yields on this Triton (tags/tag_generations regressed to
            # pre-loop values at the epilogue store) while scalars at the
            # same depth and tensors at depth 2 (protected/ages) survived.
            # All carried-tensor mutations now happen at loop-body top level
            # gated by predicates; only side-effect stores stay in the if.
            claim = slot < H
            hot_page = tl.load(HOT_SLOTS + request * HOT_PAGES + slot // PAGE,
                               mask=claim, other=0)
            if HAS_FENCE:
                fenced = tl.load(FENCE + hot_page, mask=claim, other=0) == \
                    tl.load(EXPECTED + hot_page, mask=claim, other=-1)
            else:
                fenced = claim
            ok = claim & fenced
            is_evict = ok & ~is_hit
            row_id = tl.where(ok, hot_page * PAGE + slot % PAGE,
                              -1).to(tl.int64)
            hit_count += (ok & is_hit).to(tl.int64)
            err_count += (claim & ~fenced).to(tl.int64)
            if is_evict:
                host_page = tl.load(BACKING + request * PAGES + page)
                tl.store(FILL + (state * K + miss_count) * 2,
                         host_page * PAGE + offset)
                tl.store(FILL + (state * K + miss_count) * 2 + 1,
                         hot_page * PAGE + slot % PAGE)
                miss_count += 1
            # carried-tensor updates, flattened (upd is all-false when
            # slot == BLOCK_H, so unclaimed items are safe no-ops)
            upd = h == slot
            tags = tl.where(is_evict & upd, token, tags)
            tag_generations = tl.where(is_evict & upd, generation,
                                       tag_generations)
            ages = tl.where(ok & upd, clock, ages)
            protected = protected | (ok & upd)
            tl.store(OUTPUT + state * K + mi, row_id)
        tl.debug_barrier()

        # ---- 4. epilogue (verbatim) ----
        tl.store(TAGS + state * H + h, tags, hm)
        tl.store(TAG_GENERATIONS + state * H + h, tag_generations, hm)
        tl.store(AGES + state * H + h, ages, hm)
        tl.store(CLOCK + state, clock)
        tl.store(COUNTERS + state * 3, resident_count)
        tl.store(COUNTERS + state * 3 + 1, hit_count)
        tl.store(COUNTERS + state * 3 + 2, miss_count)
        tl.store(ERRORS + state, err_count)

        # ---- 5. copy pass: drain the fill list in tiles (verbatim) ----
        # the tail [miss_count, K) is outside the contract (consumers read
        # only [0, miss_count)) but must be DETERMINISTIC for bit-exact
        # parity: zero it in the same loop (scratch buffers are reused
        # across calls within a decode step — job-656573 lesson).
        for j in range(0, K, FILL_CHUNK):
            jv = j + tl.arange(0, FILL_CHUNK)
            live = jv < miss_count
            src = tl.load(FILL + (state * K + jv) * 2, live, 0)
            dst = tl.load(FILL + (state * K + jv) * 2 + 1, live, 0)
            tl.store(FILL + (state * K + jv) * 2, 0,
                     mask=(jv >= miss_count) & (jv < K))
            tl.store(FILL + (state * K + jv) * 2 + 1, 0,
                     mask=(jv >= miss_count) & (jv < K))
            tile = live[:, None] & (d < WIDTH)[None, :]
            values = tl.load(
                HOST + layer * HOST_ROWS * WIDTH + src[:, None] * WIDTH + d[None, :],
                tile, 0)
            tl.store(
                KV + layer * GPU_ROWS * WIDTH + dst[:, None] * WIDTH + d[None, :],
                values, tile)

    return _resolve_rows_fast


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------

def _launch_checked(page_tokens, kv, host, resident, backing, generations,
                    lengths, hot_slots, selected, tags, tag_generations, ages,
                    clock, output, counters, errors, fills, bitmap, worklist,
                    fence_values, fence_expected, *, allow_padding, has_fence,
                    block_k=1024, fill_chunk=32):
    import triton

    layers, gpu_rows, width = kv.shape
    batch, pages = resident.shape
    hot_rows = tags.shape[-1]
    hot_pages = hot_slots.shape[1]
    k = selected.shape[-1]
    words = (pages * page_tokens + 31) // 32
    block_h = triton.next_power_of_2(hot_rows)
    num_warps = max(4, min(16, block_h // 512))
    _kernel()[(batch, layers)](
        kv, host, fills, resident, backing, generations, lengths, hot_slots,
        selected, tags, tag_generations, ages, clock,
        output, counters, errors, fence_values, fence_expected,
        bitmap, worklist,
        page_tokens, width, gpu_rows, host.shape[1], pages, batch,
        k, hot_rows, hot_pages, words,
        block_k, block_h, triton.next_power_of_2(width), fill_chunk,
        allow_padding, has_fence,
        num_warps=num_warps,
    )


def _validate_common(kv, host, resident, backing, generations, lengths,
                     hot_slots, selected, tags, tag_generations, ages, clock):
    def _req(cond, msg):
        if not cond:
            raise OpNotEligible(msg)

    _req(kv.is_cuda and kv.is_contiguous() and kv.dim() == 3,
         "dsa_resolve_rows contract: contiguous CUDA kv [layers, rows, width]")
    _req(host.is_cpu and host.is_pinned() and host.is_contiguous() and host.dim() == 3,
         "dsa_resolve_rows contract: contiguous PINNED CPU host [layers, host_rows, width]")
    _req(host.shape[0] == kv.shape[0] and host.shape[2] == kv.shape[2],
         "dsa_resolve_rows contract: host/KV layer and width must match")
    _req(host.dtype == kv.dtype and kv.dtype in (torch.float16, torch.bfloat16, torch.float32),
         "dsa_resolve_rows contract: matching fp16/bf16/fp32 kv+host dtype")
    dev = kv.device
    for name, t, nd in (("resident", resident, 2), ("backing", backing, 2),
                        ("generations", generations, 2), ("hot_slots", hot_slots, 2)):
        _req(t.is_cuda and t.device == dev and t.dtype == torch.int64
             and t.is_contiguous() and t.dim() == nd,
             f"dsa_resolve_rows contract: {name} must be contiguous CUDA int64")
    _req(resident.shape == backing.shape == generations.shape,
         "dsa_resolve_rows contract: resident/backing/generations shapes must match")
    _req(lengths.is_cuda and lengths.device == dev and lengths.dtype == torch.int64
         and lengths.shape == (resident.shape[0],),
         "dsa_resolve_rows contract: lengths must be CUDA int64 [batch]")
    _req(hot_slots.shape[0] == resident.shape[0],
         "dsa_resolve_rows contract: hot_slots batch must match")
    for name, t in (("tags", tags), ("tag_generations", tag_generations),
                    ("ages", ages)):
        _req(t.is_cuda and t.device == dev and t.dtype == torch.int64
             and t.is_contiguous() and t.dim() == 3,
             f"dss_resolve_rows contract: {name} must be contiguous CUDA int64 [layers, batch, hot_rows]")
    _req(clock.is_cuda and clock.device == dev and clock.dtype == torch.int64
         and clock.is_contiguous() and clock.dim() == 2,
         "dsa_resolve_rows contract: clock must be contiguous CUDA int64 [layers, batch]")
    _req(selected.is_cuda and selected.device == dev and selected.dtype == torch.int64
         and selected.is_contiguous() and selected.dim() == 2,
         "dsa_resolve_rows contract: selected must be contiguous CUDA int64 [batch, topk]")
    _req(selected.shape[0] == resident.shape[0],
         "dsa_resolve_rows contract: selected batch must match tables")
    _req(tags.shape[0] == kv.shape[0] and tags.shape[1] == resident.shape[0],
         "dsa_resolve_rows contract: tag state layers/batch must match kv/tables")


def dsa_resolve_rows(kv, host, resident, backing, generations, lengths,
                     hot_slots, selected, tags, tag_generations, ages, clock,
                     *, page_tokens, allow_padding=False, fence=None,
                     scratch=None, block_k=1024, fill_chunk=32):
    """Resolve one DSA layer's top-k selection to hot/resident row ids.

    Drop-in for the kvaas per-layer ``SparseRowResolver.resolve``: same
    tables, same bit-exact outputs, vectorized kernel (see module docstring).
    ``fence`` is an optional ``(values, expected)`` pair of CUDA int64
    ``[gpu_pages]`` tensors (``SparseHotSlotFence`` semantics). ``scratch``
    is an optional dict caching the bitmap/worklist/output/counters/errors/
    fills allocations across calls (callers inside a decode step should
    cache — see the floe patch in ROWS-REPORT.md); any missing entry is
    allocated. Returns ``(output, counters, errors, fills)`` where
    ``output`` is ``[batch, top_k]`` int64 row ids (-1 masked) and ``fills``
    is ``[batch, top_k, 2]`` (source host row, destination hot row) with
    entries at or beyond counters[..., 2] stale — identical to the
    production contract.

    Raises :class:`OpNotEligible` on any contract miss; the caller keeps
    the kvaas resolver as the fallback and the parity oracle.
    """
    _validate_common(kv, host, resident, backing, generations, lengths,
                     hot_slots, selected, tags, tag_generations, ages, clock)
    layers, gpu_rows, width = kv.shape
    batch, pages = resident.shape
    hot_rows = tags.shape[-1]
    k = selected.shape[-1]
    if scratch is None:
        scratch = {}
    dev = kv.device

    def _buf(name, shape, dtype=torch.int64, fill=None):
        t = scratch.get(name)
        if t is None or t.shape != shape or t.dtype != dtype or t.device != dev:
            t = torch.zeros(shape, device=dev, dtype=dtype) if fill == 0 else torch.empty(shape, device=dev, dtype=dtype)
            scratch[name] = t
        return t

    output = _buf("output", (batch, k))
    counters = _buf("counters", (batch, 3), fill=0)
    errors = _buf("errors", (batch,), fill=0)
    fills = _buf("fills", (batch, k, 2))
    bitmap = _buf("bitmap", (batch, (pages * page_tokens + 31) // 32),
                  dtype=torch.int32)
    worklist = _buf("worklist", (batch, k), dtype=torch.int32)
    if fence is None:
        # dummy pointers, never dereferenced with HAS_FENCE=False (kvaas trick)
        fence_values = fence_expected = tags
    else:
        fence_values, fence_expected = fence
    _launch_checked(page_tokens, kv, host, resident, backing, generations,
                    lengths, hot_slots, selected, tags, tag_generations, ages,
                    clock, output, counters, errors, fills, bitmap, worklist,
                    fence_values, fence_expected, allow_padding=bool(allow_padding),
                    has_fence=fence is not None,
                    block_k=block_k, fill_chunk=fill_chunk)
    return output, counters, errors, fills


def dsa_resolve_rows_batched(kv, host, resident, backing, generations, lengths,
                             hot_slots, selected, tags, tag_generations, ages,
                             clock, *, page_tokens, allow_padding=False,
                             fence=None, block_k=1024, fill_chunk=32):
    """One launch for ALL DSA layers: ``selected`` is ``[layers, batch, topk]``.

    Same tables/state layout as the per-layer entry (``[layers, batch, ...]``
    state — stack the per-layer resolvers' state tensors once per step), one
    ``grid = (batch, layers)`` launch. Bit-exactness is per-state identical
    to the per-layer path (the kernel is state-indexed, grid-order free).

    NOTE the call-site constraint: drop-in batching needs the two-pass
    decode restructure (all indexers run, then one resolve, then all
    attentions) because layer L's selection only exists mid-forward. Until
    that lands, the per-layer calls at ~50-60 us already meet the target.
    """
    _validate_common(kv, host, resident, backing, generations, lengths,
                     hot_slots, selected[0], tags, tag_generations, ages, clock)
    layers, gpu_rows, width = kv.shape
    batch, pages = resident.shape
    hot_rows = tags.shape[-1]
    k = selected.shape[-1]
    dev = kv.device
    output = torch.empty((layers, batch, k), device=dev, dtype=torch.int64)
    counters = torch.zeros((layers, batch, 3), device=dev, dtype=torch.int64)
    errors = torch.zeros((layers, batch), device=dev, dtype=torch.int64)
    fills = torch.empty((layers, batch, k, 2), device=dev, dtype=torch.int64)
    bitmap = torch.empty((layers, batch, (pages * page_tokens + 31) // 32),
                         device=dev, dtype=torch.int32)
    worklist = torch.empty((layers, batch, k), device=dev, dtype=torch.int32)
    if fence is None:
        fence_values = fence_expected = tags
    else:
        fence_values, fence_expected = fence
    _launch_checked(page_tokens, kv, host, resident, backing, generations,
                    lengths, hot_slots, selected, tags, tag_generations, ages,
                    clock, output, counters, errors, fills, bitmap, worklist,
                    fence_values, fence_expected, allow_padding=bool(allow_padding),
                    has_fence=fence is not None,
                    block_k=block_k, fill_chunk=fill_chunk)
    return output, counters, errors, fills


# ---------------------------------------------------------------------------
# Reference oracle (CPU, pure torch): the PRODUCTION scalar kernel's exact
# semantics, one layer at a time.
# ---------------------------------------------------------------------------

def resolve_rows_reference(resident, backing, generations, lengths, hot_slots,
                           selected, tags, tag_generations, ages, clock,
                           *, page_tokens, allow_padding=False,
                           fence_values=None, fence_expected=None):
    """Bit-exact CPU oracle of the kvaas production ``resolve_rows``.

    Single layer: state tensors are ``[batch, ...]``, ``selected`` is
    ``[batch, K]`` int64 (any device; computed on CPU). Returns a dict with
    ``output [batch, K]``, ``counters [batch, 3]``, ``errors [batch]``,
    ``fills [batch, K, 2]`` and the advanced ``tags``/``tag_generations``/
    ``ages``/``clock`` state — all bit-exact against the production kernel
    by transliteration (scalar walk over K in selection order).
    """
    resident = resident.long().cpu()
    backing = backing.long().cpu()
    generations = generations.long().cpu()
    lengths = lengths.long().cpu()
    hot_slots = hot_slots.long().cpu()
    selected = selected.long().cpu()
    tags = tags.long().cpu().clone()
    tag_generations = tag_generations.long().cpu().clone()
    ages = ages.long().cpu().clone()
    clock = clock.long().cpu().clone()
    has_fence = fence_values is not None
    if has_fence:
        fence_values = fence_values.long().cpu()
        fence_expected = fence_expected.long().cpu()
    batch = resident.shape[0]
    pages = resident.shape[1]
    hot_rows = tags.shape[-1]
    k = selected.shape[-1]
    page = page_tokens
    output = torch.full((batch, k), -1, dtype=torch.int64)
    counters = torch.zeros((batch, 3), dtype=torch.int64)
    errors = torch.zeros((batch,), dtype=torch.int64)
    fills = torch.zeros((batch, k, 2), dtype=torch.int64)
    for b in range(batch):
        length = int(lengths[b])
        protected = torch.zeros(hot_rows, dtype=torch.bool)
        t_b = tags[b].clone()
        g_b = tag_generations[b].clone()
        a_b = ages[b].clone()
        clock_b = int(clock[b]) + 1
        # production protect pre-pass: a hot lane whose tag is a VALID
        # SELECTED token at that page's current generation is immune to
        # eviction until the walk reaches it (kvaas does this as the
        # O(K·H) compare pre-pass; the kernel builds the identical set
        # from the selected-token bitmap — job-656618 follow-up: the
        # oracle was missing it, a latent kernel↔oracle divergence).
        sel_valid = selected[b][(selected[b] >= 0) & (selected[b] < length)]
        tagged = (t_b >= 0).nonzero(as_tuple=True)[0]
        if tagged.numel():
            tt = t_b[tagged]
            protected[tagged] = (
                (generations[b, tt // page] == g_b[tagged])
                & torch.isin(tt, sel_valid)
            )
        res_n = hit_n = miss_n = err_n = 0
        for i in range(k):
            token = int(selected[b, i])
            valid = 0 <= token < length
            row_id = -1
            if valid:
                pg = token // page
                off = token % page
                r = int(resident[b, pg])
                if r >= 0:
                    row_id = r * page + off
                    res_n += 1
                else:
                    gen = int(generations[b, pg])
                    hit = (t_b == token) & (g_b == gen)
                    hit_idx = hit.nonzero(as_tuple=True)[0]
                    slot = int(hit_idx[0]) if hit_idx.numel() else hot_rows
                    is_hit = slot < hot_rows
                    if not is_hit:
                        scores = torch.where(~protected, a_b,
                                             torch.full_like(a_b, 2**63 - 1))
                        oldest = int(scores.min())
                        cand = (~protected & (a_b == oldest)).nonzero(as_tuple=True)[0]
                        slot = int(cand[0]) if cand.numel() else hot_rows
                    if slot < hot_rows:
                        hot_page = int(hot_slots[b, slot // page])
                        fenced = True
                        if has_fence:
                            fenced = (int(fence_values[hot_page])
                                      == int(fence_expected[hot_page]))
                        if fenced:
                            if is_hit:
                                hit_n += 1
                            else:
                                host_page = int(backing[b, pg])
                                dst = hot_page * page + slot % page
                                fills[b, miss_n, 0] = host_page * page + off
                                fills[b, miss_n, 1] = dst
                                miss_n += 1
                                t_b[slot] = token
                                g_b[slot] = gen
                            row_id = hot_page * page + slot % page
                            protected[slot] = True
                            a_b[slot] = clock_b
                        else:
                            err_n += 1
            else:
                if not (allow_padding and token == -1):
                    err_n += 1
            output[b, i] = row_id
        tags[b] = t_b
        tag_generations[b] = g_b
        ages[b] = a_b
        clock[b] = clock_b
        counters[b, 0] = res_n
        counters[b, 1] = hit_n
        counters[b, 2] = miss_n
        errors[b] = err_n
    return {
        "output": output, "counters": counters, "errors": errors,
        "fills": fills, "tags": tags, "tag_generations": tag_generations,
        "ages": ages, "clock": clock,
    }
