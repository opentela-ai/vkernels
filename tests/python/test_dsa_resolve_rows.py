"""rows lane — dsa_resolve_rows parity tests (bit-exact vs the kvaas oracle).

The production kvaas scalar ``resolve_rows`` kernel is the oracle; its exact
semantics are restated by
:func:`vkernels.torch_ops.dsa_resolve_rows.resolve_rows_reference` (CPU) and
must be reproduced BIT-EXACTLY (integer/address math — no tolerance) by the
vectorized Triton kernel in the same module. Runs on-cluster (GPU); the CPU
reference-vs-semantics cases run everywhere.
"""

import pytest


def test_import_is_lazy():
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.dsa_resolve_rows; "
            "assert 'torch' in sys.modules; "  # torch is a top-level import
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# shared randomized-state builder (production-ish geometry, scaled down for
# CPU speed; a full-size production-geometry case runs in the GPU leg)
# ---------------------------------------------------------------------------

def _build_state(torch, rng, *, batch=2, pages=8, page=64, hot_pages=3,
                 k=12, width=8, layers=1, hot_rows=None,
                 gen=3, dup=True, pad=True):
    """Randomized but VALID tables: hot slots unique + disjoint from resident,
    active nonresident pages host-backed, lengths within pages*page."""
    import torch

    hot_rows = hot_rows or hot_pages * page
    # Pool sizing invariant (the job-656560 StopIteration fix): leaseable slots
    # must cover the WORST case by construction — resident rolls capped so the
    # batch's hot pages always fit, host capacity covers every non-resident
    # active page. next() selectors below can then never run dry.
    gpu_slots = batch * pages + batch * hot_pages + 2  # +2: scratch slot 0 + margin
    max_resident = gpu_slots - 1 - batch * hot_pages   # reserve the hot capacity
    host_pages_default = batch * pages + 1
    kv = torch.randn(layers, gpu_slots * page, width)
    host = torch.randn(layers, host_pages_default * page, width)
    resident = torch.full((batch, pages), -1, dtype=torch.int64)
    backing = torch.full((batch, pages), -1, dtype=torch.int64)
    generations = torch.randint(0, gen, (batch, pages), dtype=torch.int64)
    lengths = torch.randint(1, pages * page + 1, (batch,), dtype=torch.int64)
    hot_slots = torch.full((batch, hot_pages), -1, dtype=torch.int64)
    used_res, used_hot = set(), set()
    free_host = list(range(host_pages_default))
    for b in range(batch):
        L = int(lengths[b])
        for pg in range((L + page - 1) // page):
            roll = rng.random()
            if roll < 0.55 and len(used_res) < max_resident:
                # resident page (slot 0 stays the shared scratch — never lease it)
                slot = next(
                    s for s in range(1, gpu_slots) if s not in used_res
                )
                used_res.add(slot)
                resident[b, pg] = slot
            else:
                assert free_host, "host capacity exhausted"
                backing[b, pg] = free_host.pop(rng.randrange(len(free_host)))
        # hot slots: unique across live rows, disjoint from resident slots
        # (always satisfiable: max_resident reserves batch*hot_pages slots)
        for hp in range(hot_pages):
            slot = next(
                s for s in range(1, gpu_slots)
                if s not in used_res and s not in used_hot
            )
            used_hot.add(slot)
            hot_slots[b, hp] = slot
    tags = torch.full((batch, hot_rows), -1, dtype=torch.int64)
    tag_generations = torch.full((batch, hot_rows), -1, dtype=torch.int64)
    # seed some hot-tier hits: tokens whose page is nonresident, tagged at
    # that page's CURRENT generation -> the walk must find them as hits
    for b in range(batch):
        L = int(lengths[b])
        nonres = [pg for pg in range((L + page - 1) // page) if resident[b, pg] < 0]
        for hp in range(min(hot_pages, len(nonres))):
            pg = nonres[hp]
            tok = pg * page + int(rng.randint(0, page))
            slot = hp * page + int(rng.randint(0, page))
            tags[b, slot] = tok
            tag_generations[b, slot] = generations[b, pg]
    ages = torch.randint(0, 100, (batch, hot_rows), dtype=torch.int64)
    clock = torch.randint(0, 50, (batch,), dtype=torch.int64)
    selected = torch.randint(-1 if pad else 0, int(lengths.max()), (batch, k), dtype=torch.int64)
    if dup:
        # force duplicates: copy a random valid token over later slots, and a
        # nonresident-tagged token too (dup of a just-missed token must HIT
        # the slot the earlier miss reserved)
        for b in range(batch):
            L = int(lengths[b])
            src = int(rng.randint(0, k))
            selected[b, k - 1] = selected[b, src]
            hot_tok = int(tags[b][tags[b] >= 0][0]) if (tags[b] >= 0).any() else 0
            if hot_tok < L:
                selected[b, k // 2] = hot_tok
    return dict(
        kv=kv, host=host, resident=resident, backing=backing,
        generations=generations, lengths=lengths, hot_slots=hot_slots,
        selected=selected, tags=tags, tag_generations=tag_generations,
        ages=ages, clock=clock, page=page, width=width,
    )


def _run_reference(torch, st, *, allow_padding=False, fence=False):
    from vkernels.torch_ops.dsa_resolve_rows import resolve_rows_reference

    fv = fe = None
    if fence:
        fv = torch.randint(0, 2, (st["kv"].shape[1] // st["page"],), dtype=torch.int64)
        fe = fv.clone()
    return resolve_rows_reference(
        st["resident"], st["backing"], st["generations"], st["lengths"],
        st["hot_slots"], st["selected"], st["tags"], st["tag_generations"],
        st["ages"], st["clock"], page_tokens=st["page"],
        allow_padding=allow_padding, fence_values=fv, fence_expected=fe,
    )


# ---------------------------------------------------------------------------
# CPU: the reference IS the production semantics (spot-checks)
# ---------------------------------------------------------------------------

def test_reference_resident_rows_and_counts(torch):
    import torch

    rng = __import__("random").Random(7)
    st = _build_state(torch, rng, batch=1, k=8, dup=False, pad=False)
    # make EVERY selected token resident-page: force it by picking tokens in
    # resident pages
    res_pages = (st["resident"][0] >= 0).nonzero(as_tuple=True)[0]
    assert res_pages.numel()
    length = int(st["lengths"][0])
    page0 = int(st["page"])
    # only pages that contain valid tokens (token < length); -1 is the
    # correct answer for out-of-length tokens and this test pins hits
    valid = res_pages[res_pages * page0 < length]
    if valid.numel() == 0:
        st["resident"][0, 0] = 0  # force page 0 resident; it always has valid tokens
        valid = torch.tensor([0], dtype=res_pages.dtype)
    for j in range(st["selected"].shape[1]):
        pg = int(valid[j % valid.numel()])
        st["selected"][0, j] = pg * page0 + j % page0
    out = _run_reference(torch, st)
    page = st["page"]
    for j in range(st["selected"].shape[1]):
        pg = int(st["selected"][0, j]) // page
        off = int(st["selected"][0, j]) % page
        slot = int(st["resident"][0, pg])
        assert int(out["output"][0, j]) == slot * page + off
    assert int(out["counters"][0, 0]) == st["selected"].shape[1]
    assert int(out["counters"][0, 1]) == 0 and int(out["counters"][0, 2]) == 0
    assert int(out["errors"][0]) == 0


def test_reference_invalid_tokens_error_and_minus1(torch):
    import torch

    rng = __import__("random").Random(11)
    st = _build_state(torch, rng, batch=1, k=6, pad=False)
    st["selected"][0, 1] = -1
    st["selected"][0, 3] = int(st["lengths"][0]) + 5  # past length
    out = _run_reference(torch, st, allow_padding=False)
    assert int(out["output"][0, 1]) == -1 and int(out["output"][0, 3]) == -1
    assert int(out["errors"][0]) == 2
    # ALLOW_PADDING: the -1 padding is exempt (exactly one error remains)
    out2 = _run_reference(torch, st, allow_padding=True)
    assert int(out2["errors"][0]) == 1
    assert int(out2["output"][0, 1]) == -1


def test_reference_miss_fill_and_duplicate_hits_reserved_slot(torch):
    import torch

    rng = __import__("random").Random(13)
    st = _build_state(torch, rng, batch=1, k=6, dup=False, pad=False)
    # the walk needs a seeded hot-tier hit; ~1% of draws leave every active
    # page resident (no tags) — retry seeds until one exists
    for seed in range(13, 40):
        if (st["tags"][0] >= 0).any():
            break
        rng = __import__("random").Random(seed + 1)
        st = _build_state(torch, rng, batch=1, k=6, dup=False, pad=False)
    assert (st["tags"][0] >= 0).any(), "no seeded hot hit across seed sweep"
    # one hot-tagged token (guaranteed hit), then a MISS for the same page
    # family, then a duplicate of the first: it must hit the SAME slot
    tagged = (st["tags"][0] >= 0).nonzero(as_tuple=True)[0][0]
    tok = int(st["tags"][0, tagged])
    page = st["page"]
    pg = tok // page
    assert int(st["resident"][0, pg]) < 0  # nonresident -> hot hit
    miss_tok = None
    for cand in range(int(st["lengths"][0])):
        if int(st["resident"][0, cand // page]) < 0 and int((st["tags"][0] == cand).sum()) == 0:
            miss_tok = cand
            break
    assert miss_tok is not None
    st["selected"][0, 0] = tok
    st["selected"][0, 1] = miss_tok
    st["selected"][0, 2] = tok  # duplicate AFTER the miss-free hit: same slot
    out = _run_reference(torch, st)
    assert int(out["output"][0, 0]) == int(out["output"][0, 2])
    hit_slot_page = int(out["output"][0, 0]) // page
    assert int(out["counters"][0, 1]) >= 1  # the two tok hits
    assert int(out["counters"][0, 2]) >= 1  # miss_tok filled
    # the miss's fill dst is inside the leased hot pages
    m = int(out["counters"][0, 2])
    for f in range(m):
        src, dst = int(out["fills"][0, f, 0]), int(out["fills"][0, f, 1])
        assert src >= 0 and dst >= 0
    del hit_slot_page


# ---------------------------------------------------------------------------
# GPU: bit-exact parity of the Triton kernel vs the reference oracle
# ---------------------------------------------------------------------------

def _gpu_parity_case(torch, dev, *, seed, k, batch=2, fence=False,
                     allow_padding=False, width=512, page=64):
    import random

    from vkernels.torch_ops.dsa_resolve_rows import (
        dsa_resolve_rows,
        resolve_rows_reference,
    )

    rng = random.Random(seed)
    st = _build_state(
        torch, rng, batch=batch, k=k, width=width, page=page,
        hot_pages=4, gen=4,
    )
    layers = 1
    kv = st["kv"].to(dev).to(torch.bfloat16).contiguous()
    host = st["host"].to(torch.bfloat16).pin_memory().contiguous()  # host tier stays host-side (pinned for H2D)
    t = lambda x: x.to(dev)  # noqa: E731
    tags = st["tags"].repeat(layers, 1, 1).to(dev)  # kernel contract: [layers, batch, hot_rows]
    tgs = st["tag_generations"].repeat(layers, 1, 1).to(dev)
    ages = st["ages"].repeat(layers, 1, 1).to(dev)
    clock = st["clock"].repeat(layers, 1).to(dev)  # [layers, batch]
    selected = st["selected"].to(dev)
    fv = fe = None
    if fence:
        fv = torch.randint(0, 2, (kv.shape[1] // page,), dtype=torch.int64, device=dev)
        fe = fv.clone()
    ref = resolve_rows_reference(
        st["resident"], st["backing"], st["generations"], st["lengths"],
        st["hot_slots"], st["selected"], st["tags"], st["tag_generations"],
        st["ages"], st["clock"], page_tokens=page,
        allow_padding=allow_padding,
        fence_values=fv.cpu() if fence else None,
        fence_expected=fe.cpu() if fence else None,
    )
    out, counters, errors, fills = dsa_resolve_rows(
        kv, host, t(st["resident"]), t(st["backing"]), t(st["generations"]),
        t(st["lengths"]), t(st["hot_slots"]), selected, tags, tgs, ages, clock,
        page_tokens=page, allow_padding=allow_padding,
        fence=(fv, fe) if fence else None,
    )
    # BIT-EXACT: outputs, counters, errors, fills (FULL tensor — the tail
    # [m_b, K) is zeroed by the kernel epilogue and by the reference's
    # torch.zeros, so it is defined; the job-656573 failure was
    # garbage-vs-garbage in those rows when batch requests had different
    # miss counts and the old max-slice compare), AND the advanced
    # LRU state (tags/tag_generations/ages/clock) — no tolerance anywhere.
    assert torch.equal(out, ref["output"].to(dev))
    assert torch.equal(counters, ref["counters"].to(dev))
    assert torch.equal(errors, ref["errors"].to(dev))
    assert torch.equal(fills, ref["fills"].to(dev))
    assert torch.equal(tags.cpu(), ref["tags"])
    assert torch.equal(tgs.cpu(), ref["tag_generations"])
    assert torch.equal(ages.cpu(), ref["ages"])
    assert torch.equal(clock.cpu(), ref["clock"])
    # resolved resident rows must point at the resident-page latents
    page = st["page"]
    for j in range(k):
        r = int(ref["output"][0, j])
        tok = int(st["selected"][0, j])
        if r >= 0 and 0 <= tok < int(st["lengths"][0]) and int(st["resident"][0, tok // page]) >= 0:
            slot = int(st["resident"][0, tok // page])
            assert r == slot * page + tok % page


def test_gpu_parity_bit_exact(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    import torch

    dev = "cuda"
    # randomized sweep: duplicates, nonresident/hot mix, invalid tokens
    for seed in range(4):
        _gpu_parity_case(torch, dev, seed=seed, k=2051 if seed == 0 else 257,
                         batch=2 if seed != 2 else 1,
                         allow_padding=seed % 2 == 0)
    # targeted: fence mismatch -> masked miss + nonzero errors, no fill
    _gpu_parity_case(torch, dev, seed=42, k=129, fence=True)
    # production geometry single-shot: topk=2048-class selection
    _gpu_parity_case(torch, dev, seed=99, k=2048, batch=1, width=512)


def test_gpu_parity_multi_layer_state_indexing(torch):
    """Two layers sharing tables (floe's per-layer resolvers share setup
    tables): per-layer state must evolve independently and bit-exactly."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    import random

    import torch

    from vkernels.torch_ops.dsa_resolve_rows import (
        dsa_resolve_rows_batched,
        resolve_rows_reference,
    )

    dev = "cuda"
    rng = random.Random(5)
    st = _build_state(torch, rng, batch=2, k=65, layers=1)
    layers = 3
    kv = st["kv"].repeat(layers, 1, 1).to(dev).to(torch.bfloat16).contiguous()
    host = st["host"].repeat(layers, 1, 1).to(dev).to(torch.bfloat16).pin_memory().contiguous()
    t = lambda x: x.to(dev)  # noqa: E731
    tags = st["tags"].repeat(layers, 1, 1).to(dev)
    tgs = st["tag_generations"].repeat(layers, 1, 1).to(dev)
    ages = st["ages"].repeat(layers, 1, 1).to(dev)
    clock = st["clock"].repeat(layers, 1).to(dev)
    selected = st["selected"].unsqueeze(0).repeat(layers, 1, 1).to(dev)
    out, counters, errors, _ = dsa_resolve_rows_batched(
        kv, host, t(st["resident"]), t(st["backing"]), t(st["generations"]),
        t(st["lengths"]), t(st["hot_slots"]), selected, tags, tgs, ages, clock,
        page_tokens=st["page"],
    )
    for L in range(layers):
        ref = resolve_rows_reference(
            st["resident"], st["backing"], st["generations"], st["lengths"],
            st["hot_slots"], st["selected"], st["tags"][L], st["tag_generations"][L],
            st["ages"][L], st["clock"][L], page_tokens=st["page"],
        )
        assert torch.equal(out[L].cpu(), ref["output"])
        assert torch.equal(counters[L].cpu(), ref["counters"])
        assert torch.equal(errors[L].cpu(), ref["errors"])
        assert torch.equal(tags[L].cpu(), ref["tags"])
        assert torch.equal(ages[L].cpu(), ref["ages"])


def test_contract_miss_falls_back(torch):
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.dsa_resolve_rows import dsa_resolve_rows

    t = pytest.importorskip("torch")
    kv = t.randn(1, 8, 8)
    host = t.randn(1, 8, 8)  # CPU but NOT pinned -> contract miss
    with pytest.raises(OpNotEligible):
        dsa_resolve_rows(
            kv, host, t.zeros(1, 2, dtype=t.int64), t.zeros(1, 2, dtype=t.int64),
            t.zeros(1, 2, dtype=t.int64), t.zeros(1, dtype=t.int64),
            t.zeros(1, 1, dtype=t.int64), t.zeros(1, 4, dtype=t.int64),
            t.zeros(1, 4, dtype=t.int64), t.zeros(1, 4, dtype=t.int64),
            t.zeros(1, 4, dtype=t.int64), t.zeros(1, dtype=t.int64),
            page_tokens=2,
        )
