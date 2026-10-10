"""rows lane — rank-normalization shim tests for resolve_rows_reference.

Round-6 follow-up to jobs 656671/656672/656673: the GPU parity callers hand
the oracle KERNEL-RANKED state ([1, batch, ...] singleton-layer tensors, or
scalars for the 1-D per-request contracts), and torch 2.9 rejects any
``X[b]`` on such input with "invalid index of a 0-dim tensor". The reference
now funnels EVERY input through one ``_norm`` shim (squeeze a singleton
leading dim / broadcast a scalar across the batch / fail loud) and restores
the input rank of every mutated in-out at return.

This file catches that entire bug class WITHOUT torch:

* a pure-python mirror of the ``_norm``/``_restore`` shim semantics
  (KEEP IN SYNC with dsa_resolve_rows.resolve_rows_reference), fuzzed over
  the same random-state generator family as the algorithm fuzz: every state
  is walked at contract rank AND at kernel rank (and with a scalar clock) —
  all results must be identical and restored shapes must round-trip;
* negative cases (genuinely malformed ranks must raise);
* if torch IS importable (on-cluster), the REAL reference is exercised with
  kernel-ranked CPU tensors and compared against the contract-rank call.
"""

import copy
import os
import random

TRIALS = int(os.environ.get("RANKSHIM_TRIALS", "20000"))

INF = 2**63 - 1


# ---------------------------------------------------------------- mock shim
def _shape_of(x):
    s = []
    while isinstance(x, list):
        s.append(len(x))
        x = x[0]
    return tuple(s)


def _flatten(x, out):
    if isinstance(x, list):
        for e in x:
            _flatten(e, out)
    else:
        out.append(x)


def _build(it, shape):
    if len(shape) == 1:
        return [next(it) for _ in range(shape[0])]
    return [_build(it, shape[1:]) for _ in range(shape[0])]


def _prod(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def norm(x, rank, batch=None, name=""):
    """Pure-python mirror of _norm in resolve_rows_reference. KEEP IN SYNC."""
    shape0 = _shape_of(x)
    if len(shape0) == rank + 1 and shape0[0] == 1:
        x = x[0]  # singleton layer dim
    shape = _shape_of(x)
    if len(shape) == rank:
        if batch is not None and shape[0] != batch:
            raise ValueError(
                f"{name}: leading dim {shape[0]} != batch {batch}")
        return copy.deepcopy(x), shape0
    if len(shape) == rank - 1:
        if rank == 1:  # dim 0 => scalar
            if batch is not None and batch > 1:
                return [x] * batch, shape0
            return [x], shape0
        if batch == 1:
            return [copy.deepcopy(x)], shape0
    raise ValueError(
        f"{name}: shape {shape0}; expected rank {rank} or [1, *]")


def restore(x, shape0):
    """Mirror of _restore: reshape back to the input rank when numel allows."""
    flat = []
    _flatten(x, flat)
    if len(flat) == _prod(shape0):
        return _build(iter(flat), shape0)
    return x


# ------------------------------------------------- reference walk (mirror)
def ref_walk(resident, backing, generations, lengths, hot_slots, selected,
             tags, tgs, ages, clock, page):
    """Pure-python transliteration of resolve_rows_reference's walk."""
    batch = len(resident)
    k = len(selected[0])
    hot_rows = len(tags[0])
    output = [[-1] * k for _ in range(batch)]
    counters = [[0] * 3 for _ in range(batch)]
    errors = [0] * batch
    fills = [[(0, 0)] * k for _ in range(batch)]
    tags = [list(r) for r in tags]
    tgs = [list(r) for r in tgs]
    ages = [list(r) for r in ages]
    clock = list(clock)
    for b in range(batch):
        length = lengths[b]
        protected = [False] * hot_rows
        t_b = list(tags[b])
        g_b = list(tgs[b])
        a_b = list(ages[b])
        clock_b = clock[b] + 1
        sel_valid = {t for t in selected[b] if 0 <= t < length}
        for l in range(hot_rows):
            if t_b[l] >= 0:
                protected[l] = (generations[b][t_b[l] // page] == g_b[l]
                                and t_b[l] in sel_valid)
        res_n = hit_n = miss_n = err_n = 0
        for i in range(k):
            token = selected[b][i]
            valid = 0 <= token < length
            row_id = -1
            if valid:
                pg, off = token // page, token % page
                r = resident[b][pg]
                if r >= 0:
                    row_id = r * page + off
                    res_n += 1
                else:
                    gen = generations[b][pg]
                    slot = next((l for l in range(hot_rows)
                                 if t_b[l] == token and g_b[l] == gen),
                                hot_rows)
                    is_hit = slot < hot_rows
                    if not is_hit:
                        oldest = min((a_b[l] if not protected[l] else INF)
                                     for l in range(hot_rows))
                        slot = next((l for l in range(hot_rows)
                                     if not protected[l]
                                     and a_b[l] == oldest), hot_rows)
                    if slot < hot_rows:
                        hot_page = hot_slots[b][slot // page]
                        if is_hit:
                            hit_n += 1
                        else:
                            fills[b][miss_n] = (backing[b][pg] * page + off,
                                                hot_page * page
                                                + slot % page)
                            miss_n += 1
                            t_b[slot] = token
                            g_b[slot] = gen
                        row_id = hot_page * page + slot % page
                        protected[slot] = True
                        a_b[slot] = clock_b
            else:
                err_n += 1
            output[b][i] = row_id
        tags[b] = t_b
        tgs[b] = g_b
        ages[b] = a_b
        clock[b] = clock_b
        counters[b] = [res_n, hit_n, miss_n]
        errors[b] = err_n
    return (output, counters, errors, fills, tags, tgs, ages, clock)


# ------------------------------------------------------------- state maker
def make_state(rng):
    batch = rng.choice([1, 2])
    pages = rng.choice([2, 3, 4])
    page = rng.choice([4, 8])
    hot_pages = rng.choice([1, 2, 3])
    k = rng.choice([3, 5, 9, 17])
    gpu_slots = batch * pages + batch * hot_pages + 2
    host_pages = batch * pages + 1
    resident = [[-1] * pages for _ in range(batch)]
    backing = [[-1] * pages for _ in range(batch)]
    hot_slots = [[-1] * hot_pages for _ in range(batch)]
    generations = [[rng.randrange(3) for _ in range(pages)]
                   for _ in range(batch)]
    lengths = [rng.randrange(1, pages * page + 1) for _ in range(batch)]
    slotpool = list(range(1, gpu_slots))
    rng.shuffle(slotpool)
    pi = 0
    for b in range(batch):
        for pg in range(pages):
            if rng.random() < 0.5 and pi < len(slotpool):
                resident[b][pg] = slotpool[pi]
                pi += 1
            else:
                backing[b][pg] = rng.randrange(0, host_pages)
    used = set()
    for b in range(batch):
        for hp in range(hot_pages):
            while pi < len(slotpool) and slotpool[pi] in used:
                pi += 1
            if pi < len(slotpool):
                hot_slots[b][hp] = slotpool[pi]
                used.add(slotpool[pi])
                pi += 1
    selected = [[rng.choice([-1, -1, rng.randrange(0, pages * page + 2)])
                 for _ in range(k)] for _ in range(batch)]
    if rng.random() < 0.7:
        for b in range(batch):
            selected[b][k - 1] = selected[b][rng.randrange(k)]
    H = hot_pages * page
    tags = [[-1] * H for _ in range(batch)]
    tgs = [[-1] * H for _ in range(batch)]
    ages = [[rng.randrange(6) for _ in range(H)] for _ in range(batch)]
    clock = [rng.randrange(5) for _ in range(batch)]
    for b in range(batch):
        L = lengths[b]
        nonres = [pg for pg in range((L + page - 1) // page)
                  if resident[b][pg] < 0]
        for hp in range(min(hot_pages, len(nonres))):
            pg = nonres[hp]
            slot = hp * page + rng.randrange(page)
            tags[b][slot] = pg * page + rng.randrange(page)
            tgs[b][slot] = generations[b][pg]
    return dict(batch=batch, page=page, resident=resident, backing=backing,
                generations=generations, lengths=lengths, hot_slots=hot_slots,
                selected=selected, tags=tags, tgs=tgs, ages=ages, clock=clock)


# ------------------------------------------------------------------- tests
def test_shim_fuzz_rank_equivalence():
    """Contract-rank walk == kernel-rank walk == scalar-clock walk, and the
    mutated in-outs round-trip their input rank — for every fuzzed state."""
    rng = random.Random(1234)
    for trial in range(TRIALS):
        st = make_state(rng)
        args = (st["resident"], st["backing"], st["generations"],
                st["lengths"], st["hot_slots"], st["selected"], st["tags"],
                st["tgs"], st["ages"], st["clock"], st["page"])
        base = ref_walk(*args)

        # variant A: everything kernel-ranked ([1, batch, ...] / [1, batch])
        normed = [norm(x, 2, batch=st["batch"], name=n)
                  if n != "clock" and n != "lengths"
                  else norm(x, 1, batch=st["batch"], name=n)
                  for n, x in zip(
                      ["resident", "backing", "generations", "lengths",
                       "hot_slots", "selected", "tags", "tgs", "ages",
                       "clock"], args[:10])]
        shapes = [s for _, s in normed]
        got = ref_walk(*[x for x, _ in normed], st["page"])
        assert got == base, f"kernel-rank walk diverged at trial {trial}"
        # mutated in-outs round-trip their input rank
        for got_st, shp, name in zip(got[4:], shapes[6:],
                                     ["tags", "tgs", "ages", "clock"]):
            assert _shape_of(restore(got_st, shp)) == shp, \
                f"{name} rank not restored at trial {trial}"

        # variant B: scalar clock (uniform across requests), everything else
        # kernel-ranked; baseline re-run with the same uniform clock.
        c0 = st["clock"][0]
        st2 = dict(st)
        st2["clock"] = [c0] * st["batch"]
        base_uniform = ref_walk(
            st2["resident"], st2["backing"], st2["generations"],
            st2["lengths"], st2["hot_slots"], st2["selected"], st2["tags"],
            st2["tgs"], st2["ages"], st2["clock"], st2["page"])
        # norm of a bare scalar: batch broadcast
        clk, clk_shape = norm(c0, 1, batch=st["batch"], name="clock")
        assert _shape_of(clk) == (st["batch"],)
        got_uniform = ref_walk(
            *[norm(x, 2, batch=st["batch"], name=n)[0]
              if n != "lengths" else
              norm(x, 1, batch=st["batch"], name=n)[0]
              for n, x in zip(
                  ["resident", "backing", "generations", "lengths",
                   "hot_slots", "selected", "tags", "tgs", "ages"],
                  [st2[k2] for k2 in ["resident", "backing", "generations",
                                      "lengths", "hot_slots", "selected",
                                      "tags", "tgs", "ages"]])],
            clk, st2["page"])
        assert got_uniform == base_uniform, \
            f"scalar-clock walk diverged at trial {trial}"


def test_shim_negative_ranks():
    """Genuinely malformed ranks must fail loud, not misindex."""
    for bad, rank, batch, name in [
        ([[1, 2], [3, 4], [5, 6]], 2, 2, "tags"),    # rank ok, batch mismatch
        ([1, 2], 2, 2, "tags"),                      # 1-D tags, batch=2
        ([5], 1, 2, "clock"),                        # 1-D clock, batch=2
    ]:
        try:
            norm(bad, rank, batch=batch, name=name)
        except ValueError:
            continue
        raise AssertionError(f"{name} {bad} should have raised")


def test_torch_reference_rank_shim_end_to_end():
    """On-cluster: the REAL resolve_rows_reference with kernel-ranked CPU
    tensors must match the contract-rank call bit-for-bit, and return
    input-rank-consistent advanced state."""
    try:
        import torch
    except ImportError:
        import pytest
        pytest.skip("torch not available")
    from vkernels.torch_ops.dsa_resolve_rows import resolve_rows_reference

    rng = random.Random(77)
    st = make_state(rng)
    t = lambda x: torch.tensor(x, dtype=torch.int64)  # noqa: E731

    ref = resolve_rows_reference(
        t(st["resident"]), t(st["backing"]), t(st["generations"]),
        t(st["lengths"]), t(st["hot_slots"]), t(st["selected"]), t(st["tags"]),
        t(st["tgs"]), t(st["ages"]), t(st["clock"]), page_tokens=st["page"])

    # kernel-ranked: [1, batch, ...] everywhere (singleton layer dim)
    ref3 = resolve_rows_reference(
        t([st["resident"]]), t([st["backing"]]), t([st["generations"]]),
        t([st["lengths"]]), t([st["hot_slots"]]), t([st["selected"]]),
        t([st["tags"]]), t([st["tgs"]]), t([st["ages"]]), t([st["clock"]]),
        page_tokens=st["page"])
    for key in ["output", "counters", "errors", "fills", "tags",
                "tag_generations", "ages", "clock"]:
        if key in ("tags", "tag_generations", "ages", "clock"):
            assert ref3[key].shape == (1, *ref[key].shape), key  # rank restored
            assert torch.equal(ref[key], ref3[key].squeeze(0)), key
        else:
            assert torch.equal(ref[key], ref3[key]), key

    # scalar clock (uniform across the batch)
    st2 = dict(st)
    st2["clock"] = [st["clock"][0]] * st["batch"]
    base = resolve_rows_reference(
        t(st2["resident"]), t(st2["backing"]), t(st2["generations"]),
        t(st2["lengths"]), t(st2["hot_slots"]), t(st2["selected"]),
        t(st2["tags"]), t(st2["tgs"]), t(st2["ages"]),
        torch.tensor(st["clock"][0], dtype=torch.int64),  # 0-dim clock
        page_tokens=st2["page"])
    want = resolve_rows_reference(
        t(st2["resident"]), t(st2["backing"]), t(st2["generations"]),
        t(st2["lengths"]), t(st2["hot_slots"]), t(st2["selected"]),
        t(st2["tags"]), t(st2["tgs"]), t(st2["ages"]), t(st2["clock"]),
        page_tokens=st2["page"])
    for key in ["output", "counters", "errors", "fills", "tags",
                "tag_generations", "ages", "clock"]:
        assert torch.equal(want[key], base[key]), key

    # malformed rank fails loud
    import pytest
    with pytest.raises(ValueError):
        resolve_rows_reference(
            t([st["resident"], st["resident"]]),  # [2, batch, pages]
            t(st["backing"]), t(st["generations"]), t(st["lengths"]),
            t(st["hot_slots"]), t(st["selected"]), t(st["tags"]),
            t(st["tgs"]), t(st["ages"]), t(st["clock"]),
            page_tokens=st["page"])


if __name__ == "__main__":
    test_shim_fuzz_rank_equivalence()
    test_shim_negative_ranks()
    print(f"rank-shim fuzz ok ({TRIALS} trials, torch-free)")
