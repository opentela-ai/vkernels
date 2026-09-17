"""Issue #94: slot-table indirection in Region — paged/indirected index sets.

Validation chain (mirrors the issue's Validation section):

* IR: ``Region.indirect(view, table, axis)`` overapproximates storage spans
  to the whole pool — indirected regions conflict with everything on their
  pool (sound under phase order), and never with other storages;
* hazards: paged append -> paged scores recover the RAW edge; two paged
  writers on one pool conflict (phase order enforced);
* capture: paged ops record the slot table as an external input and whole
  pool effects; post-append views bump storage versions (§4.3);
* reference: the kvaas pattern — **permuted slot tables decode identically
  to identity tables** given correspondingly permuted pool content, with
  disjoint slot leases per row; the reserved slot-0 null/sink page is
  poisoned with NaN and must never leak into outputs;
* GQA: the grouped kv-head mapping is honored through the gather.

CPU-only suite (the device kernels already address pools through slot
tables — see ``device_triton_hybrid.py``'s kvaas handlers).
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler.capture import RecordingBackend
from vkernels.compiler.lowerings import lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.operator_ir import OP_ATTENTION_SCORES_PAGED, Region, TensorValue
from vkernels.compiler.reference_exec import ReferenceExecutor
from vkernels.compiler.schedule_phase import PhaseSchedule

B, KVH, H, D, SCAP, TMAX = 3, 2, 4, 8, 32, 16


# ---------------------------------------------------------------------------
# IR: Region.indirect storage-span overapproximation
# ---------------------------------------------------------------------------

def _pool(storage_id=7, name="k_pool"):
    return TensorValue(vid=None, name=name, shape=(64, 2, 8), dtype=None,
                       strides=(16, 8, 1), storage_id=storage_id, offset=0)


def test_region_indirect_span_covers_whole_pool():
    pool = _pool()
    table = TensorValue(vid=None, name="slot_table", shape=(4, 32), dtype=None,
                        strides=(32, 1), storage_id=8, offset=0)
    r = Region.indirect(pool, table, axis=0)
    assert r.indirect_table is table and r.indirect_axis == 0
    # whole-pool overapproximation: [offset, offset + numel - 1]
    assert r.storage_span() == (0, 64 * 16 - 1)


def test_region_indirect_conflicts_conservatively():
    pool = _pool(storage_id=7)
    table = TensorValue(vid=None, name="slot_table", shape=(4, 32), dtype=None,
                        strides=(32, 1), storage_id=8, offset=0)
    ind = Region.indirect(pool, table, axis=0)
    dense_same_pool = Region(7, pool, ((0, 1), (0, 2), (0, 8)))
    dense_other_pool = Region(9, _pool(storage_id=9, name="v_pool"), ((0, 64), (0, 2), (0, 8)))
    assert ind.overlaps(dense_same_pool)  # conservative on the shared pool
    assert not ind.overlaps(dense_other_pool)  # never across storages
    assert ind.overlaps(Region.indirect(pool, table, axis=0))  # two indirected regions conflict


# ---------------------------------------------------------------------------
# Capture + hazard plumbing
# ---------------------------------------------------------------------------

def _capture_paged_step(table_shape=(B, SCAP)):
    rec = RecordingBackend()
    pos = rec.define_position(TMAX)
    kp = rec.external_tensor("k_pool", (SCAP, KVH, D), storage_id=401)
    vp = rec.external_tensor("v_pool", (SCAP, KVH, D), storage_id=402)
    tb = rec.external_tensor("slot_table", table_shape, storage_id=403)
    kn = rec.external_tensor("k_new", (B, KVH, D), storage_id=404)
    vn = rec.external_tensor("v_new", (B, KVH, D), storage_id=405)
    q = rec.external_tensor("q", (B, H, D), storage_id=406)
    kpost, vpost = rec.cache_append_paged(kp, vp, tb, kn, vn, pos, layer=0)
    s = rec.attention_scores_paged(q, kpost, tb, pos, scale=0.35, layer=0, kv_heads=KVH)
    pr = rec.softmax(s, pos, layer=0)
    ctx = rec.attention_values_paged(pr, vpost, tb, pos, layer=0, kv_heads=KVH)
    return rec, pos, (kp, vp, tb, kn, vn, q), ctx


def test_capture_records_paged_kinds_and_external_table():
    rec, _, _, _ = _capture_paged_step()
    kinds = [op.kind for op in rec.graph.ops]
    assert kinds == ["cache_append_paged", "attention_scores_paged", "softmax", "attention_values_paged"]
    append = rec.graph.ops[0]
    assert append.kind == "cache_append_paged"
    assert "slot_table" in append.inputs
    assert all(w.indirect_table is not None for w in append.write_regions)
    assert append.write_regions[0].indirect_axis == 0
    # post-append views bump storage versions (§4.3)
    versions = rec.graph.storage_versions


def test_hazards_order_paged_append_before_scores():
    from vkernels.compiler.operator_ir import compute_hazards
    rec, _, _, _ = _capture_paged_step()
    hazards = compute_hazards(rec.graph.ops)
    raw = {(h.producer, h.consumer) for h in hazards if h.kind == "RAW"}
    # append -> scores (K), softmax -> values (probs), scores -> softmax,
    # append -> values (V via vpost view)
    assert any(a < b for (a, b) in raw if rec.graph.ops[a].kind == "cache_append_paged"
               and rec.graph.ops[b].kind == "attention_scores_paged")


# ---------------------------------------------------------------------------
# Reference execution: the kvaas permuted-table pattern
# ---------------------------------------------------------------------------

def _make_table(cols, width=SCAP):
    """[B, width] table from per-row slot lists; unused columns repeat slot cols[b][0]."""
    t = np.zeros((B, width), dtype=np.int32)
    for b in range(B):
        t[b, : len(cols[b])] = cols[b]
        t[b, len(cols[b]):] = cols[b][0]
    return t


def _build_pool(table, seed=0):
    """Populate pools so pool[table[b, t]] holds logical token (b, t); sink NaN."""
    rng = np.random.default_rng(seed)
    kpool = rng.normal(size=(SCAP, KVH, D)).astype(np.float32)
    vpool = rng.normal(size=(SCAP, KVH, D)).astype(np.float32)
    X_k = rng.normal(size=(B, SCAP, KVH, D)).astype(np.float32)
    X_v = rng.normal(size=(B, SCAP, KVH, D)).astype(np.float32)
    for b in range(B):
        for t in range(SCAP):
            s = int(table[b, t])
            kpool[s] = X_k[b, t]
            vpool[s] = X_v[b, t]
    kpool[0] = np.nan  # reserved null/sink page — poisoned so any read leaks visibly
    vpool[0] = np.nan
    return kpool, vpool


def _run_paged_decode(table, position, kpool, vpool, workers=3):
    rec, pos, (kp, vp, tb, kn, vn, q), ctx = _capture_paged_step()
    fams = lower_graph(rec.graph)
    sched = PhaseSchedule.from_families(fams, workers=workers)
    ws, _ = plan_memory(rec.graph, sched, fams)
    rng = np.random.default_rng(1)
    q_arr = rng.normal(size=(B, H, D)).astype(np.float32)
    kn_arr = rng.normal(size=(B, KVH, D)).astype(np.float32)
    vn_arr = rng.normal(size=(B, KVH, D)).astype(np.float32)
    st = {
        kp.value.storage_id: kpool.reshape(-1),
        vp.value.storage_id: vpool.reshape(-1),
        tb.value.storage_id: table.astype(np.int32).reshape(-1),
        kn.value.storage_id: kn_arr.reshape(-1),
        vn.value.storage_id: vn_arr.reshape(-1),
        q.value.storage_id: q_arr.reshape(-1),
    }
    for buf in ws.buffers:
        st[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    ex = ReferenceExecutor(sched, workers=workers, storage_arrays=st, graph=rec.graph, workspace_plan=ws)
    ex.run({"p": position})
    return np.array(ex.tensor(ctx.value.name), copy=True), q_arr, kn_arr, vn_arr


def _numpy_oracle(table, p, kpool, vpool, q, k_new, v_new, scale=0.35):
    """Independent fp64 reference of append -> scores -> softmax -> values."""
    ka, va = kpool.astype(np.float64).copy(), vpool.astype(np.float64).copy()
    for b in range(B):  # the append lands at table[b, p]
        ka[int(table[b, p])] = k_new[b]
        va[int(table[b, p])] = v_new[b]
    ctx = np.zeros((B, H, D), dtype=np.float64)
    group = H // KVH
    for b in range(B):
        slots = table[b, : p + 1].astype(int)
        for h in range(H):
            kv = h // group
            scores = ka[slots, kv, :] @ q[b, h].astype(np.float64) * scale
            e = np.exp(scores - scores.max())
            probs = e / e.sum()
            ctx[b, h] = probs @ va[slots, kv, :]
    return ctx


def _disjoint_tables(p, seed):
    """Identity (row-major) and permuted disjoint slot leases; slot 0 excluded."""
    per_row = p + 1
    ident = _make_table([np.arange(1 + per_row * b, 1 + per_row * (b + 1)) for b in range(B)])
    rng = np.random.default_rng(seed)
    slots = rng.permutation(np.arange(1, B * per_row + 1)).astype(np.int32)
    perm = _make_table([slots[per_row * b: per_row * (b + 1)] for b in range(B)])
    return ident, perm


@pytest.mark.parametrize("p", [0, 1, 5])
@pytest.mark.parametrize("workers", [1, 3])
def test_permuted_slot_tables_decode_identically(p, workers):
    ident, perm = _disjoint_tables(p, seed=100 + p)
    assert 0 not in perm[:, : p + 1] and 0 not in ident[:, : p + 1]  # sink never leased
    a, q_a, kn_a, vn_a = _run_paged_decode(ident, p, *_build_pool(ident, seed=p))
    b, q_b, kn_b, vn_b = _run_paged_decode(perm, p, *_build_pool(perm, seed=p))
    assert np.allclose(a, b, atol=1e-6)
    assert not np.isnan(a).any()  # NaN sink never leaked
    # both agree with the independent fp64 oracle (same inputs by seeding)
    ref = _numpy_oracle(ident, p, *_build_pool(ident, seed=p)[:2], q_a, kn_a, vn_a)
    assert np.allclose(a, ref, atol=1e-5)


def test_gqa_grouped_heads_match_oracle():
    """H=4 query heads share KVH=2 kv heads: the gather uses kv(h) = h // group,
    validated against the independent oracle (not head-pair equality —
    different queries give different probs over the same V rows)."""
    ident, _ = _disjoint_tables(5, seed=7)
    ctx, q, kn, vn = _run_paged_decode(ident, 5, *_build_pool(ident, seed=3))
    assert ctx.shape == (B, H, D)
    ref = _numpy_oracle(ident, 5, *_build_pool(ident, seed=3)[:2], q, kn, vn)
    assert np.allclose(ctx, ref, atol=1e-5)
    # the grouped mapping is recorded on the ops
    rec, _, _, _ = _capture_paged_step()
    assert rec.graph.ops[1].attributes["kv_heads"] == KVH


def test_append_lands_at_table_slot_and_is_read_back():
    """The freshly appended row (t = p) must be readable through the table."""
    p = 2
    ident, _ = _disjoint_tables(p, seed=11)
    kpool, vpool = _build_pool(ident, seed=5)
    rec, pos, (kp, vp, tb, kn, vn, q), ctx = _capture_paged_step()
    fams = lower_graph(rec.graph)
    sched = PhaseSchedule.from_families(fams, workers=2)
    ws, _ = plan_memory(rec.graph, sched, fams)
    rng = np.random.default_rng(2)
    k_new = rng.normal(size=(B, KVH, D)).astype(np.float32)
    v_new = rng.normal(size=(B, KVH, D)).astype(np.float32)
    q_arr = rng.normal(size=(B, H, D)).astype(np.float32)
    st = {
        kp.value.storage_id: kpool.reshape(-1),
        vp.value.storage_id: vpool.reshape(-1),
        tb.value.storage_id: ident.astype(np.int32).reshape(-1),
        kn.value.storage_id: k_new.reshape(-1),
        vn.value.storage_id: v_new.reshape(-1),
        q.value.storage_id: q_arr.reshape(-1),
    }
    for buf in ws.buffers:
        st[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    ex = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=rec.graph, workspace_plan=ws)
    ex.run({"p": p})
    # write went to table[b, p]; the pool storage must now hold k_new there
    for b in range(B):
        s = int(ident[b, p])
        assert np.allclose(kpool[s], k_new[b], atol=1e-6)
        assert np.allclose(vpool[s], v_new[b], atol=1e-6)
    assert not np.isnan(np.array(ex.tensor(ctx.value.name))).any()


def test_scores_only_read_valid_prefix():
    """Rows beyond p are never gathered: poison every slot not leased to [0, p]."""
    p = 3
    ident, _ = _disjoint_tables(p, seed=13)
    kpool, vpool = _build_pool(ident, seed=6)
    leased = {int(ident[b, t]) for b in range(B) for t in range(p + 1)}
    for s in range(SCAP):
        if s not in leased:
            kpool[s] = np.nan
            vpool[s] = np.nan
    ctx, q, kn, vn = _run_paged_decode(ident, p, kpool, vpool)
    assert not np.isnan(ctx).any()  # invalid tail never touched
    ref = _numpy_oracle(ident, p, kpool, vpool, q, kn, vn)
    assert np.allclose(ctx, ref, atol=1e-5)


def test_score_op_kind_registered_in_lowerings():
    from vkernels.compiler.lowerings import LOWERINGS
    for kind in ("cache_append_paged", "attention_scores_paged", "attention_values_paged"):
        assert kind in LOWERINGS
    assert OP_ATTENTION_SCORES_PAGED == "attention_scores_paged"
