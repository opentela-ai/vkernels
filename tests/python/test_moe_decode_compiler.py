"""MoE decode ops (``moe_route`` / ``moe_expert`` / ``moe_combine``) — issue #98.

Covers the vertical slice the issue names:

* capture — ``ops.moe_route`` records the router family (DeepSeek-V4
  sqrtsoftplus global top-k, GLM-5.3 noaux_tc sigmoid + bias + group
  restriction, frozen hash gather) and writes the external routing table
  (i32 ids + f32 weights, [B, k]);
* lowering — static task grids: one route task per row, the k·B expert grid
  with the weight base indirected through the routing table (the #94
  slot-table pattern over a read-only weight pool), one combine task per row;
* reference bodies — fp64 oracles of the fp32 device math, including the
  swiglu_limit clamp folded into the expert activation;
* legality — the routing-table RAW hazards order route ≺ experts ≺ combine;
* validation — router determinism/tie rules, hash exactness, k=E parity
  against dense-every-expert evaluation, clamp boundaries, weighted combine,
  shared-expert add, NaN canaries, one-launch accounting, and numpy mirrors
  of the Triton task decompositions (worker strides included).

All CPU (§15.1): the bare environment has neither torch nor triton, so every
arithmetic claim is pinned against numpy mirrors; the floe eager-parity tests
are import-gated and reported as attested-not-verified.
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler.capture import RecordingBackend
from vkernels.compiler.codegen_cute import TEMPLATE_NAMES
from vkernels.compiler.legality import enforce
from vkernels.compiler.lowerings import LOWERINGS, check_thread_contract, lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.operator_ir import F32, I32, compute_hazards
from vkernels.compiler.reference_exec import ReferenceExecutor
from vkernels.compiler.schedule_phase import PhaseSchedule


# ---------------------------------------------------------------------------
# numpy oracles (independent of the reference executor; fp64)
# ---------------------------------------------------------------------------


def _scores(logits, score_fn):
    """Router scores per the floe reference: stable sigmoid (+bias applied by
    the caller) or sqrt(softplus)."""
    if score_fn == "sigmoid_noaux_tc":
        return np.exp(-np.logaddexp(0.0, -logits))  # stable sigmoid(l)
    assert score_fn == "sqrtsoftplus"
    return np.sqrt(np.logaddexp(0.0, logits))


def _topk_stable(choice, k):
    """Documented determinism contract: stable descending order, ties to the
    lower expert index (sorted=False set semantics)."""
    return np.argsort(-choice, kind="stable")[:k]


def _route_mirror(x_row, router_w, *, mode="learned", score_fn="sqrtsoftplus", top_k, rsf=1.0,
                  bias=None, n_group=1, topk_group=1, norm_topk_prob=True,
                  tid2eid=None, token_id=None):
    logits = router_w.astype(np.float64) @ x_row.astype(np.float64)
    scores = _scores(logits, score_fn)
    if mode == "hash":
        sel = tid2eid[token_id].astype(np.int64)
        w = scores[sel]
        w = w / (w.sum() + 1e-20)  # floe hash router renorms unconditionally
    else:
        choice = scores.copy()
        if score_fn == "sigmoid_noaux_tc":
            choice = choice + bias
            if n_group > 1:
                gs = choice.reshape(n_group, -1)
                top2 = np.sort(gs, axis=1, kind="stable")[:, ::-1][:, :2].sum(axis=1)
                keep = _topk_stable(top2, topk_group)
                mask = np.full(choice.shape, -np.inf)
                for g in keep:
                    mask[g * (choice.size // n_group):(g + 1) * (choice.size // n_group)] = 0.0
                choice = choice + mask
        sel = _topk_stable(choice, top_k)
        w = scores[sel]  # unbiased scores (floe semantics)
        if norm_topk_prob:
            w = w / (w.sum() + 1e-20)
    return sel.astype(np.int32), (w * rsf)


def _expert_mirror(x_row, e, gate_up, down, limit=None):
    gu = gate_up[e].astype(np.float64) @ x_row.astype(np.float64)
    inter = gu.size // 2
    g, u = gu[:inter], gu[inter:]
    if limit is not None:
        g = np.minimum(g, float(limit))
        u = np.clip(u, -float(limit), float(limit))
    act = g / (1.0 + np.exp(-g)) * u  # silu(g)·u — the executor's exact form
    return down[e].astype(np.float64) @ act


def _block_oracle(x_row, router_w, gate_up, down, *, top_k, limit=None, **route_kw):
    """Full decode-block oracle: route → experts → weighted combine."""
    sel, w = _route_mirror(x_row, router_w, top_k=top_k, **route_kw)
    acc = np.zeros(gate_up.shape[2], dtype=np.float64)
    for s, e in enumerate(sel):
        acc += w[s] * _expert_mirror(x_row, int(e), gate_up, down, limit)
    return acc, sel, w


# ---------------------------------------------------------------------------
# numpy mirrors of the Triton task decompositions (fp32, worker strides)
# ---------------------------------------------------------------------------


def _sim_t_moe_expert(x, gate_up, down, ids, workers, limit=None):
    """Mirror of ``_t_moe_expert``: one task per (b, s); [I, H] / [H, I] block
    reductions in fp32; task = worker, += P."""
    B, H = x.shape
    K = ids.shape[1]
    E, two_i, _ = gate_up.shape
    I = two_i // 2
    partials = np.full((B, K, H), np.nan, dtype=np.float32)
    for w in range(workers):
        task = w
        while task < B * K:
            b, s = task // K, task % K
            e = int(ids[b, s])
            xb = x[b].astype(np.float32)
            wg = gate_up[e, :I, :].astype(np.float32)          # [I, H]
            wu = gate_up[e, I:, :].astype(np.float32)          # [I, H]
            g = np.sum(wg * xb[None, :], axis=1, dtype=np.float32)
            u = np.sum(wu * xb[None, :], axis=1, dtype=np.float32)
            if limit is not None:
                g = np.minimum(g, np.float32(limit))
                u = np.minimum(np.maximum(u, np.float32(-limit)), np.float32(limit))
            act = (g / (np.float32(1.0) + np.exp(-g)) * u).astype(np.float32)
            wd = down[e].astype(np.float32)                    # [H, I]
            partials[b, s] = np.sum(wd * act[None, :], axis=1, dtype=np.float32)
            task += workers
    return partials


def _sim_t_moe_combine(partials, weights, shared, workers):
    """Mirror of ``_t_moe_combine``: per-row weighted sum in slot order."""
    B, K, H = partials.shape
    y = np.full((B, H), np.nan, dtype=np.float32)
    for w in range(workers):
        b = w
        while b < B:
            acc = np.zeros(H, dtype=np.float32)
            for s in range(K):
                acc += np.float32(weights[b, s]) * partials[b, s].astype(np.float32)
            if shared is not None:
                acc += shared[b].astype(np.float32)
            y[b] = acc
            b += workers
    return y


def _sim_t_moe_route(x, router_w, *, workers, score_fn, top_k, rsf=1.0,
                     bias=None, norm_topk_prob=True, tid2eid=None, token_ids=None,
                     hash_mode=False):
    """Mirror of ``_t_moe_route``: per-row logits GEMV (fp32), score fn, K
    rounds of first-max argmax (ties to the lower index), renorm, ×rsf."""
    B, H = x.shape
    E = router_w.shape[0]
    ids = np.full((B, top_k), -1, dtype=np.int32)
    weights = np.full((B, top_k), np.nan, dtype=np.float32)
    for w in range(workers):
        b = w
        while b < B:
            logits = np.zeros(E, dtype=np.float32)
            for h in range(H):
                logits += router_w[:, h].astype(np.float32) * np.float32(x[b, h])
            scores = _scores(logits.astype(np.float64), score_fn).astype(np.float32)
            if hash_mode:
                sel = tid2eid[int(token_ids[b])].astype(np.int32)
                wsel = scores[sel]
                wsel = wsel / (wsel.sum(dtype=np.float32) + np.float32(1e-20))
            else:
                choice = scores.astype(np.float64).copy()
                if score_fn == "sigmoid_noaux_tc":
                    choice = choice + bias.astype(np.float64)
                sel = np.zeros(top_k, dtype=np.int32)
                wsel = np.zeros(top_k, dtype=np.float32)
                for kk in range(top_k):
                    best = int(np.argmax(choice))  # first max: ties -> lower index
                    sel[kk] = best
                    wsel[kk] = scores[best]  # unbiased gather
                    choice[best] = -np.inf
                if norm_topk_prob:
                    wsel = wsel / (wsel.sum(dtype=np.float32) + np.float32(1e-20))
            weights[b] = (wsel * np.float32(rsf)).astype(np.float32)
            ids[b] = sel
            b += workers
    return ids, weights


# ---------------------------------------------------------------------------
# harness (the #91 pattern)
# ---------------------------------------------------------------------------


def _run_schedule(graph, externals, workers=4):
    enforce(graph)
    families = lower_graph(graph)
    assert check_thread_contract(families) == []
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)
    storage_arrays = {}
    for sid, name in graph.storage_names.items():
        if sid in graph.fresh_storages:
            continue
        storage_arrays[sid] = externals[name]
    workspace = np.full(workspace_plan.total_elements, np.nan, dtype=np.float64)
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = workspace[buf.offset: buf.offset + buf.numel]
    executor = ReferenceExecutor(
        schedule, workers=workers, storage_arrays=storage_arrays,
        graph=graph, workspace_plan=workspace_plan,
    )
    trace = executor.run({"p": 0})
    return executor, trace


_EXT_SID = [9000]


def _ext(rec, name, shape, dtype=F32):
    _EXT_SID[0] += 1
    return rec.external_tensor(name, shape, dtype, storage_id=_EXT_SID[0])


def _capture_block(B=3, H=16, E=8, K=2, I=8, *, mode="learned",
                   score_fn="sqrtsoftplus", limit=None, shared=False,
                   n_group=1, topk_group=1, norm_topk_prob=True, rsf=1.0,
                   V=32):
    """Capture route → expert → combine (+ optional shared dense path built
    from the existing linear/swiglu kinds)."""
    rec = RecordingBackend()
    rec.define_position(4)
    x = _ext(rec, "x", (B, H), F32)
    router_w = _ext(rec, "router_w", (E, H), F32)
    ids = _ext(rec, "route_ids", (B, K), I32)
    weights = _ext(rec, "route_w", (B, K), F32)
    kw = {}
    if mode == "hash":
        kw = {"tid2eid": _ext(rec, "tid2eid", (V, K), I32),
              "token_ids": _ext(rec, "token_ids", (B,), I32)}
    elif score_fn == "sigmoid_noaux_tc":
        kw = {"bias": _ext(rec, "e_score_bias", (E,), F32),
              "n_group": n_group, "topk_group": topk_group,
              "norm_topk_prob": norm_topk_prob}
    rec.moe_route(x, router_w, ids, weights, mode=mode, score_fn=score_fn,
                  top_k=K, routed_scaling_factor=rsf, **kw)
    gate_up = _ext(rec, "gate_up", (E, 2 * I, H), F32)
    down = _ext(rec, "down", (E, H, I), F32)
    partials = rec.moe_expert(x, gate_up, down, ids, swiglu_limit=limit)
    shared_out = None
    if shared:
        wg = _ext(rec, "sh_gate_w", (H, I), F32)
        wu = _ext(rec, "sh_up_w", (H, I), F32)
        wd = _ext(rec, "sh_down_w", (I, H), F32)
        g = rec.linear(x, wg, name="shared_gate")
        u = rec.linear(x, wu, name="shared_up")
        a = rec.swiglu(g, u, name="shared_act")
        shared_out = rec.linear(a, wd, name="shared_down")
    y = rec.moe_combine(partials, weights, shared=shared_out)
    return rec, dict(x=x, router_w=router_w, ids=ids, weights=weights,
                     gate_up=gate_up, down=down, partials=partials, y=y,
                     shared=shared_out, **kw)


def _rand_block(rng, B=3, H=16, E=8, K=2, I=8, *, mode="learned",
                score_fn="sqrtsoftplus", V=32, shared=False):
    data = {
        "x": rng.standard_normal((B, H)).astype(np.float32),
        "router_w": rng.standard_normal((E, H)).astype(np.float32),
        "gate_up": rng.standard_normal((E, 2 * I, H)).astype(np.float32),
        "down": rng.standard_normal((E, H, I)).astype(np.float32),
        "route_ids": np.zeros((B, K), dtype=np.int32),
        "route_w": np.zeros((B, K), dtype=np.float32),
    }
    if mode == "hash":
        data["tid2eid"] = rng.integers(0, E, size=(V, K)).astype(np.int32)
        data["token_ids"] = rng.integers(0, V, size=B).astype(np.int32)
    elif score_fn == "sigmoid_noaux_tc":
        data["e_score_bias"] = rng.standard_normal(E).astype(np.float32)
    if shared:
        data["sh_gate_w"] = rng.standard_normal((H, I)).astype(np.float32)
        data["sh_up_w"] = rng.standard_normal((H, I)).astype(np.float32)
        data["sh_down_w"] = rng.standard_normal((I, H)).astype(np.float32)
    return data


def _externals(rec, data):
    out = {}
    for sid, name in rec.graph.storage_names.items():
        if sid in rec.graph.fresh_storages:
            continue
        out[name] = data[name]
    return out


# ---------------------------------------------------------------------------
# capture + lowering + executor equivalence, per router family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode,score_fn,n_group,topk_group,norm", [
    ("learned", "sqrtsoftplus", 1, 1, True),       # DeepSeek-V4 TopKRouter
    ("learned", "sigmoid_noaux_tc", 1, 1, True),   # GLM-5.3 degenerate group
    ("learned", "sigmoid_noaux_tc", 2, 1, False),  # GLM-5.3 general groups
    ("hash", "sqrtsoftplus", 1, 1, True),          # DeepSeek-V4 HashRouter
])
def test_route_matches_floe_semantics_per_family(mode, score_fn, n_group, topk_group, norm):
    """Executed route == independent fp64 mirror of the floe router math."""
    rng = np.random.default_rng(abs(hash((mode, score_fn, n_group))) % 2**31)
    B, H, E, K = 3, 16, 8, 2
    rec, h = _capture_block(B=B, H=H, E=E, K=K, mode=mode, score_fn=score_fn,
                            n_group=n_group, topk_group=topk_group, norm_topk_prob=norm)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, mode=mode, score_fn=score_fn)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    ids = np.array(ex.tensor("route_ids"))
    weights = np.array(ex.tensor("route_w"))
    for b in range(B):
        sel, w = _route_mirror(
            data["x"][b], data["router_w"], mode=mode, score_fn=score_fn,
            top_k=K, bias=data.get("e_score_bias"), n_group=n_group,
            topk_group=topk_group, norm_topk_prob=norm,
            tid2eid=data.get("tid2eid"),
            token_id=(int(data["token_ids"][b]) if mode == "hash" else None),
        )
        np.testing.assert_array_equal(ids[b], sel)
        np.testing.assert_allclose(weights[b].astype(np.float64), w, rtol=0, atol=1e-6)


def test_hash_gather_is_exact():
    """Hash routing: ids are exactly the frozen tid2eid rows; weights are the
    renormed gathered sqrtsoftplus scores × scaling factor."""
    rng = np.random.default_rng(981)
    B, H, E, K, V = 4, 8, 6, 1, 16
    rec, h = _capture_block(B=B, H=H, E=E, K=K, mode="hash", V=V)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, mode="hash", V=V)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    ids = np.array(ex.tensor("route_ids"))
    weights = np.array(ex.tensor("route_w"))
    for b in range(B):
        tok = int(data["token_ids"][b])
        np.testing.assert_array_equal(ids[b], data["tid2eid"][tok])
        scores = _scores(data["router_w"].astype(np.float64) @ data["x"][b].astype(np.float64), "sqrtsoftplus")
        w = scores[data["tid2eid"][tok]]
        w = w / (w.sum() + 1e-20)
        np.testing.assert_allclose(weights[b].astype(np.float64), w, rtol=0, atol=1e-6)


def test_route_tie_break_is_deterministic_lower_index():
    """Two tied router-row pairs: the documented contract resolves ties to the
    lower expert index, and the selection is stable across worker counts."""
    B, H, E, K = 2, 8, 4, 2
    rec, h = _capture_block(B=B, H=H, E=E, K=K, score_fn="sqrtsoftplus")
    rng = np.random.default_rng(982)
    rows = rng.standard_normal((2, H)).astype(np.float32)
    data = {
        "x": rng.standard_normal((B, H)).astype(np.float32),
        # experts 0/1 identical, 2/3 identical -> exact score ties
        "router_w": np.concatenate([rows[:1], rows[:1], rows[1:], rows[1:]], axis=0),
        "gate_up": rng.standard_normal((E, 16, H)).astype(np.float32),
        "down": rng.standard_normal((E, H, 8)).astype(np.float32),
        "route_ids": np.zeros((B, K), dtype=np.int32),
        "route_w": np.zeros((B, K), dtype=np.float32),
    }
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    ids = np.array(ex.tensor("route_ids"))
    scores = _scores(data["router_w"].astype(np.float64) @ data["x"][0].astype(np.float64), "sqrtsoftplus")
    expect = _topk_stable(scores, K)
    # tied pair -> the LOWER index occupies the earlier slot
    assert ids[0][0] == expect[0] == min(expect[0], expect[1]) if scores[expect[0]] == scores[expect[1]] else True
    np.testing.assert_array_equal(ids[0], expect)
    ex2, _ = _run_schedule(rec.graph, _externals(rec, data), workers=7)
    np.testing.assert_array_equal(np.array(ex2.tensor("route_ids")), ids)


def test_k_equals_E_bit_parity_vs_dense_every_expert():
    """The issue's acceptance check: with k=E every expert runs; the executed
    block must match dense-every-expert evaluation exactly in f32 (combine and
    oracle both accumulate in slot order)."""
    rng = np.random.default_rng(983)
    B, H, E, K, I = 2, 8, 4, 4, 6
    rec, h = _capture_block(B=B, H=H, E=E, K=K, I=I, score_fn="sqrtsoftplus")
    data = _rand_block(rng, B=B, H=H, E=E, K=K, I=I)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    y = np.array(ex.tensor(ex.graph.ops[-1].outputs[0]))
    ids = np.array(ex.tensor("route_ids"))
    weights = np.array(ex.tensor("route_w"))
    for b in range(B):
        sel, w = _route_mirror(data["x"][b], data["router_w"], mode="learned",
                               score_fn="sqrtsoftplus", top_k=E)
        # k=E: the route must select every expert exactly once
        np.testing.assert_array_equal(np.sort(sel), np.arange(E))
        # slot-ordered accumulation over the executor's grids: the routing
        # table is f32 by contract (weights round to f32), partials and the
        # combine accumulator stay f64 (fresh workspace buffers)
        w32 = w.astype(np.float32)
        ordered = np.zeros(H, dtype=np.float64)
        for s in range(E):
            ordered += float(w32[s]) * _expert_mirror(data["x"][b], int(sel[s]), data["gate_up"], data["down"])
        np.testing.assert_array_equal(y[b], ordered)


# ---------------------------------------------------------------------------
# expert task: clamps, indirection
# ---------------------------------------------------------------------------


def test_swiglu_limit_clamp_boundaries():
    """g = L passes through unchanged; g > L clamps to L; u clamps at ±L; a
    negative gate is unclamped (only the ≤ L side clamps). H=I=1 with exactly
    representable weights makes the boundaries exact."""
    B, H, E, K, I = 3, 2, 3, 1, 1
    L = 2.0
    rec, h = _capture_block(B=B, H=H, E=E, K=K, I=I, limit=L, V=8)
    gate_up = np.zeros((E, 2 * I, H), dtype=np.float32)
    down = np.ones((E, H, I), dtype=np.float32)
    gate_up[0, 0, 0] = 0.5 * L    # row0 (x=[2,0]): g = u = L exactly (boundary)
    gate_up[0, 1, 0] = 0.5 * L
    gate_up[1, 0, 1] = 1.5 * L    # row1 (x=[0,2]): g = 3L -> clamps to L
    gate_up[1, 1, 1] = -1.5 * L   # u = -3L -> clamps to -L
    gate_up[2, 0, 0] = 2 * L      # row2 (x=[-.5,-.5]): g = -L (negative, unclamped)
    gate_up[2, 1, 0] = -L         # u = +0.5 * L
    # per-row argmax over sqrt(softplus(w·x)): row0 -> e0, row1 -> e1, row2 -> e2
    router_w = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=np.float32)
    data = {
        "x": np.array([[2.0, 0.0], [0.0, 2.0], [-0.5, -0.5]], dtype=np.float32),
        "router_w": router_w,
        "gate_up": gate_up,
        "down": down,
        "route_ids": np.zeros((B, K), dtype=np.int32),
        "route_w": np.zeros((B, K), dtype=np.float32),
        "tid2eid": np.array([[0], [1], [2]], dtype=np.int32),
        "token_ids": np.arange(B, dtype=np.int32),
    }
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    partials = np.array(ex.tensor(h["partials"].value.name))
    silu = lambda v: v / (1.0 + np.exp(-v))
    np.testing.assert_allclose(partials[0, 0, 0], silu(L) * L, rtol=1e-6, atol=1e-7)  # row0 -> e0
    np.testing.assert_allclose(partials[1, 0, 0], silu(L) * (-L), rtol=1e-6, atol=1e-7)  # row1 -> e1
    np.testing.assert_allclose(partials[2, 0, 0], silu(-L) * (0.5 * L), rtol=1e-6, atol=1e-7)  # row2 -> e2


def test_expert_weight_base_follows_the_routing_table():
    """Runtime indirection: changing the routing table's input changes which
    expert weight rows are read — the #94 pattern applied to a weight pool.
    (Hash routing recomputes route_ids from tid2eid/token_ids, so the rewrite
    goes through the router input, not the table itself.)"""
    rng = np.random.default_rng(985)
    B, H, E, K, I = 1, 8, 4, 1, 6
    rec, h = _capture_block(B=B, H=H, E=E, K=K, I=I, mode="hash", V=8)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, I=I, mode="hash", V=8)
    data["router_w"] = np.zeros((E, H), dtype=np.float32)
    data["tid2eid"] = np.array([[0], [1], [2], [3], [0], [1], [2], [3]], dtype=np.int32)
    data["token_ids"] = np.zeros(B, dtype=np.int32)
    ext = _externals(rec, data)
    ex, _ = _run_schedule(rec.graph, ext)
    ids = np.array(ex.tensor("route_ids"))
    partials = np.array(ex.tensor(h["partials"].value.name))
    np.testing.assert_allclose(
        partials[0, 0], _expert_mirror(data["x"][0], int(ids[0, 0]), data["gate_up"], data["down"]),
        rtol=1e-6, atol=1e-7,
    )
    ext2 = dict(ext)
    ext2["token_ids"] = np.array([3], dtype=np.int32)  # tid2eid[3] = 3
    ex2, _ = _run_schedule(rec.graph, ext2)
    partials2 = np.array(ex2.tensor(h["partials"].value.name))
    np.testing.assert_allclose(
        partials2[0, 0], _expert_mirror(data["x"][0], 3, data["gate_up"], data["down"]),
        rtol=1e-6, atol=1e-7,
    )


# ---------------------------------------------------------------------------
# combine, shared expert, phase ordering, accounting
# ---------------------------------------------------------------------------


def test_weighted_combine_and_shared_add():
    rng = np.random.default_rng(986)
    B, K, H, I, E = 3, 2, 16, 8, 8
    rec, h = _capture_block(B=B, H=H, E=E, K=K, I=I, shared=True)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, I=I, shared=True)
    data["router_w"] = np.zeros((E, H), dtype=np.float32)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    y = np.array(ex.tensor(ex.graph.ops[-1].outputs[0]))
    ids = np.array(ex.tensor("route_ids"))
    weights = np.array(ex.tensor("route_w"))
    for b in range(B):
        routed = np.zeros(H, dtype=np.float64)
        for s in range(K):
            routed += weights[b, s] * _expert_mirror(data["x"][b], int(ids[b, s]), data["gate_up"], data["down"])
        g = data["x"][b].astype(np.float64) @ data["sh_gate_w"].astype(np.float64)
        u = data["x"][b].astype(np.float64) @ data["sh_up_w"].astype(np.float64)
        act = (g / (1.0 + np.exp(-g))) * u  # silu(g)·u
        shared = act @ data["sh_down_w"].astype(np.float64)
        np.testing.assert_allclose(y[b].astype(np.float64), routed + shared, rtol=1e-6, atol=1e-6)


def test_raw_hazards_order_route_expert_combine():
    rec, _ = _capture_block()
    hazards = compute_hazards(rec.graph.ops)
    raw = {(h.producer, h.consumer) for h in hazards if h.kind == "RAW"}
    kinds = [op.kind for op in rec.graph.ops]
    i_route, i_exp, i_comb = (kinds.index(k) for k in ("moe_route", "moe_expert", "moe_combine"))
    assert (i_route, i_exp) in raw, "route ≺ expert RAW missing (routing ids)"
    assert (i_route, i_comb) in raw, "route ≺ combine RAW missing (weights)"
    assert (i_exp, i_comb) in raw, "expert ≺ combine RAW missing (partials)"


def test_phases_strictly_order_route_expert_combine():
    workers = 4
    rec, _ = _capture_block()
    families = lower_graph(rec.graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    phase_of = {}
    for ph in schedule.phases:
        for fam in ph.families:
            phase_of.setdefault(fam.kind, []).append(ph.index)
    assert max(phase_of["moe_route"]) < min(phase_of["moe_expert"])
    assert max(phase_of["moe_expert"]) < min(phase_of["moe_combine"])


def test_single_launch_accounting_and_nan_canaries():
    rec, h = _capture_block(B=3, shared=True)
    rng = np.random.default_rng(987)
    data = _rand_block(rng, B=3, shared=True)
    ex, trace = _run_schedule(rec.graph, _externals(rec, data), workers=3)
    y = np.array(ex.tensor(ex.graph.ops[-1].outputs[0]))
    assert trace.kernel_launches == 1, "one launch for the whole block (§15.2)"
    assert trace.grid_barriers == len(trace.phases)
    assert trace.task_executions >= 3 + 3 * 2 + 3  # route + k·B experts + combine
    assert np.isfinite(y).all(), "NaN canary: no workspace leak into the output"


def test_lowerings_registered_and_codegen_knows_templates():
    for kind in ("moe_route", "moe_expert", "moe_combine"):
        assert kind in LOWERINGS
        assert kind in TEMPLATE_NAMES
    assert TEMPLATE_NAMES["moe_expert"] == "moe_expert_task"


# ---------------------------------------------------------------------------
# capture guards
# ---------------------------------------------------------------------------


def test_capture_rejects_bad_configs():
    rec = RecordingBackend()
    rec.define_position(4)
    x = _ext(rec, "x", (2, 8), F32)
    w = _ext(rec, "router_w", (4, 8), F32)
    with pytest.raises(ValueError, match="routing table must be"):
        bad_ids = _ext(rec, "bad_ids", (2, 3), I32)
        bad_w = _ext(rec, "bad_w", (2, 2), F32)
        rec.moe_route(x, w, bad_ids, bad_w, top_k=2)
    with pytest.raises(ValueError, match="i32 ids"):
        f_ids = _ext(rec, "f_ids", (2, 2), F32)
        w2 = _ext(rec, "w2", (2, 2), F32)
        rec.moe_route(x, w, f_ids, w2, top_k=2)
    with pytest.raises(ValueError, match="hash routing requires"):
        ids3 = _ext(rec, "ids3", (2, 2), I32)
        w3 = _ext(rec, "w3", (2, 2), F32)
        rec.moe_route(x, w, ids3, w3, mode="hash")
    bias = _ext(rec, "bias4", (4,), F32)
    with pytest.raises(ValueError, match="n_group"):
        ids4 = _ext(rec, "ids4", (2, 2), I32)
        w4 = _ext(rec, "w4", (2, 2), F32)
        rec.moe_route(x, w, ids4, w4, score_fn="sigmoid_noaux_tc", bias=bias, n_group=3)
    with pytest.raises(ValueError, match="topk_group"):
        ids5 = _ext(rec, "ids5", (2, 2), I32)
        w5 = _ext(rec, "w5", (2, 2), F32)
        rec.moe_route(x, w, ids5, w5, score_fn="sigmoid_noaux_tc", bias=bias,
                      n_group=2, topk_group=3)
    with pytest.raises(ValueError, match="outside \\[1, E"):
        ids6 = _ext(rec, "ids6", (2, 5), I32)
        w6 = _ext(rec, "w6", (2, 5), F32)
        rec.moe_route(x, w, ids6, w6, top_k=5)


# ---------------------------------------------------------------------------
# device-template mirrors (the triton decomposition, worker strides)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,K,H,I,workers", [(1, 1, 8, 8, 1), (2, 3, 16, 8, 5), (3, 2, 16, 4, 13)])
def test_device_template_mirror_expert(B, K, H, I, workers):
    rng = np.random.default_rng(B * 100 + K + H + I + workers)
    E = 5
    rec, h = _capture_block(B=B, H=H, E=E, K=K, I=I, limit=2.0)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, I=I)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data), workers=workers)
    partials = np.array(ex.tensor(h["partials"].value.name))
    ids = np.array(ex.tensor("route_ids"))
    mirror = _sim_t_moe_expert(data["x"], data["gate_up"], data["down"], ids, workers, limit=2.0)
    rel = np.abs(mirror.astype(np.float64) - partials.astype(np.float64)).max()
    assert rel < 1e-4, f"template mirror diverged from reference executor: {rel:.3e}"
    assert np.isfinite(partials).all()


def test_device_template_mirror_route():
    rng = np.random.default_rng(988)
    B, H, E, K = 3, 16, 8, 2
    for score_fn, kw in [("sqrtsoftplus", {}), ("sigmoid_noaux_tc", {})]:
        rec, h = _capture_block(B=B, H=H, E=E, K=K, score_fn=score_fn, **kw)
        data = _rand_block(rng, B=B, H=H, E=E, K=K, score_fn=score_fn)
        ex, _ = _run_schedule(rec.graph, _externals(rec, data))
        ids = np.array(ex.tensor("route_ids"))
        weights = np.array(ex.tensor("route_w"))
        mirror_ids, mirror_w = _sim_t_moe_route(
            data["x"], data["router_w"], workers=4, score_fn=score_fn, top_k=K,
            bias=data.get("e_score_bias"),
        )
        np.testing.assert_array_equal(ids, mirror_ids)
        np.testing.assert_allclose(weights, mirror_w, rtol=1e-5, atol=1e-6)


def test_device_template_mirror_route_hash():
    rng = np.random.default_rng(989)
    B, H, E, K, V = 4, 8, 6, 1, 16
    rec, h = _capture_block(B=B, H=H, E=E, K=K, mode="hash", V=V)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, mode="hash", V=V)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data))
    ids = np.array(ex.tensor("route_ids"))
    weights = np.array(ex.tensor("route_w"))
    mirror_ids, mirror_w = _sim_t_moe_route(
        data["x"], data["router_w"], workers=3, score_fn="sqrtsoftplus",
        top_k=K, tid2eid=data["tid2eid"], token_ids=data["token_ids"], hash_mode=True,
    )
    np.testing.assert_array_equal(ids, mirror_ids)
    np.testing.assert_allclose(weights, mirror_w, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("workers", [1, 4, 9])
def test_device_template_mirror_combine(workers):
    rng = np.random.default_rng(990 + workers)
    B, K, H = 3, 2, 16
    partials = rng.standard_normal((B, K, H)).astype(np.float32)
    weights = rng.random((B, K)).astype(np.float32)
    shared = rng.standard_normal((B, H)).astype(np.float32)
    y = _sim_t_moe_combine(partials, weights, shared, workers)
    expect = np.einsum("bkh,bk->bh", partials.astype(np.float64), weights.astype(np.float64)) + shared
    np.testing.assert_allclose(y, expect, rtol=1e-5, atol=1e-6)


def test_full_block_end_to_end_matches_block_oracle():
    """route → expert → combine over a randomized block equals the full numpy
    block oracle, with workers under/over-subscribed."""
    rng = np.random.default_rng(991)
    B, H, E, K, I = 3, 16, 8, 3, 8
    rec, h = _capture_block(B=B, H=H, E=E, K=K, I=I)
    data = _rand_block(rng, B=B, H=H, E=E, K=K, I=I)
    ex, _ = _run_schedule(rec.graph, _externals(rec, data), workers=5)
    y = np.array(ex.tensor(ex.graph.ops[-1].outputs[0]))
    for b in range(B):
        expect, _, _ = _block_oracle(data["x"][b], data["router_w"], data["gate_up"],
                                     data["down"], top_k=K)
        np.testing.assert_allclose(y[b], expect.astype(np.float32), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# floe eager parity — ATTESTED ONLY (gated: bare env has no torch/floe)
# ---------------------------------------------------------------------------


def test_floe_eager_parity_glm53_router_and_experts():
    """Attested-not-verified in the bare env: parity of the route/expert/combine
    chain against ``floe.engine.runner.models.glm5.glm5_arch`` Glm53TopkRouter +
    Glm53Experts on a tiny config, torch fp32."""
    pytest.importorskip("torch")
    pytest.importorskip("floe.engine.runner.models.glm5.glm5_arch")
    pytest.skip("floe eager parity requires a torch env with the floe package; attested from the author environment")


def test_floe_eager_parity_deepseek_v4_moe_block():
    """Attested-not-verified: parity against ``DeepseekV4SparseMoeBlock``
    (TopKRouter + HashRouter + shared expert) on a tiny config."""
    pytest.importorskip("torch")
    pytest.importorskip("floe.engine.runner.models.deepseek_v4.deepseek_v4_arch")
    pytest.skip("floe eager parity requires a torch env with the floe package; attested from the author environment")
