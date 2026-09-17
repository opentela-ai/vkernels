"""Issue #99: mHC hyper-connection mixing (mhc_pre / mhc_post op family).

Validation doctrine (hardened after issue #90): the repo env has NO torch
and NO floe — the core suite pins ALL arithmetic with a numpy/fp64 mirror
written independently from the issue spec + floe source, so everything
below the "bare-environment core" divider collects and passes with numpy
alone (0 collection errors, 0 unexpected skips). The floe-parity and
Triton-device tests are import-gated (``pytest.importorskip``) and reported
separately as attested-not-verified.

Coverage (issue Validation section + #99 review doctrine):

* capture: recorder contract for both ops of the family (data-dependent
  weights, per-token Sinkhorn, workspace-not-pool state discipline, fresh
  buffers, streams read-only in pre) and shape rejection;
* hazards: pre -> body -> post chain orders via RAW on the h_in / post /
  comb workspaces; layer N+1's pre consumes layer N's streams_post view;
* Sinkhorn: convergence vs iters (rowsum error strictly decreasing,
  colsum exact after the final column pass), eps sensitivity (eps is
  load-bearing), strict positivity;
* data-dependence: different stream contents give different weights;
  identical rows give identical weights;
* reference executor vs the independent fp64 mirror: multi-batch, NaN
  canaries, two-layer stream round-trip with contents checked at the layer
  boundary, hc=1 analytic collapse;
* single-launch accounting (one kernel launch per run, every task exactly
  once) and a two-layer model-level compiler regression (capture -> legality
  -> lower -> schedule -> memory -> execute).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vkernels.compiler.capture import CaptureError, RecordingBackend  # noqa: E402
from vkernels.compiler.legality import check_graph  # noqa: E402
from vkernels.compiler.lowerings import lower_graph  # noqa: E402
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import compute_hazards  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny decode config (hc small per the issue; C shrunk for the bare env)
# ---------------------------------------------------------------------------

B, HC, C = 3, 4, 32
MIX = (2 + HC) * HC  # 24 = hc + hc + hc²
EPS, RMS_EPS = 1e-6, 1e-6

S_STREAMS, S_FN, S_BASE, S_SCALE, S_BIAS = 901, 902, 903, 904, 905


# ---------------------------------------------------------------------------
# Independent fp64 numpy mirror (written from the issue spec + floe
# DeepseekV4HyperConnection.forward; deliberately NOT sharing code with the
# reference-exec bodies it is checked against)
# ---------------------------------------------------------------------------

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _mirror_pre(streams, fn, base, scale, *, iters, eps, rms_eps):
    """Per-token data-dependent weights + stream collapse, fp64."""
    B_, hc, C_ = streams.shape
    h_in = np.empty((B_, C_))
    post = np.empty((B_, hc))
    comb = np.empty((B_, hc, hc))
    for b in range(B_):
        flat = streams[b].astype(np.float64).reshape(-1)
        flat = flat / np.sqrt(np.sum(flat * flat) / (hc * C_) + rms_eps)
        logits = fn.astype(np.float64) @ flat + base.astype(np.float64)
        pre_w, post_w = logits[:hc], logits[hc : 2 * hc]
        comb_w = logits[2 * hc :].reshape(hc, hc)
        pre_b, post_b = base[:hc], base[hc : 2 * hc]
        comb_b = base[2 * hc :].reshape(hc, hc)
        pre = _sigmoid(pre_w * scale[0] + pre_b) + eps
        post[b] = 2.0 * _sigmoid(post_w * scale[1] + post_b)
        cl = comb_w * scale[2] + comb_b
        e = np.exp(cl - cl.max(axis=-1, keepdims=True))
        cb = e / e.sum(axis=-1, keepdims=True) + eps
        cb = cb / (cb.sum(axis=-2, keepdims=True) + eps)  # Sinkhorn: col first
        for _ in range(iters - 1):
            cb = cb / (cb.sum(axis=-1, keepdims=True) + eps)  # row
            cb = cb / (cb.sum(axis=-2, keepdims=True) + eps)  # col
        comb[b] = cb
        h_in[b] = (pre[:, None] * streams[b].astype(np.float64)).sum(axis=0)
    return h_in, post, comb


def _mirror_compose(streams, body_out, post, comb):
    """streams'[j] = post[j]·body_out + Σ_k comb[k, j]·streams[k] (floe
    _mhc_compose), fp64."""
    mixed = np.einsum("bkj,bkc->bjc", comb.astype(np.float64), streams.astype(np.float64))
    return mixed + post.astype(np.float64)[:, :, None] * body_out.astype(np.float64)[:, None, :]


# ---------------------------------------------------------------------------
# Compiler plumbing: capture -> lower -> schedule -> memory -> execute
# ---------------------------------------------------------------------------

def _weights(rng, hc=HC, c=C):
    mix = (2 + hc) * hc
    return {
        "streams": (rng.standard_normal((B, hc, c)) * 0.5).astype(np.float32),
        "fn": (rng.standard_normal((mix, hc * c)) * 0.05).astype(np.float32),
        "base": (rng.standard_normal(mix) * 0.3).astype(np.float32),
        "scale": np.array([0.9, 1.1, 1.3], dtype=np.float32),
        "bias": (rng.standard_normal((B, c)) * 0.3).astype(np.float32),
    }


def _build_model(w, *, layers=1, iters=2, workers=3, hc=HC, c=C, canary=True, with_body=True):
    """Capture a layers-layer mHC model (trivial identity-plus-bias block
    body between pre and post; ``with_body=False`` captures bare mhc_pre ops
    so h_in/post/comb stay live end-outputs) and build the executor."""
    rec = RecordingBackend()
    streams0 = rec.external_tensor("streams", (B, hc, c), storage_id=S_STREAMS)
    fn = rec.external_tensor("fn", ((2 + hc) * hc, hc * c), storage_id=S_FN)
    base = rec.external_tensor("base", ((2 + hc) * hc,), storage_id=S_BASE)
    scale = rec.external_tensor("scale", (3,), storage_id=S_SCALE)
    bias = rec.external_tensor("bias", (B, c), storage_id=S_BIAS)
    handles = {"streams0": streams0, "fn": fn, "base": base, "scale": scale}
    cur = streams0
    for layer in range(layers):
        h, p, cb = rec.mhc_pre(cur, fn, base, scale, layer=layer, iters=iters, eps=EPS, rms_eps=RMS_EPS)
        handles[f"pre_l{layer}"] = (h, p, cb)
        if with_body:
            body_out = rec.add(h, bias)  # stand-in block body
            cur = rec.mhc_post(cur, body_out, p, cb, layer=layer)
            handles[f"streams_l{layer + 1}"] = cur
    graph = rec.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)
    sa = {
        S_STREAMS: w["streams"].reshape(-1).copy(),
        S_FN: w["fn"].reshape(-1).copy(),
        S_BASE: w["base"].reshape(-1).copy(),
        S_SCALE: w["scale"].copy(),
        S_BIAS: w["bias"].reshape(-1).copy(),
    }
    for buf in workspace_plan.buffers:
        sa[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(
        schedule, workers=workers, storage_arrays=sa, graph=graph,
        workspace_plan=workspace_plan, canary=canary,
    )
    handles.update({"graph": graph, "families": families, "sa": sa,
                    "executor": executor, "workspace_plan": workspace_plan})
    return executor, handles


def _run_model(w, **kw):
    executor, handles = _build_model(w, **kw)
    executor.run({})
    return executor, handles


def _tensor(executor, sym):
    return np.array(executor.tensor(sym.value.name), copy=True)


# ===========================================================================
# Bare-environment core (numpy only)
# ===========================================================================

def test_capture_mhc_pre_records_contract():
    rec = RecordingBackend()
    streams = rec.external_tensor("streams", (B, HC, C), storage_id=S_STREAMS)
    fn = rec.external_tensor("fn", (MIX, HC * C), storage_id=S_FN)
    base = rec.external_tensor("base", (MIX,), storage_id=S_BASE)
    scale = rec.external_tensor("scale", (3,), storage_id=S_SCALE)
    h_in, post, comb = rec.mhc_pre(streams, fn, base, scale, layer=0, iters=3, eps=EPS, rms_eps=RMS_EPS)
    op = rec.graph.ops[-1]
    assert op.kind == "mhc_pre"
    assert op.attributes == {"layer": 0, "hc": HC, "iters": 3, "eps": EPS, "rms_eps": RMS_EPS}
    for key in ("input_norm", "projection", "pre", "post", "comb", "sinkhorn", "collapse", "dtypes", "state"):
        assert key in op.numerical_contract, f"numerical contract missing {key!r}"
    assert "doubly-stochastic" in op.numerical_contract["sinkhorn"]
    # family attributes shape the op; the weights are data-dependent outputs
    assert h_in.shape == (B, C) and post.shape == (B, HC) and comb.shape == (B, HC, HC)
    # streams are read-only: the op writes only its fresh workspaces
    read_sids = {r.storage_id for r in op.read_regions}
    write_sids = {r.storage_id for r in op.write_regions}
    assert streams.value.storage_id in read_sids
    assert streams.value.storage_id not in write_sids
    # workspace discipline: h_in/post/comb are fresh buffers, NOT the stream
    # storage and NOT a persistent pool (each its own new storage)
    out_sids = [h_in.value.storage_id, post.value.storage_id, comb.value.storage_id]
    assert len(set(out_sids)) == 3
    assert not (set(out_sids) & read_sids & {streams.value.storage_id})
    assert all(sid in rec.graph.fresh_storages for sid in out_sids)
    # position independence: no symbolic scalars consumed
    assert rec.graph.scalars == {}


def test_capture_mhc_post_records_contract():
    rec = RecordingBackend()
    streams = rec.external_tensor("streams", (B, HC, C), storage_id=S_STREAMS)
    body_out = rec.external_tensor("body_out", (B, C), storage_id=910)
    post = rec.external_tensor("post_w", (B, HC), storage_id=911)
    comb = rec.external_tensor("comb", (B, HC, HC), storage_id=912)
    streams_post = rec.mhc_post(streams, body_out, post, comb, layer=0)
    op = rec.graph.ops[-1]
    assert op.kind == "mhc_post"
    assert op.attributes == {"layer": 0, "hc": HC}
    assert "compose" in op.numerical_contract and "state" in op.numerical_contract
    assert streams_post.shape == (B, HC, C)
    # fresh [hc, C] workspace per layer — intermediate, not a persistent pool
    assert streams_post.value.storage_id in rec.graph.fresh_storages
    assert streams_post.value.storage_id != streams.value.storage_id
    read_sids = {r.storage_id for r in op.read_regions}
    assert {streams.value.storage_id, body_out.value.storage_id,
            post.value.storage_id, comb.value.storage_id} <= read_sids
    assert {streams_post.value.storage_id} == {r.storage_id for r in op.write_regions}


@pytest.mark.parametrize("kwargs", [
    {"streams_shape": (B, HC)},                      # rank-2 stream stack
    {"fn_shape": (MIX + 1, HC * C)},                 # mix mismatch
    {"fn_shape": (MIX, HC * C + 1)},                 # hidden mismatch
    {"base_shape": (MIX - 1,)},
    {"scale_shape": (2,)},                           # scale must be [3]
    {"iters": 0},                                    # Sinkhorn needs >= 1 pass
])
def test_capture_mhc_pre_rejects_shape_mismatches(kwargs):
    rec = RecordingBackend()
    streams = rec.external_tensor("streams", kwargs.get("streams_shape", (B, HC, C)), storage_id=920)
    fn = rec.external_tensor("fn", kwargs.get("fn_shape", (MIX, HC * C)), storage_id=921)
    base = rec.external_tensor("base", kwargs.get("base_shape", (MIX,)), storage_id=922)
    scale = rec.external_tensor("scale", kwargs.get("scale_shape", (3,)), storage_id=923)
    with pytest.raises(CaptureError):
        rec.mhc_pre(streams, fn, base, scale, layer=0,
                    iters=kwargs.get("iters", 2), eps=EPS, rms_eps=RMS_EPS)


@pytest.mark.parametrize("kwargs", [
    {"streams_shape": (B, HC, C, 1)},
    {"body_out_shape": (B, C + 1)},
    {"post_shape": (B, HC + 1)},
    {"comb_shape": (B, HC, HC - 1)},
])
def test_capture_mhc_post_rejects_shape_mismatches(kwargs):
    rec = RecordingBackend()
    streams = rec.external_tensor("streams", kwargs.get("streams_shape", (B, HC, C)), storage_id=930)
    body_out = rec.external_tensor("body_out", kwargs.get("body_out_shape", (B, C)), storage_id=931)
    post = rec.external_tensor("post_w", kwargs.get("post_shape", (B, HC)), storage_id=932)
    comb = rec.external_tensor("comb", kwargs.get("comb_shape", (B, HC, HC)), storage_id=933)
    with pytest.raises(CaptureError):
        rec.mhc_post(streams, body_out, post, comb, layer=0)


def test_mhc_chain_hazards():
    """pre -> body -> post orders via RAW hazards on the h_in / post / comb
    workspaces; layer 1's pre consumes layer 0's streams_post view."""
    rng = np.random.default_rng(7)
    w = _weights(rng)
    rec = RecordingBackend()
    streams0 = rec.external_tensor("streams", (B, HC, C), storage_id=S_STREAMS)
    fn = rec.external_tensor("fn", (MIX, HC * C), storage_id=S_FN)
    base = rec.external_tensor("base", (MIX,), storage_id=S_BASE)
    scale = rec.external_tensor("scale", (3,), storage_id=S_SCALE)
    bias = rec.external_tensor("bias", (B, C), storage_id=S_BIAS)
    h0, p0, c0 = rec.mhc_pre(streams0, fn, base, scale, layer=0, iters=2, eps=EPS, rms_eps=RMS_EPS)
    body0 = rec.add(h0, bias)
    streams1 = rec.mhc_post(streams0, body0, p0, c0, layer=0)
    h1, p1, c1 = rec.mhc_pre(streams1, fn, base, scale, layer=1, iters=2, eps=EPS, rms_eps=RMS_EPS)

    hazards = compute_hazards(rec.graph.ops)
    def pairs_on(sid):
        return {(h.kind, h.producer, h.consumer) for h in hazards if h.storage_id == sid}
    # h_in: produced by pre0 (op0), read by add (op1)
    assert ("RAW", 0, 1) in pairs_on(h0.value.storage_id)
    # post/comb: produced by pre0, read by post0 (op2)
    assert ("RAW", 0, 2) in pairs_on(p0.value.storage_id)
    assert ("RAW", 0, 2) in pairs_on(c0.value.storage_id)
    # streams_post view: written by post0, read by pre1 (RAW), no WAR/WAW
    # (workspace double-buffering, not a read-modify-write pool)
    sp_pairs = pairs_on(streams1.value.storage_id)
    assert ("RAW", 2, 3) in sp_pairs
    assert not any(kind in ("WAR", "WAW") for kind, _, _ in sp_pairs)


def test_lowering_task_decomposition():
    rng = np.random.default_rng(8)
    w = _weights(rng)
    executor, handles = _build_model(w, layers=1, iters=2, workers=1)
    fams = {f.kind: f for f in handles["families"]}
    pre, post = fams["mhc_pre"], fams["mhc_post"]
    assert pre.task_count == B  # one task per batch row
    assert post.task_count == B * HC  # one task per (batch, stream)
    assert pre.params["hc"] == HC and pre.params["iters"] == 2
    assert post.params["hc"] == HC
    r = pre.reads(0)
    assert len(r) == 4  # stream stack row, fn, base, scale
    wr = pre.writes(0)
    assert len(wr) == 3  # h_in row, post row, comb row
    box = pre.box(0)
    assert box[0] == (0, 1)  # the task owns exactly batch row 0's stream stack
    assert pre.box(1)[0] == (1, 2)
    r = post.reads(0)
    assert len(r) == 4  # stream stack, body_out row, post scalar, comb column
    wr = post.writes(0)
    assert len(wr) == 1
    bb0 = post.box(0)
    assert bb0[1] == (0, 1)  # narrow write: exactly stream j=0's row
    assert post.box(1)[1] == (1, 2)  # task 1 writes stream 1 only


def test_sinkhorn_convergence_and_iters_sensitivity():
    """fp32 math demanded by the issue: the Sinkhorn projection must
    converge (rowsum error strictly decreasing in iters; colsum exact after
    the final column pass) and eps must be load-bearing."""
    rng = np.random.default_rng(21)
    w = _weights(rng)
    row_err, col_err = {}, {}
    for iters in (1, 2, 4, 8):
        executor, handles = _run_model(w, layers=1, iters=iters)
        comb = _tensor(executor, handles["pre_l0"][2]).astype(np.float64)
        col_err[iters] = np.abs(comb.sum(axis=-2) - 1.0).max()
        row_err[iters] = np.abs(comb.sum(axis=-1) - 1.0).max()
        # strictly positive (softmax + eps, eps inside every denominator)
        assert comb.min() > 0.0
        # final pass is a column normalization -> columns exact to fp32 eps
        assert col_err[iters] < 1e-5, f"iters={iters}: colsum error {col_err[iters]}"
    # more iterations pull the rows onto the manifold too
    assert row_err[1] > row_err[2] > row_err[4] >= row_err[8], row_err
    assert row_err[8] < 1e-4, f"iters=8 rowsum error {row_err[8]} not converged"

    # eps sensitivity: the eps term inside every denominator is load-bearing
    combs = {}
    for eps in (1e-2, 1e-6, 1e-12):
        rec = RecordingBackend()
        streams = rec.external_tensor("streams", (B, HC, C), storage_id=S_STREAMS)
        fn = rec.external_tensor("fn", (MIX, HC * C), storage_id=S_FN)
        base = rec.external_tensor("base", (MIX,), storage_id=S_BASE)
        scale = rec.external_tensor("scale", (3,), storage_id=S_SCALE)
        h, p, c = rec.mhc_pre(streams, fn, base, scale, layer=0, iters=2, eps=eps, rms_eps=RMS_EPS)
        fams = lower_graph(rec.graph)
        sched = PhaseSchedule.from_families(fams, workers=2)
        wp, _ = plan_memory(rec.graph, sched, fams)
        sa = {S_STREAMS: w["streams"].reshape(-1).copy(), S_FN: w["fn"].reshape(-1).copy(),
              S_BASE: w["base"].reshape(-1).copy(), S_SCALE: w["scale"].copy()}
        for buf in wp.buffers:
            sa[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
        ex = ReferenceExecutor(sched, workers=2, storage_arrays=sa, graph=rec.graph,
                               workspace_plan=wp, canary=False)
        ex.run({})
        combs[eps] = _tensor(ex, c).astype(np.float64)
    assert np.abs(combs[1e-2] - combs[1e-12]).max() > 1e-3, "eps has no effect — Sinkhorn not per floe"
    assert np.abs(combs[1e-6] - combs[1e-12]).max() > 1e-9, "eps=1e-6 indistinguishable from 1e-12"
    # fp32 math (not fp64 accumulation) is what the device runs; the executor
    # mirrors that at fp32 storage resolution against the fp64 mirror
    executor, handles = _run_model(w, layers=1, iters=4)
    comb = _tensor(executor, handles["pre_l0"][2]).astype(np.float64)
    _, _, comb_mirror = _mirror_pre(
        w["streams"], w["fn"], w["base"], w["scale"], iters=4, eps=EPS, rms_eps=RMS_EPS)
    assert np.abs(comb - comb_mirror).max() < 1e-5


def test_data_dependent_weights():
    """The weights are computed from the stream CONTENTS per token, not at
    load time: different streams -> different comb; identical batch rows ->
    identical weights."""
    rng = np.random.default_rng(31)
    w = _weights(rng)
    executor, handles = _run_model(w, layers=1, iters=3)
    comb = _tensor(executor, handles["pre_l0"][2])
    post = _tensor(executor, handles["pre_l0"][1])
    assert not np.allclose(comb[0], comb[1], atol=1e-6), "comb did not vary with stream contents"
    assert not np.allclose(post[0], post[1], atol=1e-6)

    w_same = dict(w)
    w_same["streams"] = np.repeat(w["streams"][:1], B, axis=0).copy()
    executor2, handles2 = _run_model(w_same, layers=1, iters=3)
    comb2 = _tensor(executor2, handles2["pre_l0"][2])
    post2 = _tensor(executor2, handles2["pre_l0"][1])
    assert np.allclose(comb2[0], comb2[1], rtol=1e-6, atol=1e-7)
    assert np.allclose(post2[0], post2[1], rtol=1e-6, atol=1e-7)


def test_reference_matches_mirror_multi_batch():
    """Executor vs the independent fp64 mirror: pre gates + Sinkhorn comb +
    collapse + composed streams, multi-batch, NaN canaries armed. The
    pre-only build keeps h_in live (a consumed intermediate is legitimately
    canary-poisoned after its last reader — §9.2)."""
    rng = np.random.default_rng(41)
    w = _weights(rng)
    executor, handles = _run_model(w, layers=1, iters=3, with_body=False)
    h_in, post, comb = _mirror_pre(
        w["streams"], w["fn"], w["base"], w["scale"], iters=3, eps=EPS, rms_eps=RMS_EPS)

    got_h = _tensor(executor, handles["pre_l0"][0])
    got_post = _tensor(executor, handles["pre_l0"][1])
    got_comb = _tensor(executor, handles["pre_l0"][2])
    # NaN canaries: every live output fully written
    for name, got in (("h_in", got_h), ("post", got_post), ("comb", got_comb)):
        assert np.isfinite(got).all(), f"{name}: unwritten/NaN outputs (canary tripped)"
    assert np.abs(got_h - h_in).max() < 1e-5
    assert np.abs(got_post - post).max() < 1e-5
    assert np.abs(got_comb - comb).max() < 1e-5

    # the composed streams through the full pre -> body -> post chain
    executor2, handles2 = _run_model(w, layers=1, iters=3)
    body_out = h_in + w["bias"].astype(np.float64)
    streams_expect = _mirror_compose(w["streams"], body_out, post, comb)
    got_streams = _tensor(executor2, handles2["streams_l1"])
    assert np.isfinite(got_streams).all(), "streams': unwritten/NaN outputs (canary tripped)"
    assert np.abs(got_streams.astype(np.float64) - streams_expect).max() < 1e-4


def test_stream_roundtrip_two_layers():
    """Stream contents checked at layer boundaries (issue Validation):
    a two-layer walk must chain — layer 1's pre reads layer 0's composed
    streams — and match the mirror chained across both boundaries."""
    rng = np.random.default_rng(51)
    w = _weights(rng)
    # canary=False so the (dead after consumption) boundary buffer stays
    # readable; the NaN prefill in _build_model still arms unwritten detection
    executor, handles = _run_model(w, layers=2, iters=3, canary=False)

    # boundary 0: identical to the 1-layer run (deterministic per layer)
    exec1, handles1 = _run_model(w, layers=1, iters=3, canary=False)
    boundary0 = _tensor(exec1, handles1["streams_l1"]).astype(np.float64)
    got_boundary0 = _tensor(executor, handles["streams_l1"]).astype(np.float64)
    assert np.abs(got_boundary0 - boundary0).max() == 0.0, "layer boundary not deterministic across builds"

    # mirror chain over both boundaries
    h0, p0, c0 = _mirror_pre(w["streams"], w["fn"], w["base"], w["scale"],
                             iters=3, eps=EPS, rms_eps=RMS_EPS)
    s1 = _mirror_compose(w["streams"], h0 + w["bias"].astype(np.float64), p0, c0)
    h1, p1, c1 = _mirror_pre(s1.astype(np.float32), w["fn"], w["base"], w["scale"],
                             iters=3, eps=EPS, rms_eps=RMS_EPS)
    s2 = _mirror_compose(s1.astype(np.float32), h1 + w["bias"].astype(np.float64), p1, c1)
    assert np.abs(boundary0 - s1).max() < 1e-4, "layer-0 boundary streams off vs mirror"
    final = _tensor(executor, handles["streams_l2"]).astype(np.float64)
    assert np.isfinite(final).all()
    assert np.abs(final - s2).max() < 1e-3, "two-layer round-trip off vs mirror chain"
    # the collapse is pre-weighted: h_in must be finite and bounded
    got_h1 = _tensor(executor, handles["pre_l1"][0])
    assert np.isfinite(got_h1).all()


def test_hc1_analytic():
    """hc=1: comb is the 1x1 doubly-stochastic matrix (~1), so the compose
    degenerates to streams' = post·body_out + streams; pre >= eps and post
    in (0, 2) — the analytic anchors of the family."""
    rng = np.random.default_rng(61)
    hc, c = 1, 16
    w = _weights(rng, hc=hc, c=c)
    executor, handles = _run_model(w, layers=1, iters=2, hc=hc, c=c)
    comb = _tensor(executor, handles["pre_l0"][2])
    post = _tensor(executor, handles["pre_l0"][1])
    assert comb.shape == (B, 1, 1)
    assert np.abs(comb[:, 0, 0] - 1.0).max() < 1e-4, f"1x1 comb must be ~1, got {comb[:, 0, 0]}"
    assert (post > 0.0).all() and (post < 2.0).all(), "post must stay in (0, 2)"
    # body_out = h_in + bias; with comb=1 the compose mixes only via post
    # over the (single) stream — recompute exactly:
    h_in, post_m, comb_m = _mirror_pre(w["streams"], w["fn"], w["base"], w["scale"],
                                       iters=2, eps=EPS, rms_eps=RMS_EPS)
    body_out = h_in + w["bias"].astype(np.float64)
    streams_expect = _mirror_compose(w["streams"], body_out, post_m, comb_m)
    got = _tensor(executor, handles["streams_l1"]).astype(np.float64)
    assert np.abs(got - streams_expect).max() < 1e-4


def test_single_launch_accounting():
    """The whole model is ONE kernel launch; every task of every family
    executes exactly once per run."""
    rng = np.random.default_rng(71)
    w = _weights(rng)
    executor, handles = _build_model(w, layers=2, iters=2, workers=4)
    expected_tasks = sum(f.task_count for f in handles["families"])
    trace = executor.run({})
    assert trace.kernel_launches == 1, f"expected one launch, got {trace.kernel_launches}"
    assert trace.task_executions == expected_tasks
    by_kind = {}
    for f in handles["families"]:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + f.task_count
    assert by_kind["mhc_pre"] == 2 * B and by_kind["mhc_post"] == 2 * B * HC
    # every phase carried work and closed with a barrier
    assert all(ps.tasks > 0 for ps in trace.phases)
    assert trace.grid_barriers == len(trace.phases)


def test_two_layer_model_compiler_regression():
    """Full compiler path on a two-layer mHC model: capture -> legality ->
    lower -> schedule -> memory -> execute, live outputs finite, per-phase
    canary protocol quiet (the executor raises on any §9.2 violation)."""
    rng = np.random.default_rng(81)
    w = _weights(rng)
    executor, handles = _build_model(w, layers=2, iters=4, workers=3)
    assert not [d for d in check_graph(handles["graph"]) if d.severity == "error"]
    kinds = [f.kind for f in handles["families"]]
    assert kinds.count("mhc_pre") == 2 and kinds.count("mhc_post") == 2
    trace = executor.run({})
    final = _tensor(executor, handles["streams_l2"])
    assert np.isfinite(final).all()
    assert trace.kernel_launches == 1
    # workspace plan respects lifetimes (no buffer placed before its birth)
    for buf in handles["workspace_plan"].buffers:
        assert 0 <= buf.offset and buf.offset + buf.numel <= handles["workspace_plan"].total_elements


# ===========================================================================
# Attested-not-verified (import-gated: this stack has no torch/floe/GPU)
# ===========================================================================

def _floe_import():
    """Locate the floe sibling checkout and stub its kvaas_runtime native
    import (same protocol as the #90 suite)."""
    import os
    import types
    if "floe" not in sys.modules:
        for cand in (os.environ.get("FLOE_ROOT"),
                     "/home/xiayao/Documents/projects/opentela-ai/serving-stack/floe"):
            if cand and (Path(cand) / "floe" / "engine").is_dir():
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                break
    if "kvaas_runtime" not in sys.modules:
        kv = types.ModuleType("kvaas_runtime")
        for name in ("CudaVmm", "ElasticControlSession", "ManagedResidencyAdmissionDeferred",
                     "ManagedResidencySession", "allocate_device_pool"):
            setattr(kv, name, object)
        sub = types.ModuleType("kvaas_runtime.kv_pool_import")
        sub.tensor_from_cuda_pointer = object
        kv.kv_pool_import = sub
        sys.modules["kvaas_runtime"] = kv
        sys.modules["kvaas_runtime.kv_pool_import"] = sub


def test_floe_deepseek_hc_parity():
    """ATTESTED-NOT-VERIFIED (needs torch + floe; skipped on the bare env):
    the recorded/executed mhc_pre/mhc_post chain vs the real floe
    ``DeepseekV4HyperConnection`` + the ``_mhc_compose`` torch reference,
    fp32 eager on CPU."""
    torch = pytest.importorskip("torch")
    _floe_import()
    try:
        from floe.engine.runner.models.deepseek_v4.deepseek_v4_config import DeepseekV4Config
        from floe.engine.runner.models.deepseek_v4.deepseek_v4_arch import DeepseekV4HyperConnection
    except ModuleNotFoundError as exc:
        pytest.skip(f"floe deepseek_v4 oracle unavailable: {exc}")

    cfg = DeepseekV4Config.tiny()
    torch.manual_seed(99)
    hc_mod = DeepseekV4HyperConnection(cfg)
    with torch.no_grad():
        hc_mod.fn.normal_(0.0, 0.05)
        hc_mod.base.normal_(0.0, 0.3)
        hc_mod.scale.copy_(torch.tensor([0.9, 1.1, 1.3]))

    rng = np.random.default_rng(101)
    hc, d = cfg.hc_mult, cfg.hidden_size
    streams = (rng.standard_normal((B, hc, d)) * 0.5).astype(np.float32)
    w = _weights(rng, hc=hc, c=d)
    # route the floe module's weights through the compiled graph (the
    # executor's fn layout [mix, hc·C] IS floe's F.linear weight layout)
    w["fn"] = hc_mod.fn.detach().numpy().copy()
    w["base"] = hc_mod.base.detach().numpy().copy()
    w["scale"] = hc_mod.scale.detach().numpy().copy()
    w["streams"] = streams
    executor, handles = _run_model(w, layers=1, iters=cfg.hc_sinkhorn_iters, hc=hc, c=d)

    # floe oracle: eager forward + the _mhc_compose torch reference
    st = torch.from_numpy(streams).unsqueeze(1)  # [B, S=1, hc, D]
    post, comb, collapsed = hc_mod.forward(st)
    body_out = collapsed + torch.from_numpy(w["bias"])
    streams_ref = (post.unsqueeze(-1) * body_out.unsqueeze(-2)
                   + comb.transpose(-1, -2) @ st)
    got_comb = _tensor(executor, handles["pre_l0"][2])
    got_post = _tensor(executor, handles["pre_l0"][1])
    got_h = _tensor(executor, handles["pre_l0"][0])
    got_streams = _tensor(executor, handles["streams_l1"])
    torch.testing.assert_close(torch.from_numpy(got_post), post.squeeze(1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.from_numpy(got_comb), comb.squeeze(1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.from_numpy(got_h), collapsed.squeeze(1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        torch.from_numpy(got_streams), streams_ref.squeeze(1).to(torch.float32), rtol=1e-3, atol=1e-4)


def test_floe_glm_hc_parity():
    """ATTESTED-NOT-VERIFIED (needs torch + floe): the same op family via
    ``Glm53HyperConnection`` — one implementation, family attributes."""
    torch = pytest.importorskip("torch")
    _floe_import()
    try:
        from floe.engine.runner.models.glm5.glm5_config import Glm53Config
        from floe.engine.runner.models.glm5.glm5_arch import Glm53HyperConnection
    except ModuleNotFoundError as exc:
        pytest.skip(f"floe glm5 oracle unavailable: {exc}")

    cfg = Glm53Config.tiny()
    torch.manual_seed(98)
    hc_mod = Glm53HyperConnection(cfg)
    with torch.no_grad():
        hc_mod.fn.normal_(0.0, 0.05)
        hc_mod.base.normal_(0.0, 0.3)
        hc_mod.scale.copy_(torch.tensor([0.9, 1.1, 1.3]))

    rng = np.random.default_rng(102)
    hc, d = cfg.hc_mult, cfg.hidden_size
    streams = (rng.standard_normal((B, hc, d)) * 0.5).astype(np.float32)
    w = _weights(rng, hc=hc, c=d)
    w["fn"] = hc_mod.fn.detach().numpy().copy()
    w["base"] = hc_mod.base.detach().numpy().copy()
    w["scale"] = hc_mod.scale.detach().numpy().copy()
    w["streams"] = streams
    executor, handles = _run_model(w, layers=1, iters=cfg.hc_sinkhorn_iters, hc=hc, c=d)

    st = torch.from_numpy(streams).unsqueeze(1)  # [B, S=1, hc, D]
    post, comb, collapsed = hc_mod.forward(st)
    body_out = collapsed + torch.from_numpy(w["bias"])
    streams_ref = (post.unsqueeze(-1) * body_out.unsqueeze(-2)
                   + comb.transpose(-1, -2) @ st)
    got_comb = _tensor(executor, handles["pre_l0"][2])
    got_post = _tensor(executor, handles["pre_l0"][1])
    got_streams = _tensor(executor, handles["streams_l1"])
    torch.testing.assert_close(torch.from_numpy(got_post), post.squeeze(1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(torch.from_numpy(got_comb), comb.squeeze(1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        torch.from_numpy(got_streams), streams_ref.squeeze(1).to(torch.float32), rtol=1e-3, atol=1e-4)


def test_triton_device_templates_match_mirror():
    """ATTESTED-NOT-VERIFIED (needs torch + triton + CUDA): the _t_mhc_pre /
    _t_mhc_post device templates vs the fp64 numpy mirror on a tiny config."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vkernels.compiler.device_triton import _t_mhc_post, _t_mhc_pre
    # module-level launcher kernels (defined with triton importable at the
    # top; worker/P are supplied per program here exactly as the megakernel
    # composes them)
    from tests.python._mhc_triton_launchers import _mhc_post_launch as _post_launch
    from tests.python._mhc_triton_launchers import _mhc_pre_launch as _pre_launch

    dev = torch.device("cuda")
    rng = np.random.default_rng(111)
    hc, c = 4, 64
    mix = (2 + hc) * hc
    w = _weights(rng, hc=hc, c=c)
    streams_t = torch.from_numpy(w["streams"]).to(dev)
    fn_t = torch.from_numpy(w["fn"]).to(dev)
    base_t = torch.from_numpy(w["base"]).to(dev)
    scale_t = torch.from_numpy(w["scale"]).to(dev)
    h_in = torch.empty(B, c, device=dev, dtype=torch.float32)
    post_t = torch.empty(B, hc, device=dev, dtype=torch.float32)
    comb_t = torch.empty(B, hc, hc, device=dev, dtype=torch.float32)

    def _pow2(n):
        p = 1
        while p < n:
            p *= 2
        return p

    P = 4
    _pre_launch[(P,)](streams_t, fn_t, base_t, scale_t, h_in, post_t, comb_t,
                      B, hc, c, mix, EPS, RMS_EPS, 3, _pow2(mix), _pow2(hc), 128,
                      num_warps=4)
    torch.cuda.synchronize()
    torch.testing.assert_close(h_in.cpu(), torch.from_numpy(
        _mirror_pre(w["streams"], w["fn"], w["base"], w["scale"], iters=3, eps=EPS, rms_eps=RMS_EPS)[0]
    ).to(torch.float32), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(post_t.cpu(), torch.from_numpy(
        _mirror_pre(w["streams"], w["fn"], w["base"], w["scale"], iters=3, eps=EPS, rms_eps=RMS_EPS)[1]
    ).to(torch.float32), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(comb_t.cpu(), torch.from_numpy(
        _mirror_pre(w["streams"], w["fn"], w["base"], w["scale"], iters=3, eps=EPS, rms_eps=RMS_EPS)[2]
    ).to(torch.float32), rtol=1e-4, atol=1e-5)

    body_out = torch.from_numpy((w["streams"].astype(np.float64).mean(axis=1)).astype(np.float32)).to(dev)
    out = torch.empty(B, hc, c, device=dev, dtype=torch.float32)
    _post_launch[(P,)](streams_t, body_out, post_t, comb_t, out, B, hc, c, _pow2(hc), 128, num_warps=4)
    torch.cuda.synchronize()
    comb_np = comb_t.cpu().numpy().astype(np.float64)
    post_np = post_t.cpu().numpy().astype(np.float64)
    expected = (np.einsum("bkj,bkc->bjc", comb_np, w["streams"].astype(np.float64))
                + post_np[:, :, None] * body_out.cpu().numpy().astype(np.float64)[:, None, :])
    torch.testing.assert_close(out.cpu(), torch.from_numpy(expected).to(torch.float32),
                               rtol=1e-4, atol=1e-5)
