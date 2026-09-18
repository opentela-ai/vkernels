"""Issue #101: KDA gated delta rule decode op (GLM-5.3 linear attention).

``kda_delta`` — the element-wise-decay delta rule (Kimi delta attention),
generalizing #90's scalar-per-head ``gdn_delta``: the forget gate is
per-(head, k-dim) log-space, consumed as ``exp(g)`` row-broadcast over the
value axis of the ``[B, H, K, V]`` fp32 state. Gate conditioning
(``f_b`` row + ``dt_bias`` + ``A_log`` + ``lower_bound``·sigmoid) folds
inside the task; q/k L2-normalize inside (q carries the 1/sqrt(D) scale).

Test doctrine (hard rule learned from #90): ALL arithmetic is pinned by
fp64 numpy mirrors ported verbatim from floe
``glm5_arch.Glm53LinearAttention``/``Glm53ForgetGate``/``_kda_recurrent``
— the suite passes in the bare environment (no torch, no floe, zero
collection errors, zero unexpected skips). The real-floe oracle and the
Triton device template are import-gated individual tests, reported as
attested-not-verified when the gate skips.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from vkernels.compiler.capture import CaptureError, RecordingBackend  # noqa: E402
from vkernels.compiler.legality import check_graph  # noqa: E402
from vkernels.compiler.lowerings import lower_graph  # noqa: E402
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import compute_hazards  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402


# ===========================================================================
# fp64 numpy mirror of floe Glm53LinearAttention seq==1 (verbatim port of
# glm5_arch.py: _project_qkv + _conv_decode + Glm53ForgetGate.forward +
# _kda_recurrent + Glm53RMSNormGated.forward + o_proj). Weights in floe's
# nn.Linear [out, in] convention.
# ===========================================================================

B = 2
H = 3  # linear_num_heads (tiny)
D = 8  # linear_head_dim; KDA heads are square: K = V = D
HID = 16
KC = 4  # linear_conv_kernel_dim
CONV_DIM = 3 * H * D
LB = -5.0  # linear_lower_bound (Glm53Config default)
_EPS_L2 = 1e-6

STORAGE_CONV = 801
STORAGE_SSM = 802
STORAGE_HID = 803
STORAGE_WQKV = 804
STORAGE_WFIR = 805
STORAGE_WFA = 806
STORAGE_WFB = 807
STORAGE_WB = 808
STORAGE_DT = 809
STORAGE_ALOG = 810
STORAGE_WGA = 811
STORAGE_WGB = 812
STORAGE_GAMMA = 813
STORAGE_WOUT = 814


def _random_params(rng):
    """floe-convention weights ([out, in] linears, [C, Kc] FIR, [H, D] dt)."""
    return {
        "w_qkv": rng.standard_normal((CONV_DIM, HID)) * 0.2,  # cat(q, k, v)
        "w_fir": rng.standard_normal((CONV_DIM, KC)) * 0.5,  # conv1d.weight.squeeze(1)
        "w_fa": rng.standard_normal((D, HID)) * 0.3,  # forget_gate.f_a_proj
        "w_fb": rng.standard_normal((H * D, D)) * 0.3,  # forget_gate.f_b_proj
        "dt_bias": rng.standard_normal((H, D)) * 0.3,
        "A_log": rng.standard_normal(H) * 0.2,
        "w_b": rng.standard_normal((H, HID)) * 0.3,  # b_proj
        "w_ga": rng.standard_normal((D, HID)) * 0.3,  # g_a_proj
        "w_gb": rng.standard_normal((H * D, D)) * 0.3,  # g_b_proj
        "gamma": np.abs(rng.standard_normal(D)) * 0.5 + 0.5,  # o_norm.weight
        "w_out": rng.standard_normal((HID, H * D)) * 0.2,  # o_proj
    }


def _mirror_forget_gate(x, P, lower_bound):
    """Glm53ForgetGate.forward seq==1: log-space per-(head, k-dim) gate."""
    frow = (x @ P["w_fa"].T) @ P["w_fb"].T  # [B, H*D]
    x_dt = frow.reshape(len(x), H, D) + P["dt_bias"][None]
    A = np.exp(P["A_log"])[:, None]  # [H, 1]
    if lower_bound is not None:
        return lower_bound / (1.0 + np.exp(-(A * x_dt)))
    sp = np.where(x_dt <= 20.0, np.log(1.0 + np.exp(x_dt)), x_dt)
    return -A * sp


def _mirror_kda_recurrent(q, k, v, g, beta, ssm_state, lower_bound):
    """_kda_recurrent seq==1 (S==1): element-wise exp(g) row decay, delta
    rule, plain readout. q/k must be pre-normalized; scale inside (floe
    applies 1/sqrt(k_dim) to query inside the core). Returns (o, new_state).
    """
    scale = D ** -0.5
    new_state = np.empty_like(ssm_state)
    out = np.empty((len(q), H, D))
    for bb in range(len(q)):
        for h in range(H):
            qn = q[bb, h] / np.sqrt((q[bb, h] ** 2).sum() + _EPS_L2) * scale
            kn = k[bb, h] / np.sqrt((k[bb, h] ** 2).sum() + _EPS_L2)
            s = ssm_state[bb, h] * np.exp(g[bb, h])[:, None]
            kv_mem = (s * kn[:, None]).sum(axis=0)
            s = s + kn[:, None] * ((beta[bb, h] * (v[bb, h] - kv_mem))[None, :])
            new_state[bb, h] = s
            out[bb, h] = (s * qn[:, None]).sum(axis=0)
    return out, new_state


def _mirror_layer(x, P, conv_state, ssm_state, lower_bound=LB):
    """Full Glm53LinearAttention seq==1 in fp64. conv_state [B, Kc-1, C]
    time-major (#89 pool convention), ssm_state [B, H, D, D] fp64.
    Returns (y [B, HID], new_conv, new_ssm)."""
    qkv = x @ P["w_qkv"].T  # [B, C] packed qkv projection
    # conv decode: window = cat(state[b], x[b]) over the time axis; FIR +
    # silu (#89 reference semantics == floe _conv_decode under layout)
    window = np.concatenate([conv_state, qkv[:, None, :]], axis=1)  # [B, Kc, C]
    fir = np.einsum("bjc,cj->bc", window, P["w_fir"])
    mixed = fir / (1.0 + np.exp(-fir))  # silu
    new_conv = np.concatenate([conv_state[:, 1:, :], qkv[:, None, :]], axis=1)
    q = mixed[:, 0 : H * D].reshape(-1, H, D)
    k = mixed[:, H * D : 2 * H * D].reshape(-1, H, D)
    v = mixed[:, 2 * H * D :].reshape(-1, H, D)
    # floe conditions q/k via _l2norm BEFORE the core; scale rides on q
    scale = D ** -0.5
    q = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + _EPS_L2) * scale
    k = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + _EPS_L2)
    g = _mirror_forget_gate(x, P, lower_bound)  # [B, H, D] log-space
    beta = 1.0 / (1.0 + np.exp(-(x @ P["w_b"].T)))  # [B, H]
    o, new_ssm = _mirror_kda_recurrent(q, k, v, g, beta, ssm_state, lower_bound)
    # Glm53RMSNormGated: norm(o) * gamma * sigmoid(gate)
    gate = ((x @ P["w_ga"].T) @ P["w_gb"].T).reshape(-1, H, D)
    var = (o ** 2).mean(axis=-1, keepdims=True)
    on = o / np.sqrt(var + _EPS_L2) * P["gamma"] * (1.0 / (1.0 + np.exp(-gate)))
    y = on.reshape(len(x), H * D) @ P["w_out"].T
    return y, new_conv, new_ssm


# ===========================================================================
# Compiler plumbing (capture -> lower -> schedule -> reference executor),
# persistent external pools so a decode walk evolves them in place.
# ===========================================================================

def _ext(rec, name, shape, sid):
    return rec.external_tensor(name, shape, storage_id=sid)


def _capture_kda(rec, layer: int = 0, lower_bound=LB):
    state = _ext(rec, "kda_state", (B, H, D, D), STORAGE_SSM)
    q = _ext(rec, "kda_q", (B, H, D), 820)
    k = _ext(rec, "kda_k", (B, H, D), 821)
    v = _ext(rec, "kda_v", (B, H, D), 822)
    f = _ext(rec, "kda_f", (B, H, D), 823)
    b = _ext(rec, "kda_b", (B, H), 824)
    dt_bias = _ext(rec, "dt_bias", (H, D), 825)
    a_log = _ext(rec, "A_log", (H,), 826)
    out, state_post = rec.kda_delta(
        state, q, k, v, f, b, dt_bias, a_log,
        layer=layer, scale=D ** -0.5, lower_bound=lower_bound,
    )
    return {"state": state, "q": q, "k": k, "v": v, "f": f, "b": b,
            "dt_bias": dt_bias, "A_log": a_log, "out": out, "state_post": state_post}


def _build_executor(state_init, workers, feeds=None, lower_bound=LB):
    """Capture a single kda_delta op; return (executor, handles)."""
    feeds = feeds or {}
    recorder = RecordingBackend()  # position-independent: no define_position
    h = _capture_kda(recorder, lower_bound=lower_bound)
    graph = recorder.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)
    state_arr = state_init.astype(np.float32).reshape(-1).copy()
    storage_arrays = {STORAGE_SSM: state_arr}
    for name, sym in (("q", h["q"]), ("k", h["k"]), ("v", h["v"]), ("f", h["f"]),
                      ("b", h["b"]), ("dt_bias", h["dt_bias"]), ("A_log", h["A_log"])):
        tv = sym.value
        arr = np.full(int(np.prod(tv.shape)), np.nan, dtype=np.float32)
        if name in feeds:
            arr[:] = feeds[name].reshape(-1)
        storage_arrays[tv.storage_id] = arr
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(schedule, workers=workers, storage_arrays=storage_arrays,
                                 graph=graph, workspace_plan=workspace_plan)
    handles = dict(h)
    handles.update({"graph": graph, "families": families, "storage_arrays": storage_arrays,
                    "workspace_plan": workspace_plan, "schedule": schedule})
    return executor, handles


def _set_feeds(handles, **arrays):
    sa = handles["storage_arrays"]
    for name, arr in arrays.items():
        sa[handles[name].value.storage_id][:] = arr.reshape(-1)


# ===========================================================================
# Capture (recorder contract)
# ===========================================================================

def test_capture_kda_delta_records_contract():
    recorder = RecordingBackend()
    h = _capture_kda(recorder)
    op = recorder.graph.ops[-1]
    assert op.kind == "kda_delta"
    assert op.attributes["layer"] == 0
    assert op.attributes["scale"] == pytest.approx(D ** -0.5)
    assert op.attributes["lower_bound"] == LB
    # position-independent: no decode-position scalar in the contract
    assert all("position" not in str(r) for r in op.read_regions)
    nc = op.numerical_contract
    assert "element-wise" in nc["decay"]
    assert "lower_bound" in nc["gate"] and "softplus" in nc["gate"]
    assert "L2" in nc["qk_norm"]
    assert "read-modify-write" in nc["pool"]
    # out follows v's dtype; the post view shares the pool's storage
    assert h["out"].value.shape == (B, H, D)
    assert h["state_post"].value.storage_id == STORAGE_SSM


@pytest.mark.parametrize("kwargs", [
    {"q_shape": (B, H + 1, D)},
    {"k_shape": (B, H, D + 1)},
    {"v_shape": (B, H, D + 1)},
    {"f_shape": (B, H, D + 1)},
    {"b_shape": (B, H + 1)},
    {"dt_shape": (H, D + 1)},
    {"alog_shape": (H + 1,)},
])
def test_capture_kda_delta_rejects_shape_mismatches(kwargs):
    def _bad(**over):
        rec = RecordingBackend()
        state = _ext(rec, "kda_state", (B, H, D, D), STORAGE_SSM)
        q = _ext(rec, "q", over.get("q_shape", (B, H, D)), 830)
        k = _ext(rec, "k", over.get("k_shape", (B, H, D)), 831)
        v = _ext(rec, "v", over.get("v_shape", (B, H, D)), 832)
        f = _ext(rec, "f", over.get("f_shape", (B, H, D)), 833)
        b = _ext(rec, "b", over.get("b_shape", (B, H)), 834)
        dt = _ext(rec, "dt", over.get("dt_shape", (H, D)), 835)
        al = _ext(rec, "al", over.get("alog_shape", (H,)), 836)
        rec.kda_delta(state, q, k, v, f, b, dt, al, layer=0, scale=1.0, lower_bound=LB)
    with pytest.raises(CaptureError):
        _bad(**kwargs)


def test_capture_kda_delta_rejects_nonsquare_heads():
    rec = RecordingBackend()
    state = _ext(rec, "kda_state", (B, H, D, D + 2), STORAGE_SSM)
    q = _ext(rec, "q", (B, H, D), 830)
    k = _ext(rec, "k", (B, H, D), 831)
    v = _ext(rec, "v", (B, H, D + 2), 832)
    f = _ext(rec, "f", (B, H, D), 833)
    b = _ext(rec, "b", (B, H), 834)
    dt = _ext(rec, "dt", (H, D), 835)
    al = _ext(rec, "al", (H,), 836)
    with pytest.raises(CaptureError, match="square"):
        rec.kda_delta(state, q, k, v, f, b, dt, al, layer=0, scale=1.0, lower_bound=LB)


def test_kda_delta_two_layer_hazards_are_state_ordered():
    """Chained kda_delta ops on one pool: RAW/WAR/WAW on the state storage —
    the ordering is state hazards, not the (absent) decode position."""
    recorder = RecordingBackend()
    _h1 = _capture_kda(recorder, layer=0)
    _h2 = _capture_kda(recorder, layer=1)
    hazards = compute_hazards(recorder.graph.ops)
    pairs = {(h.kind, h.producer, h.consumer) for h in hazards if h.storage_id == STORAGE_SSM}
    assert ("RAW", 0, 1) in pairs and ("WAR", 0, 1) in pairs and ("WAW", 0, 1) in pairs


# ===========================================================================
# Lowering (task decomposition)
# ===========================================================================

def test_lowering_kda_delta_task_decomposition():
    recorder = RecordingBackend()
    h = _capture_kda(recorder)
    families = lower_graph(recorder.graph)
    kda = [f for f in families if f.kind == "kda_delta"]
    assert len(kda) == 1
    fam = kda[0]
    # one task per (batch, head); no group expansion (KDA: one q/k/v head each)
    assert fam.domain.dims == ((B, 1), (H, 1))
    assert fam.scratch_bytes == D * D * 4  # the [K, V] fp32 state slice
    assert fam.params["scale"] == pytest.approx(D ** -0.5)
    assert fam.params["lower_bound"] == LB
    # the state slice is both read and written (RMW), per (b, h) tile
    wr = fam.write_regions((0, 0))
    assert any(r.storage_id == STORAGE_SSM for r in wr)


def test_single_launch_accounting():
    """The kda_delta family is exactly B*H tasks — one launch per decode
    step, no per-k or per-v refinement tasks."""
    recorder = RecordingBackend()
    _capture_kda(recorder)
    families = lower_graph(recorder.graph)
    total = sum(int(np.prod([n for n, _ in f.domain.dims])) for f in families)
    assert total == B * H


# ===========================================================================
# Reference execution vs the fp64 numpy mirror (bare environment)
# ===========================================================================

@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("lower_bound", [LB, None])
def test_reference_kda_delta_walk_matches_numpy_mirror(workers, lower_bound):
    """Multi-step decode walk vs the fp64 mirror: outputs and BOTH evolving
    pools (the compiled graph only persists the ssm pool; the mirror's conv
    pool is exercised in the full-layer test). lower_bound=None covers the
    guarded-softplus gate branch."""
    rng = np.random.default_rng(101)
    P = _random_params(rng)
    ssm0 = (rng.standard_normal((B, H, D, D)) * 0.3).astype(np.float32)
    executor, handles = _build_executor(
        ssm0, workers,
        feeds={"dt_bias": P["dt_bias"].astype(np.float32),
               "A_log": P["A_log"].astype(np.float32)},
        lower_bound=lower_bound,
    )
    sa = handles["storage_arrays"]
    sa[handles["dt_bias"].value.storage_id][:] = P["dt_bias"].reshape(-1)
    sa[handles["A_log"].value.storage_id][:] = P["A_log"].reshape(-1)
    mirror_ssm = ssm0.astype(np.float64)

    workspace_sids = {buf.storage_id: buf.numel for buf in handles["workspace_plan"].buffers}

    def make_exec():
        arrays = dict(sa)
        for sid, numel in workspace_sids.items():
            arrays[sid] = np.full(numel, np.nan, dtype=np.float32)
        return ReferenceExecutor(schedule=handles["schedule"], workers=workers,
                                 storage_arrays=arrays, graph=handles["graph"],
                                 workspace_plan=handles["workspace_plan"])

    for step in range(5):
        q = rng.standard_normal((B, H, D)).astype(np.float32) * 0.7
        k = rng.standard_normal((B, H, D)).astype(np.float32) * 0.7
        v = rng.standard_normal((B, H, D)).astype(np.float32) * 0.7
        f = rng.standard_normal((B, H, D)).astype(np.float32) * 0.5
        b = rng.standard_normal((B, H)).astype(np.float32) * 0.5
        _set_feeds(handles, q=q, k=k, v=v, f=f, b=b)
        executor = make_exec()
        executor.run({})
        got = np.array(executor.tensor(handles["out"].value.name), copy=True)
        assert np.isfinite(got).all(), f"step {step}: NaN canary tripped"
        # fp64 mirror of the same step: q/k fed RAW — the op normalizes
        # inside (floe _l2norm boundary); gate mirrored directly from the
        # f rows (the op consumes f, not x)
        q64, k64, v64 = q.astype(np.float64), k.astype(np.float64), v.astype(np.float64)
        scale = D ** -0.5
        qn = q64 / np.sqrt((q64 ** 2).sum(-1, keepdims=True) + _EPS_L2) * scale
        kn = k64 / np.sqrt((k64 ** 2).sum(-1, keepdims=True) + _EPS_L2)
        x_dt = f.astype(np.float64) + P["dt_bias"][None]
        A = np.exp(P["A_log"])
        if lower_bound is not None:
            g = lower_bound / (1.0 + np.exp(-(A[:, None] * x_dt)))
        else:
            sp = np.where(x_dt <= 20.0, np.log(1.0 + np.exp(x_dt)), x_dt)
            g = -A[:, None] * sp
        o, new_ssm = _mirror_kda_recurrent(qn, kn, v64, g, 1.0 / (1.0 + np.exp(-b.astype(np.float64))),
                                           mirror_ssm, lower_bound)
        np.testing.assert_allclose(got, o, rtol=1e-4, atol=1e-5,
                                   err_msg=f"step {step}: output mismatch")
        compiled_state = sa[STORAGE_SSM].reshape(B, H, D, D).copy()
        np.testing.assert_allclose(compiled_state, new_ssm, rtol=1e-4, atol=1e-5,
                                   err_msg=f"step {step}: state mismatch")
        mirror_ssm = new_ssm


def test_reference_kda_delta_issue_factored_form():
    """The issue's arithmetic, stated in its exact factored form, matches
    the task body: state *= exp(g)[..., :, None]; kv_mem = Σ_k state·k;
    δ = (v − kv_mem)·β; state += k ⊗ δ; o = Σ_k state·q."""
    rng = np.random.default_rng(202)
    s = rng.standard_normal((H, D, D))
    q = rng.standard_normal((H, D))
    k = rng.standard_normal((H, D))
    v = rng.standard_normal((H, D))
    g = -np.abs(rng.standard_normal((H, D)))  # log-space decay
    beta = rng.random(H)
    scale = D ** -0.5
    qn = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + _EPS_L2) * scale
    kn = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + _EPS_L2)
    # issue-factored per-head loop
    out = np.empty((H, D))
    new_state = np.empty_like(s)
    for h in range(H):
        st = s[h] * np.exp(g[h])[:, None]
        kv_mem = (st * kn[h][:, None]).sum(axis=0)
        delta = (v[h] - kv_mem) * beta[h]
        st = st + kn[h][:, None] * delta[None, :]
        new_state[h] = st
        out[h] = (st * qn[h][:, None]).sum(axis=0)
    # vectorized einsum equivalent
    st_v = s * np.exp(g)[:, :, None]
    kv_v = np.einsum("hkv,hk->hv", st_v, kn)
    st_v = st_v + np.einsum("hk,hv->hkv", kn, beta[:, None] * (v - kv_v))
    out_v = np.einsum("hkv,hk->hv", st_v, qn)
    np.testing.assert_allclose(out_v, out, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(st_v, new_state, rtol=1e-12, atol=1e-12)


def test_reference_kda_delta_state_drift_bounds():
    """The issue demands state drift bounds: over an 8-step decode walk the
    compiled fp32 state tracks the fp64 mirror within a per-step bound that
    does not grow (bounded error, not accumulating divergence)."""
    rng = np.random.default_rng(303)
    P = _random_params(rng)
    ssm0 = (rng.standard_normal((B, H, D, D)) * 0.3).astype(np.float32)
    executor, handles = _build_executor(
        ssm0, 3,
        feeds={"dt_bias": P["dt_bias"].astype(np.float32),
               "A_log": P["A_log"].astype(np.float32)},
    )
    sa = handles["storage_arrays"]
    sa[handles["dt_bias"].value.storage_id][:] = P["dt_bias"].reshape(-1)
    sa[handles["A_log"].value.storage_id][:] = P["A_log"].reshape(-1)
    mirror_ssm = ssm0.astype(np.float64)
    workspace_sids = {buf.storage_id: buf.numel for buf in handles["workspace_plan"].buffers}

    def make_exec():
        arrays = dict(sa)
        for sid, numel in workspace_sids.items():
            arrays[sid] = np.full(numel, np.nan, dtype=np.float32)
        return ReferenceExecutor(schedule=handles["schedule"], workers=3,
                                 storage_arrays=arrays, graph=handles["graph"],
                                 workspace_plan=handles["workspace_plan"])

    drifts = []
    for step in range(8):
        feeds = {n: rng.standard_normal(sh).astype(np.float32) * 0.5
                 for n, sh in (("q", (B, H, D)), ("k", (B, H, D)), ("v", (B, H, D)),
                               ("f", (B, H, D)), ("b", (B, H)))}
        _set_feeds(handles, **feeds)
        executor = make_exec()
        executor.run({})
        A = np.exp(P["A_log"])
        x_dt = feeds["f"].astype(np.float64) + P["dt_bias"][None]
        g = LB / (1.0 + np.exp(-(A[:, None] * x_dt)))
        q64 = feeds["q"].astype(np.float64)
        k64 = feeds["k"].astype(np.float64)
        scale = D ** -0.5
        qn = q64 / np.sqrt((q64 ** 2).sum(-1, keepdims=True) + _EPS_L2) * scale
        kn = feeds["k"].astype(np.float64)
        kn = kn / np.sqrt((kn ** 2).sum(-1, keepdims=True) + _EPS_L2)
        _o, mirror_ssm = _mirror_kda_recurrent(qn, kn, feeds["v"].astype(np.float64), g,
                                               1.0 / (1.0 + np.exp(-feeds["b"].astype(np.float64))),
                                               mirror_ssm, LB)
        compiled = sa[STORAGE_SSM].reshape(B, H, D, D)
        drift = float(np.abs(compiled - mirror_ssm).max())
        drifts.append(drift)
        assert drift <= 3e-5, f"step {step}: state drift {drift:.2e} exceeds bound"
    # bounded, not accumulating: the last step's drift stays within the
    # same envelope as the first (no compounding fp32 divergence)
    assert drifts[-1] <= max(drifts[0], 3e-5) * 4 + 1e-6, f"drift grew: {drifts}"


def test_reference_full_kda_layer_matches_numpy_mirror():
    """Full GLM-5.3 KDA decode layer in-graph: packed qkv linear ->
    gdn_conv (#89, conv_dim = 3·qkv_dim) -> split -> kda_delta ->
    rms_norm_gated (#100) -> o_proj, vs the fp64 mirror of the whole
    Glm53LinearAttention seq==1 — both pools evolving in place."""
    rng = np.random.default_rng(404)
    P = _random_params(rng)
    recorder = RecordingBackend()
    conv_state = _ext(recorder, "conv_state", (B, KC - 1, CONV_DIM), STORAGE_CONV)
    ssm_state = _ext(recorder, "kda_state", (B, H, D, D), STORAGE_SSM)
    hid = _ext(recorder, "hidden", (B, HID), STORAGE_HID)
    w_qkv = _ext(recorder, "w_qkv", (HID, CONV_DIM), STORAGE_WQKV)
    w_fir = _ext(recorder, "w_fir", (CONV_DIM, KC), STORAGE_WFIR)
    qkv = recorder.linear(hid, w_qkv, name="qkv_proj")
    conv_out, conv_post = recorder.gdn_conv(conv_state, qkv, w_fir, layer=0)
    heads_flat = recorder.view_of(conv_out, "kda_heads_flat", (B, 3 * H, D))
    q = recorder.narrow(heads_flat, "kda_q", axis=1, start=0, length=H)
    k = recorder.narrow(heads_flat, "kda_k", axis=1, start=H, length=H)
    v = recorder.narrow(heads_flat, "kda_v", axis=1, start=2 * H, length=H)
    # forget-gate projection chain (floe Glm53ForgetGate projections)
    f_a = recorder.linear(hid, _ext(recorder, "w_fa", (HID, D), STORAGE_WFA), name="f_a_proj")
    f_b = recorder.linear(f_a, _ext(recorder, "w_fb", (D, H * D), STORAGE_WFB), name="f_b_proj")
    f_rows = recorder.view_of(f_b, "kda_f", (B, H, D))
    b_logits = recorder.linear(hid, _ext(recorder, "w_b", (HID, H), STORAGE_WB), name="b_proj")
    dt_bias = _ext(recorder, "dt_bias", (H, D), STORAGE_DT)
    a_log = _ext(recorder, "A_log", (H,), STORAGE_ALOG)
    o, ssm_post = recorder.kda_delta(ssm_state, q, k, v, f_rows, b_logits, dt_bias, a_log,
                                     layer=0, scale=D ** -0.5, lower_bound=LB)
    # gated output norm (#100) + o_proj
    g_a = recorder.linear(hid, _ext(recorder, "w_ga", (HID, D), STORAGE_WGA), name="g_a_proj")
    g_b = recorder.linear(g_a, _ext(recorder, "w_gb", (D, H * D), STORAGE_WGB), name="g_b_proj")
    gate = recorder.view_of(g_b, "kda_gate", (B, H, D))
    gamma = _ext(recorder, "gamma", (D,), STORAGE_GAMMA)
    on = recorder.rms_norm_gated(o, gate, gamma, _EPS_L2, name="o_norm")
    on_flat = recorder.view_of(on, "kda_on_flat", (B, H * D))
    y = recorder.linear(on_flat, _ext(recorder, "w_out", (H * D, HID), STORAGE_WOUT), name="out_proj")
    graph = recorder.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    # both pools RMW'd once per step
    assert graph.storage_versions[STORAGE_CONV] == 1 and graph.storage_versions[STORAGE_SSM] == 1

    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=3)
    workspace_plan, _ = plan_memory(graph, schedule, families)

    conv0 = (rng.standard_normal((B, KC - 1, CONV_DIM)) * 0.05).astype(np.float32)
    ssm0 = (rng.standard_normal((B, H, D, D)) * 0.3).astype(np.float32)
    sa = {
        STORAGE_CONV: conv0.reshape(-1).copy(),
        STORAGE_SSM: ssm0.reshape(-1).copy(),
        STORAGE_HID: np.full(B * HID, np.nan, dtype=np.float32),
        STORAGE_WQKV: P["w_qkv"].T.reshape(-1),
        STORAGE_WFIR: P["w_fir"].reshape(-1),
        STORAGE_WFA: P["w_fa"].T.reshape(-1),
        STORAGE_WFB: P["w_fb"].T.reshape(-1),
        STORAGE_WB: P["w_b"].T.reshape(-1),
        STORAGE_DT: P["dt_bias"].reshape(-1),
        STORAGE_ALOG: P["A_log"].reshape(-1),
        STORAGE_WGA: P["w_ga"].T.reshape(-1),
        STORAGE_WGB: P["w_gb"].T.reshape(-1),
        STORAGE_GAMMA: P["gamma"].reshape(-1),
        STORAGE_WOUT: P["w_out"].T.reshape(-1),
    }
    # single-invocation canaries: rebuild the executor per step over the
    # persistent external arrays (#90's per-step re-arm pattern)
    workspace_sids = {buf.storage_id: buf.numel for buf in workspace_plan.buffers}

    def make_exec():
        arrays = dict(sa)
        for sid, numel in workspace_sids.items():
            arrays[sid] = np.full(numel, np.nan, dtype=np.float32)
        return ReferenceExecutor(schedule=schedule, workers=3, storage_arrays=arrays,
                                 graph=graph, workspace_plan=workspace_plan)

    mirror_conv = conv0.astype(np.float64)
    mirror_ssm = ssm0.astype(np.float64)
    for step in range(4):
        hid_np = rng.standard_normal((B, HID)).astype(np.float32) * 0.5
        sa[STORAGE_HID][:] = hid_np.reshape(-1)
        executor = make_exec()
        executor.run({})
        got = np.array(executor.tensor(y.value.name), copy=True)
        assert np.isfinite(got).all(), f"step {step}: NaN canary tripped"
        want, mirror_conv, mirror_ssm = _mirror_layer(hid_np.astype(np.float64), P,
                                                      mirror_conv, mirror_ssm)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4,
                                   err_msg=f"step {step}: layer output mismatch")
    # both pools equal the mirror's final states (in-place RMW chains)
    np.testing.assert_allclose(sa[STORAGE_SSM].reshape(B, H, D, D), mirror_ssm,
                               rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(sa[STORAGE_CONV].reshape(B, KC - 1, CONV_DIM), mirror_conv,
                               rtol=1e-4, atol=1e-4)


# ===========================================================================
# Real-floe oracle + Triton device template — import-gated (attested when
# the gate skips; never at module level so the bare suite runs clean).
# ===========================================================================

def _floe_kda():
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
    try:
        torch = pytest.importorskip("torch")
        from floe.engine.runner.models.glm5.glm5_config import Glm53Config
        from floe.engine.runner.models.glm5 import glm5_arch
        from floe.engine.runner.models.glm5.glm5_arch import Glm53LinearAttention
    except ModuleNotFoundError as exc:
        pytest.skip(f"floe glm5 oracle unavailable: {exc}")
    cfg = Glm53Config.tiny()
    torch.manual_seed(101)
    la = Glm53LinearAttention(cfg, layer_idx=0)
    return torch, la, cfg, glm5_arch


def test_reference_kda_full_layer_matches_floe_oracle():
    """Attested (torch+floe gated): the in-graph layer walk vs the real
    floe Glm53LinearAttention seq==1 forward branch, weights loaded from
    the module, pools evolving in place."""
    torch, la, cfg, glm5_arch = _floe_kda()
    rng = np.random.default_rng(505)
    assert cfg.linear_lower_bound == LB
    recorder = RecordingBackend()
    conv_state = _ext(recorder, "conv_state", (B, KC - 1, CONV_DIM), STORAGE_CONV)
    ssm_state = _ext(recorder, "kda_state", (B, H, D, D), STORAGE_SSM)
    hid = _ext(recorder, "hidden", (B, HID), STORAGE_HID)
    w_qkv = _ext(recorder, "w_qkv", (HID, CONV_DIM), STORAGE_WQKV)
    w_fir = _ext(recorder, "w_fir", (CONV_DIM, KC), STORAGE_WFIR)
    qkv = recorder.linear(hid, w_qkv, name="qkv_proj")
    conv_out, _ = recorder.gdn_conv(conv_state, qkv, w_fir, layer=0)
    heads_flat = recorder.view_of(conv_out, "kda_heads_flat", (B, 3 * H, D))
    q = recorder.narrow(heads_flat, "kda_q", axis=1, start=0, length=H)
    k = recorder.narrow(heads_flat, "kda_k", axis=1, start=H, length=H)
    v = recorder.narrow(heads_flat, "kda_v", axis=1, start=2 * H, length=H)
    f_a = recorder.linear(hid, _ext(recorder, "w_fa", (HID, D), STORAGE_WFA), name="f_a_proj")
    f_b = recorder.linear(f_a, _ext(recorder, "w_fb", (D, H * D), STORAGE_WFB), name="f_b_proj")
    f_rows = recorder.view_of(f_b, "kda_f", (B, H, D))
    b_logits = recorder.linear(hid, _ext(recorder, "w_b", (HID, H), STORAGE_WB), name="b_proj")
    dt_bias = _ext(recorder, "dt_bias", (H, D), STORAGE_DT)
    a_log = _ext(recorder, "A_log", (H,), STORAGE_ALOG)
    o, _ = recorder.kda_delta(ssm_state, q, k, v, f_rows, b_logits, dt_bias, a_log,
                              layer=0, scale=D ** -0.5, lower_bound=LB)
    g_a = recorder.linear(hid, _ext(recorder, "w_ga", (HID, D), STORAGE_WGA), name="g_a_proj")
    g_b = recorder.linear(g_a, _ext(recorder, "w_gb", (D, H * D), STORAGE_WGB), name="g_b_proj")
    gate = recorder.view_of(g_b, "kda_gate", (B, H, D))
    gamma = _ext(recorder, "gamma", (D,), STORAGE_GAMMA)
    on = recorder.rms_norm_gated(o, gate, gamma, cfg.rms_norm_eps, name="o_norm")
    on_flat = recorder.view_of(on, "kda_on_flat", (B, H * D))
    y = recorder.linear(on_flat, _ext(recorder, "w_out", (H * D, HID), STORAGE_WOUT), name="out_proj")
    graph = recorder.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=3)
    workspace_plan, _ = plan_memory(graph, schedule, families)

    conv0 = (rng.standard_normal((B, KC - 1, CONV_DIM)) * 0.05).astype(np.float32)
    ssm0 = (rng.standard_normal((B, H, D, D)) * 0.3).astype(np.float32)
    sa = {
        STORAGE_CONV: conv0.reshape(-1).copy(),
        STORAGE_SSM: ssm0.reshape(-1).copy(),
        STORAGE_HID: np.full(B * HID, np.nan, dtype=np.float32),
        # packed qkv = cat(q_proj, k_proj, v_proj) over the out axis
        STORAGE_WQKV: torch.cat([la.q_proj.weight, la.k_proj.weight, la.v_proj.weight],
                                dim=0).detach().numpy().T.reshape(-1),
        STORAGE_WFIR: la.conv1d.weight.squeeze(1).detach().numpy().reshape(-1),
        STORAGE_WFA: la.forget_gate.f_a_proj.weight.detach().numpy().T.reshape(-1),
        STORAGE_WFB: la.forget_gate.f_b_proj.weight.detach().numpy().T.reshape(-1),
        STORAGE_WB: la.b_proj.weight.detach().numpy().T.reshape(-1),
        STORAGE_DT: la.forget_gate.dt_bias.detach().numpy().reshape(H, D).reshape(-1),
        STORAGE_ALOG: la.forget_gate.A_log.detach().numpy().reshape(-1),
        STORAGE_WGA: la.g_a_proj.weight.detach().numpy().T.reshape(-1),
        STORAGE_WGB: la.g_b_proj.weight.detach().numpy().T.reshape(-1),
        STORAGE_GAMMA: la.o_norm.weight.detach().numpy().reshape(-1),
        STORAGE_WOUT: la.o_proj.weight.detach().numpy().T.reshape(-1),
    }
    workspace_sids = {buf.storage_id: buf.numel for buf in workspace_plan.buffers}

    def make_exec():
        arrays = dict(sa)
        for sid, numel in workspace_sids.items():
            arrays[sid] = np.full(numel, np.nan, dtype=np.float32)
        return ReferenceExecutor(schedule=schedule, workers=3, storage_arrays=arrays,
                                 graph=graph, workspace_plan=workspace_plan)

    # floe oracle state: channel-major conv pool [B, C, Kc-1] + fp32 ssm
    oracle_conv = torch.from_numpy(conv0.copy()).transpose(1, 2).contiguous()  # [B, C, Kc-1]
    oracle_ssm = torch.from_numpy(ssm0.copy())
    for step in range(4):
        hid_np = rng.standard_normal((B, HID)).astype(np.float32) * 0.5
        sa[STORAGE_HID][:] = hid_np.reshape(-1)
        executor = make_exec()
        executor.run({})
        got = np.array(executor.tensor(y.value.name), copy=True)
        assert np.isfinite(got).all(), f"step {step}: NaN canary tripped"
        for bb in range(B):
            x = torch.from_numpy(hid_np[bb]).reshape(1, HID)
            # Glm53LinearAttention.forward seq==1 branch, verbatim
            mixed = torch.cat([la.q_proj(x), la.k_proj(x), la.v_proj(x)], dim=-1)  # [1, C]
            raw_col = mixed  # pre-conv input kept for the state update
            mixed_c = la._conv_decode(mixed.unsqueeze(-1), oracle_conv[bb].unsqueeze(0)).squeeze(-1)  # [1, C]
            oracle_conv[bb] = torch.cat([oracle_conv[bb][:, 1:], raw_col], dim=-1).contiguous()
            q_t, k_t, v_t = torch.split(mixed_c.view(1, 3 * H, D), [H, H, H], dim=1)
            q_t = q_t.squeeze(0)  # [H, D]
            k_t = k_t.squeeze(0)
            v_t = v_t.squeeze(0)
            g = la.forget_gate(x).view(H, D)  # [H, D] log-space
            beta = torch.sigmoid(la.b_proj(x)).view(H)  # [H]
            q_f = glm5_arch._l2norm(q_t.float().unsqueeze(0)).squeeze(0)
            k_f = glm5_arch._l2norm(k_t.float().unsqueeze(0)).squeeze(0)
            core, new_ssm = glm5_arch._kda_recurrent(
                q_f.unsqueeze(0), k_f.unsqueeze(0), v_t.float().unsqueeze(0),
                g.float().unsqueeze(0), beta.float().unsqueeze(0),
                initial_state=oracle_ssm[bb].unsqueeze(0), output_final_state=True,
            )
            oracle_ssm[bb] = new_ssm.squeeze(0)
            gate_t = la.g_b_proj(la.g_a_proj(x)).view(H, D)
            on = la.o_norm(core.squeeze(0), gate_t).reshape(1, -1)
            want = la.o_proj(on).reshape(-1)
            torch.testing.assert_close(torch.from_numpy(got[bb]), want, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(
        torch.from_numpy(sa[STORAGE_SSM].reshape(B, H, D, D).copy()), oracle_ssm, rtol=1e-4, atol=1e-5)


def test_device_t_kda_heads_batched_matches_numpy_mirror():
    """Attested (CUDA+triton gated): the batched device task body vs the
    fp64 numpy mirror on real per-(head, k-dim) gates — the element-wise
    decay is the whole point of the op, so the template must reproduce it
    exactly (in-place state RMW on the device pool)."""
    pytest.importorskip("torch")
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vkernels.compiler.device_triton import _t_kda_heads_batched

    rng = np.random.default_rng(606)
    P = _random_params(rng)
    dev = torch.device("cuda")
    state0 = (rng.standard_normal((B, H, D, D)) * 0.3).astype(np.float32)
    feeds = {n: (rng.standard_normal(sh) * 0.7).astype(np.float32)
             for n, sh in (("q", (B, H, D)), ("k", (B, H, D)), ("v", (B, H, D)),
                           ("f", (B, H, D)), ("b", (B, H)))}
    tensors = {n: torch.from_numpy(a).to(dev) for n, a in feeds.items()}
    state_t = torch.from_numpy(state0).to(dev)
    dt = torch.from_numpy(P["dt_bias"].astype(np.float32)).to(dev)
    alog = torch.from_numpy(P["A_log"].astype(np.float32)).to(dev)
    out = torch.empty(B, H, D, device=dev, dtype=torch.float32)
    # worker=0, P=1: a single program covers every (batch, head) task via
    # the task-striding loop; the compiled path passes worker=pid, P=grid
    # (same direct-launch contract as the #88 rope fix).
    _t_kda_heads_batched[(1,)](
        0, 1,
        tensors["q"], tensors["k"], tensors["v"], tensors["f"], tensors["b"],
        dt, alog, state_t, out, B, H, D, D, scale=D ** -0.5, lower_bound=LB, num_warps=4,
    )
    torch.cuda.synchronize()
    # numpy mirror
    x_dt = feeds["f"].astype(np.float64) + P["dt_bias"][None]
    A = np.exp(P["A_log"])
    g = LB / (1.0 + np.exp(-(A[:, None] * x_dt)))
    q64 = feeds["q"].astype(np.float64)
    k64 = feeds["k"].astype(np.float64)
    scale = D ** -0.5
    qn = q64 / np.sqrt((q64 ** 2).sum(-1, keepdims=True) + _EPS_L2) * scale
    kn = k64 / np.sqrt((k64 ** 2).sum(-1, keepdims=True) + _EPS_L2)
    o, new_ssm = _mirror_kda_recurrent(qn, kn, feeds["v"].astype(np.float64), g,
                                       1.0 / (1.0 + np.exp(-feeds["b"].astype(np.float64))),
                                       state0.astype(np.float64), LB)
    torch.testing.assert_close(out.cpu(), torch.from_numpy(o.astype(np.float32)),
                               rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(state_t.cpu(), torch.from_numpy(new_ssm.astype(np.float32)),
                               rtol=1e-5, atol=1e-6)
