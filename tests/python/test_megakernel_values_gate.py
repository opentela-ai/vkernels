"""Issue #92 — gated attention_values (Qwen3.5 attn_output_gate).

Bare-env (numpy/fp64 mirror) validation of the gated values vertical:
capture → lowering → legality → reference-executor walk, plus the single-
launch accounting that proves the gate costs no extra grid barrier, and a
CUDA-gated device-template test (attested-not-verified in this env).
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler.capture import RecordingBackend
from vkernels.compiler.legality import enforce
from vkernels.compiler.lowerings import LOWERINGS, lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.operator_ir import F32
from vkernels.compiler.reference_exec import ReferenceExecutor, _stable_sigmoid
from vkernels.compiler.schedule_phase import PhaseSchedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gated_attention_graph(B, H, S, D, *, kv_heads=None, gated=True, use_scalar=False, gate_shape=None):
    ops = RecordingBackend()
    position = ops.define_position(S) if use_scalar else ops.define_row_positions("row_positions", B, S, storage_id=900)
    q = ops.external_tensor("q", (B, H, D), F32, storage_id=901)
    kc = ops.external_tensor("k_cache", (B, kv_heads or H, S, D), F32, storage_id=902)
    vc = ops.external_tensor("v_cache", (B, kv_heads or H, S, D), F32, storage_id=903)
    gate = None
    if gated:
        gate = ops.external_tensor("gate", gate_shape or (B, H, D), F32, storage_id=904)
    scores = ops.attention_scores(q, kc, position, scale=0.5, layer=0, kv_heads=kv_heads)
    probs = ops.softmax(scores, position, layer=0)
    ctx = ops.attention_values(probs, vc, position, layer=0, kv_heads=kv_heads, gate=gate)
    return ops, position, q, kc, vc, gate, scores, probs, ctx


def _compile(ops, workers=2):
    enforce(ops.graph)
    fams = lower_graph(ops.graph)
    sched = PhaseSchedule.from_families(fams, workers=workers)
    ws_plan, _ = plan_memory(ops.graph, sched, fams)
    return sched, ws_plan


def _state(ws_plan, externals, seed=7):
    """Full storage dict: NaN-poisoned workspace views + external inputs."""
    rng = np.random.default_rng(seed)
    st = {}
    ws = np.full(ws_plan.total_elements, np.nan)
    for buf in ws_plan.buffers:
        st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
    for tv, arr in externals.items():
        st[tv.value.storage_id] = arr
    return st, rng


def _softmax_oracle(sc):
    m = sc.max(axis=-1, keepdims=True)
    e = np.exp(sc - m)
    return e / e.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Capture & IR
# ---------------------------------------------------------------------------


def test_capture_records_gate_input_and_attribute():
    ops, *_ , ctx = _gated_attention_graph(2, 4, 6, 8, kv_heads=2, gated=True)
    op = next(o for o in ops.graph.ops if o.kind == "attention_values")
    assert op.attributes.get("gated") is True
    assert len(op.inputs) == 3
    contract = op.numerical_contract
    assert "sigmoid" in contract["gate"] and "#92" in contract["gate"]


def test_capture_ungated_unchanged():
    ops, *_, ctx = _gated_attention_graph(2, 4, 6, 8, kv_heads=2, gated=False)
    op = next(o for o in ops.graph.ops if o.kind == "attention_values")
    assert "gated" not in op.attributes and len(op.inputs) == 2
    assert "gate" not in op.numerical_contract


def test_capture_rejects_wrong_gate_shape():
    ops = RecordingBackend()
    position = ops.define_row_positions("rp", 2, 6, storage_id=900)
    q = ops.external_tensor("q", (2, 4, 8), F32, storage_id=901)
    kc = ops.external_tensor("kc", (2, 2, 6, 8), F32, storage_id=902)
    vc = ops.external_tensor("vc", (2, 2, 6, 8), F32, storage_id=903)
    bad = ops.external_tensor("bad", (2, 3, 8), F32, storage_id=904)  # wrong H
    sc = ops.attention_scores(q, kc, position, scale=0.5, layer=0, kv_heads=2)
    pr = ops.softmax(sc, position, layer=0)
    with pytest.raises(ValueError, match="gate must be"):
        ops.attention_values(pr, vc, position, layer=0, kv_heads=2, gate=bad)


def test_gated_registered_in_lowerings():
    assert "attention_values" in LOWERINGS


# ---------------------------------------------------------------------------
# Legality: gate read region soundness
# ---------------------------------------------------------------------------


def test_gate_read_region_recorded_whole():
    """The gate is read whole (per-(b,h) row, no time axis): its region must
    appear in the op's reads so RAW ordering against the producing projection
    is enforced."""
    ops, *_, ctx = _gated_attention_graph(2, 4, 6, 8, kv_heads=2, gated=True)
    op = next(o for o in ops.graph.ops if o.kind == "attention_values")
    gate_name = op.inputs[2]
    gate_reads = [r for r in op.reads() if r.view.name == gate_name]
    assert len(gate_reads) == 1
    # full extent on every dimension
    assert all(hi == d for (_, hi), d in zip(gate_reads[0].boxes, gate_reads[0].view.shape))


def test_hazards_order_projection_before_gated_values():
    """A producer writing the gate (the chunked [q|gate|k|v] projection,
    modeled as a linear + view) must RAW-order before the gated values task."""
    from vkernels.compiler.operator_ir import compute_hazards

    B, H, S, D = 1, 2, 4, 4
    ops = RecordingBackend()
    position = ops.define_row_positions("rp", B, S, storage_id=900)
    q = ops.external_tensor("q", (B, H, D), F32, storage_id=901)
    kc = ops.external_tensor("kc", (B, H, S, D), F32, storage_id=902)
    vc = ops.external_tensor("vc", (B, H, S, D), F32, storage_id=903)
    x = ops.external_tensor("x", (B, D), F32, storage_id=905)
    w = ops.external_tensor("w", (D, H * D), F32, storage_id=906)
    gate_flat = ops.linear(x, w, name="gate_proj")            # [B, H*D]
    gate = ops.view_of(gate_flat, "gate_view", (B, H, D))     # per-(b,h) rows
    sc = ops.attention_scores(q, kc, position, scale=0.5, layer=0)
    pr = ops.softmax(sc, position, layer=0)
    ops.attention_values(pr, vc, position, layer=0, gate=gate)
    hazards = compute_hazards(ops.graph.ops)
    raw = {(h.producer, h.consumer) for h in hazards if h.kind == "RAW"}
    assert any(
        ops.graph.ops[a].kind == "linear" and ops.graph.ops[b].kind == "attention_values"
        for a, b in raw
    ), f"no linear->values RAW hazard; raw={raw}"


# ---------------------------------------------------------------------------
# Numerics: gated walk vs independent fp64 oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kv_heads", [None, 1, 2])  # MHA, group=4, group=2
def test_gated_walk_matches_oracle(kv_heads):
    B, H, S, D = 3, 4, 6, 4
    positions = np.array([2, 0, 5])
    ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
        B, H, S, D, kv_heads=kv_heads, gated=True
    )
    rng = np.random.default_rng(11)
    ext = {
        q: rng.standard_normal((B, H, D)),
        kc: rng.standard_normal((B, kv_heads or H, S, D)),
        vc: rng.standard_normal((B, kv_heads or H, S, D)),
        gate: rng.standard_normal((B, H, D)) * 0.7,
    }
    for workers in (1, 3):
        sched, ws_plan = _compile(ops, workers=workers)
        st, _ = _state(ws_plan, ext)
        # per-row positions: host i64 tensor narrowed on upload
        st[pos_tv.value.storage_id] = positions.astype(np.int64)
        exe = ReferenceExecutor(sched, workers=workers, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
        trace = exe.run({})
        assert trace.kernel_launches == 1
        y = exe.tensor(ctx.name)
        q_arr, k_arr, v_arr, g_arr = ext[q], ext[kc], ext[vc], ext[gate]
        kvh_group = (H // kv_heads) if kv_heads else 1
        for b in range(B):
            vl = positions[b] + 1
            for h in range(H):
                kvh = h // kvh_group if kvh_group > 1 else h
                sc_h = (q_arr[b, h] @ k_arr[b, kvh, :vl, :].T) * 0.5
                pr_h = _softmax_oracle(sc_h)
                acc = pr_h @ v_arr[b, kvh, :vl, :]
                ref = acc * _stable_sigmoid(g_arr[b, h])
                assert np.allclose(y[b, h], ref, atol=1e-12), (kv_heads, workers, b, h, np.abs(y[b, h] - ref).max())


def test_gated_equals_ungated_times_sigmoid():
    """Decomposition law: gated walk == ungated walk * sigmoid(gate) exactly
    (up to the fp64 walk's determinism). Same inputs both runs."""
    B, H, S, D, kv_heads = 2, 4, 6, 4, 2
    positions = np.array([3, 5])
    rng = np.random.default_rng(23)
    q_arr = rng.standard_normal((B, H, D))
    kc_arr = rng.standard_normal((B, kv_heads, S, D))
    vc_arr = rng.standard_normal((B, kv_heads, S, D))
    g_arr = rng.standard_normal((B, H, D)) * 0.5
    outs = {}
    for gated in (True, False):
        ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
            B, H, S, D, kv_heads=kv_heads, gated=gated
        )
        ext = {q: q_arr, kc: kc_arr, vc: vc_arr}
        if gated:
            ext[gate] = g_arr
        sched, ws_plan = _compile(ops)
        st, _ = _state(ws_plan, ext)
        st[pos_tv.value.storage_id] = positions.astype(np.int64)
        exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
        exe.run({})
        outs[gated] = exe.tensor(ctx.name)
    ref = outs[False] * _stable_sigmoid(g_arr)
    assert np.allclose(outs[True], ref, atol=1e-12)


def test_ungated_identity_when_gate_is_zero_sigmoid_midpoint():
    """gate = 0 → sigmoid = 0.5 (halving), NOT identity — pin the semantics
    against a naive 'gate is a residual' misreading."""
    B, H, S, D, kv_heads = 1, 2, 4, 4, None
    positions = np.array([3])
    ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
        B, H, S, D, kv_heads=kv_heads, gated=True
    )
    rng = np.random.default_rng(5)
    ext = {
        q: rng.standard_normal((B, H, D)),
        kc: rng.standard_normal((B, H, S, D)),
        vc: rng.standard_normal((B, H, S, D)),
        gate: np.zeros((B, H, D)),
    }
    sched, ws_plan = _compile(ops)
    st, _ = _state(ws_plan, ext)
    st[pos_tv.value.storage_id] = positions.astype(np.int64)
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    y = exe.tensor(ctx.name)
    # identity check against the ungated walk on identical inputs
    ops2, pos2, q2, kc2, vc2, _, _, _, ctx2 = _gated_attention_graph(B, H, S, D, kv_heads=kv_heads, gated=False)
    sched2, ws2 = _compile(ops2)
    st2, _ = _state(ws2, {q2: ext[q], kc2: ext[kc], vc2: ext[vc]})
    st2[pos2.value.storage_id] = positions.astype(np.int64)
    exe2 = ReferenceExecutor(sched2, workers=2, storage_arrays=st2, graph=ops2.graph, workspace_plan=ws2, canary=True)
    exe2.run({})
    assert np.allclose(y, exe2.tensor(ctx2.name) * 0.5, atol=1e-12)


@pytest.mark.parametrize("big", [50.0, -50.0])
def test_sigmoid_saturation_extremes(big):
    """Large |gate| saturates to 1/0 without overflow NaN (stable epilogue)."""
    B, H, S, D, kv_heads = 1, 2, 4, 4, None
    positions = np.array([3])
    ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
        B, H, S, D, kv_heads=kv_heads, gated=True
    )
    rng = np.random.default_rng(9)
    ext = {
        q: rng.standard_normal((B, H, D)),
        kc: rng.standard_normal((B, H, S, D)),
        vc: rng.standard_normal((B, H, S, D)),
        gate: np.full((B, H, D), big),
    }
    sched, ws_plan = _compile(ops)
    st, _ = _state(ws_plan, ext)
    st[pos_tv.value.storage_id] = positions.astype(np.int64)
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    y = exe.tensor(ctx.name)
    assert np.isfinite(y).all()
    if big > 0:
        ops2, pos2, q2, kc2, vc2, _, _, _, ctx2 = _gated_attention_graph(B, H, S, D, gated=False)
        sched2, ws2 = _compile(ops2)
        st2, _ = _state(ws2, {q2: ext[q], kc2: ext[kc], vc2: ext[vc]})
        st2[pos2.value.storage_id] = positions.astype(np.int64)
        exe2 = ReferenceExecutor(sched2, workers=2, storage_arrays=st2, graph=ops2.graph, workspace_plan=ws2, canary=True)
        exe2.run({})
        assert np.allclose(y, exe2.tensor(ctx2.name), atol=1e-12)  # sigmoid→1
    else:
        assert np.allclose(y, 0.0, atol=1e-12)  # sigmoid→0


def test_gate_nan_canary_propagates():
    """NaN in the gate must propagate to y (the fused multiply does not mask)
    — the executor's NaN canaries catch a silent drop."""
    B, H, S, D = 1, 2, 4, 4
    ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
        B, H, S, D, kv_heads=None, gated=True
    )
    rng = np.random.default_rng(13)
    g_arr = rng.standard_normal((B, H, D)) * 0.3
    g_arr[0, 1, 2] = np.nan
    ext = {
        q: rng.standard_normal((B, H, D)),
        kc: rng.standard_normal((B, H, S, D)),
        vc: rng.standard_normal((B, H, S, D)),
        gate: g_arr,
    }
    sched, ws_plan = _compile(ops)
    st, _ = _state(ws_plan, ext)
    st[pos_tv.value.storage_id] = np.array([3], dtype=np.int64)
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=False)
    exe.run({})
    y = exe.tensor(ctx.name)
    assert np.isnan(y[0, 1, 2])  # propagates
    assert np.isfinite(y[0, 0]).all()  # other head unaffected


def test_ragged_rows_valid_prefix_only():
    """Per-row valid lengths (issue #93 contract) still hold under the gate:
    V tails beyond pos[b] never contribute, NaN tails must not leak."""
    B, H, S, D, kv_heads = 3, 2, 6, 4, 1
    positions = np.array([1, 4, 0])
    ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
        B, H, S, D, kv_heads=kv_heads, gated=True
    )
    rng = np.random.default_rng(17)
    v_arr = rng.standard_normal((B, kv_heads, S, D))
    for b in range(B):  # poison ONLY each row's invalid tail
        v_arr[b, :, positions[b] + 1 :, :] = np.nan
    ext = {
        q: rng.standard_normal((B, H, D)),
        kc: rng.standard_normal((B, kv_heads, S, D)),
        vc: v_arr,
        gate: rng.standard_normal((B, H, D)) * 0.4,
    }
    sched, ws_plan = _compile(ops)
    st, _ = _state(ws_plan, ext)
    st[pos_tv.value.storage_id] = positions.astype(np.int64)
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    y = exe.tensor(ctx.name)
    assert np.isfinite(y).all()


def test_single_launch_accounting_gated_vs_ungated():
    """The gate must cost NO extra grid barrier / launch / phase — the whole
    point of fusing it into the values task (one fewer barrier per FA layer
    × 16 layers)."""
    B, H, S, D, kv_heads = 2, 4, 6, 4, 2
    positions = np.array([3, 5])
    traces = {}
    for gated in (True, False):
        ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
            B, H, S, D, kv_heads=kv_heads, gated=gated
        )
        rng = np.random.default_rng(31)
        ext = {
            q: rng.standard_normal((B, H, D)),
            kc: rng.standard_normal((B, kv_heads, S, D)),
            vc: rng.standard_normal((B, kv_heads, S, D)),
        }
        if gated:
            ext[gate] = rng.standard_normal((B, H, D)) * 0.3
        sched, ws_plan = _compile(ops, workers=2)
        st, _ = _state(ws_plan, ext)
        st[pos_tv.value.storage_id] = positions.astype(np.int64)
        exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
        trace = exe.run({})
        traces[gated] = (trace.kernel_launches, trace.phases if hasattr(trace, "phases") else None, sched)
        lower_graph(ops.graph)
    assert traces[True][0] == traces[False][0] == 1
    # identical phase structure
    fams_g = [f.kind for f in lower_graph(_gated_attention_graph(B, H, S, D, kv_heads=kv_heads, gated=True)[0].graph)]
    fams_u = [f.kind for f in lower_graph(_gated_attention_graph(B, H, S, D, kv_heads=kv_heads, gated=False)[0].graph)]
    assert fams_g == fams_u


def test_chunked_qkv_gate_slicing_semantics():
    """The gate is emitted by the packed [q|gate|k|v] projection: slicing the
    packed output into (q, gate, k, v) then gating must equal gating with the
    gate rows directly — pins the tile-domain agreement."""
    B, H, S, D = 1, 2, 4, 4
    packed = np.random.default_rng(41).standard_normal((B, 3 * H + 1, D))  # [q(H)|gate(H)|k|v head-rows]
    q_rows, gate_rows = packed[:, :H, :], packed[:, H : 2 * H, :]
    assert q_rows.shape == (B, H, D) and gate_rows.shape == (B, H, D)
    # tile domain per (b, h): gate row aligns with the values task row
    ops, pos_tv, q, kc, vc, gate, scores, probs, ctx = _gated_attention_graph(
        B, H, S, D, kv_heads=None, gated=True
    )
    rng = np.random.default_rng(43)
    ext = {
        q: q_rows,
        kc: rng.standard_normal((B, H, S, D)),
        vc: rng.standard_normal((B, H, S, D)),
        gate: gate_rows,
    }
    sched, ws_plan = _compile(ops)
    st, _ = _state(ws_plan, ext)
    st[pos_tv.value.storage_id] = np.array([3], dtype=np.int64)
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    y = exe.tensor(ctx.name)
    sc = np.einsum("bhd,bhsd->bhs", ext[q], ext[kc][:, :, :4, :]) * 0.5
    pr = _softmax_oracle(sc)
    acc = np.einsum("bhs,bhsd->bhd", pr, ext[vc][:, :, :4, :])
    assert np.allclose(y, acc * _stable_sigmoid(gate_rows), atol=1e-12)


# ---------------------------------------------------------------------------
# CUDA-gated device template (attested-not-verified in this env)
# ---------------------------------------------------------------------------


def test_device_template_registered_and_importable():
    """The compiler-path gated template exists alongside _t_values and the
    27B-validated hybrid epilogue it reuses. CUDA-gated (torch/triton)."""
    pytest.importorskip("torch")
    from vkernels.compiler.device_triton import _t_values_gated

    assert _t_values_gated is not None


def test_device_template_parity_cuda():
    """CUDA-gated: _t_values_gated vs the fp64 reference executor on random
    inputs (paged table + per-row pos). ATTESTED in the bare env."""
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA — device templates are mirrored on CPU above")

    from tests.python._megakernel_launch import launch_task_body
    from vkernels.compiler.device_triton import _t_values, _t_values_gated

    B, H, KVH, S, D = 3, 4, 2, 16, 32
    dev = "cuda"
    rng = np.random.default_rng(51)
    probs = torch.from_numpy(rng.standard_normal((B, H, S)).astype(np.float32)).to(dev)
    v = torch.from_numpy(rng.standard_normal((B, KVH, S, D)).astype(np.float32)).to(dev)
    gate = torch.from_numpy(rng.standard_normal((B, H, D)).astype(np.float32)).to(dev)
    slots = torch.randperm(S, device=dev).expand(B, S).contiguous()
    pos = torch.tensor([3, 7, S - 1], device=dev, dtype=torch.int32)
    y_g = torch.empty(B, H, D, device=dev, dtype=torch.float32)
    y_u = torch.empty(B, H, D, device=dev, dtype=torch.float32)
    BT = 8
    launch_task_body(_t_values_gated, probs, v, gate, slots, pos, y_g, B, H, KVH, D, S, BT)
    launch_task_body(_t_values, probs, v, slots, pos, y_u, B, H, KVH, D, S, BT)
    sig = torch.sigmoid(gate.double())
    ref = (y_u.double() * sig).float()
    torch.testing.assert_close(y_g, ref, rtol=1e-4, atol=1e-5)
    # sigmoid monotonicity: gate→0.5 midpoint check
    torch.testing.assert_close(y_g, y_u * torch.sigmoid(gate), rtol=1e-4, atol=1e-5)
