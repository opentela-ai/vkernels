"""Issue #93: per-row valid lengths + per-row positions (ragged decode).

Validation chain (mirrors the issue's Validation section):

* IR: ``ValidLength`` gains a row-tensor form (external i32 ``[B]`` of
  per-row positions, ``len = pos[b] + 1``); ``Region.prefix`` stores the
  row form and ``resolved_boxes`` stays conservative (full extent);
* capture: ``define_row_positions`` registers the external; every
  position consumer (attention scores / softmax / values / cache_append /
  rope / embedding) accepts it and records ``position_form="row"``;
* legality: per-row position tensors are scoped exactly like the scalar
  ``p`` — only POSITION_CONSUMERS may read them;
* reference_exec: per-row masked bodies (scores / softmax / values /
  append / rope table row) validated against an independent per-row eager
  oracle on a **mixed-length batch with NaN cache tails** — any read
  beyond a row's own valid length leaks a NaN into the output and fails
  the comparison, so the suite doubles as the §5.3 NaN-tail canary;
* scalar regression: the same graph captured with the symbolic scalar
  ``p`` still matches the same oracle (shared-mask semantics).
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler.capture import CaptureError, RecordingBackend, SymbolicTensor
from vkernels.compiler.legality import LegalityError, check_graph, enforce
from vkernels.compiler.lowerings import lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.operator_ir import F32, I32, Region, TensorValue, ValidLength
from vkernels.compiler.reference_exec import ReferenceExecutor
from vkernels.compiler.schedule_phase import PhaseSchedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_attention_graph(B, H, S, D, *, kv_heads=None, use_scalar=False):
    """Attention chain (scores -> softmax -> values) captured either with
    the shared scalar ``p`` (use_scalar) or with a per-row positions tensor."""
    ops = RecordingBackend()
    position = ops.define_position(S) if use_scalar else ops.define_row_positions("row_positions", B, S, storage_id=900)
    q = ops.external_tensor("q", (B, H, D), F32, storage_id=901)
    kc = ops.external_tensor("k_cache", (B, H, S, D), F32, storage_id=902)
    vc = ops.external_tensor("v_cache", (B, H, S, D), F32, storage_id=903)
    scores = ops.attention_scores(q, kc, position, scale=0.5, layer=0, kv_heads=kv_heads)
    probs = ops.softmax(scores, position, layer=0)
    ctx = ops.attention_values(probs, vc, position, layer=0, kv_heads=kv_heads)
    return ops, position, q, kc, vc, scores, probs, ctx


def _compile(ops, workers=2):
    enforce(ops.graph)
    fams = lower_graph(ops.graph)
    sched = PhaseSchedule.from_families(fams, workers=workers)
    ws_plan, _ = plan_memory(ops.graph, sched, fams)
    return sched, ws_plan


def _eager_attention_row(q_row, k_row, v_row, valid_len, scale):
    """Independent per-row eager oracle (fp64, NaN-tail aware).

    q_row: (H, D); k_row/v_row: (S, D) for this row's kv head."""
    sc = (q_row @ k_row[:valid_len, :].T.astype(np.float64)) * scale
    pr = np.exp(sc - sc.max(axis=-1, keepdims=True))
    pr /= pr.sum(axis=-1, keepdims=True)
    return pr @ v_row[:valid_len, :].astype(np.float64), pr


# ---------------------------------------------------------------------------
# IR: ValidLength row form
# ---------------------------------------------------------------------------


def test_valid_length_row_forms():
    vl = ValidLength.from_positions("row_pos")
    assert vl.is_row and vl.mode == "positions" and vl.expr is None
    assert str(vl) == "row:row_pos[pos[b]+1]"
    vl2 = ValidLength.from_lengths("cache_len")
    assert vl2.is_row and vl2.mode == "lengths"
    with pytest.raises(ValueError, match="per row"):
        vl.resolve({"p": 3})
    with pytest.raises(ValueError, match="row_tensor"):
        ValidLength(expr="p+1", row_tensor="row_pos")
    with pytest.raises(ValueError, match="mode"):
        ValidLength(row_tensor="row_pos", mode="bogus")
    with pytest.raises(ValueError):
        ValidLength()  # neither form


def test_valid_length_scalar_forms_unchanged():
    assert ValidLength(4).resolve({}) == 4
    assert ValidLength("p+1").resolve({"p": 3}) == 4
    assert not ValidLength("p+1").is_row


def test_region_prefix_row_form_is_conservative():
    tv = TensorValue(1, "kv", (2, 3, 8, 4), F32, (96, 32, 4, 1), 0)
    row = Region.prefix(tv, axis=2, valid=ValidLength.from_positions("row_pos"))
    assert row.boxes[2][1] is not None and isinstance(row.boxes[2][1], ValidLength)
    # Conservative: full extent regardless of scalars (same soundness as
    # the symbolic str form).
    assert row.resolved_boxes({})[2] == (0, 8)
    assert row.resolved_boxes({"p": 3})[2] == (0, 8)
    scal = Region.prefix(tv, axis=2, valid=ValidLength("p+1"))
    assert scal.resolved_boxes({"p": 3})[2] == (0, 4)


# ---------------------------------------------------------------------------
# Capture: define_row_positions + op plumbing
# ---------------------------------------------------------------------------


def test_capture_row_position_registration():
    ops = RecordingBackend()
    row_pos = ops.define_row_positions("row_positions", 3, 64, storage_id=900)
    assert isinstance(row_pos, SymbolicTensor)
    assert row_pos.value.shape == (3,) and row_pos.value.dtype == I32
    assert "row_positions" in ops.graph.row_position_tensors
    # Re-registration returns the same handle.
    again = ops.define_row_positions("row_positions", 3, 64, storage_id=900)
    assert again is row_pos
    with pytest.raises(CaptureError):
        ops.define_row_positions("bad", 0, 64, storage_id=901)


def test_capture_position_form_recorded_row_vs_scalar():
    ops_row, _, _, _, _, scores, probs, ctx = _row_attention_graph(2, 2, 8, 4)
    for op in ops_row.graph.ops:
        assert op.attributes["position_form"] == "row"
    # The softmax read region carries the row-form ValidLength object.
    softmax_read = ops_row.graph.ops[1].read_regions[0]
    assert isinstance(softmax_read.boxes[2][1], ValidLength)
    assert softmax_read.boxes[2][1].is_row
    ops_sc, _, _, _, _, scores_s, probs_s, ctx_s = _row_attention_graph(2, 2, 8, 4, use_scalar=True)
    for op in ops_sc.graph.ops:
        assert op.attributes["position_form"] == "scalar"
    scal_read = ops_sc.graph.ops[1].read_regions[0]
    assert scal_read.boxes[2][1] == "p+1"


def test_capture_rejects_unregistered_row_position():
    ops = RecordingBackend()
    ops.define_position(8)
    q = ops.external_tensor("q", (2, 2, 4), F32, storage_id=901)
    kc = ops.external_tensor("k_cache", (2, 2, 8, 4), F32, storage_id=902)
    stranger = ops.external_tensor("stranger", (2,), I32, storage_id=903)
    with pytest.raises(CaptureError, match="define_row_positions"):
        ops.attention_scores(q, kc, stranger, scale=0.5, layer=0)
    with pytest.raises(CaptureError, match="symbolic decode position"):
        ops.attention_scores(q, kc, 3, scale=0.5, layer=0)


# ---------------------------------------------------------------------------
# Legality: row-position tensors scoped like the scalar p
# ---------------------------------------------------------------------------


def test_legality_row_position_tensor_only_position_consumers():
    ops, row_pos, *_ = _row_attention_graph(2, 2, 8, 4)
    # The registered consumers pass.
    diags = check_graph(ops.graph)
    assert not [d for d in diags if d.severity == "error"]
    # A non-consumer op that references the row tensor is rejected.
    graph = ops.graph
    tv = TensorValue(99, "sneaky", (2, 4), F32, (4, 1), 900)
    graph.tensors["sneaky"] = tv
    graph.record(
        "gemm",
        inputs=("sneaky",),
        outputs=(),
        attributes={"position": "row_positions"},  # consumes the row tensor
        reads=(Region.whole(tv),),
        writes=(Region.whole(tv),),
        source_location="sneaky gemm",
    )
    diags = check_graph(graph)
    errs = [d for d in diags if d.severity == "error" and d.code == "row-position-consumer"]
    assert errs, "a gemm consuming the row-position tensor must be illegal"
    with pytest.raises(LegalityError):
        enforce(graph)


# ---------------------------------------------------------------------------
# Reference executor vs per-row eager oracle (mixed lengths + NaN tails)
# ---------------------------------------------------------------------------


def _mixed_length_state(B, H, S, D, positions, seed=7):
    rng = np.random.default_rng(seed)
    st = {
        900: positions.astype(np.int32),
        901: rng.standard_normal(B * H * D),
        902: rng.standard_normal(B * H * S * D),
        903: rng.standard_normal(B * H * S * D),
    }
    k = st[902].reshape(B, H, S, D)
    v = st[903].reshape(B, H, S, D)
    for b in range(B):
        k[b, :, positions[b] + 1 :, :] = np.nan  # §5.3 NaN tails, per row
        v[b, :, positions[b] + 1 :, :] = np.nan
    return st, rng


@pytest.mark.parametrize("kv_heads", [None, 1])  # MHA and GQA group=2
def test_mixed_length_attention_matches_per_row_oracle(kv_heads):
    B, H, S, D = 3, 2, 6, 4
    positions = np.array([2, 0, 5])
    ops, _, q, kc, vc, scores, probs, ctx = _row_attention_graph(B, H, S, D, kv_heads=kv_heads)
    for workers in (1, 3):
        sched, ws_plan = _compile(ops, workers=workers)
        st, _ = _mixed_length_state(B, H, S, D, positions)
        ws = np.full(ws_plan.total_elements, np.nan)
        for buf in ws_plan.buffers:
            st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
        exe = ReferenceExecutor(sched, workers=workers, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
        trace = exe.run({})  # no scalar p: lengths resolve per row
        assert trace.kernel_launches == 1
        q_arr = st[901].reshape(B, H, D)
        k_arr = st[902].reshape(B, H, S, D)
        v_arr = st[903].reshape(B, H, S, D)
        ctx_run = exe.tensor(ctx.name)
        probs_run = exe.tensor(probs.name)
        kvh_group = (H // kv_heads) if kv_heads else 1
        for b in range(B):
            vl = positions[b] + 1
            for h in range(H):
                kvh = h // kvh_group if kvh_group > 1 else h
                ref, pr = _eager_attention_row(q_arr[b, h], k_arr[b, kvh], v_arr[b, kvh], vl, 0.5)
                assert np.allclose(ctx_run[b, h], ref, atol=1e-12), (workers, b, h, np.abs(ctx_run[b, h] - ref).max())
                # softmax: valid prefix matches, invalid tail is EXACT zero
                assert np.allclose(probs_run[b, h, :vl], pr, atol=1e-12)
                assert (probs_run[b, h, vl:] == 0.0).all()
        # No NaN may leak anywhere in the outputs (any cross-row tail read
        # would have propagated NaN from a neighbouring row's tail).
        assert np.isfinite(ctx_run).all()
        assert np.isfinite(probs_run).all()


def test_scalar_path_regression_same_oracle():
    """The shared-scalar capture (pre-#93 form) still matches the oracle at
    a uniform position — guards the refactor against scalar drift."""
    B, H, S, D = 2, 2, 6, 4
    p = 3
    ops, position, q, kc, vc, scores, probs, ctx = _row_attention_graph(B, H, S, D, use_scalar=True)
    sched, ws_plan = _compile(ops, workers=2)
    st = {
        901: np.random.default_rng(3).standard_normal(B * H * D),
        902: np.random.default_rng(4).standard_normal(B * H * S * D),
        903: np.random.default_rng(5).standard_normal(B * H * S * D),
    }
    ws = np.full(ws_plan.total_elements, np.nan)
    for buf in ws_plan.buffers:
        st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({"p": p})
    q_arr = st[901].reshape(B, H, D)
    k_arr = st[902].reshape(B, H, S, D)
    v_arr = st[903].reshape(B, H, S, D)
    ctx_run = exe.tensor(ctx.name)
    for b in range(B):
        for h in range(H):
            ref, _ = _eager_attention_row(q_arr[b, h], k_arr[b, h], v_arr[b, h], p + 1, 0.5)
            assert np.allclose(ctx_run[b, h], ref, atol=1e-12)


def test_shared_mask_would_read_nan_tail():
    """The canary the issue demands: rows at DIFFERENT lengths, NaN tails.
    If any body used a shared batch-wide mask (max position), it would read
    a shorter row's NaN tail and produce NaN — the assert below fires."""
    B, H, S, D = 3, 2, 8, 4
    positions = np.array([0, 3, 7])  # max is 7; rows 0/1 have NaN tails
    ops, _, _, _, _, _, _, ctx = _row_attention_graph(B, H, S, D)
    sched, ws_plan = _compile(ops, workers=2)
    st, _ = _mixed_length_state(B, H, S, D, positions, seed=11)
    ws = np.full(ws_plan.total_elements, np.nan)
    for buf in ws_plan.buffers:
        st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    assert np.isfinite(exe.tensor(ctx.name)).all()


# ---------------------------------------------------------------------------
# Per-row cache append + rope table rows
# ---------------------------------------------------------------------------


def test_cache_append_per_row_slots():
    B, H, S, D = 3, 2, 8, 4
    ops = RecordingBackend()
    position = ops.define_row_positions("row_positions", B, S, storage_id=900)
    kc = ops.external_tensor("k_cache", (B, H, S, D), F32, storage_id=902)
    vc = ops.external_tensor("v_cache", (B, H, S, D), F32, storage_id=903)
    k_new = ops.external_tensor("k_new", (B, H, D), F32, storage_id=904)
    v_new = ops.external_tensor("v_new", (B, H, D), F32, storage_id=905)
    k_post, v_post = ops.cache_append(kc, vc, k_new, v_new, position, layer=0)
    # Post-append views carry the per-row valid length.
    assert k_post.value.valid_length.is_row and k_post.value.valid_length.row_tensor == "row_positions"  # type: ignore[union-attr]
    sched, ws_plan = _compile(ops, workers=2)
    positions = np.array([1, 4, 0])
    st = {
        900: positions.astype(np.int32),
        902: np.full(B * H * S * D, np.nan),
        903: np.full(B * H * S * D, np.nan),
        904: np.random.default_rng(1).standard_normal(B * H * D),
        905: np.random.default_rng(2).standard_normal(B * H * D),
    }
    ws = np.full(ws_plan.total_elements, np.nan)
    for buf in ws_plan.buffers:
        st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    k_arr = st[902].reshape(B, H, S, D)
    v_arr = st[903].reshape(B, H, S, D)
    kn = st[904].reshape(B, H, D)
    vn = st[905].reshape(B, H, D)
    for b in range(B):
        p = positions[b]
        for h in range(H):
            assert np.allclose(k_arr[b, h, p], kn[b, h], atol=1e-12)
            assert np.allclose(v_arr[b, h, p], vn[b, h], atol=1e-12)
        # Every other slot untouched: still NaN (per-row single-slot write).
        mask = np.ones(S, dtype=bool)
        mask[p] = False
        assert np.isnan(k_arr[b, :, mask, :]).all() and np.isnan(v_arr[b, :, mask, :]).all()


def test_rope_uses_each_rows_own_table_row():
    B, H, D = 3, 2, 4
    S = 8
    ops = RecordingBackend()
    position = ops.define_row_positions("row_positions", B, S, storage_id=900)
    x = ops.external_tensor("x", (B, H, D), F32, storage_id=901)
    cos_t = ops.external_tensor("cos_table", (S, D), F32, storage_id=906)
    sin_t = ops.external_tensor("sin_table", (S, D), F32, storage_id=907)
    out = ops.rope(x, cos_t, sin_t, position, layer=0, which="q")
    sched, ws_plan = _compile(ops, workers=2)
    positions = np.array([0, 2, 6])
    rng = np.random.default_rng(9)
    st = {
        900: positions.astype(np.int32),
        901: rng.standard_normal(B * H * D),
        906: rng.standard_normal(S * D),
        907: rng.standard_normal(S * D),
    }
    ws = np.full(ws_plan.total_elements, np.nan)
    for buf in ws_plan.buffers:
        st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
    exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
    exe.run({})
    x_arr = st[901].reshape(B, H, D)
    cos_arr = st[906].reshape(S, D)
    sin_arr = st[907].reshape(S, D)
    out_run = exe.tensor(out.name)
    for b in range(B):
        row = x_arr[b].astype(np.float64)
        half = D // 2
        rotated = np.concatenate((-row[:, half:], row[:, :half]), axis=-1)
        ref = row * cos_arr[positions[b]].astype(np.float64) + rotated * sin_arr[positions[b]].astype(np.float64)
        assert np.allclose(out_run[b], ref, atol=1e-12), (b, np.abs(out_run[b] - ref).max())


# ---------------------------------------------------------------------------
# Two-step ragged decode: append + attention with per-row position advance
# ---------------------------------------------------------------------------


def test_two_step_ragged_decode_matches_oracle():
    """Rows advance their own positions between steps; the compiled
    schedule (re-captured per step, positions a runtime tensor) must track
    each row independently through append + full attention chain."""
    B, H, S, D = 3, 2, 8, 4
    rng = np.random.default_rng(21)
    cache_k = np.full((B, H, S, D), np.nan)
    cache_v = np.full((B, H, S, D), np.nan)
    positions = np.array([1, 3, 4])
    q_arr = rng.standard_normal((B, H, D))
    # Prior decode history: slots [0, pos[b]] hold real values (the
    # attention-only graph assumes the current slot is already written),
    # tails NaN.
    for b in range(B):
        cache_k[b, :, : positions[b] + 1, :] = rng.standard_normal((H, positions[b] + 1, D))
        cache_v[b, :, : positions[b] + 1, :] = rng.standard_normal((H, positions[b] + 1, D))

    for step in range(2):
        ops, _, _, _, _, _, _, ctx = _row_attention_graph(B, H, S, D)
        sched, ws_plan = _compile(ops, workers=2)
        st = {
            900: positions.astype(np.int32),
            901: q_arr.reshape(-1),
            902: cache_k.reshape(-1),
            903: cache_v.reshape(-1),
        }
        ws = np.full(ws_plan.total_elements, np.nan)
        for buf in ws_plan.buffers:
            st[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]
        exe = ReferenceExecutor(sched, workers=2, storage_arrays=st, graph=ops.graph, workspace_plan=ws_plan, canary=True)
        exe.run({})
        ctx_run = exe.tensor(ctx.name)
        for b in range(B):
            vl = positions[b] + 1
            for h in range(H):
                ref, _ = _eager_attention_row(q_arr[b, h], cache_k[b, h], cache_v[b, h], vl, 0.5)
                assert np.allclose(ctx_run[b, h], ref, atol=1e-12), (step, b, h)
        assert np.isfinite(ctx_run).all()
        # Ragged advance: each row moves at its own pace (floe decode).
        positions = positions + 1
        for b in range(B):
            cache_k[b, :, positions[b], :] = rng.standard_normal((H, D))
            cache_v[b, :, positions[b], :] = rng.standard_normal((H, D))
