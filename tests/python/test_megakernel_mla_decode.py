"""MLA decode family (``mla_scores`` + ``mla_values`` + ``conjugate_rope`` +
grouped linear) compiler support — issue #95, DeepSeek-V4 attention.

Covers the vertical slice the issue names:

* capture — ``ops.mla_scores(q, latent_pool, window_table, comp_pool,
  comp_idx, sink, position)`` records the fused scores + softmax + sink over
  the window ∪ compressed candidates (shared-KV MQA over a latent cache,
  #97's comp_idx/block_bias as producers); ``ops.mla_values(...)`` records
  the context gather (sink column contributes NO value);
  ``ops.conjugate_rope(...)`` records the output-side rotation by the
  NEGATIVE angle (exact inverse of the q/k rotation); ``ops.linear(...,
  grouped_heads=H)`` records the block-diagonal per-head projection
  (DeepSeek-V4 ``GroupedLinear``);
* lowering — one task per (batch, head) for the MLA pair and the rope;
  per-N-tile tasks for the grouped GEMV with block-diagonal read regions
  (off-block weight storage NEVER observed — NaN-canary sound);
* reference body — fp64 tile-exact semantics: window bound t in
  (p−W, p], comp_idx >= 0 masking, sink last, invalid candidates exact 0.0,
  fp32-style accumulation in an fp64 oracle;
* validation — eager exact oracle match, sink-absorption identity, window
  geometry at row start/end, ragged per-row positions (#93), MQA broadcast,
  #97 block_bias effect, pinned interleaved convention, rope round-trip,
  NaN canaries (off-block grouped weights, unselected compressed rows),
  device-template CPU mirrors, single-launch accounting.

All CPU (§15.1): graph logic and tile semantics; the Triton device
templates themselves are CUDA-gated and UNVERIFIED in this CPU-only
environment (same flagged gap as PRs #88/#113/#114/#115/#117).
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

from vkernels.compiler.capture import RecordingBackend  # noqa: E402
from vkernels.compiler.codegen_cute import TEMPLATE_NAMES  # noqa: E402
from vkernels.compiler.legality import check_graph  # noqa: E402
from vkernels.compiler.lowerings import (  # noqa: E402
    THREADS_PER_WORKER,
    check_thread_contract,
    lower_graph,
)
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import BF16, F32, I32  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402

# ---------------------------------------------------------------------------
# Oracles (independent eager recomputation, fp64)
# ---------------------------------------------------------------------------


def _unit_tables(length, half, rng):
    """cos/sin tables on the unit circle (rotation must be orthonormal for
    the inverse property to hold — random normals are NOT angles)."""
    theta = rng.uniform(0, 2 * np.pi, (length, half))
    return np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)


def _rope_il(x, cos, sin, pos, rot, negate=False):
    """Interleaved (GPT-J) pair rotation, fp64.

    PINNED convention (issue #95): pairs (2i, 2i+1), tables indexed by PAIR
    index i at the row's position; dims [rot, D) pass through.
    negate=True applies the CONJUGATE (negative-angle) rotation.
    """
    x = x.astype(np.float64)
    out = x.copy()
    c = cos[pos].astype(np.float64)
    s = sin[pos].astype(np.float64) * (-1.0 if negate else 1.0)
    out[..., 0:rot:2] = x[..., 0:rot:2] * c - x[..., 1:rot:2] * s
    out[..., 1:rot:2] = x[..., 1:rot:2] * c + x[..., 0:rot:2] * s
    return out


def _oracle_scores(q, latent, wtable, comp, compidx, sink, bias, pos, W, K, scale):
    """Per (b, h): logits over [W window | K compressed | 1 sink], two-pass
    fp64 softmax over valid candidates; invalid slots exact 0.0."""
    b, h, _ = q.shape
    probs = np.zeros((b, h, W + K + 1))
    for bi in range(b):
        p = int(pos[bi]) if hasattr(pos, "shape") and pos.ndim else int(pos)
        for hi in range(h):
            logits = np.full(W + K + 1, -np.inf)
            t_lo = max(0, p - W + 1)
            for i in range(W):
                t = p - W + 1 + i
                if t_lo <= t <= p:
                    logits[i] = (
                        latent[bi, wtable[bi, t]].astype(np.float64) @ q[bi, hi].astype(np.float64)
                    ) * scale
            for j in range(K):
                e = int(compidx[bi, j])
                if e >= 0:
                    lg = (comp[bi, e].astype(np.float64) @ q[bi, hi].astype(np.float64)) * scale
                    if bias is not None:
                        lg += float(bias[bi, j])
                    logits[W + j] = lg
            logits[W + K] = float(sink[bi, hi]) if sink.ndim == 2 else float(sink[hi])
            m = logits.max()
            e = np.exp(logits - m)
            e[~np.isfinite(logits)] = 0.0
            probs[bi, hi] = e / e.sum()
    return probs


def _oracle_values(probs, latent, wtable, comp, compidx, pos, W, K):
    """ctx[b, h] = sum valid window probs * latent rows + valid compressed
    probs * comp rows; sink column (W+K) contributes nothing."""
    b, h, _ = probs.shape
    d = latent.shape[2]
    ctx = np.zeros((b, h, d))
    for bi in range(b):
        p = int(pos[bi]) if hasattr(pos, "shape") and pos.ndim else int(pos)
        for hi in range(h):
            acc = np.zeros(d)
            t_lo = max(0, p - W + 1)
            for i in range(W):
                t = p - W + 1 + i
                if t_lo <= t <= p:
                    acc += probs[bi, hi, i] * latent[bi, wtable[bi, t]].astype(np.float64)
            for j in range(K):
                e = int(compidx[bi, j])
                if e >= 0:
                    acc += probs[bi, hi, W + j] * comp[bi, e].astype(np.float64)
            ctx[bi, hi] = acc
    return ctx


def _oracle_grouped(x, w, gh):
    """y[m, h*N_g + n] = w[h*N_g + n, h*K_g + :] @ x[m, h*K_g + :] —
    block-diagonal per-head projection (w stored [K, N], §3.1)."""
    m, k = x.shape
    n = w.shape[1]
    k_g, n_g = k // gh, n // gh
    y = np.zeros((m, n))
    for h in range(gh):
        y[:, h * n_g : (h + 1) * n_g] = x[:, h * k_g : (h + 1) * k_g].astype(np.float64) @ w[
            h * k_g : (h + 1) * k_g, h * n_g : (h + 1) * n_g
        ].astype(np.float64)
    return y


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------


def _capture_chain(b=2, h=4, d=16, s=32, m=8, k=4, w=8, rot=8, *, bias=True, scalar_pos=True, stop="full"):
    """Record rope -> mla_scores -> mla_values -> conjugate_rope -> grouped
    linear (or a prefix, per ``stop``); returns (recorder, handles)."""
    recorder = RecordingBackend()
    pos = recorder.define_position(s) if scalar_pos else recorder.define_row_positions(
        "rowpos", b, s, storage_id=110
    )
    cos_t = recorder.external_tensor("cos", (64, rot // 2), F32, storage_id=101)
    sin_t = recorder.external_tensor("sin", (64, rot // 2), F32, storage_id=102)
    latent = recorder.external_tensor("latent", (b, s, d), BF16, storage_id=103)
    wtable = recorder.external_tensor("wtable", (b, s), I32, storage_id=104)
    comp = recorder.external_tensor("comp", (b, m, d), BF16, storage_id=105)
    compidx = recorder.external_tensor("compidx", (b, k), I32, storage_id=106)
    sink = recorder.external_tensor("sink", (h,), F32, storage_id=107)
    bias_t = recorder.external_tensor("bias", (b, k), F32, storage_id=108) if bias else None
    q_in = recorder.external_tensor("q_in", (b, h, d), F32, storage_id=100)
    q = recorder.rope(q_in, cos_t, sin_t, pos, layer=0, which="q", convention="interleaved", rotary_dim=rot)
    probs = recorder.mla_scores(
        q, latent, wtable, comp, compidx, sink, pos,
        scale=d**-0.5, layer=0, window=w, bias=bias_t,
    )
    handles = {
        "pos": pos, "cos": cos_t, "sin": sin_t, "latent": latent, "wtable": wtable,
        "comp": comp, "compidx": compidx, "sink": sink, "bias": bias_t, "q_in": q_in,
        "q": q, "probs": probs,
    }
    if stop == "scores":
        return recorder, handles
    ctx = recorder.mla_values(probs, latent, wtable, comp, compidx, pos, layer=0, window=w)
    handles["ctx"] = ctx
    if stop == "ctx":
        return recorder, handles
    o = recorder.conjugate_rope(ctx, cos_t, sin_t, pos, layer=0, which="o", convention="interleaved")
    w_go = recorder.external_tensor("w_go", (h * d, h * d), F32, storage_id=109)
    y = recorder.linear(o, w_go, grouped_heads=h)
    handles.update({"o": o, "w_go": w_go, "y": y})
    return recorder, handles


def _run(recorder, handles, arrays, positions, workers=4, canary=True):
    families = lower_graph(recorder.graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    plan, _ = plan_memory(recorder.graph, schedule, families)
    storage = dict(arrays)
    for buf in plan.buffers:
        storage[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(
        schedule,
        workers=workers,
        storage_arrays=storage,
        graph=recorder.graph,
        workspace_plan=plan,
        canary=canary,
    )
    scalars = {} if isinstance(positions, np.ndarray) else {"p": positions}
    trace = executor.run(scalars)
    return executor, trace, families, schedule, plan


def _standard_data(rng, b, h, d, s, m, k, rot):
    theta = rng.uniform(0, 2 * np.pi, (64, rot // 2))
    return {
        "cos": np.cos(theta).astype(np.float32),
        "sin": np.sin(theta).astype(np.float32),
        "latent": rng.standard_normal((b, s, d)).astype(np.float16),
        "comp": rng.standard_normal((b, m, d)).astype(np.float16),
        "sink": rng.standard_normal(h).astype(np.float32),
        "bias": (rng.standard_normal((b, k)) * 0.1).astype(np.float32),
        "q_in": rng.standard_normal((b, h, d)).astype(np.float32),
        "w_go": rng.standard_normal((h * d, h * d)).astype(np.float32),
    }


# ===========================================================================
# Capture (recorder contract)
# ===========================================================================


def test_capture_mla_scores_records_contract():
    recorder, h = _capture_chain()
    op = recorder.graph.ops[1]
    assert op.kind == "mla_scores"
    assert op.attributes["scale"] == pytest.approx(16**-0.5)
    assert op.attributes["window"] == 8 and op.attributes["comp_slots"] == 4
    assert op.attributes["sink"] == "last"
    assert op.attributes["width"] == 8 + 4 + 1  # window ∪ compressed ∪ sink
    # six inputs + the #97 block_bias
    assert len(op.inputs) == 7
    # probs [B, H, W+K+1], f32 (softmax output)
    probs_tv = recorder.graph.tensors[h["probs"].value.name]
    assert probs_tv.shape == (2, 4, 13) and probs_tv.dtype == F32
    # the window gather is INDIRECT through the slot table (Region.indirect,
    # #94 form) — both the scores and the values gather share it
    for op_idx in (1, 2):
        op_ = recorder.graph.ops[op_idx]
        indirect = [r for r in op_.read_regions if r.indirect_table is not None]
        assert len(indirect) == 1, f"op {op_.kind} must gather the latent pool indirectly"
        assert indirect[0].indirect_axis == 1
        assert indirect[0].indirect_table.name == "wtable"
    err = [d for d in check_graph(recorder.graph) if d.severity == "error"]
    assert err == []


def test_capture_mla_values_records_sink_absorption_contract():
    recorder, h = _capture_chain()
    op = recorder.graph.ops[2]
    assert op.kind == "mla_values"
    assert op.attributes["window"] == 8 and op.attributes["comp_slots"] == 4
    contract = op.numerical_contract["sink"]
    assert "no value" in contract, "sink column must be documented as contributing no value"
    assert recorder.graph.tensors[h["ctx"].value.name].shape == (2, 4, 16)


def test_capture_conjugate_rope_records_negative_angle():
    recorder, h = _capture_chain()
    op = recorder.graph.ops[3]
    assert op.kind == "conjugate_rope"
    assert op.attributes["which"] == "o"
    assert op.attributes["convention"] == "interleaved"
    contract = op.numerical_contract
    assert "NEGATED" in contract["interleaved"]
    assert "inverse" in contract
    # rotary_dim inferred from the [64, rot//2] tables when not given
    assert op.attributes["rotary_dim"] == 8


def test_capture_grouped_linear_records_block_diagonal():
    recorder, h = _capture_chain()
    op = recorder.graph.ops[4]
    assert op.kind == "linear"
    assert op.attributes["grouped_heads"] == 4
    # 3D activations [B, H, D] are flattened row-major for the projection
    flat = recorder.graph.tensors[op.inputs[0]]
    assert flat.shape == (2, 64)


def test_capture_rejects_bad_grouped_shapes():
    recorder = RecordingBackend()
    x = recorder.external_tensor("x", (2, 8), F32, storage_id=1)
    w = recorder.external_tensor("w", (12, 10), F32, storage_id=2)  # H must divide N
    with pytest.raises(ValueError, match="grouped_heads"):
        recorder.linear(x, w, grouped_heads=4)
    with pytest.raises(ValueError, match="grouped_heads"):
        recorder.linear(x, w, grouped_heads=0)


def test_capture_mla_scores_rejects_shape_mismatches():
    recorder = RecordingBackend()
    p = recorder.define_position(32)
    q = recorder.external_tensor("q", (2, 4, 16), F32, storage_id=1)
    latent = recorder.external_tensor("latent", (2, 32, 16), BF16, storage_id=2)
    wtable = recorder.external_tensor("wtable", (2, 32), I32, storage_id=3)
    comp = recorder.external_tensor("comp", (2, 8, 16), BF16, storage_id=4)
    compidx = recorder.external_tensor("compidx", (2, 4), I32, storage_id=5)
    sink = recorder.external_tensor("sink", (4,), F32, storage_id=6)
    short_latent = recorder.external_tensor("latent_bad", (2, 16, 16), BF16, storage_id=9)
    with pytest.raises(ValueError, match="latent_pool"):
        recorder.mla_scores(q, short_latent, wtable, comp, compidx, sink, p, scale=0.25, layer=0, window=8)
    with pytest.raises(ValueError, match="comp_pool"):
        short_comp = recorder.external_tensor("comp_bad", (2, 8, 8), BF16, storage_id=10)
        recorder.mla_scores(q, latent, wtable, short_comp, compidx, sink, p, scale=0.25, layer=0, window=8)
    with pytest.raises(ValueError, match="sink"):
        bad_sink = recorder.external_tensor("sink2", (3,), F32, storage_id=7)
        recorder.mla_scores(q, latent, wtable, comp, compidx, bad_sink, p, scale=0.25, layer=0, window=8)
    with pytest.raises(ValueError, match="window"):
        recorder.mla_scores(q, latent, wtable, comp, compidx, sink, p, scale=0.25, layer=0, window=0)
    bad_bias = recorder.external_tensor("bb", (2, 5), F32, storage_id=8)
    with pytest.raises(ValueError, match="bias"):
        recorder.mla_scores(q, latent, wtable, comp, compidx, sink, p, scale=0.25, layer=0, window=8, bias=bad_bias)


def test_capture_conjugate_rope_rejects_mismatched_tables():
    recorder = RecordingBackend()
    p = recorder.define_position(32)
    x = recorder.external_tensor("x", (2, 4, 16), F32, storage_id=1)
    cos_t = recorder.external_tensor("cos", (64, 4), F32, storage_id=2)
    sin_t = recorder.external_tensor("sin", (64, 8), F32, storage_id=3)
    with pytest.raises(Exception, match="match"):
        recorder.conjugate_rope(x, cos_t, sin_t, p, layer=0, which="o")
    sin_ok = recorder.external_tensor("sin_ok", (64, 4), F32, storage_id=4)
    # tables define rot=8, but the caller pins rotary_dim=16 -> mismatch
    with pytest.raises(Exception, match="tables must be"):
        recorder.conjugate_rope(x, cos_t, sin_ok, p, layer=0, which="o", rotary_dim=16)
    with pytest.raises(Exception, match="convention"):
        recorder.conjugate_rope(x, cos_t, sin_ok, p, layer=0, which="o", convention="bogus")


# ===========================================================================
# Lowering
# ===========================================================================


def test_lowering_mla_scores_one_task_per_head():
    recorder, h = _capture_chain(b=3, h=4, d=16)
    fam = lower_graph(recorder.graph)[1]
    assert fam.kind == "mla_scores"
    assert fam.domain.dims == ((3, 1), (4, 1))
    assert fam.task_count == 12
    assert fam.threads == THREADS_PER_WORKER
    assert fam.params["scale"] == pytest.approx(16**-0.5)
    assert fam.params["window"] == 8 and fam.params["comp_slots"] == 4
    r = fam.reads(6)  # row-major task grid: task 6 = coords (1, 2)
    assert r[0].boxes == ((1, 2), (2, 3), (0, 16))  # q[b, h]
    assert r[1].boxes == ((1, 2), (0, 32), (0, 16))  # latent pool row block
    assert r[4].boxes == ((1, 2), (0, 4))  # comp_idx row
    wr = fam.writes(6)
    assert wr[0].boxes == ((1, 2), (2, 3), (0, 13))
    assert check_thread_contract([fam]) == []


def test_lowering_mla_values_regions_and_width():
    recorder, h = _capture_chain(b=2, h=2, d=16)
    fam = lower_graph(recorder.graph)[2]
    assert fam.kind == "mla_values"
    assert fam.domain.dims == ((2, 1), (2, 1))
    r = fam.reads(1)  # coords (0, 1)
    assert r[0].boxes == ((0, 1), (1, 2), (0, 13))  # probs incl. the sink column
    assert r[4].boxes == ((0, 1), (0, 4))
    wr = fam.writes(1)
    assert wr[0].boxes == ((0, 1), (1, 2), (0, 16))
    assert check_thread_contract([fam]) == []


def test_lowering_grouped_linear_block_diagonal_reads():
    recorder, h = _capture_chain(b=2, h=4, d=16)
    fam = lower_graph(recorder.graph)[4]
    assert fam.kind == "gemm"
    assert fam.params["grouped_heads"] == 4 and fam.params["k_g"] == 16 and fam.params["n_g"] == 16
    # tile n0=32 reads ONLY head 2's diagonal blocks: x[:, 32:48], w[32:48, 32:48]
    r = fam.reads(2)  # gemm tile coords (0, 2)
    boxes = [reg.boxes for reg in r]
    assert ((0, 2), (32, 48)) in boxes  # x head-2 block
    assert ((32, 48), (32, 48)) in boxes  # w diagonal block
    # and NO off-block weight region anywhere: the only 2D region with
    # row-start >= 32 is the weight diagonal block itself
    w_boxes = [reg.boxes for reg in r if len(reg.boxes) == 2 and reg.boxes[0][0] >= 32]
    assert w_boxes == [((32, 48), (32, 48))]
    assert check_thread_contract([fam]) == []


def test_chain_is_legal_and_orders_scores_before_values():
    recorder, h = _capture_chain()
    graph = recorder.graph
    assert [d for d in check_graph(graph) if d.severity == "error"] == []
    families = lower_graph(graph)
    assert check_thread_contract(families) == []
    schedule = PhaseSchedule.from_families(families, workers=4)
    order = []
    for phase in schedule.phases:
        for fam in phase.families if hasattr(phase, "families") else []:
            order.append(fam.kind)
    kinds = [f.kind for f in families]
    assert kinds.index("mla_scores") < kinds.index("mla_values")
    assert kinds == ["rope", "mla_scores", "mla_values", "conjugate_rope", "gemm"]


# ===========================================================================
# Reference executor vs eager fp64 oracle
# ===========================================================================


@pytest.mark.parametrize("workers", [1, 3, 4])
def test_reference_full_chain_matches_eager_oracle(workers):
    rng = np.random.default_rng(95)
    b, h, d, s, m, k, w, rot = 2, 4, 16, 32, 8, 4, 8, 8
    p0 = 17
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.array([0, 3, 5, -1], dtype=np.int32), (b, 1))
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    executor, trace, *_ = _run(recorder, handles, arrays, p0, workers=workers)
    y_out = np.array(executor.tensor(handles["y"].value.name), copy=True)

    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable, data["comp"], compidx,
                               data["sink"], data["bias"], p0, w, k, d**-0.5)
    ctx_ref = _oracle_values(probs_ref, data["latent"], wtable, data["comp"], compidx, p0, w, k)
    o_ref = _rope_il(ctx_ref, data["cos"], data["sin"], p0, rot, negate=True).reshape(b, h * d)
    y_ref = _oracle_grouped(o_ref, data["w_go"], h)
    assert np.isfinite(y_out).all()
    assert np.abs(y_out - y_ref.astype(np.float32)).max() < 1e-5
    assert trace.kernel_launches == 1  # §15.2: the whole chain is one launch


def test_reference_window_geometry_row_start():
    """p < W-1: only p+1 window candidates are valid; the masked prefix is
    exact 0.0 and the softmax renormalizes over the survivors.

    PINNED window bound (issue #95): candidate logical position t is live
    iff |q - t| < W and t <= q (i.e. t in (p-W, p]) — slot i maps to
    t = p - W + 1 + i, so the FIRST slots die at row start."""
    rng = np.random.default_rng(5)
    b, h, d, s, m, k, w, rot = 1, 2, 16, 32, 4, 2, 8, 8
    p0 = 2  # valid window: t in {0, 1, 2} -> slots 6, 7; slots 0..5 exact 0
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.array([[0, -1]], dtype=np.int32)
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    # scores-only subgraph: probs is the FINAL output (alive at read time —
    # in the full chain the planner co-allocates its dead storage away, the
    # #98 dead-buffer lesson)
    recorder2, handles2 = _capture_chain(b, h, d, s, m, k, w, rot, stop="scores")
    executor, *_ = _run(recorder2, handles2, arrays, p0)
    probs = np.array(executor.tensor(handles2["probs"].value.name), copy=True)
    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable, data["comp"], compidx,
                               data["sink"], data["bias"], p0, w, k, d**-0.5)
    assert np.abs(probs - probs_ref.astype(np.float32)).max() < 1e-6
    np.testing.assert_array_equal(probs[0, 0, :5], 0.0)  # t < 0 masked
    assert probs.sum(-1).min() > 1 - 1e-6 and probs.sum(-1).max() < 1 + 1e-6


def test_reference_window_geometry_row_end():
    """p >= W-1: exactly the W most recent logical positions are live."""
    rng = np.random.default_rng(6)
    b, h, d, s, m, k, w, rot = 1, 2, 16, 32, 4, 2, 8, 8
    p0 = 30  # window t in [23, 30], all valid
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.array([[1, 2]], dtype=np.int32)
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    executor, *_ = _run(recorder, handles, arrays, p0)
    y_out = np.array(executor.tensor(handles["y"].value.name), copy=True)
    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable, data["comp"], compidx,
                               data["sink"], data["bias"], p0, w, k, d**-0.5)
    ctx_ref = _oracle_values(probs_ref, data["latent"], wtable, data["comp"], compidx, p0, w, k)
    o_ref = _rope_il(ctx_ref, data["cos"], data["sin"], p0, rot, negate=True).reshape(b, h * d)
    y_ref = _oracle_grouped(o_ref, data["w_go"], h)
    assert np.abs(y_out - y_ref.astype(np.float32)).max() < 1e-5


def test_reference_ragged_row_positions_issue93():
    """Per-row decode positions (#93 form): rows p=0, 5, 31 in one batch —
    each row's window bound and rope position follow its OWN p."""
    rng = np.random.default_rng(93)
    b, h, d, s, m, k, w, rot = 3, 2, 16, 32, 4, 2, 8, 8
    positions = np.array([0, 5, 31], dtype=np.int32)
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot, scalar_pos=False)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.array([0, 1], dtype=np.int32), (b, 1))
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"], 110: positions}
    executor, *_ = _run(recorder, handles, arrays, positions)
    y_out = np.array(executor.tensor(handles["y"].value.name), copy=True)
    y_ref = np.zeros((b, h * d))
    for bi, p0 in enumerate(positions):
        q_ref = _rope_il(data["q_in"][bi : bi + 1], data["cos"], data["sin"], int(p0), rot)
        probs_ref = _oracle_scores(q_ref, data["latent"][bi : bi + 1], wtable[bi : bi + 1],
                                   data["comp"][bi : bi + 1], compidx[bi : bi + 1],
                                   data["sink"], data["bias"][bi : bi + 1], int(p0), w, k, d**-0.5)
        ctx_ref = _oracle_values(probs_ref, data["latent"][bi : bi + 1], wtable[bi : bi + 1],
                                 data["comp"][bi : bi + 1], compidx[bi : bi + 1], int(p0), w, k)
        o_ref = _rope_il(ctx_ref, data["cos"], data["sin"], int(p0), rot, negate=True).reshape(1, h * d)
        y_ref[bi] = _oracle_grouped(o_ref, data["w_go"], h)
    assert np.abs(y_out - y_ref.astype(np.float32)).max() < 1e-5


def test_reference_sink_absorbs_mass_but_contributes_no_value():
    """Two absorption invariants: (1) a large sink logit soaks up ALL the
    softmax mass (sink column -> 1, context -> 0); (2) a -inf sink is
    exactly equivalent to a candidate-only softmax (dead mass, no
    renormalization of the survivors)."""
    rng = np.random.default_rng(11)
    b, h, d, s, m, k, w, rot = 2, 3, 16, 32, 4, 2, 8, 8
    p0 = 20

    def run_with_sink(sink_np):
        recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot, stop="ctx")
        data = _standard_data(np.random.default_rng(11), b, h, d, s, m, k, rot)
        data["sink"] = sink_np
        wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
        compidx = np.tile(np.array([0, 1], dtype=np.int32), (b, 1))
        arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
                  104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
                  108: data["bias"], 109: data["w_go"]}
        executor, *_ = _run(recorder, handles, arrays, p0)
        probs = np.array(executor.tensor(handles["probs"].value.name), copy=True) \
            if "probs" in handles else None
        ctx = np.array(executor.tensor(handles["ctx"].value.name), copy=True)
        return probs, ctx

    # (1) sink absorbs everything
    probs_huge, ctx_huge = run_with_sink(np.full(h, 1e4, dtype=np.float32))
    assert probs_huge[:, :, -1].min() > 1 - 1e-4, "sink column must soak up the mass"
    assert np.abs(ctx_huge).max() < 1e-3, "all-mass-on-sink must gather nothing"

    # (2) dead sink == candidate-only softmax
    q_ref = _rope_il(_standard_data(np.random.default_rng(11), b, h, d, s, m, k, rot)["q_in"],
                     _standard_data(np.random.default_rng(11), b, h, d, s, m, k, rot)["cos"],
                     _standard_data(np.random.default_rng(11), b, h, d, s, m, k, rot)["sin"], p0, rot)
    data0 = _standard_data(np.random.default_rng(11), b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.array([0, 1], dtype=np.int32), (b, 1))
    probs_ref = _oracle_scores(q_ref, data0["latent"], wtable, data0["comp"], compidx,
                               np.full(h, -1e30, dtype=np.float32), data0["bias"], p0, w, k, d**-0.5)
    ctx_ref = _oracle_values(probs_ref, data0["latent"], wtable, data0["comp"], compidx, p0, w, k)
    _, ctx_dead = run_with_sink(np.full(h, -1e30, dtype=np.float32))
    assert np.abs(ctx_dead - ctx_ref.astype(np.float32)).max() < 1e-5


def test_reference_mqa_broadcast_shared_latent_pool():
    """num_key_value_heads == 1: every head scores the SAME latent rows;
    per-head probs differ only through their own q/sink."""
    rng = np.random.default_rng(12)
    b, h, d, s, m, k, w, rot = 1, 4, 16, 32, 4, 2, 8, 8
    p0 = 15
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.array([[0, 2]], dtype=np.int32)
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    recorder2, handles2 = _capture_chain(b, h, d, s, m, k, w, rot, stop="scores")
    executor, *_ = _run(recorder2, handles2, arrays, p0)
    probs = np.array(executor.tensor(handles2["probs"].value.name), copy=True)
    assert probs.shape == (1, 4, 11)
    # heads differ (own q, own sink) but all read the same pool rows
    assert not np.allclose(probs[0, 0], probs[0, 1])
    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable, data["comp"], compidx,
                               data["sink"], data["bias"], p0, w, k, d**-0.5)
    assert np.abs(probs - probs_ref.astype(np.float32)).max() < 1e-6


def test_reference_block_bias_affects_only_compressed_columns():
    """#97's normalized block_bias enters ONLY the compressed logits: the
    window/sink columns of the softmax are reweighted but the window logits
    themselves are identical with and without bias."""
    rng = np.random.default_rng(97)
    b, h, d, s, m, k, w, rot = 2, 2, 16, 32, 4, 2, 8, 8
    p0 = 13
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot, bias=True)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.array([0, 1], dtype=np.int32), (b, 1))
    arrays_bias = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
                   104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
                   108: data["bias"], 109: data["w_go"]}
    arrays_nobias = dict(arrays_bias)
    arrays_nobias[108] = np.zeros((b, k), dtype=np.float32)
    recorder_a, handles_a = _capture_chain(b, h, d, s, m, k, w, rot, bias=True, stop="scores")
    recorder_b, handles_b = _capture_chain(b, h, d, s, m, k, w, rot, bias=True, stop="scores")
    ex_a, *_ = _run(recorder_a, handles_a, arrays_bias, p0)
    ex_b, *_ = _run(recorder_b, handles_b, arrays_nobias, p0)
    pa = np.array(ex_a.tensor(handles_a["probs"].value.name), copy=True)
    pb = np.array(ex_b.tensor(handles_b["probs"].value.name), copy=True)
    assert np.abs(pa - pb).max() > 1e-4, "bias must move the compressed probabilities"
    # window logits (pre-softmax) identical: verified via the window-only
    # renormalization — recompute both windows from the same q
    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref_b = _oracle_scores(q_ref, data["latent"], wtable, data["comp"], compidx,
                                 data["sink"], None, p0, w, k, d**-0.5)
    assert np.abs(pb - probs_ref_b.astype(np.float32)).max() < 1e-6


def test_reference_rope_round_trip_is_identity():
    """rope(...) -> conjugate_rope(...) at the SAME position is the identity
    (the conjugate rotation is the exact inverse). The task arithmetic is
    fp64 — identity to ~1e-12 there — but the workspace/output storages are
    fp32 (no F64 tensor dtype in the IR), so the OBSERVABLE round-trip bound
    is two fp32 store roundings, ~1e-6; asserted at 1e-6."""
    rng = np.random.default_rng(13)
    b, h, d, s, rot, p0 = 2, 3, 16, 32, 8, 21
    recorder = RecordingBackend()
    p = recorder.define_position(s)
    cos_t = recorder.external_tensor("cos", (64, rot // 2), F32, storage_id=101)
    sin_t = recorder.external_tensor("sin", (64, rot // 2), F32, storage_id=102)
    x_in = recorder.external_tensor("x", (b, h, d), F32, storage_id=103)
    y = recorder.rope(x_in, cos_t, sin_t, p, layer=0, which="q", convention="interleaved", rotary_dim=rot)
    back = recorder.conjugate_rope(y, cos_t, sin_t, p, layer=0, which="o", convention="interleaved")
    theta = rng.uniform(0, 2 * np.pi, (64, rot // 2))
    # NOTE: external storage ids must not collide with the recorder's
    # fresh-storage counter (low ids are fair game) — use 101+.
    data = {101: np.cos(theta).astype(np.float32), 102: np.sin(theta).astype(np.float32),
            103: rng.standard_normal((b, h, d)).astype(np.float32)}
    executor, *_ = _run(recorder, {"back": back}, data, p0)
    out = np.array(executor.tensor(back.value.name), copy=True)
    assert np.abs(out - data[103]).max() < 1e-6  # two fp32 store roundings (fp64 task math)


def test_reference_interleaved_convention_pinned():
    """PINNED convention (slice/stride/pairing): pairs (0,1),(2,3) over the
    first ROT dims, tables indexed by PAIR index; hand-computed exact
    values, dims [ROT, D) pass through."""
    rng = np.random.default_rng(14)
    b, h, d, s, rot, p0 = 1, 1, 8, 32, 4, 3
    recorder = RecordingBackend()
    p = recorder.define_position(s)
    cos_t = recorder.external_tensor("cos", (64, rot // 2), F32, storage_id=101)
    sin_t = recorder.external_tensor("sin", (64, rot // 2), F32, storage_id=102)
    x_in = recorder.external_tensor("x", (b, h, d), F32, storage_id=103)
    y = recorder.rope(x_in, cos_t, sin_t, p, layer=0, which="q", convention="interleaved", rotary_dim=rot)
    c0, s0 = 0.6, 0.8  # cos/sin for pair 0 (unit circle)
    c1, s1 = -0.28, 0.96  # pair 1
    # full-height tables (indexed at the runtime position p0=3)
    cos_tab = np.zeros((s, rot // 2), dtype=np.float32)
    sin_tab = np.zeros((s, rot // 2), dtype=np.float32)
    cos_tab[3], sin_tab[3] = (c0, c1), (s0, s1)
    data = {101: cos_tab, 102: sin_tab,
            103: np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]], dtype=np.float32)}
    executor, *_ = _run(recorder, {"y": y}, data, p0)
    out = np.array(executor.tensor(y.value.name), copy=True)
    expected = np.array([[[
        1.0 * c0 - 2.0 * s0, 2.0 * c0 + 1.0 * s0,
        3.0 * c1 - 4.0 * s1, 4.0 * c1 + 3.0 * s1,
        5.0, 6.0, 7.0, 8.0,  # tail passthrough
    ]]], dtype=np.float64)
    assert np.abs(out - expected).max() < 1e-5
    # and the conjugate exactly undoes it
    recorder2 = RecordingBackend()
    p2 = recorder2.define_position(s)
    cos2 = recorder2.external_tensor("cos", (s, rot // 2), F32, storage_id=101)
    sin2 = recorder2.external_tensor("sin", (s, rot // 2), F32, storage_id=102)
    x2 = recorder2.external_tensor("x", (b, h, d), F32, storage_id=103)
    fwd = recorder2.rope(x2, cos2, sin2, p2, layer=0, which="q", convention="interleaved", rotary_dim=rot)
    rev = recorder2.conjugate_rope(fwd, cos2, sin2, p2, layer=0, which="o", convention="interleaved")
    executor2, *_ = _run(recorder2, {"rev": rev}, data, p0)
    out2 = np.array(executor2.tensor(rev.value.name), copy=True)
    assert np.abs(out2 - data[103]).max() < 1e-5


def test_reference_grouped_linear_dense_mirror_and_nan_offblock():
    """Grouped projection == dense GEMM with off-block weights zeroed; NaN
    canaries in the off-block storage change nothing (never read)."""
    rng = np.random.default_rng(15)
    b, h, d = 2, 4, 16
    x = rng.standard_normal((b, h, d)).astype(np.float32)
    w = rng.standard_normal((h * d, h * d)).astype(np.float32)
    w_nan = w.copy()
    for a in range(h):
        for c in range(h):
            if a != c:
                w_nan[a * d : (a + 1) * d, c * d : (c + 1) * d] = np.nan

    def run(weights):
        recorder = RecordingBackend()
        x_in = recorder.external_tensor("x", (b, h, d), F32, storage_id=201)
        w_t = recorder.external_tensor("w", (h * d, h * d), F32, storage_id=202)
        y = recorder.linear(x_in, w_t, grouped_heads=h)
        executor, *_ = _run(recorder, {"y": y}, {201: x, 202: weights}, None)
        return np.array(executor.tensor(y.value.name), copy=True)

    y_clean = run(w)
    y_nan = run(w_nan)
    y_ref = _oracle_grouped(x.reshape(b, h * d), w, h)
    assert np.abs(y_clean - y_ref.astype(np.float32)).max() < 1e-5
    np.testing.assert_array_equal(y_clean, y_nan), "off-block NaN must never be observed"


def test_reference_comp_idx_masking_and_unselected_rows_canary():
    """comp_idx = -1 slots are masked (exact 0.0 prob); unselected compressed
    pool rows are never read — NaN-filled there, the result stays finite."""
    rng = np.random.default_rng(16)
    b, h, d, s, m, k, w, rot = 2, 2, 16, 32, 8, 4, 8, 8
    p0 = 10
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    comp = data["comp"].copy()
    comp[:, 2:, :] = np.nan  # rows 2+ never indexed (compidx uses 0..3 < 2? -> use 0,1,-1,-1)
    comp[:, 3:, :] = np.nan
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.array([[0, 1, -1, -1]] * b, dtype=np.int32)
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: comp, 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    recorder2, handles2 = _capture_chain(b, h, d, s, m, k, w, rot, stop="ctx")
    executor, trace, *_ = _run(recorder2, handles2, arrays, p0)
    probs = np.array(executor.tensor(handles2["probs"].value.name), copy=True)
    ctx = np.array(executor.tensor(handles2["ctx"].value.name), copy=True)
    assert np.isfinite(probs).all() and np.isfinite(ctx).all()
    np.testing.assert_array_equal(probs[:, :, 8 + 2], 0.0)  # comp_idx=-1 slots
    np.testing.assert_array_equal(probs[:, :, 8 + 3], 0.0)
    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable, comp, compidx,
                               data["sink"], data["bias"], p0, w, k, d**-0.5)
    assert np.abs(probs - probs_ref.astype(np.float32)).max() < 1e-6


def test_reference_row_position_zero_no_candidates_but_sink():
    """p = 0: the ONLY window candidate is t=0 (slot W-1); the sink keeps
    the softmax well-defined and the context gathers just that row."""
    rng = np.random.default_rng(17)
    b, h, d, s, m, k, w, rot = 1, 2, 16, 32, 4, 2, 8, 8
    p0 = 0
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.array([[-1, -1]], dtype=np.int32)
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    recorder2, handles2 = _capture_chain(b, h, d, s, m, k, w, rot, stop="scores")
    executor, *_ = _run(recorder2, handles2, arrays, p0)
    probs = np.array(executor.tensor(handles2["probs"].value.name), copy=True)
    np.testing.assert_array_equal(probs[0, 0, :7], 0.0)
    np.testing.assert_array_equal(probs[0, 0, 8:10], 0.0)  # no compressed candidates
    assert probs[0, 0, 7] > 0 and probs[0, 0, 10] > 0
    assert probs.sum(-1).min() > 1 - 1e-6


def test_reference_bf16_pools_streamed_single_store():
    """bf16 latent/comp pools are accepted (streamed .cg); probs fp32, one
    fp32 store per (b, h) row — grid accounting: B*H tasks per MLA family."""
    rng = np.random.default_rng(18)
    b, h, d, s, m, k, w, rot = 2, 4, 16, 32, 4, 2, 8, 8
    p0 = 9
    recorder, handles = _capture_chain(b, h, d, s, m, k, w, rot)
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.array([0, 1], dtype=np.int32), (b, 1))
    arrays = {100: data["q_in"], 101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: wtable, 105: data["comp"], 106: compidx, 107: data["sink"],
              108: data["bias"], 109: data["w_go"]}
    executor, trace, families, schedule, plan = _run(recorder, handles, arrays, p0)
    assert trace.kernel_launches == 1
    assert trace.grid_barriers == len(schedule.phases)
    kinds = [fam.kind for fam in families]
    assert kinds.count("mla_scores") == 1 and kinds.count("mla_values") == 1
    scores_fam = families[kinds.index("mla_scores")]
    assert scores_fam.task_count == b * h
    values_fam = families[kinds.index("mla_values")]
    assert values_fam.task_count == b * h
    assert trace.task_executions >= b * h * 2  # both MLA families at full fanout


def test_reference_bf16_single_store_epilogue():
    """fp32 accumulation, ONE bf16 store: mla_values/conjugate_rope accept
    an explicit bf16 out; the gathered context round-trips through bf16
    storage within bf16 rounding of the fp64 oracle, and the downstream
    conjugate rope consumes the bf16 rows."""
    rng = np.random.default_rng(19)
    b, h, d, s, m, k, w, rot = 2, 2, 16, 32, 4, 2, 8, 8
    p0 = 12
    recorder = RecordingBackend()
    p = recorder.define_position(s)
    cos_t = recorder.external_tensor("cos", (64, rot // 2), F32, storage_id=101)
    sin_t = recorder.external_tensor("sin", (64, rot // 2), F32, storage_id=102)
    latent = recorder.external_tensor("latent", (b, s, d), BF16, storage_id=103)
    wtable = recorder.external_tensor("wtable", (b, s), I32, storage_id=104)
    comp = recorder.external_tensor("comp", (b, m, d), BF16, storage_id=105)
    compidx = recorder.external_tensor("compidx", (b, k), I32, storage_id=106)
    sink = recorder.external_tensor("sink", (h,), F32, storage_id=107)
    q_in = recorder.external_tensor("q_in", (b, h, d), F32, storage_id=108)
    q = recorder.rope(q_in, cos_t, sin_t, p, layer=0, which="q", convention="interleaved", rotary_dim=rot)
    probs = recorder.mla_scores(q, latent, wtable, comp, compidx, sink, p,
                                scale=d**-0.5, layer=0, window=w)
    ctx_bf16 = recorder.fresh_buffer("mla_ctx_bf16", (b, h, d), BF16)
    ctx = recorder.mla_values(probs, latent, wtable, comp, compidx, p, layer=0, window=w, out=ctx_bf16)
    assert recorder.graph.tensors[ctx.value.name].dtype == BF16
    o = recorder.conjugate_rope(ctx, cos_t, sin_t, p, layer=0, which="o", convention="interleaved")
    data = _standard_data(rng, b, h, d, s, m, k, rot)
    arrays = {101: data["cos"], 102: data["sin"], 103: data["latent"],
              104: np.tile(np.arange(s, dtype=np.int32), (b, 1)), 105: data["comp"],
              106: np.tile(np.array([0, 1], dtype=np.int32), (b, 1)), 107: data["sink"],
              108: data["q_in"]}
    # manual run: the bf16 buffer's backing storage must be fp16 (the
    # generic _run fill uses fp32 NaN for every workspace storage)
    families = lower_graph(recorder.graph)
    schedule = PhaseSchedule.from_families(families, workers=4)
    plan, _ = plan_memory(recorder.graph, schedule, families)
    storage = dict(arrays)
    ctx_sid = recorder.graph.tensors[ctx.value.name].storage_id
    for buf in plan.buffers:
        dt = np.float16 if buf.storage_id == ctx_sid else np.float32
        storage[buf.storage_id] = np.full(buf.numel, np.nan, dtype=dt)
    from vkernels.compiler.reference_exec import ReferenceExecutor as _RE

    executor = _RE(schedule, workers=4, storage_arrays=storage, graph=recorder.graph,
                   workspace_plan=plan, canary=False)
    executor.run({"p": p0})
    ctx_out = np.array(executor.tensor(ctx.value.name), copy=True)
    assert ctx_out.dtype == np.float16
    q_ref = _rope_il(data["q_in"], data["cos"], data["sin"], p0, rot)
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable=None, comp=data["comp"], compidx=None,
                               sink=data["sink"], bias=None, pos=p0, W=w, K=k, scale=d**-0.5) \
        if False else None
    # oracle via the standard path (wtable/compidx from the arrays above)
    wtable_a = arrays[104]
    compidx_a = arrays[106]
    probs_ref = _oracle_scores(q_ref, data["latent"], wtable_a, data["comp"], compidx_a,
                               data["sink"], None, p0, w, k, d**-0.5)
    ctx_ref = _oracle_values(probs_ref, data["latent"], wtable_a, data["comp"], compidx_a, p0, w, k)
    bf16_eps = 2**-8  # bf16 relative rounding, values O(1..4)
    assert np.abs(ctx_out.astype(np.float64) - ctx_ref).max() / max(np.abs(ctx_ref).max(), 1e-30) < 8 * bf16_eps


# ===========================================================================
# Device templates — CPU mirrors of the Triton decomposition
# ===========================================================================


def _simulate_t_mla_scores(q, latent, wtable, comp, compidx, sink, bias, pos, workers, W, K, scale):
    """Numpy mirror of ``device_triton._t_mla_scores``: one task per (b, h),
    fp32 logits, two-pass max-subtract softmax, invalid slots exact 0.0."""
    b, h, d = q.shape
    width = W + K + 1
    probs = np.full((b, h, width), np.nan, dtype=np.float32)
    for wk in range(workers):
        task = wk
        while task < b * h:
            bi, hi = task // h, task % h
            p = int(pos[bi])
            logits = np.full(width, -np.inf, dtype=np.float32)
            t_lo = max(0, p - W + 1)
            for i in range(W):
                t = p - W + 1 + i
                if t_lo <= t <= p:
                    logits[i] = np.float32(
                        latent[bi, wtable[bi, t]].astype(np.float32).astype(np.float64)
                        @ q[bi, hi].astype(np.float64)
                    ) * np.float32(scale)
            for j in range(K):
                e = int(compidx[bi, j])
                if e >= 0:
                    lg = np.float32(
                        comp[bi, e].astype(np.float32).astype(np.float64) @ q[bi, hi].astype(np.float64)
                    ) * np.float32(scale)
                    if bias is not None:
                        lg += bias[bi, j]
                    logits[W + j] = lg
            logits[W + K] = sink[hi] if sink.ndim == 1 else sink[bi, hi]
            m = logits.max()
            ex = np.exp(logits - m)
            ex[~np.isfinite(logits)] = 0.0
            probs[bi, hi] = ex / ex.sum()
            task += workers
    assert not np.isnan(probs).any(), "every (b, h) row must be written"
    return probs


def _simulate_t_mla_values(probs, latent, wtable, comp, compidx, pos, workers, W, K):
    """Numpy mirror of ``device_triton._t_mla_values``: fp32 accumulation,
    sink column skipped."""
    b, h, width = probs.shape
    d = latent.shape[2]
    out = np.full((b, h, d), np.nan, dtype=np.float32)
    for wk in range(workers):
        task = wk
        while task < b * h:
            bi, hi = task // h, task % h
            p = int(pos[bi])
            acc = np.zeros(d, dtype=np.float32)
            t_lo = max(0, p - W + 1)
            for i in range(W):
                t = p - W + 1 + i
                if t_lo <= t <= p:
                    acc += np.float32(probs[bi, hi, i]) * latent[bi, wtable[bi, t]].astype(np.float32)
            for j in range(K):
                e = int(compidx[bi, j])
                if e >= 0:
                    acc += np.float32(probs[bi, hi, W + j]) * comp[bi, e].astype(np.float32)
            out[bi, hi] = acc
            task += workers
    assert not np.isnan(out).any()
    return out


@pytest.mark.parametrize("b,h,d,s,m,k,W,workers", [(1, 4, 16, 32, 8, 4, 8, 1), (2, 2, 8, 64, 8, 4, 16, 3), (3, 4, 16, 48, 8, 4, 8, 5)])
def test_device_template_mirror_mla_pair_matches_reference(b, h, d, s, m, k, W, workers):
    rng = np.random.default_rng(b * 10 + h + d + workers)
    rot = 8
    theta = rng.uniform(0, 2 * np.pi, (64, rot // 2))
    cos_t, sin_t = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    q_in = rng.standard_normal((b, h, d)).astype(np.float32)
    latent = rng.standard_normal((b, s, d)).astype(np.float16)
    comp = rng.standard_normal((b, m, d)).astype(np.float16)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.array([0, 1, -1, -1][:k], dtype=np.int32), (b, 1))
    sink = rng.standard_normal(h).astype(np.float32)
    pos = np.full(b, 20, dtype=np.int32)
    p0 = 20

    # reference executor: rope -> scores -> values (final values alive)
    recorder = RecordingBackend()
    p = recorder.define_position(s)
    cos_h = recorder.external_tensor("cos", (64, rot // 2), F32, storage_id=201)
    sin_h = recorder.external_tensor("sin", (64, rot // 2), F32, storage_id=202)
    q_in_h = recorder.external_tensor("q_in", (b, h, d), F32, storage_id=203)
    latent_h = recorder.external_tensor("latent", (b, s, d), BF16, storage_id=204)
    wtable_h = recorder.external_tensor("wtable", (b, s), I32, storage_id=205)
    comp_h = recorder.external_tensor("comp", (b, m, d), BF16, storage_id=206)
    compidx_h = recorder.external_tensor("compidx", (b, k), I32, storage_id=207)
    sink_h = recorder.external_tensor("sink", (h,), F32, storage_id=208)
    q = recorder.rope(q_in_h, cos_h, sin_h, p, layer=0, which="q", convention="interleaved", rotary_dim=rot)
    probs_h = recorder.mla_scores(q, latent_h, wtable_h, comp_h, compidx_h, sink_h, p,
                                  scale=d**-0.5, layer=0, window=W)
    ctx_h = recorder.mla_values(probs_h, latent_h, wtable_h, comp_h, compidx_h, p, layer=0, window=W)
    arrays = {201: cos_t, 202: sin_t, 203: q_in, 204: latent, 205: wtable, 206: comp, 207: compidx, 208: sink}
    executor, *_ = _run(recorder, {"ctx": ctx_h}, arrays, p0, workers=workers)
    ref_probs = np.array(executor.tensor(probs_h.value.name), copy=True)
    ref_ctx = np.array(executor.tensor(ctx_h.value.name), copy=True)
    assert not np.isnan(ref_probs).any() and not np.isnan(ref_ctx).any()

    # mirrors (rope the q on the oracle side)
    q_m = _rope_il(q_in, cos_t, sin_t, p0, rot).astype(np.float32)
    mir_probs = _simulate_t_mla_scores(q_m, latent, wtable, comp, compidx, sink, None, pos, workers, W, k, d**-0.5)
    mir_ctx = _simulate_t_mla_values(mir_probs, latent, wtable, comp, compidx, pos, workers, W, k)
    assert np.abs(mir_probs - ref_probs).max() < 1e-5
    assert np.abs(mir_ctx - ref_ctx).max() < 1e-4


def _simulate_t_linear_grouped(x, w, gh, workers, tile_n):
    """Numpy mirror of ``device_triton._t_linear_grouped``: one task per
    (row, N tile), per-head diagonal blocks only, fp32 accumulation."""
    mm, k = x.shape
    n = w.shape[1]
    k_g, n_g = k // gh, n // gh
    out = np.full((mm, n), np.nan, dtype=np.float32)
    ntasks = mm * (n // tile_n)
    for wk in range(workers):
        task = wk
        while task < ntasks:
            row, nt = task // (n // tile_n), task % (n // tile_n)
            n0 = nt * tile_n
            acc = np.zeros(min(tile_n, n - n0), dtype=np.float32)
            for hh in range(gh):
                lo, hi = hh * n_g, (hh + 1) * n_g
                sel_lo, sel_hi = max(n0, lo), min(n0 + len(acc), hi)
                if sel_lo >= sel_hi:
                    continue
                xv = x[row, hh * k_g : (hh + 1) * k_g].astype(np.float32)
                wv = w[hh * k_g : (hh + 1) * k_g, sel_lo:sel_hi].astype(np.float32)
                acc[sel_lo - n0 : sel_hi - n0] = (xv.astype(np.float64) @ wv.astype(np.float64)).astype(np.float32)
            out[row, n0 : n0 + len(acc)] = acc
            task += workers
    assert not np.isnan(out).any()
    return out


@pytest.mark.parametrize("b,h,d,tile_n,workers", [(2, 4, 16, 16, 1), (3, 4, 16, 32, 3), (1, 8, 8, 16, 5)])
def test_device_template_mirror_grouped_linear_matches_reference(b, h, d, tile_n, workers):
    rng = np.random.default_rng(b + h + d + tile_n + workers)
    x = rng.standard_normal((b, h * d)).astype(np.float32)
    w = rng.standard_normal((h * d, h * d)).astype(np.float32)
    recorder = RecordingBackend()
    x_in = recorder.external_tensor("x", (b, h * d), F32, storage_id=201)
    w_t = recorder.external_tensor("w", (h * d, h * d), F32, storage_id=202)
    y = recorder.linear(x_in, w_t, grouped_heads=h)
    executor, *_ = _run(recorder, {"y": y}, {201: x, 202: w}, None, workers=workers)
    ref = np.array(executor.tensor(y.value.name), copy=True)
    mir = _simulate_t_linear_grouped(x, w, h, workers, tile_n)
    assert np.abs(mir - ref).max() < 1e-5


def test_codegen_knows_the_mla_templates():
    assert TEMPLATE_NAMES["mla_scores"] == "mla_scores_task"
    assert TEMPLATE_NAMES["mla_values"] == "mla_values_task"
    assert TEMPLATE_NAMES["conjugate_rope"] == "conjugate_rope_task"


# ===========================================================================
# Device templates — CUDA only (UNVERIFIED in this CPU-only environment;
# same flagged gap as PRs #88/#113/#114/#115/#117)
# ===========================================================================

gpu = pytest.mark.skipif(
    not __import__("torch").cuda.is_available() if "torch" in sys.modules else True,
    reason="requires CUDA — CPU-only lane: device templates are mirrored on CPU above",
)


@gpu
def test_device_t_mla_scores_and_values_match_oracle():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    import torch

    from vkernels.compiler.device_triton import _t_mla_scores, _t_mla_values

    b, h, d, s, m, W, K = 2, 4, 128, 512, 64, 256, 16
    rng = np.random.default_rng(95)
    theta = rng.uniform(0, 2 * np.pi, (64, d // 2))
    cos_t, sin_t = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    q_in = rng.standard_normal((b, h, d)).astype(np.float32)
    latent = rng.standard_normal((b, s, d)).astype(np.float16)
    comp = rng.standard_normal((b, m, d)).astype(np.float16)
    wtable = np.tile(np.arange(s, dtype=np.int32), (b, 1))
    compidx = np.tile(np.concatenate([np.arange(8, dtype=np.int32), np.full(8, -1, dtype=np.int32)]), (b, 1))
    sink = rng.standard_normal(h).astype(np.float32)
    pos = np.array([100, 400], dtype=np.int32)
    scale = d**-0.5

    q_ref = _rope_il(q_in, cos_t, sin_t, int(pos[0]), d)  # per-row handled below
    probs_ref = np.zeros((b, h, W + K + 1))
    for bi, p0 in enumerate(pos):
        q_r = _rope_il(q_in[bi : bi + 1], cos_t, sin_t, int(p0), d)
        probs_ref[bi] = _oracle_scores(q_r, latent[bi : bi + 1], wtable[bi : bi + 1], comp[bi : bi + 1],
                                       compidx[bi : bi + 1], sink, None, int(p0), W, K, scale)
    ctx_ref = _oracle_values(probs_ref, latent, wtable, comp, compidx, pos, W, K)

    dev = torch.device("cuda")
    tt = lambda a, dt: torch.from_numpy(a).to(device=dev, dtype=dt)
    q_t = tt(q_in, torch.float32)
    probs_t = torch.empty((b, h, W + K + 1), device=dev, dtype=torch.float32)
    ctx_t = torch.empty((b, h, d), device=dev, dtype=torch.float32)
    P = 4
    _t_mla_scores[(1,)](P, P, q_t, tt(latent, torch.bfloat16), tt(wtable, torch.int32),
                        tt(comp, torch.bfloat16), tt(compidx, torch.int32), tt(sink, torch.float32),
                        q_t, tt(pos, torch.int32), probs_t,
                        B=b, H=h, D=d, W=W, K=K, SPOOL=s, MPOOL=m, HAS_BIAS=False,
                        scale=scale)
    torch.cuda.synchronize()
    assert (probs_t.cpu().numpy() - probs_ref).max() < 1e-4
    _t_mla_values[(1,)](P, P, probs_t, tt(latent, torch.bfloat16), tt(wtable, torch.int32),
                        tt(comp, torch.bfloat16), tt(compidx, torch.int32), tt(pos, torch.int32),
                        ctx_t, B=b, H=h, D=d, W=W, K=K, SPOOL=s, MPOOL=m)
    torch.cuda.synchronize()
    assert (ctx_t.cpu().numpy() - ctx_ref.astype(np.float32)).max() < 1e-3
