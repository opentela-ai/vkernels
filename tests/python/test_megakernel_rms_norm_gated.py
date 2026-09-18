"""Sigmoid-gated RMSNorm (``rms_norm_gated``, GLM o_norm) — issue #100.

Covers the vertical slice the issue names:

* capture — ``ops.rms_norm_gated`` records the ``rms_norm`` variant with an
  elementwise ``gate`` input and the ``activation`` attribute (``sigmoid``
  only in the supported subset; silu is deliberately NOT accepted —
  vkernels' ``kda_layer_norm_gated`` already covers that variant);
* lowering — the same per-row / per-head tile domain as plain ``rms_norm``
  with a second elementwise input stream;
* reference body — fp64 ``x * rsqrt(mean(x^2) + eps) * gamma * sigmoid(gate)``
  (floe ``Glm53RMSNormGated`` math, strict-fp32 device contract);
* edge cases — zero gate (fp32 sigmoid saturates to exactly 0), large
  magnitudes (RMSNorm scale invariance, ±50 gates), epsilon boundary (eps=0
  and stats-dominated eps), head-broadcast shapes, NaN canaries;
* device template — the arithmetic of ``_t_rms2d_gated`` /
  ``_t_rms_heads_gated`` pinned on CPU by a numpy mirror of the task
  decomposition (worker-stride assignment, fp32 accumulate) at <1e-5 against
  the executed schedule. The CUDA kernels themselves are unverified on this
  CPU-only stack (same flagged gap as PRs #88/#113/#114/#115).

Main suite is torch-free (the vkernels venv carries numpy only). The floe
eager-parity test (``Glm53RMSNormGated``, incl. the bf16-input fp32-stat
cast) runs only where torch + floe are importable.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from vkernels.compiler.capture import CaptureError, RecordingBackend
from vkernels.compiler.codegen_cute import TEMPLATE_NAMES
from vkernels.compiler.legality import check_graph
from vkernels.compiler.lowerings import THREADS_PER_WORKER, check_thread_contract, lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.reference_exec import ReferenceExecutor
from vkernels.compiler.schedule_phase import PhaseSchedule

Sigmoid = lambda g: 1.0 / (1.0 + np.exp(-np.asarray(g, dtype=np.float64)))


# ---------------------------------------------------------------------------
# bf16 grid (RNE) — for the fp32-stat cast behavior at bf16 input
# ---------------------------------------------------------------------------


def _bf16_round(x):
    """Round fp32 -> bf16 grid (RNE), returned as float64."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) >> 16).astype(np.uint32) << 16
    return rounded.view(np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# Capture + execution helpers
# ---------------------------------------------------------------------------


def _capture_gated(x_shape, eps=1e-6, activation="sigmoid", *, gate_shape=None, weight_shape=None, name="o_norm"):
    recorder = RecordingBackend()
    width = x_shape[-1]
    x = recorder.external_tensor("x", x_shape, storage_id=401)
    gate = recorder.external_tensor("gate", gate_shape or x_shape, storage_id=402)
    w = recorder.external_tensor("o_w", weight_shape or (width,), storage_id=403)
    y = recorder.rms_norm_gated(x, gate, w, eps, activation=activation, name=name)
    return recorder.graph, {"x": x, "gate": gate, "w": w, "y": y}


def _run_schedule(graph, externals, workers=4, canary_fill=np.nan):
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)
    storage_arrays = {}
    for sid, nm in graph.storage_names.items():
        if sid in graph.fresh_storages:
            continue
        storage_arrays[sid] = externals[nm]
    workspace = np.full(workspace_plan.total_elements, canary_fill, dtype=np.float64)
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = workspace[buf.offset : buf.offset + buf.numel]
    executor = ReferenceExecutor(
        schedule,
        workers=workers,
        storage_arrays=storage_arrays,
        graph=graph,
        workspace_plan=workspace_plan,
    )
    trace = executor.run({})
    out_name = graph.ops[-1].outputs[0]
    return np.array(executor.tensor(out_name), copy=True), trace, executor


def _fp64_oracle(x, gate, w, eps):
    """Independent fp64 o_norm oracle: strict-fp32 contract modeled in fp64."""
    x = np.asarray(x, dtype=np.float64)
    gate = np.asarray(gate, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    inv = np.reciprocal(np.sqrt((x * x).mean(-1, keepdims=True) + eps))
    return x * inv * w * Sigmoid(gate)


# ===========================================================================
# Capture (recorder contract)
# ===========================================================================


def test_capture_records_gate_input_and_activation_attribute():
    graph, h = _capture_gated((2, 64), eps=1e-5)
    assert len(graph.ops) == 1
    op = graph.ops[0]
    assert op.kind == "rms_norm_gated"
    assert op.attributes["eps"] == 1e-5
    assert op.attributes["activation"] == "sigmoid"
    assert op.inputs == ("x", "gate", "o_w")
    # the gate storage is read
    gate_sid = graph.tensor("gate").storage_id
    assert any(r.storage_id == gate_sid for r in op.read_regions)
    # numerical contract: gated formula + activation + fp32 upcast
    assert "activation(gate)" in op.numerical_contract["formula"]
    assert op.numerical_contract["activation"] == "sigmoid(g) = 1 / (1 + exp(-g))"
    assert "fp32" in op.numerical_contract["upcast"]
    assert h["y"].value.shape == (2, 64)


def test_capture_accepts_head_view_shapes():
    graph, h = _capture_gated((2, 4, 16))
    op = graph.ops[0]
    assert h["y"].value.shape == (2, 4, 16)
    assert len([d for d in check_graph(graph) if d.severity == "error"]) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gate_shape": (2, 65)},                       # gate not elementwise with x
        {"gate_shape": (2, 4, 16), "x": (2, 64)},      # rank mismatch vs x
        {"weight_shape": (65,)},                       # weight width mismatch
        {"weight_shape": (2, 64)},                     # per-row weight unsupported (broadcast contract)
    ],
)
def test_capture_gated_rejects_shape_mismatches(kwargs):
    x_shape = kwargs.pop("x", (2, 64))
    with pytest.raises(CaptureError):
        _capture_gated(x_shape, **kwargs)


def test_capture_gated_rejects_unsupported_activation():
    with pytest.raises(CaptureError, match="silu"):
        _capture_gated((2, 64), activation="silu")
    # kda_layer_norm_gated covers silu; rms_norm_gated must not silently accept it
    with pytest.raises(CaptureError):
        _capture_gated((2, 64), activation="gelu")


def test_gated_graph_is_legal_and_lowers_cleanly():
    graph, _ = _capture_gated((2, 64))
    assert [d for d in check_graph(graph) if d.severity == "error"] == []
    families = lower_graph(graph)
    assert check_thread_contract(families) == []
    assert families[0].threads == THREADS_PER_WORKER
    assert TEMPLATE_NAMES["rms_norm_gated"] == "rms_norm_gated_task"


# ===========================================================================
# Lowering (task decomposition)
# ===========================================================================


def test_lowering_matches_plain_rms_norm_domain_2d():
    rows, width = 5, 128
    graph, _ = _capture_gated((rows, width))
    fam = lower_graph(graph)[0]
    assert fam.kind == "rms_norm_gated"
    assert fam.inputs == ("x", "gate", "o_w")
    # one task per row, exactly the plain rms_norm domain
    plain_rec = RecordingBackend()
    px = plain_rec.external_tensor("x", (rows, width), storage_id=401)
    pw = plain_rec.external_tensor("o_w", (width,), storage_id=403)
    plain_rec.rms_norm(px, pw, 1e-6)
    plain = lower_graph(plain_rec.graph)[0]
    assert fam.domain.dims == plain.domain.dims
    assert fam.task_count == rows
    assert fam.params["eps"] == 1e-6 and fam.params["activation"] == "sigmoid"
    # per-task regions: x row + gate row + whole weight reads; y row write
    r = fam.reads(0)
    assert len(r) == 3
    assert r[0].boxes == ((0, 1), (0, width))
    assert r[1].boxes == ((0, 1), (0, width))
    wr = fam.writes(0)
    assert len(wr) == 1 and wr[0].boxes == ((0, 1), (0, width))
    # row 3 reads row 3's gate, not row 0's
    assert fam.reads(3)[1].boxes == ((3, 4), (0, width))


def test_lowering_head_view_domain_one_task_per_head():
    b, heads, d = 2, 7, 32
    graph, _ = _capture_gated((b, heads, d))
    fam = lower_graph(graph)[0]
    assert fam.task_count == b * heads
    assert fam.domain.dims == ((b, 1), (heads, 1))
    coords = fam.coords(9)  # task = b * heads + h -> b=1, h=2
    r = fam.reads(9)
    assert r[0].boxes == ((1, 2), (2, 3), (0, d))
    assert r[1].boxes == ((1, 2), (2, 3), (0, d))
    assert fam.writes(9)[0].boxes == ((1, 2), (2, 3), (0, d))


def test_lowering_rejects_unsupported_activation():
    graph, _ = _capture_gated((2, 64))
    from vkernels.compiler.operator_ir import OperatorGraph
    from vkernels.compiler.lowerings import lower_op

    op = graph.ops[0]
    bad = OperatorGraph()
    op2 = bad.record(
        "rms_norm_gated",
        inputs=op.inputs,
        outputs=op.outputs,
        attributes={"eps": 1e-6, "activation": "tanh"},
        reads=op.read_regions,
        writes=op.write_regions,
        source_location=op.source_location,
    )
    for name, tv in graph.tensors.items():
        bad.tensors[name] = tv
    with pytest.raises(ValueError, match="activation"):
        lower_op(op2, bad)


# ===========================================================================
# Reference execution vs the fp64 oracle (CPU)
# ===========================================================================


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("shape", [(4, 96), (2, 3, 24)])
def test_reference_gated_matches_fp64_oracle(workers, shape):
    rng = np.random.default_rng(100 + workers + sum(shape))
    x = rng.standard_normal(shape).astype(np.float32)
    gate = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal(shape[-1]).astype(np.float32)

    graph, _ = _capture_gated(shape)
    y, trace, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w}, workers=workers)

    assert trace.kernel_launches == 1
    oracle = _fp64_oracle(x, gate, w, 1e-6)
    assert not np.isnan(y).any(), "every output tile must be written exactly once"
    denom = np.abs(oracle).max()
    assert np.abs(y - oracle).max() / denom < 1e-12


def test_reference_gated_zero_gate_saturates_to_exact_zero():
    """gate <= -103: fp32 sigmoid(gate) == 0 exactly -> the row output is
    zero after the reference body's fp32 output cast. The executed schedule
    runs fp64 buffers (the fp64-oracle convention), so the saturation shows
    up as O(1e-87) residuals — anything that stops modeling the gate
    multiply (e.g. dropping it, clamping at 1e-6) lands orders of magnitude
    higher."""
    rng = np.random.default_rng(7)
    shape = (3, 48)
    x = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal(shape[-1]).astype(np.float32)
    gate = np.full(shape, -200.0, dtype=np.float32)

    graph, _ = _capture_gated(shape)
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w})
    assert np.abs(y).max() < 1e-30, f"saturated gate must zero the row; got {np.abs(y).max()}"
    # on the device the fp32 store underflows these to exactly 0.0
    assert (y.astype(np.float32) == 0.0).all()


def test_reference_gated_large_magnitude_scale_invariance():
    """RMSNorm is scale invariant: x * 1e4 must produce the same normalized
    result up to fp rounding, with extreme gates on both tails."""
    rng = np.random.default_rng(11)
    shape = (2, 64)
    x = (rng.standard_normal(shape) * 1e4).astype(np.float32)
    w = rng.standard_normal(shape[-1]).astype(np.float32)
    gate = np.where(rng.standard_normal(shape) > 0, 50.0, -50.0).astype(np.float32)

    graph, _ = _capture_gated(shape)
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w})
    oracle = _fp64_oracle(x, gate, w, 1e-6)
    denom = np.abs(oracle).max()
    assert np.abs(y - oracle).max() / denom < 1e-12
    # gates at ±50: sigmoid is 1 - 2e-22 or 2e-22 — rows are ~amplified or ~zero
    assert np.isfinite(y).all()


def test_reference_gated_epsilon_boundary():
    """eps = 0 (valid for nonzero rows) and eps = 1e6 (stats dominated)."""
    rng = np.random.default_rng(13)
    shape = (3, 32)
    x = rng.standard_normal(shape).astype(np.float32)
    gate = rng.standard_normal(shape).astype(np.float32)
    w = rng.standard_normal(shape[-1]).astype(np.float32)
    for eps in (0.0, 1e6):
        graph, _ = _capture_gated(shape, eps=eps)
        y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w})
        oracle = _fp64_oracle(x, gate, w, eps)
        denom = max(np.abs(oracle).max(), 1e-300)
        assert np.abs(y - oracle).max() / denom < 1e-12, f"eps={eps}"
        if eps == 1e6:
            # stats dominated: y ~ x/sqrt(eps) * w * sig — tiny but not zero
            assert 0 < np.abs(y).max() < 1e-2


def test_reference_gated_head_broadcast_weight_shared_across_heads():
    """[B, H, D] head view: one [D] weight broadcast over all heads, gate
    per element."""
    rng = np.random.default_rng(17)
    b, heads, d = 2, 5, 16
    x = rng.standard_normal((b, heads, d)).astype(np.float32)
    gate = rng.standard_normal((b, heads, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)

    graph, _ = _capture_gated((b, heads, d))
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w})
    oracle = _fp64_oracle(x, gate, w, 1e-6)
    assert np.abs(y - oracle).max() / np.abs(oracle).max() < 1e-12
    # per-head independence: head h's output only depends on head h's row
    x2 = x.copy()
    x2[1, 3] *= 7.0
    graph2, _ = _capture_gated((b, heads, d))
    y2, _t2, _ex2 = _run_schedule(graph2, {"x": x2, "gate": gate, "o_w": w})
    np.testing.assert_allclose(y2[0], y[0], rtol=1e-12, atol=0)
    expected_head = x2[1, 3].astype(np.float64) * np.reciprocal(np.sqrt((x2[1, 3].astype(np.float64) ** 2).mean() + 1e-6)) * w * Sigmoid(gate[1, 3])
    np.testing.assert_allclose(y2[1, 3], expected_head, rtol=1e-12, atol=0)


# ===========================================================================
# Device template mirror (CPU pin of _t_rms2d_gated / _t_rms_heads_gated)
# ===========================================================================


def _mirror_t_rms2d_gated(x, gate, w, workers, bc=None, eps=1e-6):
    """Numpy mirror of ``device_triton._t_rms2d_gated``: one task per batch
    row, worker-stride assignment, fp32 arithmetic, BC >= C vector lane
    masking (other=0.0)."""
    b, c = x.shape
    bc = bc or 1 << (c - 1).bit_length()
    y = np.zeros((b, bc), dtype=np.float32)
    wf_full = np.zeros(bc, dtype=np.float32)  # masked lanes load other=0.0
    wf_full[:c] = w.astype(np.float32)
    for worker in range(workers):  # the template's stride loop: task = worker; task += P
        task = worker
        while task < b:
            offs = np.arange(bc)
            m = offs < c
            xv = np.zeros(bc, dtype=np.float32)
            gv = np.zeros(bc, dtype=np.float32)
            xv[m] = x[task].astype(np.float32)
            gv[m] = gate[task].astype(np.float32)
            var = np.float32(np.float32(np.sum(xv * xv, dtype=np.float32)) / np.float32(c))
            sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-gv, dtype=np.float32).astype(np.float32))
            y[task] = (xv * (np.float32(1.0) / np.sqrt(var + np.float32(eps))) * wf_full * sig).astype(np.float32)
            task += workers
    return y[:, :c]


def _mirror_t_rms_heads_gated(x, gate, w, workers, eps=1e-6):
    """Numpy mirror of ``device_triton._t_rms_heads_gated``: one task per
    (batch, head), worker-stride assignment, fp32 arithmetic."""
    b, nh, d = x.shape
    y = np.empty((b, nh, d), dtype=np.float32)
    wf = w.astype(np.float32)
    for worker in range(workers):  # the template's stride loop
        task = worker
        while task < b * nh:
            bb, h = task // nh, task % nh
            xv = x[bb, h].astype(np.float32)
            gv = gate[bb, h].astype(np.float32)
            var = np.float32(np.float32(np.sum(xv * xv, dtype=np.float32)) / np.float32(d))
            sig = np.float32(1.0) / (np.float32(1.0) + np.exp(-gv, dtype=np.float32).astype(np.float32))
            y[bb, h] = (xv * (np.float32(1.0) / np.sqrt(var + np.float32(eps))) * wf * sig).astype(np.float32)
            task += workers
    return y


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("c,mask", [(64, False), (48, True)])  # pow2 lane count vs masked tail
def test_template_mirror_matches_reference_executor_2d(workers, c, mask):
    b = 3
    rng = np.random.default_rng(500 + c + workers)
    x = rng.standard_normal((b, c)).astype(np.float32)
    gate = rng.standard_normal((b, c)).astype(np.float32)
    w = rng.standard_normal(c).astype(np.float32)

    graph, _ = _capture_gated((b, c))
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w}, workers=workers)

    mirror = _mirror_t_rms2d_gated(x, gate, w, workers)
    rel = np.abs(mirror.astype(np.float64) - y.astype(np.float64)).max() / np.abs(y).max()
    assert rel < 1e-5, f"template mirror diverged from reference: {rel:.3e}"
    assert not np.isnan(y).any()


@pytest.mark.parametrize("workers", [1, 4])
def test_template_mirror_matches_reference_executor_heads(workers):
    b, heads, d = 2, 5, 32
    rng = np.random.default_rng(600 + workers)
    x = rng.standard_normal((b, heads, d)).astype(np.float32)
    gate = rng.standard_normal((b, heads, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)

    graph, _ = _capture_gated((b, heads, d))
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w}, workers=workers)

    mirror = _mirror_t_rms_heads_gated(x, gate, w, workers)
    rel = np.abs(mirror.astype(np.float64) - y.astype(np.float64)).max() / np.abs(y).max()
    assert rel < 1e-5, f"template mirror diverged from reference: {rel:.3e}"


def test_template_mirror_bf16_input_fp32_stat_contract():
    """The strict-fp32 contract at bf16 input: bf16 x/gate upcast before the
    row reduction (stats on the bf16 grid, fp32 accumulator). The fp64
    reference executed on the same bf16-rounded values must agree with the
    fp32 mirror within fp32 noise (<1e-5), and BOTH must differ from an
    oracle fed the pre-rounding fp32 values by more than bf16 rounding —
    pinning that stats are computed on the stored (bf16) values."""
    rng = np.random.default_rng(97)
    b, c = 2, 64
    x_hi = rng.standard_normal((b, c))
    gate_hi = rng.standard_normal((b, c))
    w = rng.standard_normal(c).astype(np.float32)
    x = _bf16_round(x_hi).astype(np.float32)
    gate = _bf16_round(gate_hi).astype(np.float32)

    graph, _ = _capture_gated((b, c))
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w})

    mirror = _mirror_t_rms2d_gated(x, gate, w, workers=2)
    rel = np.abs(mirror.astype(np.float64) - y.astype(np.float64)).max() / np.abs(y).max()
    assert rel < 1e-5

    # oracle on the stored (bf16-grid) values matches exactly
    oracle_stored = _fp64_oracle(x, gate, w, 1e-6)
    assert np.abs(y - oracle_stored).max() / np.abs(oracle_stored).max() < 1e-12
    # an oracle on the pre-rounding values would be a different contract
    oracle_hi = _fp64_oracle(x_hi, gate_hi, w, 1e-6)
    assert np.abs(oracle_hi - oracle_stored).max() / np.abs(oracle_stored).max() > 1e-6


# ===========================================================================
# NaN canaries + schedule accounting
# ===========================================================================


@pytest.mark.parametrize("workers", [1, 2, 5])
def test_nan_canaries_and_worker_stride_coverage(workers):
    """Workspace pre-filled with NaN: any unwritten output element or
    unsanitized scratch reuse trips the canary."""
    rng = np.random.default_rng(31 + workers)
    b, c = 6, 96
    x = rng.standard_normal((b, c)).astype(np.float32)
    gate = rng.standard_normal((b, c)).astype(np.float32)
    w = rng.standard_normal(c).astype(np.float32)

    graph, _ = _capture_gated((b, c))
    y, trace, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w}, workers=workers)
    assert np.isfinite(y).all(), "canary tripped: unwritten/NaN outputs"
    oracle = _fp64_oracle(x, gate, w, 1e-6)
    assert np.abs(y - oracle).max() / np.abs(oracle).max() < 1e-12
    assert trace.grid_barriers == 1  # single-phase single-launch slice


# ===========================================================================
# floe eager parity (torch + floe only; skipped on the vkernels-only venv)
# ---------------------------------------------------------------------------

def _floe_glm_o_norm():
    if "floe" not in sys.modules:
        from pathlib import Path

        for cand in (os.environ.get("FLOE_ROOT"), "/home/xiayao/Documents/projects/opentela-ai/serving-stack/floe"):
            if cand and (Path(cand) / "floe" / "engine").is_dir():
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                break
    try:
        from floe.engine.runner.models.glm5.glm5_arch import Glm53RMSNormGated
    except ModuleNotFoundError as exc:  # floe not on this stack
        pytest.skip(f"floe glm5 oracle unavailable: {exc}")
    torch = pytest.importorskip("torch")
    torch.manual_seed(100)
    norm = Glm53RMSNormGated(64, eps=1e-6)
    return norm, torch


def test_eager_glm53_rms_norm_gated_parity_bf16_fp32_stats():
    """Eager floe ``Glm53RMSNormGated`` parity incl. the fp32-stat cast at
    bf16 input: bf16 x/gate -> fp32 math -> bf16 out. The compiler slice
    models the fp32 math exactly; bf16 output rounding is applied on top."""
    norm, torch = _floe_glm_o_norm()
    rng = np.random.default_rng(41)
    b, c = 2, 64
    x_bf16 = torch.from_numpy(_bf16_round(rng.standard_normal((b, c)))).to(torch.bfloat16)
    gate_bf16 = torch.from_numpy(_bf16_round(rng.standard_normal((b, c)))).to(torch.bfloat16)
    w_bf16 = torch.from_numpy(_bf16_round(rng.standard_normal(c))).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(w_bf16)
        expected = norm(x_bf16, gate_bf16)  # bf16 out, fp32 stats

    x = x_bf16.float().numpy().astype(np.float32)
    gate = gate_bf16.float().numpy().astype(np.float32)
    w = w_bf16.float().numpy().astype(np.float32)
    graph, _ = _capture_gated((b, c))
    y, _t, _ex = _run_schedule(graph, {"x": x, "gate": gate, "o_w": w})

    # compiled fp32 output vs eager bf16 output: identical modulo the final
    # bf16 rounding step (relative tolerance on the bf16 grid ~2^-8)
    expected_f = expected.float().numpy()
    rel = np.abs(y.astype(np.float64) - expected_f.astype(np.float64)).max() / np.abs(expected_f).max()
    assert rel < 1e-2, f"eager o_norm parity: {rel:.3e}"
    # and vs the eager fp32 (pre-cast) values the slice must be tighter
    with torch.no_grad():
        xf = x_bf16.float()
        fp32_ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight.float() * torch.sigmoid(gate_bf16.float())).numpy()
    rel32 = np.abs(y.astype(np.float64) - fp32_ref.astype(np.float64)).max() / np.abs(fp32_ref).max()
    assert rel32 < 1e-5, f"eager fp32-stat parity: {rel32:.3e}"
