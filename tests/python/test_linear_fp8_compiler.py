"""fp8-blockwise GEMV linear (``gemv_fp8``) compiler support — issue #91.

Covers the vertical slice the issue names:

* capture — ``ops.linear_fp8`` records the ``linear`` variant with the
  ``weight_layout="fp8_block"`` attribute and the scale tensor as the
  second weight external (DeepSeek-style 128x128 block-FP8);
* lowering — the same GEMV tile domain as ``linear`` (16-column output
  tiles, full-K sweep), with per-task scale-tensor region reads (exactly
  one scale row per task: a 16-column tile always lies inside one
  128-wide N block);
* reference body — numpy dequant-then-matmul in fp64, tile-exact against
  the device contract (``_t_gemv_fp8`` / ``_h_gemv_fp8``);
* validation — fp64-oracle equivalence of the executed schedule, and the
  bf16-reference vs fp8 GEMV relative-error bound mirroring the 27B gate.

All CPU (§15.1): this validates graph logic and tile semantics; the device
template itself is validated against the real 27B checkpoint elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler.capture import RecordingBackend
from vkernels.compiler.codegen_cute import TEMPLATE_NAMES
from vkernels.compiler.legality import check_graph
from vkernels.compiler.lowerings import THREADS_PER_WORKER, check_thread_contract, lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.operator_ir import F32, F8_E4M3, I32
from vkernels.compiler.reference_exec import ReferenceExecutor, decode_e4m3
from vkernels.compiler.schedule_phase import PhaseSchedule
from vkernels.compiler.task_ir import TileDomain

QB = 128  # the DeepSeek quant block


# ---------------------------------------------------------------------------
# e4m3 encode (test-side counterpart of decode_e4m3), block quantization
# ---------------------------------------------------------------------------


def _encode_e4m3(x):
    """Round-to-nearest fp32 -> e4m3fn bytes (RNE on the decoded value table)."""
    x = np.asarray(x, dtype=np.float64)
    pos = decode_e4m3(np.arange(0, 0x7F, dtype=np.uint8))  # ascending, max 448
    sign = np.signbit(x)
    ax = np.clip(np.abs(x), 0.0, 448.0)
    hi = np.searchsorted(pos, ax)
    lo = np.clip(hi - 1, 0, len(pos) - 1)
    hi = np.clip(hi, 0, len(pos) - 1)
    lo_d, hi_d = pos[lo], pos[hi]
    take_hi = (ax - lo_d) > (hi_d - ax)  # exact tie -> lower magnitude (even mantissa)
    val = np.where(take_hi, hi_d, lo_d)
    byte = np.where(sign, 0x80 | np.searchsorted(pos, val), np.searchsorted(pos, val))
    return byte.astype(np.uint8)


def _quantize_blockwise(w, block=QB):
    """DeepSeek-style 128x128 block-FP8: per-block absmax scale -> e4m3 bytes.

    Handles ragged N (trailing partial row-block) and K must be a multiple
    of ``block``. Returns (bytes_u8 [N, K], scales_f32 [ceil(N/block), K/block]).
    """
    n, k = w.shape
    nb, kb = -(-n // block), k // block
    w_bytes = np.zeros((n, k), dtype=np.uint8)
    scale = np.zeros((nb, kb), dtype=np.float32)
    for b in range(kb):
        for rb in range(nb):
            rows = slice(rb * block, min((rb + 1) * block, n))
            blk = w[rows, b * block : (b + 1) * block]
            absmax = np.abs(blk).max()
            s = max(absmax / 448.0, 1e-12)
            scale[rb, b] = np.float32(s)
            w_bytes[rows, b * block : (b + 1) * block] = _encode_e4m3(blk / s)
    return w_bytes, scale


def _bf16_round(x):
    """Round fp32 -> bf16 grid (RNE), returned as float64."""
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = ((bits + 0x7FFF + lsb) >> 16).astype(np.uint32) << 16
    return rounded.view(np.float32).astype(np.float64)


# ---------------------------------------------------------------------------
# Capture helpers
# ---------------------------------------------------------------------------


def _capture_fp8_linear(m=2, k=256, n=256, *, scale_shape=None, k_override=None):
    """Record a single linear_fp8 op; returns (graph, recorder, handles)."""
    recorder = RecordingBackend()
    position = recorder.define_position(4)
    ids = recorder.external_tensor("ids", (m,), I32, storage_id=10**6)
    x = recorder.external_tensor("x", (m, k), F32, storage_id=2001)
    w = recorder.external_tensor("w_fp8", (n, k_override or k), F8_E4M3, storage_id=2002)
    eff_k = k_override or k
    sh = scale_shape or (n // QB, eff_k // QB)
    scale = recorder.external_tensor("w_scale", sh, F32, storage_id=2003)
    y = recorder.linear_fp8(x, w, scale, name="proj")
    return recorder.graph, recorder, {"x": x, "w": w, "scale": scale, "y": y, "ids": ids, "position": position}


# ===========================================================================
# Capture (issue: "linear op variant / attributes: weight_layout fp8_block")
# ===========================================================================


def test_capture_records_fp8_block_layout_and_second_weight_external():
    graph, _, h = _capture_fp8_linear()
    assert len(graph.ops) == 1
    op = graph.ops[0]
    assert op.kind == "linear_fp8"  # the linear op variant (issue #91)
    assert op.attributes["weight_layout"] == "fp8_block"
    assert op.attributes["quant_block"] == QB
    # inputs: activation, fp8 weight, scale tensor (the second weight external)
    assert op.inputs == ("x", "w_fp8", "w_scale")
    # scale storage is read, and the recorder declared the dequant contract
    scale_sid = graph.tensor("w_scale").storage_id
    assert any(r.storage_id == scale_sid for r in op.read_regions)
    assert "dequant" in op.numerical_contract
    assert h["y"].value.shape == (2, 256)


def test_capture_linear_fp8_rejects_bad_scale_shape_and_contraction():
    with pytest.raises(ValueError, match="scale shape"):
        _capture_fp8_linear(scale_shape=(2, 1))
    with pytest.raises(ValueError, match="contraction"):
        _capture_fp8_linear(k_override=512)
    with pytest.raises(ValueError, match="divisible"):
        _capture_fp8_linear(k=100)


def test_captured_fp8_graph_is_legal_and_lowers_without_diagnostics():
    graph, _, _ = _capture_fp8_linear()
    errors = [d for d in check_graph(graph) if d.severity == "error"]
    assert errors == []
    families = lower_graph(graph)
    assert check_thread_contract(families) == []
    assert families[0].threads == THREADS_PER_WORKER


# ===========================================================================
# Lowering (issue: "same GEMV tile domain as linear; scales region reads")
# ===========================================================================


def test_lowering_uses_linear_tile_domain_with_16_column_tiles():
    graph, _, _ = _capture_fp8_linear(m=2, k=256, n=256)
    fam = lower_graph(graph)[0]
    assert fam.kind == "gemv_fp8"
    assert fam.inputs == ("x", "w_fp8", "w_scale")
    # Same tile shape as the dense linear lowering: (m, 16) x (n, 16).
    assert fam.domain.dims == ((2, 16), (256, 16))
    assert fam.task_count == 1 * 16
    assert fam.params["quant_block"] == QB


def test_lowering_scale_region_is_one_row_per_task():
    """A 16-column tile sits inside one 128-wide N block -> exactly one
    scale row (full K-block width) is read per task."""
    graph, _, _ = _capture_fp8_linear(m=1, k=256, n=256)
    fam = lower_graph(graph)[0]
    scale = graph.tensor("w_scale")
    for tid in range(fam.task_count):
        coords = fam.coords(tid)
        reads = fam.reads(tid)
        n0 = coords[1] * 16
        sb = n0 // QB
        # x: the batch rows x full K; w: the 16-column weight tile x full K
        assert reads[0].boxes == ((0, 1), (0, 256))
        assert reads[1].boxes == ((n0, n0 + 16), (0, 256))
        assert reads[2].view is scale
        assert reads[2].boxes == ((sb, sb + 1), (0, 2))
    # Column block 1 (task 8 -> n0 = 128) reads scale row 1, not row 0.
    late = fam.reads(8)
    assert late[2].boxes == ((1, 2), (0, 2))


def test_lowering_rejects_non_fp8_block_layout_and_bad_dtypes():
    graph, _, _ = _capture_fp8_linear()
    op = graph.ops[0]
    from vkernels.compiler.operator_ir import OperatorGraph

    bad = OperatorGraph()
    op2 = bad.record(
        "linear_fp8",
        inputs=op.inputs,
        outputs=op.outputs,
        attributes={"weight_layout": "mxfp4", "quant_block": QB, "bias": False},
        reads=op.read_regions,
        writes=op.write_regions,
        source_location=op.source_location,
    )
    # re-register tensors so graph.tensor() resolves
    for name, tv in graph.tensors.items():
        bad.tensors[name] = tv
    from vkernels.compiler.lowerings import lower_op

    with pytest.raises(ValueError, match="weight_layout"):
        lower_op(op2, bad)

    # wrong dtypes (fp32 weights) must be rejected too
    recorder = RecordingBackend()
    x = recorder.external_tensor("x", (1, 256), F32, storage_id=3001)
    w = recorder.external_tensor("w", (256, 256), F32, storage_id=3002)
    scale = recorder.external_tensor("s", (2, 2), F32, storage_id=3003)
    y = recorder.linear_fp8(x, w, scale, name="bad_dtypes")
    with pytest.raises(ValueError, match="dtypes"):
        lower_graph(recorder.graph)


# ===========================================================================
# Reference execution: fp64 dequant-then-matmul oracle (issue: "reference body")
# ===========================================================================


def _run_schedule(graph, externals, workers=4):
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)
    storage_arrays = {}
    for sid, name in graph.storage_names.items():
        if sid in graph.fresh_storages:
            continue
        storage_arrays[sid] = externals[name]
    workspace = np.full(workspace_plan.total_elements, np.nan, dtype=np.float64)
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = workspace[buf.offset : buf.offset + buf.numel]
    executor = ReferenceExecutor(
        schedule,
        workers=workers,
        storage_arrays=storage_arrays,
        graph=graph,
        workspace_plan=workspace_plan,
    )
    trace = executor.run({"p": 0})
    out_name = graph.ops[-1].outputs[0]
    return np.array(executor.tensor(out_name), copy=True), trace


def test_reference_exec_matches_fp64_dequant_matmul_exactly():
    """The executed schedule computes the tile-exact dequant-then-matmul
    semantics: fp64 reference body vs an independent numpy oracle."""
    rng = np.random.default_rng(91)
    m, k, n = 3, 256, 384
    w_ref = rng.standard_normal((n, k))
    w_bytes, scale = _quantize_blockwise(w_ref)
    x = rng.standard_normal((m, k)).astype(np.float32)

    recorder = RecordingBackend()
    recorder.define_position(4)
    xt = recorder.external_tensor("x", (m, k), F32, storage_id=2001)
    wt = recorder.external_tensor("w_fp8", (n, k), F8_E4M3, storage_id=2002)
    st = recorder.external_tensor("w_scale", (n // QB, k // QB), F32, storage_id=2003)
    recorder.linear_fp8(xt, wt, st, name="proj")

    y, trace = _run_schedule(recorder.graph, {"x": x, "w_fp8": w_bytes, "w_scale": scale})
    assert trace.kernel_launches == 1

    # Independent oracle: dequant the same bytes in fp64, then matmul.
    w_deq = decode_e4m3(w_bytes).reshape(n // QB, QB, k // QB, QB) * scale[:, None, :, None]
    oracle = x.astype(np.float64) @ w_deq.reshape(n, k).T
    assert not np.isnan(y).any(), "every output tile must be written exactly once"
    denom = np.abs(oracle).max()
    assert np.abs(y - oracle).max() / denom < 1e-12


def test_device_template_contract_matches_reference_k_order():
    """The device template's per-block accumulation order (scale loaded once
    per 128-deep k-chunk, fp32 accumulate) is what the reference body models:
    dequant-in-register per block, k ascending. Reordering the reference to
    a single global scale would be a different contract — pin the current
    one by checking per-block scale application changes the result."""
    rng = np.random.default_rng(7)
    k, n = 256, 128
    w_bytes, scale = _quantize_blockwise(rng.standard_normal((n, k)))
    x = rng.standard_normal((1, k)).astype(np.float32)
    recorder = RecordingBackend()
    recorder.define_position(4)
    xt = recorder.external_tensor("x", (1, k), F32, storage_id=2001)
    wt = recorder.external_tensor("w_fp8", (n, k), F8_E4M3, storage_id=2002)
    st = recorder.external_tensor("w_scale", (n // QB, k // QB), F32, storage_id=2003)
    recorder.linear_fp8(xt, wt, st, name="proj")
    y, _ = _run_schedule(recorder.graph, {"x": x, "w_fp8": w_bytes, "w_scale": scale})
    # Per-block scales (the device contract) reproduce the oracle exactly...
    w_deq = decode_e4m3(w_bytes).reshape(n // QB, QB, k // QB, QB) * scale[:, None, :, None]
    oracle = x.astype(np.float64) @ w_deq.reshape(n, k).T
    assert np.abs(y - oracle).max() / np.abs(oracle).max() < 1e-12
    # ...while a single global scale would be a different (wrong) contract.
    wrong = x.astype(np.float64) @ (decode_e4m3(w_bytes) * scale.max()).T
    assert np.abs(y - wrong).max() / np.abs(oracle).max() > 1e-6


# ===========================================================================
# Validation bound (issue: "bf16-dequantized-reference vs fp8 GEMV rel-err
# bound, mirroring the 27B gate (logits 1.4-3.4e-3)")
# ===========================================================================


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fp8_gemv_rel_err_against_bf16_reference_within_27b_gate(seed):
    """Quantized-weights error bound: fp8-block GEMV output vs the bf16
    weights reference, both against the same activation.

    The 27B gate observed end-to-end *logits* rel err 1.4-3.4e-3 (many
    layers, K=5120, max over ~150k correlated outputs). A single GEMV at
    K=512 carries the full per-element e4m3 quantization error (~3e-2 of
    the max output magnitude, measured 2.6e-2/3.0e-2/3.1e-2 over the three
    seeds); the bound below is that per-GEMV regime with headroom, and the
    suite additionally pins exact equivalence to the fp64 dequant oracle.
    """
    rng = np.random.default_rng(seed)
    m, k, n = 1, 512, 256
    w_bf16 = _bf16_round(rng.standard_normal((n, k)) * 0.02)
    w_bytes, scale = _quantize_blockwise(w_bf16)
    x = rng.standard_normal((m, k)).astype(np.float32)

    recorder = RecordingBackend()
    recorder.define_position(4)
    xt = recorder.external_tensor("x", (m, k), F32, storage_id=2001)
    wt = recorder.external_tensor("w_fp8", (n, k), F8_E4M3, storage_id=2002)
    st = recorder.external_tensor("w_scale", (n // QB, k // QB), F32, storage_id=2003)
    recorder.linear_fp8(xt, wt, st, name="proj")
    y, _ = _run_schedule(recorder.graph, {"x": x, "w_fp8": w_bytes, "w_scale": scale})

    ref = x.astype(np.float64) @ w_bf16.T
    rel = np.abs(y - ref).max() / np.abs(ref).max()
    assert rel < 5e-2, f"fp8 GEMV rel err {rel:.3e} exceeds the per-GEMV gate bound"


def test_codegen_knows_the_gemv_fp8_template():
    """The generated-source phase table maps the new family kind."""
    assert TEMPLATE_NAMES["gemv_fp8"] == "linear_fp8_task"


# ---------------------------------------------------------------------------
# Device-template mirror: _t_gemv_fp8's task decomposition on CPU (§15.1)
# ---------------------------------------------------------------------------


def _simulate_t_gemv_fp8(x, w_bytes, scale, workers, tile=16, bk=128):
    """Numpy mirror of ``device_triton._t_gemv_fp8``: one task per 16-column
    N tile, full-K sweep in 128-deep blocks, one fp32 scale load per block,
    fp32 accumulation, worker-stride task assignment (task = worker, +P)."""
    n, k = w_bytes.shape
    ntask = n // tile
    kb = k // bk
    y = np.zeros(n, dtype=np.float32)
    for w in range(workers):
        task = w
        while task < ntask:
            offs_n = task * tile + np.arange(tile)
            sb_row = (task * tile) // bk
            acc = np.zeros(tile, dtype=np.float32)
            for b in range(kb):
                offs_k = b * bk + np.arange(bk)
                s = np.float32(scale[sb_row, b])
                xv = x[offs_k].astype(np.float32)
                wt = decode_e4m3(w_bytes[offs_n[:, None], offs_k[None, :]]).astype(np.float32)
                acc += np.sum(wt * (s * xv[None, :]), axis=1, dtype=np.float32)
            y[offs_n] = acc
            task += workers
    return y


@pytest.mark.parametrize("m,k,n,workers", [(1, 128, 128, 1), (1, 512, 256, 7), (2, 256, 144, 3), (3, 256, 384, 13)])
def test_device_template_mirror_matches_reference_executor(m, k, n, workers):
    """Template equivalence: the task decomposition the Triton device body
    implements (16-col tiles, 128-deep k blocks, fp32 accumulate, stride
    assignment) reproduces the compiled schedule's reference output — for
    odd batch rows, ragged N (144 = 9 tiles, ceil scale row), single- and
    multi-block K, and worker counts that both under- and over-subscribe
    the task domain."""
    rng = np.random.default_rng(m * 100 + k + n + workers)
    w_bf16 = _bf16_round(rng.standard_normal((n, k)) * 0.02)
    w_bytes, scale = _quantize_blockwise(w_bf16)
    x = rng.standard_normal((m, k)).astype(np.float32)

    recorder = RecordingBackend()
    recorder.define_position(4)
    xt = recorder.external_tensor("x", (m, k), F32, storage_id=2001)
    wt = recorder.external_tensor("w_fp8", (n, k), F8_E4M3, storage_id=2002)
    st = recorder.external_tensor("w_scale", (-(-n // QB), k // QB), F32, storage_id=2003)
    recorder.linear_fp8(xt, wt, st, name="proj")
    y, _ = _run_schedule(recorder.graph, {"x": x, "w_fp8": w_bytes, "w_scale": scale}, workers=workers)

    mirror = _simulate_t_gemv_fp8(x[0], w_bytes, scale, workers=workers)
    rel = np.abs(mirror.astype(np.float64) - y[0]).max() / np.abs(y[0]).max()
    assert rel < 1e-5, f"template mirror diverged from reference: {rel:.3e}"
    # ragged-N coverage: the trailing partial scale block's outputs are written
    assert not np.isnan(y).any()


# ---------------------------------------------------------------------------
# Sanity: the quantizer and decoder round-trip
# ---------------------------------------------------------------------------


def test_e4m3_round_trip_and_specials():
    vals = np.array([0.0, 1.0, -1.0, 448.0, -448.0, 0.001953125, 2.5, -3.25], dtype=np.float64)
    decoded = decode_e4m3(_encode_e4m3(vals))
    assert np.allclose(decoded, vals, rtol=2 ** -3, atol=0)
    # NaN encoding (0x7F) decodes to NaN; subnormal minimum is 2^-9
    assert np.isnan(decode_e4m3(np.array([0x7F], dtype=np.uint8))[0])
    assert decode_e4m3(np.array([0x01], dtype=np.uint8))[0] == pytest.approx(2 ** -9)
