"""Issue #88: `rope` op with partial rotation (rotary_dim) + NeoX convention.

Validation chain (mirrors the issue's Validation section):

* capture: the recorder's ``rope(...)`` gains ``rotary_dim`` / ``convention``
  parameters; the Qwen3 default capture is unchanged (full-width
  ``rotate_half``);
* reference_exec: the NeoX partial body (split-half over the first
  ``rotary_dim`` dims, pass-through tail) is validated against the floe
  oracle — ``floe/engine/runner/models/qwen35/qwen35_arch.py``'s
  ``PartialRotaryEmbedding.apply`` — on a tiny config (CPU);
* convention equivalence: at ``rotary_dim == head_dim`` the NeoX partial form
  agrees with the full-width rotate_half semantics (duplicated tables);
* device: the generic Triton template ``_t_rope`` (ported from the
  27B-validated ``_h_rope_append``) matches the same oracle on CUDA; skipped
  on CPU-only stacks (the full-model GPU paths in test_megakernel_triton.py
  already cover the ``ROT=D`` call).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vkernels.compiler.capture import CaptureError, RecordingBackend  # noqa: E402
from vkernels.compiler.lowerings import lower_graph  # noqa: E402
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402


# ---------------------------------------------------------------------------
# floe oracle: PartialRotaryEmbedding (qwen35). The floe package imports
# kvaas_runtime at its top level (KV residency); that native module is
# irrelevant to the RoPE math, so stub it before importing.
# ---------------------------------------------------------------------------

def _floe_partial_rope():
    if "kvaas_runtime" not in sys.modules:
        kv = types.ModuleType("kvaas_runtime")
        for name in ("CudaVmm", "ElasticControlSession", "ManagedResidencyAdmissionDeferred", "ManagedResidencySession", "allocate_device_pool"):
            setattr(kv, name, object)
        sub = types.ModuleType("kvaas_runtime.kv_pool_import")
        sub.tensor_from_cuda_pointer = object
        kv.kv_pool_import = sub
        sys.modules["kvaas_runtime"] = kv
        sys.modules["kvaas_runtime.kv_pool_import"] = sub
    try:
        from floe.engine.runner.models.qwen35.qwen35_arch import PartialRotaryEmbedding
    except ModuleNotFoundError as exc:  # floe not on this stack (vkernels-only venv)
        pytest.skip(f"floe qwen35 oracle unavailable: {exc}")

    return PartialRotaryEmbedding


def _neox_tables(rotary_dim: int, max_pos: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """fp32 cos/sin [max_pos, rotary_dim//2] (floe PartialRotaryEmbedding._build)."""
    inv_freq = theta ** (-np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
    freqs = np.outer(np.arange(max_pos, dtype=np.float64), inv_freq)  # [S, rot/2]
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


# Tiny config (issue: rotary_dim 64 of head_dim 256, scaled down for CPU).
HEAD_DIM = 16
ROTARY_DIM = 8
MAX_POS = 32
THETA = 1e6


# ---------------------------------------------------------------------------
# Compiler plumbing: capture -> lower -> schedule -> reference execution of a
# single rope op, exactly as compile_model wires the pieces (§13).
# ---------------------------------------------------------------------------

def _compile_and_run_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, position: int, *, rotary_dim=None, convention="rotate_half", workers=3) -> np.ndarray:
    B, H, D = x.shape
    recorder = RecordingBackend()
    pos = recorder.define_position(MAX_POS)
    xt = recorder.external_tensor("x", (B, H, D), storage_id=101)
    ct = recorder.external_tensor("cos", cos.shape, storage_id=102)
    st = recorder.external_tensor("sin", sin.shape, storage_id=103)
    out = recorder.rope(xt, ct, st, pos, layer=0, which="q", rotary_dim=rotary_dim, convention=convention)
    graph = recorder.graph

    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)

    storage_arrays = {
        101: x.astype(np.float32).reshape(-1),
        102: cos.astype(np.float32).reshape(-1),
        103: sin.astype(np.float32).reshape(-1),
    }
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(schedule, workers=workers, storage_arrays=storage_arrays, graph=graph, workspace_plan=workspace_plan)
    executor.run({"p": position})
    return np.array(executor.tensor(out.value.name), copy=True)


# ===========================================================================
# Capture (recorder parameters; Qwen3 default unchanged)
# ===========================================================================

def test_capture_rope_default_is_full_width_rotate_half():
    recorder = RecordingBackend()
    pos = recorder.define_position(MAX_POS)
    x = recorder.external_tensor("x", (1, 4, HEAD_DIM), storage_id=201)
    cos = recorder.external_tensor("cos", (MAX_POS, HEAD_DIM), storage_id=202)
    sin = recorder.external_tensor("sin", (MAX_POS, HEAD_DIM), storage_id=203)
    recorder.rope(x, cos, sin, pos, layer=0, which="q")
    op = recorder.graph.ops[-1]
    assert op.kind == "rope"
    assert op.attributes["convention"] == "rotate_half"
    assert "rotary_dim" not in op.attributes
    assert op.numerical_contract["rotate_half"] == "cat(-x[D/2:], x[:D/2])"


def test_capture_rope_neox_partial_records_attributes():
    recorder = RecordingBackend()
    pos = recorder.define_position(MAX_POS)
    x = recorder.external_tensor("x", (1, 4, HEAD_DIM), storage_id=211)
    cos = recorder.external_tensor("cos", (MAX_POS, ROTARY_DIM // 2), storage_id=212)
    sin = recorder.external_tensor("sin", (MAX_POS, ROTARY_DIM // 2), storage_id=213)
    recorder.rope(x, cos, sin, pos, layer=2, which="k", rotary_dim=ROTARY_DIM, convention="neox_partial")
    op = recorder.graph.ops[-1]
    assert op.attributes["convention"] == "neox_partial"
    assert op.attributes["rotary_dim"] == ROTARY_DIM
    assert op.attributes["which"] == "k"
    assert "pass_through" in op.numerical_contract


@pytest.mark.parametrize("kwargs", [
    {"convention": "neox_partial", "rotary_dim": None},
    {"convention": "neox_partial", "rotary_dim": 7},  # odd
    {"convention": "neox_partial", "rotary_dim": HEAD_DIM + 4},  # > head_dim
    {"convention": "gneox"},  # unknown convention
])
def test_capture_rope_rejects_invalid_conventions(kwargs):
    recorder = RecordingBackend()
    pos = recorder.define_position(MAX_POS)
    x = recorder.external_tensor("x", (1, 1, HEAD_DIM), storage_id=221)
    cos = recorder.external_tensor("cos", (MAX_POS, ROTARY_DIM // 2), storage_id=222)
    sin = recorder.external_tensor("sin", (MAX_POS, ROTARY_DIM // 2), storage_id=223)
    with pytest.raises(CaptureError):
        recorder.rope(x, cos, sin, pos, layer=0, which="q", **kwargs)


def test_capture_rope_neox_rejects_wrong_table_width():
    recorder = RecordingBackend()
    pos = recorder.define_position(MAX_POS)
    x = recorder.external_tensor("x", (1, 1, HEAD_DIM), storage_id=231)
    cos = recorder.external_tensor("cos", (MAX_POS, HEAD_DIM), storage_id=232)  # full-width
    sin = recorder.external_tensor("sin", (MAX_POS, HEAD_DIM), storage_id=233)
    with pytest.raises(CaptureError):
        recorder.rope(x, cos, sin, pos, layer=0, which="q", rotary_dim=ROTARY_DIM, convention="neox_partial")


# ===========================================================================
# Reference execution vs the floe oracle (CPU, tiny config)
# ===========================================================================

@pytest.mark.parametrize("position", [0, 1, 3, 17, 31])
@pytest.mark.parametrize("workers", [1, 3])
def test_reference_rope_neox_matches_floe_oracle(position, workers):
    PartialRotaryEmbedding = _floe_partial_rope()
    rng = np.random.default_rng(position * 7 + workers)
    x = rng.standard_normal((2, 5, HEAD_DIM)).astype(np.float32)
    cos, sin = _neox_tables(ROTARY_DIM, MAX_POS, THETA)

    rope = PartialRotaryEmbedding(HEAD_DIM, ROTARY_DIM, MAX_POS, THETA, torch.device("cpu"), torch.float32)
    rope._build(torch.device("cpu"))
    q_ref, _ = rope.apply(torch.from_numpy(x), torch.zeros_like(torch.from_numpy(x)), position)
    expected = q_ref.numpy()

    got = _compile_and_run_rope(x, cos, sin, position, rotary_dim=ROTARY_DIM, convention="neox_partial", workers=workers)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


def test_reference_rope_neox_pass_through_tail_is_untouched():
    """Dims [rotary_dim, head_dim) must be a bit-exact copy of the input."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, HEAD_DIM)).astype(np.float32)
    cos, sin = _neox_tables(ROTARY_DIM, MAX_POS, THETA)
    got = _compile_and_run_rope(x, cos, sin, 11, rotary_dim=ROTARY_DIM, convention="neox_partial")
    np.testing.assert_array_equal(got[..., ROTARY_DIM:], x[..., ROTARY_DIM:])


def test_reference_rope_neox_full_width_agrees_with_rotate_half():
    """At rotary_dim == head_dim the NeoX split-half form equals the
    rotate_half semantics (with duplicated cat([f, f]) tables)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((1, 2, HEAD_DIM)).astype(np.float32)
    inv_freq = THETA ** (-np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)
    freqs = np.outer(np.arange(MAX_POS, dtype=np.float64), inv_freq)
    half_tables_cos = np.cos(freqs).astype(np.float32)  # [S, D/2]
    half_tables_sin = np.sin(freqs).astype(np.float32)
    # rotate_half body: full-width duplicated tables
    cos_dup = np.concatenate([half_tables_cos, half_tables_cos], axis=-1)
    sin_dup = np.concatenate([half_tables_sin, half_tables_sin], axis=-1)
    got_neox = _compile_and_run_rope(x, half_tables_cos, half_tables_sin, 9, rotary_dim=HEAD_DIM, convention="neox_partial")
    got_rh = _compile_and_run_rope(x, cos_dup, sin_dup, 9, convention="rotate_half")
    np.testing.assert_allclose(got_neox, got_rh, rtol=1e-6, atol=1e-7)


# ===========================================================================
# Device template (_t_rope, ported from _h_rope_append) — CUDA only
# ===========================================================================

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA (the ROT=D path is covered by the full-model tests in test_megakernel_triton.py)",
)


@gpu
def test_device_t_rope_partial_matches_floe_oracle():
    pytest.importorskip("triton")
    from vkernels.compiler.device_triton import _t_rope

    PartialRotaryEmbedding = _floe_partial_rope()
    dev = torch.device("cuda")
    rng = np.random.default_rng(5)
    B, H, D = 2, 3, HEAD_DIM
    x = torch.from_numpy(rng.standard_normal((B, H, D)).astype(np.float32)).to(dev).to(torch.bfloat16)
    positions = torch.tensor([4, 11], device=dev, dtype=torch.int32)
    cos_np, sin_np = _neox_tables(ROTARY_DIM, MAX_POS, THETA)
    cos = torch.from_numpy(cos_np).to(dev)
    sin = torch.from_numpy(sin_np).to(dev)

    y = torch.empty_like(x)
    _t_rope[(4,)](x, cos, sin, positions, y, B, H, D, ROT=ROTARY_DIM, TSTRIDE=ROTARY_DIM // 2)

    rope = PartialRotaryEmbedding(D, ROTARY_DIM, MAX_POS, THETA, dev, torch.float32)
    rope._build(dev)
    # per-row positions: build the expected rows individually
    expected = torch.stack([rope.apply(x[b:b + 1].float(), torch.zeros(1, H, D, device=dev), int(positions[b]))[0] for b in range(B)]).squeeze(1)
    torch.testing.assert_close(y.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)


@gpu
def test_device_t_rope_full_width_matches_oracle():
    """The Qwen3 call shape (ROT=D, TSTRIDE=D) is unchanged by the port."""
    pytest.importorskip("triton")
    from vkernels.compiler.device_triton import _t_rope

    dev = torch.device("cuda")
    rng = np.random.default_rng(6)
    B, H, D = 2, 3, HEAD_DIM
    x = torch.from_numpy(rng.standard_normal((B, H, D)).astype(np.float32)).to(dev).to(torch.bfloat16)
    positions = torch.tensor([0, 7], device=dev, dtype=torch.int32)
    inv_freq = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    freqs = np.outer(np.arange(MAX_POS, dtype=np.float64), inv_freq)
    cos = torch.from_numpy(np.cos(np.concatenate([freqs, freqs], -1)).astype(np.float32)).to(dev)
    sin = torch.from_numpy(np.sin(np.concatenate([freqs, freqs], -1)).astype(np.float32)).to(dev)

    y = torch.empty_like(x)
    _t_rope[(4,)](x, cos, sin, positions, y, B, H, D, ROT=D, TSTRIDE=D)

    for b in range(B):
        p = int(positions[b])
        row = x[b].float()
        half = D // 2
        x1, x2 = row[:, :half], row[:, half:]
        expected = torch.cat([x1 * cos[p, :half] - x2 * sin[p, :half], x2 * cos[p, half:] + x1 * sin[p, half:]], -1)
        torch.testing.assert_close(y[b].float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
