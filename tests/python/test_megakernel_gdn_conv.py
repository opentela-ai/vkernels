"""Issue #89: `gdn_conv` op — causal depthwise conv1d + silu decode step.

Validation chain (mirrors the issue's Validation section):

* capture: the recorder's ``gdn_conv(...)`` records the FIR + state-shift
  contract with read-modify-write effects on the external [B, K-1, C] fp32
  state pool and returns a post-step state view (cache_append's §4.3
  pattern); the op is position-independent — capture works with no decode
  position defined at all;
* hazards: two chained gdn_conv ops on the same pool order via state-storage
  RAW/WAR/WAW hazards, not the decode position;
* reference executor vs the floe oracle (``qwen35_gdn.GatedDeltaNet`` seq==1
  decode branch, CPU tiny config) over a random-state walk of several decode
  steps, with NaN canaries on the workspace;
* device: the generic template ``_t_gdn_conv_tiled`` (arithmetically the
  27B-validated ``_t_gdn_conv`` over a batched pool) matches the same oracle
  on CUDA; skipped on CPU-only stacks.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src" / "python"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

torch = pytest.importorskip("torch")

from vkernels.compiler.capture import CaptureError, RecordingBackend  # noqa: E402
from vkernels.compiler.legality import check_graph  # noqa: E402
from vkernels.compiler.lowerings import lower_graph  # noqa: E402
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import compute_hazards  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402
from vkernels.compiler.task_ir import TileDomain  # noqa: E402


# ---------------------------------------------------------------------------
# floe oracle: GatedDeltaNet (qwen35). The floe package imports kvaas_runtime
# at its top level (KV residency); that native module is irrelevant to the
# conv math, so stub it before importing.
# ---------------------------------------------------------------------------

def _floe_gdn():
    if "floe" not in sys.modules:
        # floe is a sibling repo on this stack, not a venv dependency; probe
        # FLOE_ROOT then the standard serving-stack checkout location.
        import os
        from pathlib import Path
        for cand in (os.environ.get("FLOE_ROOT"), "/home/xiayao/Documents/projects/opentela-ai/serving-stack/floe", "/local/home/xiayao/Documents/code/floe"):
            if cand and (Path(cand) / "floe" / "engine").is_dir():
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                break
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
        # floe refactored the qwen35 modules: qwen35_config -> qwen35.config,
        # qwen35_gdn -> qwen35.gdn. Try the historical paths first, then the
        # current layout.
        try:
            from floe.engine.runner.models.qwen35.qwen35_config import Qwen35Config
            from floe.engine.runner.models.qwen35.qwen35_gdn import GatedDeltaNet
        except ModuleNotFoundError:
            from floe.engine.runner.models.qwen35.config import Qwen35Config
            from floe.engine.runner.models.qwen35.gdn import GatedDeltaNet
    except ModuleNotFoundError as exc:  # floe not on this stack (vkernels-only venv)
        pytest.skip(f"floe qwen35 oracle unavailable: {exc}")

    cfg = Qwen35Config.tiny()
    torch.manual_seed(89)
    gdn = GatedDeltaNet(cfg, torch.device("cpu"), torch.float32)
    return gdn, cfg


def _floe_decode_step(gdn: "torch.nn.Module", conv_state: "torch.Tensor", mixed_qkv: "torch.Tensor"):
    """floe GatedDeltaNet.forward seq==1 branch (qwen35_gdn.py), verbatim:

        full = cat(conv_state, mixed_qkv)          # [K, C]
        conv_out = silu((full * w.t()).sum(0))     # depthwise FIR
        new_conv_state = full[1:]                  # time-major shift
    """
    # floe keeps the seq dim at seq==1: mixed_qkv arrives as [1, C].
    mixed_qkv = mixed_qkv.reshape(1, -1)
    full = torch.cat([conv_state, mixed_qkv], dim=0)
    w = gdn.conv1d.weight.squeeze(1)
    conv_out = torch.nn.functional.silu((full * w.t()).sum(dim=0))
    return conv_out, full[1:]


# Tiny floe config: conv_dim = 2*key_dim + value_dim = 64, K = 4.
K = 4
CONV_DIM = 64
B = 2
STEPS = 6
STORAGE_STATE, STORAGE_X, STORAGE_W = 301, 302, 303


# ---------------------------------------------------------------------------
# Compiler plumbing: capture -> lower -> schedule -> reference executor, with
# the state pool as external persistent storage so a decode walk evolves it
# in place across run() invocations.
# ---------------------------------------------------------------------------

def _build_executor(state_init: np.ndarray, workers: int):
    """Capture a single gdn_conv op and return (executor, out_name, handles)."""
    recorder = RecordingBackend()
    # No define_position call: gdn_conv is position-independent.
    state = recorder.external_tensor("conv_state", state_init.shape, storage_id=STORAGE_STATE)
    x = recorder.external_tensor("mixed_qkv", (B, CONV_DIM), storage_id=STORAGE_X)
    w = recorder.external_tensor("fir_w", (CONV_DIM, K), storage_id=STORAGE_W)
    out, state_post = recorder.gdn_conv(state, x, w, layer=0)
    graph = recorder.graph

    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"

    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)

    state_arr = state_init.astype(np.float32).reshape(-1).copy()
    storage_arrays = {
        STORAGE_STATE: state_arr,
        STORAGE_X: np.full(B * CONV_DIM, np.nan, dtype=np.float32),
        STORAGE_W: np.tile(np.arange(K, dtype=np.float32), CONV_DIM) * 0.25,
    }
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(schedule, workers=workers, storage_arrays=storage_arrays, graph=graph, workspace_plan=workspace_plan)
    handles = {"state": state, "x": x, "w": w, "out": out, "state_post": state_post, "graph": graph, "families": families, "storage_arrays": storage_arrays}
    return executor, handles


# ===========================================================================
# Capture (recorder contract)
# ===========================================================================

def test_capture_gdn_conv_records_contract():
    recorder = RecordingBackend()
    state = recorder.external_tensor("conv_state", (B, K - 1, CONV_DIM), storage_id=401)
    x = recorder.external_tensor("mixed_qkv", (B, CONV_DIM), storage_id=402)
    w = recorder.external_tensor("fir_w", (CONV_DIM, K), storage_id=403)
    out, state_post = recorder.gdn_conv(state, x, w, layer=3)
    op = recorder.graph.ops[-1]
    assert op.kind == "gdn_conv"
    assert op.attributes["layer"] == 3
    assert op.attributes["conv_kernel"] == K
    assert "fir" in op.numerical_contract and "state_shift" in op.numerical_contract
    # read-modify-write on the pool: the state storage is both read and written
    state_sids = {r.storage_id for r in op.read_regions} & {r.storage_id for r in op.write_regions}
    assert 401 in state_sids
    # post-step state view: same storage, bumped version
    assert state_post.value.storage_id == 401
    assert state_post.value.name != state.value.name
    assert recorder.graph.storage_versions[401] == 1
    # position independence: no symbolic scalars consumed
    assert recorder.graph.scalars == {}


@pytest.mark.parametrize("kwargs", [
    {"state_shape": (B, K, CONV_DIM)},          # rank-2 pool (K not K-1)
    {"x_shape": (B, CONV_DIM + 1)},             # row width mismatch
    {"w_shape": (CONV_DIM, K + 1)},             # wrong tap count
    {"w_shape": (CONV_DIM + 8, K)},             # wrong channel count
])
def test_capture_gdn_conv_rejects_shape_mismatches(kwargs):
    recorder = RecordingBackend()
    state = recorder.external_tensor("conv_state", kwargs.get("state_shape", (B, K - 1, CONV_DIM)), storage_id=501)
    x = recorder.external_tensor("mixed_qkv", kwargs.get("x_shape", (B, CONV_DIM)), storage_id=502)
    w = recorder.external_tensor("fir_w", kwargs.get("w_shape", (CONV_DIM, K)), storage_id=503)
    with pytest.raises(CaptureError):
        recorder.gdn_conv(state, x, w, layer=0)


def test_gdn_conv_two_layer_hazards_are_state_ordered():
    """Chained gdn_conv ops on one pool: RAW/WAR/WAW on the state storage —
    the ordering is state hazards, not the (absent) decode position."""
    recorder = RecordingBackend()
    state = recorder.external_tensor("conv_state", (B, K - 1, CONV_DIM), storage_id=601)
    x1 = recorder.external_tensor("mixed_qkv_1", (B, CONV_DIM), storage_id=602)
    w = recorder.external_tensor("fir_w", (CONV_DIM, K), storage_id=603)
    _out1, state_post = recorder.gdn_conv(state, x1, w, layer=0)
    x2 = recorder.external_tensor("mixed_qkv_2", (B, CONV_DIM), storage_id=604)
    recorder.gdn_conv(state_post, x2, w, layer=1)
    hazards = compute_hazards(recorder.graph.ops)
    pairs = {(h.kind, h.producer, h.consumer) for h in hazards if h.storage_id == 601}
    assert ("RAW", 0, 1) in pairs and ("WAR", 0, 1) in pairs and ("WAW", 0, 1) in pairs


# ===========================================================================
# Lowering (task decomposition)
# ===========================================================================

def test_lowering_gdn_conv_task_decomposition():
    gdn, _cfg = _floe_gdn()
    executor, handles = _build_executor(np.zeros((B, K - 1, CONV_DIM), dtype=np.float32), workers=1)
    fam = handles["families"][0]
    assert fam.kind == "gdn_conv"
    assert fam.params["conv_kernel"] == K
    assert fam.params["tile"] * fam.task_count >= CONV_DIM * B  # every channel covered
    assert fam.task_count == B * ((CONV_DIM + fam.params["tile"] - 1) // fam.params["tile"])
    # per-task regions: state tile + row tile + weight tile reads;
    # state tile + out tile writes
    r = fam.reads(0)
    assert len(r) == 3
    wr = fam.writes(0)
    assert len(wr) == 2


# ===========================================================================
# Reference execution vs the floe oracle (CPU, tiny config, decode walk)
# ===========================================================================

@pytest.mark.parametrize("workers", [1, 3])
def test_reference_gdn_conv_walk_matches_floe_oracle(workers):
    gdn, _cfg = _floe_gdn()
    rng = np.random.default_rng(41 + workers)
    state0 = rng.standard_normal((B, K - 1, CONV_DIM)).astype(np.float32) * 0.05

    executor, handles = _build_executor(state0, workers=workers)
    out_name = handles["out"].value.name

    # Oracle side: per-batch fp32 torch states, floe decode branch each step.
    oracle_states = torch.from_numpy(state0.copy())  # [B, K-1, C]
    w_t = gdn.conv1d.weight.squeeze(1)  # [C, K]
    # The executor must run the same FIR taps as the oracle.
    handles["storage_arrays"][STORAGE_W][:] = w_t.detach().numpy().reshape(-1)

    for step in range(STEPS):
        x_np = rng.standard_normal((B, CONV_DIM)).astype(np.float32)
        handles["storage_arrays"][STORAGE_X][:] = x_np.reshape(-1)
        executor.run({})

        got = np.array(executor.tensor(out_name), copy=True)
        # NaN canary: the workspace out buffer must be fully written
        assert np.isfinite(got).all(), f"step {step}: unwritten/NaN outputs (canary tripped)"

        x_t = torch.from_numpy(x_np)
        expected = torch.empty(B, CONV_DIM)
        for b in range(B):
            conv_out, new_state = _floe_decode_step(gdn, oracle_states[b], x_t[b])
            expected[b] = conv_out
            oracle_states[b] = new_state
        torch.testing.assert_close(torch.from_numpy(got), expected, rtol=1e-5, atol=1e-6)

    # The persistent pool must equal the oracle's final states (shift chain).
    pool = handles["storage_arrays"][STORAGE_STATE].reshape(B, K - 1, CONV_DIM)
    torch.testing.assert_close(torch.from_numpy(pool.copy()), oracle_states, rtol=1e-6, atol=1e-6)


def test_reference_gdn_conv_state_shift_semantics():
    """After one step, pool row j holds old row j+1 and row K-2 holds x —
    checked directly against a hand-rolled shift."""
    rng = np.random.default_rng(7)
    state0 = rng.standard_normal((B, K - 1, CONV_DIM)).astype(np.float32)
    executor, handles = _build_executor(state0, workers=2)
    x_np = rng.standard_normal((B, CONV_DIM)).astype(np.float32)
    handles["storage_arrays"][STORAGE_X][:] = x_np.reshape(-1)
    executor.run({})
    pool = handles["storage_arrays"][STORAGE_STATE].reshape(B, K - 1, CONV_DIM)
    np.testing.assert_array_equal(pool[:, : K - 2, :], state0[:, 1:, :])
    np.testing.assert_array_equal(pool[:, K - 2, :], x_np)


def test_reference_gdn_conv_silu_is_elementwise_fir():
    """Out must be silu of the FIR accumulation: compare against an
    independent fp64 recomputation from the same storages."""
    gdn, _cfg = _floe_gdn()
    rng = np.random.default_rng(13)
    state0 = rng.standard_normal((B, K - 1, CONV_DIM)).astype(np.float32) * 0.3
    executor, handles = _build_executor(state0, workers=2)
    x_np = rng.standard_normal((B, CONV_DIM)).astype(np.float32)
    w_np = handles["storage_arrays"][STORAGE_W].reshape(CONV_DIM, K)
    handles["storage_arrays"][STORAGE_X][:] = x_np.reshape(-1)
    executor.run({})
    got = np.array(executor.tensor(handles["out"].value.name), copy=True)
    for b in range(B):
        full = np.concatenate([state0[b], x_np[b][None, :]], axis=0).astype(np.float64)
        acc = (full * np.asarray(w_np, dtype=np.float64).T).sum(axis=0)
        expected = acc / (1.0 + np.exp(-acc))
        np.testing.assert_allclose(got[b], expected, rtol=1e-5, atol=1e-6)


# ===========================================================================
# Device template (_t_gdn_conv_tiled) — CUDA only
# ===========================================================================

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA (the B=1 path is covered by the 27B tests in test_megakernel_27b.py)",
)


@gpu
def test_device_t_gdn_conv_tiled_matches_oracle():
    """Same oracle as the CPU executor tests above: the kernel is given a
    random FIR weight ``w`` and must reproduce silu(FIR) + the time-major
    state shift for exactly that ``w`` (fp64 recomputation).

    Launch contract: ``worker``/``P`` are the runtime worker id / worker
    count; the kernel mutates the state pool in place, so the test runs a
    single worker (grid (1,), worker=0, P=1) that walks all (batch, tile)
    tasks sequentially — deterministic, no double-shift races.
    """
    pytest.importorskip("triton")
    from vkernels.compiler.device_triton import _t_gdn_conv_tiled

    dev = torch.device("cuda")
    rng = np.random.default_rng(97)
    state = (rng.standard_normal((B, K - 1, CONV_DIM)) * 0.05).astype(np.float32)
    x = rng.standard_normal((B, CONV_DIM)).astype(np.float32)
    w = rng.standard_normal((CONV_DIM, K)).astype(np.float32)

    state_t = torch.from_numpy(state).to(dev)
    x_t = torch.from_numpy(x).to(dev)
    w_t = torch.from_numpy(w).to(dev)
    out = torch.empty(B, CONV_DIM, device=dev, dtype=torch.float32)

    ELEM = 32  # exact tiling of the tiny conv_dim
    _t_gdn_conv_tiled[(1,)](0, 1, state_t, w_t, x_t, out, B, CONV_DIM, ELEM, K, num_warps=4)
    torch.cuda.synchronize()

    # fp64 oracle on the SAME random w the kernel received (matches the CPU
    # executor tests): full = cat(state0[b], x[b]); silu((full * w.T).sum(0));
    # new_state = full[1:] (time-major shift).
    expected = torch.empty(B, CONV_DIM, dtype=torch.float64)
    expected_state = torch.empty(B, K - 1, CONV_DIM, dtype=torch.float64)
    w64 = torch.from_numpy(w).double()
    for b in range(B):
        full = torch.cat([torch.from_numpy(state)[b].double(), torch.from_numpy(x)[b][None, :].double()], dim=0)
        acc = (full * w64.t()).sum(dim=0)
        expected[b] = acc / (1.0 + torch.exp(-acc))
        expected_state[b] = full[1:]
    torch.testing.assert_close(out.double().cpu(), expected, rtol=1e-4, atol=1e-5)
    # in-place state shift on the device pool
    torch.testing.assert_close(state_t.double().cpu(), expected_state, rtol=1e-5, atol=1e-6)
