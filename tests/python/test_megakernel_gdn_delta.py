"""Issue #90: `gdn_delta` op — per-head gated delta rule decode step.

Validation chain (mirrors the issue's Validation section):

* capture: the recorder's ``gdn_delta(...)`` records the decay + delta-rule
  + RMSNorm/z-gate contract with read-modify-write effects on the external
  [B, NV, HV, HK] fp32 SSM state pool and returns a post-step state view
  (cache_append's §4.3 pattern); the op is position-independent — capture
  works with no decode position defined at all;
* hazards: two chained gdn_delta ops on the same pool order via state-
  storage RAW/WAR/WAW hazards, not the decode position;
* reference executor vs the per-token eager recurrence oracle (floe
  ``qwen35_gdn._gdn_delta_rule_recurrent`` + the conditioning/RMSNorm tail
  of ``GatedDeltaNet.forward`` verbatim, CPU tiny config) over a random
  multi-step decode walk, multi-head/multi-batch, with NaN canaries and an
  explicit state-drift bound (the 27B-test convention);
* device: the batched template ``_t_gdn_heads_batched`` (arithmetically the
  27B-validated ``_t_gdn_heads`` over a batched pool) matches the same
  oracle on CUDA; skipped on CPU-only stacks.
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


# ---------------------------------------------------------------------------
# floe oracle: GatedDeltaNet (qwen35). The floe package imports kvaas_runtime
# at its top level (KV residency); that native module is irrelevant to the
# delta-rule math, so stub it before importing.
# ---------------------------------------------------------------------------

def _floe_gdn():
    if "floe" not in sys.modules:
        # floe is a sibling repo on this stack, not a venv dependency; probe
        # FLOE_ROOT then the standard serving-stack checkout location.
        import os
        for cand in (os.environ.get("FLOE_ROOT"), "/home/xiayao/Documents/projects/opentela-ai/serving-stack/floe"):
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
        from floe.engine.runner.models.qwen35.qwen35_config import Qwen35Config
        from floe.engine.runner.models.qwen35 import qwen35_gdn
        from floe.engine.runner.models.qwen35.qwen35_gdn import GatedDeltaNet
    except ModuleNotFoundError as exc:  # floe not on this stack (vkernels-only venv)
        pytest.skip(f"floe qwen35 oracle unavailable: {exc}")

    cfg = Qwen35Config.tiny()
    torch.manual_seed(90)
    gdn = GatedDeltaNet(cfg, torch.device("cpu"), torch.float32)
    return gdn, cfg, qwen35_gdn


def _floe_condition_and_recurse(gdn, qwen35_gdn, q, k, v, z, a, b, ssm_state):
    """floe GatedDeltaNet.forward conditioning + seq==1 recurrence, verbatim
    (qwen35_gdn.py): per-token fp32 conditioning then the eager per-token
    recurrence, then the per-head RMSNorm + z-gate readout. Returns
    ``(o_gated [nv, hv], new_ssm_state [nv, hv, hk])`` for one batch row.
    """
    scale, eps = gdn.scale, gdn.cfg.rms_norm_eps
    A = torch.exp(gdn.A_log)
    a_f, b_f = a.to(torch.float32), b.to(torch.float32)
    q_f32, k_f32, v_f32 = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)
    x_dt = a_f + gdn.dt_bias
    softplus_x = torch.where(x_dt <= 20.0, torch.log(1.0 + torch.exp(x_dt)), x_dt)
    g = -A * softplus_x  # [nv] log-decay
    beta = torch.sigmoid(b_f)  # [nv]
    q_n = q_f32 / torch.sqrt((q_f32 * q_f32).sum(dim=-1, keepdim=True) + 1e-6) * scale
    k_n = k_f32 / torch.sqrt((k_f32 * k_f32).sum(dim=-1, keepdim=True) + 1e-6)
    q_exp = q_n.repeat_interleave(gdn.group_size, dim=0)  # [nv, hk]
    k_exp = k_n.repeat_interleave(gdn.group_size, dim=0)  # [nv, hk]
    o, ssm_state = qwen35_gdn._gdn_delta_rule_recurrent(
        q_exp.unsqueeze(0), k_exp.unsqueeze(0), v_f32.unsqueeze(0), beta.unsqueeze(0), torch.exp(g).unsqueeze(0), ssm_state
    )
    o_f = o.squeeze(0).to(torch.float32)
    var = o_f.pow(2).mean(-1, keepdim=True)
    o_norm = o_f * torch.rsqrt(var + gdn.norm.eps)
    o_norm = gdn.norm.weight * o_norm
    z_f = z.to(torch.float32)
    o_gated = o_norm * (z_f * torch.sigmoid(z_f))
    return o_gated, ssm_state


# Tiny floe config: nk=2, nv=4 (group_size 2), hk=hv=8, hidden=16.
B = 2
NK, NV, HV, HK = 2, 4, 8, 8
STEPS = 6
_SCALE = 8**-0.5
_EPS = 1e-6
STORAGE_STATE = 901


# ---------------------------------------------------------------------------
# Compiler plumbing: capture -> lower -> schedule -> reference executor, with
# the state pool as external persistent storage so a decode walk evolves it
# in place across run() invocations.
# ---------------------------------------------------------------------------

def _ext(rec, name, shape, sid):
    return rec.external_tensor(name, shape, storage_id=sid)


def _capture_delta(rec, layer: int = 0):
    state = _ext(rec, "ssm_state", (B, NV, HV, HK), STORAGE_STATE)
    q = _ext(rec, "gdn_q", (B, NK, HK), 902)
    k = _ext(rec, "gdn_k", (B, NK, HK), 903)
    v = _ext(rec, "gdn_v", (B, NV, HV), 904)
    z = _ext(rec, "gdn_z", (B, NV, HV), 905)
    a = _ext(rec, "gdn_a", (B, NV), 906)
    b = _ext(rec, "gdn_b", (B, NV), 907)
    a_log = _ext(rec, "A_log", (NV,), 908)
    dt_bias = _ext(rec, "dt_bias", (NV,), 909)
    norm_w = _ext(rec, "norm_w", (HV,), 910)
    out, state_post = rec.gdn_delta(state, q, k, v, z, a, b, a_log, dt_bias, norm_w, layer=layer, scale=_SCALE, eps=_EPS)
    return {"state": state, "q": q, "k": k, "v": v, "z": z, "a": a, "b": b,
            "a_log": a_log, "dt_bias": dt_bias, "norm_w": norm_w, "out": out, "state_post": state_post}


def _build_executor(state_init: np.ndarray, workers: int, feeds: dict | None = None):
    feeds = feeds or {}
    """Capture a single gdn_delta op and return (executor, handles)."""
    recorder = RecordingBackend()
    # No define_position call: gdn_delta is position-independent.
    h = _capture_delta(recorder)
    graph = recorder.graph

    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"

    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)

    state_arr = state_init.astype(np.float32).reshape(-1).copy()
    storage_arrays = {STORAGE_STATE: state_arr}
    for name, sym in (("q", h["q"]), ("k", h["k"]), ("v", h["v"]), ("z", h["z"]),
                      ("a", h["a"]), ("b", h["b"]), ("a_log", h["a_log"]),
                      ("dt_bias", h["dt_bias"]), ("norm_w", h["norm_w"])):
        tv = sym.value
        arr = np.full(int(np.prod(tv.shape)), np.nan, dtype=np.float32)
        if name in feeds:
            arr[:] = feeds[name].reshape(-1)
        storage_arrays[tv.storage_id] = arr
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(schedule, workers=workers, storage_arrays=storage_arrays, graph=graph, workspace_plan=workspace_plan)
    handles = dict(h)
    handles.update({"graph": graph, "families": families, "storage_arrays": storage_arrays, "executor": executor})
    return executor, handles


def _set_feeds(handles, q, k, v, z, a, b):
    sa = handles["storage_arrays"]
    for name, arr in (("q", q), ("k", k), ("v", v), ("z", z), ("a", a), ("b", b)):
        sa[handles[name].value.storage_id][:] = arr.reshape(-1)


def _load_head_params(gdn, handles):
    """Copy the floe module's per-layer params into the compiled storages."""
    sa = handles["storage_arrays"]
    sa[handles["a_log"].value.storage_id][:] = gdn.A_log.detach().numpy()
    sa[handles["dt_bias"].value.storage_id][:] = gdn.dt_bias.detach().numpy()
    sa[handles["norm_w"].value.storage_id][:] = gdn.norm.weight.detach().numpy()


# ===========================================================================
# Capture (recorder contract)
# ===========================================================================

def test_capture_gdn_delta_records_contract():
    recorder = RecordingBackend()
    h = _capture_delta(recorder)
    op = recorder.graph.ops[-1]
    assert op.kind == "gdn_delta"
    assert op.attributes["layer"] == 0
    assert abs(op.attributes["scale"] - 8**-0.5) < 1e-12
    assert "decay" in op.numerical_contract and "delta_rule" in op.numerical_contract
    assert "out_gate" in op.numerical_contract and "qk_norm" in op.numerical_contract
    # read-modify-write on the pool: the state storage is both read and written
    state_sids = {r.storage_id for r in op.read_regions} & {r.storage_id for r in op.write_regions}
    assert STORAGE_STATE in state_sids
    # post-step state view: same storage, bumped version
    assert h["state_post"].value.storage_id == STORAGE_STATE
    assert h["state_post"].value.name != h["state"].value.name
    assert recorder.graph.storage_versions[STORAGE_STATE] == 1
    # position independence: no symbolic scalars consumed
    assert recorder.graph.scalars == {}


@pytest.mark.parametrize("kwargs", [
    {"state_shape": (B, NV, HV)},                # rank-3 pool
    {"state_shape": (B, NV, HV, HK + 1)},        # head dim mismatch vs q/k
    {"q_shape": (B, 3, HK)},                     # NK does not divide NV
    {"k_shape": (B, NK, HK + 2)},
    {"v_shape": (B, NV + 1, HV)},
    {"z_shape": (B, NV, HV + 1)},
    {"a_shape": (B, NV + 1)},
    {"b_shape": (B + 1, NV)},
    {"a_log_shape": (NV + 1,)},
    {"dt_bias_shape": (NV - 1,)},
    {"norm_w_shape": (HV + 1,)},
])
def test_capture_gdn_delta_rejects_shape_mismatches(kwargs):
    recorder = RecordingBackend()
    state = _ext(recorder, "ssm_state", kwargs.get("state_shape", (B, NV, HV, HK)), 951)
    q = _ext(recorder, "gdn_q", kwargs.get("q_shape", (B, NK, HK)), 952)
    k = _ext(recorder, "gdn_k", kwargs.get("k_shape", (B, NK, HK)), 953)
    v = _ext(recorder, "gdn_v", kwargs.get("v_shape", (B, NV, HV)), 954)
    z = _ext(recorder, "gdn_z", kwargs.get("z_shape", (B, NV, HV)), 955)
    a = _ext(recorder, "gdn_a", kwargs.get("a_shape", (B, NV)), 956)
    b = _ext(recorder, "gdn_b", kwargs.get("b_shape", (B, NV)), 957)
    a_log = _ext(recorder, "A_log", kwargs.get("a_log_shape", (NV,)), 958)
    dt_bias = _ext(recorder, "dt_bias", kwargs.get("dt_bias_shape", (NV,)), 959)
    norm_w = _ext(recorder, "norm_w", kwargs.get("norm_w_shape", (HV,)), 960)
    with pytest.raises(CaptureError):
        recorder.gdn_delta(state, q, k, v, z, a, b, a_log, dt_bias, norm_w, layer=0, scale=1.0, eps=1e-6)


def test_gdn_delta_two_layer_hazards_are_state_ordered():
    """Chained gdn_delta ops on one pool: RAW/WAR/WAW on the state storage —
    the ordering is state hazards, not the (absent) decode position."""
    recorder = RecordingBackend()
    _h1 = _capture_delta(recorder, layer=0)
    _h2 = _capture_delta(recorder, layer=1)
    hazards = compute_hazards(recorder.graph.ops)
    pairs = {(h.kind, h.producer, h.consumer) for h in hazards if h.storage_id == STORAGE_STATE}
    assert ("RAW", 0, 1) in pairs and ("WAR", 0, 1) in pairs and ("WAW", 0, 1) in pairs


# ===========================================================================
# Lowering (task decomposition)
# ===========================================================================

def test_lowering_gdn_delta_task_decomposition():
    executor, handles = _build_executor(np.zeros((B, NV, HV, HK), dtype=np.float32), workers=1)
    fam = handles["families"][0]
    assert fam.kind == "gdn_delta"
    assert fam.params["scale"] == pytest.approx(8**-0.5)
    assert fam.task_count == B * NV  # one task per (batch, value head)
    r = fam.reads(0)
    assert len(r) == 10  # state slice, q, k, v, z, a, b, A_log, dt_bias, norm_w
    wr = fam.writes(0)
    assert len(wr) == 2  # state slice (RMW) + out slice
    # per-task regions own exactly one (b, head) state slice
    box = fam.box(0)
    assert box[0][1] - box[0][0] == 1 and box[1][1] - box[1][0] == 1


# ===========================================================================
# Reference execution vs the floe oracle (CPU, tiny config, decode walk)
# ===========================================================================

@pytest.mark.parametrize("workers", [1, 3])
def test_reference_gdn_delta_walk_matches_floe_oracle(workers):
    gdn, _cfg, qwen35_gdn = _floe_gdn()
    rng = np.random.default_rng(40 + workers)
    state0 = (rng.standard_normal((B, NV, HV, HK)) * 0.3).astype(np.float32)

    executor, handles = _build_executor(state0, workers=workers)
    out_name = handles["out"].value.name
    _load_head_params(gdn, handles)

    # Oracle side: per-batch fp32 torch states, floe eager recurrence each step.
    oracle_states = torch.from_numpy(state0.copy())  # [B, NV, HV, HK]
    drift = 0.0
    for step in range(STEPS):
        feeds = {
            "q": rng.standard_normal((B, NK, HK)) * 0.7,
            "k": rng.standard_normal((B, NK, HK)) * 0.7,
            "v": rng.standard_normal((B, NV, HV)) * 0.7,
            "z": rng.standard_normal((B, NV, HV)) * 0.7,
            "a": rng.standard_normal((B, NV)) * 0.7,
            "b": rng.standard_normal((B, NV)) * 0.7,
        }
        _set_feeds(handles, **feeds)
        executor.run({})

        got = np.array(executor.tensor(out_name), copy=True)
        # NaN canary: the workspace out buffer must be fully written
        assert np.isfinite(got).all(), f"step {step}: unwritten/NaN outputs (canary tripped)"

        expected = torch.empty(B, NV, HV)
        for bb in range(B):
            o_gated, new_state = _floe_condition_and_recurse(
                gdn, qwen35_gdn,
                torch.from_numpy(feeds["q"][bb]), torch.from_numpy(feeds["k"][bb]),
                torch.from_numpy(feeds["v"][bb]), torch.from_numpy(feeds["z"][bb]),
                torch.from_numpy(feeds["a"][bb]), torch.from_numpy(feeds["b"][bb]),
                oracle_states[bb],
            )
            expected[bb] = o_gated
            oracle_states[bb] = new_state.detach()
        torch.testing.assert_close(torch.from_numpy(got), expected, rtol=1e-5, atol=1e-6)

        pool = handles["storage_arrays"][STORAGE_STATE].reshape(B, NV, HV, HK)
        drift = max(drift, float(np.abs(pool - oracle_states.numpy()).max()))

    # State-drift bound (the 27B-test convention): compiled pool vs oracle
    # states must stay tight across the whole walk.
    assert drift < 1e-5, f"ssm state drift {drift} exceeds the 1e-5 bound"
    pool = handles["storage_arrays"][STORAGE_STATE].reshape(B, NV, HV, HK)
    torch.testing.assert_close(torch.from_numpy(pool.copy()), oracle_states, rtol=1e-6, atol=1e-6)


def test_reference_gdn_delta_issue_factored_form():
    """Independent recomputation in the issue's factored form
    ``s_new = diag(exp(g)) · s (I − β k kᵀ) + β k vᵀ`` — algebraically the
    same update floe applies, rearranged."""
    gdn, _cfg, qwen35_gdn = _floe_gdn()
    rng = np.random.default_rng(11)
    state0 = (rng.standard_normal((B, NV, HV, HK)) * 0.5).astype(np.float32)
    executor, handles = _build_executor(state0, workers=2)
    _load_head_params(gdn, handles)
    q = rng.standard_normal((B, NK, HK)) * 0.5
    k = rng.standard_normal((B, NK, HK)) * 0.5
    v = rng.standard_normal((B, NV, HV)) * 0.5
    z = rng.standard_normal((B, NV, HV)) * 0.5
    a = rng.standard_normal((B, NV)) * 0.5
    b = rng.standard_normal((B, NV)) * 0.5
    _set_feeds(handles, q, k, v, z, a, b)
    executor.run({})
    got = np.array(executor.tensor(handles["out"].value.name), copy=True)

    for bb in range(B):
        for h in range(NV):
            kh = h // (NV // NK)
            x_dt = a[bb, h] + gdn.dt_bias[h].item()
            sp = np.log(1.0 + np.exp(x_dt)) if x_dt <= 20.0 else x_dt
            decay = np.exp(-np.exp(gdn.A_log[h].item()) * sp)
            beta = 1.0 / (1.0 + np.exp(-b[bb, h]))
            qf, kf, vf = q[bb, kh].astype(np.float64), k[bb, kh].astype(np.float64), v[bb, h].astype(np.float64)
            qn = qf / np.sqrt(qf @ qf + 1e-6) * gdn.scale
            kn = kf / np.sqrt(kf @ kf + 1e-6)
            s = state0[bb, h].astype(np.float64)
            # issue-factored update: decay * (s - beta*(s@kn)kn^T) + beta*vf kn^T
            s_new = decay * (s - beta * np.outer(s @ kn, kn)) + beta * np.outer(vf, kn)
            o = s_new @ qn
            on = o / np.sqrt(np.mean(o * o) + gdn.norm.eps) * gdn.norm.weight.detach().numpy()
            zf = z[bb, h].astype(np.float64)
            expected = on * (zf / (1.0 + np.exp(-zf)))
            np.testing.assert_allclose(got[bb, h], expected, rtol=1e-5, atol=1e-6)


def test_reference_gdn_delta_group_expansion():
    """Value heads in one group must read the SAME normalized key head: two
    decode chains identical except for a perturbation of key head 0 must
    produce identical state/output evolution for the value heads OUTSIDE
    its group ({2, 3}) at every step, and differ inside it ({0, 1})."""
    gdn, _cfg, qwen35_gdn = _floe_gdn()
    rng = np.random.default_rng(23)
    state0 = (rng.standard_normal((B, NV, HV, HK)) * 0.3).astype(np.float32)
    base = {n: rng.standard_normal(sh) * 0.5 for n, sh in
            (("q", (B, NK, HK)), ("k", (B, NK, HK)), ("v", (B, NV, HV)), ("z", (B, NV, HV)),
             ("a", (B, NV)), ("b", (B, NV)))}

    def chain(k_feed, steps=3):
        executor, handles = _build_executor(state0.copy(), workers=2)
        _load_head_params(gdn, handles)
        outs, pools = [], []
        for _ in range(steps):
            _set_feeds(handles, **{**base, "k": k_feed})
            executor.run({})
            outs.append(np.array(executor.tensor(handles["out"].value.name), copy=True))
            pools.append(handles["storage_arrays"][STORAGE_STATE].reshape(B, NV, HV, HK).copy())
        return outs, pools

    k2 = base["k"].copy()
    k2[:, 0, :] += 0.9  # perturb key head 0 only (its group: value heads {0, 1})
    outs_a, pools_a = chain(base["k"])
    outs_b, pools_b = chain(k2)
    for step in range(len(outs_a)):
        for bb in range(B):
            # group members move; non-members are bitwise unaffected
            assert not np.allclose(outs_a[step][bb, 0], outs_b[step][bb, 0])
            assert not np.allclose(outs_a[step][bb, 1], outs_b[step][bb, 1])
            np.testing.assert_array_equal(outs_a[step][bb, 2:], outs_b[step][bb, 2:])
            assert not np.allclose(pools_a[step][bb, 0], pools_b[step][bb, 0])
            np.testing.assert_array_equal(pools_a[step][bb, 2:], pools_b[step][bb, 2:])


def test_reference_gdn_delta_in_place_state_rmw():
    """The pool must be updated in place from its pre-step contents (the
    task reads the OLD state, writes the NEW state to the same storage)."""
    gdn, _cfg, qwen35_gdn = _floe_gdn()
    rng = np.random.default_rng(31)
    state0 = (rng.standard_normal((B, NV, HV, HK)) * 0.3).astype(np.float32)
    executor, handles = _build_executor(state0, workers=2)
    _load_head_params(gdn, handles)
    feeds = {n: rng.standard_normal(sh) * 0.5 for n, sh in
             (("q", (B, NK, HK)), ("k", (B, NK, HK)), ("v", (B, NV, HV)), ("z", (B, NV, HV)),
              ("a", (B, NV)), ("b", (B, NV)))}
    _set_feeds(handles, **feeds)
    executor.run({})
    pool = handles["storage_arrays"][STORAGE_STATE].reshape(B, NV, HV, HK)
    assert not np.allclose(pool, state0)  # genuinely rewritten
    # and a second run must chain from the new state (recurrent semantics)
    oracle = torch.from_numpy(pool.copy())
    _set_feeds(handles, **feeds)
    executor.run({})
    pool2 = handles["storage_arrays"][STORAGE_STATE].reshape(B, NV, HV, HK)
    for bb in range(B):
        _, new_state = _floe_condition_and_recurse(
            gdn, qwen35_gdn,
            torch.from_numpy(feeds["q"][bb]), torch.from_numpy(feeds["k"][bb]),
            torch.from_numpy(feeds["v"][bb]), torch.from_numpy(feeds["z"][bb]),
            torch.from_numpy(feeds["a"][bb]), torch.from_numpy(feeds["b"][bb]),
            oracle[bb],
        )
        torch.testing.assert_close(torch.from_numpy(pool2[bb].copy()), new_state, rtol=1e-6, atol=1e-6)


# ===========================================================================
# Integration: gdn_conv -> gdn_delta chained decode vs the full floe forward
# (runs only once the #89 gdn_conv op is on this base)
# ===========================================================================

def _gdn_conv_available() -> bool:
    return hasattr(RecordingBackend, "gdn_conv")


@pytest.mark.skipif(not _gdn_conv_available(), reason="requires the #89 gdn_conv op on this base")
def test_reference_gdn_conv_then_delta_matches_full_floe_forward():
    """Full Qwen3.5 GDN decode step: linear qkv -> gdn_conv -> split ->
    gdn_delta (+ z/a/b/out projections) vs floe GatedDeltaNet.forward seq==1,
    state and conv pools evolving in place across steps."""
    gdn, cfg, qwen35_gdn = _floe_gdn()
    conv_dim = gdn.conv_dim
    key_dim, value_dim, hidden = gdn.key_dim, gdn.value_dim, cfg.hidden_size
    K = gdn.conv_kernel
    rng = np.random.default_rng(55)
    S_CONV, S_SSM, S_HID, S_WQKV, S_WFIR, S_WZ, S_WA, S_WB, S_WOUT = range(701, 710)

    recorder = RecordingBackend()
    conv_state = recorder.external_tensor("conv_state", (B, K - 1, conv_dim), storage_id=S_CONV)
    ssm_state = recorder.external_tensor("ssm_state", (B, NV, HV, HK), storage_id=S_SSM)
    hid = recorder.external_tensor("hidden", (B, hidden), storage_id=S_HID)
    w_qkv = recorder.external_tensor("w_qkv", (hidden, conv_dim), storage_id=S_WQKV)
    w_fir = recorder.external_tensor("w_fir", (conv_dim, K), storage_id=S_WFIR)
    qkv = recorder.linear(hid, w_qkv, name="qkv_proj")
    conv_out, conv_post = recorder.gdn_conv(conv_state, qkv, w_fir, layer=0)
    q = recorder.view_of(recorder.narrow(conv_out, "q_flat", axis=1, start=0, length=key_dim), "gdn_q", (B, NK, HK))
    k = recorder.view_of(recorder.narrow(conv_out, "k_flat", axis=1, start=key_dim, length=key_dim), "gdn_k", (B, NK, HK))
    v = recorder.view_of(recorder.narrow(conv_out, "v_flat", axis=1, start=2 * key_dim, length=value_dim), "gdn_v", (B, NV, HV))
    z = recorder.view_of(recorder.linear(hid, recorder.external_tensor("w_z", (hidden, value_dim), storage_id=S_WZ), name="z_proj"), "gdn_z", (B, NV, HV))
    a = recorder.linear(hid, recorder.external_tensor("w_a", (hidden, NV), storage_id=S_WA), name="a_proj")
    b = recorder.linear(hid, recorder.external_tensor("w_b", (hidden, NV), storage_id=S_WB), name="b_proj")
    a_log = recorder.external_tensor("A_log", (NV,), storage_id=720)
    dt_bias = recorder.external_tensor("dt_bias", (NV,), storage_id=721)
    norm_w = recorder.external_tensor("norm_w", (HV,), storage_id=722)
    out, ssm_post = recorder.gdn_delta(ssm_state, q, k, v, z, a, b, a_log, dt_bias, norm_w, layer=0, scale=gdn.scale, eps=cfg.rms_norm_eps)
    out_flat = recorder.view_of(out, "gdn_out_flat", (B, value_dim))
    y = recorder.linear(out_flat, recorder.external_tensor("w_out", (value_dim, hidden), storage_id=S_WOUT), name="out_proj")
    graph = recorder.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    # two state pools, each RMW'd once per layer
    assert recorder.graph.storage_versions[S_CONV] == 1 and recorder.graph.storage_versions[S_SSM] == 1

    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=3)
    workspace_plan, _ = plan_memory(graph, schedule, families)

    conv0 = (rng.standard_normal((B, K - 1, conv_dim)) * 0.05).astype(np.float32)
    ssm0 = (rng.standard_normal((B, NV, HV, HK)) * 0.3).astype(np.float32)
    sa = {
        S_CONV: conv0.reshape(-1).copy(),
        S_SSM: ssm0.reshape(-1).copy(),
        S_HID: np.full(B * hidden, np.nan, dtype=np.float32),
        S_WQKV: gdn.in_proj_qkv.weight.detach().numpy().T.reshape(-1),
        S_WFIR: gdn.conv1d.weight.squeeze(1).detach().numpy().reshape(-1),
        S_WZ: gdn.in_proj_z.weight.detach().numpy().T.reshape(-1),
        S_WA: gdn.in_proj_a.weight.detach().numpy().T.reshape(-1),
        S_WB: gdn.in_proj_b.weight.detach().numpy().T.reshape(-1),
        S_WOUT: gdn.out_proj.weight.detach().numpy().T.reshape(-1),
        720: gdn.A_log.detach().numpy().copy(),
        721: gdn.dt_bias.detach().numpy().copy(),
        722: gdn.norm.weight.detach().numpy().copy(),
    }
    for buf in workspace_plan.buffers:
        sa[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(schedule, workers=3, storage_arrays=sa, graph=graph, workspace_plan=workspace_plan)

    # oracle: real floe module, per batch row, real module states
    oracle_conv = torch.from_numpy(conv0.copy())  # [B, K-1, C]
    oracle_ssm = torch.from_numpy(ssm0.copy())  # [B, NV, HV, HK]
    for step in range(4):
        hid_np = rng.standard_normal((B, hidden)).astype(np.float32) * 0.5
        sa[S_HID][:] = hid_np.reshape(-1)
        executor.run({})
        got = np.array(executor.tensor(y.value.name), copy=True)
        assert np.isfinite(got).all(), f"step {step}: NaN canary tripped"
        for bb in range(B):
            output, new_ssm, new_conv = gdn.forward(
                torch.from_numpy(hid_np[bb]).reshape(1, hidden),  # seq==1 branch
                oracle_ssm[bb], oracle_conv[bb],
            )
            torch.testing.assert_close(torch.from_numpy(got[bb]), output.reshape(-1), rtol=1e-4, atol=1e-5)
            oracle_ssm[bb], oracle_conv[bb] = new_ssm.detach(), new_conv.detach()
    # both pools must equal the oracle's final states (in-place RMW chains)
    torch.testing.assert_close(
        torch.from_numpy(sa[S_SSM].reshape(B, NV, HV, HK).copy()), oracle_ssm, rtol=1e-5, atol=1e-5,
    )
    torch.testing.assert_close(
        torch.from_numpy(sa[S_CONV].reshape(B, K - 1, conv_dim).copy()), oracle_conv, rtol=1e-5, atol=1e-5,
    )


# ===========================================================================
# Device template (_t_gdn_heads_batched) — CUDA only
# ===========================================================================

gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires CUDA (the B=1 path is covered by the 27B tests in test_megakernel_27b.py)",
)


@gpu
def test_device_t_gdn_heads_batched_matches_floe_oracle():
    pytest.importorskip("triton")
    from vkernels.compiler.device_triton import _t_gdn_heads_batched

    gdn, _cfg, qwen35_gdn = _floe_gdn()
    dev = torch.device("cuda")
    rng = np.random.default_rng(99)
    state = (rng.standard_normal((B, NV, HV, HK)) * 0.3).astype(np.float32)
    feeds = {n: (rng.standard_normal(sh) * 0.7).astype(np.float32) for n, sh in
             (("q", (B, NK, HK)), ("k", (B, NK, HK)), ("v", (B, NV, HV)), ("z", (B, NV, HV)),
              ("a", (B, NV)), ("b", (B, NV)))}

    tensors = {n: torch.from_numpy(arr).to(dev) for n, arr in feeds.items()}
    state_t = torch.from_numpy(state).to(dev)
    a_log = torch.from_numpy(gdn.A_log.detach().numpy().copy()).to(dev)
    dt_bias = torch.from_numpy(gdn.dt_bias.detach().numpy().copy()).to(dev)
    norm_w = torch.from_numpy(gdn.norm.weight.detach().numpy().copy()).to(dev)
    out = torch.empty(B, NV, HV, device=dev, dtype=torch.float32)

    _t_gdn_heads_batched[(4,)](
        tensors["q"], tensors["k"], tensors["v"], tensors["z"], tensors["a"], tensors["b"],
        a_log, dt_bias, norm_w, state_t, out, B, NV, NK, HV, HK,
        eps=_EPS, scale=_SCALE, num_warps=4,
    )
    torch.cuda.synchronize()

    expected = torch.empty(B, NV, HV)
    expected_state = torch.empty_like(torch.from_numpy(state))
    for bb in range(B):
        o_gated, new_state = _floe_condition_and_recurse(
            gdn, qwen35_gdn,
            tensors["q"][bb].cpu(), tensors["k"][bb].cpu(), tensors["v"][bb].cpu(),
            tensors["z"][bb].cpu(), tensors["a"][bb].cpu(), tensors["b"][bb].cpu(),
            torch.from_numpy(state)[bb],
        )
        expected[bb] = o_gated
        expected_state[bb] = new_state
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-4, atol=1e-5)
    # in-place state update on the device pool
    torch.testing.assert_close(state_t.cpu(), expected_state, rtol=1e-5, atol=1e-6)
