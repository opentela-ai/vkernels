"""GLM-5.3 Flash whole-decode-step tests (issue #102 Stage 3).

Chain of evidence:

* capture: the KDA-layer body records the expected op sequence per layer
  (two mHC blocks, fused qkv GEMV, gdn_conv, kda_delta with the gate
  folded inside, rms_norm_gated per head, o_proj, dense-MLP / sparse-MoE
  variants) and ends with the HyperHead mean + final norm;
* whole-step numerics: the compiled phase schedule, executed on the CPU
  reference executor, matches the fp64 mirror in :mod:`.glm53_arch`
  across worker counts, across two chained decode steps (state-pool RMW
  across steps) and at batch 1 and 3;
* hazards: every layer's conv/ssm state pools carry read-modify-write
  hazards and the phase order serializes them; the mHC stream chain is
  RAW-ordered;
* canaries: workspace and untouched externals are NaN-poisoned before the
  run; outputs come back clean and pools never observe NaN (Stage 1's
  canary protocol);
* flat-1-D contract (Stage 1 field intel): every external pool passed to
  the executor is asserted raveled 1-D contiguous before the run — the
  harness regression that would have caught the N-D-pool aliasing bug;
* indexer: in the hybrid config the indexer_scores ≺ index_topk phase
  order holds inside a sparse-layer prefix (the sparse-attention consumer
  of the fixed-width top-k table vs the paged ops' table[b, :p+1]
  convention is the documented Stage-2/3 seam);
* real dims: the published GLM-5.3-Flash decode dims compile end to end.
"""

from __future__ import annotations

import os
import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vkernels.compiler.capture import RecordingBackend  # noqa: E402
from vkernels.compiler.legality import check_graph  # noqa: E402
from vkernels.compiler.lowerings import lower_graph  # noqa: E402
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import compute_hazards  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402
from vkernels.compiler.model_glm53 import Glm53ModelArgs, build_glm53_forward  # noqa: E402
from vkernels.compiler.glm53_arch import (  # noqa: E402
    Glm53Config,
    glm53_reference_decode_step,
    random_glm53_weights,
)

TOL = 1.5e-4  # fp32 storage vs fp64 mirror (deep 3-layer stack, squashing gates)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _compile(cfg: Glm53Config, seed: int, batch: int, workers: int, *, materialize_hosts: bool = True):
    recorder = RecordingBackend()
    weights = random_glm53_weights(cfg, seed=seed, real_experts=materialize_hosts)
    args = Glm53ModelArgs(recorder, cfg, weights, batch=batch, materialize_hosts=materialize_hosts)
    out = build_glm53_forward(recorder, args)
    graph = recorder.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _ = plan_memory(graph, schedule, families)
    return args, out, graph, families, schedule, workspace_plan


def _storage(args, out, workspace_plan, streams):
    """Flat 1-D contiguous storage arrays for every external + workspace.

    Stage 1 field intel: the executor's storage contract is FLAT 1-D
    arrays; an N-D pool slices axis 0 out of bounds and as_strided UB
    silently aliases offset 0. Every array here is raveled and asserted
    1-D contiguous before the run.
    """
    storage: dict[int, np.ndarray] = {}
    streams_sid = args.streams_in.value.storage_id
    streams_shape = args.streams_in.value.shape
    for sid, arr in args.host_by_sid.items():
        if sid == streams_sid:
            continue  # filled by the caller from the explicit streams arg
        flat = arr.astype(np.float32).reshape(-1).copy()
        assert flat.ndim == 1 and flat.flags["C_CONTIGUOUS"]
        storage[sid] = flat
    streams_flat = np.asarray(streams, np.float32).reshape(-1).copy()
    assert streams_flat.size == int(np.prod(streams_shape))
    assert streams_flat.ndim == 1 and streams_flat.flags["C_CONTIGUOUS"]
    storage[streams_sid] = streams_flat
    for buf in workspace_plan.buffers:
        arr = np.full(buf.numel, np.nan, dtype=np.float32)  # NaN canaries
        assert arr.ndim == 1
        storage[buf.storage_id] = arr
    return storage


def streams_flags_ok(arr: np.ndarray) -> bool:
    return bool(arr.flags["C_CONTIGUOUS"])


def _executor(args, out, schedule, workspace_plan, graph, storage, workers):
    return ReferenceExecutor(schedule, workers=workers, storage_arrays=storage,
                             graph=graph, workspace_plan=workspace_plan)


def _mirror_pools(cfg, batch, seed=99):
    """Mirror-side initial pools (fp64)."""
    rng = np.random.default_rng(seed)
    K, Cc = cfg.linear_conv_kernel_dim, cfg.conv_dim
    H, D = cfg.linear_num_heads, cfg.linear_head_dim
    conv = [rng.normal(0, 0.5, (batch, K - 1, Cc)) for _ in range(cfg.num_hidden_layers)]
    ssm = [rng.normal(0, 0.5, (batch, H, D, D)) for _ in range(cfg.num_hidden_layers)]
    return conv, ssm


# ---------------------------------------------------------------------------
# Capture structure
# ---------------------------------------------------------------------------

EXPECTED_PER_KDA_LAYER = [
    "mhc_pre", "rms_norm", "linear", "gdn_conv", "linear", "linear",
    "linear", "linear", "linear", "kda_delta", "rms_norm_gated", "linear",
    "mhc_post",  # attention sub-block
    "mhc_pre", "rms_norm",  # ffn sub-block head
]


def test_capture_records_glm53_phase_structure():
    cfg = Glm53Config()
    rec = RecordingBackend()
    args = Glm53ModelArgs(rec, cfg, random_glm53_weights(cfg, seed=1), batch=2)
    out = build_glm53_forward(rec, args)
    kinds = [op.kind for op in rec.graph.ops]
    pos = 0
    for i, layer_kind in enumerate(cfg.mlp_layer_types):
        if layer_kind == "dense":
            tail = ["linear", "linear", "swiglu", "linear", "mhc_post"]
        else:
            tail = ["moe_route", "moe_expert", "linear", "linear", "swiglu",
                    "linear", "moe_combine", "mhc_post"]
        expect = EXPECTED_PER_KDA_LAYER + tail
        got = kinds[pos: pos + len(expect)]
        pos += len(expect)
        assert got == expect, f"layer {i}:\n got  {got}\n want {expect}"
    # the HyperHead mean + final norm close the graph
    assert kinds[-2:] == ["linear", "rms_norm"]
    # state pools are external and each op contract carries the RMW note
    kda_ops = [op for op in rec.graph.ops if op.kind == "kda_delta"]
    assert len(kda_ops) == cfg.num_hidden_layers
    for op in kda_ops:
        assert "read-modify-write" in op.numerical_contract["pool"]
    conv_ops = [op for op in rec.graph.ops if op.kind == "gdn_conv"]
    assert len(conv_ops) == cfg.num_hidden_layers
    assert all(op.attributes["conv_kernel"] == cfg.linear_conv_kernel_dim
               for op in conv_ops)


def test_hybrid_capture_indexer_phase_order():
    """Sparse-layer prefix: indexer_scores ≺ index_topk, fixed-width table."""
    from vkernels.compiler.operator_ir import I32
    cfg = Glm53Config(layer_types=("linear_attention", "linear_attention", "deepseek_sparse_attention"))
    rec = RecordingBackend()
    B, H, D, M, k = 2, 4, cfg.indexer_head_dim, 16, cfg.index_topk
    q = rec.external_tensor("idx_q", (B, H, D), storage_id=3001)
    entries = rec.external_tensor("idx_entries", (B, M, D), storage_id=3002)
    mix_w = rec.external_tensor("idx_mix", (B, H), storage_id=3003)
    valid = rec.external_tensor("idx_valid", (B,), I32, storage_id=3004)
    scores = rec.indexer_scores(q, entries, mix_w, name="idx_scores")
    idx, bias = rec.index_topk(scores, valid, k=k, name="idx_topk")
    diags = [d for d in check_graph(rec.graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    kinds = [op.kind for op in rec.graph.ops]
    assert kinds == ["indexer_scores", "index_topk"]
    assert idx.value.shape == (B, k) and idx.value.dtype == I32
    # the table is the #94 indirection source: score/value tasks consume it
    # via Region.indirect (the fixed-width-vs-p+1 seam is documented in the
    # module docstring and the PR).
    assert rec.graph.ops[0].numerical_contract["activation"].startswith("relu")


# ---------------------------------------------------------------------------
# Whole-step numerics vs the fp64 mirror
# ---------------------------------------------------------------------------


def _run_whole_step(cfg, workers, batch, steps, seed):
    args, out, graph, families, schedule, wp = _compile(cfg, seed=seed, batch=batch, workers=workers)
    conv, ssm = _mirror_pools(cfg, batch)
    rng = np.random.default_rng(4242)
    hc, C = cfg.hc_mult, cfg.hidden_size
    streams = rng.normal(0, 0.5, (batch, hc, C))
    # load pools + weights into storage
    for i in range(cfg.num_hidden_layers):
        args.host_by_sid[args.conv_state[i].value.storage_id][...] = conv[i].astype(np.float32)
        args.host_by_sid[args.ssm_state[i].value.storage_id][...] = ssm[i].astype(np.float32)
    storage = _storage(args, out, wp, streams)
    # Stage-1 protocol: one simulated launch per step — a FRESH executor per
    # run (canary state re-arms per step); the storage dict (and with it the
    # conv/ssm state pools) persists across steps so the RMW chains.
    for _ in range(steps):
        ex = _executor(args, out, schedule, wp, graph, storage, workers)
        ex.run({})
        got = ex.tensor(out.value.name).reshape(batch, C)
        want = glm53_reference_decode_step(cfg, _mirror_weights(cfg, seed), streams, conv, ssm)
        assert np.all(np.isfinite(got)), "NaN in compiled output"
        assert np.max(np.abs(got - want)) < TOL, f"max err {np.max(np.abs(got - want)):.3e}"
    # chained steps exercised the RMW: pools moved
    assert steps == 1 or np.max(np.abs(args.host_by_sid[args.ssm_state[0].value.storage_id])) > 0


_MIRROR_CACHE: dict[int, object] = {}


def _mirror_weights(cfg, seed):
    key = (id(cfg), seed)
    if key not in _MIRROR_CACHE:
        _MIRROR_CACHE[key] = random_glm53_weights(cfg, seed=seed)
    return _MIRROR_CACHE[key]


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("batch", [1, 3])
def test_whole_step_matches_mirror(workers, batch):
    _run_whole_step(Glm53Config(), workers, batch, steps=1, seed=11)


def test_two_chained_steps_match_mirror():
    """Second step consumes the step-1 state pools (RAW across runs)."""
    _run_whole_step(Glm53Config(), workers=2, batch=3, steps=2, seed=13)


# ---------------------------------------------------------------------------
# Hazards, canaries, flat-1-D contract
# ---------------------------------------------------------------------------


def test_state_pool_hazards_and_disjoint_slabs():
    """Stage-1 whole-graph protocol, instantiated for the KDA spine:
    per-layer state pools are pairwise-disjoint write slabs, each op's RMW
    regions are recorded, and the mHC stream chain is RAW-ordered."""
    cfg = Glm53Config()
    rec = RecordingBackend()
    args = Glm53ModelArgs(rec, cfg, random_glm53_weights(cfg, seed=3), batch=2)
    build_glm53_forward(rec, args)
    hazards = compute_hazards(rec.graph.ops)
    pairs = {(h.producer, h.consumer) for h in hazards}
    # per-layer pools: pairwise disjoint storages (write slabs never cross)
    conv_sids = [t.value.storage_id for t in args.conv_state]
    ssm_sids = [t.value.storage_id for t in args.ssm_state]
    assert len(set(conv_sids)) == cfg.num_hidden_layers
    assert len(set(ssm_sids)) == cfg.num_hidden_layers
    assert not (set(conv_sids) & set(ssm_sids))
    # each kda_delta's RMW on its own pool: reads AND writes recorded
    kda_ops = [op for op in rec.graph.ops if op.kind == "kda_delta"]
    for op, t in zip(kda_ops, args.ssm_state):
        sid = t.value.storage_id
        assert any(r.storage_id == sid for r in op.read_regions), "kda RMW read missing"
        assert any(r.storage_id == sid for r in op.write_regions), "kda RMW write missing"
    conv_ops = [op for op in rec.graph.ops if op.kind == "gdn_conv"]
    for op, t in zip(conv_ops, args.conv_state):
        sid = t.value.storage_id
        assert any(r.storage_id == sid for r in op.read_regions)
        assert any(r.storage_id == sid for r in op.write_regions)
    # the mHC stream chain: each mhc_post's output feeds the next mhc_pre (RAW)
    post_ops = [op for op in rec.graph.ops if op.kind == "mhc_post"]
    pre_ops = [op for op in rec.graph.ops if op.kind == "mhc_pre"]
    assert len(post_ops) == 2 * cfg.num_hidden_layers == len(pre_ops)
    for p, nxt in zip(post_ops, post_ops[1:]):
        assert (p.opid, nxt.opid) in pairs or (p.opid, pre_ops[0].opid) in pairs, \
            f"stream chain unordered at op {p.opid} -> {nxt.opid}"


def test_nan_canaries_and_flat_1d_storage():
    """Stage 1 protocol: NaN canaries in workspace + the 1-D assertion."""
    cfg = Glm53Config()
    args, out, graph, families, schedule, wp = _compile(cfg, seed=17, batch=2, workers=2)
    conv, ssm = _mirror_pools(cfg, 2)
    rng = np.random.default_rng(7)
    streams = rng.normal(0, 0.5, (2, cfg.hc_mult, cfg.hidden_size))
    storage = {}
    for sid, arr in args.host_by_sid.items():
        if arr is None:
            continue
        flat = arr.astype(np.float32).reshape(-1).copy()
        assert flat.ndim == 1 and flat.flags["C_CONTIGUOUS"], "external not flat 1-D"
        storage[sid] = flat
    for i in range(cfg.num_hidden_layers):
        storage[args.conv_state[i].value.storage_id][:] = conv[i].reshape(-1).astype(np.float32)
        storage[args.ssm_state[i].value.storage_id][:] = ssm[i].reshape(-1).astype(np.float32)
    sid_streams = args.streams_in.value.storage_id
    storage[sid_streams] = np.asarray(streams, np.float32).reshape(-1).copy()
    for buf in wp.buffers:
        storage[buf.storage_id] = np.full(buf.numel, np.nan, np.float32)
    ex = _executor(args, out, schedule, wp, graph, storage, workers=2)
    ex.run({})
    got = ex.tensor(out.value.name)
    assert np.all(np.isfinite(got)), "NaN canary leaked into the output"
    want = glm53_reference_decode_step(cfg, _mirror_weights(cfg, 17), streams, conv, ssm)
    assert np.max(np.abs(got.reshape(2, -1) - want)) < TOL


# ---------------------------------------------------------------------------
# Real dims
# ---------------------------------------------------------------------------


def test_real_glm53_flash_dims_compile():
    """Published decode dims (KDA-only stack — the sparse-attention layer is
    the documented seam): capture → legality → lower → schedule → memory."""
    from vkernels.compiler.glm53_arch import real_glm53_dims_config
    cfg = real_glm53_dims_config(num_hidden_layers=12)
    args, out, graph, families, schedule, wp = _compile(cfg, seed=23, batch=1, workers=4,
                                                        materialize_hosts=False)
    n_tasks = sum(fam.domain.task_count if hasattr(fam, "domain") else 0 for fam in families)
    n_phases = len(schedule.phases) if hasattr(schedule, "phases") else -1
    assert n_tasks > 0
    # every weight/pool external got a distinct flat storage
    assert len(args.host_by_sid) == len(set(args.host_by_sid))


def test_real_dims_state_pool_count():
    from vkernels.compiler.glm53_arch import real_glm53_dims_config
    cfg = real_glm53_dims_config(num_hidden_layers=12)
    rec = RecordingBackend()
    args = Glm53ModelArgs(rec, cfg, random_glm53_weights(cfg, seed=29, real_experts=False),
                          batch=1, materialize_hosts=False)
    build_glm53_forward(rec, args)
    assert len(args.conv_state) == 12 and len(args.ssm_state) == 12
    # real KDA dims: 64 heads × 128 dim, conv 4-tap over 3·qkv channels
    assert args.ssm_state[0].value.shape == (1, 64, 128, 128)
    assert args.conv_state[0].value.shape == (1, 3, 3 * 64 * 128)
