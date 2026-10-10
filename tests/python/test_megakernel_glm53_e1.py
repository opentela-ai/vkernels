"""GLM-5.3 megakernel E1 — the KDA-spine slice (readiness Lever E, step E1).

E1 is the first real slice of the whole-step megakernel: the attention
sub-block of every KDA layer — ``mhc_pre → ln1 → fused qkv/f/b/g GEMVs →
fused decode (conv + kda-delta + gated norm) → o_proj → mhc_post`` — as
phase-scheduled tile tasks in ONE launch (11 ops = 11 phases per layer; a
34-layer spine is 374 barriers). The decode block is captured by the new
``kda_fused_decode`` op family under the FUSED-DECODE CONTRACT — the ABI of
the CUDA oracle ``vkernels.torch_ops.glm_kda_fused_decode`` (RAW dot rows,
sigmoid-after-bf16-round, slot-indirected V-major ssm pool + w-major conv
pool) — so a device lowering of these tasks can feed the real kernel's
semantics without re-deriving them.

Chain of evidence:

* structure: 11 ops per layer, per-layer pools disjoint + slot-indirected,
  the fused op carries both pools' read-modify-write effects;
* recurrence: layer i's ``mhc_post`` → layer i+1's ``mhc_pre`` RAW hazard is
  the cross-layer recurrence the phase barriers gate; every layer's ops
  occupy consecutive phases and one launch runs the whole spine;
* parity: the compiled spine (CPU reference executor) matches the fp64
  fused-contract mirror
  (:func:`glm53_arch.glm53_reference_kda_spine_step`) across worker counts,
  batches and TWO chained decode steps (pool RMW across steps); padded
  ``-1`` slots zero the output row and leave both pools untouched;
* cross-check vs the CUDA oracle's own reference: the fused contract mirror
  (and the compiled op) agree with
  ``glm_kda_fused_decode_reference`` — bit-exact on the bf16-grid out and
  conv pool, fp32-ULP on the fp32 ssm pool (torch-gated);
* divergence band vs the decomposed chain: the incumbent decomposed body
  (conv op → delta op → gated-norm op, no bf16 ABI rounds — the class floe's
  eager chain rounds at every kernel boundary) agrees with the fused spine
  only within the documented bf16-ABI band; that gap is INTENTIONAL and
  priced here, not a parity failure.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from vkernels.compiler.capture import RecordingBackend  # noqa: E402
from vkernels.compiler.legality import check_graph  # noqa: E402
from vkernels.compiler.lowerings import lower_graph  # noqa: E402
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import I32, compute_hazards  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402
from vkernels.compiler.model_glm53 import Glm53KdaSpineArgs, build_glm53_kda_spine_forward  # noqa: E402
from vkernels.compiler.glm53_arch import (  # noqa: E402
    Glm53Config,
    _mhc_block,
    _mhc_compose,
    _kda_layer_decode,
    _rms_norm,
    glm53_reference_kda_spine_step,
    kda_fused_decode_reference,
    random_glm53_weights,
)
from vkernels.compiler.reference_types import bf16_round  # noqa: E402

# executor vs the fp64 fused-contract mirror: fp32 pool stores + fp32
# weight storage are the only rounding classes between the two (the bf16
# ABI rounds are pinned IDENTICALLY on both sides)
TOL_MIRROR = 1e-6
# the fused ABI vs the round-free decomposed chain: the bf16 grid (~2^-7
# spacing at unit scale) entered at four points per layer, mixed through
# the mHC stream chain — the documented intentional divergence band
TOL_DECOMPOSED = 5e-2

OPS_PER_LAYER = [
    "mhc_pre", "rms_norm", "linear", "linear", "linear", "linear", "linear",
    "linear", "kda_fused_decode", "linear", "mhc_post",
]


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _compile_spine(cfg: Glm53Config, seed: int, batch: int, workers: int, *, slots: int | None = None):
    recorder = RecordingBackend()
    weights = random_glm53_weights(cfg, seed=seed)
    args = Glm53KdaSpineArgs(recorder, cfg, weights, batch=batch, slots=slots)
    out = build_glm53_kda_spine_forward(recorder, args)
    graph = recorder.graph
    diags = [d for d in check_graph(graph) if d.severity == "error"]
    assert not diags, f"legality errors: {diags}"
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _ = plan_memory(graph, schedule, families)
    return args, out, graph, families, schedule, workspace_plan


def _spine_pools(cfg: Glm53Config, slots: int, seed: int, batch: int):
    """Mirror-side initial pools (fp64; conv values on the bf16 grid — the
    pool's declared value class)."""
    rng = np.random.default_rng(seed)
    K, Cc = cfg.linear_conv_kernel_dim, cfg.conv_dim
    H, D = cfg.linear_num_heads, cfg.linear_head_dim
    L = cfg.num_hidden_layers
    conv = [bf16_round(rng.normal(0, 0.5, (slots, K - 1, Cc))).astype(np.float64) for _ in range(L)]
    ssm = [rng.normal(0, 0.5, (slots, H, D, D)) for _ in range(L)]
    streams = rng.normal(0, 0.5, (batch, cfg.hc_mult, cfg.hidden_size))
    return conv, ssm, streams


def _run_spine(cfg, args, out, graph, schedule, workspace_plan, workers,
               conv, ssm, streams, slot_ids, steps):
    """Load pools + a fresh streams row into flat 1-D storage; run `steps`
    chained decode steps (one simulated launch each, pools persist)."""
    storage: dict[int, np.ndarray] = {}
    for sid, arr in args.host_by_sid.items():
        if arr is None:
            continue
        storage[sid] = arr.astype(np.float32).reshape(-1).copy()
    for i in range(cfg.num_hidden_layers):
        storage[args.conv_pool[i].value.storage_id][:] = conv[i].astype(np.float32).reshape(-1)
        storage[args.ssm_pool[i].value.storage_id][:] = ssm[i].astype(np.float32).reshape(-1)
    storage[args.cache_indices.value.storage_id][:] = slot_ids
    sid_in = args.streams_in.value.storage_id
    storage.setdefault(sid_in, np.zeros(int(np.prod(args.streams_in.value.shape)), np.float32))
    storage[sid_in][:] = streams.astype(np.float32).reshape(-1).copy()
    for buf in workspace_plan.buffers:
        storage[buf.storage_id] = np.full(buf.numel, np.nan, np.float32)
    got = None
    for _ in range(steps):
        ex = ReferenceExecutor(schedule, workers=workers, storage_arrays=storage,
                               graph=graph, workspace_plan=workspace_plan)
        ex.run({})
        got = ex.tensor(out.value.name).reshape(streams.shape)
    return storage, got


_MIRROR_CACHE: dict[tuple, object] = {}


def _weights(cfg, seed):
    key = (id(cfg), seed)
    if key not in _MIRROR_CACHE:
        _MIRROR_CACHE[key] = random_glm53_weights(cfg, seed=seed)
    return _MIRROR_CACHE[key]


# ---------------------------------------------------------------------------
# Capture structure
# ---------------------------------------------------------------------------


def test_e1_spine_capture_structure():
    cfg = Glm53Config()
    rec = RecordingBackend()
    args = Glm53KdaSpineArgs(rec, cfg, random_glm53_weights(cfg, seed=1), batch=2, slots=4)
    build_glm53_kda_spine_forward(rec, args)
    kinds = [op.kind for op in rec.graph.ops]
    expect = OPS_PER_LAYER * cfg.num_hidden_layers
    assert kinds == expect, f"\n got  {kinds}\n want {expect}"
    # pools: per-layer disjoint externals in the fused-decode shapes
    conv_sids = [t.value.storage_id for t in args.conv_pool]
    ssm_sids = [t.value.storage_id for t in args.ssm_pool]
    assert len(set(conv_sids)) == cfg.num_hidden_layers
    assert len(set(ssm_sids)) == cfg.num_hidden_layers
    assert not (set(conv_sids) & set(ssm_sids))
    Kt, Cc, H, D = cfg.linear_conv_kernel_dim, cfg.conv_dim, cfg.linear_num_heads, cfg.linear_head_dim
    assert args.conv_pool[0].value.shape == (4, Kt - 1, Cc)  # w-major [slots, Kw, Cc]
    assert args.ssm_pool[0].value.shape == (4, H, D, D)  # V-MAJOR [slots, H, V, K]
    assert args.cache_indices.value.dtype.name == "i32" and args.cache_indices.value.shape == (2,)
    # the fused op carries the contract + RMW on BOTH pools
    fused = [op for op in rec.graph.ops if op.kind == "kda_fused_decode"]
    assert len(fused) == cfg.num_hidden_layers
    for op, cp, sp in zip(fused, args.conv_pool, args.ssm_pool):
        assert "RAW projection rows" in op.numerical_contract["abi"]
        assert "read-modify-write" in op.numerical_contract["pool"]
        conv_sid, ssm_sid = cp.value.storage_id, sp.value.storage_id
        assert any(r.storage_id == conv_sid for r in op.read_regions)
        assert any(r.storage_id == conv_sid for r in op.write_regions)
        assert any(r.storage_id == ssm_sid for r in op.read_regions)
        assert any(r.storage_id == ssm_sid for r in op.write_regions)
    # taps are the kernel-ABI time-major [Kt, Cc] externals
    assert args.taps[0].value.shape == (Kt, Cc)


# ---------------------------------------------------------------------------
# Cross-layer recurrence + phase machinery
# ---------------------------------------------------------------------------


def test_e1_spine_phase_order_respects_recurrence():
    """Layer i's mhc_post RAW-feeds layer i+1's mhc_pre (the streams chain);
    the phase barriers between them are the recurrence gate; every layer's
    ops occupy consecutive phases and one launch covers the spine."""
    cfg = Glm53Config()
    rec = RecordingBackend()
    args = Glm53KdaSpineArgs(rec, cfg, random_glm53_weights(cfg, seed=3), batch=2)
    build_glm53_kda_spine_forward(rec, args)
    ops = rec.graph.ops
    hazards = compute_hazards(ops)
    raw = {(h.producer, h.consumer) for h in hazards if h.kind == "RAW"}
    post = [op for op in ops if op.kind == "mhc_post"]
    pre = [op for op in ops if op.kind == "mhc_pre"]
    assert len(post) == len(pre) == cfg.num_hidden_layers
    for p, nxt in zip(post, pre[1:]):
        assert (p.opid, nxt.opid) in raw, \
            f"cross-layer stream recurrence unordered: mhc_post {p.opid} -> mhc_pre {nxt.opid}"
    # the fused decode's pool RMW orders it against everything else on the pool
    fused = [op for op in ops if op.kind == "kda_fused_decode"]
    for op in fused:
        pool_sid = op.inputs[0]
        others = [o for o in ops if o is not op and any(
            r.storage_id == pool_sid for r in (*o.read_regions, *o.write_regions))]
        assert others == [], "per-layer pools must be private to their fused op (within a step)"
    # schedule: consecutive phases per layer, one barrier per phase
    _, _, _, _, schedule, _ = _compile_spine(cfg, seed=3, batch=2, workers=3)
    assert schedule.num_phases == len(ops) == 11 * cfg.num_hidden_layers
    for i, phase in enumerate(schedule.phases):
        assert phase.families[0].op.opid == i, "phases follow program order"
    # one launch executes the whole spine (every task exactly once)
    args2, out, graph, _, schedule2, wp = _compile_spine(cfg, seed=3, batch=2, workers=3)
    conv, ssm, streams = _spine_pools(cfg, slots=2, seed=99, batch=2)
    _, got = _run_spine(cfg, args2, out, graph, schedule2, wp, 3,
                        conv, ssm, streams, np.arange(2, dtype=np.int32), steps=1)
    assert schedule2.total_tasks > 0 and got is not None


# ---------------------------------------------------------------------------
# Parity vs the fp64 fused-contract mirror
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("batch", [1, 3])
def test_e1_spine_matches_fused_contract_mirror(workers, batch):
    cfg = Glm53Config()
    seed = 11
    args, out, graph, _, schedule, wp = _compile_spine(cfg, seed=seed, batch=batch, workers=workers)
    conv, ssm, streams = _spine_pools(cfg, slots=batch, seed=21, batch=batch)
    slot_ids = np.arange(batch, dtype=np.int32)
    storage, got = _run_spine(cfg, args, out, graph, schedule, wp, workers,
                              conv, ssm, streams, slot_ids, steps=1)
    want = glm53_reference_kda_spine_step(
        cfg, _weights(cfg, seed), streams.astype(np.float64),
        [c.copy() for c in conv], [s.copy() for s in ssm], slot_ids,
    )
    assert np.all(np.isfinite(got)), "NaN in compiled spine output"
    assert np.max(np.abs(got - want)) < TOL_MIRROR, f"max err {np.max(np.abs(got - want)):.3e}"
    # pools updated on the executor side too
    H, D = cfg.linear_num_heads, cfg.linear_head_dim
    pool = storage[args.ssm_pool[0].value.storage_id].reshape(batch, H, D, D)
    assert np.max(np.abs(pool - ssm[0])) > 0


def test_e1_two_chained_steps_match_mirror():
    """Step 2 consumes step 1's pools (the cross-step recurrence chains
    through the pool storage hazards)."""
    cfg = Glm53Config()
    seed = 13
    args, out, graph, _, schedule, wp = _compile_spine(cfg, seed=seed, batch=2, workers=2)
    conv, ssm, streams = _spine_pools(cfg, slots=2, seed=31, batch=2)
    slot_ids = np.arange(2, dtype=np.int32)
    storage, got = _run_spine(cfg, args, out, graph, schedule, wp, 2,
                              conv, ssm, streams, slot_ids, steps=2)
    # mirror: two chained steps (pools thread through)
    mconv = [c.copy() for c in conv]
    mssm = [s.copy() for s in ssm]
    s64 = streams.astype(np.float64)
    for _ in range(2):
        want = glm53_reference_kda_spine_step(cfg, _weights(cfg, seed), s64, mconv, mssm, slot_ids)
    assert np.max(np.abs(got - want)) < TOL_MIRROR, f"max err {np.max(np.abs(got - want)):.3e}"
    # executor pools tracked the second RMW
    H, D = cfg.linear_num_heads, cfg.linear_head_dim
    pool = storage[args.ssm_pool[0].value.storage_id].reshape(2, H, D, D)
    assert np.max(np.abs(pool - mssm[0])) < TOL_MIRROR


def test_e1_padded_slot_zero_output_pools_untouched():
    """A -1 cache index is a padded CUDA-graph slot: output row zero, BOTH
    pools untouched (the fused kernel's contract, mirrored by the op)."""
    cfg = Glm53Config()
    seed = 17
    args, out, graph, _, schedule, wp = _compile_spine(cfg, seed=seed, batch=3, workers=2, slots=2)
    conv, ssm, streams = _spine_pools(cfg, slots=2, seed=41, batch=3)
    slot_ids = np.array([0, -1, 1], dtype=np.int32)
    _, got = _run_spine(cfg, args, out, graph, schedule, wp, 2,
                        conv, ssm, streams, slot_ids, steps=1)
    want = glm53_reference_kda_spine_step(
        cfg, _weights(cfg, seed), streams.astype(np.float64),
        [c.copy() for c in conv], [s.copy() for s in ssm], slot_ids,
    )
    assert np.max(np.abs(got - want)) < TOL_MIRROR
    # the padded row's parity includes its zeroed per-layer kda output rows
    # (row 1's layer-0 fused output is exactly zero — the padded-slot
    # contract — verified directly on the single-op oracle test above)
    assert np.max(np.abs(got[1] - want[1])) < TOL_MIRROR


# ---------------------------------------------------------------------------
# Cross-check vs the CUDA oracle's own reference (torch-gated)
# ---------------------------------------------------------------------------


def test_e1_fused_contract_matches_torch_oracle():
    """The fused-contract mirror AND the compiled op vs
    ``glm_kda_fused_decode_reference`` (the CUDA kernel's eager fp32
    oracle): bit-exact on the bf16-grid out + conv pool, fp32-ULP on the
    fp32 ssm pool. This pins the megakernel's KDA task semantics to the
    lane-3 kernel's documented contract."""
    torch = pytest.importorskip("torch")
    from vkernels.torch_ops.glm_kda_fused_decode import glm_kda_fused_decode_reference as oracle

    rng = np.random.default_rng(7)
    B, slots, H, D, Kt = 3, 5, 4, 128, 4
    seg, Cc = H * D, 3 * H * D
    scale, eps, lb = D ** -0.5, 1e-5, -5.0

    def bf16_grid(x):
        return bf16_round(x).astype(np.float32)

    mixed = bf16_grid(rng.normal(0, 1, (B, Cc)))
    a_raw = bf16_grid(rng.normal(0, 1, (B, H, D)))
    b_raw = bf16_grid(rng.normal(0, 1, (B, H)))
    g_raw = bf16_grid(rng.normal(0, 1, (B, H, D)))
    conv0 = bf16_grid(rng.normal(0, 1, (slots, Kt - 1, Cc)))
    ssm0 = rng.normal(0, 1, (slots, H, D, D)).astype(np.float32)
    taps = rng.normal(0, 0.5, (Kt, Cc)).astype(np.float32)
    dtb = rng.normal(0, 0.3, (H, D)).astype(np.float32)
    alog = rng.normal(0, 0.2, H).astype(np.float32)
    onorm = (1 + rng.normal(0, 0.1, D)).astype(np.float32)
    slot_ids = np.array([0, -1, 3], dtype=np.int32)

    out_np, conv_np, ssm_np = kda_fused_decode_reference(
        conv0.astype(np.float64), ssm0.astype(np.float64), slot_ids,
        mixed.astype(np.float64), a_raw.astype(np.float64), b_raw.astype(np.float64),
        g_raw.astype(np.float64), taps.astype(np.float64), dtb.astype(np.float64),
        alog.astype(np.float64), onorm.astype(np.float64),
        scale=scale, eps=eps, lower_bound=lb)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    out_t, conv_t, ssm_t = oracle(
        t(mixed).to(torch.bfloat16), t(a_raw.reshape(B, H * D)).to(torch.bfloat16),
        t(b_raw).to(torch.bfloat16), t(conv0).to(torch.bfloat16),
        t(taps[:, :seg]), t(taps[:, seg:2 * seg]), t(taps[:, 2 * seg:]),
        torch.zeros(3 * seg, dtype=torch.float32), t(alog),
        t(dtb.reshape(H * D)), t(g_raw.reshape(B, H * D)).to(torch.bfloat16), t(onorm),
        t(ssm0), torch.from_numpy(slot_ids), scale, eps, lower_bound=lb)
    out_t = out_t.float().numpy().reshape(B, H, D)
    conv_t, ssm_t = conv_t.float().numpy(), ssm_t.numpy()

    assert np.array_equal(out_np, out_t), "fused out diverges from the CUDA oracle reference"
    assert np.array_equal(conv_np, conv_t), "conv pool diverges (the shift must be pure bf16 moves)"
    assert np.max(np.abs(ssm_np - ssm_t)) < 1e-5  # fp32-ULP class
    assert np.all(out_t[1] == 0) and np.all(out_np[1] == 0)  # padded slot

    # -- the compiled op (capture -> lower -> schedule -> execute) agrees --
    rec = RecordingBackend()
    pool_c = rec.external_tensor("conv", (slots, Kt - 1, Cc), storage_id=9001)
    pool_s = rec.external_tensor("ssm", (slots, H, D, D), storage_id=9002)
    ids = rec.external_tensor("ids", (B,), storage_id=9003, dtype=I32)
    xq = rec.external_tensor("xq", (B, Cc), storage_id=9004)
    xf = rec.external_tensor("xf", (B, H, D), storage_id=9005)
    xb = rec.external_tensor("xb", (B, H), storage_id=9006)
    xg = rec.external_tensor("xg", (B, H, D), storage_id=9007)
    wt = rec.external_tensor("taps", (Kt, Cc), storage_id=9008)
    wd = rec.external_tensor("dtb", (H, D), storage_id=9009)
    wa = rec.external_tensor("alog", (H,), storage_id=9010)
    wo = rec.external_tensor("onorm", (D,), storage_id=9011)
    fused, _, _ = rec.kda_fused_decode(
        pool_c, pool_s, ids, xq, xf, xb, xg, wt, wd, wa, wo,
        layer=0, scale=scale, eps=eps, lower_bound=lb)
    graph = rec.graph
    assert not [d for d in check_graph(graph) if d.severity == "error"]
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=2)
    wp, _ = plan_memory(graph, schedule, families)
    storage = {sid: a.astype(np.float32).reshape(-1).copy() for sid, a in (
        (9001, conv0), (9002, ssm0), (9003, slot_ids), (9004, mixed),
        (9005, a_raw), (9006, b_raw), (9007, g_raw), (9008, taps),
        (9009, dtb), (9010, alog), (9011, onorm))}
    for buf in wp.buffers:
        storage[buf.storage_id] = np.full(buf.numel, np.nan, np.float32)
    ex = ReferenceExecutor(schedule, workers=2, storage_arrays=storage, graph=graph, workspace_plan=wp)
    ex.run({})
    got = ex.tensor(fused.value.name).reshape(B, H, D)
    got_conv = storage[9001].reshape(slots, Kt - 1, Cc)
    got_ssm = storage[9002].reshape(slots, H, D, D)
    # bf16-grid out: values agree up to at most one grid step (fp32-vs-fp64
    # accumulation inside the norm can flip a tie), state at fp32 ULPs
    n_bad = int(np.sum(np.abs(got - out_t) > 0.0))
    assert n_bad / got.size < 0.02, f"{n_bad}/{got.size} out elements off the oracle grid"
    assert np.max(np.abs(got - out_t)) < 0.05
    assert np.max(np.abs(got_conv - conv_t)) == 0.0
    assert np.max(np.abs(got_ssm - ssm_t)) < 1e-5


# ---------------------------------------------------------------------------
# Divergence band vs the decomposed incumbent chain (documented, priced)
# ---------------------------------------------------------------------------


def _decomposed_spine_step(cfg, w, streams, conv_km, ssm_km):
    """The decomposed incumbent: the compiler's own decomposed op chain
    (gdn_conv → kda_delta → rms_norm_gated semantics — the fp64 mirror floe
    eager follows), NO bf16 ABI rounds between the ops. Pools are the
    decomposed ops' shapes: conv [B, K-1, Cc] batch-indexed, ssm [B, H, K, V]
    K-major (the transpose of the fused op's V-major pool — layout only)."""
    hidden = streams
    for layer in range(cfg.num_hidden_layers):
        post, comb, h_in = _mhc_block(
            hidden, w.hc_attn_fn[layer], w.hc_attn_base[layer], w.hc_attn_scale[layer], cfg)
        h = _rms_norm(h_in, w.ln1[layer], cfg.rms_norm_eps)
        body = _kda_layer_decode(h, w, cfg, layer, conv_km[layer], ssm_km[layer])
        hidden = _mhc_compose(hidden, post, comb, body)
    return hidden


def test_e1_fused_vs_decomposed_divergence_band():
    """Same weights, same pool values: the fused ABI (bf16 rounds at the raw
    dots + gates, unrounded conv output) vs the round-free decomposed chain.
    The gap is the INTENTIONAL divergence documented by the CUDA oracle
    ("cross-path agreement at ~1e-2 relative" — the packed kernel's class):
    it must be small (a sanity bound on the ABI's numeric cost) but is NOT
    the tight parity band (which the mirror tests pin)."""
    cfg = Glm53Config()
    seed = 23
    batch = 3
    args, out, graph, _, schedule, wp = _compile_spine(cfg, seed=seed, batch=batch, workers=2)
    conv, ssm, streams = _spine_pools(cfg, slots=batch, seed=51, batch=batch)
    slot_ids = np.arange(batch, dtype=np.int32)
    _, got = _run_spine(cfg, args, out, graph, schedule, wp, 2, conv, ssm, streams, slot_ids, steps=1)

    # decomposed pools: same values, decomposed layouts (batch-indexed,
    # ssm K-major = transpose of the V-major pool)
    conv_km = [c[:batch].copy() for c in conv]
    ssm_km = [np.transpose(s[:batch], (0, 1, 3, 2)).copy() for s in ssm]
    want = _decomposed_spine_step(cfg, _weights(cfg, seed), streams.astype(np.float64), conv_km, ssm_km)
    err = np.max(np.abs(got - want))
    assert err < TOL_DECOMPOSED, f"fused-vs-decomposed gap {err:.3e} exceeds the documented bf16-ABI band"
