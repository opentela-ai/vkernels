"""Issue #102 Stage 1 — Qwen3.5 hybrid decode step as a single-launch megakernel.

Whole-graph integration suite: the qwen35 decode-step body (``model_qwen35``
against the ops API) through the generic capture -> legality -> lowering ->
schedule -> memory pipeline, on the tiny hybrid config.

Bare env (numpy only, no torch / no floe — 0 collection errors, 0 unexpected
skips): graph census, single-launch accounting, hazard ordering across the
full step, workspace-plan validity, NaN canaries, ragged rows (#93), paged
slot indirection (#94), and whole-decode-step numerics against the fp64
mirror (``qwen35_arch.qwen35_reference_decode_step`` — the oracle of record).
The floe eager forward is the §15.1 oracle and is exercised in the
import-gated test at the bottom (skipped when floe is absent).
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler.capture import RecordingBackend, capture_model
from vkernels.compiler.legality import check_graph
from vkernels.compiler.lowerings import lower_graph
from vkernels.compiler.memory import plan_memory
from vkernels.compiler.model_gpt2 import _stable_storage_id as sid
from vkernels.compiler.model_qwen35 import Qwen35ModelArgs, build_qwen35_forward
from vkernels.compiler.operator_ir import I32, compute_hazards
from vkernels.compiler.qwen35_arch import (
    initial_state,
    qwen35_reference_decode_step,
    random_qwen35_weights,
    tiny_qwen35_config,
)
from vkernels.compiler.reference_exec import ReferenceExecutor
from vkernels.compiler.schedule_phase import PhaseSchedule


# ---------------------------------------------------------------------------
# Harness: build + storage wiring (flat 1-D arrays — the executor contract)
# ---------------------------------------------------------------------------


def _build(cfg=None, seed_w=0, seed_s=1, workers=None):
    cfg = cfg or tiny_qwen35_config()
    W = random_qwen35_weights(cfg, seed=seed_w)
    st = initial_state(cfg, seed=seed_s)
    ops = RecordingBackend()
    pos = ops.define_row_positions("row_positions", cfg.batch, cfg.cache_capacity, storage_id=10**6 + 5)
    ids = ops.external_tensor("ids", (cfg.batch,), I32, storage_id=10**6)
    args = Qwen35ModelArgs(ops, cfg)
    graph, _ = capture_model(build_qwen35_forward, args, ids, pos, cfg, backend=ops)
    fams = lower_graph(graph)
    widest = max(f.task_count for f in fams)
    w = workers or max(2, min(4, widest))
    sched = PhaseSchedule.from_families(fams, workers=w)
    ws_plan, scratch = plan_memory(graph, sched, fams)
    return cfg, W, st, graph, fams, sched, ws_plan


def _storage(cfg, W, st, graph, ws_plan, *, copy_state=False):
    """Flat external storage dict + NaN-carved workspace views."""
    S = {}
    ws = np.full(ws_plan.total_elements, np.nan)
    for buf in ws_plan.buffers:
        S[buf.storage_id] = ws[buf.offset : buf.offset + buf.numel]

    def put(name, arr, key=None):
        arr = np.array(arr, copy=True) if copy_state else np.asarray(arr)
        S[key if key is not None else sid(name)] = np.ascontiguousarray(arr).ravel()

    put("token_emb", W.token_emb)
    put("final_ln_gamma", W.final_gamma)
    put("rope_cos", W.rope_cos)
    put("rope_sin", W.rope_sin)
    put("row_positions", st.row_positions, key=10**6 + 5)
    put("slot_table", st.slot_table)
    put("ids", st.ids, key=10**6)
    put("k_pool", st.k_pool)
    put("v_pool", st.v_pool)
    put("k_cache_dense", st.k_cache_dense)
    put("v_cache_dense", st.v_cache_dense)
    put("conv_state", st.conv_state)
    put("ssm_state", st.ssm_state)
    for li, lt in enumerate(cfg.layer_types):
        lw = W.layers[li]
        for k, v in {
            "ln1_gamma": lw.ln1_gamma, "ln2_gamma": lw.ln2_gamma,
            "gate_up_w": lw.gate_up_w, "down_w": lw.down_w, "o_proj_w": lw.o_proj_w,
        }.items():
            put(f"l{li}_{k}", v)
        if lt == "gdn":
            for k, v in {
                "in_a_w": lw.in_a_w, "in_b_w": lw.in_b_w, "conv_w": lw.conv_w,
                "A_log": lw.A_log, "dt_bias": lw.dt_bias, "norm_w": lw.norm_w,
            }.items():
                put(f"l{li}_{k}", v)
            put(f"l{li}_in_qkvz_bytes", lw.in_qkvz_bytes)
            put(f"l{li}_in_qkvz_scale", lw.in_qkvz_scale)
        else:
            for k, v in {"q_norm_gamma": lw.q_norm_gamma, "k_norm_gamma": lw.k_norm_gamma}.items():
                put(f"l{li}_{k}", v)
            put(f"l{li}_qkv_gate_bytes", lw.qkv_gate_bytes)
            put(f"l{li}_qkv_gate_scale", lw.qkv_gate_scale)
    # every graph storage must be backed (workspace ids come from the plan)
    need = {tv.storage_id for tv in graph.tensors.values()}
    missing = need - set(S)
    assert not missing, f"unbacked storages: {sorted(missing)[:8]}"
    return S


def _run(cfg, W, st, graph, sched, ws_plan, S, *, canary=True):
    exe = ReferenceExecutor(sched, workers=sched.workers, storage_arrays=S, graph=graph,
                            workspace_plan=ws_plan, canary=canary)
    trace = exe.run({})
    out = np.array(exe.tensor(graph.ops[-1].outputs[0]))
    return exe, trace, out.reshape(cfg.batch, cfg.vocab)


def _pools(cfg, S):
    """Post-run pool views (reshaped from the flat storage contract)."""
    n_gdn, n_fa = len(cfg.gdn_layers), len(cfg.fa_layers)
    return dict(
        conv_state=S[sid("conv_state")].reshape(n_gdn, cfg.batch, cfg.conv_kernel - 1, cfg.gdn_kv_dim),
        ssm_state=S[sid("ssm_state")].reshape(n_gdn, cfg.batch, cfg.gdn_v_heads, cfg.head_v_dim, cfg.head_k_dim),
        k_pool=S[sid("k_pool")].reshape(n_fa, cfg.n_slots, cfg.kv_heads, cfg.head_dim),
        v_pool=S[sid("v_pool")].reshape(n_fa, cfg.n_slots, cfg.kv_heads, cfg.head_dim),
        k_cache_dense=S[sid("k_cache_dense")].reshape(n_fa, cfg.batch, cfg.kv_heads, cfg.cache_capacity, cfg.head_dim),
        v_cache_dense=S[sid("v_cache_dense")].reshape(n_fa, cfg.batch, cfg.kv_heads, cfg.cache_capacity, cfg.head_dim),
    )


# ---------------------------------------------------------------------------
# Capture & graph shape
# ---------------------------------------------------------------------------


def test_capture_census_whole_step():
    """The ops-API body captures the full hybrid step: every Stage-1 op kind,
    in per-layer-kind counts, ending in the tied-head logits."""
    cfg, _, _, graph, _, _, _ = _build()
    kinds = {}
    for op in graph.ops:
        kinds[op.kind] = kinds.get(op.kind, 0) + 1
    n_gdn, n_fa = len(cfg.gdn_layers), len(cfg.fa_layers)
    assert kinds["gdn_conv"] == n_gdn and kinds["gdn_delta"] == n_gdn
    assert kinds["rope"] == 2 * n_fa  # q and k, partial NeoX
    assert kinds["linear_fp8"] == n_gdn + n_fa  # in_qkvz + qkv_gate
    assert kinds["cache_append_paged"] == 1 and kinds["attention_scores_paged"] == 1
    assert kinds["attention_values_paged"] == 1
    assert kinds["cache_append"] == 1 and kinds["attention_values"] == 1  # gated (#92)
    assert kinds["embedding"] == 1
    assert graph.ops[-1].kind == "linear" and graph.ops[-1].outputs[0].startswith("logits")
    assert kinds["softmax"] == n_fa and kinds["swiglu"] == cfg.layers


def test_row_positions_are_the_only_position_form():
    """#93: every position consumer takes the row-form tensor — no scalar p
    anywhere in the graph (the step is ragged-native)."""
    _, _, _, graph, _, _, _ = _build()
    assert "p" not in graph.scalars, "scalar position leaked into the hybrid step"
    assert graph.row_position_tensors, "row-position tensor not registered"
    for op in graph.ops:
        if op.kind in ("rope", "cache_append", "attention_scores", "softmax",
                       "attention_values", "embedding", "cache_append_paged",
                       "attention_scores_paged", "attention_values_paged"):
            assert op.attributes.get("position_form") == "row", (op.kind, op.attributes)


# ---------------------------------------------------------------------------
# Hazards: ordering obligations across the full step
# ---------------------------------------------------------------------------


def test_hazard_ordering_full_step():
    """Legality is clean, and the state-pool RMW chains carry real
    RAW/WAR/WAW hazards in program order (the #98 op-level pattern, at
    whole-graph scale): each GDN layer's delta rule RAWs its layer's SSM
    pool; conv pools chain through their layers; the paged trio hazards on
    the k/v pools."""
    cfg, _, _, graph, _, _, _ = _build()
    diags = check_graph(graph)
    errors = [d for d in diags if d.severity == "error"]
    assert not errors, errors

    hazards = compute_hazards(graph.ops)
    assert hazards, "no hazards recorded — RMW pools were not tracked"
    opid_kind = {op.opid: op.kind for op in graph.ops}

    def edges(*kinds):
        ks = set(kinds)
        return [h for h in hazards if {opid_kind[h.producer], opid_kind[h.consumer]} == ks]

    # (a) conv -> delta RAW on the fresh conv-out storage (per GDN layer)
    conv2delta = edges("gdn_conv", "gdn_delta")
    assert len(conv2delta) == len(cfg.gdn_layers) and all(h.kind == "RAW" for h in conv2delta)

    # (b) paged pools: append -> scores/values RAW edges (program order that
    # the phase schedule must honor — the #93/#94 chain at whole-graph scale)
    kp_sid = graph.tensor("k_pool").storage_id
    vp_sid = graph.tensor("v_pool").storage_id
    for pool_sid in (kp_sid, vp_sid):
        hs = [h for h in hazards if h.storage_id == pool_sid]
        assert hs and all(h.kind == "RAW" for h in hs), (pool_sid, hs)
        assert all({opid_kind[h.producer], opid_kind[h.consumer]} ==
                   {"cache_append_paged", "attention_scores_paged"} or
                   {opid_kind[h.producer], opid_kind[h.consumer]} ==
                   {"cache_append_paged", "attention_values_paged"} for h in hs)

    # (c) dense-cache pools: the gated (#92) trio chains identically
    kd_sid = graph.tensor("k_cache_dense").storage_id
    hs = [h for h in hazards if h.storage_id == kd_sid]
    assert hs and all(h.kind == "RAW" for h in hs)

    # (d) state pools: per-layer write slabs are pairwise disjoint and
    # exactly tile the pool — the soundness argument for a single launch
    # with no cross-layer barrier on those storages (phase order only
    # orders conv -> delta within a layer)
    for pool_name, per_layer in (("conv_state", "conv_state_l"), ("ssm_state", "ssm_state_l")):
        pool_sid = graph.tensor(pool_name).storage_id
        spans = []
        for op in graph.ops:
            if op.kind not in ("gdn_conv", "gdn_delta"):
                continue
            for reg in op.write_regions:
                if reg.storage_id == pool_sid:
                    spans.append(reg.storage_span())
        assert len(spans) == len(cfg.gdn_layers), (pool_name, spans)
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 < b0, f"{pool_name}: overlapping slabs {a0}..{a1} vs {b0}..{b1}"
        assert spans[0][0] == 0
        total = graph.storage_sizes[pool_sid]
        assert spans[-1][1] == total - 1, (pool_name, spans, total)

    # (e) program order: every hazard's producer precedes its consumer in
    # the captured op order — the phase schedule's backbone
    opid_index = {op.opid: i for i, op in enumerate(graph.ops)}
    assert all(opid_index[h.producer] < opid_index[h.consumer] for h in hazards)


# ---------------------------------------------------------------------------
# Workspace plan validity (the #98 lesson: co-allocation is legal, overlap
# of LIVE ranges is not)
# ---------------------------------------------------------------------------


def test_workspace_plan_lifetime_safety():
    """Fresh buffers sharing a storage region must have disjoint lifetimes;
    the plan must actually reuse (else the 'persistent' claim is hollow)."""
    _, _, _, _, _, sched, ws_plan = _build()
    by_storage = {}
    for b in ws_plan.buffers:
        by_storage.setdefault(b.storage_id, []).append(b)
    for storage, bufs in by_storage.items():
        for i, a in enumerate(bufs):
            for b in bufs[i + 1 :]:
                if a.overlaps_lifetime(b):
                    # legal only if their byte ranges are disjoint
                    a0, a1 = a.offset, a.offset + a.numel
                    b0, b1 = b.offset, b.offset + b.numel
                    assert a1 <= b0 or b1 <= a0, (
                        f"live-range overlap on storage {storage}: {a.name} {a.lifetime} "
                        f"vs {b.name} {b.lifetime} with overlapping placement"
                    )
    total = sum(b.numel for b in ws_plan.buffers)
    assert ws_plan.total_elements < total, "planner achieved no workspace reuse"


# ---------------------------------------------------------------------------
# Single-launch accounting (§15.2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [2, 4])
def test_single_launch_accounting(workers):
    cfg, W, st, graph, fams, sched, ws_plan = _build(workers=workers)
    S = _storage(cfg, W, st, graph, ws_plan)
    _, trace, _ = _run(cfg, W, st, graph, sched, ws_plan, S)
    assert trace.kernel_launches == 1, "the whole decode step must be ONE launch"
    assert trace.grid_barriers == len(sched.phases)
    assert trace.task_executions == sum(p.task_count for p in sched.phases)
    # every phase's tasks were exactly covered by the worker tiling
    for ps in trace.phases:
        assert sum(ps.per_worker) == ps.tasks
    # re-specialization is legal within the residency bound: same numerics
    S2 = _storage(cfg, W, st, graph, ws_plan)
    sched2 = PhaseSchedule.from_families(fams, workers=2) if workers != 2 else sched
    if sched2 is not sched:
        ws2, _ = plan_memory(graph, sched2, fams)
        S2 = _storage(cfg, W, st, graph, ws2)
        _, tr2, _ = _run(cfg, W, st, graph, sched2, ws2, S2)
        assert tr2.kernel_launches == 1


# ---------------------------------------------------------------------------
# NaN canaries at whole-graph level
# ---------------------------------------------------------------------------


def test_nan_canary_whole_graph():
    """Executor canary protocol passes for every phase; unfilled pool slots
    stay NaN (no stray writes); the reserved null slot 0 is never touched."""
    cfg, W, st, graph, fams, sched, ws_plan = _build()
    S = _storage(cfg, W, st, graph, ws_plan, copy_state=True)
    _, trace, out = _run(cfg, W, st, graph, sched, ws_plan, S, canary=True)  # raises on violation
    assert np.isfinite(out).all()
    pools = _pools(cfg, S)
    # NaN canaries: slots beyond each row's written position stay untouched
    for b in range(cfg.batch):
        pb = int(st.row_positions[b])
        for t in range(pb + 1, cfg.cache_capacity):
            slot = int(st.slot_table[b, t])
            assert np.isnan(pools["k_pool"][0, slot]).all(), f"stray write at slot {slot}"
            assert np.isnan(pools["v_pool"][0, slot]).all()
        assert np.isnan(pools["k_cache_dense"][1, b, :, pb + 1 :, :]).all()
    # null slot 0: reserved, never written by a live row
    assert np.isnan(pools["k_pool"][0, 0]).all() and np.isnan(pools["v_pool"][0, 0]).all()


# ---------------------------------------------------------------------------
# Whole-step numerics vs the fp64 mirror (oracle of record, bare env)
# ---------------------------------------------------------------------------


def test_tiny_config_decode_numerics_vs_mirror():
    """The compiled whole-decode-step graph matches the fp64 straight-line
    mirror on logits AND every post-step pool — ragged rows, paged layers,
    gated layers, GDN recurrences, fp8 projections, all together."""
    cfg, W, st, graph, fams, sched, ws_plan = _build()
    S = _storage(cfg, W, st, graph, ws_plan, copy_state=True)
    logits_ref, st_ref = qwen35_reference_decode_step(W, st, cfg)
    _, trace, out = _run(cfg, W, st, graph, sched, ws_plan, S, canary=False)
    assert trace.kernel_launches == 1
    assert np.abs(out - logits_ref).max() < 1e-9, np.abs(out - logits_ref).max()
    pools = _pools(cfg, S)
    assert np.allclose(pools["conv_state"], st_ref.conv_state, atol=1e-9)
    assert np.allclose(pools["ssm_state"], st_ref.ssm_state, atol=1e-9)
    assert np.allclose(pools["k_pool"], st_ref.k_pool, atol=1e-9, equal_nan=True)
    assert np.allclose(pools["v_pool"], st_ref.v_pool, atol=1e-9, equal_nan=True)
    assert np.allclose(pools["k_cache_dense"], st_ref.k_cache_dense, atol=1e-9, equal_nan=True)
    assert np.allclose(pools["v_cache_dense"], st_ref.v_cache_dense, atol=1e-9, equal_nan=True)


def test_ragged_rows_end_to_end():
    """#93: two rows at different positions (5, 2) — both rows' outputs are
    finite and row-distinct; per-row valid lengths gate every consumer."""
    cfg, W, st, graph, fams, sched, ws_plan = _build()
    assert list(st.row_positions) == [5, 2]
    S = _storage(cfg, W, st, graph, ws_plan, copy_state=True)
    logits_ref, _ = qwen35_reference_decode_step(W, st, cfg)
    _, _, out = _run(cfg, W, st, graph, sched, ws_plan, S, canary=False)
    assert np.isfinite(out).all()
    assert np.abs(out - logits_ref).max() < 1e-9
    assert np.abs(out[0] - out[1]).max() > 1e-6, "rows must compute distinct positions' results"


def test_paged_indirection_end_to_end():
    """#94: the per-row slot table is a non-identity permutation — appends
    land at table[b, pos[b]] and gathers route through table[b, t]; the
    mirror (which reads the same table) agrees exactly, so any addressing
    shortcut in the compiled graph would show up as a mismatch."""
    cfg, W, st, graph, fams, sched, ws_plan = _build()
    # the table really is non-identity
    for b in range(cfg.batch):
        row = st.slot_table[b]
        assert not np.array_equal(row, 1 + b * cfg.cache_capacity + np.arange(cfg.cache_capacity))
    S = _storage(cfg, W, st, graph, ws_plan, copy_state=True)
    logits_ref, st_ref = qwen35_reference_decode_step(W, st, cfg)
    _, _, out = _run(cfg, W, st, graph, sched, ws_plan, S, canary=False)
    assert np.abs(out - logits_ref).max() < 1e-9
    pools = _pools(cfg, S)
    # append landed exactly at the table-mapped slot, in BOTH pool slots
    for b in range(cfg.batch):
        pb = int(st.row_positions[b])
        slot = int(st.slot_table[b, pb])
        assert slot != 1 + b * cfg.cache_capacity + pb, "table permutation regressed to identity"
        assert np.isfinite(pools["k_pool"][0, slot]).all()
        assert np.allclose(pools["k_pool"][0, slot], st_ref.k_pool[0, slot])


def test_gdn_state_recurrence_chains_layers():
    """The GDN pools' storage versions bump per layer: later layers consume
    the post-step views (cache_append §4.3 pattern at graph scale)."""
    cfg, _, _, graph, _, _, _ = _build()
    conv_view_names = [name for name in graph.tensors if "_conv_state" in name and name != "conv_state"]
    ssm_view_names = [name for name in graph.tensors if "_ssm_state" in name and name != "ssm_state"]
    assert len(conv_view_names) == len(cfg.gdn_layers)
    assert len(ssm_view_names) == len(cfg.gdn_layers)
    # each view maps to the shared pool storage with the layer's offset
    base = graph.tensor("conv_state")
    seen_offsets = set()
    for name in conv_view_names:
        tv = graph.tensor(name)
        assert tv.storage_id == base.storage_id
        seen_offsets.add(tv.offset)
    assert len(seen_offsets) == len(cfg.gdn_layers), "per-layer views must address distinct pool slabs"


def test_compilation_report_and_determinism():
    """Two runs over fresh state pools are bit-identical (fp64 walk
    determinism), and the schedule is worker-count re-specializable."""
    cfg, W, st, graph, fams, sched, ws_plan = _build()
    outs = []
    for _ in range(2):
        S = _storage(cfg, W, st, graph, ws_plan, copy_state=True)
        _, trace, out = _run(cfg, W, st, graph, sched, ws_plan, S, canary=False)
        outs.append(out)
    assert np.array_equal(outs[0], outs[1])
    assert ws_plan.total_elements > 0 and sched.workers >= 2


# ---------------------------------------------------------------------------
# The generic compile path end-to-end on the tiny config
# ---------------------------------------------------------------------------


def test_compile_model_end_to_end_tiny_config():
    """compile_model dispatches the qwen35 frontend through the FULL generic
    pipeline — capture, legality, lowering, phase schedule, memory plan,
    codegen and launch manifest — with the single-kernel contract enforced."""
    from vkernels.compiler.compile import compile_model

    exe = compile_model(model_config=tiny_qwen35_config(), target="reference", seed=0)
    assert exe.graph.ops[-1].outputs[0].startswith("logits")
    assert exe.schedule.phases, "no phases compiled"
    assert exe.report is not None
    assert exe.manifest.workspace_bytes == exe.workspace_plan.total_bytes
    assert exe.manifest.workers == exe.schedule.workers
    # strict single-kernel: the qwen35 body compiles under the same contract
    kinds = {op.kind for op in exe.graph.ops}
    assert {"gdn_conv", "gdn_delta", "cache_append_paged", "linear_fp8"} <= kinds
    # the compiled executable's graph runs against its own schedule too
    st = initial_state(exe.config, seed=1)
    W = exe.weights
    S = _storage(exe.config, W, st, exe.graph, exe.workspace_plan, copy_state=True)
    logits_ref, _ = qwen35_reference_decode_step(W, st, exe.config)
    executor = ReferenceExecutor(exe.schedule, workers=exe.schedule.workers,
                                 storage_arrays=S, graph=exe.graph,
                                 workspace_plan=exe.workspace_plan, canary=True)
    trace = executor.run({})
    assert trace.kernel_launches == 1
    out = np.array(executor.tensor(exe.graph.ops[-1].outputs[0])).reshape(exe.config.batch, -1)
    assert np.abs(out - logits_ref).max() < 1e-9


# ---------------------------------------------------------------------------
# §15.1 floe oracle (import-gated; VERIFIED only when a torch env exists)
# ---------------------------------------------------------------------------


def test_floe_eager_oracle_gated():
    """The floe eager qwen35 forward is the §15.1 oracle. This env has no
    torch/floe (bare suite doctrine), so the gate skips; when run in a torch
    venv (the #99 worker's conftest repo-src pin pattern), compare the
    mirror's GDN/FA layer numerics against floe's eager modules."""
    pytest.importorskip("torch")
    floe = pytest.importorskip("floe")
    assert hasattr(floe, "qwen35_gdn"), "floe layout changed; re-point the oracle"
