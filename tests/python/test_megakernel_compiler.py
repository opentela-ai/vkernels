"""Validation suite for the CuTe DSL megakernel model compiler.

Mirrors the design doc's §15.1 validation levels — IR tests, task tests,
whole-model tests, schedule tests — all runnable on CPU (§15.1: "randomized
CPU execution can check graph logic, but cannot validate compiled GPU
synchronization or memory ordering"; the device-side obligations are
exercised through the capability report and the refusal paths).
"""

from __future__ import annotations

import numpy as np
import pytest

from vkernels.compiler import (
    GPT2Config,
    KVCache,
    PhaseSchedule,
    ScheduleInvariantError,
    capture_model,
    compile_model,
    compute_hazards,
    random_weights,
    reference_forward,
)
from vkernels.compiler.capture import CaptureError, RecordingBackend, UnsupportedOperator
from vkernels.compiler.codegen_cute import TEMPLATE_NAMES
from vkernels.compiler.model_gpt2 import SymbolicModelArgs, build_forward
from vkernels.compiler.runtime.synchronization import (
    CapabilityError,
    SimulatedGridBarrier,
)
from vkernels.compiler.runtime.task_queue import TaskDeps, simulate_fifo_execution
from vkernels.compiler.schedule_static import (
    build_static_worker_schedule,
    build_tile_dependencies,
)

# ===========================================================================
# IR + capture tests (Milestone 1)
# ===========================================================================


def _capture_default():
    cfg = GPT2Config()
    weights = random_weights(cfg)
    recorder = RecordingBackend()
    position = recorder.define_position(cfg.cache_capacity)
    ids = recorder.external_tensor("ids", (cfg.batch,), storage_id=10**6)
    args = SymbolicModelArgs(recorder, cfg, weights)
    graph, recorder = capture_model(build_forward, args, ids, position, cfg, backend=recorder)
    return graph, recorder, cfg, weights


EXPECTED_LAYER_KINDS = [
    "layer_norm",  # ln1
    "linear",  # qkv
    "cache_append",
    "attention_scores",
    "softmax",
    "attention_values",
    "linear",  # attn_out
    "add",
    "layer_norm",  # ln2
    "linear",  # mlp up
    "gelu",
    "linear",  # mlp down
    "add",
]


def test_capture_records_29_unfused_phases():
    """Milestone 1 acceptance: 29 arithmetic/cache phases for the default model."""
    graph, _, cfg, _ = _capture_default()
    kinds = [op.kind for op in graph.ops]
    expected = ["embedding"]
    for _ in range(cfg.layers):
        expected += EXPECTED_LAYER_KINDS
    expected += ["layer_norm", "linear"]  # final ln + tied head
    assert kinds == expected
    assert len(kinds) == 29


def test_capture_is_deterministic_and_versions_cache():
    """Capture fidelity: identical re-capture; appends bump storage versions (§4.3)."""
    graph1, rec1, cfg, weights = _capture_default()
    graph2, rec2, _, _ = _capture_default()
    assert [op.kind for op in graph1.ops] == [op.kind for op in graph2.ops]
    assert len(graph1.tensors) == len(graph2.tensors)
    # Each layer's append bumps its packed cache storage version exactly once.
    k_sid = next(tv.storage_id for name, tv in graph1.tensors.items() if name == "k_cache")
    assert graph1.storage_versions[k_sid] == cfg.layers
    # Capture never advanced any runtime cache length: no KVCache was involved
    # at all — the recorder only ever sees symbolic views.


def test_tied_head_is_transposed_alias():
    """The tied head reads a transposed view of the token embedding (§5.2)."""
    graph, _, _, _ = _capture_default()
    tied = graph.tensor("token_emb.T")
    token = graph.tensor("token_emb")
    assert tied.storage_id == token.storage_id
    assert tied.shape == (token.shape[1], token.shape[0])
    assert tied.strides == (token.strides[1], token.strides[0])
    # The final linear's read region overlaps the embedding's read storage.
    final_op = graph.ops[-1]
    assert any(r.storage_id == token.storage_id for r in final_op.read_regions)


def test_hazards_recover_cache_effect_ordering():
    """RAW/WAR/WAW edges express the §4.3 ordering (whole-operator regions)."""
    graph, _, _, _ = _capture_default()
    hazards = compute_hazards(graph.ops)
    kinds = [op.kind for op in graph.ops]
    # qkv -> cache_append (RAW through the k/v views of the qkv buffer)
    raw_to_append = [h for h in hazards if kinds[h.consumer] == "cache_append" and h.kind == "RAW"]
    assert raw_to_append, "append must depend on the qkv projection (RAW)"
    # append -> attention scores / values (RAW on the cache storage)
    raw_from_append = [h for h in hazards if kinds[h.producer] == "cache_append" and h.kind == "RAW"]
    consumers = {kinds[h.consumer] for h in raw_from_append}
    assert "attention_scores" in consumers
    assert "attention_values" in consumers


def test_unsupported_operator_is_rejected_with_diagnostic():
    """§3.3: unsupported inputs must be explicit, not silently external."""

    def body_with_custom_op(ops, args, ids, position, cfg):
        x = ops.embedding(ids, args.token, args.position, position)
        y = ops.custom_mystery_op(x)  # not in the supported subset
        return y

    cfg = GPT2Config()
    weights = random_weights(cfg)
    recorder = RecordingBackend()
    position = recorder.define_position(cfg.cache_capacity)
    ids = recorder.external_tensor("ids", (cfg.batch,), storage_id=10**6)
    args = SymbolicModelArgs(recorder, cfg, weights)
    with pytest.raises(UnsupportedOperator) as ei:
        capture_model(body_with_custom_op, args, ids, position, cfg, backend=recorder)
    assert "supported subset" in str(ei.value)


def test_frozen_position_is_rejected():
    """§4.2: a concrete position would freeze the cache length into the graph."""
    graph, recorder, cfg, weights = _capture_default()
    ids = recorder.external_tensor("ids2", (cfg.batch,), storage_id=10**6 + 1)
    args = SymbolicModelArgs(recorder, cfg, weights)
    with pytest.raises(CaptureError):
        capture_model(build_forward, args, ids, 7, cfg, backend=recorder)


def test_symbolic_item_is_rejected():
    from vkernels.compiler.capture import SymbolicTensor

    graph, recorder, cfg, weights = _capture_default()
    sym = SymbolicTensor(graph.tensor("token_emb"))
    with pytest.raises(CaptureError):
        sym.item()


# ===========================================================================
# Task lowering tests (Milestone 2)
# ===========================================================================


@pytest.fixture(scope="module")
def compiled_default():
    cfg = GPT2Config()
    return compile_model(model_config=cfg, weights=random_weights(cfg), workers=4)


def test_worked_example_tile_counts(compiled_default):
    """§6.4: QKV 24 tiles, MLP up 32, MLP down 8 — counts follow the output
    dimensions, not a desired SM count."""
    fams = {f.op.source_location: f for f in compiled_default.families}
    assert fams["l0_qkv"].task_count == 24  # ceil(384/16) * ceil(1/16)
    assert fams["l0_mlp_up"].task_count == 32  # ceil(512/16)
    assert fams["l0_mlp_down"].task_count == 8  # ceil(128/16)
    assert fams["l0_attn_out"].task_count == 8
    # Attention/LN/append tasks are per (row|head): B*H = 4, B = 1.
    assert fams["layer 0 attention scores"].task_count == 4
    assert fams["layer 0 softmax"].task_count == 4
    assert fams["layer 0 kv cache append"].task_count == 4
    assert fams["l0_ln1"].task_count == 1


def test_thread_contract_is_uniform_256(compiled_default):
    """§7.1: one persistent worker cannot change its block size between tasks."""
    assert {f.threads for f in compiled_default.families} == {256}


def test_task_region_precision_for_one_attention_head(compiled_default):
    """§10.3 prerequisite: per-task regions are tile-precise, not whole-op."""
    fams = {f.family_id: f for f in compiled_default.families}
    append = fams["ph03_cache_append"]
    # Head 0's append task reads only head 0's K/V rows of the qkv buffer.
    regions = append.reads(0)
    qkv = compiled_default.graph.tensor(compiled_default.graph.ops[2].outputs[0])
    from vkernels.compiler import Region

    q_head0 = Region.tile(qkv, ((0, 1), (0, 32)))  # Q columns of head 0
    k_head0 = Region.tile(qkv, ((0, 1), (128, 160)))  # K columns of head 0
    touched = [r for r in regions if r.storage_id == qkv.storage_id]
    assert all(r.overlaps(k_head0) or r.overlaps(Region.tile(qkv, ((0, 1), (256, 288)))) for r in touched)
    assert not any(r.overlaps(q_head0) for r in touched)


def test_gemm_task_regions_cover_outputs_exactly(compiled_default):
    """§12 coverage/ownership: output tiles tile the output without overlap."""
    fams = {f.family_id: f for f in compiled_default.families}
    qkv = fams["ph02_linear"]
    boxes = [qkv.writes(t)[0].resolved_boxes({}) for t in range(qkv.task_count)]
    # Collect column ranges; rows are trivially [0,1).
    cols = sorted((b[1][0], b[1][1]) for b in boxes)
    assert cols[0][0] == 0
    for (_, hi), (lo, _) in zip(cols, cols[1:]):
        assert lo == hi  # dense, disjoint cover
    assert cols[-1][1] == 384


# ===========================================================================
# Schedule tests (Milestone 3)
# ===========================================================================


def test_coverage_and_barriers_for_all_worker_counts(compiled_default):
    """§8.2/§8.3: every tile executes exactly once for any legal P; idle
    workers still meet every barrier."""
    families = compiled_default.families
    for p in (1, 2, 3, 4, 5, 7, 13, 32):
        sched = PhaseSchedule.from_families(families, workers=p)
        assert sched.total_tasks == sum(f.task_count for f in families)
        # from_families already ran check_invariants; also exercise stats.
        widest = max(ps.task_count for ps in sched.phases)
        assert widest <= max(f.task_count for f in families)


def test_invalid_worker_count_rejected(compiled_default):
    with pytest.raises(ScheduleInvariantError):
        PhaseSchedule.from_families(compiled_default.families, workers=0)


def test_simulated_barrier_detects_protocol_violations():
    barrier = SimulatedGridBarrier(workers=3)
    barrier.arrive(0)
    barrier.arrive(1)
    with pytest.raises(Exception):
        barrier.release()  # worker 2 missing (§8.3)
    barrier.arrive(2)
    barrier.release()
    barrier.arrive(0)
    with pytest.raises(Exception):
        barrier.arrive(0)  # double arrival


# ===========================================================================
# Memory planning tests (Milestone 4)
# ===========================================================================


def test_workspace_reuse_is_lifetime_sound(compiled_default):
    """§9.2: reused storage only between disjoint phase lifetimes."""
    plan = compiled_default.workspace_plan
    assert plan.total_bytes < plan.naive_bytes, "lifetime reuse must kick in"
    for i, a in enumerate(plan.buffers):
        for b in plan.buffers[i + 1 :]:
            ranges_overlap = a.offset < b.offset + b.numel and b.offset < a.offset + a.numel
            if ranges_overlap:
                assert not a.overlaps_lifetime(b), f"buffers {a.name} and {b.name} share storage with overlapping lifetimes"


def test_scratch_follows_sequential_max_rule(compiled_default):
    """§7.2: S_block = S_runtime + max_j S_task,j."""
    scratch = compiled_default.scratch_plan
    max_task = max(f.scratch_bytes for f in compiled_default.families)
    assert scratch.max_task_scratch == max_task
    assert scratch.per_worker_bytes >= scratch.runtime_bytes + max_task


def test_kv_cache_formula():
    """§9.1: M_KV = 2 L B S C * 4 bytes."""
    from vkernels.compiler.memory import kv_cache_bytes

    cfg = GPT2Config()
    assert kv_cache_bytes(cfg.layers, cfg.batch, cfg.cache_capacity, cfg.hidden) == 2 * 2 * 1 * 64 * 128 * 4
    # The reference KVCache uses f64 elements; same formula, doubled width.
    assert KVCache(cfg).storage_bytes() == kv_cache_bytes(cfg.layers, cfg.batch, cfg.cache_capacity, cfg.hidden, element_size=8)


# ===========================================================================
# Whole-model tests (§15.1): compiled schedule vs. independent oracle
# ===========================================================================


def _decode_compare(cfg, *, workers, steps):
    weights = random_weights(cfg)
    exe = compile_model(model_config=cfg, weights=weights, workers=workers)
    rng = np.random.default_rng(7)
    ids_all = rng.integers(0, cfg.vocab, size=(steps, cfg.batch))
    cache_ref, cache_run = KVCache(cfg), KVCache(cfg)
    worst = 0.0
    for p in range(steps):
        ids = ids_all[p]
        logits_ref, _ = reference_forward(weights, ids, cache_ref, p, cfg)
        logits, trace = exe.run(ids, cache_run, p)
        worst = max(worst, float(np.abs(logits - logits_ref).max()))
        # §15.2 launch accounting: exactly one kernel event per invocation.
        assert trace.kernel_launches == 1
        assert trace.grid_barriers == exe.schedule.num_phases
        # §5.3: uninitialized tails must never be read.
        assert np.isnan(cache_run.k[:, :, :, p + 1 :, :]).all()
        assert np.isnan(cache_run.v[:, :, :, p + 1 :, :]).all()
    assert worst < 1e-9, f"logits diverged from oracle: {worst}"
    # Full cache comparison including the appended rows.
    assert np.allclose(cache_run.k, cache_ref.k, equal_nan=True)
    assert np.allclose(cache_run.v, cache_ref.v, equal_nan=True)
    return worst


def test_sequential_decode_matches_oracle_full_capacity():
    """Every position 0..S-1 (near-capacity append included), B=1."""
    _decode_compare(GPT2Config(), workers=4, steps=GPT2Config().cache_capacity)


def test_batch2_matches_oracle():
    cfg = GPT2Config(batch=2, cache_capacity=8)
    _decode_compare(cfg, workers=3, steps=cfg.cache_capacity)


def test_worker_count_does_not_change_results():
    """Any legal P computes the same schedule semantics (§7.4: P is tuning)."""
    cfg = GPT2Config(cache_capacity=6)
    weights = random_weights(cfg)
    results = {}
    for p in (1, 2, 3, 5, 9):
        exe = compile_model(model_config=cfg, weights=weights, workers=p)
        cache = KVCache(cfg)
        logits = None
        for pos in range(cfg.cache_capacity):
            logits, _ = exe.run(np.array([3]), cache, pos)
        results[p] = logits
    for p, lg in results.items():
        assert np.allclose(lg, results[1], atol=0), f"P={p} changed results"


def test_position_guard_rejects_out_of_range():
    cfg = GPT2Config()
    exe = compile_model(model_config=cfg, weights=random_weights(cfg), workers=2)
    cache = KVCache(cfg)
    with pytest.raises(ValueError):
        exe.run(np.array([0]), cache, -1)
    with pytest.raises(ValueError):
        exe.run(np.array([0]), cache, cfg.cache_capacity)


def test_cache_valid_len_advances_only_after_success():
    cfg = GPT2Config()
    exe = compile_model(model_config=cfg, weights=random_weights(cfg), workers=2)
    cache = KVCache(cfg)
    exe.run(np.array([1]), cache, 0)
    assert cache.valid_len == 1
    with pytest.raises(ValueError):
        exe.run(np.array([1]), cache, 99)
    assert cache.valid_len == 1  # §13.4: host bookkeeping after success only


# ===========================================================================
# Static fine-grained scheduling (§10) and the ready-task model (§11)
# ===========================================================================


def _deps_of(exe):
    deps = build_tile_dependencies(exe.schedule)
    deps.check_acyclic()
    return deps


def test_mlp_dependencies_match_design_example(compiled_default):
    """§10.2: GELU needs only its producing tiles; the down projection needs
    the full K range (all U tiles) — barrier removal cannot start it early."""
    deps = _deps_of(compiled_default)
    gelu_preds = deps.preds[("ph11_gelu", 0)]
    assert gelu_preds, "gelu must depend on the up projection"
    up_tiles = {p for p in gelu_preds if p[0] == "ph10_linear"}
    assert len(up_tiles) == 32  # the [1,512] gelu task spans all F tiles

    # Down tile (0,0): direct dep is the gelu output it reads...
    direct = deps.preds[("ph12_linear", 0)]
    assert ("ph11_gelu", 0) in direct
    # ...and transitively every up tile (full-K reduction, §10.2).
    seen, stack = set(), [("ph12_linear", 0)]
    while stack:
        k = stack.pop()
        for pr in deps.preds[k]:
            if pr not in seen:
                seen.add(pr)
                stack.append(pr)
    assert len([k for k in seen if k[0] == "ph10_linear"]) == 32


def test_attention_head_readiness_is_tile_precise(compiled_default):
    """§10.3: head h's tasks depend only on head h's QKV tiles — readiness
    can precede completion of the whole QKV projection."""
    deps = _deps_of(compiled_default)
    append_preds = deps.preds[("ph03_cache_append", 0)]
    qkv_tiles = sorted(p[1] for p in append_preds if p[0] == "ph02_linear")
    # Head 0: K columns [128,160) -> tiles 8,9; V columns [256,288) -> 16,17.
    assert qkv_tiles == [8, 9, 16, 17]

    scores_preds = deps.preds[("ph04_attn_scores", 0)]
    q_tiles = sorted(p[1] for p in scores_preds if p[0] == "ph02_linear")
    assert q_tiles == [0, 1]  # Q columns [0,32)
    assert ("ph03_cache_append", 0) in scores_preds

    values_preds = deps.preds[("ph06_attn_values", 0)]
    assert ("ph05_softmax", 0) in values_preds
    assert ("ph03_cache_append", 0) in values_preds


def test_static_worker_schedule_is_acyclic_for_several_p(compiled_default):
    """§10.4: fixed worker sequences must not introduce wait cycles."""
    deps = _deps_of(compiled_default)
    for p in (1, 2, 4, 8):
        sched = build_static_worker_schedule(deps, p)
        assert len(sched.order) == len(deps.tasks)
        sched.check_augmented_acyclic()  # must not raise


def test_ready_task_simulation_completes_and_detects_deadlock(compiled_default):
    """§11.4: FIFO dispatch retires every task; an unsatisfiable graph raises."""
    deps = _deps_of(compiled_default)
    dep_map = {k: set(ps) for k, ps in deps.preds.items()}
    result = simulate_fifo_execution(TaskDeps(preds=dep_map), num_workers=4)
    assert result == {"retired": len(dep_map), "total": len(dep_map)}

    cyclic = TaskDeps(preds={("a", 0): {("b", 0)}, ("b", 0): {("a", 0)}})
    from vkernels.compiler.runtime.task_queue import ProgressError as PE

    with pytest.raises(PE):
        simulate_fifo_execution(cyclic, num_workers=2)


# ===========================================================================
# Codegen + report + strict-mode capability tests (§8.4, §13, §15.2)
# ===========================================================================


def test_generated_source_is_valid_python_with_structure(compiled_default):
    src = compiled_default.source
    compile(src, "<generated>", "exec")  # syntax gate
    assert "NUM_PHASES = 29" in src
    assert src.count("grid_sync()") >= 1  # the single synchronization seam
    assert "@cute.kernel" in src
    assert "while task_id < task_count" in src  # §8.1 stride loop
    assert "WORKSPACE_BUFFERS" in src
    # One phase-table row per phase (each row carries a phase comment).
    assert src.count("    # phase ") == 29
    # Every template referenced exists in the template registry.
    for name in set(TEMPLATE_NAMES.values()):
        assert name in src


def test_strict_mode_refuses_device_execution(compiled_default):
    """§3.3/§8.4: no silent fallback; the device path demands the verified
    Milestone-0 grid-sync primitive."""
    from vkernels.compiler.runtime.synchronization import GRID_SYNC_BACKEND_VERIFIED

    assert GRID_SYNC_BACKEND_VERIFIED is False
    with pytest.raises(CapabilityError) as ei:
        compiled_default.run(np.array([0]), KVCache(GPT2Config()), 0, mode="device")
    assert "Milestone 0" in str(ei.value) or "Milestone-0" in str(ei.value)


def test_compilation_report_contents(compiled_default):
    r = compiled_default.report
    assert r.num_phases == 29
    assert r.total_tasks == 252
    assert r.workspace_bytes < r.workspace_naive_bytes
    assert r.kv_cache_bytes == 2 * 2 * 1 * 64 * 128 * 4
    assert any("grid synchronization" in n for n in r.capability_notes)
    assert any(d.code == "numerical-contract" for d in r.diagnostics)
    assert len(r.specialization_key) == 64


def test_specialization_key_depends_on_dims():
    cfg1 = GPT2Config()
    cfg2 = GPT2Config(hidden=256, heads=8)
    exe1 = compile_model(model_config=cfg1, weights=random_weights(cfg1), workers=2)
    exe2 = compile_model(model_config=cfg2, weights=random_weights(cfg2), workers=2)
    exe3 = compile_model(model_config=cfg1, weights=random_weights(cfg1), workers=3)
    assert exe1.report.specialization_key != exe2.report.specialization_key
    # Worker count is launch-time, not a specialization (§13.2).
    assert exe1.report.specialization_key == exe3.report.specialization_key


def test_schedule_argument_validation():
    with pytest.raises(ValueError):
        compile_model(schedule="dynamic")  # M5 not implemented (§11 is optional)


# ===========================================================================
# Reuse-of-workspace robustness (§15.1: "repeated reuse of runtime buffers")
# ===========================================================================


def test_repeated_invocations_reuse_workspace_consistently():
    """Decode twice through the same executable+workspace object: the canary
    re-arm protocol must keep holding (§9.2)."""
    cfg = GPT2Config(cache_capacity=10)
    exe = compile_model(model_config=cfg, weights=random_weights(cfg), workers=4)
    workspace = exe.allocate_workspace()
    cache = KVCache(cfg)
    for p in range(cfg.cache_capacity):
        logits, _ = exe.run(np.array([5]), cache, p, workspace=workspace)
        assert np.isfinite(logits).all()
    # A second, fresh decode over the same (now dirty) workspace.
    cache2 = KVCache(cfg)
    ref_cache = KVCache(cfg)
    weights = exe.weights
    worst = 0.0
    for p in range(cfg.cache_capacity):
        ids = np.array([5])
        lg, _ = exe.run(ids, cache2, p, workspace=workspace)
        ref, _ = reference_forward(weights, ids, ref_cache, p, cfg)
        worst = max(worst, float(np.abs(lg - ref).max()))
    assert worst < 1e-9
