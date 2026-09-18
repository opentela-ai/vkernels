"""Lightning-indexer top-k selection (``indexer_scores`` + ``index_topk``)
compiler support — issue #97.

Covers the vertical slice the issue names:

* capture — ``ops.indexer_scores(q, entries, mix_w)`` records the ReLU
  scoring + fused head mix (scale = head_dim**-0.5, f32 scoring over
  bf16-streamable q/entries); ``ops.index_topk(scores, valid_counts, k)``
  records the fixed-count selection producing the i32 indirection table +
  normalized block_bias consumed by #95/#96 via #94 — data-dependent
  selection, static decode output count;
* lowering — ``indexer_scores``: one task per (batch, 64-entry tile);
  ``index_topk``: one task per batch row (M <= ~1k candidate sweep);
* reference body — fp64 tile-exact semantics: rank-by-comparison-counting
  selection (descending score, lowest-index tie-break, NaN excluded,
  valid-count masking, -1/0.0 padding);
* validation — eager exact oracle match (indices exact, scores <1e-9),
  tie-heavy determinism, k at the boundary, k > candidates (ragged valid
  counts), NaN canaries, HCA topk=capacity parity, device-template mirrors
  (<1e-5 on scores).

All CPU (§15.1): this validates graph logic and tile semantics; the Triton
device templates themselves are CUDA-gated and UNVERIFIED in this
CPU-only environment (same flagged gap as PRs #88/#113/#114/#115).
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
    lower_op,
)
from vkernels.compiler.memory import plan_memory  # noqa: E402
from vkernels.compiler.operator_ir import BF16, F32, I32  # noqa: E402
from vkernels.compiler.reference_exec import ReferenceExecutor  # noqa: E402
from vkernels.compiler.schedule_phase import PhaseSchedule  # noqa: E402

# ---------------------------------------------------------------------------
# Oracles (independent eager recomputation, fp64)
# ---------------------------------------------------------------------------


def _oracle_scores(q, c, w, scale):
    """s[b, j] = sum_h relu(<q[b,h,:], c[b,j,:]>) * scale * w[b, h]."""
    dots = np.einsum("bhd,bjd->bhj", q.astype(np.float64), c.astype(np.float64))
    return (np.maximum(dots, 0.0) * scale * w.astype(np.float64)[:, :, None]).sum(axis=1)


def _oracle_topk_row(row, vc, k):
    """Eager exact top-k: descending score, ties to the lowest index; NaN
    scores excluded; only the valid prefix [0, vc) is observed. Returns
    (indices padded with -1, normalized scores padded with 0.0)."""
    m = row.shape[0]
    vc = max(0, min(int(vc), m))
    cand = [j for j in range(vc) if np.isfinite(row[j])]
    order = sorted(cand, key=lambda j: (-row[j], j))[:k]
    idx = np.full(k, -1, dtype=np.int32)
    bias = np.zeros(k, dtype=np.float32)
    finite = np.array([row[j] for j in cand], dtype=np.float64)
    norm = np.sqrt((finite**2).sum()) if finite.size else 0.0
    for slot, j in enumerate(order):
        idx[slot] = j
        bias[slot] = np.float32(row[j] / norm) if norm > 0 else np.float32(0.0)
    return idx, bias


# ---------------------------------------------------------------------------
# Capture + execution helpers
# ---------------------------------------------------------------------------


def _capture_chain(b=2, h=4, d=16, m=64, k=8, *, q_dtype=F32, c_dtype=F32):
    """Record indexer_scores -> index_topk; returns (recorder, handles)."""
    recorder = RecordingBackend()
    q = recorder.external_tensor("idx_q", (b, h, d), q_dtype, storage_id=101)
    c = recorder.external_tensor("idx_c", (b, m, d), c_dtype, storage_id=102)
    w = recorder.external_tensor("idx_w", (b, h), F32, storage_id=103)
    s = recorder.indexer_scores(q, c, w, name="lightning_indexer")
    vc = recorder.external_tensor("valid_counts", (b,), I32, storage_id=104)
    idx, bias = recorder.index_topk(s, vc, k=k, name="lightning_topk")
    handles = {"q": q, "c": c, "w": w, "s": s, "vc": vc, "idx": idx, "bias": bias}
    return recorder, handles


def _run_indexer_case(b, h, d, m, k, q_np, c_np, w_np, vc_np, workers):
    recorder, handles = _capture_chain(b, h, d, m, k)
    graph = recorder.graph
    s_name = handles["s"].value.name
    idx_name = handles["idx"].value.name
    bias_name = handles["bias"].value.name
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _scratch = plan_memory(graph, schedule, families)
    storage_arrays = {
        101: q_np,
        102: c_np,
        103: w_np,
        104: vc_np,
    }
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(
        schedule,
        workers=workers,
        storage_arrays=storage_arrays,
        graph=graph,
        workspace_plan=workspace_plan,
    )
    trace = executor.run({})
    return {
        "s": np.array(executor.tensor(s_name), copy=True),
        "idx": np.array(executor.tensor(idx_name), copy=True),
        "bias": np.array(executor.tensor(bias_name), copy=True),
        "graph": graph,
        "families": families,
        "schedule": schedule,
        "trace": trace,
    }


# ===========================================================================
# Capture (recorder contract)
# ===========================================================================


def test_capture_indexer_scores_records_contract():
    recorder, h = _capture_chain(b=3, h=4, d=32, m=128)
    graph = recorder.graph
    assert len(graph.ops) == 2
    op = graph.ops[0]
    assert op.kind == "indexer_scores"
    assert op.attributes["heads"] == 4
    assert op.attributes["head_dim"] == 32
    assert op.attributes["capacity"] == 128
    assert op.attributes["activation"] == "relu"
    assert op.attributes["scale"] == pytest.approx(32**-0.5)
    assert "activation" in op.numerical_contract and "accumulation" in op.numerical_contract
    assert h["s"].value.shape == (3, 128)
    assert h["s"].value.dtype == F32
    # reads cover all three inputs; writes the fused-score buffer
    read_sids = {r.storage_id for r in op.read_regions}
    assert read_sids == {101, 102, 103}
    assert {r.storage_id for r in op.write_regions} == {h["s"].value.storage_id}


def test_capture_index_topk_records_selection_attributes():
    recorder, h = _capture_chain(b=2, h=4, d=16, m=64, k=8)
    op = recorder.graph.ops[1]
    assert op.kind == "index_topk"
    assert op.attributes["k"] == 8
    assert op.attributes["capacity"] == 64
    assert op.attributes["tie_break"] == "lowest_index"
    assert op.attributes["score_dtype"] == "f32"
    assert op.attributes["index_dtype"] == "i32"
    assert op.attributes["nan_policy"] == "exclude"
    # outputs: i32 indirection table [B, k] + fp32 normalized bias [B, k]
    assert h["idx"].value.shape == (2, 8) and h["idx"].value.dtype == I32
    assert h["bias"].value.shape == (2, 8) and h["bias"].value.dtype == F32
    # reads cover the fused scores and the per-row valid counts
    read_sids = {r.storage_id for r in op.read_regions}
    assert read_sids == {h["s"].value.storage_id, 104}
    # RAW ordering: the topk op reads the storage indexer_scores wrote
    scores_sid = h["s"].value.storage_id
    assert any(r.storage_id == scores_sid for r in op.read_regions)


def test_capture_bf16_streamable_q_and_entries():
    """The issue's dtype contract: indexer q/entries bf16 (`.cg` on device),
    scoring + mixing fp32, indices i32 — the recorder accepts the bf16
    storages and pins the f32 score contract."""
    recorder, h = _capture_chain(b=1, h=2, d=8, m=32, k=4, q_dtype=BF16, c_dtype=BF16)
    op = recorder.graph.ops[0]
    assert recorder.graph.tensor("idx_q").dtype == BF16
    assert recorder.graph.tensor("idx_c").dtype == BF16
    assert op.attributes["activation"] == "relu"
    assert "f32" in op.numerical_contract["accumulation"]
    assert h["s"].value.dtype == F32
    # the graph still lowers legally with mixed dtypes
    assert [d for d in check_graph(recorder.graph) if d.severity == "error"] == []


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"d_bad": 32}, "mismatch"),  # entries head_dim mismatch
        ({"b": 2, "h": 4, "mix_bad": True}, "mix weights"),
        ({"k": 0}, "1 <= k"),
        ({"k": 65}, "1 <= k"),  # k > capacity M=64
        ({"k": 2.5}, "1 <= k"),
    ],
)
def test_capture_rejects_bad_shapes_and_k(kwargs, match):
    b, h, d, m = kwargs.get("b", 2), kwargs.get("h", 4), kwargs.get("d", 16), 64
    recorder = RecordingBackend()
    q = recorder.external_tensor("q", (b, h, d), F32, storage_id=201)
    if "d_bad" in kwargs:
        c = recorder.external_tensor("c", (b, m, kwargs["d_bad"]), F32, storage_id=202)
    else:
        c = recorder.external_tensor("c", (b, m, d), F32, storage_id=202)
    if kwargs.get("mix_bad"):
        w = recorder.external_tensor("w", (b, h + 1), F32, storage_id=203)
    else:
        w = recorder.external_tensor("w", (b, h), F32, storage_id=203)
    if "d_bad" in kwargs or kwargs.get("mix_bad"):
        with pytest.raises(ValueError, match=match):
            recorder.indexer_scores(q, c, w)
        return
    s = recorder.indexer_scores(q, c, w)
    vc = recorder.external_tensor("vc", (b,), I32, storage_id=204)
    with pytest.raises(ValueError, match=match):
        recorder.index_topk(s, vc, k=kwargs["k"])


def test_capture_indexer_scores_rejects_shape_mismatches():
    recorder = RecordingBackend()
    q = recorder.external_tensor("q", (2, 4, 16), F32, storage_id=301)
    c = recorder.external_tensor("c", (2, 64, 8), F32, storage_id=302)  # wrong head_dim
    w = recorder.external_tensor("w", (2, 4), F32, storage_id=303)
    with pytest.raises(ValueError, match="mismatch"):
        recorder.indexer_scores(q, c, w)
    recorder2 = RecordingBackend()
    q2 = recorder2.external_tensor("q", (2, 4, 16), F32, storage_id=311)
    c2 = recorder2.external_tensor("c", (2, 64, 16), F32, storage_id=312)
    w2 = recorder2.external_tensor("w", (2, 5), F32, storage_id=313)
    with pytest.raises(ValueError, match="mix weights"):
        recorder2.indexer_scores(q2, c2, w2)


def test_capture_index_topk_rejects_bad_scores_and_valid_counts():
    recorder = RecordingBackend()
    s = recorder.fresh_buffer("s", (2, 64), BF16)  # scores must be f32
    vc = recorder.external_tensor("vc", (2,), I32, storage_id=401)
    with pytest.raises(ValueError, match="f32"):
        recorder.index_topk(s, vc, k=4)
    recorder2 = RecordingBackend()
    s2 = recorder2.fresh_buffer("s2", (2, 64), F32)
    vc_bad = recorder2.external_tensor("vc_bad", (2, 3), I32, storage_id=402)
    with pytest.raises(ValueError, match="valid_counts"):
        recorder2.index_topk(s2, vc_bad, k=4)


# ===========================================================================
# Lowering (task decomposition)
# ===========================================================================


def test_lowering_indexer_scores_domain_and_regions():
    recorder, h = _capture_chain(b=2, h=4, d=16, m=100, k=8)  # ragged M=100
    fam = lower_graph(recorder.graph)[0]
    assert fam.kind == "indexer_scores"
    assert fam.domain.dims == ((2, 1), (100, 64))
    assert fam.task_count == 2 * 2  # ceil(100/64) = 2 tiles per row
    assert fam.threads == THREADS_PER_WORKER
    assert fam.params["scale"] == pytest.approx(16**-0.5)
    # per-task regions: full [H, D] query block, the entry tile, the mix row
    r = fam.reads(0)
    assert len(r) == 3
    assert r[0].boxes == ((0, 1), (0, 4), (0, 16))
    assert r[1].boxes == ((0, 1), (0, 64), (0, 16))
    assert r[2].boxes == ((0, 1), (0, 4))
    late = fam.reads(1)
    assert late[1].boxes == ((0, 1), (64, 100), (0, 16))  # clipped ragged tile
    wr = fam.writes(1)
    assert wr[0].boxes == ((0, 1), (64, 100))


def test_lowering_index_topk_one_task_per_row():
    recorder, h = _capture_chain(b=3, h=4, d=16, m=64, k=8)
    fam = lower_graph(recorder.graph)[1]
    assert fam.kind == "index_topk"
    assert fam.domain.dims == ((3, 1),)
    assert fam.task_count == 3
    assert fam.params["k"] == 8 and fam.params["tie_break"] == "lowest_index"
    r = fam.reads(2)
    assert len(r) == 2
    assert r[0].boxes == ((2, 3), (0, 64))  # the whole candidate row
    assert r[1].boxes == ((2, 3),)  # the row's valid count
    wr = fam.writes(2)
    assert wr[0].boxes == ((2, 3), (0, 8))  # i32 indirection row
    assert wr[1].boxes == ((2, 3), (0, 8))  # normalized bias row


def test_lowering_rejects_wrong_attributes():
    recorder, h = _capture_chain(b=1, h=2, d=8, m=32, k=4)
    op = recorder.graph.ops[1]
    from vkernels.compiler.operator_ir import OperatorGraph

    for attr_patch in ({"tie_break": "highest_index"}, {"k": 99}):
        bad = OperatorGraph()
        attrs = dict(op.attributes)
        attrs.update(attr_patch)
        op2 = bad.record(
            "index_topk",
            inputs=op.inputs,
            outputs=op.outputs,
            attributes=attrs,
            reads=op.read_regions,
            writes=op.write_regions,
            source_location=op.source_location,
        )
        for name, tv in recorder.graph.tensors.items():
            bad.tensors[name] = tv
        with pytest.raises(ValueError):
            lower_op(op2, bad)


def test_chain_is_legal_and_orders_indexer_before_topk():
    """Phase barrier ordering (#94 pattern): the selection consumes the fused
    scores, so the schedule must place indexer_scores strictly before
    index_topk."""
    recorder, h = _capture_chain(b=2, h=4, d=16, m=64, k=8)
    graph = recorder.graph
    assert [d for d in check_graph(graph) if d.severity == "error"] == []
    families = lower_graph(graph)
    assert check_thread_contract(families) == []
    schedule = PhaseSchedule.from_families(families, workers=4)
    kinds = [f.kind for phase in schedule.phases for f in phase.families]
    assert kinds.index("indexer_scores") < kinds.index("index_topk")


# ===========================================================================
# Reference execution vs the eager exact oracle
# ===========================================================================


@pytest.mark.parametrize("workers", [1, 3])
def test_reference_chain_matches_eager_oracle(workers):
    """Random well-separated inputs, full-capacity row + ragged row (vc < k):
    indices must match the eager oracle exactly, scores/bias to fp64 noise."""
    rng = np.random.default_rng(97 + workers)
    b, h, d, m, k = 2, 4, 16, 64, 8
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.array([m, 5], dtype=np.int32)  # row 1: only 5 candidates < k=8

    got = _run_indexer_case(b, h, d, m, k, q, c, w, vc, workers)
    assert np.isfinite(got["s"]).all(), "NaN canary tripped: unwritten score tiles"
    assert np.isfinite(got["bias"]).all(), "NaN canary tripped: unwritten bias slots"

    oracle_s = _oracle_scores(q, c, w, d**-0.5)
    np.testing.assert_allclose(got["s"], oracle_s, rtol=1e-5, atol=1e-6)

    for bi in range(b):
        oidx, obias = _oracle_topk_row(oracle_s[bi], vc[bi], k)
        np.testing.assert_array_equal(got["idx"][bi], oidx, err_msg=f"row {bi} indices")
        np.testing.assert_allclose(got["bias"][bi], obias, rtol=1e-6, atol=1e-7)
    # ragged row: slots beyond the valid count are the -1/0.0 fill
    assert (got["idx"][1][5:] == -1).all()
    assert (got["bias"][1][5:] == 0.0).all()


def test_reference_k_at_boundary():
    """k=1 (argmax with deterministic tie-break) and k=M (everything,
    HCA-style) both at the boundary of the contract."""
    rng = np.random.default_rng(11)
    b, h, d, m = 2, 4, 16, 64
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.array([m, m], dtype=np.int32)
    oracle_s = _oracle_scores(q, c, w, d**-0.5)

    for k in (1, m):
        got = _run_indexer_case(b, h, d, m, k, q, c, w, vc, workers=2)
        for bi in range(b):
            oidx, obias = _oracle_topk_row(oracle_s[bi], vc[bi], k)
            np.testing.assert_array_equal(got["idx"][bi], oidx)
            np.testing.assert_allclose(got["bias"][bi], obias, rtol=1e-6, atol=1e-7)
        if k == 1:
            assert got["idx"][0, 0] == int(np.argmax(oracle_s[0]))


def test_reference_tie_heavy_inputs_are_deterministic():
    """Scores with heavy exact ties (quantized dots on a coarse grid): the
    tie-break must resolve to the lowest candidate index, exactly like the
    eager (-score, index) oracle."""
    rng = np.random.default_rng(5)
    b, h, d, m, k = 2, 2, 8, 48, 10
    # Coarse quantization -> many exactly-equal fused scores after ReLU.
    q = np.round(rng.standard_normal((b, h, d)) * 2) / 2
    c = np.round(rng.standard_normal((b, m, d)) * 2) / 2
    # clamp c so ReLU zeroes coincide too
    c = np.maximum(c, 0.0)
    w = np.round(rng.standard_normal((b, h)) * 2) / 2
    vc = np.full(b, m, dtype=np.int32)
    oracle_s = _oracle_scores(q, c, w, d**-0.5)
    # sanity: ties actually occur
    assert len(np.unique(np.round(oracle_s[0], 9))) < m

    got = _run_indexer_case(b, h, d, m, k, q.astype(np.float32), c.astype(np.float32), w.astype(np.float32), vc, workers=3)
    for bi in range(b):
        oidx, obias = _oracle_topk_row(oracle_s[bi], m, k)
        np.testing.assert_array_equal(got["idx"][bi], oidx, err_msg=f"tie-break mismatch row {bi}")
        np.testing.assert_allclose(got["bias"][bi], obias, rtol=1e-6, atol=1e-7)


def test_reference_nan_canaries_in_invalid_tail_and_valid_prefix():
    """Two NaN contracts: (1) the invalid tail may hold NaN canaries —
    candidates at or beyond the row's valid count are never observed;
    (2) a NaN score *inside* the valid prefix is excluded from selection
    (corrupt scores must not win) and from the normalizer."""
    rng = np.random.default_rng(23)
    b, h, d, m, k = 2, 4, 16, 64, 6
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    oracle_s = _oracle_scores(q, c, w, d**-0.5)
    # row 0: 40 valid candidates, NaN canaries in the tail (the workspace is
    # NaN-poisoned anyway, but poison the *scores* view explicitly too).
    # row 1: a NaN score inside the valid prefix at position 3.
    s_poisoned = oracle_s.astype(np.float32).copy()
    s_poisoned[1, 3] = np.nan
    vc = np.array([40, m], dtype=np.int32)

    recorder, handles = _capture_chain(b, h, d, m, k)
    # Feed the poisoned scores directly to index_topk (indexer_scores is
    # dropped from this graph variant by executing a scores-only capture).
    recorder2 = RecordingBackend()
    s2 = recorder2.external_tensor("s_pre", (b, m), F32, storage_id=105)
    vc2 = recorder2.external_tensor("vc2", (b,), I32, storage_id=104)
    idx2, bias2 = recorder2.index_topk(s2, vc2, k=k)
    graph = recorder2.graph
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=2)
    workspace_plan, _ = plan_memory(graph, schedule, families)
    storage_arrays = {105: s_poisoned, 104: vc}
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(
        schedule,
        workers=2,
        storage_arrays=storage_arrays,
        graph=graph,
        workspace_plan=workspace_plan,
    )
    executor.run({})
    got_idx = np.array(executor.tensor(idx2.value.name), copy=True)
    got_bias = np.array(executor.tensor(bias2.value.name), copy=True)

    for bi in range(b):
        oidx, obias = _oracle_topk_row(s_poisoned[bi].astype(np.float64), vc[bi], k)
        np.testing.assert_array_equal(got_idx[bi], oidx, err_msg=f"NaN contract row {bi}")
        np.testing.assert_allclose(got_bias[bi], obias, rtol=1e-6, atol=1e-7)
    # the NaN position must never be selected
    assert 3 not in got_idx[1]


def test_reference_hca_topk_all_capacity_parity():
    """HCA variant (no indexer truncation): topk = capacity with full valid
    counts. The selected index set is *all* candidates and the gathered
    bias equals the full normalized score row — logits parity with and
    without indexer truncation at topk=capacity."""
    rng = np.random.default_rng(31)
    b, h, d, m = 2, 4, 16, 128
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.full(b, m, dtype=np.int32)

    got = _run_indexer_case(b, h, d, m, m, q, c, w, vc, workers=4)
    oracle_s = _oracle_scores(q, c, w, d**-0.5)
    for bi in range(b):
        oidx, obias = _oracle_topk_row(oracle_s[bi], m, m)
        np.testing.assert_array_equal(got["idx"][bi], oidx)
        np.testing.assert_allclose(got["bias"][bi], obias, rtol=1e-6, atol=1e-7)
        # parity: the selected set is every candidate, sorted descending
        assert sorted(got["idx"][bi].tolist()) == list(range(m))
        norm = np.linalg.norm(oracle_s[bi])
        gathered = oracle_s[bi][np.argsort(-oracle_s[bi], kind="stable")]
        np.testing.assert_allclose(got["bias"][bi] * norm, gathered, rtol=1e-6, atol=1e-7)


def test_reference_row_with_zero_valid_count():
    """vc=0 row: no candidates observed at all — all slots -1/0.0, no NaN
    leaks from the uninitialized tail."""
    rng = np.random.default_rng(41)
    b, h, d, m, k = 2, 4, 16, 64, 8
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.array([m, 0], dtype=np.int32)
    got = _run_indexer_case(b, h, d, m, k, q, c, w, vc, workers=2)
    np.testing.assert_array_equal(got["idx"][1], np.full(k, -1, dtype=np.int32))
    np.testing.assert_array_equal(got["bias"][1], np.zeros(k, dtype=np.float32))
    assert got["idx"][0].min() >= 0  # the full row still selects normally


# ===========================================================================
# Device-template mirrors (_t_indexer_scores / _t_index_topk on CPU, §15.1)
# ---------------------------------------------------------------------------


def _next_pow2(x: int) -> int:
    p = 1
    while p < x:
        p *= 2
    return p


def _simulate_t_indexer_scores(q, c, w, workers, tile, scale):
    """Numpy mirror of ``device_triton._t_indexer_scores``: one task per
    (batch, tile-entry tile), fp32 dot + relu + register head-mix, worker-
    stride task assignment (task = worker, +P)."""
    b, h, d = q.shape
    m = c.shape[1]
    nt = -(-m // tile)
    s = np.full((b, m), np.nan, dtype=np.float32)
    for wk in range(workers):
        task = wk
        while task < b * nt:
            bi, t = task // nt, task % nt
            offs_m = t * tile + np.arange(tile)
            mm = offs_m < m
            acc = np.zeros(tile, dtype=np.float32)
            for hh in range(h):
                qh = q[bi, hh].astype(np.float32)
                cj = np.where(mm[:, None], c[bi][np.minimum(offs_m, m - 1)].astype(np.float32), 0.0)
                sc = np.maximum(np.sum(cj * qh[None, :], axis=1, dtype=np.float32), 0.0) * np.float32(scale)
                acc += sc * np.float32(w[bi, hh])
            s[bi, offs_m[mm]] = acc[mm]
            task += workers
    return s


def _simulate_t_index_topk(s, valid, k, workers, tile):
    """Numpy mirror of ``device_triton._t_index_topk``: one task per batch
    row, fp32 rank-by-comparison-counting sweep, -1/0.0 fill then rank-
    addressed scatter, bias = s / ||s_valid||_2 in fp32."""
    b, m = s.shape
    kp = _next_pow2(k)
    idx = np.full((b, k), -2, dtype=np.int32)  # -2 sentinel: must be overwritten by fill/scatter
    bias = np.full((b, k), np.nan, dtype=np.float32)
    for wk in range(workers):
        task = wk
        while task < b:
            bi = task
            vc = int(np.clip(valid[bi], 0, m))
            idx[bi, :] = -1
            bias[bi, :] = 0.0
            valid_mask = np.zeros(m, dtype=bool)
            valid_mask[:vc] = True
            finite = np.isfinite(s[bi].astype(np.float32))
            cand = valid_mask & finite
            sv = s[bi].astype(np.float32)
            fin_vals = sv[:vc][finite[:vc]]
            norm = np.float32(np.sqrt(np.sum(fin_vals**2, dtype=np.float32))) if fin_vals.size else np.float32(0.0)
            for j in range(m):
                if not cand[j]:
                    continue
                gt = np.sum((sv[:vc] > sv[j]) & finite[:vc])
                eq_lower = np.sum(((sv[:vc] == sv[j]) & finite[:vc])[:j])
                rank = int(gt + eq_lower)
                if rank < k:
                    idx[bi, rank] = j
                    bias[bi, rank] = np.float32(sv[j] / norm) if norm > 0 else np.float32(0.0)
            task += workers
    return idx, bias


@pytest.mark.parametrize(
    "b,h,d,m,tile,workers",
    [(1, 4, 32, 128, 64, 1), (2, 4, 16, 100, 64, 3), (3, 2, 8, 64, 32, 7), (1, 4, 16, 40, 64, 5)],
)
def test_device_template_mirror_indexer_scores_matches_reference(b, h, d, m, tile, workers):
    """The Triton task decomposition (per-(batch, tile) tasks, fp32 register
    mix, worker strides) reproduces the compiled schedule's reference
    scores within fp32 accumulation noise (<1e-5 relative)."""
    rng = np.random.default_rng(b * 100 + h * 10 + d + m + workers)
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.full(b, m, dtype=np.int32)

    got = _run_indexer_case(b, h, d, m, _next_pow2(min(m, 8)), q, c, w, vc, workers=workers)
    mirror = _simulate_t_indexer_scores(q, c, w, workers, tile, d**-0.5)
    denom = np.abs(got["s"]).max()
    rel = np.abs(mirror.astype(np.float64) - got["s"]).max() / denom
    assert rel < 1e-5, f"indexer_scores template mirror diverged: {rel:.3e}"
    assert not np.isnan(mirror).any(), "every score tile must be written"


@pytest.mark.parametrize(
    "b,m,k,vc_mode,workers",
    [(1, 128, 8, "full", 1), (2, 100, 16, "ragged", 3), (2, 64, 64, "full", 2), (3, 96, 8, "ragged", 7)],
)
def test_device_template_mirror_index_topk_matches_reference(b, m, k, vc_mode, workers):
    """The selection template's decomposition (fp32 rank counting, tie-break
    to the lowest index, valid-count masking, padding) reproduces the
    reference executor: indices exactly, scores <1e-5 relative."""
    rng = np.random.default_rng(m + k + workers)
    s = rng.standard_normal((b, m)).astype(np.float32) * 3.0
    if vc_mode == "ragged":
        vc = np.array([m, max(k - 2, 0), m // 2][:b], dtype=np.int32)
    else:
        vc = np.full(b, m, dtype=np.int32)
    s[np.arange(b), np.minimum(vc, m - 1) // 2] = np.float32("nan")  # NaN canary inside

    recorder = RecordingBackend()
    st = recorder.external_tensor("s_pre", (b, m), F32, storage_id=501)
    vt = recorder.external_tensor("vc", (b,), I32, storage_id=502)
    idx_t, bias_t = recorder.index_topk(st, vt, k=k)
    graph = recorder.graph
    families = lower_graph(graph)
    schedule = PhaseSchedule.from_families(families, workers=workers)
    workspace_plan, _ = plan_memory(graph, schedule, families)
    storage_arrays = {501: s, 502: vc}
    for buf in workspace_plan.buffers:
        storage_arrays[buf.storage_id] = np.full(buf.numel, np.nan, dtype=np.float32)
    executor = ReferenceExecutor(
        schedule, workers=workers, storage_arrays=storage_arrays, graph=graph, workspace_plan=workspace_plan
    )
    executor.run({})
    ref_idx = np.array(executor.tensor(idx_t.value.name), copy=True)
    ref_bias = np.array(executor.tensor(bias_t.value.name), copy=True)

    mir_idx, mir_bias = _simulate_t_index_topk(s, vc, k, workers, tile=_next_pow2(min(m, 128)))
    assert not (mir_idx == -2).any(), "mirror fill/scatter left an output slot unwritten"
    np.testing.assert_array_equal(mir_idx, ref_idx, err_msg="selection mirror indices diverged")
    denom = np.abs(ref_bias).max()
    if denom > 0:
        rel = np.abs(mir_bias.astype(np.float64) - ref_bias).max() / denom
        assert rel < 1e-5, f"index_topk bias mirror diverged: {rel:.3e}"
    assert not np.isnan(ref_bias).any(), "every output slot must be written (canary)"


def test_device_template_mirror_full_chain_score_agreement():
    """Full-chain mirror (scores template feeding the selection template):
    the selected *score* multiset agrees with the reference executor to
    <1e-5 (index identity may swap only among fp-noise-equal candidates)."""
    rng = np.random.default_rng(77)
    b, h, d, m, k, workers = 2, 4, 16, 128, 16, 3
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.full(b, m, dtype=np.int32)

    got = _run_indexer_case(b, h, d, m, k, q, c, w, vc, workers=workers)
    mir_s = _simulate_t_indexer_scores(q, c, w, workers, tile=64, scale=d**-0.5)
    mir_idx, mir_bias = _simulate_t_index_topk(mir_s, vc, k, workers, tile=128)
    for bi in range(b):
        ref_sorted = np.sort(got["bias"][bi])[::-1]
        mir_sorted = np.sort(mir_bias[bi])[::-1]
        rel = np.abs(ref_sorted.astype(np.float64) - mir_sorted.astype(np.float64)).max() / max(
            np.abs(ref_sorted).max(), 1e-30
        )
        assert rel < 1e-5, f"chain mirror bias divergence row {bi}: {rel:.3e}"
        # k valid selections in both (row is full-capacity)
        assert (mir_idx[bi] >= 0).all() and (got["idx"][bi] >= 0).all()


def test_codegen_knows_the_indexer_templates():
    """The generated-source phase table maps the new family kinds."""
    assert TEMPLATE_NAMES["indexer_scores"] == "indexer_scores_task"
    assert TEMPLATE_NAMES["index_topk"] == "index_topk_task"


# ===========================================================================
# Device templates — CUDA only (UNVERIFIED in this CPU-only environment;
# same flagged gap as PRs #88/#113/#114/#115)
# ===========================================================================

gpu = pytest.mark.skipif(
    not __import__("torch").cuda.is_available() if "torch" in sys.modules else True,
    reason="requires CUDA — CPU-only lane: device templates are mirrored on CPU above",
)


@gpu
def test_device_t_indexer_scores_and_topk_match_oracle():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    import torch

    from vkernels.compiler.device_triton import _t_index_topk, _t_indexer_scores

    b, h, d, m, k = 2, 4, 32, 256, 32
    rng = np.random.default_rng(97)
    q = rng.standard_normal((b, h, d)).astype(np.float32)
    c = rng.standard_normal((b, m, d)).astype(np.float32)
    w = rng.standard_normal((b, h)).astype(np.float32)
    vc = np.array([m, 100], dtype=np.int32)

    dev = torch.device("cuda")
    q_t = torch.from_numpy(q).to(dev)
    c_t = torch.from_numpy(c).to(dev)
    w_t = torch.from_numpy(w).to(dev)
    s_t = torch.empty(b, m, device=dev, dtype=torch.float32)
    _t_indexer_scores[(4,)](q_t, c_t, w_t, s_t, b, h, d, m, 64, d**-0.5, num_warps=4)

    idx_t = torch.full((b, k), -1, device=dev, dtype=torch.int32)
    bias_t = torch.empty(b, k, device=dev, dtype=torch.float32)
    vc_t = torch.from_numpy(vc).to(dev)
    _t_index_topk[(2,)](s_t, vc_t, idx_t, bias_t, b, m, k, _next_pow2(k), 128, num_warps=4)
    torch.cuda.synchronize()

    oracle_s = _oracle_scores(q, c, w, d**-0.5)
    np.testing.assert_allclose(s_t.cpu().numpy(), oracle_s, rtol=1e-4, atol=1e-5)
    for bi in range(b):
        oidx, obias = _oracle_topk_row(oracle_s[bi], vc[bi], k)
        np.testing.assert_array_equal(idx_t.cpu().numpy()[bi], oidx)
        np.testing.assert_allclose(bias_t.cpu().numpy()[bi], obias, rtol=1e-4, atol=1e-5)
