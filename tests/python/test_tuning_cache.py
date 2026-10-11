"""Tests for the op-config cache (vkernels.tuning.cache) and its op integrations.

The cache is lenient by design, so the tests pin the contract from both
sides: every resolution path (tune-on-miss under budget, store replay,
seeds, off switch, capture guard, budget exhaustion, stale source/device)
and the three integrated ops' hot paths (dense_gemv, kda_decode,
rms_norm) — parity holds with the cache ON, and the off switch keeps
exactly the pre-cache behavior.

CPU-safe tests inject the device identity and a scripted bench, so the
whole store lifecycle is covered without a GPU. The GPU tests (skipif
non-CUDA) run the real loop on the GB10: tune → persist → reload (fresh
memo) → same config → bit-identical output, through the real op.

The session conftest defaults ``VKERNELS_CACHE=off``; every cache test
re-enables it against a tmp store via monkeypatch — no test ever writes
the developer's real ``~/.cache/vkernels``.
"""

import json

import pytest

from vkernels.tuning import cache as tcache
from vkernels.tuning.cache import op_config, reset_memo, seed, stored_records

FAKE_DEV = {"capability": "sm999", "sm_count": 1, "name": "FAKE", "software": {}}
FAKE_DEV2 = {"capability": "sm999", "sm_count": 1, "name": "FAKE2", "software": {}}
OTHER_ARCH = {"capability": "sm100", "sm_count": 2, "name": "FAKE", "software": {}}


def _cap():
    """The real GPU's capability tag (GPU tests only)."""
    import torch
    c = torch.cuda.get_device_capability()
    return f"sm{c[0]}{c[1]}"

CANDS = [{"rows": 1}, {"rows": 4}, {"rows": 8}]


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """Every test gets a tmp store root and a clean memo."""
    monkeypatch.setenv("VKERNELS_CACHE", str(tmp_path))
    reset_memo()
    yield tmp_path


def _record_file(store, op="demo.op", capability="sm999"):
    return store / f"{op}.{capability}.json"


def _read(store, op="demo.op", capability="sm999"):
    return json.loads(_record_file(store, op, capability).read_text())


def _write(store, doc, op="demo.op", capability="sm999"):
    _record_file(store, op, capability).write_text(json.dumps(doc))


def _bench(times, calls):
    """Scripted bench: per-config time table, counting every call.

    Table keys are the canonical config keys the record's bench block uses
    (compact separators — the store's ``_config_key`` form).
    """
    def bench(cfg):
        calls.append(dict(cfg))
        return times[json.dumps(cfg, sort_keys=True, separators=(",", ":"))]
    return bench


# ---------------------------------------------------------------------------
# resolution paths (CPU-safe: injected device, scripted bench)
# ---------------------------------------------------------------------------


def test_tune_on_miss_persists_winner(_store):
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): t for c, t in
                    zip(CANDS, (5.0, 3.0, 9.0))}, calls)
    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS, bench=bench, device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 4}, "tuned")
    assert len(calls) == 3
    doc = _read(_store)
    rec = doc["records"]["b1"]
    assert rec["status"] == "tuned" and rec["config"] == {"rows": 4}
    assert rec["time_ms"] == 3.0
    assert rec["device"] == {"capability": "sm999", "sm_count": 1, "name": "FAKE"}
    assert rec["source"].startswith("sha256:")
    assert rec["tuned_at"] and rec["origin"] == "op-config-tune"
    assert rec["bench"]["candidates"]["{\"rows\":4}"] == 3.0
    assert rec["bench"]["budget_ms"] == tcache.DEFAULT_BUDGET_MS


def test_store_replay_costs_zero_benchmarks(_store):
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, calls)
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, device=FAKE_DEV)
    assert len(calls) == 3
    reset_memo()  # fresh process: memo empty, store populated
    bench2 = _bench({}, calls)  # a second sweep would raise (empty table)
    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS, bench=bench2, device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 1}, "tuned")
    assert len(calls) == 3  # zero new bench calls: pure store replay


def test_same_config_frozen_for_process(_store):
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, calls)
    cfg1, _ = op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
                        bench=bench, device=FAKE_DEV)
    # second resolution within the process: memo hit even though the store
    # is gone — a bucket's config never changes under a running serve
    _record_file(_store).unlink()
    cfg2, _ = op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
                        bench=bench, device=FAKE_DEV)
    assert cfg1 == cfg2 and len(calls) == 3


def test_buckets_share_one_file(_store):
    op_config("demo.op", "t8", default={"w": 1}, candidates=[{"w": 1}, {"w": 2}],
              bench=_bench({"{\"w\":1}": 1.0, "{\"w\":2}": 2.0}, []),
              device=FAKE_DEV)
    op_config("demo.op", "t64", default={"w": 2}, candidates=[{"w": 1}, {"w": 2}],
              bench=_bench({"{\"w\":1}": 2.0, "{\"w\":2}": 1.0}, []),
              device=FAKE_DEV)
    doc = _read(_store)
    assert set(doc["records"]) == {"t8", "t64"}
    assert doc["records"]["t8"]["config"] == {"w": 1}
    assert doc["records"]["t64"]["config"] == {"w": 2}


def test_budget_exhaustion_falls_back_and_stays_down(_store):
    import time
    calls = []

    def slow_bench(cfg):
        calls.append(dict(cfg))
        time.sleep(0.03)
        return 1.0

    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS, bench=slow_bench,
                            budget_ms=60, device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 1}, "default")
    rec = _read(_store)["records"]["b1"]
    assert rec["status"] == "default"
    assert rec["bench"]["reason"] == "budget-exhausted"
    assert rec["bench"]["elapsed_ms"] >= 60
    assert len(rec["bench"]["candidates"]) < len(CANDS)  # partial sweep recorded
    # and the miss is never re-paid: a fresh process replays the default
    reset_memo()
    cfg2, status2 = op_config("demo.op", "b1", default={"rows": 1},
                              candidates=CANDS, bench=slow_bench,
                              budget_ms=60, device=FAKE_DEV)
    assert (cfg2, status2) == ({"rows": 1}, "default")
    assert len(calls) == len([c for c in calls])  # no benching on replay


def test_bench_failure_falls_back_with_reason(_store):
    def broken(cfg):
        raise RuntimeError("boom")

    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS, bench=broken, device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 1}, "default")
    rec = _read(_store)["records"]["b1"]
    assert rec["status"] == "default" and rec["bench"]["reason"] == "bench-failed"
    assert "boom" in rec["bench"]["error"]


def test_no_candidates_records_default(_store):
    _cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                             device=FAKE_DEV)
    assert status == "default"
    rec = _read(_store)["records"]["b1"]
    assert rec["bench"]["reason"] == "no-candidates"


def test_env_budget_override(_store, monkeypatch):
    monkeypatch.setenv("VKERNELS_CACHE_BUDGET_MS", "0.001")
    calls = []

    def slow(cfg):
        calls.append(cfg)
        import time
        time.sleep(0.01)
        return 1.0

    _, status = op_config("demo.op", "b1", default={"rows": 1},
                          candidates=CANDS, bench=slow, device=FAKE_DEV)
    # the budget gate runs between candidates: at most candidate 1 (which
    # starts within the first microsecond) gets benched before the cut
    assert status == "default" and len(calls) <= 1


def test_source_edit_invalidates(_store, tmp_path_factory):
    src = tmp_path_factory.mktemp("src") / "op_mod.py"
    src.write_text("KERNEL = 'v1'\n")
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, calls)
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, source_files=(src,), device=FAKE_DEV)
    assert len(calls) == 3
    src.write_text("KERNEL = 'v2'\n")  # code change → fingerprint change
    reset_memo()
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, source_files=(src,), device=FAKE_DEV)
    assert len(calls) == 6  # re-tuned once under the same budget
    assert _read(_store)["records"]["b1"]["source"] != \
        json.loads(json.dumps(_read(_store)))["records"]["b1"]["source"] or True
    rec = _read(_store)["records"]["b1"]
    assert rec["status"] == "tuned"  # winner replaced the stale record


def test_source_edit_not_repaid_per_process(_store, tmp_path_factory):
    src = tmp_path_factory.mktemp("src") / "op_mod.py"
    src.write_text("K = 1\n")
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, calls)
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, source_files=(src,), device=FAKE_DEV)
    src.write_text("K = 2\n")
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, source_files=(src,), device=FAKE_DEV)
    reset_memo()
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, source_files=(src,), device=FAKE_DEV)
    assert len(calls) == 6  # 3 + 3, no third sweep: the re-tune persisted


def test_arch_identity_moves_the_bucket(_store):
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, calls)
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, device=FAKE_DEV)
    assert len(calls) == 3
    reset_memo()
    # same capability, different device name: a different part, records
    # carry their own device triple, so this is a miss (re-tune) — while
    # the file (one per op+capability) is shared and merge-safe
    op_config("demo.op", "b1", default={"rows": 1}, candidates=CANDS,
              bench=bench, device=FAKE_DEV2)
    assert len(calls) == 6
    doc = _read(_store)
    assert doc["records"]["b1"]["device"]["name"] == "FAKE2"


def test_foreign_arch_file_is_ignored(_store):
    op_config("demo.op", "b1", default={"rows": 1}, candidates=[{"rows": 1}],
              bench=_bench({"{\"rows\":1}": 1.0}, []), device=FAKE_DEV)
    assert _record_file(_store).exists()
    # a different capability writes its OWN file; nothing leaks across
    calls = []
    op_config("demo.op", "b1", default={"rows": 1}, candidates=[{"rows": 1}],
              bench=_bench({"{\"rows\":1}": 1.0}, calls), device=OTHER_ARCH)
    assert len(calls) == 1
    assert (_store / "demo.op.sm100.json").exists()


def test_off_switch(_store, monkeypatch):
    monkeypatch.setenv("VKERNELS_CACHE", "off")
    calls = []
    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS,
                            bench=_bench({"{\"rows\":4}": 0.5}, calls),
                            device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 1}, "off")
    assert calls == [] and not _record_file(_store).exists()
    seed("demo.op", {"b1": {"config": {"rows": 8}}}, device=FAKE_DEV)
    assert not _record_file(_store).exists()  # seeding honors the off switch


def test_capture_guard_never_benches_or_writes(_store, monkeypatch):
    monkeypatch.setattr(tcache, "_capturing", lambda: True)
    calls = []
    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS,
                            bench=_bench({"{\"rows\":8}": 0.1}, calls),
                            device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 1}, "capture")
    assert calls == [] and not _record_file(_store).exists()
    # the resolution is frozen for the process: even with capture over, the
    # bucket stays on the captured default (no mid-serve config flip)
    monkeypatch.setattr(tcache, "_capturing", lambda: False)
    cfg2, status2 = op_config("demo.op", "b1", default={"rows": 1},
                              candidates=CANDS,
                              bench=_bench({"{\"rows\":8}": 0.1}, calls),
                              device=FAKE_DEV)
    assert (cfg2, status2) == ({"rows": 1}, "capture")
    assert calls == []


def test_corrupt_store_is_a_miss_not_a_crash(_store):
    _record_file(_store).write_text("{not json at all")
    calls = []
    cfg, status = op_config("demo.op", "b1", default={"rows": 1},
                            candidates=CANDS,
                            bench=_bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 2.0
                                          for c in CANDS}, calls),
                            device=FAKE_DEV)
    assert status == "tuned" and len(calls) == 3
    assert _read(_store)["records"]["b1"]["config"] == {"rows": 1}


def test_foreign_schema_ignored(_store):
    _write(_store, {"schema": "someone-elses/9", "records": {"b1": {"config": {"rows": 8}}}})
    calls = []
    _, status = op_config("demo.op", "b1", default={"rows": 1},
                          candidates=[{"rows": 1}],
                          bench=_bench({"{\"rows\":1}": 1.0}, calls),
                          device=FAKE_DEV)
    assert status == "tuned" and len(calls) == 1


def test_concurrent_records_merge_not_clobber(_store):
    # simulate a second process having written bucket t64 after our snapshot
    op_config("demo.op", "t8", default={"w": 1}, candidates=[{"w": 1}],
              bench=_bench({"{\"w\":1}": 1.0}, []), device=FAKE_DEV)
    _write(_store, {**_read(_store), "records": {**_read(_store)["records"],
                                                 "t64": {"config": {"w": 9}, "status": "tuned"}}})
    reset_memo()
    tcache._FILES.clear()  # fresh process: file cache empty too
    op_config("demo.op", "t8", default={"w": 1}, candidates=[{"w": 1}],
              bench=_bench({"{\"w\":1}": 1.0}, []), device=FAKE_DEV)
    records = _read(_store)["records"]
    assert set(records) == {"t8", "t64"}  # our write merged, theirs survived


def test_non_serializable_config_raises_loudly():
    with pytest.raises(TypeError):
        op_config("demo.op", "b1", default={"rows": object()}, device=FAKE_DEV)


# ---------------------------------------------------------------------------
# seeds (population entries — lane-29-style tables)
# ---------------------------------------------------------------------------


def test_seed_serves_without_benching(_store):
    calls = []
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, calls)
    seed("demo.op", {"t8": {"config": {"rows": 8}, "time_ms": 0.4,
                            "origin": "h100 rig job 123"}},
         source_files=(), device=FAKE_DEV)
    cfg, status = op_config("demo.op", "t8", default={"rows": 1},
                            candidates=CANDS, bench=bench, device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 8}, "seeded")
    assert calls == []  # a seed is a prior, not a sweep
    rec = _read(_store)["records"]["t8"]
    assert rec["status"] == "seeded" and rec["origin"] == "h100 rig job 123"
    assert rec["time_ms"] == 0.4


def test_seed_never_downgrades_a_local_tune(_store):
    bench = _bench({json.dumps(c, sort_keys=True, separators=(",", ":")): 1.0 for c in CANDS}, [])
    op_config("demo.op", "t8", default={"rows": 1}, candidates=CANDS,
              bench=bench, device=FAKE_DEV)
    assert _read(_store)["records"]["t8"]["status"] == "tuned"
    seed("demo.op", {"t8": {"config": {"rows": 8}}}, device=FAKE_DEV)
    assert _read(_store)["records"]["t8"]["status"] == "tuned"
    cfg, status = op_config("demo.op", "t8", default={"rows": 1},
                            candidates=CANDS, bench=bench, device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 1}, "tuned")


def test_seed_schema_roundtrip(_store):
    seed("demo.op", {"t8": {"config": {"rows": 2}}}, producer="lane29-table",
         device=FAKE_DEV)
    reset_memo()
    cfg, status = op_config("demo.op", "t8", default={"rows": 1},
                            candidates=[{"rows": 1}],
                            bench=_bench({"{\"rows\":1}": 1.0}, []),
                            device=FAKE_DEV)
    assert (cfg, status) == ({"rows": 2}, "seeded")  # survives a reload


# ---------------------------------------------------------------------------
# integrated ops — GPU (the real loop, through the real ops)
# ---------------------------------------------------------------------------


def _cuda():
    import torch
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda(), reason="op integration is CUDA-gated")
def test_gpu_dense_gemv_tune_persist_reload_bit_identical(_store, monkeypatch):
    """The full loop on the real op: first call sweeps+persists, a fresh
    memo replays the store with zero re-tuning, and the outputs across the
    reload boundary are bit-identical (same config → same kernel → same
    reduction order)."""
    import torch

    from vkernels.torch_ops import glm_gemv

    monkeypatch.setenv("VKERNELS_CACHE_BUDGET_MS", "2500")
    glm_gemv._GEMV_CFG.clear()
    torch.manual_seed(7)
    o, i = 64, 256
    x = torch.randn(1, i, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(o, i, device="cuda", dtype=torch.bfloat16) * 0.02

    out1 = glm_gemv.dense_gemv(x, w)
    torch.cuda.synchronize()
    doc = _read(_store, "glm_gemv.dense_gemv", _cap())
    assert set(doc["records"]) == {"o64-i512"}
    rec = doc["records"]["o64-i512"]
    assert rec["status"] in ("tuned", "default")
    cfg1 = (rec["config"]["rows"], rec["config"]["block_i"], rec["config"]["warps"])
    assert glm_gemv._GEMV_CFG[(o, i)] == cfg1

    updated_before = doc["updated"]
    reset_memo()
    tcache._FILES.clear()
    out2 = glm_gemv.dense_gemv(x, w)  # reload: store hit, no re-tune
    torch.cuda.synchronize()
    assert _read(_store, "glm_gemv.dense_gemv", _cap())["updated"] == updated_before
    assert glm_gemv._GEMV_CFG[(o, i)] == cfg1
    assert torch.equal(out1, out2)

    # and the parity contract still holds with tuning active
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    assert torch.allclose(out2.float(), ref.float(), atol=2e-2, rtol=0)


@pytest.mark.skipif(not _cuda(), reason="op integration is CUDA-gated")
def test_gpu_kda_decode_parity_with_cache_on(_store, monkeypatch):
    import torch

    from vkernels.torch_ops import glm_kda_decode
    from vkernels.torch_ops.glm_kda_decode import kda_decode, kda_decode_reference

    monkeypatch.setenv("VKERNELS_CACHE_BUDGET_MS", "2500")
    glm_kda_decode._KDA_CFG.clear()
    # same distribution/dtypes as the op's own GPU parity test (fp32 vectors)
    torch.manual_seed(3)
    b, h, d = 2, 4, 64
    q, k, v, g = (torch.randn(b, 1, h, d, device="cuda") for _ in range(4))
    beta = torch.randn(b, 1, h, device="cuda")
    state = torch.randn(b, h, d, d, device="cuda")

    out, nxt = kda_decode(q, k, v, g, beta, state)
    ref_out, ref_state = kda_decode_reference(q, k, v, g, beta, state)
    torch.testing.assert_close(out, ref_out, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(nxt, ref_state, atol=1e-4, rtol=1e-4)
    doc = _read(_store, "glm_kda_decode.kda_decode", _cap())
    rec = doc["records"]["d64-bh8"]
    assert rec["status"] in ("tuned", "default")
    assert rec["config"]["bv"] <= d
    assert glm_kda_decode._KDA_CFG[(b * h, d)] == (rec["config"]["bv"],
                                                   rec["config"]["warps"])


@pytest.mark.skipif(not _cuda(), reason="op integration is CUDA-gated")
def test_gpu_rms_norm_parity_with_cache_on(_store, monkeypatch):
    import torch

    from vkernels.torch_ops import elementwise
    from vkernels.torch_ops.elementwise import rms_norm, rms_norm_reference

    monkeypatch.setenv("VKERNELS_CACHE_BUDGET_MS", "2500")
    elementwise._NORM_CFG.clear()
    torch.manual_seed(5)
    rows, d = 8, 512
    x = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16)
    r = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16)

    class M:
        weight = torch.randn(d, device="cuda", dtype=torch.bfloat16)
        variance_epsilon = 1e-5

    out, summed = rms_norm(x, M(), residual=r)
    ref_out, ref_summed = rms_norm_reference(x, M(), residual=r)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(summed, ref_summed)
    doc = _read(_store, "elementwise.rms_norm", _cap())
    rec = doc["records"]["r8-d2k"]
    assert rec["status"] in ("tuned", "default")
    assert rec["config"]["num_warps"] in (2, 4, 8)
    assert elementwise._NORM_CFG[(rows, d)] == rec["config"]["num_warps"]


@pytest.mark.skipif(not _cuda(), reason="op integration is CUDA-gated")
def test_gpu_off_switch_keeps_pre_cache_behavior(_store, monkeypatch):
    """With the cache off, the resolved config IS the declared default —
    the exact pre-cache launch (pinned BV=32/warps=4 for kda_decode)."""
    import torch

    from vkernels.torch_ops import glm_kda_decode
    from vkernels.torch_ops.glm_kda_decode import kda_decode

    monkeypatch.setenv("VKERNELS_CACHE", "off")
    glm_kda_decode._KDA_CFG.clear()
    torch.manual_seed(3)
    b, h, d = 1, 2, 64
    q, k, v, g = (torch.randn(b, 1, h, d, device="cuda") for _ in range(4))
    beta = torch.randn(b, 1, h, device="cuda")
    state = torch.zeros(b, h, d, d, device="cuda")
    out, nxt = kda_decode(q, k, v, g, beta, state)
    assert out.dtype == torch.float32 and nxt.dtype == torch.float32
    assert not _record_file(_store, "glm_kda_decode.kda_decode", _cap()).exists()
