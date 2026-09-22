"""Tests for the persistent autotune store (:mod:`vkernels.torch_ops.tuning_cache`).

The store plumbing (record/lookup/schema/fingerprint/staleness) is
device-independent and runs anywhere. The kernel-level tests — the
tune-then-replay round trip through a real Triton kernel, and the
``mhc_projection`` integration — require CUDA and skip elsewhere.

Contract under test:

* a sweep's winner persists to ``<store>/<kernel>.<arch>.json`` and a
  *fresh* tuner instance replays it with zero benchmarking;
* device/software mismatch or changed producer sources are cache misses
  (re-tune) in lenient mode and raises in strict mode;
* ``VKERNELS_TUNING_CACHE=off`` disables persistence entirely.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
import torch
import triton
import triton.language as tl

from importlib import import_module

from vkernels.torch_ops.tuning_cache import (
    SCHEMA,
    TuningCache,
    TuningCacheError,
    device_fingerprint,
    persistent_autotune,
    tuning_enabled,
)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture
def source_file(tmp_path):
    path = tmp_path / "producer.py"
    path.write_text("version = 1\n")
    return path


@pytest.fixture
def cache(tmp_path, source_file):
    return TuningCache("k", [source_file], store_dir=tmp_path / "store",
                       device={"name": "fake", "arch": "sm00", "cu_count": 1,
                               "software": {"torch": "0", "triton": "0"}})


# ---------------------------------------------------------------------------
# store plumbing (no GPU)
# ---------------------------------------------------------------------------


def test_record_lookup_roundtrip(cache, tmp_path):
    cache.record((7, 1), kwargs={"BLOCK": 64}, num_warps=2, num_stages=3, time_ms=0.5)
    path = next((tmp_path / "store").iterdir())
    assert path.name.startswith("k.sm00")
    doc = json.loads(path.read_text())
    assert doc["schema"] == SCHEMA and doc["kernel"] == "k"
    record = cache.lookup((7, 1))
    assert record["config"] == {"kwargs": {"BLOCK": 64}, "num_warps": 2, "num_stages": 3}
    assert record["time_ms"] == 0.5
    assert list(record["producer"]["fingerprints"]) == [str(cache.source_files[0])]


def test_unknown_key_and_clear(cache):
    assert cache.lookup((1, 2)) is None
    cache.record((1,), kwargs={}, num_warps=1, num_stages=1, time_ms=1.0)
    cache.clear()
    assert cache.lookup((1,)) is None
    assert not cache._path().exists()


def test_fingerprint_staleness_is_a_miss(cache, source_file):
    cache.record((1,), kwargs={}, num_warps=1, num_stages=1, time_ms=1.0)
    source_file.write_text("version = 2\n")  # producer changed
    assert cache.lookup((1,)) is None  # lenient: re-tune
    cache.strict = True
    with pytest.raises(TuningCacheError, match="changed"):
        cache.lookup((1,))


def test_device_mismatch_is_a_miss(cache):
    cache.record((1,), kwargs={}, num_warps=1, num_stages=1, time_ms=1.0)
    cache._device = {**cache.device, "cu_count": 999}
    assert cache.lookup((1,)) is None
    cache.strict = True
    with pytest.raises(TuningCacheError, match="device"):
        cache.lookup((1,))


def test_unsupported_schema_is_ignored_lenient(cache, tmp_path):
    cache.record((1,), kwargs={}, num_warps=1, num_stages=1, time_ms=1.0)
    path = cache._path()
    doc = json.loads(path.read_text())
    doc["schema"] = "vk-tuning-store/0"
    path.write_text(json.dumps(doc))
    cache._records = None  # force reload
    assert cache.lookup((1,)) is None
    cache.strict = True
    cache._records = None  # strict re-read of the on-disk schema
    with pytest.raises(TuningCacheError, match="schema"):
        cache.lookup((1,))


def test_off_switch_disables_persistence(cache, monkeypatch):
    monkeypatch.setenv("VKERNELS_TUNING_CACHE", "off")
    assert not tuning_enabled()
    cache.record((1,), kwargs={}, num_warps=1, num_stages=1, time_ms=1.0)
    assert not cache._path().exists()
    assert cache.lookup((1,)) is None


def test_device_fingerprint_shape():
    fp = device_fingerprint()
    assert fp["arch"] and fp["cu_count"] > 0
    assert "torch" in fp["software"] and "triton" in fp["software"]
    assert "index" not in fp  # ordinals must not leak into the identity


# ---------------------------------------------------------------------------
# tune-then-replay through a real kernel
# ---------------------------------------------------------------------------


def _make_add_kernel(configs, **tuner_kwargs):
    @persistent_autotune(configs=configs, key=["N"], **tuner_kwargs)
    @triton.jit
    def add_one(X, Y, N, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        tl.store(Y + offs, tl.load(X + offs, mask=mask, other=0) + 1, mask=mask)

    return add_one


_CONFIGS = [triton.Config({"BLOCK": b}, num_warps=1) for b in (64, 128)]


@gpu
def test_first_run_tunes_and_persists(tmp_path):
    add = _make_add_kernel(_CONFIGS, kernel_name="add_one", store_dir=tmp_path)
    x = torch.arange(100, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    counts = []
    add._bench = lambda call: (counts.append(1), 2.0 - len(counts) * 0.5)[1]
    add[lambda meta: (1,)](x, y, 100)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1)
    assert len(counts) == len(_CONFIGS)  # every config benchmarked exactly once
    assert len(add.cache) == 1
    doc = json.loads(next(tmp_path.iterdir()).read_text())
    assert doc["schema"] == SCHEMA and doc["records"]


@gpu
def test_fresh_tuner_replays_store_without_benchmarking(tmp_path):
    # N=50 (not 100): the constant-bench stub makes configs[0] (BLOCK=64)
    # the deterministic winner, and the launch must cover N on its own —
    # with N=100 this test only passed when the unwritten tail of the
    # empty output buffer happened to hold x+1 from an earlier trial.
    add = _make_add_kernel(_CONFIGS, kernel_name="add_one", store_dir=tmp_path)
    x = torch.arange(50, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add._bench = lambda call: 1.0
    add[lambda meta: (1,)](x, y, 50)  # tunes + persists

    # A fresh tuner (fresh process analog): benchmarking raises — a store
    # hit must never bench.
    add2 = _make_add_kernel(_CONFIGS, kernel_name="add_one", store_dir=tmp_path)
    add2._bench = lambda call: pytest.fail("replayed tuner must not benchmark")
    y2 = torch.empty_like(x)
    add2[lambda meta: (1,)](x, y2, 50)
    torch.cuda.synchronize()
    torch.testing.assert_close(y2, x + 1)
    winner = add.cache[(50,)]
    assert add2.cache[(50,)].kwargs == winner.kwargs
    assert add2.cache[(50,)].num_warps == winner.num_warps


@gpu
def test_fingerprint_change_forces_retune(tmp_path):
    source = tmp_path / "k.py"
    source.write_text("v1")
    add = _make_add_kernel(_CONFIGS, kernel_name="add_one", store_dir=tmp_path,
                           source_files=[source])
    x = torch.arange(100, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add._bench = lambda call: 1.0
    add[lambda meta: (1,)](x, y, 100)

    source.write_text("v2")  # producer changed: next tuner must re-tune
    add2 = _make_add_kernel(_CONFIGS, kernel_name="add_one", store_dir=tmp_path,
                            source_files=[source])
    bench_calls = []
    add2._bench = lambda call: (bench_calls.append(1), 1.0)[1]
    y2 = torch.empty_like(x)
    add2[lambda meta: (1,)](x, y2, 100)
    assert bench_calls  # re-tuned, not replayed


@gpu
def test_off_switch_keeps_autotune_in_process_only(tmp_path, monkeypatch):
    monkeypatch.setenv("VKERNELS_TUNING_CACHE", "off")
    add = _make_add_kernel(_CONFIGS, kernel_name="add_one", store_dir=tmp_path)
    x = torch.arange(50, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    add._bench = lambda call: 1.0
    add[lambda meta: (1,)](x, y, 50)
    torch.testing.assert_close(y, x + 1)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    assert list(add.cache) == [(50,)]  # in-process choice still made


# ---------------------------------------------------------------------------
# mhc_projection integration (the exemplar migration)
# ---------------------------------------------------------------------------


@gpu
def test_mhc_projection_persists_and_replays(tmp_path, monkeypatch):
    # ``torch_ops/__init__`` re-exports the mhc_projection *function*, which
    # shadows the submodule attribute — import the module explicitly.
    mhc = import_module("vkernels.torch_ops.mhc_projection")

    monkeypatch.setenv("VKERNELS_TUNING_CACHE", str(tmp_path))
    mhc._kernels.cache_clear()
    mhc._WARMED.clear()
    x = torch.randn(2, 16384, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(24, 16384, device="cuda", dtype=torch.bfloat16)
    out = mhc.mhc_projection(x, w)
    ref = mhc.mhc_projection_reference(x, w)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    assert mhc.mhc_projection_tuning_metadata()["choices"]

    # Fresh-process analog: with benchmarking disabled, the stored config
    # must carry the call.
    mhc._kernels.cache_clear()
    mhc._WARMED.clear()
    monkeypatch.setattr(triton.testing, "do_bench",
                        lambda *a, **k: pytest.fail("replay must not benchmark"))
    out2 = mhc.mhc_projection(x, w)
    torch.testing.assert_close(out2, ref, rtol=2e-2, atol=2e-2)
