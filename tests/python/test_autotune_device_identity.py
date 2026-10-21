"""The persistent autotuner memoizes its per-device store identity.

``PersistentAutotuner.run`` sits on every launch of an autotuned Triton
kernel, so the device fingerprint (a device-property query) and its
canonical JSON are computed once per device index, not per call. The
identity string keeps its ``json.dumps(fingerprint, sort_keys=True)`` form:
callers (e.g. test_kda_solve_tuning_reproducibility) seed ``_memory_configs``
by it. CPU-safe: the fingerprint and current device are injected.
"""

from __future__ import annotations

import json
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from vkernels.torch_ops import tuning_cache  # noqa: E402


@pytest.fixture()
def fingerprints(monkeypatch):
    calls = []

    def fake_fingerprint(device=None):
        calls.append(device)
        return {"name": f"FAKE{device}", "arch": "sm999", "cu_count": 1, "software": {}}

    monkeypatch.setattr(tuning_cache, "device_fingerprint", fake_fingerprint)
    monkeypatch.setattr(tuning_cache, "_DEVICE_IDENTITIES", {})
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return calls


def test_identity_is_memoized_per_device_index(fingerprints):
    first = tuning_cache._device_identity(types.SimpleNamespace(index=1))
    again = tuning_cache._device_identity(types.SimpleNamespace(index=1))
    other = tuning_cache._device_identity(types.SimpleNamespace(index=2))
    assert first is again
    assert first[1] == json.dumps(first[0], sort_keys=True)
    assert other[0]["name"] == "FAKE2"
    assert fingerprints == [1, 2]


def test_unindexed_device_resolves_to_current(fingerprints):
    current = tuning_cache._device_identity(None)
    bare = tuning_cache._device_identity(types.SimpleNamespace(index=None))
    assert current is bare
    assert fingerprints == [0]


def test_run_fingerprints_once_across_launches(fingerprints, monkeypatch):
    monkeypatch.setenv("VKERNELS_TUNING_CACHE", "off")
    launches = []
    fn = types.SimpleNamespace(arg_names=["x", "N"], run=lambda *args, **kwargs: launches.append(kwargs))
    config = types.SimpleNamespace(kwargs={"BLOCK": 8}, num_warps=4, num_stages=2)
    tuner = tuning_cache.PersistentAutotuner(fn, [config], ["N"], kernel_name="identity_demo")
    tuner._bench = lambda call: 1.0
    x = types.SimpleNamespace(is_cuda=True, device=types.SimpleNamespace(index=3))
    for _ in range(5):
        tuner.run(x, 16, grid=(1,))
    assert fingerprints == [3]
    identity = json.dumps(tuning_cache._DEVICE_IDENTITIES[3][0], sort_keys=True)
    assert list(tuner._stores) == [identity]
    assert tuner._memory_configs[identity] == {(16,): config}
    assert len(launches) == 5 and launches[-1]["BLOCK"] == 8
