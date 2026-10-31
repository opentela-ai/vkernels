"""ctypes prototypes of the HIP C ABI are declared once, at library load.

Host-only: a stand-in library object replaces libvkernels_hip.so.
"""

from __future__ import annotations

import ctypes
import types

import pytest

pytest.importorskip("torch")

from vkernels import hip_dsa_mhc  # noqa: E402


class _Fn:
    restype = "unset"
    argtypes = "unset"


def test_declare_prototypes_sets_present_symbols_and_skips_missing():
    present = ["vk_hip_mhc_post_stream", "vk_hip_kda_chunked_scratch_floats"]
    lib = types.SimpleNamespace(**{name: _Fn() for name in present})
    hip_dsa_mhc._declare_prototypes(lib)
    post = lib.vk_hip_mhc_post_stream
    assert post.restype is ctypes.c_int
    assert post.argtypes == [ctypes.c_int] * 3 + [ctypes.c_void_p] * 6
    floats = lib.vk_hip_kda_chunked_scratch_floats
    assert floats.restype is ctypes.c_ulonglong
    assert floats.argtypes == [ctypes.c_int] * 4


def test_load_declares_prototypes_once(monkeypatch):
    lib = types.SimpleNamespace(vk_hip_mhc_post=_Fn())
    calls = []
    monkeypatch.setattr(hip_dsa_mhc, "_LIB", None)
    monkeypatch.setattr(hip_dsa_mhc, "_LIB_FAILED", False)
    monkeypatch.setattr(hip_dsa_mhc, "_declare_prototypes", lambda lb: calls.append(lb))
    import vkernels.vllm_experts as ve

    monkeypatch.setattr(ve, "load_libvkernels_hip", lambda: lib)
    assert hip_dsa_mhc._load() is lib
    assert hip_dsa_mhc._load() is lib
    assert calls == [lib]
