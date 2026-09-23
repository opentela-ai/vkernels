"""Tests for the M-bucketed tile-config sidecars (:mod:`vkernels.torch_ops.tile_configs`).

The lookup plumbing (nearest-M selection, env/dir resolution, cache,
manifest audit) is device-independent and runs anywhere. The kernel-level
tests — a real launch whose tiles come from a sidecar — require CUDA and
skip elsewhere.

Contract under test:

* no sidecar for the op → ``None`` (caller keeps built-in defaults);
* nearest-M bucket selection: exact hit, below-min, above-max, mid-band,
  ties to the smaller bucket;
* a malformed sidecar raises :class:`TileConfigError`; a stale or foreign
  manifest raises :class:`tuning_manifest.ManifestError` — never a silent
  downgrade;
* ``VK_TILE_CONFIGS_DIR`` selects the sidecar directory; edits are picked
  up by mtime without a process restart;
* the ``_tiles`` launcher precedence: ``VK_FP8GEMM_TILES`` > sidecar >
  built-in default.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")

from vkernels.torch_ops.tile_configs import (
    TileConfigError,
    build_tile_manifest,
    lookup_tiles,
    write_tile_manifest,
)


@pytest.fixture()
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VK_TILE_CONFIGS_DIR", str(tmp_path))
    return tmp_path


def _write(dir_path, op, buckets):
    path = dir_path / f"{op}.json"
    path.write_text(json.dumps({"op": op, "buckets": buckets}))
    return path


def test_no_config_returns_none(config_dir):
    assert lookup_tiles("glm_fp8_dense_fnuz", 123) is None


def test_exact_and_nearest_buckets(config_dir):
    _write(config_dir, "glm_fp8_dense_fnuz", {
        "16": [16, 128, 4, 3],
        "128": [64, 128, 4, 3],
        "4096": [128, 256, 8, 3],
    })
    assert lookup_tiles("glm_fp8_dense_fnuz", 128) == (64, 128, 4, 3)
    assert lookup_tiles("glm_fp8_dense_fnuz", 129) == (64, 128, 4, 3)    # nearest bucket: 128
    assert lookup_tiles("glm_fp8_dense_fnuz", 1) == (16, 128, 4, 3)      # below min → smallest
    assert lookup_tiles("glm_fp8_dense_fnuz", 100000) == (128, 256, 8, 3)
    assert lookup_tiles("glm_fp8_dense_fnuz", 60) == (16, 128, 4, 3)     # |60-16|=44 < |60-128|=68


def test_tie_picks_smaller_bucket(config_dir):
    _write(config_dir, "glm_fp8_dense_fnuz", {"8": [32, 128, 4, 2], "32": [64, 128, 4, 3]})
    # m=20: |20-8|=12 == |20-32|=12 → smaller bucket wins
    assert lookup_tiles("glm_fp8_dense_fnuz", 20) == (32, 128, 4, 2)


def test_malformed_sidecar_raises(config_dir):
    path = config_dir / "glm_fp8_dense_fnuz.json"
    path.write_text(json.dumps({"op": "glm_fp8_dense_fnuz"}))  # no buckets
    with pytest.raises(TileConfigError):
        lookup_tiles("glm_fp8_dense_fnuz", 64)
    path.write_text("{not json")
    with pytest.raises(Exception):
        lookup_tiles("glm_fp8_dense_fnuz", 64)


def test_bad_tile_tuple_raises(config_dir):
    _write(config_dir, "glm_fp8_dense_fnuz", {"16": [0, 128, 4, 3]})
    with pytest.raises(TileConfigError):
        lookup_tiles("glm_fp8_dense_fnuz", 16)


def test_manifest_audit_catches_edit(config_dir):
    path = _write(config_dir, "glm_fp8_dense_fnuz", {"16": [16, 128, 4, 3]})
    producer = config_dir / "producer_ref.py"
    producer.write_text("# reference source\n")
    manifest = build_tile_manifest(
        path, kernel="glm_fp8_blockwise_gemm", op="glm_fp8_dense_fnuz",
        producer_paths=[str(producer)],
    )
    write_tile_manifest(path, manifest)
    assert lookup_tiles("glm_fp8_dense_fnuz", 16) == (16, 128, 4, 3)
    # Edit the artifact after production → sha256 mismatch → loud failure.
    path.write_text(json.dumps({"op": "glm_fp8_dense_fnuz", "buckets": {"16": [64, 128, 4, 3]}}))
    with pytest.raises(Exception):
        lookup_tiles("glm_fp8_dense_fnuz", 16)


def test_mtime_edit_is_picked_up(config_dir):
    path = _write(config_dir, "glm_fp8_dense_fnuz", {"16": [16, 128, 4, 3]})
    assert lookup_tiles("glm_fp8_dense_fnuz", 16) == (16, 128, 4, 3)
    import os
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    path.write_text(json.dumps({"op": "glm_fp8_dense_fnuz", "buckets": {"16": [64, 128, 4, 3]}}))
    os.utime(path, ns=(2_000_000_000, 2_000_000_000))
    assert lookup_tiles("glm_fp8_dense_fnuz", 16) == (64, 128, 4, 3)


def test_launcher_precedence_env_over_sidecar_over_default(config_dir, monkeypatch):
    from vkernels.torch_ops import glm_fp8_blockwise_gemm as gemm
    _write(config_dir, "glm_fp8_dense_fnuz", {"16": [32, 64, 2, 2]})
    # Sidecar wins over the built-in default.
    assert gemm._tiles("glm_fp8_dense_fnuz", 16, (128, 128, 4, 3)) == (32, 64, 2, 2)
    # Env override beats the sidecar.
    monkeypatch.setenv("VK_FP8GEMM_TILES", "96,96,4,2")
    assert gemm._tiles("glm_fp8_dense_fnuz", 16, (128, 128, 4, 3)) == (96, 96, 4, 2)
    # No sidecar for this op → default.
    monkeypatch.delenv("VK_FP8GEMM_TILES")
    assert gemm._tiles("glm_fp8_dense_fn", 16, (128, 128, 8, 3)) == (128, 128, 8, 3)
