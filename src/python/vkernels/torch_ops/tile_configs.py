"""M-bucketed triton tile-config sidecars (round-8 M3, mechanism only).

SGLang ships per-(E,N,device,dtype) fused-MoE JSON configs keyed by M
buckets with a nearest-M lookup (``fused_moe_triton_config.get_moe_configs``).
vkernels' fp8 triton launchers instead take ONE global tile tuple from
``VK_FP8GEMM_TILES`` — every launch shape pays the same tiles the offline
sweep happened to pick for one shape (kernels-infra.md item 3).

This module adds the lookup mechanism:

* a JSON sidecar per op (``<op>.json``) carrying an ``M -> (BM, BN, warps,
  stages)`` bucket table,
* SGLang's nearest-M semantics (closest bucket by absolute distance, ties
  to the smaller bucket),
* defaults preserved on any miss (absent file → caller keeps its defaults;
  ``VK_FP8GEMM_TILES`` still overrides everything for live experiments),
* artifacts audited through the tuning-manifest contract: when a sibling
  ``<op>.manifest.json`` exists it is validated like a TunableOp artifact
  (schema, sha256 of the tile JSON, device/software match, producer
  fingerprints resolved against the vkernels tree). A stale or foreign
  artifact raises :class:`tuning_manifest.ManifestError` — never a silent
  downgrade. A *missing* config is not an error: it means "use defaults".

Producer sidecar layout::

    {"op": "glm_fp8_dense_fnuz",
     "buckets": {"1": [16, 128, 4, 3], "2048": [128, 128, 4, 3]},
     "notes": "..."}

The offline sweep that produces the buckets is GPU work (rig job); this
module only makes an artifact loadable, auditable, and M-addressable.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .tuning_manifest import (
    ManifestError,
    SCHEMA,
    collect_environment,
    default_quality_gates,
    fingerprint_files,
    load_manifest,
    sha256_file,
    validate_manifest,
)

__all__ = ["TileConfigError", "build_tile_manifest", "lookup_tiles", "write_tile_manifest"]


class TileConfigError(ValueError):
    """A tile-config sidecar is present but malformed."""


def _vk_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_manifest_fingerprints(manifest: dict) -> list[str]:
    """Producer fingerprints recorded relative to the vkernels tree root."""
    fingerprints = (manifest.get("producer") or {}).get("fingerprints") or {}
    root = _vk_root()
    resolved = []
    for path in fingerprints:
        candidate = Path(path)
        if not candidate.is_absolute() and not candidate.is_file():
            candidate = root / path
        resolved.append(str(candidate))
    return resolved


def _config_path(op: str) -> Path | None:
    """Env dir wins; else the packaged ``configs/`` dir beside this module."""
    env_dir = os.environ.get("VK_TILE_CONFIGS_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir) / f"{op}.json"
        return candidate if candidate.is_file() else None
    candidate = Path(__file__).resolve().parent / "configs" / f"{op}.json"
    return candidate if candidate.is_file() else None


def _load_json(path: Path):
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _load_json._cache:
        try:
            _load_json._cache[key] = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(f"tile config {path} is unreadable: {error}") from error
        for stale in [k for k in _load_json._cache if k[0] == str(path) and k != key]:
            _load_json._cache.pop(stale)
    return _load_json._cache[key]


_load_json._cache = {}


def _parse_tiles(entry) -> tuple[int, int, int, int]:
    try:
        bm, bn, warps, stages = (int(x) for x in entry)
    except (TypeError, ValueError) as error:
        raise TileConfigError(f"tile entry {entry!r} is not (BM, BN, warps, stages)") from error
    if bm <= 0 or bn <= 0 or warps <= 0 or stages <= 0:
        raise TileConfigError(f"tile entry {entry!r} has a non-positive component")
    return (bm, bn, warps, stages)


def _nearest_bucket(buckets: dict, m: int):
    """SGLang semantics: closest bucket by |bucket - m|, ties to the smaller."""
    ordered = sorted(int(k) for k in buckets)
    if not ordered:
        return None
    return min(ordered, key=lambda b: (abs(b - int(m)), b))


def lookup_tiles(op: str, m: int, *, config_path=None) -> tuple[int, int, int, int] | None:
    """Return ``(BM, BN, warps, stages)`` for ``m`` rows, or ``None``.

    ``None`` = no sidecar for this op → the caller keeps its built-in
    defaults (or its ``VK_FP8GEMM_TILES`` override). A present-but-invalid
    sidecar or a stale manifest raises — an audited artifact must never
    silently downgrade.
    """
    path = Path(config_path) if config_path is not None else _config_path(op)
    if path is None:
        return None
    data = _load_json(path)
    if not isinstance(data, dict):
        raise TileConfigError(f"tile config {path} must be a JSON object")
    buckets = data.get("buckets")
    if not isinstance(buckets, dict) or not buckets:
        raise TileConfigError(f"tile config {path} has no usable 'buckets' table")

    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        manifest = load_manifest(path, manifest_path)
        validate_manifest(
            manifest,
            csv_path=path,
            environment=None,
            producer_paths=_resolve_manifest_fingerprints(manifest),
        )

    bucket = _nearest_bucket(buckets, m)
    if bucket is None:
        return None
    return _parse_tiles(buckets[str(bucket)] if str(bucket) in buckets else buckets[bucket])


def build_tile_manifest(config_path, *, kernel: str, op: str, producer_paths) -> dict:
    """Describe a finished tile-config sidecar (tuning-manifest shape)."""
    config_path = Path(config_path)
    data = json.loads(config_path.read_text())
    buckets = data.get("buckets") or {}
    return {
        "schema": SCHEMA,
        "kernel": kernel,
        "csv": config_path.name,
        "csv_sha256": sha256_file(config_path),
        "op": op,
        "shapes": {str(k): {"M": int(k)} for k in buckets},
        "algos": {str(k): list(v) for k, v in buckets.items()},
        "validators": {},
        "device": {},
        "software": {},
        "producer": {"fingerprints": fingerprint_files(producer_paths)},
        "quality_gates": dict(default_quality_gates()),
        "policy": {"default_winners": "recorded"},
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job": None,
        "notes": "tile-config sidecar (round-8 M3)",
    }


def write_tile_manifest(config_path, manifest: dict) -> Path:
    path = Path(config_path).with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
