"""Sidecar manifests that make frozen tuning artifacts auditable (#67).

A PyTorch TunableOp CSV records *which* implementation won, but nothing about
the producer, the producing source, the device beyond one arch validator, or
the quality gates a promotion must pass. This module pairs every tuning CSV
with a sibling ``<csv-stem>.manifest.json`` carrying:

* the exact CSV it describes (name + sha256 — an edited CSV invalidates it),
* the CSV validator block (cross-checked on load),
* per-shape chosen algorithms, **including** ``Default`` winners (recording a
  choice is reproducibility; gating it is quality — issue #67 separates them),
* device (arch + CU count) and software versions,
* producer fingerprints (sha256 of the producing/consuming sources), so an
  artifact produced by different source bytes is rejected as stale,
* quality gates **declared before evaluation**, which the qualification
  harness enforces without a waiver path.

Validation never silently downgrades a mismatch into a warning: anything the
manifest claims and the environment contradicts raises ``ManifestError``.
Torch is imported lazily and is optional until an API needing it is called.
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "vk-tuning-manifest/1"

_REQUIRED_KEYS = ("schema", "kernel", "csv", "csv_sha256", "op", "shapes",
                  "algos", "validators", "device", "software", "producer",
                  "quality_gates", "created")


class ManifestError(ValueError):
    """A tuning manifest is missing, stale, or contradicts its artifact."""


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def fingerprint_files(paths):
    """sha256 each existing path; keys are the paths as given."""
    return {str(path): sha256_file(path) for path in paths}


def hip_version_code(hip_version):
    """Encode a HIP version string as PyTorch's ``HIP_VERSION`` validator.

    PyTorch emits ``major*100+minor`` of ``torch.version.hip`` (e.g.
    ``6.3.42134-a9a80e791`` -> ``603``). There is no ``ROCM_VERSION`` validator.
    """
    major, minor = str(hip_version).split(".")[:2]
    return int(major) * 100 + int(minor)


def collect_environment(device_index=0):
    """Device + software provenance, or ``None`` when CUDA is unavailable.

    The loader skips environment cross-checks when this returns ``None``
    (CPU-only hosts can still audit manifest/CSV consistency and source
    fingerprints; a GPU host always checks device and versions).
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "arch": getattr(properties, "gcnArchName", properties.name),
        "cu_count": properties.multi_processor_count,
        "torch": torch.__version__.split("+")[0],
        "hip": hip_version_code(torch.version.hip) if torch.version.hip else None,
    }


def read_csv_block(csv_path):
    """Return (validators dict, {(op, signature): winner}) from a TunableOp CSV."""
    with open(csv_path, newline="") as source:
        rows = list(csv.reader(source))
    validators = {row[1]: row[2] for row in rows if len(row) >= 3 and row[0] == "Validator"}
    results = {(row[0], row[1]): row[2] for row in rows
               if len(row) >= 4 and row[0] != "Validator"}
    return validators, results


def default_quality_gates():
    """Pre-declared QKV promotion thresholds. Recorded in every manifest.

    These are fixed *before* any evaluation runs; the harness has no waiver
    flag. ``model_nll_regression_max`` is the allowed mean-NLL increase of the
    candidate over the baseline on held-out prompts (None = the gate needs a
    floe-side measurement and fails closed until one is supplied).
    """
    return {
        "max_abs_vs_fp64_oracle": 0.008,
        "max_rel_vs_fp64_oracle": 0.008,
        "min_argmax_agreement": 0.95,
        "require_reference_repeat_identical": True,
        "model_nll_regression_max": 0.005,
    }


def build_manifest(csv_path, *, kernel, op, shapes, producer_paths,
                   quality_gates=None, job=None, notes=""):
    """Describe a finished TunableOp CSV. ``shapes`` maps signature -> dims."""
    csv_path = Path(csv_path)
    validators, results = read_csv_block(csv_path)
    algos = {signature: winner for (name, signature), winner in results.items() if name == op}
    return {
        "schema": SCHEMA,
        "kernel": kernel,
        "csv": csv_path.name,
        "csv_sha256": sha256_file(csv_path),
        "op": op,
        "shapes": shapes,
        "algos": algos,
        "validators": validators,
        "device": {}, "software": {},  # filled below when a GPU is present
        "producer": {"fingerprints": fingerprint_files(producer_paths)},
        "quality_gates": dict(default_quality_gates(), **(quality_gates or {})),
        "policy": {"default_winners": "recorded"},
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "job": str(job) if job else None,
        "notes": notes,
    }


def write_manifest(csv_path, manifest):
    path = Path(csv_path).with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def load_manifest(csv_path, manifest_path=None):
    path = Path(manifest_path) if manifest_path else \
        Path(csv_path).with_suffix(".manifest.json")
    if not path.is_file():
        raise ManifestError(
            f"missing tuning manifest {path} beside {csv_path}; produce one with "
            "tuning_manifest.build_manifest/write_manifest (or pass require_manifest=False "
            "to load an unmanifested artifact for interactive debugging only)")
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ManifestError(f"tuning manifest {path} is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ManifestError(f"tuning manifest {path} must be a JSON object")
    return manifest


def _reject_environment_mismatch(manifest, environment):
    live = collect_environment() if environment is None else environment
    if live is None:
        return  # CPU-only host: device/version cross-check is impossible here
    for key, label in (("arch", "device architecture"), ("cu_count", "CU count")):
        recorded = manifest.get("device", {}).get(key)
        if recorded is not None and recorded != live[key]:
            raise ManifestError(
                f"stale tuning artifact: produced on {label} {recorded!r}, "
                f"running on {live[key]!r}")
    for key, label in (("torch", "PyTorch"), ("hip", "HIP")):
        recorded = manifest.get("software", {}).get(key)
        if recorded is not None and recorded != live[key]:
            raise ManifestError(
                f"stale tuning artifact: produced with {label} {recorded!r}, running on {live[key]!r}")


def validate_manifest(manifest, *, csv_path=None, csv_rows=None, environment=None,
                      producer_paths=None):
    """Cross-check a manifest against its CSV, this host, and this source tree.

    ``csv_rows`` is the (validators, results) pair from :func:`read_csv_block`;
    passing it skips re-reading. ``producer_paths`` re-fingerprints the named
    sources and rejects the artifact when any byte changed since production.
    """
    for key in _REQUIRED_KEYS:
        if key not in manifest:
            raise ManifestError(f"tuning manifest is missing required key {key!r}")
    if manifest["schema"] != SCHEMA:
        raise ManifestError(
            f"unsupported tuning manifest schema {manifest['schema']!r} (expected {SCHEMA!r})")

    if csv_path is not None:
        recorded_name = manifest.get("csv")
        if recorded_name and recorded_name != Path(csv_path).name:
            raise ManifestError(
                f"tuning manifest describes CSV {recorded_name!r}, was pointed at "
                f"{Path(csv_path).name!r}")
        recorded_digest = manifest.get("csv_sha256")
        if recorded_digest and recorded_digest != sha256_file(csv_path):
            raise ManifestError(
                f"stale tuning artifact: CSV {Path(csv_path).name} changed since the "
                "manifest was written (sha256 mismatch)")

    if csv_rows is not None:
        validators, results = csv_rows
        recorded = manifest.get("validators") or {}
        missing = sorted(set(validators) - set(recorded))
        differing = sorted(key for key in set(validators) & set(recorded)
                           if validators[key] != recorded[key])
        if missing or differing:
            raise ManifestError(
                f"tuning manifest validator block disagrees with the CSV "
                f"(missing: {missing}, differing: {differing})")
        algos = manifest.get("algos") or {}
        expected = {signature: winner for (name, signature), winner in results.items()
                    if name == manifest.get("op")}
        if expected != algos:
            raise ManifestError(
                "tuning manifest algorithms disagree with the CSV rows "
                f"(manifest: {algos!r}, CSV: {expected!r})")

    _reject_environment_mismatch(manifest, environment)

    fingerprints = (manifest.get("producer") or {}).get("fingerprints") or {}
    if producer_paths is not None:
        fingerprints = dict(fingerprints)
        fingerprints.update(fingerprint_files(producer_paths))
    for path, digest in fingerprints.items():
        source = Path(path)
        if not source.is_file():
            raise ManifestError(
                f"cannot verify tuning artifact provenance: fingerprinted source {path!r} "
                "is absent from this checkout")
        actual = sha256_file(source)
        if actual != digest:
            raise ManifestError(
                f"stale tuning artifact: source {path} changed since the artifact was "
                f"produced ({digest} -> {actual})")
