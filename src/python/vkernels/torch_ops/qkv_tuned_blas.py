"""Opt-in, pre-tuned BLAS QKV for serialized inference only.

Configuration reads a PyTorch-version-validated TunableOp artifact plus its
sidecar manifest (see ``tuning_manifest``) eagerly. Calls never tune or read
files. The manifest binds the CSV to the producing source, device, software,
and the exact recorded winners — including ``Default``: reproducibility means
re-running the autotuner's choice, while quality is a separate, pre-declared
gate (``accept_default=False`` refuses Default winners for a promotion that
requires a tuned implementation). Stale or mismatched artifacts are rejected,
never downgraded to warnings.

TunableOp flags and its result table are process-global: configuration is
serialized here by a lock, but calls cannot isolate concurrent threads, remove
loaded results, or undo an already captured graph's choices. Serving rule:
configure once at startup, single-threaded, before any capture, and serialize
forward execution against all other BLAS work in this process. Torch remains
optional until an API is invoked.
"""

import csv
import threading
from contextlib import contextmanager

from .qkv_projection import _validate
from . import tuning_manifest

_OP = "GemmTunableOp_BFloat16_TN"
_CONFIGURED = {}
_CONFIGURED_META = {}
_CONFIGURE_LOCK = threading.Lock()


def _signature(tokens):
    return f"tn_8192_{tokens}_4096_ld_4096_4096_8192"


@contextmanager
def _enabled_without_tuning(tunable):
    enabled, tuning = tunable.is_enabled(), tunable.tuning_is_enabled()
    try:
        tunable.tuning_enable(False)
        tunable.enable(True)
        if not tunable.is_enabled() or tunable.tuning_is_enabled():
            raise RuntimeError("TunableOp environment overrides prevent frozen QKV execution")
        yield
    finally:
        try:
            tunable.enable(enabled)
        finally:
            tunable.tuning_enable(tuning)


def configure_qkv_tuned_blas(path, *, manifest=None, accept_default=True,
                             require_manifest=True):
    """Load an artifact (CSV + manifest) before capture; record the winners.

    PyTorch read_file performs hardware/library validator checks; the manifest
    adds producer-source, device, software, and CSV-integrity checks that
    reject stale artifacts. Winners are recorded exactly as the autotuner
    chose them — a ``Default`` winner is a legitimate reproducible choice when
    ``accept_default`` is true, and a rejected one when a promotion demands a
    tuned implementation. Results are checked against this exact file so a
    preexisting global result cannot silently stand in for the requested
    winner. Returns the configured row counts. A failed configuration leaves
    this wrapper disabled (global results may still contain entries loaded by
    PyTorch).
    """
    import torch
    import torch.cuda.tunable as tunable

    global _CONFIGURED, _CONFIGURED_META
    with _CONFIGURE_LOCK:
        _CONFIGURED, _CONFIGURED_META = {}, {}
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("configure qkv_tuned_blas eagerly before capture")
        with open(path, newline="") as source:
            rows = list(csv.reader(source))
        validators = {row[1] for row in rows if len(row) >= 3 and row[0] == "Validator"}
        # PyTorch's ``torch.cuda.tunable.write_file`` emits exactly these validators
        # on an AMD/HIP target (HIP_VERSION encodes the ROCm stack as
        # ``major*100+minor``; there is no ``ROCM_VERSION`` validator). Requiring
        # ``ROCM_VERSION`` here rejected every genuine artifact (job 628932,
        # section 5); ``read_file`` below still enforces version equality.
        required = {"PT_VERSION", "HIP_VERSION", "HIPBLASLT_VERSION", "GCN_ARCH_NAME", "ROCBLAS_VERSION"}
        if not required.issubset(validators):
            raise ValueError("QKV tuning artifact must include PyTorch/HIP/library/device validators")

        loaded_manifest = manifest if manifest is not None else (
            tuning_manifest.load_manifest(path) if require_manifest else None)
        csv_rows = tuning_manifest.read_csv_block(path)
        if loaded_manifest is not None:
            tuning_manifest.validate_manifest(
                loaded_manifest, csv_path=path, csv_rows=csv_rows)

        winners = {
            tokens: row[2]
            for tokens in (1, 2) for row in rows
            if len(row) >= 3 and row[0] == _OP and row[1] == _signature(tokens) and row[2]
        }
        if 1 not in winners:
            raise ValueError("QKV tuning artifact requires a one-token shape entry")
        defaults = sorted(tokens for tokens, winner in winners.items() if winner == "Default")
        if defaults and not accept_default:
            raise ValueError(
                f"QKV tuning artifact records Default winners for token counts {defaults}; "
                "a promotion requiring tuned implementations must reject them "
                "(reproducibility records the choice, quality gates it — issue #67)")
        with _enabled_without_tuning(tunable):
            if not tunable.read_file(str(path)):
                raise RuntimeError("PyTorch rejected QKV tuning artifact validators")
            loaded = {(row[0], row[1]): row[2] for row in tunable.get_results()}
            if any(loaded.get((_OP, _signature(tokens))) != winner for tokens, winner in winners.items()):
                raise RuntimeError("loaded QKV tuning results do not match the requested artifact")
        _CONFIGURED = dict(winners)
        _CONFIGURED_META = {
            "csv": str(path),
            "winners": dict(winners),
            "manifest": None if loaded_manifest is None else {
                "schema": loaded_manifest.get("schema"),
                "kernel": loaded_manifest.get("kernel"),
                "created": loaded_manifest.get("created"),
                "job": loaded_manifest.get("job"),
                "device": loaded_manifest.get("device"),
                "software": loaded_manifest.get("software"),
                "quality_gates": loaded_manifest.get("quality_gates"),
            },
        }
        return tuple(sorted(winners))


def qkv_tuned_blas_state():
    """JSON-safe configuration provenance for benchmark/deployment reports."""
    return {
        "configured": dict(_CONFIGURED),
        "manifest": _CONFIGURED_META.get("manifest"),
        "csv": _CONFIGURED_META.get("csv"),
        "concurrency": (
            "TunableOp flags and results are process-global; configuration is "
            "lock-serialized but forwards must be serialized against all other "
            "BLAS work. Configure once at startup, single-threaded, before any "
            "graph capture; captured graphs keep their loaded choices."),
    }


def qkv_tuned_blas(x, q_weight, k_weight, v_weight):
    """Compute BF16 Q/K/V with preloaded winners, restoring global flags."""
    import torch
    import torch.cuda.tunable as tunable

    weights = (q_weight, k_weight, v_weight)
    tokens = _validate(x, weights, gpu=True)
    if tokens not in _CONFIGURED:
        raise RuntimeError("configure qkv_tuned_blas eagerly for this row count before use")
    with torch.cuda.device(x.device), _enabled_without_tuning(tunable):
        return torch.cat([torch.nn.functional.linear(x, weight) for weight in weights], dim=-1)
