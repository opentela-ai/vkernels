"""Opt-in, pre-tuned BLAS QKV for serialized inference only.

Configuration reads a PyTorch-version-validated TunableOp artifact eagerly.
Calls never tune or read files. TunableOp flags and its result table are
process-global: callers must serialize configuration/inference against all
other BLAS work in this process. Flag restoration does not isolate concurrent
threads, remove loaded results, or undo an already captured graph's choices.
Torch remains optional until an API is invoked.
"""

import csv
from contextlib import contextmanager

from .qkv_projection import _validate

_OP = "GemmTunableOp_BFloat16_TN"
_CONFIGURED = {}


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


def configure_qkv_tuned_blas(path):
    """Load an artifact before capture; reject missing/default QKV winners.

    PyTorch read_file performs hardware/library validator checks. Results are
    then checked against this exact file so a preexisting global result cannot
    silently stand in for the requested winner. Returns configured row counts.
    A failed configuration leaves this wrapper disabled (global results may
    still contain entries loaded by PyTorch).
    """
    import torch
    import torch.cuda.tunable as tunable

    global _CONFIGURED
    _CONFIGURED = {}
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
    expected = {
        tokens: row[2]
        for tokens in (1, 2) for row in rows
        if len(row) >= 3 and row[0] == _OP and row[1] == _signature(tokens)
        and row[2] and row[2] != "Default"
    }
    if 1 not in expected:
        raise ValueError("QKV tuning artifact requires a non-Default one-token shape entry")
    with _enabled_without_tuning(tunable):
        if not tunable.read_file(str(path)):
            raise RuntimeError("PyTorch rejected QKV tuning artifact validators")
        loaded = {(row[0], row[1]): row[2] for row in tunable.get_results()}
        if any(loaded.get((_OP, _signature(tokens))) != winner for tokens, winner in expected.items()):
            raise RuntimeError("loaded QKV tuning results do not match the requested artifact")
    _CONFIGURED = expected
    return tuple(sorted(expected))


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
