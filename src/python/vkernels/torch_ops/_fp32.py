"""Canonical torch fp32-matmul precision pin (shared numerics invariant).

NGC/NVIDIA containers default torch matmuls to tf32
(``fp32_precision='tf32'``, ``allow_tf32=True``); stock torch defaults to
ieee (true fp32). Any eager ``*_reference`` oracle whose parity bar is
tighter than tf32's 10-bit mantissa must therefore pin the matmul
precision for the duration of the call and restore it afterwards —
otherwise the container silently degrades the *oracle* while the fused
kernel (which pins its own Triton dots via ``input_precision="ieee"``)
is the accurate side. This was round-7 lane C/I's GH200 "final-state
drift" root cause: a 7.3e-4 delta attributed to the kernel that was
actually the tf32-degraded eager oracle. One canonical pin, one place to
update when torch renames the knob again.

Torch loads lazily (importing :mod:`vkernels.torch_ops` loads no torch).
"""

__all__ = ["pin_fp32_matmul"]


def pin_fp32_matmul(mode="ieee"):
    """Pin torch CUDA matmuls; returns a callable restoring the prior state.

    ``mode`` is ``"ieee"`` (true fp32 — the oracle default) or ``"tf32"``.
    Both knobs are pinned: the legacy ``allow_tf32`` flag and, on builds
    that have it, ``fp32_precision`` (newer torch's source of truth, which
    containers may set directly). Call the returned restore in a
    ``finally``; deliberately keeping it is a mode switch (the
    ``bench/kda_state_bisect.py`` oracle A/B does that).
    """
    import torch

    if mode not in ("ieee", "tf32"):
        raise ValueError(f"mode must be 'ieee' or 'tf32'; got {mode!r}")
    matmul = torch.backends.cuda.matmul
    prev = (matmul.allow_tf32, getattr(matmul, "fp32_precision", None))
    matmul.allow_tf32 = mode == "tf32"
    if prev[1] is not None:
        matmul.fp32_precision = mode

    def _restore():
        matmul.allow_tf32 = prev[0]
        if prev[1] is not None:
            matmul.fp32_precision = prev[1]

    return _restore
