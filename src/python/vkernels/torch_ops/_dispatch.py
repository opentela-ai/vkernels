"""The torch_ops calling convention: ops validate themselves.

Every public op in :mod:`vkernels.torch_ops` owns its full eligibility
check — device/arch, dtypes, shapes, contiguity, backend availability —
so callers never pre-gate on hardware or layout. The convention:

* **Contract miss -> ``OpNotEligible``.** The op cannot take the fused
  path for this input (CPU tensor, unexpected dtype, shape outside the
  validated envelope, missing Triton, ...). This is an *expected*,
  routine outcome: callers catch it and fall back to their eager torch
  path (or the op's ``*_reference`` oracle). ``OpNotEligible`` subclasses
  both :class:`ValueError` and :class:`TypeError`, so older call sites and
  tests that pinned either historical flavor keep working unchanged.
* **Anything else is a caller bug** (``TypeError`` for structurally wrong
  arguments that no fallback can fix, ``RuntimeError`` for operational
  failures such as an unconfigured tuned backend) and propagates.

This is what keeps model-side call sites to one line — ``if knob: return
op(...)`` — with the hardware/backend knowledge living here, next to the
kernels it describes (the deepseek_v41 arch already calls its ops exactly
this way).
"""

__all__ = ["OpNotEligible", "require", "same_gpu_contiguous"]


class OpNotEligible(ValueError, TypeError):
    """The fused kernel cannot run these inputs; the caller should fall back.

    Raised by the op's own validation for every routine miss — device/arch
    (CPU tensor on a CUDA kernel, HIP vs NVIDIA fp8 flavours), dtype,
    shape/contiguity outside the validated envelope, or a missing optional
    backend. Dual-inherits :class:`ValueError` and :class:`TypeError`
    because the contract historically escaped ops as either flavor and
    existing ``except`` clauses / tests pin both — every such site keeps
    working unchanged, while new call sites catch this exact class.
    """


def require(condition: bool, message: str) -> None:
    """Raise :class:`OpNotEligible` with ``message`` unless ``condition``."""
    if not condition:
        raise OpNotEligible(message)


def same_gpu_contiguous(*tensors) -> None:
    """Shared eligibility floor: CUDA-resident, one device, contiguous.

    The most common per-op validation prefix — reuse it instead of
    hand-rolling the loop in each launcher.
    """
    require(
        tensors[0].is_cuda,
        "inputs must be CUDA-resident on the same GPU "
        "(CPU callers take the *_reference/eager path)",
    )
    require(all(t.device == tensors[0].device for t in tensors),
            "inputs must share one GPU device")
    require(all(t.is_contiguous() for t in tensors),
            "inputs must be contiguous on the same GPU")
