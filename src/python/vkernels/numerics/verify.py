"""Run any registered kernel against its pure-Python oracle and report.

The two-implementation model gives every public kernel a fallback oracle
(:mod:`vkernels._fallback`). :func:`verify` generates a standard seeded
input, pushes it through the *active* backend
(:mod:`vkernels.kernels` — compiled extension when built, fallback
otherwise) and the oracle, and reports closeness without raising, so the
CLI and the bisect driver can consume the verdict programmatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vkernels.numerics.compare import assert_close, calc_diff, mismatch_stats
from vkernels.numerics.generators import make_case
from vkernels.numerics.tolerance import tolerance_for

__all__ = ["VerificationResult", "verify"]


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of one kernel-vs-oracle comparison."""

    op: str
    size: int
    backend: str
    ok: bool
    exact: bool
    stats: dict = field(default_factory=dict)
    message: str = ""

    def __str__(self) -> str:
        status = "ok" if self.ok else "FAILED"
        head = f"{self.op}[size={self.size}] backend={self.backend} {status}"
        if self.ok:
            return head
        return f"{head}: {self.message}"


def _run(op: str, args: tuple, kwargs: dict) -> Any:
    module = _kernel_module()
    return getattr(module, op)(*args, **kwargs)


def _oracle_add(args: tuple, kwargs: dict):
    a, b = args
    out = np.empty_like(a)
    _oracle_module().add(a, b, out)
    return out


def _oracle_scale(args: tuple, kwargs: dict):
    (x,) = args
    out = np.empty_like(x)
    _oracle_module().scale(x, kwargs.get("alpha", 1.0), out)
    return out


def _oracle_relu(args: tuple, kwargs: dict):
    (x,) = args
    out = np.empty_like(x)
    _oracle_module().relu(x, out)
    return out


def _oracle_sum(args: tuple, kwargs: dict):
    return _oracle_module().sum(*args)


def _oracle_max(args: tuple, kwargs: dict):
    return _oracle_module().max(*args)


def _oracle_gemm(args: tuple, kwargs: dict):
    a, b = args
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.float32)
    _oracle_module().gemm(
        m, n, k, kwargs.get("alpha", 1.0), a, b, kwargs.get("beta", 0.0), c
    )
    return c


#: The fallback oracle exposes the low-level in-place API (``add(a, b,
#: out)``); these adapters give it the public calling convention the
#: generators produce.
_ORACLE_ADAPTERS = {
    "add": _oracle_add,
    "scale": _oracle_scale,
    "relu": _oracle_relu,
    "sum": _oracle_sum,
    "max": _oracle_max,
    "gemm": _oracle_gemm,
}


def _kernel_module():
    from vkernels import kernels

    return kernels


def _oracle_module():
    from vkernels import _fallback

    return _fallback


def verify(
    op: str,
    size: int,
    *,
    seed: int = 0,
    exact: bool | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    **extra: Any,
) -> VerificationResult:
    """Compare the active backend's ``op`` against the fallback oracle.

    Args:
        op: Kernel name (see :data:`vkernels.numerics.generators.CASES`).
        size: Scalar shape parameter the generator varies.
        seed: Seed for the standard input generator.
        exact: Force bit-identical comparison. ``None`` (default) applies
            the dtype policy: exact for float32, scaled otherwise.
        atol, rtol: Override the tolerance policy for this call.
        **extra: Forwarded to the generator (e.g. ``n=``/``k=`` for gemm).

    The backend is whatever :func:`vkernels._backend.load_extension`
    resolves to; comparing fallback-vs-fallback still exercises the
    generator and comparison plumbing (and is what host CI does).
    """
    from vkernels import _backend

    args, kwargs = make_case(op, size, seed=seed, **extra)
    actual = _run(op, args, kwargs)
    try:
        adapter = _ORACLE_ADAPTERS[op]
    except KeyError:
        raise KeyError(
            f"no oracle adapter for {op!r}; available: {', '.join(sorted(_ORACLE_ADAPTERS))}"
        ) from None
    expected = adapter(args, kwargs)

    backend = _backend.backend_name()
    stats = mismatch_stats(actual, expected)
    if exact is None:
        exact = stats.get("dtype") == "float32"
    policy_atol, policy_rtol = tolerance_for(
        stats.get("dtype", "float32"), exact=bool(exact)
    )
    try:
        assert_close(
            actual,
            expected,
            atol=atol if atol is not None else policy_atol,
            rtol=rtol if rtol is not None else policy_rtol,
            label=f"{op}[size={size}]",
        )
    except AssertionError as exc:
        stats["calc_diff"] = calc_diff(actual, expected)
        return VerificationResult(op, size, backend, False, bool(exact), stats, str(exc))
    stats["calc_diff"] = calc_diff(actual, expected)
    return VerificationResult(op, size, backend, True, bool(exact), stats)
