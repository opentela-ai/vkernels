"""Standard input cases for the public kernels (seeded, shape-parameterized).

Each generator is a pure function of ``(size, seed, **extra)`` returning
``(args, kwargs)`` for the corresponding :mod:`vkernels.kernels` function,
so verification and bisection run the *same* inputs through the active
backend and the pure-Python oracle. ``size`` is the single scalar the
bisect driver varies: the element count for elementwise/reduce ops and the
leading matrix dimension for ``gemm``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

_F32 = np.dtype(np.float32)

__all__ = ["CASES", "available_ops"]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _case_add(size: int, seed: int) -> tuple[tuple, dict]:
    rng = _rng(seed)
    return (
        rng.standard_normal(size).astype(_F32) * 8,
        rng.standard_normal(size).astype(_F32) * 8,
    ), {}


def _case_scale(size: int, seed: int) -> tuple[tuple, dict]:
    rng = _rng(seed)
    return (rng.standard_normal(size).astype(_F32) * 16,), {"alpha": -0.5}


def _case_relu(size: int, seed: int) -> tuple[tuple, dict]:
    rng = _rng(seed)
    return (rng.standard_normal(size).astype(_F32) * 32,), {}


def _case_sum(size: int, seed: int) -> tuple[tuple, dict]:
    rng = _rng(seed)
    return (rng.standard_normal(size).astype(_F32) * 4,), {}


def _case_max(size: int, seed: int) -> tuple[tuple, dict]:
    rng = _rng(seed)
    return (rng.standard_normal(size).astype(_F32) * 64,), {}


def _case_gemm(size: int, seed: int, n: int = 16, k: int = 16) -> tuple[tuple, dict]:
    rng = _rng(seed)
    return (
        rng.standard_normal((size, k)).astype(_F32) * 2,
        rng.standard_normal((k, n)).astype(_F32) * 2,
    ), {"alpha": 1.0, "beta": 0.0}


#: op name -> generator. Ops beyond the original six gain entries as their
#: compiled/fallback pairs stabilize; the CLI lists exactly these. gemm's
#: ``n``/``k`` default to 16 because the pure-Python oracle is a scalar
#: triple loop (~4M MAC/s) — raise them for representative shapes.
CASES: dict[str, Callable[..., tuple[tuple, dict]]] = {
    "add": _case_add,
    "scale": _case_scale,
    "relu": _case_relu,
    "sum": _case_sum,
    "max": _case_max,
    "gemm": _case_gemm,
}


def available_ops() -> tuple[str, ...]:
    return tuple(sorted(CASES))


def make_case(op: str, size: int, *, seed: int = 0, **extra: Any) -> tuple[tuple, dict]:
    """Generate ``(args, kwargs)`` for ``op`` at ``size`` (raises KeyError
    with the available ops on an unknown name)."""
    try:
        generator = CASES[op]
    except KeyError:
        raise KeyError(
            f"no numerics case generator for {op!r}; available: {', '.join(available_ops())}"
        ) from None
    return generator(size, seed, **extra)
