"""Comparison utilities for kernel verification.

``calc_diff``, ``count_bytes`` and ``assert_bitwise_equal`` are ports of
``deep_gemm/testing/numeric.py`` (DeepGEMM, MIT, Copyright (c) 2025
DeepSeek) from torch to numpy; ``mismatch_stats`` and ``assert_close``
extend them with the first-mismatch reporting this repo's oracle contracts
need. Kept numpy-only so the whole numerics layer works without a GPU
runtime.
"""

from __future__ import annotations

import numpy as np

from vkernels.numerics.tolerance import tolerance_for

__all__ = [
    "calc_diff",
    "count_bytes",
    "assert_bitwise_equal",
    "assert_close",
    "mismatch_stats",
]


def calc_diff(x: np.ndarray, y: np.ndarray) -> float:
    """Relative difference via cosine similarity (DeepGEMM's ``calc_diff``).

    ``1 - 2<x,y> / (||x||^2 + ||y||^2)`` in float64: 0.0 for identical
    inputs, ~1.0 for uncorrelated ones, and exactly 0.0 when both inputs
    are all zeros. Suitable for a quick "did the kernel blow up" check that
    does not over-punish large-magnitude outputs.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    denominator = (x * x + y * y).sum()
    if denominator == 0:  # all elements of both x and y are zero
        return 0.0
    return float(1.0 - 2.0 * (x * y).sum() / denominator)


def count_bytes(*values) -> int:
    """Total bytes of the arrays (or nested tuples/lists thereof)."""
    total = 0
    for value in values:
        if isinstance(value, (tuple, list)):
            total += count_bytes(*value)
        elif value is not None:
            array = np.asarray(value)
            total += array.size * array.itemsize
    return total


def _first_mismatch(
    actual: np.ndarray, expected: np.ndarray
) -> tuple[int, tuple[int, ...], float, float] | None:
    """Index, coordinates and values of the first mismatching element."""
    flat_a = actual.reshape(-1)
    flat_e = expected.reshape(-1)
    if flat_a.size != flat_e.size:
        return None
    with np.errstate(invalid="ignore"):
        differs = (flat_a != flat_e) & ~(np.isnan(flat_a) & np.isnan(flat_e))
    hits = np.flatnonzero(differs)
    if hits.size == 0:
        return None
    index = int(hits[0])
    coords = tuple(int(c) for c in np.unravel_index(index, actual.shape))
    return index, coords, float(flat_a[index]), float(flat_e[index])


def _bad_mask(a64: np.ndarray, e64: np.ndarray, atol: float, rtol: float) -> np.ndarray:
    """Elements outside tolerance; NaN pairs count as equal (like BLAS)."""
    with np.errstate(invalid="ignore"):
        return (np.abs(a64 - e64) > atol + rtol * np.abs(e64)) | (
            np.isnan(a64) != np.isnan(e64)
        )


def mismatch_stats(actual, expected) -> dict:
    """Summary of how two numeric results differ (never raises on shape)."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    stats: dict = {"shape": actual.shape, "dtype": str(actual.dtype)}
    if actual.shape != expected.shape:
        stats.update(
            ok=False,
            reason=f"shape mismatch: {actual.shape} vs {expected.shape}",
            calc_diff=float("nan"),
        )
        return stats
    a64, e64 = actual.astype(np.float64), expected.astype(np.float64)
    abs_err = np.abs(a64 - e64)
    with np.errstate(invalid="ignore"):
        nan_mismatch = np.isnan(a64) != np.isnan(e64)
    denom = np.maximum(np.abs(e64), np.finfo(np.float64).tiny)
    first = _first_mismatch(actual, expected)
    finite = ~nan_mismatch
    stats.update(
        max_abs=float(abs_err[finite].max()) if finite.any() else 0.0,
        max_rel=float((abs_err[finite] / denom[finite]).max()) if finite.any() else 0.0,
        num_nan_mismatch=int(nan_mismatch.sum()),
        calc_diff=calc_diff(actual, expected),
        num_elements=int(actual.size),
        first_mismatch=None
        if first is None
        else {
            "flat_index": first[0],
            "coords": first[1],
            "actual": first[2],
            "expected": first[3],
        },
    )
    return stats


def assert_bitwise_equal(x: np.ndarray, y: np.ndarray, label: str = "") -> None:
    """Assert identical bytes, reporting the first mismatch (DeepGEMM port)."""
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        raise AssertionError(
            f"bitwise mismatch{f' ({label})' if label else ''}: "
            f"{x.dtype}{x.shape} vs {y.dtype}{y.shape}"
        )
    xb = np.ascontiguousarray(x).view(np.uint8) if x.dtype != np.uint8 else x
    yb = np.ascontiguousarray(y).view(np.uint8) if y.dtype != np.uint8 else y
    if np.array_equal(xb, yb):
        return
    first = _first_mismatch(xb, yb)
    assert first is not None
    index, _byte_coords, xb_val, yb_val = first
    elem = index // x.itemsize
    coords = tuple(int(c) for c in np.unravel_index(elem, x.shape))
    raise AssertionError(
        f"bitwise mismatch{f' ({label})' if label else ''}: "
        f"first_byte={index}, element={elem}, coord={coords}, "
        f"byte_in_elem={index % x.itemsize}, "
        f"x_byte={xb_val}, y_byte={yb_val}, "
        f"x_val={x.reshape(-1)[elem]}, y_val={y.reshape(-1)[elem]}"
    )


def assert_close(
    actual,
    expected,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    dtype=None,
    exact: bool = False,
    label: str = "",
) -> None:
    """Assert closeness under the dtype-aware tolerance policy.

    ``atol``/``rtol`` override the policy for one call; ``dtype`` defaults
    to the array dtype. The failure message names the worst element, not
    just the bound that was crossed.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"shape mismatch{f' ({label})' if label else ''}: "
            f"{actual.shape} vs {expected.shape}"
        )
    default_atol, default_rtol = tolerance_for(
        dtype if dtype is not None else actual.dtype, exact=exact
    )
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol
    a64, e64 = actual.astype(np.float64), expected.astype(np.float64)
    bad = _bad_mask(a64, e64, atol, rtol)
    if not bad.any():
        return
    index = int(np.flatnonzero(bad.reshape(-1))[0])
    coords = tuple(int(c) for c in np.unravel_index(index, actual.shape))
    raise AssertionError(
        f"not close{f' ({label})' if label else ''}: {int(bad.sum())} of "
        f"{bad.size} elements exceed atol={atol}, rtol={rtol}; first at "
        f"{coords}: actual={a64.reshape(-1)[index]}, "
        f"expected={e64.reshape(-1)[index]}"
    )
