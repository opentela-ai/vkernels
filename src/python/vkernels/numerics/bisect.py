"""Bisect a failing kernel-vs-oracle verification down to the smallest case.

Generalizes the ad-hoc bisect scripts in ``bench/`` (kda_fault_bisect.py
and friends) into the numerics layer: when a verification fails at one
size, the interesting repro is the *smallest* size that still fails.
:class:`bisect_size` binary-searches the generator's scalar ``size``
parameter under the invariant that ``verify(op, hi)`` fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vkernels.numerics.verify import VerificationResult, verify

__all__ = ["BisectReport", "bisect_size"]


@dataclass(frozen=True)
class BisectReport:
    """Result of a bisection over the size parameter."""

    op: str
    lo: int
    hi: int
    smallest_failing: int | None
    trials: tuple[VerificationResult, ...] = field(default_factory=tuple)

    def __str__(self) -> str:
        if self.smallest_failing is None:
            return (
                f"{self.op}: no failure in ({self.lo}, {self.hi}] "
                f"({len(self.trials)} trials)"
            )
        failing = next(t for t in self.trials if t.size == self.smallest_failing and not t.ok)
        return (
            f"{self.op}: smallest failing size={self.smallest_failing} "
            f"in ({self.lo}, {self.hi}] after {len(self.trials)} trials\n"
            f"  {failing.message}"
        )


def bisect_size(
    op: str,
    lo: int,
    hi: int,
    *,
    seed: int = 0,
    exact: bool | None = None,
    **extra: Any,
) -> BisectReport:
    """Find the smallest ``size`` in ``(lo, hi]`` where ``verify`` fails.

    Runs ``verify(op, hi)`` first: a passing apex short-circuits with
    ``smallest_failing=None``. Then binary-searches with the invariant
    ``fails(hi_known_bad)`` / ``passes(lo)`` — callers therefore pass a
    ``lo`` that is known (or assumed) good, e.g. ``1`` or the largest size
    covered by the unit tests. ``exact``/``seed``/``**extra`` forward to
    :func:`vkernels.numerics.verify.verify`.
    """
    if lo < 1:
        raise ValueError(f"lo must be >= 1, got {lo}")
    if hi <= lo:
        raise ValueError(f"hi must be > lo, got {hi}")

    trials: list[VerificationResult] = []

    apex = verify(op, hi, seed=seed, exact=exact, **extra)
    trials.append(apex)
    if apex.ok:
        return BisectReport(op, lo, hi, None, tuple(trials))

    good, bad = lo, hi
    while bad - good > 1:
        mid = (good + bad) // 2
        result = verify(op, mid, seed=seed, exact=exact, **extra)
        trials.append(result)
        if result.ok:
            good = mid
        else:
            bad = mid
    return BisectReport(op, lo, hi, bad, tuple(trials))
