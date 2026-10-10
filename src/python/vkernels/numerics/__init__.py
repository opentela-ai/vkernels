"""Standalone kernel verification: tolerances, comparison, bisect, CLI.

The numerics layer generalizes the ad-hoc bisect scripts that used to live
in ``bench/`` into a module any kernel can use:

* :mod:`vkernels.numerics.tolerance` — dtype-aware comparison policy
  (exact for the documented bit-identical float32 oracle pairs, scaled
  tolerances for reduced precision).
* :mod:`vkernels.numerics.compare` — ``calc_diff`` / ``count_bytes`` /
  ``assert_bitwise_equal`` (ported from DeepGEMM's testing utils) plus
  first-mismatch-reporting ``assert_close``.
* :mod:`vkernels.numerics.generators` — standard seeded inputs for the
  public kernels, parameterized by one scalar ``size``.
* :mod:`vkernels.numerics.verify` — run the active backend against the
  pure-Python oracle and report without raising.
* :mod:`vkernels.numerics.bisect` — shrink a failing comparison to the
  smallest failing size.

Command line::

    python -m vkernels.numerics ops
    python -m vkernels.numerics verify add --size 4096
    python -m vkernels.numerics bisect gemm --lo 1 --hi 2048 --n 64 --k 64

Design borrowed from tokenspeed-kernel's ``numerics`` package (MIT,
LightSeek Foundation); the CPU-oracle authority stays with this repo's
two-implementation model.
"""

from __future__ import annotations

from vkernels.numerics.bisect import BisectReport, bisect_size
from vkernels.numerics.compare import (
    assert_bitwise_equal,
    assert_close,
    calc_diff,
    count_bytes,
    mismatch_stats,
)
from vkernels.numerics.generators import CASES, available_ops, make_case
from vkernels.numerics.tolerance import EXACT, tolerance_for
from vkernels.numerics.verify import VerificationResult, verify

__all__ = [
    "EXACT",
    "BisectReport",
    "CASES",
    "VerificationResult",
    "assert_bitwise_equal",
    "assert_close",
    "available_ops",
    "bisect_size",
    "calc_diff",
    "count_bytes",
    "make_case",
    "mismatch_stats",
    "tolerance_for",
    "verify",
]
