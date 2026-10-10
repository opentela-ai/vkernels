"""Dtype-aware comparison tolerances for kernel-vs-oracle verification.

The two-implementation model (``docs/README.md``) makes the CPU reference
the correctness oracle. Compiled-vs-fallback float32 kernels are documented
bit-identical, so the default policy for full precision is *exact*; reduced
precision and GPU paths get scaled tolerances. Policies are keyed by dtype
name so they apply equally to numpy dtypes and registry metadata strings.
"""

from __future__ import annotations

import numpy as np

__all__ = ["tolerance_for", "EXACT"]

#: Sentinel for bit-identical comparison (atol == rtol == 0).
EXACT = (0.0, 0.0)

# atol / rtol per dtype name. Values follow the conventions used across
# tests/python (e.g. the moe_combine bf16 replay check at 2e-2) and
# DeepGEMM's testing practice of loose-tolerance comparisons for
# reduced-precision tensor-core paths.
_TOLERANCES: dict[str, tuple[float, float]] = {
    "float64": (1e-12, 1e-12),
    "float32": EXACT,  # documented bit-identical against the oracle
    "float16": (1e-3, 1e-3),
    "bfloat16": (2e-2, 2e-2),
    "complex64": (1e-5, 1e-5),
    "complex128": (1e-12, 1e-12),
}


def tolerance_for(
    dtype: "str | np.dtype | type",
    *,
    exact: bool = False,
) -> tuple[float, float]:
    """Return ``(atol, rtol)`` for comparing tensors of ``dtype``.

    Integer and boolean dtypes compare bit-exactly regardless of ``exact``.
    ``exact=True`` forces ``(0, 0)`` for any dtype; ``exact=None``-style
    tri-states are deliberately not offered — callers that know their path
    is bit-identical say so.
    """
    if exact:
        return EXACT
    name = np.dtype(dtype).name if not isinstance(dtype, str) else dtype
    if name.startswith(("int", "uint", "bool")):
        return EXACT
    return _TOLERANCES.get(name, (1e-5, 1e-5))
