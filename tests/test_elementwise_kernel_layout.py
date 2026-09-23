"""Layout contract for elementwise._kernels() — guards against tuple-position drift.

The GLM glue lane (lvr4-glue) appended ``_swiglu_limit`` to the ``_kernels()``
tuple; two launch sites still used tail unpacks (``*_, unw, gated = ...``),
which silently rebind on ANY append and only detonate on a CUDA rig
(triton launch TypeError). Local gates never caught it because the triton
path is CPU-untestable. These tests are import-only (no GPU) and pin the
tuple by index -> kernel identity, so reordering/appending without auditing
all call sites fails here first.

Contract: indices 0-7 are the ORIGINAL elementwise kernels in their historical
order; index 8 is ``_swiglu_limit`` (append-only extension). New kernels must
be APPENDED and every consumer must use index access, never tail unpacks.
"""

import pytest

vk = pytest.importorskip("vkernels.torch_ops.elementwise")


def _names():
    ks = vk._kernels()
    return tuple(getattr(k, "fn", None).__name__ if hasattr(k, "fn") else type(k).__name__ for k in ks)


EXPECTED = (
    "rms_norm",          # [0]  norm
    "qk_norm",           # [1]  qk_n
    "rope",              # [2]  rope
    "silu_mul",          # [3]  sm (classic silu*mul)
    "store_kv",          # [4]  sk
    "qk_norm_rope",      # [5]  qk_rope
    "rms_norm_unweighted",  # [6]  unw  (rms_norm_unweighted launch site)
    "gated_norm",        # [7]  gated (gated-norm launch site)
    "swiglu_limit",      # [8]  lvr4-glue append
)


def test_kernels_tuple_matches_pinned_layout():
    names = _names()
    assert len(names) == len(EXPECTED), (
        f"_kernels() grew/shrank ({len(names)} != {len(EXPECTED)}): if you added a "
        "kernel, APPEND it and audit every consumer — tail unpacks silently rebind. "
        f"got={names}"
    )
    for i, (got, want) in enumerate(zip(names, EXPECTED)):
        assert got == want, f"_kernels()[{i}] is {got!r}, contract says {want!r} (got all={names})"


def test_no_tail_unpacks_of_kernels_tuple():
    """Grep-guard: elementwise.py must not unpack _kernels() with star patterns."""
    import inspect
    src = inspect.getsource(vk)
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("*") and "_kernels()" in line:
            pytest.fail(
                f"elementwise.py:{lineno} uses a tail unpack of _kernels() "
                f"({stripped!r}) — this silently rebinds when the tuple grows. "
                "Use index access: unw, gated = _kernels()[6], _kernels()[7]"
            )
