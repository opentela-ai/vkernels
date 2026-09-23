"""Layout contract for elementwise._kernels() — guards against tuple-position drift.

The GLM glue lane (lvr4-glue) appended ``_swiglu_limit`` to the ``_kernels()``
tuple; two launch sites still used tail unpacks (``*_, unw, gated = ...``),
which silently rebind on ANY append and only detonate on a CUDA rig
(triton launch TypeError). Local gates never caught it because ``_kernels()``
needs triton and is therefore CPU-untestable — these tests are STATIC
(source inspection, no triton import) so they run and fail locally.

Contract: indices 0-7 are the ORIGINAL elementwise kernels in their historical
order; index 8 is ``_swiglu_limit`` (append-only extension). New kernels must
be APPENDED and every consumer must use index access, never tail unpacks.
"""

import inspect
import re

import pytest

vk = pytest.importorskip("vkernels.torch_ops.elementwise")

EXPECTED_RETURN = (
    "_norm", "_qk_norm", "_rope", "_silu_mul", "_store_kv",
    "_qk_norm_rope", "_norm_uw", "_norm_gated", "_swiglu_limit",
)


def _returned_tuple():
    src = inspect.getsource(vk._kernels)
    m = re.search(r"^\s*return\s+(.+)$", src, re.M)
    assert m, "_kernels() has no return statement?"
    return tuple(s.strip().rstrip(",") for s in m.group(1).split(","))


def test_kernels_return_matches_pinned_layout():
    got = _returned_tuple()
    assert got == EXPECTED_RETURN, (
        f"_kernels() return changed: {got}\n"
        "If you ADDED a kernel: APPEND it at the end AND audit every consumer —\n"
        "tail unpacks ('*_, unw, gated = ...') silently rebind on any append\n"
        "(this killed clariden job 3499983 at engine startup). Use index access:\n"
        "  unw, gated = _kernels()[6], _kernels()[7]"
    )


def test_returned_names_are_defined_kernels():
    mod_src = inspect.getsource(vk)
    for name in _returned_tuple():
        assert re.search(rf"^\s*def {re.escape(name)}\(", mod_src, re.M), (
            f"_kernels() returns {name!r} which is not a module-level def — "
            "the layout contract cannot be trusted; fix _kernels()."
        )


def test_no_tail_unpacks_of_kernels_tuple():
    """Grep-guard: elementwise.py must not unpack _kernels() with star patterns."""
    src = inspect.getsource(vk)
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("*") and "_kernels()" in line:
            pytest.fail(
                f"elementwise.py:{lineno} uses a tail unpack of _kernels() "
                f"({stripped!r}) — this silently rebinds when the tuple grows. "
                "Use index access: unw, gated = _kernels()[6], _kernels()[7]"
            )
