"""Shared fast-path gate for the torch ops that fall back to ``*_reference``.

The V4.1 ops own Triton accelerators but promise "always correct": instead of
raising off-GPU (the GLM set's contract) they route to their exact torch
reference. This helper keeps that availability probe — every tensor on CUDA,
triton importable — in one place; the op-specific shape gates (power-of-two
arange dims, small-tile cutoffs, scale-block shapes) stay next to their
kernels.

Importing this module loads no Torch and no Triton.
"""


def fast_path(*tensors) -> bool:
    """True when the Triton fast path may be taken: every tensor is on CUDA
    and triton importable. The caller still applies its own shape gates."""
    for t in tensors:
        if not t.is_cuda:
            return False
    try:
        import triton  # noqa: F401  (availability probe)
    except Exception:
        return False
    return True
