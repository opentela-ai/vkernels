"""Element-wise, reduction and GEMM kernels (Python side of
``src/c/vkernels/kernels``).

Every function mirrors a C++ kernel and accepts numpy arrays. Inputs may be
anything convertible to a C-contiguous ``float32`` array (lists, other float
dtypes — a copy is made when needed); ``out`` buffers must already be
writable, C-contiguous, ``float32`` numpy arrays.

Contract violations raise ``ValueError`` (the C++ ``VK_EXPECTS`` checks
surface the same way through the compiled backend). Results are bit-identical
to the C++ CPU references.

Example:
    >>> import numpy as np
    >>> a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    >>> add(a, a)
    array([2., 4., 6.], dtype=float32)
    >>> relu(np.array([-1.0, 2.0], dtype=np.float32))
    array([0., 2.], dtype=float32)
    >>> sum(np.array([1.0, 2.0], dtype=np.float32))
    3.0
"""

from __future__ import annotations

import numpy as np

from vkernels import _backend

_core = _backend.load_extension()
_COMPILED = _core is not None
if _COMPILED:
    _impl = _core.kernels
else:
    from vkernels import _fallback

    _impl = _fallback

_F32 = np.dtype(np.float32)


def _as_input(x, name: str) -> np.ndarray:
    """Coerce ``x`` to a C-contiguous float32 array (copy only if needed)."""
    arr = np.ascontiguousarray(x, dtype=_F32)
    if arr.dtype != _F32:
        # np.ascontiguousarray never changes dtype; guard future changes.
        arr = np.ascontiguousarray(arr, dtype=_F32)  # pragma: no cover
    return arr


def _as_out(n: int, out, name: str = "out") -> np.ndarray:
    """Validate (or allocate) the ``out`` buffer for ``n`` elements.

    ``out`` must be a writable, C-contiguous float32 numpy array of exactly
    ``n`` elements; anything else raises TypeError/ValueError up front so a
    silent copy can never swallow the kernel's result.
    """
    if out is None:
        return np.empty(n, dtype=_F32)
    if not isinstance(out, np.ndarray):
        raise TypeError(f"{name} must be a numpy array (or None)")
    if out.dtype != _F32:
        raise TypeError(f"{name} must have dtype float32, got {out.dtype}")
    if not out.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not out.flags.writeable:
        raise ValueError(f"{name} must be writable")
    if out.size != n:
        raise ValueError(f"{name} must hold {n} elements, got {out.size}")
    return out


def add(a, b, out=None) -> np.ndarray:
    """Compute ``out = a + b`` element-wise.

    Args:
        a: first operand, any shape (treated flat).
        b: second operand, same number of elements as ``a``.
        out: optional writable float32 numpy array of ``a.size`` elements;
            a new array is allocated when omitted.

    Returns:
        The output array (``out`` if given, otherwise a new float32 array).

    Raises:
        ValueError: if ``a`` and ``b`` have different lengths.

    Example:
        >>> add([1.0, 2.0], [10.0, 20.0])
        array([11., 22.], dtype=float32)
    """
    a_arr = _as_input(a, "a")
    b_arr = _as_input(b, "b")
    if a_arr.size != b_arr.size:
        raise ValueError("a and b must have equal length")
    out_arr = _as_out(a_arr.size, out)
    _impl.add(a_arr, b_arr, out_arr)
    return out_arr


def scale(x, alpha: float, out=None) -> np.ndarray:
    """Compute ``out = alpha * x`` element-wise.

    Args:
        x: input array, any shape (treated flat).
        alpha: scalar multiplier (converted to float32).
        out: optional writable float32 array of ``x.size`` elements.

    Returns:
        The output array (``out`` if given, otherwise a new float32 array).

    Raises:
        ValueError: if ``x`` and ``out`` have different lengths.

    Example:
        >>> scale([1.0, 2.0, 3.0], 2.0)
        array([2., 4., 6.], dtype=float32)
    """
    x_arr = _as_input(x, "x")
    out_arr = _as_out(x_arr.size, out)
    _impl.scale(x_arr, float(np.float32(alpha)), out_arr)
    return out_arr


def relu(x, out=None) -> np.ndarray:
    """Compute ``out = max(x, 0)`` element-wise.

    Args:
        x: input array, any shape (treated flat).
        out: optional writable float32 array of ``x.size`` elements.

    Returns:
        The output array (``out`` if given, otherwise a new float32 array).

    Example:
        >>> relu([-1.0, 0.5, 2.0])
        array([0. , 0.5, 2. ], dtype=float32)
    """
    x_arr = _as_input(x, "x")
    out_arr = _as_out(x_arr.size, out)
    _impl.relu(x_arr, out_arr)
    return out_arr


def sum(x) -> float:
    """Sum of all elements, accumulated in float32 (like the C++ kernel).

    Args:
        x: input array, any shape (treated flat); must be non-empty.

    Returns:
        The sum as a Python float.

    Raises:
        ValueError: if ``x`` is empty.

    Example:
        >>> sum([1.0, 2.0, 3.0])
        6.0
    """
    return float(_impl.sum(_as_input(x, "x")))


def max(x) -> float:
    """Maximum of all elements (like the C++ kernel).

    Args:
        x: input array, any shape (treated flat); must be non-empty.

    Returns:
        The maximum as a Python float.

    Raises:
        ValueError: if ``x`` is empty.

    Example:
        >>> max([1.0, 5.0, 3.0])
        5.0
    """
    return float(_impl.max(_as_input(x, "x")))


def gemm(A, B, alpha: float = 1.0, beta: float = 0.0, out=None) -> np.ndarray:
    """Compute ``C = alpha * A @ B + beta * C`` (SGEMM, row-major).

    This is the Pythonic form of the C++ ``gemm(M, N, K, alpha, A, B, beta,
    C)``: the matrix dimensions are inferred from the input shapes.

    Args:
        A: float32 matrix of shape ``(M, K)``.
        B: float32 matrix of shape ``(K, N)``.
        alpha: scalar for the product term (default 1.0).
        beta: scalar for the accumulator term (default 0.0).
        out: optional writable float32 array of shape ``(M, N)`` (or flat
            ``(M*N,)``); a new ``(M, N)`` array is allocated when omitted.

    Returns:
        The output array (``out`` if given, otherwise a new ``(M, N)``
        float32 array).

    Raises:
        ValueError: if the inner dimensions disagree, or ``out`` has the
            wrong number of elements.

    Example:
        >>> gemm([[1.0, 2.0], [3.0, 4.0]], [[1.0], [1.0]])
        array([[ 3.],
               [ 7.]], dtype=float32)
    """
    a_arr = _as_input(A, "A")
    b_arr = _as_input(B, "B")
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("A and B must be 2-D (MxK and KxN)")
    M, K = a_arr.shape
    k2, N = b_arr.shape
    if K != k2:
        raise ValueError(
            f"inner dimensions must match: A is {M}x{K} but B is {k2}x{N}"
        )
    out_arr = _as_out(M * N, out, "out")
    _impl.gemm(M, N, K, float(np.float32(alpha)), a_arr, b_arr,
               float(np.float32(beta)), out_arr)
    if out_arr.shape != (M, N):
        out_arr = out_arr.reshape(M, N)
    return out_arr
