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
    return _as_out_typed(out, _F32, n, name)


def _as_out_typed(out, dtype, n: int, name: str = "out") -> np.ndarray:
    """Validate a caller-provided output buffer (writable, C-contiguous,
    exact dtype and length). Used for non-float32 outputs (bf16, uint8…)."""
    if not isinstance(out, np.ndarray):
        raise TypeError(f"{name} must be a numpy array (or None)")
    if out.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {out.dtype}")
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


# ---------------------------------------------------------------------------
# MoE / AMD gfx942 low-level primitives
# ---------------------------------------------------------------------------


def direct_lds_fill_bf16(lds_dst: int, global_src: int,
                         elements: int) -> None:
    """Copy ``elements`` bf16 values from a global-memory address to an LDS
    address (plain memcpy on the host; vectorised loads + LDS stores in the
    HIP path).

    This is the host replacement for CDNA4's ``raw_ptr_buffer_load_lds``.

    Args:
        lds_dst: byte address of the LDS destination.
        global_src: byte address of the global-memory source.
        elements: number of bf16 (2-byte) values to copy.

    Raises:
        ValueError: if either address is null (with non-zero elements).

    Example:
        >>> import ctypes
        >>> src = (ctypes.c_uint16 * 4)(0x3F80, 0x4000, 0x4040, 0x4080)
        >>> dst = (ctypes.c_uint16 * 4)()
        >>> direct_lds_fill_bf16(ctypes.addressof(dst),
        ...                      ctypes.addressof(src), 4)
        >>> list(dst)
        [16256, 16384, 16448, 16512]
    """
    _impl.direct_lds_fill_bf16(int(lds_dst), int(global_src),
                               int(elements))


def fp4_to_bf16_dequant(packed, scale: float = 1.0) -> np.ndarray:
    """Convert packed fp4 (E2M1 microscaling format, two values per byte,
    low nibble first) to bf16 (uint16 bit patterns).

    Args:
        packed: uint8 array (or anything convertible to uint8) of packed
            fp4 values.
        scale: per-block scale factor (default 1.0).

    Returns:
        A uint16 numpy array of bf16 bit patterns, length = 2 × len(packed).

    Raises:
        ValueError: if ``packed`` is empty.

    Example:
        >>> fp4_to_bf16_dequant(np.array([0x00], dtype=np.uint8))
        array([0, 0], dtype=uint16)
    """
    packed_arr = np.ascontiguousarray(packed, dtype=np.uint8)
    return _impl.fp4_to_bf16_dequant(packed_arr, float(np.float32(scale)))


def use_async_copy_default() -> bool:
    """Return True if async copy should be used by default.

    On gfx942 (CDNA3) it misbehaves and the HIP implementation defaults to
    OFF. On other architectures — and on the host build — it defaults to ON.
    The ``K3_NO_ASYNC`` environment variable overrides: ``"0"`` = ON,
    ``"1"`` = OFF.

    Example:
        >>> use_async_copy_default()
        True
    """
    return _impl.use_async_copy_default()


def mfma_f32_16x16x16bf16(c: list[float], a: list[int], b: list[int],
                          cbsz: int = 0, abid: int = 0,
                          blgp: int = 0) -> None:
    """K16 bf16 MFMA: ``C[0..3] += A[0..1] × B[0..1]`` (16×16×16 bf16,
    accumulator fp32).

    On gfx942, the CDNA4-only K32 bf16 MFMA is emulated by calling this
    K16 function twice (once for K=0..15, once for K=16..31).

    Args:
        c: list of 4 float accumulators (updated in-place).
        a: list of 2 uint32_t — packed bf16 A fragment.
        b: list of 2 uint32_t — packed bf16 B fragment.
        cbsz: MFMA control flag (typically 0), ignored on the host path.
        abid: MFMA control flag (typically 0), ignored on the host path.
        blgp: MFMA control flag (typically 0), ignored on the host path.

    Example:
        >>> c = [0.0, 0.0, 0.0, 0.0]
        >>> mfma_f32_16x16x16bf16(c, [0x3F803F80, 0x3F803F80],
        ...                      [0x3F803F80, 0x3F803F80])
        >>> c
        [1.0, 1.0, 1.0, 1.0]
    """
    c_copy = [float(np.float32(v)) for v in c]
    _impl.mfma_f32_16x16x16bf16(
        c_copy,
        [int(v) & 0xFFFFFFFF for v in a],
        [int(v) & 0xFFFFFFFF for v in b],
        int(cbsz), int(abid), int(blgp),
    )
    for i in range(4):
        c[i] = c_copy[i]


# ---------------------------------------------------------------------------
# MoE fused — expert alignment + fused MXFP4 grouped GEMM
# ---------------------------------------------------------------------------

_BLOCK_M = 16


def moe_align_block_size(topk_ids, num_experts: int,
                         block_size: int = _BLOCK_M):
    """Map the ``[M, top_k]`` token→expert routing table to the block-aligned
    sorted layout consumed by :func:`fused_moe_mxfp4`.

    Args:
        topk_ids: int32 array of shape ``(M, top_k)`` (or a flat ``(M*top_k,)``
            view) — which expert each ``(token, selection)`` maps to.
        num_experts: total number of experts (``E``).
        block_size: block alignment (``BLOCK_M``, default 16).

    Returns:
        ``(sorted_ids, expert_ids, EM)`` where

        * ``sorted_ids`` — ``[EM]`` int32 flat topk indices
          (``token*top_k + sel``), grouped by expert and padded per expert
          with ``M*top_k``;
        * ``expert_ids`` — ``[EM // block_size]`` int32, the expert per block
          (``-1`` for pure-padding blocks);
        * ``EM`` — the padded row count (a multiple of ``block_size``).

    Raises:
        ValueError: if ``topk_ids`` is not 2-D, or ``block_size`` is not
            positive.

    Example:
        >>> ids = np.array([[0, 1], [1, 1]], dtype=np.int32)  # M=2, top_k=2
        >>> sorted_ids, expert_ids, EM = moe_align_block_size(ids, 2, 16)
        >>> EM
        32
    """
    ids = np.ascontiguousarray(topk_ids, dtype=np.int32)
    if ids.ndim != 2:
        raise ValueError("topk_ids must be 2-D [M, top_k]")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    M, top_k = ids.shape
    sorted_ids, expert_ids, EM = _impl.moe_align_block_size(
        ids, int(M), int(top_k), int(block_size), int(num_experts))
    return sorted_ids, expert_ids, int(EM)


def fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale,
                    sorted_ids, topk_w, expert_ids,
                    act_scratch=None, out=None, *,
                    top_k: int = 1, group_size: int = 32,
                    swiglu_limit: float = 0.0,
                    activation: str = "swiglu",
                    beta: float = 4.0,
                    linear_beta: float = 25.0,
                    b13=None, b2=None) -> np.ndarray:
    """Fused MXFP4 MoE grouped GEMM (CPU reference oracle).

    Stage 0 — gate_up + activation. With the default ``activation="swiglu"``::

        act[EM, ispp] = silu(clamp(A_sorted @ w13_gate + b13_gate, L))
                      · clamp(A_sorted @ w13_up   + b13_up,   L)

    With ``activation="situ"`` (Kimi-K3, matches vLLM's ``situ_and_mul``):

        gate' = beta · tanh(gate / beta) · sigmoid(gate)
        up'   = linear_beta · tanh(up / linear_beta)   (linear_beta > 0)
        act[EM, ispp] = gate' · up'

    No ``swiglu_limit`` clamp is applied on the SiTU path.

    Stage 1 — down + routed combine::

        out[M, hidden] += act @ w2^T · topk_w_sorted + b2

    Args:
        A: uint16 bf16 activations of shape ``(M, hidden)``.
        w13: uint8 packed E2M1 weights ``(E, 2*ispp, hidden//2)``
            (gate rows then up rows).
        w13_scale: uint8 ue8m0 scales ``(E, 2*ispp, hidden//group_size)``.
        w2: uint8 packed E2M1 weights ``(E, hidden, ispp//2)``.
        w2_scale: uint8 ue8m0 scales ``(E, hidden, ispp//group_size)``.
        sorted_ids: int32 ``[EM]`` flat topk indices (token*top_k + sel) —
            from :func:`moe_align_block_size`.
        topk_w: float32 ``[EM]`` routing weights, sorted to match
            ``sorted_ids``.
        expert_ids: int32 ``[EM // 16]`` expert per block (``-1`` = padding).
        act_scratch: optional writable uint16 ``[EM * ispp]`` buffer for the
            activation intermediate; allocated when omitted.
        out: optional writable float32 ``[M * hidden]`` buffer; the caller
            must zero-initialise it (stage 1 accumulates). A zeroed
            ``(M, hidden)`` array is allocated when omitted.
        top_k: number of experts selected per token (required to decode the
            flat index: ``token = flat / top_k``).
        group_size: ue8m0 scale group size (typically 32).
        swiglu_limit: SwiGLU clamp limit ``L`` (``<= 0`` disables clamping;
            ignored for ``activation="situ"``).
        activation: epilogue tag, ``"swiglu"`` or ``"situ"``.
        beta: SiTU gate softcap (used when ``activation="situ"``).
        linear_beta: SiTU up softcap (``<= 0`` passes ``up`` through
            unmodified; used when ``activation="situ"``).
        b13: optional float32 ``[E * 2 * ispp]`` gate/up bias.
        b2: optional float32 ``[E * hidden]`` down bias.

    Returns:
        The ``out`` array (``(M, hidden)`` float32).

    Raises:
        ValueError: on inconsistent shapes/dimensions.

    Example:
        >>> sorted_ids, expert_ids, EM = moe_align_block_size(topk_ids, E)
        >>> out = fused_moe_mxfp4(A, w13, w13s, w2, w2s,
        ...                       sorted_ids, topk_w, expert_ids, top_k=top_k)
    """
    A_arr = np.ascontiguousarray(A, dtype=np.uint16)
    if A_arr.ndim != 2:
        raise ValueError("A must be 2-D [M, hidden]")
    M, hidden = A_arr.shape

    w13_arr = np.ascontiguousarray(w13, dtype=np.uint8)
    if w13_arr.ndim != 3:
        raise ValueError("w13 must be 3-D [E, 2*ispp, hidden/2]")
    E, two_ispp, h2 = w13_arr.shape
    if h2 * 2 != hidden:
        raise ValueError(
            f"w13 last dim must be hidden/2 = {hidden // 2}, got {h2}")
    if two_ispp % 2 != 0:
        raise ValueError("w13 second dim must be even (2*ispp)")
    ispp = two_ispp // 2

    w13_scale_arr = np.ascontiguousarray(w13_scale, dtype=np.uint8)
    w2_arr = np.ascontiguousarray(w2, dtype=np.uint8)
    if w2_arr.ndim != 3:
        raise ValueError("w2 must be 3-D [E, hidden, ispp/2]")
    E2, h2_, i2 = w2_arr.shape
    if E2 != E or h2_ != hidden or i2 * 2 != ispp:
        raise ValueError(
            f"w2 must be [E, hidden, ispp/2] = [{E}, {hidden}, {ispp // 2}], "
            f"got {w2_arr.shape}")
    w2_scale_arr = np.ascontiguousarray(w2_scale, dtype=np.uint8)

    sorted_ids_arr = np.ascontiguousarray(sorted_ids, dtype=np.int32)
    if sorted_ids_arr.ndim != 1:
        raise ValueError("sorted_ids must be 1-D [EM]")
    EM = sorted_ids_arr.size
    if EM % _BLOCK_M != 0:
        raise ValueError(f"EM must be a multiple of {_BLOCK_M}")

    topk_w_arr = np.ascontiguousarray(topk_w, dtype=np.float32)
    if topk_w_arr.size != EM:
        raise ValueError(
            f"topk_w must have EM = {EM} elements, got {topk_w_arr.size}")

    expert_ids_arr = np.ascontiguousarray(expert_ids, dtype=np.int32)
    if expert_ids_arr.size != EM // _BLOCK_M:
        raise ValueError(
            f"expert_ids must have EM/{_BLOCK_M} = {EM // _BLOCK_M} elements, "
            f"got {expert_ids_arr.size}")

    b13_arr = None
    if b13 is not None:
        b13_arr = np.ascontiguousarray(b13, dtype=np.float32)
        if b13_arr.size != E * 2 * ispp:
            raise ValueError(
                f"b13 must have E*2*ispp = {E * 2 * ispp} elements, "
                f"got {b13_arr.size}")
    b2_arr = None
    if b2 is not None:
        b2_arr = np.ascontiguousarray(b2, dtype=np.float32)
        if b2_arr.size != E * hidden:
            raise ValueError(
                f"b2 must have E*hidden = {E * hidden} elements, "
                f"got {b2_arr.size}")

    if act_scratch is None:
        act_scratch_arr = np.empty(EM * ispp, dtype=np.uint16)
    else:
        act_scratch_arr = _as_out_typed(act_scratch, np.uint16, EM * ispp,
                                        "act_scratch")

    if out is None:
        out_arr = np.zeros(M * hidden, dtype=np.float32)
    else:
        out_arr = _as_out_typed(out, np.float32, M * hidden, "out")

    act_key = activation.lower()
    if act_key not in ("swiglu", "situ"):
        raise ValueError(
            f"activation must be 'swiglu' or 'situ', got {activation!r}")
    act_tag = 0 if act_key == "swiglu" else 1

    _impl.fused_moe_mxfp4(
        A_arr, w13_arr, w13_scale_arr, w2_arr, w2_scale_arr,
        sorted_ids_arr, topk_w_arr, expert_ids_arr,
        act_scratch_arr, out_arr,
        int(M), int(hidden), int(ispp), int(top_k), int(EM),
        int(group_size), float(np.float32(swiglu_limit)), act_tag,
        float(np.float32(beta)), float(np.float32(linear_beta)),
        b13_arr, b2_arr)

    return out_arr.reshape(M, hidden)
