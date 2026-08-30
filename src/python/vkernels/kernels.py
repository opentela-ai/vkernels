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


def _as_out_any(out, dtype, n: int, name: str = "out", *, zero: bool = False):
    """Allocate (when ``out is None``) or validate a non-float32 output.

    ``zero`` requests a zero-initialised buffer (for state that must start
    at zero, e.g. the KDA inter-state carry); otherwise an uninitialised
    buffer is allocated, matching :func:`_as_out`.
    """
    if out is None:
        return np.zeros(n, dtype=dtype) if zero else np.empty(n, dtype=dtype)
    return _as_out_typed(out, dtype, n, name)


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
        raise ValueError(f"inner dimensions must match: A is {M}x{K} but B is {k2}x{N}")
    out_arr = _as_out(M * N, out, "out")
    _impl.gemm(
        M,
        N,
        K,
        float(np.float32(alpha)),
        a_arr,
        b_arr,
        float(np.float32(beta)),
        out_arr,
    )
    if out_arr.shape != (M, N):
        out_arr = out_arr.reshape(M, N)
    return out_arr


# ---------------------------------------------------------------------------
# MoE / AMD gfx942 low-level primitives
# ---------------------------------------------------------------------------


def direct_lds_fill_bf16(lds_dst: int, global_src: int, elements: int) -> None:
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
    _impl.direct_lds_fill_bf16(int(lds_dst), int(global_src), int(elements))


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


def mfma_f32_16x16x16bf16(
    c: list[float],
    a: list[int],
    b: list[int],
    cbsz: int = 0,
    abid: int = 0,
    blgp: int = 0,
) -> None:
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
        int(cbsz),
        int(abid),
        int(blgp),
    )
    for i in range(4):
        c[i] = c_copy[i]


# ---------------------------------------------------------------------------
# MoE fused — expert alignment + fused MXFP4 grouped GEMM
# ---------------------------------------------------------------------------

_BLOCK_M = 16


def moe_align_block_size(topk_ids, num_experts: int, block_size: int = _BLOCK_M):
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
        ids, int(M), int(top_k), int(block_size), int(num_experts)
    )
    return sorted_ids, expert_ids, int(EM)


def fused_moe_mxfp4(
    A,
    w13,
    w13_scale,
    w2,
    w2_scale,
    sorted_ids,
    topk_w,
    expert_ids,
    act_scratch=None,
    out=None,
    *,
    top_k: int = 1,
    group_size: int = 32,
    swiglu_limit: float = 0.0,
    activation: str = "swiglu",
    beta: float = 4.0,
    linear_beta: float = 25.0,
    b13=None,
    b2=None,
) -> np.ndarray:
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
        raise ValueError(f"w13 last dim must be hidden/2 = {hidden // 2}, got {h2}")
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
            f"got {w2_arr.shape}"
        )
    w2_scale_arr = np.ascontiguousarray(w2_scale, dtype=np.uint8)

    sorted_ids_arr = np.ascontiguousarray(sorted_ids, dtype=np.int32)
    if sorted_ids_arr.ndim != 1:
        raise ValueError("sorted_ids must be 1-D [EM]")
    EM = sorted_ids_arr.size
    if EM % _BLOCK_M != 0:
        raise ValueError(f"EM must be a multiple of {_BLOCK_M}")

    topk_w_arr = np.ascontiguousarray(topk_w, dtype=np.float32)
    if topk_w_arr.size != EM:
        raise ValueError(f"topk_w must have EM = {EM} elements, got {topk_w_arr.size}")

    expert_ids_arr = np.ascontiguousarray(expert_ids, dtype=np.int32)
    if expert_ids_arr.size != EM // _BLOCK_M:
        raise ValueError(
            f"expert_ids must have EM/{_BLOCK_M} = {EM // _BLOCK_M} elements, "
            f"got {expert_ids_arr.size}"
        )

    b13_arr = None
    if b13 is not None:
        b13_arr = np.ascontiguousarray(b13, dtype=np.float32)
        if b13_arr.size != E * 2 * ispp:
            raise ValueError(
                f"b13 must have E*2*ispp = {E * 2 * ispp} elements, got {b13_arr.size}"
            )
    b2_arr = None
    if b2 is not None:
        b2_arr = np.ascontiguousarray(b2, dtype=np.float32)
        if b2_arr.size != E * hidden:
            raise ValueError(
                f"b2 must have E*hidden = {E * hidden} elements, got {b2_arr.size}"
            )

    if act_scratch is None:
        act_scratch_arr = np.empty(EM * ispp, dtype=np.uint16)
    else:
        act_scratch_arr = _as_out_typed(
            act_scratch, np.uint16, EM * ispp, "act_scratch"
        )

    if out is None:
        out_arr = np.zeros(M * hidden, dtype=np.float32)
    else:
        out_arr = _as_out_typed(out, np.float32, M * hidden, "out")

    act_key = activation.lower()
    if act_key not in ("swiglu", "situ"):
        raise ValueError(f"activation must be 'swiglu' or 'situ', got {activation!r}")
    act_tag = 0 if act_key == "swiglu" else 1

    _impl.fused_moe_mxfp4(
        A_arr,
        w13_arr,
        w13_scale_arr,
        w2_arr,
        w2_scale_arr,
        sorted_ids_arr,
        topk_w_arr,
        expert_ids_arr,
        act_scratch_arr,
        out_arr,
        int(M),
        int(hidden),
        int(ispp),
        int(top_k),
        int(EM),
        int(group_size),
        float(np.float32(swiglu_limit)),
        act_tag,
        float(np.float32(beta)),
        float(np.float32(linear_beta)),
        b13_arr,
        b2_arr,
    )

    return out_arr.reshape(M, hidden)


# ---------------------------------------------------------------------------
# MXFP4 MoE orchestration ops (issue #22)
#
# Portable (gfx942) replacement for AITER's gfx950-only `module_moe_mxfp4_aux`.
# These bracket a grouped MXFP4 GEMM to build a W4A4 MoE serving path:
#
#   align  (moe_align_block_size)
#     -> mxfp4_moe_sort          gather A    [M, hidden]   -> [EM, hidden]
#     -> mxfp4_moe_quant         E2M1 + ue8m0 per token/block
#     -> mxfp4_moe_sort_scales   gather scales to sorted row order
#     -> grouped GEMM (fused_moe_mxfp4)
#     -> mxfp4_moe_scatter_reduce[_q]   routed combine -> [M, hidden]
#
# The fp4 (E2M1) and scale (ue8m0, `s << 23`) layouts are identical to the
# weight decode in moe.cpp, so a W4A4 GEMM using the same format for A and W
# is numerically symmetric.
# ---------------------------------------------------------------------------


def mxfp4_moe_quant(A, *, group_size: int = 32):
    """MXFP4 (E2M1 + ue8m0) per-token, per-group activation quantization.

    The exact inverse of the weight decode in :mod:`vkernels` — bf16
    activations ``A`` of shape ``(M, hidden)`` are quantized to packed
    E2M1 (two nibbles per byte, low nibble = even K) plus per-group ue8m0
    scales, the same layout the grouped GEMM consumes for the weights. A
    W4A4 GEMM that uses this output for ``A`` is therefore numerically
    symmetric with the W4A16 weight path.

    Per group of ``group_size`` consecutive hidden elements:
      * ``amax = max |A[m, g*gs + i]|``
      * if ``amax == 0`` (or non-finite): scale byte ``0xFF`` (decodes to 0)
      * else ``e = ceil(log2(amax / 3))``, ``scale = 2^e`` (clamped so the
        scale byte is in ``[0, 254]``), and each ``A[...] / scale`` is rounded
        to the nearest representable E2M1 value (ties round to the larger
        magnitude).

    Args:
        A: uint16 bf16 array of shape ``(M, hidden)``. ``hidden`` must be
            divisible by ``group_size`` and even.
        group_size: ue8m0 scale group length (default 32, the K3 setting).

    Returns:
        ``(packed, scales)`` — ``packed`` is uint8 ``[M, hidden/2]`` E2M1 and
        ``scales`` is uint8 ``[M, hidden/group_size]`` ue8m0.

    Raises:
        ValueError: if ``A`` is not 2-D uint16-compatible, or ``hidden`` is
            not divisible by ``group_size``.
    """
    A_arr = np.ascontiguousarray(A, dtype=np.uint16)
    if A_arr.ndim != 2:
        raise ValueError("A must be 2-D [M, hidden]")
    M, hidden = A_arr.shape
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    if hidden % int(group_size) != 0:
        raise ValueError(
            f"hidden ({hidden}) must be a multiple of group_size ({int(group_size)})"
        )
    packed, scales = _impl.mxfp4_moe_quant(A_arr, int(M), int(hidden), int(group_size))
    return packed.reshape(M, hidden // 2), scales.reshape(M, hidden // group_size)


def mxfp4_moe_sort(A, sorted_ids, *, top_k: int):
    """Gather bf16 activations into the block-aligned sorted row order.

    Row ``r`` of the output is ``A[sorted_ids[r] / top_k]`` when
    ``sorted_ids[r] < M * top_k`` (a real ``(token, sel)`` pair), and zero
    for padding rows (``sorted_ids[r] >= M * top_k``). The gather is exact
    for bf16 (a 16-bit copy), so a subsequent :func:`mxfp4_moe_quant`
    produces a ``0xFF`` scale and zero nibbles for padding rows.

    Args:
        A: uint16 bf16 array of shape ``(M, hidden)``.
        sorted_ids: int32 array of shape ``(EM,)`` from
            :func:`moe_align_block_size`.
        top_k: experts selected per token (decodes ``token = flat / top_k``).

    Returns:
        uint16 bf16 array of shape ``(EM, hidden)`` in sorted row order.
    """
    A_arr = np.ascontiguousarray(A, dtype=np.uint16)
    if A_arr.ndim != 2:
        raise ValueError("A must be 2-D [M, hidden]")
    M, hidden = A_arr.shape
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32)
    if ids.ndim != 1:
        raise ValueError("sorted_ids must be 1-D [EM]")
    EM = ids.size
    out = _impl.mxfp4_moe_sort(A_arr, ids, int(M), int(hidden), int(top_k), int(EM))
    return out.reshape(EM, hidden)


def mxfp4_moe_sort_scales(scales, sorted_ids, *, top_k: int):
    """Gather per-token ue8m0 scales into sorted row order.

    Structurally identical to :func:`mxfp4_moe_sort` on the per-token scale
    tensor: ``scales [M, n_groups] -> [EM, n_groups]`` by ``sorted_ids``, so
    quantized activations and their scales share a row order. Padding rows
    are zeroed.

    Args:
        scales: uint8 ue8m0 array of shape ``(M, n_groups)`` (e.g. the
            second return value of :func:`mxfp4_moe_quant`).
        sorted_ids: int32 array of shape ``(EM,)``.
        top_k: experts selected per token.

    Returns:
        uint8 array of shape ``(EM, n_groups)``.
    """
    s = np.ascontiguousarray(scales, dtype=np.uint8)
    if s.ndim != 2:
        raise ValueError("scales must be 2-D [M, n_groups]")
    M, n_groups = s.shape
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32)
    if ids.ndim != 1:
        raise ValueError("sorted_ids must be 1-D [EM]")
    EM = ids.size
    out = _impl.mxfp4_moe_sort_scales(
        s, ids, int(M), int(n_groups), int(top_k), int(EM)
    )
    return out.reshape(EM, n_groups)


def mxfp4_moe_scatter_reduce(
    partial, topk_w, sorted_ids, *, M: int, width: int, top_k: int
):
    """Routed output combine (float32 partials).

    ``partial`` is the per-expert GEMM output in sorted row order
    ``(EM, width)``; for each real sorted row ``r`` (``sorted_ids[r] <
    M * top_k``), ``out[token] += partial[r] * topk_w[r]`` with
    ``token = sorted_ids[r] / top_k``. ``out`` is zero-initialised and
    accumulated with atomic semantics (several sorted rows map to one token).

    This is the bias-free form of the fused-path combine; bias is added
    separately in the W4A4 path. Matches AITER ``mxfp4_moe_scatter_reduce``.

    Args:
        partial: float32 array of shape ``(EM, width)``.
        topk_w: float32 array of shape ``(EM,)`` routing weights, sorted to
            match ``sorted_ids``.
        sorted_ids: int32 array of shape ``(EM,)``.
        M: original token count (``out`` has ``M`` rows).
        width: output / partial column count (``hidden`` for the down output,
            ``2*ispp`` for the gate/up intermediate).
        top_k: experts selected per token.

    Returns:
        float32 array of shape ``(M, width)`` (the reduced output).
    """
    p = np.ascontiguousarray(partial, dtype=np.float32)
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32)
    w = np.ascontiguousarray(topk_w, dtype=np.float32)
    if ids.ndim != 1:
        raise ValueError("sorted_ids must be 1-D [EM]")
    EM = ids.size
    if int(M) <= 0:
        raise ValueError("M must be positive")
    if int(width) <= 0:
        raise ValueError("width must be positive")
    if int(top_k) <= 0:
        raise ValueError("top_k must be positive")
    if p.size != EM * int(width):
        raise ValueError(
            f"partial must have EM*width = {EM * int(width)} elements, got {p.size}"
        )
    if w.size != EM:
        raise ValueError(f"topk_w must have EM = {EM} elements, got {w.size}")
    out = _impl.mxfp4_moe_scatter_reduce(
        p, w, ids, int(M), int(width), int(top_k), int(EM)
    )
    return out.reshape(M, width)


def mxfp4_moe_scatter_reduce_q(
    partial_q,
    partial_s,
    topk_w,
    sorted_ids,
    *,
    M: int,
    width: int,
    top_k: int,
    group_size: int = 32,
):
    """Routed output combine of a quantized partial (E2M1 + ue8m0).

    Identical to :func:`mxfp4_moe_scatter_reduce` except the per-expert
    partial is kept in the MXFP4 layout and dequantized inline — the "Q"
    combine AITER ships on gfx950 to cut the scatter bandwidth.

    Args:
        partial_q: uint8 packed E2M1 array of shape ``(EM, width/2)``.
        partial_s: uint8 ue8m0 array of shape ``(EM, width/group_size)``.
        topk_w: float32 array of shape ``(EM,)``.
        sorted_ids: int32 array of shape ``(EM,)``.
        M: original token count (``out`` has ``M`` rows).
        width: output column count (must be divisible by ``group_size``).
        top_k: experts selected per token.
        group_size: ue8m0 group length (default 32).

    Returns:
        float32 array of shape ``(M, width)`` (the reduced output).
    """
    pq = np.ascontiguousarray(partial_q, dtype=np.uint8)
    ps = np.ascontiguousarray(partial_s, dtype=np.uint8)
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32)
    w = np.ascontiguousarray(topk_w, dtype=np.float32)
    if ids.ndim != 1:
        raise ValueError("sorted_ids must be 1-D [EM]")
    EM = ids.size
    if int(M) <= 0 or int(width) <= 0 or int(top_k) <= 0:
        raise ValueError("M, width, top_k must be positive")
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    if int(width) % int(group_size) != 0:
        raise ValueError(
            f"width ({int(width)}) must be a multiple of group_size ({int(group_size)})"
        )
    if pq.size != EM * (int(width) // 2):
        raise ValueError(
            f"partial_q must have EM*width/2 = "
            f"{EM * (int(width) // 2)} elements, got {pq.size}"
        )
    if ps.size != EM * (int(width) // int(group_size)):
        raise ValueError(
            f"partial_s must have EM*width/group = "
            f"{EM * (int(width) // int(group_size))} elements, "
            f"got {ps.size}"
        )
    if w.size != EM:
        raise ValueError(f"topk_w must have EM = {EM} elements, got {w.size}")
    out = _impl.mxfp4_moe_scatter_reduce_q(
        pq, ps, w, ids, int(M), int(width), int(top_k), int(EM), int(group_size)
    )
    return out.reshape(M, width)


# ---------------------------------------------------------------------------
# bf16 GEMM (gfx942 projection reference, issue #29)
# ---------------------------------------------------------------------------


def gemm_bf16(A, B, alpha: float = 1.0, beta: float = 0.0, out=None):
    """Compute ``C = alpha * A @ B + beta * C`` (bf16 in/out, fp32 accumulate,
    single round-to-nearest-even on store).

    This is the Pythonic form of the C++ ``gemm_bf16(M, N, K, alpha, A, B,
    beta, C)``: ``A``, ``B`` are bf16 (uint16 bit patterns), the inner
    product accumulates in fp32, and the result is stored back as bf16 with
    the same RNE as the HIP MFMA kernel. It is the bit-faithful host oracle
    the gfx942 bf16 GEMM is checked against.

    Args:
        A: uint16 bf16 matrix of shape ``(M, K)``.
        B: uint16 bf16 matrix of shape ``(K, N)``.
        alpha: scalar for the product term (default 1.0).
        beta: scalar for the accumulator term (default 0.0; ``beta == 0``
            skips the prior-``C`` read, matching the kernel).
        out: optional writable uint16 array of shape ``(M, N)`` (or flat
            ``(M*N,)``); a new ``(M, N)`` uint16 array is allocated when
            omitted.

    Returns:
        The output array (``out`` if given, otherwise a new ``(M, N)``
        uint16 array of bf16 bit patterns).

    Raises:
        ValueError: if the inner dimensions disagree, or ``out`` has the
            wrong dtype/length.
    """
    a_arr = np.ascontiguousarray(A, dtype=np.uint16)
    b_arr = np.ascontiguousarray(B, dtype=np.uint16)
    if a_arr.ndim != 2 or b_arr.ndim != 2:
        raise ValueError("A and B must be 2-D (MxK and KxN)")
    M, K = a_arr.shape
    k2, N = b_arr.shape
    if K != k2:
        raise ValueError(f"inner dimensions must match: A is {M}x{K} but B is {k2}x{N}")
    out_arr = _as_out_any(out, np.uint16, M * N, "out")
    _impl.gemm_bf16(
        int(M),
        int(N),
        int(K),
        float(np.float32(alpha)),
        a_arr,
        b_arr,
        float(np.float32(beta)),
        out_arr,
    )
    if out_arr.shape != (M, N):
        out_arr = out_arr.reshape(M, N)
    return out_arr


def gemm_bf16_config(M, N, K):
    """Per-shape ``(bm, bn, bk, threads)`` MFMA tile the HIP bf16 GEMM should
    launch for ``(M, N, K)``.

    ``BK`` is fixed at 64. ``M <= 64`` (serving / decode) selects
    ``(16, 16, 64, 64)`` (one wavefront per 16-row fragment); larger ``M``
    (warmup / prefill) selects ``(64, 64, 64, 256)`` (four wavefronts).

    Args:
        M, N, K: integer matrix dimensions.

    Returns:
        The ``(bm, bn, bk, threads)`` tuple (all ``int``).
    """
    return tuple(int(v) for v in _impl.gemm_bf16_config(int(M), int(N), int(K)))


# ---------------------------------------------------------------------------
# MLA: absorbed-form Multi-head Latent Attention (issue #21)
# ---------------------------------------------------------------------------


def mla_fwd(q, k_c, k_pe, v_c, *, q_start=0, kv_start=0, scale=1.0, out=None):
    """Absorbed-form Multi-head Latent Attention forward (fp32 two-pass
    stable softmax, causal via ``(q_start, kv_start)``).

    This is the CPU reference the gfx942 MLA kernel is checked against. The
    batch / head / sequence / latent dimensions are inferred from the input
    shapes:

    * ``q``    : ``(B, H, S_q, kv_lora_rank + qk_rope_head_dim)``
    * ``k_c``  : ``(B, S_kv, kv_lora_rank)``
    * ``k_pe`` : ``(B, S_kv, qk_rope_head_dim)``
    * ``v_c``  : ``(B, S_kv, kv_lora_rank)``
    * ``out``  : ``(B, H, S_q, kv_lora_rank)``

    For query ``i`` (global position ``q_start + i``) the attended keys are
    those ``j`` with ``kv_start + j <= q_start + i``; every other key is
    masked to ``-inf``. When all keys are masked the output row is zero.

    Args:
        q: float32 ``(B, H, S_q, lr+rhd)`` queries (nope prefix then rope).
        k_c: float32 ``(B, S_kv, lr)`` compressed keys.
        k_pe: float32 ``(B, S_kv, rhd)`` decoupled RoPE key positions.
        v_c: float32 ``(B, S_kv, lr)`` compressed values.
        q_start: global position of the first query row (default 0).
        kv_start: global position of the first key row (default 0).
        scale: softmax pre-scale, typically ``1/sqrt(lr+rhd)`` (default 1.0).
        out: optional writable float32 ``(B, H, S_q, lr)`` buffer; allocated
            (zero-initialised) when omitted.

    Returns:
        The output array (``out`` if given, otherwise a new
        ``(B, H, S_q, kv_lora_rank)`` float32 array).

    Raises:
        ValueError: on inconsistent shapes or non-positive ranks.
    """
    q_arr = _as_input(q, "q")
    kc_arr = _as_input(k_c, "k_c")
    kp_arr = _as_input(k_pe, "k_pe")
    vc_arr = _as_input(v_c, "v_c")
    if q_arr.ndim != 4:
        raise ValueError("q must be 4-D [B, H, S_q, lr+rhd]")
    if kc_arr.ndim != 3:
        raise ValueError("k_c must be 3-D [B, S_kv, lr]")
    if kp_arr.ndim != 3:
        raise ValueError("k_pe must be 3-D [B, S_kv, rhd]")
    if vc_arr.ndim != 3:
        raise ValueError("v_c must be 3-D [B, S_kv, lr]")
    B, H, S_q, D_q = q_arr.shape
    Bk, S_kv, lr = kc_arr.shape
    Bp, S_kvp, rhd = kp_arr.shape
    Bv, S_kv2, lr2 = vc_arr.shape
    if not (lr > 0 and rhd > 0):
        raise ValueError("kv_lora_rank and qk_rope_head_dim must be positive")
    if D_q != lr + rhd:
        raise ValueError(f"q last dim {D_q} != lr+rhd = {lr+rhd}")
    if Bk != B or Bp != B or Bv != B:
        raise ValueError(f"batch mismatch: q={B} k_c={Bk} k_pe={Bp} v_c={Bv}")
    if S_kv != S_kvp or S_kv != S_kv2:
        raise ValueError(f"S_kv mismatch: k_c={S_kv} k_pe={S_kvp} v_c={S_kv2}")
    if lr2 != lr:
        raise ValueError(f"kv_lora_rank mismatch: k_c={lr} v_c={lr2}")
    out_arr = _as_out(B * H * S_q * lr, out, "out")
    _impl.mla_fwd(
        int(B), int(H), int(S_q), int(S_kv), int(q_start), int(kv_start),
        int(lr), int(rhd), float(np.float32(scale)), q_arr, kc_arr, kp_arr,
        vc_arr, out_arr,
    )
    if out_arr.shape != (B, H, S_q, lr):
        out_arr = out_arr.reshape(B, H, S_q, lr)
    return out_arr


def mla_config(S_q, kv_lora_rank, qk_rope_head_dim):
    """Per-shape ``(bq, bn_kv, threads)`` tile selector for the HIP MLA
    kernel.

    Decode (``S_q <= 8``) selects ``(1, 64, 64)`` (one query row per block,
    one wavefront); prefill (``S_q >= 9``) selects ``(4, 64, 256)``. The
    latent ranks only affect the runtime, not the tile.

    Args:
        S_q: query sequence length per head.
        kv_lora_rank: compressed latent rank (unused by the heuristic).
        qk_rope_head_dim: decoupled RoPE head dim (unused by the heuristic).

    Returns:
        The ``(bq, bn_kv, threads)`` tuple (all ``int``).
    """
    return tuple(
        int(v)
        for v in _impl.mla_config(int(S_q), int(kv_lora_rank), int(qk_rope_head_dim))
    )


# ---------------------------------------------------------------------------
# DSA: DeepseekSparseAttn sparse-MLA forward (issue #51)
# ---------------------------------------------------------------------------


# log2(e): the DSA softmax scale folds this so the weights are the standard
# natural-exp softmax expressed in base-2 (FlashAttention's log2 trick).
_LOG2E = float(np.log2(np.e))


def dsa_sparse_fwd(q, kv, indices, *, dim, tail_dim, topk=None,
                   kv_group=1, block_I=64, inner_iter=1, sm_scale=None,
                   return_lse=False, out=None, lse=None):
    """DeepseekSparseAttn sparse-MLA forward (GLM-5.3-Flash / DeepSeek-V3),
    fp32 two-pass BASE-2 stable softmax over the ``topk`` indexer-selected
    keys. This is the CPU reference the gfx942 HIP kernel is checked
    against; the device ABI is bf16 (the integrator converts).

    The indexer has already selected, per query token, the ``topk`` most
    relevant KV tiles (``indices``); this kernel scores each query against
    those keys and produces the combined attention output.

    * ``q``       : ``(1, S_q,  H,  dim + tail_dim)`` float32 —
                   ``[q_main (dim) | q_tail (tail_dim)]`` (tail may be 0).
    * ``kv``      : ``(1, S_kv, kv_group, dim + tail_dim)`` float32 (kv_group==1)
                   ``kv[j] = [v (d_v) | k_nope_extra (tail_dim) | k_rope (tail_dim)]``
                   so ``v = kv[j][0:d_v]``, ``k_main = kv[j][0:dim]``,
                   ``k_tail = kv[j][dim:dim+tail_dim]``, with ``d_v = dim - tail_dim``.
    * ``indices`` : ``(1, S_q, kv_group, topk)`` int32 — selected key ids;
                   entries ``< 0`` or ``>= S_kv`` are masked kpool tails.
    * ``out``     : ``(1, S_q, H, d_v)`` float32 (combined).
    * ``lse``     : ``(1, S_q, H)`` float32 (base-2 log-sum-exp; optional).

    ``dim`` and ``tail_dim`` are passed explicitly (they are not separately
    recoverable from ``W = dim + tail_dim``). ``d_v = dim - tail_dim`` must be
    positive. ``sm_scale`` defaults to ``(1/sqrt(dim + tail_dim)) * log2(e)``
    (the base-2 fold); ``tail_dim == 0`` skips the rope-tail dot entirely —
    the exact case the tilelang code path cannot compile.

    Args:
        q: float32 ``(1, S_q, H, dim + tail_dim)`` queries.
        kv: float32 ``(1, S_kv, kv_group, dim + tail_dim)`` keys/values.
        indices: int32 ``(1, S_q, kv_group, topk)`` selected key ids.
        dim: the main key/query width (``qk_nope_head_dim + qk_rope_head_dim``
            in the absorbed form).
        tail_dim: the decoupled RoPE tail width (may be 0).
        topk: number of selected keys per query. Inferred from ``indices``
            when omitted.
        kv_group: number of key/value heads (must be 1).
        block_I, inner_iter: kernel group-tiling configuration (the indexer
            pads ``topk`` to a multiple of ``block_I`` in the sglang layout).
        sm_scale: softmax pre-scale (folds ``log2(e)``); defaults to
            ``(1/sqrt(dim + tail_dim)) * log2(e)``.
        return_lse: when True, also compute the base-2 log-sum-exp.
        out: optional writable float32 ``(1, S_q, H, d_v)`` buffer.
        lse: optional writable float32 ``(1, S_q, H)`` buffer (required
            only when ``return_lse=True``).

    Returns:
        ``out`` when ``return_lse=False``, otherwise ``(out, lse)``.

    Raises:
        ValueError: on inconsistent shapes, ``d_v <= 0``, or ``kv_group != 1``.
    """
    q_arr = _as_input(q, "q")
    kv_arr = _as_input(kv, "kv")
    idx_arr = np.ascontiguousarray(indices, dtype=np.int32)
    if idx_arr.dtype != np.int32:
        raise TypeError("indices must be an integer array")
    if q_arr.ndim != 4:
        raise ValueError("q must be 4-D [1, S_q, H, dim + tail_dim]")
    if kv_arr.ndim != 4:
        raise ValueError("kv must be 4-D [1, S_kv, kv_group, dim + tail_dim]")
    if idx_arr.ndim != 4:
        raise ValueError("indices must be 4-D [1, S_q, kv_group, topk]")
    B, S_q, H, W = (int(v) for v in q_arr.shape)
    Bk, S_kv, kvg, W2 = (int(v) for v in kv_arr.shape)
    Bi, S_q2, kvg2, tk = (int(v) for v in idx_arr.shape)
    if B != 1:
        raise ValueError(f"DSA forward is batch-1 (got B={B})")
    if not (dim > 0 and tail_dim >= 0):
        raise ValueError("dim must be > 0 and tail_dim >= 0")
    d_v = dim - tail_dim
    if d_v <= 0:
        raise ValueError(f"dim - tail_dim must be > 0 (got d_v={d_v})")
    if W != dim + tail_dim or W2 != dim + tail_dim:
        raise ValueError(
            f"q/kv last dim {W}/{W2} != dim + tail_dim = {dim + tail_dim}"
        )
    if kvg != kv_group or kvg2 != kv_group:
        raise ValueError(f"kv_group mismatch: kv={kvg} indices={kvg2} arg={kv_group}")
    if kv_group != 1:
        raise ValueError("kv_group must be 1 (a single shared head_kv)")
    if S_q != S_q2:
        raise ValueError(f"S_q mismatch: q={S_q} indices={S_q2}")
    if topk is None:
        topk = tk
    elif int(topk) != tk:
        raise ValueError(f"topk mismatch: arg={topk} indices={tk}")
    if block_I <= 0 or inner_iter <= 0:
        raise ValueError("block_I and inner_iter must be positive")
    if sm_scale is None:
        sm_scale = (1.0 / float(np.sqrt(float(dim + tail_dim)))) * _LOG2E

    out_arr = _as_out(S_q * H * d_v, out, "out")
    if return_lse:
        lse_arr = _as_out(S_q * H, lse, "lse")
    else:
        lse_arr = np.empty(0, dtype=_F32)  # placeholder (never written)
    _impl.dsa_sparse_fwd(
        int(S_q), int(S_kv), int(H), int(dim), int(tail_dim), int(topk),
        int(kv_group), int(block_I), int(inner_iter), float(np.float32(sm_scale)),
        bool(return_lse), q_arr, kv_arr, idx_arr, out_arr, lse_arr,
    )
    out_arr = out_arr.reshape(1, S_q, H, d_v)
    if return_lse:
        return out_arr, lse_arr.reshape(1, S_q, H)
    return out_arr


def dsa_config(S_q, H, dim, topk):
    """Per-shape ``(bq, threads, block_I, inner_iter)`` tile selector for
    the HIP DSA kernel.

    Decode (``S_q <= 8``) selects ``(1, 64, 64, inner_iter)`` (one query row
    per block, one wavefront); prefill (``S_q >= 9``) selects
    ``(4, 256, 64, inner_iter)``. ``inner_iter`` grows while ``topk`` stays
    divisible by ``block_I * inner_iter`` (so the group tiling aligns with
    the indexer's padded-`topk` layout).

    Returns:
        The ``(bq, threads, block_I, inner_iter)`` tuple (all ``int``).
    """
    return tuple(
        int(v)
        for v in _impl.dsa_config(int(S_q), int(H), int(dim), int(topk))
    )


def dsa_topk_transform(score, lengths, *, pool_size, token_topk,
                       out_cols=None, page_table=None,
                       page_table_row_index=None, topk_indices_offset=None,
                       row_starts=None, seq_lens=None, out=None):
    """DeepseekSparseAttn pool-level top-k transform.

    ``score`` is ``(B, score_stride)`` float32 pool-group logits and
    ``lengths`` is ``(B,)`` int32 valid group counts. The transform selects
    ``token_topk / pool_size`` pool groups per row, expands each winner to
    ``pool_size`` token ids, optionally remaps them through ``page_table`` or
    adds a ragged ``topk_indices_offset``, and optionally appends the
    ``seq_len % pool_size`` tail tokens.
    """
    score_arr = _as_input(score, "score")
    if score_arr.ndim != 2:
        raise ValueError("score must be 2-D [batch_size, score_stride]")
    batch_size, score_stride = (int(v) for v in score_arr.shape)
    lengths_arr = np.ascontiguousarray(lengths, dtype=np.int32)
    if lengths_arr.ndim != 1 or lengths_arr.size != batch_size:
        raise ValueError("lengths must be 1-D with one entry per score row")
    if pool_size <= 1:
        raise ValueError("pool_size must be greater than one")
    if token_topk <= 0 or token_topk % pool_size != 0:
        raise ValueError("token_topk must be a positive multiple of pool_size")
    group_topk = int(token_topk) // int(pool_size)
    if not bool(_impl.dsa_topk_group_topk_supported(group_topk)):
        raise ValueError("unsupported pool-level top-k")
    if page_table is not None and topk_indices_offset is not None:
        raise ValueError("page_table and topk_indices_offset are mutually exclusive")

    def _opt_i32(name, value):
        if value is None:
            return None
        arr = np.ascontiguousarray(value, dtype=np.int32)
        if arr.ndim != 1 or arr.size != batch_size:
            raise ValueError(f"{name} must be 1-D with {batch_size} elements")
        return arr

    page_table_arr = None
    page_table_stride = 0
    if page_table is not None:
        page_table_arr = np.ascontiguousarray(page_table, dtype=np.int32)
        if page_table_arr.ndim != 2:
            raise ValueError("page_table must be 2-D [rows, page_table_stride]")
        if page_table_arr.shape[0] != batch_size:
            raise ValueError("page_table must have one row per score row")
        page_table_stride = int(page_table_arr.shape[1])
    page_row_arr = _opt_i32("page_table_row_index", page_table_row_index)
    if page_table_arr is not None and page_row_arr is not None:
        if np.any(page_row_arr < 0) or np.any(page_row_arr >= page_table_arr.shape[0]):
            raise ValueError("page_table_row_index entries must address a page-table row")
    offsets_arr = _opt_i32("topk_indices_offset", topk_indices_offset)
    row_starts_arr = _opt_i32("row_starts", row_starts)
    seq_lens_arr = _opt_i32("seq_lens", seq_lens)
    if out_cols is None:
        out_cols = int(token_topk) + (int(pool_size) - 1 if seq_lens_arr is not None else 0)
    out_arr = _as_out_typed(out, np.int32, batch_size * int(out_cols), "out") if out is not None else np.empty((batch_size, int(out_cols)), dtype=np.int32)
    out_flat = out_arr.reshape(batch_size * int(out_cols))
    _impl.dsa_topk_transform(
        int(batch_size), int(score_stride), int(pool_size), int(token_topk),
        int(out_cols), score_arr, lengths_arr, out_flat,
        page_table_arr, page_row_arr, offsets_arr, row_starts_arr,
        seq_lens_arr, int(page_table_stride),
    )
    return out_flat.reshape(batch_size, int(out_cols))


# ---------------------------------------------------------------------------
# MHC: multi-head hybrid-attention pre-norm (issue #51, part 2)
# ---------------------------------------------------------------------------


def mhc_pre_gemm_sqrsum(x, fn, *, hc_mult, hidden_size, out=None,
                        sqrsum=None):
    """MHC pre-norm GEMM + squared-sum (GLM-5.3-Flash), fp32 CPU reference.

    Computes, per token ``n`` over ``hc_hidden_size = hc_mult * hidden_size``
    hidden units,

    * ``out[n, o]     = sum_h x[n, h] * fn[o, h]``  for ``o`` in
      ``[0, hc_mult3)``  with ``hc_mult3 = hc_mult * (2 + hc_mult)``, and
    * ``sqrsum[n]     = sum_h x[n, h] ** 2``.

    This is the first half of the tilelang
    ``mhc_pre_gemm_sqrsum_splitk_kernel`` (a GEMM against the MHC reshape
    weight plus the per-token squared-sum fed to the RMS-norm); the HIP
    device path (``vk_hip_mhc_pre_gemm_sqrsum``) is bf16 and is checked
    against this fp32 reference.

    Args:
        x: float32 ``(num_tokens, hc_mult * hidden_size)`` activations.
        fn: float32 ``(hc_mult * (2 + hc_mult), hc_mult * hidden_size)``
            reshape weights (the `hc_mult3` per-token output channels).
        hc_mult: MHC head-count multiplier (``hc = hc_mult``).
        hidden_size: per-head hidden width (``hc_hidden_size = hc_mult *
            hidden_size``).
        out: optional writable float32 ``(num_tokens, hc_mult3)`` buffer.
        sqrsum: optional writable float32 ``(num_tokens,)`` buffer.

    Returns:
        ``(out, sqrsum)`` reshaped to ``(num_tokens, hc_mult3)`` and
        ``(num_tokens,)``.

    Raises:
        ValueError: on inconsistent shapes or ``hc_mult/hidden_size <= 0``.
    """
    if hc_mult <= 0:
        raise ValueError(f"hc_mult must be > 0 (got {hc_mult})")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be > 0 (got {hidden_size})")
    hc_hidden_size = int(hc_mult) * int(hidden_size)
    hc_mult3 = int(hc_mult) * (2 + int(hc_mult))
    x_arr = _as_input(x, "x")
    fn_arr = _as_input(fn, "fn")
    if x_arr.size % hc_hidden_size != 0:
        raise ValueError(
            f"x has {x_arr.size} elements, not a multiple of "
            f"hc_mult*hidden_size = {hc_hidden_size}"
        )
    num_tokens = x_arr.size // hc_hidden_size
    if fn_arr.size != hc_mult3 * hc_hidden_size:
        raise ValueError(
            f"fn has {fn_arr.size} elements, expected "
            f"hc_mult3*hc_hidden_size = {hc_mult3}*{hc_hidden_size} = "
            f"{hc_mult3 * hc_hidden_size}"
        )
    out_arr = _as_out(num_tokens * hc_mult3, out, "out")
    sqrsum_arr = _as_out(num_tokens, sqrsum, "sqrsum")
    _impl.mhc_pre_gemm_sqrsum(
        int(num_tokens), int(hc_mult), int(hidden_size),
        x_arr, fn_arr, out_arr, sqrsum_arr,
    )
    return out_arr.reshape(num_tokens, hc_mult3), sqrsum_arr.reshape(num_tokens)


def mhc_post(a, b, c, d, *, hc, hidden, out=None):
    """MHC post-attention combine (GLM-5.3-Flash), fp32 CPU reference.

    Computes, per token ``n`` and output head ``j`` over hidden dim
    ``hidden``,

    .. code-block::

        out[n, j, h] = c[n, j] * d[n, h] + sum_k a[n, k, j] * b[n, k, h]

    where ``a`` is the (post-attention) ``comb_res_mix`` mixing matrix
    ``(num_tokens, hc, hc)``, ``b`` the residual `(num_tokens, hc,
    hidden)` (the ``x`` going into the next sub-layer), ``c`` the scalar
    ``post_layer_mix`` gate `(num_tokens, hc)`, and ``d`` the bypassed
    ``x` `(num_tokens, hidden)``. The HIP device path
    (``vk_hip_mhc_post``) is bf16 (``b``/``d`` in, ``out`` out) and is
    checked against this fp32 reference.

    Args:
        a: float32 `(num_tokens, hc, hc)`` mixing matrix (flat).
        b: float32 `(num_tokens, hc, hidden)`` residual (flat).
        c: float32 `(num_tokens, hc)`` per-head scalar gate (flat).
        d: float32 `(num_tokens, hidden)`` bypass input (flat).
        hc: number of MHC heads (the matrix head dimension).
        hidden: per-head hidden width.
        out: optional writable float32 `(num_tokens, hc, hidden)`` buffer.

    Returns:
        The output array reshaped to `(num_tokens, hc, hidden)``.

    Raises:
        ValueError: on inconsistent shapes or ``hc/hidden <= 0``.
    """
    if hc <= 0:
        raise ValueError(f"hc must be > 0 (got {hc})")
    if hidden <= 0:
        raise ValueError(f"hidden must be > 0 (got {hidden})")
    a_arr = _as_input(a, "a")
    b_arr = _as_input(b, "b")
    c_arr = _as_input(c, "c")
    d_arr = _as_input(d, "d")
    if a_arr.size % (hc * hc) != 0:
        raise ValueError(
            f"a has {a_arr.size} elements, not a multiple of hc*hc = {hc * hc}"
        )
    num_tokens = a_arr.size // (hc * hc)
    if b_arr.size != num_tokens * hc * hidden:
        raise ValueError(
            f"b has {b_arr.size} elements, expected "
            f"{num_tokens}*{hc}*{hidden} = {num_tokens * hc * hidden}"
        )
    if c_arr.size != num_tokens * hc:
        raise ValueError(
            f"c has {c_arr.size} elements, expected {num_tokens * hc}"
        )
    if d_arr.size != num_tokens * hidden:
        raise ValueError(
            f"d has {d_arr.size} elements, expected {num_tokens * hidden}"
        )
    out_arr = _as_out(num_tokens * hc * hidden, out, "out")
    _impl.mhc_post(
        int(num_tokens), int(hc), int(hidden),
        a_arr, b_arr, c_arr, d_arr, out_arr,
    )
    return out_arr.reshape(num_tokens, hc, hidden)


# ---------------------------------------------------------------------------
# KDA: Kimi Delta Attention (issue #21)
# ---------------------------------------------------------------------------


def _kda_dims(q, g, name="q"):
    """Validate ``q`` [B,H,S,D] / ``g`` [B,H,S] and return ``(B, H, S, D)``."""
    q = np.asarray(q)
    g = np.asarray(g)
    if q.ndim != 4:
        raise ValueError(f"{name} must be 4-D [B, H, S, D]")
    if g.ndim != 3:
        raise ValueError("g must be 3-D [B, H, S]")
    B, H, S, D = q.shape
    if g.shape != (B, H, S):
        raise ValueError(f"g must be [{B}, {H}, {S}], got {g.shape}")
    return int(B), int(H), int(S), int(D)


def _kda_dims3(g, beta, name="g"):
    """Validate ``g`` [B,H,S] (and ``beta``) and return ``(B, H, S)``."""
    g_arr = np.asarray(g)
    beta_arr = np.asarray(beta)
    if g_arr.ndim != 3:
        raise ValueError(f"{name} must be 3-D [B, H, S]")
    if beta_arr.shape != g_arr.shape:
        raise ValueError(f"beta must match g shape {g_arr.shape}, got {beta_arr.shape}")
    return int(g_arr.shape[0]), int(g_arr.shape[1]), int(g_arr.shape[2])


def _require_chunk(S, chunk_size):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if int(S) % chunk_size != 0:
        raise ValueError(f"chunk_size ({chunk_size}) must divide S ({int(S)})")
    return int(S) // chunk_size


def kda_layer_norm_gated(x, weight, gate, *, out=None, eps: float = 1e-6):
    """Gated RMSNorm: ``out[n,d] = (x[n]/rms_n) * weight[d] * silu(gate[n,d])``
    with ``rms_n = sqrt(mean(x[:]^2) + eps)`` (the K3 layer-normalization).

    Args:
        x: float32 ``(N, D)`` input.
        weight: float32 ``(D,)`` scale.
        gate: float32 ``(N, D)`` gate (silu is applied per element).
        out: optional writable float32 ``(N, D)`` buffer; a new ``(N, D)``
            array is allocated when omitted.
        eps: RMS epsilon (default 1e-6, must be non-negative).

    Returns:
        The output array (``out`` if given, otherwise a new ``(N, D)``
        float32 array).
    """
    x_arr = _as_input(x, "x")
    weight_arr = _as_input(weight, "weight")
    gate_arr = _as_input(gate, "gate")
    if x_arr.ndim != 2:
        raise ValueError("x must be 2-D [N, D]")
    N, D = x_arr.shape
    if weight_arr.shape != (D,):
        raise ValueError(f"weight must be ({D},), got {weight_arr.shape}")
    if gate_arr.shape != (N, D):
        raise ValueError(f"gate must be ({N}, {D}), got {gate_arr.shape}")
    out_arr = _as_out(N * D, out, "out")
    _impl.kda_layer_norm_gated(
        x_arr, weight_arr, gate_arr, out_arr, int(N), int(D),
        float(np.float32(eps)),
    )
    if out_arr.shape != (N, D):
        out_arr = out_arr.reshape(N, D)
    return out_arr


def kda_gate_chunk_cumsum(g):
    """Log-gate cumulative sums (KDA L2).

    ``g`` ``(B, H, n_chunks, chunk_size)`` (normal space, ``g > 0``; ``g == 0``
    is clamped to ``-1e9`` so a fully-forgotten gate yields ~0 weight) is
    split into a within-chunk INCLUSIVE log-cumsum and a cross-chunk
    EXCLUSIVE log-cumsum. Gate products recover as ``exp(L_b - L_{a-1})``
    (``L_{-1} = 0``).

    Args:
        g: float32 ``(B, H, n_chunks, chunk_size)`` gates.

    Returns:
        ``(intra, inter)`` where ``intra`` is float32
        ``(B, H, n_chunks, chunk_size)`` and ``inter`` is float32
        ``(B, H, n_chunks)``.
    """
    g_arr = _as_input(g, "g")
    if g_arr.ndim != 4:
        raise ValueError("g must be 4-D [B, H, n_chunks, chunk_size]")
    B, H, n_chunks, chunk_size = g_arr.shape
    intra, inter = _impl.kda_gate_chunk_cumsum(
        g_arr, int(B), int(H), int(n_chunks), int(chunk_size)
    )
    return intra.reshape(B, H, n_chunks, chunk_size), inter.reshape(B, H, n_chunks)


def kda_naive_delta_rule_fwd(q, k, v, g, beta, *, out=None):
    """Per-token delta-rule oracle (O(S*D^2) per head).

    Implements the recurrence ``S_t = g_t S_{t-1} + beta_t (v_t -
    S_{t-1} k_t) k_t^T`` then ``o_t = S_t q_t`` per token, per head. This is
    the slow but obviously-correct reference the chunked forward (and the
    HIP kernel) are checked against.

    Args:
        q, k, v: float32 ``(B, H, S, D)``.
        g, beta: float32 ``(B, H, S)``.
        out: optional writable float32 ``(B, H, S, D)`` buffer; a new
            ``(B, H, S, D)`` array is allocated when omitted.

    Returns:
        The output array (``out`` if given, otherwise a new
        ``(B, H, S, D)`` float32 array).
    """
    B, H, S, D = _kda_dims(q, g)
    q_arr = _as_input(q, "q")
    k_arr = _as_input(k, "k")
    v_arr = _as_input(v, "v")
    g_arr = _as_input(g, "g")
    beta_arr = _as_input(beta, "beta")
    out_arr = _as_out(B * H * S * D, out, "out")
    _impl.kda_naive_delta_rule_fwd(
        q_arr, k_arr, v_arr, g_arr, beta_arr, int(B), int(H), int(S),
        int(D), out_arr,
    )
    if out_arr.shape != (B, H, S, D):
        out_arr = out_arr.reshape(B, H, S, D)
    return out_arr


def kda_delta_rule_fwd(q, k, v, g, beta, *, chunk_size, out=None):
    """Chunked delta-rule forward (KDA L2..L6 pipeline).

    Orchestrates the gate log-cumsum (L2), the within-chunk delta-corrected
    value solve (L4), the cross-chunk state propagation (L5) and the final
    intra+inter output combine (L6). ``chunk_size`` must divide ``S``. The
    result matches :func:`kda_naive_delta_rule_fwd` within fp32 round-off
    scaled by the sequence length.

    Args:
        q, k, v: float32 ``(B, H, S, D)``.
        g, beta: float32 ``(B, H, S)``.
        chunk_size: per-chunk token count (must divide ``S``).
        out: optional writable float32 ``(B, H, S, D)`` buffer; a new
            ``(B, H, S, D)`` array is allocated when omitted.

    Returns:
        The output array (``out`` if given, otherwise a new
        ``(B, H, S, D)`` float32 array).
    """
    B, H, S, D = _kda_dims(q, g)
    q_arr = _as_input(q, "q")
    k_arr = _as_input(k, "k")
    v_arr = _as_input(v, "v")
    g_arr = _as_input(g, "g")
    beta_arr = _as_input(beta, "beta")
    out_arr = _as_out(B * H * S * D, out, "out")
    _impl.kda_delta_rule_fwd(
        q_arr, k_arr, v_arr, g_arr, beta_arr, int(B), int(H), int(S), int(D),
        int(chunk_size), out_arr,
    )
    if out_arr.shape != (B, H, S, D):
        out_arr = out_arr.reshape(B, H, S, D)
    return out_arr


def kda_delta_rule_intra(
    q, k, v, g, beta, intra_log, inter_state, *, u=None, chunk_size, chunk_idx
):
    """Within-chunk delta-corrected value solve (KDA L4), one chunk.

    For token ``t`` in chunk ``chunk_idx``:
    ``u_t = v_t - G_{0,t-1}(C_{c-1} k_t) - sum_{j<t} G_{j+1,t-1} b_j (k_j.k_t) u_j``
    where ``C_{c-1}`` is read from ``inter_state[..., chunk_idx]``.

    This is a low-level stage of the chunked pipeline, exposed so the HIP
    kernel's per-stage output can be cross-checked. For the full layer use
    :func:`kda_delta_rule_fwd`; to call the stages manually you must pass the
    SAME persistent ``u`` and ``inter_state`` buffers across ``chunk_idx``.

    Args:
        q, k, v: float32 ``(B, H, S, D)``.
        g, beta: float32 ``(B, H, S)``.
        intra_log: float32 ``(B, H, n_chunks, chunk_size)`` (from
            :func:`kda_gate_chunk_cumsum`).
        inter_state: float32 ``(B, H, n_chunks+1, D, D)``; row ``chunk_idx``
            (``C_{c-1}``) is read.
        u: optional writable float32 ``(B, H, S, D)`` buffer; a zeroed array
            is allocated when omitted (correct for a single chunk).
        chunk_size: per-chunk token count (must divide ``S``).
        chunk_idx: which chunk to solve (``0 <= chunk_idx*chunk_size < S``).

    Returns:
        The ``u`` array (``u`` if given, otherwise a new zeroed
        ``(B, H, S, D)`` float32 array).
    """
    B, H, S, D = _kda_dims(q, g)
    q_arr = _as_input(q, "q")
    k_arr = _as_input(k, "k")
    v_arr = _as_input(v, "v")
    g_arr = _as_input(g, "g")
    beta_arr = _as_input(beta, "beta")
    intra_arr = _as_input(intra_log, "intra_log")
    inter_arr = _as_input(inter_state, "inter_state")
    n_chunks = _require_chunk(S, chunk_size)
    if intra_arr.shape != (B, H, n_chunks, int(chunk_size)):
        raise ValueError(
            f"intra_log must be [{B}, {H}, {n_chunks}, {chunk_size}], got {intra_arr.shape}"
        )
    if inter_arr.shape != (B, H, n_chunks + 1, D, D):
        raise ValueError(
            f"inter_state must be [{B}, {H}, {n_chunks + 1}, {D}, {D}], got {inter_arr.shape}"
        )
    if not (0 <= int(chunk_idx) * int(chunk_size) < S):
        raise ValueError("chunk_idx out of range")
    u_arr = _as_out_any(u, _F32, B * H * S * D, "u", zero=True)
    _impl.kda_delta_rule_intra(
        q_arr, k_arr, v_arr, g_arr, beta_arr, intra_arr, inter_arr, u_arr,
        int(B), int(H), int(S), int(D), int(chunk_size), int(chunk_idx),
    )
    if u_arr.shape != (B, H, S, D):
        u_arr = u_arr.reshape(B, H, S, D)
    return u_arr


def kda_delta_rule_inter(
    k, v, g, beta, intra_log, u, *, inter_state=None, chunk_size, chunk_idx
):
    """Cross-chunk state propagation (KDA L5), one chunk.

    Fills ``inter_state[..., chunk_idx+1] = C_c`` where
    ``C_c = G_{0,C-1} C_{c-1} + sum_t G_{t+1,C-1} b_t u_t k_t^T``. The read
    row ``C_{c-1}`` is ``inter_state[..., chunk_idx]`` (must be zero for
    ``chunk_idx == 0``).

    Low-level pipeline stage — see :func:`kda_delta_rule_intra` for the
    buffer-ownership contract and :func:`kda_delta_rule_fwd` for the whole
    layer.

    Args:
        k, v: float32 ``(B, H, S, D)`` (``v`` is accepted for signature
            symmetry but unused by this stage).
        g, beta: float32 ``(B, H, S)``.
        intra_log: float32 ``(B, H, n_chunks, chunk_size)``.
        u: float32 ``(B, H, S, D)`` (from :func:`kda_delta_rule_intra`).
        inter_state: optional writable float32
            ``(B, H, n_chunks+1, D, D)`` buffer; a zeroed array is allocated
            when omitted (correct only for ``chunk_idx == 0``).
        chunk_size: per-chunk token count (must divide ``S``).
        chunk_idx: which chunk to propagate (``0 <= chunk_idx*chunk_size < S``).

    Returns:
        The ``inter_state`` array (``inter_state`` if given, otherwise a new
        zeroed ``(B, H, n_chunks+1, D, D)`` float32 array).
    """
    B, H, S = _kda_dims3(g, beta)
    D = np.asarray(u).shape[-1]
    k_arr = _as_input(k, "k")
    v_arr = _as_input(v, "v")
    g_arr = _as_input(g, "g")
    beta_arr = _as_input(beta, "beta")
    u_arr = _as_input(u, "u")
    intra_arr = _as_input(intra_log, "intra_log")
    n_chunks = _require_chunk(S, chunk_size)
    if u_arr.shape != (B, H, S, D):
        raise ValueError(f"u must be [{B}, {H}, {S}, {D}], got {u_arr.shape}")
    if intra_arr.shape != (B, H, n_chunks, int(chunk_size)):
        raise ValueError(
            f"intra_log must be [{B}, {H}, {n_chunks}, {chunk_size}], got {intra_arr.shape}"
        )
    if not (0 <= int(chunk_idx) * int(chunk_size) < S):
        raise ValueError("chunk_idx out of range")
    inter_arr = _as_out_any(
        inter_state, _F32, B * H * (n_chunks + 1) * D * D, "inter_state", zero=True
    )
    _impl.kda_delta_rule_inter(
        k_arr, v_arr, g_arr, beta_arr, intra_arr, u_arr, inter_arr, int(B),
        int(H), int(S), int(D), int(chunk_size), int(chunk_idx),
    )
    if inter_arr.shape != (B, H, n_chunks + 1, D, D):
        inter_arr = inter_arr.reshape(B, H, n_chunks + 1, D, D)
    return inter_arr


def kda_gla_fwd_o(q, k, g, beta, intra_log, inter_state, u, *, out=None, chunk_size):
    """Output (intra + inter) combine (KDA L6).

    ``o_t = G_{0,t}(C_{c-1} q_t) + sum_{j<=t} G_{j+1,t} b_j (k_j.q_t) u_j``
    where ``C_{c-1}`` is ``inter_state[..., c]`` for ``t`` in chunk ``c``.

    Low-level pipeline stage — see :func:`kda_delta_rule_fwd` for the whole
    layer.

    Args:
        q, k: float32 ``(B, H, S, D)``.
        g, beta: float32 ``(B, H, S)``.
        intra_log: float32 ``(B, H, n_chunks, chunk_size)``.
        inter_state: float32 ``(B, H, n_chunks+1, D, D)``.
        u: float32 ``(B, H, S, D)``.
        out: optional writable float32 ``(B, H, S, D)`` buffer; a new
            ``(B, H, S, D)`` array is allocated when omitted.
        chunk_size: per-chunk token count (must divide ``S``).

    Returns:
        The output array (``out`` if given, otherwise a new
        ``(B, H, S, D)`` float32 array).
    """
    B, H, S, D = _kda_dims(q, g)
    q_arr = _as_input(q, "q")
    k_arr = _as_input(k, "k")
    g_arr = _as_input(g, "g")
    beta_arr = _as_input(beta, "beta")
    intra_arr = _as_input(intra_log, "intra_log")
    inter_arr = _as_input(inter_state, "inter_state")
    u_arr = _as_input(u, "u")
    n_chunks = _require_chunk(S, chunk_size)
    if intra_arr.shape != (B, H, n_chunks, int(chunk_size)):
        raise ValueError(
            f"intra_log must be [{B}, {H}, {n_chunks}, {chunk_size}], got {intra_arr.shape}"
        )
    if inter_arr.shape != (B, H, n_chunks + 1, D, D):
        raise ValueError(
            f"inter_state must be [{B}, {H}, {n_chunks + 1}, {D}, {D}], got {inter_arr.shape}"
        )
    out_arr = _as_out(B * H * S * D, out, "out")
    _impl.kda_gla_fwd_o(
        q_arr, k_arr, g_arr, beta_arr, intra_arr, inter_arr, u_arr, int(B),
        int(H), int(S), int(D), int(chunk_size), out_arr,
    )
    if out_arr.shape != (B, H, S, D):
        out_arr = out_arr.reshape(B, H, S, D)
    return out_arr


def kda_pack_bitmatrix(bits, *, n_bits=None):
    """Pack a binary bit array into bytes, MSB-first (KDA ``pack_bitmatrix``).

    Bit ``k`` (``0`` or ``1``) is written to byte ``k//8``, bit
    ``7 - k%8``. The result has ``ceil(n_bits/8)`` bytes; trailing bits of
    the final byte are zero.

    Args:
        bits: uint8 array (each element ``0`` or ``1``).
        n_bits: number of leading bits to pack (default ``bits.size``; must
            not exceed it).

    Returns:
        A uint8 array of length ``ceil(n_bits/8)``.

    Raises:
        ValueError: if ``n_bits`` exceeds ``bits.size``.

    Example:
        >>> list(kda_pack_bitmatrix(np.array([1, 0, 1, 1, 0, 0, 0, 1],
        ...                                dtype=np.uint8)))
        [177]
    """
    bits_arr = np.ascontiguousarray(bits, dtype=np.uint8).ravel()
    n = bits_arr.size if n_bits is None else int(n_bits)
    if n < 0:
        raise ValueError("n_bits must be non-negative")
    if n > bits_arr.size:
        raise ValueError(f"n_bits ({n}) exceeds bits.size ({bits_arr.size})")
    return _impl.kda_pack_bitmatrix(bits_arr, n)
