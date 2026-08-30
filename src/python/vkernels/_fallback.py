"""Pure-Python reference implementations of the vkernels API.

This module mirrors the CPU reference implementations in ``src/c/vkernels``
(the "oracle" side of the repository's two-implementation model) so the
Python package remains fully importable and testable on machines where the
compiled backend (:mod:`vkernels._core`) was not built. It is *not* the
primary path: when the extension is available the public API dispatches to it
instead.

Faithfulness notes
------------------
* All arithmetic is done in ``float32`` with the same operation order as the
  C++ references, so results are bit-identical for element-wise ops, GEMM
  and reductions (sequential accumulation for ``sum``).
* Contract violations raise ``ValueError`` exactly like the C++
  ``VK_EXPECTS`` checks, which pybind11 maps to ``ValueError`` in the
  compiled backend.
* ``Stream`` uses one worker thread per stream, mirroring the host C++
  ``Stream`` semantics (in-order tasks, concurrent across streams).
"""

from __future__ import annotations

import ctypes
import math
import threading
from collections import deque

import numpy as np

from vkernels._types import Gather2DRun, Result, StagedRun1D, StagedRun2D, Topology

# ---------------------------------------------------------------------------
# core: Device / Stream
# ---------------------------------------------------------------------------


class Device:
    """Host-only device: index storage, every operation is a no-op."""

    def __init__(self, index: int = -1):
        self._index = index

    def index(self) -> int:
        return self._index

    def set_current(self) -> None:
        pass

    def sync(self) -> None:
        pass

    def supports_peer(self, other) -> bool:
        return False

    def __eq__(self, other) -> bool:
        return isinstance(other, Device) and self._index == other._index

    def __repr__(self) -> str:
        return f"Device(index={self._index})"


def default_device() -> Device:
    return Device()


class Stream:
    """Ordered, asynchronous queue of tasks backed by one worker thread.

    Mirrors the host C++ ``vkernels::Stream``: tasks run in submission
    order; distinct streams run concurrently.
    """

    def __init__(self):
        self._queue: deque = deque()
        self._cv = threading.Condition()
        self._outstanding = 0
        self._total = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._stop or self._queue)
                if self._stop and not self._queue:
                    return
                task = self._queue.popleft()
            task()
            with self._cv:
                self._outstanding -= 1
                if self._outstanding == 0:
                    self._cv.notify_all()

    def submit(self, task) -> None:
        with self._cv:
            self._queue.append(task)
            self._outstanding += 1
            self._total += 1
            self._cv.notify()

    def wait(self) -> None:
        with self._cv:
            self._cv.wait_for(lambda: self._outstanding == 0)

    def submitted(self) -> int:
        with self._cv:
            return self._total

    def close(self) -> None:
        with self._cv:
            if self._stop:
                return
            self._stop = True
            self._cv.notify()
        self._thread.join()


# ---------------------------------------------------------------------------
# kernels: elementwise, reduce, gemm
# ---------------------------------------------------------------------------


def add(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
    if a.size != b.size or a.size != out.size:
        raise ValueError("a and b must have equal length")
    np.add(a, b, out=out)


def scale(x: np.ndarray, alpha, out: np.ndarray) -> None:
    if x.size != out.size:
        raise ValueError("x and out must have equal length")
    np.multiply(np.float32(alpha), x, out=out)


def relu(x: np.ndarray, out: np.ndarray) -> None:
    if x.size != out.size:
        raise ValueError("x and out must have equal length")
    np.maximum(x, np.float32(0.0), out=out)


def sum(x: np.ndarray) -> float:
    if x.size == 0:
        raise ValueError("cannot reduce an empty span")
    acc = np.float32(0.0)
    for v in x.flat:
        acc = np.float32(acc + v)
    return float(acc)


def max(x: np.ndarray) -> float:
    if x.size == 0:
        raise ValueError("cannot reduce an empty span")
    m = np.float32(x.flat[0])
    for v in x.flat[1:]:
        if v > m:
            m = v
    return float(m)


def gemm(
    M: int, N: int, K: int, alpha, A: np.ndarray, B: np.ndarray, beta, C: np.ndarray
) -> None:
    if A.size != M * K:
        raise ValueError("A must be M*K")
    if B.size != K * N:
        raise ValueError("B must be K*N")
    if C.size != M * N:
        raise ValueError("C must be M*N")
    alpha = np.float32(alpha)
    beta = np.float32(beta)
    a = A.flat
    b = B.flat
    c = C.flat
    for i in range(M):
        for j in range(N):
            acc = np.float32(0.0)
            for k in range(K):
                acc = np.float32(acc + a[i * K + k] * b[k * N + j])
            c[i * N + j] = np.float32(alpha * acc + beta * c[i * N + j])


# --- moe: fp4 dequant, LDS fill, MFMA ---------------------------------------


def direct_lds_fill_bf16(lds_dst: int, global_src: int, elements: int) -> None:
    """Copy `elements` bf16 values from global memory to LDS.

    Uses a plain ctypes memmove; on the host this is the reference for the
    HIP vectorised-loads + LDS-stores path.
    """
    if elements == 0:
        return
    if lds_dst == 0:
        raise ValueError("lds_dst must not be null")
    if global_src == 0:
        raise ValueError("global_src must not be null")
    nbytes = elements * 2  # bf16 = 2 bytes
    ctypes.memmove(ctypes.c_void_p(lds_dst), ctypes.c_void_p(global_src), nbytes)


# fp4 E2M1 representable values, indexed by nibble.
# Nibble layout: [s:1][e1:1][e0:1][m:1] → 4-bit value = s*8 + e*2 + m.
# Positive values (s=0) occupy indices 0-7; negative (s=1) occupy 8-15.
_FP4_NIBBLE_VALUES: list[float] = [
    # s=0, e=0: zero / subnormal (m=1 → 0.25)
    0.0,
    0.25,
    # s=0, e=1: normal ×2^(1-1)= ×1
    1.0,
    1.5,
    # s=0, e=2: normal ×2^(2-1)= ×2
    2.0,
    3.0,
    # s=0, e=3: inf / NaN
    float("inf"),
    float("nan"),
    # s=1, e=0: negative zero / subnormal
    -0.0,
    -0.25,
    # s=1, e=1: negative normal
    -1.0,
    -1.5,
    # s=1, e=2: negative normal
    -2.0,
    -3.0,
    # s=1, e=3: negative inf / NaN
    -float("inf"),
    float("nan"),
]


def _float_to_bf16_bits(f: float) -> int:
    """Round a float32 to bf16 and return the uint16 bit pattern."""
    bits = int.from_bytes(np.float32(f).tobytes(), "little")
    # Round-to-nearest-even: add half-ulp of the truncated 16 LSBs.
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFFFFFF
    return (bits >> 16) & 0xFFFF


def fp4_to_bf16_dequant(packed: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Convert packed fp4 (E2M1, two values per byte, low nibble first) to
    bf16 (uint16 bit patterns).

    Args:
        packed: uint8 array of packed fp4 values.
        scale: per-block scale factor (default 1.0).

    Returns:
        uint16 array of bf16 bit patterns, length = 2 × len(packed).
    """
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty(packed.size * 2, dtype=np.uint16)
    scale_f = float(scale)
    for i in range(packed.size):
        byte = int(packed[i])
        lo_val = _FP4_NIBBLE_VALUES[byte & 0x0F] * scale_f
        hi_val = _FP4_NIBBLE_VALUES[(byte >> 4) & 0x0F] * scale_f
        out[i * 2] = _float_to_bf16_bits(lo_val)
        out[i * 2 + 1] = _float_to_bf16_bits(hi_val)
    return out


def use_async_copy_default() -> bool:
    """Return True if async copy should be used by default.

    On gfx942 (CDNA3) it misbehaves and defaults to OFF; elsewhere ON.
    The K3_NO_ASYNC env var overrides: '0'=ON, '1'=OFF.
    On the host fallback path this always returns True.
    """
    import os

    env = os.environ.get("K3_NO_ASYNC", "")
    if env == "1":
        return False
    return True


def mfma_f32_16x16x16bf16(
    c: list[float],
    a: list[int],
    b: list[int],
    cbsz: int = 0,
    abid: int = 0,
    blgp: int = 0,
) -> None:
    """K16 bf16 MFMA: C[0..3] += A[0..1] × B[0..1] (16×16×16 bf16, acc fp32).

    `c` is a list of 4 floats updated in-place.
    `a` and `b` are lists of 2 uint32_t each, packing 2 bf16 values per
    uint32_t (low 16 bits, high 16 bits).

    The control flags (cbsz, abid, blgp) are accepted but ignored on the
    host reference path.
    """
    if len(c) < 4 or len(a) < 2 or len(b) < 2:
        raise ValueError("c must have 4 floats; a and b must each have 2 uint32_t")

    # Unpack a[2] → 4 bf16 values, b[2] → 4 bf16 values.
    def _unpack_bf16(v32: int) -> tuple[float, float]:
        lo_bits = (v32 & 0xFFFF) << 16
        hi_bits = (v32 >> 16) << 16
        lo = np.frombuffer(lo_bits.to_bytes(4, "little"), dtype=np.float32)[0]
        hi = np.frombuffer(hi_bits.to_bytes(4, "little"), dtype=np.float32)[0]
        return float(lo), float(hi)

    a_lo, a_hi = _unpack_bf16(int(a[0]))
    a2_lo, a2_hi = _unpack_bf16(int(a[1]))
    b_lo, b_hi = _unpack_bf16(int(b[0]))
    b2_lo, b2_hi = _unpack_bf16(int(b[1]))

    a_f32 = [a_lo, a_hi, a2_lo, a2_hi]
    b_f32 = [b_lo, b_hi, b2_lo, b2_hi]

    # Per-thread dot: C[i] += A[i] × B[i]
    for i in range(4):
        c[i] = float(np.float32(c[i]) + np.float32(a_f32[i] * b_f32[i]))


# --- moe fused: expert alignment + grouped GEMM ---------------------------------


def _ue8m0_to_float(s: int) -> np.float32:
    """Decode a ue8m0 scale byte to float32 (0xFF → 0.0, else 2^(s-127))."""
    if s == 0xFF:
        return np.float32(0.0)
    unbiased = s - 127
    if unbiased >= -126:
        bits = s << 23
        return np.frombuffer(bits.to_bytes(4, "little"), dtype=np.float32)[0]
    shift = -(unbiased + 126)
    if shift < 32:
        mant = 0x00800000 >> shift
        return np.frombuffer(mant.to_bytes(4, "little"), dtype=np.float32)[0]
    return np.float32(0.0)


def _bf16_to_float(bits: int) -> np.float32:
    """Reinterpret a bf16 uint16 bit pattern as float32."""
    return np.frombuffer((bits << 16).to_bytes(4, "little"), dtype=np.float32)[0]


def _dequant_weight_tile(
    packed, scale, p_base, s_base, N, K, group_size, stride_packed, stride_scale_n
):
    """Dequant a [N, K] packed fp4 + ue8m0 tile into out[K, N] bf16.

    Mirrors ``dequant_weight_tile`` in moe_fused.cpp: the result is rounded
    to bf16 (RNE) exactly once, indexed transposed (out[k][n]).
    """
    out = np.empty((K, N), dtype=np.uint16)
    for n in range(N):
        for kp in range(K // 2):
            pb = int(packed[p_base + n * stride_packed + kp])
            gi = (kp * 2) // group_size
            sc = int(scale[s_base + n * stride_scale_n + gi])
            s = _ue8m0_to_float(sc)
            lo = np.float32(_FP4_NIBBLE_VALUES[pb & 0x0F]) * s
            hi = np.float32(_FP4_NIBBLE_VALUES[(pb >> 4) & 0x0F]) * s
            out[kp * 2, n] = _float_to_bf16_bits(float(lo))
            out[kp * 2 + 1, n] = _float_to_bf16_bits(float(hi))
    return out


def moe_align_block_size(topk_ids, M, top_k, block_size, num_experts):
    """Map the flat [M*top_k] token→expert routing to block-aligned
    ``sorted_ids`` / ``expert_ids``. Returns (sorted_ids, expert_ids, EM)."""
    ids = np.asarray(topk_ids, dtype=np.int32).ravel()
    if ids.size != M * top_k:
        raise ValueError("topk_ids must have M*top_k elements")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    N = M * top_k
    per_expert = [[] for _ in range(num_experts)]
    for i in range(N):
        e = int(ids[i])
        if 0 <= e < num_experts:
            per_expert[e].append(i)  # flat index

    EM = 0
    for v in per_expert:
        EM += ((len(v) + block_size - 1) // block_size) * block_size
    sorted_ids = np.empty(EM, dtype=np.int32)
    idx = 0
    for e in range(num_experts):
        tokens = per_expert[e]
        nt = len(tokens)
        for t in tokens:
            sorted_ids[idx] = t
            idx += 1
        padded = ((nt + block_size - 1) // block_size) * block_size
        for _ in range(nt, padded):
            sorted_ids[idx] = N  # padding sentinel
            idx += 1

    num_blocks = EM // block_size
    expert_ids = np.full(num_blocks, -1, dtype=np.int32)
    idx = 0
    for e in range(num_experts):
        nt = len(per_expert[e])
        padded_blocks = (nt + block_size - 1) // block_size
        for b in range(padded_blocks):
            expert_ids[idx] = e if b * block_size < nt else -1
            idx += 1
    return sorted_ids, expert_ids, EM


def fused_moe_mxfp4(
    A,
    w13,
    w13_scale,
    w2,
    w2_scale,
    sorted_ids,
    topk_w,
    expert_ids,
    act_scratch,
    out,
    M,
    hidden,
    ispp,
    top_k,
    EM,
    group_size,
    swiglu_limit,
    activation=0,
    beta=4.0,
    linear_beta=25.0,
    b13=None,
    b2=None,
):
    """Fused MXFP4 MoE grouped GEMM (CPU-reference oracle, in place).

    Writes the activation intermediate into ``act_scratch`` [EM*ispp] bf16
    (SwiGLU by default, or Kimi-K3 SiTU via ``activation=1``) and accumulates
    into ``out`` [M*hidden] fp32 (which the caller must zero-initialise).
    Mirrors ``fused_moe_mxfp4_cpu`` block for block.
    """
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 64
    N = M * top_k

    A = np.asarray(A, dtype=np.uint16).ravel()
    w13 = np.asarray(w13, dtype=np.uint8).ravel()
    w13_scale = np.asarray(w13_scale, dtype=np.uint8).ravel()
    w2 = np.asarray(w2, dtype=np.uint8).ravel()
    w2_scale = np.asarray(w2_scale, dtype=np.uint8).ravel()
    sorted_ids = np.asarray(sorted_ids, dtype=np.int32).ravel()
    topk_w = np.asarray(topk_w, dtype=np.float32).ravel()
    expert_ids = np.asarray(expert_ids, dtype=np.int32).ravel()
    act = np.asarray(act_scratch, dtype=np.uint16).ravel()
    out_arr = np.asarray(out, dtype=np.float32).ravel()
    b13 = np.asarray(b13, dtype=np.float32).ravel() if b13 is not None else None
    b2 = np.asarray(b2, dtype=np.float32).ravel() if b2 is not None else None

    def _f2bf(v: float) -> np.uint16:
        return np.uint16(_float_to_bf16_bits(v))

    # ===== Stage 0: gate_up + activation → act [EM, ispp] =====
    w13_expert_bytes = 2 * ispp * (hidden // 2)
    w13s_expert_bytes = 2 * ispp * (hidden // group_size)
    num_m_blocks = EM // BLOCK_M
    num_n_blocks = ispp // BLOCK_N
    num_k_blocks = hidden // BLOCK_K

    for mb in range(num_m_blocks):
        expert = int(expert_ids[mb])
        if expert < 0:
            continue
        token_base = mb * BLOCK_M
        for nb in range(num_n_blocks):
            acc_gate = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)
            acc_up = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)
            for kb in range(num_k_blocks):
                k_start = kb * BLOCK_K
                tile_A = np.zeros((BLOCK_M, BLOCK_K), dtype=np.uint16)
                for m in range(BLOCK_M):
                    flat = int(sorted_ids[token_base + m])
                    if flat < N:
                        token = flat // top_k
                        tile_A[m] = A[
                            token * hidden + k_start : token * hidden
                            + k_start
                            + BLOCK_K
                        ]

                # gate half
                p_gate = expert * w13_expert_bytes + nb * BLOCK_N * (hidden // 2)
                s_gate = expert * w13s_expert_bytes + nb * BLOCK_N * (
                    hidden // group_size
                )
                tile_gate = _dequant_weight_tile(
                    w13,
                    w13_scale,
                    p_gate + k_start // 2,
                    s_gate + k_start // group_size,
                    BLOCK_N,
                    BLOCK_K,
                    group_size,
                    hidden // 2,
                    hidden // group_size,
                )
                # up half (offset by ispp rows in w13)
                p_up = expert * w13_expert_bytes + (nb * BLOCK_N + ispp) * (hidden // 2)
                s_up = expert * w13s_expert_bytes + (nb * BLOCK_N + ispp) * (
                    hidden // group_size
                )
                tile_up = _dequant_weight_tile(
                    w13,
                    w13_scale,
                    p_up + k_start // 2,
                    s_up + k_start // group_size,
                    BLOCK_N,
                    BLOCK_K,
                    group_size,
                    hidden // 2,
                    hidden // group_size,
                )

                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        dg = np.float32(0.0)
                        du = np.float32(0.0)
                        for k in range(BLOCK_K):
                            a = _bf16_to_float(int(tile_A[m, k]))
                            dg = np.float32(
                                dg
                                + np.float32(a * _bf16_to_float(int(tile_gate[k, n])))
                            )
                            du = np.float32(
                                du + np.float32(a * _bf16_to_float(int(tile_up[k, n])))
                            )
                        acc_gate[m, n] = np.float32(acc_gate[m, n] + dg)
                        acc_up[m, n] = np.float32(acc_up[m, n] + du)

            # Activation epilogue (skip padding rows)
            for m in range(BLOCK_M):
                flat = int(sorted_ids[token_base + m])
                if flat >= N:
                    continue
                for n in range(BLOCK_N):
                    g = float(acc_gate[m, n])
                    u = float(acc_up[m, n])
                    if b13 is not None:
                        g += float(b13[expert * 2 * ispp + nb * BLOCK_N + n])
                        u += float(b13[expert * 2 * ispp + nb * BLOCK_N + n + ispp])
                    if activation == 1:
                        # Kimi-K3 SiTU (situ_and_mul): no clamp.
                        sig = 1.0 / (1.0 + np.exp(-g))
                        gate_out = beta * np.tanh(g / beta) * sig
                        up_out = (
                            linear_beta * np.tanh(u / linear_beta)
                            if linear_beta > 0.0
                            else u
                        )
                        result = gate_out * up_out
                    else:
                        if swiglu_limit > 0.0:
                            g = min(g, swiglu_limit)
                            u = min(u, swiglu_limit)
                            u = -min(-u, swiglu_limit)  # max(u, -limit)
                        silu_g = g / (1.0 + np.exp(-g))
                        result = silu_g * u
                    act[(token_base + m) * ispp + nb * BLOCK_N + n] = _f2bf(result)

    # ===== Stage 1: down + combine → out [M, hidden] =====
    w2_expert_bytes = hidden * (ispp // 2)
    w2s_expert_bytes = hidden * (ispp // group_size)
    num_down_n_blocks = hidden // BLOCK_N
    num_down_k_blocks = ispp // BLOCK_K

    for mb in range(num_m_blocks):
        expert = int(expert_ids[mb])
        if expert < 0:
            continue
        token_base = mb * BLOCK_M
        for nb in range(num_down_n_blocks):
            acc_down = np.zeros((BLOCK_M, BLOCK_N), dtype=np.float32)
            for kb in range(num_down_k_blocks):
                k_start = kb * BLOCK_K
                tile_A = np.zeros((BLOCK_M, BLOCK_K), dtype=np.uint16)
                for m in range(BLOCK_M):
                    flat = int(sorted_ids[token_base + m])
                    if flat < N:
                        tile_A[m] = act[
                            (token_base + m) * ispp + k_start : (token_base + m) * ispp
                            + k_start
                            + BLOCK_K
                        ]

                p_down = expert * w2_expert_bytes + nb * BLOCK_N * (ispp // 2)
                s_down = expert * w2s_expert_bytes + nb * BLOCK_N * (ispp // group_size)
                tile_down = _dequant_weight_tile(
                    w2,
                    w2_scale,
                    p_down + k_start // 2,
                    s_down + k_start // group_size,
                    BLOCK_N,
                    BLOCK_K,
                    group_size,
                    ispp // 2,
                    ispp // group_size,
                )

                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        dot = np.float32(0.0)
                        for k in range(BLOCK_K):
                            a = _bf16_to_float(int(tile_A[m, k]))
                            dot = np.float32(
                                dot
                                + np.float32(a * _bf16_to_float(int(tile_down[k, n])))
                            )
                        acc_down[m, n] = np.float32(acc_down[m, n] + dot)

            # Combine: bias + weight + scatter-add
            for m in range(BLOCK_M):
                flat = int(sorted_ids[token_base + m])
                if flat >= N:
                    continue
                token = flat // top_k
                weight = float(topk_w[token_base + m])
                for n in range(BLOCK_N):
                    val = float(acc_down[m, n])
                    if b2 is not None:
                        val += float(b2[expert * hidden + nb * BLOCK_N + n])
                    val *= weight
                    out_arr[token * hidden + nb * BLOCK_N + n] += np.float32(val)


# ---------------------------------------------------------------------------
# MoE orchestration (mxfp4_moe_aux): per-block quant, gather, scatter-reduce.
# Mirrors src/c/vkernels/kernels/moe_aux.cpp byte-for-byte so the pure-Python
# backend is a faithful oracle for the HIP kernels.
# ---------------------------------------------------------------------------
# Largest finite E2M1 value (matches _FP4_NIBBLE_VALUES[5] and moe.cpp).
_MXFP4_MAX = 3.0


def _bf16_to_float_vec(bits: np.ndarray) -> np.ndarray:
    """Reinterpret a uint16 bf16 array as float32 (zero-extend; exact)."""
    u = bits.astype(np.uint32) << 16
    return u.view(np.float32).copy()


def _float_to_fp4_nib_vec(x: np.ndarray) -> np.ndarray:
    """Round (|x| <= MXFP4_MAX) to the nearest E2M1 nibble code.

    Ties round to the LARGER magnitude, exactly like the C++ reference
    (``float_to_fp4_nib`` in moe_aux.cpp: first matching ``if / else if`` wins).
      |v| < 0.125 -> 0 (0.0); < 0.625 -> 1 (0.25); < 1.25 -> 2 (1.0);
      < 1.75 -> 3 (1.5); < 2.5 -> 4 (2.0); else -> 5 (3.0)."""
    v = np.abs(x)
    # np.select evaluates conditions left-to-right and returns the first
    # match, mirroring the C++ if/else-if chain (strict <, ties to larger).
    mag = np.select(
        [v < 0.125, v < 0.625, v < 1.25, v < 1.75, v < 2.5, True], [0, 1, 2, 3, 4, 5]
    ).astype(np.uint8)
    sign = np.where(x < 0.0, np.uint8(0x8), np.uint8(0))
    return (mag | sign).astype(np.uint8)


def mxfp4_moe_quant(A, M: int, hidden: int, group_size: int = 32):
    """MXFP4 activation quant — the inverse of the weight decode.

    Args:
        A: uint16 bf16 array of shape ``(M, hidden)``.
        M, hidden: dimensions (``hidden`` must be divisible by ``group_size``).
        group_size: ue8m0 scale group length (default 32).

    Returns:
        ``(packed, scales)`` where ``packed`` is ``[M, hidden/2]`` uint8 E2M1
        (low nibble = even K) and ``scales`` is ``[M, hidden/group_size]``
        uint8 ue8m0 (``2^(s-127)``; ``0xFF`` = zero group).
    """
    A_arr = np.ascontiguousarray(A, dtype=np.uint16)
    if A_arr.shape != (M, hidden):
        raise ValueError(f"A must be [{M}, {hidden}], got {A_arr.shape}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if hidden % group_size != 0:
        raise ValueError("hidden must be a multiple of group_size")
    if hidden % 2 != 0:
        raise ValueError("hidden must be even")

    n_groups = hidden // group_size
    x = _bf16_to_float_vec(A_arr).reshape(M, hidden)
    # Per-group amax (max absolute value over `group_size` consecutive cols).
    xg = x.reshape(M, n_groups, group_size)
    amax = np.max(np.abs(xg), axis=2)  # [M, n_groups]

    scales = np.empty((M, n_groups), dtype=np.uint8)
    packed = np.zeros((M, hidden // 2), dtype=np.uint8)

    finite = np.isfinite(amax) & (amax > 0.0)
    # e = ceil(log2(amax / MXFP4_MAX)); sb = clamp(e + 127, 0, 254).
    ratio = np.where(finite, amax / _MXFP4_MAX, 1.0)
    e = np.ceil(np.log2(ratio)).astype(np.int64)
    sb = np.clip(e + 127, 1, 254).astype(np.uint8)
    scales[:] = np.where(finite, sb, np.uint8(0xFF))

    # scale value = 2^(sb - 127); zero groups (0xFF) contribute scale = 0.
    scale_val = np.where(finite, (2.0 ** (sb.astype(np.int32) - 127)), 0.0)
    scale_val = scale_val.astype(np.float32)  # [M, n_groups]

    # Quantize each element: nibble = round(x / scale) to nearest E2M1.
    # Zero-scale groups (0xFF) contribute 0; suppress the divide-by-zero /
    # invalid-value warnings since np.where masks them anyway.
    sv = np.broadcast_to(scale_val[:, :, None], (M, n_groups, group_size))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(sv > 0.0, xg / sv, 0.0)
    nibs = _float_to_fp4_nib_vec(q)  # [M, n_groups, group_size] uint8
    lo = nibs[:, :, 0::2] & 0x0F
    hi = nibs[:, :, 1::2] & 0x0F
    packed = (hi << 4) | lo  # [M, n_groups, group_size/2] -> reshape
    packed = packed.reshape(M, hidden // 2)
    return packed, scales


def mxfp4_moe_sort(A, sorted_ids, M: int, hidden: int, top_k: int, EM: int):
    """Gather bf16 A [M, hidden] into sorted row order [EM, hidden]."""
    A_arr = np.ascontiguousarray(A, dtype=np.uint16)
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32).ravel()
    if A_arr.shape != (M, hidden):
        raise ValueError(f"A must be [{M}, {hidden}], got {A_arr.shape}")
    if ids.size != EM:
        raise ValueError(f"sorted_ids must have EM = {EM} elements")
    out = np.zeros((EM, hidden), dtype=np.uint16)
    for r in range(EM):
        flat = int(ids[r])
        if 0 <= flat < M * top_k:
            out[r] = A_arr[flat // top_k]
    return out.reshape(EM * hidden) if False else out.ravel()


def mxfp4_moe_sort_scales(
    scales, sorted_ids, M: int, n_groups: int, top_k: int, EM: int
):
    """Gather per-token ue8m0 scales [M, n_groups] into [EM, n_groups]."""
    s = np.ascontiguousarray(scales, dtype=np.uint8)
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32).ravel()
    if s.shape != (M, n_groups):
        raise ValueError(f"scales must be [{M}, {n_groups}], got {s.shape}")
    if ids.size != EM:
        raise ValueError(f"sorted_ids must have EM = {EM} elements")
    out = np.zeros((EM, n_groups), dtype=np.uint8)
    for r in range(EM):
        flat = int(ids[r])
        if 0 <= flat < M * top_k:
            out[r] = s[flat // top_k]
    return out.ravel()


def mxfp4_moe_scatter_reduce(
    partial, topk_w, sorted_ids, M: int, width: int, top_k: int, EM: int
):
    """Routed combine of float32 partials [EM, width] -> out [M, width]."""
    p = np.ascontiguousarray(partial, dtype=np.float32).reshape(EM, width)
    w = np.ascontiguousarray(topk_w, dtype=np.float32).ravel()
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32).ravel()
    if p.shape != (EM, width):
        raise ValueError(f"partial must be [{EM}, {width}]")
    if w.size != EM or ids.size != EM:
        raise ValueError("topk_w and sorted_ids must each have EM elements")
    out = np.zeros((M, width), dtype=np.float32)
    for r in range(EM):
        flat = int(ids[r])
        if 0 <= flat < M * top_k:
            np.add.at(out, flat // top_k, p[r] * w[r])
    return out.ravel()


def mxfp4_moe_scatter_reduce_q(
    partial_q,
    partial_s,
    topk_w,
    sorted_ids,
    M: int,
    width: int,
    top_k: int,
    EM: int,
    group_size: int = 32,
):
    """Routed combine of a quantized partial (E2M1 + ue8m0) -> out [M, width]."""
    pq = np.ascontiguousarray(partial_q, dtype=np.uint8).reshape(EM, width // 2)
    ps = np.ascontiguousarray(partial_s, dtype=np.uint8).reshape(
        EM, width // group_size
    )
    w = np.ascontiguousarray(topk_w, dtype=np.float32).ravel()
    ids = np.ascontiguousarray(sorted_ids, dtype=np.int32).ravel()
    if pq.shape != (EM, width // 2):
        raise ValueError(f"partial_q must be [{EM}, {width // 2}]")
    if ps.shape != (EM, width // group_size):
        raise ValueError(f"partial_s must be [{EM}, {width // group_size}]")
    if w.size != EM or ids.size != EM:
        raise ValueError("topk_w and sorted_ids must each have EM elements")
    n_groups = width // group_size
    out = np.zeros((M, width), dtype=np.float32)
    for r in range(EM):
        flat = int(ids[r])
        if not (0 <= flat < M * top_k):
            continue
        token = flat // top_k
        wr = float(w[r])
        for g in range(n_groups):
            sc = _ue8m0_to_float(int(ps[r, g]))
            base = g * group_size
            for i in range(0, group_size, 2):
                byte = int(pq[r, base // 2 + i // 2])
                lo = float(np.float32(_FP4_NIBBLE_VALUES[byte & 0x0F]) * sc)
                hi = float(np.float32(_FP4_NIBBLE_VALUES[(byte >> 4) & 0x0F]) * sc)
                out[token, base + i] += np.float32(lo * wr)
                out[token, base + i + 1] += np.float32(hi * wr)
    return out.ravel()


# ---------------------------------------------------------------------------
# bf16 GEMM (gfx942 projection reference, issue #29)
# ---------------------------------------------------------------------------


def gemm_bf16(M, N, K, alpha, A, B, beta, C):
    """C = alpha * A @ B + beta * C, bf16 (uint16) in/out, fp32 accumulate.

    Mirrors ``gemm_bf16_cpu`` exactly: a sequential k-accumulation in
    float32 and a single round-to-nearest-even on store (``beta == 0`` skips
    the prior-C read). Intended for small shapes — the O(M*N*K) loop is the
    bit-faithful reference the K3 HIP MFMA kernel is checked against.
    """
    A = np.asarray(A, dtype=np.uint16).ravel()
    B = np.asarray(B, dtype=np.uint16).ravel()
    C = np.asarray(C, dtype=np.uint16).ravel()
    if A.size != M * K:
        raise ValueError("A must be M*K")
    if B.size != K * N:
        raise ValueError("B must be K*N")
    if C.size != M * N:
        raise ValueError("C must be M*N")
    a = np.float32(alpha)
    b = np.float32(beta)
    read_prev = b != np.float32(0.0)
    for i in range(int(M)):
        for j in range(int(N)):
            acc = np.float32(0.0)
            for k in range(int(K)):
                av = np.float32(_bf16_to_float(int(A[i * K + k])))
                bv = np.float32(_bf16_to_float(int(B[k * N + j])))
                acc = np.float32(acc + av * bv)
            prev = (
                np.float32(_bf16_to_float(int(C[i * N + j])))
                if read_prev
                else np.float32(0.0)
            )
            C[i * N + j] = _float_to_bf16_bits(float(np.float32(a * acc + b * prev)))
    return C


def gemm_bf16_config(M, N, K):
    """Per-shape (bm, bn, bk, threads) MFMA tile selector (mirrors
    ``gemm_bf16_config_for``; BK is fixed at 64 for the K3 shapes)."""
    bk = 64
    if M <= 64:
        return (16, 16, bk, 64)  # serving / decode: 1 wavefront
    return (64, 64, bk, 256)  # warmup / prefill: 4 wavefronts


# ---------------------------------------------------------------------------
# MLA: absorbed-form Multi-head Latent Attention (issue #21)
# ---------------------------------------------------------------------------


def mla_fwd(B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank,
            qk_rope_head_dim, scale, q, k_c, k_pe, v_c, out):
    """Absorbed-form MLA forward, fp32 two-pass stable softmax (mirrors
    ``mla_fwd_cpu``). q [B,H,S_q,lr+rhd], k_c [B,S_kv,lr], k_pe [B,S_kv,rhd],
    v_c [B,S_kv,lr] -> out [B,H,S_q,lr], all float32, written in place.
    """
    D_q = kv_lora_rank + qk_rope_head_dim
    q = np.ascontiguousarray(q, dtype=np.float32).ravel()
    k_c = np.ascontiguousarray(k_c, dtype=np.float32).ravel()
    k_pe = np.ascontiguousarray(k_pe, dtype=np.float32).ravel()
    v_c = np.ascontiguousarray(v_c, dtype=np.float32).ravel()
    out = np.asarray(out, dtype=np.float32).ravel()
    if q.size != B * H * S_q * D_q:
        raise ValueError("q must be B*H*S_q*(lr+rhd)")
    if k_c.size != B * S_kv * kv_lora_rank:
        raise ValueError("k_c must be B*S_kv*lr")
    if k_pe.size != B * S_kv * qk_rope_head_dim:
        raise ValueError("k_pe must be B*S_kv*rhd")
    if v_c.size != B * S_kv * kv_lora_rank:
        raise ValueError("v_c must be B*S_kv*lr")
    if out.size != B * H * S_q * kv_lora_rank:
        raise ValueError("out must be B*H*S_q*lr")
    if B == 0 or H == 0 or S_q == 0 or S_kv == 0:
        return out
    sc = np.float32(scale)
    qr = q.reshape(B, H, S_q, D_q)
    kc = k_c.reshape(B, S_kv, kv_lora_rank)
    kp = k_pe.reshape(B, S_kv, qk_rope_head_dim)
    vc = v_c.reshape(B, S_kv, kv_lora_rank)
    oh = out.reshape(B, H, S_q, kv_lora_rank)
    lr = kv_lora_rank
    rhd = qk_rope_head_dim
    for b in range(B):
        for h in range(H):
            for i in range(S_q):
                gqi = q_start + i
                qi = qr[b, h, i]
                qn = qi[:lr]
                qrp = qi[lr:]
                s = np.full(S_kv, -np.inf, dtype=np.float32)
                mx = -np.inf
                for j in range(S_kv):
                    if kv_start + j > gqi:
                        continue
                    dot = np.float32(0.0)
                    for d in range(lr):
                        dot = np.float32(dot + qn[d] * kc[b, j, d])
                    for d in range(rhd):
                        dot = np.float32(dot + qrp[d] * kp[b, j, d])
                    sj = np.float32(sc * dot)
                    s[j] = sj
                    if sj > mx:
                        mx = sj
                if mx == -np.inf:  # every key masked
                    oh[b, h, i, :] = np.float32(0.0)
                    continue
                sumv = np.float32(0.0)
                w = np.empty(S_kv, dtype=np.float32)
                for j in range(S_kv):
                    if s[j] == -np.inf:
                        continue
                    w[j] = np.float32(math.exp(float(s[j] - mx)))
                    sumv = np.float32(sumv + w[j])
                inv = np.float32(1.0 / sumv)
                oi = np.zeros(lr, dtype=np.float32)
                for j in range(S_kv):
                    if s[j] == -np.inf:
                        continue
                    a = np.float32(w[j] * inv)
                    vj = vc[b, j]
                    for d in range(lr):
                        oi[d] = np.float32(oi[d] + a * vj[d])
                oh[b, h, i, :] = oi
    return out


def mla_config(S_q, kv_lora_rank, qk_rope_head_dim):
    """Per-shape (bq, bn_kv, threads) tile selector (mirrors
    ``mla_config_for``; decode S_q<=8 -> (1,64,64), prefill -> (4,64,256))."""
    if S_q <= 8:
        return (1, 64, 64)
    return (4, 64, 256)


# ---------------------------------------------------------------------------
# DSA: DeepseekSparseAttn sparse-MLA forward (issue #51)
# ---------------------------------------------------------------------------

_LOG2E = np.float32(math.log2(math.e))
_DSA_NEG_INF = np.float32(-np.inf)


def dsa_sparse_fwd(S_q, S_kv, H, dim, tail_dim, topk, kv_group, block_I,
                   inner_iter, sm_scale, return_lse, q, kv, indices, out,
                   lse):
    """Pure-Python (numpy) DSA sparse-MLA forward; mirrors
    ``dsa_sparse_fwd_cpu`` exactly (two-pass BASE-2 stable softmax over the
    ``topk`` selected keys, fp32 throughout). ``q`` ``(1,S_q,H,W)``,
    ``kv`` ``(1,S_kv,1,W)``, ``indices`` ``(1,S_q,1,topk)`` int32
    (``<0/>=S_kv`` masked), ``W = dim + tail_dim``, ``d_v = dim - tail_dim``.
    """
    W = dim + tail_dim
    d_v = dim - tail_dim
    q = np.ascontiguousarray(q, dtype=np.float32).reshape(S_q, H, W)
    kv = np.ascontiguousarray(kv, dtype=np.float32).reshape(S_kv, W)
    idx = np.ascontiguousarray(indices, dtype=np.int32).reshape(S_q, topk)
    out = np.ascontiguousarray(out, dtype=np.float32).reshape(S_q, H, d_v)
    lse = (
        np.ascontiguousarray(lse, dtype=np.float32).reshape(S_q, H)
        if return_lse
        else None
    )
    for i in range(S_q):
        idx_i = idx[i]  # (topk,)
        valid = (idx_i >= 0) & (idx_i < S_kv)  # (topk,)
        sel = np.where(valid, idx_i, 0)  # safe gather index
        kv_sel = kv[sel]  # (topk, W)
        v_sel = kv_sel[:, :d_v]  # (topk, d_v)
        k_main = kv_sel[:, :dim]  # (topk, dim)
        q_i = q[i]  # (H, W)
        q_main = q_i[:, :dim]  # (H, dim)
        s = q_main @ k_main.T  # (H, topk)
        if tail_dim > 0:
            k_tail = kv_sel[:, dim:dim + tail_dim]  # (topk, tail_dim)
            q_tail = q_i[:, dim:dim + tail_dim]  # (H, tail_dim)
            s = s + q_tail @ k_tail.T
        s = s * np.float32(sm_scale)
        s = np.where(valid[None, :], s, _DSA_NEG_INF)
        # kv_group == 1: every H head shares the same selected keys, so the
        # all-masked condition is a single scalar for this query row.
        row_valid = bool(valid.any())
        mx = s.max(axis=1, keepdims=True)  # (H, 1)
        safe_mx = np.where(row_valid, mx, np.float32(0.0))
        e = np.where(
            valid[None, :],
            np.power(np.float32(2.0), s - safe_mx),
            np.float32(0.0),
        )
        sm = e.sum(axis=1, keepdims=True)  # (H, 1)
        sm_safe = np.where(sm > 0, sm, np.float32(1.0))
        w = e / sm_safe  # (H, topk)
        out_i = (w @ v_sel).astype(np.float32)  # (H, d_v)
        out[i] = np.where(row_valid, out_i, np.float32(0.0)).astype(np.float32)
        if return_lse:
            row_lse = safe_mx[:, 0] + np.log2(np.where(sm[:, 0] > 0, sm[:, 0], np.float32(1.0)))
            lse[i] = np.where(row_valid, row_lse.astype(np.float32), _DSA_NEG_INF).astype(np.float32)
    return out


def dsa_config(S_q, H, dim, topk):
    """Per-shape (bq, threads, block_I, inner_iter) tile selector (mirrors
    ``dsa_config_for``; decode S_q<=8 -> (1,64,64,i), prefill ->
    (4,256,64,i); inner_iter grows while topk % (block_I*inner_iter) == 0)."""
    bq = 1 if S_q <= 8 else 4
    threads = 64 if S_q <= 8 else 256
    block_I = 64
    inner_iter = 1
    while inner_iter < 8 and topk % (block_I * (inner_iter * 2)) == 0:
        inner_iter *= 2
    return (bq, threads, block_I, inner_iter)


# ---------------------------------------------------------------------------
# MHC: multi-head hybrid-attention pre-norm (issue #51, part 2)
# ---------------------------------------------------------------------------


def mhc_pre_gemm_sqrsum(num_tokens, hc_mult, hidden_size, x, fn, out,
                        sqrsum):
    """Pure-Python (numpy) MHC pre-norm GEMM + squared-sum; mirrors
    ``mhc_pre_gemm_sqrsum_cpu`` exactly (fp32 throughout). ``x``
    ``(num_tokens, hc_hidden)`` with ``hc_hidden = hc_mult*hidden``,
    ``fn`` `(hc_mult3, hc_hidden)` with ``hc_mult3 = hc_mult*(2+hc_mult)``. """
    hc_hidden = hc_mult * hidden_size
    hc_mult3 = hc_mult * (2 + hc_mult)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(num_tokens,
                                                          hc_hidden)
    fn = np.ascontiguousarray(fn, dtype=np.float32).reshape(hc_mult3,
                                                           hc_hidden)
    out = np.ascontiguousarray(out, dtype=np.float32).reshape(num_tokens,
                                                              hc_mult3)
    sqrsum = np.ascontiguousarray(sqrsum, dtype=np.float32).reshape(
        num_tokens
    )
    np.dot(x, fn.T, out=out)
    np.einsum("nh,nh->n", x, x, out=sqrsum)
    return out, sqrsum


def mhc_post(num_tokens, hc, hidden, a, b, c, d, out):
    """Pure-Python (numpy) MHC post-attention combine; mirrors
    ``mhc_post_cpu`` exactly (fp32 throughout). ``a`` `(n,hc,hc)`,
    ``b`` `(n,hc,hidden)`, ``c`` `(n,hc)`, ``d`` `(n,hidden)` ->
    ``out[n,j,h] = c[n,j]*d[n,h] + sum_k a[n,k,j]*b[n,k,h]``. """
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(num_tokens, hc, hc)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(num_tokens, hc,
                                                           hidden)
    c = np.ascontiguousarray(c, dtype=np.float32).reshape(num_tokens, hc)
    d = np.ascontiguousarray(d, dtype=np.float32).reshape(num_tokens, hidden)
    out = np.ascontiguousarray(out, dtype=np.float32).reshape(
        num_tokens, hc, hidden
    )
    # a[n,k,j] * b[n,k,h] summed over k -> (n, j, h); then + c[n,j]*d[n,h].
    mixed = np.einsum("nkj,nkh->njh", a, b)
    out[:] = (mixed + c[:, :, None] * d[:, None, :]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# KDA: Kimi Delta Attention (issue #21)
# ---------------------------------------------------------------------------

_KDA_LOG_FLOOR = np.float32(-1.0e9)  # log(0) clamp (fully forgotten)


def _kda_log_gate(g):
    g = np.float32(g)
    if g <= np.float32(0.0):
        return _KDA_LOG_FLOOR
    return np.float32(math.log(float(g)))


def _kda_gate_prod_intra(L, a, b):
    """G_{a,b} = exp(L_b - L_{a-1}) with L_{-1}=0; 1.0 when a > b."""
    if a > b:
        return np.float32(1.0)
    return np.float32(math.exp(float(L[b] - (L[a - 1] if a > 0 else np.float32(0.0)))))


def kda_layer_norm_gated(x, weight, gate, out, N, D, eps):
    """Gated RMSNorm: out[n,d] = (x[n]/rms_n) * weight[d] * silu(gate[n,d]),
    rms_n = sqrt(mean(x[:]^2) + eps). Mirrors ``kda_layer_norm_gated_cpu``."""
    x = np.ascontiguousarray(x, dtype=np.float32).ravel()
    weight = np.ascontiguousarray(weight, dtype=np.float32).ravel()
    gate = np.ascontiguousarray(gate, dtype=np.float32).ravel()
    out = np.asarray(out, dtype=np.float32).ravel()
    if weight.size != D:
        raise ValueError("weight must be D")
    if x.size != N * D:
        raise ValueError("x must be N*D")
    if gate.size != N * D:
        raise ValueError("gate must be N*D")
    if out.size != N * D:
        raise ValueError("out must be N*D")
    if D <= 0:
        raise ValueError("D must be positive")
    if eps < 0.0:
        raise ValueError("eps must be non-negative")
    for n in range(N):
        xn = x[n * D : (n + 1) * D]
        sq = np.float32(0.0)
        for d in range(D):
            sq = np.float32(sq + xn[d] * xn[d])
        inv = np.float32(1.0 / math.sqrt(float(sq / np.float32(D) + np.float32(eps))))
        for d in range(D):
            g = np.float32(gate[n * D + d])
            silu = np.float32(g / (np.float32(1.0) + np.float32(math.exp(-float(g)))))
            out[n * D + d] = np.float32(xn[d] * inv * weight[d] * silu)
    return out


def kda_gate_chunk_cumsum(g, B, H, n_chunks, chunk_size):
    """Log-gate cumulative sums (mirrors ``kda_gate_chunk_cumsum_cpu``):
    g [B,H,n_chunks,chunk_size] -> (intra within-chunk INCLUSIVE log-cumsum,
    inter cross-chunk EXCLUSIVE log-cumsum)."""
    g = np.ascontiguousarray(g, dtype=np.float32).ravel()
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if g.size != B * H * n_chunks * chunk_size:
        raise ValueError("g must be B*H*n_chunks*chunk_size")
    intra = np.empty(B * H * n_chunks * chunk_size, dtype=np.float32)
    inter = np.empty(B * H * n_chunks, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            base = (b * H + h) * n_chunks
            inter_acc = np.float32(0.0)
            for c in range(n_chunks):
                Lc = intra[(base + c) * chunk_size : (base + c + 1) * chunk_size]
                gc = g[(base + c) * chunk_size : (base + c + 1) * chunk_size]
                acc = np.float32(0.0)
                for t in range(chunk_size):
                    acc = np.float32(acc + _kda_log_gate(gc[t]))
                    Lc[t] = acc
                inter[base + c] = inter_acc
                inter_acc = np.float32(inter_acc + acc)
    return intra, inter


def kda_naive_delta_rule_fwd(q, k, v, g, beta, B, H, S, D, out):
    """Per-token delta-rule oracle (mirrors ``kda_naive_delta_rule_fwd_cpu``,
    O(S*D^2)). q,k,v [B,H,S,D], g,beta [B,H,S] -> out [B,H,S,D] fp32."""
    q = np.ascontiguousarray(q, dtype=np.float32).ravel()
    k = np.ascontiguousarray(k, dtype=np.float32).ravel()
    v = np.ascontiguousarray(v, dtype=np.float32).ravel()
    g = np.ascontiguousarray(g, dtype=np.float32).ravel()
    beta = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    out = np.asarray(out, dtype=np.float32).ravel()
    if D <= 0:
        raise ValueError("D must be positive")
    for nm, arr, exp in (
        ("q", q, B * H * S * D),
        ("k", k, B * H * S * D),
        ("v", v, B * H * S * D),
        ("out", out, B * H * S * D),
        ("g", g, B * H * S),
        ("beta", beta, B * H * S),
    ):
        if arr.size != exp:
            raise ValueError(f"{nm} has {arr.size}, expected {exp}")
    if B == 0 or H == 0 or S == 0:
        return out
    state = np.zeros(D * D, dtype=np.float32)
    a = np.zeros(D, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            bh = (b * H + h) * S
            state[:] = 0.0
            for t in range(S):
                kt = k[(bh + t) * D : (bh + t + 1) * D]
                vt = v[(bh + t) * D : (bh + t + 1) * D]
                qt = q[(bh + t) * D : (bh + t + 1) * D]
                gt = g[bh + t]
                bt = beta[bh + t]
                for d in range(D):
                    s = np.float32(0.0)
                    Srow = state[d * D : (d + 1) * D]
                    for e in range(D):
                        s = np.float32(s + Srow[e] * kt[e])
                    a[d] = s
                for d in range(D):
                    ud = np.float32(vt[d] - a[d])
                    Srow = state[d * D : (d + 1) * D]
                    for e in range(D):
                        Srow[e] = np.float32(gt * Srow[e] + bt * ud * kt[e])
                    s = np.float32(0.0)
                    for e in range(D):
                        s = np.float32(s + Srow[e] * qt[e])
                    out[(bh + t) * D + d] = s
    return out


def kda_delta_rule_intra(q, k, v, g, beta, intra_log, inter_state, u,
                         B, H, S, D, chunk_size, chunk_idx):
    """Within-chunk delta-corrected value solve u_t for chunk ``chunk_idx``
    (mirrors ``kda_delta_rule_intra_cpu``). Writes into ``u`` in place."""
    q = np.ascontiguousarray(q, dtype=np.float32).ravel()
    k = np.ascontiguousarray(k, dtype=np.float32).ravel()
    v = np.ascontiguousarray(v, dtype=np.float32).ravel()
    g = np.ascontiguousarray(g, dtype=np.float32).ravel()
    beta = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    u = np.asarray(u, dtype=np.float32).ravel()
    intra_log = np.ascontiguousarray(intra_log, dtype=np.float32).ravel()
    inter_state = np.ascontiguousarray(inter_state, dtype=np.float32).ravel()
    _kda_check_chunk(S, D, chunk_size)
    n_chunks = S // chunk_size
    if not (0 <= chunk_idx * chunk_size < S):
        raise ValueError("chunk_idx out of range")
    for b in range(B):
        for h in range(H):
            bh = (b * H + h) * S
            Lc = intra_log[((b * H + h) * n_chunks + chunk_idx) * chunk_size:
                           ((b * H + h) * n_chunks + chunk_idx + 1) * chunk_size]
            Cin = inter_state[((b * H + h) * (n_chunks + 1) + chunk_idx) * D * D:
                              ((b * H + h) * (n_chunks + 1) + chunk_idx + 1) * D * D]
            for t in range(chunk_size):
                tau = chunk_idx * chunk_size + t
                kt = k[(bh + tau) * D : (bh + tau + 1) * D]
                G0 = _kda_gate_prod_intra(Lc, 0, t - 1)
                pred = np.zeros(D, dtype=np.float32)
                for d in range(D):
                    s = np.float32(0.0)
                    for e in range(D):
                        s = np.float32(s + Cin[d * D + e] * kt[e])
                    pred[d] = np.float32(G0 * s)
                corr = np.zeros(D, dtype=np.float32)
                for j in range(t):
                    tauj = chunk_idx * chunk_size + j
                    kj = k[(bh + tauj) * D : (bh + tauj + 1) * D]
                    uj = u[(bh + tauj) * D : (bh + tauj + 1) * D]
                    Gj = _kda_gate_prod_intra(Lc, j + 1, t - 1)
                    djk = np.float32(0.0)
                    for e in range(D):
                        djk = np.float32(djk + kj[e] * kt[e])
                    cjk = np.float32(Gj * beta[bh + tauj] * djk)
                    for d in range(D):
                        corr[d] = np.float32(corr[d] + cjk * uj[d])
                ut = u[(bh + tau) * D : (bh + tau + 1) * D]
                for d in range(D):
                    ut[d] = np.float32(v[(bh + tau) * D + d] - pred[d] - corr[d])
    return u


def kda_delta_rule_inter(k, v, g, beta, intra_log, u, inter_state,
                         B, H, S, D, chunk_size, chunk_idx):
    """Cross-chunk state propagation for chunk ``chunk_idx`` (mirrors
    ``kda_delta_rule_inter_cpu``): fills inter_state[..,chunk_idx+1] = C_c.
    Writes into ``inter_state`` in place."""
    k = np.ascontiguousarray(k, dtype=np.float32).ravel()
    u = np.ascontiguousarray(u, dtype=np.float32).ravel()
    beta = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    intra_log = np.ascontiguousarray(intra_log, dtype=np.float32).ravel()
    inter_state = np.asarray(inter_state, dtype=np.float32).ravel()
    _kda_check_chunk(S, D, chunk_size)
    n_chunks = S // chunk_size
    if not (0 <= chunk_idx * chunk_size < S):
        raise ValueError("chunk_idx out of range")
    for b in range(B):
        for h in range(H):
            bh = (b * H + h) * S
            Lc = intra_log[((b * H + h) * n_chunks + chunk_idx) * chunk_size:
                           ((b * H + h) * n_chunks + chunk_idx + 1) * chunk_size]
            Cin = inter_state[((b * H + h) * (n_chunks + 1) + chunk_idx) * D * D:
                              ((b * H + h) * (n_chunks + 1) + chunk_idx + 1) * D * D]
            Cout = inter_state[((b * H + h) * (n_chunks + 1) + chunk_idx + 1) * D * D:
                               ((b * H + h) * (n_chunks + 1) + chunk_idx + 2) * D * D]
            Gfull = _kda_gate_prod_intra(Lc, 0, chunk_size - 1)
            for d in range(D):
                Cinrow = Cin[d * D : (d + 1) * D]
                Crow = Cout[d * D : (d + 1) * D]
                for e in range(D):
                    Crow[e] = np.float32(Gfull * Cinrow[e])
            for t in range(chunk_size):
                tau = chunk_idx * chunk_size + t
                kt = k[(bh + tau) * D : (bh + tau + 1) * D]
                Gt = _kda_gate_prod_intra(Lc, t + 1, chunk_size - 1)
                w = np.float32(Gt * beta[bh + tau])
                ut = u[(bh + tau) * D : (bh + tau + 1) * D]
                for d in range(D):
                    Crow = Cout[d * D : (d + 1) * D]
                    wd = np.float32(w * ut[d])
                    for e in range(D):
                        Crow[e] = np.float32(Crow[e] + wd * kt[e])
    return inter_state


def kda_gla_fwd_o(q, k, g, beta, intra_log, inter_state, u,
                  B, H, S, D, chunk_size, out):
    """Output (intra + inter) combine (mirrors ``kda_gla_fwd_o_cpu``):
    o_t = G_{0,t}(C_{c-1} q_t) + sum_{j<=t} G_{j+1,t} b_j (k_j.q_t) u_j.
    Writes into ``out`` in place."""
    q = np.ascontiguousarray(q, dtype=np.float32).ravel()
    k = np.ascontiguousarray(k, dtype=np.float32).ravel()
    beta = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    intra_log = np.ascontiguousarray(intra_log, dtype=np.float32).ravel()
    inter_state = np.ascontiguousarray(inter_state, dtype=np.float32).ravel()
    u = np.ascontiguousarray(u, dtype=np.float32).ravel()
    out = np.asarray(out, dtype=np.float32).ravel()
    _kda_check_chunk(S, D, chunk_size)
    n_chunks = S // chunk_size
    inter_out = np.zeros(D, dtype=np.float32)
    intra_out = np.zeros(D, dtype=np.float32)
    for b in range(B):
        for h in range(H):
            bh = (b * H + h) * S
            for c in range(n_chunks):
                Lc = intra_log[((b * H + h) * n_chunks + c) * chunk_size:
                               ((b * H + h) * n_chunks + c + 1) * chunk_size]
                Cin = inter_state[((b * H + h) * (n_chunks + 1) + c) * D * D:
                                  ((b * H + h) * (n_chunks + 1) + c + 1) * D * D]
                for t in range(chunk_size):
                    tau = c * chunk_size + t
                    qt = q[(bh + tau) * D : (bh + tau + 1) * D]
                    G0t = _kda_gate_prod_intra(Lc, 0, t)
                    for d in range(D):
                        s = np.float32(0.0)
                        for e in range(D):
                            s = np.float32(s + Cin[d * D + e] * qt[e])
                        inter_out[d] = np.float32(G0t * s)
                    for d in range(D):
                        intra_out[d] = np.float32(0.0)
                    for j in range(t + 1):
                        tauj = c * chunk_size + j
                        kj = k[(bh + tauj) * D : (bh + tauj + 1) * D]
                        uj = u[(bh + tauj) * D : (bh + tauj + 1) * D]
                        Gj = _kda_gate_prod_intra(Lc, j + 1, t)
                        djq = np.float32(0.0)
                        for e in range(D):
                            djq = np.float32(djq + kj[e] * qt[e])
                        cjq = np.float32(Gj * beta[bh + tauj] * djq)
                        for d in range(D):
                            intra_out[d] = np.float32(intra_out[d] + cjq * uj[d])
                    for d in range(D):
                        out[(bh + tau) * D + d] = np.float32(inter_out[d] + intra_out[d])
    return out


def kda_delta_rule_fwd(q, k, v, g, beta, B, H, S, D, chunk_size, out):
    """Chunked delta-rule forward (mirrors ``kda_delta_rule_fwd_cpu``, the
    L2..L6 pipeline). chunk_size must divide S. Writes into ``out`` in place."""
    q = np.ascontiguousarray(q, dtype=np.float32).ravel()
    k = np.ascontiguousarray(k, dtype=np.float32).ravel()
    v = np.ascontiguousarray(v, dtype=np.float32).ravel()
    g = np.ascontiguousarray(g, dtype=np.float32).ravel()
    beta = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    out = np.asarray(out, dtype=np.float32).ravel()
    _kda_check_chunk(S, D, chunk_size)
    if q.size != B * H * S * D:
        raise ValueError("q must be B*H*S*D")
    if g.size != B * H * S:
        raise ValueError("g must be B*H*S")
    if out.size != B * H * S * D:
        raise ValueError("out must be B*H*S*D")
    if B == 0 or H == 0 or S == 0:
        return out
    n_chunks = S // chunk_size
    intra_log, inter_log = kda_gate_chunk_cumsum(g, B, H, n_chunks, chunk_size)
    u = np.zeros(B * H * S * D, dtype=np.float32)
    inter_state = np.zeros(B * H * (n_chunks + 1) * D * D, dtype=np.float32)
    for c in range(n_chunks):
        kda_delta_rule_intra(q, k, v, g, beta, intra_log, inter_state, u,
                             B, H, S, D, chunk_size, c)
        kda_delta_rule_inter(k, v, g, beta, intra_log, u, inter_state,
                             B, H, S, D, chunk_size, c)
    kda_gla_fwd_o(q, k, g, beta, intra_log, inter_state, u,
                  B, H, S, D, chunk_size, out)
    return out


def kda_pack_bitmatrix(bits, n_bits):
    """Pack a binary [n_bits] uint8 array into ceil(n/8) bytes, MSB first
    (bit k -> byte k/8, bit 7 - k%8). Mirrors ``kda_pack_bitmatrix_cpu``."""
    bits = np.ascontiguousarray(bits, dtype=np.uint8).ravel()
    n_bytes = (n_bits + 7) // 8
    packed = np.zeros(n_bytes, dtype=np.uint8)
    for k in range(n_bits):
        if bits[k] != 0:
            packed[k // 8] |= np.uint8(1 << (7 - (k % 8)))
    return packed


def _kda_check_chunk(S, D, chunk_size):
    if D <= 0:
        raise ValueError("D must be positive")
    if chunk_size <= 0 or S % chunk_size != 0:
        raise ValueError("chunk_size must divide S")


# ---------------------------------------------------------------------------
# comm: topology, channels, ring all-reduce
# ---------------------------------------------------------------------------


def ring_rank(rank: int, world: int) -> Topology:
    if world <= 0:
        raise ValueError("world must be positive")
    if not (0 <= rank < world):
        raise ValueError("rank out of range")
    return Topology(rank, world, (rank + 1) % world, (rank - 1) % world)


def build_ring_topology(world: int) -> list[Topology]:
    if world <= 0:
        raise ValueError("world must be positive")
    return [ring_rank(r, world) for r in range(world)]


class BlockingQueue:
    """Thread-safe blocking queue of float32 chunks (mirrors C++)."""

    def __init__(self):
        self._items: deque = deque()
        self._cv = threading.Condition()
        self._closed = False

    def push(self, chunk) -> None:
        with self._cv:
            self._items.append(np.asarray(chunk, dtype=np.float32))
            self._cv.notify()

    def pop(self) -> np.ndarray:
        with self._cv:
            self._cv.wait_for(lambda: bool(self._items))
            return self._items.popleft()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def closed(self) -> bool:
        with self._cv:
            return self._closed


class MockChannel:
    """In-process channel: send into ``out``, receive from ``in``."""

    def __init__(self, out: BlockingQueue, in_: BlockingQueue):
        self._out = out
        self._in = in_

    def send(self, chunk) -> None:
        self._out.push(chunk)

    def recv(self) -> np.ndarray:
        return self._in.pop()

    def closed(self) -> bool:
        return self._in.closed()


def make_ring_channels(world: int) -> list[MockChannel]:
    if world <= 0:
        raise ValueError("world must be positive")
    edges = [BlockingQueue() for _ in range(world)]
    return [MockChannel(edges[r], edges[(r - 1) % world]) for r in range(world)]


def ring_allreduce_rank(
    local: np.ndarray, rank: int, world: int, next_ch, prev_ch
) -> None:
    """Run one rank of a ring all-reduce; ``local`` is summed in place."""
    if world <= 0:
        raise ValueError("world must be positive")
    if not (0 <= rank < world):
        raise ValueError("rank out of range")
    n = local.size
    if n % world != 0:
        raise ValueError("local length must be divisible by world")
    if world == 1:
        return

    chunk = n // world

    def chunk_of(c: int) -> np.ndarray:
        return local[c * chunk : (c + 1) * chunk].copy()

    def replace_chunk(c: int, v: np.ndarray) -> None:
        local[c * chunk : (c + 1) * chunk] = v

    # Phase 1: reduce-scatter (world-1 steps).
    for t in range(world - 1):
        send_c = (rank - t) % world
        recv_c = (rank - t - 1) % world
        next_ch.send(chunk_of(send_c))
        got = prev_ch.recv()
        replace_chunk(recv_c, chunk_of(recv_c) + got)

    # Phase 2: all-gather (world-1 steps), copying the reduced chunks.
    for t in range(world - 1):
        send_c = (rank - t + 1) % world
        recv_c = (rank - t) % world
        next_ch.send(chunk_of(send_c))
        got = prev_ch.recv()
        replace_chunk(recv_c, got)


def ring_allreduce(locals_: list[np.ndarray]) -> list[np.ndarray]:
    world = len(locals_)
    if world <= 0:
        raise ValueError("locals must be non-empty")
    length = locals_[0].size
    for v in locals_:
        if v.size != length:
            raise ValueError("all locals must have equal length")
    if length % world != 0:
        raise ValueError("local length must be divisible by world")

    channels = make_ring_channels(world)
    buffers = [np.array(v, dtype=np.float32, copy=True) for v in locals_]
    threads = [
        threading.Thread(
            target=ring_allreduce_rank,
            args=(buffers[r], r, world, channels[r], channels[r]),
        )
        for r in range(world)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return buffers


class OverlapExecutor:
    """Compute on one stream, comm on a second; data dependency via events."""

    def __init__(self):
        self._compute = Stream()
        self._comm = Stream()

    def uses_two_streams(self) -> bool:
        return True

    def run(self, iters: int, compute, comm_fn) -> Result:
        for i in range(iters):
            event = threading.Event()
            cell: dict = {}

            def do_compute(i=i, event=event, cell=cell):
                cell["value"] = compute(i)
                event.set()

            self._compute.submit(do_compute)

            def do_comm(i=i, event=event, cell=cell):
                event.wait()
                comm_fn(i, cell["value"])

            self._comm.submit(do_comm)
        self._compute.wait()
        self._comm.wait()
        return Result(iters, iters)


# ---------------------------------------------------------------------------
# comm: p2p run-list gather
# ---------------------------------------------------------------------------


def _dst_buffer(dst: np.ndarray) -> ctypes.Array:
    """Writable ctypes view of a uint8 numpy array (same base address)."""
    return (ctypes.c_ubyte * dst.nbytes).from_buffer(dst)


def stage_runs_1d(dst: np.ndarray, src_ptrs, dst_offsets, lengths) -> list[StagedRun1D]:
    n = len(src_ptrs)
    if len(dst_offsets) != n or len(lengths) != n:
        raise ValueError("src_ptrs/dst_offsets/lengths must all have the same length")
    base = ctypes.addressof(_dst_buffer(dst))
    capacity = dst.nbytes

    runs: list[StagedRun1D] = []
    for i in range(n):
        length = int(lengths[i])
        off = int(dst_offsets[i])
        if length == 0:
            continue  # empty run: no source needed, nothing to copy
        if off > capacity or capacity - off < length:
            raise ValueError("run exceeds dst capacity")
        s = int(src_ptrs[i])
        if s == 0:
            raise ValueError("src_ptrs[i] must be non-null for a non-empty run")
        d = base + off
        if not (s + length <= d or s >= d + length):
            raise ValueError("src and dst ranges must not overlap")
        runs.append(StagedRun1D(s, off, length))

    # Output runs must be mutually disjoint (checked in sorted order).
    spans = sorted((r.dst_offset, r.dst_offset + r.length) for r in runs)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        if end > start:
            raise ValueError("output runs must be disjoint")
    return runs


def stage_runs_2d(dst: np.ndarray, runs) -> list[StagedRun2D]:
    capacity = dst.nbytes
    out: list[StagedRun2D] = []
    for g in runs:
        if not isinstance(g, Gather2DRun):
            raise TypeError("runs must be Gather2DRun instances")
        if g.width == 0 or g.height == 0:
            continue  # empty tile: nothing to copy
        if g.width > g.src_stride:
            raise ValueError("width must not exceed src_stride")
        if g.width > g.dst_stride:
            raise ValueError("width must not exceed dst_stride")
        if g.src == 0:
            raise ValueError("src must be non-null for a non-empty tile")
        off = g.dst_offset
        if off > capacity:
            raise ValueError("2-D run starts past dst capacity")
        rows = g.height - 1
        if rows > (capacity - off) // g.dst_stride:
            raise ValueError("2-D row stride exceeds dst capacity")
        row_span = rows * g.dst_stride
        if g.width > capacity - off - row_span:
            raise ValueError("2-D run exceeds dst capacity")
        out.append(
            StagedRun2D(g.src, off, g.src_stride, g.dst_stride, g.width, g.height)
        )
    return out


def p2p_gather_runs(
    dst: np.ndarray, src_ptrs, dst_offsets, lengths, stream=None
) -> None:
    staged = stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)
    if not staged:
        return  # 0 runs, or every run empty: enqueue nothing
    buf = _dst_buffer(dst)

    def copy_all() -> None:
        for r in staged:
            ctypes.memmove(
                ctypes.byref(buf, r.dst_offset), ctypes.c_void_p(r.src), r.length
            )

    if stream is None:
        copy_all()
    else:
        stream.submit(copy_all)


def p2p_gather_runs_2d(dst: np.ndarray, runs, stream=None) -> None:
    staged = stage_runs_2d(dst, runs)
    if not staged:
        return
    buf = _dst_buffer(dst)

    def copy_all() -> None:
        for r in staged:
            for y in range(r.height):
                ctypes.memmove(
                    ctypes.byref(buf, r.dst_offset + y * r.dst_stride),
                    ctypes.c_void_p(r.src + y * r.src_stride),
                    r.width,
                )

    if stream is None:
        copy_all()
    else:
        stream.submit(copy_all)


def memcpy_peer_batch_async(
    dst: np.ndarray, src_ptrs, dst_offsets, lengths, stream=None
) -> None:
    staged = stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)
    if not staged:
        return
    buf = _dst_buffer(dst)

    def copy_one(r: StagedRun1D) -> None:
        ctypes.memmove(
            ctypes.byref(buf, r.dst_offset), ctypes.c_void_p(r.src), r.length
        )

    if stream is None:
        for r in staged:
            copy_one(r)
    else:
        # Legacy seam: one stream task per run.
        for r in staged:
            stream.submit(lambda r=r: copy_one(r))


# ---------------------------------------------------------------------------
# comm: fused indexed K/V layer gather (issue #2)
# ---------------------------------------------------------------------------
#
# Packs arbitrary SGLang KV-pool slots into a contiguous per-layer page
# buffer before peer donation/eviction, fusing the two separate advanced-
# index gathers (one for K, one for V) into a single operation. The
# reference is byte-exact:
#
#     dst[:, :, 0] = k_src[slot_ids]
#     dst[:, :, 1] = v_src[slot_ids]
#
# Because the operation is a raw-byte copy (no type-specific arithmetic),
# the same implementation is correct for both BF16 and FP16: numpy has no
# native BF16, so callers pass a ``uint16`` view of BF16 storage and FP16
# as ``np.float16``. Both have ``itemsize == 2``.
_KV_DTYPES = (np.dtype(np.float16), np.dtype(np.uint16), np.dtype(np.int16))


def _as_kv_array(x, name: str) -> np.ndarray:
    """Coerce ``x`` to a C-contiguous 2-byte, 3-D KV source array."""
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.dtype not in _KV_DTYPES:
        raise TypeError(
            f"{name} must be a BF16 (uint16 view) or FP16 array, "
            f"got dtype={arr.dtype}"
        )
    if arr.itemsize != 2:
        raise TypeError(f"{name} must have itemsize 2, got {arr.itemsize}")
    if arr.ndim != 3:
        raise ValueError(
            f"{name} must be [num_slots, num_kv_heads, head_dim] (3-D), "
            f"got shape {arr.shape}"
        )
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return arr


def _as_slot_ids(x, name: str = "slot_ids") -> np.ndarray:
    """Coerce ``x`` to a C-contiguous int32/int64 2-D slot map."""
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
        raise TypeError(
            f"{name} must be int32 or int64, got dtype={arr.dtype}"
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be [num_pages, page_size] (2-D), got shape {arr.shape}"
        )
    if not arr.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    return arr


def kv_gather_layer(k_src, v_src, slot_ids, dst, *, stream=None) -> None:
    """Fused indexed K/V gather for one layer (issue #2).

    Reads ``num_pages * page_size`` indexed source slots from ``k_src`` /
    ``v_src`` and writes the packed ``[num_pages, page_size, 2, num_kv_heads,
    head_dim]`` destination, exactly:

        dst[:, :, 0] = k_src[slot_ids]
        dst[:, :, 1] = v_src[slot_ids]

    Replaces the two separate advanced-index gathers the KVAAS ``pack_pages``
    path performs today. ``num_pages == 0`` is a valid no-op. ``slot_ids`` may
    repeat (gather semantics) and be in any order (non-monotonic); only the
    range ``[0, num_slots)`` is enforced. Enqueued on ``stream`` (one task)
    and returns without synchronising.

    Args:
        k_src, v_src: ``[num_slots, num_kv_heads, head_dim]`` C-contiguous
            arrays with ``itemsize == 2`` (BF16 as a ``uint16`` view, FP16 as
            ``np.float16``).
        slot_ids: ``[num_pages, page_size]`` C-contiguous ``int32`` or
            ``int64`` array of source slots in ``[0, num_slots)``.
        dst: ``[num_pages, page_size, 2, num_kv_heads, head_dim]``
            C-contiguous writable array of the same dtype as ``k_src``/
            ``v_src``.
        stream: optional :class:`~vkernels.core.Stream`; when given the
            gather runs as a single asynchronous task.

    Raises:
        TypeError: if any array has the wrong dtype.
        ValueError: if shapes/strides are inconsistent, ``slot_ids`` is out
            of range, or ``dst`` is not writable.
    """
    k = _as_kv_array(k_src, "k_src")
    v = _as_kv_array(v_src, "v_src")
    slots = _as_slot_ids(slot_ids, "slot_ids")
    if isinstance(dst, np.ndarray):
        d = dst
    else:
        d = np.asarray(dst)
    if d.dtype != k.dtype:
        raise TypeError(
            f"dst must have the same dtype as k_src/v_src, got {d.dtype} vs {k.dtype}"
        )
    if d.ndim != 5:
        raise ValueError(
            f"dst must be [num_pages, page_size, 2, num_kv_heads, head_dim] "
            f"(5-D), got shape {d.shape}"
        )
    if not d.flags.c_contiguous:
        raise ValueError("dst must be C-contiguous")
    if not d.flags.writeable:
        raise ValueError("dst must be writable")
    if k.shape != v.shape:
        raise ValueError(
            f"k_src and v_src must have the same shape, got {k.shape} vs {v.shape}"
        )
    if k.dtype != v.dtype:
        raise TypeError(
            f"k_src and v_src must have the same dtype, got {k.dtype} vs {v.dtype}"
        )

    num_slots, num_kv_heads, head_dim = k.shape
    num_pages, page_size = slots.shape
    if d.shape != (num_pages, page_size, 2, num_kv_heads, head_dim):
        raise ValueError(
            f"dst shape {d.shape} does not match (num_pages={num_pages}, "
            f"page_size={page_size}, 2, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim})"
        )

    if num_pages == 0:
        return  # valid no-op (also covers page_size == 0)

    if page_size == 0:
        return
    if num_slots == 0:
        raise ValueError("num_slots must be positive when num_pages > 0")
    if num_kv_heads == 0 or head_dim == 0:
        raise ValueError("num_kv_heads and head_dim must be positive")

    if slots.size and (slots.min() < 0 or slots.max() >= num_slots):
        raise ValueError(f"slot_ids must be in [0, {num_slots})")

    slot_bytes = num_kv_heads * head_dim * 2  # elem_size == 2
    k_flat = k.view(np.uint8).reshape(num_slots, slot_bytes)
    v_flat = v.view(np.uint8).reshape(num_slots, slot_bytes)
    d_flat = d.view(np.uint8).reshape(num_pages * page_size, 2, slot_bytes)
    flat_ids = slots.reshape(num_pages * page_size)

    def gather() -> None:
        gathered_k = k_flat[flat_ids]  # [N, slot_bytes]
        gathered_v = v_flat[flat_ids]
        # Copy into the destination; assignment is a byte-exact memmove for
        # matching dtypes (uint8 views here).
        d_flat[:, 0, :] = gathered_k
        d_flat[:, 1, :] = gathered_v

    if stream is None:
        gather()
    else:
        stream.submit(gather)



# ---------------------------------------------------------------------------
# Fused indexed K/V layer scatter (issue #1)
# ---------------------------------------------------------------------------
#
# The restore-side reverse of kv_gather (issue #2): scatters a contiguous,
# already-gathered per-layer scratch buffer back into the paged KV pool at
# arbitrary, non-contiguous destination slots. The current fallback performs
# two PyTorch advanced-index writes (one for K, one for V); this primitive
# fuses both writes (and the K/V split) into a single operation, byte-exact
# against:
#
#     k_dst[slot_ids] = src[:, :, 0]
#     v_dst[slot_ids] = src[:, :, 1]
#
# Because the destination slots are DISJOINT (uniqueness is enforced, unlike
# the gather's repeatable sources), every byte is written exactly once and
# the write order is irrelevant. Same raw-byte-copy reasoning as kv_gather:
# correct for both BF16 (uint16 view) and FP16.

def kv_scatter_layer(k_dst, v_dst, slot_ids, src, *, stream=None) -> None:
    """Fused indexed K/V scatter for one layer (issue #1).

    Reads a contiguous ``[num_pages, page_size, 2, num_kv_heads, head_dim]``
    ``src`` and writes K and V into the indexed destination slots of
    ``k_dst`` / ``v_dst``, exactly:

        k_dst[slot_ids] = src[:, :, 0]
        v_dst[slot_ids] = src[:, :, 1]

    Replaces the two separate advanced-index writes the KVAAS ``scatter_layer``
    path performs today. ``num_pages == 0`` is a valid no-op. ``slot_ids``
    must be UNIQUE (scatter writes disjoint destinations; duplicates are a
    contract violation) and in range ``[0, num_slots)``; any order is allowed
    (non-monotonic).

    Args:
        k_dst, v_dst: ``[num_slots, num_kv_heads, head_dim]`` C-contiguous
            writable arrays with ``itemsize == 2`` (BF16 as a ``uint16``
            view, FP16 as ``np.float16``).
        slot_ids: ``[num_pages, page_size]`` C-contiguous ``int32`` or
            ``int64`` array of UNIQUE destination slots in ``[0, num_slots)``.
        src: ``[num_pages, page_size, 2, num_kv_heads, head_dim]``
            C-contiguous array of the same dtype as ``k_dst``/``v_dst``.
        stream: optional :class:`~vkernels.core.Stream`; when given the
            scatter runs as a single asynchronous task.

    Raises:
        TypeError: if any array has the wrong dtype.
        ValueError: if shapes/strides are inconsistent, ``slot_ids`` is out
            of range or non-unique, or ``k_dst``/``v_dst`` is not writable.
    """
    k = _as_kv_array(k_dst, "k_dst")
    v = _as_kv_array(v_dst, "v_dst")
    slots = _as_slot_ids(slot_ids, "slot_ids")
    if isinstance(src, np.ndarray):
        s = src
    else:
        s = np.asarray(src)
    if s.dtype != k.dtype:
        raise TypeError(
            f"src must have the same dtype as k_dst/v_dst, got {s.dtype} vs {k.dtype}"
        )
    if s.ndim != 5:
        raise ValueError(
            f"src must be [num_pages, page_size, 2, num_kv_heads, head_dim] "
            f"(5-D), got shape {s.shape}"
        )
    if not s.flags.c_contiguous:
        raise ValueError("src must be C-contiguous")
    if not k.flags.writeable:
        raise ValueError("k_dst must be writable")
    if not v.flags.writeable:
        raise ValueError("v_dst must be writable")
    if k.shape != v.shape:
        raise ValueError(
            f"k_dst and v_dst must have the same shape, got {k.shape} vs {v.shape}"
        )
    if k.dtype != v.dtype:
        raise TypeError(
            f"k_dst and v_dst must have the same dtype, got {k.dtype} vs {v.dtype}"
        )

    num_slots, num_kv_heads, head_dim = k.shape
    num_pages, page_size = slots.shape
    if s.shape != (num_pages, page_size, 2, num_kv_heads, head_dim):
        raise ValueError(
            f"src shape {s.shape} does not match (num_pages={num_pages}, "
            f"page_size={page_size}, 2, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim})"
        )

    if num_pages == 0:
        return  # valid no-op (also covers page_size == 0)

    if page_size == 0:
        return
    if num_slots == 0:
        raise ValueError("num_slots must be positive when num_pages > 0")
    if num_kv_heads == 0 or head_dim == 0:
        raise ValueError("num_kv_heads and head_dim must be positive")

    # Scatter writes disjoint destinations: uniqueness is required (unlike
    # the gather, which may repeat source slots). Range is also enforced.
    flat_ids = slots.reshape(num_pages * page_size)
    if flat_ids.size:
        if flat_ids.min() < 0 or flat_ids.max() >= num_slots:
            raise ValueError(f"slot_ids must be in [0, {num_slots})")
        if np.unique(flat_ids).size != flat_ids.size:
            raise ValueError("slot_ids must be unique (scatter writes disjoint destinations)")

    slot_bytes = num_kv_heads * head_dim * 2  # elem_size == 2
    k_flat = k.view(np.uint8).reshape(num_slots, slot_bytes)
    v_flat = v.view(np.uint8).reshape(num_slots, slot_bytes)
    s_flat = s.view(np.uint8).reshape(num_pages * page_size, 2, slot_bytes)

    def scatter() -> None:
        # Byte-exact memmove into the indexed destinations (matching dtypes
        # via uint8 views). Uniqueness guarantees no two writes alias.
        k_flat[flat_ids] = s_flat[:, 0, :]
        v_flat[flat_ids] = s_flat[:, 1, :]

    if stream is None:
        scatter()
    else:
        stream.submit(scatter)
