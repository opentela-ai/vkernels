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


def gemm(M: int, N: int, K: int, alpha, A: np.ndarray, B: np.ndarray, beta,
         C: np.ndarray) -> None:
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
    0.0, 0.25,
    # s=0, e=1: normal ×2^(1-1)= ×1
    1.0, 1.5,
    # s=0, e=2: normal ×2^(2-1)= ×2
    2.0, 3.0,
    # s=0, e=3: inf / NaN
    float("inf"), float("nan"),
    # s=1, e=0: negative zero / subnormal
    -0.0, -0.25,
    # s=1, e=1: negative normal
    -1.0, -1.5,
    # s=1, e=2: negative normal
    -2.0, -3.0,
    # s=1, e=3: negative inf / NaN
    -float("inf"), float("nan"),
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


def mfma_f32_16x16x16bf16(c: list[float], a: list[int], b: list[int],
                          cbsz: int = 0, abid: int = 0, blgp: int = 0) -> None:
    """K16 bf16 MFMA: C[0..3] += A[0..1] × B[0..1] (16×16×16 bf16, acc fp32).

    `c` is a list of 4 floats updated in-place.
    `a` and `b` are lists of 2 uint32_t each, packing 2 bf16 values per
    uint32_t (low 16 bits, high 16 bits).

    The control flags (cbsz, abid, blgp) are accepted but ignored on the
    host reference path.
    """
    if len(c) < 4 or len(a) < 2 or len(b) < 2:
        raise ValueError(
            "c must have 4 floats; a and b must each have 2 uint32_t"
        )

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
    return np.frombuffer((bits << 16).to_bytes(4, "little"),
                         dtype=np.float32)[0]


def _dequant_weight_tile(packed, scale, p_base, s_base,
                         N, K, group_size, stride_packed, stride_scale_n):
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


def fused_moe_mxfp4(A, w13, w13_scale, w2, w2_scale, sorted_ids, topk_w,
                    expert_ids, act_scratch, out,
                    M, hidden, ispp, top_k, EM, group_size,
                    swiglu_limit, b13=None, b2=None):
    """Fused MXFP4 MoE grouped GEMM (CPU-reference oracle, in place).

    Writes the SwiGLU intermediate into ``act_scratch`` [EM*ispp] bf16 and
    accumulates into ``out`` [M*hidden] fp32 (which the caller must
    zero-initialise). Mirrors ``fused_moe_mxfp4_cpu`` block for block.
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

    # ===== Stage 0: gate_up + SwiGLU → act [EM, ispp] =====
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
                        tile_A[m] = A[token * hidden + k_start:
                                      token * hidden + k_start + BLOCK_K]

                # gate half
                p_gate = expert * w13_expert_bytes + nb * BLOCK_N * (hidden // 2)
                s_gate = expert * w13s_expert_bytes + nb * BLOCK_N * (hidden // group_size)
                tile_gate = _dequant_weight_tile(
                    w13, w13_scale, p_gate + k_start // 2,
                    s_gate + k_start // group_size,
                    BLOCK_N, BLOCK_K, group_size, hidden // 2,
                    hidden // group_size)
                # up half (offset by ispp rows in w13)
                p_up = expert * w13_expert_bytes + (nb * BLOCK_N + ispp) * (hidden // 2)
                s_up = expert * w13s_expert_bytes + (nb * BLOCK_N + ispp) * (hidden // group_size)
                tile_up = _dequant_weight_tile(
                    w13, w13_scale, p_up + k_start // 2,
                    s_up + k_start // group_size,
                    BLOCK_N, BLOCK_K, group_size, hidden // 2,
                    hidden // group_size)

                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        dg = np.float32(0.0)
                        du = np.float32(0.0)
                        for k in range(BLOCK_K):
                            a = _bf16_to_float(int(tile_A[m, k]))
                            dg = np.float32(
                                dg + np.float32(a * _bf16_to_float(int(tile_gate[k, n]))))
                            du = np.float32(
                                du + np.float32(a * _bf16_to_float(int(tile_up[k, n]))))
                        acc_gate[m, n] = np.float32(acc_gate[m, n] + dg)
                        acc_up[m, n] = np.float32(acc_up[m, n] + du)

            # SwiGLU epilogue (skip padding rows)
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
                    if swiglu_limit > 0.0:
                        g = min(g, swiglu_limit)
                        u = min(u, swiglu_limit)
                        u = max(u, -swiglu_limit)
                    silu_g = g / (1.0 + np.exp(-g))
                    act[(token_base + m) * ispp + nb * BLOCK_N + n] = \
                        _f2bf(silu_g * u)

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
                        tile_A[m] = act[(token_base + m) * ispp + k_start:
                                        (token_base + m) * ispp + k_start + BLOCK_K]

                p_down = expert * w2_expert_bytes + nb * BLOCK_N * (ispp // 2)
                s_down = expert * w2s_expert_bytes + nb * BLOCK_N * (ispp // group_size)
                tile_down = _dequant_weight_tile(
                    w2, w2_scale, p_down + k_start // 2,
                    s_down + k_start // group_size,
                    BLOCK_N, BLOCK_K, group_size, ispp // 2,
                    ispp // group_size)

                for m in range(BLOCK_M):
                    for n in range(BLOCK_N):
                        dot = np.float32(0.0)
                        for k in range(BLOCK_K):
                            a = _bf16_to_float(int(tile_A[m, k]))
                            dot = np.float32(
                                dot + np.float32(a * _bf16_to_float(int(tile_down[k, n]))))
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


def ring_allreduce_rank(local: np.ndarray, rank: int, world: int, next_ch,
                        prev_ch) -> None:
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
        return local[c * chunk:(c + 1) * chunk].copy()

    def replace_chunk(c: int, v: np.ndarray) -> None:
        local[c * chunk:(c + 1) * chunk] = v

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
        threading.Thread(target=ring_allreduce_rank,
                         args=(buffers[r], r, world, channels[r], channels[r]))
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
        out.append(StagedRun2D(g.src, off, g.src_stride, g.dst_stride,
                               g.width, g.height))
    return out


def p2p_gather_runs(dst: np.ndarray, src_ptrs, dst_offsets, lengths,
                    stream=None) -> None:
    staged = stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)
    if not staged:
        return  # 0 runs, or every run empty: enqueue nothing
    buf = _dst_buffer(dst)

    def copy_all() -> None:
        for r in staged:
            ctypes.memmove(ctypes.byref(buf, r.dst_offset),
                           ctypes.c_void_p(r.src), r.length)

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
                ctypes.memmove(ctypes.byref(buf, r.dst_offset + y * r.dst_stride),
                               ctypes.c_void_p(r.src + y * r.src_stride),
                               r.width)

    if stream is None:
        copy_all()
    else:
        stream.submit(copy_all)


def memcpy_peer_batch_async(dst: np.ndarray, src_ptrs, dst_offsets, lengths,
                            stream=None) -> None:
    staged = stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)
    if not staged:
        return
    buf = _dst_buffer(dst)

    def copy_one(r: StagedRun1D) -> None:
        ctypes.memmove(ctypes.byref(buf, r.dst_offset),
                       ctypes.c_void_p(r.src), r.length)

    if stream is None:
        for r in staged:
            copy_one(r)
    else:
        # Legacy seam: one stream task per run.
        for r in staged:
            stream.submit(lambda r=r: copy_one(r))
