"""Pure-Python reference implementations of the vkernels API.

This module mirrors the CPU reference implementations in ``csrc/vkernels``
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
