"""Communication primitives (Python side of ``src/c/vkernels/comm``).

Topology, in-process channels, ring all-reduce, compute/communication
overlap, and the p2p run-list gather. Everything here is host-side and
testable without a GPU; the compiled backend calls the C++ implementations
and the fallback mirrors them in pure Python.

The p2p gather API matches the contract documented in ``docs/p2p-gather.md``:

    p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, *, stream=None)

where ``src_ptrs`` are raw byte addresses (e.g. ``ctypes.addressof(...)`` or
``arr.__array_interface__["data"][0]``) of peer-accessible memory. On the
host build the sources must simply be readable memory.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from vkernels import _backend
from vkernels._types import (
    Gather2DRun,
    Result,
    StagedRun1D,
    StagedRun2D,
    Topology,
)

_core = _backend.load_extension()
_COMPILED = _core is not None
if _COMPILED:
    _impl = _core.comm
else:
    from vkernels import _fallback

    _impl = _fallback

__all__ = [
    "Topology",
    "Gather2DRun",
    "StagedRun1D",
    "StagedRun2D",
    "Result",
    "ring_rank",
    "build_ring_topology",
    "BlockingQueue",
    "MockChannel",
    "make_ring_channels",
    "ring_allreduce_rank",
    "ring_allreduce",
    "OverlapExecutor",
    "stage_runs_1d",
    "stage_runs_2d",
    "p2p_gather_runs",
    "p2p_gather_runs_2d",
    "memcpy_peer_batch_async",
    "kv_gather_layer",
]


def _as_addrs(values: Sequence[int], name: str) -> list[int]:
    """Coerce a sequence of addresses to non-negative Python ints."""
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"{name} must contain integer addresses")
        i = int(v)
        if i < 0:
            raise ValueError(f"{name} addresses must be non-negative")
        out.append(i)
    return out


def _as_counts(values: Sequence[int], name: str) -> list[int]:
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise TypeError(f"{name} must contain integers")
        i = int(v)
        if i < 0:
            raise ValueError(f"{name} values must be non-negative")
        out.append(i)
    return out


def _as_dst(dst, writable: bool = True) -> np.ndarray:
    """Validate a uint8 destination buffer (no silent dtype copies)."""
    if not isinstance(dst, np.ndarray):
        dst = np.asarray(dst)
    if dst.dtype != np.uint8:
        raise TypeError(f"dst must be a uint8 numpy array, got {dst.dtype}")
    if not dst.flags.c_contiguous:
        raise ValueError("dst must be C-contiguous")
    if writable and not dst.flags.writeable:
        raise ValueError("dst must be writable")
    return dst


def _impl_stream(stream):
    """The backend stream object behind a public :class:`~vkernels.core.Stream`."""
    if stream is None:
        return None
    impl = getattr(stream, "_impl", None)
    if impl is None:
        raise TypeError("stream must be a vkernels.core.Stream or None")
    return impl


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


def ring_rank(rank: int, world: int) -> Topology:
    """Ring topology for one rank.

    Args:
        rank: this rank's id, in ``[0, world)``.
        world: number of ranks in the ring (>= 1).

    Returns:
        A :class:`Topology` with ``next = (rank + 1) % world`` and
        ``prev = (rank - 1) % world``.

    Raises:
        ValueError: if ``world`` is not positive or ``rank`` is out of range.

    Example:
        >>> ring_rank(1, 4)
        Topology(rank=1, world=4, next=2, prev=0)
    """
    return Topology(*_topology_tuple(rank, world))


def build_ring_topology(world: int) -> list[Topology]:
    """One :class:`Topology` per rank, for a ring of ``world`` ranks.

    Args:
        world: number of ranks (>= 1).

    Returns:
        A list of ``world`` topologies, in rank order.

    Raises:
        ValueError: if ``world`` is not positive.

    Example:
        >>> build_ring_topology(2)
        [Topology(rank=0, world=2, next=1, prev=1), Topology(rank=1, world=2, next=0, prev=0)]
    """
    if world <= 0:
        raise ValueError("world must be positive")
    return [ring_rank(r, world) for r in range(world)]


def _topology_tuple(rank: int, world: int):
    """(rank, world, next, prev) from the active backend."""
    t = _impl.ring_rank(rank, world)
    return t.rank, t.world, t.next, t.prev


# ---------------------------------------------------------------------------
# channels + ring all-reduce
# ---------------------------------------------------------------------------


class BlockingQueue:
    """Thread-safe blocking queue of float32 chunks.

    Links two mock channels together; one channel's ``out`` is another's
    ``in``. Constructed internally by :func:`make_ring_channels`.
    """

    def __init__(self):
        if _COMPILED:
            self._impl = _impl.BlockingQueue()
        else:
            self._impl = _fallback.BlockingQueue()

    def push(self, chunk) -> None:
        """Append one chunk (converted to float32)."""
        self._impl.push(chunk)

    def pop(self) -> np.ndarray:
        """Remove and return the next chunk; blocks until one is available."""
        return self._impl.pop()

    def close(self) -> None:
        """Mark the queue closed."""
        self._impl.close()

    def closed(self) -> bool:
        """True once :meth:`close` has been called."""
        return self._impl.closed()


class MockChannel:
    """In-process channel: sends go to ``out``, receives come from ``in``.

    Obtained from :func:`make_ring_channels`; also constructible directly
    from two :class:`BlockingQueue` objects.
    """

    def __init__(self, out: BlockingQueue, in_: BlockingQueue):
        if _COMPILED:
            self._impl = _impl.MockChannel(out._impl, in_._impl)
        else:
            self._impl = _fallback.MockChannel(out._impl, in_._impl)

    def send(self, chunk) -> None:
        """Blocking send of one chunk to the peer on the other end."""
        self._impl.send(chunk)

    def recv(self) -> np.ndarray:
        """Blocking receive of the next chunk from the peer."""
        return self._impl.recv()

    def closed(self) -> bool:
        """True once the peer has finished producing for this rank."""
        return self._impl.closed()


def make_ring_channels(world: int) -> list[MockChannel]:
    """Build ``world`` mock channels arranged in a ring.

    ``channels[r].send(chunk)`` is received by ``channels[(r + 1) %
    world].recv()``. This is the transport used by :func:`ring_allreduce`.

    Args:
        world: number of channels (>= 1).

    Returns:
        A list of ``world`` channels, one per rank.
    """
    if _COMPILED:
        return list(_impl.make_ring_channels(world))
    return _impl.make_ring_channels(world)


def ring_allreduce_rank(local: np.ndarray, rank: int, world: int, next_ch,
                        prev_ch) -> None:
    """Run one rank of a ring all-reduce; ``local`` is summed in place.

    On return, ``local`` holds the element-wise sum across every rank.

    Args:
        local: writable float32 numpy array; its length must be divisible by
            ``world``.
        rank: this rank's id in ``[0, world)``.
        world: number of ranks.
        next_ch: channel sending to ``rank + 1`` (see :func:`make_ring_channels`).
        prev_ch: channel receiving from ``rank - 1``.

    Raises:
        ValueError: on an invalid ``world``/``rank`` or a length not
            divisible by ``world``.

    Note:
        For a multi-rank simulation, prefer :func:`ring_allreduce`, which
        drives every rank concurrently in one process.
    """
    if not isinstance(local, np.ndarray) or local.dtype != np.float32:
        raise TypeError("local must be a float32 numpy array")
    if not local.flags.writeable:
        raise ValueError("local must be writable")
    # make_ring_channels returns raw backend channels (no `_impl`); accept
    # those as well as wrapped MockChannel objects.
    _impl.ring_allreduce_rank(local, rank, world,
                              getattr(next_ch, "_impl", next_ch),
                              getattr(prev_ch, "_impl", prev_ch))


def ring_allreduce(locals_: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Simulate a ring all-reduce across all ``world`` ranks in one process.

    Args:
        locals_: exactly ``world`` float32 arrays of equal length, divisible
            by ``world``; each is one rank's local buffer.

    Returns:
        A list of ``world`` arrays, each holding the element-wise sum of all
        inputs (``locals_[i]`` is untouched).

    Raises:
        ValueError: if the input list is empty or the buffers disagree in
            length.

    Example:
        >>> a = np.array([1.0, 2.0], dtype=np.float32)
        >>> b = np.array([3.0, 4.0], dtype=np.float32)
        >>> ring_allreduce([a, b])
        [array([4., 6.], dtype=float32), array([4., 6.], dtype=float32)]
    """
    if _COMPILED:
        return list(_impl.ring_allreduce(locals_))
    return _impl.ring_allreduce(list(locals_))


class OverlapExecutor:
    """Overlap compute and communication across two streams.

    ``run(iters, compute, comm)`` submits ``compute(i)`` on stream A and
    ``comm(i, value)`` on stream B for each iteration. The per-iteration data
    dependency (comm needs compute's output) is honoured with a future, so
    iteration ``i+1``'s compute can proceed while iteration ``i``'s comm is
    still in flight.
    """

    def __init__(self):
        if _COMPILED:
            self._impl = _impl.OverlapExecutor()
        else:
            self._impl = _fallback.OverlapExecutor()

    def uses_two_streams(self) -> bool:
        """Always True: two distinct backing streams are in use."""
        return self._impl.uses_two_streams()

    def run(self, iters: int, compute, comm) -> Result:
        """Run ``iters`` iterations of compute/comm overlap.

        Args:
            iters: number of iterations.
            compute: ``compute(i) -> int`` — produces iteration ``i``'s value
                on the compute stream.
            comm: ``comm(i, value)`` — consumes the value on the comm stream.

        Returns:
            A :class:`Result` with ``compute_count == comm_count == iters``.

        Example:
            >>> ex = OverlapExecutor()
            >>> res = ex.run(4, lambda i: i * 2, lambda i, v: None)
            >>> res.compute_count
            4
        """
        if _COMPILED:
            a, b = self._impl.run(iters, compute, comm)
            return Result(int(a), int(b))
        return self._impl.run(iters, compute, comm)


# ---------------------------------------------------------------------------
# p2p run-list gather
# ---------------------------------------------------------------------------


def _unpack_runs(runs: Sequence[Gather2DRun]) -> list[Gather2DRun]:
    """Normalise a run list: tuples ``(src, src_stride, dst_offset,
    dst_stride, width, height)`` become :class:`Gather2DRun`."""
    out = []
    for r in runs:
        if isinstance(r, Gather2DRun):
            out.append(r)
        elif isinstance(r, (tuple, list)) and len(r) == 6:
            out.append(Gather2DRun(*r))
        else:
            raise TypeError(
                "runs must be Gather2DRun instances or 6-tuples "
                "(src, src_stride, dst_offset, dst_stride, width, height)"
            )
    return out


def _parallel_arrays(runs: Sequence[Gather2DRun]):
    src_ptrs = [r.src for r in runs]
    src_strides = [r.src_stride for r in runs]
    dst_offsets = [r.dst_offset for r in runs]
    dst_strides = [r.dst_stride for r in runs]
    widths = [r.width for r in runs]
    heights = [r.height for r in runs]
    return src_ptrs, src_strides, dst_offsets, dst_strides, widths, heights


def stage_runs_1d(dst, src_ptrs: Sequence[int], dst_offsets: Sequence[int],
                  lengths: Sequence[int]) -> list[StagedRun1D]:
    """Validate a 1-D run list against ``dst`` and stage it.

    Performs the full contract check — non-null sources, destination
    capacity, source/destination non-overlap, mutually disjoint output runs —
    and raises ``ValueError`` on violation. Empty runs are dropped.

    Args:
        dst: the uint8 destination buffer (capacity and base address come
            from it).
        src_ptrs: ``num_runs`` byte addresses of the source ranges.
        dst_offsets: byte offsets into ``dst`` for each run.
        lengths: byte lengths of each run.

    Returns:
        A list of validated :class:`StagedRun1D` descriptors.

    Example:
        >>> src = np.array([1, 2, 3], dtype=np.uint8)
        >>> dst = np.zeros(4, dtype=np.uint8)
        >>> stage_runs_1d(dst, [src.__array_interface__["data"][0]], [1], [3])
        [StagedRun1D(src=..., dst_offset=1, length=3)]
    """
    dst = _as_dst(dst, writable=False)
    src_ptrs = _as_addrs(src_ptrs, "src_ptrs")
    dst_offsets = _as_counts(dst_offsets, "dst_offsets")
    lengths = _as_counts(lengths, "lengths")
    if _COMPILED:
        return [
            StagedRun1D(*t) for t in _impl.stage_runs_1d(
                dst, src_ptrs, dst_offsets, lengths)
        ]
    return _impl.stage_runs_1d(dst, src_ptrs, dst_offsets, lengths)


def stage_runs_2d(dst, runs: Sequence[Gather2DRun]) -> list[StagedRun2D]:
    """Validate a 2-D run list against ``dst`` and stage it.

    Args:
        dst: the uint8 destination buffer.
        runs: a sequence of :class:`Gather2DRun` (or 6-tuples).

    Returns:
        A list of validated :class:`StagedRun2D` descriptors.

    Raises:
        ValueError: if a tile exceeds the destination, or ``width`` exceeds
            a stride.
    """
    dst = _as_dst(dst, writable=False)
    runs = _unpack_runs(runs)
    if _COMPILED:
        arrays = _parallel_arrays(runs)
        return [
            StagedRun2D(*t) for t in _impl.stage_runs_2d(dst, *arrays)
        ]
    return _impl.stage_runs_2d(dst, runs)


def p2p_gather_runs(dst, src_ptrs: Sequence[int], dst_offsets: Sequence[int],
                    lengths: Sequence[int], *, stream=None) -> None:
    """Copy every 1-D run into ``dst`` in a single operation.

    For each ``i``, copies ``lengths[i]`` bytes from address ``src_ptrs[i]``
    to ``dst[dst_offsets[i]:dst_offsets[i] + lengths[i]]``. The run list is
    validated up front (capacity, disjoint outputs, non-overlap with
    sources); ``num_runs == 0`` is a valid no-op.

    Args:
        dst: writable uint8 numpy array — the local destination buffer.
        src_ptrs: byte addresses of the source ranges (peer-accessible UVA
            or IPC-mapped pointers under CUDA; readable memory on the host
            build). The addresses must stay valid until the work completes.
        dst_offsets: byte offsets into ``dst`` per run.
        lengths: byte lengths per run.
        stream: optional :class:`~vkernels.core.Stream`. When omitted the
            copies run to completion before returning; otherwise the work is
            enqueued and the call returns immediately.

    Raises:
        ValueError: if the run list violates the contract.

    Example:
        >>> src = np.arange(6, dtype=np.uint8)
        >>> dst = np.zeros(6, dtype=np.uint8)
        >>> p2p_gather_runs(dst, [src.__array_interface__["data"][0]], [2], [6])
        >>> dst.tolist()
        [0, 0, 0, 1, 2, 3]
    """
    dst = _as_dst(dst)
    src_ptrs = _as_addrs(src_ptrs, "src_ptrs")
    dst_offsets = _as_counts(dst_offsets, "dst_offsets")
    lengths = _as_counts(lengths, "lengths")
    impl_stream = _impl_stream(stream)
    _impl.p2p_gather_runs(dst, src_ptrs, dst_offsets, lengths, impl_stream)


def p2p_gather_runs_2d(dst, runs: Sequence[Gather2DRun], *, stream=None) -> None:
    """Copy every 2-D strided tile into ``dst`` in a single operation.

    Each run copies a ``height`` x ``width``-byte tile from a row-major
    region, honouring its own source and destination row strides.

    Args:
        dst: writable uint8 numpy array — the local destination buffer.
        runs: a sequence of :class:`Gather2DRun` (or 6-tuples
            ``(src, src_stride, dst_offset, dst_stride, width, height)``).
        stream: optional :class:`~vkernels.core.Stream` (see
            :func:`p2p_gather_runs`).

    Raises:
        ValueError: if a tile violates the contract.
    """
    dst = _as_dst(dst)
    runs = _unpack_runs(runs)
    impl_stream = _impl_stream(stream)
    if _COMPILED:
        _impl.p2p_gather_runs_2d(dst, *_parallel_arrays(runs), impl_stream)
    else:
        _impl.p2p_gather_runs_2d(dst, runs, impl_stream)


def memcpy_peer_batch_async(dst, src_ptrs: Sequence[int],
                            dst_offsets: Sequence[int],
                            lengths: Sequence[int], *, stream=None) -> None:
    """Legacy seam: one copy per run (the predecessor of
    :func:`p2p_gather_runs`).

    Same contract and lifetime rules as :func:`p2p_gather_runs`, kept so
    benchmarks can sweep run counts against the single-launch version. With
    a stream, ``stream.submitted()`` grows by the number of runs here versus
    by 1 for :func:`p2p_gather_runs`.
    """
    dst = _as_dst(dst)
    src_ptrs = _as_addrs(src_ptrs, "src_ptrs")
    dst_offsets = _as_counts(dst_offsets, "dst_offsets")
    lengths = _as_counts(lengths, "lengths")
    impl_stream = _impl_stream(stream)
    _impl.memcpy_peer_batch_async(dst, src_ptrs, dst_offsets, lengths,
                                  impl_stream)


def kv_gather_layer(k_src, v_src, slot_ids, dst, *, stream=None) -> None:
    """Fused indexed K/V gather for one layer (issue #2).

    Reads ``num_pages * page_size`` indexed source slots from ``k_src`` /
    ``v_src`` and writes the packed ``[num_pages, page_size, 2, num_kv_heads,
    head_dim]`` destination, exactly:

        dst[:, :, 0] = k_src[slot_ids]
        dst[:, :, 1] = v_src[slot_ids]

    This fuses the two separate advanced-index gathers the KVAAS
    ``pack_pages`` path performs today into a single operation, the reverse
    of :func:`p2p_kv_restore` (issue #4) and the building block of the
    peer-KV donation/eviction paths (issue #36).

    Args:
        k_src, v_src: ``[num_slots, num_kv_heads, head_dim]`` C-contiguous
            arrays with ``itemsize == 2``. BF16 is passed as a ``uint16``
            view (numpy has no native BF16); FP16 as ``np.float16``. The
            gather is a raw-byte copy, so the bit pattern is preserved
            exactly for both dtypes.
        slot_ids: ``[num_pages, page_size]`` C-contiguous ``int32`` or
            ``int64`` array of source slots in ``[0, num_slots)``. Slots may
            repeat (gather semantics, unlike the restore's unique-destination
            scatter) and be in any order (non-monotonic).
        dst: ``[num_pages, page_size, 2, num_kv_heads, head_dim]``
            C-contiguous writable array of the same dtype as ``k_src``/
            ``v_src``. Non-default strides are rejected explicitly (a silent
            copy would write into a throwaway buffer).
        stream: optional :class:`~vkernels.core.Stream`. When omitted the
            gather runs to completion before returning; otherwise it is
            enqueued as a single task and the call returns immediately (no
            device-wide synchronization).

    Raises:
        TypeError: if any array has the wrong dtype.
        ValueError: if shapes/strides are inconsistent, ``slot_ids`` is out
            of range, or ``dst`` is not writable.

    Example:
        >>> import numpy as np
        >>> k = np.arange(2*1*4, dtype=np.float16).reshape(2, 1, 4)
        >>> v = np.full((2, 1, 4), 99, dtype=np.float16)
        >>> slot_ids = np.array([[0, 1]], dtype=np.int32)
        >>> dst = np.zeros((1, 2, 2, 1, 4), dtype=np.float16)
        >>> kv_gather_layer(k, v, slot_ids, dst)
        >>> dst[0, 0, 0].tolist(), dst[0, 1, 1].tolist()
        ([0.0, 1.0, 2.0, 3.0], [99.0, 99.0, 99.0, 99.0])
    """
    impl_stream = _impl_stream(stream)
    if _COMPILED:
        _impl.kv_gather_layer(k_src, v_src, slot_ids, dst, impl_stream)
    else:
        _impl.kv_gather_layer(k_src, v_src, slot_ids, dst, stream=impl_stream)
