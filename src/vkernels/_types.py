"""Shared value types for the vkernels Python bindings.

These dataclasses mirror the structs in the C++ headers (``Topology``,
``Gather2DRun``, ``StagedRun1D``, ``StagedRun2D``) plus the small result
record returned by :class:`~vkernels.comm.OverlapExecutor`. They live in
their own module (no dependencies) so both the compiled backend and the
pure-Python fallback can construct them without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topology:
    """Ring topology for one rank (mirrors ``comm::Topology``).

    Attributes:
        rank: this rank's id in ``[0, world)``.
        world: number of ranks in the ring.
        next: the rank this rank sends to, ``(rank + 1) % world``.
        prev: the rank this rank receives from, ``(rank - 1) % world``.
    """

    rank: int
    world: int
    next: int
    prev: int


@dataclass(frozen=True)
class Gather2DRun:
    """One strided 2-D copy run (mirrors ``comm::Gather2DRun``).

    Copies a ``height`` x ``width``-byte tile from a row-major region at
    address ``src`` (row stride ``src_stride`` bytes) into the destination
    buffer at byte offset ``dst_offset`` (row stride ``dst_stride`` bytes).
    ``width`` must not exceed either stride.

    Attributes:
        src: byte address of the source region's first row.
        src_stride: bytes between source rows (>= width).
        dst_offset: byte offset of the tile's first row in the destination.
        dst_stride: bytes between destination rows (>= width).
        width: bytes copied per row.
        height: number of rows copied.
    """

    src: int
    src_stride: int
    dst_offset: int
    dst_stride: int
    width: int
    height: int


@dataclass(frozen=True)
class StagedRun1D:
    """A validated, owned 1-D copy run (mirrors ``comm::StagedRun1D``).

    Produced by :func:`vkernels.comm.stage_runs_1d` after the full contract
    check; safe to hand to a kernel launcher.

    Attributes:
        src: byte address of the source range.
        dst_offset: byte offset into the destination buffer.
        length: number of bytes copied.
    """

    src: int
    dst_offset: int
    length: int


@dataclass(frozen=True)
class StagedRun2D:
    """A validated, owned 2-D copy run (mirrors ``comm::StagedRun2D``).

    Attributes:
        src: byte address of the source region's first row.
        dst_offset: byte offset of the tile's first row in the destination.
        src_stride: bytes between source rows.
        dst_stride: bytes between destination rows.
        width: bytes copied per row.
        height: number of rows copied.
    """

    src: int
    dst_offset: int
    src_stride: int
    dst_stride: int
    width: int
    height: int


@dataclass(frozen=True)
class Result:
    """Outcome of :meth:`vkernels.comm.OverlapExecutor.run`.

    Attributes:
        compute_count: number of compute tasks that ran.
        comm_count: number of comm tasks that ran.
    """

    compute_count: int
    comm_count: int
