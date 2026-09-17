"""Task IR: logical tiles and their execution contracts.

The central abstraction:

> A task describes what to compute for a logical tile. A persistent worker
> supplies the physical threads that execute it.

A :class:`TaskFamily` is a *compact* description of many task instances: the logical domain is a regular tile grid, ``decode(task_id)``
recovers tile coordinates, and region callbacks express exactly which
memory the task instance reads and writes. Region precision is what later
passes need to replace phase barriers with task dependencies.

Every task body obeys the initial contract: finite, block-local,
reentrant at task boundaries, no internal waits on unscheduled producers,
no reliance on fresh kernel-launch state or an assumed SM assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .operator_ir import Operator, Region

__all__ = ["TileDomain", "TaskFamily", "TASK_CONTRACT"]

TASK_CONTRACT = "block-local-finite-reentrant"


@dataclass(frozen=True)
class TileDomain:
    """A regular tile grid: per-dimension (extent, tile_size)."""

    dims: tuple[tuple[int, int], ...]

    @property
    def task_grid(self) -> tuple[int, ...]:
        return tuple((e + t - 1) // t for e, t in self.dims)

    @property
    def task_count(self) -> int:
        n = 1
        for g in self.task_grid:
            n *= g
        return n

    @property
    def extent(self) -> tuple[int, ...]:
        return tuple(e for e, _ in self.dims)

    def coords(self, task_id: int) -> tuple[int, ...]:
        """Tile indices (row-major over the tile grid)."""
        grid = self.task_grid
        out = [0] * len(grid)
        rem = task_id
        for i in range(len(grid) - 1, -1, -1):
            out[i] = rem % grid[i]
            rem //= grid[i]
        if rem != 0:
            raise IndexError(f"task id {task_id} outside domain of {self.task_count} tasks")
        return tuple(out)

    def box(self, task_id: int) -> tuple[tuple[int, int], ...]:
        """Per-dimension [lo, hi) element ranges, clipped to the extent."""
        out = []
        for (extent, tile), c in zip(self.dims, self.coords(task_id)):
            lo = c * tile
            hi = min(lo + tile, extent)
            out.append((lo, hi))
        return tuple(out)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"TileDomain(dims={self.dims}, tasks={self.task_count})"


# Region callbacks: task coords -> regions touched by that task instance.
RegionFn = Callable[[tuple[int, ...]], tuple[Region, ...]]


@dataclass
class TaskFamily:
    """A compact description of many task instances."""

    family_id: str
    kind: str  # lowering kind: gemm | layernorm | elementwise | embedding | ...
    op: Operator
    domain: TileDomain
    inputs: tuple[str, ...]  # tensor names, in lowering-defined order
    outputs: tuple[str, ...]
    params: dict = field(default_factory=dict)
    threads: int = 256  # thread-group contract: the whole worker block
    scratch_bytes: int = 0
    read_regions: Optional[RegionFn] = None
    write_regions: Optional[RegionFn] = None
    completion_contract: str = TASK_CONTRACT

    # -- task enumeration ---------------------------------------------------

    @property
    def task_count(self) -> int:
        return self.domain.task_count

    def coords(self, task_id: int) -> tuple[int, ...]:
        return self.domain.coords(task_id)

    def box(self, task_id: int) -> tuple[tuple[int, int], ...]:
        return self.domain.box(task_id)

    def reads(self, task_id: int) -> tuple[Region, ...]:
        if self.read_regions is None:
            return ()
        return self.read_regions(self.coords(task_id))

    def writes(self, task_id: int) -> tuple[Region, ...]:
        if self.write_regions is None:
            return ()
        return self.write_regions(self.coords(task_id))

    def describe(self) -> str:
        return f"{self.family_id}: kind={self.kind} tasks={self.task_count} domain={self.domain} threads={self.threads} scratch={self.scratch_bytes}B"
