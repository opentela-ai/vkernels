"""Memory planning: global workspace with lifetime-based reuse (§9).

First-target policy (§9.1): all intermediates live in a preallocated global
workspace; parameters and persistent K/V storage are separate caller-owned
allocations; each worker gets a private scratch region sized by the
sequential-max rule of §7.2:

    S_block = S_runtime + max_j S_task,j

Reuse requires an ordering proof (§9.2): with phase-synchronous execution,
an allocation may be recycled once the last consuming *phase* has completed
(the grid barrier orders all readers before any later writer). Buffers with
disjoint phase lifetimes may therefore share storage; buffers with
overlapping lifetimes never do. This is the interval-coloring described in
§9.2 — memory planning evaluated together with the schedule, not in
isolation.

All sizes are computed in *elements* with 64-element alignment (256 B at
f32); :attr:`WorkspacePlan.total_bytes` converts to bytes at the target
dtype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .operator_ir import OperatorGraph, Region
from .schedule_phase import PhaseSchedule
from .task_ir import TaskFamily

__all__ = ["BufferPlan", "WorkspacePlan", "ScratchPlan", "plan_memory"]

ALIGN_ELEMENTS = 64  # 64 * 4 B = 256 B alignment
RUNTIME_SCRATCH_BYTES = 256  # per-worker runtime state (barrier flags, etc.)


def _align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class BufferPlan:
    """One intermediate's placement inside the workspace."""

    name: str  # tensor (view) name that created the storage
    storage_id: int
    numel: int
    first_phase: int
    last_phase: int
    offset: int = 0  # element offset, assigned by the planner
    live_at_end: bool = False  # e.g. the logits output

    @property
    def lifetime(self) -> tuple[int, int]:
        return (self.first_phase, self.last_phase)

    def overlaps_lifetime(self, other: "BufferPlan") -> bool:
        a0, a1 = self.lifetime
        b0, b1 = other.lifetime
        return a0 <= b1 and b0 <= a1


@dataclass
class WorkspacePlan:
    buffers: list[BufferPlan] = field(default_factory=list)
    total_elements: int = 0
    element_size: int = 4  # f32 target (§16.1 initial assumption: FP32)

    @property
    def total_bytes(self) -> int:
        return self.total_elements * self.element_size

    @property
    def naive_bytes(self) -> int:
        """Sum without reuse — shows what lifetime planning saves (§9.2)."""
        return sum(b.numel for b in self.buffers) * self.element_size


@dataclass
class ScratchPlan:
    """Per-worker scratch: runtime state + the max task footprint (§7.2)."""

    runtime_bytes: int = RUNTIME_SCRATCH_BYTES
    max_task_scratch: int = 0

    @property
    def per_worker_bytes(self) -> int:
        return _align_up(self.runtime_bytes + self.max_task_scratch, 128)

    def describe(self) -> str:
        return f"scratch: runtime={self.runtime_bytes}B + max(task)={self.max_task_scratch}B -> {self.per_worker_bytes}B per worker (sequential-max rule, §7.2)"


def _region_overlaps_storage(regions: tuple[Region, ...], storage_id: int) -> bool:
    for r in regions:
        if r.storage_id == storage_id:
            return True
    return False


def plan_memory(graph: OperatorGraph, schedule: PhaseSchedule, families: list[TaskFamily]) -> tuple[WorkspacePlan, ScratchPlan]:
    """Compute buffer lifetimes from the schedule and pack with reuse.

    Lifetimes: a fresh-buffer storage is born at the phase of the operator
    that writes it and dies after the last phase whose operators read or
    write it (effects are recovered from recorded regions, so *view* reads
    of an aliased storage — e.g. the qkv projection consumed through
    q/k/v slices — extend the lifetime correctly).
    """
    # Phase index per operator id.
    phase_of_op = {}
    for phase in schedule.phases:
        for fam in phase.families:
            phase_of_op[fam.op.opid] = phase.index

    # Fresh (compiler-planned) storages vs. external (caller) storages.
    fresh: dict[int, tuple[str, int]] = {}
    for name, tv in graph.tensors.items():
        if tv.storage_id in graph.fresh_storages:
            fresh.setdefault(tv.storage_id, (name, tv.numel))

    # Liveness per fresh storage from operator effects.
    lifetimes: dict[int, list[int]] = {sid: [] for sid in fresh}
    for op in graph.ops:
        ph = phase_of_op[op.opid]
        for sid in fresh:
            if _region_overlaps_storage(op.read_regions, sid) or _region_overlaps_storage(op.write_regions, sid):
                lifetimes[sid].append(ph)

    # The output tensor(s) of the final phase stay live at the end.
    final_phase = schedule.num_phases - 1

    buffers: list[BufferPlan] = []
    for sid, (name, numel) in fresh.items():
        phases_touched = lifetimes[sid]
        if not phases_touched:
            # Never touched after creation (dead write) — keep a one-phase life.
            phases_touched = [0]
        first = min(phases_touched)
        last = max(phases_touched)
        live_at_end = last == final_phase
        buffers.append(BufferPlan(name=name, storage_id=sid, numel=numel, first_phase=first, last_phase=last, live_at_end=live_at_end))

    # Interval packing with reuse (§9.2): process in first-phase order;
    # place each buffer at the lowest aligned offset that does not collide
    # with any already-placed buffer whose lifetime overlaps.
    buffers.sort(key=lambda b: (b.first_phase, -b.numel))
    placed: list[BufferPlan] = []
    for buf in buffers:
        offset = 0
        colliding = [p for p in placed if p.overlaps_lifetime(buf)]
        # First-fit over candidate offsets.
        candidate = 0
        while True:
            candidate = _align_up(candidate, ALIGN_ELEMENTS)
            conflict = None
            for p in colliding:
                if candidate < p.offset + _align_up(p.numel, ALIGN_ELEMENTS) and p.offset < candidate + _align_up(buf.numel, ALIGN_ELEMENTS):
                    conflict = p
                    break
            if conflict is None:
                offset = candidate
                break
            candidate = conflict.offset + _align_up(conflict.numel, ALIGN_ELEMENTS)
        buf.offset = offset
        placed.append(buf)

    total = 0
    for b in placed:
        total = max(total, b.offset + b.numel)
    plan = WorkspacePlan(buffers=placed, total_elements=total)

    scratch = ScratchPlan(max_task_scratch=max((f.scratch_bytes for f in families), default=0))
    return plan, scratch


def kv_cache_bytes(layers: int, batch: int, capacity: int, hidden: int, element_size: int = 4) -> int:
    """Dense FP32 KV storage: M_KV = 2 L B S C * 4 bytes (§9.1)."""
    return 2 * layers * batch * capacity * hidden * element_size
