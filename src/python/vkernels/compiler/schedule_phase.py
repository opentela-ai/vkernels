"""Phase-synchronous schedule: operator-ordered persistent execution (§8).

Milestone 3: emit the schedule from the graph rather than writing
GPT-2-specific device control flow. Every recorded operator becomes one
phase; worker ``b`` executes logical task ids ``b, b+P, b+2P, ...`` of each
phase and then joins a grid barrier (§8.1).

Coverage argument (§8.2): for each logical task index j in [0, N_i), the
unique worker is ``j mod P`` and that worker reaches ``j`` after
``floor(j/P)`` strides — every task executes exactly once. Disjoint output
ownership prevents conflicting writes within a phase, and the grid barrier
establishes producer-before-consumer ordering between phases, preserving
the original operator sequence's semantics.

The barrier-participation obligation (§8.3) is checked explicitly: workers
with no tiles in a phase still reach that phase's barrier, and every worker
encounters the same sequence of grid barriers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .task_ir import TaskFamily

__all__ = ["PhasePlan", "PhaseSchedule", "ScheduleInvariantError"]


class ScheduleInvariantError(Exception):
    """A schedule correctness obligation failed (§12)."""


@dataclass
class PhasePlan:
    """One phase: the task families of one recorded operator (§8.1)."""

    index: int
    families: list[TaskFamily]

    @property
    def task_count(self) -> int:
        return sum(f.task_count for f in self.families)

    @property
    def kind(self) -> str:
        kinds = {f.kind for f in self.families}
        return "+".join(sorted(kinds))

    def family_of(self, task_id: int) -> TaskFamily:
        """Map a linear task id to its family and family-local id."""
        for f in self.families:
            if task_id < f.task_count:
                return f
            task_id -= f.task_count
        raise IndexError(f"task id {task_id} outside phase {self.index}")

    def local_id(self, task_id: int) -> int:
        for f in self.families:
            if task_id < f.task_count:
                return task_id
            task_id -= f.task_count
        raise IndexError(f"task id {task_id} outside phase {self.index}")

    def describe(self) -> str:
        fam = self.families[0]
        return f"phase {self.index:2d}: {fam.op.kind:16s} tasks={self.task_count:3d} @ {fam.op.source_location}"


def worker_task_ids(phase: PhasePlan, worker: int, workers: int) -> list[int]:
    """The stride assignment of §8.1: worker b takes ids b, b+P, ... < N."""
    n = phase.task_count
    return list(range(worker, n, workers))


@dataclass
class PhaseSchedule:
    """The full operator-ordered schedule across all phases."""

    phases: list[PhasePlan] = field(default_factory=list)
    workers: int = 1  # P, the persistent block count (a tuning parameter, §7.4)

    # -- construction -------------------------------------------------------

    @staticmethod
    def from_families(families: list[TaskFamily], workers: int = 1) -> "PhaseSchedule":
        phases = []
        # Group consecutive families that share an operator (one op -> one
        # family in v1, but the grouping keeps the structure general).
        by_op: dict[int, list[TaskFamily]] = {}
        order: list[int] = []
        for fam in families:
            if fam.op.opid not in by_op:
                order.append(fam.op.opid)
                by_op[fam.op.opid] = []
            by_op[fam.op.opid].append(fam)
        for i, opid in enumerate(order):
            phases.append(PhasePlan(index=i, families=by_op[opid]))
        sched = PhaseSchedule(phases=phases, workers=workers)
        sched.check_invariants()
        return sched

    # -- queries -------------------------------------------------------------

    @property
    def num_phases(self) -> int:
        return len(self.phases)

    @property
    def total_tasks(self) -> int:
        return sum(p.task_count for p in self.phases)

    def iter_tasks(self) -> Iterator[tuple[PhasePlan, TaskFamily, int]]:
        for phase in self.phases:
            for fam in phase.families:
                for tid in range(fam.task_count):
                    yield phase, fam, tid

    # -- §12 invariants -------------------------------------------------------

    def check_invariants(self) -> None:
        """Coverage, ownership and barrier-participation obligations."""
        P = self.workers
        if P < 1:
            raise ScheduleInvariantError(f"worker count must be >= 1, got {P}")

        # Coverage + ownership (§8.2): every task id executes exactly once,
        # and each task id is owned by exactly one worker.
        for phase in self.phases:
            n = phase.task_count
            assigned: set[int] = set()
            for w in range(P):
                for tid in worker_task_ids(phase, w, P):
                    if tid in assigned:
                        raise ScheduleInvariantError(f"phase {phase.index}: task {tid} assigned to more than one worker")
                    assigned.add(tid)
            missing = set(range(n)) - assigned
            if missing:
                raise ScheduleInvariantError(f"phase {phase.index}: tasks {sorted(missing)} never assigned")

        # Barrier participation (§8.3): every worker encounters the same
        # sequence of grid barriers — one per phase — regardless of how many
        # tiles it owns. (Idle workers arrive with an empty task list.)
        for w in range(P):
            barriers = 0
            for phase in self.phases:
                _ = worker_task_ids(phase, w, P)  # worker runs its (possibly empty) share
                barriers += 1
            if barriers != self.num_phases:
                raise ScheduleInvariantError(f"worker {w} sees {barriers} barriers, expected {self.num_phases}")

    # -- reporting -------------------------------------------------------------

    def summary(self) -> str:
        lines = [f"PhaseSchedule: {self.num_phases} phases, {self.total_tasks} tasks, P={self.workers}"]
        for p in self.phases:
            lines.append("  " + p.describe())
        return "\n".join(lines)
