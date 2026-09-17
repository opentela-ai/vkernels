"""Static fine-grained scheduling: tile-region dependencies (§10, Milestone 5).

After the phase target is correct, each full phase barrier can be relaxed
into only the dependencies its consumers need: a consumer task becomes
executable when all producers of its read regions have completed (§10.1).

This module builds the task-level dependency graph from the *tile regions*
declared by the task families, restricted to operator pairs that actually
have a whole-operator hazard (§5.2). It also implements the static option
of §10.4: fixed per-worker task sequences with waits on producer events,
including the augmented-graph acyclicity check (worker order can introduce
wait cycles even when the dataflow graph is acyclic).

Two worked examples from §10.2 are directly testable:

* a GELU tile depends only on the up-projection tile that produced its
  elements (elementwise RAW, per-tile precision);
* a down-projection output tile depends on *all* up-projection tiles that
  cover its K range — the full-K reduction cannot start from one
  available U tile. Starting earlier would require a different algorithm
  (split-K partials), not merely barrier elimination.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .operator_ir import HAZARD_RAW, HAZARD_WAW, HAZARD_WAR, compute_hazards
from .schedule_phase import PhaseSchedule
from .task_ir import TaskFamily

__all__ = ["TaskKey", "TaskDependencyGraph", "build_tile_dependencies", "StaticWorkerSchedule", "CyclicScheduleError"]

TaskKey = tuple[str, int]  # (family_id, task_id)


class CyclicScheduleError(Exception):
    """The worker-order-augmented dependency graph has a cycle (§10.4)."""


@dataclass
class TaskDependencyGraph:
    """Per-task predecessor sets derived from tile-region hazards (§10.1)."""

    tasks: dict[TaskKey, int] = field(default_factory=dict)  # key -> phase index
    preds: dict[TaskKey, set[TaskKey]] = field(default_factory=dict)

    def check_acyclic(self) -> None:
        indeg = {k: len(ps) for k, ps in self.preds.items()}
        from collections import deque

        succ: dict[TaskKey, set[TaskKey]] = {k: set() for k in self.preds}
        for k, ps in self.preds.items():
            for p in ps:
                succ[p].add(k)
        q = deque(k for k, d in indeg.items() if d == 0)
        seen = 0
        while q:
            k = q.popleft()
            seen += 1
            for s in succ[k]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    q.append(s)
        if seen != len(self.preds):
            raise CyclicScheduleError("tile dependency graph contains a cycle")

    def edges(self) -> int:
        return sum(len(ps) for ps in self.preds.values())


def build_tile_dependencies(
    schedule: PhaseSchedule,
    *,
    max_span: int | None = None,
) -> TaskDependencyGraph:
    """Expand whole-operator hazards into per-task edges (§10.1).

    ``max_span`` limits how many phases back a producer may be considered
    (None = unbounded). Only operator pairs with a recorded hazard are
    expanded, which keeps the pair count small for feed-forward graphs.
    """
    graph_dep = TaskDependencyGraph()
    families = [fam for phase in schedule.phases for fam in phase.families]
    fam_by_opid = {fam.op.opid: fam for fam in families}
    phase_of = {}
    for phase in schedule.phases:
        for fam in phase.families:
            phase_of[fam.op.opid] = phase.index
            for tid in range(fam.task_count):
                graph_dep.tasks[(fam.family_id, tid)] = phase.index
                graph_dep.preds.setdefault((fam.family_id, tid), set())

    hazards = compute_hazards([f.op for f in families])
    for hz in hazards:
        u, v = fam_by_opid[hz.producer], fam_by_opid[hz.consumer]
        if max_span is not None and phase_of[v.op.opid] - phase_of[u.op.opid] > max_span:
            continue
        for tu in range(u.task_count):
            for tv in range(v.task_count):
                if _task_pair_hazard(hz.kind, u, tu, v, tv):
                    graph_dep.preds[(v.family_id, tv)].add((u.family_id, tu))
    return graph_dep


def _task_pair_hazard(kind: str, u: TaskFamily, tu: int, v: TaskFamily, tv: int) -> bool:
    if kind == HAZARD_RAW:
        return any(wu.overlaps(rv) for wu in u.writes(tu) for rv in v.reads(tv))
    if kind == HAZARD_WAR:
        return any(ru.overlaps(wv) for ru in u.reads(tu) for wv in v.writes(tv))
    if kind == HAZARD_WAW:
        return any(wu.overlaps(wv) for wu in u.writes(tu) for wv in v.writes(tv))
    return False


@dataclass
class StaticWorkerSchedule:
    """Fixed per-worker task sequences with producer waits (§10.4)."""

    workers: int
    order: list[TaskKey]  # global topological execution order
    assignment: dict[TaskKey, int]  # task -> worker
    deps: TaskDependencyGraph

    def worker_sequence(self, w: int) -> list[TaskKey]:
        return [k for k in self.order if self.assignment[k] == w]

    def check_augmented_acyclic(self) -> None:
        """Worker-order edges + dependency edges must stay acyclic (§10.4)."""
        # Build combined edge set: dependencies plus same-worker sequence edges.
        edges: dict[TaskKey, set[TaskKey]] = {k: set() for k in self.order}
        for k, ps in self.deps.preds.items():
            for p in ps:
                edges[p].add(k)
        for w in range(self.workers):
            seq = self.worker_sequence(w)
            for a, b in zip(seq, seq[1:]):
                edges[a].add(b)
        # Kahn.
        from collections import deque

        indeg = {k: 0 for k in edges}
        for k, outs in edges.items():
            for o in outs:
                indeg[o] += 1
        q = deque(k for k, d in indeg.items() if d == 0)
        seen = 0
        while q:
            k = q.popleft()
            seen += 1
            for o in edges[k]:
                indeg[o] -= 1
                if indeg[o] == 0:
                    q.append(o)
        if seen != len(edges):
            raise CyclicScheduleError("static schedule infeasible: fixed worker order introduced a wait cycle (§10.4)")


def build_static_worker_schedule(
    deps: TaskDependencyGraph,
    workers: int,
) -> StaticWorkerSchedule:
    """Round-robin topological assignment: the simplest static policy.

    Tasks are appended to workers in global topological order (round-robin
    over workers); each worker's sequence is therefore a valid interleaving
    candidate, and :meth:`StaticWorkerSchedule.check_augmented_acyclic`
    verifies no wait cycle was introduced.
    """
    from collections import deque

    deps.check_acyclic()
    succ: dict[TaskKey, set[TaskKey]] = {k: set() for k in deps.preds}
    for k, ps in deps.preds.items():
        for p in ps:
            succ[p].add(k)
    indeg = {k: len(ps) for k, ps in deps.preds.items()}
    q = deque(sorted((k for k, d in indeg.items() if d == 0), key=lambda k: (deps.tasks[k], k)))
    order: list[TaskKey] = []
    while q:
        k = q.popleft()
        order.append(k)
        for s in succ[k]:
            indeg[s] -= 1
            if indeg[s] == 0:
                q.append(s)
    if len(order) != len(deps.preds):  # pragma: no cover - checked above
        raise CyclicScheduleError("dependency cycle during topological sort")
    assignment = {k: i % workers for i, k in enumerate(order)}
    sched = StaticWorkerSchedule(workers=workers, order=order, assignment=assignment, deps=deps)
    sched.check_augmented_acyclic()
    return sched
