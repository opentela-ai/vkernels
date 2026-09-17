"""Task queue: the later ready-task runtime, specified and CPU-modeled (§11).

This is *not* needed for the phase-synchronous milestone. It exists so the
finite-DAG progress argument of §11.4 can be exercised on CPU:

* state: for each task ``t`` the number of unfinished predecessors
  ``r_t = |pred(t)|``; a task is enqueued at most once, when its remaining
  count reaches zero; roots are seeded at initialization;
* dispatch: only an elected thread performs queue operations, then
  broadcasts the selected task to its block (§11.2);
* completion: declared only when *all* tasks have retired, where retirement
  follows publication of outputs and successor notifications (§11.4) —
  an empty ready queue alone does not mean finished.

:func:`simulate_fifo_execution` is the CPU oracle for those rules; a real
device queue additionally needs the multi-producer/multi-consumer
publication protocol and acquire/release visibility argument of §11.3,
which no CPU simulation can establish.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

__all__ = ["TaskDeps", "ReadyTaskState", "simulate_fifo_execution", "ProgressError"]


class ProgressError(Exception):
    """The ready-task progress argument failed (§11.4)."""


@dataclass
class TaskDeps:
    """Predecessor/successor lists over task keys ``(family_id, task_id)``."""

    preds: dict[tuple[str, int], set[tuple[str, int]]] = field(default_factory=dict)

    def successors(self) -> dict[tuple[str, int], set[tuple[str, int]]]:
        succ: dict[tuple[str, int], set[tuple[str, int]]] = {k: set() for k in self.preds}
        for k, ps in self.preds.items():
            for p in ps:
                succ[p].add(k)
        return succ

    @property
    def tasks(self) -> list[tuple[str, int]]:
        return list(self.preds)

    def check_acyclic(self) -> None:
        """Kahn's algorithm; raises on cycles (a wait cycle deadlocks, §11.4)."""
        indeg = {k: len(ps) for k, ps in self.preds.items()}
        succ = self.successors()
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
            raise ProgressError("dependency graph contains a cycle")


@dataclass
class ReadyTaskState:
    """The §11.1 state: remaining counts, ready queue, retired count."""

    deps: TaskDeps
    remaining: dict = field(default_factory=dict)
    ready: deque = field(default_factory=deque)
    retired: int = 0
    _enqueued: set = field(default_factory=set)

    def initialize(self) -> "ReadyTaskState":
        """Seed roots — the in-kernel initialization phase of §11.1."""
        for k, ps in self.deps.preds.items():
            self.remaining[k] = len(ps)
            if not ps:
                self._enqueue(k)
        return self

    def _enqueue(self, key) -> None:
        if key not in self._enqueued:  # each task enqueued at most once
            self._enqueued.add(key)
            self.ready.append(key)

    def claim(self):
        """Pop one ready task or None (empty queue != finished, §11.4)."""
        return self.ready.popleft() if self.ready else None

    @property
    def finished(self) -> bool:
        return self.retired == len(self.deps.preds)

    def complete(self, key) -> list:
        """Publish outputs, notify successors, retire (§11.2)."""
        succ = self.deps.successors()[key]
        for s in succ:
            self.remaining[s] -= 1
            if self.remaining[s] == 0:
                self._enqueue(s)
        self.retired += 1
        return sorted(succ)


def simulate_fifo_execution(deps: TaskDeps, *, num_workers: int = 4) -> dict:
    """Model §11.2's block-cooperative dispatch loop on CPU.

    Workers spin on ``claim -> execute -> complete``; execution finishes
    only when all tasks retired. Raises :class:`ProgressError` if the
    workers deadlock (queue empty, work unfinished — §11.4's failure mode).
    """
    deps.check_acyclic()
    state = ReadyTaskState(deps=deps).initialize()
    spins_without_progress = 0
    max_spins = len(deps.preds) * num_workers + 16
    while not state.finished:
        progressed = False
        for _ in range(num_workers):
            key = state.claim()
            if key is None:
                spins_without_progress += 1
                continue
            state.complete(key)  # task bodies are trivial in this model
            progressed = True
        if not progressed:
            if state.finished:
                break
            raise ProgressError(f"ready queue empty with {len(deps.preds) - state.retired} unfinished tasks")
        spins_without_progress = 0
        if spins_without_progress > max_spins:  # pragma: no cover - defensive
            raise ProgressError("runaway spin")
    return {"retired": state.retired, "total": len(deps.preds)}
