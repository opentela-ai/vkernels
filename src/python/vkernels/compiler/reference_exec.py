"""Reference executor: interpret the compiled schedule on CPU (§15.1).

This is the validation vehicle the design explicitly endorses:

> Randomized CPU execution can check graph logic, but cannot validate
> compiled GPU synchronization or memory ordering. (§15.1)

What it *does* validate, end to end against an independent oracle:

* capture fidelity — the schedule computes the model's arithmetic;
* coverage/ownership — the stride assignment runs each logical task
  exactly once and every worker meets every phase barrier (§8.2, §8.3);
* memory planning — all tensor accesses go through storage-offset views
  mirroring device pointer math, so wrong buffer reuse corrupts results or
  trips the NaN canaries;
* cache-validity masking — attention never reads beyond position ``p``
  even when cache tails hold NaN (§5.3);
* one-launch accounting — a single invocation is exactly one simulated
  kernel with one grid barrier per phase (§15.2's event-count check,
  modulo the device itself).

The executor models the *device* memory precisely: one flat array per
storage; every tensor (including aliased views like the transposed tied
head or the q/k/v slices of the QKV projection) is materialized as a
strided view at its recorded offset. Reference arithmetic runs in float64
for oracle stability; the IR's numerical contract (f32 accumulation,
declared reduction orders) is what the generated device code implements.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .memory import WorkspacePlan
from .schedule_phase import PhasePlan, PhaseSchedule, worker_task_ids
from .task_ir import TaskFamily

__all__ = ["ExecutionTrace", "PhaseStat", "ReferenceExecutor", "ExecutorError", "decode_e4m3"]


def _sigmoid(g):
    """Overflow-stable elementwise sigmoid in fp64: 1/(1+e^-g) for g>=0,
    e^g/(1+e^g) otherwise (avoids exp overflow warnings for gate <= -103,
    where the device fp32 sigmoid saturates to exactly 0)."""













from .reference_types import ExecutorError as ExecutorError, ExecutionTrace as ExecutionTrace, PhaseStat as PhaseStat, decode_e4m3 as decode_e4m3
from .reference_dense import DenseReference
from .reference_attention import AttentionReference
from .reference_recurrent import RecurrentReference

class ReferenceExecutor(DenseReference, AttentionReference, RecurrentReference):
    """Executes a :class:`PhaseSchedule` over numpy storage arrays.

    ``storage_arrays`` maps storage id -> flat 1-D numpy array (externals
    use the caller's arrays; fresh buffers are carved from the workspace).
    ``tensor_arrays`` (name -> ndarray) is derived by layout math, exactly
    as device pointers would be derived (offset + strides).
    """

    def __init__(
        self,
        schedule: PhaseSchedule,
        *,
        workers: int,
        storage_arrays: dict[int, np.ndarray],
        graph,
        workspace_plan: Optional[WorkspacePlan] = None,
        canary: bool = True,
    ):
        self.schedule = schedule
        self.workers = workers
        self.graph = graph
        self.workspace_plan = workspace_plan
        self.canary = canary
        self.storage_arrays = dict(storage_arrays)
        self.trace = ExecutionTrace()
        if canary and workspace_plan is not None:
            # A new invocation re-arms every canary: buffer contents do not
            # survive across invocations (the workspace is call-private).
            for buf in workspace_plan.buffers:
                self.storage_arrays[buf.storage_id].fill(np.nan)
        self._tensor_arrays = self._materialize_tensors()
        from .contracts import CONTRACTS
        self._bodies = {spec.task_kind: getattr(self, spec.reference_body) for spec in CONTRACTS.values()}
        self._barrier_state = None

    # ------------------------------------------------------------------
    # Memory model
    # ------------------------------------------------------------------

    def _materialize_tensors(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, tv in self.graph.tensors.items():
            base = self.storage_arrays.get(tv.storage_id)
            if base is None:
                raise ExecutorError(f"no storage array for tensor {name!r} (storage s{tv.storage_id})")
            itemsize = base.dtype.itemsize
            window = base[tv.offset :]
            out[name] = np.lib.stride_tricks.as_strided(
                window,
                shape=tv.shape,
                strides=tuple(s * itemsize for s in tv.strides),
                writeable=True,
            )
        return out

    def tensor(self, name: str) -> np.ndarray:
        return self._tensor_arrays[name]

    # ------------------------------------------------------------------
    # The generated megakernel's device algorithm (§8.1)
    # ------------------------------------------------------------------

    def run(self, scalars: dict[str, int]) -> ExecutionTrace:
        from .runtime.synchronization import SimulatedGridBarrier

        self.trace.kernel_launches += 1  # ONE launch for the whole model (§15.2)
        barrier = SimulatedGridBarrier(workers=self.workers)
        for phase in self.schedule.phases:
            # Worker b executes logical task ids b, b+P, ... (§8.1).
            per_worker = []
            for w in range(self.workers):
                count = 0
                for tid in worker_task_ids(phase, w, self.workers):
                    self._execute_task(phase, tid, scalars)
                    count += 1
                per_worker.append(count)
                barrier.arrive(w)  # idle workers participate too (§8.3)
            gen = barrier.release()
            self.trace.grid_barriers += 1
            self.trace.task_executions += sum(per_worker)
            self.trace.phases.append(
                PhaseStat(
                    index=phase.index,
                    kind=phase.kind,
                    tasks=phase.task_count,
                    per_worker=per_worker,
                    barrier_generation=gen,
                )
            )
            if self.canary and self.workspace_plan is not None:
                self._canary_check(phase)
        return self.trace

    def _execute_task(self, phase: PhasePlan, task_id: int, scalars: dict[str, int]) -> None:
        family = phase.family_of(task_id)
        local = phase.local_id(task_id)
        coords = family.coords(local)
        body = self._bodies.get(family.kind)
        if body is None:
            raise ExecutorError(f"no reference body for task kind {family.kind!r}")
        body(family, coords, scalars)

    # ------------------------------------------------------------------
    # Canary: workspace-reuse safety (§9.2)
    # ------------------------------------------------------------------

    def _canary_check(self, phase: PhasePlan) -> None:
        """Workspace-reuse safety (§9.2) via a re-arming NaN canary.

        Protocol: every fresh buffer starts as all-NaN (the workspace is
        NaN-initialized). After each phase's barrier:

        * every buffer whose lifetime has not started yet must still be
          all-NaN — *unless* a buffer that is live right now legally shares
          its region (the planner only shares storage between disjoint
          lifetimes, so a live co-occupant explains any non-NaN content);
        * every buffer that just died (``last_phase == phase.index`` and
          not live at the end) is re-poisoned to NaN, re-arming the canary
          for whichever later buffer reuses its storage.

        Because the planner only shares storage between buffers with
        disjoint lifetimes, poisoning a dead buffer cannot corrupt a live
        one — if that proof were wrong, the poison would corrupt data and
        the oracle comparison would fail. Both failure directions are
        therefore observable.
        """
        ws = self.workspace_plan
        if not self.canary or ws is None:
            return
        for buf in ws.buffers:
            arr = self.storage_arrays[buf.storage_id]
            if buf.first_phase > phase.index:
                if self._has_live_cooccupant(buf, phase.index):
                    continue
                if not np.isnan(arr).all():
                    raise ExecutorError(f"phase {phase.index} wrote into buffer {buf.name!r} before its lifetime starts at phase {buf.first_phase} (§9.2 reuse proof)")
            elif buf.last_phase == phase.index and not buf.live_at_end:
                arr.fill(np.nan)  # free the storage; re-arm the canary

    def _has_live_cooccupant(self, buf, phase_index: int) -> bool:
        lo0, hi0 = buf.offset, buf.offset + buf.numel
        for other in self.workspace_plan.buffers:
            if other is buf or not (other.first_phase <= phase_index <= other.last_phase):
                continue
            lo1, hi1 = other.offset, other.offset + other.numel
            if lo0 < hi1 and lo1 < hi0:
                return True
        return False

    # ------------------------------------------------------------------
    # Task bodies: tile-exact reference semantics
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Position resolution: shared scalar p or per-row ragged decode (#93)
    # ------------------------------------------------------------------

    def _row_pos_value(self, fam: TaskFamily, b: int, scalars) -> int:
        """Row ``b``'s decode position: the scalar ``p`` or, in per-row form
        (issue #93), ``positions[b]`` read from the external row tensor."""
        if fam.params.get("position_form", "scalar") == "row":
            pos_t = self.tensor(fam.params["position"])
            return int(pos_t[b])
        return scalars["p"]

    def _valid_len(self, fam: TaskFamily, b: int, scalars) -> int:
        """Row ``b``'s valid cache length: ``pos[b] + 1`` per row (§5.3,
        per-row NaN-tail contract) or the scalar form ``p + 1``."""
        return self._row_pos_value(fam, b, scalars) + 1


    @staticmethod
    def _gemm_box(fam: TaskFamily, coords):
        (m_extent, m_tile), (n_extent, n_tile) = fam.domain.dims
        m0 = coords[0] * m_tile
        n0 = coords[1] * n_tile
        return (m0, min(m0 + m_tile, m_extent)), (n0, min(n0 + n_tile, n_extent))

























    # ------------------------------------------------------------------
    # MoE decode reference bodies (issue #98)
    # ------------------------------------------------------------------








from .reference_types import _stable_sigmoid as _stable_sigmoid
