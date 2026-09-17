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

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .memory import WorkspacePlan
from .schedule_phase import PhasePlan, PhaseSchedule, worker_task_ids
from .task_ir import TaskFamily

__all__ = ["ExecutionTrace", "PhaseStat", "ReferenceExecutor", "ExecutorError"]


def _sigmoid(g):
    """Overflow-stable elementwise sigmoid in fp64: 1/(1+e^-g) for g>=0,
    e^g/(1+e^g) otherwise (avoids exp overflow warnings for gate <= -103,
    where the device fp32 sigmoid saturates to exactly 0)."""
    g = np.asarray(g, dtype=np.float64)
    out = np.empty_like(g)
    pos = g >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-g[pos]))
    eg = np.exp(g[~pos])
    out[~pos] = eg / (1.0 + eg)
    return out


class ExecutorError(Exception):
    """The executed schedule violated an obligation (§12)."""


@dataclass
class PhaseStat:
    index: int
    kind: str
    tasks: int
    per_worker: list[int]
    barrier_generation: int

    @property
    def idle_workers(self) -> int:
        return sum(1 for c in self.per_worker if c == 0)

    @property
    def max_tasks_per_worker(self) -> int:
        return max(self.per_worker)


@dataclass
class ExecutionTrace:
    kernel_launches: int = 0
    grid_barriers: int = 0
    phases: list[PhaseStat] = field(default_factory=list)
    task_executions: int = 0

    def summary(self) -> str:
        lines = [f"execution: {self.kernel_launches} kernel launch(s), {self.grid_barriers} grid barriers, {self.task_executions} task executions"]
        for st in self.phases:
            lines.append(f"  phase {st.index:2d} {st.kind:18s} tasks={st.tasks:3d} per-worker={st.per_worker} idle={st.idle_workers}")
        return "\n".join(lines)


class ReferenceExecutor:
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
        self._bodies: dict[str, Callable] = {
            "gemm": self._body_gemm,
            "layernorm": self._body_layernorm,
            "rms_norm": self._body_rms_norm,
            "rms_norm_gated": self._body_rms_norm_gated,
            "rope": self._body_rope,
            "elementwise": self._body_elementwise,
            "embedding": self._body_embedding,
            "cache_append": self._body_cache_append,
            "gdn_conv": self._body_gdn_conv,
            "gdn_delta": self._body_gdn_delta,
            "kda_delta": self._body_kda_delta,
            "attention_scores": self._body_attention_scores,
            "softmax": self._body_softmax,
            "attention_values": self._body_attention_values,
        }
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

    def _body_gemm(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0])
        w = self.tensor(fam.inputs[1])
        y = self.tensor(fam.outputs[0])
        bias = self.tensor(fam.inputs[2]) if fam.params["bias"] else None
        (m0, m1), (n0, n1) = self._gemm_box(fam, coords)
        acc = x[m0:m1, :].astype(np.float64) @ w[:, n0:n1].astype(np.float64)
        if bias is not None:
            acc = acc + bias[n0:n1]
        y[m0:m1, n0:n1] = acc.astype(y.dtype)

    @staticmethod
    def _gemm_box(fam: TaskFamily, coords):
        (m_extent, m_tile), (n_extent, n_tile) = fam.domain.dims
        m0 = coords[0] * m_tile
        n0 = coords[1] * n_tile
        return (m0, min(m0 + m_tile, m_extent)), (n0, min(n0 + n_tile, n_extent))

    def _body_layernorm(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        g = self.tensor(fam.inputs[1])
        b = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        eps = fam.params["eps"]
        r = coords[0]
        row = x[r]
        mean = row.mean()
        var = ((row - mean) ** 2).mean()
        y[r] = ((row - mean) / np.sqrt(var + eps) * g + b).astype(y.dtype)

    def _body_rms_norm(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        g = self.tensor(fam.inputs[1])
        y = self.tensor(fam.outputs[0])
        eps = fam.params["eps"]
        if len(x.shape) == 3:
            b, h = coords
            row = x[b, h]
            y[b, h] = (row * np.reciprocal(np.sqrt((row * row).mean() + eps)) * g).astype(y.dtype)
        else:
            r = coords[0]
            row = x[r]
            y[r] = (row * np.reciprocal(np.sqrt((row * row).mean() + eps)) * g).astype(y.dtype)

    def _body_rms_norm_gated(self, fam: TaskFamily, coords, scalars) -> None:
        """Sigmoid-gated RMSNorm (issue #100, GLM o_norm), one task per row
        or head row. Strict-fp32 semantics in the device contract (floe
        Glm53RMSNormGated); the reference body runs the same math in fp64
        for oracle stability:

            o = x * rsqrt(mean(x^2) + eps) * gamma * sigmoid(gate)
        """
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        gate = self.tensor(fam.inputs[1]).astype(np.float64)
        g = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        eps = fam.params["eps"]
        if len(x.shape) == 3:
            b, h = coords
            row, g_row = x[b, h], gate[b, h]
            y[b, h] = (
                row * np.reciprocal(np.sqrt((row * row).mean() + eps)) * g * _sigmoid(g_row)
            ).astype(y.dtype)
        else:
            r = coords[0]
            row, g_row = x[r], gate[r]
            y[r] = (
                row * np.reciprocal(np.sqrt((row * row).mean() + eps)) * g * _sigmoid(g_row)
            ).astype(y.dtype)

    def _body_rope(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        cos_t = self.tensor(fam.inputs[1])
        sin_t = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        p = scalars["p"]
        row = x[b, h]
        half = row.shape[-1] // 2
        rotated = np.concatenate((-row[half:], row[:half]), axis=-1)
        out = row * cos_t[p].astype(np.float64) + rotated * sin_t[p].astype(np.float64)
        y[b, h] = out.astype(y.dtype)

    def _body_elementwise(self, fam: TaskFamily, coords, scalars) -> None:
        op = fam.params["op"]
        tile = fam.domain.dims[0][1]
        lo = coords[0] * tile
        hi = min(lo + tile, self.tensor(fam.inputs[0]).size)
        if op == "gelu":
            x = self.tensor(fam.inputs[0]).astype(np.float64)
            y = self.tensor(fam.outputs[0])
            flat_x = x.reshape(-1)[lo:hi]
            t = np.sqrt(2.0 / np.pi) * (flat_x + 0.044715 * flat_x**3)
            y.reshape(-1)[lo:hi] = (0.5 * flat_x * (1.0 + np.tanh(t))).astype(y.dtype)
        elif op == "swiglu":
            gate = self.tensor(fam.inputs[0]).astype(np.float64)
            up = self.tensor(fam.inputs[1]).astype(np.float64)
            out = self.tensor(fam.outputs[0])
            g = gate.reshape(-1)[lo:hi]
            u = up.reshape(-1)[lo:hi]
            out.reshape(-1)[lo:hi] = (g / (1.0 + np.exp(-g)) * u).astype(out.dtype)
        else:  # add
            a = self.tensor(fam.inputs[0])
            b = self.tensor(fam.inputs[1])
            out = self.tensor(fam.outputs[0])
            out.reshape(-1)[lo:hi] = (a.reshape(-1)[lo:hi] + b.reshape(-1)[lo:hi]).astype(out.dtype)

    def _body_embedding(self, fam: TaskFamily, coords, scalars) -> None:
        ids = self.tensor(fam.inputs[0])
        token = self.tensor(fam.inputs[1])
        pos = self.tensor(fam.inputs[2]) if len(fam.inputs) > 2 else None
        y = self.tensor(fam.outputs[0])
        b = coords[0]
        width = y.shape[-1]
        tile_c = fam.domain.dims[1][1]
        c0 = 0 if fam.domain.task_grid[1] == 1 else coords[1] * tile_c
        c1 = width if fam.domain.task_grid[1] == 1 else min(c0 + tile_c, width)
        row = token[int(ids[b]), c0:c1].astype(np.float64)
        if pos is not None:
            row = row + pos[scalars["p"], c0:c1].astype(np.float64)
        y[b, c0:c1] = row.astype(y.dtype)

    def _body_cache_append(self, fam: TaskFamily, coords, scalars) -> None:
        k_cache = self.tensor(fam.inputs[0])
        v_cache = self.tensor(fam.inputs[1])
        k_new = self.tensor(fam.inputs[2])
        v_new = self.tensor(fam.inputs[3])
        b, h = coords
        p = scalars["p"]
        k_cache[b, h, p, :] = k_new[b, h, :]
        v_cache[b, h, p, :] = v_new[b, h, :]

    def _body_gdn_conv(self, fam: TaskFamily, coords, scalars) -> None:
        """GDN short-conv decode step over one (batch, channel-tile) task:
        fp32-accumulated depthwise FIR + silu, then the time-major state
        shift (drop oldest tap, append the new row) — in place, since the
        pool is external persistent storage.
        """
        state = self.tensor(fam.inputs[0])
        x = self.tensor(fam.inputs[1]).astype(np.float64)
        w = self.tensor(fam.inputs[2]).astype(np.float64)
        out = self.tensor(fam.outputs[0])
        b, c = coords
        tile = fam.params["tile"]
        K = fam.params["conv_kernel"]
        C = state.shape[-1]
        c0, c1 = c * tile, min((c + 1) * tile, C)
        st = state[b, :, c0:c1].astype(np.float64)  # [K-1, T] time-major
        xs = x[b, c0:c1]  # [T]
        ws = w[c0:c1, :]  # [T, K]
        acc = np.einsum("jt,tj->t", st, ws[:, :-1]) + xs * ws[:, K - 1]
        out[b, c0:c1] = (acc / (1.0 + np.exp(-acc))).astype(out.dtype)
        # state shift: state[j] <- state[j+1]; state[K-2] <- x
        state[b, :, c0:c1] = np.concatenate([st[1:], xs[None, :]], axis=0).astype(state.dtype)
    def _body_gdn_delta(self, fam: TaskFamily, coords, scalars) -> None:
        """Gated delta rule decode step over one (batch, value head) task.

        fp64 oracle arithmetic mirroring floe ``qwen35_gdn.py`` seq==1:
        guarded softplus decay, per-key-head L2 q/k (group-expanded),
        delta-rule outer-product state update, per-head RMSNorm + z-gate.
        The head's ``[HV, HK]`` fp32 state slice is updated in place (the
        pool is external persistent storage).
        """
        state = self.tensor(fam.inputs[0])  # [B, NV, HV, HK]
        q = self.tensor(fam.inputs[1])  # [B, NK, HK]
        k = self.tensor(fam.inputs[2])
        v = self.tensor(fam.inputs[3])  # [B, NV, HV]
        z = self.tensor(fam.inputs[4])
        a = self.tensor(fam.inputs[5])  # [B, NV]
        b = self.tensor(fam.inputs[6])
        a_log = self.tensor(fam.inputs[7])  # [NV]
        dt_bias = self.tensor(fam.inputs[8])  # [NV]
        norm_w = self.tensor(fam.inputs[9])  # [HV]
        out = self.tensor(fam.outputs[0])
        bb, h = coords
        NV, HV, HK = state.shape[1], state.shape[2], state.shape[3]
        NK = q.shape[1]
        kh = h // (NV // NK)
        scale, eps = fam.params["scale"], fam.params["eps"]
        s = state[bb, h].astype(np.float64)  # [HV, HK]
        qf = q[bb, kh].astype(np.float64)
        kf = k[bb, kh].astype(np.float64)
        vf = v[bb, h].astype(np.float64)
        zf = z[bb, h].astype(np.float64)
        # per-head gating scalars (floe: log(1+exp) with the x>20 guard)
        x_dt = float(a[bb, h]) + float(dt_bias[h])
        softplus_x = np.log(1.0 + np.exp(x_dt)) if x_dt <= 20.0 else x_dt
        decay = float(np.exp(-float(np.exp(a_log[h])) * softplus_x))
        beta = 1.0 / (1.0 + np.exp(-float(b[bb, h])))
        # per-key-head normalization (group-expanded)
        qn = qf / np.sqrt(np.dot(qf, qf) + 1e-6) * scale
        kn = kf / np.sqrt(np.dot(kf, kf) + 1e-6)
        # delta-rule state update
        s = s * decay
        sk = s @ kn
        s = s + (beta * (vf - sk))[:, None] * kn[None, :]
        state[bb, h] = s.astype(state.dtype)
        # readout + per-head RMSNorm over HV + z gate
        o = s @ qn
        var = np.mean(o * o)
        on = o / np.sqrt(var + eps) * norm_w.astype(np.float64)
        og = on * (zf / (1.0 + np.exp(-zf)))
        out[bb, h] = og.astype(out.dtype)

    def _body_kda_delta(self, fam: TaskFamily, coords, scalars) -> None:
        """KDA gated delta rule decode step over one (batch, head) task.

        fp64 oracle arithmetic mirroring floe ``Glm53LinearAttention``
        seq==1: the per-(head, k-dim) forget gate (lower_bound·sigmoid of
        exp(A_log)·(f + dt_bias); lower_bound None -> guarded softplus),
        element-wise exp(g) row decay over the [K, V] state, L2 q/k with
        the 1/sqrt(D) scale on q, delta-rule outer-product update, plain
        state readout (the gated norm lives in the separate rms_norm_gated
        op). The head's ``[K, V]`` fp32 state slice is updated in place
        (the pool is external persistent storage).
        """
        state = self.tensor(fam.inputs[0])  # [B, H, K, V]
        q = self.tensor(fam.inputs[1])  # [B, H, K]
        k = self.tensor(fam.inputs[2])  # [B, H, K]
        v = self.tensor(fam.inputs[3])  # [B, H, V]
        f = self.tensor(fam.inputs[4])  # [B, H, K] f_b projection row
        b = self.tensor(fam.inputs[5])  # [B, H] beta logits
        dt_bias = self.tensor(fam.inputs[6])  # [H, K]
        a_log = self.tensor(fam.inputs[7])  # [H]
        out = self.tensor(fam.outputs[0])
        bb, h = coords
        K, V = state.shape[2], state.shape[3]
        scale = fam.params["scale"]
        lower_bound = fam.params.get("lower_bound")
        s = state[bb, h].astype(np.float64)  # [K, V]
        qf = q[bb, h].astype(np.float64)
        kf = k[bb, h].astype(np.float64)
        vf = v[bb, h].astype(np.float64)
        ff = f[bb, h].astype(np.float64)
        dt = dt_bias[h].astype(np.float64)
        A = float(np.exp(a_log[h]))
        x_dt = ff + dt
        if lower_bound is not None:
            g = lower_bound / (1.0 + np.exp(-A * x_dt))  # log-space [K]
        else:
            sp = np.where(x_dt <= 20.0, np.log(1.0 + np.exp(x_dt)), x_dt)
            g = -A * sp
        beta = 1.0 / (1.0 + np.exp(-float(b[bb, h])))
        # L2 conditioning (floe _l2norm, eps 1e-6 inside the sqrt)
        qn = qf / np.sqrt(np.dot(qf, qf) + 1e-6) * scale
        kn = kf / np.sqrt(np.dot(kf, kf) + 1e-6)
        # element-wise decay: exp(g) broadcasts over the value axis
        s = s * np.exp(g)[:, None]
        kv_mem = (s * kn[:, None]).sum(axis=0)  # [V] = sum_k s[k,v]*kn[k]
        s = s + kn[:, None] * ((beta * (vf - kv_mem))[None, :])
        state[bb, h] = s.astype(state.dtype)
        # plain readout; gated norm (rms_norm_gated) is a separate op
        o = (s * qn[:, None]).sum(axis=0)  # [V] = sum_k s[k,v]*qn[k]
        out[bb, h] = o.astype(out.dtype)

    def _body_attention_scores(self, fam: TaskFamily, coords, scalars) -> None:
        q = self.tensor(fam.inputs[0]).astype(np.float64)
        k_cache = self.tensor(fam.inputs[1]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        kvh = h // fam.params["group"] if fam.params.get("group", 1) > 1 else h
        p = scalars["p"]
        scale = fam.params["scale"]
        # Masked load: only rows [0, p] are read (§5.3); the tail is never touched.
        y[b, h, : p + 1] = (k_cache[b, kvh, : p + 1, :] @ q[b, h, :] * scale).astype(y.dtype)

    def _body_softmax(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        p = scalars["p"]
        row = x[b, h, : p + 1]
        row = row - row.max()
        e = np.exp(row)
        y[b, h, : p + 1] = (e / e.sum()).astype(y.dtype)
        y[b, h, p + 1 :] = 0.0  # invalid tail written to exact zero (§4.3 contract)

    def _body_attention_values(self, fam: TaskFamily, coords, scalars) -> None:
        probs = self.tensor(fam.inputs[0]).astype(np.float64)
        v_cache = self.tensor(fam.inputs[1]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        kvh = h // fam.params["group"] if fam.params.get("group", 1) > 1 else h
        p = scalars["p"]
        # Masked: V rows beyond p are never loaded (NaN tails must not leak).
        y[b, h, :] = (probs[b, h, : p + 1] @ v_cache[b, kvh, : p + 1, :]).astype(y.dtype)
