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

__all__ = ["ExecutionTrace", "PhaseStat", "ReferenceExecutor", "ExecutorError", "decode_e4m3"]


class ExecutorError(Exception):
    """The executed schedule violated an obligation (§12)."""


def decode_e4m3(bytes_u8):
    """Decode fp8 e4m3fn bytes (uint8 array) to float64, vectorized.

    e4m3fn: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits; no
    infinities — exponent 0xF with mantissa 0x7 encodes NaN, and the
    maximum finite magnitude is 448. Subnormals (exponent 0) are
    mant/8 * 2^-6. This is the numpy-side counterpart of the checkpoint's
    ``torch.float8_e4m3fn`` weights and of ``_fp8_e4m3fn_encode``.
    """
    b = np.asarray(bytes_u8, dtype=np.uint8).astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    mant = b & 0x7
    is_nan = (exp == 0xF) & (mant == 0x7)
    normal = np.exp2((exp - 7).astype(np.float64)) * (1.0 + mant / 8.0)
    sub = np.exp2(-6.0) * (mant / 8.0)
    val = sign * np.where(exp == 0, sub, normal)
    return np.where(is_nan, np.nan, val)


_decode_e4m3 = decode_e4m3


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
            "rope": self._body_rope,
            "elementwise": self._body_elementwise,
            "embedding": self._body_embedding,
            "cache_append": self._body_cache_append,
            "cache_append_paged": self._body_cache_append_paged,
            "attention_scores": self._body_attention_scores,
            "attention_scores_paged": self._body_attention_scores_paged,
            "softmax": self._body_softmax,
            "attention_values": self._body_attention_values,
            "gemv_fp8": self._body_gemv_fp8,
            "attention_values_paged": self._body_attention_values_paged,
            "moe_route": self._body_moe_route,
            "moe_expert": self._body_moe_expert,
            "moe_combine": self._body_moe_combine,
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

    def _body_gemv_fp8(self, fam: TaskFamily, coords, scalars) -> None:
        """fp8-blockwise GEMV reference (issue #91): dequant-then-matmul in fp64.

        Mirrors the device contract of ``_t_gemv_fp8`` / ``_h_gemv_fp8``:
        weights are e4m3 bytes [N, K] row-major (decoded tile-exactly for the
        task's 16-column N tile), one fp32 scale per 128x128 block, fp64
        accumulation standing in for the device's fp32 (oracle stability).
        """
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        w = self.tensor(fam.inputs[1])  # e4m3 bytes materialize as uint8 storage
        scale = self.tensor(fam.inputs[2]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        qb = int(fam.params.get("quant_block", 128))
        (m0, m1), (n0, n1) = self._gemm_box(fam, coords)
        wt = _decode_e4m3(w[n0:n1, :])
        # Broadcast each 128-deep k-block's scale over its columns, exactly as
        # the device loads one scale per k-chunk of the tile.
        k = wt.shape[1]
        s = np.repeat(scale[n0 // qb, :], qb)[:k]
        acc = (x[m0:m1, :] @ (wt * s[None, :]).T)
        y[m0:m1, n0:n1] = acc.astype(y.dtype)

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

    def _body_rope(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        cos_t = self.tensor(fam.inputs[1])
        sin_t = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        # Row b rotates at its OWN runtime position (issue #93 ragged form).
        p = self._row_pos_value(fam, b, scalars)
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
            # Row b adds its own position row (issue #93 ragged form).
            row = row + pos[self._row_pos_value(fam, b, scalars), c0:c1].astype(np.float64)
        y[b, c0:c1] = row.astype(y.dtype)

    def _body_cache_append(self, fam: TaskFamily, coords, scalars) -> None:
        k_cache = self.tensor(fam.inputs[0])
        v_cache = self.tensor(fam.inputs[1])
        k_new = self.tensor(fam.inputs[2])
        v_new = self.tensor(fam.inputs[3])
        b, h = coords
        # Row b appends at its own position (issue #93 ragged form).
        p = self._row_pos_value(fam, b, scalars)
        k_cache[b, h, p, :] = k_new[b, h, :]
        v_cache[b, h, p, :] = v_new[b, h, :]

    def _body_cache_append_paged(self, fam: TaskFamily, coords, scalars) -> None:
        k_pool = self.tensor(fam.inputs[0])
        v_pool = self.tensor(fam.inputs[1])
        table = self.tensor(fam.inputs[2])
        k_new = self.tensor(fam.inputs[3])
        v_new = self.tensor(fam.inputs[4])
        b, h = coords
        p = int(scalars["p"])
        slot = int(table[b, p])  # write lands at slot_table[b, p_row] (#94)
        k_pool[slot, h, :] = k_new[b, h, :]
        v_pool[slot, h, :] = v_new[b, h, :]

    def _body_attention_scores_paged(self, fam: TaskFamily, coords, scalars) -> None:
        q = self.tensor(fam.inputs[0]).astype(np.float64)
        k_pool = self.tensor(fam.inputs[1]).astype(np.float64)
        table = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        kvh = h // fam.params["group"] if fam.params.get("group", 1) > 1 else h
        p = int(scalars["p"])
        scale = fam.params["scale"]
        # Gathered masked load: only positions [0, p] map through the table;
        # slot 0 is the reserved null/sink page (never written by a live row).
        slots = table[b, : p + 1].astype(np.int64)
        y[b, h, : p + 1] = (k_pool[slots, kvh, :] @ q[b, h, :] * scale).astype(y.dtype)

    def _body_attention_values_paged(self, fam: TaskFamily, coords, scalars) -> None:
        probs = self.tensor(fam.inputs[0]).astype(np.float64)
        v_pool = self.tensor(fam.inputs[1]).astype(np.float64)
        table = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        kvh = h // fam.params["group"] if fam.params.get("group", 1) > 1 else h
        p = int(scalars["p"])
        # Gathered masked: V rows beyond p are never gathered (NaN slots must not leak).
        slots = table[b, : p + 1].astype(np.int64)
        y[b, h, :] = (probs[b, h, : p + 1] @ v_pool[slots, kvh, :]).astype(y.dtype)

    def _body_attention_scores(self, fam: TaskFamily, coords, scalars) -> None:
        q = self.tensor(fam.inputs[0]).astype(np.float64)
        k_cache = self.tensor(fam.inputs[1]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        kvh = h // fam.params["group"] if fam.params.get("group", 1) > 1 else h
        vlen = self._valid_len(fam, b, scalars)
        scale = fam.params["scale"]
        # Masked load: only rows [0, pos[b]] are read (§5.3, per-row); the
        # NaN tail beyond each row's valid length is never touched.
        y[b, h, :vlen] = (k_cache[b, kvh, :vlen, :] @ q[b, h, :] * scale).astype(y.dtype)

    # ------------------------------------------------------------------
    # MoE decode reference bodies (issue #98)
    # ------------------------------------------------------------------

    def _moe_scores(self, logits: np.ndarray, fam: TaskFamily) -> np.ndarray:
        """Router scores, fp64 standing in for the fp32 device math."""
        if fam.params["score_fn"] == "sigmoid_noaux_tc":
            # Stable sigmoid: no overflow for large |logit|.
            return 1.0 / (1.0 + np.exp(-np.logaddexp(0.0, -logits)))
        # sqrtsoftplus (DeepSeek-V4): sqrt(softplus(l)), stable softplus.
        return np.sqrt(np.logaddexp(0.0, logits))

    @staticmethod
    def _moe_topk_stable(choice: np.ndarray, k: int) -> np.ndarray:
        """Top-k with the documented determinism contract: stable descending
        order, ties to the lower expert index (sorted=False set semantics)."""
        order = np.argsort(-choice, kind="stable")
        return order[:k]

    def _body_moe_route(self, fam: TaskFamily, coords, scalars) -> None:
        """Router decode step (issue #98), fp64 oracle of the fp32 device math.

        Learned noaux_tc (GLM-5.3): biased choice scores restricted to the
        top-2-sum groups, top-k over masked scores, weights from the UNBIASED
        scores. Learned sqrtsoftplus (DeepSeek-V4): global top-k. Hash: frozen
        ``tid2eid[token_id]`` gather. Renorm ``w/(Σw+1e-20)`` (unconditional
        for sqrtsoftplus/hash; gated on norm_topk_prob for noaux_tc), then ×
        routed_scaling_factor.
        """
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        router_w = self.tensor(fam.inputs[1]).astype(np.float64)
        ids = self.tensor(fam.inputs[2])
        weights = self.tensor(fam.inputs[3])
        r = coords[0]
        k = int(fam.params["top_k"])
        rsf = float(fam.params["routed_scaling_factor"])
        logits = router_w @ x[r]
        scores = self._moe_scores(logits, fam)
        if fam.params["mode"] == "hash":
            tid2eid = self.tensor(fam.inputs[4])
            token_ids = self.tensor(fam.inputs[5])
            sel = tid2eid[int(token_ids[r]), :].astype(np.int64)
            w = scores[sel]
            w = w / (w.sum() + 1e-20)
        else:
            choice = scores.copy()
            if fam.params.get("routed_bias"):
                choice = choice + self.tensor(fam.inputs[4]).astype(np.float64)
            if fam.params["score_fn"] == "sigmoid_noaux_tc" and int(fam.params["n_group"]) > 1:
                n_group = int(fam.params["n_group"])
                topk_group = int(fam.params["topk_group"])
                e = choice.shape[0]
                groups = choice.reshape(n_group, e // n_group)
                # Top-2 sum per group (stable descending; ties to lower index).
                top2 = np.sort(groups, axis=1, kind="stable")[:, ::-1][:, :2]
                group_scores = top2.sum(axis=1)
                g_order = self._moe_topk_stable(group_scores, topk_group)
                mask = np.full(e, -np.inf)
                for g in g_order:
                    mask[g * (e // n_group) : (g + 1) * (e // n_group)] = 0.0
                choice = choice + mask
            sel = self._moe_topk_stable(choice, k)
            w = scores[sel]  # unbiased scores (floe semantics)
            if fam.params.get("norm_topk_prob", False):
                w = w / (w.sum() + 1e-20)
        ids[r, :] = sel.astype(ids.dtype)
        weights[r, :] = (w * rsf).astype(weights.dtype)

    def _body_moe_expert(self, fam: TaskFamily, coords, scalars) -> None:
        """One (row, slot) expert FFN task (issue #98): runtime-indirected
        weight base ``e = ids[b, slot]`` (#94 pattern), swiglu_limit clamp
        folded into the activation, fp64 oracle of the fp32 device math."""
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        gate_up = self.tensor(fam.inputs[1]).astype(np.float64)
        down = self.tensor(fam.inputs[2]).astype(np.float64)
        ids = self.tensor(fam.inputs[3])
        partials = self.tensor(fam.outputs[0])
        r, s = coords
        e = int(ids[r, s])
        gu = gate_up[e] @ x[r]
        inter = gu.shape[0] // 2
        g, u = gu[:inter], gu[inter:]
        limit = fam.params.get("swiglu_limit")
        if limit is not None:
            g = np.minimum(g, float(limit))
            u = np.clip(u, -float(limit), float(limit))
        act = g / (1.0 + np.exp(-np.logaddexp(0.0, -g))) * u  # silu(g) · u
        partials[r, s, :] = (down[e] @ act).astype(partials.dtype)

    def _body_moe_combine(self, fam: TaskFamily, coords, scalars) -> None:
        """Weighted scatter-add per row (issue #98): Σ_k w_k·h_k in slot order
        (+ the dense shared-expert path when recorded), fp64 accumulate."""
        partials = self.tensor(fam.inputs[0]).astype(np.float64)
        weights = self.tensor(fam.inputs[1]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        r = coords[0]
        k = partials.shape[1]
        acc = np.zeros(partials.shape[2], dtype=np.float64)
        for s in range(k):  # documented contract: accumulate in slot order
            acc += weights[r, s] * partials[r, s]
        if len(fam.inputs) > 2:
            acc = acc + self.tensor(fam.inputs[2]).astype(np.float64)[r]
        y[r, :] = acc.astype(y.dtype)

    def _body_softmax(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        vlen = self._valid_len(fam, b, scalars)
        row = x[b, h, :vlen]
        row = row - row.max()
        e = np.exp(row)
        y[b, h, :vlen] = (e / e.sum()).astype(y.dtype)
        y[b, h, vlen:] = 0.0  # invalid tail written to exact zero (§4.3 contract)

    def _body_attention_values(self, fam: TaskFamily, coords, scalars) -> None:
        probs = self.tensor(fam.inputs[0]).astype(np.float64)
        v_cache = self.tensor(fam.inputs[1]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        kvh = h // fam.params["group"] if fam.params.get("group", 1) > 1 else h
        vlen = self._valid_len(fam, b, scalars)
        # Masked: V rows beyond pos[b] are never loaded (per-row NaN-tail
        # contract, issue #93).
        y[b, h, :] = (probs[b, h, :vlen] @ v_cache[b, kvh, :vlen, :]).astype(y.dtype)
