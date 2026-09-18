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


def _stable_sigmoid(g: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid, fp64 mirror of the device epilogue
    ``1 / (1 + exp(-g))`` (issue #92). Overflow-safe for large |g|: the
    device computes exp(-g) directly, so huge negative g overflows to inf
    and the ratio still rounds to 0; huge positive g underflows to 0 and
    the ratio rounds to 1 — same saturating semantics, no NaN."""
    g = np.asarray(g, dtype=np.float64)
    out = np.empty_like(g)
    pos = g >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-g[pos]))
    eg = np.exp(g[~pos])
    out[~pos] = eg / (1.0 + eg)
    return out


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
            "indexer_scores": self._body_indexer_scores,
            "index_topk": self._body_index_topk,
            "layernorm": self._body_layernorm,
            "rms_norm": self._body_rms_norm,
            "rms_norm_gated": self._body_rms_norm_gated,
            "rope": self._body_rope,
            "elementwise": self._body_elementwise,
            "embedding": self._body_embedding,
            "cache_append": self._body_cache_append,
            "gdn_conv": self._body_gdn_conv,
            "compressor_append": self._body_compressor_append,
            "cache_append_paged": self._body_cache_append_paged,
            "mhc_pre": self._body_mhc_pre,
            "mhc_post": self._body_mhc_post,
            "gdn_delta": self._body_gdn_delta,
            "kda_delta": self._body_kda_delta,
            "attention_scores": self._body_attention_scores,
            "attention_scores_paged": self._body_attention_scores_paged,
            "softmax": self._body_softmax,
            "attention_values": self._body_attention_values,
            "attention_values_paged": self._body_attention_values_paged,
            "gemv_fp8": self._body_gemv_fp8,
            "moe_route": self._body_moe_route,
            "moe_expert": self._body_moe_expert,
            "moe_combine": self._body_moe_combine,
            "mla_scores": self._body_mla_scores,
            "mla_values": self._body_mla_values,
            "conjugate_rope": self._body_conjugate_rope,
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
        gh = fam.params.get("grouped_heads")
        if gh:
            # Block-diagonal per-head projection (issue #95 GroupedLinear):
            # output n-range may span several heads; each head contributes
            # only its own diagonal block (off-block entries NEVER read —
            # the storage may hold NaN canaries there).
            k_g = x.shape[1] // gh
            n_g = w.shape[1] // gh
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float64)
            for h in range(n0 // n_g, min(gh, (n1 + n_g - 1) // n_g)):
                h_n0, h_n1 = max(n0, h * n_g), min(n1, (h + 1) * n_g)
                x_blk = x[m0:m1, h * k_g:(h + 1) * k_g].astype(np.float64)
                # w is stored [Cin, Cout]: block rows are the head's K slice,
                # block columns the head's N slice.
                w_blk = w[h * k_g:(h + 1) * k_g,
                          h * n_g + (h_n0 - h * n_g):h * n_g + (h_n1 - h * n_g)].astype(np.float64)
                acc[:, h_n0 - n0:h_n1 - n0] = x_blk @ w_blk
            if bias is not None:
                acc = acc + bias[n0:n1]
            y[m0:m1, n0:n1] = acc.astype(y.dtype)
            return
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

    def _body_conjugate_rope(self, fam: TaskFamily, coords, scalars) -> None:
        """Conjugate (output-side) rope (issue #95): rotation by the NEGATIVE
        angle — sin negated. Exact inverse of the q/k rotation."""
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        cos_t = self.tensor(fam.inputs[1])
        sin_t = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        p = self._row_pos_value(fam, b, scalars)
        rot = fam.params["rotary_dim"]
        half = rot // 2
        c = cos_t[p][:half].astype(np.float64)
        s = sin_t[p][:half].astype(np.float64)  # applied NEGATED below
        row = x[b, h]
        if fam.params.get("convention", "interleaved") == "interleaved":
            x_even, x_odd = row[0:rot:2], row[1:rot:2]
            out = row.copy()
            out[0:rot:2] = x_even * c + x_odd * s
            out[1:rot:2] = x_odd * c - x_even * s
        else:  # rotate_half conjugate
            hh = row.shape[-1] // 2
            rotated = np.concatenate((-row[hh:], row[:hh]), axis=-1)
            out = row * cos_t[p].astype(np.float64) - rotated * sin_t[p].astype(np.float64)
        y[b, h] = out.astype(y.dtype)

    def _body_mla_scores(self, fam: TaskFamily, coords, scalars) -> None:
        """MLA fused scores + softmax + sink (issue #95), fp64 oracle.

        Candidate layout per (b, h): [W window | K compressed | 1 sink].
        Window slot i holds logical cache position t = p - W + 1 + i
        (sliding-window bound |q - t| < W, t <= q); slots with t < 0 or
        t > p are invalid. Compressed slot j holds comp_idx[b, j] (valid
        iff >= 0). fp64 two-pass softmax over valid candidates ∪ sink;
        invalid slots exact 0.0 (§4.3).
        """
        q = self.tensor(fam.inputs[0]).astype(np.float64)
        latent = self.tensor(fam.inputs[1]).astype(np.float64)
        window_table = self.tensor(fam.inputs[2])
        comp_pool = self.tensor(fam.inputs[3]).astype(np.float64)
        comp_idx = self.tensor(fam.inputs[4])
        sink = self.tensor(fam.inputs[5]).astype(np.float64)
        bias = None
        if len(fam.inputs) > 6:
            bias = self.tensor(fam.inputs[6]).astype(np.float64)
        y = self.tensor(fam.outputs[0])
        b, h = coords
        p = self._row_pos_value(fam, b, scalars)
        W = fam.params["window"]
        K = fam.params["comp_slots"]
        scale = fam.params["scale"]
        qb = q[b, h]
        logits = np.full(W + K + 1, -np.inf, dtype=np.float64)
        # window candidates: logical t in [max(0, p-W+1), p]
        t_lo = max(0, p - W + 1)
        for i in range(W):
            t = p - W + 1 + i
            if t_lo <= t <= p:
                row = latent[b, int(window_table[b, t])]
                lg = float(row @ qb) * scale
                logits[i] = lg
        # compressed candidates via the #97 indirection table
        for j in range(K):
            e = int(comp_idx[b, j])
            if e >= 0:
                lg = float(comp_pool[b, e] @ qb) * scale
                if bias is not None:
                    lg += float(bias[b, j])
                logits[W + j] = lg
        # sink: per-head learnable logit, always valid, LAST slot
        logits[W + K] = float(sink[b, h]) if sink.ndim == 2 else float(sink[h])
        m = logits.max()
        e = np.exp(logits - m)
        e[~np.isfinite(logits)] = 0.0  # invalid slots (logit -inf) exact zero
        y[b, h, :] = (e / e.sum()).astype(y.dtype)

    def _body_mla_values(self, fam: TaskFamily, coords, scalars) -> None:
        """MLA context gather (issue #95): window + compressed pools, sink
        column contributes no value. fp64 accumulation oracle."""
        probs = self.tensor(fam.inputs[0]).astype(np.float64)
        latent = self.tensor(fam.inputs[1]).astype(np.float64)
        window_table = self.tensor(fam.inputs[2])
        comp_pool = self.tensor(fam.inputs[3]).astype(np.float64)
        comp_idx = self.tensor(fam.inputs[4])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        p = self._row_pos_value(fam, b, scalars)
        W = fam.params["window"]
        K = fam.params["comp_slots"]
        acc = np.zeros(y.shape[2], dtype=np.float64)
        t_lo = max(0, p - W + 1)
        for i in range(W):
            t = p - W + 1 + i
            if t_lo <= t <= p:
                acc += probs[b, h, i] * latent[b, int(window_table[b, t])]
        for j in range(K):
            e = int(comp_idx[b, j])
            if e >= 0:
                acc += probs[b, h, W + j] * comp_pool[b, e]
        y[b, h, :] = acc.astype(y.dtype)

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
    def _body_indexer_scores(self, fam: TaskFamily, coords, scalars) -> None:
        """Lightning-indexer scoring reference (issue #97).

        Mirrors the device contract of ``_t_indexer_scores``: per (batch,
        entry tile), relu(<q_h, c_j>) * head_dim**-0.5 per indexer head,
        then the f32/fp64 weighted head mix. fp64 accumulation stands in
        for the device's f32 (oracle stability).
        """
        q = self.tensor(fam.inputs[0]).astype(np.float64)
        c = self.tensor(fam.inputs[1]).astype(np.float64)
        w = self.tensor(fam.inputs[2]).astype(np.float64)
        s = self.tensor(fam.outputs[0])
        scale = fam.params["scale"]
        (m_extent, m_tile) = fam.domain.dims[1]
        b, t = coords
        m0, m1 = t * m_tile, min(t * m_tile + m_tile, m_extent)
        # [H, tile]: relu of the per-head dots, scaled.
        scores = np.maximum(q[b] @ c[b, m0:m1].T, 0.0) * scale
        s[b, m0:m1] = (scores * w[b][:, None]).sum(axis=0).astype(s.dtype)

    def _body_index_topk(self, fam: TaskFamily, coords, scalars) -> None:
        """Fixed-count top-k selection reference (issue #97).

        Mirrors the device contract of ``_t_index_topk`` exactly:
        rank(j) = #{valid i : s_i > s_j} + #{valid i < j : s_i == s_j} —
        descending score with deterministic lowest-index tie-break; NaN
        scores inside the valid prefix are excluded; candidates at or
        beyond the row's valid count are never observed; slots beyond a
        row's valid count are idx=-1 / bias=0.0; bias = s_j / ||s_valid||_2.
        """
        s = self.tensor(fam.inputs[0]).astype(np.float64)
        valid = self.tensor(fam.inputs[1])
        idx = self.tensor(fam.outputs[0])
        bias = self.tensor(fam.outputs[1])
        k = fam.params["k"]
        (b,) = coords
        row = s[b]
        m = row.shape[0]
        vc = int(valid[b])
        vc = max(0, min(vc, m))
        valid_mask = np.zeros(m, dtype=bool)
        valid_mask[:vc] = True
        finite = np.isfinite(row)
        cand = valid_mask & finite  # NaN canaries inside the prefix lose
        # rank by comparison counting (the template's exact tie-break).
        gt = (row[None, :] > row[:, None]) & cand[None, :]
        eq_lower = (row[None, :] == row[:, None]) & cand[None, :] & (np.arange(m)[None, :] < np.arange(m)[:, None])
        rank = gt.sum(axis=1) + eq_lower.sum(axis=1)
        sel = cand & (rank < k)
        idx_row = np.full(k, -1, dtype=np.int32)
        bias_row = np.zeros(k, dtype=bias.dtype)
        sel_idx = np.nonzero(sel)[0]
        idx_row[rank[sel_idx]] = sel_idx.astype(np.int32)
        if vc > 0:
            valid_finite = row[:vc][finite[:vc]]
            norm = np.sqrt((valid_finite**2).sum()) if valid_finite.size else 0.0
        else:
            norm = 0.0
        if norm > 0.0:
            bias_row[rank[sel_idx]] = (row[sel_idx] / norm).astype(bias.dtype)
        idx[b, :] = idx_row
        bias[b, :] = bias_row

    def _body_indexer_scores(self, fam: TaskFamily, coords, scalars) -> None:
        """Lightning-indexer scoring reference (issue #97).

        Mirrors the device contract of ``_t_indexer_scores``: per (batch,
        entry tile), relu(<q_h, c_j>) * head_dim**-0.5 per indexer head,
        then the f32/fp64 weighted head mix. fp64 accumulation stands in
        for the device's f32 (oracle stability).
        """
        q = self.tensor(fam.inputs[0]).astype(np.float64)
        c = self.tensor(fam.inputs[1]).astype(np.float64)
        w = self.tensor(fam.inputs[2]).astype(np.float64)
        s = self.tensor(fam.outputs[0])
        scale = fam.params["scale"]
        (m_extent, m_tile) = fam.domain.dims[1]
        b, t = coords
        m0, m1 = t * m_tile, min(t * m_tile + m_tile, m_extent)
        # [H, tile]: relu of the per-head dots, scaled.
        scores = np.maximum(q[b] @ c[b, m0:m1].T, 0.0) * scale
        s[b, m0:m1] = (scores * w[b][:, None]).sum(axis=0).astype(s.dtype)

    def _body_index_topk(self, fam: TaskFamily, coords, scalars) -> None:
        """Fixed-count top-k selection reference (issue #97).

        Mirrors the device contract of ``_t_index_topk`` exactly:
        rank(j) = #{valid i : s_i > s_j} + #{valid i < j : s_i == s_j} —
        descending score with deterministic lowest-index tie-break; NaN
        scores inside the valid prefix are excluded; candidates at or
        beyond the row's valid count are never observed; slots beyond a
        row's valid count are idx=-1 / bias=0.0; bias = s_j / ||s_valid||_2.
        """
        s = self.tensor(fam.inputs[0]).astype(np.float64)
        valid = self.tensor(fam.inputs[1])
        idx = self.tensor(fam.outputs[0])
        bias = self.tensor(fam.outputs[1])
        k = fam.params["k"]
        (b,) = coords
        row = s[b]
        m = row.shape[0]
        vc = int(valid[b])
        vc = max(0, min(vc, m))
        valid_mask = np.zeros(m, dtype=bool)
        valid_mask[:vc] = True
        finite = np.isfinite(row)
        cand = valid_mask & finite  # NaN canaries inside the prefix lose
        # rank by comparison counting (the template's exact tie-break).
        gt = (row[None, :] > row[:, None]) & cand[None, :]
        eq_lower = (row[None, :] == row[:, None]) & cand[None, :] & (np.arange(m)[None, :] < np.arange(m)[:, None])
        rank = gt.sum(axis=1) + eq_lower.sum(axis=1)
        sel = cand & (rank < k)
        idx_row = np.full(k, -1, dtype=np.int32)
        bias_row = np.zeros(k, dtype=bias.dtype)
        sel_idx = np.nonzero(sel)[0]
        idx_row[rank[sel_idx]] = sel_idx.astype(np.int32)
        if vc > 0:
            valid_finite = row[:vc][finite[:vc]]
            norm = np.sqrt((valid_finite**2).sum()) if valid_finite.size else 0.0
        else:
            norm = 0.0
        if norm > 0.0:
            bias_row[rank[sel_idx]] = (row[sel_idx] / norm).astype(bias.dtype)
        idx[b, :] = idx_row
        bias[b, :] = bias_row


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
                row * np.reciprocal(np.sqrt((row * row).mean() + eps)) * g * _stable_sigmoid(g_row)
            ).astype(y.dtype)
        else:
            r = coords[0]
            row, g_row = x[r], gate[r]
            y[r] = (
                row * np.reciprocal(np.sqrt((row * row).mean() + eps)) * g * _stable_sigmoid(g_row)
            ).astype(y.dtype)

    def _body_rope(self, fam: TaskFamily, coords, scalars) -> None:
        x = self.tensor(fam.inputs[0]).astype(np.float64)
        cos_t = self.tensor(fam.inputs[1])
        sin_t = self.tensor(fam.inputs[2])
        y = self.tensor(fam.outputs[0])
        b, h = coords
        # Row b rotates at its OWN runtime position (issue #93 ragged form).
        p = self._row_pos_value(fam, b, scalars)
        row = x[b, h]
        if fam.params.get("convention", "rotate_half") == "neox_partial":
            # NeoX split-half over the first rotary_dim dims (floe Qwen3.5
            # PartialRotaryEmbedding): x1' = x1*c - x2*s / x2' = x2*c + x1*s
            # with half = rotary_dim//2; dims [rotary_dim, D) pass through.
            rot = fam.params["rotary_dim"]
            half = rot // 2
            c = cos_t[p][:half].astype(np.float64)
            s = sin_t[p][:half].astype(np.float64)
            x1, x2 = row[:half], row[half:rot]
            out = row.copy()
            out[:half] = x1 * c - x2 * s
            out[half:rot] = x2 * c + x1 * s
            y[b, h] = out.astype(y.dtype)
            return
        if fam.params.get("convention", "rotate_half") == "interleaved":
            # GPT-J adjacent-pair rotation (issue #95, DeepSeek-V4): pairs
            # (2i, 2i+1), c/s indexed by PAIR index i at the row's position;
            # dims [rotary_dim, D) pass through. PINNED convention: pair
            # stride 2, tables [max_pos, rotary_dim//2], c_i = cos[p, i],
            # s_i = sin[p, i].
            rot = fam.params["rotary_dim"]
            half = rot // 2
            c = cos_t[p][:half].astype(np.float64)
            s = sin_t[p][:half].astype(np.float64)
            x_even, x_odd = row[0:rot:2], row[1:rot:2]
            out = row.copy()
            out[0:rot:2] = x_even * c - x_odd * s
            out[1:rot:2] = x_odd * c + x_even * s
            y[b, h] = out.astype(y.dtype)
            return
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

    def _body_compressor_append(self, fam: TaskFamily, coords, scalars) -> None:
        """Issue #96: emit one compressed entry at the m-token boundary.

        One task per (batch, layer). Rows not at a boundary (``p % m !=
        m-1``) are exact no-ops. Emission (fp32 accumulated, mirrored in
        fp64 here — tile-exact against the device contract):

            w = softmax(gates[b]) ; e = Σ_t w_t·window[b,t]
            e = e/sqrt(mean(e²)+eps)·rms_weight ; e = rotate_half(e, cos[b], sin[b])
            entry_pool[b,l,slot,cb_len,:] = bf16(e)

        then series bookkeeping: cb_len += 1; at cb_len == r//m the
        completed Cb becomes Ca (slots ping-pong) and Cb restarts.
        """
        pool = self.tensor(fam.inputs[0])
        state = self.tensor(fam.inputs[1])
        window = self.tensor(fam.inputs[2])
        gates = self.tensor(fam.inputs[3])
        rms_w = self.tensor(fam.inputs[4])
        cos = self.tensor(fam.inputs[5])
        sin = self.tensor(fam.inputs[6])
        b, l = coords
        p = self._row_pos_value(fam, b, scalars)
        m = fam.params["m"]
        if p % m != m - 1:
            return  # not a boundary token for this row: exact no-op
        R = fam.params["r"] // m
        eps = fam.params["eps"]
        slot = int(state[b, l, 0])
        cb_len = int(state[b, l, 1])
        # fp32-accumulated emission (fp64 mirror here, tile-exact rounding
        # applied only at the bf16 store).
        g = gates[b].astype(np.float64)
        gmax = g.max()
        ex = np.exp(g - gmax)
        w = ex / ex.sum()
        e = (w[:, None] * window[b].astype(np.float64)).sum(axis=0)
        e = e * np.reciprocal(np.sqrt((e * e).mean() + eps)) * rms_w.astype(np.float64)
        half = e.shape[0] // 2
        ch, sh = cos[b].astype(np.float64), sin[b].astype(np.float64)
        e1, e2 = e[:half], e[half:]
        e_rot = np.concatenate([e1 * ch - e2 * sh, e2 * ch + e1 * sh])
        pool[b, l, slot, cb_len, :] = e_rot.astype(pool.dtype)
        cb_len += 1
        if cb_len == R:
            state[b, l, 0] = 1 - slot
            state[b, l, 1] = 0
        else:
            state[b, l, 1] = cb_len

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

    def _body_cache_append_paged(self, fam: TaskFamily, coords, scalars) -> None:
        k_pool = self.tensor(fam.inputs[0])
        v_pool = self.tensor(fam.inputs[1])
        table = self.tensor(fam.inputs[2])
        k_new = self.tensor(fam.inputs[3])
        v_new = self.tensor(fam.inputs[4])
        b, h = coords
        p = self._row_pos_value(fam, b, scalars)  # #93 ragged: per-row bound
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
        p = self._row_pos_value(fam, b, scalars)  # #93 ragged: per-row bound
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
        p = self._row_pos_value(fam, b, scalars)  # #93 ragged: per-row bound
        # Gathered masked: V rows beyond p are never gathered (NaN slots must not leak).
        slots = table[b, : p + 1].astype(np.int64)
        y[b, h, :] = (probs[b, h, : p + 1] @ v_pool[slots, kvh, :]).astype(y.dtype)
    def _body_mhc_pre(self, fam: TaskFamily, coords, scalars) -> None:
        """mHC hyper-connection pre-mix over one batch row.

        fp64 oracle arithmetic mirroring floe
        ``DeepseekV4HyperConnection.forward`` / ``Glm53HyperConnection``
        (device math is fp32; the fp64 mirror is the bare-environment
        oracle pin — issue #99 validation doctrine): unweighted RMSNorm
        over the flattened streams, one [mix, hc·C] GEMV projection,
        sigmoid pre/post gates, softmax + Sinkhorn-Knopp alternate row/col
        normalization (eps inside every denominator) and the pre-weighted
        stream collapse.
        """
        streams = self.tensor(fam.inputs[0])  # [B, hc, C]
        fn = self.tensor(fam.inputs[1])  # [mix, hc·C]
        base = self.tensor(fam.inputs[2])  # [mix]
        scale = self.tensor(fam.inputs[3])  # [3]
        h_in = self.tensor(fam.outputs[0])  # [B, C]
        post_o = self.tensor(fam.outputs[1])  # [B, hc]
        comb_out = self.tensor(fam.outputs[2])  # [B, hc, hc]
        (bb,) = coords
        hc, C = streams.shape[1], streams.shape[2]
        iters, eps = int(fam.params["iters"]), float(fam.params["eps"])
        rms_eps = float(fam.params["rms_eps"])

        flat = streams[bb].astype(np.float64).reshape(-1)  # [hc·C]
        flat = flat / np.sqrt(np.mean(flat * flat) + rms_eps)  # unweighted RMSNorm
        # floe F.linear(flat, fn) — NO bias on the projection; base enters
        # only inside the gates below (adding it here double-counts it)
        logits = fn.astype(np.float64) @ flat  # [mix]
        pre_w, post_w, comb_w = (
            logits[:hc], logits[hc : 2 * hc], logits[2 * hc :].reshape(hc, hc),
        )
        pre_s, post_s, comb_s = (float(scale[0]), float(scale[1]), float(scale[2]))
        pre_b, post_b, comb_b = (
            base.astype(np.float64)[:hc],
            base.astype(np.float64)[hc : 2 * hc],
            base.astype(np.float64)[2 * hc :].reshape(hc, hc),
        )

        pre = 1.0 / (1.0 + np.exp(-(pre_w * pre_s + pre_b))) + eps
        post = 2.0 / (1.0 + np.exp(-(post_w * post_s + post_b)))
        comb_logits = comb_w * comb_s + comb_b
        comb_logits = comb_logits - comb_logits.max(axis=-1, keepdims=True)
        comb = np.exp(comb_logits)
        comb = comb / comb.sum(axis=-1, keepdims=True) + eps
        # Sinkhorn-Knopp: initial column normalization, then (iters−1)
        # alternate row/col passes — eps inside every denominator, exactly
        # as floe conditions them.
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
        for _ in range(iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)

        h_in[bb] = (pre[:, None] * streams[bb].astype(np.float64)).sum(axis=0).astype(h_in.dtype)
        post_o[bb] = post.astype(post_o.dtype)
        comb_out[bb] = comb.astype(comb_out.dtype)

    def _body_mhc_post(self, fam: TaskFamily, coords, scalars) -> None:
        """mHC post-compose over one (batch, stream j) task (fp64 mirror of
        floe ``_mhc_compose``):
        ``streams'[j] = post[j]·body_out + Σ_k comb[k, j]·streams[k]``."""
        streams = self.tensor(fam.inputs[0])  # [B, hc, C]
        body_out = self.tensor(fam.inputs[1])  # [B, C]
        post_w = self.tensor(fam.inputs[2])  # [B, hc]
        comb = self.tensor(fam.inputs[3])  # [B, hc, hc]
        streams_post = self.tensor(fam.outputs[0])  # [B, hc, C]
        bb, j = coords
        acc = (comb[bb, :, j].astype(np.float64)[:, None] * streams[bb].astype(np.float64)).sum(axis=0)
        acc = acc + float(post_w[bb, j]) * body_out[bb].astype(np.float64)
        streams_post[bb, j] = acc.astype(streams_post.dtype)

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
            # Stable sigmoid: exp(-softplus(-l)) == sigmoid(l), no overflow.
            return np.exp(-np.logaddexp(0.0, -logits))
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
        act = g / (1.0 + np.exp(-g)) * u  # silu(g) · u
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
        acc = probs[b, h, :vlen] @ v_cache[b, kvh, :vlen, :]
        if fam.params.get("gated", False):
            # Issue #92: per-head sigmoid output gate fused into the values
            # task — fp64 reference of the device's fp32 epilogue
            # y = acc * sigmoid(gate[b,h,:]) with no extra barrier.
            gate = self.tensor(fam.inputs[2]).astype(np.float64)
            acc = acc * _stable_sigmoid(gate[b, h, :])
        y[b, h, :] = acc.astype(y.dtype)
