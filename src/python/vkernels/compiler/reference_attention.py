"""Attention operator numerical references."""
from __future__ import annotations



import numpy as np



from .task_ir import TaskFamily

from .reference_types import _stable_sigmoid

class AttentionReference:
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

