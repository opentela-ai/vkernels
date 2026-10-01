"""Dense operator numerical references."""
from __future__ import annotations



import numpy as np



from .task_ir import TaskFamily

from .reference_types import _stable_sigmoid, _decode_e4m3

class DenseReference:
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

