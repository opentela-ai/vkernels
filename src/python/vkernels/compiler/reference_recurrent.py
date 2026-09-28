"""Recurrent operator numerical references."""
from __future__ import annotations



import numpy as np



from .task_ir import TaskFamily


class RecurrentReference:
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
        hc = streams.shape[1]
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
        NV = state.shape[1]
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

