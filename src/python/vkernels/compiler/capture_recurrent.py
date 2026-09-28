"""Recurrent operator capture rules."""
from __future__ import annotations

from typing import Optional

from .operator_ir import (
    F32,
    Region,
)

from .capture_types import SymbolicTensor, CaptureError, _regional_reads

class RecurrentRecording:
    def compressor_append(
        self,
        entry_pool: SymbolicTensor,
        series_state: SymbolicTensor,
        window: SymbolicTensor,
        gates: SymbolicTensor,
        rms_weight: SymbolicTensor,
        cos: SymbolicTensor,
        sin: SymbolicTensor,
        position,
        *,
        m: int,
        r: int,
        eps: float,
        name: str = "compressor_append",
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Emit one compressed entry at the m-token boundary (issue #96).

        DeepSeek Sparse Attention's compressor (floe
        ``deepseek_v4/forward.py::_BaseCompressor``): every ``m`` tokens
        compress into one cached entry, kept in a **two-series** per-layer
        pool — the prior completed series ``Ca`` and the in-flight series
        ``Cb`` (width ``2r`` stride ``r`` tokens: at each ``r``-boundary the
        just-completed ``Cb`` becomes the new ``Ca`` and ``Cb`` restarts, so
        the attention window [Ca ∪ Cb] slides by ``r`` per rotation). Entries
        are rope-rotated **once at emission** (at the emitting row's own
        position), so decode only rotates the query.

        Emission math for a boundary row ``b`` (fp32 accumulated, per floe
        ``Compressor.emit``)::

            w = softmax(gates[b, :])                    # over the m window
            e = Σ_t w_t · window[b, t, :]               # weighted latent fold
            e = e / sqrt(mean(e²) + eps) · rms_weight   # rms_norm
            e = rotate_half(e, cos[b], sin[b])          # rotated ONCE, at emission
            entry_pool[b, l, slot, cb_len, :] = bf16(e)

        with ``slot``/``cb_len`` from the persistent per-(row, layer) series
        state, then the rotation bookkeeping (``cb_len += 1``; at ``cb_len ==
        r // m`` the completed ``Cb`` becomes ``Ca`` — slot roles ping-pong —
        and ``Cb`` restarts). Rows whose position is not at a boundary
        (``p[b] % m != m-1``) are exact no-ops — the per-row mask comes from
        the issue #93 positions tensor, so ragged rows emit at their own
        cadence.

        Tensor layout:

        * ``window`` [B, m, D] — the row's in-flight window of shared-latent
          kv vectors (compressed-entry source), from the compressor state
          pool; bf16 stored, fp32-accumulated;
        * ``gates`` [B, m] — fp32 compression gates over the window;
        * ``entry_pool`` [B, L, 2, R, D] bf16 — persistent, two series slots
          of ``R = r // m`` entries each; slot roles ping-pong;
        * ``series_state`` [B, L, 2] i32 — persistent ``[active_slot,
          cb_len]`` per (row, layer);
        * ``cos``/``sin`` [B, D//2] fp32 — gathered by the caller at each
          row's emission position (per-row ragged cadence).

        Reads the series state (RAW), writes the entry pool slab and the
        series state (RMW, ``cache_append`` §4.3 pattern); returns
        *post-append views* — same storages, bumped versions.

        block_bias contract (issue #97, producer of record): entries are
        appended in global emission order, so flat entry index ``j`` is a
        stable per-row identity across the pool; the indexer's
        ``block_bias[b, j]`` aligns 1:1 with that index, and the Ca/Cb
        attention window selects contiguous ``j`` ranges via the #94 slot
        indirection at scores time (#95's integration).
        """
        if m <= 0 or r <= 0 or r % m:
            raise CaptureError(
                f"{name}: series rotation needs r % m == 0; got m={m}, r={r}"
            )
        b, mw, d = window.value.shape
        if mw != m:
            raise CaptureError(f"{name}: window token axis {mw} != m={m}")
        bg, mg = gates.value.shape
        if (bg, mg) != (b, m):
            raise CaptureError(f"{name}: gates {gates.value.shape} != [{b}, {m}]")
        R = r // m
        be, le, slots, re, de = entry_pool.value.shape
        if (be, slots, re, de) != (b, 2, R, d):
            raise CaptureError(
                f"{name}: entry_pool {entry_pool.value.shape} != [B={b}, L, 2, R={R}, D={d}]"
            )
        if series_state.value.shape != (b, le, 2):
            raise CaptureError(
                f"{name}: series_state {series_state.value.shape} != [{b}, {le}, 2]"
            )
        if rms_weight.value.shape != (d,):
            raise CaptureError(f"{name}: rms_weight {rms_weight.value.shape} != [{d}]")
        dh = cos.value.shape[-1]
        if cos.value.shape != (b, dh) or sin.value.shape != (b, dh) or 2 * dh != d:
            raise CaptureError(
                f"{name}: cos/sin must be [B={b}, D/2={d // 2}]; got {cos.value.shape} / {sin.value.shape}"
            )
        valid = self._valid_plus_one(position, name)
        p = self._position_name(position)
        writes = (
            Region.tile(
                entry_pool.value,
                ((0, b), (0, le), (0, 2), (0, R), (0, d)),
            ),
            Region.tile(series_state.value, ((0, b), (0, le), (0, 2))),
        )
        position_form = "row" if valid.is_row else "scalar"
        self._record(
            "compressor_append",
            inputs=(entry_pool, series_state, window, gates, rms_weight, cos, sin),
            outputs=(),  # mutation through storage effects; views returned below
            attributes={
                "m": m,
                "r": r,
                "eps": eps,
                "position": p,
                "position_form": position_form,
            },
            reads=(
                _regional_reads(window.value),
                _regional_reads(gates.value),
                _regional_reads(rms_weight.value),
                _regional_reads(cos.value),
                _regional_reads(sin.value),
                _regional_reads(series_state.value),
            ),
            writes=writes,
            source_location=name,
            numerical_contract={
                "emission": (
                    "boundary rows (p[b] % m == m-1): w = softmax(gates[b]) fp32; "
                    "e = sum_t w_t*window[b,t] fp32; e = e/sqrt(mean(e^2)+eps)*rms_weight; "
                    "e = rotate_half(e, cos[b], sin[b]) — rotated once at emission; "
                    "entry_pool[b,l,slot,cb_len,:] = bf16(e)"
                ),
                "boundary_mask": "rows with p[b] % m != m-1 are exact no-ops (per-row, issue #93)",
                "series_rotation": (
                    "cb_len += 1 after append; at cb_len == r//m the completed Cb "
                    "becomes Ca (slot roles swap) and Cb restarts empty"
                ),
                "block_bias": (
                    "entry flat index j = global emission order (stable per row); "
                    "block_bias[b, j] aligns 1:1 with that index (issue #97 producer "
                    "contract, non-finite-valid normalization as landed); Ca/Cb "
                    "windows select contiguous j ranges via #94 indirection at "
                    "scores time (#95 integration)"
                ),
            },
        )
        pool_post = self.graph.add_tensor(
            f"entry_pool_s{entry_pool.value.storage_id}_v{self.graph.storage_versions[entry_pool.value.storage_id]}",
            entry_pool.value.shape,
            entry_pool.value.dtype,
            storage_id=entry_pool.value.storage_id,
            strides=entry_pool.value.strides,
            offset=entry_pool.value.offset,
            valid_length=valid,
        )
        state_post = self.graph.add_tensor(
            f"series_state_s{series_state.value.storage_id}_v{self.graph.storage_versions[series_state.value.storage_id]}",
            series_state.value.shape,
            series_state.value.dtype,
            storage_id=series_state.value.storage_id,
            strides=series_state.value.strides,
            offset=series_state.value.offset,
            valid_length=valid,
        )
        return SymbolicTensor(pool_post), SymbolicTensor(state_post)


    def gdn_conv(
        self,
        conv_state: SymbolicTensor,
        x: SymbolicTensor,
        w: SymbolicTensor,
        *,
        layer: int,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """GDN short-conv decode step (Qwen3.5 ``GatedDeltaNet``, seq==1 path).

        Causal depthwise conv1d over the packed qkv row, folding the
        persistent conv state::

            full = cat(conv_state[b], x[b])      # [K, C]
            out[b] = silu(sum_k full[:, k] * w[:, k])   # depthwise FIR
            conv_state'[b] = full[1:]            # time-major shift

        ``conv_state`` is an external caller-owned pool [B, K-1, C], fp32,
        time-major (floe's eager init uses the embed dtype — the compiled
        pool is fp32; oracle tolerance ~1e-3). ``x`` is the mixed qkv row
        [B, C] and ``w`` the FIR weights [C, K] (grouped conv, one tap
        vector per channel).

        The op is position-independent: it consumes no decode-position
        scalar, and its ordering obligation is the read-modify-write on
        the state pool (RAW/WAR/WAW hazards vs any other op touching that
        storage). Records write effects on the pool and returns
        ``(out, conv_state_post)`` — the post-step state view (same
        storage, bumped version) that later layers must read.
        """
        sv, xv, wv = conv_state.value, x.value, w.value
        if len(sv.shape) != 3:
            raise CaptureError(f"gdn_conv state pool must be [B, K-1, C]; got shape {sv.shape}")
        if len(xv.shape) != 2:
            raise CaptureError(f"gdn_conv input row must be [B, C]; got shape {xv.shape}")
        B, Km1, C = sv.shape
        K = Km1 + 1
        if xv.shape != (B, C):
            raise CaptureError(f"gdn_conv input row shape {xv.shape} does not match state pool [B, C] = {(B, C)}")
        if tuple(wv.shape) != (C, K):
            raise CaptureError(f"gdn_conv FIR weights must be [C, K] = [{C}, {K}]; got shape {wv.shape}")
        out = self.fresh_buffer(f"gdn_conv_l{layer}{self._suffix()}", (B, C), dtype=xv.dtype)
        self._record(
            "gdn_conv",
            inputs=(conv_state, x, w),
            outputs=(out,),
            attributes={"layer": layer, "conv_kernel": K},
            reads=(_regional_reads(sv), _regional_reads(xv), _regional_reads(wv)),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} gdn conv",
            numerical_contract={
                "fir": "out[b] = silu(sum_{j<K-1} w[:, j] * state[b, j, :] + w[:, K-1] * x[b, :])",
                "state_shift": "state'[b, j, :] = state[b, j+1, :] for j < K-2; state'[b, K-2, :] = x[b, :]",
                "silu": "x / (1 + exp(-x))",
                "accumulate": "fp32 accumulation",
                "dtypes": "x/w bf16 (.cg loads), state pool fp32 [B, K-1, C] time-major (floe eager init uses the embed dtype — compiled pool is fp32; oracle tolerance ~1e-3)",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"conv_state_l{layer}_v{self.graph.storage_versions[sid]}",
            sv.shape,
            sv.dtype,
            storage_id=sid,
            strides=sv.strides,
            offset=sv.offset,
        )
        return out, SymbolicTensor(post)


    def mhc_pre(
        self,
        streams: SymbolicTensor,
        fn: SymbolicTensor,
        base: SymbolicTensor,
        scale: SymbolicTensor,
        *,
        layer: int,
        iters: int,
        eps: float,
        rms_eps: float,
    ) -> tuple[SymbolicTensor, SymbolicTensor, SymbolicTensor]:
        """mHC hyper-connection pre-mix (issue #99; floe
        ``DeepseekV4HyperConnection.forward`` / ``Glm53HyperConnection.forward``
        — one op family, family attributes).

        Per decode token (row ``b`` of the ``[B, hc, C]`` stream stack), with
        ``mix = (2 + hc)·hc``:

            flat    = unweighted_rms_norm(flatten(streams[b]))    # [hc·C], fp32
            logits  = fn @ flat                                   # [mix] GEMV — NO bias:
                                                                  # base enters only in the gates
            pre_w, post_w, comb_w = split(logits, [hc, hc, hc²])
            pre     = sigmoid(pre_w·pre_s + pre_b) + eps          # [hc]
            post    = 2·sigmoid(post_w·post_s + post_b)           # [hc], (0, 2)
            comb    = softmax((comb_w·comb_s + comb_b).view(hc, hc), -1) + eps
            comb    = comb / (colsum(comb) + eps)                 # Sinkhorn-Knopp:
            for _ in range(iters − 1):                            # alternate row/col
                comb = comb / (rowsum(comb) + eps)                # normalization to
                comb = comb / (colsum(comb) + eps)                # doubly-stochastic
            h_in   = Σ_h pre_h · streams[b, h, :]                 # stream collapse

        ``pre``/``post``/``comb`` are DATA-DEPENDENT — computed from the stream
        contents every step, so the Sinkhorn projection runs per token, not at
        load time. ``base`` splits into (pre_b, post_b, comb_b) with comb_b
        viewed ``[hc, hc]``; ``scale`` = (pre_s, post_s, comb_s) scalars. All
        mixing math is fp32 (Sinkhorn numerics demand it); ``h_in`` follows the
        streams dtype.

        Stream state is intermediate workspace: ``streams`` is read-only here
        (the composing ``mhc_post`` writes the updated stack to a fresh buffer),
        and ``h_in``/``post``/``comb`` are fresh compiler-planned buffers — no
        persistent state pool, no read-modify-write effects.

        Returns ``(h_in, post, comb)``; the block body consumes ``h_in`` and
        ``mhc_post`` consumes ``(streams, body_out, post, comb)``.
        """
        sv, fnv, bv, scv = (
            streams.value, fn.value, base.value, scale.value,
        )
        if len(sv.shape) != 3:
            raise CaptureError(f"mhc_pre streams must be [B, hc, C]; got shape {sv.shape}")
        B, hc, C = sv.shape
        mix = (2 + hc) * hc
        if fnv.shape != (mix, hc * C):
            raise CaptureError(
                f"mhc_pre fn must be [{mix}, {hc * C}] = [(2+hc)·hc, hc·C]; got shape {fnv.shape}"
            )
        if bv.shape != (mix,):
            raise CaptureError(f"mhc_pre base must be [{mix}]; got shape {bv.shape}")
        if scv.shape != (3,):
            raise CaptureError(f"mhc_pre scale must be [3] = (pre_s, post_s, comb_s); got shape {scv.shape}")
        if iters < 1:
            raise CaptureError(f"mhc_pre Sinkhorn iters must be >= 1; got {iters}")
        h_in = self.fresh_buffer(f"mhc_h_in_l{layer}{self._suffix()}", (B, C), dtype=sv.dtype)
        post = self.fresh_buffer(f"mhc_post_w_l{layer}{self._suffix()}", (B, hc), dtype=F32)
        comb = self.fresh_buffer(f"mhc_comb_l{layer}{self._suffix()}", (B, hc, hc), dtype=F32)
        self._record(
            "mhc_pre",
            inputs=(streams, fn, base, scale),
            outputs=(h_in, post, comb),
            attributes={"layer": layer, "hc": hc, "iters": iters, "eps": eps, "rms_eps": rms_eps},
            reads=(
                _regional_reads(sv), _regional_reads(fnv),
                _regional_reads(bv), _regional_reads(scv),
            ),
            writes=(
                _regional_reads(h_in.value), _regional_reads(post.value),
                _regional_reads(comb.value),
            ),
            source_location=f"layer {layer} mhc pre-mix",
            numerical_contract={
                "input_norm": "flat = flatten(streams[b]) * rsqrt(mean(flat²) + rms_eps), fp32 (unweighted RMSNorm over the full hc·C vector)",
                "projection": "logits = fn @ flat (F.linear, NO projection bias), one [mix=(2+hc)·hc, hc·C] GEMV per token (data-dependent); base enters only inside the gates",
                "pre": "pre = sigmoid(pre_w·pre_s + pre_b) + eps, [hc]",
                "post": "post = 2·sigmoid(post_w·post_s + post_b), [hc], range (0, 2)",
                "comb": "comb = softmax((comb_w·comb_s + comb_b.view(hc, hc)), dim=-1) + eps",
                "sinkhorn": "comb /= (colsum(comb) + eps); then (iters−1)× [comb /= (rowsum+eps); comb /= (colsum+eps)] — alternate normalization to doubly-stochastic, eps inside every denominator",
                "collapse": "h_in = Σ_h pre_h · streams[b, h, :] (raw streams, not the normalized flat)",
                "dtypes": "streams bf16 workspace loads, all mixing math fp32 (Sinkhorn numerics demand it), h_in follows the streams dtype",
                "state": "streams read-only workspace; h_in/post/comb fresh buffers — intermediate, NOT a persistent state pool",
            },
        )
        return h_in, post, comb


    def mhc_post(
        self,
        streams: SymbolicTensor,
        body_out: SymbolicTensor,
        post: SymbolicTensor,
        comb: SymbolicTensor,
        *,
        layer: int,
    ) -> SymbolicTensor:
        """mHC hyper-connection post-compose (issue #99; floe
        ``_mhc_compose``): place the sublayer output back onto the hc parallel
        residual streams with the Sinkhorn-projected mixer —

            streams'[j, :] = post[j]·body_out[:] + Σ_k comb[k, j]·streams[k, :]

        (floe torch reference: ``post.unsqueeze(-1)·sublayer_out.unsqueeze(-2)
        + combᵀ @ residual``). ``post``/``comb`` are the data-dependent weights
        this token's ``mhc_pre`` produced. The updated stream stack is written
        to a FRESH ``[B, hc, C]`` workspace buffer (intermediate, NOT a
        persistent state pool — the next layer's ``mhc_pre`` consumes the
        returned view through an ordinary RAW dependency).
        """
        sv, ov, pv, cv = (
            streams.value, body_out.value, post.value, comb.value,
        )
        if len(sv.shape) != 3:
            raise CaptureError(f"mhc_post streams must be [B, hc, C]; got shape {sv.shape}")
        B, hc, C = sv.shape
        if ov.shape != (B, C):
            raise CaptureError(f"mhc_post body_out must be [B, {C}]; got shape {ov.shape}")
        if pv.shape != (B, hc):
            raise CaptureError(f"mhc_post post weights must be [B, {hc}]; got shape {pv.shape}")
        if cv.shape != (B, hc, hc):
            raise CaptureError(f"mhc_post comb must be [B, {hc}, {hc}]; got shape {cv.shape}")
        streams_post = self.fresh_buffer(
            f"mhc_streams_l{layer}{self._suffix()}", (B, hc, C), dtype=sv.dtype
        )
        self._record(
            "mhc_post",
            inputs=(streams, body_out, post, comb),
            outputs=(streams_post,),
            attributes={"layer": layer, "hc": hc},
            reads=(
                _regional_reads(sv), _regional_reads(ov),
                _regional_reads(pv), _regional_reads(cv),
            ),
            writes=(_regional_reads(streams_post.value),),
            source_location=f"layer {layer} mhc post-compose",
            numerical_contract={
                "compose": "streams'[j] = post[j]·body_out + Σ_k comb[k, j]·streams[k] (combᵀ @ streams + post ⊙ body_out)",
                "dtypes": "fp32 compose arithmetic, streams' follows the streams dtype",
                "state": "fresh [B, hc, C] workspace buffer per layer — intermediate, NOT a persistent state pool",
            },
        )
        return streams_post


    def gdn_delta(
        self,
        ssm_state: SymbolicTensor,
        q: SymbolicTensor,
        k: SymbolicTensor,
        v: SymbolicTensor,
        z: SymbolicTensor,
        a: SymbolicTensor,
        b: SymbolicTensor,
        a_log: SymbolicTensor,
        dt_bias: SymbolicTensor,
        norm_w: SymbolicTensor,
        *,
        layer: int,
        scale: float,
        eps: float,
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """Gated delta rule decode step (Qwen3.5 ``GatedDeltaNet``, seq==1).

        Per value head ``h`` over its ``[HV, HK]`` fp32 SSM state slice,
        with ``GROUP = NV // NK`` and ``kh = h // GROUP``::

            g      = -exp(A_log[h]) * softplus(a[b,h] + dt_bias[h])
            beta   = sigmoid(b[b,h])
            q_n    = q[b,kh] / sqrt(|q[b,kh]|^2 + 1e-6) * scale
            k_n    = k[b,kh] / sqrt(|k[b,kh]|^2 + 1e-6)
            s     *= exp(g)                            # per-head scalar decay
            sk     = s @ k_n                           # state read
            s     += beta * (v[b,h] - sk) outer k_n     # delta-rule update
            o      = s @ q_n                           # state readout
            out    = RMSNorm_HV(o) * norm_w * (z * sigmoid(z))

        ``ssm_state`` is an external caller-owned pool ``[B, NV, HV, HK]``,
        fp32, read-modify-write per (batch, head) row. ``q``/``k`` are the
        post-conv rows ``[B, NK, HK]`` (normalized *inside* the op, exactly
        as floe ``qwen35_gdn.GatedDeltaNet.forward`` conditions them);
        ``v``/``z`` are ``[B, NV, HV]``; ``a``/``b`` are ``[B, NV]``;
        ``A_log``/``dt_bias`` are ``[NV]`` and ``norm_w`` ``[HV]`` per-layer
        parameters broadcast over the batch. All fp32 arithmetic between
        the gates and the state (floe keeps the recurrence fp32; inputs may
        arrive bf16 from the projections).

        The op is position-independent: its ordering obligation is the
        read-modify-write on the state pool (RAW/WAR/WAW hazards vs any
        other op touching that storage). Records write effects on the pool
        and returns ``(out, ssm_state_post)`` -- the post-step state view
        (same storage, bumped version) that later layers must read.
        """
        sv, qv, kv, vv, zv, av, bv = (
            ssm_state.value, q.value, k.value, v.value, z.value, a.value, b.value,
        )
        if len(sv.shape) != 4:
            raise CaptureError(f"gdn_delta state pool must be [B, NV, HV, HK]; got shape {sv.shape}")
        B, NV, HV, HK = sv.shape
        if len(qv.shape) != 3 or qv.shape[0] != B or qv.shape[2] != HK:
            raise CaptureError(f"gdn_delta q rows must be [B, NK, {HK}]; got shape {qv.shape}")
        NK = qv.shape[1]
        if NK < 1 or NV % NK:
            raise CaptureError(f"gdn_delta head group must divide: NV={NV} not divisible by NK={NK}")
        if kv.shape != (B, NK, HK):
            raise CaptureError(f"gdn_delta k rows must be [B, {NK}, {HK}]; got shape {kv.shape}")
        if vv.shape != (B, NV, HV):
            raise CaptureError(f"gdn_delta v rows must be [B, {NV}, {HV}]; got shape {vv.shape}")
        if zv.shape != (B, NV, HV):
            raise CaptureError(f"gdn_delta z gate rows must be [B, {NV}, {HV}]; got shape {zv.shape}")
        if av.shape != (B, NV) or bv.shape != (B, NV):
            raise CaptureError(f"gdn_delta a/b rows must be [B, {NV}]; got {av.shape} / {bv.shape}")
        if a_log.value.shape != (NV,) or dt_bias.value.shape != (NV,):
            raise CaptureError(f"gdn_delta A_log/dt_bias must be [{NV}]; got {a_log.value.shape} / {dt_bias.value.shape}")
        if norm_w.value.shape != (HV,):
            raise CaptureError(f"gdn_delta norm_w must be [{HV}]; got shape {norm_w.value.shape}")
        out = self.fresh_buffer(f"gdn_delta_l{layer}{self._suffix()}", (B, NV, HV), dtype=vv.dtype)
        self._record(
            "gdn_delta",
            inputs=(ssm_state, q, k, v, z, a, b, a_log, dt_bias, norm_w),
            outputs=(out,),
            attributes={"layer": layer, "scale": scale, "eps": eps},
            reads=(
                _regional_reads(sv), _regional_reads(qv), _regional_reads(kv),
                _regional_reads(vv), _regional_reads(zv), _regional_reads(av),
                _regional_reads(bv), _regional_reads(a_log.value),
                _regional_reads(dt_bias.value), _regional_reads(norm_w.value),
            ),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} gdn delta rule",
            numerical_contract={
                "decay": "g = -exp(A_log) * softplus(a + dt_bias); s *= exp(g) (softplus guarded: x>20 -> x)",
                "qk_norm": "q_n = L2(q)*scale, k_n = L2(k) per key head (group-expanded), eps 1e-6 inside the sqrt",
                "delta_rule": "sk = s @ k_n; s += beta*(v - sk) outer k_n with beta = sigmoid(b)",
                "readout": "o = s @ q_n",
                "out_gate": "out = RMSNorm_HV(o) * norm_w * (z * sigmoid(z)), eps inside the rsqrt",
                "accumulate": "fp32 arithmetic between gates and state",
                "dtypes": "q/k/v/z/a/b workspace loads (.cg), A_log/dt_bias/norm_w params, SSM state pool fp32 [B, NV, HV, HK] read-modify-write, out follows v's dtype",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"ssm_state_l{layer}_v{self.graph.storage_versions[sid]}",
            sv.shape,
            sv.dtype,
            storage_id=sid,
            strides=sv.strides,
            offset=sv.offset,
        )
        return out, SymbolicTensor(post)


    def kda_delta(
        self,
        ssm_state: SymbolicTensor,
        q: SymbolicTensor,
        k: SymbolicTensor,
        v: SymbolicTensor,
        f: SymbolicTensor,
        b: SymbolicTensor,
        dt_bias: SymbolicTensor,
        A_log: SymbolicTensor,
        *,
        layer: int,
        scale: float,
        lower_bound: Optional[float],
    ) -> tuple[SymbolicTensor, SymbolicTensor]:
        """KDA gated delta rule decode step with element-wise decay
        (GLM-5.3 ``Glm53LinearAttention``, seq==1 — Kimi delta attention).

        Per head ``h`` over its ``[K, V]`` fp32 state slice (KDA heads are
        square: K = V = head_dim; one q/k/v head each, no group expansion)::

            g      = lower_bound * sigmoid(exp(A_log[h]) * (f[b,h] + dt_bias[h]))
                     # log-space [K]; lower_bound None -> -exp(A_log)*softplus
            beta   = sigmoid(b[b,h])
            q_n    = q[b,h] / sqrt(|q|^2 + 1e-6) * scale
            k_n    = k[b,h] / sqrt(|k|^2 + 1e-6)
            s     *= exp(g)[None over V per k-row]      # element-wise decay
            kv     = sum_k s * k_n                      # [V]
            s     += k_n outer (beta * (v - kv))
            o      = sum_k s * q_n                      # [V]

        ``ssm_state`` is an external caller-owned pool ``[B, H, K, V]`` fp32,
        read-modify-write per (batch, head) row — same §4.3 pattern as
        gdn_conv/gdn_delta. ``q``/``k`` are the post-conv rows ``[B, H, K]``
        (normalized *inside* the op, exactly as floe ``_l2norm`` conditions
        them before ``_kda_recurrent``); ``v`` is ``[B, H, V]``; ``f`` is
        the ``f_b(f_a(x))`` projection row ``[B, H, K]`` with ``dt_bias``
        ``[H, K]`` and ``A_log`` ``[H]`` folded inside (the gate arithmetic
        is per-(head, k-dim), so it rides with the task — unlike gdn_delta's
        per-head scalars it cannot stay outside without a materialized
        ``[B, H, K]`` intermediate). ``lower_bound`` is the config scalar
        (``Glm53Config.linear_lower_bound``; ``None`` selects the guarded
        softplus branch). All gate/state arithmetic fp32; output follows
        ``v``'s dtype.

        The op is position-independent: its ordering obligation is the
        read-modify-write on the state pool (RAW/WAR/WAW hazards vs any
        other op touching that storage). Records write effects on the pool
        and returns ``(out, ssm_state_post)`` — the post-step state view
        (same storage, bumped version) that later layers must read.
        """
        sv, qv, kv, vv, fv, bv = (
            ssm_state.value, q.value, k.value, v.value, f.value, b.value,
        )
        if len(sv.shape) != 4:
            raise CaptureError(f"kda_delta state pool must be [B, H, K, V]; got shape {sv.shape}")
        B, H, K, V = sv.shape
        if K != V:
            raise CaptureError(f"kda_delta heads must be square (K = V = head_dim); got K={K}, V={V}")
        if qv.shape != (B, H, K):
            raise CaptureError(f"kda_delta q rows must be [B, {H}, {K}]; got shape {qv.shape}")
        if kv.shape != (B, H, K):
            raise CaptureError(f"kda_delta k rows must be [B, {H}, {K}]; got shape {kv.shape}")
        if vv.shape != (B, H, V):
            raise CaptureError(f"kda_delta v rows must be [B, {H}, {V}]; got shape {vv.shape}")
        if fv.shape != (B, H, K):
            raise CaptureError(f"kda_delta f rows (f_b projection) must be [B, {H}, {K}]; got shape {fv.shape}")
        if bv.shape != (B, H):
            raise CaptureError(f"kda_delta b logits must be [B, {H}]; got shape {bv.shape}")
        if dt_bias.value.shape != (H, K):
            raise CaptureError(f"kda_delta dt_bias must be [{H}, {K}]; got {dt_bias.value.shape}")
        if A_log.value.shape != (H,):
            raise CaptureError(f"kda_delta A_log must be [{H}]; got {A_log.value.shape}")
        out = self.fresh_buffer(f"kda_delta_l{layer}{self._suffix()}", (B, H, V), dtype=vv.dtype)
        self._record(
            "kda_delta",
            inputs=(ssm_state, q, k, v, f, b, dt_bias, A_log),
            outputs=(out,),
            attributes={"layer": layer, "scale": scale, "lower_bound": lower_bound},
            reads=(
                _regional_reads(sv), _regional_reads(qv), _regional_reads(kv),
                _regional_reads(vv), _regional_reads(fv), _regional_reads(bv),
                _regional_reads(dt_bias.value), _regional_reads(A_log.value),
            ),
            writes=(_regional_reads(sv), _regional_reads(out.value)),
            source_location=f"layer {layer} kda delta rule",
            numerical_contract={
                "gate": "g = lower_bound * sigmoid(exp(A_log[h]) * (f[b,h] + dt_bias[h])) per (head, k-dim); lower_bound None -> g = -exp(A_log[h]) * softplus(f + dt_bias) with the x>20 guard",
                "decay": "s *= exp(g) broadcast over the value axis (element-wise per k-row; richer than gdn_delta's scalar-per-head decay)",
                "qk_norm": "q_n = L2(q)*scale, k_n = L2(k) per head, eps 1e-6 inside the sqrt (floe _l2norm)",
                "beta": "beta = sigmoid(b[b,h])",
                "delta_rule": "kv = sum_k s * k_n; s += k_n outer (beta * (v - kv))",
                "readout": "o = sum_k s * q_n",
                "accumulate": "fp32 arithmetic between gates and state (floe keeps the KDA recurrence fp32 unconditionally)",
                "dtypes": "q/k/v/f/b workspace loads (.cg), dt_bias/A_log params, state pool fp32 [B, H, K, V] read-modify-write, out follows v's dtype",
                "pool": "external caller-owned state pool; read-modify-write per decode step",
            },
        )
        # Post-step state view: same pool, bumped version (cache_append's
        # §4.3 pattern). Later layers must consume this view.
        sid = sv.storage_id
        post = self.graph.add_tensor(
            f"kda_state_l{layer}_v{self.graph.storage_versions[sid]}",
            sv.shape,
            sv.dtype,
            storage_id=sid,
            strides=sv.strides,
            offset=sv.offset,
        )
        return out, SymbolicTensor(post)

