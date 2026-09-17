"""DeepSeek-V4 tiny fixture for the Stage-2 megakernel (issue #102 Stage 2).

Model definition only: config, random weights, decode-state pools, and the
fp64 whole-decode-step mirror (the bare-environment oracle of record). The
compiler capture layer lives in :mod:`.model_deepseek_v4`; the eager floe
``deepseek_v4/forward.py`` is the §15.1 oracle (import-gated).

Every stage of the mirror transcribes the LANDED op contracts verbatim
(#93 per-row positions, #94 slot-table indirection, #95 MLA scores/values +
conjugate rope + grouped o-projection, #96 two-series compressor, #97
indexer scores/top-k with block bias, #98 routed MoE, #99 hyper-connection
mixing) — the compiled whole-step graph and this mirror are two readings of
the same contracts, so numerics disagreements localize the bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4Config:
    """Tiny DeepSeek-V4 decode fixture (MLA + DSA + MoE + mHC)."""

    batch: int = 2
    layers: int = 2
    vocab: int = 64
    hidden: int = 32  # C — block body width (h_in, moe in/out)

    # hyper-connections (#99)
    hc: int = 2  # parallel residual streams
    mhc_iters: int = 2  # Sinkhorn iterations
    mhc_eps: float = 1e-6
    mhc_rms_eps: float = 1e-6

    # MLA (#95): MQA over a shared latent cache; q head dim == latent dim so
    # the conjugate output-side rope and the latent gather share one width.
    heads: int = 4  # query heads H
    latent_dim: int = 16  # D — shared KV-latent width (== q head dim)
    window: int = 4  # W — sliding-window candidate count (recent tokens)
    attention_scale: float = 0.25  # 1/sqrt(D) precomputed

    # DSA (#96/#97)
    compress_m: int = 4  # m — window tokens per emitted entry
    compress_r: int = 8  # r — series span; R = r // m entries per series
    index_heads: int = 2  # H_i — indexer heads (head dim == latent_dim)
    index_k: int = 3  # K — fixed top-k selected compressed entries
    compressor_eps: float = 1e-6

    # MoE (#98)
    n_experts: int = 4
    moe_top_k: int = 2
    moe_intermediate: int = 24
    routed_scaling_factor: float = 1.3
    shared_intermediate: int = 24

    rms_eps: float = 1e-6
    max_positions: int = 8
    cache_capacity: int = 8  # S — per-row latent slots (paged, #94)

    def validate(self) -> None:
        if self.batch < 1 or self.layers < 1:
            raise ValueError("batch/layers must be >= 1")
        if self.hc < 1:
            raise ValueError("hc must be >= 1")
        if self.latent_dim % 2:
            raise ValueError("latent_dim must be even (interleaved rope)")
        if self.compress_r % self.compress_m:
            raise ValueError("compress_r must be a multiple of compress_m")
        if self.window < 1 or self.index_k < 1:
            raise ValueError("window and index_k must be >= 1")
        if self.cache_capacity < self.max_positions:
            raise ValueError("cache_capacity must cover max_positions")
        if self.moe_top_k > self.n_experts:
            raise ValueError("moe_top_k exceeds n_experts")
        if self.heads * self.latent_dim % self.heads:
            raise ValueError("grouped o-proj requires N % H == 0")

    @property
    def entries_per_series(self) -> int:
        return self.compress_r // self.compress_m  # R

    @property
    def entry_capacity(self) -> int:
        return 2 * self.entries_per_series  # M — Ca ∪ Cb flat width


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def _rng_weights(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


@dataclass
class MHCWeights:
    """One mHC pre pair (#99): fn [(2+hc)·hc, hc·C], base [mix], scale [3]."""

    fn: np.ndarray  # [(2+hc)*hc, hc*C]
    base: np.ndarray  # [mix]
    scale: np.ndarray  # [3] = (pre_s, post_s, comb_s)


@dataclass
class LayerWeights:
    # mHC pairs: attention sublayer + MoE sublayer
    mhc_attn: MHCWeights
    mhc_moe: MHCWeights
    # MLA
    attn_ln_gamma: np.ndarray  # [C]
    q_proj_w: np.ndarray  # [C, H*D]
    kv_a_w: np.ndarray  # [C, D]  (single shared latent head, MQA)
    o_proj_w: np.ndarray  # [H*(C//H), H*D] block-diagonal (grouped, #95)
    # DSA indexer
    idx_q_w: np.ndarray  # [C, H_i*D]
    idx_mix_w: np.ndarray  # [C, H_i]  (1/sqrt(H_i) folded in)
    compressor_rms_w: np.ndarray  # [D]
    mla_sink: np.ndarray  # [H]
    # MoE
    moe_ln_gamma: np.ndarray  # [C]
    router_w: np.ndarray  # [E, C]
    expert_gate_up: np.ndarray  # [E, 2I, C] fp32 stack
    expert_down: np.ndarray  # [E, C, I] fp32 stack
    shared_gate_up: np.ndarray  # [2I_s, C]  (same layout as the expert stack)
    shared_down: np.ndarray  # [C, I_s]


@dataclass
class DeepseekV4Weights:
    token_emb: np.ndarray  # [V, hc*C] — embedding writes the stream stack
    final_gamma: np.ndarray  # [hc*C]
    layers: list  # [LayerWeights] * layers


def random_deepseek_weights(config: DeepseekV4Config, seed: int = 0) -> DeepseekV4Weights:
    cfg = config
    rng = np.random.default_rng(seed)
    C, D, H, HI = cfg.hidden, cfg.latent_dim, cfg.heads, cfg.index_heads
    mix = (2 + cfg.hc) * cfg.hc

    def mhc_pair() -> MHCWeights:
        return MHCWeights(
            fn=_rng_weights(rng, mix, cfg.hc * C),
            base=_rng_weights(rng, mix) * 0.1,
            scale=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

    layers = []
    for _ in range(cfg.layers):
        layers.append(
            LayerWeights(
                mhc_attn=mhc_pair(),
                mhc_moe=mhc_pair(),
                attn_ln_gamma=(1.0 + 0.05 * rng.standard_normal(C)).astype(np.float32),
                q_proj_w=_rng_weights(rng, C, H * D),
                kv_a_w=_rng_weights(rng, C, D),
                # block-diagonal grouped o-proj: only w[g*N:(g+1)*N, g*K:(g+1)*K]
                # is ever read; fill the off-blocks too (they must be IGNORED —
                # the grouped-linear contract is exercised by the numerics).
                o_proj_w=_rng_weights(rng, H * (C // H), H * D),
                idx_q_w=_rng_weights(rng, C, HI * D),
                idx_mix_w=_rng_weights(rng, C, HI) * (HI**-0.5),
                compressor_rms_w=(1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32),
                mla_sink=_rng_weights(rng, H) * 0.1,
                moe_ln_gamma=(1.0 + 0.05 * rng.standard_normal(C)).astype(np.float32),
                router_w=_rng_weights(rng, cfg.n_experts, C),
                expert_gate_up=_rng_weights(rng, cfg.n_experts, 2 * cfg.moe_intermediate, C),
                expert_down=_rng_weights(rng, cfg.n_experts, C, cfg.moe_intermediate),
                shared_gate_up=_rng_weights(rng, 2 * cfg.shared_intermediate, C),
                shared_down=_rng_weights(rng, C, cfg.shared_intermediate),
            )
        )
    return DeepseekV4Weights(
        token_emb=_rng_weights(rng, cfg.vocab, cfg.hc * C),
        final_gamma=(1.0 + 0.05 * rng.standard_normal(cfg.hc * C)).astype(np.float32),
        layers=layers,
    )


# ---------------------------------------------------------------------------
# Decode state (persistent pools the step reads / read-modify-writes)
# ---------------------------------------------------------------------------


@dataclass
class DeepseekV4DecodeState:
    row_positions: np.ndarray  # [B] i32
    ids: np.ndarray  # [B] i32
    # #94: global paged table (appends) + row-local view (mla window_table).
    # global[b, t] = b*S + local[b, t]; local is a NON-identity permutation.
    slot_table_global: np.ndarray  # [B, S] i32
    slot_table_local: np.ndarray  # [B, S] i32
    latent_k: np.ndarray  # [layers, B*S, 1, D] f32 — NaN-carved, paged appends
    latent_v: np.ndarray  # [layers, B*S, 1, D] f32
    entry_pool: np.ndarray  # [B, layers, 2, R, D] f32 — NaN-carved
    series_state: np.ndarray  # [B, layers, 2] i32 — [active_slot, cb_len]
    # per-step host-side inputs (#96 landed pattern): the in-flight compressor
    # window over the row's last m latent rows, its gates, and the per-row
    # emission rope rows.
    comp_window: np.ndarray  # [B, m, D] f32
    comp_gates: np.ndarray  # [B, m] f32
    comp_cos: np.ndarray  # [B, D//2] f32
    comp_sin: np.ndarray  # [B, D//2] f32
    valid_counts: np.ndarray  # [B] i32 — emitted entries so far (per row)


def initial_state(config: DeepseekV4Config, seed: int = 1) -> DeepseekV4DecodeState:
    cfg = config
    rng = np.random.default_rng(seed)
    B, S, L, D = cfg.batch, cfg.cache_capacity, cfg.layers, cfg.latent_dim
    # positions (5, 2): ragged rows; positions small enough that some rows sit
    # on a compression boundary and others do not.
    row_positions = np.array([5, 2][:B], dtype=np.int32)
    if B > 2:
        row_positions = np.concatenate([row_positions, rng.integers(0, 6, size=B - 2).astype(np.int32)])
    ids = rng.integers(0, cfg.vocab, size=B).astype(np.int32)
    # #94 non-identity per-row slot permutation (slot 0 reserved as null)
    local = np.zeros((B, S), dtype=np.int32)
    glob = np.zeros((B, S), dtype=np.int32)
    for b in range(B):
        perm = rng.permutation(S)  # includes the reserved slot 0
        local[b] = perm
        glob[b] = b * S + perm
    carved = lambda *shape: np.full(shape, np.nan, dtype=np.float32)  # noqa: E731
    st = DeepseekV4DecodeState(
        row_positions=row_positions,
        ids=ids,
        slot_table_global=glob,
        slot_table_local=local,
        latent_k=carved(L, B * S, 1, D),
        latent_v=carved(L, B * S, 1, D),
        entry_pool=carved(B, L, 2, cfg.entries_per_series, D),
        series_state=np.zeros((B, L, 2), dtype=np.int32),
        comp_window=np.zeros((B, cfg.compress_m, D), dtype=np.float32),
        comp_gates=rng.standard_normal((B, cfg.compress_m)).astype(np.float32),
        comp_cos=np.cos(rng.standard_normal((B, D // 2)) * 0.3).astype(np.float32),
        comp_sin=np.sin(rng.standard_normal((B, D // 2)) * 0.3).astype(np.float32),
        valid_counts=np.zeros(B, dtype=np.int32),
    )
    return _seed_history(config, st, rng)


def _seed_history(config: DeepseekV4Config, st: DeepseekV4DecodeState, rng: np.random.Generator) -> DeepseekV4DecodeState:
    """Pre-fill pools with the rows' PRIOR tokens (t < p) so the decode step
    attends over a non-trivial history: latent appends land at the table
    slots, compressor emissions fire at every boundary t < p, and the window
    holds the row's last m latents."""
    cfg = config
    B, S, L, D = cfg.batch, cfg.cache_capacity, cfg.layers, cfg.latent_dim
    m, R = cfg.compress_m, cfg.entries_per_series
    for b in range(B):
        p = int(st.row_positions[b])
        hist = (rng.standard_normal((p, D)) * 0.5).astype(np.float32)  # t = 0..p-1
        for t in range(p):
            slot = int(st.slot_table_global[b, t])
            row = hist[t]
            for l in range(L):
                st.latent_k[l, slot, 0, :] = row
                st.latent_v[l, slot, 0, :] = row
            # compressor emissions at boundaries t % m == m-1
            if t % m == m - 1:
                e_idx = t // m  # emission order j (global per row)
                series, off = divmod(e_idx, R)
                for l in range(L):
                    st.entry_pool[b, l, series, off, :] = row * (1.0 + 0.1 * l)
                # series state after emitting entry e_idx
                nxt = e_idx + 1
                for l in range(L):
                    st.series_state[b, l, 0] = (nxt // R) % 2
                    st.series_state[b, l, 1] = nxt % R
        st.valid_counts[b] = p // m  # emissions at t = m-1, 2m-1, … < p
        # window = the row's last m latents (t = p-m .. p-1); t < 0 rows zero
        win = np.zeros((m, D), dtype=np.float32)
        for i, t in enumerate(range(p - m, p)):
            if t >= 0:
                win[i] = hist[t]
        st.comp_window[b] = win
        st.comp_gates[b] = rng.standard_normal(m).astype(np.float32)
    return st


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------


def _rms(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def _rope_interleaved(x: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray) -> np.ndarray:
    """x[..., D] rotated with the interleaved convention (issue #95)."""
    x2 = x.reshape(*x.shape[:-1], -1, 2)
    lead = x2.shape[:-2]  # e.g. (H,)
    c = np.broadcast_to(cos_row.reshape((1,) * len(lead) + (-1,)), lead + cos_row.shape)
    s = np.broadcast_to(sin_row.reshape((1,) * len(lead) + (-1,)), lead + sin_row.shape)
    x0, x1 = x2[..., 0], x2[..., 1]
    return np.stack([x0 * c - x1 * s, x1 * c + x0 * s], axis=-1).reshape(x.shape)


def _rope_interleaved_inverse(x: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray) -> np.ndarray:
    """conjugate_rope: same rotation with sin NEGATED (#95)."""
    return _rope_interleaved(x, cos_row, -sin_row)


def _rotate_half(x: np.ndarray, cos_row: np.ndarray, sin_row: np.ndarray) -> np.ndarray:
    """rotate_half convention (compressor emission, #96 contract)."""
    d = x.shape[-1]
    x0, x1 = x[..., : d // 2], x[..., d // 2 :]
    return np.concatenate([x0 * cos_row - x1 * sin_row, x1 * cos_row + x0 * sin_row], axis=-1)


def _softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# The fp64 whole-step mirror (oracle of record)
# ---------------------------------------------------------------------------


def deepseek_reference_decode_step(
    W: DeepseekV4Weights, st: DeepseekV4DecodeState, cfg: DeepseekV4Config
) -> tuple[np.ndarray, DeepseekV4DecodeState]:
    """One DeepSeek-V4 decode step in fp64 straight-line numpy.

    Returns (logits [B, V], post-step state). Transcribes the landed op
    contracts in program order: per layer — mhc_pre, compressor emission,
    indexer scores + top-k, latent paged append, MLA scores/values +
    conjugate rope + grouped o-proj, mhc compose; then the MoE sublayer;
    finally the flattened-stream head.
    """
    B, C, D, H = cfg.batch, cfg.hidden, cfg.latent_dim, cfg.heads
    L, S = cfg.layers, cfg.cache_capacity
    m, R = cfg.compress_m, cfg.entries_per_series
    M = 2 * R
    HI, K = cfg.index_heads, cfg.index_k

    # pools we mutate (fp64 working copies)
    lat_k = st.latent_k.astype(np.float64).copy()
    lat_v = st.latent_v.astype(np.float64).copy()
    entry = st.entry_pool.astype(np.float64).copy()
    series = st.series_state.copy()

    logits_out = None
    for b in range(B):
        p = int(st.row_positions[b])
        cos_q = np.cos(np.arange(D // 2) * 0.3 + p * 0.1).astype(np.float64)
        sin_q = np.sin(np.arange(D // 2) * 0.3 + p * 0.1).astype(np.float64)

        # embedding → initial stream stack [hc, C]
        streams = W.token_emb[st.ids[b]].astype(np.float64).reshape(cfg.hc, C)

        for l in range(L):
            lw = W.layers[l]
            # ---- mHC pre (attention sublayer) ----
            streams, h_in, post_a, comb_a = _mhc_pre(streams, lw.mhc_attn, cfg)
            # ---- compressor emission (boundary rows only, #96) ----
            if p % m == m - 1:
                e = _compressor_emit(st.comp_window[b], st.comp_gates[b], lw.compressor_rms_w, st.comp_cos[b], st.comp_sin[b], cfg)
                slot_active = int(series[b, l, 0])
                off = int(series[b, l, 1])
                entry[b, l, slot_active, off, :] = e
                nxt = (slot_active * R + off) + 1
                series[b, l, 0] = (nxt // R) % 2
                series[b, l, 1] = nxt % R
            # ---- latent paged append (current token, #94) ----
            n1 = _rms(h_in, lw.attn_ln_gamma, cfg.rms_eps)
            lat = n1 @ lw.kv_a_w.astype(np.float64)  # [D]
            slot = int(st.slot_table_global[b, p])
            lat_k[l, slot, 0, :] = lat
            lat_v[l, slot, 0, :] = lat
            kv_rows = lat_k[l].reshape(B, S, D)[b]  # row-local [S, D]
            # ---- indexer scores over this layer's entries (#97) ----
            entries = entry[b, l].reshape(M, D)  # flat Ca|Cb view
            q_idx = (n1 @ lw.idx_q_w.astype(np.float64)).reshape(HI, D)
            mix = n1 @ lw.idx_mix_w.astype(np.float64)  # [HI]
            hscores = np.maximum(q_idx @ entries.T, 0.0) * (D**-0.5)  # [HI, M]
            idx_scores = hscores.T @ mix  # [M]
            vc = int(st.valid_counts[b])
            # deterministic top-k over the valid prefix (NaN-excluded), ties → lowest j
            cand = [j for j in range(min(vc, M)) if np.isfinite(idx_scores[j])]
            order = sorted(cand, key=lambda j: (-idx_scores[j], j))
            sel = order[:K]
            comp_idx = np.full(K, -1, dtype=np.int64)
            comp_idx[: len(sel)] = sel
            valid_scores = np.array([idx_scores[j] for j in cand]) if cand else np.zeros(1)
            norm = np.linalg.norm(valid_scores)
            block_bias = np.zeros(K)
            for i, j in enumerate(sel):
                block_bias[i] = idx_scores[j] / norm if norm > 0 else 0.0
            # ---- MLA scores + softmax + sink (#95) ----
            q = (n1 @ lw.q_proj_w.astype(np.float64)).reshape(H, D)
            q_rot = _rope_interleaved(q, cos_q, sin_q)
            width = cfg.window + K + 1
            logits = np.full((H, width), -np.inf)
            for i in range(cfg.window):
                t = p - cfg.window + 1 + i
                if 0 <= t <= p:
                    kslot = int(st.slot_table_local[b, t])
                    logits[:, i] = cfg.attention_scale * (q_rot * kv_rows[kslot]).sum(axis=1)
            for j in range(K):
                if comp_idx[j] >= 0:
                    logits[:, cfg.window + j] = (
                        cfg.attention_scale * (q_rot * entries[comp_idx[j]]).sum(axis=1) + block_bias[j]
                    )
            logits[:, -1] = lw.mla_sink  # sink column is ALWAYS valid [H]
            valid = np.isfinite(logits)
            masked = np.where(valid, logits, -np.inf)
            mx = masked.max(axis=1, keepdims=True)  # finite: the sink is always valid
            e = np.where(valid, np.exp(masked - mx), 0.0)
            probs = e / e.sum(axis=1, keepdims=True)  # invalid candidates exact 0.0
            # ---- MLA values: sink contributes NO value (#95) ----
            ctx = np.zeros((H, D))
            for i in range(cfg.window):
                t = p - cfg.window + 1 + i
                if 0 <= t <= p:
                    vslot = int(st.slot_table_local[b, t])
                    ctx += probs[:, i, None] * lat_v[l].reshape(B, S, D)[b][vslot]
            for j in range(K):
                if comp_idx[j] >= 0:
                    ctx += probs[:, cfg.window + j, None] * entries[comp_idx[j]]
            # ---- conjugate rope (inverse of the q rotation) + grouped o-proj ----
            ctx_c = _rope_interleaved_inverse(ctx, cos_q, sin_q)
            flat = ctx_c.reshape(H * D)
            attn_out = _grouped_linear(flat, lw.o_proj_w.astype(np.float64), H)
            # ---- mHC compose (attention sublayer) ----
            streams = _mhc_post(streams, attn_out, (post_a, comb_a), cfg)
            # ---- MoE sublayer: second mHC pre pair on the updated streams ----
            streams, h_in2, post_m, comb_m = _mhc_pre(streams, lw.mhc_moe, cfg)
            n2 = _rms(h_in2, lw.moe_ln_gamma, cfg.rms_eps)
            moe_out = _moe_block(n2, lw, cfg)
            streams = _mhc_post(streams, moe_out, (post_m, comb_m), cfg)
        # ---- head: flattened streams → rms → tied logits ----
        flat = streams.reshape(cfg.hc * C)
        n = _rms(flat, W.final_gamma, cfg.rms_eps)
        row_logits = n @ W.token_emb.astype(np.float64).T
        logits_out = row_logits if logits_out is None else np.vstack([logits_out, row_logits])

    st_out = DeepseekV4DecodeState(
        row_positions=st.row_positions,
        ids=st.ids,
        slot_table_global=st.slot_table_global,
        slot_table_local=st.slot_table_local,
        latent_k=lat_k.astype(np.float32),
        latent_v=lat_v.astype(np.float32),
        entry_pool=entry.astype(np.float32),
        series_state=series,
        comp_window=st.comp_window,
        comp_gates=st.comp_gates,
        comp_cos=st.comp_cos,
        comp_sin=st.comp_sin,
        valid_counts=st.valid_counts,
    )
    return logits_out, st_out


# ---------------------------------------------------------------------------
# Mirror internals (mHC / MoE / compressor) — transcriptions of the landed
# #96/#98/#99 contracts.
# ---------------------------------------------------------------------------


def _sinkhorn(comb: np.ndarray, iters: int, eps: float) -> np.ndarray:
    c = comb / (comb.sum(axis=0, keepdims=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(axis=1, keepdims=True) + eps)
        c = c / (c.sum(axis=0, keepdims=True) + eps)
    return c


def _mhc_pre(streams: np.ndarray, w: MHCWeights, cfg: DeepseekV4Config):
    """#99 mhc_pre in fp64. Returns (streams_unchanged, h_in, post, comb)."""
    hc, C = cfg.hc, cfg.hidden
    flat = streams.reshape(hc * C)
    flat_n = flat / np.sqrt(np.mean(flat * flat) + cfg.mhc_rms_eps)
    lg = w.fn.astype(np.float64) @ flat_n + w.base.astype(np.float64)
    pre_w, post_w, comb_w = lg[:hc], lg[hc : 2 * hc], lg[2 * hc :].reshape(hc, hc)
    pre = 1.0 / (1.0 + np.exp(-(pre_w * w.scale[0]))) + cfg.mhc_eps
    post = 2.0 / (1.0 + np.exp(-(post_w * w.scale[1])))
    comb = _softmax((comb_w * w.scale[2]).reshape(hc, hc)) + cfg.mhc_eps
    comb = _sinkhorn(comb, cfg.mhc_iters, cfg.mhc_eps)
    h_in = (pre[:, None] * streams).sum(axis=0)
    return streams, h_in, post, comb


def _mhc_post(streams: np.ndarray, body_out: np.ndarray, gates, cfg: DeepseekV4Config) -> np.ndarray:
    """#99 mhc_post: streams'[j] = post[j]·body_out + Σ_k comb[k, j]·streams[k]."""
    post, comb = gates
    hc, C = cfg.hc, cfg.hidden
    out = np.zeros((hc, C))
    for j in range(hc):
        out[j] = post[j] * body_out + (comb[:, j, None] * streams).sum(axis=0)
    return out


def _compressor_emit(window, gates, rms_w, cos_row, sin_row, cfg) -> np.ndarray:
    """#96 emission: w=softmax(gates); e=Σ w_t·window_t; rms; rotate_half."""
    g = np.asarray(gates, dtype=np.float64)
    w = _softmax(g)
    e = (w[:, None] * np.asarray(window, dtype=np.float64)).sum(axis=0)
    e = e / np.sqrt(np.mean(e * e) + cfg.compressor_eps) * np.asarray(rms_w, dtype=np.float64)
    return _rotate_half(e, np.asarray(cos_row, dtype=np.float64), np.asarray(sin_row, dtype=np.float64))


def _grouped_linear(x: np.ndarray, w: np.ndarray, H: int) -> np.ndarray:
    """#95 grouped-linear: block-diagonal per-head projection (fp64)."""
    N, K = w.shape[0] // H, w.shape[1] // H
    y = np.zeros(N * H)
    for h in range(H):
        y[h * N : (h + 1) * N] = w[h * N : (h + 1) * N, h * K : (h + 1) * K] @ x[h * K : (h + 1) * K]
    return y


def _moe_block(x: np.ndarray, lw: LayerWeights, cfg: DeepseekV4Config) -> np.ndarray:
    """#98 router (sqrtsoftplus) + indirected expert FFN + combine (+ shared)."""
    E, k, I = cfg.n_experts, cfg.moe_top_k, cfg.moe_intermediate
    lg = x @ lw.router_w.astype(np.float64).T  # [E]
    sp = np.log1p(np.exp(lg))  # softplus
    scores = np.sqrt(sp)
    order = sorted(range(E), key=lambda e: (-scores[e], e))
    sel = order[:k]
    w = scores[sel]
    w = w / (w.sum() + 1e-20) * cfg.routed_scaling_factor
    C = cfg.hidden
    partials = np.zeros((k, C))
    for i, e in enumerate(sel):
        gu = lw.expert_gate_up[e].astype(np.float64) @ x  # [2I]
        g, u = gu[:I], gu[I:]
        act = _silu(g) * u
        partials[i] = lw.expert_down[e].astype(np.float64) @ act
    y = (w[:, None] * partials).sum(axis=0)
    gu = lw.shared_gate_up.astype(np.float64) @ x
    y += lw.shared_down.astype(np.float64) @ (_silu(gu[: cfg.shared_intermediate]) * gu[cfg.shared_intermediate :])
    return y
