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
    cache_capacity: int = 9  # S — per-row latent slots (paged, #94; slot 0 reserved)

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
        if self.cache_capacity <= self.max_positions:
            raise ValueError("cache_capacity must EXCEED max_positions (slot 0 is the reserved null page)")
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
    o_proj_w: np.ndarray  # [H*D, H*(C//H)] block-diagonal, [Cin, Cout] (grouped, #95)
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
    shared_gate_up: np.ndarray  # [C, 2I_s]  ([Cin, Cout] — plain linear convention)
    shared_down: np.ndarray  # [I_s, C]


@dataclass
class DeepseekV4Weights:
    token_emb: np.ndarray  # [V, hc*C] — embedding writes the stream stack
    final_gamma: np.ndarray  # [hc*C]
    rope_cos: np.ndarray  # [max_positions, D/2] — interleaved query tables
    rope_sin: np.ndarray  # [max_positions, D/2]
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
                o_proj_w=_rng_weights(rng, H * D, H * (C // H)),
                idx_q_w=_rng_weights(rng, C, HI * D),
                idx_mix_w=_rng_weights(rng, C, HI) * (HI**-0.5),
                compressor_rms_w=(1.0 + 0.05 * rng.standard_normal(D)).astype(np.float32),
                mla_sink=_rng_weights(rng, H) * 0.1,
                moe_ln_gamma=(1.0 + 0.05 * rng.standard_normal(C)).astype(np.float32),
                router_w=_rng_weights(rng, cfg.n_experts, C),
                expert_gate_up=_rng_weights(rng, cfg.n_experts, 2 * cfg.moe_intermediate, C),
                expert_down=_rng_weights(rng, cfg.n_experts, C, cfg.moe_intermediate),
                shared_gate_up=_rng_weights(rng, C, 2 * cfg.shared_intermediate),
                shared_down=_rng_weights(rng, cfg.shared_intermediate, C),
            )
        )
    idx = np.arange(cfg.max_positions)[:, None] * 0.1 + np.arange(cfg.latent_dim // 2)[None, :] * 0.3
    return DeepseekV4Weights(
        token_emb=_rng_weights(rng, cfg.vocab, cfg.hc * C),
        final_gamma=(1.0 + 0.05 * rng.standard_normal(cfg.hc * C)).astype(np.float32),
        rope_cos=np.cos(idx).astype(np.float32),
        rope_sin=np.sin(idx).astype(np.float32),
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
    row_positions = np.array([5, 7][:B], dtype=np.int32)
    if B > 2:
        row_positions = np.concatenate([row_positions, rng.integers(0, 6, size=B - 2).astype(np.int32)])
    ids = rng.integers(0, cfg.vocab, size=B).astype(np.int32)
    # #94 non-identity per-row slot permutation over the LIVE slots
    # 1..S-1 (slot 0 is the reserved null/sink page, never written)
    local = np.zeros((B, S), dtype=np.int32)
    glob = np.zeros((B, S), dtype=np.int32)
    for b in range(B):
        # positions 0..max_positions-1 map into live slots 1..max_positions;
        # the tail of the table stays 0 (never dereferenced beyond p)
        perm = rng.permutation(S - 1) + 1  # live slots only
        local[b, : S - 1] = perm
        glob[b, : S - 1] = b * S + perm
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
        st.valid_counts[b] = (p + 1) // m  # emissions through this step's boundary (if p % m == m-1)
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
    """One DeepSeek-V4 decode step — the storage-precision oracle of record.

    Per-op fp64 arithmetic with **fp32 rounding at every op boundary**,
    exactly as the reference executor stores each op's output into its
    F32 workspace buffer (there is no F64 tensor dtype in the IR). The
    compiled whole-step graph and this mirror are two readings of the same
    landed op contracts; numerics disagreements localize the bug.

    Returns (logits [B, V], post-step state).
    """
    B, C, D = cfg.batch, cfg.hidden, cfg.latent_dim
    L, S = cfg.layers, cfg.cache_capacity
    m, R = cfg.compress_m, cfg.entries_per_series
    M = 2 * R
    HI, K = cfg.index_heads, cfg.index_k
    H = cfg.heads
    f32 = lambda a: np.asarray(a, dtype=np.float32)  # noqa: E731 — op-boundary store

    lat_k = st.latent_k.copy()  # fp32 pools, mutated in place semantics
    lat_v = st.latent_v.copy()
    entry = st.entry_pool.copy()
    series = st.series_state.copy()

    logits_out = None
    for b in range(B):
        p = int(st.row_positions[b])
        cos_q = W.rope_cos[p].astype(np.float64)  # [D/2] pair-indexed
        sin_q = W.rope_sin[p].astype(np.float64)
        vc = int(st.valid_counts[b])

        # embedding (no positional row) → initial stream stack [hc, C]
        emb = f32(W.token_emb[int(st.ids[b])])
        streams = emb.reshape(cfg.hc, C)

        for l in range(L):
            lw = W.layers[l]
            # ================= attention sublayer =================
            flat = streams.reshape(-1).astype(np.float64)
            flat_n = flat / np.sqrt(np.mean(flat * flat) + cfg.mhc_rms_eps)
            lg = lw.mhc_attn.fn.astype(np.float64) @ flat_n  # no bias on projection
            pre_w, post_w, comb_w = lg[: cfg.hc], lg[cfg.hc : 2 * cfg.hc], lg[2 * cfg.hc :].reshape(cfg.hc, cfg.hc)
            base = lw.mhc_attn.base.astype(np.float64)
            sc = lw.mhc_attn.scale.astype(np.float64)
            pre = 1.0 / (1.0 + np.exp(-(pre_w * sc[0] + base[: cfg.hc]))) + cfg.mhc_eps
            post = 2.0 / (1.0 + np.exp(-(post_w * sc[1] + base[cfg.hc : 2 * cfg.hc])))
            cl = comb_w * sc[2] + base[2 * cfg.hc :].reshape(cfg.hc, cfg.hc)
            cl = cl - cl.max(axis=-1, keepdims=True)
            comb = np.exp(cl) / np.exp(cl).sum(axis=-1, keepdims=True) + cfg.mhc_eps
            comb = comb / (comb.sum(axis=-2, keepdims=True) + cfg.mhc_eps)
            for _ in range(cfg.mhc_iters - 1):
                comb = comb / (comb.sum(axis=-1, keepdims=True) + cfg.mhc_eps)
                comb = comb / (comb.sum(axis=-2, keepdims=True) + cfg.mhc_eps)
            h_in = f32((pre[:, None] * streams.astype(np.float64)).sum(axis=0))
            post_a, comb_a = f32(post), f32(comb)

            n1 = f32(
                h_in.astype(np.float64)
                / np.sqrt(np.mean(h_in.astype(np.float64) ** 2) + cfg.rms_eps)
                * lw.attn_ln_gamma.astype(np.float64)
            )

            # --- DSA compressor emission (#96): boundary rows only ---
            if p % m == m - 1:
                g = st.comp_gates[b].astype(np.float64)
                ex = np.exp(g - g.max())
                wgt = ex / ex.sum()
                e = (wgt[:, None] * st.comp_window[b].astype(np.float64)).sum(axis=0)
                e = e * np.reciprocal(np.sqrt((e * e).mean() + cfg.compressor_eps)) * lw.compressor_rms_w.astype(np.float64)
                ch, sh = st.comp_cos[b].astype(np.float64), st.comp_sin[b].astype(np.float64)
                half = e.shape[0] // 2
                e_rot = np.concatenate([e[:half] * ch - e[half:] * sh, e[half:] * ch + e[:half] * sh])
                slot, cb = int(series[b, l, 0]), int(series[b, l, 1])
                entry[b, l, slot, cb, :] = f32(e_rot)
                cb += 1
                if cb == R:
                    series[b, l, 0], series[b, l, 1] = 1 - slot, 0
                else:
                    series[b, l, 1] = cb
            entries = entry[b, l].reshape(M, D)  # flat slot-major view

            # --- paged latent append (#94): the current token's latent row ---
            lat = f32(n1.astype(np.float64) @ lw.kv_a_w.astype(np.float64))  # [D]
            slot_p = int(st.slot_table_global[b, p])
            lat_k[l, slot_p, 0, :] = lat
            lat_v[l, slot_p, 0, :] = lat
            kv_rows_k = lat_k[l].reshape(B, S, D)[b]  # row-local [S, D]
            kv_rows_v = lat_v[l].reshape(B, S, D)[b]

            # --- Lightning indexer (#97) ---
            q_idx = f32(n1.astype(np.float64) @ lw.idx_q_w.astype(np.float64)).reshape(HI, D)
            mix = f32(n1.astype(np.float64) @ lw.idx_mix_w.astype(np.float64))  # [HI]
            hs = np.maximum(q_idx.astype(np.float64) @ entries.astype(np.float64).T, 0.0) * (D**-0.5)
            idx_scores = f32((hs * mix.astype(np.float64)[:, None]).sum(axis=0))  # [M]
            # rank-by-comparison top-k (descending, ties → lowest j), NaN excluded
            row = idx_scores.astype(np.float64)
            cand = np.array([j for j in range(min(vc, M)) if np.isfinite(row[j])], dtype=int)
            rank = {j: i for i, j in enumerate(sorted(cand, key=lambda j: (-row[j], j)))}
            comp_idx = np.full(K, -1, dtype=np.int32)
            block_bias = np.zeros(K, dtype=np.float32)
            valid_fin = row[[j for j in range(min(vc, M)) if np.isfinite(row[j])]] if vc > 0 else np.zeros(0)
            norm = np.sqrt((valid_fin**2).sum()) if valid_fin.size else 0.0
            for j, rk in rank.items():
                if rk < K:
                    comp_idx[rk] = j
                    if norm > 0:
                        block_bias[rk] = f32(row[j] / norm)
            sel_bias = block_bias.astype(np.float64)

            # --- MLA (#95) ---
            q = f32(n1.astype(np.float64) @ lw.q_proj_w.astype(np.float64)).reshape(H, D)
            q_rot = np.empty_like(q, dtype=np.float64)
            for h in range(H):
                xe, xo = q[h][0::2], q[h][1::2]
                q_rot[h][0::2] = xe * cos_q - xo * sin_q
                q_rot[h][1::2] = xo * cos_q + xe * sin_q
            q_rot = f32(q_rot)
            width = cfg.window + K + 1
            logits = np.full((H, width), -np.inf, dtype=np.float64)
            t_lo = max(0, p - cfg.window + 1)
            for i in range(cfg.window):
                t = p - cfg.window + 1 + i
                if t_lo <= t <= p:
                    krow = kv_rows_k[int(st.slot_table_local[b, t])].astype(np.float64)
                    logits[:, i] = (q_rot.astype(np.float64) * krow).sum(axis=1) * cfg.attention_scale
            for j in range(K):
                if comp_idx[j] >= 0:
                    logits[:, cfg.window + j] = (
                        (q_rot.astype(np.float64) * entries[comp_idx[j]].astype(np.float64)).sum(axis=1)
                        * cfg.attention_scale
                        + sel_bias[j]
                    )
            logits[:, -1] = lw.mla_sink.astype(np.float64)  # sink ALWAYS valid
            mx = logits.max(axis=1, keepdims=True)
            e = np.exp(logits - mx)
            e[~np.isfinite(logits)] = 0.0  # invalid slots exact zero
            probs = f32(e / e.sum(axis=1, keepdims=True))

            ctx = np.zeros((H, D), dtype=np.float64)
            for i in range(cfg.window):
                t = p - cfg.window + 1 + i
                if t_lo <= t <= p:
                    vrow = kv_rows_v[int(st.slot_table_local[b, t])].astype(np.float64)
                    ctx += probs[:, i].astype(np.float64)[:, None] * vrow
            for j in range(K):
                if comp_idx[j] >= 0:
                    ctx += probs[:, cfg.window + j].astype(np.float64)[:, None] * entries[comp_idx[j]].astype(np.float64)
            ctx = f32(ctx)
            # conjugate rope: sin NEGATED (inverse of the q rotation)
            ctx_c = np.empty_like(ctx)
            for h in range(H):
                xe, xo = ctx[h][0::2], ctx[h][1::2]
                ctx_c[h][0::2] = xe * cos_q + xo * sin_q
                ctx_c[h][1::2] = xo * cos_q - xe * sin_q
            ctx_c = f32(ctx_c)
            # grouped o-proj: block-diagonal [Cin, Cout] = [H*D, H*(C//H)]
            xf = ctx_c.reshape(H * D).astype(np.float64)
            ow = lw.o_proj_w.astype(np.float64)
            Kg, Ng = ow.shape[0] // H, ow.shape[1] // H
            attn = np.zeros(H * Ng, dtype=np.float64)
            for h in range(H):
                attn[h * Ng : (h + 1) * Ng] = xf[h * Kg : (h + 1) * Kg] @ ow[h * Kg : (h + 1) * Kg, h * Ng : (h + 1) * Ng]
            attn = f32(attn)
            # mhc compose (attention sublayer)
            streams = np.empty_like(streams)
            for j in range(cfg.hc):
                acc = (comb_a[:, j].astype(np.float64)[:, None] * streams.astype(np.float64)).sum(axis=0)
                acc = acc + float(post_a[j]) * attn.astype(np.float64)
                streams[j] = f32(acc)

            # ================= MoE sublayer =================
            flat = streams.reshape(-1).astype(np.float64)
            flat_n = flat / np.sqrt(np.mean(flat * flat) + cfg.mhc_rms_eps)
            lg = lw.mhc_moe.fn.astype(np.float64) @ flat_n
            pre_w, post_w, comb_w = lg[: cfg.hc], lg[cfg.hc : 2 * cfg.hc], lg[2 * cfg.hc :].reshape(cfg.hc, cfg.hc)
            base = lw.mhc_moe.base.astype(np.float64)
            sc = lw.mhc_moe.scale.astype(np.float64)
            pre = 1.0 / (1.0 + np.exp(-(pre_w * sc[0] + base[: cfg.hc]))) + cfg.mhc_eps
            post = 2.0 / (1.0 + np.exp(-(post_w * sc[1] + base[cfg.hc : 2 * cfg.hc])))
            cl = comb_w * sc[2] + base[2 * cfg.hc :].reshape(cfg.hc, cfg.hc)
            cl = cl - cl.max(axis=-1, keepdims=True)
            comb = np.exp(cl) / np.exp(cl).sum(axis=-1, keepdims=True) + cfg.mhc_eps
            comb = comb / (comb.sum(axis=-2, keepdims=True) + cfg.mhc_eps)
            for _ in range(cfg.mhc_iters - 1):
                comb = comb / (comb.sum(axis=-1, keepdims=True) + cfg.mhc_eps)
                comb = comb / (comb.sum(axis=-2, keepdims=True) + cfg.mhc_eps)
            h_in2 = f32((pre[:, None] * streams.astype(np.float64)).sum(axis=0))
            post_m, comb_m = f32(post), f32(comb)

            n2 = f32(
                h_in2.astype(np.float64)
                / np.sqrt(np.mean(h_in2.astype(np.float64) ** 2) + cfg.rms_eps)
                * lw.moe_ln_gamma.astype(np.float64)
            )
            # router: sqrt(softplus(l)) stable, global top-k, unconditional renorm
            lgr = lw.router_w.astype(np.float64) @ n2.astype(np.float64)
            scores = np.sqrt(np.logaddexp(0.0, lgr))
            order = np.argsort(-scores, kind="stable")
            sel = order[: cfg.moe_top_k]
            wsel = scores[sel]
            wsel = wsel / (wsel.sum() + 1e-20) * cfg.routed_scaling_factor
            partials = np.zeros((cfg.moe_top_k, C), dtype=np.float64)
            for i, e_idx in enumerate(sel):
                gu = lw.expert_gate_up[e_idx].astype(np.float64) @ n2.astype(np.float64)
                g, u = gu[: cfg.moe_intermediate], gu[cfg.moe_intermediate :]
                act = g / (1.0 + np.exp(-g)) * u
                partials[i] = lw.expert_down[e_idx].astype(np.float64) @ act
            partials = f32(partials)
            r_w = f32(wsel)
            acc = np.zeros(C, dtype=np.float64)
            for i in range(cfg.moe_top_k):  # slot order
                acc += r_w[i].astype(np.float64) * partials[i].astype(np.float64)
            sgu = f32(n2.astype(np.float64) @ lw.shared_gate_up.astype(np.float64))
            sg = sgu[: cfg.shared_intermediate].astype(np.float64)
            su = sgu[cfg.shared_intermediate :].astype(np.float64)
            shared = f32((sg / (1.0 + np.exp(-sg)) * su) @ lw.shared_down.astype(np.float64))
            moe = f32(acc + shared.astype(np.float64))
            # mhc compose (MoE sublayer)
            new_streams = np.empty_like(streams)
            for j in range(cfg.hc):
                accc = (comb_m[:, j].astype(np.float64)[:, None] * streams.astype(np.float64)).sum(axis=0)
                accc = accc + float(post_m[j]) * moe.astype(np.float64)
                new_streams[j] = f32(accc)
            streams = new_streams

        # ---- head: flattened streams → RMSNorm → tied logits ----
        flat = streams.reshape(-1).astype(np.float64)
        n = f32(flat / np.sqrt(np.mean(flat * flat) + cfg.rms_eps) * W.final_gamma.astype(np.float64))
        row_logits = f32(n.astype(np.float64) @ W.token_emb.astype(np.float64).T)
        logits_out = row_logits if logits_out is None else np.vstack([logits_out, row_logits])

    st_out = DeepseekV4DecodeState(
        row_positions=st.row_positions,
        ids=st.ids,
        slot_table_global=st.slot_table_global,
        slot_table_local=st.slot_table_local,
        latent_k=lat_k,
        latent_v=lat_v,
        entry_pool=entry,
        series_state=series,
        comp_window=st.comp_window,
        comp_gates=st.comp_gates,
        comp_cos=st.comp_cos,
        comp_sin=st.comp_sin,
        valid_counts=st.valid_counts,
    )
    return logits_out, st_out
