"""Qwen3.5 hybrid model definition for the megakernel compiler (issue #102 Stage 1).

Tiny-but-faithful Qwen3.5-family hybrid decode config: GatedDeltaNet (GDN)
layers interleaved with gated full-attention layers, partial NeoX RoPE
(``rotary_dim``), fp8-blockwise projections (#91), per-row ragged decode
positions (#93) and paged KV pools with slot-table indirection (#94).

The module owns the **whole-decode-step fp64 mirror** — the bare-env oracle
of record for the compiled graph (the floe eager ``qwen35`` forward is the
§15.1 oracle and is exercised in the import-gated test).

Layer-type split used by the tiny fixture: ``("gdn", "fa", "gdn", "fa")``.
FA layers alternate KV plumbing to exercise both landed patterns: the first
FA layer uses **paged** pools + slot tables (#94, plain
``attention_values_paged``), the second uses a **dense** cache with the
fused sigmoid output gate (#92, ``attention_values(..., gate=...)``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reference_exec import decode_e4m3

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Qwen35Config:
    """Hybrid Qwen3.5 dimensions (these specialize compilation, §13.3)."""

    vocab: int
    layers: int
    layer_types: tuple[str, ...]  # "gdn" | "fa" per layer
    hidden: int  # C
    # GDN (GatedDeltaNet) dims
    gdn_k_heads: int  # NK
    gdn_v_heads: int  # NV
    head_k_dim: int  # HK
    head_v_dim: int  # HV
    conv_kernel: int  # K (FIR taps; state keeps K-1)
    # full-attention dims
    heads: int  # H (query heads)
    kv_heads: int  # KVH (GQA)
    head_dim: int  # D
    rotary_dim: int  # partial NeoX rope width (even, <= D)
    # MLP
    intermediate: int  # F (SwiGLU)
    # decode-step plumbing
    batch: int = 2  # 2 rows -> ragged positions exercise #93
    cache_capacity: int = 16  # tokens per row
    max_positions: int = 32
    rope_theta: float = 10_000.0
    rms_eps: float = 1e-6
    delta_scale: float = 1.0  # q_n scale in the delta rule
    delta_eps: float = 1e-6  # l2-norm eps inside the delta rule
    fp8_quant_block: int = 32  # tiny-config block (real models: 128)

    @property
    def gdn_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "gdn")

    @property
    def fa_layers(self) -> tuple[int, ...]:
        return tuple(i for i, t in enumerate(self.layer_types) if t == "fa")

    @property
    def gdn_kv_dim(self) -> int:
        """Width of the packed post-conv row: q|k|v|z = 2*NK*HK + 2*NV*HV."""
        return 2 * self.gdn_k_heads * self.head_k_dim + 2 * self.gdn_v_heads * self.head_v_dim

    @property
    def fa_proj_dim(self) -> int:
        """Width of the chunked q|gate|k|v projection: (2H + 2KVH) * D."""
        return (2 * self.heads + 2 * self.kv_heads) * self.head_dim

    @property
    def n_slots(self) -> int:
        """Paged pool slots: slot 0 reserved (null/sink), then B rows x capacity."""
        return 1 + self.batch * self.cache_capacity

    @property
    def attention_scale(self) -> float:
        return 1.0 / np.sqrt(self.head_dim)

    def validate(self) -> None:
        assert self.layers == len(self.layer_types), "layer_types must cover every layer"
        assert self.heads % self.kv_heads == 0, "GQA: H must be a multiple of KVH"
        assert self.rotary_dim % 2 == 0 and 0 < self.rotary_dim <= self.head_dim
        assert self.intermediate % 2 == 0, "fused gate-up splits [B, 2F]"
        assert set(self.layer_types) <= {"gdn", "fa"}
        assert self.gdn_layers and self.fa_layers, "hybrid fixture needs both layer kinds"


def tiny_qwen35_config(*, batch: int = 2) -> Qwen35Config:
    """Small-but-faithful hybrid fixture: GDN+FA mix, partial rope, fp8, ragged rows."""
    cfg = Qwen35Config(
        vocab=256,
        layers=4,
        layer_types=("gdn", "fa", "gdn", "fa"),
        hidden=64,
        gdn_k_heads=2,
        gdn_v_heads=4,
        head_k_dim=8,
        head_v_dim=8,
        conv_kernel=4,
        heads=4,
        kv_heads=2,
        head_dim=16,
        rotary_dim=8,
        intermediate=64,
        batch=batch,
        cache_capacity=16,
        max_positions=32,
    )
    cfg.validate()
    return cfg


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


@dataclass
class GDNLayerWeights:
    ln1_gamma: np.ndarray  # [C]
    in_qkvz_w: np.ndarray  # fp8-projected: [C_conv, C] fp64 reference (dequant target)
    in_a_w: np.ndarray  # [C, NV]
    in_b_w: np.ndarray  # [C, NV]
    conv_w: np.ndarray  # [C_conv, K]
    A_log: np.ndarray  # [NV]
    dt_bias: np.ndarray  # [NV]
    norm_w: np.ndarray  # [HV]
    ln2_gamma: np.ndarray  # [C]
    gate_up_w: np.ndarray  # [C, 2F]
    down_w: np.ndarray  # [F, C]
    o_proj_w: np.ndarray  # [NV*HV, C]
    in_qkvz_bytes: np.ndarray = None  # [C_conv, C] uint8 e4m3 (set by quantize)
    in_qkvz_scale: np.ndarray = None  # [ceil(C_conv/qb), C/qb] fp32


@dataclass
class FALayerWeights:
    ln1_gamma: np.ndarray  # [C]
    qkv_gate_w: np.ndarray  # fp8-projected: [(2H+2KVH)*D, C] fp64 reference
    q_norm_gamma: np.ndarray  # [D]
    k_norm_gamma: np.ndarray  # [D]
    o_proj_w: np.ndarray  # [H*D, C]
    ln2_gamma: np.ndarray  # [C]
    gate_up_w: np.ndarray  # [C, 2F]
    down_w: np.ndarray  # [F, C]
    qkv_gate_bytes: np.ndarray = None
    qkv_gate_scale: np.ndarray = None


@dataclass
class Qwen35Weights:
    token_emb: np.ndarray  # [V, C]
    final_gamma: np.ndarray  # [C]
    layers: list  # GDNLayerWeights | FALayerWeights per config.layer_types
    rope_cos: np.ndarray = None  # [max_positions, RD//2]
    rope_sin: np.ndarray = None


def rope_tables_neox(config: Qwen35Config) -> tuple[np.ndarray, np.ndarray]:
    """Partial-NeoX tables [max_positions, rotary_dim // 2] (fp64)."""
    half = config.rotary_dim // 2
    inv_freq = config.rope_theta ** (-2.0 * np.arange(half) / config.rotary_dim)
    pos = np.arange(config.max_positions, dtype=np.float64)
    angles = np.outer(pos, inv_freq)  # [P, half]
    return np.cos(angles), np.sin(angles)


def _rng_weights(rng, *shape):
    return rng.standard_normal(shape) * 0.08


def quantize_fp8(w: np.ndarray, quant_block: int) -> tuple[np.ndarray, np.ndarray]:
    """DeepSeek-style block-FP8: per-block absmax scale -> e4m3 bytes.

    Returns (bytes_u8 [N, K], scales fp32 [ceil(N/qb), K/qb]). K must be a
    multiple of the block (tiny fixture guarantees it).
    """
    n, k = w.shape
    assert k % quant_block == 0, f"K={k} must be a multiple of quant_block={quant_block}"
    nb, kb = -(-n // quant_block), k // quant_block
    pos = decode_e4m3(np.arange(0, 0x7F, dtype=np.uint8))
    w_bytes = np.zeros((n, k), dtype=np.uint8)
    scale = np.zeros((nb, kb), dtype=np.float32)
    for b in range(kb):
        for rb in range(nb):
            rows = slice(rb * quant_block, min((rb + 1) * quant_block, n))
            blk = w[rows, b * quant_block : (b + 1) * quant_block]
            s = max(np.abs(blk).max() / 448.0, 1e-12)
            scale[rb, b] = np.float32(s)
            q = blk / s
            hi = np.clip(np.searchsorted(pos, np.abs(q)), 0, len(pos) - 1)
            lo = np.clip(hi - 1, 0, len(pos) - 1)
            take_hi = (np.abs(q) - pos[lo]) > (pos[hi] - np.abs(q))
            val = np.where(take_hi, pos[hi], pos[lo])
            byte = np.where(np.signbit(q), 0x80 | np.searchsorted(pos, val), np.searchsorted(pos, val))
            w_bytes[rows, b * quant_block : (b + 1) * quant_block] = byte.astype(np.uint8)
    return w_bytes, scale


def dequant_fp8(w_bytes: np.ndarray, scale: np.ndarray, quant_block: int) -> np.ndarray:
    """Mirror-side dequant: y-target weights [N, K] fp64 from bytes + block scales."""
    n, k = w_bytes.shape
    out = np.empty((n, k), dtype=np.float64)
    for b in range(k // quant_block):
        for rb in range(-(-n // quant_block)):
            rows = slice(rb * quant_block, min((rb + 1) * quant_block, n))
            out[rows, b * quant_block : (b + 1) * quant_block] = (
                decode_e4m3(w_bytes[rows, b * quant_block : (b + 1) * quant_block]).astype(np.float64)
                * float(scale[rb, b])
            )
    return out


def random_qwen35_weights(config: Qwen35Config, seed: int = 0) -> Qwen35Weights:
    """Random tiny weights; fp8 projections quantized against the same seed."""
    rng = np.random.default_rng(seed)
    C, F = config.hidden, config.intermediate
    qb = config.fp8_quant_block
    layers = []
    for lt in config.layer_types:
        if lt == "gdn":
            lw = GDNLayerWeights(
                ln1_gamma=1.0 + 0.1 * rng.standard_normal(C),
                in_qkvz_w=_rng_weights(rng, config.gdn_kv_dim, C),
                in_a_w=_rng_weights(rng, C, config.gdn_v_heads),
                in_b_w=_rng_weights(rng, C, config.gdn_v_heads),
                conv_w=_rng_weights(rng, config.gdn_kv_dim, config.conv_kernel),
                A_log=rng.standard_normal(config.gdn_v_heads) * 0.3,
                dt_bias=rng.standard_normal(config.gdn_v_heads) * 0.3,
                norm_w=1.0 + 0.1 * rng.standard_normal(config.head_v_dim),
                ln2_gamma=1.0 + 0.1 * rng.standard_normal(C),
                gate_up_w=_rng_weights(rng, C, 2 * F),
                down_w=_rng_weights(rng, F, C),
                o_proj_w=_rng_weights(rng, config.gdn_v_heads * config.head_v_dim, C),
            )
            lw.in_qkvz_bytes, lw.in_qkvz_scale = quantize_fp8(lw.in_qkvz_w, qb)
            layers.append(lw)
        else:
            fw = FALayerWeights(
                ln1_gamma=1.0 + 0.1 * rng.standard_normal(C),
                qkv_gate_w=_rng_weights(rng, config.fa_proj_dim, C),
                q_norm_gamma=1.0 + 0.1 * rng.standard_normal(config.head_dim),
                k_norm_gamma=1.0 + 0.1 * rng.standard_normal(config.head_dim),
                o_proj_w=_rng_weights(rng, config.heads * config.head_dim, C),
                ln2_gamma=1.0 + 0.1 * rng.standard_normal(C),
                gate_up_w=_rng_weights(rng, C, 2 * F),
                down_w=_rng_weights(rng, F, C),
            )
            fw.qkv_gate_bytes, fw.qkv_gate_scale = quantize_fp8(fw.qkv_gate_w, qb)
            layers.append(fw)
    cos, sin = rope_tables_neox(config)
    return Qwen35Weights(
        token_emb=_rng_weights(rng, config.vocab, C),
        final_gamma=1.0 + 0.1 * rng.standard_normal(C),
        layers=layers,
        rope_cos=cos,
        rope_sin=sin,
    )


# ---------------------------------------------------------------------------
# Decode-step state (pools + routing plumbing)
# ---------------------------------------------------------------------------


@dataclass
class Qwen35DecodeState:
    """External persistent pools for one decode step (caller-owned, §4.3)."""

    conv_state: np.ndarray  # [n_gdn, B, K-1, C_conv] fp64 (device contract: fp32)
    ssm_state: np.ndarray  # [n_gdn, B, NV, HV, HK] fp64 (device contract: fp32)
    k_pool: np.ndarray  # [n_fa, slots, KVH, D]
    v_pool: np.ndarray  # [n_fa, slots, KVH, D]
    k_cache_dense: np.ndarray  # [n_fa_dense, B, KVH, cap, D] (gated FA layers)
    v_cache_dense: np.ndarray  # same
    slot_table: np.ndarray  # [B, cap] i32 — slot per (row, token position)
    row_positions: np.ndarray  # [B] i32 — per-row decode position (#93)
    ids: np.ndarray  # [B] i32

    def copy(self) -> "Qwen35DecodeState":
        import copy as _copy

        return _copy.deepcopy(self)


def initial_state(config: Qwen35Config, seed: int = 1) -> Qwen35DecodeState:
    """Fresh pools: zeros where legal, NaN canaries where reads are forbidden.

    Ragged rows: row 0 at position 5, row 1 at position 2 (both < capacity).
    Paged pools carry NaN in every slot that is not the null slot and not
    owned by a live row slot — a read of those through the slot table must
    surface as NaN in the output (canary), never silently as stale data.
    """
    B, cap = config.batch, config.cache_capacity
    rng = np.random.default_rng(1000 + seed)
    pos = np.array([min(5, cap - 1), min(2, cap - 1)], dtype=np.int32)[:B]
    if B > 2:  # deterministic extension for larger batches
        pos = np.array([min(2 * b + 1, cap - 1) for b in range(B)], dtype=np.int32)
    slot_table = np.zeros((B, cap), dtype=np.int32)
    for b in range(B):
        # Non-identity per-row permutation (rotate by 3): the compiled graph
        # must route every gather/append through the table — an identity
        # table would silently pass a slot==position aliasing bug.
        for t in range(cap):
            slot_table[b, t] = 1 + b * cap + ((t + 3) % cap)
    n_gdn, n_fa = len(config.gdn_layers), len(config.fa_layers)
    conv = np.zeros((n_gdn, B, config.conv_kernel - 1, config.gdn_kv_dim))
    ssm = np.zeros((n_gdn, B, config.gdn_v_heads, config.head_v_dim, config.head_k_dim))
    nan = float("nan")
    k_pool = np.full((n_fa, config.n_slots, config.kv_heads, config.head_dim), nan)
    v_pool = np.full_like(k_pool, nan)
    k_dense = np.full((n_fa, B, config.kv_heads, cap, config.head_dim), nan)
    v_dense = np.full_like(k_dense, nan)
    # Prior-step K/V: positions [0, pos[b]) are legal history (random, seeded);
    # position pos[b] is written by THIS step; everything beyond stays NaN —
    # an over-gather (contract violation) leaks NaN into the output canary.
    for b in range(B):
        pb = int(pos[b])
        if pb > 0:
            for t in range(pb):
                slot = slot_table[b, t]
                k_pool[0, slot] = rng.standard_normal(k_pool.shape[2:]) * 0.1
                v_pool[0, slot] = rng.standard_normal(v_pool.shape[2:]) * 0.1
            k_dense[0, b, :, :pb, :] = rng.standard_normal((k_dense.shape[2], pb, k_dense.shape[4])) * 0.1
            v_dense[0, b, :, :pb, :] = rng.standard_normal((v_dense.shape[2], pb, v_dense.shape[4])) * 0.1
    ids = np.array([7, 3][:B] + [ (11 * (b + 1)) % config.vocab for b in range(2, B)], dtype=np.int32)
    return Qwen35DecodeState(conv, ssm, k_pool, v_pool, k_dense, v_dense, slot_table, pos, ids)


# ---------------------------------------------------------------------------
# The fp64 whole-decode-step mirror (oracle of record, bare env)
# ---------------------------------------------------------------------------


def _rms(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps) * gamma


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _rope_neox(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, pos, rotary_dim: int) -> np.ndarray:
    """NeoX split-half over the first rotary_dim dims; per-row positions.

    x: [B, H, D]; tables [P, RD//2]; pos: [B] ints. x1' = x1*c - x2*s;
    x2' = x2*c + x1*s; dims [RD, D) pass through.
    """
    B, H, D = x.shape
    half = rotary_dim // 2
    out = x.copy()
    for b in range(B):
        c, s = cos[pos[b]], sin[pos[b]]  # [half]
        x1 = x[b, :, :half]  # [H, half]
        x2 = x[b, :, half:rotary_dim]
        out[b, :, :half] = x1 * c - x2 * s
        out[b, :, half:rotary_dim] = x2 * c + x1 * s
    return out


def _fp8_linear(x: np.ndarray, w_bytes: np.ndarray, scale: np.ndarray, qb: int) -> np.ndarray:
    """y = x @ dequant(w, scale)^T (fp64; the executor decodes e4m3 in-walk)."""
    w = dequant_fp8(w_bytes, scale, qb)
    return x @ w.T


def qwen35_reference_decode_step(
    weights: Qwen35Weights,
    state: Qwen35DecodeState,
    config: Qwen35Config,
) -> tuple[np.ndarray, Qwen35DecodeState]:
    """Straight-line NumPy whole-decode-step mirror; mutates pool copies.

    Returns (logits [B, V], post-step state). Every formula mirrors the
    corresponding op's ``numerical_contract`` verbatim; the layer-type
    dispatch and view layout mirror ``build_qwen35_forward`` exactly.
    """
    st = state.copy()
    cfg = config
    B = cfg.batch
    NK, HK, NV, HV = cfg.gdn_k_heads, cfg.head_k_dim, cfg.gdn_v_heads, cfg.head_v_dim
    H, KVH, D, RD = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.rotary_dim
    F = cfg.intermediate
    qb = cfg.fp8_quant_block

    x = weights.token_emb[st.ids].astype(np.float64)  # [B, C]
    gdn_i = 0
    fa_paged_i = 0
    fa_dense_i = 0

    for li, lt in enumerate(cfg.layer_types):
        lw = weights.layers[li]
        if lt == "gdn":
            # --- projections (normed hidden feeds all three) ---
            n1 = _rms(x, lw.ln1_gamma, cfg.rms_eps)
            qkvz_row = _fp8_linear(n1, lw.in_qkvz_bytes, lw.in_qkvz_scale, qb)  # [B, C_conv]
            a = n1 @ lw.in_a_w  # [B, NV]
            b = n1 @ lw.in_b_w  # [B, NV]
            # --- short conv + state shift (gdn_conv contract) ---
            sv = st.conv_state[gdn_i]  # [B, K-1, C_conv]
            full = np.concatenate([sv, qkvz_row[:, None, :]], axis=1)  # [B, K, C_conv]
            fir = np.einsum("ck,bkc->bc", lw.conv_w.astype(np.float64), full)  # out[b,c] = sum_k w[c,k]*full[b,k,c]
            conv_out = _silu(fir)
            st.conv_state[gdn_i] = full[:, 1:]  # time-major shift
            # --- views: q|k|v|z ---
            q = conv_out[:, : NK * HK].reshape(B, NK, HK)
            k = conv_out[:, NK * HK : 2 * NK * HK].reshape(B, NK, HK)
            v = conv_out[:, 2 * NK * HK : 2 * NK * HK + NV * HV].reshape(B, NV, HV)
            z = conv_out[:, 2 * NK * HK + NV * HV :].reshape(B, NV, HV)
            # --- gated delta rule (gdn_delta contract) ---
            GROUP = NV // NK
            s = st.ssm_state[gdn_i].copy()  # [B, NV, HV, HK]
            out = np.empty((B, NV, HV))
            for hh in range(NV):
                kh = hh // GROUP
                g = -np.exp(lw.A_log[hh]) * _softplus(a[:, hh] + lw.dt_bias[hh])  # [B]
                beta = _sigmoid(b[:, hh])  # [B]
                qn = q[:, kh] / np.sqrt(np.sum(q[:, kh] ** 2, axis=-1, keepdims=True) + cfg.delta_eps) * cfg.delta_scale
                kn = k[:, kh] / np.sqrt(np.sum(k[:, kh] ** 2, axis=-1, keepdims=True) + cfg.delta_eps)
                for bb in range(B):
                    s_b = s[bb, hh] * np.exp(g[bb])  # decay [HV, HK]
                    sk = s_b @ kn[bb]  # [HV]
                    s_b = s_b + beta[bb] * np.outer(v[bb, hh] - sk, kn[bb])
                    o = s_b @ qn[bb]  # [HV]
                    s[bb, hh] = s_b
                    out[bb, hh] = o
            st.ssm_state[gdn_i] = s
            o_norm = out / np.sqrt(np.mean(out**2, axis=-1, keepdims=True) + cfg.rms_eps) * lw.norm_w
            gated = o_norm * (z * _sigmoid(z))
            proj = gated.reshape(B, NV * HV) @ lw.o_proj_w  # [B, C]
            x = x + proj
            # --- SwiGLU MLP ---
            n2 = _rms(x, lw.ln2_gamma, cfg.rms_eps)
            gu = n2 @ lw.gate_up_w  # [B, 2F]
            gate, up = gu[:, :F], gu[:, F:]
            act = _silu(gate) * up
            x = x + act @ lw.down_w
            gdn_i += 1
        else:
            # --- projections: chunked q|gate|k|v (fp8) ---
            n1 = _rms(x, lw.ln1_gamma, cfg.rms_eps)
            qkv_gate = _fp8_linear(n1, lw.qkv_gate_bytes, lw.qkv_gate_scale, qb)  # [B, fa_proj]
            q = qkv_gate[:, : H * D].reshape(B, H, D)
            gate = qkv_gate[:, H * D : 2 * H * D].reshape(B, H, D)
            k = qkv_gate[:, 2 * H * D : (2 * H + KVH) * D].reshape(B, KVH, D)
            v = qkv_gate[:, (2 * H + KVH) * D :].reshape(B, KVH, D)
            q = _rms(q, lw.q_norm_gamma, cfg.rms_eps)
            k = _rms(k, lw.k_norm_gamma, cfg.rms_eps)
            q = _rope_neox(q, weights.rope_cos, weights.rope_sin, st.row_positions, RD)
            k = _rope_neox(k, weights.rope_cos, weights.rope_sin, st.row_positions, RD)
            paged = li == cfg.fa_layers[0]  # first FA layer: paged plumbing (#94)
            if paged:
                for bb in range(B):
                    slot = st.slot_table[bb, st.row_positions[bb]]
                    st.k_pool[fa_paged_i, slot, :, :] = k[bb]
                    st.v_pool[fa_paged_i, slot, :, :] = v[bb]
                ctx = np.zeros((B, H, D))
                for bb in range(B):
                    pb = st.row_positions[bb]
                    for h in range(H):
                        kvh = h // (H // KVH)
                        s_row = np.empty(pb + 1)
                        for t in range(pb + 1):
                            slot = st.slot_table[bb, t]
                            s_row[t] = np.dot(q[bb, h], st.k_pool[fa_paged_i, slot, kvh]) * cfg.attention_scale
                        s_row -= s_row.max()
                        e = np.exp(s_row)
                        probs = e / e.sum()
                        acc = np.zeros(D)
                        for t in range(pb + 1):
                            slot = st.slot_table[bb, t]
                            acc += probs[t] * st.v_pool[fa_paged_i, slot, kvh]
                        ctx[bb, h] = acc
                fa_paged_i += 1
            else:
                # dense cache + fused sigmoid output gate (#92)
                for bb in range(B):
                    st.k_cache_dense[fa_dense_i, bb, :, st.row_positions[bb], :] = k[bb]
                    st.v_cache_dense[fa_dense_i, bb, :, st.row_positions[bb], :] = v[bb]
                ctx = np.zeros((B, H, D))
                for bb in range(B):
                    pb = st.row_positions[bb]
                    for h in range(H):
                        kvh = h // (H // KVH)
                        s_row = (
                            st.k_cache_dense[fa_dense_i, bb, kvh, : pb + 1, :] @ q[bb, h]
                        ) * cfg.attention_scale
                        s_row -= s_row.max()
                        e = np.exp(s_row)
                        probs = e / e.sum()
                        acc = probs @ st.v_cache_dense[fa_dense_i, bb, kvh, : pb + 1, :]
                        ctx[bb, h] = acc * _sigmoid(gate[bb, h])
                fa_dense_i += 1
            proj = ctx.reshape(B, H * D) @ lw.o_proj_w
            x = x + proj
            n2 = _rms(x, lw.ln2_gamma, cfg.rms_eps)
            gu = n2 @ lw.gate_up_w
            gate2, up = gu[:, :F], gu[:, F:]
            act = _silu(gate2) * up
            x = x + act @ lw.down_w

    x = _rms(x, weights.final_gamma, cfg.rms_eps)
    logits = x @ weights.token_emb.T  # tied head
    return logits, st
