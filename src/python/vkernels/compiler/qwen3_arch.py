"""Dense Qwen3 model definition (Qwen3-0.6B architecture).

Config, weights, KV cache, RoPE tables, and a reference NumPy forward pass.
This module is the single source of truth for the Qwen3 architecture; both
the naive runner and the megakernel compiler import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

__all__ = [
    "Qwen3Config",
    "QWEN3_06B",
    "tiny_qwen3_config",
    "Qwen3LayerWeights",
    "Qwen3Weights",
    "rope_tables",
    "random_qwen3_weights",
    "weights_from_hf",
    "Qwen3KVCache",
    "_rms",
    "_rotate_half",
    "qwen3_reference_forward",
]


@dataclass(frozen=True)
class Qwen3Config:
    """Dense Qwen3 dimensions (these specialize compilation, §13.3)."""

    vocab: int
    layers: int
    hidden: int  # C
    heads: int  # H (query heads)
    kv_heads: int  # KVH (GQA)
    head_dim: int  # D (explicit in Qwen3; independent of hidden/heads)
    intermediate: int  # F (SwiGLU)
    batch: int = 1
    cache_capacity: int = 512
    max_positions: int = 4096
    rope_theta: float = 1_000_000.0
    rms_eps: float = 1e-6

    @property
    def group(self) -> int:
        return self.heads // self.kv_heads

    @property
    def attention_scale(self) -> float:
        return 1.0 / np.sqrt(self.head_dim)

    def validate(self) -> None:
        assert self.heads % self.kv_heads == 0, "GQA: H must be a multiple of KVH"
        assert self.heads * self.head_dim >= 1
        assert self.intermediate % 2 == 0, "fused gate-up splits [B, 2F]"


def QWEN3_06B(*, batch: int = 1, cache_capacity: int = 512) -> Qwen3Config:  # noqa: N802
    """The published Qwen3-0.6B dimensions (HF config.json)."""
    cfg = Qwen3Config(
        vocab=151936,
        layers=28,
        hidden=1024,
        heads=16,
        kv_heads=8,
        head_dim=128,
        intermediate=3072,
        batch=batch,
        cache_capacity=cache_capacity,
        max_positions=4096,
    )
    cfg.validate()
    return cfg


def tiny_qwen3_config(*, layers: int = 2, batch: int = 1, cache_capacity: int = 16) -> Qwen3Config:
    """Small-but-faithful fixture: GQA (4Q/2KV), QK-norm, RoPE, SwiGLU."""
    cfg = Qwen3Config(
        vocab=512,
        layers=layers,
        hidden=128,
        heads=4,
        kv_heads=2,
        head_dim=32,
        intermediate=256,
        batch=batch,
        cache_capacity=cache_capacity,
        max_positions=64,
    )
    cfg.validate()
    return cfg


@dataclass
class Qwen3LayerWeights:
    ln1_gamma: np.ndarray  # [C]
    qkv_w: np.ndarray  # [C, (H+2*KVH)*D] — packed [q | k | v] (no bias)
    q_norm_gamma: np.ndarray  # [D] per-head QK-norm
    k_norm_gamma: np.ndarray  # [D]
    o_proj_w: np.ndarray  # [H*D, C] = [C_q, C]
    ln2_gamma: np.ndarray  # [C]
    gate_up_w: np.ndarray  # [C, 2F] — packed [gate | up]
    down_w: np.ndarray  # [F, C]
    # (no biases anywhere in Qwen3)


@dataclass
class Qwen3Weights:
    token_emb: np.ndarray  # [V, C] (tied head)
    layers: list[Qwen3LayerWeights]
    final_gamma: np.ndarray  # [C]
    cos: np.ndarray  # [max_positions, D]
    sin: np.ndarray  # [max_positions, D]

    def arrays(self) -> dict[str, np.ndarray]:
        out = {
            "token_emb": self.token_emb,
            "final_ln_gamma": self.final_gamma,
            "rope_cos": self.cos,
            "rope_sin": self.sin,
        }
        for li, lw in enumerate(self.layers):
            for f_ in fields(Qwen3LayerWeights):
                out[f"l{li}_{f_.name}"] = getattr(lw, f_.name)
        return out


def rope_tables(config: Qwen3Config) -> tuple[np.ndarray, np.ndarray]:
    """Precomputed rotate_half RoPE tables (HF Qwen3 convention).

    inv_freq_i = theta^(-2i/D); freqs[p] = p * inv_freq;
    cos = sin tables of full width D = cat([f, f]).
    """
    D = config.head_dim
    inv_freq = config.rope_theta ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    pos = np.arange(config.max_positions, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)  # [S, D/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [S, D]
    return np.cos(emb), np.sin(emb)


def random_qwen3_weights(config: Qwen3Config, seed: int = 0) -> Qwen3Weights:
    rng = np.random.default_rng(seed)
    C, F, H, KVH, D = config.hidden, config.intermediate, config.heads, config.kv_heads, config.head_dim

    def x(shape):
        return rng.standard_normal(shape).astype(np.float64)

    layers = [
        Qwen3LayerWeights(
            ln1_gamma=1.0 + 0.1 * x((C,)),
            qkv_w=0.05 * x((C, (H + 2 * KVH) * D)),
            q_norm_gamma=1.0 + 0.1 * x((D,)),
            k_norm_gamma=1.0 + 0.1 * x((D,)),
            o_proj_w=0.05 * x((H * D, C)),
            ln2_gamma=1.0 + 0.1 * x((C,)),
            gate_up_w=0.05 * x((C, 2 * F)),
            down_w=0.05 * x((F, C)),
        )
        for _ in range(config.layers)
    ]
    cos, sin = rope_tables(config)
    return Qwen3Weights(
        token_emb=0.1 * x((config.vocab, C)),
        layers=layers,
        final_gamma=1.0 + 0.1 * x((C,)),
        cos=cos,
        sin=sin,
    )


def weights_from_hf(hf_state: dict, config: Qwen3Config) -> Qwen3Weights:
    """Adapt an HF ``Qwen3ForCausalLM`` state dict into compiler weights.

    HF stores projections as [out, in]; this compiler's convention is
    y = x @ w with w [in, out], so every 2-D projection is transposed.
    The fused QKV is unpacked into [q | k | v] groups of heads.
    """
    import torch

    def t(name):
        return torch.as_tensor(hf_state[name]).detach().cpu().to(torch.float64).numpy()

    layers = []
    for li in range(config.layers):
        p = f"model.layers.{li}."
        wq = t(p + "self_attn.q_proj.weight")  # [H*D, C]
        wk = t(p + "self_attn.k_proj.weight")  # [KVH*D, C]
        wv = t(p + "self_attn.v_proj.weight")  # [KVH*D, C]
        qkv_w = np.concatenate([wq, wk, wv], axis=0).T  # [C, (H+2KVH)*D]
        gate = t(p + "mlp.gate_proj.weight")  # [F, C]
        up = t(p + "mlp.up_proj.weight")  # [F, C]
        gate_up_w = np.concatenate([gate, up], axis=0).T  # [C, 2F] packed [gate | up]
        layers.append(
            Qwen3LayerWeights(
                ln1_gamma=t(p + "input_layernorm.weight"),
                qkv_w=qkv_w,
                q_norm_gamma=t(p + "self_attn.q_norm.weight"),
                k_norm_gamma=t(p + "self_attn.k_norm.weight"),
                o_proj_w=t(p + "self_attn.o_proj.weight").T,  # HF [C, H*D] -> our [H*D, C]
                ln2_gamma=t(p + "post_attention_layernorm.weight"),
                gate_up_w=gate_up_w,  # [C, 2F] packed [gate | up]
                down_w=t(p + "mlp.down_proj.weight").T,  # HF [C, F] -> our [F, C]
            )
        )
    cos, sin = rope_tables(config)
    return Qwen3Weights(
        token_emb=t("model.embed_tokens.weight"),
        layers=layers,
        final_gamma=t("model.norm.weight"),
        cos=cos,
        sin=sin,
    )


class Qwen3KVCache:
    """Packed [L, B, KVH, S, D] K/V storages (GQA: KV heads only)."""

    def __init__(self, config: Qwen3Config, *, fill: float = np.nan):
        self.config = config
        shape = (config.layers, config.batch, config.kv_heads, config.cache_capacity, config.head_dim)
        self.k = np.full(shape, fill, dtype=np.float64)
        self.v = np.full(shape, fill, dtype=np.float64)
        self.valid_len = 0

    def storage_bytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


def _rms(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    var = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(var + eps) * gamma


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def qwen3_reference_forward(
    weights: Qwen3Weights,
    ids: np.ndarray,
    cache: Qwen3KVCache,
    position: int,
    config: Qwen3Config,
) -> tuple[np.ndarray, Qwen3KVCache]:
    """Straight-line NumPy dense-Qwen3 decode step; appends at position p."""
    cfg = config
    if not (0 <= position < cfg.cache_capacity):
        raise ValueError(f"position {position} out of range [0, {cfg.cache_capacity})")
    B, H, KVH, D = cfg.batch, cfg.heads, cfg.kv_heads, cfg.head_dim
    G = H // KVH

    x = weights.token_emb[ids]  # [B, C]

    for li, lw in enumerate(weights.layers):
        n = _rms(x, lw.ln1_gamma, cfg.rms_eps)
        qkv = n @ lw.qkv_w  # [B, (H+2KVH)*D]
        q = qkv.reshape(B, H + 2 * KVH, D)[:, :H]  # [B, H, D]
        k = qkv.reshape(B, H + 2 * KVH, D)[:, H : H + KVH]
        v = qkv.reshape(B, H + 2 * KVH, D)[:, H + KVH :]

        q = _rms(q, lw.q_norm_gamma, cfg.rms_eps)
        k = _rms(k, lw.k_norm_gamma, cfg.rms_eps)
        q = q * weights.cos[position] + _rotate_half(q) * weights.sin[position]
        k = k * weights.cos[position] + _rotate_half(k) * weights.sin[position]

        cache.k[li, :, :, position, :] = k
        cache.v[li, :, :, position, :] = v

        K = cache.k[li, :, :, : position + 1, :]  # [B, KVH, p+1, D]
        V = cache.v[li, :, :, : position + 1, :]
        ctx = np.zeros((B, H, D))
        for h in range(H):
            kvh = h // G
            s = np.einsum("bd,btd->bt", q[:, h], K[:, kvh]) * cfg.attention_scale
            s = s - s.max(axis=-1, keepdims=True)
            e = np.exp(s)
            probs = e / e.sum(axis=-1, keepdims=True)
            ctx[:, h] = np.einsum("bt,btd->d", probs, V[:, kvh])
        attn = ctx.reshape(B, H * D) @ lw.o_proj_w  # o_proj input width is H*D (head_dim is explicit)
        x = x + attn

        n2 = _rms(x, lw.ln2_gamma, cfg.rms_eps)
        gu = n2 @ lw.gate_up_w  # [B, 2F]
        gate, up = gu.reshape(B, 2, -1)[:, 0], gu.reshape(B, 2, -1)[:, 1]
        act = gate / (1.0 + np.exp(-gate)) * up
        x = x + (act @ lw.down_w)

    xf = _rms(x, weights.final_gamma, cfg.rms_eps)
    logits = xf @ weights.token_emb.T
    cache.valid_len = position + 1
    return logits, cache
