"""GPT-2 model definition: config, weights, KV cache, and reference forward.

Default dimensions follow the worked example in §6.4: B=1, C=128, H=4
(D=32), F=512; L=2 layers reproduce the expected 29 unfused arithmetic/cache
phases (13 per layer + embedding + final LayerNorm + tied head).

:func:`reference_forward` is a deliberately *independent* straight-line
NumPy implementation so tests can compare the compiled schedule's execution
against an oracle that shares no code with the compiler (§15.1).

This module is the single source of truth for the GPT-2 architecture; both
the naive runner and the megakernel compiler import from here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

__all__ = [
    "GPT2Config",
    "LayerWeights",
    "GPT2Weights",
    "KVCache",
    "LAYER_NORM_EPS",
    "default_config",
    "random_weights",
    "reference_forward",
]

LAYER_NORM_EPS = 1e-5


@dataclass(frozen=True)
class GPT2Config:
    """Static model dimensions; these specialize compilation (§1.3, §13.3)."""

    vocab: int = 1024
    layers: int = 2
    hidden: int = 128  # C
    heads: int = 4  # H
    mlp: int = 512  # F
    batch: int = 1  # B
    cache_capacity: int = 64  # S
    max_positions: int = 64  # learned position-embedding table size

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def attention_scale(self) -> float:
        return 1.0 / np.sqrt(self.head_dim)

    def validate(self) -> None:
        assert self.hidden % self.heads == 0, "C must divide into H heads"
        assert self.cache_capacity <= self.max_positions, "cache cannot exceed position table"
        assert self.vocab > 0 and self.layers > 0


def default_config() -> GPT2Config:
    cfg = GPT2Config()
    cfg.validate()
    return cfg


@dataclass
class LayerWeights:
    ln1_gamma: np.ndarray
    ln1_beta: np.ndarray
    qkv_w: np.ndarray  # [C, 3C]
    qkv_b: np.ndarray  # [3C]
    attn_out_w: np.ndarray  # [C, C]
    attn_out_b: np.ndarray  # [C]
    ln2_gamma: np.ndarray
    ln2_beta: np.ndarray
    mlp_up_w: np.ndarray  # [C, F]
    mlp_up_b: np.ndarray  # [F]
    mlp_down_w: np.ndarray  # [F, C]
    mlp_down_b: np.ndarray  # [C]


@dataclass
class GPT2Weights:
    token_emb: np.ndarray  # [V, C]
    position_emb: np.ndarray  # [max_positions, C]
    layers: list[LayerWeights]
    final_ln_gamma: np.ndarray
    final_ln_beta: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        """Flat name -> array map (used by the reference executor)."""
        out = {
            "token_emb": self.token_emb,
            "position_emb": self.position_emb,
            "final_ln_gamma": self.final_ln_gamma,
            "final_ln_beta": self.final_ln_beta,
        }
        for li, lw in enumerate(self.layers):
            for f_ in fields(LayerWeights):
                out[f"l{li}_{f_.name}"] = getattr(lw, f_.name)
        return out


def random_weights(config: GPT2Config, seed: int = 0) -> GPT2Weights:
    """Deterministic synthetic weights — no checkpoint download required."""
    rng = np.random.default_rng(seed)
    C, F, V, L = config.hidden, config.mlp, config.vocab, config.layers

    def x(shape) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float64)

    layers = []
    for _ in range(L):
        layers.append(
            LayerWeights(
                ln1_gamma=1.0 + 0.1 * x((C,)),
                ln1_beta=0.1 * x((C,)),
                qkv_w=0.05 * x((C, 3 * C)),
                qkv_b=0.05 * x((3 * C,)),
                attn_out_w=0.05 * x((C, C)),
                attn_out_b=0.05 * x((C,)),
                ln2_gamma=1.0 + 0.1 * x((C,)),
                ln2_beta=0.1 * x((C,)),
                mlp_up_w=0.05 * x((C, F)),
                mlp_up_b=0.05 * x((F,)),
                mlp_down_w=0.05 * x((F, C)),
                mlp_down_b=0.05 * x((C,)),
            )
        )
    return GPT2Weights(
        token_emb=0.1 * x((V, C)),
        position_emb=0.1 * x((config.max_positions, C)),
        layers=layers,
        final_ln_gamma=1.0 + 0.1 * x((C,)),
        final_ln_beta=0.1 * x((C,)),
    )


class KVCache:
    """Host-side KV cache: storage plus valid-length bookkeeping (§4.3).

    ``valid_len`` is *host metadata only* — the compiler treats the position
    as a runtime scalar and guards ``0 <= p < S``. Unused tails may contain
    garbage or NaN; attention must never read them (§5.3).
    """

    def __init__(self, config: GPT2Config, *, fill: float = np.nan):
        self.config = config
        shape = (config.layers, config.batch, config.heads, config.cache_capacity, config.head_dim)
        self.k = np.full(shape, fill, dtype=np.float64)
        self.v = np.full(shape, fill, dtype=np.float64)
        self.valid_len = 0

    def capacity(self) -> int:
        return self.config.cache_capacity

    def storage_bytes(self) -> int:
        return self.k.nbytes + self.v.nbytes


# ---------------------------------------------------------------------------
# Independent reference oracle (§15.1)
# ---------------------------------------------------------------------------


def _layer_norm_ref(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _gelu_ref(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def reference_forward(
    weights: GPT2Weights,
    ids: np.ndarray,
    cache: KVCache,
    position: int,
    config: GPT2Config,
) -> tuple[np.ndarray, KVCache]:
    """Straight-line NumPy GPT-2 decode step; shares no code with the compiler.

    Appends to ``cache`` in place (advancing ``valid_len``), matching the
    execution contract of §1.1: inputs at positions [0, p), append p,
    return logits for p.
    """
    cfg = config
    if not (0 <= position < cfg.cache_capacity):
        raise ValueError(f"position {position} out of range [0, {cfg.cache_capacity})")
    B, C, H, D = cfg.batch, cfg.hidden, cfg.heads, cfg.head_dim
    scale = cfg.attention_scale

    x = weights.token_emb[ids] + weights.position_emb[position]  # [B, C]

    for li, lw in enumerate(weights.layers):
        n = _layer_norm_ref(x, lw.ln1_gamma, lw.ln1_beta, LAYER_NORM_EPS)
        qkv = n @ lw.qkv_w + lw.qkv_b  # [B, 3C]
        qkv3 = qkv.reshape(B, 3, H, D)
        q, k_new, v_new = qkv3[:, 0], qkv3[:, 1], qkv3[:, 2]  # each [B, H, D]

        # Cache append at position p (mutates the caller's cache).
        cache.k[li, :, :, position, :] = k_new
        cache.v[li, :, :, position, :] = v_new

        # Attention over valid prefix [0, p].
        K = cache.k[li, :, :, : position + 1, :]  # [B, H, p+1, D]
        V = cache.v[li, :, :, : position + 1, :]
        scores = np.einsum("bhd,bhtd->bht", q, K) * scale  # [B, H, p+1]
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        probs = e / e.sum(axis=-1, keepdims=True)
        ctx = np.einsum("bht,bhtd->bhd", probs, V)  # [B, H, D]

        attn = ctx.reshape(B, C) @ lw.attn_out_w + lw.attn_out_b
        x = x + attn

        n2 = _layer_norm_ref(x, lw.ln2_gamma, lw.ln2_beta, LAYER_NORM_EPS)
        up = _gelu_ref(n2 @ lw.mlp_up_w + lw.mlp_up_b)
        x = x + (up @ lw.mlp_down_w + lw.mlp_down_b)

    xf = _layer_norm_ref(x, weights.final_ln_gamma, weights.final_ln_beta, LAYER_NORM_EPS)
    logits = xf @ weights.token_emb.T  # tied head
    cache.valid_len = position + 1
    return logits, cache
