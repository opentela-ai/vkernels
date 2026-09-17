"""GPT-2 compiler frontend for the megakernel compiler.

The model definition (config, weights, KV cache, reference forward) lives in
:mod:`.gpt2_arch` — this module provides the
compiler-specific capture layer: ``SymbolicModelArgs`` (external tensors for
the recording backend) and ``build_forward`` / ``build_transformer_layer``
(the model body expressed against the §4.1 ``ops`` interface so the
capture → lowering → schedule → codegen pipeline applies).

All model-level symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Re-export the model definition so existing ``from .model_gpt2 import …``
# continues to work everywhere (compiler, tests, bench).
# ---------------------------------------------------------------------------
from .gpt2_arch import (  # noqa: F401
    GPT2Config,
    GPT2Weights,
    KVCache,
    LAYER_NORM_EPS,
    LayerWeights,
    default_config,
    random_weights,
    reference_forward,
)

__all__ = [
    # Model definition (re-exported from runner.models.gpt2)
    "GPT2Config",
    "GPT2Weights",
    "KVCache",
    "LayerWeights",
    "LAYER_NORM_EPS",
    "default_config",
    "random_weights",
    "reference_forward",
    # Compiler-specific
    "SymbolicModelArgs",
    "build_forward",
    "build_transformer_layer",
]


# ---------------------------------------------------------------------------
# Compiler utility: deterministic storage ids for symbolic captures
# ---------------------------------------------------------------------------

_STORAGE_SEED = 0x5EED
_storage_counter: dict[str, int] = {}


def _stable_storage_id(name: str) -> int:
    """Deterministic storage id per parameter name (stable across captures)."""
    global _storage_counter
    if name not in _storage_counter:
        _storage_counter[name] = _STORAGE_SEED + len(_storage_counter)
    return _storage_counter[name]


# ---------------------------------------------------------------------------
# Compiler-specific: symbolic model args for the recording backend
# ---------------------------------------------------------------------------


class SymbolicModelArgs:
    """Assembles external tensors/scalars for capture (§4.1).

    Weights are pointer-like arguments: the graph records identity + layout
    only, so a new checkpoint with identical shapes does not force
    recompilation (§13.3). Storage ids are derived deterministically from
    names so repeated captures produce identical graphs.
    """

    def __init__(self, ops, config: GPT2Config, weights: GPT2Weights):
        self.ops = ops
        self.config = config
        self.weights = weights
        w = weights
        cfg = config
        self.token = ops.external_tensor("token_emb", w.token_emb.shape, storage_id=_stable_storage_id("token_emb"))
        self.position = ops.external_tensor("position_emb", w.position_emb.shape, storage_id=_stable_storage_id("position_emb"))
        self.final_gamma = ops.external_tensor("final_ln_gamma", w.final_ln_gamma.shape, storage_id=_stable_storage_id("final_ln_gamma"))
        self.final_beta = ops.external_tensor("final_ln_beta", w.final_ln_beta.shape, storage_id=_stable_storage_id("final_ln_beta"))
        self.layers = []
        for li, lw in enumerate(w.layers):
            views = {}
            for fname in (
                "ln1_gamma",
                "ln1_beta",
                "qkv_w",
                "qkv_b",
                "attn_out_w",
                "attn_out_b",
                "ln2_gamma",
                "ln2_beta",
                "mlp_up_w",
                "mlp_up_b",
                "mlp_down_w",
                "mlp_down_b",
            ):
                arr = getattr(lw, fname)
                key = f"l{li}_{fname}"
                views[fname] = ops.external_tensor(key, arr.shape, storage_id=_stable_storage_id(key))
            self.layers.append(type("LayerViews", (), views))
        # KV cache storages: one packed [L, B, H, S, D] per side (§9.1).
        cache_shape = (cfg.layers, cfg.batch, cfg.heads, cfg.cache_capacity, cfg.head_dim)
        self.k_cache = ops.external_tensor("k_cache", cache_shape, storage_id=_stable_storage_id("k_cache"))
        self.v_cache = ops.external_tensor("v_cache", cache_shape, storage_id=_stable_storage_id("v_cache"))


# ---------------------------------------------------------------------------
# Compiler-specific: model body against the §4.1 ops interface
# ---------------------------------------------------------------------------


def build_forward(ops, args, ids, position, config: GPT2Config):
    """The single-token decode body (§1.1): F(W, ids, K, p) -> (logits, K').

    ``ops`` is any backend implementing the §4.1 interface (recorder or
    numerical); ``args`` supplies symbolic/numeric handles for parameters,
    the packed KV caches, etc.
    """
    cfg = config
    hidden = ops.embedding(ids, args.token, args.position, position)

    for layer_id in range(cfg.layers):
        hidden = build_transformer_layer(ops, args, hidden, position, cfg, layer_id)

    hidden = ops.layer_norm(hidden, args.final_gamma, args.final_beta, LAYER_NORM_EPS, name="final_ln")
    # Tied head: read a transposed view of the token embedding (§5.2).
    tied_w = ops.transpose_view(args.token, "token_emb.T")  # [C, V]
    return ops.linear(hidden, tied_w, None, name="logits")


def build_transformer_layer(ops, args, hidden, position, config: GPT2Config, layer_id: int):
    """One transformer block: 13 unfused arithmetic/cache operations (§6.4)."""
    cfg = config
    lv = args.layers[layer_id]

    # --- attention sublayer ---
    n = ops.layer_norm(hidden, lv.ln1_gamma, lv.ln1_beta, LAYER_NORM_EPS, name=f"l{layer_id}_ln1")
    qkv = ops.linear(n, lv.qkv_w, lv.qkv_b, name=f"l{layer_id}_qkv")  # [B, 3C]
    B = cfg.batch
    C, H, D = cfg.hidden, cfg.heads, cfg.head_dim
    qkv3 = ops.view_of(qkv, f"l{layer_id}_qkv3", (B, 3, H, D))
    q = ops.narrow(qkv3, f"l{layer_id}_q", axis=1, start=0, length=1)
    k_new = ops.narrow(qkv3, f"l{layer_id}_k_new", axis=1, start=1, length=1)
    v_new = ops.narrow(qkv3, f"l{layer_id}_v_new", axis=1, start=2, length=1)
    q_bh = ops.view_of(q, f"l{layer_id}_q_bh", (B, H, D))

    S, D = cfg.cache_capacity, cfg.head_dim
    k_view = ops.view_of(ops.narrow(args.k_cache, f"l{layer_id}_k_l", axis=0, start=layer_id, length=1), f"l{layer_id}_k", (B, H, S, D))
    v_view = ops.view_of(ops.narrow(args.v_cache, f"l{layer_id}_v_l", axis=0, start=layer_id, length=1), f"l{layer_id}_v", (B, H, S, D))
    k_post, v_post = ops.cache_append(k_view, v_view, ops.view_of(k_new, f"l{layer_id}_k_row", (B, H, D)), ops.view_of(v_new, f"l{layer_id}_v_row", (B, H, D)), position, layer=layer_id)

    scores = ops.attention_scores(q_bh, k_post, position, scale=cfg.attention_scale, layer=layer_id)
    probs = ops.softmax(scores, position, layer=layer_id)
    ctx = ops.attention_values(probs, v_post, position, layer=layer_id)  # [B, H, D]
    ctx_flat = ops.view_of(ctx, f"l{layer_id}_ctx_flat", (B, C))
    attn_out = ops.linear(ctx_flat, lv.attn_out_w, lv.attn_out_b, name=f"l{layer_id}_attn_out")
    hidden = ops.add(hidden, attn_out, name=f"l{layer_id}_attn_res")

    # --- MLP sublayer ---
    n2 = ops.layer_norm(hidden, lv.ln2_gamma, lv.ln2_beta, LAYER_NORM_EPS, name=f"l{layer_id}_ln2")
    up = ops.linear(n2, lv.mlp_up_w, lv.mlp_up_b, name=f"l{layer_id}_mlp_up")  # [B, F]
    act = ops.gelu(up, name=f"l{layer_id}_gelu")
    down = ops.linear(act, lv.mlp_down_w, lv.mlp_down_b, name=f"l{layer_id}_mlp_down")  # [B, C]
    return ops.add(hidden, down, name=f"l{layer_id}_mlp_res")
