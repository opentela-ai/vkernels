"""Qwen3 compiler frontend for the megakernel compiler.

The model definition (config, weights, KV cache, reference forward) lives in
:mod:`.qwen3_arch` — this module provides the
compiler-specific capture layer: ``Qwen3ModelArgs`` (symbolic external
tensors for the recording backend) and ``build_qwen3_forward`` /
``_qwen3_layer`` (the model body expressed against the §4.1 ``ops``
interface so the capture → lowering → schedule → codegen pipeline applies).

All model-level symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

from .capture import RecordingBackend
from .model_gpt2 import _stable_storage_id

# ---------------------------------------------------------------------------
# Re-export the model definition so existing ``from .model_qwen3 import …``
# continues to work everywhere (compiler, tests, bench).
# ---------------------------------------------------------------------------
from .qwen3_arch import (  # noqa: F401
    QWEN3_06B,
    Qwen3Config,
    Qwen3KVCache,
    Qwen3LayerWeights,
    Qwen3Weights,
    qwen3_reference_forward,
    random_qwen3_weights,
    rope_tables,
    tiny_qwen3_config,
    weights_from_hf,
)

__all__ = [
    # Model definition (re-exported from runner.models.qwen3)
    "Qwen3Config",
    "Qwen3LayerWeights",
    "Qwen3Weights",
    "Qwen3KVCache",
    "QWEN3_06B",
    "random_qwen3_weights",
    "rope_tables",
    "Qwen3ModelArgs",
    "build_qwen3_forward",
    "qwen3_reference_forward",
    "weights_from_hf",
    "tiny_qwen3_config",
]


# ---------------------------------------------------------------------------
# Compiler-specific: symbolic model args for the recording backend
# ---------------------------------------------------------------------------


class Qwen3ModelArgs:
    """External tensors for capture: parameters, RoPE tables, packed caches."""

    def __init__(self, ops: RecordingBackend, config: Qwen3Config):
        self.config = config
        ops._externals = getattr(ops, "_externals", {})
        self.token = ops.external_tensor("token_emb", (config.vocab, config.hidden), storage_id=_stable_storage_id("token_emb"))
        self.final_gamma = ops.external_tensor("final_ln_gamma", (config.hidden,), storage_id=_stable_storage_id("final_ln_gamma"))
        self.rope_cos = ops.external_tensor("rope_cos", (config.max_positions, config.head_dim), storage_id=_stable_storage_id("rope_cos"))
        self.rope_sin = ops.external_tensor("rope_sin", (config.max_positions, config.head_dim), storage_id=_stable_storage_id("rope_sin"))
        self.layers = []
        for li in range(config.layers):
            views = {}
            for fname in ("ln1_gamma", "qkv_w", "q_norm_gamma", "k_norm_gamma", "o_proj_w", "ln2_gamma", "gate_up_w", "down_w"):
                shape = _param_shape(config, fname)
                views[fname] = ops.external_tensor(f"l{li}_{fname}", shape, storage_id=_stable_storage_id(f"l{li}_{fname}"))
            self.layers.append(type("LayerViews", (), views))
        cache_shape = (config.layers, config.batch, config.kv_heads, config.cache_capacity, config.head_dim)
        self.k_cache = ops.external_tensor("k_cache", cache_shape, storage_id=_stable_storage_id("k_cache"))
        self.v_cache = ops.external_tensor("v_cache", cache_shape, storage_id=_stable_storage_id("v_cache"))


def _param_shape(config: Qwen3Config, fname: str) -> tuple[int, ...]:
    C, H, KVH, D, F = config.hidden, config.heads, config.kv_heads, config.head_dim, config.intermediate
    return {
        "ln1_gamma": (C,),
        "qkv_w": (C, (H + 2 * KVH) * D),
        "q_norm_gamma": (D,),
        "k_norm_gamma": (D,),
        "o_proj_w": (H * D, C),
        "ln2_gamma": (C,),
        "gate_up_w": (C, 2 * F),
        "down_w": (F, C),
    }[fname]


# ---------------------------------------------------------------------------
# Compiler-specific: model body against the §4.1 ops interface
# ---------------------------------------------------------------------------


def build_qwen3_forward(ops, args, ids, position, config: Qwen3Config):
    """One dense-Qwen3 single-token decode: F(W, ids, K, p) -> logits."""
    cfg = config
    hidden = ops.embedding(ids, args.token, None, position)  # token only (RoPE carries position)

    for layer_id in range(cfg.layers):
        hidden = _qwen3_layer(ops, args, hidden, position, cfg, layer_id)

    hidden = ops.rms_norm(hidden, args.final_gamma, cfg.rms_eps, name="final_rms")
    tied_w = ops.transpose_view(args.token, "token_emb.T")  # [C, V], tied head
    return ops.linear(hidden, tied_w, None, name="logits")


def _qwen3_layer(ops, args, hidden, position, config: Qwen3Config, layer_id: int):
    cfg = config
    B, H, KVH, D, F = cfg.batch, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.intermediate
    lv = args.layers[layer_id]

    # --- attention sublayer ---
    n = ops.rms_norm(hidden, lv.ln1_gamma, cfg.rms_eps, name=f"l{layer_id}_rms1")
    qkv = ops.linear(n, lv.qkv_w, None, name=f"l{layer_id}_qkv")  # [B, (H+2KVH)*D]
    qkv3 = ops.view_of(qkv, f"l{layer_id}_qkv3", (B, H + 2 * KVH, D))
    q = ops.view_of(ops.narrow(qkv3, f"l{layer_id}_q_n", axis=1, start=0, length=H), f"l{layer_id}_q", (B, H, D))
    k = ops.view_of(ops.narrow(qkv3, f"l{layer_id}_k_n", axis=1, start=H, length=KVH), f"l{layer_id}_k", (B, KVH, D))
    v = ops.view_of(ops.narrow(qkv3, f"l{layer_id}_v_n", axis=1, start=H + KVH, length=KVH), f"l{layer_id}_v", (B, KVH, D))

    # QK-norm (per head over D), then RoPE at position p.
    q = ops.rms_norm(q, lv.q_norm_gamma, cfg.rms_eps, name=f"l{layer_id}_qnorm")
    k = ops.rms_norm(k, lv.k_norm_gamma, cfg.rms_eps, name=f"l{layer_id}_knorm")
    q = ops.rope(q, args.rope_cos, args.rope_sin, position, layer=layer_id, which="q")
    k = ops.rope(k, args.rope_cos, args.rope_sin, position, layer=layer_id, which="k")

    k_view = ops.view_of(ops.narrow(args.k_cache, f"l{layer_id}_kc_l", axis=0, start=layer_id, length=1), f"l{layer_id}_kcache", (B, KVH, cfg.cache_capacity, D))
    v_view = ops.view_of(ops.narrow(args.v_cache, f"l{layer_id}_vc_l", axis=0, start=layer_id, length=1), f"l{layer_id}_vcache", (B, KVH, cfg.cache_capacity, D))
    k_post, v_post = ops.cache_append(k_view, v_view, k, v, position, layer=layer_id)

    scores = ops.attention_scores(q, k_post, position, scale=cfg.attention_scale, layer=layer_id, kv_heads=KVH)
    probs = ops.softmax(scores, position, layer=layer_id)
    ctx = ops.attention_values(probs, v_post, position, layer=layer_id, kv_heads=KVH)  # [B, H, D]
    attn = ops.linear(ops.view_of(ctx, f"l{layer_id}_ctx_flat", (B, H * D)), lv.o_proj_w, None, name=f"l{layer_id}_o_proj")
    hidden = ops.add(hidden, attn, name=f"l{layer_id}_attn_res")

    # --- SwiGLU MLP ---
    n2 = ops.rms_norm(hidden, lv.ln2_gamma, cfg.rms_eps, name=f"l{layer_id}_rms2")
    gu = ops.linear(n2, lv.gate_up_w, None, name=f"l{layer_id}_gate_up")  # [B, 2F]
    gu3 = ops.view_of(gu, f"l{layer_id}_gu3", (B, 2, F))
    gate = ops.view_of(ops.narrow(gu3, f"l{layer_id}_gate_n", axis=1, start=0, length=1), f"l{layer_id}_gate", (B, F))
    up = ops.view_of(ops.narrow(gu3, f"l{layer_id}_up_n", axis=1, start=1, length=1), f"l{layer_id}_up", (B, F))
    act = ops.swiglu(gate, up, name=f"l{layer_id}_swiglu")
    down = ops.linear(act, lv.down_w, None, name=f"l{layer_id}_down")
    return ops.add(hidden, down, name=f"l{layer_id}_mlp_res")
