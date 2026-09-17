"""GLM-5.3 Flash compiler arch: config, tiny weights, fp64 decode-step mirror.

The mirror reproduces the floe ``engine/runner/models/glm5/glm5_arch.py``
decode path exactly as the landed op contracts pin it (issues #88–#101):

* mHC hyper-connections, TWO blocks per layer (attn side + ffn side) —
  ``Glm53HyperConnection.forward``: unweighted-RMSNorm over the flattened
  streams, bias-free ``fn`` projection, sigmoid/2-sigmoid gates, Sinkhorn
  doubly-stochastic ``comb``; ``_mhc_compose`` is the layer residual path.
* KDA linear-attention layer: fused qkv GEMV → depthwise causal FIR conv
  (raw-input state convention) → per-(head,dim) forget gate (lower-bound
  sigmoid form) → element-wise-decay delta rule (L2-normed q/k, fp32
  state RMW) → sigmoid-gated RMSNorm per head → o_proj.
* MoE: noaux_tc sigmoid router (biased choice scores, top-2-sum groups
  degenerate at n_group=1, renorm, routed_scaling_factor) → per-(row,
  slot) expert FFN with swiglu_limit clamping → weighted combine + the
  shared dense expert.
* Output head: unweighted mean over the hc streams (``Glm53HyperHead``)
  then the final RMSNorm.

Everything is float64: the mirror is the bare-environment oracle the
compiled graph is validated against (no torch, no floe needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Glm53Config:
    """Tiny-but-faithful GLM-5.3 decode config (floe ``Glm53Config`` fields)."""

    hidden_size: int = 64
    num_hidden_layers: int = 3
    layer_types: tuple[str, ...] = ("linear_attention", "linear_attention", "linear_attention")
    mlp_layer_types: tuple[str, ...] = ("dense", "sparse", "sparse")
    # KDA linear attention
    linear_num_heads: int = 2
    linear_head_dim: int = 8  # KDA heads are square: K = V = head_dim
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float | None = -5.0
    # mHC
    hc_mult: int = 2
    hc_sinkhorn_iters: int = 3
    hc_eps: float = 1.0e-6
    # norms
    rms_norm_eps: float = 1.0e-5
    # MoE
    n_routed_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 32
    n_shared_experts: int = 1
    intermediate_size: int = 48  # dense MLP
    swiglu_limit: float = 10.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    first_k_dense_replace: int = 1
    # DSA indexer (sparse layers; capture-structure scope in Stage 3)
    index_topk: int = 4
    index_kpool: int = 2
    indexer_head_dim: int = 8
    indexer_types: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must cover every layer")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError("mlp_layer_types must cover every layer")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError("index_topk must be divisible by index_kpool")

    @property
    def qkv_dim(self) -> int:
        return self.linear_num_heads * self.linear_head_dim

    @property
    def conv_dim(self) -> int:
        return self.qkv_dim * 3

    @property
    def hc_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult


def real_glm53_dims_config(num_hidden_layers: int = 46) -> Glm53Config:
    """Published GLM-5.3-Flash decode dims (compile smoke target)."""
    return Glm53Config(
        hidden_size=5120,
        num_hidden_layers=num_hidden_layers,
        layer_types=tuple(
            "deepseek_sparse_attention" if ((i + 1) % 4 == 0 and i >= 3) else "linear_attention"
            for i in range(num_hidden_layers)
        ),
        mlp_layer_types=tuple("dense" if i < 3 else "sparse" for i in range(num_hidden_layers)),
        linear_num_heads=64,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        n_routed_experts=288,
        num_experts_per_tok=8,
        moe_intermediate_size=1536,
        intermediate_size=1536,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        index_topk=2048,
        index_kpool=4,
    )


# ---------------------------------------------------------------------------
# fp64 reference decode step (the mirror)
# ---------------------------------------------------------------------------


def _rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    return x * np.rsqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _swiglu(gate: np.ndarray, up: np.ndarray, limit: float) -> np.ndarray:
    """GLM swiglu: silu(clamp(gate, max=limit)) * clamp(up, ±limit)."""
    return _silu(np.minimum(gate, limit)) * np.clip(up, -limit, limit)


@dataclass
class Glm53Weights:
    """Numpy weight container (layout mirrors the floe module attributes)."""

    embed: np.ndarray  # [V, C]
    # per layer
    hc_attn_fn: list  # [(2+hc)*hc, hc*C]
    hc_attn_base: list  # [(2+hc)*hc]
    hc_attn_scale: list  # [3]
    hc_ffn_fn: list
    hc_ffn_base: list
    hc_ffn_scale: list
    ln1: list  # [C]
    ln2: list  # [C]
    # KDA
    qkv_proj: list  # [3*qkv, C] (fused q|k|v)
    conv_w: list  # [conv_dim, K]
    f_a: list  # [D, C]
    f_b: list  # [qkv, D]
    dt_bias: list  # [qkv] == [H, K] flattened
    A_log: list  # [H]
    b_proj: list  # [H, C]
    g_a: list  # [D, C]
    g_b: list  # [qkv, D]
    o_norm: list  # [D]
    o_proj: list  # [C, qkv]
    # MLP / MoE
    gate_proj: list  # dense layers: [I, C]
    up_proj: list
    down_proj: list  # [C, I]
    router_w: list  # sparse layers: [E, C]
    router_bias: list  # [E]
    expert_gate_up: list  # [E, 2*Imoe, C]
    expert_down: list  # [E, C, Imoe]
    shared_gate: list  # [Imoe, C]
    shared_up: list
    shared_down: list  # [C, Imoe]
    final_norm: list  # [C]


def random_glm53_weights(cfg: Glm53Config, seed: int = 20260917) -> Glm53Weights:
    rng = np.random.default_rng(seed)

    def n(*shape: int, s: float = 0.05) -> np.ndarray:
        return rng.normal(0.0, s, size=shape)

    C, H, D = cfg.hidden_size, cfg.linear_num_heads, cfg.linear_head_dim
    qkv, I, Imoe, E = cfg.qkv_dim, cfg.intermediate_size, cfg.moe_intermediate_size, cfg.n_routed_experts
    hc, mix = cfg.hc_mult, cfg.hc_mix
    L = cfg.num_hidden_layers
    w = Glm53Weights(
        embed=n(cfg.num_experts_per_tok * 32, C, s=0.2),  # vocab ample for tiny ids
        hc_attn_fn=[n(mix, hc * C) for _ in range(L)],
        hc_attn_base=[n(mix, s=0.1) for _ in range(L)],
        hc_attn_scale=[np.array([1.0, 1.0, 1.0]) for _ in range(L)],
        hc_ffn_fn=[n(mix, hc * C) for _ in range(L)],
        hc_ffn_base=[n(mix, s=0.1) for _ in range(L)],
        hc_ffn_scale=[np.array([1.0, 1.0, 1.0]) for _ in range(L)],
        ln1=[np.ones(C) + n(C, s=0.02) for _ in range(L)],
        ln2=[np.ones(C) + n(C, s=0.02) for _ in range(L)],
        qkv_proj=[n(3 * qkv, C) for _ in range(L)],
        conv_w=[n(cfg.conv_dim, cfg.linear_conv_kernel_dim, s=0.3) for _ in range(L)],
        f_a=[n(D, C) for _ in range(L)],
        f_b=[n(qkv, D) for _ in range(L)],
        dt_bias=[n(qkv, s=0.1) for _ in range(L)],
        A_log=[n(H, s=0.2) for _ in range(L)],
        b_proj=[n(H, C) for _ in range(L)],
        g_a=[n(D, C) for _ in range(L)],
        g_b=[n(qkv, D) for _ in range(L)],
        o_norm=[np.ones(D) + n(D, s=0.02) for _ in range(L)],
        o_proj=[n(C, qkv) for _ in range(L)],
        gate_proj=[n(I, C) for _ in range(L)],
        up_proj=[n(I, C) for _ in range(L)],
        down_proj=[n(C, I) for _ in range(L)],
        router_w=[n(E, C, s=0.3) for _ in range(L)],
        router_bias=[np.zeros(E) for _ in range(L)],
        expert_gate_up=[n(E, 2 * Imoe, C) for _ in range(L)],
        expert_down=[n(E, C, Imoe) for _ in range(L)],
        shared_gate=[n(Imoe, C) for _ in range(L)],
        shared_up=[n(Imoe, C) for _ in range(L)],
        shared_down=[n(C, Imoe) for _ in range(L)],
        final_norm=[np.ones(C) + n(C, s=0.02)],
    )
    return w


def _mhc_block(
    streams: np.ndarray,  # [B, hc, C]
    fn: np.ndarray, base: np.ndarray, scale: np.ndarray,
    cfg: Glm53Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """floe Glm53HyperConnection.forward: returns (post, comb, h_in)."""
    B, hc, C = streams.shape
    eps, rms_eps = cfg.hc_eps, cfg.rms_norm_eps
    flat = streams.reshape(B, hc * C)
    flat_n = flat * np.rsqrt(np.mean(flat * flat, axis=-1, keepdims=True) + rms_eps)
    logits = flat_n @ fn.T  # [B, mix] — no projection bias
    pre_w, post_w, comb_w = np.split(logits, [hc, 2 * hc], axis=-1)
    pre_b, post_b, comb_b = np.split(base, [hc, 2 * hc])
    pre_s, post_s, comb_s = scale
    pre = _sigmoid(pre_w * pre_s + pre_b) + eps
    post = 2.0 * _sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.reshape(B, hc, hc) * comb_s + comb_b.reshape(hc, hc)
    comb = _softmax(comb_logits, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    h_in = (pre[:, :, None] * streams).sum(axis=1)  # [B, C]
    return post, comb, h_in


def _mhc_compose(
    streams: np.ndarray, post: np.ndarray, comb: np.ndarray, body: np.ndarray,
) -> np.ndarray:
    """streams'[j] = post[j]·body + Σ_k comb[k, j]·streams[k]."""
    return post[:, :, None] * body[:, None, :] + np.einsum("bkj,bkc->bjc", comb, streams)


def _kda_layer_decode(
    h: np.ndarray,  # [B, C] post-input_layernorm row
    w: Glm53Weights, cfg: Glm53Config, layer: int,
    conv_state: np.ndarray, ssm_state: np.ndarray,  # pools, mutated
) -> np.ndarray:
    C, H, D = cfg.hidden_size, cfg.linear_num_heads, cfg.linear_head_dim
    qkv, K = cfg.qkv_dim, cfg.linear_conv_kernel_dim
    B = h.shape[0]
    # fused qkv projection + depthwise causal FIR conv (raw-input state)
    mixed = h @ w.qkv_proj[layer].T  # [B, 3qkv]
    full = np.concatenate([conv_state, mixed[:, None, :]], axis=1)  # [B, K, Cc]
    # per channel: out[b, c] = silu(Σ_j w[c, j] * full[b, j, c])
    conv_out = _silu(np.einsum("bjc,cj->bc", full, w.conv_w[layer]))
    conv_state[:, : K - 1, :] = full[:, 1:, :]
    q, k, v = np.split(conv_out, [qkv, 2 * qkv], axis=-1)
    q = q.reshape(B, H, D)
    k = k.reshape(B, H, D)
    v = v.reshape(B, H, D)
    # forget gate (lower-bound sigmoid form) + beta
    f_mid = h @ w.f_a[layer].T
    f = (f_mid @ w.f_b[layer].T).reshape(B, H, D)
    g = cfg.linear_lower_bound * _sigmoid(
        np.exp(w.A_log[layer])[None, :, None] * (f + w.dt_bias[layer].reshape(H, D)[None])
    )  # [B, H, D] log-space
    beta = _sigmoid(h @ w.b_proj[layer].T)  # [B, H]
    # element-wise-decay delta rule (fp32 in floe, fp64 here), per (b, head)
    scale = D ** -0.5
    q_n = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * scale
    k_n = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    s = ssm_state.copy()
    s = s * np.exp(g)[:, :, :, None]  # decay rows, broadcast over V
    kv = np.einsum("bhk,bhk->bhv", s, k_n)
    delta = beta[:, :, None] * (v - kv)
    s = s + k_n[:, :, :, None] * delta[:, :, None, :]
    out = np.einsum("bhk,bhk->bhv", s, q_n)
    ssm_state[:] = s
    # gated output norm (per head row) + o_proj
    gate = (h @ w.g_a[layer].T @ w.g_b[layer].T).reshape(B, H, D)
    on = _rms_norm(out, w.o_norm[layer], cfg.rms_norm_eps) * _sigmoid(gate)
    return on.reshape(B, qkv) @ w.o_proj[layer].T


def _moe_block(
    h: np.ndarray, w: Glm53Weights, cfg: Glm53Config, layer: int,
) -> np.ndarray:
    E, Imoe = cfg.n_routed_experts, cfg.moe_intermediate_size
    logits = h @ w.router_w[layer].T
    scores = _sigmoid(logits)
    choice = scores + w.router_bias[layer]
    # n_group == 1: group mask degenerates (floe + landed op agree)
    k = cfg.num_experts_per_tok
    order = np.argsort(-choice, axis=-1, kind="stable")  # ties -> lower index
    sel = order[:, :k]
    wk = np.take_along_axis(scores, sel, axis=-1)
    if cfg.norm_topk_prob:
        wk = wk / (wk.sum(axis=-1, keepdims=True) + 1.0e-20)
    wk = wk * cfg.routed_scaling_factor
    routed = np.zeros((h.shape[0], cfg.hidden_size))
    for b in range(h.shape[0]):
        acc = np.zeros(cfg.hidden_size)
        for slot in range(k):
            e = int(sel[b, slot])
            gu = w.expert_gate_up[layer][e] @ h[b]
            gpt, upt = np.split(gu, [Imoe])
            act = _swiglu(gpt, upt, cfg.swiglu_limit)
            acc = acc + wk[b, slot] * (w.expert_down[layer][e] @ act)
        routed[b] = acc
    # shared dense expert
    sh = w.shared_down[layer] @ _swiglu(w.shared_gate[layer] @ h.T, w.shared_up[layer] @ h.T, cfg.swiglu_limit).T
    return routed + sh.T


def glm53_reference_decode_step(
    cfg: Glm53Config, w: Glm53Weights,
    streams: np.ndarray,  # [B, hc, C]
    conv_states: list, ssm_states: list,  # per-layer pools (mutated)
) -> np.ndarray:
    """One decode step over the whole stack; returns the normed hidden [B, C]."""
    hidden = streams
    for layer in range(cfg.num_hidden_layers):
        # attention sub-block
        residual = hidden
        post, comb, h_in = _mhc_block(hidden, w.hc_attn_fn[layer], w.hc_attn_base[layer], w.hc_attn_scale[layer], cfg)
        h = _rms_norm(h_in, w.ln1[layer], cfg.rms_norm_eps)
        body = _kda_layer_decode(h, w, cfg, layer, conv_states[layer], ssm_states[layer])
        hidden = _mhc_compose(residual, post, comb, body)
        # ffn sub-block
        residual = hidden
        post, comb, h_in = _mhc_block(hidden, w.hc_ffn_fn[layer], w.hc_ffn_base[layer], w.hc_ffn_scale[layer], cfg)
        h = _rms_norm(h_in, w.ln2[layer], cfg.rms_norm_eps)
        if cfg.mlp_layer_types[layer] == "dense":
            body = w.down_proj[layer] @ _swiglu(w.gate_proj[layer] @ h.T, w.up_proj[layer] @ h.T, cfg.swiglu_limit).T
            body = body.T
        else:
            body = _moe_block(h, w, cfg, layer)
        hidden = _mhc_compose(residual, post, comb, body)
    # HyperHead: unweighted mean over hc, then final norm
    head = hidden.mean(axis=1)
    return _rms_norm(head, w.final_norm[0], cfg.rms_norm_eps)
