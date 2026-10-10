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
    return x * _rsqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gamma


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _rsqrt(x: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(x)


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


class _LazyWeight:
    """Shape-only stand-in for the huge MoE weight tensors. The real-dims
    compile smoke never executes on the host — 288 experts × [2·Imoe, C]
    fp32 is ~18 GB per layer — so only the shape is consulted."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape

    def astype(self, *a, **k):  # pragma: no cover - guard
        raise RuntimeError("lazy weight materialized outside compile-only mode")


def random_glm53_weights(
    cfg: Glm53Config, seed: int = 20260917, *, real_experts: bool = True, realize: bool = True
) -> Glm53Weights:
    """Random GLM-5.3 weights. ``realize=False`` returns lazy ``np.zeros``
    placeholders with the real shapes instead of drawn values — the
    real-dims compile smoke (~20 GB of fp64 weight draws at the published
    dims, 12 layers) only exercises shapes, never values; lazy zeros keep
    the compile-only path at ~0 RSS (untouched calloc pages)."""
    rng = np.random.default_rng(seed)

    def n(*shape: int, s: float = 0.05) -> np.ndarray:
        return rng.normal(0.0, s, size=shape) if realize else np.zeros(shape)

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
        expert_gate_up=[n(E, 2 * Imoe, C) if real_experts else _LazyWeight((E, 2 * Imoe, C)) for _ in range(L)],
        expert_down=[n(E, C, Imoe) if real_experts else _LazyWeight((E, C, Imoe)) for _ in range(L)],
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
    flat_n = flat * _rsqrt(np.mean(flat * flat, axis=-1, keepdims=True) + rms_eps)
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
    H, D = cfg.linear_num_heads, cfg.linear_head_dim
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
    kv = np.einsum("bhkv,bhk->bhv", s, k_n)
    delta = beta[:, :, None] * (v - kv)
    s = s + k_n[:, :, :, None] * delta[:, :, None, :]
    out = np.einsum("bhkv,bhk->bhv", s, q_n)
    ssm_state[:] = s
    # gated output norm (per head row) + o_proj
    gate = (h @ w.g_a[layer].T @ w.g_b[layer].T).reshape(B, H, D)
    on = _rms_norm(out, w.o_norm[layer], cfg.rms_norm_eps) * _sigmoid(gate)
    return on.reshape(B, qkv) @ w.o_proj[layer].T


def _moe_block(
    h: np.ndarray, w: Glm53Weights, cfg: Glm53Config, layer: int,
) -> np.ndarray:
    Imoe = cfg.moe_intermediate_size
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
    act = _swiglu(w.shared_gate[layer] @ h.T, w.shared_up[layer] @ h.T, cfg.swiglu_limit)  # [I, B]
    sh = w.shared_down[layer] @ act  # [C, B]
    return routed + sh.T


def kda_fused_decode_reference(
    conv_pool: np.ndarray,  # [slots, Kw, Cc] time-major (w-major) pool, bf16-grid values
    ssm_pool: np.ndarray,  # [slots, H, V, K] V-MAJOR fp32 pool
    slot_ids: np.ndarray,  # i32 [B]; -1 = padded slot
    qkv_raw: np.ndarray,  # [B, Cc] RAW pre-conv fused-projection rows (bf16 round at entry)
    f_raw: np.ndarray,  # [B, H, K] RAW f_b(f_a(x)) dots
    b_raw: np.ndarray,  # [B, H] RAW b_proj dots
    g_raw: np.ndarray,  # [B, H, V] RAW g_b(g_a(x)) o-norm gate dots
    taps: np.ndarray,  # [Kt, Cc] TIME-MAJOR fp32 conv taps (q|k|v channel segments)
    dt_bias: np.ndarray,  # [H, K]
    A_log: np.ndarray,  # [H]
    o_norm: np.ndarray,  # [V] shared across heads
    *,
    scale: float,
    eps: float,
    lower_bound: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """fp64 mirror of the E1 fused-decode contract (one layer's decode op).

    Pins EXACTLY the semantics documented by the CUDA oracle
    ``vkernels.torch_ops.glm_kda_fused_decode`` (the correctness oracle for
    the megakernel's KDA tasks), in the kernel's ABI and op order:

    * RAW dot contract — the op consumes the RAW pre-conv fused-projection
      row, the RAW ``f_b``/``b_proj``/``g_b`` dots; each is rounded to the
      bf16 grid at task entry (the kernel's bf16 ABI), and every gate
      nonlinearity (beta, o-norm gate) is applied AFTER that round;
    * conv: fp32 time-major taps over the bf16-valued w-major window
      ``[pool | x]``, SiLU in fp32; the conv output is NOT rounded to bf16
      before the recurrence (the incumbent chain's conv store rounds — that
      bf16-round class is the documented cross-path divergence, ~1e-2 rel);
      the pool shift ``[old w1, old w2, new raw x]`` is pure bf16 moves;
      floe's conv1d is bias-free (the CUDA ABI's ``conv_bias`` is zeros);
    * ssm pool V-MAJOR ``[slots, H, V, K]`` fp32 (rows = v, cols = k — the
      transpose of the decomposed ``kda_delta`` op's [B, H, K, V] pool;
      layout only, no numeric divergence), decay per (head, k) broadcast
      over v-rows, delta rule, plain readout;
    * gated per-head RMSNorm with the [V] weight shared across heads and
      the sigmoid gate applied after the bf16 round; the output row is
      stored on the bf16 grid (the kernel's out ABI);
    * ``-1`` slot ids are padded slots: zero output row, both pools
      untouched. Duplicate live slot ids in one step are UB (same contract
      as the fused kernel).

    Returns fresh ``(out [B, H, V], conv_pool', ssm_pool')`` — the caller's
    pools are never mutated (the executor mutates in place; this mirror
    copies so both stay comparable after the fact).
    """
    from .reference_types import bf16_round

    B, H = b_raw.shape
    Cc = conv_pool.shape[2]
    D = V = ssm_pool.shape[-1]
    seg = Cc // 3
    conv = conv_pool.copy()
    ssm = ssm_pool.copy()
    out = np.zeros((B, H, V), np.float64)
    for b in range(B):
        slot = int(slot_ids[b])
        if slot < 0:
            continue  # padded graph slot: output row stays zero, pools untouched
        x = bf16_round(qkv_raw[b])  # [Cc] on the bf16 grid
        acc_all = np.einsum("wc,wc->c", taps, np.concatenate([conv[slot], x[None, :]], axis=0))
        y = acc_all * _sigmoid(acc_all)  # SiLU, fp64 (fp32 on device), NOT rounded
        conv[slot] = np.concatenate([conv[slot][1:], x[None, :]], axis=0)  # pure bf16 moves
        for h in range(H):
            q = y[h * D : (h + 1) * D]
            k = y[seg + h * D : seg + (h + 1) * D]
            v = y[2 * seg + h * D : 2 * seg + (h + 1) * D]
            xx = bf16_round(f_raw[b, h]) + dt_bias[h]
            A = np.exp(A_log[h])
            if lower_bound is not None:
                decay = np.exp(lower_bound * _sigmoid(A * xx))  # [K] log-space
            else:
                sp = np.where(xx <= 20.0, np.log1p(np.exp(np.minimum(xx, 20.0))), xx)
                decay = np.exp(-A * sp)
            beta = _sigmoid(bf16_round(b_raw[b, h]))  # sigmoid AFTER the bf16 round
            qn = q / np.sqrt((q * q).sum() + 1e-6) * scale
            kn = k / np.sqrt((k * k).sum() + 1e-6)
            s = ssm[slot, h] * decay[None, :]  # [V, K], decay broadcast over v-rows
            t = (s * kn[None, :]).sum(axis=1)  # t[v] = sum_k s[v,k]*kn[k]
            delta = (v - t) * beta
            s = s + delta[:, None] * kn[None, :]
            ssm[slot, h] = s  # fp32 pool store class
            o = (s * qn[None, :]).sum(axis=1)  # o[v] = sum_k s[v,k]*qn[k]
            rstd = 1.0 / np.sqrt((o * o).mean() + eps)
            out[b, h] = bf16_round(o * rstd * o_norm * _sigmoid(bf16_round(g_raw[b, h])))
    return out, conv, ssm


def glm53_reference_kda_spine_step(
    cfg: Glm53Config,
    w: Glm53Weights,
    streams: np.ndarray,  # [B, hc, C]
    conv_pools: list,  # per-layer [slots, Kw, Cc] (NOT mutated; returns updated)
    ssm_pools: list,  # per-layer [slots, H, V, K] V-major
    slot_ids: np.ndarray | None = None,  # i32 [B]; default identity
) -> np.ndarray:
    """E1 KDA-spine mirror: the 34-KDA-layer attention sub-blocks chained
    through the mHC streams, each layer's decode block under the FUSED
    contract (``kda_fused_decode_reference``).

    The spine is the E1 slice: ``mhc_pre → ln1 → fused qkv/f/b/g GEMVs →
    fused decode (conv + kda-delta + gated norm) → o_proj → mhc_post`` per
    layer, no FFN sub-block (that is E3's slice) — the exact body the
    compiled spine (``build_glm53_kda_spine_forward``) captures. Returns
    the post-spine streams ``[B, hc, C]`` and the updated pool lists.
    """
    B = streams.shape[0]
    H, D = cfg.linear_num_heads, cfg.linear_head_dim
    qkv = cfg.qkv_dim
    if slot_ids is None:
        slot_ids = np.arange(B, dtype=np.int32)
    hidden = streams
    new_conv, new_ssm = [], []
    for layer in range(cfg.num_hidden_layers):
        post, comb, h_in = _mhc_block(
            hidden, w.hc_attn_fn[layer], w.hc_attn_base[layer], w.hc_attn_scale[layer], cfg
        )
        h = _rms_norm(h_in, w.ln1[layer], cfg.rms_norm_eps)
        # RAW dots (fp64 here; the fused op rounds to bf16 at entry)
        mixed = h @ w.qkv_proj[layer].T
        f_raw = (h @ w.f_a[layer].T @ w.f_b[layer].T).reshape(B, H, D)
        b_raw = h @ w.b_proj[layer].T
        g_raw = (h @ w.g_a[layer].T @ w.g_b[layer].T).reshape(B, H, D)
        fused, conv_pools[layer], ssm_pools[layer] = kda_fused_decode_reference(
            conv_pools[layer], ssm_pools[layer], slot_ids,
            mixed, f_raw, b_raw, g_raw,
            w.conv_w[layer].T.copy(),  # [C, Kt] -> [Kt, C] time-major taps
            w.dt_bias[layer].reshape(H, D), w.A_log[layer], w.o_norm[layer],
            scale=D ** -0.5, eps=cfg.rms_norm_eps, lower_bound=cfg.linear_lower_bound,
        )
        new_conv.append(conv_pools[layer])
        new_ssm.append(ssm_pools[layer])
        body = fused.reshape(B, qkv) @ w.o_proj[layer].T
        hidden = _mhc_compose(hidden, post, comb, body)
    conv_pools[:] = new_conv
    ssm_pools[:] = new_ssm
    return hidden


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
            act = _swiglu(w.gate_proj[layer] @ h.T, w.up_proj[layer] @ h.T, cfg.swiglu_limit)  # [I, B]
            body = (w.down_proj[layer] @ act).T  # [B, C]
        else:
            body = _moe_block(h, w, cfg, layer)
        hidden = _mhc_compose(residual, post, comb, body)
    # HyperHead: unweighted mean over hc, then final norm
    head = hidden.mean(axis=1)
    return _rms_norm(head, w.final_norm[0], cfg.rms_norm_eps)
