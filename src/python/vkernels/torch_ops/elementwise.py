"""Inference-only Triton fusions for standard HF Llama/Qwen RMSNorm blocks.

Keep the low-precision intermediate roundings of the eager HF expressions.
In particular, RMSNorm casts before multiplying its weight, RoPE rounds each
product, and SiLU rounds before multiplying the up projection. No input (or
residual) is mutated, so graph padding and projection views remain safe.

Adopted from floe's ``engine/runner/kernels/elementwise.py`` (vkernels owns
the kernels; floe imports them back through a thin adapter — the #64/#65
thin-adapter model). Torch and Triton load lazily. Inference-only, no
autograd backward.

``supports_model`` is the fail-closed capability gate for these fusions and
is deliberately owned here, next to the kernels it gates: it encodes which
HF blocks/heads the kernels cover (and rejects Gemma-style offset-weight
norms), keeping floe's adapter thin. Extend it whenever ``transformers``
renames or restructures the covered classes.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _norm(
        X,
        W,
        R,
        Y,
        SUM,
        D: tl.constexpr,
        EPS: tl.constexpr,
        ADD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
        if ADD:
            r = tl.load(R + row * D + col, col < D, 0).to(tl.float32)
            x = (x + r).to(X.dtype.element_ty).to(tl.float32)
            tl.store(SUM + row * D + col, x, col < D)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        w = tl.load(W + col, col < D, 0).to(tl.float32)
        y = (x * inv).to(X.dtype.element_ty).to(tl.float32) * w
        tl.store(Y + row * D + col, y, col < D)

    @triton.jit
    def _qk_norm(
        Q,
        K,
        WQ,
        WK,
        OQ,
        OK,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        QS: tl.constexpr,
        KS: tl.constexpr,
        QE: tl.constexpr,
        KE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, BLOCK)
        if head < HQ:
            x = tl.load(Q + row * QS + head * D + col, col < D, 0).to(tl.float32)
            w = tl.load(WQ + col, col < D, 0).to(tl.float32)
            y = (x * tl.rsqrt(tl.sum(x * x, 0) / D + QE)).to(Q.dtype.element_ty)
            tl.store(OQ + (row * HQ + head) * D + col, y.to(tl.float32) * w, col < D)
        else:
            h = head - HQ
            x = tl.load(K + row * KS + h * D + col, col < D, 0).to(tl.float32)
            w = tl.load(WK + col, col < D, 0).to(tl.float32)
            y = (x * tl.rsqrt(tl.sum(x * x, 0) / D + KE)).to(K.dtype.element_ty)
            tl.store(OK + (row * HK + h) * D + col, y.to(tl.float32) * w, col < D)

    @triton.jit
    def _rope(
        Q,
        K,
        C,
        S,
        OQ,
        OK,
        HQ: tl.constexpr,
        HK: tl.constexpr,
        D: tl.constexpr,
        QS: tl.constexpr,
        KS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, BLOCK)
        other = (col + D // 2) % D
        c = tl.load(C + row * D + col, col < D, 0).to(tl.float32)
        s = tl.load(S + row * D + col, col < D, 0).to(tl.float32)
        sign = tl.where(col < D // 2, -1.0, 1.0)
        if head < HQ:
            x = tl.load(Q + row * QS + head * D + col, col < D, 0).to(tl.float32)
            r = tl.load(Q + row * QS + head * D + other, col < D, 0).to(tl.float32)
            a = (x * c).to(Q.dtype.element_ty).to(tl.float32)
            b = (r * sign * s).to(Q.dtype.element_ty).to(tl.float32)
            tl.store(OQ + (row * HQ + head) * D + col, a + b, col < D)
        else:
            h = head - HQ
            x = tl.load(K + row * KS + h * D + col, col < D, 0).to(tl.float32)
            r = tl.load(K + row * KS + h * D + other, col < D, 0).to(tl.float32)
            a = (x * c).to(K.dtype.element_ty).to(tl.float32)
            b = (r * sign * s).to(K.dtype.element_ty).to(tl.float32)
            tl.store(OK + (row * HK + h) * D + col, a + b, col < D)

    @triton.jit
    def _silu_mul(
        G,
        U,
        Y,
        D: tl.constexpr,
        GS: tl.constexpr,
        US: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        g = tl.load(G + row * GS + col, col < D, 0).to(tl.float32)
        u = tl.load(U + row * US + col, col < D, 0).to(tl.float32)
        act = (g / (1.0 + tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * D + col, act * u, col < D)

    @triton.jit(do_not_specialize=["N"])
    def _store_kv(
        K,
        V,
        KC,
        VC,
        BT,
        SL,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        KS: tl.constexpr,
        VS: tl.constexpr,
        BTS: tl.constexpr,
        SLS: tl.constexpr,
        SCRATCH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch, token, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        row = batch * N + token
        col = tl.arange(0, BLOCK)
        pos = tl.load(SL + batch * SLS) + token
        slot = tl.load(BT + batch * BTS + pos).to(tl.int64)
        # Padded requests/tokens may all reference the same null page. Never
        # race writes there, or turn the shared zero page into a real KV token.
        mask = (col < D) & (slot != SCRATCH)
        k = tl.load(K + row * KS + head * D + col, mask, 0)
        v = tl.load(V + row * VS + head * D + col, mask, 0)
        tl.store(KC + (slot * H + head) * D + col, k, mask)
        tl.store(VC + (slot * H + head) * D + col, v, mask)

    return _norm, _qk_norm, _rope, _silu_mul, _store_kv


def rms_norm(x, module, residual=None):
    """Return (normalized value, rounded residual sum), without aliasing writes."""
    import triton  # lazy: launches need triton only
    import torch

    x = x.contiguous()
    out = torch.empty_like(x)
    summed = x if residual is None else torch.empty_like(x)
    r = x if residual is None else residual.contiguous()
    d = x.shape[-1]
    norm, _, _, _, _ = _kernels()
    norm[(x.numel() // d,)](
        x,
        module.weight,
        r,
        out,
        summed,
        d,
        module.variance_epsilon,
        residual is not None,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return out, summed


def qk_norm(q, k, q_norm, k_norm):
    import triton  # lazy: launches need triton only
    import torch

    d, hq, hk = q.shape[-1], q.shape[-2], k.shape[-2]
    q3, k3 = q.reshape(-1, hq, d), k.reshape(-1, hk, d)
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _, qk_n, _, _, _ = _kernels()
    qk_n[(q3.shape[0], hq + hk)](
        q3,
        k3,
        q_norm.weight,
        k_norm.weight,
        oq,
        ok,
        hq,
        hk,
        d,
        q3.stride(0),
        k3.stride(0),
        q_norm.variance_epsilon,
        k_norm.variance_epsilon,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return oq, ok


def rotary(q, k, cos, sin):
    import triton  # lazy: launches need triton only
    import torch

    d, hq, hk = q.shape[-1], q.shape[-2], k.shape[-2]
    q3, k3 = q.reshape(-1, hq, d), k.reshape(-1, hk, d)
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    _, _, rope, _, _ = _kernels()
    rope[(q3.shape[0], hq + hk)](
        q3,
        k3,
        cos.contiguous(),
        sin.contiguous(),
        oq,
        ok,
        hq,
        hk,
        d,
        q3.stride(0),
        k3.stride(0),
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return oq, ok


def silu_mul(gate, up):
    import triton  # lazy: launches need triton only
    import torch

    d = gate.shape[-1]
    g, u = gate.reshape(-1, d), up.reshape(-1, d)
    out = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    _, _, _, sm, _ = _kernels()
    sm[(g.shape[0], triton.cdiv(d, 256))](
        g,
        u,
        out,
        d,
        g.stride(0),
        u.stride(0),
        256,
        enable_fp_fusion=False,
    )
    return out


def store_kv(k, v, kc, vc, block_table, seqlens, scratch_slot):
    """Resolve virtual slots and scatter K and V in one graph-safe launch."""
    import triton  # lazy: launches need triton only

    b, n, h, d = k.shape
    k3, v3 = k.reshape(-1, h, d), v.reshape(-1, h, d)
    # Token count is runtime data, not a JIT specialization: captured decode
    # initializes the same kernel used for arbitrary-length fresh prefills.
    _, _, _, _, sk = _kernels()
    sk[(b, n, h)](
        k3,
        v3,
        kc,
        vc,
        block_table,
        seqlens,
        n,
        h,
        d,
        k3.stride(0),
        v3.stride(0),
        block_table.stride(0),
        seqlens.stride(0),
        scratch_slot,
        triton.next_power_of_2(d),
    )


def rms_norm_reference(x, module, residual=None):
    """Eager HF oracle for :func:`rms_norm` (exact intermediate rounding)."""
    import torch

    x32 = x.float()
    if residual is None:
        summed = x
    else:
        x32 = (x32 + residual.float()).to(x.dtype).float()
        summed = x32.to(x.dtype)
    inv = torch.rsqrt(
        x32.pow(2).sum(-1, keepdim=True) / x.shape[-1] + module.variance_epsilon
    )
    y = ((x32 * inv).to(x.dtype).float() * module.weight.float()).to(x.dtype)
    return y, summed


def qk_norm_reference(q, k, q_norm, k_norm):
    """Eager HF oracle for :func:`qk_norm`."""
    import torch

    def _one(t, w, eps):
        t32 = t.float()
        inv = torch.rsqrt(t32.pow(2).sum(-1, keepdim=True) / t.shape[-1] + eps)
        return ((t32 * inv).to(t.dtype).float() * w.float()).to(t.dtype)

    return _one(q, q_norm.weight, q_norm.variance_epsilon), _one(
        k, k_norm.weight, k_norm.variance_epsilon
    )


def rotary_reference(q, k, cos, sin):
    """Eager HF oracle for :func:`rotary` (rounded halves, rotated pairs)."""
    import torch

    d, hq, hk = q.shape[-1], q.shape[-2], k.shape[-2]
    q3, k3 = q.reshape(-1, hq, d).float(), k.reshape(-1, hk, d).float()
    other = torch.cat([torch.arange(d // 2, d), torch.arange(0, d // 2)])
    sign = torch.cat([-torch.ones(d // 2), torch.ones(d - d // 2)]).to(q.device)
    c, s = cos.float()[:, None, :], sin.float()[:, None, :]
    aq = (q3 * c).to(q.dtype).float()
    bq = (q3[:, :, other] * sign * s).to(q.dtype).float()
    ak = (k3 * c).to(k.dtype).float()
    bk = (k3[:, :, other] * sign * s).to(k.dtype).float()
    return (aq + bq).to(q.dtype).reshape(q.shape), (ak + bk).to(k.dtype).reshape(
        k.shape
    )


def silu_mul_reference(gate, up):
    """Eager HF oracle for :func:`silu_mul` (SiLU rounds before the product)."""
    import torch

    act = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(gate.dtype)
    return (act.float() * up.float()).to(gate.dtype)


def store_kv_reference(k, v, kc, vc, block_table, seqlens, scratch_slot):
    """Eager oracle for :func:`store_kv`; returns copies, inputs untouched."""
    b, n = k.shape[0], k.shape[1]
    kc, vc = kc.clone(), vc.clone()
    for bi in range(b):
        for t in range(n):
            slot = int(block_table[bi, int(seqlens[bi]) + t])
            if slot == scratch_slot:
                continue
            kc[slot] = k[bi, t]
            vc[slot] = v[bi, t]
    return kc, vc


def supports_model(model):
    """Fail closed: Gemma's offset-weight norms and custom blocks are not RMSNorm."""
    import torch

    norm_types = {
        (f"transformers.models.{name}.modeling_{name}", cls)
        for name, cls in (
            ("llama", "LlamaRMSNorm"),
            ("qwen2", "Qwen2RMSNorm"),
            ("qwen3", "Qwen3RMSNorm"),
        )
    }
    if model.config.model_type not in ("llama", "qwen2", "qwen3"):
        return False
    head_dim = getattr(model.config, "head_dim", None) or (
        model.config.hidden_size // model.config.num_attention_heads
    )
    norms = [model.model.norm]
    for layer in model.model.layers:
        mlp_type = type(layer.mlp)
        if (mlp_type.__module__, mlp_type.__name__) not in {
            ("transformers.models.llama.modeling_llama", "LlamaMLP"),
            ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
            ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
        }:
            return False
        if getattr(layer, "pre_feedforward_layernorm", None) is not None:
            return False
        norms.extend((layer.input_layernorm, layer.post_attention_layernorm))
        attn = layer.self_attn
        if getattr(attn, "q_norm", None) is not None:
            norms.extend((attn.q_norm, attn.k_norm))
        if getattr(attn, "rotary_dim", head_dim) != head_dim or head_dim % 2:
            return False
        activation = type(layer.mlp.act_fn)
        if activation is not torch.nn.SiLU and (
            activation.__module__,
            activation.__name__,
        ) != ("transformers.activations", "SiLUActivation"):
            return False
    return all(
        (type(n).__module__, type(n).__name__) in norm_types
        and n.weight.is_cuda
        and n.weight.is_contiguous()
        and n.weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and n.weight.dtype == model.model.embed_tokens.weight.dtype
        and n.weight.ndim == 1
        and n.weight.numel() <= 8192
        for n in norms
    )
