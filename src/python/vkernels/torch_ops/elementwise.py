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

Every public op also validates its own per-call eligibility (CUDA residency,
dtype, shape agreement — the ``torch_ops`` dispatch convention) and raises
:class:`OpNotEligible` when the fused launch cannot take the inputs, so
callers fall back to the eager path instead of pre-gating on hardware.
"""

from functools import lru_cache

from ._dispatch import require


def _same_gpu(*tensors):
    """Shared eligibility floor: every tensor CUDA-resident on one device."""
    require(tensors[0].is_cuda,
            "inputs must be CUDA-resident (CPU callers take the *_reference/eager path)")
    require(all(t.device == tensors[0].device for t in tensors),
            "inputs must share one GPU device")


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

    @triton.jit
    def _qk_norm_rope(
        Q,
        K,
        WQ,
        WK,
        C,
        S,
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
        mask = col < D
        other = (col + D // 2) % D
        c = tl.load(C + row * D + col, mask, 0).to(tl.float32)
        s = tl.load(S + row * D + col, mask, 0).to(tl.float32)
        sign = tl.where(col < D // 2, -1.0, 1.0)
        if head < HQ:
            x = tl.load(Q + row * QS + head * D + col, mask, 0).to(tl.float32)
            xp = tl.load(Q + row * QS + head * D + other, mask, 0).to(tl.float32)
            w = tl.load(WQ + col, mask, 0).to(tl.float32)
            # the PAIRED element keeps ITS OWN norm weight (the chain norms
            # every element with its own column weight before rotary), so
            # yp's weight loads at `other`, not at `col`.
            wp = tl.load(WQ + other, mask, 0).to(tl.float32)
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + QE)
            y = ((x * inv).to(Q.dtype.element_ty).to(tl.float32) * w).to(Q.dtype.element_ty).to(tl.float32)
            yp = ((xp * inv).to(Q.dtype.element_ty).to(tl.float32) * wp).to(Q.dtype.element_ty).to(tl.float32)
            a = (y * c).to(Q.dtype.element_ty).to(tl.float32)
            b = (yp * sign * s).to(Q.dtype.element_ty).to(tl.float32)
            tl.store(OQ + (row * HQ + head) * D + col, a + b, mask)
        else:
            h = head - HQ
            x = tl.load(K + row * KS + h * D + col, mask, 0).to(tl.float32)
            xp = tl.load(K + row * KS + h * D + other, mask, 0).to(tl.float32)
            w = tl.load(WK + col, mask, 0).to(tl.float32)
            wp = tl.load(WK + other, mask, 0).to(tl.float32)
            inv = tl.rsqrt(tl.sum(x * x, 0) / D + KE)
            y = ((x * inv).to(K.dtype.element_ty).to(tl.float32) * w).to(K.dtype.element_ty).to(tl.float32)
            yp = ((xp * inv).to(K.dtype.element_ty).to(tl.float32) * wp).to(K.dtype.element_ty).to(tl.float32)
            a = (y * c).to(K.dtype.element_ty).to(tl.float32)
            b = (yp * sign * s).to(K.dtype.element_ty).to(tl.float32)
            tl.store(OK + (row * HK + h) * D + col, a + b, mask)

    @triton.jit
    def _norm_uw(
        X,
        Y,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < D
        x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
        # The eager form is `x * rsqrt(...).to(dtype)`: the inverse is rounded
        # to the storage dtype BEFORE the multiply, so round here too.
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS).to(Y.dtype.element_ty).to(tl.float32)
        tl.store(Y + row * D + col, (x * inv).to(Y.dtype.element_ty), mask)

    @triton.jit
    def _norm_gated(
        X,
        W,
        G,
        Y,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < D
        x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
        w = tl.load(W + col, mask, 0).to(tl.float32)
        gate = tl.load(G + row * D + col, mask, 0).to(tl.float32)
        # strict fp32 throughout, one rounding on store -- and the same
        # left-to-right multiply order as the eager reference
        tl.store(Y + row * D + col, (((x * inv) * w) * tl.sigmoid(gate)).to(Y.dtype.element_ty), mask)

    return _norm, _qk_norm, _rope, _silu_mul, _store_kv, _qk_norm_rope, _norm_uw, _norm_gated


def rms_norm(x, module, residual=None):
    """Return (normalized value, rounded residual sum), without aliasing writes.

    Eligibility: CUDA bf16/fp16 ``x`` whose ``module.weight`` shares its dtype
    (the eager expression's output dtype follows the weight's, so a mismatched
    weight would change the result dtype, not just the speed) and an optional
    ``residual`` of the same shape/dtype/device. Anything else raises
    :class:`OpNotEligible` — callers fall back to :func:`rms_norm_reference`.
    """
    import torch

    _same_gpu(x, module.weight, *((residual,) if residual is not None else ()))
    require(x.dtype in (torch.float32, torch.bfloat16, torch.float16),
            f"rms_norm runs fp32/bf16/fp16, got {x.dtype}")
    require(module.weight.dtype == x.dtype,
            f"weight dtype {module.weight.dtype} must match x ({x.dtype})")
    require(module.weight.is_contiguous(), "weight must be contiguous")
    if residual is not None:
        require(residual.shape == x.shape and residual.dtype == x.dtype,
                "residual must match x's shape and dtype")
    import triton  # lazy: launches need triton only

    x = x.contiguous()
    out = torch.empty_like(x)
    summed = x if residual is None else torch.empty_like(x)
    r = x if residual is None else residual.contiguous()
    d = x.shape[-1]
    norm = _kernels()[0]
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
    """Per-head QK-RMSNorm in one launch over both projections.

    Eligibility: CUDA bf16/fp16 ``q``/``k`` sharing device, token grid and
    head dim; norm weights contiguous. Raises :class:`OpNotEligible`
    otherwise (the eager two-norm chain is the fallback).
    """
    import torch

    _same_gpu(q, k, q_norm.weight, k_norm.weight)
    require(q.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and k.dtype in (torch.float32, torch.bfloat16, torch.float16),
            f"qk_norm runs fp32/bf16/fp16, got {q.dtype}/{k.dtype}")
    require(q.shape[:-2] == k.shape[:-2] and q.shape[-1] == k.shape[-1],
            "q and k must share their token grid and head dim")
    require(q_norm.weight.is_contiguous() and k_norm.weight.is_contiguous(),
            "norm weights must be contiguous")
    import triton  # lazy: launches need triton only

    d, hq, hk = q.shape[-1], q.shape[-2], k.shape[-2]
    q3, k3 = q.reshape(-1, hq, d), k.reshape(-1, hk, d)
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    qk_n = _kernels()[1]
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
    """Half-rotation rotary embedding over every head in one launch.

    Eligibility: CUDA ``q``/``k``/``cos``/``sin`` on one device, ``q``/``k``
    bf16/fp16 sharing their token grid and (even) head dim, ``cos``/``sin``
    contiguous with one row per token. Raises :class:`OpNotEligible`
    otherwise (:func:`rotary_reference` is the eager form).
    """
    import torch

    _same_gpu(q, k, cos, sin)
    require(q.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and k.dtype in (torch.float32, torch.bfloat16, torch.float16),
            f"rotary runs fp32/bf16/fp16, got {q.dtype}/{k.dtype}")
    require(q.shape[:-2] == k.shape[:-2] and q.shape[-1] == k.shape[-1]
            and q.shape[-1] % 2 == 0,
            "q and k must share their token grid and an even head dim")
    d = q.shape[-1]
    tokens = q.numel() // (q.shape[-2] * d)
    require(cos.is_contiguous() and sin.is_contiguous()
            and cos.numel() == tokens * d and sin.numel() == tokens * d,
            "cos/sin must be contiguous with one row per token")
    import triton  # lazy: launches need triton only

    d, hq, hk = q.shape[-1], q.shape[-2], k.shape[-2]
    q3, k3 = q.reshape(-1, hq, d), k.reshape(-1, hk, d)
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    rope = _kernels()[2]
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


def qk_norm_rope(q, k, q_norm, k_norm, cos, sin):
    """Fused per-head QK-RMSNorm + rotary embedding (one kernel).

    Equivalent to :func:`qk_norm` followed by :func:`rotary` on the normed
    tensors (the rounding sequence matches: norm result is rounded to the
    input dtype before the rope multiply, then each rope half is rounded
    before the add). Removes one launch and one full read+write of q/k
    per layer on the decode hot path.

    Eligibility: the :func:`qk_norm` constraints plus :func:`rotary`'s
    cos/sin constraints; raises :class:`OpNotEligible` otherwise (the eager
    two-kernel chain is the fallback).
    """
    import torch
    import triton  # lazy: launches need triton only

    _same_gpu(q, k, q_norm.weight, k_norm.weight, cos, sin)
    require(q.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and k.dtype in (torch.float32, torch.bfloat16, torch.float16),
            f"qk_norm_rope runs fp32/bf16/fp16, got {q.dtype}/{k.dtype}")
    require(q.shape[:-2] == k.shape[:-2] and q.shape[-1] == k.shape[-1]
            and q.shape[-1] % 2 == 0,
            "q and k must share their token grid and an even head dim")
    require(q_norm.weight.is_contiguous() and k_norm.weight.is_contiguous(),
            "norm weights must be contiguous")
    d = q.shape[-1]
    tokens = q.numel() // (q.shape[-2] * d)
    require(cos.is_contiguous() and sin.is_contiguous()
            and cos.numel() == tokens * d and sin.numel() == tokens * d,
            "cos/sin must be contiguous with one row per token")

    d, hq, hk = q.shape[-1], q.shape[-2], k.shape[-2]
    q3, k3 = q.reshape(-1, hq, d), k.reshape(-1, hk, d)
    oq = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    ok = torch.empty(k.shape, device=k.device, dtype=k.dtype)
    qk_rope = _kernels()[5]
    qk_rope[(q3.shape[0], hq + hk)](
        q3,
        k3,
        q_norm.weight,
        k_norm.weight,
        cos.contiguous(),
        sin.contiguous(),
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


def silu_mul(gate, up):
    """SiLU(gate) * up in one launch (SiLU rounds before the product).

    Eligibility: CUDA bf16/fp16 ``gate``/``up`` of the same shape and dtype
    on one device; raises :class:`OpNotEligible` otherwise.
    """
    import torch
    import triton  # lazy: launches need triton only

    _same_gpu(gate, up)
    require(gate.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and up.dtype == gate.dtype,
            f"silu_mul runs fp32/bf16/fp16 with matching dtypes, got {gate.dtype}/{up.dtype}")
    require(up.shape == gate.shape, "gate and up must share a shape")

    d = gate.shape[-1]
    g, u = gate.reshape(-1, d), up.reshape(-1, d)
    out = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    sm = _kernels()[3]
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
    """Resolve virtual slots and scatter K and V in one graph-safe launch.

    Eligibility: CUDA-resident tensors on one device; ``kc``/``vc`` pools
    contiguous with the incoming ``k``/``v`` dtypes (the scatter is a raw
    copy — a dtype mismatch would corrupt the pool, so it is rejected, not
    launched); integer ``block_table``/``seqlens``. Raises
    :class:`OpNotEligible` otherwise.
    """
    import triton  # lazy: launches need triton only

    _same_gpu(k, v, kc, vc, block_table, seqlens)
    require(k.dim() == 4, "expected k/v [B, n, H, D]")
    require(kc.dtype == k.dtype and vc.dtype == v.dtype,
            "cache pools must store k/v's dtype (raw copy)")
    # kc/vc are indexed with raw pointer math (no stride args) — contiguous
    # pools only. block_table may be a column slice of a wider table: the
    # kernel takes its row stride, but walks the slot dimension with a unit
    # inner stride; seqlens' 1-D stride is a kernel parameter.
    require(kc.is_contiguous() and vc.is_contiguous(),
            "cache pools must be contiguous")
    require(block_table.stride(-1) == 1,
            "block_table must be unit-stride in the slot dimension")
    require(block_table.dtype.is_floating_point is False
            and seqlens.dtype.is_floating_point is False,
            "block_table and seqlens must be integer tensors")

    b, n, h, d = k.shape
    k3, v3 = k.reshape(-1, h, d), v.reshape(-1, h, d)
    # Token count is runtime data, not a JIT specialization: captured decode
    # initializes the same kernel used for arbitrary-length fresh prefills.
    sk = _kernels()[4]
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


def rms_norm_unweighted(x, module):
    """RMSNorm without a learned scale (mHC ``input_norm``), returning a new tensor.

    Eligibility: CUDA bf16/fp16 ``x``; raises :class:`OpNotEligible`
    otherwise (:func:`rms_norm_unweighted_reference` is the eager form).
    """
    import torch
    import triton  # lazy: launches need triton only

    _same_gpu(x)
    require(x.dtype in (torch.float32, torch.bfloat16, torch.float16),
            f"rms_norm_unweighted runs fp32/bf16/fp16, got {x.dtype}")

    x = x.contiguous()
    out = torch.empty_like(x)
    d = x.shape[-1]
    *_, unw, _gated = _kernels()
    unw[(x.numel() // d,)](
        x,
        out,
        d,
        module.eps,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return out


def rms_norm_gated(x, gate, module):
    """``weight * norm(x) * sigmoid(gate)`` — one launch, strict fp32 math.

    ``gate`` must match ``x`` exactly (same shape — the kernel does not
    broadcast). Eligibility: CUDA bf16/fp16 ``x``/``gate`` of one shape and
    dtype on one device, contiguous ``module.weight``; raises
    :class:`OpNotEligible` otherwise
    (:func:`rms_norm_gated_reference` is the eager form).
    """
    import torch
    import triton  # lazy: launches need triton only

    _same_gpu(x, gate, module.weight)
    require(x.dtype in (torch.float32, torch.bfloat16, torch.float16),
            f"rms_norm_gated runs fp32/bf16/fp16, got {x.dtype}")
    require(gate.shape == x.shape and gate.dtype == x.dtype,
            "gate must match x's shape and dtype (no broadcast)")
    require(module.weight.is_contiguous(), "weight must be contiguous")

    x = x.contiguous()
    gate = gate.contiguous()
    out = torch.empty_like(x)
    d = x.shape[-1]
    *_, _unw, gated = _kernels()
    gated[(x.numel() // d,)](
        x,
        module.weight,
        gate,
        out,
        d,
        module.variance_epsilon,
        triton.next_power_of_2(d),
        enable_fp_fusion=False,
    )
    return out


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


def rms_norm_unweighted_reference(x, module):
    """Eager HF oracle for :func:`rms_norm_unweighted` (exact intermediate rounding)."""
    import torch

    inv = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + module.eps).to(x.dtype)
    return x * inv


def rms_norm_gated_reference(x, gate, module):
    """Eager HF oracle for :func:`rms_norm_gated` (exact intermediate rounding)."""
    import torch

    x32 = x.float()
    inv = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
    y = (x32 * inv) * module.weight.float()
    return (y * torch.sigmoid(gate.float())).to(x.dtype)


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
