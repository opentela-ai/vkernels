"""GLM-5 mHC gate and Sinkhorn fusion (inference-only).

Computes the mHC control planes for contiguous ``[..., hc*(hc+2)]`` logits:
``pre = sigmoid(...)+eps``, ``post = 2*sigmoid(...)``, and the combiner
matrix normalized by row softmax + eps, column normalization, then
``sinkhorn_iters - 1`` row/column normalizations — matching
Glm53HyperConnection's ordering. GEMM and stream collapse remain with the
caller; all control math here uses FP32.

Adopted from floe's ``engine/runner/kernels/glm5_mhc.py`` (vkernels owns
the kernel; floe imports it back through a thin adapter — the #64/#65
thin-adapter model extended to the whole GLM-5 Triton set).

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

import math
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _mix(
        L,
        B,
        S,
        PRE,
        POST,
        COMB,
        HC: tl.constexpr,
        EPS: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        token = tl.program_id(0)
        k = tl.arange(0, HC)
        width: tl.constexpr = HC * (HC + 2)
        pre = tl.load(L + token * width + k) * tl.load(S) + tl.load(B + k)
        post = tl.load(L + token * width + HC + k) * tl.load(S + 1) + tl.load(
            B + HC + k
        )
        tl.store(PRE + token * HC + k, tl.sigmoid(pre) + EPS)
        tl.store(POST + token * HC + k, 2.0 * tl.sigmoid(post))
        offset = k[:, None] * HC + k[None, :]
        logits = tl.load(L + token * width + 2 * HC + offset) * tl.load(
            S + 2
        ) + tl.load(B + 2 * HC + offset)
        value = tl.exp(logits - tl.max(logits, 1)[:, None])
        value = tl.div_rn(value, tl.sum(value, 1)[:, None]) + EPS
        value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
        for _ in range(ITERS - 1):
            value = tl.div_rn(value, tl.sum(value, 1)[:, None] + EPS)
            value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
        tl.store(COMB + token * HC * HC + offset, value)

    return _mix


def mhc_mix(logits, base, scale, hc=4, eps=1e-6, sinkhorn_iters=20):
    """Return FP32 ``(pre, post, comb)`` for contiguous ``[..., hc*(hc+2)]``.

    Matches Glm53HyperConnection's ordering: row softmax + eps, column
    normalization, then ``sinkhorn_iters - 1`` row/column normalizations.
    This is an inference kernel without autograd support.
    """
    import torch

    if hc not in (2, 4):
        raise ValueError("supported hc values are 2 and 4")
    if not isinstance(sinkhorn_iters, int) or sinkhorn_iters < 1:
        raise ValueError("sinkhorn_iters must be a positive integer")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    width = hc * (hc + 2)
    if logits.ndim < 1 or logits.shape[-1] != width:
        raise ValueError("incorrect logits width")
    if base.shape != (width,) or scale.shape != (3,):
        raise ValueError("expected base [hc*(hc+2)] and scale [3]")
    for x in (logits, base, scale):
        if x.dtype != torch.float32:
            raise TypeError("mHC control tensors must be FP32")
        if not x.is_cuda or x.device != logits.device or not x.is_contiguous():
            raise ValueError("inputs must be contiguous tensors on the same GPU")
    pre = torch.empty(
        (*logits.shape[:-1], hc), device=logits.device, dtype=torch.float32
    )
    post = torch.empty_like(pre)
    comb = torch.empty(
        (*logits.shape[:-1], hc, hc), device=logits.device, dtype=torch.float32
    )
    tokens = logits.numel() // width
    if tokens:
        with torch.cuda.device(logits.device):
            _kernel()[(tokens,)](
                logits,
                base,
                scale,
                pre,
                post,
                comb,
                hc,
                eps,
                sinkhorn_iters,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return pre, post, comb


def mhc_mix_reference(logits, base, scale, hc=4, eps=1e-6, sinkhorn_iters=20):
    """Eager FP32 oracle mirroring the kernel's exact normalization order."""
    import torch

    planes = logits.reshape(*logits.shape[:-1], 2 * hc + hc * hc)
    pre = torch.sigmoid(planes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(planes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb_logits = planes[..., 2 * hc :].reshape(*logits.shape[:-1], hc, hc) * scale[
        2
    ] + base[2 * hc :].reshape(hc, hc)
    value = torch.exp(comb_logits - comb_logits.max(dim=-1, keepdim=True).values)
    value = value / value.sum(dim=-1, keepdim=True) + eps
    value = value / (value.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        value = value / (value.sum(dim=-1, keepdim=True) + eps)
        value = value / (value.sum(dim=-2, keepdim=True) + eps)
    return pre, post, value
