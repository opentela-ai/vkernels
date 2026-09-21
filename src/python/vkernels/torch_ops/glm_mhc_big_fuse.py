"""GLM-5 mHC pre "big fuse": gates + Sinkhorn + collapse + RMSNorm (inference-only).

One launch per mHC pre site, replacing the post-GEMM epilogue chain of
``Glm53HyperConnection.forward`` + the decoder layer's RMSNorm:

* gates + Sinkhorn combiner from the mix logits (exactly the
  ``glm_mhc_mix._mix`` math and normalization order),
* stream collapse ``sum_k pre[k] * streams[k, :]`` in fp32 with a single
  round on store (exactly the ``mhc_compose._collapse`` reduction),
* weighted RMSNorm of the collapsed streams (exactly ``Glm53RMSNorm``'s
  rounding points: variance on the bf16-rounded collapse re-upcast to
  fp32, the normalized value rounded to the stream dtype, then the
  fp32-opmath weight multiply rounded once on store).

The mix GEMM stays with the caller (``_mix_logits``); this is the sglang
``mhc_pre_big_fuse_with_norm_tilelang`` design (one kernel per token
block, GEMMs kept) ported to floe's norm-then-GEMM logits contract, where
the logits arrive already RMS-rescaled and only the fp32 gate math, the
Sinkhorn loop, the collapse and the post-collapse norm are left to fuse.

fp32 intermediate contract: every control-plane and reduction intermediate
is fp32; the only intermediate round is the collapse store, which the
eager chain also pays (the RMSNorm reads the bf16 collapsed tensor).
Residual deviation vs the eager chain is bounded by fp32 reduction-order
(tree vs sequential) in the 4-way collapse sum and the D-way variance sum
— at most last-ulp fp32 before each single rounding, i.e. 0 or 1 ulp of
the stored dtype. The parity test pins the max abs drift.

Decode-shaped: static per-bucket shapes (token count is the grid), plain
current-stream Triton launches, no host syncs, no device scalars — CUDA
graph-capture safe.

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

import math

from ._dispatch import OpNotEligible
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _big_fuse(
        L,
        B,
        S,
        STREAMS,
        NORM_W,
        POST,
        COMB,
        OUT,
        D,
        HC: tl.constexpr,
        EPS: tl.constexpr,
        ITERS: tl.constexpr,
        NORM_EPS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        k = tl.arange(0, HC)
        width: tl.constexpr = HC * (HC + 2)
        s0 = tl.load(S)
        s1 = tl.load(S + 1)
        s2 = tl.load(S + 2)
        # -- gates + Sinkhorn combiner (glm_mhc_mix._mix order) ------------
        pre = tl.load(L + token * width + k).to(tl.float32) * s0 + tl.load(B + k)
        post = tl.load(L + token * width + HC + k).to(tl.float32) * s1 + tl.load(
            B + HC + k
        )
        pre_gated = tl.sigmoid(pre) + EPS
        tl.store(POST + token * HC + k, 2.0 * tl.sigmoid(post))
        offset = k[:, None] * HC + k[None, :]
        logits = tl.load(L + token * width + 2 * HC + offset).to(tl.float32) * s2 + tl.load(
            B + 2 * HC + offset
        )
        value = tl.exp(logits - tl.max(logits, 1)[:, None])
        value = tl.div_rn(value, tl.sum(value, 1)[:, None]) + EPS
        value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
        for _ in range(ITERS - 1):
            value = tl.div_rn(value, tl.sum(value, 1)[:, None] + EPS)
            value = tl.div_rn(value, tl.sum(value, 0)[None, :] + EPS)
        tl.store(COMB + token * HC * HC + offset, value)
        # -- collapse: sum_k pre[k] * streams[k, :] (fp32, one round) ------
        # The gate vector is re-read per k through the where-select so the
        # reduction is the same [HC, BLOCK_D] tree tl.sum the landed
        # mhc_collapse kernel uses (pre[i] scalar indexing is not portable).
        offs = tl.arange(0, BLOCK_D)
        streams = tl.load(
            STREAMS + token * HC * D + k[:, None] * D + offs[None, :]
        ).to(tl.float32)
        acc = tl.sum(pre_gated[:, None] * streams, axis=0)
        # The eager chain rounds the collapse to the stream dtype before the
        # RMSNorm reads it back up — pay the same round here.
        collapsed = acc.to(OUT.dtype.element_ty).to(tl.float32)
        # -- weighted RMSNorm (Glm53RMSNorm rounding points) ---------------
        rstd = tl.rsqrt(tl.sum(collapsed * collapsed, axis=0) / D + NORM_EPS)
        w = tl.load(NORM_W + offs).to(tl.float32)
        y = (collapsed * rstd).to(OUT.dtype.element_ty).to(tl.float32)
        tl.store(OUT + token * D + offs, (w * y).to(OUT.dtype.element_ty))

    return _big_fuse


def mhc_pre_big_fuse(
    logits,
    base,
    scale,
    streams,
    norm_weight,
    hc=4,
    eps=1e-6,
    sinkhorn_iters=20,
    norm_eps=1e-6,
):
    """Fused mHC pre epilogue: return fp32 ``(post, comb)`` plus the
    RMSNorm'd collapsed streams.

    ``logits`` is the mix-GEMM output ``[..., hc*(hc+2)]`` (bf16/fp16/fp32,
    the gate math upcasts to fp32 exactly like the caller's
    ``logits.float()``); ``streams`` is ``[..., hc, d]`` bf16/fp16;
    ``base``/``scale`` fp32; ``norm_weight`` ``[d]`` (the post-collapse
    RMSNorm's learned scale). Shapes of ``logits`` and ``streams`` must
    share every leading dim. Returns ``(post [..., hc] fp32,
    comb [..., hc, hc] fp32, layer_input [..., d] in streams.dtype)``.
    """
    import torch

    if hc not in (2, 4):
        raise OpNotEligible("supported hc values are 2 and 4")
    if not isinstance(sinkhorn_iters, int) or sinkhorn_iters < 1:
        raise OpNotEligible("sinkhorn_iters must be a positive integer")
    if not math.isfinite(eps) or eps < 0:
        raise OpNotEligible("eps must be finite and nonnegative")
    width = hc * (hc + 2)
    if logits.ndim < 2 or logits.shape[-1] != width:
        raise OpNotEligible("incorrect logits width")
    if streams.ndim != logits.ndim + 1 or streams.shape[-2] != hc:
        raise OpNotEligible("streams must be [..., hc, d] matching the logits' leading dims")
    d = streams.shape[-1]
    if base.shape != (width,) or scale.shape != (3,):
        raise OpNotEligible("expected base [hc*(hc+2)] and scale [3]")
    if norm_weight.shape != (d,):
        raise OpNotEligible("expected norm_weight [d]")
    if logits.shape[:-1] != streams.shape[:-2]:
        raise OpNotEligible("logits and streams leading dims must match")
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise OpNotEligible("logits must be bf16/fp16/fp32")
    if base.dtype != torch.float32 or scale.dtype != torch.float32:
        raise OpNotEligible("mHC control tensors must be FP32")
    if streams.dtype not in (torch.bfloat16, torch.float16):
        raise OpNotEligible("streams must be bf16/fp16")
    if d != _pow2(d) or d < 512 or d > 8192:
        raise OpNotEligible("d must be a power of two in [512, 8192]")
    for x in (logits, base, scale, streams, norm_weight):
        if not x.is_cuda or not x.is_contiguous():
            raise OpNotEligible("inputs must be contiguous tensors on a GPU")
    if not (logits.device == streams.device == base.device == scale.device == norm_weight.device):
        raise OpNotEligible("inputs must live on one GPU")
    post = torch.empty(
        (*logits.shape[:-1], hc), device=logits.device, dtype=torch.float32
    )
    comb = torch.empty(
        (*logits.shape[:-1], hc, hc), device=logits.device, dtype=torch.float32
    )
    out = torch.empty(
        (*logits.shape[:-1], d), device=logits.device, dtype=streams.dtype
    )
    tokens = logits.numel() // width
    if tokens:
        with torch.cuda.device(logits.device):
            _kernel()[(tokens,)](
                logits,
                base,
                scale,
                streams,
                norm_weight,
                post,
                comb,
                out,
                d,
                hc,
                eps,
                sinkhorn_iters,
                norm_eps,
                d,
                num_warps=8,
                enable_fp_fusion=False,
            )
    return post, comb, out


def _pow2(d):
    return 1 << (d.bit_length() - 1)


def mhc_pre_big_fuse_reference(
    logits,
    base,
    scale,
    streams,
    norm_weight,
    hc=4,
    eps=1e-6,
    sinkhorn_iters=20,
    norm_eps=1e-6,
):
    """Eager oracle: the exact chain the kernel replaces, composed from the
    landed reference pieces (``mhc_mix_reference`` + eager collapse +
    eager ``Glm53RMSNorm``), so parity here is parity with the whole
    epilogue chain, not a second hand-rolled kernel oracle."""
    import torch

    from .glm_mhc_mix import mhc_mix_reference

    pre, post, comb = mhc_mix_reference(
        logits.float(), base, scale, hc=hc, eps=eps, sinkhorn_iters=sinkhorn_iters
    )
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
    x = collapsed.float()
    var = x.pow(2).mean(-1, keepdim=True)
    normalized = (x * torch.rsqrt(var + norm_eps)).to(streams.dtype)
    return post, comb, norm_weight * normalized
