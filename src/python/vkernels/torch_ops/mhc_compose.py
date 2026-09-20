"""GLM-5 mHC compose/collapse fusion (inference-only).

Two stream-datapath fusions for manifold-constrained hyper-connections,
replacing the eager torch expressions in ``Glm53HyperConnection.forward``
(stream collapse) and ``_mhc_compose`` (attention/FFN writeback):

* ``mhc_collapse``: ``collapsed[j] = sum_k pre[k] * streams[k, j]``
  (eager: ``(pre.unsqueeze(-1) * streams).sum(dim=2)``).
* ``mhc_compose``: ``out[j] = post[j] * sublayer_out[j] + sum_k comb[k, j] * residual[k, j]``
  (eager: ``post.unsqueeze(-1).float() * sublayer_out.unsqueeze(-2).float()
  + comb.transpose(-1, -2).float() @ residual.float()``).

Both eager forms materialize an FP32 ``[n, hc, d]`` intermediate and pay
several dtype-cast launches per site (90 compose sites + 90 collapses per
decode step at hc=4, d=4096). The kernels accumulate in FP32 and round once
on store, matching the reference's rounding points. ``hc`` is small and a
power of two (4 in GLM-5), so the ``k`` reduction is a fully unrolled
multiply-add loop — no ``tl.dot``, hence no TF32 detour and no MMA layout
requirements.

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _collapse(
        PRE,
        STREAMS,
        OUT,
        D,
        HC: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        k = tl.arange(0, HC)
        pre = tl.load(PRE + token * HC + k)
        streams = tl.load(
            STREAMS + token * HC * D + k[:, None] * D + offs[None, :],
            mask=mask[None, :],
            other=0.0,
        )
        acc = tl.sum(pre[:, None] * streams.to(tl.float32), axis=0)
        tl.store(OUT + token * D + offs, acc.to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def _compose(
        POST,
        COMB,
        SUB,
        RESIDUAL,
        OUT,
        D,
        HC: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        token = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = offs < D
        j = tl.arange(0, HC)
        post = tl.load(POST + token * HC + j)
        sub = tl.load(SUB + token * D + offs, mask=mask, other=0.0)
        residual = RESIDUAL + token * HC * D
        acc = tl.zeros([HC, BLOCK_D], dtype=tl.float32)
        for k in tl.static_range(HC):
            comb = tl.load(COMB + token * HC * HC + k * HC + j)
            value = tl.load(residual + k * D + offs, mask=mask, other=0.0)
            acc += comb[:, None] * value[None, :].to(tl.float32)
        # The eager reference computes the FP32 matmul and the FP32 product
        # separately, then adds them; adding the product *after* the k-sum
        # keeps the same association (and therefore bit-equal rounding).
        acc += post[:, None] * sub[None, :].to(tl.float32)
        tl.store(
            OUT + token * HC * D + j[:, None] * D + offs[None, :],
            acc.to(OUT.dtype.element_ty),
            mask=mask[None, :],
        )

    return _collapse, _compose


def _check_common(hc, d, *tensors):

    if not isinstance(hc, int) or hc < 2 or hc & (hc - 1):
        raise OpNotEligible("hc must be a power of two >= 2")
    if d <= 0:
        raise OpNotEligible("hidden dimension must be positive")
    for x in tensors:
        if not x.is_cuda or not x.is_contiguous():
            raise OpNotEligible("inputs must be contiguous tensors on a GPU")


def mhc_collapse(pre, streams, hc=4):
    """Return FP32-summed, output-rounded ``sum_k pre[..., k] * streams[..., k, :]``.

    ``pre`` is FP32 ``[..., hc]``, ``streams`` bf16/fp16 ``[..., hc, d]``;
    the result has ``streams``' dtype and shape ``[..., d]``.
    """
    import torch

    d = streams.shape[-1]
    _check_common(hc, d, pre, streams)
    if pre.shape != (*streams.shape[:-2], hc):
        raise OpNotEligible("expected pre [..., hc] matching streams [..., hc, d]")
    if pre.dtype != torch.float32:
        raise OpNotEligible("mHC control tensors must be FP32")
    if streams.dtype not in (torch.bfloat16, torch.float16):
        raise OpNotEligible("streams must be bf16/fp16")
    import triton

    out = torch.empty(streams.shape[:-2] + (d,), device=streams.device, dtype=streams.dtype)
    tokens = streams.numel() // (hc * d)
    if tokens:
        collapse, _ = _kernels()
        block = min(4096, triton.next_power_of_2(d))
        with torch.cuda.device(streams.device):
            collapse[(tokens, triton.cdiv(d, block))](
                pre,
                streams,
                out,
                d,
                hc,
                block,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out


def mhc_compose(post, comb, sublayer_out, residual, hc=4):
    """Return the FP32-accumulated, output-rounded stream writeback.

    ``post`` FP32 ``[..., hc]``, ``comb`` FP32 ``[..., hc, hc]``,
    ``sublayer_out`` ``[..., d]``, ``residual`` ``[..., hc, d]`` (same dtype
    as ``sublayer_out``); the result has ``residual``'s dtype and shape
    ``[..., hc, d]``.
    """
    import torch

    d = residual.shape[-1]
    _check_common(hc, d, post, comb, sublayer_out, residual)
    if post.shape != (*residual.shape[:-2], hc):
        raise OpNotEligible("expected post [..., hc] matching residual [..., hc, d]")
    if comb.shape != (*residual.shape[:-2], hc, hc):
        raise OpNotEligible("expected comb [..., hc, hc] matching residual [..., hc, d]")
    if sublayer_out.shape != residual.shape[:-2] + (d,):
        raise OpNotEligible("expected sublayer_out [..., d] matching residual [..., hc, d]")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise OpNotEligible("mHC control tensors must be FP32")
    if residual.dtype != sublayer_out.dtype:
        raise OpNotEligible("residual and sublayer_out must share a dtype")
    import triton

    out = torch.empty_like(residual)
    tokens = residual.numel() // (hc * d)
    if tokens:
        _, compose = _kernels()
        block = min(4096, triton.next_power_of_2(d))
        with torch.cuda.device(residual.device):
            compose[(tokens, triton.cdiv(d, block))](
                post,
                comb,
                sublayer_out,
                residual,
                out,
                d,
                hc,
                block,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out


def mhc_collapse_reference(pre, streams, hc=4):
    """Eager oracle: FP32 multiply/sum, one round on store."""

    return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)


def _exact_fp32_matmul(a, b):
    """``a @ b`` in true fp32, never TF32.

    NGC torch and any container that sets ``fp32_precision="tf32"`` (or the
    legacy ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``) silently lower fp32 matmuls
    to TF32, which moves this oracle by ~2**-11 relative — far more than the
    kernels' own rounding. The oracle must be the *exact* expression, so pin
    the precision for the call and restore it afterwards.
    """
    import torch

    cuda_backend = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    previous = None
    if cuda_backend is not None and hasattr(cuda_backend, "allow_tf32"):
        previous = cuda_backend.allow_tf32
        cuda_backend.allow_tf32 = False
    try:
        return torch.matmul(a, b)
    finally:
        if previous is not None:
            cuda_backend.allow_tf32 = previous


def mhc_compose_reference(post, comb, sublayer_out, residual, hc=4):
    """Eager oracle: FP32 multiply/add + FP32 matmul, one round on store."""

    out = post.unsqueeze(-1).float() * sublayer_out.unsqueeze(-2).float() + _exact_fp32_matmul(
        comb.transpose(-1, -2).float(), residual.float()
    )
    return out.to(residual.dtype)
