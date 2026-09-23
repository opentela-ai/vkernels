"""GLM-5 mHC pre-GEMV fusion: unweighted RMSNorm + mix GEMV (inference-only).

One launch per mHC pre site, replacing the two-launch head of the pre
chain (``Glm53HyperConnection._mix_logits``'s torch path):

* ``Glm53UnweightedRMSNorm``: ``y = x * rsqrt(mean(x_f32²) + eps)``,
  the rsqrt factor rounded to the stream dtype before the multiply and
  the normalized value rounded again on store (bf16/fp16, fp32 opmath) —
  exactly the eager rounding points;
* the mix GEMV ``F.linear(y, fn)`` (``[mix, hc·D] @ [hc·D]``): bf16/fp16
  inputs, FP32 accumulation, one round on store — the same precision
  class as cuBLAS/the landed Triton dense GEMV, differing only in fp32
  reduction order (bounded by last-ulp fp32 before each single rounding).

Per decode step this removes ~90 ``_norm_uw`` launches (~2.1 us each) and
folds the mix GEMV into the same launch: a pre site drops from 5 kernels
(norm → GEMV → ``_mix`` gates → ``_collapse`` → post-collapse ``_norm``,
~12-13 us of launch-bound single-CTA work at bs=1) to 2
(``mhc_pre_gemv`` + ``mhc_pre_big_fuse``) under floe's ``mhc_big_fuse``
knob. Grid ``(tokens, mix)``: each program computes one output row,
re-deriving the (tiny) row statistic — the fn stack (the only real
traffic, ~786 KB at the deployed [24, 16384] shape) is read exactly once
across the row programs, in parallel.

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible
import math
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _pre_gemv(
        X,            # [T, K] streams (bf16/fp16)
        FN,           # [MIX, K] mix weights (same dtype)
        OUT,          # [T, MIX] logits (same dtype)
        K,
        MIX,
        EPS: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        # -- pass 1: the unweighted-RMS statistic (fp32) -------------------
        ss = 0.0
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(X + token * K + offs, mask=mask, other=0.0).to(tl.float32)
            ss += tl.sum(x * x, axis=0)
        # eager: the rsqrt factor is rounded to the stream dtype BEFORE the
        # multiply; keep that round (and re-upcast) exactly.
        rstd = tl.rsqrt(ss / K + EPS).to(OUT.dtype.element_ty).to(tl.float32)
        # -- pass 2: normalized GEMV row, FP32 accumulation ----------------
        acc = 0.0
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            mask = offs < K
            x = tl.load(X + token * K + offs, mask=mask, other=0.0).to(tl.float32)
            xn = (x * rstd).to(OUT.dtype.element_ty).to(tl.float32)
            w = tl.load(FN + row * K + offs, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(xn * w, axis=0)
        tl.store(OUT + token * MIX + row, acc.to(OUT.dtype.element_ty))

    return _pre_gemv


def mhc_pre_gemv(streams_flat, fn, hc=4, hidden_size=4096, eps=1e-6):
    """Return the mix logits ``RMSNorm(streams_flat) @ fnᵀ`` in one launch.

    ``streams_flat`` is ``[..., hc·hidden]`` bf16/fp16 (the flattened
    streams), ``fn`` ``[hc·(hc+2), hc·hidden]`` in the same dtype; the
    result has ``streams_flat``'s dtype and shape ``[..., mix]`` with
    ``mix = hc·(hc+2)``. The normalization statistic is the exact
    ``Glm53UnweightedRMSNorm`` math (fp32 mean of squares, the rsqrt
    factor rounded to the stream dtype); the GEMV accumulates in FP32 and
    rounds once on store.
    """
    import torch

    if not isinstance(hc, int) or hc < 2 or hc & (hc - 1):
        raise OpNotEligible("hc must be a power of two >= 2")
    width = hc * (hc + 2)
    k = hc * hidden_size
    if streams_flat.ndim < 1 or streams_flat.shape[-1] != k:
        raise OpNotEligible("streams_flat must be [..., hc*hidden]")
    if fn.ndim != 2 or fn.shape != (width, k):
        raise OpNotEligible("expected fn [hc*(hc+2), hc*hidden]")
    if streams_flat.dtype not in (torch.bfloat16, torch.float16):
        raise OpNotEligible("streams must be bf16/fp16")
    rows = streams_flat.numel() // k
    if rows > 2:
        # Decode-sized rows only (the mhc_projection envelope): at prefill
        # row counts the per-row GEMV grid wastes the machine AND the
        # reduction-order drift surface grows with the row count, so
        # prefill keeps the BLAS GEMM (bit-identical to the eager chain).
        raise OpNotEligible("decode-sized rows only (<= 2)")
    if fn.dtype != streams_flat.dtype:
        raise OpNotEligible("fn must share the streams' dtype")
    for x in (streams_flat, fn):
        if not x.is_cuda or not x.is_contiguous():
            raise OpNotEligible("inputs must be contiguous tensors on a GPU")
    if not math.isfinite(eps):
        raise OpNotEligible("eps must be finite")
    import triton

    lead = streams_flat.shape[:-1]
    x2 = streams_flat.reshape(-1, k)
    tokens = x2.shape[0]
    out = torch.empty(*lead, width, device=streams_flat.device, dtype=streams_flat.dtype)
    if tokens:
        kernel = _kernels()
        block = min(2048, triton.next_power_of_2(k))
        with torch.cuda.device(streams_flat.device):
            kernel[(tokens, width)](
                x2,
                fn,
                out.reshape(-1, width),
                k,
                width,
                eps,
                block,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out.view(*lead, width)


def mhc_pre_gemv_reference(streams_flat, fn, hc=4, hidden_size=4096, eps=1e-6):
    """Eager oracle: the exact two-launch chain the kernel replaces."""
    import torch

    k = hc * hidden_size
    x = streams_flat.reshape(-1, k)
    rstd = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)
    y = x * rstd
    return torch.nn.functional.linear(y, fn).view(*streams_flat.shape[:-1], -1)
