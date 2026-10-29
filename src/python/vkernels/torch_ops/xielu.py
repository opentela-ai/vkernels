"""Inference xIELU preserving each eager intermediate dtype rounding."""
from functools import lru_cache

from ._dispatch import require


@lru_cache(maxsize=1)
def _kernel():
    global tl, libdevice
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit(do_not_specialize=["N"])
    def run(X, AP, AN, BETA, EPS, Y, N, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        dtype = X.dtype.element_ty
        x = tl.load(X + index, index < N, 0).to(tl.float32)
        ap, an = tl.load(AP).to(tl.float32), tl.load(AN).to(tl.float32)
        beta, eps = tl.load(BETA).to(tl.float32), tl.load(EPS).to(tl.float32)
        # F.softplus uses beta=1 and threshold=20. Read learned parameters
        # on every launch, including replay, rather than caching scalars.
        ap = tl.where(ap > 20, ap, libdevice.log1p(libdevice.exp(ap))).to(dtype).to(tl.float32)
        an = tl.where(an > 20, an, libdevice.log1p(libdevice.exp(an))).to(dtype).to(tl.float32)
        an = (beta + an).to(dtype).to(tl.float32)
        bx = (beta * x).to(dtype).to(tl.float32)
        positive = (ap * x).to(dtype).to(tl.float32)
        positive = (positive * x).to(dtype).to(tl.float32)
        positive = (positive + bx).to(dtype).to(tl.float32)
        negative = libdevice.expm1(tl.minimum(x, eps, propagate_nan=tl.PropagateNan.ALL)).to(dtype).to(tl.float32)
        negative = (negative - x).to(dtype).to(tl.float32)
        negative = (negative * an).to(dtype).to(tl.float32)
        negative = (negative + bx).to(dtype).to(tl.float32)
        tl.store(Y + index, tl.where(x > 0, positive, negative), index < N)

    return triton, run


def xielu(x, alpha_p, alpha_n, beta, eps):
    """Fuse native learned xIELU; scalar parameters share the input dtype."""
    import torch

    values = (x, alpha_p, alpha_n, beta, eps)
    require(x.is_cuda and all(t.device == x.device for t in values), "xIELU inputs must share a GPU")
    require(x.dtype in (torch.float16, torch.bfloat16, torch.float32) and all(t.dtype == x.dtype for t in values), "xIELU inputs must share a floating-point dtype")
    require(x.is_contiguous() and all(t.numel() == 1 for t in values[1:]), "xIELU requires contiguous input and scalar parameters")
    require(not torch.is_grad_enabled() or not any(t.requires_grad for t in values), "xIELU fusion is inference-only")
    output = torch.empty_like(x)
    if x.numel():
        triton, kernel = _kernel()
        # Small decode vectors need enough independent programs; large
        # prefills amortize program/scalar-parameter overhead with more work
        # per program. Arithmetic and rounding remain elementwise-identical.
        block = 2048 if x.numel() >= 1 << 20 else 256
        with torch.cuda.device(x.device):
            kernel[(triton.cdiv(x.numel(), block),)](x, alpha_p, alpha_n, beta, eps, output, x.numel(), block, enable_fp_fusion=False)
    return output
