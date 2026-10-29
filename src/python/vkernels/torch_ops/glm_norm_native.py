"""Native-order RMSNorm glue for the pinned Beverin inference stack.

Torch retains the mean/epsilon/rsqrt chain and its reduction layout. Two
Triton kernels fuse BF16 widening/square and the native-rounded epilogue.
The plain, gated and unweighted cases keep their distinct rounding points.
"""
from functools import lru_cache
from ._dispatch import OpNotEligible


@lru_cache(maxsize=1)
def _kernels():
    global tl, libdevice
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def square_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, 0).to(tl.float32)
        tl.store(Y + i, x * x, i < N)

    @triton.jit
    def epilogue(X, G, W, INV, Y, D: tl.constexpr, R1: tl.constexpr,
                 R2: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                 S2: tl.constexpr, S3: tl.constexpr, GS0: tl.constexpr,
                 GS1: tl.constexpr, GS2: tl.constexpr, GS3: tl.constexpr,
                 KIND: tl.constexpr, FP32_WEIGHT: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        d = tl.arange(0, BLOCK)
        r0 = row // (R1 * R2)
        r1 = (row // R2) % R1
        r2 = row % R2
        offset = r0 * S0 + r1 * S1 + r2 * S2 + d * S3
        x = tl.load(X + offset, d < D, 0).to(tl.float32)
        inv = tl.load(INV + row)
        if KIND == 2:
            inv = inv.to(tl.bfloat16).to(tl.float32)
            y = x * inv
        else:
            y = x * inv
            if KIND == 0 and not FP32_WEIGHT:
                y = y.to(tl.bfloat16).to(tl.float32)
            w = tl.load(W + d, d < D, 0).to(tl.float32)
            y = w * y
            if KIND == 1:
                goffset = r0 * GS0 + r1 * GS1 + r2 * GS2 + d * GS3
                g = tl.load(G + goffset, d < D, 0).to(tl.float32)
                sigmoid = tl.div_rn(1., 1. + libdevice.exp(-g))
                y = y * sigmoid
        tl.store(Y + row * D + d, y, d < D)

    return square_kernel, epilogue


def rms_norm_native(x, module, gate=None, *, fp32_weight=False, canonical_rows=False):
    """Return native-rounded BF16 output; unsupported contracts decline.

    Eligibility covers measured dense BF16 matrices and head layouts on
    gfx942/Torch2.9-HIP7/Triton3.4. There are no global backend changes,
    cache writes, input mutations or data-dependent synchronization.
    ``canonical_rows`` keeps contiguous weighted width-1536 reductions in
    the one-to-three-row reference regime on this stack. Callers must opt
    in only where short-block outputs must agree with single-token decode.
    """
    import math
    import torch

    weighted = hasattr(module, "weight")
    gated = gate is not None
    if not 2 <= x.ndim <= 4 or x.dtype != torch.bfloat16:
        raise OpNotEligible("requires a BF16 tensor with two to four dimensions")
    dim = x.shape[-1]
    rows = x.numel() // dim if dim else 0
    if ((gated and dim != 128) or (not gated and weighted and dim not in (512, 1536, 4096))
            or (not weighted and dim != 4096) or not 1 <= rows <= 256):
        raise OpNotEligible("unmeasured norm geometry")
    # A compact physical layout lets the square kernel traverse storage in
    # order while preserving exactly the native pow(2) output strides.
    extent = 1
    for stride, size in sorted(zip(x.stride(), x.shape)):
        if size > 1:
            if stride != extent:
                raise OpNotEligible("requires a dense non-overlapping layout")
            extent *= size
    if not x.is_cuda:
        raise OpNotEligible("requires a HIP device")
    if weighted and (module.weight.shape != (dim,) or module.weight.dtype != torch.bfloat16
                     or module.weight.device != x.device or not module.weight.is_contiguous()):
        raise OpNotEligible("requires contiguous BF16 weights on the input device")
    if gated and (not weighted or gate.shape != x.shape or gate.dtype != x.dtype
                  or gate.device != x.device or any(s < 0 for s in gate.stride())):
        raise OpNotEligible("requires a matching BF16 sigmoid gate")
    tensors = [x] + ([module.weight] if weighted else []) + ([gate] if gated else [])
    if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
        raise OpNotEligible("inference only")
    eps = module.variance_epsilon if weighted else module.eps
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise OpNotEligible("requires a positive finite scalar epsilon")
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        raise OpNotEligible("unqualified Torch/HIP stack")
    if torch.cuda.get_device_properties(x.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("qualified on gfx942 only")
    import triton
    if triton.__version__ != "3.4.0":
        raise OpNotEligible("unqualified Triton version")
    square, finish = _kernels()
    shape = (1,) * (4 - x.ndim) + tuple(x.shape)
    strides = (0,) * (4 - x.ndim) + x.stride()
    gs = (0,) * (4 - gate.ndim) + gate.stride() if gated else (0, 0, 0, 0)
    kind = 1 if gated else 0 if weighted else 2
    with torch.cuda.device(x.device):
        squared = torch.empty_like(x, dtype=torch.float32)
        square[(triton.cdiv(x.numel(), 256),)](
            x, squared, x.numel(), 256, enable_fp_fusion=False)
        if canonical_rows and weighted and not gated and dim == 1536 and rows >= 4 and x.is_contiguous():
            # HIP Torch selects a different mean reduction at four rows.
            # Two-row groups retain the reference reduction and rounding;
            # square and the BF16 epilogue remain unchanged.
            inv = torch.cat([
                torch.rsqrt(part.mean(-1, keepdim=True) + eps)
                for part in squared.view(rows, dim).split(2)
            ], dim=0).contiguous()
        else:
            inv = torch.rsqrt(squared.mean(-1, keepdim=True) + eps).contiguous()
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        finish[(rows,)](
            x, gate if gated else x, module.weight if weighted else x, inv, out,
            dim, shape[1], shape[2], *strides, *gs, kind, fp32_weight,
            triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return out
