"""DeepSeek-V4.1 fp8 block-scaled GEMM with UE8M0 (power-of-two) scales.

V4.1's backbone projections run fp8 GEMMs whose scales are **UE8M0-rounded**
(power of two) at a configurable block (32 in the reference; §2.9), unlike the
GLM-5 path's 128x128 fp32 scales. The block-scaled GEMM math is the same
DeepSeek scheme (``w = q_fp8 * scale[·//block]``), so this module adds:

* ``quantize_fp8_ue8m0`` — bf16 -> (e4m3, UE8M0 block scales): per (block x
  block) tile, ``scale = 2**ceil(log2(amax/448))`` (a power of two, i.e. an
  E8M0-representable exponent), payload ``round(x/scale)`` clamped to e4m3.
* ``fp8_block_gemm`` / ``*_reference`` — general block-size block-scaled GEMM.
  For ``block == 128`` on GPU it delegates to the proven GLM Hopper kernel
  (``glm_fp8_blockwise_gemm``); otherwise it uses the torch oracle.

CPU ``*_reference`` is the always-tested oracle. Torch/Triton load lazily.
Inference-only, no autograd backward.
"""


def quantize_fp8_ue8m0(x, block: int = 32):
    """BF16/FP32 ``[M, K]`` -> (e4m3 ``[M, K]``, fp32 UE8M0 scales
    ``[ceil(M/block), K//block]``). Scales are exact powers of two."""
    import torch
    import torch.nn.functional as F

    if x.ndim != 2:
        raise ValueError("expected a 2-D [M, K] tensor")
    m, k = x.shape
    if k % block:
        raise ValueError(f"K ({k}) must be a multiple of block ({block})")
    mb = (m + block - 1) // block
    pad = mb * block - m
    xp = F.pad(x, (0, 0, 0, pad)) if pad else x
    xg = xp.view(mb, block, k // block, block).float()
    amax = xg.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    exponent = torch.ceil(torch.log2(amax / 448.0))  # UE8M0: scale is 2**e
    scale = torch.exp2(exponent)
    q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.reshape(mb * block, k)[:m].contiguous(), scale.reshape(mb, k // block).contiguous()


def _dequant_blocks(q_fp8, scales, block, rows, cols):
    full = scales.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    return q_fp8.float() * full[:rows, :cols]


def fp8_block_gemm_reference(a_fp8, a_scales, b_fp8, b_scales, *, block: int = 32, out_dtype=None):
    """``D = dequant(A) @ dequant(B).T`` in fp32, rounded to ``out_dtype``
    (bf16). ``A``: ``[M, K]`` with scales ``[ceil(M/block), K//block]``;
    ``B``: ``[N, K]`` (weights, transposed-use) with scales
    ``[ceil(N/block), K//block]``."""
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16
    m, k = a_fp8.shape
    n, k2 = b_fp8.shape
    if k != k2:
        raise ValueError(f"K mismatch: A K={k}, B K={k2}")
    a = _dequant_blocks(a_fp8, a_scales, block, m, k)
    b = _dequant_blocks(b_fp8, b_scales, block, n, k)
    return (a @ b.t()).to(out_dtype)


def fp8_block_gemm(a_fp8, a_scales, b_fp8, b_scales, *, block: int = 32, out_dtype=None):
    """Device block-scaled fp8 GEMM. For ``block == 128`` on GPU, delegates to
    the GLM Hopper kernel; otherwise the torch reference (always correct)."""
    import torch

    if out_dtype is None:
        out_dtype = torch.bfloat16
    on_gpu = a_fp8.is_cuda and b_fp8.is_cuda
    if block == 128 and on_gpu:
        try:
            from .glm_fp8_blockwise_gemm import fp8_blockwise_gemm

            return fp8_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out_dtype=out_dtype)
        except Exception:
            # house idiom (glm_fp8_blockwise_gemm): no triton, or a shape the
            # GLM kernel validates against, translates to the torch oracle
            pass
    return fp8_block_gemm_reference(a_fp8, a_scales, b_fp8, b_scales, block=block, out_dtype=out_dtype)
