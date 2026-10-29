"""CDNA3 storage for the dense projection's E4M3FN activation contract."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X, Q, S, eps, fp8_min, fp8_max, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    offsets = group * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets).to(tl.float32)
    # Keep the CUDA donor's floor and reciprocal/multiply order. The older
    # grouped-MoE FNUZ quantizer uses a different 1e-4 floor and division.
    scale = tl.maximum(tl.max(tl.abs(values)), eps) / fp8_max
    inverse = 1.0 / scale
    payload = tl.clamp(values * inverse, fp8_min, fp8_max)
    half = (payload * 0.5).to(tl.float8e4b8)
    tl.store(Q + offsets, half)
    tl.store(S + group, scale * 2.0)


def quantize_fn_as_fnuz(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FN-equivalent values stored as half-valued FNUZ with doubled scales.

    The caller validates gfx942 and inference projection inputs. No weight
    conversion, cache, or host read of activation values is involved.
    """
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("expected contiguous BF16 [M, K] activations")
    if x.shape[1] <= 0 or x.shape[1] % 128:
        raise ValueError("activation width must be a positive multiple of 128")
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fnuz)
    scales = torch.empty((x.shape[0], x.shape[1] // 128), device=x.device, dtype=torch.float32)
    if x.numel():
        _quantize[(x.numel() // 128,)](x, quantized, scales, 1e-10, -448.0, 448.0, BLOCK=128,
                                     num_warps=1, num_stages=1)
    return quantized, scales
