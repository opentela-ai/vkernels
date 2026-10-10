"""Opt-in SM90 block128 FP8 dense decode, using pinned SGL DeepGEMM."""
from __future__ import annotations

from functools import lru_cache

import torch

_WARMED: set[tuple] = set()


@lru_cache(maxsize=1)
def _quantizer():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def quantize(X, Q, S, GROUPS: tl.constexpr, SCALE_STRIDE: tl.constexpr):
        # Same E4M3FN/group128 arithmetic as sgl_moe's vendored SGLang quantizer;
        # write scale columns directly in the MN-major TMA layout.
        group = tl.program_id(0)
        offsets = group * 128 + tl.arange(0, 128)
        x = tl.load(X + offsets).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x)), 1e-10) / 448.0
        inverse = 1.0 / scale
        q = tl.clamp(x * inverse, -448.0, 448.0).to(Q.dtype.element_ty)
        tl.store(Q + offsets, q)
        tl.store(S + group // GROUPS + (group % GROUPS) * SCALE_STRIDE, scale)
    return quantize


def quantize_tma(x):
    """Contiguous BF16 [M,K] to FP8 payload and owned, aligned FP32 scales."""
    if (not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2
            or not x.is_contiguous() or x.shape[1] % 128):
        raise ValueError("TMA quantization requires contiguous CUDA BF16 [M,K], K multiple128")
    m, k = x.shape
    aligned_m = ((m + 3) // 4) * 4
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    owner = torch.empty((k // 128, aligned_m), device=x.device, dtype=torch.float32)
    scales = owner.t()[:m]
    if m:
        _quantizer()[(m * (k // 128),)](x, q, scales, k // 128, aligned_m,
                                       num_warps=1, num_stages=1)
    return q, scales


@lru_cache(maxsize=1)
def _backend():
    import deep_gemm

    if not callable(getattr(deep_gemm, "fp8_gemm_nt", None)):
        raise RuntimeError("DeepGEMM dense decode requires fp8_gemm_nt")
    return deep_gemm


def deepgemm_linear(x, weight, scales, bias=None):
    """Preserve group128 quantization and BF16 outputs; accumulation may differ.

    Activation scales use owned Torch MN-major, 16-byte-aligned storage,
    avoiding non-owning TVM-FFI layout aliases. Weight scales stay block128.
    Backend/JIT errors propagate; this explicit experiment never silently falls
    back to the ordered-group implementation.
    """
    if (not x.is_cuda or torch.version.hip is not None
            or torch.cuda.get_device_capability(x.device)[0] != 9
            or x.dtype != torch.bfloat16 or x.ndim < 2 or weight.ndim != 2
            or weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32
            or x.device != weight.device or x.device != scales.device
            or x.shape[-1] != weight.shape[1]
            or not weight.is_contiguous() or not scales.is_contiguous()):
        raise ValueError("DeepGEMM dense decode requires SM90 CUDA BF16/block128 E4M3FN inputs")
    n, k = weight.shape
    m = x.numel() // k
    if (not 1 <= m <= 8 or n % 128 or k % 128
            or scales.shape != (n // 128, k // 128)):
        raise ValueError("DeepGEMM dense decode requires M1-8 and block128-aligned N/K/scales")
    key = (x.device, m, n, k)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and key not in _WARMED:
        raise RuntimeError("DeepGEMM dense decode must be warmed before graph capture")
    backend = _backend()
    flat = x.reshape(m, k).contiguous()
    quantized, sf_tma = quantize_tma(flat)
    result = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    backend.fp8_gemm_nt((quantized, sf_tma), (weight, scales), result,
                        compiled_dims="nk", disable_ue8m0_cast=True)
    if not capturing:
        _WARMED.add(key)
    result = result.view(*x.shape[:-1], n)
    return result if bias is None else result + bias
