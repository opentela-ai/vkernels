"""Gather checkpoint E4M3FN experts and block-dequantize to BF16/FP16.

Decode the stored bytes explicitly: gfx942's native E4M3FNUZ conversion has
different zero/NaN encodings and exponent bias from checkpoint E4M3FN.
Only the output is allocated; FP32 products remain in kernel registers.

Adopted from floe's ``engine/runner/kernels/fp8_experts.py`` (vkernels owns
the kernel; floe imports it back through a thin adapter — the #64/#65
thin-adapter model extended to the whole GLM-5 Triton set).

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_dequant(
        W, S, IDX, Y, O: tl.constexpr, I: tl.constexpr, BLOCK: tl.constexpr,
        FNUZ: tl.constexpr,
    ):
        selected = tl.program_id(0)
        offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offset < O * I
        expert = tl.load(IDX + selected).to(tl.int64)
        raw = tl.load(W + expert * (O * I) + offset, mask, 0).to(tl.int32)
        exponent = (raw >> 3) & 15
        mantissa = raw & 7
        if FNUZ:
            # e4m3fnuz storage (issue #71 in-place conversion): bias 8, NaN
            # only at 0x80, subnormals m*2^-10; stored scales are the
            # DOUBLED scales, reproducing the original e4m3fn values
            # exactly (verified exhaustive over all 256 bytes).
            bits = ((exponent + 119) << 23) | (mantissa << 20)
            value = tl.where(
                exponent == 0,
                mantissa.to(tl.float32) * 0.0009765625,
                bits.to(tl.float32, bitcast=True),
            )
            value = tl.where(raw == 128, float("nan"), value)
        else:
            # Normal E4M3FN numbers have bias 7; all exponent-15 values
            # except mantissa 7 are finite. Bit construction is exact in
            # FP32.
            bits = ((exponent + 120) << 23) | (mantissa << 20)
            value = tl.where(
                exponent == 0,
                mantissa.to(tl.float32) * 0.001953125,
                bits.to(tl.float32, bitcast=True),
            )
            value = tl.where((raw & 127) == 127, float("nan"), value)
        # Preserve -0 explicitly: arithmetic negation can be folded to 0-x,
        # which produces +0 for a zero input on the HIP compilation path.
        value = (value.to(tl.int32, bitcast=True) | ((raw & 128) << 24)).to(
            tl.float32, bitcast=True
        )
        scale_offset = (expert * (O // 128) + offset // I // 128) * (I // 128)
        scale_offset += (offset % I) // 128
        scale = tl.load(S + scale_offset, mask, 0)
        tl.store(Y + selected.to(tl.int64) * (O * I) + offset, value * scale, mask)

    return _gather_dequant


def gather_dequant(weights, scales, indices, dtype=None, storage="e4m3fn"):
    """Return contiguous [T,K,O,I] selected weights with 128x128 scales.

    ``storage="e4m3fnuz"`` (issue #71) reads the IN-PLACE converted stacks
    from ``e4m3fn_to_fnuz_inplace`` — fnuz bytes with DOUBLED fp32 scales
    sharing the checkpoint's storage; values decode identically to the
    e4m3fn path (verified exhaustive over all 256 bytes).

    Inputs must be contiguous GPU tensors on the same device. Indices must
    be valid expert IDs (as supplied by topk); bounds are not checked on the
    host, keeping this launch asynchronous and graph-capturable.
    ``dtype`` defaults to BF16.
    """
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    if weights.ndim != 3 or indices.ndim != 2:
        raise OpNotEligible("expected weights [E,O,I] and indices [T,K]")
    e, o, i = weights.shape
    if not e or not o or not i or o % 128 or i % 128:
        raise OpNotEligible("expert dimensions must be positive multiples of 128")
    if scales.shape != (e, o // 128, i // 128):
        raise OpNotEligible("expected scales [E,O/128,I/128]")
    if storage not in ("e4m3fn", "e4m3fnuz"):
        raise OpNotEligible(f"unknown weight storage {storage!r}")
    fnuz = storage == "e4m3fnuz"
    want = torch.float8_e4m3fnuz if fnuz else torch.float8_e4m3fn
    if weights.dtype != want or scales.dtype != torch.float32:
        raise OpNotEligible(
            f"expected {'E4M3FNUZ' if fnuz else 'E4M3FN'} weights and FP32 scales"
        )
    if indices.dtype != torch.int64 or dtype not in (torch.bfloat16, torch.float16):
        raise OpNotEligible("expected int64 indices and BF16 or FP16 output")
    if any(
        not x.is_cuda or x.device != weights.device for x in (weights, scales, indices)
    ):
        raise OpNotEligible("inputs must share a GPU device")
    if any(not x.is_contiguous() for x in (weights, scales, indices)):
        raise OpNotEligible("inputs must be contiguous")
    out = torch.empty((*indices.shape, o, i), device=weights.device, dtype=dtype)
    if indices.numel():
        import triton  # lazy: validation above needs only torch

        with torch.cuda.device(weights.device):
            _kernel()[(indices.numel(), triton.cdiv(o * i, 1024))](
                weights.view(torch.uint8),
                scales,
                indices,
                out,
                o,
                i,
                1024,
                fnuz,
            )
    return out


def gather_dequant_reference(weights, scales, indices, dtype=None, storage="e4m3fn"):
    """Gather + explicit E4M3FN/FNUZ block-dequant in torch (the oracle).

    Mirrors the kernel's byte decode: bias-7 (e4m3fn) or bias-8 (e4m3fnuz,
    NaN only at 0x80, subnormals m*2^-10) exponent reconstruction,
    subnormal mantissa scaling, NaN on the encoding-specific pattern, and
    the explicit sign bit (preserving -0).
    """
    import torch

    if storage not in ("e4m3fn", "e4m3fnuz"):
        raise OpNotEligible(f"unknown weight storage {storage!r}")
    fnuz = storage == "e4m3fnuz"
    if dtype is None:
        dtype = torch.bfloat16
    raw = weights.view(torch.uint8).to(torch.int32)
    exponent, mantissa = (raw >> 3) & 15, raw & 7
    bias_shift = 119 if fnuz else 120
    sub_step = 0.0009765625 if fnuz else 0.001953125
    bits = (((exponent + bias_shift) << 23) | (mantissa << 20)).to(torch.int32)
    value = bits.view(torch.float32)
    value = torch.where(exponent == 0, mantissa.to(torch.float32) * sub_step, value)
    value = torch.where(
        (raw == 128) if fnuz else ((raw & 127) == 127), float("nan"), value
    )
    value = value * torch.where((raw & 128) != 0, -1.0, 1.0)
    scale = scales.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
    weight = (value * scale[:, : weights.shape[1], : weights.shape[2]]).to(dtype)
    return weight[indices]
