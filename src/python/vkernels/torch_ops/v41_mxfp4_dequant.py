"""Dequantize MXFP4 (packed E2M1 + E8M0 microscale) weights to BF16/FP16.

DeepSeek-V4.1-Flash stores its routed/shared experts (and the fp4 KV/index
latents) as packed E2M1: two 4-bit codes per byte, low nibble first, with a
per-group (default 32) microscale. The 16-entry E2M1 value table is fixed by
the format (1 sign, 2 exponent, 1 mantissa bits); the scale is applied per
group along the input dimension.

Adopted into vkernels as the V4.1 counterpart of the GLM-5 E4M3
``glm_expert_gather_dequant`` kernel (vkernels owns the kernel; floe imports
it through ``vkl_ops`` and falls back to the reference when unavailable).

Torch and Triton load lazily. Inference-only, no autograd backward. The CPU
``*_reference`` is the always-compiled correctness oracle; the Triton path
mirrors it and is validated on GPU CI.
"""

from functools import lru_cache

from ._fastpath import fast_path

# E2M1 code (4 bits) -> value. High bit is the sign; low 3 bits are the
# magnitude {0, .5, 1, 1.5, 2, 3, 4, 6}. Format-level constant — the decode
# GEMV shares it, so it is public (unlike the kernels' private helpers).
FP4_VALUES = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def decode_scale(torch, scale):
    """Return the microscale as an fp32 multiplier.

    Accepts an already-decoded float scale (the common loader path), a raw
    uint8 E8M0 exponent code (value ``2**(code-127)``), or torch's native
    ``float8_e8m0fnu`` (cast handles the exponent decode). Public: the
    device GEMV needs the same decode as its reference on every path."""
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.float() - 127.0)
    return scale.float()


def mxfp4_dequant_reference(packed, scale, *, group: int = 32, dtype=None):
    """Unpack E2M1 codes and apply the per-``group`` microscale (the oracle).

    ``packed``: ``[..., I//2]`` uint8 / ``float4_e2m1fn_x2`` (two codes/byte,
    low nibble first). ``scale``: ``[..., I//group]``. Returns ``[..., I]``."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    u8 = packed.view(torch.uint8)
    table = torch.tensor(FP4_VALUES, dtype=torch.float32, device=u8.device)
    low = table[(u8 & 0x0F).long()]
    high = table[((u8 >> 4) & 0x0F).long()]
    vals = torch.stack([low, high], dim=-1).flatten(-2)  # [..., I]
    sc = decode_scale(torch, scale)
    width = vals.shape[-1]
    if sc.shape[-1] == width // group:
        sc = sc.repeat_interleave(group, dim=-1)
    elif sc.shape[-1] != width:
        raise ValueError(f"scale last dim {sc.shape[-1]} != I//group ({width // group}) or I ({width})")
    return (vals * sc).to(dtype)


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _mxfp4(PACKED, SCALE, TABLE, Y, I: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr):
        # int64: the flattened full-stack shape (E*O rows, e.g. 884_736 x 5120)
        # overflows int32 byte offsets.
        row = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)  # output column in [0, I)
        mask = col < I
        byte = tl.load(PACKED + row * (I // 2) + col // 2, mask, 0).to(tl.int32)
        nib = tl.where(col % 2 == 0, byte & 0x0F, (byte >> 4) & 0x0F)
        val = tl.load(TABLE + nib)  # fp32 E2M1 value
        sc = tl.load(SCALE + row * (I // GROUP) + col // GROUP, mask, 0)
        tl.store(Y + row * I + col, val * sc, mask)

    return _mxfp4


def mxfp4_dequant(packed, scale, *, group: int = 32, dtype=None):
    """Device MXFP4 dequant. Falls back to the reference off-GPU / without
    Triton so the op is always correct; the Triton path is the accelerator."""
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    if packed.view(torch.uint8).shape[-1] * 2 % group:
        raise ValueError("I must be a multiple of the scale group")
    if not fast_path(packed, scale):
        return mxfp4_dequant_reference(packed, scale, group=group, dtype=dtype)

    u8 = packed.view(torch.uint8)
    lead = u8.shape[:-1]
    o = 1
    for d in lead:
        o *= int(d)
    i = int(u8.shape[-1]) * 2
    u8 = u8.reshape(o, i // 2).contiguous()
    sc = decode_scale(torch, scale).reshape(o, i // group).contiguous()
    table = torch.tensor(FP4_VALUES, dtype=torch.float32, device=u8.device)
    out = torch.empty((o, i), device=u8.device, dtype=dtype)
    import triton

    with torch.cuda.device(u8.device):
        _kernel()[(o, triton.cdiv(i, 256))](u8, sc, table, out, i, group, 256)
    return out.reshape(*lead, i)
