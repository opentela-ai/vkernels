"""Selected-expert MXFP4 (E2M1 + microscale) decode GEMV for DeepSeek-V4.1.

The V4.1 counterpart of the GLM-5 ``glm_expert_gemv`` (E4M3 block-fp8): decode
each selected expert's packed E2M1 weight on the fly in a BF16 GEMV, with no
gathered/dequantized weight tensor materialized. One matrix per call (an
expert FFN composes three: gate ``w1``, up ``w3``, down ``w2``).

Weights are ``[E, O, I//2]`` packed E2M1 (two codes/byte, low nibble first),
scales ``[E, O, I//group]`` (E8M0 or float microscale, ``group`` default 32).
``x`` is ``[T, I]`` or ``[T, K, I]`` bf16; ``indices`` ``[T, K]`` int64;
output ``[T, K, O]`` bf16 (fp32 dot, bf16-rounded weight — matching
gather-dequant then GEMV).

CPU ``*_reference`` is the oracle (reuses the dequant reference); the Triton
path mirrors it and is validated on GPU CI. Torch/Triton load lazily.
"""

from functools import lru_cache

from .v41_mxfp4_dequant import _FP4_VALUES, mxfp4_dequant_reference


def mxfp4_expert_gemv_reference(x, weights, scales, indices, *, group: int = 32):
    """Gather-dequant + einsum (the oracle): decode all experts to bf16, gather
    the selected ones, and do an fp32 dot rounded to bf16."""
    import torch

    e, o, ip = weights.shape
    i = ip * 2
    t, k = indices.shape
    weight = mxfp4_dequant_reference(weights, scales, group=group, dtype=torch.bfloat16)  # [E,O,I]
    selected = weight[indices]  # [T,K,O,I]
    x3 = x if x.ndim == 3 else x[:, None, :].expand(t, k, i)
    out = torch.einsum("tki,tkoi->tko", x3.float(), selected.float())
    return out.to(torch.bfloat16)


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _mxfp4_gemv(
        X, W, S, TABLE, IDX, Y,
        O: tl.constexpr, I: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr,
        BROADCAST: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
    ):
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        rmask = row[:, None] < O
        cmask = col[None, :] < I
        mask = rmask & cmask
        expert = tl.load(IDX + selected).to(tl.int64)
        byte = tl.load(W + expert * (O * (I // 2)) + row[:, None] * (I // 2) + col[None, :] // 2, mask, 0).to(tl.int32)
        nib = tl.where(col[None, :] % 2 == 0, byte & 0x0F, (byte >> 4) & 0x0F)
        val = tl.load(TABLE + nib)
        scale = tl.load(S + expert * (O * (I // GROUP)) + row[:, None] * (I // GROUP) + col[None, :] // GROUP, mask, 0)
        weight = (val * scale).to(tl.bfloat16).to(tl.float32)
        x_row = selected // K if BROADCAST else selected
        x = tl.load(X + x_row * I + col, col < I, 0).to(tl.float32)
        result = tl.sum(weight * x[None, :], axis=1)
        tl.store(Y + selected * O + row, result, row < O)

    return _mxfp4_gemv


def mxfp4_expert_gemv(x, weights, scales, indices, *, group: int = 32):
    """Device MXFP4 decode GEMV; falls back to the reference off-GPU / without
    Triton so the op is always correct."""
    import torch

    if weights.ndim != 3 or indices.ndim != 2:
        raise ValueError("expected weights [E,O,I//2] and indices [T,K]")
    e, o, ip = weights.shape
    i = ip * 2
    t, k = indices.shape
    if x.shape not in ((t, i), (t, k, i)):
        raise ValueError("expected x[T,I] or x[T,K,I]")
    if scales.shape != (e, o, i // group):
        raise ValueError(f"expected scales [E,O,I//group] = {(e, o, i // group)}")
    on_gpu = x.is_cuda and weights.is_cuda and scales.is_cuda and indices.is_cuda
    try:
        import triton  # noqa: F401
    except Exception:
        on_gpu = False
    if not on_gpu:
        return mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group)

    xb = x.to(torch.bfloat16).contiguous()
    u8 = weights.view(torch.uint8).contiguous()
    sc = scales.float().contiguous()
    idx = indices.to(torch.int64).contiguous()
    table = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=u8.device)
    out = torch.empty((t, k, o), device=xb.device, dtype=torch.bfloat16)
    if t and k:
        import triton

        with torch.cuda.device(xb.device):
            _kernel()[(t * k, triton.cdiv(o, 4))](
                xb, u8, sc, table, idx, out,
                o, i, k, group, xb.ndim == 2, 4, triton.next_power_of_2(i),
                num_warps=4, enable_fp_fusion=False,
            )
    return out
