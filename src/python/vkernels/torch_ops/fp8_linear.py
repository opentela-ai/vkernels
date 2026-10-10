"""Block-scaled dense projections with per-token FP8 activations."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from .sgl_moe import per_token_group_quant_fp8


@triton.jit
def _linear(A, W, SA, SW, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, 128)
    accumulator = tl.zeros((BM, BN), tl.float32)
    for group in range(K // 128):
        a = tl.load(A + rows[:, None] * K + group * 128 + reduction[None, :],
                    rows[:, None] < M, other=0.0)
        w = tl.load(W + columns[None, :] * K + group * 128 + reduction[:, None],
                    columns[None, :] < N, other=0.0)
        sa = tl.load(SA + rows * (K // 128) + group, rows < M, other=0)
        sw = tl.load(SW + (columns // 128) * (K // 128) + group,
                     columns < N, other=0)
        accumulator += tl.dot(a, w) * sa[:, None] * sw[None, :]
    tl.store(Y + rows[:, None] * N + columns[None, :], accumulator,
             (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def _group_products(A, W, P, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, 128)
    group = tl.program_id(2)
    a = tl.load(A + rows[:, None] * K + group * 128 + reduction[None, :],
                rows[:, None] < M, other=0.0)
    w = tl.load(W + columns[None, :] * K + group * 128 + reduction[:, None],
                columns[None, :] < N, other=0.0)
    product = tl.dot(a, w)
    tl.store(P + (group * M + rows[:, None]) * N + columns[None, :], product,
             (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def _accumulate_groups(P, SA, SW, Y, M: tl.constexpr, N: tl.constexpr,
                       GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    COUNT: tl.constexpr = M * N
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // N
    columns = offsets % N
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for group in range(GROUPS):
        product = tl.load(P + group * COUNT + offsets, offsets < COUNT, other=0.0)
        sa = tl.load(SA + rows * GROUPS + group, offsets < COUNT, other=0.0)
        sw = tl.load(SW + (columns // 128) * GROUPS + group, offsets < COUNT, other=0.0)
        accumulator += product * sa * sw
    tl.store(Y + offsets, accumulator, offsets < COUNT)


def fp8_linear(x: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
               bias: torch.Tensor | None = None) -> torch.Tensor:
    """BF16 inputs, E4M3FN weights and FP32 block-128 scales; BF16 output."""
    if (not x.is_cuda or torch.version.hip is not None or x.dtype != torch.bfloat16
            or weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32
            or weight.ndim != 2 or x.ndim < 2 or x.shape[-1] != weight.shape[1]
            or x.device != weight.device or x.device != scales.device
            or not weight.is_contiguous() or not scales.is_contiguous()):
        raise ValueError("FP8 activation projections require CUDA BF16 inputs and contiguous E4M3FN weights with FP32 scales")
    outputs, inputs = weight.shape
    if inputs <= 0 or inputs % 128 or outputs <= 0 or scales.shape != (triton.cdiv(outputs, 128), inputs // 128):
        raise ValueError("FP8 projection scales must cover the weight's 128x128 blocks")
    flat = x.reshape(-1, inputs).contiguous()
    result = torch.empty((flat.shape[0], outputs), device=x.device, dtype=x.dtype)
    if flat.shape[0]:
        quantized, activation_scales = per_token_group_quant_fp8(flat, 128)
        block_rows = 16 if flat.shape[0] < 32 else 32
        parallel = flat.shape[0] <= 8 and inputs > 512
        if parallel:
            groups = inputs // 128
            partial = torch.empty((groups, *result.shape), device=x.device, dtype=torch.float32)
            _group_products[(triton.cdiv(flat.shape[0], block_rows), triton.cdiv(outputs, 128), groups)](
                quantized, weight, partial, flat.shape[0], outputs, inputs, block_rows, 128, num_warps=4, num_stages=2)  # ty: ignore[invalid-argument-type,unknown-argument]  # Triton launch.
            _accumulate_groups[(triton.cdiv(result.numel(), 256),)](
                partial, activation_scales, scales, result, flat.shape[0], outputs, groups, 256, num_warps=4)  # ty: ignore[invalid-argument-type,unknown-argument]  # Triton launch.
        else:
            _linear[(triton.cdiv(flat.shape[0], block_rows), triton.cdiv(outputs, 128))](
                quantized, weight, activation_scales, scales, result,
                flat.shape[0], outputs, inputs, block_rows, 128, num_warps=4, num_stages=2)  # ty: ignore[invalid-argument-type,unknown-argument]  # Triton launch.
    result = result.view(*x.shape[:-1], outputs)
    return result if bias is None else result + bias
