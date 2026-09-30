"""Weighted top-k combine for MoE decode — one Triton launch.

The routed-expert decode paths produce the per-assignment activations
``[T, K, H]`` and finish with ``(out * w.unsqueeze(-1)).sum(dim=1)`` — an
eager broadcast multiply plus a reduce, two TensorIterator launches per MoE
layer (84/step at T<=8 decode, ~168 kernels when both the fused-GEMV and the
einsum paths count). This module replaces that pair with ONE kernel.

The kernel is a weighted adaptation of SGLang's ``_moe_sum_reduce_kernel``
(``sglang.kernels.ops.moe.fused_moe_triton_kernels``, itself credited to
lightllm; Apache-2.0), measured as the donor's combine in the sgs-gpu07
nsys donor profile (42 calls/step, 0.26 ms at B=1 — SGLang applies the
routing weights in the down-GEMM epilogue, so their reduce is an unweighted
sum; floe's expert-GEMV op has no weight argument, hence the weighted load
added here — the only functional change from the donor kernel).

Adopted from floe (``floe/engine/runner/kernels/moe_combine.py``, the
sgs-gpu07 perf campaign) under the #64/#65 re-export model: vkernels owns
the kernel and the contract, floe re-imports it through its ``vkl_ops``
adapter. The only adoption delta besides the move: a contract miss raises
:class:`~vkernels.torch_ops._dispatch.OpNotEligible` (which subclasses
``TypeError``, the exception the floe-local original raised — every
existing catch keeps working).

Rounding contract: eager rounds ``w`` to the activation dtype, rounds the
product, then reduces (torch bf16 reductions accumulate in fp32); this
kernel multiplies and accumulates in fp32 and rounds ONCE at the store —
the same contract the vkernels grouped path already ships
(``restored[order] = weighted`` precedent). Parity is tolerance-gated
(bf16 ulp), not bit-identical; the eager expression stays the oracle and
the fallback.

Stdlib+torch+triton only; the JIT compiles on first CUDA call (the eager
decode warmup therefore covers graph capture).
"""

from __future__ import annotations

import functools

from typing import Optional

import torch

from ._dispatch import OpNotEligible

__all__ = ["moe_weighted_sum", "moe_weighted_sum_eligible"]


def moe_weighted_sum_eligible(out: torch.Tensor, weights: torch.Tensor) -> bool:
    """Contract check for :func:`moe_weighted_sum` (cheap, no device sync)."""
    return (
        out.is_cuda
        and out.dim() == 3
        and weights.dim() == 2
        and weights.shape[0] == out.shape[0]
        and weights.shape[1] == out.shape[1]
        and out.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and weights.dtype == torch.float32
        and out.shape[2] > 0
    )


@functools.lru_cache(maxsize=1)
def _kernel():
    """JIT-scoped kernel definition (import triton lazily like vkernels)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _weighted_moe_sum_reduce_kernel(
        input_ptr,
        input_stride_0,
        input_stride_1,
        input_stride_2,
        weights_ptr,
        weights_stride_0,
        weights_stride_1,
        output_ptr,
        output_stride_0,
        output_stride_1,
        token_num,
        topk_num,
        hidden_dim,
        BLOCK_M: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
        input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
        input_stride_2 = tl.cast(input_stride_2, dtype=tl.int64)
        weights_stride_0 = tl.cast(weights_stride_0, dtype=tl.int64)
        weights_stride_1 = tl.cast(weights_stride_1, dtype=tl.int64)
        output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)
        output_stride_1 = tl.cast(output_stride_1, dtype=tl.int64)

        token_block_id = tl.program_id(0)
        dim_block_id = tl.program_id(1)

        offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        mask_token = offs_token < token_num
        mask_dim = offs_dim < hidden_dim

        accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
        base_ptrs = (
            input_ptr
            + offs_token[:, None] * input_stride_0
            + offs_dim[None, :] * input_stride_2
        )
        for i in range(0, topk_num):
            w = tl.load(
                weights_ptr + offs_token * weights_stride_0 + i * weights_stride_1,
                mask=mask_token,
                other=0.0,
            ).to(tl.float32)
            tile = tl.load(
                base_ptrs + i * input_stride_1,
                mask=mask_token[:, None] & mask_dim[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator += w[:, None] * tile

        store_ptrs = (
            output_ptr
            + offs_token[:, None] * output_stride_0
            + offs_dim[None, :] * output_stride_1
        )
        tl.store(
            store_ptrs,
            accumulator.to(output_ptr.dtype.element_ty),
            mask=mask_token[:, None] & mask_dim[None, :],
        )

    return _weighted_moe_sum_reduce_kernel


def moe_weighted_sum(out: torch.Tensor, weights: torch.Tensor, *, result: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Combine ``out`` [T, K, H] with routing ``weights`` [T, K] (fp32).

    Returns ``[T, H]``: fp32 multiply-accumulate over K, one rounding at
    the store dtype. ``result`` may pre-allocate the output (graph-friendly
    reuse); otherwise it is allocated. Raises :class:`OpNotEligible` when
    the contract in :func:`moe_weighted_sum_eligible` is not met — callers
    gate on that check and keep their eager expression as the fallback.
    """
    if not moe_weighted_sum_eligible(out, weights):
        raise OpNotEligible(
            f"moe_weighted_sum contract: CUDA [T,K,H] bf16/fp16/fp32 tensor with "
            f"fp32 [T,K] weights (got {out.device} {tuple(out.shape)} {out.dtype}, "
            f"{tuple(weights.shape)} {weights.dtype})"
        )
    import triton

    if result is None:
        result = torch.empty(out.shape[0], out.shape[2], device=out.device, dtype=out.dtype)
    token_num, topk_num, hidden_dim = out.shape
    block_dim = min(triton.next_power_of_2(hidden_dim), 4096)
    grid = (token_num, triton.cdiv(hidden_dim, block_dim))
    _kernel()[grid](
        out,
        *out.stride(),
        weights,
        *weights.stride(),
        result,
        *result.stride(),
        token_num,
        topk_num,
        hidden_dim,
        BLOCK_M=1,
        BLOCK_DIM=block_dim,
        num_warps=16,
    )
    return result
