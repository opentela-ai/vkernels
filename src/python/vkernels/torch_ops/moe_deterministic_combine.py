"""Default-off deterministic prefill MoE reduction with unchanged products.

Original flat route slots are unique. Scatter assigns, never atomically adds;
each token's route contributions are then added in fixed slot order in FP32.
"""
from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def inverse_order(order, inverse, size: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0) * block + tl.arange(0, block)
        original = tl.load(order + row, row < size, 0)
        tl.store(inverse + original, row, row < size)

    @triton.jit
    def combine(down, inverse, weights, output,
                hidden: tl.constexpr, top_k: tl.constexpr, block: tl.constexpr):
        token = tl.program_id(0)
        col = tl.program_id(1) * block + tl.arange(0, block)
        acc = tl.zeros((block,), tl.float32)
        for slot in range(top_k):
            original = token * top_k + slot
            row = tl.load(inverse + original)
            value = tl.load(down + row * hidden + col, col < hidden, 0).to(tl.float32)
            weight = tl.load(weights + original).to(tl.float32)
            product = value * weight
            acc = acc + product
        tl.store(output + token * hidden + col, acc, col < hidden)

    return triton, inverse_order, combine


def _fused_route_combine(down, order, weights):
    triton, inverse_kernel, combine_kernel = _kernels()
    tokens, top_k = weights.shape
    hidden = down.shape[1]
    output = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=down.device)
    if tokens == 0 or hidden == 0:
        return output
    inverse = torch.empty_like(order)
    inverse_kernel[(triton.cdiv(order.numel(), 256),)](
        order, inverse, size=order.numel(), block=256)
    # The separate FP32 multiply rounding and original-slot addition order
    # are part of the reference contract. FMA would change that contract.
    combine_kernel[(tokens, triton.cdiv(hidden, 256))](
        down, inverse, weights, output, hidden=hidden, top_k=top_k,
        block=256, enable_fp_fusion=False)
    return output


def _fixed_sum(rows):
    out = torch.zeros((rows.shape[0], rows.shape[2]), dtype=torch.float32, device=rows.device)
    for slot in range(rows.shape[1]):
        out.add_(rows[:, slot])
    return out.to(torch.bfloat16)


def deterministic_route_combine(down, order, topk_weights):
    """BF16 expert-sorted rows -> original-route FP32 products -> BF16 sum.

    Caller supplies the stable-sort permutation of all T*K original slots.
    No synchronization or data-dependent order/permutation validation occurs.
    """
    if (down.ndim != 2 or topk_weights.ndim != 2 or order.ndim != 1
            or down.dtype != torch.bfloat16 or order.dtype != torch.int64
            or not topk_weights.is_floating_point()
            or len({down.device, order.device, topk_weights.device}) != 1):
        raise ValueError("deterministic combine requires BF16 rows, int64 permutation, same-device floating weights")
    t, k = topk_weights.shape
    if k < 1 or down.shape[0] != t * k or order.shape != (t * k,):
        raise ValueError("deterministic combine requires one unique permutation slot per routed row")
    if (down.is_cuda and down.is_contiguous() and order.is_contiguous()
            and topk_weights.is_contiguous()):
        return _fused_route_combine(down, order, topk_weights)
    # Exactly the frozen native expression's FP32 products, before changing
    # ONLY reduction order. scatter has unique original-route destinations.
    weighted = down.float() * topk_weights.reshape(-1)[order].float()[:, None]
    original = torch.empty_like(weighted)
    original.index_copy_(0, order, weighted)
    return _fixed_sum(original.reshape(t, k, down.shape[1]))


def deterministic_compacted_combine(down, flat_slots, *, tokens, top_k):
    """EP outer combine: rows already carry the native BF16 weighted result.

    flat_slots are unique live positions from the original [tokens, top_k]
    routing. Missing (exact-zero weight) positions are initialized to zero.
    """
    if (down.ndim != 2 or down.dtype != torch.bfloat16 or flat_slots.ndim != 1
            or flat_slots.dtype != torch.int64 or down.device != flat_slots.device
            or flat_slots.numel() != down.shape[0] or tokens < 0 or top_k < 1):
        raise ValueError("deterministic EP combine requires BF16 live rows and unique int64 original slots")
    original = torch.zeros((tokens * top_k, down.shape[1]), dtype=torch.float32, device=down.device)
    original.index_copy_(0, flat_slots, down.float())
    return _fixed_sum(original.reshape(tokens, top_k, down.shape[1]))
