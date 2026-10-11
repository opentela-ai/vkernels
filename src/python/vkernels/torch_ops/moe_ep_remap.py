"""Fuse contiguous-block EP route translation without changing route slots."""
from functools import lru_cache

import torch

from ._dispatch import OpNotEligible


def ep_remap_eligible(ids, weights, *, rank, local_experts):
    return (
        ids.is_cuda and ids.device == weights.device
        and ids.ndim == 2 and ids.shape == weights.shape
        and ids.dtype in (torch.int32, torch.int64)
        and weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and ids.is_contiguous() and weights.is_contiguous()
        and rank >= 0 and local_experts > 0
    )


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=['size'])
    def remap(ids, weights, local_ids, local_weights, size,
              offset: tl.constexpr, experts: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)*block + tl.arange(0, block)
        source = tl.load(ids+row, row < size, 0)
        local = source-offset
        valid = (local >= 0) & (local < experts)
        weight = tl.load(weights+row, row < size, 0)
        masked = weight * valid.to(weight.dtype)
        tl.store(local_ids+row, tl.minimum(tl.maximum(local, 0), experts-1), row < size)
        tl.store(local_weights+row, masked, row < size)

    return triton, remap


def ep_remap(ids, weights, *, rank, local_experts):
    """Return local clamped IDs and exactly the old zero-weight masking.

    Slot order, dtypes and shapes stay unchanged. Non-local IDs clamp to a
    valid stack index and weights multiply by zero, including IEEE NaN and
    signed-zero behavior. No live count or device scalar is read on the host.
    """
    if not ep_remap_eligible(ids, weights, rank=rank, local_experts=local_experts):
        raise OpNotEligible('EP remap requires same-device contiguous CUDA routing matrices')
    local_ids = torch.empty_like(ids)
    local_weights = torch.empty_like(weights)
    if ids.numel():
        triton, kernel = _kernel()
        kernel[(triton.cdiv(ids.numel(), 256),)](
            ids, weights, local_ids, local_weights, ids.numel(),
            offset=rank*local_experts, experts=local_experts, block=256)
    return local_ids, local_weights
