"""Bounded selected attention without a per-query gathered KV tensor.

Selections are multisets: repeated indices retain their softmax multiplicity.
The BF16 additive mask is log(count) for each cache row, and -inf for absent rows.
SDPA keeps scores/probabilities inside the fused attention kernel, changing
rounding boundaries relative to a BF16 QK/scale/probability reference.
All inputs are read-only. This operator has no cache ownership or causal policy.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F

from ._dispatch import OpNotEligible


def selected_attention_sdpa_eligible(q, kv, indices) -> bool:
    """Accept CUDA BF16 q[B,S,H,D], kv[B,N,D], indices[B,S,W]."""
    if not (
        q.is_cuda
        and torch.version.hip is None
        and q.ndim == 4
        and kv.ndim == 3
        and indices.ndim == 3
        and q.dtype == kv.dtype == torch.bfloat16
        and indices.dtype in (torch.int32, torch.int64)
        and q.device == kv.device == indices.device
    ):
        return False
    batch, seq, heads, dim = q.shape
    if not (
        kv.shape[0] == indices.shape[0] == batch
        and indices.shape[1] == seq
        and kv.shape[2] == dim
        and 0 < batch <= 32
        and 0 <= seq <= 512
        and 0 < heads <= 64
        and 8 <= dim <= 512
        and dim % 8 == 0
        and kv.shape[1] <= 512
        and indices.shape[2] <= 8192
    ):
        return False
    return not (torch.is_grad_enabled() and (q.requires_grad or kv.requires_grad))


@functools.lru_cache(maxsize=1)
def _mask_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def mask(
        IDX,
        BIAS,
        SEQ: tl.constexpr,
        N: tl.constexpr,
        W: tl.constexpr,
        IB: tl.constexpr,
        IS: tl.constexpr,
        IW: tl.constexpr,
        PAD_N: tl.constexpr,
        BLOCK_W: tl.constexpr,
        BINS: tl.constexpr,
    ):
        row = tl.program_id(0)
        slots = tl.arange(0, BLOCK_W)
        selected = tl.load(IDX + (row // SEQ) * IB + (row % SEQ) * IS + slots * IW, slots < W, other=-1)
        # Invalid entries go to an extra bin, never another request's cache.
        selected = tl.where((selected >= 0) & (selected < N), selected, N).to(tl.int32)
        counts = tl.histogram(selected, BINS)
        keys = tl.arange(0, BINS)
        bias = tl.where(counts > 0, tl.log(counts.to(tl.float32)), float("-inf"))
        tl.store(BIAS + row * PAD_N + keys, bias, keys < N)

    return mask


def selected_attention_sdpa(q, kv, indices, scale: float) -> torch.Tensor:
    """Return [B,S,H,D], preserving selections, duplicates and empty rows.

    Negative and out-of-range indices are padding. This inference-only path
    explicitly requires the memory-efficient CUDA SDPA backend; it never silently
    falls back to materializing a large math attention operation.
    """
    if not selected_attention_sdpa_eligible(q, kv, indices) or not math.isfinite(scale):
        raise OpNotEligible("selected_attention_sdpa requires bounded CUDA BF16 inference tensors")
    batch, seq, heads, _ = q.shape
    count, width = kv.shape[1], indices.shape[2]
    if seq == 0 or count == 0 or width == 0:
        return torch.zeros_like(q)
    import triton
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # SDPA accepts broadcast heads; do not replicate the compact cache.
    query = q.transpose(1, 2)
    if query.stride(-1) != 1:
        query = query.contiguous()
    cache = kv if kv.stride(-1) == 1 else kv.contiguous()
    cache = cache[:, None].expand(-1, heads, -1, -1)
    # The efficient backend requires an aligned bias row stride.
    padded_count = triton.cdiv(count, 8) * 8
    storage = torch.empty((batch, seq, padded_count), device=q.device, dtype=q.dtype)
    bias = storage[:, None, :, :count]
    params = torch.backends.cuda.SDPAParams(query, cache, cache, bias, 0.0, False, False)
    if not torch.backends.cuda.can_use_efficient_attention(params):
        raise OpNotEligible("memory-efficient SDPA is unavailable for these tensors")
    with torch.cuda.device(q.device):
        _mask_kernel()[(batch * seq,)](
            indices,
            storage,
            seq,
            count,
            width,
            *indices.stride(),
            padded_count,
            triton.next_power_of_2(width),
            triton.next_power_of_2(count + 1),
            num_warps=4,
        )
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            result = F.scaled_dot_product_attention(query, cache, cache, attn_mask=bias, scale=scale)
    return result.transpose(1, 2)
