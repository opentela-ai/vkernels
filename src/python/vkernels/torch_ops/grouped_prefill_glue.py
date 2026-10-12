"""Launch fusions for grouped GEMM, retaining the eager integer/FP8 results.

No routing policy, weight storage, reduction order or cache state lives here.
The quantizer deliberately uses the eager scalar-division reciprocal rounding
for scales and round-to-nearest tensor division for payloads.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ._dispatch import OpNotEligible


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def tile_map(C, S, E, R, M, NC: tl.constexpr, CS: tl.constexpr,
                 SS: tl.constexpr, BM: tl.constexpr, CAP: tl.constexpr,
                 EXPERTS: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr):
        experts = tl.arange(0, EXPERTS)
        counts = tl.load(C + experts * CS, experts < NC, 0)
        tiles = (counts + BM - 1) // BM
        ends = tl.cumsum(tiles)
        offsets = ends - tiles
        total = tl.sum(tiles)
        tid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        # Upper bound (right=True), including consecutive empty experts.
        lo = tl.full((BLOCK,), 0, tl.int32)
        hi = tl.full((BLOCK,), NC, tl.int32)
        for _ in range(STEPS):
            mid = (lo + hi) // 2
            end = tl.gather(ends, tl.minimum(mid, NC - 1), axis=0)
            right = (mid < NC) & (end <= tid)
            active = lo < hi
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
        expert = tl.minimum(lo, NC - 1)
        local = tid - tl.gather(offsets, expert, axis=0)
        count = tl.gather(counts, expert, axis=0)
        start = tl.load(S + expert * SS) + local * BM
        size = tl.minimum(tl.maximum(count - local * BM, 0), BM)
        tl.store(E + tid, expert, tid < CAP)
        tl.store(R + tid, tl.where(tid < total, start, 0), tid < CAP)
        tl.store(M + tid, tl.where(tid < total, size, 0), tid < CAP)

    @triton.jit
    def quantize(X, Q, S, K: tl.constexpr, XS0: tl.constexpr,
                 XS1: tl.constexpr, GROUP: tl.constexpr):
        row = tl.program_id(0)
        group = tl.program_id(1)
        col = group * GROUP + tl.arange(0, GROUP)
        x = tl.load(X + row * XS0 + col * XS1).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-4)
        # PyTorch's scalar DivFunctor precomputes an FP32 reciprocal.
        reciprocal = tl.full((), 1.0 / 448.0, tl.float32)
        scale = tl.maximum(amax * reciprocal, 1.0e-12)
        q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
        tl.store(Q + row * K + col, q)
        tl.store(S + row * (K // GROUP) + group, scale)

    return triton, tile_map, quantize


def grouped_tile_map(counts, segments, block_rows: int, capacity: int):
    """Exact static tile map from nonnegative counts and exclusive segments.

    Counts/segments are read-only int64 vectors; empty experts and padded
    tiles retain the eager searchsorted/clamp semantics. Capacity is a host
    shape bound. No device values are read by Python, including under replay.
    """
    if not (
        counts.is_cuda and torch.version.hip is None
        and counts.ndim == segments.ndim == 1
        and counts.shape == segments.shape
        and counts.dtype == segments.dtype == torch.int64
        and counts.device == segments.device
        and 0 < counts.numel() <= 1024
        and isinstance(block_rows, int) and 0 < block_rows <= 1024
        and isinstance(capacity, int) and 0 <= capacity <= 131072
    ):
        raise OpNotEligible("tile map requires bounded CUDA int64 counts/segments")
    triton, kernel, _ = _kernels()
    outputs = tuple(torch.empty(capacity, dtype=torch.int64, device=counts.device) for _ in range(3))
    if capacity:
        with torch.cuda.device(counts.device):
            kernel[(triton.cdiv(capacity, 256),)](
                counts, segments, *outputs, counts.numel(), counts.stride(0),
                segments.stride(0), block_rows, capacity,
                triton.next_power_of_2(counts.numel()), 256,
                counts.numel().bit_length() + 1, num_warps=4)
    return outputs


def quantize_per_token_fp8_exact(x, group_size: int = 128):
    """BF16 matrix -> E4M3FN bytes and FP32 scales, matching eager quantization.

    Finite activations, including BF16 subnormals and zero groups, use the
    unchanged amax floor 1e-4, scale floor 1e-12 and saturation at +/-448.
    Arbitrary input strides are accepted; returned tensors are contiguous.
    """
    if not (
        x.is_cuda and torch.version.hip is None and x.ndim == 2
        and x.dtype == torch.bfloat16
        and not (torch.is_grad_enabled() and x.requires_grad)
        and isinstance(group_size, int) and 0 < group_size <= 1024
        and group_size & (group_size - 1) == 0
        and x.shape[1] > 0 and x.shape[1] % group_size == 0
    ):
        raise OpNotEligible("exact quantizer requires CUDA BF16 inference matrix and power-of-two groups")
    _, _, kernel = _kernels()
    rows, columns = x.shape
    q = torch.empty((rows, columns), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((rows, columns // group_size), dtype=torch.float32, device=x.device)
    if rows:
        with torch.cuda.device(x.device):
            kernel[(rows, columns // group_size)](
                x, q, scales, columns, *x.stride(), group_size,
                num_warps=4 if group_size >= 128 else 1, enable_fp_fusion=False)
    return q, scales
