"""Measured gfx942 BF16 small-batch projections, with FP32 accumulation."""
from functools import lru_cache

from ._dispatch import OpNotEligible, require, same_gpu_contiguous
from .glm_gemv import _is_gfx942

# (outputs, inputs) -> (pipeline stages, K partitions), M2–4.
# Confirmed with >512 MiB rotating weights (Beverin 696978/697320). One layout
# per matrix keeps the arithmetic consistent across these batch sizes.
# M1 retains the original GEMV. Unlisted matrices fall back at the caller.
_CONFIGS = {
    (21504, 4096): (2, 1),
    (4096, 21504): (1, 8),
    (10752, 8192): (2, 2),
    (8192, 10752): (1, 4),
    (6144, 4096): (2, 4),
    (4096, 4096): (2, 8),
    (2560, 8192): (2, 16),
    (8192, 2048): (2, 4),
    (131072, 4096): (2, 1),
    (131072, 8192): (2, 1),
}

# Beverin 697973/697984: cold rotating weights, FP64 oracle, M2–32.
# Existing M2–4 reductions above keep their original partitioning.
_HIGH_CONFIGS = {
    # SP up-projection keeps BLAS above M4: the M32 sweep did not win.
    (4096, 21504): (1, 8, 128),
    (6144, 4096): (2, 4, 128),
    (4096, 4096): (1, 8, 128),
    (5376, 4096): (2, 4, 128),
    (4096, 5376): (2, 4, 128),
    (1536, 4096): (2, 4, 64),
    (4096, 1024): (1, 4, 128),
    (10752, 8192): (1, 4, 128),
    (8192, 10752): (1, 4, 128),
    (2560, 8192): (2, 4, 128),
    (8192, 2048): (1, 4, 128),
    (131072, 4096): (1, 1, 128),
    (131072, 8192): (1, 1, 128),
}


@lru_cache(maxsize=1)
def _kernels():
    global tl
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise OpNotEligible("BF16 small-batch matrix kernels require Triton") from error

    @triton.jit
    def _gemm_bf16_small(X, W, P, M: tl.constexpr, O: tl.constexpr,
                         I: tl.constexpr, SPLITS: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr):
        row = tl.arange(0, BM)
        column = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.program_id(1)
        k = split * (I // SPLITS) + tl.arange(0, 128)
        end = (split + 1) * (I // SPLITS)
        acc = tl.zeros((BM, BN), tl.float32)
        for offset in range(0, I // SPLITS, 128):
            a = tl.load(X + row[:, None] * I + (k + offset)[None, :],
                        (row[:, None] < M) & ((k + offset)[None, :] < end), 0)
            b = tl.load(W + column[None, :] * I + (k + offset)[:, None],
                        (column[None, :] < O) & ((k + offset)[:, None] < end), 0)
            acc = tl.dot(a, b, acc)
        tl.store(P + split * M * O + row[:, None] * O + column[None, :], acc,
                 (row[:, None] < M) & (column[None, :] < O))

    @triton.jit
    def _gemm_bf16_merge(P, Y, N: tl.constexpr, SPLITS: tl.constexpr):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        split = tl.arange(0, SPLITS)
        partial = tl.load(P + split[:, None] * N + index[None, :], index[None, :] < N, 0)
        tl.store(Y + index, tl.sum(partial, axis=0), index < N)

    return triton, _gemm_bf16_small, _gemm_bf16_merge


def dense_bf16_small_batch(x, weight):
    """Use measured MFMA matrices for M2–32 on gfx942.

    Bias-free contiguous BF16 inputs/output; products and split partials
    accumulate in FP32. The caller must distinguish this numerical recipe.
    Unsupported contracts raise OpNotEligible for the caller's fallback.
    """
    import torch

    require(x.ndim == weight.ndim == 2, "expected BF16 [M,I] and [O,I]")
    m, i = x.shape
    shape = tuple(weight.shape)
    config = _CONFIGS.get(shape) if 2 <= m <= 4 else None
    bn = 64
    if config is None and 2 <= m <= 32 and shape in _HIGH_CONFIGS:
        stages, splits, bn = _HIGH_CONFIGS[shape]
        if m <= 16:
            stages, bn = 2, 64
        config = stages, splits
    if config is None:
        raise OpNotEligible("no measured BF16 small-batch matrix for this shape")
    same_gpu_contiguous(x, weight)
    require(_is_gfx942(x.device), "measured small-batch matrices require gfx942")
    require(x.dtype == weight.dtype == torch.bfloat16 and i == weight.shape[1],
            "small-batch matrices require matching BF16 operands")
    require(not torch.is_grad_enabled() or not (x.requires_grad or weight.requires_grad),
            "small-batch matrices are inference-only")
    o = weight.shape[0]
    stages, splits = config
    triton, matmul, merge = _kernels()
    with torch.cuda.device(x.device):
        output = torch.empty((m, o), device=x.device, dtype=x.dtype)
        partial = output if splits == 1 else torch.empty((splits, m, o), device=x.device, dtype=torch.float32)
        matmul[(triton.cdiv(o, bn), splits)](x, weight, partial, m, o, i, splits,
                                            32 if m > 16 else 16, bn,
                                            num_warps=4, num_stages=stages)
        if splits > 1:
            merge[(triton.cdiv(m * o, 256),)](partial, output, m * o, splits, num_warps=4)
    return output
