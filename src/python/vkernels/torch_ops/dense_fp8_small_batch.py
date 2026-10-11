"""Small-M tensor-core GEMM with FP8 weight dequantization inside each tile.

Weights round to BF16 at the loader boundary, then BF16 dot products accumulate
in FP32. Split-K uses an owned FP32 buffer and a separate reduction, not atomics.
Target shape selection requires independent accelerator and serving validation.
"""
from functools import lru_cache

from ._dispatch import require, same_gpu_contiguous

# Historical SM90 campaign picks; combined-tree serving/NLL qualification
# remains outstanding. This explicit experimental API is absent from dispatch.
SM90_CONFIGS = {
    (3, 1536, 4096): (64, 16, 4),
    (3, 3072, 4096): (128, 16, 4),
    (3, 4096, 4096): (128, 16, 4),
    (4, 1536, 4096): (64, 16, 4),
    (4, 3072, 4096): (128, 16, 4),
    (4, 4096, 4096): (128, 16, 4),
}


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def dot_kernel(X, W, S, P, M: tl.constexpr, O: tl.constexpr, K: tl.constexpr,
                   BN: tl.constexpr, SPLIT: tl.constexpr):
        n = tl.program_id(0) * BN + tl.arange(0, BN)
        split = tl.program_id(1)
        m = tl.arange(0, 16)
        k = tl.arange(0, 128)
        iterations: tl.constexpr = triton.cdiv(K, SPLIT * 128)
        acc = tl.zeros((16, BN), tl.float32)
        for iteration in range(iterations):
            kk = (split * iterations + iteration) * 128 + k
            group = (split * iterations + iteration)
            raw = tl.load(W + n[None, :] * K + kk[:, None],
                          (n[None, :] < O) & (kk[:, None] < K), other=0)
            scale = tl.load(S + (n // 128) * (K // 128) + group,
                            (n < O) & (group < K // 128), other=0)
            weight = (raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                      * scale[None, :]).to(tl.bfloat16)
            x = tl.load(X + m[:, None] * K + kk[None, :],
                        (m[:, None] < M) & (kk[None, :] < K), other=0)
            acc = tl.dot(x, weight, acc)
        tl.store(P + split * M * O + m[:, None] * O + n[None, :], acc,
                 (m[:, None] < M) & (n[None, :] < O))

    @triton.jit
    def reduce_kernel(P, Y, M: tl.constexpr, O: tl.constexpr,
                      SPLIT: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        splits = tl.arange(0, SPLIT)
        values = tl.load(P + splits[:, None] * M * O + index[None, :],
                         index[None, :] < M * O, other=0)
        result = tl.sum(values, axis=0)
        tl.store(Y + index, result.to(tl.bfloat16), index < M * O)

    return triton, dot_kernel, reduce_kernel


def _validate(x, weight, scales):
    import torch

    require(x.ndim == weight.ndim == scales.ndim == 2, "expected 2D operands")
    require(x.dtype == torch.bfloat16 and weight.dtype == torch.float8_e4m3fn
            and scales.dtype == torch.float32, "expected BF16/FP8/FP32 operands")
    require(torch.version.hip is None, "CUDA implementation only")
    same_gpu_contiguous(x, weight, scales)
    m, k = x.shape
    o, wk = weight.shape
    require(2 <= m <= 4 and k == wk and 0 < k <= 16384 and 0 < o <= 16384
            and k % 128 == o % 128 == 0, "small-M block-scaled shape required")
    require(scales.shape == (o // 128, k // 128), "scale grid mismatch")
    require(not any(t.requires_grad for t in (x, weight, scales)), "inference only")
    require(torch.cuda.get_device_capability(x.device)[0] >= 9, "SM90 or newer required")
    return m, o, k


def _run(x, weight, scales, config):
    import torch

    m, o, k = _validate(x, weight, scales)
    bn, split, warps = config
    require(bn in (32, 64, 128) and split in (1, 2, 4, 8, 16, 32)
            and warps in (4, 8), "unsupported tile configuration")
    triton, dot, reduce = _kernels()
    out = torch.empty((m, o), dtype=torch.bfloat16, device=x.device)
    partial = torch.empty((split, m, o), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        dot[(triton.cdiv(o, bn), split)](x, weight.view(torch.uint8), scales, partial,
                                        m, o, k, bn, split, num_warps=warps, num_stages=2)
        reduce[(triton.cdiv(m * o, 128),)](partial, out, m, o, split, 128, num_warps=4)
    return out


def dense_fp8_small_batch(x, weight, scales):
    """Experimental BF16 [M,O]; decline shapes outside the SM90 candidate table."""
    import torch

    m, o, k = _validate(x, weight, scales)
    config = SM90_CONFIGS.get((m, o, k))
    require(torch.cuda.get_device_capability(x.device) == (9, 0) and config is not None,
            "shape/device has no qualified small-batch tile")
    return _run(x, weight, scales, config)
