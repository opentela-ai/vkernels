"""Experimental BF16 tensor-core expert projection with tiled FP8 decoding.

This module is deliberately absent from dispatch. Weight decoding rounds to
BF16 before FP32 accumulation, but tensor-core reduction order differs from
the serving GEMV oracle. Operator and end-to-end qualification are separate.
No sorting, activation quantization, resident weight copy or atomic reduction.
"""
from functools import lru_cache

from ._dispatch import require, same_gpu_contiguous


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def project(X, W, S, IDX, Y, O: tl.constexpr, I: tl.constexpr,
                K: tl.constexpr, ROUTED: tl.constexpr, BN: tl.constexpr):
        selected = tl.program_id(0)
        expert = tl.load(IDX + selected).to(tl.int64)
        source = selected if ROUTED else selected // K
        n = tl.program_id(1) * BN + tl.arange(0, BN)
        m = tl.arange(0, 16)
        k = tl.arange(0, 128)
        acc = tl.zeros((16, BN), tl.float32)
        for group in range(I // 128):
            kk = group * 128 + k
            raw = tl.load(W + expert * O * I + n[None, :] * I + kk[:, None],
                          n[None, :] < O, other=0)
            scale = tl.load(S + (expert * (O // 128) + n // 128) * (I // 128) + group,
                            n < O, other=0)
            weight = (raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                      * scale[None, :]).to(tl.bfloat16)
            # Only one token is active per selected route. Duplicate grouping
            # is a separate experiment; padding does not imply weight reuse.
            x = tl.load(X + source * I + kk[None, :], m[:, None] == 0, other=0)
            acc = tl.dot(x, weight, acc)
        result = tl.sum(tl.where(m[:, None] == 0, acc, 0.0), axis=0)
        tl.store(Y + selected * O + n, result, n < O)

    return triton, project


def expert_fp8_tensorcore(x, weight, scales, indices, *, tile=32):
    """Project [T,I] or [T,K,I] to BF16 [T,K,O]; routed IDs must be valid.

    No host read of device routing is permitted during CUDA graph capture.
    The caller owns the same valid-ID contract as expert_gemv.
    """
    import torch

    require(x.ndim in (2, 3) and weight.ndim == scales.ndim == 3
            and indices.ndim == 2, "expected token/routed activations and expert stacks")
    require(x.dtype == torch.bfloat16 and weight.dtype == torch.float8_e4m3fn
            and scales.dtype == torch.float32 and indices.dtype == torch.int64,
            "expected BF16/FP8/FP32/int64 operands")
    require(torch.version.hip is None, "CUDA implementation only")
    same_gpu_contiguous(x, weight, scales, indices)
    t, k = indices.shape
    e, o, i = weight.shape
    require(0 < t <= 8 and 0 < k <= 32 and e > 0 and 0 < o <= 16384
            and 0 < i <= 16384 and o % 128 == i % 128 == 0,
            "unsupported block-scaled decode shape")
    require(x.shape == ((t, i) if x.ndim == 2 else (t, k, i)),
            "activation/routing shape mismatch")
    require(scales.shape == (e, o // 128, i // 128), "scale grid mismatch")
    require(tile in (32, 64, 128), "unsupported output tile")
    require(not any(v.requires_grad for v in (x, weight, scales)), "inference only")
    require(torch.cuda.get_device_capability(x.device)[0] >= 9, "SM90 or newer required")
    triton, project = _kernel()
    out = torch.empty((t, k, o), device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        project[(t * k, triton.cdiv(o, tile))](
            x, weight.view(torch.uint8), scales, indices, out, o, i, k,
            x.ndim == 3, tile, num_warps=4, num_stages=2,
        )
    return out
