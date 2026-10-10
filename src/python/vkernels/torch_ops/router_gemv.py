"""FP32 router scoring GEMV — the Glm53TopkRouter logits GEMM off BLAS.

The GLM-5.3 router runs ``F.linear(x.float(), w_f32)`` per MoE layer per
decode step: ``[1, 4096] @ [288, 4096]`` in fp32 — a 4.7 MB weight read
that rocBLAS executes as a small-tile Cijk GEMM at ~64 us (~74 GB/s, the
M=1 skinny-GEMM heuristic misfire; the 658395 trace: the persistent
~80-call MT128x32x32 block, ~2.7 ms/step across 42 MoE layers). The
``fused_router`` op only fuses the POST-GEMM glue (sigmoid/bias/top-k/
normalize) — the logits GEMM itself was always plain BLAS.

This op computes the same logits with a row-block GEMV kernel: one CTA
per ``ROWS`` experts, the 4096-wide fp32 dot per row, fp32 accumulation
throughout, ONE fp32 round exactly where cuBLAS's fp32 GEMM rounds (the
store) — the same rounding class with a different reduction order (the
standard perf-only class; the HF-parity concern is bf16-rounding logits
BEFORE sigmoid, which neither path does).

House pattern (``kda_small_gemv``): lazy Triton, OpNotEligible contract,
warmup tracking for graph capture, inference-only.
"""

from ._dispatch import OpNotEligible
from functools import lru_cache

_WARMED = set()
_MAX_M = 8  # the decode-GEMV m<=8 policy

__all__ = ["router_gemv", "router_gemv_eligible", "router_gemv_reference"]


@lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def _router_gemv(X, W, Y, M, E, K, MP: tl.constexpr, ROWS: tl.constexpr, BK: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        rmask = rows < E
        acc = tl.zeros((MP, ROWS), dtype=tl.float32)
        for k0 in range(0, K, BK):
            ks = k0 + tl.arange(0, BK)
            x = tl.load(X + tl.arange(0, MP)[:, None] * K + ks[None, :],
                        mask=(tl.arange(0, MP)[:, None] < M) & (ks[None, :] < K), other=0.0)
            w = tl.load(W + rows[:, None] * K + ks[None, :],
                        mask=rmask[:, None] & (ks[None, :] < K), other=0.0)
            acc += tl.sum(x[:, None, :] * w[None, :, :], 2)
        tl.store(Y + tl.arange(0, MP)[:, None] * E + rows[None, :], acc,
                 mask=(tl.arange(0, MP)[:, None] < M) & rmask[None, :])

    return triton, _router_gemv


def router_gemv_eligible(x, w) -> None:
    """Raise :class:`OpNotEligible` unless the contract holds."""
    import torch

    if x.ndim not in (1, 2) or w.ndim != 2:
        raise OpNotEligible("expected x [M, K] (or [K]) and router weight [E, K], both fp32")
    m = 1 if x.ndim == 1 else x.shape[0]
    e, k = w.shape
    if m < 1 or m > _MAX_M:
        raise OpNotEligible(f"M={m} outside the 1..{_MAX_M} decode-GEMV policy")
    if k < 1 or k < 64 or k % 64:
        raise OpNotEligible(f"K={k} outside the 64-multiple router contract")
    if e < 1:
        raise OpNotEligible(f"E={e} must be positive")
    if x.dtype != torch.float32 or w.dtype != torch.float32:
        raise OpNotEligible("router_gemv requires FP32 inputs (the HF router parity contract)")
    if not (x.is_cuda and x.device == w.device):
        raise OpNotEligible("inputs must share a GPU device")
    if not (x.is_contiguous() and w.is_contiguous()):
        raise OpNotEligible("router_gemv requires contiguous inputs")


def router_gemv(x, w):
    """Logits ``[M, E]`` fp32 for x ``[M, K]`` fp32, w ``[E, K]`` fp32."""
    import torch

    router_gemv_eligible(x, w)
    m = 1 if x.ndim == 1 else x.shape[0]
    e, k = w.shape
    triton, kern = _kernel()
    # MI300A sweep (658479, [288,4096] fp32 M=1): ROWS=1/BK=4096 = 3.06us
    # (1.56 TB/s, 288 CTAs) vs cuBLAS fp32 18.77us -- one full-width dot
    # per expert row maximizes CTA count; multi-row tiles starve the grid.
    y = torch.empty(m, e, device=x.device, dtype=torch.float32)
    mp = triton.next_power_of_2(m)
    # Register budget: the x tile is [MP, BK] fp32 -- full-width BK at
    # MP>=4 spills (the 658480 width-4 conc regression, 41.3s vs 16.6).
    # Keep MP*BK <= 8192 floats per CTA.
    bk = triton.next_power_of_2(k)
    while mp * bk > 8192 and bk > 512:
        bk //= 2
    with torch.cuda.device(x.device):
        kern[(e,)](
            x.reshape(m, k), w, y, m, e, k,
            MP=mp, ROWS=1, BK=bk,
        )
    _WARMED.add((x.device.index, m, e, k))
    return y


def router_gemv_reference(x, w):
    """The eager parity oracle: fp32 F.linear (cuBLAS fp32 GEMM)."""
    import torch.nn.functional as F

    return F.linear(x, w)
