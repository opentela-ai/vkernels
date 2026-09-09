"""GLM-5.3-Flash fp8 blockwise-scaled grouped GEMM (sm_90, CuTe DSL).

The MoE layers keep routed experts in checkpoint fp8-e4m3 with per-128x128
DeepSeek-scheme block scales (``w_bf16 = w_fp8 * S[n//128, k//128]``). The
decode/verify path streams them through :mod:`glm_expert_gemv` (one GEMV per
(token, expert) pair, bandwidth-bound at small M); PREFILL and batched serving
need a tensor-core grouped GEMM instead. This module provides it in three
layers:

1. ``quantize_activations_fp8`` — per-token per-128-group e4m3 activation
   quantization (the DeepSeek/VLLM serving contract): returns (x_fp8,
   x_scales[m, K/128]). Required because Hopper WGMMA needs BOTH operands in
   fp8 (no bf16 x fp8 mixed MMA on sm_90).
2. ``fp8_blockwise_gemm`` — the single-expert primitive:
   ``D[m,n] = (sum_k A_fp8[m,k] * B_fp8[n,k]) elementwise-scaled by
   Sa[m, k//128] * Sb[n//128, k//128]`` accumulated per K-block, BF16 output.
   Dispatches to the CuTe DSL kernel below when available/importable, else to
   a pure-torch reference path with identical semantics (block-dequant +
   per-block-scaled accumulation).
3. ``glm_moe_grouped_gemm`` — the MoE-shaped orchestration: host-side
   sort-by-expert of the (token, slot) routing, one blockwise GEMM per ACTIVE
   expert over its gathered activations, epilogue applying routing weights.
   The persistent fused grouped kernel (ptr-array scheduling inside one
   launch) replaces the per-expert loop later WITHOUT changing this API.

Numerics contract (vs the bf16-activation GEMV path): activation e4m3
quantization adds ~1e-2-class relative noise on top of the weight-dequant
rounding the GEMV already does. The MoE parity gate (floe loop) therefore
uses a looser tolerance for this path; logit-level effects are measured by
the NLL/quality gates, as with any fp8-activation serving stack.

The CuTe DSL kernel (``HopperFP8BlockwiseGemmKernel``) is a lean adaptation
of cutlass's ``dense_gemm_fp8_2xacc.py`` example: warp-specialized
TMA+WGMMA with the two-level (2xAcc) accumulator, where the promotion every
``mma_promotion_interval`` k-MMAs (aligned to 128-wide K blocks) multiplies
``accum_temp`` by the per-block scale tiles ``Sa[:, kb] (x) Sb[:, kb]``
instead of a scalar. VALIDATION: first compile/run happens on a GH200 job
(``tests/python/test_glm_fp8_blockwise_gemm.py`` gates on CUDA + the DSL);
the torch path is the oracle.

Requires the optional ``cute`` extra: ``uv pip install '.[cute]'``
(nvidia-cutlass-dsl). Import of this module stays dependency-free; only the
kernel path imports the DSL lazily.
"""

from __future__ import annotations

import os
from functools import lru_cache


# ---------------------------------------------------------------------------
# 1. activation quantization (DeepSeek per-token per-128-group e4m3)
# ---------------------------------------------------------------------------


def quantize_activations_fp8(x, group_size: int = 128):
    """BF16 [M, K] -> (fp8-e4m3 [M, K], fp32 scales [M, ceil(K/group)]).

    Per-token per-group scaling with the e4m3 finite amax (448): each group's
    scale = amax/448 (min 1e-4 for denormal safety), the fp8 payload is
    x/scale. Mirrors vLLM's per-token-group quantization.
    """
    import torch

    if x.ndim != 2:
        raise ValueError("expected x [M, K]")
    m, k = x.shape
    if group_size <= 0 or k % group_size:
        # tail groups are not supported by the kernel's power-of-2 tiling
        raise ValueError(f"K ({k}) must be a multiple of group_size ({group_size})")
    xg = x.float().view(m, k // group_size, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = (amax / 448.0).clamp_min(1e-12)
    q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return q, scale.squeeze(-1).contiguous()


# ---------------------------------------------------------------------------
# 2. single-expert blockwise GEMM primitive (kernel + torch oracle)
# ---------------------------------------------------------------------------


def fp8_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out_dtype=None):
    """D[m,n] = sum_kb Sa[m,kb] * Sb[n//128,kb] * (sum_{k in kb} A[m,k]*B[n,k]).

    A: fp8-e4m3 [M, K] row-major with scales [M, K/128] fp32.
    B: fp8-e4m3 [N, K] row-major (weights, transposed-use) with scales
       [N/128, K/128] fp32 (the checkpoint layout).
    Returns BF16 [M, N] (or ``out_dtype``).
    """
    import torch

    if a_fp8.dtype != torch.float8_e4m3fn or b_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("both operands must be fp8-e4m3")
    m, k = a_fp8.shape
    n, k2 = b_fp8.shape
    if k != k2 or k % 128 or n % 128:
        raise ValueError(
            f"shape mismatch A[{m},{k}] B[{n},{k2}] (K,N multiples of 128)"
        )
    if a_scales.shape != (m, k // 128) or b_scales.shape != (n // 128, k // 128):
        raise ValueError(f"scale shapes {a_scales.shape}/{b_scales.shape} do not match")
    out_dtype = out_dtype or torch.bfloat16
    out = torch.empty((m, n), device=a_fp8.device, dtype=out_dtype)

    kernel = _cute_kernel() if a_fp8.is_cuda else None
    if kernel is not None:
        kernel(a_fp8, a_scales, b_fp8, b_scales, out)
        return out
    if a_fp8.is_cuda and os.environ.get(
            "VKERNELS_FP8_GEMM_BACKEND", "auto") in ("auto", "triton"):
        try:
            import triton  # noqa: F401

            return _triton_backend(a_fp8, a_scales, b_fp8, b_scales, out)
        except Exception:
            pass  # no triton (or launch rejected) -> torch oracle
    return _torch_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out)


def _torch_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out):
    """Reference path: block-dequant + per-K-block scaled fp32 accumulation."""
    import torch

    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    kb = k // 128
    acc = torch.zeros((m, n), device=a_fp8.device, dtype=torch.float32)
    for b in range(kb):
        a_blk = (
            a_fp8[:, b * 128 : (b + 1) * 128].to(torch.float32) * a_scales[:, b : b + 1]
        )
        w_blk = b_fp8[:, b * 128 : (b + 1) * 128].to(torch.float32)
        w_blk = w_blk.view(n // 128, 128, 128) * b_scales[:, b : b + 1].unsqueeze(-1)
        acc[:, :] += a_blk @ w_blk.reshape(n, 128).T
    out.copy_(acc.to(out.dtype))
    return out


# ---------------------------------------------------------------------------
# 3. MoE orchestration (sort by expert -> per-active-expert blockwise GEMM)
# ---------------------------------------------------------------------------


def glm_moe_grouped_gemm(
    x,
    gate_up_fp8,
    gate_up_scales,
    down_fp8,
    down_scales,
    topk_index,
    topk_weights,
    swiglu_limit=7.0,
    intermediate_dtype=None,
):
    """Routed expert MLP via blockwise fp8 GEMMs (API-stable; the internal
    per-expert loop is replaced by the fused grouped kernel later).

    x:            BF16 [T, H]
    gate_up_fp8:  [E, 2I, H] fp8, gate_up_scales [E, 2I/128, H/128]
    down_fp8:     [E, H, I] fp8,  down_scales   [E, H/128, I/128]
    topk_index:   int64 [T, K];  topk_weights: [T, K]
    Returns BF16 [T, H] (weighted sum of the K routed experts per token).
    """
    import torch

    e_gate, two_i, h = gate_up_fp8.shape
    e_down, h2, i = down_fp8.shape
    t, k = topk_index.shape
    if two_i != 2 * i or h != h2 or e_gate != e_down:
        raise ValueError("expert stack shapes inconsistent")
    if x.shape != (t, h):
        raise ValueError(
            f"x {tuple(x.shape)} does not match routing [{t}, {k}] and hidden {h}"
        )
    t = x.shape[0]

    flat_expert = topk_index.reshape(-1)  # [T*K]
    token_of = torch.arange(t, device=x.device).repeat_interleave(k)
    sort_idx = torch.argsort(flat_expert, stable=True)
    experts_sorted = flat_expert[sort_idx]
    token_sorted = token_of[sort_idx]
    uniq, starts = torch.unique_consecutive(experts_sorted, return_counts=True)
    offsets = torch.zeros(len(uniq) + 1, device=x.device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(starts, 0)

    y = torch.zeros((t, h), device=x.device, dtype=torch.float32)
    for j in range(len(uniq)):
        expert = int(uniq[j])
        rows = sort_idx[offsets[j] : offsets[j + 1]]
        toks = token_sorted[offsets[j] : offsets[j + 1]]
        xa = x[toks]  # [m_j, H]
        a8, asc = quantize_activations_fp8(xa)
        gu = fp8_blockwise_gemm(a8, asc, gate_up_fp8[expert], gate_up_scales[expert])
        gate, up = gu[:, :i].float(), gu[:, i:].float()
        # SwiGLU with the GLM limit (mirrors floe's _swiglu)
        act = (
            (gate / swiglu_limit).sigmoid() * up * swiglu_limit
            if swiglu_limit
            else torch.nn.functional.silu(gate) * up
        )
        a8d, ascd = quantize_activations_fp8(act.to(torch.bfloat16))
        dn = fp8_blockwise_gemm(
            a8d, ascd, down_fp8[expert], down_scales[expert]
        ).float()
        w = topk_weights.reshape(-1)[rows].float().unsqueeze(-1)
        y.index_add_(0, toks, dn * w)
    return y.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Triton backend (gfx942/MI300A + portable): per-128-K-block scaled fp8 GEMM
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _triton_kernel():
    """Decode-fused blockwise fp8 GEMM: per-K-block tl.dot with the block
    scales applied to the block partial sums (identical math to the torch
    oracle). The raw e4m3FN bytes are loaded as uint8 and cast through
    tl.float8e4nv — Triton's e4m3 decode — which is exact for the FN byte
    layout on both gfx942 and sm_90 backends."""
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _e4m3fn_to_f32(raw):
        # Manual E4M3FN bit-decode: gfx942's native fp8 conversions/convert
        # instructions are FNUZ (different bias/NaN encodings), so both the
        # bitcast-to-fp8-dtype path AND hardware fp8 MMA produce wrong
        # values for checkpoint e4m3fn bytes. Decode: exponent bits +120
        # into the f32 exponent, mantissa <<20, subnormals = m/512, the
        # 0x7F/0xFF encodings are NaN (never present in weights).
        r = raw.to(tl.int32)          # promote: shifts below exceed 8 bits
        e = (r >> 3) & 15
        m = r & 7
        bits = ((e + 120) << 23) | (m << 20)
        v = bits.to(tl.float32, bitcast=True)
        v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, v)
        sign = tl.where((r & 128) != 0, -1.0, 1.0)
        return v * sign

    @triton.jit
    def _blockwise_gemm(
        A, SA, B, SB, D,
        M, N, K,
        sam, sakb, sbn, sbkb,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // 128):
            rk = kb * 128 + tl.arange(0, 128)
            # decode to fp32, round ONCE to bf16 for the tensor-core dot:
            # e4m3fn -> bf16 is EXACT (3 mantissa bits fit in 8), so the
            # dot sees the same values the fp32 oracle does.
            a = _e4m3fn_to_f32(tl.load(
                A + rm[:, None] * K + rk[None, :],
                mask=rm[:, None] < M, other=0,
            )).to(tl.bfloat16)
            b = _e4m3fn_to_f32(tl.load(
                B + rn[:, None] * K + rk[None, :],
                mask=rn[:, None] < N, other=0,
            )).to(tl.bfloat16)
            # row stride * row + col stride * col (a row/col stride swap
            # here reads scale[kb] of the WRONG row — row 0 still looks
            # correct, which is why small probes passed)
            sa = tl.load(SA + rm * sam + kb * sakb, mask=rm < M, other=0.0)
            sb = tl.load(SB + (rn // 128) * sbn + kb * sbkb,
                         mask=rn < N, other=0.0)
            blk = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            acc += blk * (sa[:, None] * sb[None, :])
        tl.store(
            D + rm[:, None] * N + rn[None, :], acc.to(D.dtype.element_ty),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

    return _blockwise_gemm


def _triton_backend(a_fp8, a_scales, b_fp8, b_scales, out):
    import torch

    kfn = _triton_kernel()
    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    a8 = a_fp8.view(torch.uint8)
    b8 = b_fp8.view(torch.uint8)
    import triton

    BM, BN = 64, 128
    grid = (triton.cdiv(n, BN), triton.cdiv(m, BM))
    kfn[grid](
        a8, a_scales, b8, b_scales, out,
        m, n, k,
        a_scales.stride(0), a_scales.stride(1),
        b_scales.stride(0), b_scales.stride(1),
        BM=BM, BN=BN, BK=128,
        num_warps=8, num_stages=3,
    )
    return out


# ---------------------------------------------------------------------------
# CuTe DSL kernel (sm_90) — lazy import; None when the DSL is unavailable.
# ---------------------------------------------------------------------------

_CUTE_KERNEL = None
_CUTE_TRIED = False


def _cute_kernel():
    global _CUTE_KERNEL, _CUTE_TRIED
    if _CUTE_TRIED:
        return _CUTE_KERNEL
    _CUTE_TRIED = True
    if os.environ.get("VKERNELS_DISABLE_CUTE", "") not in ("", "0"):
        return None
    try:
        _CUTE_KERNEL = _build_cute_kernel()
    except Exception as exc:  # noqa: BLE001 - optional accelerator path
        print(
            f"[glm_fp8_blockwise_gemm] CuTe DSL unavailable ({type(exc).__name__}: {exc}); torch reference path active"
        )
        _CUTE_KERNEL = None
    return _CUTE_KERNEL


def _build_cute_kernel():
    """Import + JIT-compile the Hopper blockwise kernel lazily; return a
    (a8, asc, b8, bsc, out) -> None launcher. Compilation happens on first
    GPU call (cute.compile); the pure-torch reference path is the oracle."""
    from vkernels.torch_ops._glm_fp8_sm90_gemm import launch

    def kernel(a8, asc, b8, bsc, out):
        launch(a8, asc, b8, bsc, out)

    return kernel
