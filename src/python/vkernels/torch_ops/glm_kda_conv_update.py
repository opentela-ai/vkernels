"""KDA decode conv + SiLU + state roll — one Triton launch.

The single-token decode step of every GLM-5.3 KDA layer runs

    window = torch.cat([conv_state, x], dim=-1)        # kernel 1 (cat)
    act    = F.silu((window * w).sum(dim=-1))          # kernels 2-4 (mul, sum, silu)
    cache.update_conv_state(layer, x)                  # kernel 5 (cat) + D2D memcpy

— measured on the sgs-gpu07 floe-base0 nsys census at ~10.7 us of kernels
plus the state-roll memcpy per layer, 34 KDA layers per decode step. The
donor (SGLang, and through it every vLLM-family engine) runs this whole
step as ONE ``causal_conv1d_update`` launch — the conv output, the SiLU,
and the in-kernel state shift — a design credited to Tri Dao's
causal-conv1d (sgl-kernels ``aot/csrc/mamba/causal_conv1d.cu``,
``silu_activation=true``; Apache-2.0 / MIT upstream).

This module vendors that design as a Triton kernel: the conv math is the
vkernels ``kda_conv_decode`` contract verbatim (bf16 product rounding per
tap, fp32 tail accumulation, one round on the conv store, SiLU in fp32 on
the stored value, one round — the four rounding points of the eager chain,
so the fused activation is BIT-IDENTICAL to it), and the state roll is
pure storage-dtype moves (``state[..., :k-2] = state[..., 1:]``,
``state[..., k-2] = x`` — exactly the cat the cache performed, no
rounding at all). The kernel CONSUMES AND ROLLS the state buffer in
place: every program owns its channels exclusively and reads the full
window before any state store, so the in-place shift is race-free, and
writing the cache's own backing tensor is what deletes the roll's cat +
memcpy (and its per-step graph-pool allocation) — the same in-place
contract floe's decode-graph bank's state-restore writes already follow
(``cache._conv[li][:B] = ...`` in floe's graphs.py).

Adopted from floe (``floe/engine/runner/kernels/kda_conv_update.py``, the
sgs-gpu07 perf campaign) under the #64/#65 re-export model; the module
sits beside its ``kda_conv_decode`` contract sibling. The only adoption
delta besides the move: a contract miss raises
:class:`~vkernels.torch_ops._dispatch.OpNotEligible` (which subclasses
``TypeError``, the exception the floe-local original raised — every
existing catch keeps working).

Stdlib+torch at import; triton JIT-compiles on first CUDA call (floe's
eager decode warmup precedes graph capture, so a cold compile never
happens under capture).
"""

from __future__ import annotations

import torch

from ._dispatch import OpNotEligible

__all__ = ["kda_conv_update", "kda_conv_update_eligible"]

# The per-tap kernel widths the static_range loop supports (mirrors the
# vkernels kda_conv_decode envelope; the real checkpoint uses K=4).
_SUPPORTED_K = (2, 3, 4, 8)


def kda_conv_update_eligible(
    conv_state: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> bool:
    """Contract check for :func:`kda_conv_update` (cheap, no device sync)."""
    k = weight.shape[-1] if weight.dim() == 3 else 0
    return (
        conv_state.is_cuda
        and x.is_cuda
        and weight.is_cuda
        and weight.dim() == 3
        and weight.shape[1] == 1
        and k in _SUPPORTED_K
        and conv_state.dim() == 3
        and conv_state.shape[-1] == k - 1
        and x.dim() == 3
        and x.shape[-1] == 1
        and x.shape[:2] == conv_state.shape[:2]
        and weight.shape[0] == conv_state.shape[1]
        and conv_state.dtype in (torch.bfloat16, torch.float16)
        and x.dtype == conv_state.dtype
        and weight.dtype == conv_state.dtype
        and conv_state.is_contiguous()
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def _kernel():
    """JIT-scoped kernel definition (import triton lazily, moe_combine style)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _kda_conv_update_kernel(
        STATE,
        X,
        W,
        OUT,
        C,
        KS: tl.constexpr,
        BC: tl.constexpr,
    ):
        b = tl.program_id(0)
        cb = tl.program_id(1)
        c = cb * BC + tl.arange(0, BC)
        cmask = c < C
        # The four rounding points of the eager cat -> mul -> sum(-1) ->
        # silu chain (vkernels kda_conv_decode contract): bf16 product
        # rounding per tap, fp32 tail accumulation, one round on the conv
        # store, SiLU in fp32 on the stored value with one round.
        base = b * C * (KS - 1) + c * (KS - 1)
        acc = tl.zeros((BC,), tl.float32)
        for i in tl.static_range(KS):
            w = tl.load(W + c * KS + i, cmask, 0).to(tl.float32)
            if i < KS - 1:
                v = tl.load(STATE + base + i, cmask, 0)
            else:
                v = tl.load(X + b * C + c, cmask, 0)
            acc += (v.to(tl.float32) * w).to(OUT.dtype.element_ty).to(tl.float32)
        out = acc.to(OUT.dtype.element_ty).to(tl.float32)
        act = (out / (1.0 + tl.exp(-out))).to(OUT.dtype.element_ty)
        tl.store(OUT + b * C + c, act, mask=cmask)
        # State roll, in place: the whole window was read above (the conv
        # loop's loads), so shifting the tail left one tap and appending x
        # cannot clobber a value still needed — and channels are owned
        # exclusively by this program, so no cross-program hazard either.
        # Storage-dtype moves only: bit-identical to the cache's cat roll.
        for j in tl.static_range(KS - 2):
            tl.store(STATE + base + j, tl.load(STATE + base + j + 1, cmask, 0), mask=cmask)
        tl.store(STATE + base + (KS - 2), tl.load(X + b * C + c, cmask, 0), mask=cmask)

    return _kda_conv_update_kernel


def kda_conv_update(
    conv_state: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Single-token depthwise causal conv + SiLU, rolling ``conv_state`` in place.

    ``conv_state`` ``[B, C, K-1]`` is CONSUMED and updated to
    ``[state[..., 1:], x]`` by the same launch that computes the output
    ``silu((cat([state, x], -1) * weight).sum(-1))`` — ``[B, C]`` in the
    input dtype, bit-identical to that eager chain. ``weight`` is the
    ``nn.Conv1d`` ``[C, 1, K]`` layout. Raises :class:`OpNotEligible` when
    the contract in :func:`kda_conv_update_eligible` is not met — callers
    gate on that check and keep the eager chain (the oracle and the
    fallback).
    """
    if not kda_conv_update_eligible(conv_state, x, weight):
        raise OpNotEligible(
            "kda_conv_update contract: CUDA contiguous bf16/fp16 state [B,C,K-1], "
            "x [B,C,1] and weight [C,1,K] of one dtype with K in (2, 3, 4, 8) "
            "(see kda_conv_update_eligible)"
        )
    import triton

    batch, channels = conv_state.shape[0], conv_state.shape[1]
    out = torch.empty((batch, channels), device=conv_state.device, dtype=conv_state.dtype)
    if batch * channels:
        with torch.cuda.device(conv_state.device):
            _kernel()[(batch, triton.cdiv(channels, 1024))](
                conv_state,
                x,
                weight,
                out,
                channels,
                weight.shape[2],
                1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
    else:
        # Degenerate shape: still perform the (empty) state roll contract.
        conv_state.copy_(torch.cat([conv_state[..., 1:], x], dim=-1))
    return out
