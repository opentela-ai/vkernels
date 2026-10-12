"""Inference causal depthwise convolution with one rounding after SiLU.

Like SGLang's GLM causal-conv prefill (whose weights are stored in FP32),
tap products accumulate in FP32 and SiLU precedes the output's BF16 store.
Inputs and optional preceding raw-input tail are
read-only; the caller retains ownership of recurrent state and its commit.
This intentionally differs from a BF16 Conv1d followed by a separate SiLU.
"""

from __future__ import annotations

import functools

import torch

from ._dispatch import OpNotEligible


def causal_conv1d_silu_eligible(x, weight, state=None) -> bool:
    """Accept strided BF16 [B,C,S], [C,1,K] and optional [B,C,K-1]."""
    if not (
        x.is_cuda
        and x.ndim == 3
        and x.dtype == torch.bfloat16
        and weight.ndim == 3
        and weight.shape[:2] == (x.shape[1], 1)
        and weight.shape[-1] in (2, 3, 4)
        and weight.device == x.device
        and weight.dtype == x.dtype
    ):
        return False
    if state is not None and not (
        state.shape == (*x.shape[:2], weight.shape[-1] - 1) and state.device == x.device and state.dtype == x.dtype
    ):
        return False
    tensors = (x, weight) if state is None else (x, weight, state)
    return not (torch.is_grad_enabled() and any(t.requires_grad for t in tensors))


@functools.lru_cache(maxsize=1)
def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def conv_silu(
        X,
        W,
        STATE,
        OUT,
        C: tl.constexpr,
        S: tl.constexpr,
        K: tl.constexpr,
        XB: tl.constexpr,
        XC: tl.constexpr,
        XS: tl.constexpr,
        WC: tl.constexpr,
        WK: tl.constexpr,
        SB: tl.constexpr,
        SC: tl.constexpr,
        SS: tl.constexpr,
        HAS_STATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(1)
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        channel, token = offsets % C, offsets // C
        valid = offsets < C * S
        acc = tl.zeros((BLOCK,), tl.float32)
        for tap in tl.static_range(K):
            source_token = token + tap - (K - 1)
            value = tl.load(
                X + batch * XB + channel * XC + source_token * XS,
                valid & (source_token >= 0),
                other=0,
            ).to(tl.float32)
            if HAS_STATE:
                previous = tl.load(
                    STATE + batch * SB + channel * SC + (source_token + K - 1) * SS,
                    valid & (source_token < 0),
                    other=0,
                ).to(tl.float32)
                value = tl.where(source_token < 0, previous, value)
            weight = tl.load(W + channel * WC + tap * WK, valid, other=0).to(tl.float32)
            # GLM's donor weights are FP32, even with BF16 activations.
            # Widen checkpoint BF16 values before the multiplication.
            acc = tl.fma(value, weight, acc)
        activated = acc / (1 + tl.exp(-acc))
        # Channel-last output keeps the next projection's input contiguous.
        tl.store(OUT + batch * C * S + offsets, activated, valid)

    return conv_silu


def causal_conv1d_silu(x, weight, state=None) -> torch.Tensor:
    """Return [B,C,S]; never update the supplied preceding input tail."""
    if not causal_conv1d_silu_eligible(x, weight, state):
        raise OpNotEligible("causal_conv1d_silu requires CUDA BF16 inference tensors, K=2/3/4")
    import triton

    batch, channels, sequence = x.shape
    storage = torch.empty((batch, sequence, channels), dtype=x.dtype, device=x.device)
    if batch * channels * sequence:
        strides = (0, 0, 0) if state is None else state.stride()
        with torch.cuda.device(x.device):
            _kernel()[(triton.cdiv(channels * sequence, 256), batch)](
                x,
                weight,
                x if state is None else state,
                storage,
                channels,
                sequence,
                weight.shape[-1],
                *x.stride(),
                weight.stride(0),
                weight.stride(2),
                *strides,
                state is not None,
                256,
                num_warps=4,
            )
    return storage.transpose(1, 2)
