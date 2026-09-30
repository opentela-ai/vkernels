"""GLM swiglu activation (clamp + clamp + SiLU + mul) — one Triton launch.

Between the two grouped fp8 GEMMs of the routed-expert MoE
(``_moe_gate_up`` -> activation -> ``_moe_down``), the reference chain is

    gate = gu[:, :I].float().clamp(max=limit)              # cast + clamp
    up   = gu[:, I:].float().clamp(min=-limit, max=limit)  # cast + clamp
    act  = (F.silu(gate) * up).to(bfloat16)                # silu + mul + cast

— five-to-seven elementwise kernels per MoE layer per forward (the casts
materialize fp32 copies of both halves; the moegrp2-nsys census on
sgs-gpu07 shows the resulting eager elementwise rows behind the 10.1% /
6.1% grouped-GEMM pair). SGLang runs this activation as ONE kernel
everywhere its fused MoE does not already absorb it (sgl-kernel
``silu_and_mul_with_thresh``, csrc/elementwise/; Apache-2.0 upstream) —
this module vendors that design.

Adopted from floe (``floe/engine/runner/kernels/silu_mul_clamp.py``, the
sgs-gpu07 perf campaign) under the #64/#65 re-export model. floe injects
the op through the ``swiglu_fn`` seam of
:func:`~vkernels.torch_ops.glm_fp8_blockwise_gemm.glm_moe_grouped_gemm_native`
(the callable owns its eligibility and never raises — an internal miss
falls to the exact eager expression). The only adoption delta besides
the move: a contract miss raises
:class:`~vkernels.torch_ops._dispatch.OpNotEligible` (which subclasses
``TypeError``, the exception the floe-local original raised — every
existing catch keeps working).

Rounding contract: the reference computes in fp32 after exact bf16->fp32
upcasts, with rounding points at sigmoid (exp, add, divide), the silu
multiply, the up-multiply, and one bf16 round on the final cast; this
kernel reproduces that exact sequence (``s = 1 / (1 + exp(-g))`` then
``g * s`` then ``* u`` then one bf16 store), so the fused activation is
BIT-IDENTICAL to the eager chain wherever ``tl.exp`` and torch's CUDA
``expf`` agree bitwise (they did for every tested input on GB200-class
and GB10 hardware; the eager chain stays the oracle and the fallback).

The caller passes ``gu`` as the grouped stage-1 output — rows are
per-slot in sort order, ``[slots, 2*I]`` bf16 contiguous — and receives
``[slots, I]`` bf16 contiguous, ready for the stage-2 activation quant.

Stdlib+torch at import; triton JIT-compiles on first CUDA call (the
eager decode warmup precedes graph capture, so a cold compile never
happens under capture).
"""

from __future__ import annotations

import functools

import torch

from ._dispatch import OpNotEligible

__all__ = ["silu_mul_clamp", "silu_mul_clamp_eligible"]


def silu_mul_clamp_eligible(gate_up: torch.Tensor) -> bool:
    """Contract check for :func:`silu_mul_clamp` (cheap, no device sync)."""
    return (
        gate_up.is_cuda
        and gate_up.dim() == 2
        and gate_up.shape[1] > 0
        and gate_up.shape[1] % 2 == 0
        and gate_up.dtype == torch.bfloat16
        and gate_up.is_contiguous()
    )


@functools.lru_cache(maxsize=1)
def _kernel():
    """JIT-scoped kernel definition (import triton lazily, moe_combine style)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _silu_mul_clamp_kernel(
        GU,
        OUT,
        total,
        I,
        LIMIT,
        stride_gu,
        stride_out,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        row = offs // I
        col = offs % I
        # The eager chain's exact rounding sequence: exact bf16->fp32
        # upcast, exact clamps, sigmoid (exp, add, divide), silu multiply,
        # up multiply — all fp32 — then ONE bf16 round at the store.
        g = tl.load(GU + row * stride_gu + col, mask=mask, other=0).to(tl.float32)
        u = tl.load(GU + row * stride_gu + I + col, mask=mask, other=0).to(tl.float32)
        g = tl.minimum(g, LIMIT)
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        s = 1.0 / (1.0 + tl.exp(-g))
        a = (g * s) * u
        tl.store(OUT + row * stride_out + col, a.to(OUT.dtype.element_ty), mask=mask)

    return _silu_mul_clamp_kernel


def silu_mul_clamp(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    """``silu(clamp(gate, max=limit)) * clamp(up, ±limit)`` — one launch.

    ``gate_up`` is ``[N, 2I]`` bf16 (gate half first, matching the
    chunk(2, dim=-1) split of the eager chain); returns ``[N, I]`` bf16,
    bit-identical to

        gate, up = gate_up.chunk(2, dim=-1)
        (F.silu(gate.float().clamp(max=limit))
         * up.float().clamp(min=-limit, max=limit)).to(torch.bfloat16)

    Raises :class:`OpNotEligible` when the contract in
    :func:`silu_mul_clamp_eligible` is not met — callers gate on that
    check and keep the eager chain (the oracle and the fallback).
    """
    if not silu_mul_clamp_eligible(gate_up):
        raise OpNotEligible(
            "silu_mul_clamp contract: CUDA contiguous bf16 gate_up [N, 2I] "
            "with an even inner dim (see silu_mul_clamp_eligible)"
        )
    import triton

    rows, two_i = gate_up.shape
    inner = two_i // 2
    out = torch.empty((rows, inner), device=gate_up.device, dtype=gate_up.dtype)
    if rows * inner:
        with torch.cuda.device(gate_up.device):
            _kernel()[(triton.cdiv(rows * inner, 4096),)](
                gate_up,
                out,
                rows * inner,
                inner,
                float(limit),
                gate_up.stride(0),
                out.stride(0),
                4096,
                num_warps=4,
            )
    return out
