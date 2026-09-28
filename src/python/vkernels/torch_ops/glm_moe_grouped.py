"""Grouped MoE decode for the GLM-5.3 block-FP8 expert stacks — 2 launches.

The B=1..8 decode regime (``T <= moe_decode_max_tokens``) walks a four-launch
per-layer ladder::

    gu   = expert_gemv(x, w13, s13, idx)          # [T,K,2*IA] bf16
    act  = swiglu_limit(gu[..., :IA], gu[..., IA:], limit)
    out  = expert_gemv(act, w2, s2, idx)          # [T,K,H] bf16
    y    = moe_weighted_sum(out, topk_w)          # [T,H]

(plus a fifth launch when ``fused_moe_combine`` is off). This module folds
the segment into TWO grouped Triton launches that consume the serving
tensors DIRECTLY — the stacked fp8-e4m3fn expert blocks with their
[O/128, I/128] fp32 scales, the int64 ``top_k_index`` and the fp32
``top_k_weights`` — with no moe_align/sort step, no merged-step
dependency, no host syncs, and static per-bucket shapes (CUDA-graph
capture safe).

* Stage 1 (``_moe_gate_up``): grid ``(T*K, IA/ROWS)``; each program pairs
  the gate rows and the matching up rows (offset +IA) of one selected
  expert, dots both against the token's bf16 activation, applies the GLM
  ``swiglu_limit`` clamps + silu IN-KERNEL, and writes ``act[T,K,IA]``.
* Stage 2 (``_moe_down``): grid ``(T, H/ROWS)``; each program loops the
  token's K experts, dots the staged ``act`` against the down-stack rows,
  and accumulates ``w[t,k] * round_bf16(dot)`` in fp32 across K — the
  ``expert_gemv`` bf16 store boundary and the ``moe_weighted_sum``
  fp32-accumulate contract reproduced in registers — storing ``[T,H]`` once.

Numerics contract (bit-exactness with the ladder, on a given device):
weight decode (fp8e4nv bitcast on CUDA), per-element ``value * scale``
rounded to bf16 then fp32, the fp32 ``tl.sum`` dot (order depends only on
the reduction-axis length — inherited from ``glm_expert_gemv``/``glm_expert_gemv_fused``
provenance), the bf16 rounding at every stage boundary, the
``propagate_nan`` clamp semantics of ``elementwise._swiglu_limit``, and the
ascending-k fp32 accumulation of ``_weighted_moe_sum_reduce_kernel`` are
reproduced verbatim; ``enable_fp_fusion=False`` keeps the dots un-FMA'd.
The ladder stays the oracle and the fallback (parity-oracle pattern).

Adopted from floe (``floe/engine/runner/kernels/moe_grouped.py``, the
wave-integration campaign) under the #64/#65 re-export model. Adoption
deltas besides the move: ``OpNotEligible`` comes from this package's
``_dispatch`` directly, and the tile-override env var is
``VK_MOE_GROUPED_CFG`` (the floe-local original read
``FLOE_MOE_GROUPED_CFG``; sweep tooling that set the old name must set
the new one).

Stdlib+torch at import (triton lazily, moe_combine house style — CPU-only
hosts can import the module for the eligibility surface). Inputs are
read-only; inference-only, no autograd backward. Graph capture: the decode
warmup's eager pass compiles both kernels per shape before capture.
"""

from __future__ import annotations

import os

import torch

from ._dispatch import OpNotEligible

__all__ = [
    "moe_grouped_decode",
    "moe_grouped_decode_eligible",
]

# (ROWS, num_warps) per stage; env-overridable JSON {"rows1": .., "warps1": ..,
# "rows2": .., "warps2": ..} for the microbench sweep. Defaults: the ladder's
# proven stage-1 geometry (ROWS=4/warps=4 over the gate rows) and a stage-2
# row-group chosen by the sgs-gpu07 microbench (see RESULTS.md).
_DEF = {"rows1": 4, "warps1": 4, "rows2": 8, "warps2": 4}


def _cfg() -> dict:
    out = dict(_DEF)
    raw = os.environ.get("VK_MOE_GROUPED_CFG")
    if raw:
        import json

        try:
            for k, v in json.loads(raw).items():
                if k in out:
                    out[k] = int(v)
        except (ValueError, TypeError):
            pass
    return out


def moe_grouped_decode_eligible(
    x: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_s: torch.Tensor,
    down_w: torch.Tensor,
    down_s: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    t_cap: int,
) -> bool:
    """Cheap (no device sync) contract check for :func:`moe_grouped_decode`.

    Mirrors the ladder wrapper's envelope: bf16 activations, e4m3fn stacked
    weights (fp32 block scales), int64 indices, fp32 routing weights,
    block-128 dims, ``T <= t_cap``, all-CUDA contiguous same-device inputs.
    """
    e, o, i = (gate_up_w.shape[0], gate_up_w.shape[1], gate_up_w.shape[2])
    ia = o // 2
    t, k = top_k_index.shape
    return bool(
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape == (t, i)
        and gate_up_w.dtype == torch.float8_e4m3fn
        and down_w.dtype == torch.float8_e4m3fn
        and gate_up_s.dtype == torch.float32
        and down_s.dtype == torch.float32
        and top_k_index.dtype == torch.int64
        and top_k_weights.dtype == torch.float32
        and gate_up_s.shape == (e, o // 128, i // 128)
        and down_s.shape == (e, down_w.shape[1] // 128, down_w.shape[2] // 128)
        and down_w.shape == (e, x.shape[1], ia)
        and ia > 0
        and ia % 128 == 0
        and i % 128 == 0
        and down_w.shape[1] % 128 == 0
        and t <= t_cap
        and t > 0
        and k > 0
        and all(v.is_contiguous() for v in (x, gate_up_w, gate_up_s, down_w, down_s, top_k_index, top_k_weights))
        and len({v.device for v in (x, gate_up_w, gate_up_s, down_w, down_s, top_k_index, top_k_weights)}) == 1
    )


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _moe_gate_up(
        X,
        W,
        S,
        IDX,
        Y,
        LIMIT,
        IA: tl.constexpr,
        I: tl.constexpr,
        K: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        """Stage 1: act[t,k,row] = swiglu_limit(gate_dot, up_dot) per pair.

        Program (pair, block) owns act columns [block*ROWS, ...): it loads
        the gate weight rows AND the matching up weight rows (offset +IA)
        of the selected expert, dots both against x[token], and applies the
        GLM clamped-swiglu epilogue with the ladder's rounding boundaries:
        dot -> bf16 (the expert_gemv store), clamp -> bf16, silu -> bf16,
        product stored bf16.
        """
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        expert = tl.load(IDX + selected).to(tl.int64)
        inbounds = row[:, None] < IA  # COLS == I always (I is a pow-2 multiple of 128 up to 4096; masked below anyway)
        base = W + expert * (2 * IA * I)
        x_row = selected // K
        x = tl.load(X + x_row * I + col, col < I, 0).to(tl.float32)
        scol = tl.arange(0, COLS // 128)

        raw_gate = tl.load(base + row[:, None] * I + col[None, :], inbounds & (col[None, :] < I), 0)
        scale = tl.load(
            S + (expert * (2 * IA // 128) + row[:, None] // 128) * (I // 128) + scol[None, :],
            (row[:, None] < IA) & (scol[None, :] < I // 128),
            0,
        )
        weight = (raw_gate.to(tl.float8e4nv, bitcast=True).to(tl.float32).reshape(ROWS, COLS // 128, 128) * scale[:, :, None]).reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
        gate = tl.sum(weight * x[None, :], axis=1)

        raw_up = tl.load(base + (row[:, None] + IA) * I + col[None, :], inbounds & (col[None, :] < I), 0)
        scale = tl.load(
            S + (expert * (2 * IA // 128) + (row[:, None] + IA) // 128) * (I // 128) + scol[None, :],
            (row[:, None] < IA) & (scol[None, :] < I // 128),
            0,
        )
        weight = (raw_up.to(tl.float8e4nv, bitcast=True).to(tl.float32).reshape(ROWS, COLS // 128, 128) * scale[:, :, None]).reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
        up = tl.sum(weight * x[None, :], axis=1)

        # Verbatim elementwise._swiglu_limit epilogue (clamp max=limit gate,
        # clamp +-limit up, silu gate; propagate_nan=ALL) with the ladder's
        # bf16 storage boundaries: gate/up were stored bf16 by expert_gemv,
        # then swiglu_limit re-rounds the clamped values and the silu result.
        gate = tl.minimum(gate.to(tl.bfloat16).to(tl.float32), LIMIT, propagate_nan=tl.PropagateNan.ALL)
        gate = gate.to(tl.bfloat16).to(tl.float32)
        upv = up.to(tl.bfloat16).to(tl.float32)
        upv = (
            tl.minimum(
                tl.maximum(upv, -LIMIT, propagate_nan=tl.PropagateNan.ALL),
                LIMIT,
                propagate_nan=tl.PropagateNan.ALL,
            )
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        act = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + selected * IA + row, act * upv, row < IA)

    @triton.jit
    def _moe_down(
        ACT,
        W,
        S,
        IDX,
        WTS,
        Y,
        IA: tl.constexpr,
        H: tl.constexpr,
        K: tl.constexpr,
        ROWS: tl.constexpr,
        COLS: tl.constexpr,
    ):
        """Stage 2: y[t, row] = sum_k wts[t,k] * round_bf16(down dot).

        Program (token, block) owns H-rows [block*ROWS, ...): for each of
        the token's K experts it dots the staged act row against the down
        weight rows (fp32, bf16-rounded weights), rounds the dot to bf16
        (the expert_gemv [T,K,H] store boundary), and accumulates
        wts[t,k] * value in fp32 in ascending k — the
        _weighted_moe_sum_reduce_kernel contract — rounding once at the
        bf16 store.
        """
        token = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        scol = tl.arange(0, COLS // 128)
        inbounds = row[:, None] < H
        acc = tl.zeros((ROWS,), tl.float32)
        for k in range(K):
            expert = tl.load(IDX + token * K + k).to(tl.int64)
            base = W + expert * (H * IA)
            raw = tl.load(base + row[:, None] * IA + col[None, :], inbounds, 0)
            scale = tl.load(
                S + (expert * (H // 128) + row[:, None] // 128) * (IA // 128) + scol[None, :],
                (row[:, None] < H) & (scol[None, :] < IA // 128),
                0,
            )
            weight = (raw.to(tl.float8e4nv, bitcast=True).to(tl.float32).reshape(ROWS, COLS // 128, 128) * scale[:, :, None]).reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
            a = tl.load(ACT + token * K * IA + k * IA + col, col < IA, 0).to(tl.float32)
            dot = tl.sum(weight * a[None, :], axis=1)
            w = tl.load(WTS + token * K + k)
            acc += w * dot.to(tl.bfloat16).to(tl.float32)
        tl.store(Y + token * H + row, acc.to(tl.bfloat16), row < H)

    return triton, _moe_gate_up, _moe_down


def moe_grouped_decode(
    x: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_s: torch.Tensor,
    down_w: torch.Tensor,
    down_s: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    swiglu_limit: float,
    t_cap: int = 8,
) -> torch.Tensor:
    """Grouped decode MoE: return ``[T, H]`` bf16 for x ``[T, I]`` bf16.

    Bit-exact (same device) with the four-launch ladder::

        gu  = expert_gemv(x, gate_up_w, gate_up_s, top_k_index)   # [T,K,2IA]
        act = swiglu_limit(gu[..., :IA], gu[..., IA:], limit)
        out = expert_gemv(act, down_w, down_s, top_k_index)       # [T,K,H]
        y   = moe_weighted_sum(out, top_k_weights)

    Raises ``OpNotEligible`` when the inputs fall outside the contract (the
    caller falls through to the ladder). No host syncs; capture-safe.
    """
    if not moe_grouped_decode_eligible(x, gate_up_w, gate_up_s, down_w, down_s, top_k_index, top_k_weights, t_cap):
        raise OpNotEligible(
            f"moe_grouped_decode contract: bf16 x[T,I], e4m3fn [E,2IA,I]/[E,H,IA] "
            f"stacks with fp32 block-128 scales, int64 [T,K] indices, fp32 [T,K] "
            f"weights, T<={t_cap} (got x{x.shape} {x.dtype}, w13{tuple(gate_up_w.shape)} "
            f"{gate_up_w.dtype}, w2{tuple(down_w.shape)} {down_w.dtype}, "
            f"idx{tuple(top_k_index.shape)} {top_k_index.dtype})"
        )
    import triton

    t, k = top_k_index.shape
    e, o, i = gate_up_w.shape
    ia = o // 2
    h = down_w.shape[1]
    cfg = _cfg()
    rows1 = cfg["rows1"]
    rows2 = cfg["rows2"]
    act = torch.empty((t, k, ia), device=x.device, dtype=torch.bfloat16)
    out = torch.empty((t, h), device=x.device, dtype=torch.bfloat16)
    _, gate_up, down = _kernels()
    with torch.cuda.device(x.device):
        # COLS = next_pow2 of the reduction axis, the ladder convention (the
        # tl.sum order depends only on COLS); I and IA are powers of two
        # multiples of 128 in the serving shapes so the masks are all-true.
        gate_up[(t * k, triton.cdiv(ia, rows1))](
            x,
            gate_up_w.view(torch.uint8),
            gate_up_s,
            top_k_index,
            act,
            float(swiglu_limit),
            ia,
            i,
            k,
            rows1,
            triton.next_power_of_2(i),
            num_warps=cfg["warps1"],
            enable_fp_fusion=False,
        )
        down[(t, triton.cdiv(h, rows2))](
            act,
            down_w.view(torch.uint8),
            down_s,
            top_k_index,
            top_k_weights,
            out,
            ia,
            h,
            k,
            rows2,
            triton.next_power_of_2(ia),
            # Default fp-fusion: the accumulate must contract to FMA exactly
            # like _weighted_moe_sum_reduce_kernel (enable_fp_fusion default
            # True) for the w[t,k]*dot term to round identically. The dot
            # itself stays bit-identical — verified by the bit-parity tests.
            num_warps=cfg["warps2"],
        )
    return out
