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

Storage flavour (mirrors ``glm_expert_gemv``'s ``storage`` kwarg, issue #71):
``storage="e4m3fn"`` (default) reads the checkpoint e4m3fn stacks (fp8e4nv
bitcast); ``storage="e4m3fnuz"`` reads the IN-PLACE converted stacks produced
by ``e4m3fn_to_fnuz_inplace`` — fnuz bytes (bias 8, NaN only at 0x80) with
the DOUBLED fp32 block scales, sharing the checkpoint's storage — decoded by
the manual bit-decode on BOTH backends (CUDA has no fnuz dtype), so
``value * scale`` reproduces the original e4m3fn weight bit-exactly.

Numerics contract (bit-exactness with the ladder, on a given device):
weight decode (fp8e4nv bitcast on CUDA / manual bit-decode for fnuz),
per-element ``value * scale``
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

import functools

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
    storage: str = "e4m3fn",
) -> bool:
    """Cheap (no device sync) contract check for :func:`moe_grouped_decode`.

    Mirrors the ladder wrapper's envelope: bf16 activations, block-FP8
    stacked weights in EITHER storage flavour (``storage="e4m3fn"`` — the
    checkpoint bytes — or ``storage="e4m3fnuz"`` — the issue #71 in-place
    rewrite views with their doubled scales; the flavour-agnostic gate
    mirrors ``expert_gemv``, which decodes fnuz bytes on both backends,
    so no CDNA/CUDA gating is needed here), fp32 block scales, int64
    indices, fp32 routing weights, block-128 dims, ``T <= t_cap``,
    all-CUDA contiguous same-device inputs.
    """
    if storage not in ("e4m3fn", "e4m3fnuz"):
        return False
    want = torch.float8_e4m3fnuz if storage == "e4m3fnuz" else torch.float8_e4m3fn
    e, o, i = (gate_up_w.shape[0], gate_up_w.shape[1], gate_up_w.shape[2])
    ia = o // 2
    t, k = top_k_index.shape
    return bool(
        x.is_cuda
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and x.shape == (t, i)
        and gate_up_w.dtype == want
        and down_w.dtype == want
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


@functools.lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _decode_w8(raw, FNUZ: tl.constexpr):
        """fp8 weight byte -> fp32 value, per storage flavour.

        Mirrors ``glm_expert_gemv._expert_gemv`` verbatim. ``FNUZ=0``
        (e4m3fn checkpoint bytes): the ``tl.float8e4nv`` bitcast decodes
        bias-7/IEEE-NaN semantics on both backends (Triton's fp8 casts are
        software — no hardware-fp8 shortcut is taken here, unlike
        ``_expert_gemv_native``). ``FNUZ=1`` (issue #71 in-place fnuz
        rewrite): manual bit-decode — bias 8, max finite 240, NaN ONLY at
        0x80 (no -0), subnormals ``m * 2**-10`` — against the DOUBLED
        scales from ``e4m3fn_to_fnuz_inplace``, so ``value * scale``
        reproduces the original e4m3fn weight exactly (verified exhaustive
        over all 256 bytes; the doubled-scale convention is part of the
        in-place conversion contract).
        """
        if FNUZ:
            ri = raw.to(tl.int32)
            exponent, mantissa = (ri >> 3) & 15, ri & 7
            bits = ((exponent + 119) << 23) | (mantissa << 20)
            value = tl.where(
                exponent == 0,
                mantissa.to(tl.float32) * 0.0009765625,
                bits.to(tl.float32, bitcast=True),
            )
            value = tl.where(ri == 128, float("nan"), value)
            return (value.to(tl.int32, bitcast=True) | ((ri & 128) << 24)).to(
                tl.float32, bitcast=True
            )
        else:
            return raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

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
        FNUZ: tl.constexpr,
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
        weight = (_decode_w8(raw_gate, FNUZ).reshape(ROWS, COLS // 128, 128) * scale[:, :, None]).reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
        gate = tl.sum(weight * x[None, :], axis=1)

        raw_up = tl.load(base + (row[:, None] + IA) * I + col[None, :], inbounds & (col[None, :] < I), 0)
        scale = tl.load(
            S + (expert * (2 * IA // 128) + (row[:, None] + IA) // 128) * (I // 128) + scol[None, :],
            (row[:, None] < IA) & (scol[None, :] < I // 128),
            0,
        )
        weight = (_decode_w8(raw_up, FNUZ).reshape(ROWS, COLS // 128, 128) * scale[:, :, None]).reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
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
        FNUZ: tl.constexpr,
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
            weight = (_decode_w8(raw, FNUZ).reshape(ROWS, COLS // 128, 128) * scale[:, :, None]).reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
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
    storage: str = "e4m3fn",
) -> torch.Tensor:
    """Grouped decode MoE: return ``[T, H]`` bf16 for x ``[T, I]`` bf16.

    ``storage`` selects the stack flavour, mirroring ``expert_gemv``:
    ``"e4m3fn"`` (default) reads the checkpoint e4m3fn bytes;
    ``"e4m3fnuz"`` reads the issue #71 in-place converted stacks (fnuz
    bytes + DOUBLED fp32 block scales — pass the ``_fp8_stack_view``
    fnuz-dtype reinterpret of the same storage). Both decode bit-exactly
    to the same weights, so the ladder oracle is flavour-independent.

    Bit-exact (same device) with the four-launch ladder::

        gu  = expert_gemv(x, gate_up_w, gate_up_s, top_k_index)   # [T,K,2IA]
        act = swiglu_limit(gu[..., :IA], gu[..., IA:], limit)
        out = expert_gemv(act, down_w, down_s, top_k_index)       # [T,K,H]
        y   = moe_weighted_sum(out, top_k_weights)

    Raises ``OpNotEligible`` when the inputs fall outside the contract (the
    caller falls through to the ladder). No host syncs; capture-safe.
    """
    if not moe_grouped_decode_eligible(x, gate_up_w, gate_up_s, down_w, down_s, top_k_index, top_k_weights, t_cap, storage):
        raise OpNotEligible(
            f"moe_grouped_decode contract: bf16 x[T,I], e4m3fn-or-e4m3fnuz "
            f"[E,2IA,I]/[E,H,IA] stacks (storage={storage!r}) with fp32 "
            f"block-128 scales, int64 [T,K] indices, fp32 [T,K] "
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
    # Flavour rides the wrapper (expert_gemv precedent): fnuz STORAGE always
    # takes the manual bit-decode (FNUZ=1) — CUDA has no fnuz dtype to
    # bitcast to, and the doubled-scale convention is part of the in-place
    # conversion contract (issue #71). e4m3fn bytes keep the float8e4nv
    # bitcast on both backends (Triton's fp8 casts are software).
    fnuz = storage == "e4m3fnuz"
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
            fnuz,
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
            fnuz,
            # Default fp-fusion: the accumulate must contract to FMA exactly
            # like _weighted_moe_sum_reduce_kernel (enable_fp_fusion default
            # True) for the w[t,k]*dot term to round identically. The dot
            # itself stays bit-identical — verified by the bit-parity tests.
            num_warps=cfg["warps2"],
        )
    return out
