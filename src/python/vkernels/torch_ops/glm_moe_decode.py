"""Three-launch FP8-weight/BF16-activation MoE decode.

Fuse gate/up with clamped SwiGLU, and down projection with the route multiply.
The final torch reduction is retained verbatim from Floe's serving oracle:
routes and down outputs round to BF16, each product rounds to BF16, and
``sum(dim=1)`` accumulates those products. No activation quantization, expert
reordering, atomic combine, resident BF16 weight copy, or host synchronization.

The gate kernel comes from glm_moe_grouped; down uses glm_expert_gemv's CUDA
native FP8 decode and compact block-scale load. JIT objects are cached and
must be warmed before capture. CUDA E4M3FN only; other devices/formats decline.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ._dispatch import OpNotEligible

__all__ = ["moe_decode", "moe_decode_eligible"]


def moe_decode_eligible(x, gate_up_w, gate_up_s, down_w, down_s, indices, routes, t_cap=8):
    """Shape-only contract; routing IDs must already address valid experts."""
    if torch.version.hip or any(t.ndim != n for t, n in (
        (x, 2), (gate_up_w, 3), (gate_up_s, 3), (down_w, 3),
        (down_s, 3), (indices, 2), (routes, 2),
    )):
        return False
    if routes.shape != indices.shape or gate_up_w.shape[1] % 2 or gate_up_w.shape[0] == 0:
        return False
    from .glm_moe_grouped import moe_grouped_decode_eligible

    return moe_grouped_decode_eligible(
        x, gate_up_w, gate_up_s, down_w, down_s, indices, routes, t_cap,
    )


@lru_cache(maxsize=1)
def _down_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def down_weighted(
        X, W, S, IDX, ROUTES, Y,
        O: tl.constexpr, I: tl.constexpr, K: tl.constexpr,
        ROWS: tl.constexpr, COLS: tl.constexpr,
    ):
        selected = tl.program_id(0)
        row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, COLS)
        expert = tl.load(IDX + selected).to(tl.int64)
        raw = tl.load(
            W + expert * (O * I) + row[:, None] * I + col[None, :],
            (row[:, None] < O) & (col[None, :] < I), 0,
        )
        value = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scol = tl.arange(0, COLS // 128)
        scale = tl.load(
            S + (expert * (O // 128) + row[:, None] // 128) * (I // 128)
            + scol[None, :],
            (row[:, None] < O) & (scol[None, :] < I // 128), 0,
        )
        weight = (
            (value.reshape(ROWS, COLS // 128, 128) * scale[:, :, None])
            .reshape(ROWS, COLS).to(tl.bfloat16).to(tl.float32)
        )
        x = tl.load(X + selected * I + col, col < I, 0).to(tl.float32)
        dot = tl.sum(weight * x[None, :], axis=1)
        route = tl.load(ROUTES + selected).to(tl.bfloat16).to(tl.float32)
        product = dot.to(tl.bfloat16).to(tl.float32) * route
        tl.store(Y + selected * O + row, product, row < O)

    return down_weighted


def moe_decode(
    x, gate_up_w, gate_up_s, down_w, down_s, indices, routes, swiglu_limit,
    t_cap=8, *, gate_rows=4, gate_warps=4, down_rows=4, down_warps=4,
):
    """Return BF16 [T,H], matching the unfused BF16 route-combine chain.

    Geometry is explicit and static; benchmark overrides do not alter global
    tuning state. Defaults retain the oracle's four-row/four-warp reduction
    layout; other geometries require separate strict device validation.
    """
    if not moe_decode_eligible(
        x, gate_up_w, gate_up_s, down_w, down_s, indices, routes, t_cap,
    ):
        raise OpNotEligible("moe_decode requires contiguous CUDA BF16 inputs, E4M3FN stacks and block-128 scales")
    if gate_rows not in (1, 2, 4, 8) or down_rows not in (4, 8, 16, 32):
        raise OpNotEligible("unsupported MoE row geometry")
    # Changing warp count changes the gate dot's reduction order. An eight-warp
    # probe differs from the four-warp serving oracle even at BF16 stores.
    if gate_warps != 4 or down_warps != 4:
        raise OpNotEligible("MoE decode preserves the four-warp reduction order")
    import triton
    from .glm_moe_grouped import _kernels

    _, gate, _ = _kernels()
    down = _down_kernel()
    t, k = indices.shape
    _, twice_ia, width = gate_up_w.shape
    ia = twice_ia // 2
    hidden = down_w.shape[1]
    act = torch.empty((t, k, ia), device=x.device, dtype=torch.bfloat16)
    weighted = torch.empty((t, k, hidden), device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        gate[(t * k, triton.cdiv(ia, gate_rows))](
            x, gate_up_w.view(torch.uint8), gate_up_s, indices, act,
            float(swiglu_limit), ia, width, k, gate_rows,
            triton.next_power_of_2(width), False,
            num_warps=gate_warps, enable_fp_fusion=False,
        )
        down[(t * k, triton.cdiv(hidden, down_rows))](
            act, down_w.view(torch.uint8), down_s, indices, routes, weighted,
            hidden, ia, k, down_rows, triton.next_power_of_2(ia),
            num_warps=down_warps, enable_fp_fusion=False,
        )
        return weighted.sum(dim=1)
