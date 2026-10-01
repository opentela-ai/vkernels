"""Fused KDA small-projection GEMV: b | f_a | g_a in ONE launch (node-cuts E).

The GLM-5.3 KDA decode step runs three tiny bf16 GEMVs against the same
activation ``x`` every layer: the beta projection ``b_proj`` [OB, K] (its
output goes straight through ``sigmoid`` — eager rounds the dot to bf16
FIRST, then applies the sigmoid), the forget-gate input projection ``f_a``
[OFA, K] (bf16 dots, consumed by :func:`~vkernels.torch_ops.glm_forget_gate`
as ``h``), and the o-norm gate head ``g_a`` [OGA, K] (plain bf16 dots).
Per layer per step that is 3 launch-bound GEMVs (+ the sigmoid); 34 layers
of them ≈ 102 graph nodes the node-cuts census prices at ~0.15-0.25 ms.

This op takes the three weights ROW-CONCATENATED at load time
(:func:`kda_small_stack`) and computes all slices in one launch sharing the
single ``x`` read: the GLM-5.3-Flash TP4 stack is [16 + 128 + 128, 4096].

Numerics contract (the node-cuts E class):
- per-slice dots are IDENTICAL to ``dense_gemv`` — fp32 accumulation over
  the same product tile with the same ``tl.sum`` reduction tree, rounded
  once to bf16 at the store;
- the beta slice applies ``sigmoid`` AFTER the bf16 round (the eager
  rounding point pinned by ``beta = torch.sigmoid(linear(x, b_proj))``);
  the transcendental itself may differ from torch by ulps in fp32 before
  the final bf16 round (the standard perf-only class).

Torch and Triton load lazily. Warm up each (device, M) key eagerly before
capture. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible
from functools import lru_cache

_WARMED = set()
_MAX_M = 8  # the decode-GEMV m<=8 policy (fp8_dense_gemv_max_m)

__all__ = [
    "kda_small_gemv",
    "kda_small_gemv_eligible",
    "kda_small_gemv_reference",
    "kda_small_stack",
]


def _validate(x, stack, ob, ofa, oga, *, gpu):
    import torch

    if x.ndim not in (1, 2) or stack.ndim != 2:
        raise OpNotEligible("expected x [M, K] (or [K]) and stacked weights [OB+OFA+OGA, K]")
    k = stack.shape[1]
    total = ob + ofa + oga
    if min(ob, ofa, oga) < 1 or stack.shape[0] != total:
        raise OpNotEligible(
            f"expected row-concat stack [{ob}+{ofa}+{oga}={total}, K], got {tuple(stack.shape)}"
        )
    if x.shape[-1] != k:
        raise OpNotEligible(f"x width {x.shape[-1]} != stack width {k}")
    if k < 1 or k & (k - 1):
        raise OpNotEligible("stack width must be a power of two (tl.arange constraint)")
    m = 1 if x.ndim == 1 else x.shape[0]
    if m < 1 or m > _MAX_M:
        raise OpNotEligible(f"M={m} outside the 1..{_MAX_M} decode-GEMV policy")
    if x.dtype != torch.bfloat16 or stack.dtype != torch.bfloat16:
        raise OpNotEligible("kda_small_gemv requires BF16 inputs")
    if (gpu and not x.is_cuda) or x.device != stack.device:
        raise OpNotEligible("inputs must share a GPU device" if gpu else "inputs must share a device")
    if not x.is_contiguous() or not stack.is_contiguous():
        raise OpNotEligible("kda_small_gemv requires contiguous inputs")
    return m, total, k


def kda_small_stack(b_weight, fa_weight, ga_weight):
    """Row-concatenate the three projection weights at load time (any device).

    ``[OB, K] + [OFA, K] + [OGA, K] -> [OB+OFA+OGA, K]`` — a plain cat the
    loader/checkpoint side can pre-bake; provided for callers that build the
    stack at runtime.
    """
    import torch

    if b_weight.shape[1:] != fa_weight.shape[1:] or b_weight.shape[1:] != ga_weight.shape[1:]:
        raise OpNotEligible(
            f"weights must share width, got {b_weight.shape}, {fa_weight.shape}, {ga_weight.shape}"
        )
    return torch.cat([b_weight, fa_weight, ga_weight], dim=0).contiguous()


def kda_small_gemv_reference(x, stack, ob, ofa, oga, sigmoid_beta=True):
    """Eager oracle: the model's per-slice chain (any device).

    Slices of the returned ``[M, OB+OFA+OGA]`` bf16 tensor: with
    ``sigmoid_beta=True`` (the incumbent model chain) ``beta`` is
    ``sigmoid`` of the bf16-rounded dots and ``h`` (f_a) / ``g_a`` are the
    plain bf16 dots; with ``sigmoid_beta=False`` EVERY slice is the plain
    bf16 dot (the kda_packed_decode contract: raw b_proj dots, sigmoid
    applied in-kernel there). fp32-accumulated linears, one bf16 round
    each — the dense_gemv class.
    """
    import torch

    m, _total, _k = _validate(x, stack, ob, ofa, oga, gpu=False)
    x2 = x.reshape(m, x.shape[-1]).float()
    dots = torch.nn.functional.linear(x2, stack.float()).to(torch.bfloat16)
    if sigmoid_beta:
        beta = torch.sigmoid(dots[:, :ob])
        return torch.cat([beta, dots[:, ob:]], dim=1)
    return dots


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def small_gemv(X, W, Y, M, O: tl.constexpr, I: tl.constexpr, OB: tl.constexpr,
                   MP: tl.constexpr, ROWS: tl.constexpr, BLOCK_I: tl.constexpr,
                   SIG: tl.constexpr):
        """Y[m, row] for m < M; sigmoid epilogue on rows < OB, raw bf16 after.

        W: bf16 [O, K] row-concat stack. The reduction streams BLOCK_I chunks
        with fp32 accumulation — the same product tile + tl.sum tree as
        dense_gemv (bit-identical per dot), vectorized over the MP token
        lanes exactly like dense_gemv_fp8. ``SIG=0`` stores the RAW bf16
        dots everywhere (the kda_packed_decode wiring: its kernel applies
        the sigmoid in-kernel and wants the pre-fold slice).
        """
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        rmask = rows < O
        m = tl.arange(0, MP)
        mmask = m < M
        acc = tl.zeros((ROWS, MP), dtype=tl.float32)
        for k0 in range(0, I, BLOCK_I):
            cols = k0 + tl.arange(0, BLOCK_I)
            cmask = rmask[:, None] & (cols[None, :] < I)
            w = tl.load(W + rows[:, None] * I + cols[None, :], mask=cmask, other=0).to(tl.float32)
            x = tl.load(
                X + m[:, None] * I + cols[None, :],
                mask=mmask[:, None] & (cols[None, :] < I),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w[:, None, :] * x[None, :, :], axis=2)
        # beta rows: round the dot to bf16 FIRST (the eager linear output),
        # then sigmoid in fp32, one final bf16 round.
        if SIG:
            beta = tl.sigmoid(acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
            tl.store(Y + m[None, :] * O + rows[:, None], beta,
                     mask=rmask[:, None] & mmask[None, :] & (rows[:, None] < OB))
            tl.store(Y + m[None, :] * O + rows[:, None], acc.to(tl.bfloat16),
                     mask=rmask[:, None] & mmask[None, :] & (rows[:, None] >= OB))
        else:
            tl.store(Y + m[None, :] * O + rows[:, None], acc.to(tl.bfloat16),
                     mask=rmask[:, None] & mmask[None, :])

    return small_gemv


def _tiles(o: int) -> tuple[int, int]:
    """(ROWS, num_warps) heuristic — the dense_gemv default class (O=272:
    4-row programs; per-shape H100 pins land here once swept)."""
    return (4 if o % 4 == 0 else 1), 4


def kda_small_gemv_eligible(x, stack, ob, ofa, oga) -> bool:
    """Contract check for :func:`kda_small_gemv` (no device sync)."""
    try:
        _validate(x, stack, ob, ofa, oga, gpu=True)
    except OpNotEligible:
        return False
    return True


def kda_small_gemv(x, stack, ob, ofa, oga, m_cap=None, sigmoid_beta=True):
    """Return BF16 ``[M, OB+OFA+OGA]`` — one launch over the row-concat stack.

    Output layout (zero-copy slices for the caller): ``beta`` = ``[:, :ob]``
    (sigmoid already folded, the eager rounding point), ``h`` (f_a) =
    ``[:, ob:ob+ofa]`` — feed to :func:`~vkernels.torch_ops.glm_forget_gate`'s
    h-consuming call site — and ``g_a`` = ``[:, ob+ofa:]``. ``m_cap``
    overrides the M<=8 decode policy (floe passes its ``moe_decode_max_tokens``
    knob). ``sigmoid_beta=False`` returns the RAW bf16 dots for every slice
    (the kda_packed_decode wiring wants pre-fold b_proj dots; its kernel
    applies the sigmoid in-kernel). Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` outside the
    contract so the caller keeps the 3-GEMV eager chain.
    """
    import torch

    m, total, k = _validate(x, stack, ob, ofa, oga, gpu=True)
    cap = int(m_cap) if m_cap is not None else _MAX_M
    if m > cap:
        raise OpNotEligible(f"M={m} above the caller cap {cap}")
    triton = __import__("triton")
    rows, warps = _tiles(total)
    mp = triton.next_power_of_2(m)
    with torch.cuda.device(x.device):
        key = (x.device.index, m)
        if torch.cuda.is_current_stream_capturing() and key not in _WARMED:
            raise RuntimeError("warm up kda_small_gemv eagerly on this device/row-count before capture")
        out = torch.empty(m, total, dtype=torch.bfloat16, device=x.device)
        _kernel()[(triton.cdiv(total, rows),)](
            x.reshape(m, k), stack, out, m, total, k, ob, mp, rows,
            min(k, 4096), bool(sigmoid_beta),
            num_warps=warps,
        )
        _WARMED.add(key)
    return out
