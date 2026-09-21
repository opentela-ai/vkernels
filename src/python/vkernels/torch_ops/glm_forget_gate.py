"""Fused GLM-5.3 KDA forget gate: second projection + gate math in one launch.

The eager chain per decode step is: ``h = x @ f_a^T`` (GEMV), ``fg = h @ f_b^T``
(GEMV), then ``g = fg + dt_bias``, ``decay = exp(A_log)`` per head,
``out = lower_bound * sigmoid(decay * g)`` (or the softplus branch), rounded
once to the serving dtype. That is ~8 small launches per layer per step on
tiny tensors. This op keeps the first GEMV on ``dense_gemv`` (its numerics are
the adopted class) and fuses everything after it into one Triton kernel.

Numerics contract: the fused kernel accumulates the second projection in FP32
and re-rounds it through BF16, reproducing dense_gemv's output rounding before
the gate math; the gate math is FP32 elementwise in the same order as eager.
vs the eager chain the only rounding-class differences are the GEMV reduction
split and libdevice exp/sigmoid ulps — the same class the dense_gemv A/B
passed the shadow gate with.

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def fgate(H, WB, DT, ALOG, OUT, LB, HAS_LB: tl.constexpr,
              HD: tl.constexpr, BLOCK_O: tl.constexpr):
        offs = tl.program_id(0) * BLOCK_O + tl.arange(0, BLOCK_O)
        h = tl.load(H + tl.arange(0, HD)).to(tl.float32)
        wb = tl.load(WB + offs[:, None] * HD + tl.arange(0, HD)[None, :]).to(tl.float32)
        fg = tl.sum(wb * h[None, :], 1)
        # dense_gemv rounds its output to the serving dtype before the gate
        # math consumes it widened — reproduce that round exactly.
        g = fg.to(tl.bfloat16).to(tl.float32) + tl.load(DT + offs).to(tl.float32)
        decay = tl.exp(tl.load(ALOG + offs // HD).to(tl.float32))
        if HAS_LB:
            out = LB * tl.sigmoid(decay * g)
        else:
            gs = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(g)))
            out = -decay * gs
        tl.store(OUT + offs, out.to(tl.bfloat16))

    return fgate


def forget_gate(x, fa_weight, fb_weight, dt_bias, a_log, lower_bound):
    """Return BF16 [1, S, H, D] log-gates for x [1, S, I].

    ``lower_bound=None`` selects the softplus branch (mirrors the eager
    ``safe_gate_lower_bound is None`` path). Anything outside the fused
    contract raises ``OpNotEligible`` so the caller falls back to eager.
    """
    import torch

    if x.ndim != 3 or x.shape[0] != 1:
        raise OpNotEligible("forget gate fusion supports one sequence [1, S, I]")
    if x.shape[1] != 1:
        raise OpNotEligible("forget gate fusion is decode-only (S == 1); dense_gemv is M == 1")
    if x.dtype != torch.bfloat16 or any(
        w.dtype != torch.bfloat16 for w in (fa_weight, fb_weight)
    ):
        raise OpNotEligible("forget gate fusion requires BF16 x and weights")
    if not (x.is_cuda and x.device == fa_weight.device == fb_weight.device
            and x.device == dt_bias.device == a_log.device):
        raise OpNotEligible("forget gate inputs must share one GPU device")
    if not all(t.is_contiguous() for t in (x, fa_weight, fb_weight, dt_bias, a_log)):
        raise OpNotEligible("forget gate inputs must be contiguous")
    head_dim, hidden = fa_weight.shape
    qkv_dim = fb_weight.shape[0]
    heads = a_log.shape[0]
    if head_dim * heads != qkv_dim or qkv_dim != dt_bias.shape[0] or hidden != x.shape[-1]:
        raise OpNotEligible("forget gate shape mismatch (fa [HD,I], fb [QKV,HD], A_log [H])")
    if qkv_dim % 64:
        raise OpNotEligible("QKV dim must be a multiple of the 64-wide block")

    from vkernels.torch_ops.glm_gemv import dense_gemv

    with torch.cuda.device(x.device):
        h = dense_gemv(x.view(1, hidden), fa_weight)  # [1, HD] BF16
        out = torch.empty(1, qkv_dim, dtype=torch.bfloat16, device=x.device)
        kern = _kernel()
        kern[(qkv_dim // 64,)](
            h, fb_weight, dt_bias, a_log, out,
            1.0 if lower_bound is None else float(lower_bound),
            lower_bound is not None, head_dim, 64,
            num_warps=4, enable_fp_fusion=False,
        )
    return out.view(1, x.shape[1], heads, head_dim)


def forget_gate_reference(x, fa_weight, fb_weight, dt_bias, a_log, lower_bound):
    """Eager FP32-math reference (the model's chain), CPU/GPU agnostic."""
    import torch

    h = torch.nn.functional.linear(x.float(), fa_weight.float()).to(torch.bfloat16)
    fg = torch.nn.functional.linear(h.float(), fb_weight.float()).to(torch.bfloat16)
    heads = a_log.shape[0]
    head_dim = fb_weight.shape[0] // heads
    g = (fg.float() + dt_bias.float().view(1, 1, -1)).view(1, -1, heads, head_dim)
    decay = torch.exp(a_log.float()).view(1, 1, heads, 1)
    if lower_bound is not None:
        out = lower_bound * torch.sigmoid(decay * g)
    else:
        gs = torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))
        out = -decay * gs
    return out.to(x.dtype)
