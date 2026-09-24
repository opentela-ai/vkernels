"""Single-token FP32 per-dimension-gated KDA inference kernel (optional).

Normalizes raw q/k (L2 with eps, then q scaled by ``D**-0.5``) and advances
the per-dimension-gated delta-rule state by one decode token, returning
``(output, next_state)`` without mutating any input.

The kernel's *accumulation* is FP32 unconditionally (the delta rule's
contract), but its ABI accepts the serving dtype directly: ``q/k/v/gate/beta``
may be BF16/FP16 and are widened in-kernel — an exact conversion, so it is bit
identical to the caller widening first, minus the cast launches and the fp32
temporaries (5 cast kernels x one KDA layer per decode step). The output
therefore comes back in the vector dtype (one round on store, exactly where
the serving path used to round it back) while ``next_state`` stays FP32.

Adopted from floe's ``engine/runner/kernels/glm5_kda_decode.py`` (vkernels
owns the kernel; floe imports it back through a thin adapter — the #64/#65
thin-adapter model extended to the whole GLM-5 Triton set).

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from ._dispatch import OpNotEligible, same_gpu_contiguous
from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _decode(
        Q,
        K,
        V,
        G,
        BETA,
        STATE,
        OUT,
        NEXT,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BV: tl.constexpr,
    ):
        head = tl.program_id(0)
        block = tl.program_id(1)
        rows = tl.arange(0, D)
        cols = block * BV + tl.arange(0, BV)
        q = tl.load(Q + head * D + rows).to(tl.float32)
        k = tl.load(K + head * D + rows).to(tl.float32)
        # Reference divides by sqrt(sum(x*x) + eps), then scales q.
        q = tl.div_rn(q, tl.sqrt(tl.sum(q * q, 0) + EPS))
        k = tl.div_rn(k, tl.sqrt(tl.sum(k * k, 0) + EPS))
        q = q * (D**-0.5)
        decay = tl.exp(tl.load(G + head * D + rows).to(tl.float32))
        state_offset = head * D * D + rows[:, None] * D + cols[None, :]
        state = tl.load(STATE + state_offset, cols[None, :] < D, 0)
        state = state * decay[:, None]
        memory = tl.sum(state * k[:, None], 0)
        value = tl.load(V + head * D + cols, cols < D, 0).to(tl.float32)
        delta = (value - memory) * tl.load(BETA + head).to(tl.float32)
        state = state + k[:, None] * delta[None, :]
        output = tl.sum(state * q[:, None], 0)
        tl.store(NEXT + state_offset, state, cols[None, :] < D)
        tl.store(OUT + head * D + cols, output, cols < D)

    return _decode


def kda_decode(query, key, value, gate, beta, initial_state, *, out_fp32=False):
    """Normalize raw q/k and return ``(output, next_state)`` without mutation.

    Vectors are contiguous GPU tensors [B,1,H,D] (FP32/BF16/FP16, one dtype),
    beta [B,1,H] in that dtype, state [B,H,D,D] FP32. The output has the vector
    shape and dtype (the widened accumulation is rounded once on store);
    ``next_state`` is FP32. Inference only; no autograd.
    The normalization epsilon is pinned to 1e-6 (kda_decode_reference's
    default); experiment via the reference's ``eps`` argument.

    ``out_fp32=True`` allocates the output in FP32 regardless of the vector
    dtype: the in-kernel widening of BF16/FP16 vectors is exact and the
    accumulation is FP32 unconditionally, so the FP32 output is bit-identical
    to the caller widening the vectors first — it only skips the caller-side
    cast launches. The default (False) keeps the historical ABI where the
    output shares the vector dtype.
    """
    import torch

    if query.ndim != 4 or query.shape[1] != 1:
        raise OpNotEligible("expected vectors [B,1,H,D]")
    batch, _, heads, dim = query.shape
    if dim not in (32, 64, 128):
        raise OpNotEligible("supported head dimensions are 32, 64, 128")
    if any(x.shape != query.shape for x in (key, value, gate)):
        raise OpNotEligible("query/key/value/gate shapes must match")
    if beta.shape != (batch, 1, heads) or initial_state.shape != (
        batch,
        heads,
        dim,
        dim,
    ):
        raise OpNotEligible("incorrect beta or state shape")
    vectors = (query, key, value, gate, beta)
    if any(x.dtype != query.dtype for x in vectors):
        raise OpNotEligible("query/key/value/gate/beta must share one dtype")
    if query.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise OpNotEligible("vectors must be FP32, BF16 or FP16")
    if initial_state.dtype != torch.float32:
        raise OpNotEligible("the recurrent state must be FP32")
    same_gpu_contiguous(query, key, value, gate, beta, initial_state)
    if out_fp32:
        out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    else:
        out = torch.empty_like(query)
    state = torch.empty_like(initial_state)
    if batch * heads:
        import triton  # lazy: validation above needs only torch

        with torch.cuda.device(query.device):
            _kernel()[(batch * heads, triton.cdiv(dim, 32))](
                query,
                key,
                value,
                gate,
                beta,
                initial_state,
                out,
                state,
                dim,
                1e-6,
                32,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out, state


def kda_decode_reference(query, key, value, gate, beta, initial_state, eps=1e-6):
    """Eager FP32 oracle mirroring the kernel's exact math (no mutation)."""
    import torch

    q = query[:, 0]
    k = key[:, 0]
    v = value[:, 0]
    g = gate[:, 0]
    b = beta[:, 0]
    # Reference divides by sqrt(sum(x*x) + eps), then scales q.
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + eps) * query.shape[-1] ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps)
    state = initial_state * g.exp()[..., None]
    memory = (state * k[..., :, None]).sum(-2)
    state = state + k[..., :, None] * ((v - memory) * b[..., None])[..., None, :]
    out = (state * q[..., :, None]).sum(-2)
    return out.unsqueeze(1), state


@lru_cache(maxsize=1)
def _conv_decode_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _conv_decode(S, X, W, OUT, C, KS: tl.constexpr, BC: tl.constexpr):
        b = tl.program_id(0)
        cb = tl.program_id(1)
        c = cb * BC + tl.arange(0, BC)
        cmask = c < C
        # (state[:, :, i] * w[:, i]) products run bf16 (one round per
        # product, exactly the eager broadcast-mul store), the k=4 tail
        # sum accumulates in FP32 (torch's bf16 acc_type) and rounds once
        # on store, then SiLU computes in FP32 on the stored value and
        # rounds again — the same four rounding points as the eager
        # cat -> mul -> sum(-1) -> silu chain it replaces.
        acc = tl.zeros((BC,), tl.float32)
        for i in tl.static_range(KS):
            w = tl.load(W + c * KS + i, cmask, 0).to(tl.float32)
            if i < KS - 1:
                v = tl.load(S + b * C * (KS - 1) + c * (KS - 1) + i, cmask, 0)
            else:
                v = tl.load(X + b * C + c, cmask, 0)
            acc += (v.to(tl.float32) * w).to(OUT.dtype.element_ty).to(tl.float32)
        out = acc.to(OUT.dtype.element_ty).to(tl.float32)
        act = (out / (1.0 + tl.exp(-out))).to(OUT.dtype.element_ty)
        tl.store(OUT + b * C + c, act, mask=cmask)

    return _conv_decode


def kda_conv_decode(state, x, weight):
    """Single-token causal depthwise conv + SiLU in one launch (GLM KDA).

    Replaces the eager four-kernel decode step
    ``silu((cat([state, x], -1) * weight).sum(-1))``:
    ``state`` ``[B, C, K-1]``, ``x`` ``[B, C, 1]``, ``weight`` ``[C, 1, K]``
    (``nn.Conv1d`` layout — the weight carries the kernel size K), output
    ``[B, C]`` in the input dtype. Bit-identical rounding contract vs the
    eager chain: bf16 product rounding per tap, FP32 tail accumulation, one
    round on the conv store, SiLU in FP32 on the stored value with one round
    (see the kernel comment).

    Eligibility: CUDA bf16/fp16 inputs of one dtype on one device,
    contiguous, ``K = weight.shape[-1]`` in (2, 3, 4, 8); raises
    :class:`OpNotEligible` otherwise (fp32 eager callers keep the torch
    path — the decode site only ever runs the serving dtype).
    """
    import torch

    if (
        weight.dim() != 3
        or weight.shape[1] != 1
        or weight.shape[2] not in (2, 3, 4, 8)
    ):
        raise OpNotEligible(
            "weight must be the nn.Conv1d [C, 1, K] layout with K in (2, 3, 4, 8)"
        )
    if state.dim() != 3 or state.shape[-1] != weight.shape[2] - 1:
        raise OpNotEligible("state must be [B, C, K-1] with K = weight.shape[-1]")
    if x.dim() != 3 or x.shape[-1] != 1 or x.shape[:2] != state.shape[:2]:
        raise OpNotEligible("x must be [B, C, 1] matching the state batch/channels")
    if weight.shape[0] != state.shape[1]:
        raise OpNotEligible("weight must have one filter per state channel")
    if (
        state.dtype not in (torch.bfloat16, torch.float16)
        or x.dtype != state.dtype
        or weight.dtype != state.dtype
    ):
        raise OpNotEligible("state/x/weight must be BF16 or FP16 of one dtype")
    same_gpu_contiguous(state, x, weight)
    batch, channels = state.shape[0], state.shape[1]
    out = torch.empty((batch, channels), device=state.device, dtype=state.dtype)
    if batch * channels:
        import triton  # lazy: validation above needs only torch

        with torch.cuda.device(state.device):
            _conv_decode_kernel()[(batch, triton.cdiv(channels, 1024))](
                state,
                x,
                weight,
                out,
                channels,
                weight.shape[2],
                1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out


def kda_conv_decode_reference(state, x, weight):
    """Eager oracle for :func:`kda_conv_decode` — the exact torch chain."""
    import torch
    import torch.nn.functional as F

    window = torch.cat([state, x], dim=-1)
    out = (window * weight.squeeze(1)).sum(dim=-1)
    return F.silu(out)
