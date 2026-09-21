"""GLM-5 sigmoid top-k router fusion (inference-only).

Replaces the eager ``Glm53TopkRouter`` glue after the fp32 scoring GEMM:
sigmoid, bias add, group masking, the three ``topk`` calls, the weight
gather, the ``norm_topk_prob`` normalization and the routed scaling — all
in one launch over the fp32 ``[T, E]`` logits.

Supports the shipped configuration (``n_group == 1``, where the group mask
is the identity); other group configurations raise :class:`OpNotEligible`
so the caller's grouped eager path runs — the kernel does not implement the
group mask, so a non-degenerate config would silently misroute. Selection
uses a lowest-index tie-break, so a tie in ``scores + bias`` resolves
deterministically instead of inheriting ``torch.topk``'s unspecified
ordering.

The returned index order is by descending selection value (not
``sorted=False`` order); the weights tensor is gathered in that same
order, so the (index, weight) pairing is exact and only the K-term
normalization sum is reassociated in fp32.

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from functools import lru_cache

from ._dispatch import OpNotEligible


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _router(
        LOGITS,
        BIAS,
        IDX,
        WGT,
        E,
        SCALING,
        K: tl.constexpr,
        NORM: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        e = tl.arange(0, BLOCK_E)
        mask = e < E
        logits = tl.load(LOGITS + row * E + e, mask=mask, other=0.0)
        bias = tl.load(BIAS + e, mask=mask, other=0.0).to(tl.float32)
        scores = tl.sigmoid(logits)
        choice = tl.where(mask, scores + bias, float("-inf"))
        ks = tl.arange(0, BLOCK_K)
        kmask = ks < K
        used = e < 0  # all-false
        weights = tl.zeros([BLOCK_K], dtype=tl.float32)
        total = 0.0
        for i in tl.static_range(K):
            cand = tl.where(used, float("-inf"), choice)
            best = tl.max(cand, axis=0)
            # lowest index attaining the max -> deterministic tie-break
            first = tl.min(tl.where(cand == best, e, BLOCK_E), axis=0)
            used = used | (e == first)
            value = tl.sum(tl.where(e == first, scores, 0.0), axis=0)
            weights = tl.where(ks == i, value, weights)
            total += value
            tl.store(IDX + row * K + i, first)
        if NORM:
            weights = weights / (total + 1.0e-20)
        tl.store(WGT + row * K + ks, weights * SCALING, mask=kmask)

    return _router


def fused_router(logits, bias, top_k, scaling, norm_topk_prob=True,
                 *, num_group=1, topk_group=1):
    """Return ``(indices [T, K] int32, weights [T, K] fp32)``.

    ``logits`` is the fp32 router GEMM output ``[T, E]``, ``bias`` the fp32
    ``e_score_correction_bias`` ``[E]``. Mirrors ``Glm53TopkRouter.forward``
    for ``n_group == 1``: ``weights = scores[indices]``, normalized across
    the selected experts when ``norm_topk_prob``, then scaled.

    Eligibility (all checked here — callers call this unconditionally under
    their policy knob and fall back on :class:`OpNotEligible`): CUDA fp32
    logits, float ``bias``, and the degenerate group config
    ``num_group == topk_group == 1`` — the kernel computes the
    group-mask-identity routing, so any other config must stay on the
    caller's grouped eager path.
    """
    import torch

    if (num_group, topk_group) != (1, 1):
        raise OpNotEligible(
            f"the fused router computes the n_group == 1 (identity group mask) "
            f"routing; got n_group={num_group}, topk_group={topk_group} — "
            f"use the eager grouped path")
    if logits.ndim != 2:
        raise OpNotEligible("logits must be 2-D [tokens, experts]")
    tokens, experts = logits.shape
    if bias.shape != (experts,):
        raise OpNotEligible("expected bias [experts] matching logits")
    if not (1 <= top_k <= experts):
        raise OpNotEligible("top_k must be within [1, experts]")
    if logits.dtype != torch.float32:
        raise OpNotEligible("router logits must be FP32")
    if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise OpNotEligible("bias must be a float dtype")
    if not logits.is_cuda or logits.device != bias.device or not logits.is_contiguous() or not bias.is_contiguous():
        raise OpNotEligible("inputs must be contiguous tensors on the same GPU")
    indices = torch.empty(tokens, top_k, device=logits.device, dtype=torch.int32)
    weights = torch.empty(tokens, top_k, device=logits.device, dtype=torch.float32)
    if tokens:
        import triton

        _kernel()[(tokens,)](
            logits,
            bias,
            indices,
            weights,
            experts,
            float(scaling),
            top_k,
            bool(norm_topk_prob),
            triton.next_power_of_2(experts),
            triton.next_power_of_2(top_k),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return indices, weights


def fused_router_reference(logits, bias, top_k, scaling, norm_topk_prob=True):
    """Eager oracle mirroring ``Glm53TopkRouter.forward`` at ``n_group == 1``.

    Returns ``(indices, weights)`` with indices ordered by descending
    selection value (the kernel's order), so the two are directly
    comparable; ties break to the lowest index.
    """
    import torch

    scores = logits.float().sigmoid()
    choice = scores + bias
    # numpy-free lowest-index tie-break: sort by (value desc, index asc).
    order = torch.argsort(choice, dim=-1, descending=True, stable=True)
    indices = order[:, :top_k]
    weights = scores.gather(1, indices)
    if norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    return indices, weights * scaling
