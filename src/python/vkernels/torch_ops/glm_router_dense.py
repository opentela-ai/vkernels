"""GLM-5 router scoring-GEMV fold (node-cut D): dense GEMV + incumbent router.

Replaces the per-MoE-layer chain ``x.float()`` cast -> cuBLAS fp32 GEMV
(``F.linear(x.float(), w)`` over the ``[E, H]`` router weight) -> the fused
``_router`` kernel with TWO launches that take ``x`` bf16 directly:

1. a GEMV kernel over the ``[E, H]`` router weight that widens ``x``
   in-kernel (EXACTLY the eager cast's math — bf16/fp16 to fp32 is
   lossless) and writes the fp32 ``[T, E]`` logits;
2. the incumbent router kernel (:mod:`vkernels.torch_ops.glm_router`),
   imported and launched verbatim — sigmoid, bias, deterministic
   lowest-index top-k, ``norm_topk_prob`` renormalization, routed
   scaling, and (shared variant) lane 12's append column.

Per layer this deletes the ``x.float()`` cast materialization (a ``[T, H]``
fp32 tensor) and one launch (node-cuts.md step D: −84 nodes across 42 MoE
layers). The routing core is deliberately NOT duplicated here: the
epilogue kernels are the incumbent module's own, so a numerics or
tie-break fix in the incumbent propagates to this fold, and the epilogue
is bit-identical to the incumbent's for identical logits by construction.

Two entry points, sharing one GEMV kernel and one eligibility helper:

* :func:`fused_router_dense` — ``(indices [T, K] int32, weights [T, K]
  fp32)``: the incumbent :func:`vkernels.torch_ops.glm_router.fused_router`
  ABI.
* :func:`fused_router_dense_shared` — ``(indices [T, K+1] int64, weights
  [T, K+1] fp32)``: lane-12's
  :func:`vkernels.torch_ops.glm_router.fused_router_shared` ABI — the
  shared-expert slot ``E`` appended in-kernel after the normalize + scale
  with weight EXACTLY 1.0, int64 emitted directly.

Numerics contract (the load-bearing part — see ``forward.py``
``Glm53TopkRouter.forward``: "rounding logits before sigmoid can change
expert selection near a top-k boundary"):

* **Widening is exact.** bf16/fp16 inputs (x and weight) widen to fp32
  losslessly, so the multiplied VALUES are bit-identical to the eager
  ``x.float()`` / ``w.float()`` casts regardless of where the widening
  happens.
* **Deterministic GEMV reduction order.** Each expert's dot is computed
  wholly inside one program (expert slices are disjoint — no atomics, no
  split-K, no cross-CTA communication): ``acc = 0.0; for h0 in range(0, H,
  BLOCK_H): acc += sum_h(w[e, h0:h0+BH] * x[h0:h0+BH])`` — chunks
  accumulated sequentially in ascending-``h0`` order, each chunk reduced
  by Triton's pairwise tree sum over the ``BLOCK_H`` lane axis, every
  product rounded to fp32 (``enable_fp_fusion=False``, the incumbent's
  rounding discipline, so no FMA contraction reassociates the products).
  The shipped ``(E, H)`` config is pinned in ``_CFG`` so the order is
  stable across launches. This order is NOT cuBLAS ``gemvx``'s (an
  unknowable implementation detail): logits differ by fp32
  reassociation only — measured max ~1.8e-4 ABSOLUTE at the production
  ``[288, 4096]`` shape (cancellation-dominated; relative error is
  meaningless near-zero logits). A top-k boundary tie closer than that
  band CAN flip selection (the docstring-warned ~1e-7-rel ulp class).
  The parity tests pin selection-set equality against the exact eager
  reference on fixed seeds and characterize the near-tie band explicitly.
* **Router math rounding points are the incumbent's.** The epilogue IS
  the incumbent kernel (same launch params, ``num_warps=4``,
  ``enable_fp_fusion=False``): fed bit-identical logits it is
  bit-identical — pinned by all-tie tests, where identical weight rows
  make both the eager GEMM and this kernel's reduction reproduce the tie
  exactly.

Eligibility (all checked here — callers call this unconditionally under
their policy knob and fall back on :class:`OpNotEligible`): CUDA tensors
on one device, contiguous; ``x`` [T, H] in {bf16, fp16, fp32}; weight
[E, H] in {fp32, bf16, fp16} (fp32 is the reference contract; bf16/fp16
widen exactly in-kernel and halve the dominant weight traffic — the
checkpoint dtype); bias [E] float; ``1 <= top_k <= E``; the degenerate
group config ``num_group == topk_group == 1`` — the kernel computes the
group-mask-identity routing, so any other config must stay on the
caller's grouped eager path; and **T <= 8** (the decode regime — batch
decode / speculative-k). The token cap is a measured perf contract, not
an arbitrary limit: at T <= 8 this fold matches or beats the eager chain
device-time (GB10 profiler, production [288, 4096] shape: T=1 12.0 vs
10.1 us at one fewer launch; T=4 16.2 vs 19.7 us; T=8 24.9 vs 25.6 us)
and deletes the ``x.float()`` cast node per layer, while at T >= 16
cuBLAS's register-tiled fp32 GEMM is compute-bound (~75% of GB10's fp32
peak on 48 SMs) and 3-7x faster than any Triton fp32-FMA GEMV variant
measured (per-row tiles have no weight reuse; ieee ``tl.dot`` variants
were not better). Prefill must keep the eager chain. An optional
``return_logits`` emits this kernel's own [T, E] fp32 logits (already
materialized for the epilogue launch — no extra node) for callers that
must keep the ``Glm53TopkRouter`` return signature.

Capture safety: static per-call shapes, plain current-stream launch, no
host syncs, JIT compiled during the eager warmup that precedes capture.
Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from functools import lru_cache

from ._dispatch import OpNotEligible

# (E, H) -> (BLOCK_E, BLOCK_H, num_warps) for the GEMV kernel, measured
# on GB10 (sm_121) at the production router shape (288, 4096); everything
# else takes the fallback heuristic below. The pin is part of the
# numerics contract: it fixes the GEMV reduction order (module docstring).
_CFG = {
    (288, 4096): (16, 512, 8),
}


@lru_cache(maxsize=1)
def _gemv_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _router_dense_gemv(
        X,
        W,
        LOGITS,
        E,
        H,
        TAIL: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # One program per (token row, expert slice). BLOCK_E is a SMALL
        # slice of experts (16 at the production shape): the [BLOCK_E,
        # BLOCK_H] fp32 weight tile stays in registers (a whole-E tile
        # spills and thrashes DRAM), and the grid's expert dimension
        # keeps the GPU busy even at T == 1 (decode), where a
        # one-program-per-row GEMV is latency-bound on a single SM.
        row = tl.program_id(0)
        es = tl.program_id(1)
        e = es * BLOCK_E + tl.arange(0, BLOCK_E)
        emask = e < E
        acc = tl.zeros([BLOCK_E], dtype=tl.float32)
        for h0 in range(0, H, BLOCK_H):
            h = h0 + tl.arange(0, BLOCK_H)
            if TAIL:
                hm = h < H
                x = tl.load(X + row * H + h, mask=hm, other=0.0).to(tl.float32)
                w = tl.load(W + e[:, None] * H + h[None, :],
                            mask=emask[:, None] & hm[None, :], other=0.0).to(tl.float32)
            else:
                x = tl.load(X + row * H + h).to(tl.float32)
                w = tl.load(W + e[:, None] * H + h[None, :],
                            mask=emask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(LOGITS + row * E + e, acc, mask=emask)

    return _router_dense_gemv


def _validate(x, weight, bias, top_k, num_group, topk_group):
    """Shared eligibility surface for both entry points.

    Messages overlap the incumbent op verbatim where the checks coincide,
    so callers see identical rejections; the dense-specific checks (x /
    weight dtype) name the in-kernel-widening contract instead.
    """
    import torch

    if (num_group, topk_group) != (1, 1):
        raise OpNotEligible(
            f"the fused router computes the n_group == 1 (identity group mask) "
            f"routing; got n_group={num_group}, topk_group={topk_group} — "
            f"use the eager grouped path")
    if x.ndim != 2:
        raise OpNotEligible("x must be 2-D [tokens, hidden]")
    if weight.ndim != 2 or weight.shape[1] != x.shape[1]:
        raise OpNotEligible(
            f"expected weight [experts, hidden] matching x{x.shape}, got {tuple(weight.shape)}")
    tokens, hidden = x.shape
    experts = weight.shape[0]
    if bias.shape != (experts,):
        raise OpNotEligible("expected bias [experts] matching weight")
    if not (1 <= top_k <= experts):
        raise OpNotEligible("top_k must be within [1, experts]")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise OpNotEligible("x must be BF16/FP16/FP32 (widened exactly in-kernel)")
    if weight.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise OpNotEligible("router weight must be FP32/BF16/FP16 (widened exactly in-kernel)")
    if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise OpNotEligible("bias must be a float dtype")
    if experts <= 0 or hidden <= 0:
        raise OpNotEligible("empty router GEMV")
    if tokens > 8:
        raise OpNotEligible(
            f"decode-regime fold: tokens={tokens} > 8 — cuBLAS's register-tiled "
            f"fp32 GEMM is compute-bound and 3-7x faster at prefill T on GB10 "
            f"(48 SMs); keep the eager x.float() + F.linear chain there")
    if not x.is_cuda or len({x.device, weight.device, bias.device}) != 1:
        raise OpNotEligible("inputs must be contiguous tensors on the same GPU")
    if not (x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous()):
        raise OpNotEligible("inputs must be contiguous tensors on the same GPU")
    return tokens, experts, hidden


def _launch(x, weight, bias, top_k, scaling, norm_topk_prob, *, num_group, topk_group,
            append, return_logits):
    import torch
    import triton

    # The epilogue kernels are the incumbent module's own — imported, not
    # copied, so the routing core stays single-source (module docstring).
    from .glm_router import _kernel as _router_epilogue
    from .glm_router import _shared_kernel as _router_shared_epilogue

    tokens, experts, hidden = _validate(x, weight, bias, top_k, num_group, topk_group)
    out_k = top_k + (1 if append else 0)
    indices = torch.empty(tokens, out_k, device=x.device,
                          dtype=torch.int64 if append else torch.int32)
    weights = torch.empty(tokens, out_k, device=x.device, dtype=torch.float32)
    logits = torch.empty(tokens, experts, device=x.device, dtype=torch.float32)
    if tokens:
        cfg = _CFG.get((experts, hidden))
        if cfg is not None:
            block_e, block_h, warps = cfg
        else:
            # Generic fallback: a 16-wide expert slice keeps the fp32
            # weight tile register-resident; BLOCK_H = 512 when it divides
            # H (elides the tail mask), else the next power of two.
            block_e = min(16, triton.next_power_of_2(experts))
            block_h = 512 if hidden % 512 == 0 else triton.next_power_of_2(hidden)
            warps = 8 if block_e * block_h >= 4096 else 4
        _gemv_kernel()[(tokens, triton.cdiv(experts, block_e))](
            x,
            weight,
            logits,
            experts,
            hidden,
            hidden % block_h != 0,
            block_e,
            block_h,
            num_warps=warps,
            enable_fp_fusion=False,
        )
        # The incumbent's epilogue launch, verbatim (same params its
        # wrapper uses — including BLOCK_E = next_pow2(E), num_warps=4):
        # fed MY logits it reproduces the incumbent's routing exactly.
        epilogue = _router_shared_epilogue() if append else _router_epilogue()
        epilogue[(tokens,)](
            logits,
            bias,
            indices,
            weights,
            experts,
            float(scaling),
            K=top_k,
            NORM=bool(norm_topk_prob),
            BLOCK_E=triton.next_power_of_2(experts),
            BLOCK_K=triton.next_power_of_2(top_k),
            **({"IDX_STRIDE": indices.stride(0), "WGT_STRIDE": weights.stride(0)} if not append else {}),
            num_warps=4,
            enable_fp_fusion=False,
        )
    if return_logits:
        return indices, weights, logits
    return indices, weights


def fused_router_dense(x, weight, bias, top_k, scaling, norm_topk_prob=True,
                       *, num_group=1, topk_group=1, return_logits=False):
    """Return ``(indices [T, K] int32, weights [T, K] fp32)`` from bf16 x.

    The node-cut-D fold of ``F.linear(x.float(), w)`` +
    :func:`vkernels.torch_ops.glm_router.fused_router`: ``x`` is the layer
    hidden state ``[T, H]`` (bf16 at the production site), ``weight`` the
    router's ``[E, H]`` weight, ``bias`` the fp32
    ``e_score_correction_bias`` ``[E]``. Mirrors ``Glm53TopkRouter.forward``
    at ``n_group == 1``: in-kernel widening + fp32-accum GEMV (deterministic
    reduction order — module docstring), then the incumbent router kernel
    verbatim on the resulting logits. Selection-set parity with the eager
    reference is pinned by the test suite; sub-noise logit ties can
    legitimately flip (the ~1e-7-rel ulp class the HF router comment warns
    about).

    ``return_logits=True`` additionally returns this kernel's own fp32
    ``[T, E]`` logits for callers keeping the ``Glm53TopkRouter``
    ``(logits, weights, indices)`` signature.
    """
    return _launch(x, weight, bias, top_k, scaling, norm_topk_prob,
                   num_group=num_group, topk_group=topk_group,
                   append=False, return_logits=return_logits)


def fused_router_dense_shared(x, weight, bias, top_k, scaling, norm_topk_prob=True,
                              *, num_group=1, topk_group=1, return_logits=False):
    """Return ``(indices [T, K+1] int64, weights [T, K+1] fp32)`` from bf16 x.

    The node-cut-D fold of ``F.linear(x.float(), w)`` +
    :func:`vkernels.torch_ops.glm_router.fused_router_shared` (lane 12's
    fused-shared-expert companion, node-cut A): the routed K columns are
    computed exactly like :func:`fused_router_dense` on the same inputs —
    bit-identically, same GEMV + incumbent epilogue kernel — and the
    shared-expert column is appended IN-KERNEL, reproducing floe's
    ``Glm53TopkRouter._append_shared_slot`` semantics bit for bit: index
    ``E`` (``weight.shape[0]`` — the shared expert stacked as the LAST
    expert row) with weight EXACTLY 1.0, appended AFTER the normalize +
    scale, int64 emitted in-kernel so the caller's
    ``indices.to(torch.int64)`` becomes a no-op.

    Same eligibility surface as :func:`fused_router_dense`.
    """
    return _launch(x, weight, bias, top_k, scaling, norm_topk_prob,
                   num_group=num_group, topk_group=topk_group,
                   append=True, return_logits=return_logits)


def fused_router_dense_reference(x, weight, bias, top_k, scaling, norm_topk_prob=True):
    """Eager oracle: the EXACT production chain this op folds.

    ``F.linear(x.float(), w)`` (cuBLAS fp32 GEMV/GEMM over the widened
    weight — the same logits ``Glm53TopkRouter.forward`` produces) fed to
    the incumbent :func:`vkernels.torch_ops.glm_router.fused_router_reference`.
    Returns ``(logits, indices, weights)`` with indices ordered by
    descending selection value, ties to the lowest index.
    """
    import torch.nn.functional as F

    from .glm_router import fused_router_reference

    logits = F.linear(x.float(), weight.float())
    indices, weights = fused_router_reference(logits, bias, top_k, scaling, norm_topk_prob)
    return logits, indices, weights
