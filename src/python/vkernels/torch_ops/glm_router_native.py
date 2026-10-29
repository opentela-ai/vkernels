"""Native-order FP32 router fusion for the measured Beverin inference cells.

The pinned ROCm top-k writes experts above the selection threshold in index
order, then enough threshold ties in index order. Preserve that order and
native sigmoid/adjacent-pair FP32 normalization, including the 1e-20 term.
This operation is separate from the descending-order general fused router.
Torch, Triton and device math load lazily. Eager warmup precedes capture.
"""
from functools import lru_cache
from ._dispatch import OpNotEligible

_ROWS = frozenset((1, 2, 3, 4, 16, 128))

@lru_cache(maxsize=1)
def _kernel():
    global tl, libdevice
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _router_native(X, BIAS, I, W, SCALE, BLOCK: tl.constexpr=512):
        row = tl.program_id(0)
        e = tl.arange(0, BLOCK)
        mask = e < 288
        x = tl.load(X + row * 288 + e, mask, 0).to(tl.float32)
        bias = tl.load(BIAS + e, mask, 0).to(tl.float32)
        scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-x))
        choice = tl.where(mask, scores + bias, float('-inf'))
        used = e < 0
        threshold = 0.0
        for _ in tl.static_range(8):
            candidate = tl.where(used, float('-inf'), choice)
            threshold = tl.max(candidate, 0)
            first = tl.min(tl.where(candidate == threshold, e, BLOCK), 0)
            used = used | (e == first)
        greater = mask & (choice > threshold)
        ties = mask & (choice == threshold)
        count = tl.sum(greater.to(tl.int32), 0)
        positions = tl.where(greater, tl.cumsum(greater.to(tl.int32), 0) - 1, count + tl.cumsum(ties.to(tl.int32), 0) - 1)
        selected = greater | ties & (positions < 8)
        k = tl.arange(0, 8)
        hits = selected[:, None] & (positions[:, None] == k[None, :])
        idx = tl.sum(tl.where(hits, e[:, None], 0), 0)
        w = tl.sum(tl.where(hits, scores[:, None], 0.0), 0)
        a = tl.sum(tl.where(k == 0, w, 0), 0)
        b = tl.sum(tl.where(k == 1, w, 0), 0)
        c = tl.sum(tl.where(k == 2, w, 0), 0)
        d = tl.sum(tl.where(k == 3, w, 0), 0)
        ee = tl.sum(tl.where(k == 4, w, 0), 0)
        f = tl.sum(tl.where(k == 5, w, 0), 0)
        g = tl.sum(tl.where(k == 6, w, 0), 0)
        h = tl.sum(tl.where(k == 7, w, 0), 0)
        total = a + b + (c + d) + (ee + f + (g + h))
        weights = tl.div_rn(w, total + 1e-20) * SCALE
        tl.store(I + row * 8 + k, idx)
        tl.store(W + row * 8 + k, weights)

    return _router_native


def fused_router_native(logits, bias, top_k, scaling, norm_topk_prob=True,
                        *, num_group=1, topk_group=1):
    """Return native-order int64 indices and FP32 weights, without writes.

    The measured contract is FP32 contiguous [M,288] logits, contiguous
    BF16/FP32 bias [288], top8, normalized probabilities, one group, and
    M in 1/2/3/4/16/128 on gfx942/Torch2.9-HIP7/Triton3.4. Valid production
    logits/bias are finite. Other contracts decline to the caller's native
    implementation. No global backend settings or weight caches change.
    """
    import math
    import torch

    if (num_group, topk_group, top_k) != (1, 1, 8) or norm_topk_prob is not True:
        raise OpNotEligible("native router fusion requires normalized top8 and one group")
    if logits.ndim != 2 or logits.shape[1] != 288 or logits.shape[0] not in _ROWS:
        raise OpNotEligible("unmeasured router geometry")
    if logits.dtype != torch.float32 or bias.shape != (288,) or bias.dtype not in (torch.float32, torch.bfloat16):
        raise OpNotEligible("requires FP32 logits and BF16/FP32 bias")
    if not (logits.is_cuda and bias.device == logits.device and logits.is_contiguous() and bias.is_contiguous()):
        raise OpNotEligible("requires contiguous inputs on one HIP device")
    if torch.is_grad_enabled() and (logits.requires_grad or bias.requires_grad):
        raise OpNotEligible("inference only")
    if not isinstance(scaling, (int, float)) or not math.isfinite(scaling):
        raise OpNotEligible("requires a finite scalar routing scale")
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        raise OpNotEligible("unqualified Torch/HIP stack")
    if torch.cuda.get_device_properties(logits.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("qualified on gfx942 only")
    import triton
    if triton.__version__ != "3.4.0":
        raise OpNotEligible("unqualified Triton version")
    rows = logits.shape[0]
    indices = torch.empty((rows, 8), device=logits.device, dtype=torch.int64)
    weights = torch.empty((rows, 8), device=logits.device, dtype=torch.float32)
    with torch.cuda.device(logits.device):
        _kernel()[(rows,)](logits, bias, indices, weights, float(scaling),
                          num_warps=4, enable_fp_fusion=False)
    return indices, weights
