"""DeepSeek-V4.1 sparse attention with an attention sink (post-gather).

The ``_attend`` core: shared-KV (MQA) attention of ``q`` over the concatenated
window ++ selected-compressed keys, with a per-head **sink** logit that joins
the softmax denominator but contributes no value. Selection/causal/window
masking is handed in as a dense ``[B,S,N]`` keep-mask (the arch gathers the
indexer's top-k into it); this op is the masked-softmax-with-sink attention.

CPU ``*_reference`` is the oracle; the Triton path is a flash-style online
softmax that folds the sink into the denominator. Torch/Triton load lazily.
Inference-only, no autograd backward.
"""

from functools import lru_cache

from ._fastpath import fast_path

_NEG = -1.0e30

# Measured GB10 crossover (see meta/benchmarks/bench_v41_triton.py): the
# Triton kernel wins at S <= 2 and loses beyond (S=4: 0.89x, S=64: 0.31x).
_SPARSE_ATTN_MAX_S = 2


def sparse_attention_reference(q, kv, mask, sink, scale):
    """``q`` ``[B,H,S,D]``, ``kv`` ``[B,N,D]`` (shared over heads), ``mask``
    ``[B,S,N]`` (``>0`` keep), ``sink`` ``[H]``, ``scale`` float ->
    ``[B,H,S,D]`` fp32. Mirrors the arch's ``_attend``."""
    import torch

    scores = torch.matmul(q.float(), kv.float().unsqueeze(1).transpose(-1, -2)) * scale  # [B,H,S,N]
    scores = torch.where(mask.unsqueeze(1) > 0, scores, torch.full_like(scores, _NEG))
    b, h, s, n = scores.shape
    sink_col = sink.view(1, h, 1, 1).expand(b, h, s, 1).float()
    probs = torch.softmax(torch.cat([scores, sink_col], dim=-1), dim=-1)
    return torch.matmul(probs[..., :-1], kv.float().unsqueeze(1))  # [B,H,S,D]


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _sparse_attn(Q, KV, MASK, SINK, Out, H: tl.constexpr, S: tl.constexpr, N: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr, BT: tl.constexpr):
        bh = tl.program_id(0)
        s = tl.program_id(1)
        b = bh // H
        h = bh % H
        d = tl.arange(0, D)
        q = tl.load(Q + (bh * S + s) * D + d).to(tl.float32)  # [D]
        # running max starts at the -1e30 sentinel (not -inf: a fully masked
        # row would hit exp(-inf - -inf) = NaN); sc uses the same sentinel.
        m = tl.full((), -1.0e30, tl.float32)
        l = tl.zeros((), tl.float32)
        acc = tl.zeros((D,), tl.float32)
        for t0 in range(0, N, BT):
            t = t0 + tl.arange(0, BT)
            tmask = t < N
            kv = tl.load(KV + (b * N + t[:, None]) * D + d[None, :], tmask[:, None], 0.0).to(tl.float32)  # [BT,D]
            sc = tl.sum(q[None, :] * kv, axis=1) * SCALE  # [BT]
            keep = tl.load(MASK + (b * S + s) * N + t, tmask, 0.0) > 0
            sc = tl.where(keep & tmask, sc, -1.0e30)
            m_new = tl.maximum(m, tl.max(sc, axis=0))
            p = tl.exp(sc - m_new)
            corr = tl.exp(m - m_new)
            l = l * corr + tl.sum(p, axis=0)
            acc = acc * corr + tl.sum(p[:, None] * kv, axis=0)  # [D]
            m = m_new
        sink = tl.load(SINK + h).to(tl.float32)  # raw logit (unscaled)
        m_final = tl.maximum(m, sink)
        corr = tl.exp(m - m_final)
        l = l * corr + tl.exp(sink - m_final)  # sink joins the denominator only
        acc = acc * corr
        tl.store(Out + (bh * S + s) * D + d, acc / l)

    return _sparse_attn


def sparse_attention(q, kv, mask, sink, scale):
    """Device sparse attention with sink. Shape-gated backend on the measured
    GB10 crossover: the one-query-per-program Triton kernel wins for tiny
    query lengths (decode ``S <= 2``; S=1 1.73x) and loses to the batched
    cuBLAS reference beyond (S=64: reference 3.2x faster — the kernel
    re-reads the whole KV block per query position and its [BT,D] fp32 tile
    register-spills at D=512). ``VKERNELS_V41_SPARSE_ATTN_BACKEND`` forces
    ``triton``/``reference``; default ``auto``. Falls back to the reference
    off-GPU / without Triton (or sub-tile ``D``)."""
    import os

    import torch

    if q.ndim != 4 or kv.ndim != 3 or mask.ndim != 3 or sink.ndim != 1:
        raise ValueError("expected q[B,H,S,D], kv[B,N,D], mask[B,S,N], sink[H]")
    b, h, s, d = q.shape
    bk, n, dk = kv.shape
    if bk != b or dk != d or sink.shape[0] != h or mask.shape != (b, s, n):
        raise ValueError("q/kv/mask/sink shapes are inconsistent")
    if not fast_path(q, kv, mask, sink) or (d & (d - 1)) or d < 16 or n == 0:
        return sparse_attention_reference(q, kv, mask, sink, scale)
    backend = os.environ.get("VKERNELS_V41_SPARSE_ATTN_BACKEND", "auto")
    if backend == "reference" or (backend == "auto" and s > _SPARSE_ATTN_MAX_S):
        return sparse_attention_reference(q, kv, mask, sink, scale)

    qf = q.reshape(b * h, s, d).contiguous()
    kvf = kv.contiguous()
    mf = mask.float().contiguous()
    sf = sink.float().contiguous()
    out = torch.empty((b * h, s, d), device=q.device, dtype=torch.float32)
    BT = 32
    with torch.cuda.device(q.device):
        _kernel()[(b * h, s)](qf, kvf, mf, sf, out, h, s, n, d, float(scale), BT)
    return out.reshape(b, h, s, d)
