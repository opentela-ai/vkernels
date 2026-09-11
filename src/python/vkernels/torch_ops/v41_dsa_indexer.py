"""DeepSeek-V4.1 DSA (Lightning) indexer scores.

The hot part of the sparse-attention indexer: for each query position the
per-entry logit ``index_score[b,s,t] = sum_h w[b,s,h] * relu(q[b,s,h,:] .
k[b,t,:])`` over the compressed index keys. The RoPE, causal/candidate
masking, and the stable top-k selection stay in the model (they carry V4.1's
tie-break/candidate semantics); this kernel is the O(B*S*H*T*D) reduction.

CPU ``*_reference`` is the oracle (the exact einsum the arch uses); the Triton
path fuses the head reduction so the ``[B,S,H,T]`` intermediate never lands.
Torch/Triton load lazily. Inference-only, no autograd backward.
"""

from functools import lru_cache


def indexer_scores_reference(q, index_k, weights):
    """``q`` ``[B,S,H,D]``, ``index_k`` ``[B,T,D]``, ``weights`` ``[B,S,H]``
    -> ``index_score`` ``[B,S,T]`` (fp32). Matches the arch's
    ``einsum('bshd,btd->bsht').relu()`` then head-weighted sum."""
    import torch

    scores = torch.einsum("bshd,btd->bsht", q.float(), index_k.float()).relu()
    return (scores * weights.float().unsqueeze(-1)).sum(dim=2)


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _dsa_scores(Q, K, W, Y, S: tl.constexpr, H: tl.constexpr, D: tl.constexpr, T: tl.constexpr, BT: tl.constexpr):
        bs = tl.program_id(0)
        b = bs // S
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        d = tl.arange(0, D)
        h = tl.arange(0, H)
        tmask = t < T
        q = tl.load(Q + bs * (H * D) + h[:, None] * D + d[None, :]).to(tl.float32)  # [H,D]
        w = tl.load(W + bs * H + h).to(tl.float32)  # [H]
        k = tl.load(K + b * (T * D) + t[:, None] * D + d[None, :], tmask[:, None], 0.0).to(tl.float32)  # [BT,D]
        sc = tl.dot(q, tl.trans(k), input_precision="ieee")  # [H,BT]=q[H,D]@k.T[D,BT], full fp32
        sc = tl.maximum(sc, 0.0) * w[:, None]
        y = tl.sum(sc, axis=0)  # [BT]
        tl.store(Y + bs * T + t, y, tmask)

    return _dsa_scores


def indexer_scores(q, index_k, weights):
    """Device DSA indexer scores; falls back to the reference off-GPU /
    without Triton (or for shapes below ``tl.dot``'s 16-min tiles)."""
    import torch

    if q.ndim != 4 or index_k.ndim != 3 or weights.ndim != 3:
        raise ValueError("expected q[B,S,H,D], index_k[B,T,D], weights[B,S,H]")
    b, s, h, d = q.shape
    bk, t, dk = index_k.shape
    if bk != b or dk != d or weights.shape != (b, s, h):
        raise ValueError("q/index_k/weights shapes are inconsistent")
    on_gpu = q.is_cuda and index_k.is_cuda and weights.is_cuda
    try:
        import triton  # noqa: F401
    except Exception:
        on_gpu = False
    # tl.dot needs >=16 in each tiled dim; small (tiny-config) shapes use the
    # reference, which is exact and cheap there.
    if not on_gpu or min(h, d) < 16 or t == 0:
        return indexer_scores_reference(q, index_k, weights)

    qf = q.reshape(b * s, h, d).contiguous()
    wf = weights.reshape(b * s, h).contiguous()
    kf = index_k.contiguous()
    out = torch.empty((b * s, t), device=q.device, dtype=torch.float32)
    import triton

    BT = 64
    with torch.cuda.device(q.device):
        _kernel()[(b * s, triton.cdiv(t, BT))](qf, kf, wf, out, s, h, d, t, BT)
    return out.reshape(b, s, t)
