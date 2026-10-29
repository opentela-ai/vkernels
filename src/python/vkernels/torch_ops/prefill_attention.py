"""Qualified gfx942 paged prefill with FP32 softmax and compensated BF16 dots.

Queries use explicit positions; K/V is read through the runner's token-slot
block table. Torch/Triton load lazily. The caller owns valid table indices.
"""
from functools import lru_cache

from ._dispatch import OpNotEligible, require, same_gpu_contiguous
from .glm_gemv import _is_gfx942


@lru_cache(maxsize=1)
def _kernel():
    global tl
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise OpNotEligible("paged prefill requires Triton") from error

    @triton.jit(do_not_specialize=["N", "QB", "BTS", "PS"],
                do_not_specialize_on_alignment=["N", "QB", "BTS", "PS"])
    def paged_prefill(Q, K, V, BT, LENGTHS, POS, OUT,
                      N, HQ: tl.constexpr, HK: tl.constexpr,
                      QB, QT: tl.constexpr, QH: tl.constexpr,
                      KS: tl.constexpr, KH: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
                      BTS, PS,
                      BM: tl.constexpr, BN: tl.constexpr, LARGE_PLANE: tl.constexpr):
        batch, head = tl.program_id(1).to(tl.int64), tl.program_id(2)
        rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
        dims = tl.arange(0, 128)
        kvhead = head // (HQ // HK)
        q = tl.load(Q + batch * QB + rows[:, None] * QT + head * QH + dims[None, :], rows[:, None] < N, 0)
        pos = tl.load(POS + batch * PS + rows, rows < N, -1)
        length = tl.load(LENGTHS + batch)
        limit = tl.minimum(length, tl.max(pos, 0) + 1)
        maximum = tl.full((BM,), float('-inf'), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        acc = tl.zeros((BM, 128), tl.float32)
        for start in range(0, limit, BN):
            columns = start + tl.arange(0, BN)
            slots = tl.load(BT + batch * BTS + columns, columns < length, 0).to(tl.int64)
            if LARGE_PLANE:
                kp = (K.to(tl.uint64) + (slots[None, :] * KS + kvhead * KH + dims[:, None]) * 2).to(tl.pointer_type(tl.bfloat16))
                vp = (V.to(tl.uint64) + (slots[:, None] * VS + kvhead * VH + dims[None, :]) * 2).to(tl.pointer_type(tl.bfloat16))
            else:
                kp = K + slots[None, :] * KS + kvhead * KH + dims[:, None]
                vp = V + slots[:, None] * VS + kvhead * VH + dims[None, :]
            k = tl.load(kp, columns[None, :] < length, 0)
            score = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
            allowed = (columns[None, :] < length) & (columns[None, :] <= pos[:, None]) & (rows[:, None] < N)
            score = tl.where(allowed, score, float('-inf'))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            new_max = tl.where(new_max == float('-inf'), 0., new_max)
            correction = tl.exp2(maximum - new_max)
            prob = tl.exp2(score - new_max[:, None])
            denom = denom * correction + tl.sum(prob, 1)
            acc *= correction[:, None]
            v = tl.load(vp, columns[:, None] < length, 0)
            hi = prob.to(q.dtype)
            lo = (prob - hi.to(tl.float32)).to(q.dtype)
            acc = tl.dot(hi, v, acc)
            acc = tl.dot(lo, v, acc)
            maximum = new_max
        acc = acc / tl.where(denom > 0, denom, 1.)[:, None]
        tl.store(OUT + ((batch * N + rows[:, None]) * HQ + head) * 128 + dims[None, :], acc, rows[:, None] < N)
    return triton, paged_prefill


def prefill_attention(q, kc, vc, block_table, seq_lens, positions):
    """Read BF16 paged K/V for causal prefill through 262144 on gfx942.

    q is [B,N,Hq,128], cache is [slots,Hkv,128], positions [B,N], lengths
    [B], and block_table [B,width] contains valid token slots. Lengths are
    nonnegative and at most width. Tables may include ragged scratch padding
    beyond the 262144-token context, up to twice that width.
    Large KV planes retain 64-bit byte addresses through compiler lowering.
    Explicit positions preserve prefix offsets,
    ragged padding and fully masked rows; the operation never writes cache.
    Unsupported metadata raises OpNotEligible for the caller's SDPA fallback.
    """
    import torch

    require(q.ndim == 4 and kc.ndim == vc.ndim == 3, "expected query and paged KV tensors")
    batch, n, heads, dim = q.shape
    require(kc.shape == vc.shape and kc.shape[0] > 0 and kc.shape[2] == dim == 128,
            "paged prefill requires matching head-128 KV pools")
    require((heads, kc.shape[1]) in ((32, 8), (16, 2), (8, 2)) and 1 <= batch <= 32,
            "no qualified paged prefill geometry")
    require(block_table.ndim == 2 and block_table.shape[0] == batch
            and 1 <= n <= 262144 and n <= block_table.shape[1] <= 524288,
            "paged prefill requires query/table widths in the qualified range")
    require(seq_lens.shape == (batch,) and positions.shape == (batch, n),
            "positions and lengths must match the query batch")
    same_gpu_contiguous(kc, vc, block_table, seq_lens, positions)
    require(q.device == kc.device and q.stride(-1) == 1,
            "query must share the KV GPU with contiguous head dimensions")
    require(_is_gfx942(q.device), "paged prefill requires gfx942")
    require(q.dtype == kc.dtype == vc.dtype == torch.bfloat16,
            "paged prefill requires BF16 query and KV")
    require(block_table.dtype == torch.int32
            and seq_lens.dtype in (torch.int32, torch.int64)
            and positions.dtype in (torch.int32, torch.int64),
            "paged prefill requires integer slot, length and position metadata")
    require(not torch.is_grad_enabled() or not any(t.requires_grad for t in (q, kc, vc)),
            "paged prefill is inference-only")
    if n >= 1024:
        from .prefill_attention_prefix import _kernel as prefix_kernel
        triton, kernel = prefix_kernel()
    else:
        triton, kernel = _kernel()
    # Cold-KV confirmation: Beverin 697478. Keep each shape's measured
    # reduction width fixed; small batches need enough query-tile CTAs.
    bm, bn = 128, 64
    if n <= 128:
        bm, bn = (32, 64) if batch == 1 else ((128, 64) if heads == 32 else (32, 32))
    elif n <= 512 and heads == 16 and batch == 1:
        bm, bn = 32, 32
    with torch.cuda.device(q.device):
        output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        kernel[(triton.cdiv(n, bm), batch, heads)](
            q, kc, vc, block_table, seq_lens, positions, output,
            n, heads, kc.shape[1], *q.stride()[:3], *kc.stride()[:2], *vc.stride()[:2],
            block_table.stride(0), positions.stride(0), bm, bn,
            kc.numel() * kc.element_size() >= (1 << 31), num_warps=4, num_stages=2)
    return output
