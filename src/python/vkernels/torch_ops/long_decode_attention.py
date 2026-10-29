"""gfx942 grouped split-KV decode with FP32 softmax and compensated dots.

Historical slots are read-only. One split owns each fresh KV write, while
all readers substitute fresh registers at the final logical position.
"""

from functools import lru_cache
from ._dispatch import OpNotEligible, require, same_gpu_contiguous
from .glm_gemv import _is_gfx942


@lru_cache(maxsize=1)
def _kernels():
    global tl
    try:
        import triton
        import triton.language as tl
    except ImportError as error:
        raise OpNotEligible("long decode requires Triton") from error

    @triton.jit
    def grouped_split(
        Q,
        K,
        V,
        BT,
        LENGTHS,
        KN,
        VN,
        A,
        M,
        L,
        QB: tl.constexpr,
        QH: tl.constexpr,
        KS: tl.constexpr,
        KH: tl.constexpr,
        VS: tl.constexpr,
        VH: tl.constexpr,
        BTS: tl.constexpr,
        KNB: tl.constexpr,
        KNH: tl.constexpr,
        VNB: tl.constexpr,
        VNH: tl.constexpr,
        HQ: tl.constexpr,
        G: tl.constexpr,
        S: tl.constexpr,
        PART: tl.constexpr,
        BN: tl.constexpr,
        HAS_NEW: tl.constexpr,
        SCRATCH: tl.constexpr,
        LARGE_PLANE: tl.constexpr,
    ):
        batch, kvhead, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        rows = tl.arange(0, 16)
        dims = tl.arange(0, 128)
        head = kvhead * G + rows
        query = tl.load(Q + batch * QB + head[:, None] * QH + dims[None, :], rows[:, None] < G, 0)
        seq = tl.load(LENGTHS + batch)
        fresh = 0
        if HAS_NEW:
            fresh = tl.load(BT + batch * BTS + seq - 1, seq > 0, 0).to(tl.int64)
            if split == 0:
                knew = tl.load(KN + batch * KNB + kvhead * KNH + dims)
                vnew = tl.load(VN + batch * VNB + kvhead * VNH + dims)
                if LARGE_PLANE:
                    fresh_kp = (K.to(tl.uint64) + (fresh * KS + kvhead * KH + dims) * 2).to(tl.pointer_type(tl.bfloat16))
                    fresh_vp = (V.to(tl.uint64) + (fresh * VS + kvhead * VH + dims) * 2).to(tl.pointer_type(tl.bfloat16))
                else:
                    fresh_kp = K + fresh * KS + kvhead * KH + dims
                    fresh_vp = V + fresh * VS + kvhead * VH + dims
                tl.store(fresh_kp, knew, (seq > 0) & (fresh != SCRATCH))
                tl.store(fresh_vp, vnew, (seq > 0) & (fresh != SCRATCH))
        maximum = tl.full((16,), float("-inf"), tl.float32)
        denom = tl.zeros((16,), tl.float32)
        acc = tl.zeros((16, 128), tl.float32)
        begin = split * PART
        end = tl.minimum(begin + PART, seq)
        for offset in range(begin, end, BN):
            cols = offset + tl.arange(0, BN)
            slots = tl.load(BT + batch * BTS + cols, cols < end, 0).to(tl.int64)
            if LARGE_PLANE:
                # Integer byte addresses keep the pinned compiler from
                # narrowing broadcast pointer offsets during buffer lowering.
                kp = (K.to(tl.uint64) + (slots[None, :] * KS + kvhead * KH + dims[:, None]) * 2).to(tl.pointer_type(tl.bfloat16))
                vp = (V.to(tl.uint64) + (slots[:, None] * VS + kvhead * VH + dims[None, :]) * 2).to(tl.pointer_type(tl.bfloat16))
            else:
                kp = K + slots[None, :] * KS + kvhead * KH + dims[:, None]
                vp = V + slots[:, None] * VS + kvhead * VH + dims[None, :]
            key = tl.load(kp, cols[None, :] < end, 0)
            value = tl.load(vp, cols[:, None] < end, 0)
            if HAS_NEW:
                knew = tl.load(KN + batch * KNB + kvhead * KNH + dims)
                vnew = tl.load(VN + batch * VNB + kvhead * VNH + dims)
                key = tl.where((cols == seq - 1)[None, :], knew[:, None], key)
                value = tl.where((cols == seq - 1)[:, None], vnew[None, :], value)
            score = tl.dot(query, key) * (0.08838834764831845 * 1.4426950408889634)
            score = tl.where(cols[None, :] < end, score, float("-inf"))
            new_max = tl.maximum(maximum, tl.max(score, 1))
            correction = tl.exp2(maximum - new_max)
            probability = tl.exp2(score - new_max[:, None])
            denom = denom * correction + tl.sum(probability, 1)
            acc *= correction[:, None]
            hi = probability.to(query.dtype)
            lo = (probability - hi.to(tl.float32)).to(query.dtype)
            acc = tl.dot(hi, value, acc)
            acc = tl.dot(lo, value, acc)
            maximum = new_max
        base = (batch * HQ + head) * S + split
        tl.store(M + base, maximum, rows < G)
        tl.store(L + base, denom, rows < G)
        tl.store(A + base[:, None] * 128 + dims[None, :], acc, rows[:, None] < G)

    @triton.jit
    def merge(A, M, L, Y, HQ: tl.constexpr, S: tl.constexpr):
        batch, head = tl.program_id(0), tl.program_id(1)
        splits = tl.arange(0, S)
        dims = tl.arange(0, 128)
        base = (batch * HQ + head) * S + splits
        maximum = tl.load(M + base)
        denominator = tl.load(L + base)
        overall = tl.max(maximum, 0)
        overall = tl.where(overall == float("-inf"), 0.0, overall)
        weights = tl.exp2(maximum - overall)
        total = tl.sum(denominator * weights, 0)
        partial = tl.load(A + base[:, None] * 128 + dims[None, :])
        output = tl.sum(partial * weights[:, None], 0) / tl.where(total > 0, total, 1.0)
        tl.store(Y + (batch * HQ + head) * 128 + dims, output)

    return triton, grouped_split, merge


def decode_attention_long(q, k, v, table, lengths, *, max_len_hint, k_new=None, v_new=None, scratch_slot=-1):
    """BF16 [B,Hq,128] attention through a token-slot table of width 4097–262144.

    B=1..32, (Hq,Hkv)=(32,8),(16,2),(8,2). Lengths are nonnegative
    and bounded by max_len_hint == table width. The caller owns valid slots.
    Fresh K/V are optional, paired [B,Hkv,128] tensors; scratch is never written.
    Large K/V planes retain 64-bit byte addresses through compiler lowering.
    """
    import torch

    require(q.ndim == k.ndim == v.ndim == 3, "expected query and KV pool tensors")
    batch, heads, dim = q.shape
    require(
        k.shape == v.shape and k.shape[0] > 0 and dim == k.shape[2] == 128,
        "long decode requires matching head-128 KV pools",
    )
    require(1 <= batch <= 32 and (heads, k.shape[1]) in ((32, 8), (16, 2), (8, 2)), "no qualified long decode geometry")
    require(
        table.ndim == 2
        and table.shape[0] == batch
        and lengths.shape == (batch,)
        and 4096 < max_len_hint == table.shape[1] <= 262144,
        "unqualified long decode width",
    )
    same_gpu_contiguous(k, v, table, lengths)
    require(q.device == k.device and q.stride(-1) == 1, "query must share GPU and contiguous dimensions")
    require(_is_gfx942(q.device), "long decode requires gfx942")
    require(q.dtype == k.dtype == v.dtype == torch.bfloat16, "long decode requires BF16")
    require(table.dtype == torch.int32 and lengths.dtype in (torch.int32, torch.int64), "invalid integer metadata")
    require((k_new is None) == (v_new is None), "fresh K/V must be paired")
    if k_new is not None:
        for fresh in (k_new, v_new):
            require(
                fresh.shape == (batch, k.shape[1], 128)
                and fresh.device == q.device
                and fresh.dtype == q.dtype
                and fresh.stride(-1) == 1,
                "invalid fresh KV tensor",
            )
    operands = (q, k, v) if k_new is None else (q, k, v, k_new, v_new)
    require(not torch.is_grad_enabled() or not any(t.requires_grad for t in operands), "long decode is inference-only")
    triton, grouped_split, merge = _kernels()
    # Beverin 697623: fixed selection, no serve-time autotuning.
    splits, bn, warps, stages = 128, (128 if heads == 16 and batch == 1 else 64), 4, 1
    hint = max_len_hint
    part = triton.cdiv(hint, splits * bn) * bn
    with torch.cuda.device(q.device):
        a = torch.empty(batch, heads, splits, 128, device=q.device, dtype=torch.float32)
        m = torch.empty(batch, heads, splits, device=q.device, dtype=torch.float32)
        l = torch.empty_like(m)
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
        has_new = k_new is not None
        grouped_split[(batch, k.shape[1], splits)](
            q,
            k,
            v,
            table,
            lengths,
            k_new if has_new else q,
            v_new if has_new else q,
            a,
            m,
            l,
            *q.stride()[:2],
            *k.stride()[:2],
            *v.stride()[:2],
            table.stride(0),
            k_new.stride(0) if has_new else 0,
            k_new.stride(1) if has_new else 0,
            v_new.stride(0) if has_new else 0,
            v_new.stride(1) if has_new else 0,
            heads,
            heads // k.shape[1],
            splits,
            part,
            bn,
            has_new,
            scratch_slot,
            k.numel() * k.element_size() >= (1 << 31),
            num_warps=warps,
            num_stages=stages,
        )
        merge[(batch, heads)](a, m, l, out, heads, splits, num_warps=4 if splits <= 32 else 8)
        return out
