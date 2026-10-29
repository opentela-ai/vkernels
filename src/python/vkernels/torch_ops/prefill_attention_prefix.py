"""Skip redundant masking over each query tile's fully attended KV prefix."""
from functools import lru_cache

from ._dispatch import OpNotEligible


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
    def paged_prefill_masked_tail(Q, K, V, BT, LENGTHS, POS, OUT,
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
        full_limit = tl.maximum(0, tl.minimum(length, tl.min(pos, 0) + 1)) // BN * BN
        for phase in tl.static_range(2):
            lo_bound = 0 if phase == 0 else full_limit
            hi_bound = full_limit if phase == 0 else limit
            for start in range(lo_bound, hi_bound, BN):
                columns = start + tl.arange(0, BN)
                if phase == 0:
                    slots = tl.load(BT + batch * BTS + columns).to(tl.int64)
                else:
                    slots = tl.load(BT + batch * BTS + columns, columns < length, 0).to(tl.int64)
                if LARGE_PLANE:
                    kp = (K.to(tl.uint64) + (slots[None, :] * KS + kvhead * KH + dims[:, None]) * 2).to(tl.pointer_type(tl.bfloat16))
                    vp = (V.to(tl.uint64) + (slots[:, None] * VS + kvhead * VH + dims[None, :]) * 2).to(tl.pointer_type(tl.bfloat16))
                else:
                    kp = K + slots[None, :] * KS + kvhead * KH + dims[:, None]
                    vp = V + slots[:, None] * VS + kvhead * VH + dims[None, :]
                if phase == 0:
                    k = tl.load(kp)
                else:
                    k = tl.load(kp, columns[None, :] < length, 0)
                score = tl.dot(q, k) * (0.08838834764831845 * 1.4426950408889634)
                if phase == 1:
                    allowed = (columns[None, :] < length) & (columns[None, :] <= pos[:, None]) & (rows[:, None] < N)
                    score = tl.where(allowed, score, float('-inf'))
                new_max = tl.maximum(maximum, tl.max(score, 1))
                new_max = tl.where(new_max == float('-inf'), 0., new_max)
                correction = tl.exp2(maximum - new_max)
                prob = tl.exp2(score - new_max[:, None])
                denom = denom * correction + tl.sum(prob, 1)
                acc *= correction[:, None]
                if phase == 0:
                    v = tl.load(vp)
                else:
                    v = tl.load(vp, columns[:, None] < length, 0)
                hi = prob.to(q.dtype)
                lo = (prob - hi.to(tl.float32)).to(q.dtype)
                acc = tl.dot(hi, v, acc)
                acc = tl.dot(lo, v, acc)
                maximum = new_max
        acc = acc / tl.where(denom > 0, denom, 1.)[:, None]
        tl.store(OUT + ((batch * N + rows[:, None]) * HQ + head) * 128 + dims[None, :], acc, rows[:, None] < N)
    return triton, paged_prefill_masked_tail

