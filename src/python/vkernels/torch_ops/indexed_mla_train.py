"""Gathered-key MLA for training, with gradients to a learned per-key gate.

Q is in absorbed latent form [B,S,H,R] with a separate RoPE tail. K/V share
the normalized latent [B,S,R]. Each query supplies its own key indices and
additive gate [B,S,K]. This avoids expanded per-head gathered K/V tensors.
The trainer owns the autograd adapter and weight absorption/expansion.

The CPU oracle is torch-composed and differentiable. Device forward/backward
use online softmax and FP32 control/accumulation. Backward atomically sums
shared key gradients in FP32; its reduction order is nondeterministic.
"""

from functools import lru_cache


def indexed_mla_reference(q, qr, k, kr, indices, gate, scale):
    """Canonical differentiable oracle; invalid key indices contribute zero."""
    import torch

    b, s, h, r = q.shape
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    valid = (indices >= 0) & (indices < k.shape[1])
    safe = indices.clamp(0, k.shape[1] - 1).long()
    batch = torch.arange(b, device=q.device)[:, None, None]
    keys, ropes = k[batch, safe].to(dtype), kr[batch, safe].to(dtype)
    scores = torch.einsum("bshr,bskr->bshk", q.to(dtype), keys)
    if qr.shape[-1]:
        scores = scores + torch.einsum("bshp,bskp->bshk", qr.to(dtype), ropes)
    scores = scores * scale + gate.to(dtype).unsqueeze(2)
    scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
    nonempty = valid.any(-1, keepdim=True).unsqueeze(2)
    probs = torch.softmax(torch.where(nonempty, scores, 0.0), dim=-1)
    probs = probs.masked_fill(~valid.unsqueeze(2), 0.0)
    return torch.einsum("bshk,bskr->bshr", probs, keys).to(q.dtype)


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def forward(
        Q,
        QR,
        KV,
        KR,
        IX,
        G,
        O,
        O32,
        LSE,
        S: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        P: tl.constexpr,
        K: tl.constexpr,
        SCALE: tl.constexpr,
        BH: tl.constexpr,
        BR: tl.constexpr,
        BP: tl.constexpr,
        BN: tl.constexpr,
    ):
        row, batch = tl.program_id(0), tl.program_id(1)
        heads, rr = tl.arange(0, BH), tl.arange(0, BR)
        pp, nn = tl.arange(0, BP), tl.arange(0, BN)
        q = tl.load(
            Q + ((batch * S + row) * H + heads[:, None]) * R + rr[None, :], (heads[:, None] < H) & (rr[None, :] < R), 0
        )
        if P > 0:
            qr = tl.load(
                QR + ((batch * S + row) * H + heads[:, None]) * P + pp[None, :],
                (heads[:, None] < H) & (pp[None, :] < P),
                0,
            )
        m = tl.full((BH,), -float("inf"), tl.float32)
        total = tl.zeros((BH,), tl.float32)
        acc = tl.zeros((BH, BR), tl.float32)
        for start in range(0, K, BN):
            cols = start + nn
            key = tl.load(IX + (batch * S + row) * K + cols, cols < K, -1).to(tl.int32)
            valid = (cols < K) & (key >= 0) & (key < S)
            kv = tl.load(KV + (batch * S + key[:, None]) * R + rr[None, :], valid[:, None] & (rr[None, :] < R), 0)
            score = tl.dot(q, tl.trans(kv)).to(tl.float32)
            if P > 0:
                kr = tl.load(KR + (batch * S + key[:, None]) * P + pp[None, :], valid[:, None] & (pp[None, :] < P), 0)
                score += tl.dot(qr, tl.trans(kr))
            bias = tl.load(G + (batch * S + row) * K + cols, cols < K, 0).to(tl.float32)
            score = tl.where(valid[None, :] & (heads[:, None] < H), score * SCALE + bias[None, :], -float("inf"))
            new_m = tl.maximum(m, tl.max(score, axis=1))
            safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
            correction = tl.exp(m - safe_m)
            probs = tl.exp(score - safe_m[:, None])
            acc = acc * correction[:, None] + tl.dot(probs.to(kv.dtype), kv)
            total = total * correction + tl.sum(probs, axis=1)
            m = new_m
        result = acc / tl.where(total > 0, total, 1.0)[:, None]
        off = ((batch * S + row) * H + heads[:, None]) * R + rr[None, :]
        mask = (heads[:, None] < H) & (rr[None, :] < R)
        tl.store(O + off, result, mask)
        tl.store(O32 + off, result, mask)
        tl.store(LSE + (batch * S + row) * H + heads, m + tl.log(total), heads < H)

    @triton.jit
    def backward(
        DO,
        Q,
        QR,
        KV,
        KR,
        IX,
        G,
        O32,
        LSE,
        DQ,
        DQR,
        DKV,
        DKR,
        DG,
        B: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        R: tl.constexpr,
        P: tl.constexpr,
        K: tl.constexpr,
        SCALE: tl.constexpr,
        SPLITS: tl.constexpr,
        BH: tl.constexpr,
        BR: tl.constexpr,
        BP: tl.constexpr,
        BN: tl.constexpr,
    ):
        row, batch = tl.program_id(0), tl.program_id(1)
        heads, rr = tl.arange(0, BH), tl.arange(0, BR)
        pp, nn = tl.arange(0, BP), tl.arange(0, BN)
        off = ((batch * S + row) * H + heads[:, None]) * R + rr[None, :]
        hm = (heads[:, None] < H) & (rr[None, :] < R)
        q, do = tl.load(Q + off, hm, 0), tl.load(DO + off, hm, 0)
        out = tl.load(O32 + off, hm, 0)
        delta = tl.sum(out * do.to(tl.float32), axis=1)
        lse = tl.load(LSE + (batch * S + row) * H + heads, heads < H, -float("inf"))
        dq = tl.zeros((BH, BR), tl.float32)
        if P > 0:
            roff = ((batch * S + row) * H + heads[:, None]) * P + pp[None, :]
            rm = (heads[:, None] < H) & (pp[None, :] < P)
            qr = tl.load(QR + roff, rm, 0)
            dqr = tl.zeros((BH, BP), tl.float32)
        for start in range(0, K, BN):
            cols = start + nn
            key = tl.load(IX + (batch * S + row) * K + cols, cols < K, -1).to(tl.int32)
            valid = (cols < K) & (key >= 0) & (key < S)
            koff = (batch * S + key[:, None]) * R + rr[None, :]
            km = valid[:, None] & (rr[None, :] < R)
            kv = tl.load(KV + koff, km, 0)
            score = tl.dot(q, tl.trans(kv)).to(tl.float32)
            if P > 0:
                kroff = (batch * S + key[:, None]) * P + pp[None, :]
                krm = valid[:, None] & (pp[None, :] < P)
                kr = tl.load(KR + kroff, krm, 0)
                score += tl.dot(qr, tl.trans(kr))
            bias = tl.load(G + (batch * S + row) * K + cols, cols < K, 0).to(tl.float32)
            probs = tl.exp(score * SCALE + bias[None, :] - lse[:, None])
            probs = tl.where(valid[None, :] & (heads[:, None] < H), probs, 0.0)
            dp = tl.dot(do, tl.trans(kv)).to(tl.float32)
            ds = probs * (dp - delta[:, None])
            ds16, p16 = ds.to(q.dtype), probs.to(q.dtype)
            dq += tl.dot(ds16, kv) * SCALE
            dkv = tl.dot(tl.trans(ds16), q) * SCALE + tl.dot(tl.trans(p16), do)
            grad_base = (row % SPLITS) * B * S
            tl.atomic_add(DKV + grad_base * R + koff, dkv, km, sem="relaxed")
            if P > 0:
                dqr += tl.dot(ds16, kr) * SCALE
                dkr = tl.dot(tl.trans(ds16), qr) * SCALE
                tl.atomic_add(DKR + grad_base * P + kroff, dkr, krm, sem="relaxed")
            tl.store(DG + (batch * S + row) * K + cols, tl.sum(ds, axis=0), cols < K)
        tl.store(DQ + off, dq, hm)
        if P > 0:
            tl.store(DQR + roff, dqr, rm)

    return forward, backward


def _validate(q, qr, kv, kr, indices, gate):
    import torch

    b, s, h, r = q.shape
    p = qr.shape[-1]
    if qr.shape != (b, s, h, p) or kv.shape != (b, s, r) or kr.shape != (b, s, p):
        raise ValueError("inconsistent shared-latent MLA shapes")
    if indices.ndim != 3 or indices.shape[:2] != (b, s) or gate.shape != indices.shape:
        raise ValueError("indices and gate must have shape [B,S,K]")
    if not indices.shape[-1] or not s or not h or not r:
        raise ValueError("empty sequence, head, latent or selected-key dimensions")
    tensors = (q, qr, kv, kr, indices, gate)
    if any(not t.is_cuda or t.device != q.device or not t.is_contiguous() for t in tensors):
        raise ValueError("device kernels require contiguous tensors on one CUDA/ROCm device")
    if q.dtype not in (torch.bfloat16, torch.float16) or any(t.dtype != q.dtype for t in (qr, kv, kr)):
        raise ValueError("device Q/K/V tensors must share BF16 or FP16 dtype")
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be int32 or int64")
    return b, s, h, r, p, indices.shape[-1]


def _launch_args(h, r, p):
    import triton

    return dict(
        BH=max(16, triton.next_power_of_2(h)),
        BR=triton.next_power_of_2(r),
        BP=max(16, triton.next_power_of_2(max(p, 1))),
        BN=16,
        num_warps=4,
        num_stages=1,
    )


def indexed_mla_forward(q, qr, kv, kr, indices, gate, scale):
    """Device forward; also return FP32 output/LSE for the backward kernel."""
    import torch

    b, s, h, r, p, k = _validate(q, qr, kv, kr, indices, gate)
    out = torch.empty_like(q)
    out32 = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty((b, s, h), device=q.device, dtype=torch.float32)
    with torch.cuda.device(q.device):
        _kernels()[0][(s, b)](
            q, qr, kv, kr, indices, gate, out, out32, lse, s, h, r, p, k, float(scale), **_launch_args(h, r, p)
        )
    return out, out32, lse


def indexed_mla_backward(do, q, qr, kv, kr, indices, gate, out32, lse, scale, *, grad_splits=None):
    """Device backward: Q, RoPE Q, shared KV, RoPE K, and gate gradients."""
    import torch

    b, s, h, r, p, k = _validate(q, qr, kv, kr, indices, gate)
    # Several independent accumulators reduce contention on popular keys.
    # Each holds at most roughly 256 query rows until the bounded split
    # count is reached; the final FP32 reduction retains the same gradient.
    splits = min(8, (s + 255) // 256) if grad_splits is None else int(grad_splits)
    if splits < 1:
        raise ValueError("grad_splits must be positive")
    dq, dqr = torch.empty_like(q), torch.empty_like(qr)
    dkv = torch.zeros((splits, *kv.shape), device=q.device, dtype=torch.float32)
    dkr = torch.zeros((splits, *kr.shape), device=q.device, dtype=torch.float32)
    dg = torch.empty_like(gate)
    with torch.cuda.device(q.device):
        _kernels()[1][(s, b)](
            do,
            q,
            qr,
            kv,
            kr,
            indices,
            gate,
            out32,
            lse,
            dq,
            dqr,
            dkv,
            dkr,
            dg,
            b,
            s,
            h,
            r,
            p,
            k,
            float(scale),
            splits,
            **_launch_args(h, r, p),
        )
    if splits > 1:
        dkv, dkr = dkv.sum(0), dkr.sum(0)
    else:
        dkv, dkr = dkv[0], dkr[0]
    return dq, dqr, dkv.to(kv.dtype), dkr.to(kr.dtype), dg
