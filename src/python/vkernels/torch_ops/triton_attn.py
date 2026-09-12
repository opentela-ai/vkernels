"""Triton paged single-token decode attention (fp32 accumulation).

One program per (batch, query head): online-softmax pass over the paged
KV cache gathered through the block table, fp32 score/accumulator math,
output cast to the query dtype. The fallback attention backend for HF
models when FlashInfer is unavailable.

Adopted from floe's ``engine/runner/kernels/triton_attn.py`` (vkernels owns
the kernel; floe imports it back through a thin adapter — the #64/#65
thin-adapter model). Torch and Triton load lazily. Inference-only.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_decode_attn(
        q_ptr,
        k_ptr,
        v_ptr,
        bt_ptr,
        sl_ptr,
        o_ptr,
        scale,
        stride_qb,
        stride_qh,
        stride_kb,
        stride_kh,
        stride_vb,
        stride_vh,
        stride_btb,
        stride_ob,
        stride_oh,
        G: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        hkv = pid_h // G
        offs_d = tl.arange(0, D)
        qt = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d).to(
            tl.float32
        )
        seq = tl.load(sl_ptr + pid_b)
        m = float("-inf")
        l = 0.0
        acc = tl.zeros([D], dtype=tl.float32)
        for t0 in range(0, seq, BLOCK_N):
            offs = t0 + tl.arange(0, BLOCK_N)
            mask = offs < seq
            idx = tl.load(bt_ptr + pid_b * stride_btb + offs, mask=mask, other=0)
            kblk = tl.load(
                k_ptr + idx[:, None] * stride_kb + hkv * stride_kh + offs_d[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(kblk * qt[None, :], axis=1) * scale
            scores = tl.where(mask, scores, float("-inf"))
            m_new = tl.maximum(m, tl.max(scores, axis=0))
            alpha = tl.exp(m - m_new)
            p = tl.exp(scores - m_new)
            l = l * alpha + tl.sum(p, axis=0)
            vblk = tl.load(
                v_ptr + idx[:, None] * stride_vb + hkv * stride_vh + offs_d[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * vblk, axis=0)
            m = m_new
        tl.store(
            o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d,
            (acc / l).to(o_ptr.dtype.element_ty),
        )

    return _paged_decode_attn


def decode_attention(q, kc, vc, block_table, seq_lens):
    """Decode attention for one token per batch: q [B, n_q, D], KV pages
    ``kc``/``vc`` [max_total, n_kv, D], per-batch page table and lengths.
    Returns [B, n_q, D] in the query dtype."""
    import math

    import torch

    B, n_q, D = q.shape
    n_kv = kc.shape[1]
    if n_q % n_kv:
        raise ValueError(f"n_q ({n_q}) must be a multiple of n_kv ({n_kv})")
    G = n_q // n_kv
    bt = block_table.to(torch.int32)
    if not bt.is_contiguous():
        bt = bt.contiguous()
    sl = seq_lens.to(torch.int32)
    if not sl.is_contiguous():
        sl = sl.contiguous()
    out = torch.empty_like(q)

    _kernel()[(B, n_q)](
        q,
        kc,
        vc,
        bt,
        sl,
        out,
        1.0 / math.sqrt(D),
        q.stride(0),
        q.stride(1),
        kc.stride(0),
        kc.stride(1),
        vc.stride(0),
        vc.stride(1),
        bt.stride(0),
        out.stride(0),
        out.stride(1),
        G=G,
        D=D,
        BLOCK_N=128,
        num_warps=4,
    )
    return out


def decode_attention_reference(q, kc, vc, block_table, seq_lens):
    """Eager oracle: gather the paged KV per batch and run masked SDPA."""
    import math

    import torch

    B, n_q, D = q.shape
    n_kv = kc.shape[1]
    G = n_q // n_kv
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    for b in range(B):
        s = int(seq_lens[b])
        idx = block_table[b, :s].to(torch.long)
        k = kc[idx, :, :].transpose(0, 1).float()  # [n_kv, s, D]
        v = vc[idx, :, :].transpose(0, 1).float()
        qb = q[b].float().view(n_kv, G, D)  # [n_kv, G, D]
        scores = torch.einsum("ngd,nsd->ngs", qb, k) * scale
        p = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("ngs,nsd->ngd", p, v).reshape(n_q, D).to(q.dtype)
    return out
