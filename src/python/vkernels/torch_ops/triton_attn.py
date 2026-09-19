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
        kn_ptr,
        vn_ptr,
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
        stride_knb,
        stride_knh,
        G: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_NEW: tl.constexpr,
        SCRATCH: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        hkv = pid_h // G
        offs_d = tl.arange(0, D)
        qt = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d).to(
            tl.float32
        )
        seq = tl.load(sl_ptr + pid_b)
        slot = 0
        if HAS_NEW:
            # Fused new-token store: this program owns kv head ``hkv`` for
            # batch ``pid_b``; the fresh k/v is substituted into the
            # attention math from registers (below) and written to the
            # pool by one q-head program per group.
            slot = tl.load(bt_ptr + pid_b * stride_btb + (seq - 1))
            if slot != SCRATCH:
                if pid_h % G == 0:
                    kn = tl.load(kn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d)
                    tl.store(k_ptr + slot * stride_kb + hkv * stride_kh + offs_d, kn)
                    vn = tl.load(vn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d)
                    tl.store(v_ptr + slot * stride_vb + hkv * stride_vh + offs_d, vn)
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
            if HAS_NEW:
                kn = tl.load(kn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d).to(tl.float32)
                kblk = tl.where((idx == slot)[:, None], kn[None, :], kblk)
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
            if HAS_NEW:
                vn = tl.load(vn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d).to(tl.float32)
                vblk = tl.where((idx == slot)[:, None], vn[None, :], vblk)
            acc = acc * alpha + tl.sum(p[:, None] * vblk, axis=0)
            m = m_new
        tl.store(
            o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d,
            (acc / l).to(o_ptr.dtype.element_ty),
        )

    return _paged_decode_attn


def decode_attention(q, kc, vc, block_table, seq_lens, *, k_new=None, v_new=None, scratch_slot=None):
    """Decode attention for one token per batch: q [B, n_q, D], KV pages
    ``kc``/``vc`` [max_total, n_kv, D], per-batch page table and lengths.
    Returns [B, n_q, D] in the query dtype.

    ``k_new``/``v_new`` ([B, n_kv, D]) fuse the new token's KV store into
    this kernel: the fresh values are used directly in the attention math
    and written to the pool by one program per (batch, kv head), replacing
    the separate ``store_kv`` launch. ``scratch_slot`` (the shared null
    page) is never written."""
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
    has_new = k_new is not None and v_new is not None
    if has_new and k_new.shape != (B, n_kv, D):
        raise ValueError(f"k_new must be [B, n_kv, D], got {tuple(k_new.shape)}")

    _kernel()[(B, n_q)](
        q,
        kc,
        vc,
        bt,
        sl,
        out,
        k_new if has_new else q,  # unused dummy pointer when HAS_NEW=0
        v_new if has_new else q,
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
        k_new.stride(0) if has_new else 0,
        k_new.stride(1) if has_new else 0,
        G=G,
        D=D,
        BLOCK_N=128,
        HAS_NEW=has_new,
        SCRATCH=scratch_slot if scratch_slot is not None else -1,
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


# --- split-KV (flash-decoding) decode attention --------------------------------
# Two-phase scheme (same design as SGLang's decode_attention stage1/stage2,
# itself adapted from lightllm's gqa_flash_decoding; Apache-2.0):
#   stage 1: grid (B, H, SPLITS) — each program runs online softmax over its
#            slice of the sequence and writes partial (acc, m, l) buffers;
#   stage 2: grid (B, H) — log-sum-exp merge of the SPLITS partials.
# The extra split axis multiplies CTA count so short batches (B=1..8) can
# still saturate device bandwidth; the unsplitted kernel above serializes
# the whole sequence per (batch, head) program.

_MAX_KV_SPLITS = 8


@lru_cache(maxsize=1)
def _split_kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _paged_decode_attn_s1(
        q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr,
        acc_ptr, m_ptr, l_ptr,
        kn_ptr, vn_ptr,
        scale,
        split_len,
        stride_qb, stride_qh,
        stride_kb, stride_kh,
        stride_vb, stride_vh,
        stride_btb,
        stride_ab, stride_ah, stride_as,
        stride_mb, stride_mh,
        stride_knb, stride_knh,
        G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
        HAS_NEW: tl.constexpr, SCRATCH: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_s = tl.program_id(2)
        hkv = pid_h // G
        offs_d = tl.arange(0, D)
        seq = tl.load(sl_ptr + pid_b)
        start = pid_s * split_len
        out_off = pid_b * stride_ab + pid_h * stride_ah + pid_s * stride_as
        m_off = pid_b * stride_mb + pid_h * stride_mh + pid_s
        slot = 0
        owns_new = False
        if HAS_NEW:
            # Exactly one split covers the new token's position (seq-1);
            # that program substitutes the fresh k/v into its attention
            # math and (one q-head per group) writes the pool. No cross-CTA
            # ordering needed: no other program reads this slot.
            slot = tl.load(bt_ptr + pid_b * stride_btb + (seq - 1))
            end_all = tl.minimum(start + split_len, seq)
            owns_new = (seq - 1) >= start and (seq - 1) < end_all
            if slot != SCRATCH and owns_new:
                if pid_h % G == 0:
                    kn = tl.load(kn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d)
                    tl.store(k_ptr + slot * stride_kb + hkv * stride_kh + offs_d, kn)
                    vn = tl.load(vn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d)
                    tl.store(v_ptr + slot * stride_vb + hkv * stride_vh + offs_d, vn)
        if start < seq:
            end = tl.minimum(start + split_len, seq)
            qt = tl.load(q_ptr + pid_b * stride_qb + pid_h * stride_qh + offs_d).to(tl.float32)
            m = float("-inf")
            l = 0.0
            acc = tl.zeros([D], dtype=tl.float32)
            for t0 in range(start, end, BLOCK_N):
                offs = t0 + tl.arange(0, BLOCK_N)
                mask = offs < end
                idx = tl.load(bt_ptr + pid_b * stride_btb + offs, mask=mask, other=0)
                kblk = tl.load(
                    k_ptr + idx[:, None] * stride_kb + hkv * stride_kh + offs_d[None, :],
                    mask=mask[:, None], other=0.0,
                ).to(tl.float32)
                if HAS_NEW:
                    kn = tl.load(kn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d).to(tl.float32)
                    kblk = tl.where((idx == slot)[:, None], kn[None, :], kblk)
                scores = tl.sum(kblk * qt[None, :], axis=1) * scale
                scores = tl.where(mask, scores, float("-inf"))
                m_new = tl.maximum(m, tl.max(scores, axis=0))
                alpha = tl.exp(m - m_new)
                p = tl.exp(scores - m_new)
                l = l * alpha + tl.sum(p, axis=0)
                vblk = tl.load(
                    v_ptr + idx[:, None] * stride_vb + hkv * stride_vh + offs_d[None, :],
                    mask=mask[:, None], other=0.0,
                ).to(tl.float32)
                if HAS_NEW:
                    vn = tl.load(vn_ptr + pid_b * stride_knb + hkv * stride_knh + offs_d).to(tl.float32)
                    vblk = tl.where((idx == slot)[:, None], vn[None, :], vblk)
                acc = acc * alpha + tl.sum(p[:, None] * vblk, axis=0)
                m = m_new
            tl.store(acc_ptr + out_off + offs_d, acc)
            tl.store(m_ptr + m_off, m)
            tl.store(l_ptr + m_off, l)
        else:
            tl.store(acc_ptr + out_off + offs_d, tl.zeros([D], dtype=tl.float32))
            tl.store(m_ptr + m_off, float("-inf"))
            tl.store(l_ptr + m_off, 0.0)

    @triton.jit
    def _paged_decode_attn_s2(
        acc_ptr, m_ptr, l_ptr, o_ptr,
        stride_ab, stride_ah, stride_as,
        stride_mb, stride_mh,
        stride_ob, stride_oh,
        D: tl.constexpr, SPLITS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = tl.arange(0, D)
        m = float("-inf")
        l = 0.0
        acc = tl.zeros([D], dtype=tl.float32)
        for s in range(SPLITS):
            base = pid_b * stride_ab + pid_h * stride_ah + s * stride_as
            m_off = pid_b * stride_mb + pid_h * stride_mh + s
            ms = tl.load(m_ptr + m_off)
            ls = tl.load(l_ptr + m_off)
            a = tl.load(acc_ptr + base + offs_d)
            m_new = tl.maximum(m, ms)
            alpha = tl.exp(m - m_new)
            beta = tl.exp(ms - m_new)
            l = l * alpha + ls * beta
            acc = acc * alpha + a * beta
            m = m_new
        tl.store(
            o_ptr + pid_b * stride_ob + pid_h * stride_oh + offs_d,
            (acc / l).to(o_ptr.dtype.element_ty),
        )

    return _paged_decode_attn_s1, _paged_decode_attn_s2


def decode_attention_split(q, kc, vc, block_table, seq_lens, *, max_splits=_MAX_KV_SPLITS, block_n=64, max_len_hint=None, k_new=None, v_new=None, scratch_slot=None):
    """Split-KV decode attention: same contract as :func:`decode_attention`.

    Splits each sequence's KV range across ``max_splits`` programs per
    (batch, head) and merges the partials, so batch-1..8 decoding keeps
    enough CTAs in flight to use the device bandwidth.

    ``max_len_hint`` (host int) is an upper bound on the longest sequence
    (e.g. the block-table width); when given it avoids a ``sl.max().item()``
    device sync on the decode hot path. Split count adapts down from the
    bound; stage-1 programs whose slice is empty write neutral partials."""
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
    dev = q.device
    max_len = int(max_len_hint) if max_len_hint is not None else (int(sl.max().item()) if B else 1)
    splits = max(1, min(max_splits, (max_len + block_n - 1) // block_n))
    split_len = (max_len + splits - 1) // splits
    acc = torch.empty(B, n_q, splits, D, dtype=torch.float32, device=dev)
    ml = torch.empty(2, B, n_q, splits, dtype=torch.float32, device=dev)
    out = torch.empty_like(q)
    has_new = k_new is not None and v_new is not None
    if has_new and k_new.shape != (B, n_kv, D):
        raise ValueError(f"k_new must be [B, n_kv, D], got {tuple(k_new.shape)}")
    k1, k2 = _split_kernels()
    k1[(B, n_q, splits)](
        q, kc, vc, bt, sl,
        acc, ml[0], ml[1],
        k_new if has_new else q,  # unused dummy pointer when HAS_NEW=0
        v_new if has_new else q,
        1.0 / math.sqrt(D),
        split_len,
        q.stride(0), q.stride(1),
        kc.stride(0), kc.stride(1),
        vc.stride(0), vc.stride(1),
        bt.stride(0),
        acc.stride(0), acc.stride(1), acc.stride(2),
        ml[0].stride(0), ml[0].stride(1),
        k_new.stride(0) if has_new else 0,
        k_new.stride(1) if has_new else 0,
        G=G, D=D, BLOCK_N=block_n, SPLITS=splits,
        HAS_NEW=has_new,
        SCRATCH=scratch_slot if scratch_slot is not None else -1,
        num_warps=2,
    )
    k2[(B, n_q)](
        acc, ml[0], ml[1], out,
        acc.stride(0), acc.stride(1), acc.stride(2),
        ml[0].stride(0), ml[0].stride(1),
        out.stride(0), out.stride(1),
        D=D, SPLITS=splits,
        num_warps=2,
    )
    return out
