"""Triton hybrid megakernel: Qwen3.8-27B-FP8 (Qwen3.5-family) decode step.

One persistent launch per decode token covering all 64 layers of the real
27B target: 48 GDN linear-attention layers + 16 full-attention layers,
DeepSeek-style 128x128 block-FP8 projections, persistent GDN states (ssm
[48, 48, 128, 128] fp32 + conv [48, 3, 10240] fp32), a full-attn KV pool
[16, S, 4, 256], untied BF16 embed/lm_head — on the same §8.1 phase-
synchronous schedule and monotonic grid barrier as the dense backend.

Task bodies reuse the pieces validated against the real checkpoint in
``tests/test_megakernel_27b.py`` (``_t_gemv_fp8``, ``_t_gdn_conv``,
``_t_gdn_heads`` match the repo reference to <1e-4); the full-attention
path adds partial RoPE (rotary_dim 64 of head_dim 256, theta 1e7),
per-head QK-norm, GQA 24Q/4KV, and the fused per-head sigmoid output
gate. Weights are packed layer-uniformly per class (one base pointer +
stride per class) so the runtime layer loop addresses them without
per-layer kernel arguments.

B=1 decode is the contract here (batched hybrid decode and the DFlash2
drafter integration are follow-ups; the drafter reads target hiddens at
layers [5,19,33,47,61], which this kernel's workspace naturally exposes
after those layers' phases).
"""

from __future__ import annotations

import glob
import os

import torch
import triton
import triton.language as tl

from .device_triton import _t_gdn_conv, _t_gdn_heads, grid_barrier

__all__ = ["HybridMegakernel", "qwen35_hybrid_megakernel"]


# ---------------------------------------------------------------------------
# Task bodies (layer-offset addressing; K/N constexpr, strides runtime)
# ---------------------------------------------------------------------------


@triton.jit
def _h_embed(worker: tl.int32, P: tl.int32, token, emb_ptr, out_ptr, C: tl.constexpr, BC: tl.constexpr):
    task = worker
    while task < 1:
        offs = tl.arange(0, BC)
        m = offs < C
        row = tl.load(emb_ptr + token.to(tl.int64) * C + offs, mask=m, other=0.0)
        tl.store(out_ptr + offs, row.to(tl.float32), mask=m)
        task += P


@triton.jit
def _h_rms(worker: tl.int32, P: tl.int32, x_ptr, g_ptr, y_ptr, C: tl.constexpr, BC: tl.constexpr, eps: tl.constexpr):
    task = worker
    while task < 1:
        offs = tl.arange(0, BC)
        m = offs < C
        x = tl.load(x_ptr + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / C
        tl.store(y_ptr + offs, x * (1.0 / tl.sqrt(var + eps)) * (1.0 + g), mask=m)
        task += P


@triton.jit
def _h_gemv_fp8(worker: tl.int32, P: tl.int32, x_ptr, w_ptr, sc_ptr, y_ptr, WSTRIDE, SSTRIDE, K: tl.constexpr, N: tl.constexpr):
    """y = dequant(w) @ x over layer-offset fp8 weights (128-block scales)."""
    TILE: tl.constexpr = 16
    BK: tl.constexpr = 128
    NTASK: tl.constexpr = N // TILE
    KB: tl.constexpr = K // BK
    task = worker
    while task < NTASK:
        offs_n = task * TILE + tl.arange(0, TILE)
        sb_row = (task * TILE) // BK
        acc = tl.zeros([TILE], tl.float32)
        for kb in range(0, KB):
            offs_k = kb * BK + tl.arange(0, BK)
            s = tl.load(sc_ptr + sb_row * KB + kb).to(tl.float32)
            xv = tl.load(x_ptr + offs_k, cache_modifier=".cg").to(tl.float32)
            wt = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.float32)
            acc += tl.sum(wt * (s * xv[None, :]), axis=1)
        tl.store(y_ptr + offs_n, acc)
        task += P


@triton.jit
def _h_gemv_bf16(worker: tl.int32, P: tl.int32, x_ptr, w_ptr, y_ptr, WSTRIDE, K: tl.constexpr, N: tl.constexpr):
    """y = w(bf16, [N, K]) @ x — layer-offset addressing (embed/lm_head)."""
    TILE: tl.constexpr = 16
    BK: tl.constexpr = 256
    NTASK: tl.constexpr = N // TILE
    task = worker
    while task < NTASK:
        offs_n = task * TILE + tl.arange(0, TILE)
        acc = tl.zeros([TILE], tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            xv = tl.load(x_ptr + offs_k, cache_modifier=".cg").to(tl.float32)
            wt = tl.load(w_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :]).to(tl.float32)
            acc += tl.sum(wt * xv[None, :], axis=1)
        tl.store(y_ptr + offs_n, acc)
        task += P


@triton.jit
def _h_gemv_bf16_small(worker: tl.int32, P: tl.int32, x_ptr, w_ptr, y_ptr, WSTRIDE, K: tl.constexpr, N: tl.constexpr):
    """Small-N bf16 gemv (gdn in_proj_a/b: [48, 5120]) — one task."""
    task = worker
    while task < 1:
        offs_n = tl.arange(0, 64)
        mn = offs_n < N
        acc = tl.zeros([64], tl.float32)
        for k0 in range(0, K, 256):
            offs_k = k0 + tl.arange(0, 256)
            xv = tl.load(x_ptr + offs_k, cache_modifier=".cg").to(tl.float32)
            wt = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mn[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(wt * xv[None, :], axis=1)
        tl.store(y_ptr + offs_n, acc, mask=mn)
        task += P


@triton.jit
def _h_add(worker: tl.int32, P: tl.int32, a_ptr, b_ptr, y_ptr, C: tl.constexpr, BC: tl.constexpr):
    task = worker
    while task < 1:
        offs = tl.arange(0, BC)
        m = offs < C
        a = tl.load(a_ptr + offs, mask=m, other=0.0, cache_modifier=".cg")
        b = tl.load(b_ptr + offs, mask=m, other=0.0, cache_modifier=".cg")
        tl.store(y_ptr + offs, a + b, mask=m)
        task += P


@triton.jit
def _h_swiglu(worker: tl.int32, P: tl.int32, gu_ptr, y_ptr, F: tl.constexpr):
    task = worker
    while task < 1:
        for f0 in range(0, F, 4096):
            offs = f0 + tl.arange(0, 4096)
            m = offs < F
            g = tl.load(gu_ptr + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
            u = tl.load(gu_ptr + F + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
            tl.store(y_ptr + offs, g / (1.0 + tl.exp(-g)) * u, mask=m)
        task += P


# ---- full-attention tasks --------------------------------------------------


@triton.jit
def _h_qknorm(worker: tl.int32, P: tl.int32, qkv_ptr, qn_ptr, kn_ptr, out_ptr, H: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, eps: tl.constexpr):
    """Per-head RMSNorm of q (with its fused gate passed through) and k.

    Input layout at ``qkv_ptr``: q+gate per head [q(D) | gate(D)] x H at
    offset h*2D; k per kv-head at H*2D + kh*D. Outputs: normed q at
    ``out_ptr + h*D``; gate copied to ``out_ptr + H*D + h*D``; normed k at
    ``out_ptr + H*D + H*D + kh*D``.
    """
    task = worker
    while task < H + NKV:
        offs = tl.arange(0, D)
        if task < H:
            h = task
            q = tl.load(qkv_ptr + h * 2 * D + offs, cache_modifier=".cg").to(tl.float32)
            var = tl.sum(q * q, axis=0) / D
            w = tl.load(qn_ptr + offs).to(tl.float32)
            tl.store(out_ptr + h * D + offs, q * (1.0 / tl.sqrt(var + eps)) * (1.0 + w))
            g = tl.load(qkv_ptr + h * 2 * D + D + offs, cache_modifier=".cg").to(tl.float32)
            tl.store(out_ptr + H * D + h * D + offs, g)
        else:
            kh = task - H
            k = tl.load(qkv_ptr + H * 2 * D + kh * D + offs, cache_modifier=".cg").to(tl.float32)
            var = tl.sum(k * k, axis=0) / D
            w = tl.load(kn_ptr + offs).to(tl.float32)
            tl.store(out_ptr + 2 * H * D + kh * D + offs, k * (1.0 / tl.sqrt(var + eps)) * (1.0 + w))
        task += P


@triton.jit
def _h_rope_append(
    worker: tl.int32,
    P: tl.int32,
    post_ptr,
    cos_ptr,
    sin_ptr,
    k_cache_ptr,
    v_src_ptr,
    v_cache_ptr,
    pos,
    slot,
    H: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    SCAP: tl.constexpr,
):
    """Partial RoPE (first ROT of D dims) on q (in ws) and k (append to KV).

    q heads rotate in place at ``post_ptr + h*D``; k heads rotate and are
    appended to the KV pool at ``slot`` (token-major [S, NKV, D] per fa
    layer). Dims >= ROT pass through unchanged.  v heads (no RoPE) are
    copied from ``v_src_ptr`` to the V pool at the same slot.
    """
    task = worker
    half: tl.constexpr = ROT // 2
    while task < H + 2 * NKV:
        d = tl.arange(0, half)
        p = pos.to(tl.int64)
        c = tl.load(cos_ptr + p * ROT + d).to(tl.float32)
        s = tl.load(sin_ptr + p * ROT + d).to(tl.float32)
        if task < H:
            h = task
            base = post_ptr + h * D
            x1 = tl.load(base + d, cache_modifier=".cg").to(tl.float32)
            x2 = tl.load(base + half + d, cache_modifier=".cg").to(tl.float32)
            tl.store(base + d, x1 * c - x2 * s)
            tl.store(base + half + d, x2 * c + x1 * s)
        elif task < H + NKV:
            kh = task - H
            base = post_ptr + 2 * H * D + kh * D
            x1 = tl.load(base + d, cache_modifier=".cg").to(tl.float32)
            x2 = tl.load(base + half + d, cache_modifier=".cg").to(tl.float32)
            r1 = x1 * c - x2 * s
            r2 = x2 * c + x1 * s
            pass_all = tl.arange(0, D)
            mt = (pass_all >= ROT) & (pass_all < D)
            tail = tl.load(base + pass_all, mask=mt, other=0.0, cache_modifier=".cg").to(tl.float32)
            dst = k_cache_ptr + slot.to(tl.int64) * (NKV * D) + kh * D
            tl.store(dst + d, r1)
            tl.store(dst + half + d, r2)
            tl.store(dst + pass_all, tail, mask=mt)
        else:
            # v copy (no RoPE) -- raw v_proj output to V pool
            vh = task - H - NKV
            offs_d = tl.arange(0, D)
            v_val = tl.load(v_src_ptr + vh * D + offs_d, cache_modifier=".cg").to(tl.float32)
            v_dst = v_cache_ptr + slot.to(tl.int64) * (NKV * D) + vh * D
            tl.store(v_dst + offs_d, v_val)
        task += P


@triton.jit
def _h_scores(worker: tl.int32, P: tl.int32, q_ptr, k_cache_ptr, y_ptr, pos, H: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, SCAP: tl.constexpr, BT: tl.constexpr, scale: tl.constexpr):
    task = worker
    GROUP: tl.constexpr = H // NKV
    while task < H:
        h = task
        kvh = h // GROUP
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + h * D + offs_d, cache_modifier=".cg").to(tl.float32)
        p1 = pos.to(tl.int64) + 1
        kbase = k_cache_ptr + kvh * D
        t0 = tl.zeros((), tl.int64)
        while t0 < p1:
            offs_t = t0 + tl.arange(0, BT)
            m = offs_t < p1
            kt = tl.load(kbase + offs_t[:, None] * (NKV * D) + offs_d[None, :], mask=m[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
            s = tl.sum(kt * q[None, :], axis=1) * scale
            tl.store(y_ptr + h * SCAP + offs_t, s, mask=m)
            t0 += BT
        task += P


@triton.jit
def _h_softmax(worker: tl.int32, P: tl.int32, x_ptr, pos, y_ptr, H: tl.constexpr, SCAP: tl.constexpr, BTR: tl.constexpr):
    task = worker
    while task < H:
        offs_t = tl.arange(0, BTR)
        m = offs_t < (pos.to(tl.int64) + 1)
        s = tl.load(x_ptr + task * SCAP + offs_t, mask=m, other=-float("inf"), cache_modifier=".cg")
        mr = tl.max(s, axis=0)
        e = tl.exp(s - mr)
        inv = 1.0 / tl.sum(e, axis=0)
        tl.store(y_ptr + task * SCAP + offs_t, tl.where(m, e * inv, 0.0), mask=offs_t < SCAP)
        task += P


@triton.jit
def _h_values_gate(worker: tl.int32, P: tl.int32, probs_ptr, v_cache_ptr, gate_ptr, y_ptr, pos, H: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, SCAP: tl.constexpr, BT: tl.constexpr):
    """ctx_h = sum probs*V, then per-head sigmoid output gate."""
    task = worker
    GROUP: tl.constexpr = H // NKV
    while task < H:
        h = task
        kvh = h // GROUP
        offs_d = tl.arange(0, D)
        acc = tl.zeros([D], tl.float32)
        p1 = pos.to(tl.int64) + 1
        vbase = v_cache_ptr + kvh * D
        t0 = tl.zeros((), tl.int64)
        while t0 < p1:
            offs_t = t0 + tl.arange(0, BT)
            m = offs_t < p1
            pv = tl.load(probs_ptr + h * SCAP + offs_t, mask=m, other=0.0, cache_modifier=".cg")
            vv = tl.load(vbase + offs_t[:, None] * (NKV * D) + offs_d[None, :], mask=m[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
            acc += tl.sum(pv[:, None] * vv, axis=0)
            t0 += BT
        gate = tl.load(gate_ptr + h * D + offs_d, cache_modifier=".cg").to(tl.float32)
        tl.store(y_ptr + h * D + offs_d, acc * (1.0 / (1.0 + tl.exp(-gate))))
        task += P


# ---------------------------------------------------------------------------
# The hybrid megakernel
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["token", "pos", "slot", "bar_base", "L"])
def qwen35_hybrid_megakernel(
    token,
    pos,
    slot,
    bar_ptr,
    bar_base: tl.int64,
    L,
    # weight base pointers (layer-uniform packing; class strides runtime)
    ln1_ptr,
    ln2_ptr,
    finaln_ptr,
    emb_ptr,
    lmh_ptr,
    gqkv_w,
    gqkv_s,
    gz_w,
    gz_s,
    ga_w,
    gb_w,
    gout_w,
    gout_s,
    gconv_w,
    galog,
    gdtb,
    gnormw,
    fq_w,
    fq_s,
    fk_w,
    fk_s,
    fv_w,
    fv_s,
    fo_w,
    fo_s,
    fqn,
    fkn,
    mgate_w,
    mgate_s,
    mup_w,
    mup_s,
    mdown_w,
    mdown_s,
    # persistent state
    ssm_ptr,
    conv_ptr,
    k_cache_ptr,
    v_cache_ptr,
    cos_ptr,
    sin_ptr,
    # workspace + outputs
    ws_ptr,
    logits_ptr,
    # dims
    C: tl.constexpr,
    F: tl.constexpr,
    V: tl.constexpr,
    SCAP: tl.constexpr,
    H: tl.constexpr,
    NKV: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    NKH: tl.constexpr,
    NVH: tl.constexpr,
    HK: tl.constexpr,
    HV: tl.constexpr,
    GQKV_N: tl.constexpr,
    GZ_N: tl.constexpr,
    FQ_N: tl.constexpr,
    FK_N: tl.constexpr,
    FV_N: tl.constexpr,
    EPS: tl.constexpr,
    FSCALE: tl.constexpr,
    GSCALE: tl.constexpr,
    BT: tl.constexpr,
    BTR: tl.constexpr,
    O_HIDDEN_A: tl.constexpr,
    O_HIDDEN_B: tl.constexpr,
    O_RMS1: tl.constexpr,
    O_PROJ: tl.constexpr,
    O_ZAB: tl.constexpr,
    O_POST: tl.constexpr,
    O_HEADS: tl.constexpr,
    O_ATTN: tl.constexpr,
    O_RMS2: tl.constexpr,
    O_GU: tl.constexpr,
    O_ACT: tl.constexpr,
    O_SCORES: tl.constexpr,
    O_PROBS: tl.constexpr,
    KILL_AFTER: tl.constexpr,
):
    worker = tl.program_id(0)
    P = tl.num_programs(0)

    _h_embed(worker, P, token, emb_ptr, ws_ptr + O_HIDDEN_A, C, 8192)
    grid_barrier(bar_ptr, bar_base + P)

    # Barrier numbering: embed is #1; GDN layers use 11 barriers, full-attn
    # 14, so layer l's block starts at (11*l + 3*(l//4) + 1) + 1 (layer j is
    # full-attention iff (j+1) % 4 == 0, count among 0..l-1 = l//4).
    for l in range(L):
        if (KILL_AFTER < 0) or (l < KILL_AFTER):
            is_full = ((l + 1) % 4) == 0
            gi = l - (l + 1) // 4
            fi = (l + 1) // 4 - 1
            bi = (11 * l + 3 * (l // 4) + 1) * P + bar_base

            _h_rms(worker, P, ws_ptr + O_HIDDEN_A, ln1_ptr + l * C, ws_ptr + O_RMS1, C, 8192, EPS)
            grid_barrier(bar_ptr, bi + P)

            if is_full:
                # ---- full-attention layer (14 barriers) ----------------------
                _h_gemv_fp8(worker, P, ws_ptr + O_RMS1, fq_w + fi.to(tl.int64) * (FQ_N * C), fq_s + fi.to(tl.int64) * ((FQ_N // 128) * (C // 128)), ws_ptr + O_PROJ, 0, 0, C, FQ_N)
                _h_gemv_fp8(worker, P, ws_ptr + O_RMS1, fk_w + fi.to(tl.int64) * (FK_N * C), fk_s + fi.to(tl.int64) * ((FK_N // 128) * (C // 128)), ws_ptr + O_ZAB, 0, 0, C, FK_N)
                _h_gemv_fp8(worker, P, ws_ptr + O_RMS1, fv_w + fi.to(tl.int64) * (FV_N * C), fv_s + fi.to(tl.int64) * ((FV_N // 128) * (C // 128)), ws_ptr + O_ZAB + FK_N, 0, 0, C, FV_N)
                grid_barrier(bar_ptr, bi + 2 * P)
                _h_qknorm(worker, P, ws_ptr + O_PROJ, fqn + fi * D, fkn + fi * D, ws_ptr + O_POST, H, NKV, D, EPS)
                grid_barrier(bar_ptr, bi + 3 * P)
                _h_rope_append(worker, P, ws_ptr + O_POST, cos_ptr, sin_ptr, k_cache_ptr + fi.to(tl.int64) * (SCAP * NKV * D), ws_ptr + O_ZAB + FK_N, v_cache_ptr + fi.to(tl.int64) * (SCAP * NKV * D), pos, slot, H, NKV, D, ROT, SCAP)
                grid_barrier(bar_ptr, bi + 4 * P)
                _h_scores(worker, P, ws_ptr + O_POST, k_cache_ptr + fi.to(tl.int64) * (SCAP * NKV * D), ws_ptr + O_SCORES, pos, H, NKV, D, SCAP, BT, FSCALE)
                grid_barrier(bar_ptr, bi + 5 * P)
                _h_softmax(worker, P, ws_ptr + O_SCORES, pos, ws_ptr + O_PROBS, H, SCAP, BTR)
                grid_barrier(bar_ptr, bi + 6 * P)
                _h_values_gate(worker, P, ws_ptr + O_PROBS, v_cache_ptr + fi.to(tl.int64) * (SCAP * NKV * D), ws_ptr + O_POST + H * D, ws_ptr + O_HEADS, pos, H, NKV, D, SCAP, BT)
                grid_barrier(bar_ptr, bi + 7 * P)
                _h_gemv_fp8(worker, P, ws_ptr + O_HEADS, fo_w + fi.to(tl.int64) * (C * (H * D)), fo_s + fi.to(tl.int64) * ((C // 128) * ((H * D) // 128)), ws_ptr + O_ATTN, 0, 0, H * D, C)
                grid_barrier(bar_ptr, bi + 8 * P)
                _h_add(worker, P, ws_ptr + O_HIDDEN_A, ws_ptr + O_ATTN, ws_ptr + O_HIDDEN_B, C, 8192)
                grid_barrier(bar_ptr, bi + 9 * P)
            else:
                # ---- GDN layer (11 barriers) --------------------------------
                _h_gemv_fp8(worker, P, ws_ptr + O_RMS1, gqkv_w + gi.to(tl.int64) * (GQKV_N * C), gqkv_s + gi.to(tl.int64) * ((GQKV_N // 128) * (C // 128)), ws_ptr + O_PROJ, 0, 0, C, GQKV_N)
                _h_gemv_fp8(worker, P, ws_ptr + O_RMS1, gz_w + gi.to(tl.int64) * (GZ_N * C), gz_s + gi.to(tl.int64) * ((GZ_N // 128) * (C // 128)), ws_ptr + O_ZAB, 0, 0, C, GZ_N)
                _h_gemv_bf16_small(worker, P, ws_ptr + O_RMS1, ga_w + gi.to(tl.int64) * (NVH * C), ws_ptr + O_ZAB + GZ_N, 0, C, NVH)
                _h_gemv_bf16_small(worker, P, ws_ptr + O_RMS1, gb_w + gi.to(tl.int64) * (NVH * C), ws_ptr + O_ZAB + GZ_N + NVH, 0, C, NVH)
                grid_barrier(bar_ptr, bi + 2 * P)
                _t_gdn_conv(worker, P, conv_ptr + gi.to(tl.int64) * (3 * (2 * NKH * HK + NVH * HV)), gconv_w + gi.to(tl.int64) * ((2 * NKH * HK + NVH * HV) * 4), ws_ptr + O_PROJ, ws_ptr + O_POST, 2 * NKH * HK + NVH * HV, 1024, 4)
                grid_barrier(bar_ptr, bi + 3 * P)
                _t_gdn_heads(
                    worker, P,
                    ws_ptr + O_POST,
                    ws_ptr + O_POST + NKH * HK,
                    ws_ptr + O_POST + 2 * NKH * HK,
                    ws_ptr + O_ZAB,
                    ws_ptr + O_ZAB + GZ_N,
                    ws_ptr + O_ZAB + GZ_N + NVH,
                    galog + gi * NVH,
                    gdtb + gi * NVH,
                    gnormw + gi * HV,
                    ssm_ptr + gi.to(tl.int64) * (NVH * HV * HK),
                    ws_ptr + O_HEADS,
                    NVH, NKH, HV, HK, EPS, GSCALE,
                )
                grid_barrier(bar_ptr, bi + 4 * P)
                _h_gemv_fp8(worker, P, ws_ptr + O_HEADS, gout_w + gi.to(tl.int64) * (C * (NVH * HV)), gout_s + gi.to(tl.int64) * ((C // 128) * ((NVH * HV) // 128)), ws_ptr + O_ATTN, 0, 0, NVH * HV, C)
                grid_barrier(bar_ptr, bi + 5 * P)
                _h_add(worker, P, ws_ptr + O_HIDDEN_A, ws_ptr + O_ATTN, ws_ptr + O_HIDDEN_B, C, 8192)
                grid_barrier(bar_ptr, bi + 6 * P)

            # ---- shared MLP (5 barriers, both layer types) -------------------
            # Continues the layer's own numbering contiguously: GDN ended at
            # bi+6, FA at bi+9, so the MLP barriers are type-relative.
            m0 = 7 if not is_full else 10
            _h_rms(worker, P, ws_ptr + O_HIDDEN_B, ln2_ptr + l * C, ws_ptr + O_RMS2, C, 8192, EPS)
            grid_barrier(bar_ptr, bi + m0 * P)
            _h_gemv_fp8(worker, P, ws_ptr + O_RMS2, mgate_w + l.to(tl.int64) * (F * C), mgate_s + l.to(tl.int64) * ((F // 128) * (C // 128)), ws_ptr + O_GU, 0, 0, C, F)
            _h_gemv_fp8(worker, P, ws_ptr + O_RMS2, mup_w + l.to(tl.int64) * (F * C), mup_s + l.to(tl.int64) * ((F // 128) * (C // 128)), ws_ptr + O_GU + F, 0, 0, C, F)
            grid_barrier(bar_ptr, bi + (m0 + 1) * P)
            _h_swiglu(worker, P, ws_ptr + O_GU, ws_ptr + O_ACT, F)
            grid_barrier(bar_ptr, bi + (m0 + 2) * P)
            _h_gemv_fp8(worker, P, ws_ptr + O_ACT, mdown_w + l.to(tl.int64) * (C * F), mdown_s + l.to(tl.int64) * ((C // 128) * (F // 128)), ws_ptr + O_ATTN, 0, 0, F, C)
            grid_barrier(bar_ptr, bi + (m0 + 3) * P)
            _h_add(worker, P, ws_ptr + O_HIDDEN_B, ws_ptr + O_ATTN, ws_ptr + O_HIDDEN_A, C, 8192)
            grid_barrier(bar_ptr, bi + (m0 + 4) * P)

    # total barriers = 1 (embed) + 48*11 + 16*14 (layers) + 1 (final rms) = 754
    if KILL_AFTER < 0:
        _h_rms(worker, P, ws_ptr + O_HIDDEN_A, finaln_ptr, ws_ptr + O_RMS1, C, 8192, EPS)
        grid_barrier(bar_ptr, bar_base + (11 * L + 3 * (L // 4) + 2) * P)
        _h_gemv_bf16(worker, P, ws_ptr + O_RMS1, lmh_ptr, logits_ptr, 0, C, V)


# ---------------------------------------------------------------------------
# Host side: pack the real checkpoint into layer-uniform classes + run
# ---------------------------------------------------------------------------

TARGET_DEFAULT = "/local/home/xiayao/minisgl-ds5/models/Qwen3.8-27B-FP8"

# workspace element offsets (fp32); one layer slot reused across layers
C, F = 5120, 17408
_HS, _HF, _HNKV, _HD = 24, 17408, 4, 256
_O = {
    "hidden_a": 0,
    "hidden_b": C,
    "rms1": 2 * C,
    "proj": 3 * C,  # gdn qkv 10240 | fa q+gate 12288
    "zab": 3 * C + 12288,  # gdn z 6144, a 48, b 48 | fa k 1024, v 1024
    "post": 3 * C + 12288 + 6272,  # gdn conv_out 10240 | fa normq 6144, gate 6144, normk 1024
    "heads": 3 * C + 12288 + 6272 + 20480,  # gdn out 6144 | fa ctx 6144
    "attn": 3 * C + 12288 + 6272 + 20480 + 12288,
    "rms2": 3 * C + 12288 + 6272 + 20480 + 12288 + C,
    "gu": 3 * C + 12288 + 6272 + 20480 + 12288 + 2 * C,
    "act": 3 * C + 12288 + 6272 + 20480 + 12288 + 2 * C + 2 * F,
    "scores": 3 * C + 12288 + 6272 + 20480 + 12288 + 2 * C + 2 * F + F,
    "probs": 3 * C + 12288 + 6272 + 20480 + 12288 + 2 * C + 2 * F + F + 24 * 512,
}
WS_TOTAL = _O["probs"] + 24 * 512


class HybridMegakernel:
    """The 27B hybrid target as one persistent launch per decode step."""

    NUM_BARRIERS = 754

    def __init__(self, checkpoint: str = TARGET_DEFAULT, *, capacity: int = 512, workers: int = 48, device: str = "cuda"):
        self.device = device
        self.capacity = capacity
        self.workers = workers
        C_, F_ = C, F
        nk, nv, hk, hv = 16, 48, 128, 128
        self.dims = dict(C=C_, F=F_, H=_HS, NKV=_HNKV, D=_HD, ROT=64, NKH=nk, NVH=nv, HK=hk, HV=hv)
        gqkv_n, gz_n = 2 * nk * hk + nv * hv, nv * hv
        fq_n, fk_n = _HS * 2 * _HD, _HNKV * _HD

        dev = torch.device(device)
        shards = sorted(glob.glob(os.path.join(checkpoint, "layers-*.safetensors")), key=lambda p: int(p.rsplit("-", 1)[1].split(".")[0]))
        outside = [p for p in glob.glob(os.path.join(checkpoint, "*.safetensors")) if "layers" not in p][0]
        from safetensors.torch import load_file

        def alloc(shape, dtype):
            return torch.empty(shape, device=dev, dtype=dtype)

        L = 64
        LG, LF = 48, 16
        w = {}
        w["ln1"] = alloc((L, C_), torch.bfloat16)
        w["ln2"] = alloc((L, C_), torch.bfloat16)
        w["gqkv_w"], w["gqkv_s"] = alloc((LG, gqkv_n, C_), torch.float8_e4m3fn), alloc((LG, gqkv_n // 128, C_ // 128), torch.bfloat16)
        w["gz_w"], w["gz_s"] = alloc((LG, gz_n, C_), torch.float8_e4m3fn), alloc((LG, gz_n // 128, C_ // 128), torch.bfloat16)
        w["ga"], w["gb"] = alloc((LG, nv, C_), torch.bfloat16), alloc((LG, nv, C_), torch.bfloat16)
        w["gout_w"], w["gout_s"] = alloc((LG, C_, gz_n), torch.float8_e4m3fn), alloc((LG, C_ // 128, gz_n // 128), torch.bfloat16)
        w["gconv"] = alloc((LG, 2 * nk * hk + nv * hv, 4), torch.bfloat16)
        w["galog"], w["gdtb"] = alloc((LG, nv), torch.float32), alloc((LG, nv), torch.float32)
        w["gnormw"] = alloc((LG, hv), torch.bfloat16)
        w["fq_w"], w["fq_s"] = alloc((LF, fq_n, C_), torch.float8_e4m3fn), alloc((LF, fq_n // 128, C_ // 128), torch.bfloat16)
        w["fk_w"], w["fk_s"] = alloc((LF, fk_n, C_), torch.float8_e4m3fn), alloc((LF, fk_n // 128, C_ // 128), torch.bfloat16)
        w["fv_w"], w["fv_s"] = alloc((LF, fk_n, C_), torch.float8_e4m3fn), alloc((LF, fk_n // 128, C_ // 128), torch.bfloat16)
        w["fo_w"], w["fo_s"] = alloc((LF, C_, _HS * _HD), torch.float8_e4m3fn), alloc((LF, C_ // 128, _HS * _HD // 128), torch.bfloat16)
        w["fqn"], w["fkn"] = alloc((LF, _HD), torch.bfloat16), alloc((LF, _HD), torch.bfloat16)
        w["mgate_w"], w["mgate_s"] = alloc((L, F_, C_), torch.float8_e4m3fn), alloc((L, F_ // 128, C_ // 128), torch.bfloat16)
        w["mup_w"], w["mup_s"] = alloc((L, F_, C_), torch.float8_e4m3fn), alloc((L, F_ // 128, C_ // 128), torch.bfloat16)
        w["mdown_w"], w["mdown_s"] = alloc((L, C_, F_), torch.float8_e4m3fn), alloc((L, C_ // 128, F_ // 128), torch.bfloat16)

        # fill layer-uniform classes from the per-layer shards
        for l, path in enumerate(shards):
            sd = load_file(path)
            pre = f"model.language_model.layers.{l}."
            g = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            is_full = (l + 1) % 4 == 0
            gi, fi = l - (l + 1) // 4, (l + 1) // 4 - 1
            w["ln1"][l].copy_(g["input_layernorm.weight"])
            w["ln2"][l].copy_(g["post_attention_layernorm.weight"])
            if is_full:
                att = "self_attn."
                w["fq_w"][fi].copy_(g[att + "q_proj.weight"])
                w["fq_s"][fi].copy_(g[att + "q_proj.weight_scale_inv"])
                w["fk_w"][fi].copy_(g[att + "k_proj.weight"])
                w["fk_s"][fi].copy_(g[att + "k_proj.weight_scale_inv"])
                w["fv_w"][fi].copy_(g[att + "v_proj.weight"])
                w["fv_s"][fi].copy_(g[att + "v_proj.weight_scale_inv"])
                w["fo_w"][fi].copy_(g[att + "o_proj.weight"])
                w["fo_s"][fi].copy_(g[att + "o_proj.weight_scale_inv"])
                w["fqn"][fi].copy_(g[att + "q_norm.weight"])
                w["fkn"][fi].copy_(g[att + "k_norm.weight"])
            else:
                att = "linear_attn."
                w["gqkv_w"][gi].copy_(g[att + "in_proj_qkv.weight"])
                w["gqkv_s"][gi].copy_(g[att + "in_proj_qkv.weight_scale_inv"])
                w["gz_w"][gi].copy_(g[att + "in_proj_z.weight"])
                w["gz_s"][gi].copy_(g[att + "in_proj_z.weight_scale_inv"])
                w["ga"][gi].copy_(g[att + "in_proj_a.weight"])
                w["gb"][gi].copy_(g[att + "in_proj_b.weight"])
                w["gout_w"][gi].copy_(g[att + "out_proj.weight"])
                w["gout_s"][gi].copy_(g[att + "out_proj.weight_scale_inv"])
                w["gconv"][gi].copy_(g[att + "conv1d.weight"].squeeze(1))
                w["galog"][gi].copy_(g[att + "A_log"].float())
                w["gdtb"][gi].copy_(g[att + "dt_bias"].float())
                w["gnormw"][gi].copy_(g[att + "norm.weight"])
            w["mgate_w"][l].copy_(g["mlp.gate_proj.weight"])
            w["mgate_s"][l].copy_(g["mlp.gate_proj.weight_scale_inv"])
            w["mup_w"][l].copy_(g["mlp.up_proj.weight"])
            w["mup_s"][l].copy_(g["mlp.up_proj.weight_scale_inv"])
            w["mdown_w"][l].copy_(g["mlp.down_proj.weight"])
            w["mdown_s"][l].copy_(g["mlp.down_proj.weight_scale_inv"])
            del sd, g

        od = load_file(outside)
        self.emb = od["model.language_model.embed_tokens.weight"].to(dev)
        self.lmh = od["lm_head.weight"].to(dev)
        self.finaln = od["model.language_model.norm.weight"].to(dev)
        self.w = w

        # persistent state
        self.ssm = torch.zeros(LG, nv, hv, hk, device=dev, dtype=torch.float32)
        self.conv = torch.zeros(LG, 3, 2 * nk * hk + nv * hv, device=dev, dtype=torch.float32)
        self.k_cache = torch.zeros(LF, capacity, _HNKV, _HD, device=dev, dtype=torch.float32)
        self.v_cache = torch.zeros(LF, capacity, _HNKV, _HD, device=dev, dtype=torch.float32)
        inv = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
        t = torch.arange(capacity, dtype=torch.float64)
        freqs = torch.outer(t, inv)
        emb_t = torch.cat([freqs, freqs], -1)
        self.cos = emb_t.cos().float().to(dev)
        self.sin = emb_t.sin().float().to(dev)

        self.ws = torch.zeros(WS_TOTAL, device=dev, dtype=torch.float32)
        self.logits = torch.zeros(248320, device=dev, dtype=torch.float32)
        self.bar = torch.zeros(1, device=dev, dtype=torch.int64)
        self._bar_base = 0

    def run(self, token: int, position: int, *, check_counter: bool = False):
        """One 27B decode step = one kernel launch (B=1)."""
        if not (0 <= position < self.capacity):
            raise ValueError(f"position {position} out of [0, {self.capacity})")
        w, d = self.w, self.dims
        qwen35_hybrid_megakernel[(self.workers,)](
            int(token),
            int(position),
            int(position),  # identity slot table (kvaas wiring: follow-up)
            self.bar,
            self._bar_base,
            64,
            w["ln1"], w["ln2"], self.finaln, self.emb, self.lmh,
            w["gqkv_w"], w["gqkv_s"], w["gz_w"], w["gz_s"], w["ga"], w["gb"],
            w["gout_w"], w["gout_s"], w["gconv"], w["galog"], w["gdtb"], w["gnormw"],
            w["fq_w"], w["fq_s"], w["fk_w"], w["fk_s"], w["fv_w"], w["fv_s"], w["fo_w"], w["fo_s"], w["fqn"], w["fkn"],
            w["mgate_w"], w["mgate_s"], w["mup_w"], w["mup_s"], w["mdown_w"], w["mdown_s"],
            self.ssm, self.conv, self.k_cache, self.v_cache, self.cos, self.sin,
            self.ws, self.logits,
            C=d["C"], F=d["F"], V=248320, SCAP=self.capacity,
            H=d["H"], NKV=d["NKV"], D=d["D"], ROT=d["ROT"],
            NKH=d["NKH"], NVH=d["NVH"], HK=d["HK"], HV=d["HV"],
            GQKV_N=2 * 16 * 128 + 48 * 128, GZ_N=48 * 128,
            FQ_N=24 * 2 * 256, FK_N=4 * 256, FV_N=4 * 256,
            EPS=1e-6, FSCALE=256**-0.5, GSCALE=128**-0.5,
            BT=64, BTR=min(1024, triton.next_power_of_2(self.capacity)),
            O_HIDDEN_A=_O["hidden_a"], O_HIDDEN_B=_O["hidden_b"], O_RMS1=_O["rms1"],
            O_PROJ=_O["proj"], O_ZAB=_O["zab"], O_POST=_O["post"], O_HEADS=_O["heads"],
            O_ATTN=_O["attn"], O_RMS2=_O["rms2"], O_GU=_O["gu"], O_ACT=_O["act"],
            O_SCORES=_O["scores"], O_PROBS=_O["probs"], KILL_AFTER=getattr(self, "kill_after", -1),
            num_warps=8,
        )
        ka = getattr(self, "kill_after", -1)
        if ka >= 0:
            executed = 1 + sum(14 if (l + 1) % 4 == 0 else 11 for l in range(ka))
        else:
            executed = self.NUM_BARRIERS
        self._bar_base += executed * self.workers
        if check_counter:
            torch.cuda.synchronize()
            got = int(self.bar[0])
            if got != self._bar_base:
                raise RuntimeError(f"barrier counter {got} != {self._bar_base}")
        return self.logits
