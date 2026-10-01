"""Recurrent Triton device task bodies."""
from __future__ import annotations
import triton
import triton.language as tl

@triton.jit
def _t_gdn_conv(
    worker: tl.int32,
    P: tl.int32,
    state_ptr,
    w_ptr,
    x_ptr,
    out_ptr,
    C: tl.constexpr,
    ELEM: tl.constexpr,
    KTAPS: tl.constexpr,
):
    """GDN short-conv decode step over channel tiles.

    ``state`` is the persistent [KTAPS-1, C] fp32 channel state (time-major);
    ``x`` is this token's mixed qkv row [C]; ``w`` is the FIR [C, KTAPS]
    (grouped conv weights, one tap vector per channel). Computes
    ``out = silu(sum_j w[:, j] * in_j)`` and shifts the state. One task per
    ELEM-channel tile.
    """
    NTASK: tl.constexpr = C // ELEM
    task = worker
    while task < NTASK:
        offs = task * ELEM + tl.arange(0, ELEM)
        acc = tl.zeros([ELEM], tl.float32)
        for j in tl.static_range(KTAPS - 1):
            wj = tl.load(w_ptr + offs * KTAPS + j).to(tl.float32)
            sj = tl.load(state_ptr + j * C + offs, cache_modifier=".cg")
            acc += wj * sj
        wj = tl.load(w_ptr + offs * KTAPS + (KTAPS - 1)).to(tl.float32)
        xn = tl.load(x_ptr + offs, cache_modifier=".cg").to(tl.float32)
        acc += wj * xn
        tl.store(out_ptr + offs, acc / (1.0 + tl.exp(-acc)))
        # state shift: drop the oldest tap, append the new row
        for j in tl.static_range(KTAPS - 2):
            sj1 = tl.load(state_ptr + (j + 1) * C + offs, cache_modifier=".cg")
            tl.store(state_ptr + j * C + offs, sj1)
        tl.store(state_ptr + (KTAPS - 2) * C + offs, xn)
        task += P


@triton.jit
def _t_gdn_conv_tiled(
    worker: tl.int32,
    P: tl.int32,
    state_ptr,
    w_ptr,
    x_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    ELEM: tl.constexpr,
    KTAPS: tl.constexpr,
):
    """Generic gdn_conv decode-step task body (issue #89): one task per
    (batch, ELEM-channel tile) over the batched persistent state pool
    [B, KTAPS-1, C] (fp32, time-major), the mixed qkv rows [B, C] and the
    FIR weights [C, KTAPS]. Same arithmetic as the 27B-validated
    ``_t_gdn_conv`` (which is a single flattened batch row of this
    template), generalized to per-task (b, tile) addressing. Requires
    C % ELEM == 0 (the lowering picks an exact tiling).
    """
    NTILE: tl.constexpr = C // ELEM
    task = worker
    while task < B * NTILE:
        b = task // NTILE
        t = task % NTILE
        offs = t * ELEM + tl.arange(0, ELEM)
        sbase = state_ptr + b.to(tl.int64) * ((KTAPS - 1) * C)
        acc = tl.zeros([ELEM], tl.float32)
        for j in tl.static_range(KTAPS - 1):
            wj = tl.load(w_ptr + offs * KTAPS + j).to(tl.float32)
            sj = tl.load(sbase + j * C + offs, cache_modifier=".cg")
            acc += wj * sj
        wj = tl.load(w_ptr + offs * KTAPS + (KTAPS - 1)).to(tl.float32)
        xn = tl.load(x_ptr + b.to(tl.int64) * C + offs, cache_modifier=".cg").to(tl.float32)
        acc += wj * xn
        tl.store(out_ptr + b.to(tl.int64) * C + offs, acc / (1.0 + tl.exp(-acc)))
        # state shift: drop the oldest tap, append the new row
        for j in tl.static_range(KTAPS - 2):
            sj1 = tl.load(sbase + (j + 1) * C + offs, cache_modifier=".cg")
            tl.store(sbase + j * C + offs, sj1)
        tl.store(sbase + (KTAPS - 2) * C + offs, xn)
        task += P


@triton.jit
def _t_gdn_heads(
    worker: tl.int32,
    P: tl.int32,
    q_ptr,
    k_ptr,
    v_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    alog_ptr,
    dtb_ptr,
    normw_ptr,
    state_ptr,
    out_ptr,
    NH: tl.constexpr,
    NK: tl.constexpr,
    HV: tl.constexpr,
    HK: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
):
    """Per-value-head gated delta rule (decode step), matching the repo's
    ``_gdn_delta_rule_recurrent`` reference exactly:

        s *= exp(g);  s += beta*(v - s.k) outer k;  o = s.q
        o <- RMSNorm(o) * norm_w * (z * sigmoid(z))

    with g = -exp(A_log)*softplus(a + dt_bias), beta = sigmoid(b), q/k
    L2-normalized per *key* head (group-expanded), fp32 state [NH, HV, HK].
    One task per value head; the [HV, HK] state slice stays in registers.
    """
    GROUP: tl.constexpr = NH // NK
    offs_v = tl.arange(0, HV)
    offs_k = tl.arange(0, HK)
    task = worker
    while task < NH:
        h = task
        kh = h // GROUP
        q = tl.load(q_ptr + kh * HK + offs_k, cache_modifier=".cg").to(tl.float32)
        k = tl.load(k_ptr + kh * HK + offs_k, cache_modifier=".cg").to(tl.float32)
        v = tl.load(v_ptr + h * HV + offs_v, cache_modifier=".cg").to(tl.float32)
        z = tl.load(z_ptr + h * HV + offs_v, cache_modifier=".cg").to(tl.float32)
        # per-head scalars
        a_ = tl.load(a_ptr + h).to(tl.float32)
        b_ = tl.load(b_ptr + h).to(tl.float32)
        A = tl.exp(tl.load(alog_ptr + h).to(tl.float32))
        x_dt = a_ + tl.load(dtb_ptr + h).to(tl.float32)
        sp = tl.where(x_dt <= 20.0, tl.log(1.0 + tl.exp(x_dt)), x_dt)
        beta = 1.0 / (1.0 + tl.exp(-b_))
        # per-key-head normalization (computed redundantly per value head)
        qn = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)) * scale
        kn = k * (1.0 / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6))
        # state update over this head's [HV, HK] slice
        sbase = state_ptr + h * HV * HK
        s = tl.load(sbase + offs_v[:, None] * HK + offs_k[None, :], cache_modifier=".cg")
        s = s * tl.exp(-A * sp)
        sk = tl.sum(s * kn[None, :], axis=1)
        vd = beta * (v - sk)
        s = s + vd[:, None] * kn[None, :]
        o = tl.sum(s * qn[None, :], axis=1)
        tl.store(sbase + offs_v[:, None] * HK + offs_k[None, :], s)
        # per-head RMSNorm over hv + z gate
        var = tl.sum(o * o, axis=0) / HV
        nw = tl.load(normw_ptr + offs_v).to(tl.float32)
        on = o * (1.0 / tl.sqrt(var + eps)) * nw
        og = on * (z * (1.0 / (1.0 + tl.exp(-z))))
        tl.store(out_ptr + h * HV + offs_v, og)
        task += P


@triton.jit
def _t_mhc_pre(
    worker: tl.int32,
    P: tl.int32,
    streams_ptr,
    fn_ptr,
    base_ptr,
    scale_ptr,
    hin_ptr,
    post_ptr,
    comb_ptr,
    B: tl.constexpr,
    HC: tl.constexpr,
    C: tl.constexpr,
    MIX: tl.constexpr,  # (2 + HC) * HC
    EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    ITERS: tl.constexpr,
    MIXP: tl.constexpr,  # pow2 pad of MIX
    HCP: tl.constexpr,  # pow2 pad of HC
    BLOCK_K: tl.constexpr,
):
    """mhc_pre task body (issue #99): one task per batch row over the
    [B, HC, C] stream stack. fp32 throughout: unweighted RMSNorm over the
    flattened hc·C row, the folded [MIX, hc·C] fn GEMV (one K-reduction
    per mix row), sigmoid pre/post gates, softmax + Sinkhorn-Knopp
    alternate row/col normalization (eps inside every denominator, floe
    DeepseekV4HyperConnection.forward verbatim) and the pre-weighted
    stream collapse."""
    HCK: tl.constexpr = HC * C
    offs_m = tl.arange(0, MIXP)
    offs_h = tl.arange(0, HCP)
    offs_k = tl.arange(0, HCP)
    m_mask = offs_m < MIX
    h_mask = offs_h < HC
    kj_mask = (offs_k[:, None] < HC) & (offs_h[None, :] < HC)
    task = worker
    while task < B:
        b = task.to(tl.int64)
        # pass 1: sqrsum over the flattened [HC, C] row (unweighted RMSNorm)
        ss = 0.0
        for k0 in range(0, HCK, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(streams_ptr + b * HCK + offs, mask=offs < HCK, other=0.0).to(tl.float32)
            ss += tl.sum(v * v, axis=0)
        rstd = 1.0 / tl.sqrt(ss / HCK + RMS_EPS)
        # pass 2: the folded fn projection — logits[m] = <fn[m, :], flat>
        # with flat = streams * rstd. NO projection bias: floe's F.linear
        # carries none; base enters only inside the gates below.
        logits = tl.zeros([MIXP], dtype=tl.float32)
        for k0 in range(0, HCK, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            kmask = offs < HCK
            flat = tl.load(streams_ptr + b * HCK + offs, mask=kmask, other=0.0).to(tl.float32) * rstd
            frows = tl.load(fn_ptr + offs_m[:, None] * HCK + offs[None, :],
                            mask=m_mask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
            logits += tl.sum(frows * flat[None, :], axis=1)
        # split [MIX] -> pre_w [HC] | post_w [HC] | comb_w [HC, HC]
        pre_w = tl.sum(tl.where((offs_m[:, None] == offs_h[None, :]) & h_mask[None, :],
                                logits[:, None], 0.0), axis=0)
        post_w = tl.sum(tl.where((offs_m[:, None] == HC + offs_h[None, :]) & h_mask[None, :],
                                 logits[:, None], 0.0), axis=0)
        comb_rows = 2 * HC + offs_k[:, None] * HC + offs_h[None, :]
        comb_w = tl.sum(tl.where(offs_m[:, None, None] == comb_rows[None, :, :],
                                 logits[:, None, None], 0.0), axis=0)
        pre_s = tl.load(scale_ptr).to(tl.float32)
        post_s = tl.load(scale_ptr + 1).to(tl.float32)
        comb_s = tl.load(scale_ptr + 2).to(tl.float32)
        pre = 1.0 / (1.0 + tl.exp(-(pre_w * pre_s + tl.load(base_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)))) + EPS
        post = 2.0 / (1.0 + tl.exp(-(post_w * post_s + tl.load(base_ptr + HC + offs_h, mask=h_mask, other=0.0).to(tl.float32))))
        comb_b = tl.load(base_ptr + 2 * HC + offs_k[:, None] * HC + offs_h[None, :],
                         mask=kj_mask, other=0.0).to(tl.float32)
        # softmax over j (rows), masked to the valid hc×hc block
        cl = comb_w * comb_s + comb_b
        cl = tl.where(kj_mask, cl, float("-inf"))
        cm = tl.max(cl, axis=1)
        ce = tl.exp(cl - cm[:, None])
        ce = tl.where(kj_mask, ce, 0.0)
        # Padded rows (k or j >= HC) must stay EXACTLY 0 through the whole
        # Sinkhorn recursion: an unmasked 0/0 here becomes NaN and the
        # unmasked column sums below poison the entire valid block; even
        # without the NaN, EPS-floored padding inflates the column
        # denominators (~17-20% at hc=3, HCP=4). All denominators therefore
        # sum only the valid hc×hc block, matching the reference recursion
        # on the exact matrix.
        row_den = tl.sum(ce, axis=1)[:, None]
        comb = tl.where(kj_mask, ce / row_den + EPS, 0.0)
        # Sinkhorn-Knopp: initial column normalization, then (ITERS−1)
        # alternate row/col passes — eps inside every denominator (floe).
        comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
        for _ in tl.static_range(ITERS - 1):
            comb = comb / (tl.sum(comb, axis=1)[:, None] + EPS)
            comb = comb / (tl.sum(comb, axis=0)[None, :] + EPS)
        comb = tl.where(kj_mask, comb, 0.0)
        tl.store(post_ptr + b * HC + offs_h, post, mask=h_mask)
        tl.store(comb_ptr + b * HC * HC + offs_k[:, None] * HC + offs_h[None, :], comb, mask=kj_mask)
        # stream collapse: h_in[c] = Σ_h pre[h] · streams[b, h, c]
        for c0 in range(0, C, BLOCK_K):
            offs_c = c0 + tl.arange(0, BLOCK_K)
            cmask = offs_c < C
            acc = tl.zeros([BLOCK_K], dtype=tl.float32)
            for h in tl.static_range(HC):
                ph = tl.sum(tl.where(offs_h == h, pre, 0.0), axis=0)
                sv = tl.load(streams_ptr + b * HCK + h * C + offs_c, mask=cmask, other=0.0).to(tl.float32)
                acc += ph * sv
            tl.store(hin_ptr + b * C + offs_c, acc, mask=cmask)
        task += P


@triton.jit
def _t_mhc_post(
    worker: tl.int32,
    P: tl.int32,
    streams_ptr,
    body_out_ptr,
    post_ptr,
    comb_ptr,
    out_ptr,
    B: tl.constexpr,
    HC: tl.constexpr,
    C: tl.constexpr,
    HCP: tl.constexpr,  # pow2 pad of HC
    BLOCK_C: tl.constexpr,
):
    """mhc_post task body (issue #99): one task per (batch, stream j);
    streams'[j] = post[j]·body_out + Σ_k comb[k, j]·streams[k] (floe
    _mhc_compose), fp32 compose arithmetic, one output stream row per task."""
    offs_k = tl.arange(0, HCP)
    k_mask = offs_k < HC
    task = worker
    while task < B * HC:
        b = (task // HC).to(tl.int64)
        j = task % HC
        pj = tl.load(post_ptr + b * HC + j).to(tl.float32)
        # comb column j: comb[k, j] weights source stream k
        ck = tl.load(comb_ptr + b * HC * HC + offs_k * HC + j, mask=k_mask, other=0.0).to(tl.float32)
        for c0 in range(0, C, BLOCK_C):
            offs_c = c0 + tl.arange(0, BLOCK_C)
            cmask = offs_c < C
            acc = tl.zeros([BLOCK_C], dtype=tl.float32)
            for k in tl.static_range(HC):
                w = tl.sum(tl.where(offs_k == k, ck, 0.0), axis=0)
                sv = tl.load(streams_ptr + b * HC * C + k * C + offs_c, mask=cmask, other=0.0).to(tl.float32)
                acc += w * sv
            bo = tl.load(body_out_ptr + b * C + offs_c, mask=cmask, other=0.0).to(tl.float32)
            acc = pj * bo + acc
            tl.store(out_ptr + b * HC * C + j * C + offs_c, acc, mask=cmask)
        task += P


@triton.jit
def _t_gdn_heads_batched(
    worker: tl.int32,
    P: tl.int32,
    q_ptr,
    k_ptr,
    v_ptr,
    z_ptr,
    a_ptr,
    b_ptr,
    alog_ptr,
    dtb_ptr,
    normw_ptr,
    state_ptr,
    out_ptr,
    B: tl.constexpr,
    NH: tl.constexpr,
    NK: tl.constexpr,
    HV: tl.constexpr,
    HK: tl.constexpr,
    eps: tl.constexpr,
    scale: tl.constexpr,
):
    """Batched gdn_delta decode-step task body (issue #90): one task per
    (batch, value head) over the batched persistent state pool
    [B, NH, HV, HK] (fp32, read-modify-write). Arithmetic identical to the
    27B-validated ``_t_gdn_heads`` (which is the B=1, task==head flattening
    of this template), generalized to per-task (b, head) addressing:
    A_log/dt_bias/norm_w are per-layer params broadcast over the batch.
    """
    GROUP: tl.constexpr = NH // NK
    offs_v = tl.arange(0, HV)
    offs_k = tl.arange(0, HK)
    task = worker
    while task < B * NH:
        bb = task // NH
        h = task % NH
        kh = h // GROUP
        q = tl.load(q_ptr + bb.to(tl.int64) * (NK * HK) + kh * HK + offs_k, cache_modifier=".cg").to(tl.float32)
        k = tl.load(k_ptr + bb.to(tl.int64) * (NK * HK) + kh * HK + offs_k, cache_modifier=".cg").to(tl.float32)
        v = tl.load(v_ptr + bb.to(tl.int64) * (NH * HV) + h * HV + offs_v, cache_modifier=".cg").to(tl.float32)
        z = tl.load(z_ptr + bb.to(tl.int64) * (NH * HV) + h * HV + offs_v, cache_modifier=".cg").to(tl.float32)
        # per-head scalars
        a_ = tl.load(a_ptr + bb.to(tl.int64) * NH + h).to(tl.float32)
        b_ = tl.load(b_ptr + bb.to(tl.int64) * NH + h).to(tl.float32)
        A = tl.exp(tl.load(alog_ptr + h).to(tl.float32))
        x_dt = a_ + tl.load(dtb_ptr + h).to(tl.float32)
        sp = tl.where(x_dt <= 20.0, tl.log(1.0 + tl.exp(x_dt)), x_dt)
        beta = 1.0 / (1.0 + tl.exp(-b_))
        # per-key-head normalization (computed redundantly per value head)
        qn = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)) * scale
        kn = k * (1.0 / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6))
        # state update over this head's [HV, HK] slice
        sbase = state_ptr + bb.to(tl.int64) * (NH * HV * HK) + h * HV * HK
        s = tl.load(sbase + offs_v[:, None] * HK + offs_k[None, :], cache_modifier=".cg")
        s = s * tl.exp(-A * sp)
        sk = tl.sum(s * kn[None, :], axis=1)
        vd = beta * (v - sk)
        s = s + vd[:, None] * kn[None, :]
        o = tl.sum(s * qn[None, :], axis=1)
        tl.store(sbase + offs_v[:, None] * HK + offs_k[None, :], s)
        # per-head RMSNorm over hv + z gate
        var = tl.sum(o * o, axis=0) / HV
        nw = tl.load(normw_ptr + offs_v).to(tl.float32)
        on = o * (1.0 / tl.sqrt(var + eps)) * nw
        og = on * (z * (1.0 / (1.0 + tl.exp(-z))))
        tl.store(out_ptr + bb.to(tl.int64) * (NH * HV) + h * HV + offs_v, og)
        task += P


@triton.jit
def _t_kda_heads_batched(
    worker: tl.int32,
    P: tl.int32,
    q_ptr,
    k_ptr,
    v_ptr,
    f_ptr,
    b_ptr,
    dtb_ptr,
    alog_ptr,
    state_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    scale: tl.constexpr,
    lower_bound: tl.constexpr,  # float; NaN sentinel selects the softplus branch
):
    """Batched KDA gated delta-rule decode-step task body (issue #101):
    one task per (batch, head) over the persistent fp32 state pool
    [B, H, K, V] (read-modify-write). GLM-5.3 ``Glm53LinearAttention``
    seq==1 arithmetic, ported from floe's eager ``Glm53ForgetGate`` +
    ``_kda_recurrent`` (the fused kda_decode Triton path no longer ships in
    floe — kernels live in vkernels):

        g    = lower_bound * sigmoid(exp(A_log[h]) * (f + dt_bias))  # [K]
           (lower_bound NaN sentinel -> g = -exp(A_log)*softplus(f + dt_bias))
        s   *= exp(g)[:, None]                     # element-wise row decay
        kv   = sum_k s * k_n
        s   += k_n outer (sigmoid(b) * (v - kv))
        o    = sum_k s * q_n

    q/k L2-normalized per head (eps 1e-6 inside the sqrt); q carries the
    1/sqrt(K) scale. Plain readout — the gated norm is the separate
    rms_norm_gated op. Unlike _t_gdn_heads_batched the decay is
    element-wise per k-row (broadcast over V), and there is no group
    expansion (KDA: one q/k/v head each).
    """
    offs_k = tl.arange(0, K)
    offs_v = tl.arange(0, V)
    task = worker
    while task < B * H:
        bb = task // H
        h = task % H
        q = tl.load(q_ptr + bb.to(tl.int64) * (H * K) + h * K + offs_k, cache_modifier=".cg").to(tl.float32)
        k = tl.load(k_ptr + bb.to(tl.int64) * (H * K) + h * K + offs_k, cache_modifier=".cg").to(tl.float32)
        v = tl.load(v_ptr + bb.to(tl.int64) * (H * V) + h * V + offs_v, cache_modifier=".cg").to(tl.float32)
        f = tl.load(f_ptr + bb.to(tl.int64) * (H * K) + h * K + offs_k, cache_modifier=".cg").to(tl.float32)
        # gate conditioning: per-(head, k-dim) log gate, folded in-task
        A = tl.exp(tl.load(alog_ptr + h).to(tl.float32))
        x_dt = f + tl.load(dtb_ptr + h * K + offs_k).to(tl.float32)
        if lower_bound == lower_bound:  # NaN sentinel: finite -> lower_bound branch
            g = lower_bound * (1.0 / (1.0 + tl.exp(-A * x_dt)))
        else:
            sp = tl.where(x_dt <= 20.0, tl.log(1.0 + tl.exp(x_dt)), x_dt)
            g = -A * sp
        beta = 1.0 / (1.0 + tl.exp(-tl.load(b_ptr + bb.to(tl.int64) * H + h).to(tl.float32)))
        # L2 conditioning (floe _l2norm)
        qn = q * (1.0 / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)) * scale
        kn = k * (1.0 / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6))
        # state update over this head's [K, V] slice — element-wise decay
        sbase = state_ptr + bb.to(tl.int64) * (H * K * V) + h * K * V
        s = tl.load(sbase + offs_k[:, None] * V + offs_v[None, :], cache_modifier=".cg")
        s = s * tl.exp(g)[:, None]
        kv = tl.sum(s * kn[:, None], axis=0)
        s = s + kn[:, None] * (beta * (v - kv))[None, :]
        tl.store(sbase + offs_k[:, None] * V + offs_v[None, :], s)
        # plain readout (gated norm is the separate rms_norm_gated op)
        o = tl.sum(s * qn[:, None], axis=0)
        tl.store(out_ptr + bb.to(tl.int64) * (H * V) + h * V + offs_v, o)
        task += P

