"""Attention Triton device task bodies."""
from __future__ import annotations
import triton
import triton.language as tl

@triton.jit
def _t_rope(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    NHEAD: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    TSTRIDE: tl.constexpr,
):
    """RoPE at row b's runtime position: one task per (b, head).

    Ported from the 27B-validated ``_h_rope_append`` (device_triton_hybrid):
    fp32 loads from the (bf16) workspace, NeoX split-half over the first
    ``ROT`` dims — ``x1' = x1*c - x2*s ; x2' = x2*c + x1*s`` with
    ``half = ROT // 2`` — and dims ``[ROT, D)`` pass through unchanged.

    ``TSTRIDE`` is the cos/sin table row stride (fp32 tables indexed at the
    per-row runtime position). The Qwen3 call passes ``ROT=D, TSTRIDE=D``:
    with the full-width cat([f, f]) tables this reduces exactly to the
    full-width rotate-half form.
    """
    task = worker
    while task < B * NHEAD:
        b = task // NHEAD
        h = task % NHEAD
        half: tl.constexpr = ROT // 2
        d = tl.arange(0, half)
        p = tl.load(pos_ptr + b).to(tl.int64)
        base = (b * NHEAD + h) * D
        x1 = tl.load(x_ptr + base + d, cache_modifier=".cg").to(tl.float32)
        x2 = tl.load(x_ptr + base + half + d, cache_modifier=".cg").to(tl.float32)
        c = tl.load(cos_ptr + p * TSTRIDE + d).to(tl.float32)
        s = tl.load(sin_ptr + p * TSTRIDE + d).to(tl.float32)
        tl.store(y_ptr + base + d, x1 * c - x2 * s)
        tl.store(y_ptr + base + half + d, x2 * c + x1 * s)
        if ROT < D:  # pass-through tail (empty for the full-width Qwen3 form)
            offs_d = tl.arange(0, D)
            mt = offs_d >= ROT
            tail = tl.load(x_ptr + base + offs_d, mask=mt, other=0.0, cache_modifier=".cg").to(tl.float32)
            tl.store(y_ptr + base + offs_d, tail, mask=mt)
        task += P


@triton.jit
def _t_append(
    worker: tl.int32,
    P: tl.int32,
    k_ptr,
    v_ptr,
    table_ptr,
    pos_ptr,
    k_new_ptr,
    v_new_ptr,
    B: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    QKVW: tl.constexpr,
    SCAP: tl.constexpr,
):
    """K/V append into slot table[b, pos[b]]: one task per (b, kv-head).

    The new k rows come from the packed roped-k buffer [B, KVH, D]; the v
    rows are read from the packed QKV projection (v at row offset
    HD + KVH*D). The cache destination is token-slot-major
    ``slot * KVH * D + kvh * D``.
    """
    task = worker
    while task < B * KVH:
        b = task // KVH
        kvh = task % KVH
        offs = tl.arange(0, D)
        p = tl.load(pos_ptr + b).to(tl.int64)
        slot = tl.load(table_ptr + b * SCAP + p)
        dst = slot * KVH * D + kvh * D
        kn = tl.load(k_new_ptr + (b * KVH + kvh) * D + offs, cache_modifier=".cg").to(tl.float32)
        vn = tl.load(v_new_ptr + b * QKVW + kvh * D + offs, cache_modifier=".cg").to(tl.float32)
        tl.store(k_ptr + dst + offs, kn.to(k_ptr.dtype.element_ty))
        tl.store(v_ptr + dst + offs, vn.to(v_ptr.dtype.element_ty))
        task += P


@triton.jit
def _t_scores(
    worker: tl.int32,
    P: tl.int32,
    q_ptr,
    k_ptr,
    table_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    SCAP: tl.constexpr,
    BT: tl.constexpr,
    scale: tl.constexpr,
):
    """scores[b, h, t] = scale * <q_bh, K[table[b, t]]>, t in [0, pos[b]]."""
    task = worker
    GROUP: tl.constexpr = H // KVH
    while task < B * H:
        b = task // H
        h = task % H
        kvh = h // GROUP
        offs_d = tl.arange(0, D)
        q = tl.load(q_ptr + (b * H + h) * D + offs_d, cache_modifier=".cg").to(tl.float32)
        p1 = tl.load(pos_ptr + b).to(tl.int64) + 1
        kbase = k_ptr + kvh * D  # head offset inside each token row
        trow = table_ptr + b * SCAP
        yrow = y_ptr + (b * H + h) * SCAP
        t0 = tl.zeros((), tl.int64)
        while t0 < p1:
            offs_t = t0 + tl.arange(0, BT)
            m = offs_t < p1
            slots = tl.load(trow + offs_t, mask=m, other=0)
            kt = tl.load(kbase + slots[:, None] * (KVH * D) + offs_d[None, :], mask=m[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
            s = tl.sum(kt * q[None, :], axis=1) * scale
            tl.store(yrow + offs_t, s, mask=m)
            t0 += BT
        task += P


@triton.jit
def _t_softmax(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    SCAP: tl.constexpr,
    BTR: tl.constexpr,
):
    """Row softmax over the valid prefix (per-row length); tail written 0.

    Loop-free on purpose: a scalar carried out of a runtime-bound loop and
    reused in later loops miscompiles inside the task loop on this
    Triton/GB10 target (verified with a minimal repro), so the whole row is
    processed as one masked block. BTR = next_pow2(SCAP) <= 1024.
    """
    task = worker
    while task < B * H:
        b = task // H
        h = task % H
        row = (b * H + h) * SCAP
        offs_t = tl.arange(0, BTR)
        m = offs_t < (tl.load(pos_ptr + b).to(tl.int64) + 1)
        s = tl.load(x_ptr + row + offs_t, mask=m, other=-float("inf"), cache_modifier=".cg")
        m_run = tl.max(s, axis=0)
        e = tl.exp(s - m_run)  # masked lanes: exp(-inf) = 0
        inv = 1.0 / tl.sum(e, axis=0)
        tl.store(y_ptr + row + offs_t, tl.where(m, e * inv, 0.0), mask=offs_t < SCAP)
        task += P


@triton.jit
def _t_values(
    worker: tl.int32,
    P: tl.int32,
    probs_ptr,
    v_ptr,
    table_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    SCAP: tl.constexpr,
    BT: tl.constexpr,
):
    """ctx[b, h, :] = sum_{t<=pos[b]} probs * V[table[b, t]] (masked)."""
    task = worker
    GROUP: tl.constexpr = H // KVH
    while task < B * H:
        b = task // H
        h = task % H
        kvh = h // GROUP
        offs_d = tl.arange(0, D)
        acc = tl.zeros([D], tl.float32)
        p1 = tl.load(pos_ptr + b).to(tl.int64) + 1
        vbase = v_ptr + kvh * D
        trow = table_ptr + b * SCAP
        prow = probs_ptr + (b * H + h) * SCAP
        t0 = tl.zeros((), tl.int64)
        while t0 < p1:
            offs_t = t0 + tl.arange(0, BT)
            m = offs_t < p1
            slots = tl.load(trow + offs_t, mask=m, other=0)
            pv = tl.load(prow + offs_t, mask=m, other=0.0, cache_modifier=".cg")
            vv = tl.load(vbase + slots[:, None] * (KVH * D) + offs_d[None, :], mask=m[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
            acc += tl.sum(pv[:, None] * vv, axis=0)
            t0 += BT
        tl.store(y_ptr + (b * H + h) * D + offs_d, acc)
        task += P


@triton.jit
def _t_values_gated(
    worker: tl.int32,
    P: tl.int32,
    probs_ptr,
    v_ptr,
    gate_ptr,
    table_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    SCAP: tl.constexpr,
    BT: tl.constexpr,
):
    """Issue #92: _t_values + per-head sigmoid output gate fused —

    ``y[b,h,:] = (sum_{t<=pos[b]} probs * V[table[b,t]]) * sigmoid(gate[b,h,:])``

    Gate epilogue is the validated ``_h_values_gate`` one
    (device_triton_hybrid.py, 27B): gate loaded once per head, sigmoid and
    multiply in f32, single bf16 store — no extra grid barrier per FA layer.
    """
    task = worker
    GROUP: tl.constexpr = H // KVH
    while task < B * H:
        b = task // H
        h = task % H
        kvh = h // GROUP
        offs_d = tl.arange(0, D)
        acc = tl.zeros([D], tl.float32)
        p1 = tl.load(pos_ptr + b).to(tl.int64) + 1
        vbase = v_ptr + kvh * D
        trow = table_ptr + b * SCAP
        prow = probs_ptr + (b * H + h) * SCAP
        t0 = tl.zeros((), tl.int64)
        while t0 < p1:
            offs_t = t0 + tl.arange(0, BT)
            m = offs_t < p1
            slots = tl.load(trow + offs_t, mask=m, other=0)
            pv = tl.load(prow + offs_t, mask=m, other=0.0, cache_modifier=".cg")
            vv = tl.load(vbase + slots[:, None] * (KVH * D) + offs_d[None, :], mask=m[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
            acc += tl.sum(pv[:, None] * vv, axis=0)
            t0 += BT
        gate = tl.load(gate_ptr + (b * H + h) * D + offs_d, cache_modifier=".cg").to(tl.float32)
        tl.store(y_ptr + (b * H + h) * D + offs_d, acc * (1.0 / (1.0 + tl.exp(-gate))))
        task += P


@triton.jit
def _t_rope_interleaved(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    NHEAD: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    TSTRIDE: tl.constexpr,
    NEGATE_SIN: tl.constexpr,
):
    """Interleaved (GPT-J style) rope — issue #95, DeepSeek-V4 q/latent-k.

    One task per (b, head); pairs ``(2i, 2i+1)`` over the first ``ROT``
    dims, cos/sin indexed by PAIR index at the row's runtime position::

        x[2i]'   = x[2i]*c_i - x[2i+1]*s_i
        x[2i+1]' = x[2i+1]*c_i + x[2i]*s_i

    dims ``[ROT, D)`` pass through. ``NEGATE_SIN=True`` turns this into the
    CONJUGATE rotation (output-side, negative angle) — the exact inverse of
    the q/k rotation at the same position; implemented as one template so
    the round-trip property is structurally guaranteed on device.
    UNVERIFIED in this environment: CUDA-gated (same flagged gap as PRs
    #88/#113/#114/#115/#117).
    """
    task = worker
    while task < B * NHEAD:
        b = task // NHEAD
        h = task % NHEAD
        half: tl.constexpr = ROT // 2
        i = tl.arange(0, half)
        p = tl.load(pos_ptr + b).to(tl.int64)
        base = (b * NHEAD + h) * D
        x_even = tl.load(x_ptr + base + 2 * i, cache_modifier=".cg").to(tl.float32)
        x_odd = tl.load(x_ptr + base + 2 * i + 1, cache_modifier=".cg").to(tl.float32)
        c = tl.load(cos_ptr + p * TSTRIDE + i).to(tl.float32)
        s = tl.load(sin_ptr + p * TSTRIDE + i).to(tl.float32)
        if NEGATE_SIN:
            s = -s
        tl.store(y_ptr + base + 2 * i, x_even * c - x_odd * s)
        tl.store(y_ptr + base + 2 * i + 1, x_odd * c + x_even * s)
        if ROT < D:
            offs_d = tl.arange(0, D)
            mt = offs_d >= ROT
            tail = tl.load(x_ptr + base + offs_d, mask=mt, other=0.0, cache_modifier=".cg").to(tl.float32)
            tl.store(y_ptr + base + offs_d, tail, mask=mt)
        task += P


@triton.jit
def _t_mla_scores(
    worker: tl.int32,
    P: tl.int32,
    q_ptr,
    latent_ptr,
    wtable_ptr,
    comp_ptr,
    compidx_ptr,
    sink_ptr,
    bias_ptr,
    pos_ptr,
    probs_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    SPOOL: tl.constexpr,
    MPOOL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    scale: tl.constexpr,
):
    """MLA fused scores + softmax + sink (issue #95). One task per (b, head).

    Candidate layout per (b, h): [W window | K compressed | 1 sink (last)].
    Window slot i -> logical cache position t = p - W + 1 + i, gathered from
    the per-row latent pool via the slot table (masked to [0, p]);
    compressed slot j -> comp_idx[b, j] (masked to >= 0; logits + bias when
    HAS_BIAS); sink logit per head, always valid. fp32 two-pass softmax over
    valid candidates; invalid slots exact 0.0. UNVERIFIED: CUDA-gated.
    """
    task = worker
    WIDTH: tl.constexpr = W + K + 1
    while task < B * H:
        b = task // H
        h = task % H
        offs_w = tl.arange(0, W)
        offs_k = tl.arange(0, K)
        offs_d = tl.arange(0, D)
        p = tl.load(pos_ptr + b).to(tl.int64)
        qb = tl.load(q_ptr + (b * H + h) * D + offs_d, cache_modifier=".cg").to(tl.float32)
        # window logits: logical t = p - W + 1 + i
        t = p - W + 1 + offs_w.to(tl.int64)
        m_w = (t >= 0) & (t <= p)
        slots_w = tl.load(wtable_ptr + b * SPOOL + t, mask=m_w, other=0).to(tl.int64)
        lat = tl.load(latent_ptr + b * SPOOL * D + slots_w[:, None] * D + offs_d[None, :],
                      mask=m_w[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        lw = tl.sum(lat * qb[None, :], axis=1) * scale
        # compressed logits
        e = tl.load(compidx_ptr + b * K + offs_k).to(tl.int64)
        m_k = e >= 0
        comp = tl.load(comp_ptr + b * MPOOL * D + e[:, None] * D + offs_d[None, :],
                       mask=m_k[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        lc = tl.sum(comp * qb[None, :], axis=1) * scale
        if HAS_BIAS:
            lc += tl.load(bias_ptr + b * K + offs_k, mask=m_k, other=0.0).to(tl.float32)
        # sink (per-head, always valid, LAST slot)
        lsink = tl.load(sink_ptr + h).to(tl.float32)
        # fused two-pass softmax over valid candidates ∪ sink. The union
        # axis (W+K+1) is not a power of two, so it is never materialized:
        # the window [W], compressed [K] and sink pieces are reduced
        # separately and share the global max / denominator (identical
        # result to a single fused pass — max and Σexp are order-free).
        mx = tl.maximum(tl.max(tl.where(m_w, lw, -float("inf"))),
                        tl.max(tl.where(m_k, lc, -float("inf"))))
        mx = tl.maximum(mx, lsink)
        ex_w = tl.where(m_w, tl.exp(lw - mx), 0.0)
        ex_k = tl.where(m_k, tl.exp(lc - mx), 0.0)
        ex_sink = tl.exp(lsink - mx)
        denom = tl.sum(ex_w) + tl.sum(ex_k) + ex_sink
        prow = probs_ptr + (b * H + h) * WIDTH
        tl.store(prow + offs_w, ex_w / denom)
        tl.store(prow + W + offs_k, ex_k / denom)
        tl.store(prow + (W + K), ex_sink / denom)
        task += P


@triton.jit
def _t_mla_values(
    worker: tl.int32,
    P: tl.int32,
    probs_ptr,
    latent_ptr,
    wtable_ptr,
    comp_ptr,
    compidx_ptr,
    pos_ptr,
    y_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    SPOOL: tl.constexpr,
    MPOOL: tl.constexpr,
):
    """MLA context gather (issue #95): window + compressed pools; the sink
    column (W+K) contributes NO value. fp32 accumulation, single store.
    UNVERIFIED: CUDA-gated."""
    task = worker
    while task < B * H:
        b = task // H
        h = task % H
        offs_w = tl.arange(0, W)
        offs_k = tl.arange(0, K)
        offs_d = tl.arange(0, D)
        p = tl.load(pos_ptr + b).to(tl.int64)
        prow = probs_ptr + (b * H + h) * (W + K + 1)
        t = p - W + 1 + offs_w.to(tl.int64)
        m_w = (t >= 0) & (t <= p)
        slots_w = tl.load(wtable_ptr + b * SPOOL + t, mask=m_w, other=0).to(tl.int64)
        pw = tl.load(prow + offs_w, mask=m_w, other=0.0, cache_modifier=".cg")
        lat = tl.load(latent_ptr + b * SPOOL * D + slots_w[:, None] * D + offs_d[None, :],
                      mask=m_w[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        acc = tl.sum(pw[:, None] * lat, axis=0)
        e = tl.load(compidx_ptr + b * K + offs_k).to(tl.int64)
        m_k = e >= 0
        pc = tl.load(prow + W + offs_k, mask=m_k, other=0.0, cache_modifier=".cg")
        comp = tl.load(comp_ptr + b * MPOOL * D + e[:, None] * D + offs_d[None, :],
                       mask=m_k[:, None], other=0.0, cache_modifier=".cg").to(tl.float32)
        acc += tl.sum(pc[:, None] * comp, axis=0)
        tl.store(y_ptr + (b * H + h) * D + offs_d, acc)
        task += P


@triton.jit
def _t_indexer_scores(
    worker: tl.int32,
    P: tl.int32,
    q_ptr,
    c_ptr,
    w_ptr,
    s_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
    TILE: tl.constexpr,
    scale: tl.constexpr,
):
    """Lightning-indexer fused scoring (issue #97), one task per
    (batch, TILE-entry tile):

        s[b, j] = sum_h relu(<q[b, h, :], c[b, j, :]>) * scale * mix_w[b, h]

    with ``scale = head_dim**-0.5``. The head loop streams the row's full
    query block [H, D] against the tile and accumulates the per-head mix in
    registers (f32); q/entries may be stored bf16 (``.cg`` streamed). The
    full capacity M is scored — masking by the per-row valid candidate
    count happens in ``_t_index_topk``. Requires D and TILE to be powers of
    two (tl.arange); the lowering picks exact tiles for ragged M via masks.
    """
    NT: tl.constexpr = (M + TILE - 1) // TILE
    offs_d = tl.arange(0, D)
    task = worker
    while task < B * NT:
        b = task // NT
        t = task % NT
        offs_m = t * TILE + tl.arange(0, TILE)
        mm = offs_m < M
        cbase = c_ptr + b.to(tl.int64) * (M * D)
        acc = tl.zeros([TILE], tl.float32)
        for h in range(0, H):
            qh = tl.load(q_ptr + (b * H + h) * D + offs_d, cache_modifier=".cg").to(tl.float32)
            cj = tl.load(
                cbase + offs_m[:, None].to(tl.int64) * D + offs_d[None, :],
                mask=mm[:, None],
                other=0.0,
                cache_modifier=".cg",
            ).to(tl.float32)
            sc = tl.maximum(tl.sum(cj * qh[None, :], axis=1), 0.0) * scale
            wv = tl.load(w_ptr + b * H + h).to(tl.float32)
            acc += sc * wv
        tl.store(s_ptr + b * M + offs_m, acc, mask=mm)
        task += P


@triton.jit
def _t_index_topk(
    worker: tl.int32,
    P: tl.int32,
    s_ptr,
    valid_ptr,
    idx_ptr,
    bias_ptr,
    B: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TILE: tl.constexpr,
):
    """Fixed-count top-k selection (issue #97), one task per batch row.

    Rank by comparison counting over the row's valid prefix:

        rank(j) = #{valid i : s_i > s_j} + #{valid i < j : s_i == s_j}

    so the rank *is* the output slot — descending score with deterministic
    lowest-index tie-break, no sort and no scratch buffer. NaN scores inside
    the valid prefix are excluded (``s == s`` fails); candidates at or
    beyond the row's valid count are never observed (masked loads — the
    uninitialized tail may hold NaN canaries). Slots beyond a row's valid
    count keep the up-front -1 / 0.0 fill. ``bias = s_j / ||s_valid||_2``
    in fp32. M <= ~1k candidates: the O(M^2/TILE) comparison sweep is a
    few dozen register-block reductions. KP is the power-of-two pad of K
    (tl.arange); TILE a power of two.
    """
    offs_k = tl.arange(0, KP)
    km = offs_k < K
    task = worker
    while task < B:
        b = task
        vc = tl.load(valid_ptr + b)
        vc = tl.minimum(tl.maximum(vc, 0), M)
        # Deterministic fill first; selected slots are overwritten by the
        # rank-addressed scatter below (same-thread program order).
        tl.store(idx_ptr + b * K + offs_k, tl.full([KP], -1, tl.int32), mask=km)
        tl.store(bias_ptr + b * K + offs_k, tl.zeros([KP], tl.float32), mask=km)
        # Normalizer: ||s||_2 over the row's valid finite prefix.
        nacc = tl.zeros([TILE], tl.float32)
        t0 = 0
        while t0 < M:
            offs = t0 + tl.arange(0, TILE)
            sm = (offs < M) & (offs < vc)
            sv = tl.load(s_ptr + b * M + offs, mask=sm, other=0.0).to(tl.float32)
            nacc += tl.where(sv == sv, sv * sv, 0.0)
            t0 += TILE
        norm = tl.sqrt(tl.sum(nacc, axis=0))
        # Rank counting + rank-addressed scatter, chunk pair by chunk pair.
        t0 = 0
        while t0 < M:
            offs_i = t0 + tl.arange(0, TILE)
            mi = (offs_i < M) & (offs_i < vc)
            si = tl.load(s_ptr + b * M + offs_i, mask=mi, other=0.0).to(tl.float32)
            ci = mi & (si == si)
            rank = tl.zeros([TILE], tl.int32)
            t1 = 0
            while t1 < M:
                offs_j = t1 + tl.arange(0, TILE)
                mj = (offs_j < M) & (offs_j < vc)
                sj = tl.load(s_ptr + b * M + offs_j, mask=mj, other=0.0).to(tl.float32)
                cj = mj & (sj == sj)
                gt = (sj[None, :] > si[:, None]) & cj[None, :] & ci[:, None]
                eq = (sj[None, :] == si[:, None]) & cj[None, :] & ci[:, None] & (offs_j[None, :] < offs_i[:, None])
                rank += tl.sum((gt | eq).to(tl.int32), axis=1)
                t1 += TILE
            sel = ci & (rank < K)
            tl.store(idx_ptr + b * K + rank, offs_i.to(tl.int32), mask=sel)
            tl.store(bias_ptr + b * K + rank, si / norm, mask=sel)
            t0 += TILE
        task += P


@triton.jit
def _t_compressor_append(
    worker: tl.int32,
    P: tl.int32,
    pool_ptr,
    state_ptr,
    win_ptr,
    gates_ptr,
    rmsw_ptr,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    p_scalar,
    B: tl.constexpr,
    L: tl.constexpr,
    M: tl.constexpr,
    R: tl.constexpr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    POS_ROW: tl.constexpr,
):
    """Issue #96: DSA compressor entry emission — one task per (b, layer).

    Rows at the m-token boundary (``p[b] % m == m-1``, per-row positions
    from issue #93) fold their m-token window: fp32 softmax over the gates,
    weighted latent fold, rms_norm, rotate_half rope at the emitting row's
    own position (entries rotate ONCE at emission — decode rotates only the
    query), stored bf16 into the row's active Ca/Cb series slot. Series
    bookkeeping ping-pongs slot roles at ``cb_len == R``. Non-boundary rows
    are exact no-ops (masked stores only).
    """
    task = worker
    while task < B * L:
        b = task // L
        l = task % L
        if POS_ROW:
            p = tl.load(pos_ptr + b)
        else:
            p = p_scalar
        boundary = (p % M) == (M - 1)
        # --- gated softmax fold over the m-token window (fp32) ---
        gmax = tl.zeros([1], tl.float32) - float("inf")
        t = 0
        while t < M:
            gt = tl.load(gates_ptr + b * M + t, mask=boundary, other=0.0).to(tl.float32)
            gmax = tl.maximum(gmax, gt)
            t += 1
        denom = tl.zeros([1], tl.float32)
        t = 0
        while t < M:
            gt = tl.load(gates_ptr + b * M + t, mask=boundary, other=0.0).to(tl.float32)
            denom += tl.exp(gt - gmax)
            t += 1
        acc = tl.zeros([D], tl.float32)
        t = 0
        while t < M:
            gt = tl.load(gates_ptr + b * M + t, mask=boundary, other=0.0).to(tl.float32)
            wt = tl.exp(gt - gmax) / denom
            offs = tl.arange(0, D)
            wv = tl.load(win_ptr + (b * M + t) * D + offs, mask=boundary, other=0.0, cache_modifier=".cg").to(tl.float32)
            acc += wt * wv
            t += 1
        # --- rms_norm ---
        ms = tl.sum(acc * acc, axis=0) / D
        rmsw = tl.load(rmsw_ptr + tl.arange(0, D), cache_modifier=".cg").to(tl.float32)
        e = acc * (1.0 / tl.sqrt(ms + EPS)) * rmsw
        # --- rotate_half rope at the emitting row's position (once) ---
        # out[i] = e[i]*cos[i mod D/2] + sign·e[partner(i)]·sin[i mod D/2],
        # partner(i) = i+D/2 for the first half, i-D/2 for the second;
        # partner gather via lane-compare reduction (D small: head_dim ≤ 128).
        i = tl.arange(0, D)
        pair = tl.where(i < D // 2, i + D // 2, i - D // 2)
        cmp = tl.arange(0, D)[None, :] == pair[:, None]
        pv = tl.sum(tl.where(cmp, e[None, :], 0.0), axis=1)
        chf = tl.load(cos_ptr + b * (D // 2) + (i % (D // 2)), mask=boundary, other=0.0).to(tl.float32)
        shf = tl.load(sin_ptr + b * (D // 2) + (i % (D // 2)), mask=boundary, other=0.0).to(tl.float32)
        sign = tl.where(i < D // 2, -1.0, 1.0)
        e_rot = e * chf + sign * pv * shf
        # --- store entry + series bookkeeping (masked: boundary rows only) ---
        slot = tl.load(state_ptr + (b * L + l) * 2 + 0, mask=boundary, other=0)
        cb = tl.load(state_ptr + (b * L + l) * 2 + 1, mask=boundary, other=0)
        dst = ((b * L + l) * 2 + slot) * R * D + cb * D + i
        tl.store(pool_ptr + dst, e_rot.to(pool_ptr.dtype.element_ty), mask=boundary)
        ncb = tl.where(cb + 1 == R, 0, cb + 1)
        nslot = tl.where(cb + 1 == R, 1 - slot, slot)
        tl.store(state_ptr + (b * L + l) * 2 + 0, nslot, mask=boundary)
        tl.store(state_ptr + (b * L + l) * 2 + 1, ncb, mask=boundary)
        task += P

