"""Dense Triton device task bodies."""
from __future__ import annotations
import triton
import triton.language as tl

@triton.jit
def _t_embed(worker: tl.int32, P: tl.int32, ids_ptr, tok_ptr, out_ptr, B: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    """hidden[b, :] = tok[ids[b], :] — one task per batch row."""
    task = worker
    while task < B:
        offs = tl.arange(0, BC)
        m = offs < C
        tok_id = tl.load(ids_ptr + task).to(tl.int64)
        row = tl.load(tok_ptr + tok_id * C + offs, mask=m, other=0.0)
        tl.store(out_ptr + task * C + offs, row.to(tl.float32), mask=m)
        task += P


@triton.jit
def _t_rms2d(worker: tl.int32, P: tl.int32, x_ptr, g_ptr, y_ptr, B: tl.constexpr, C: tl.constexpr, BC: tl.constexpr, eps: tl.constexpr):
    """RMSNorm over one hidden row: one task per batch row."""
    task = worker
    while task < B:
        offs = tl.arange(0, BC)
        m = offs < C
        x = tl.load(x_ptr + task * C + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
        var = tl.sum(x * x, axis=0) / C
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + task * C + offs, x * (1.0 / tl.sqrt(var + eps)) * g, mask=m)
        task += P


@triton.jit
def _t_rms_heads(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    g_ptr,
    y_ptr,
    B: tl.constexpr,
    NHEAD: tl.constexpr,
    D: tl.constexpr,
    ROWSTRIDE: tl.constexpr,
    eps: tl.constexpr,
):
    """Per-head RMSNorm (Qwen3 q_norm/k_norm): one task per (batch, head).

    ``ROWSTRIDE`` is the source row stride: the q/k slices live interleaved
    inside each batch row's packed QKV projection, while the output is a
    packed [B, NHEAD, D] buffer.
    """
    task = worker
    while task < B * NHEAD:
        b = task // NHEAD
        h = task % NHEAD
        offs = tl.arange(0, D)
        x = tl.load(x_ptr + b * ROWSTRIDE + h * D + offs, cache_modifier=".cg").to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        g = tl.load(g_ptr + offs).to(tl.float32)
        tl.store(y_ptr + (b * NHEAD + h) * D + offs, x * (1.0 / tl.sqrt(var + eps)) * g)
        task += P


@triton.jit
def _t_rms2d_gated(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    gate_ptr,
    g_ptr,
    y_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    eps: tl.constexpr,
):
    """Sigmoid-gated RMSNorm over one hidden row (issue #100, GLM o_norm):
    one task per batch row. Strict-fp32 math (floe Glm53RMSNormGated):

        y = x * rsqrt(mean(x^2) + eps) * g * sigmoid(gate)

    Same task decomposition as ``_t_rms2d`` with a second elementwise
    input stream; the gate multiply folds after the weight multiply so a
    saturated gate (sigmoid fp32 -> 0) zeroes the row exactly.
    """
    task = worker
    while task < B:
        offs = tl.arange(0, BC)
        m = offs < C
        x = tl.load(x_ptr + task * C + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
        var = tl.sum(x * x, axis=0) / C
        g = tl.load(g_ptr + offs, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + task * C + offs, mask=m, other=0.0, cache_modifier=".cg").to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        tl.store(y_ptr + task * C + offs, x * (1.0 / tl.sqrt(var + eps)) * g * sig, mask=m)
        task += P


@triton.jit
def _t_rms_heads_gated(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    gate_ptr,
    g_ptr,
    y_ptr,
    B: tl.constexpr,
    NHEAD: tl.constexpr,
    D: tl.constexpr,
    ROWSTRIDE: tl.constexpr,
    eps: tl.constexpr,
):
    """Per-head sigmoid-gated RMSNorm (issue #100): the linear-attention
    output norm folded before ``o_proj`` — one task per (batch, head), the
    ``_t_rms_heads`` decomposition with a second per-head gate stream.
    """
    task = worker
    while task < B * NHEAD:
        b = task // NHEAD
        h = task % NHEAD
        offs = tl.arange(0, D)
        x = tl.load(x_ptr + b * ROWSTRIDE + h * D + offs, cache_modifier=".cg").to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        g = tl.load(g_ptr + offs).to(tl.float32)
        gate = tl.load(gate_ptr + (b * NHEAD + h) * D + offs, cache_modifier=".cg").to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))
        tl.store(y_ptr + (b * NHEAD + h) * D + offs, x * (1.0 / tl.sqrt(var + eps)) * g * sig)
        task += P


@triton.jit
def _t_gemv(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    w_ptr,
    y_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    BK: tl.constexpr,
):
    """y[b, n] = sum_k x[b, k] * w[k, n] — 16-column tiles, full-K (§6.2).

    One task per output tile; the task computes all B rows (weight tile
    loaded once per k-chunk is re-read per row from L2; B == 1 compiles to
    the original single-row path).
    """
    NTASK: tl.constexpr = N // TILE
    task = worker
    while task < NTASK:
        offs_n = task * TILE + tl.arange(0, TILE)
        for b in tl.static_range(B):
            acc = tl.zeros([TILE], tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                xv = tl.load(x_ptr + b * K + offs_k, mask=offs_k < K, other=0.0, cache_modifier=".cg").to(tl.float32)
                wt = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=offs_k[:, None] < K, other=0.0).to(tl.float32)
                acc += tl.sum(wt * xv[:, None], axis=0)
            tl.store(y_ptr + b * N + offs_n, acc)
        task += P


@triton.jit
def _t_gemv_transposed(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    tok_ptr,
    y_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    BK: tl.constexpr,
):
    """Tied head: logits[b, n] = sum_c x[b, c] * tok[n, c]."""
    NTASK: tl.constexpr = N // TILE
    task = worker
    while task < NTASK:
        offs_n = task * TILE + tl.arange(0, TILE)
        for b in tl.static_range(B):
            acc = tl.zeros([TILE], tl.float32)
            for k0 in range(0, K, BK):
                offs_k = k0 + tl.arange(0, BK)
                xv = tl.load(x_ptr + b * K + offs_k, mask=offs_k < K, other=0.0, cache_modifier=".cg").to(tl.float32)
                wt = tl.load(tok_ptr + offs_n[:, None].to(tl.int64) * K + offs_k[None, :], mask=offs_k[None, :] < K, other=0.0).to(tl.float32)
                acc += tl.sum(wt * xv[None, :], axis=1)
            tl.store(y_ptr + b * N + offs_n, acc)
        task += P


@triton.jit
def _t_swiglu(worker: tl.int32, P: tl.int32, gu_ptr, y_ptr, B: tl.constexpr, F: tl.constexpr, F2: tl.constexpr, ELEM: tl.constexpr):
    """act = silu(gate) * up over [B, F]; gate/up interleaved per row in the
    fused [B, 2F] gate_up output (ELEM divides F, so tiles stay in-row)."""
    NTASK: tl.constexpr = (B * F) // ELEM
    task = worker
    while task < NTASK:
        lo = task * ELEM
        b = lo // F
        c = (lo % F) + tl.arange(0, ELEM)
        row = b * F2
        g = tl.load(gu_ptr + row + c, cache_modifier=".cg").to(tl.float32)
        u = tl.load(gu_ptr + row + F + c, cache_modifier=".cg").to(tl.float32)
        tl.store(y_ptr + b * F + c, g / (1.0 + tl.exp(-g)) * u)
        task += P


@triton.jit
def _t_add(worker: tl.int32, P: tl.int32, a_ptr, b_ptr, y_ptr, BC: tl.constexpr, ELEM: tl.constexpr):
    """y = a + b over [B, C] (flat)."""
    NTASK: tl.constexpr = BC // ELEM
    task = worker
    while task < NTASK:
        offs = task * ELEM + tl.arange(0, ELEM)
        a = tl.load(a_ptr + offs, cache_modifier=".cg")
        b = tl.load(b_ptr + offs, cache_modifier=".cg")
        tl.store(y_ptr + offs, a + b)
        task += P


@triton.jit
def _t_linear_grouped(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    w_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    GH: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """Block-diagonal per-head GEMV (issue #95 GroupedLinear). One task per
    (row, N tile): each output n reads only its owning head's diagonal
    blocks — off-block weight storage is never touched. fp32 accumulation.
    UNVERIFIED: CUDA-gated."""
    task = worker
    K_G: tl.constexpr = K // GH
    N_G: tl.constexpr = N // GH
    while task < M * (N // TILE_N):
        m = task // (N // TILE_N)
        nt = task % (N // TILE_N)
        n0 = nt * TILE_N
        offs_n = n0 + tl.arange(0, TILE_N)
        acc = tl.zeros([TILE_N], tl.float32)
        for h in range(GH):
            lo = h * N_G
            hi = (h + 1) * N_G
            sel = (offs_n >= lo) & (offs_n < hi)
            if tl.sum(sel.to(tl.int32)) > 0:
                offs_k = h * K_G + tl.arange(0, K_G)
                xv = tl.load(x_ptr + m * K + offs_k, cache_modifier=".cg").to(tl.float32)
                nn = tl.where(sel, offs_n, 0) - h * N_G
                # w is stored [Cin, Cout] = [K, N] (§3.1, same as the base
                # linear): the head's diagonal block is
                # w[h*K_G:(h+1)*K_G, h*N_G:(h+1)*N_G] — rows are the head's
                # K slice, columns its N slice. y[n] = <w[:, n], x>.
                wv = tl.load(w_ptr + offs_k[:, None] * N + (h * N_G + nn)[None, :],
                             mask=sel[None, :], other=0.0, cache_modifier=".cg").to(tl.float32)
                acc += tl.sum(xv[:, None] * wv, axis=0)
        tl.store(y_ptr + m * N + offs_n, acc)
        task += P


@triton.jit
def _t_gemv_fp8(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    w_ptr,
    scale_ptr,
    y_ptr,
    K: tl.constexpr,
    N: tl.constexpr,
    TILE: tl.constexpr,
    BK: tl.constexpr,
):
    """y[n] = sum_k x[k] * dequant(w)[k, n] with 128x128 block FP8 scales.

    ``BK`` must equal the quant block width (128) so every k-chunk of a
    16-column tile falls in exactly one scale block. Weights are streamed
    as e4m3 and dequantized in-register (scale * fp8 -> fp32 accumulate).
    """
    NTASK: tl.constexpr = N // TILE
    KB: tl.constexpr = K // BK
    task = worker
    while task < NTASK:
        offs_n = task * TILE + tl.arange(0, TILE)
        sb_row = (task * TILE) // BK
        acc = tl.zeros([TILE], tl.float32)
        for kb in range(0, KB):
            offs_k = kb * BK + tl.arange(0, BK)
            s = tl.load(scale_ptr + sb_row * KB + kb).to(tl.float32)
            xv = tl.load(x_ptr + offs_k).to(tl.float32)
            # checkpoint layout is nn.Linear [out, in] row-major
            wt = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :]).to(tl.float32)
            acc += tl.sum(wt * (s * xv[None, :]), axis=1)
        tl.store(y_ptr + offs_n, acc)
        task += P


@triton.jit
def _t_moe_expert(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    gate_up_ptr,
    down_ptr,
    ids_ptr,
    partials_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    LIMIT: tl.constexpr,  # 0.0 encodes "unclamped"
):
    """partials[b, s, :] = down[e] @ (silu(clamp(g, ≤L)) · clamp(u, ±L)),
    (g, u) = gate_up[e] @ x[b, :], e = ids[b, s] loaded at run time.

    gate_up is the Mixtral-layout stack [E, 2I, H] row-major: the gate rows
    sit at e·2I·H + i·H + h, the up rows I slots later. down is [E, H, I]."""
    offs_i = tl.arange(0, I)
    offs_h = tl.arange(0, H)
    ntask: tl.constexpr = B * K
    task = worker
    while task < ntask:
        b = task // K
        s = task % K
        e = tl.load(ids_ptr + b * K + s).to(tl.int32)
        xb = tl.load(x_ptr + b * H + offs_h, cache_modifier=".cg").to(tl.float32)
        wbase = (e * 2 * I) * H
        wg = tl.load(gate_up_ptr + wbase + offs_i[:, None] * H + offs_h[None, :], cache_modifier=".cg").to(tl.float32)
        wu = tl.load(gate_up_ptr + wbase + (I + offs_i)[:, None] * H + offs_h[None, :], cache_modifier=".cg").to(tl.float32)
        g = tl.sum(wg * xb[None, :], axis=1)
        u = tl.sum(wu * xb[None, :], axis=1)
        if LIMIT > 0.0:
            g = tl.minimum(g, LIMIT)
            u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        act = g / (1.0 + tl.exp(-g)) * u
        wd = tl.load(down_ptr + (e * H) * I + offs_h[:, None] * I + offs_i[None, :], cache_modifier=".cg").to(tl.float32)
        acc = tl.sum(wd * act[None, :], axis=1)
        tl.store(partials_ptr + (b * K + s) * H + offs_h, acc)
        task += P


@triton.jit
def _t_moe_combine(
    worker: tl.int32,
    P: tl.int32,
    partials_ptr,
    weights_ptr,
    shared_ptr,
    y_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    HAS_SHARED: tl.constexpr,
):
    """y[b, :] = Σ_k weights[b, k] · partials[b, k, :] (slot order, fp32
    accumulate) + shared[b, :] when HAS_SHARED."""
    offs_h = tl.arange(0, H)
    task = worker
    while task < B:
        b = task
        acc = tl.zeros([H], tl.float32)
        for s in range(K):
            w = tl.load(weights_ptr + b * K + s).to(tl.float32)
            acc += w * tl.load(partials_ptr + (b * K + s) * H + offs_h, cache_modifier=".cg").to(tl.float32)
        if HAS_SHARED:
            acc += tl.load(shared_ptr + b * H + offs_h, cache_modifier=".cg").to(tl.float32)
        tl.store(y_ptr + b * H + offs_h, acc)
        task += P


@triton.jit
def _t_moe_route(
    worker: tl.int32,
    P: tl.int32,
    x_ptr,
    w_ptr,
    bias_ptr,
    tok_ptr,
    t2e_ptr,
    ids_ptr,
    weights_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,  # padded expert block, power of two, E <= EP
    K: tl.constexpr,  # power of two (block width for the selection state)
    MODE_HASH: tl.constexpr,
    SQRTSP: tl.constexpr,  # sqrtsoftplus (DeepSeek-V4) vs sigmoid noaux_tc
    NORM_TOPK: tl.constexpr,
    RSF: tl.constexpr,
):
    """Router decode step — degenerate-group and hash device paths.

    scores = sqrt(softplus(logits)) (SQRTSP) or sigmoid(logits) + bias
    (noaux_tc, n_group == 1: the single group always wins — floe's own
    degeneracy note). Selection = K masked-argmax rounds over the choice
    scores; ``tl.argmax`` returns the first maximal index, which IS the
    documented tie rule (ties resolve to the lower expert index). Weights
    gather the UNBIASED scores, renorm w/(Σw+1e-20) iff NORM_TOPK, × RSF.
    Hash mode replaces selection with the frozen tid2eid[token_id] gather
    (renorm unconditional, floe semantics)."""
    offs_e = tl.arange(0, EP)
    offs_k = tl.arange(0, K)
    emask = offs_e < E
    neg_inf: tl.constexpr = -1.0e38
    task = worker
    while task < B:
        b = task
        se = tl.zeros([EP], tl.float32)
        for h in range(H):
            xv = tl.load(x_ptr + b * H + h, cache_modifier=".cg").to(tl.float32)
            wv = tl.load(w_ptr + offs_e * H + h, mask=emask, other=0.0, cache_modifier=".cg").to(tl.float32)
            se += wv * xv
        if SQRTSP:
            scores = tl.sqrt(tl.where(se > 20.0, se, tl.log(1.0 + tl.exp(se))))
            choice = scores
        else:
            sig = 1.0 / (1.0 + tl.exp(-se))
            bias = tl.load(bias_ptr + offs_e, mask=emask, other=0.0).to(tl.float32)
            scores = sig  # weights gather the UNBIASED scores (floe semantics)
            choice = sig + bias
        if MODE_HASH:
            tok = tl.load(tok_ptr + b).to(tl.int32)
            sel = tl.load(t2e_ptr + tok * K + offs_k).to(tl.int32)
            # gather scores at the selected experts: one [EP, K] masked sum
            wsel = tl.sum(tl.where(offs_e[:, None] == sel[None, :], scores[:, None], 0.0), axis=0)
            wsel = wsel / (tl.sum(wsel, axis=0) + 1e-20)
        else:
            choice = tl.where(emask, choice, neg_inf)
            sel = tl.zeros([K], tl.int32)
            wsel = tl.zeros([K], tl.float32)
            for kk in range(K):
                best = tl.argmax(choice, axis=0)  # first max index: ties -> lower expert
                hit = offs_e == best
                w_best = tl.sum(tl.where(hit, scores, 0.0), axis=0)
                sel = tl.where(offs_k == kk, best + tl.zeros([K], tl.int32), sel)
                wsel = tl.where(offs_k == kk, w_best + tl.zeros([K], tl.float32), wsel)
                choice = tl.where(hit, neg_inf, choice)
            if NORM_TOPK:
                wsel = wsel / (tl.sum(wsel, axis=0) + 1e-20)
        wsel = wsel * RSF
        tl.store(ids_ptr + b * K + offs_k, sel)
        tl.store(weights_ptr + b * K + offs_k, wsel)
        task += P

