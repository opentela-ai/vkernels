"""Triton device backend: the phase-synchronous megakernel on a real GPU.

This closes the loop the design doc leaves open at §8.4: a *verified* grid
synchronization path on the actual stack. Triton 3.7 on this GB10 provides
GPU-scope acquire/release atomics (``tl.atomic_add(..., sem=..., scope="gpu")``)
and data-dependent control flow, which is exactly what the §8.1 device
algorithm needs:

    for phase in compiled_phases:
        task_id = worker
        while task_id < phase.task_count:
            run_task(...)
            task_id += workers
        grid_sync()          # outside the tile loop: idle workers participate

The barrier is a monotonic-counter spin barrier: each block publishes its
prior writes with an ``acq_rel`` arrival and re-acquires with an atomic
poll; the host passes a per-invocation counter base (deterministic: the
base grows by ``num_barriers * P`` per step), so no reset kernel and no
extra launch is needed between invocations — **one kernel launch per
decode step**, verifiable by trace (§15.2).

Batched decode (B>1, the §1.1 "each sequence supplies one token" contract,
one launch serving B sequences): task domains grow by B where per-row or
per-head (RMSNorm per row, QK-norm/RoPE/attention per (batch, head),
append per (batch, kv-head)); projection tiles compute B rows per task
(shared weight tile, one row at a time — B=1 compiles to the original
single-row path). Every sequence carries its **own runtime position** and
its **own slot-table row**, so a batch is naturally ragged: sequences at
different cache lengths, some continuing committed kvaas prefixes, some
starting cold — all in the same persistent launch.

The KV cache is addressed through the kvaas data plane: token-slot-major
pool buffers (one slot = one token's K or V per layer,
``n_kv_heads * head_dim * dtype`` bytes) indexed by ``table[b, t]`` — the
identity table locally, ``Admission.block_table()`` rows under a lease.

The CPU reference executor remains the schedule-logic oracle; this backend
is validated against the HF-checked NumPy oracle (fp32 ~1e-6, bf16 ~5e-3
through 28 layers).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

__all__ = [
    "grid_barrier",
    "qwen3_megakernel",
    "TritonMegakernel",
    "triton_available",
    "milestone0_barrier_test",
    "attach_megakernel_pool",
]


def triton_available() -> bool:
    try:
        import triton  # noqa: F401

        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Milestone-0 primitive: the grid barrier (§8.4)
# ---------------------------------------------------------------------------


@triton.jit
def grid_barrier(bar_ptr, target: tl.int64):
    """Monotonic spin barrier over all resident blocks.

    ``bar_ptr`` is a single int64 counter that never resets within (or
    across) invocations; ``target`` is the absolute arrival count that
    releases this barrier instance (``base + (k+1) * P`` for barrier k).
    The block-level fence first orders every thread's prior global stores;
    arrival uses acq_rel at gpu scope (publishing them, acquiring the
    counter); the poll uses acquire, ordering all post-barrier loads after
    the release wave.
    """
    tl.debug_barrier()  # all threads' stores issued before the arrival
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def milestone0_barrier_test(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64, IDLE: tl.constexpr):
    """§8.4 producer-barrier-consumer capability microprogram.

    Every round: each block writes a round-tagged value, barrier, then
    reads a neighbour's value (cross-block visibility through the
    barrier). With ``IDLE`` set, odd-numbered blocks skip the write work
    but still arrive at every barrier (§8.3 idle-worker participation).
    Accumulated neighbour values are checksummed per block.
    """
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        if (not IDLE) or (pid % 2 == 0):
            tl.store(out_ptr + pid, pid * 1000.0 + r)
        grid_barrier(bar_ptr, base + (r + 1) * P)
        nb = (pid + 1) % P
        v = tl.load(out_ptr + nb)
        if (not IDLE) or (nb % 2 == 0):
            acc += v
        r += 1
    tl.store(out_ptr + P + pid, acc)


# ---------------------------------------------------------------------------
# Task bodies (§6), batched: each runs tasks worker, worker+P, ... of its
# phase. B is constexpr; B == 1 compiles the original single-row paths.
# Row/head tasks decode (b, h) from the flat task id; every sequence has
# its own runtime position (pos_ptr[b]) and slot-table row (table[b, t]).
# ---------------------------------------------------------------------------


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
):
    """Rotate-half RoPE at row b's runtime position: one task per (b, head)."""
    task = worker
    while task < B * NHEAD:
        b = task // NHEAD
        h = task % NHEAD
        half: tl.constexpr = D // 2
        d = tl.arange(0, half)
        p = tl.load(pos_ptr + b).to(tl.int64)
        base = (b * NHEAD + h) * D
        x1 = tl.load(x_ptr + base + d, cache_modifier=".cg").to(tl.float32)
        x2 = tl.load(x_ptr + base + half + d, cache_modifier=".cg").to(tl.float32)
        c = tl.load(cos_ptr + p * D + d).to(tl.float32)  # cos = cat([f, f])
        s = tl.load(sin_ptr + p * D + d).to(tl.float32)
        tl.store(y_ptr + base + d, x1 * c - x2 * s)
        tl.store(y_ptr + base + half + d, x2 * c + x1 * s)
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


# ---------------------------------------------------------------------------
# The megakernel (§8.1): embedding | L x 17 phases | final norm + tied head
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["bar_base", "L", "WQKV_L", "WOP_L", "WGU_L", "WDOWN_L", "LNLN_L", "KCBASE_L"])
def qwen3_megakernel(
    ids_ptr,
    pos_ptr,
    table_ptr,
    tok_ptr,
    final_g_ptr,
    cos_ptr,
    sin_ptr,
    ln1_ptr,
    qkv_w_ptr,
    qn_ptr,
    kn_ptr,
    op_ptr,
    ln2_ptr,
    gu_ptr,
    down_ptr,
    k_cache_ptr,
    v_cache_ptr,
    ws_ptr,
    logits_ptr,
    bar_ptr,
    bar_base: tl.int64,
    L,
    WQKV_L,
    WOP_L,
    WGU_L,
    WDOWN_L,
    LNLN_L,
    KCBASE_L,
    # dims
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
    V: tl.constexpr,
    SCAP: tl.constexpr,
    QKVW: tl.constexpr,
    HD: tl.constexpr,
    F2: tl.constexpr,
    WS_LAYER: tl.constexpr,
    O_HIDDEN_A: tl.constexpr,
    O_HIDDEN_B: tl.constexpr,
    O_FINAL: tl.constexpr,
    O_RMS1: tl.constexpr,
    O_QKV: tl.constexpr,
    O_QN: tl.constexpr,
    O_KN: tl.constexpr,
    O_RQ: tl.constexpr,
    O_RK: tl.constexpr,
    O_SCORES: tl.constexpr,
    O_PROBS: tl.constexpr,
    O_CTX: tl.constexpr,
    O_ATTN: tl.constexpr,
    O_RMS2: tl.constexpr,
    O_GU: tl.constexpr,
    O_ACT: tl.constexpr,
    O_DOWN: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BTR: tl.constexpr,
    ELEM: tl.constexpr,
):
    worker = tl.program_id(0)
    P = tl.num_programs(0)
    BC: tl.constexpr = B * C

    # ---- phase 0: embedding -------------------------------------------
    _t_embed(worker, P, ids_ptr, tok_ptr, ws_ptr + O_HIDDEN_A, B, C, C)
    grid_barrier(bar_ptr, bar_base + P)

    for l in range(L):
        li = l.to(tl.int64)
        base = ws_ptr + li * WS_LAYER
        # phase 1: rms1
        _t_rms2d(worker, P, ws_ptr + O_HIDDEN_A, ln1_ptr + li * LNLN_L, base + O_RMS1, B, C, C, EPS)
        grid_barrier(bar_ptr, bar_base + (2 + 17 * l) * P)
        # phase 2: qkv projection [C -> QKVW]
        _t_gemv(worker, P, base + O_RMS1, qkv_w_ptr + li * WQKV_L, base + O_QKV, B, C, QKVW, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (3 + 17 * l) * P)
        # phases 3-4: q/k RMSNorm per head (v needs no processing); the q/k
        # slices are interleaved inside each row's packed QKV projection.
        _t_rms_heads(worker, P, base + O_QKV, qn_ptr + li * D, base + O_QN, B, H, D, QKVW, EPS)
        grid_barrier(bar_ptr, bar_base + (4 + 17 * l) * P)
        _t_rms_heads(worker, P, base + O_QKV + HD, kn_ptr + li * D, base + O_KN, B, KVH, D, QKVW, EPS)
        grid_barrier(bar_ptr, bar_base + (5 + 17 * l) * P)
        # phases 5-6: rope q/k at each row's runtime position
        _t_rope(worker, P, base + O_QN, cos_ptr, sin_ptr, pos_ptr, base + O_RQ, B, H, D)
        grid_barrier(bar_ptr, bar_base + (6 + 17 * l) * P)
        _t_rope(worker, P, base + O_KN, cos_ptr, sin_ptr, pos_ptr, base + O_RK, B, KVH, D)
        grid_barrier(bar_ptr, bar_base + (7 + 17 * l) * P)
        # phase 7: cache append (k roped; v straight from the qkv buffer)
        kcl = k_cache_ptr + li * KCBASE_L
        vcl = v_cache_ptr + li * KCBASE_L
        _t_append(worker, P, kcl, vcl, table_ptr, pos_ptr, base + O_RK, base + O_QKV + HD + KVH * D, B, KVH, D, QKVW, SCAP)
        grid_barrier(bar_ptr, bar_base + (8 + 17 * l) * P)
        # phases 8-10: attention (GQA, per-row valid lengths)
        _t_scores(worker, P, base + O_RQ, kcl, table_ptr, pos_ptr, base + O_SCORES, B, H, KVH, D, SCAP, BT, SCALE)
        grid_barrier(bar_ptr, bar_base + (9 + 17 * l) * P)
        _t_softmax(worker, P, base + O_SCORES, pos_ptr, base + O_PROBS, B, H, SCAP, BTR)
        grid_barrier(bar_ptr, bar_base + (10 + 17 * l) * P)
        _t_values(worker, P, base + O_PROBS, vcl, table_ptr, pos_ptr, base + O_CTX, B, H, KVH, D, SCAP, BT)
        grid_barrier(bar_ptr, bar_base + (11 + 17 * l) * P)
        # phase 11: o_proj [H*D -> C], phase 12: residual add
        _t_gemv(worker, P, base + O_CTX, op_ptr + li * WOP_L, base + O_ATTN, B, HD, C, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (12 + 17 * l) * P)
        _t_add(worker, P, ws_ptr + O_HIDDEN_A, base + O_ATTN, ws_ptr + O_HIDDEN_B, BC, ELEM)
        grid_barrier(bar_ptr, bar_base + (13 + 17 * l) * P)
        # phase 13: rms2, phase 14: gate_up [C -> 2F]
        _t_rms2d(worker, P, ws_ptr + O_HIDDEN_B, ln2_ptr + li * LNLN_L, base + O_RMS2, B, C, C, EPS)
        grid_barrier(bar_ptr, bar_base + (14 + 17 * l) * P)
        _t_gemv(worker, P, base + O_RMS2, gu_ptr + li * WGU_L, base + O_GU, B, C, F2, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (15 + 17 * l) * P)
        # phase 15: swiglu, phase 16: down [F -> C], phase 17: residual add
        _t_swiglu(worker, P, base + O_GU, base + O_ACT, B, F, F2, ELEM)
        grid_barrier(bar_ptr, bar_base + (16 + 17 * l) * P)
        _t_gemv(worker, P, base + O_ACT, down_ptr + li * WDOWN_L, base + O_DOWN, B, F, C, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (17 + 17 * l) * P)
        _t_add(worker, P, ws_ptr + O_HIDDEN_B, base + O_DOWN, ws_ptr + O_HIDDEN_A, BC, ELEM)
        grid_barrier(bar_ptr, bar_base + (18 + 17 * l) * P)

    # ---- final phases: norm + tied head (no barrier after the last) ------
    _t_rms2d(worker, P, ws_ptr + O_HIDDEN_A, final_g_ptr, ws_ptr + O_FINAL, B, C, C, EPS)
    grid_barrier(bar_ptr, bar_base + (17 * L + 2) * P)
    _t_gemv_transposed(worker, P, ws_ptr + O_FINAL, tok_ptr, logits_ptr, B, C, V, TILE, BK)


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    """Layer-uniform fp32 workspace layout (element offsets, batch-sized)."""

    C: int
    H: int
    KVH: int
    D: int
    F: int
    V: int
    SCAP: int
    L: int
    B: int
    offsets: dict
    ws_layer: int
    ws_total: int
    num_phases: int  # 1 + 17L + 2
    num_barriers: int  # one after every phase except the last


def _plan(config, capacity: int, batch: int) -> _Plan:
    C, H, KVH, D, F, V, L = (
        config.hidden,
        config.heads,
        config.kv_heads,
        config.head_dim,
        config.intermediate,
        config.vocab,
        config.layers,
    )
    HD, F2, QKVW = H * D, 2 * F, (H + 2 * KVH) * D
    off: dict[str, int] = {}
    o = 0

    def take(name, n):
        nonlocal o
        off[name] = o
        o += n

    take("hidden_a", batch * C)
    take("hidden_b", batch * C)
    take("final", batch * C)
    o = (o + 63) // 64 * 64  # align the layer block
    layer_base = o
    for name, n in [
        ("rms1", batch * C),
        ("qkv", batch * QKVW),
        ("qn", batch * HD),
        ("kn", batch * KVH * D),
        ("rq", batch * HD),
        ("rk", batch * KVH * D),
        ("scores", batch * H * capacity),
        ("probs", batch * H * capacity),
        ("ctx", batch * HD),
        ("attn", batch * C),
        ("rms2", batch * C),
        ("gu", batch * F2),
        ("act", batch * F),
        ("down", batch * C),
    ]:
        take(name, n)
    ws_layer = o - layer_base
    ws_total = layer_base + ws_layer * L
    num_phases = 1 + 17 * L + 2
    return _Plan(C, H, KVH, D, F, V, capacity, L, batch, off, ws_layer, ws_total, num_phases, num_phases - 1)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _elem_tile(C: int, F: int) -> int:
    """Largest power-of-two elementwise tile dividing both C and F (<=1024)."""
    t = 1024
    while t > 1 and (C % t or F % t):
        t //= 2
    return t


class TritonMegakernel:
    """Host runner: one persistent launch per step, B sequences per launch.

    ``run(tokens, positions)`` executes one decode step for every batch row:
    sequence ``b`` appends token ``tokens[b]`` at its own runtime position
    ``positions[b]``, addressed through slot-table row ``b``. Rows may be at
    different cache lengths (ragged batch) and may mix cold sequences with
    ones continuing committed kvaas prefixes.
    """

    def __init__(
        self,
        config,
        weights,
        *,
        capacity: int = 256,
        workers: int = 48,
        dtype=torch.bfloat16,
        device="cuda",
        batch: int = 1,
        k_cache=None,
        v_cache=None,
    ):
        assert config.heads % config.kv_heads == 0
        assert 1 <= batch
        self.config = config
        self.workers = workers
        self.device = device
        self.dtype = dtype
        self.batch = batch
        self.plan = _plan(config, capacity, batch)
        p = self.plan
        self.elem = _elem_tile(config.hidden, config.intermediate)
        self.btcap = 64  # attention chunk; [64, D] fp32 tiles stay in registers
        assert capacity <= 1024, "loop-free softmax needs the row in one block"
        assert config.hidden % self.elem == 0 and config.intermediate % self.elem == 0

        def stack(getfn):
            return torch.stack([torch.from_numpy(getfn(l)) for l in range(config.layers)]).to(device, dtype).contiguous()

        w = weights
        self.tok = torch.from_numpy(w.token_emb).to(device, dtype).contiguous()
        self.final_g = torch.from_numpy(w.final_gamma).to(device, dtype).contiguous()
        # RoPE tables stay fp32 regardless of weight dtype (fp32 math inside).
        self.cos = torch.from_numpy(w.cos[:capacity]).to(device, torch.float32).contiguous()
        self.sin = torch.from_numpy(w.sin[:capacity]).to(device, torch.float32).contiguous()
        self.ln1 = stack(lambda l: w.layers[l].ln1_gamma)
        self.qkv_w = stack(lambda l: w.layers[l].qkv_w)
        self.qn = stack(lambda l: w.layers[l].q_norm_gamma)
        self.kn = stack(lambda l: w.layers[l].k_norm_gamma)
        self.op_w = stack(lambda l: w.layers[l].o_proj_w)
        self.ln2 = stack(lambda l: w.layers[l].ln2_gamma)
        self.gu_w = stack(lambda l: w.layers[l].gate_up_w)
        self.down_w = stack(lambda l: w.layers[l].down_w)
        self.ws = torch.zeros(p.ws_total, device=device, dtype=torch.float32)
        # KV cache: token-slot-major [layers, max_tokens, KVH, D] — the kvaas
        # pool granularity — either locally allocated or caller-provided
        # (daemon-owned, imported via CUDA IPC). Addressing always goes
        # through the per-row slot table; the DEFAULT table gives each batch
        # row a disjoint slot range (row b -> slots [b*capacity, (b+1)*capacity))
        # so independent sequences never alias. A per-layer buffer tuple is
        # accepted when the layer stride is uniform (attach_megakernel_pool's
        # layout); the kernel then gets layer 0's pointer + the stride.
        need_slots = capacity * batch
        shape = (config.layers, need_slots, config.kv_heads, config.head_dim)
        if k_cache is None or v_cache is None:
            self.k_cache = torch.zeros(shape, device=device, dtype=torch.float32)
            self.v_cache = torch.zeros(shape, device=device, dtype=torch.float32)
            self.pool_slots = need_slots
            self._k_launch, self._v_launch = self.k_cache, self.v_cache
            self._layer_stride = need_slots * config.kv_heads * config.head_dim
        elif isinstance(k_cache, torch.Tensor):
            if tuple(k_cache.shape[0:1]) + tuple(k_cache.shape[2:]) != (config.layers, config.kv_heads, config.head_dim):
                raise ValueError(f"pool shape {tuple(k_cache.shape)} incompatible with ([layers, >= {need_slots}, kv_heads, head_dim])")
            if k_cache.shape[1] < need_slots or v_cache.shape[1] < need_slots:
                raise ValueError(f"pool has {min(k_cache.shape[1], v_cache.shape[1])} slots < {need_slots} (batch {batch} x capacity {capacity})")
            self.k_cache, self.v_cache = k_cache, v_cache
            self.pool_slots = k_cache.shape[1]
            self._k_launch, self._v_launch = k_cache, v_cache
            self._layer_stride = k_cache.shape[1] * config.kv_heads * config.head_dim
        else:
            kl, vl = tuple(k_cache), tuple(v_cache)
            if len(kl) != config.layers or len(vl) != config.layers:
                raise ValueError(f"pool layer counts {(len(kl), len(vl))} != {(config.layers, config.layers)}")
            k0 = kl[0]
            stride_bytes = k0.numel() * k0.element_size()
            for bufs in (kl, vl):
                for l in range(len(bufs) - 1):
                    if bufs[l + 1].data_ptr() - bufs[l].data_ptr() != stride_bytes:
                        raise ValueError("non-uniform pool layer stride (daemon-placed buffers); use attach_megakernel_pool()")
            self.k_cache, self.v_cache = kl, vl
            self.pool_slots = k0.shape[0]
            self._k_launch, self._v_launch = k0, vl[0]
            self._layer_stride = k0.numel()
        self.table = torch.arange(need_slots, device=device, dtype=torch.int64).view(batch, capacity).contiguous()
        self._table_row_lens = [capacity] * batch
        # Persistent PINNED staging for the per-step tokens/positions: the H2D
        # copies stay non_blocking without the pageable-temporary lifetime
        # hazard (a freed CPU source read by a still-queued async copy).
        self._ids_host = torch.zeros(batch, dtype=torch.int64, pin_memory=True)
        self._pos_host = torch.zeros(batch, dtype=torch.int32, pin_memory=True)
        self.ids = torch.zeros(batch, device=device, dtype=torch.int64)
        self.pos = torch.zeros(batch, device=device, dtype=torch.int32)
        self.logits = torch.zeros(batch, config.vocab, device=device, dtype=torch.float32)
        self.barrier_counter = torch.zeros(1, device=device, dtype=torch.int64)
        self._bar_base = 0

    # -- kvaas integration --------------------------------------------------

    def set_slot_table(self, slots):
        """Install per-row slot tables (e.g. ``Admission.block_table`` rows).

        Accepts a 1-D table (batch size must be 1) or ``[B, n]`` rows where
        ``slots[b, i]`` is the pool slot holding sequence ``b``'s logical
        position ``i``. Rows are zero-padded to the cache capacity (the
        kernel's row stride is a compile-time constant); padding is never
        read because each row's accesses are masked to its runtime position,
        and :meth:`run` bounds-checks positions against the *logical* row
        lengths recorded here.
        """
        t = slots if torch.is_tensor(slots) else torch.as_tensor(slots, dtype=torch.int64)
        t = t.to(torch.int64) if torch.is_tensor(slots) else t
        if t.dim() == 1:
            if self.batch != 1:
                raise ValueError(f"1-D slot table given for batch {self.batch}; pass [B, n] rows")
            t = t.unsqueeze(0)
        if t.shape[0] != self.batch:
            raise ValueError(f"slot table has {t.shape[0]} rows, batch is {self.batch}")
        if t.numel() == 0:
            raise ValueError("slot table is empty")
        if int(t.max()) >= self.pool_slots:
            raise ValueError(f"slot {int(t.max())} outside pool of {self.pool_slots} slots")
        self._table_row_lens = [t.shape[1]] * self.batch
        cap = self.plan.SCAP
        if t.shape[1] < cap:
            pad = torch.zeros((self.batch, cap - t.shape[1]), dtype=torch.int64)
            t = torch.cat([t, pad], dim=1)
        elif t.shape[1] > cap:
            raise ValueError(f"slot table covers {t.shape[1]} positions > capacity {cap}")
        self.table = t.to(device=self.device, dtype=torch.int64).contiguous()
        return self

    @classmethod
    def from_pool_view(cls, config, view, weights, *, capacity: int, workers: int = 48, dtype=torch.bfloat16, device="cuda", batch: int = 1):
        """Build the megakernel over a kvaas :class:`PoolView`'s buffers.

        The pool must expose per-layer buffers with a uniform layer stride
        (:func:`attach_megakernel_pool` requests exactly that layout: one
        packed buffer per side). Daemon placement of individually planned
        buffers is not uniform, so those are rejected with a clear
        diagnostic rather than silently misaddressed.
        """
        if len(view.k_buffers) != config.layers or len(view.v_buffers) != config.layers:
            raise ValueError(f"pool has {len(view.k_buffers)} layers, config wants {config.layers}")
        if view.n_kv_heads != config.kv_heads or view.head_dim != config.head_dim:
            raise ValueError("pool geometry (kv_heads/head_dim) does not match the config")
        k0 = view.k_buffers[0]
        expect = k0.numel() * k0.element_size()
        for bufs, name in ((view.k_buffers, "k"), (view.v_buffers, "v")):
            for l in range(len(bufs) - 1):
                if bufs[l + 1].data_ptr() - bufs[l].data_ptr() != expect:
                    raise ValueError(f"pool {name} layer strides are not uniform (daemon-placed buffers); use attach_megakernel_pool(), which requests one packed buffer per side")
        return cls(
            config,
            weights,
            capacity=capacity,
            workers=workers,
            dtype=dtype,
            device=device,
            batch=batch,
            k_cache=tuple(view.k_buffers),
            v_cache=tuple(view.v_buffers),
        )

    # -- schedule facts (cross-checkable against the compiled program) ------

    @property
    def num_phases(self) -> int:
        return self.plan.num_phases

    @property
    def num_barriers(self) -> int:
        return self.plan.num_barriers

    @property
    def barrier_base(self) -> int:
        return self._bar_base

    def task_counts(self) -> dict[str, int]:
        """Per-phase task counts at this batch size (§6.4 tile contract)."""
        cfg, B = self.config, self.batch
        f2 = 2 * cfg.intermediate
        layer = {
            "rms1": B,
            "qkv": ((cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim) // 16,
            "qnorm": B * cfg.heads,
            "knorm": B * cfg.kv_heads,
            "rope_q": B * cfg.heads,
            "rope_k": B * cfg.kv_heads,
            "append": B * cfg.kv_heads,
            "scores": B * cfg.heads,
            "softmax": B * cfg.heads,
            "values": B * cfg.heads,
            "o_proj": cfg.hidden // 16,
            "add1": (B * cfg.hidden) // self.elem,
            "rms2": B,
            "gate_up": f2 // 16,
            "swiglu": (B * cfg.intermediate) // self.elem,
            "down": cfg.hidden // 16,
            "add2": (B * cfg.hidden) // self.elem,
        }
        out = {"embedding": B}
        for l in range(cfg.layers):
            out.update({f"l{l}_{k}": v for k, v in layer.items()})
        out["final_rms"] = B
        out["logits"] = cfg.vocab // 16
        return out

    # -- launch -------------------------------------------------------------

    def run(self, token_ids, positions, *, check_counter: bool = False) -> torch.Tensor:
        """One batched decode step = exactly one kernel launch (§15.2).

        ``token_ids``/``positions`` are per-row (int/list/tensor); sequence
        ``b`` appends ``token_ids[b]`` at ``positions[b]``. Returns the
        ``[B, V]`` logits. Two tiny H2D copies carry the step's tokens and
        positions (disclosed: one kernel launch + two ≤8·B-byte copies).
        """
        cfg = self.config
        p = self.plan
        ids = self._as_row(token_ids, self.ids)
        pos = self._as_row(positions, self.pos)
        for b in range(self.batch):
            if not (0 <= int(pos[b]) < p.SCAP):
                raise ValueError(f"row {b}: position {int(pos[b])} out of [0, {p.SCAP})")
            if int(pos[b]) >= self._table_row_lens[b]:
                raise ValueError(f"row {b}: position {int(pos[b])} beyond its slot table ({self._table_row_lens[b]} entries)")
        self._ids_host.copy_(ids)
        self._pos_host.copy_(pos)
        # BLOCKING device copies (deliberate): non_blocking=True from the
        # pinned staging raced the kernel queue on this stack and corrupted
        # per-step tokens/positions under deep async pipelines (diagnosed:
        # appends landing in wild slots / K rows zeroed, fully reproducible
        # with sync-per-step). The copies are two <=8*B-byte transfers; the
        # per-step block drains a queue that decode steps sync anyway.
        self.ids.copy_(self._ids_host, non_blocking=False)
        self.pos.copy_(self._pos_host, non_blocking=False)
        qwen3_megakernel[(self.workers,)](
            self.ids,
            self.pos,
            self.table,
            self.tok,
            self.final_g,
            self.cos,
            self.sin,
            self.ln1,
            self.qkv_w,
            self.qn,
            self.kn,
            self.op_w,
            self.ln2,
            self.gu_w,
            self.down_w,
            self._k_launch,
            self._v_launch,
            self.ws,
            self.logits,
            self.barrier_counter,
            self._bar_base,
            cfg.layers,
            cfg.hidden * ((cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim),
            (cfg.heads * cfg.head_dim) * cfg.hidden,
            cfg.hidden * (2 * cfg.intermediate),
            cfg.intermediate * cfg.hidden,
            cfg.hidden,
            self._layer_stride,
            B=self.batch,
            C=cfg.hidden,
            H=cfg.heads,
            KVH=cfg.kv_heads,
            D=cfg.head_dim,
            F=cfg.intermediate,
            V=cfg.vocab,
            SCAP=p.SCAP,
            QKVW=(cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim,
            HD=cfg.heads * cfg.head_dim,
            F2=2 * cfg.intermediate,
            WS_LAYER=p.ws_layer,
            O_HIDDEN_A=p.offsets["hidden_a"],
            O_HIDDEN_B=p.offsets["hidden_b"],
            O_FINAL=p.offsets["final"],
            O_RMS1=p.offsets["rms1"],
            O_QKV=p.offsets["qkv"],
            O_QN=p.offsets["qn"],
            O_KN=p.offsets["kn"],
            O_RQ=p.offsets["rq"],
            O_RK=p.offsets["rk"],
            O_SCORES=p.offsets["scores"],
            O_PROBS=p.offsets["probs"],
            O_CTX=p.offsets["ctx"],
            O_ATTN=p.offsets["attn"],
            O_RMS2=p.offsets["rms2"],
            O_GU=p.offsets["gu"],
            O_ACT=p.offsets["act"],
            O_DOWN=p.offsets["down"],
            EPS=cfg.rms_eps,
            SCALE=cfg.attention_scale,
            TILE=16,
            BK=256,
            BT=self.btcap,
            BTR=_pow2(p.SCAP),
            ELEM=self.elem,
            num_warps=8,  # §7.1: T = 256 threads per persistent worker
        )
        self._bar_base += self.num_barriers * self.workers
        if check_counter:
            torch.cuda.synchronize()
            got = int(self.barrier_counter[0])
            if got != self._bar_base:
                raise RuntimeError(f"barrier counter {got} != expected {self._bar_base} (progress/protocol failure)")
        return self.logits

    def _as_row(self, val, like):
        t = torch.as_tensor(val)
        t = t.reshape(-1)
        if t.numel() != self.batch:
            raise ValueError(f"expected {self.batch} rows, got {t.numel()}")
        return t.to(like.dtype)


def attach_megakernel_pool(
    socket_path: str,
    engine_id: str,
    gpu: int,
    *,
    layers: int,
    max_tokens: int,
    kv_heads: int,
    head_dim: int,
    dtype_label: str = "torch.bfloat16",
    pool=None,
):
    """Attach a daemon-owned KV pool in the megakernel's packed layout.

    Requests **one packed buffer per side** (``[layers * max_tokens,
    kv_heads, head_dim]``) instead of the floe-canonical per-layer buffer
    plans: the daemon's VMM placement of individually planned buffers is
    not uniformly strided, while a single planned buffer per side gives the
    kernel one base pointer + a uniform layer stride by construction. The
    slot granularity is unchanged (one token's K/V per slot), so
    ``Admission`` slot tables address it identically.

    ``pool`` optionally carries a floe-side ``DensePoolLeases``
    session (the unified BlockTable's dense view). When given, the physical
    attach is unchanged but the session's geometry is validated against the
    pool being attached, so lease extents can never describe a different
    pool than the tensors the megakernel writes. Ships dark: ``None``
    (the default) keeps today's behavior exactly.

    Returns ``(k_tensor, v_tensor)``: non-owning torch views over
    daemon-owned device memory (CUDA IPC import), shape
    ``[layers, max_tokens, kv_heads, head_dim]``.
    """
    import os

    import torch

    from kvaas_runtime import CudaVmm, allocate_device_pool
    from kvaas_runtime import kv_pool_import as kpi
    from kvaas_runtime.kv_pool_import import import_pool_buffers

    if pool is not None:
        if pool.kv_width != kv_heads * head_dim:
            raise ValueError(
                f"lease session kv_width {pool.kv_width} != pool row width {kv_heads * head_dim}"
            )
        if max_tokens % pool.page_tokens:
            raise ValueError(
                f"--max-total ({max_tokens}) must be a whole multiple of the "
                f"{pool.page_tokens}-token dense page"
            )
        if pool.gpu != gpu:
            raise ValueError(f"lease session gpu {pool.gpu} != attach gpu {gpu}")

    itemsize = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}[dtype_label]
    dims = (layers * max_tokens, kv_heads, head_dim)
    plans = [
        kpi.BufferPlan("k.all", dims, itemsize, dtype_label),
        kpi.BufferPlan("v.all", dims, itemsize, dtype_label),
    ]

    class _Deps:
        pass

    deps = _Deps()
    deps.torch = torch
    deps.cuda = CudaVmm(libpath=os.environ.get("KVAAS_CUDA_DRV_PATH"))

    response = allocate_device_pool(
        socket_path,
        engine_id=engine_id,
        gpu=gpu,
        buffers=[(p.name, p.nbytes) for p in plans],
    )
    records = import_pool_buffers(deps, gpu, plans, response.get("buffers") or [])
    by_name = {rec["plan"].name: rec for rec in records}
    if set(by_name) != {"k.all", "v.all"}:
        raise RuntimeError(f"attach_megakernel_pool: unexpected buffers {sorted(by_name)}")
    k = by_name["k.all"]["tensor"].view(layers, max_tokens, kv_heads, head_dim)
    v = by_name["v.all"]["tensor"].view(layers, max_tokens, kv_heads, head_dim)
    # Retain the import records on the tensors so the IPC mappings stay open
    # for the tensors' lifetime (the records own the handles).
    k._kvaas_records, v._kvaas_records = records, records
    return k, v


# ---------------------------------------------------------------------------
# Qwen3.8-27B feasibility building blocks (hybrid GDN target, FP8 weights).
#
# These task bodies implement the pieces the dense-Qwen3 megakernel lacks
# for the 27B target; they are validated standalone (tests/test_megakernel_27b.py)
# against the real checkpoint tensors and the repo's GatedDeltaNet reference:
#
# * ``_t_gemv_fp8`` — GEMV over DeepSeek-style block-quantized weights
#   (F8_E4M3 [N, K] + bf16 ``weight_scale_inv`` [N/128, K/128]; BK == the
#   128-wide quant block, 16-column output tiles as in §6.4);
# * ``_t_gdn_conv`` — the GDN short-conv state update (silu FIR over a
#   [3, C] channel state + the new mixed qkv row);
# * ``_t_gdn_heads`` — the per-value-head gated delta rule (normalize q/k,
#   decay/outer-product state update, per-head RMSNorm + z-gate), one task
#   per value head over its [hv, hk] fp32 state slice.
# ---------------------------------------------------------------------------


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
