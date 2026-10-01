"""Dense BF16 decode GEMV (M <= m_cap, pinned per shape) — one Triton
launch per matrix.

cuBLAS's skinny-GEMM heuristic picks a splitK kernel for every M==1
linear at decode: the GLM-5.3-Flash bs=1 step runs ~460 nvjet splitK
launches plus ~465 ``splitKreduce`` launches (~4.1 ms). A plain one-
program-per-row (or per-4-rows) Triton GEMV with FP32 accumulation beats
that heuristic on every dense decode shape measured on GH200
(graph-replayed micro-bench, jobs 3460700/3460716/3460776): 2-4.7x on
the small-O shapes the heuristic misfires on (shared experts / b_proj /
kv_a at ~700 GB/s) and ~1.5x on the big ones (kda q/k/v 6.99 -> 4.0 us
at ~4.2 TB/s). Only [8192, 512]-class shapes tie; the dispatch sends
everything and eats the noise-level delta to keep call sites simple.

Row-group / block / warps table (best-of the graph-replayed autotune sweep on
GH200, job 3462837; keyed by the production (O, I) with >4% gains — everything
else keeps the fallback heuristic below): four rows per program when O is large
(fewer x re-reads, bandwidth-bound) or the reduction axis is narrow (I <= 512);
one row per program otherwise (more programs in flight, latency-bound).
num_warps=4 wins on wide reductions (I >= 512, O >= 2048) — fewer warps, longer
per-warp streams; narrow-I shapes want ROWS=8 with warps=8.

Numerics: BF16 weights, FP32 product reduction rounded once to BF16 —
the same kernel class as the ``skinny_gemv`` torch.mv policy (perf-only
per the correctness plan; the reduction order can differ from BLAS).

Torch and Triton load lazily. Inputs are read-only; inference-only, no
autograd backward. Graph capture: warm up eagerly per (O, I) shape first
(the kernel compiles per (O, I, ROWS) on first launch; the GLM decode
step has ~15 distinct shapes -> ~15 compilations).
"""

from ._dispatch import OpNotEligible
from functools import lru_cache

# (O, I) -> (ROWS, BLOCK_I, num_warps); graph-replayed autotune, GH200
# (clariden job 3462837). Only shapes where the best beat the heuristic by
# >4% are pinned: kv_b -22%, dense_gu_fused -34%, dense_dn -12%, g_b/f_b
# -10%, dense_gu -8%, kda_q_ctl -5%, kda_o -4%, q_b/wq_b -5..-10% (warps).
#
# H100 (sgs-gpu07) refinements from the wrapper-confirmed tile sweep
# (/tmp/gemv-sweep_results.json + wrapper-level /tmp/gemv-confirm_results.json,
# GLM-5.3-Flash TP4 decode shapes, M=1): only confirm-corroborated >4% wins are
# merged — mHC mix (24,16384) wants the full-I unmasked BLOCK_I (-25%), KDA
# q/k/v (2048,4096) moves to rows=4 (-4.8%), KDA o_proj (4096,2048) to
# rows=8/BLOCK_I=256 (-4.5%, ~4.2 TB/s effective — the 16 MB weight is L2
# resident: above the 3.35 TB/s HBM floor but under the L2 ceiling).
# (16,4096)/(32,4096)/(128,4096)/(2048,128) and warps on (4096,1536) measured
# at/below the noise floor: the heuristic or the existing pin already stands.
#
# M > 1 (decode B in 2..8) H100 pins from the dense-m8 tile sweep, keyed
# (O, I, M) -> (ROWS, BLOCK_I, num_warps): only wrapper-confirm-corroborated
# cells are listed (see _CFG_M comments). Lookup order in dense_gemv for
# M > 1: _CFG_M[(O, I, M)] > heuristic. M=1 never consults this table.
#
# DENSE-M8 CONFIRM VERDICT (m8_confirm_results.json, archived under
# scratch/dense-m8/): the confirm pass measured cublas + default only for
# every bf16 cell — NO bf16 pin candidate was ever wrapper-confirmed at
# M > 1, so no bf16 M > 1 pin is corroborated and this table stays
# INTENTIONALLY EMPTY. The corroborated winner for bf16 M in 2..8 is cuBLAS
# on every swept shape (floe _skinny_gemm already keeps M > 1 on F.linear).
# Do NOT route bf16 M > 1 through the heuristic below as a substitute: at
# the wide-O shapes it is a large regression vs cuBLAS at the wrapper level
# (e.g. (2048,4096) m=8: 46.2 us vs 7.6; (38720,4096) m=8: 278 us vs 94).
_CFG_M = {}

_CFG = {
    (4096, 1536): (4, 2048, 4),
    (4096, 3072): (1, 4096, 4),
    (4096, 2048): (8, 256, 4),
    (8192, 512): (8, 512, 4),
    (3072, 4096): (2, 4096, 4),
    (6144, 4096): (4, 4096, 4),
    (2048, 128): (8, 128, 8),
    (2048, 4096): (4, 4096, 4),
    (24, 16384): (1, 16384, 8),
}

# MI300A (gfx942) overrides — CUDA-graph-replayed autotune on beverin
# (E13 job 647045, 8-call graphs x 50 replays; SCR/k8e13/gemv_best_table.json).
# Only shapes beating the GH200 pin by >7% are overridden ((4096, 4096) at 5.3%
# rides the noise floor but pays for itself across the ~15-site family):
# the MI300A win pattern wants MORE programs in flight than GH200 — rows=4
# with wide reductions, rows=1 only on the widest O. Measured per-shape wins
# (graph GPU time): (4096,2048) 5.13->3.95us, (2048,4096) 4.62->3.48,
# (8192,512) 4.91->3.64, (4096,3072) 7.67->6.45, (3072,4096) 5.98->5.09,
# (4096,512) 3.32->2.93, (2048,128) 3.72->3.38, (128,4096) 3.57->3.28,
# (4096,4096) 9.26->8.77, (6144,4096) 16.39->15.17. Full 15-shape step total
# (vs cuBLAS splitK): 685.7 -> 90.0 us = 7.6x.
_CFG_MI300A = {
    (4096, 4096): (4, 4096, 2),
    (4096, 3072): (4, 4096, 8),
    (4096, 2048): (4, 2048, 8),
    (4096, 512): (2, 512, 2),
    (8192, 512): (4, 512, 2),
    (3072, 4096): (4, 4096, 8),
    (6144, 4096): (1, 4096, 2),
    (2048, 128): (4, 128, 8),
    (2048, 4096): (1, 4096, 2),
    (128, 4096): (8, 4096, 8),
}


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv_bf16(X, W, Y, O: tl.constexpr, I: tl.constexpr,
                   ROWS: tl.constexpr, BLOCK_I: tl.constexpr):
        """Y[row] = sum_i W[row, i] * X[i], fp32 accumulation.

        One program handles ROWS consecutive output rows and streams the
        reduction axis in BLOCK_I chunks. Two compile-time elisions: the
        all-true tail mask (BLOCK_I == I — it otherwise costs ~0.6 us per
        launch on GH200) and the row-bound mask (the wrapper only picks
        ROWS=4 when O % 4 == 0; ROWS=1 has no tail by construction)."""
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        acc = tl.zeros((ROWS,), tl.float32)
        for k0 in range(0, I, BLOCK_I):
            cols = k0 + tl.arange(0, BLOCK_I)
            if BLOCK_I == I:
                w = tl.load(W + row[:, None] * I + cols[None, :]).to(tl.float32)
                x = tl.load(X + cols).to(tl.float32)
            else:
                m = cols < I
                w = tl.load(W + row[:, None] * I + cols[None, :],
                            m[None, :], other=0).to(tl.float32)
                x = tl.load(X + cols, m, other=0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        tl.store(Y + row, acc.to(tl.bfloat16))

    @triton.jit
    def _gemv_bf16_m(X, W, Y, M, O: tl.constexpr, I: tl.constexpr,
                     MP: tl.constexpr, ROWS: tl.constexpr,
                     BLOCK_I: tl.constexpr):
        """Y[m, row] = sum_i W[row, i] * X[m, i] for m < M, fp32 accum.

        The M > 1 sibling of ``_gemv_bf16`` (same ROWS/BLOCK_I scheme; the
        weight tile is read ONCE per program and reused across the MP
        rows, so the kernel stays weight-bandwidth-bound up to the cap —
        the same structure as glm_dense_fp8_gemv's M-batched kernel).
        MP is the padded constexpr power of two; the M mask folds away
        when M == MP. The row-bound mask stays elided (the wrapper only
        picks ROWS > 1 when O % ROWS == 0).
        """
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        m = tl.arange(0, MP)
        mmask = m < M
        acc = tl.zeros((ROWS, MP), tl.float32)
        for k0 in range(0, I, BLOCK_I):
            cols = k0 + tl.arange(0, BLOCK_I)
            imask = cols < I
            if BLOCK_I == I:
                w = tl.load(W + row[:, None] * I + cols[None, :]).to(tl.float32)
            else:
                w = tl.load(W + row[:, None] * I + cols[None, :],
                            imask[None, :], other=0).to(tl.float32)
            x = tl.load(X + m[:, None] * I + cols[None, :],
                        mmask[:, None] & imask[None, :],
                        other=0).to(tl.float32)
            acc += tl.sum(w[:, None, :] * x[None, :, :], axis=2)
        tl.store(Y + m[None, :] * O + row[:, None], acc.to(tl.bfloat16),
                 mask=mmask[None, :])

    return tl, triton, _gemv_bf16, _gemv_bf16_m


def dense_gemv(x, w, m_cap=None):
    """Return BF16 [M, O] (or [O] for 1-D x) for x [M, I] and w [O, I].

    ``m_cap`` bounds the accepted M (default 1 — the historical decode
    GEMV contract; callers that route M > 1 here pass their decode cap,
    the ``dense_gemv_fp8`` convention). M == 1 rides the pinned single-
    token kernel and config table exactly as before; M in 2..m_cap rides
    the M-batched sibling (the weight tile is read once for all M rows).
    Contiguous CUDA BF16 inputs are required; anything else raises
    ``OpNotEligible`` so the caller can fall back to its BLAS path."""
    import torch

    if x.ndim not in (1, 2) or w.ndim != 2:
        raise OpNotEligible("expected x [I] or [M, I] and w [O, I]")
    m = 1 if x.ndim == 1 else x.shape[0]
    cap = 1 if m_cap is None else max(1, int(m_cap))
    if m < 1 or m > cap:
        raise OpNotEligible(f"M={m} outside the 1..{cap} dense-GEMV cap")
    if x.shape[-1] != w.shape[1]:
        raise OpNotEligible(f"shape mismatch x{x.shape} w{w.shape}")
    if x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise OpNotEligible("requires BF16 activations and weights")
    if not (x.is_cuda and w.is_cuda and x.device == w.device):
        raise OpNotEligible("inputs must share a GPU device")
    if not (x.is_contiguous() and w.is_contiguous()):
        raise OpNotEligible("inputs must be contiguous")
    o, i = w.shape
    if o <= 0 or i <= 0:
        raise OpNotEligible("empty GEMV")
    tl, triton, kern, kern_m = _kernel()
    if m == 1:
        cfg = _CFG.get((o, i))
        if cfg is not None and torch.version.hip:
            cfg = _CFG_MI300A.get((o, i), cfg)
        if cfg is not None:
            rows, block_i, warps = cfg
        else:
            rows = 4 if (o % 4 == 0 and (o > 2048 or i <= 512)) else 1
            block_i = triton.next_power_of_2(min(i, 4096))
            warps = 8
        y = torch.empty(1, o, device=x.device, dtype=torch.bfloat16)
        with torch.cuda.device(x.device):
            kern[((o + rows - 1) // rows,)](
                x.reshape(i), w, y, o, i, rows, block_i,
                num_warps=warps,
            )
        return y if x.ndim == 2 else y[0]
    cfg = _CFG_M.get((o, i, m))
    if cfg is not None:
        rows, block_i, warps = cfg
    else:
        rows = 4 if (o % 4 == 0 and (o > 2048 or i <= 512)) else 1
        block_i = triton.next_power_of_2(min(i, 4096))
        warps = 4
    if rows > 1 and o % rows:
        rows = 1
    y = torch.empty(m, o, device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        kern_m[(o // rows,)](
            x.reshape(m, i), w, y, m, o, i,
            triton.next_power_of_2(m), rows, block_i,
            num_warps=warps,
        )
    return y
