"""Dense BF16 decode GEMV (M==1) — one Triton launch per matrix.

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

Row-group heuristic (best-of the measured sweep): four rows per program
when O is large (fewer x re-reads, bandwidth-bound) or the reduction
axis is narrow (I <= 512); one row per program otherwise (more programs
in flight, latency-bound).

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

    return tl, triton, _gemv_bf16


def dense_gemv(x, w):
    """Return BF16 [1, O] (or [O] for 1-D x) for x [1, I] and w [O, I].

    Contiguous CUDA BF16 inputs are required; anything else raises
    ``OpNotEligible`` so the caller can fall back to its BLAS path."""
    import torch

    if x.ndim not in (1, 2) or w.ndim != 2:
        raise OpNotEligible("expected x [I] or [1, I] and w [O, I]")
    if x.shape[-1] != w.shape[1] or (x.ndim == 2 and x.shape[0] != 1):
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
    tl, triton, kern = _kernel()
    rows = 4 if (o % 4 == 0 and (o > 2048 or i <= 512)) else 1
    block_i = triton.next_power_of_2(min(i, 4096))
    y = torch.empty(1, o, device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        kern[((o + rows - 1) // rows,)](
            x.reshape(i), w, y, o, i, rows, block_i,
            num_warps=8,
        )
    return y if x.ndim == 2 else y[0]
