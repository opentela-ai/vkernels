"""Dense block-fp8 decode GEMV (small M) — one Triton launch per matrix.

The GLM-5.3-Flash checkpoint stores most non-expert weights as fp8-e4m3fn
with DeepSeek-style per-128x128 block scales, but the serving stack
dequantizes them to bf16 at load time, so every M==1 decode linear reads
2x the bytes the checkpoint needs. This op keeps the weights fp8-resident
and dequantizes IN the kernel:

    y[m, n] = sum_k x[m, k] * bf16(fp8(w[n, k]) * scale[n//128, k//128])

with fp32 accumulation and ONE bf16 round at the output — the same
rounding class as the bf16 ``dense_gemv`` it replaces. The per-element
weight rounding point matches floe's loader dequant
(``dequant_block_fp8``: fp32 product -> one bf16 cast), so the dequantized
weight VALUES are bit-identical to the bf16-resident path; only the fp32
reduction order differs. e4m3fn -> bf16 is exact (3 mantissa bits), so the
dequant product is exact in fp32 before the bf16 round.

Weight bytes on the decode GEMV slice halve: the fp8-resident family
(DSA q_a/q_b/kv_a/o_proj, dense-MLP first layers, shared experts) reads
~1 B/element instead of ~2.

Kernel shape: one program per ROWS output rows, K streamed in 128-wide
tiles (the scale block); per (row, k-tile) the block scale is a scalar
gather. The e4m3fn bytes are loaded as uint8 and bitcast to
``tl.float8e4nv`` — the hardware decode (checkpoint e4m3FN == e4m3nv
semantics on NVIDIA), the same ~8x-faster-than-bit-twiddle trick as
``glm_expert_gemv._expert_gemv_native``. M is a masked constexpr power of
two (decode buckets 1/2/4 and DFlash2 verify blocks up to
``GLM53_MOE_DECODE_MAX_TOKENS``); the weight tile is read ONCE per
program for all M rows, so the kernel stays bandwidth-bound up to the cap.

Torch and Triton load lazily. Inputs are read-only; inference-only.
Graph capture: single launch, no host syncs; per-(M, O, I) compilation
happens in the eager warmup passes that precede every capture.
"""

from ._dispatch import OpNotEligible
from functools import lru_cache


def _m_cap() -> int:
    """Row cap: mirrors glm_expert_gemv's ``GLM53_MOE_DECODE_MAX_TOKENS``
    bridge (floe's ``_sync_vkernels_env`` exports the moe_decode_max_tokens
    knob here), so decode and verify blocks route consistently."""
    import os

    try:
        return max(2, int(os.environ.get("GLM53_MOE_DECODE_MAX_TOKENS", "2")))
    except ValueError:
        return 2


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _gemv_fp8(
        X, W, S, Y,
        M, O, I,
        MP: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK_I: tl.constexpr,
    ):
        """Y[m, n] for m < M, n = pid*ROWS + arange(ROWS).

        W: fp8-e4m3fn [O, I] read through a uint8 view (direct fp8-pointer
        loads are rejected by this Triton); S: fp32 [O//128, I//128] block
        scales; X: bf16 [M, I]. The reduction axis streams in BLOCK_I
        chunks; each chunk carries BLOCK_I//128 scale groups, applied with
        the compact gather + reshape-broadcast of glm_expert_gemv's native
        CUDA GEMV (a per-element scale gather costs more bytes than the
        fp8 weights themselves when uncoalesced). The (v * s) product is
        rounded to bf16 before the fp32 accumulation — the
        loader-dequant rounding point.
        """
        pid = tl.program_id(0)
        rows = pid * ROWS + tl.arange(0, ROWS)
        rmask = rows < O
        m = tl.arange(0, MP)
        mmask = m < M
        acc = tl.zeros((ROWS, MP), dtype=tl.float32)
        NG: tl.constexpr = BLOCK_I // 128
        srow = rows // 128
        for k0 in range(0, I, BLOCK_I):
            cols = k0 + tl.arange(0, BLOCK_I)
            cmask = (rmask[:, None]) & (cols[None, :] < I)
            raw = tl.load(
                W + rows[:, None] * I + cols[None, :],
                mask=cmask,
                other=0,
            )
            w = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            # compact scale load: [ROWS, BLOCK_I//128] gathered once (the
            # per-element scale gather costs more bytes than the fp8
            # weights themselves when uncoalesced)
            gs = k0 // 128 + tl.arange(0, NG)
            s = tl.load(
                S + srow[:, None] * (I // 128) + gs[None, :],
                mask=rmask[:, None] & (gs[None, :] < I // 128),
                other=0.0,
            )
            # group-domain dequant: reshape the LOADED tile to the scale
            # grid and broadcast the scale over its 128-wide groups there
            # (reshape(broadcast_to(s)) is NOT safe — reshaping a stride-0
            # view has no defined row-major order in this Triton). The
            # dequantized tile reshapes back to [ROWS, BLOCK_I] — a real
            # (materialized) tile, so the row-major reshape is exact.
            wv = (tl.reshape(w, (ROWS, NG, 128)) * s[:, :, None]).to(
                tl.bfloat16).to(tl.float32)  # [ROWS, NG, 128]
            wv = tl.reshape(wv, (ROWS, BLOCK_I))
            # bf16-rounded weights (bit-identical to the loader's
            # dequant_block_fp8 rounding point) times bf16 activations,
            # fp32-accumulated
            x = tl.load(
                X + m[:, None] * I + cols[None, :],
                mask=mmask[:, None] & (cols[None, :] < I),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(wv[:, None, :] * x[None, :, :], axis=2)
        tl.store(
            Y + m[None, :] * O + rows[:, None],
            acc.to(tl.bfloat16),
            mask=rmask[:, None] & mmask[None, :],
        )

    return tl, triton, _gemv_fp8


def _tiles(o: int) -> tuple[int, int]:
    """(ROWS, num_warps); ``VK_FP8_DENSE_GEMV_TILES='ROWS,warps'`` overrides
    (live single-shape tuning knob, the VK_FP8GEMM_TILES convention)."""
    import os

    cfg = os.environ.get("VK_FP8_DENSE_GEMV_TILES", "")
    if cfg:
        parts = [int(x) for x in cfg.split(",")]
        if len(parts) == 2:
            return parts[0], parts[1]
    return (4 if o % 4 == 0 else 1), 4


def dense_gemv_fp8(x, w8, scales):
    """Return bf16 [M, O] for bf16 x [M, I] (or [I]), fp8-e4m3fn w8 [O, I]
    and fp32 block scales [O//128, I//128].

    M <= the ``GLM53_MOE_DECODE_MAX_TOKENS`` cap (the padded constexpr is
    the next power of two). Anything outside the contract raises
    ``OpNotEligible`` so the caller can fall back to its BLAS/dequant path.
    """
    import torch

    if x.ndim not in (1, 2) or w8.ndim != 2 or scales.ndim != 2:
        raise OpNotEligible("expected x [M, I] (or [I]), w8 [O, I], scales [O//128, I//128]")
    m = 1 if x.ndim == 1 else x.shape[0]
    o, i = w8.shape
    if x.shape[-1] != i:
        raise OpNotEligible(f"shape mismatch x{tuple(x.shape)} w8{w8.shape}")
    cap = _m_cap()
    if m < 1 or m > cap:
        raise OpNotEligible(f"M={m} outside the 1..{cap} decode-GEMV cap")
    if i % 128 or o % 128 or i <= 0 or o <= 0:
        raise OpNotEligible("O and I must be positive multiples of 128 (block-scale grid)")
    if scales.shape != (o // 128, i // 128):
        raise OpNotEligible(
            f"scale shape {tuple(scales.shape)} != {(o // 128, i // 128)}")
    if x.dtype != torch.bfloat16 or w8.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        raise OpNotEligible("requires bf16 activations, e4m3fn weights, fp32 scales")
    if not (x.is_cuda and w8.is_cuda and x.device == w8.device and scales.device == w8.device):
        raise OpNotEligible("inputs must share a GPU device")
    if not (x.is_contiguous() and w8.is_contiguous() and scales.is_contiguous()):
        raise OpNotEligible("inputs must be contiguous")
    tl, triton, kern = _kernel()
    rows, warps = _tiles(o)
    block_i = triton.next_power_of_2(min(i, 4096))
    mp = triton.next_power_of_2(m)
    y = torch.empty(m, o, device=x.device, dtype=torch.bfloat16)
    with torch.cuda.device(x.device):
        kern[(triton.cdiv(o, rows),)](
            x.reshape(m, i),
            w8.view(torch.uint8),
            scales,
            y,
            m,
            o,
            i,
            MP=mp,
            ROWS=rows,
            BLOCK_I=block_i,
            num_warps=warps,
        )
    return y


def dense_gemv_fp8_reference(x, w8, scales):
    """Oracle: loader-dequant (bf16 weights) + fp32 GEMM, bf16 output.

    Mirrors the serving-stack reference exactly: dequant_block_fp8's
    fp32-product-then-bf16-cast, then F.linear in bf16 (fp32 internal
    accumulation). The kernel differs only in reduction order."""
    import torch

    o, i = w8.shape
    w = (
        w8.float().view(o // 128, 128, i // 128, 128)
        * scales.float()[:, None, :, None]
    ).reshape(o, i).to(torch.bfloat16)
    x2 = x.reshape(-1, i) if x.ndim == 2 else x.reshape(1, i)
    return torch.nn.functional.linear(x2, w).reshape(x.shape[:-1] + (o,))
