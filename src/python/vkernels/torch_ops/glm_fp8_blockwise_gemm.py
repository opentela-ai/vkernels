"""GLM-5.3-Flash fp8 blockwise-scaled grouped GEMM (sm_90, CuTe DSL).

The MoE layers keep routed experts in checkpoint fp8-e4m3 with per-128x128
DeepSeek-scheme block scales (``w_bf16 = w_fp8 * S[n//128, k//128]``). The
decode/verify path streams them through :mod:`glm_expert_gemv` (one GEMV per
(token, expert) pair, bandwidth-bound at small M); PREFILL and batched serving
need a tensor-core grouped GEMM instead. This module provides it in three
layers:

1. Activation quantization — e4m3 with per-group amax/448 scaling (required
   because Hopper WGMMA needs BOTH operands in fp8 — no bf16 x fp8 mixed MMA
   on sm_90), at two granularities:
   ``quantize_activations_fp8`` — per-row per-128-group (the DeepSeek/VLLM
   serving contract; feeds the fnuz-conversion grouped path), and
   ``quantize_activations_fp8_blockwise`` — per-128x128-BLOCK, matching the
   weight scale grid (feeds ``fp8_blockwise_gemm``; degenerates to a
   per-tensor scale for M <= 128).
2. ``fp8_blockwise_gemm`` — the single-expert primitive:
   ``D[m,n] = sum_kb Sa[m//128, kb] * Sb[n//128, kb] *
   (sum_{k in kb} A_fp8[m,k] * B_fp8[n,k])``, BF16 output. A scales share
   the weights' 128-row block grid: [ceil(M/128), K/128]. Dispatches to the
   CuTe DSL kernel when available/importable, else to a pure-torch reference
   path with identical semantics (block-dequant + per-block-scaled
   accumulation).
3. ``glm_moe_grouped_gemm`` — the MoE-shaped orchestration: host-side
   sort-by-expert of the (token, slot) routing, one blockwise GEMM per ACTIVE
   expert over its gathered activations, epilogue applying routing weights.
   The persistent fused grouped kernel (ptr-array scheduling inside one
   launch) replaces the per-expert loop later WITHOUT changing this API.

Numerics contract (vs the bf16-activation GEMV path): activation e4m3
quantization adds ~1e-2-class relative noise on top of the weight-dequant
rounding the GEMV already does. The MoE parity gate (floe loop) therefore
uses a looser tolerance for this path; logit-level effects are measured by
the NLL/quality gates, as with any fp8-activation serving stack.

The CuTe DSL kernel (``HopperFP8BlockwiseGemmKernel``) is a lean adaptation
of cutlass's ``dense_gemm_fp8_2xacc.py`` example: warp-specialized
TMA+WGMMA with the two-level (2xAcc) accumulator, where the promotion every
k-tile (tile_k == 128 == the scale block, i.e. 4 WGMMA k-blocks) multiplies
``accum_temp`` by the per-(tile, k-block) scalar ``Sa[m_blk, kb] *
Sb[n_blk, kb]``. VALIDATION: first compile/run happens on a GH200 job
(``tests/python/test_glm_fp8_blockwise_gemm.py`` gates on CUDA + the DSL);
the torch path is the oracle.

Requires the optional ``cute`` extra: ``uv pip install '.[cute]'``
(nvidia-cutlass-dsl). Import of this module stays dependency-free; only the
kernel path imports the DSL lazily.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache

from ._dispatch import OpNotEligible

# One-shot log for the triton-blockwise -> torch-oracle fallback in
# fp8_blockwise_gemm: the fallback is correct by contract, but a silent one
# would hide a genuine kernel failure behind an (only) slower result.
_TRITON_FALLBACK_WARNED = threading.Event()


# ---------------------------------------------------------------------------
# 1. activation quantization (DeepSeek per-token per-128-group e4m3)
# ---------------------------------------------------------------------------


def quantize_activations_fp8(x, group_size: int = 128):
    """BF16 [M, K] -> (fp8-e4m3 [M, K], fp32 scales [M, ceil(K/group)]).

    Per-token per-group scaling with the e4m3 finite amax (448): each group's
    scale = amax/448 (min 1e-4 for denormal safety), the fp8 payload is
    x/scale. Mirrors vLLM's per-token-group quantization.
    """
    import torch

    if x.ndim != 2:
        raise ValueError("expected x [M, K]")
    m, k = x.shape
    if group_size <= 0 or k % group_size:
        # tail groups are not supported by the kernel's power-of-2 tiling
        raise ValueError(f"K ({k}) must be a multiple of group_size ({group_size})")
    xg = x.float().view(m, k // group_size, group_size)
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = (amax / 448.0).clamp_min(1e-12)
    q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, k)
    return q, scale.squeeze(-1).contiguous()


def quantize_activations_fp8_blockwise(x, block: int = 128):
    """BF16 [M, K] -> (fp8-e4m3 [M, K], fp32 scales [ceil(M/block), K/block]).

    Per-128x128-BLOCK activation quantization, matching the weight scale
    granularity so the sm_90 kernel applies BOTH scales as one scalar per
    (128-tile, k-block) -- no per-fragment scale addressing. Coarser than the
    per-row DeepSeek contract (quantize_activations_fp8); for the MoE verify
    path (M <= 128 per expert) this degenerates to a per-tensor scale.
    """
    import torch

    if x.ndim != 2:
        raise ValueError("expected x [M, K]")
    m, k = x.shape
    if k % block:
        raise ValueError(f"K ({k}) must be a multiple of {block}")
    mb = (m + block - 1) // block
    pad = mb * block - m
    xg = x.float()
    if pad:
        xg = torch.nn.functional.pad(xg, (0, 0, 0, pad))
    xg = xg.view(mb, block, k // block, block)
    amax = xg.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    scale = (amax / 448.0).clamp_min(1e-12)
    q = (xg / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.view(-1, k)[:m], scale.view(mb, k // block)


# ---------------------------------------------------------------------------
# 2. single-expert blockwise GEMM primitive (kernel + torch oracle)
# ---------------------------------------------------------------------------


def fp8_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out_dtype=None):
    """D[m,n] = sum_kb Sa[m//128,kb] * Sb[n//128,kb] * (sum_{k in kb} A[m,k]*B[n,k]).

    A: fp8-e4m3 [M, K] row-major with 128x128-block scales
       [ceil(M/128), K/128] fp32 (quantize_activations_fp8_blockwise).
    B: fp8-e4m3 [N, K] row-major (weights, transposed-use) with scales
       [N/128, K/128] fp32 (the checkpoint layout).
    Returns BF16 [M, N] (or ``out_dtype``).
    """
    import torch

    if a_fp8.dtype != torch.float8_e4m3fn or b_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError("both operands must be fp8-e4m3")
    m, k = a_fp8.shape
    n, k2 = b_fp8.shape
    if k != k2 or k % 128 or n % 128:
        raise ValueError(
            f"shape mismatch A[{m},{k}] B[{n},{k2}] (K,N multiples of 128)"
        )
    mb = (m + 127) // 128
    if a_scales.shape != (mb, k // 128) or b_scales.shape != (n // 128, k // 128):
        raise ValueError(
            f"scale shapes {a_scales.shape}/{b_scales.shape} do not match "
            f"(expected [{mb}, {k // 128}] / [{n // 128}, {k // 128}])"
        )
    out_dtype = out_dtype or torch.bfloat16
    out = torch.empty((m, n), device=a_fp8.device, dtype=out_dtype)

    kernel = _cute_kernel() if a_fp8.is_cuda else None
    if kernel is not None:
        kernel(a_fp8, a_scales, b_fp8, b_scales, out)
        return out
    if a_fp8.is_cuda and os.environ.get("VKERNELS_FP8_GEMM_BACKEND", "auto") in (
        "auto",
        "triton",
    ):
        try:
            import triton  # noqa: F401

            return _triton_backend(a_fp8, a_scales, b_fp8, b_scales, out)
        except Exception as exc:
            # no triton (or launch rejected) -> torch oracle. Behavior is
            # unchanged; log the first occurrence once so the fallback (and
            # any genuine kernel failure behind it) is visible.
            if not _TRITON_FALLBACK_WARNED.is_set():
                _TRITON_FALLBACK_WARNED.set()
                print(
                    f"[glm_fp8_blockwise_gemm] triton blockwise kernel "
                    f"unavailable or failed ({type(exc).__name__}: {exc}); "
                    f"torch reference path active",
                    flush=True,
                )
    return _torch_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out)


def _torch_blockwise_gemm(a_fp8, a_scales, b_fp8, b_scales, out):
    """Reference path: block-dequant + per-K-block scaled fp32 accumulation.
    Both scale grids are 128x128-block granular (scalar per tile x k-block)."""
    import torch

    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    kb = k // 128
    mb = (m + 127) // 128
    pad = mb * 128 - m
    # zero-pad the rows once so every 128-row block is dense on the scale grid
    a32 = (
        torch.nn.functional.pad(a_fp8.float(), (0, 0, 0, pad))
        if pad
        else a_fp8.float()
    )
    acc = torch.zeros((m, n), device=a_fp8.device, dtype=torch.float32)
    for b in range(kb):
        a_blk = a32[:, b * 128 : (b + 1) * 128].view(mb, 128, 128)
        a_blk = (a_blk * a_scales[:, b].view(mb, 1, 1)).view(mb * 128, 128)[:m]
        w_blk = b_fp8[:, b * 128 : (b + 1) * 128].to(torch.float32)
        w_blk = w_blk.view(n // 128, 128, 128) * b_scales[:, b].view(n // 128, 1, 1)
        acc[:, :] += a_blk @ w_blk.reshape(n, 128).T
    out.copy_(acc.to(out.dtype))
    return out


# ---------------------------------------------------------------------------
# 3. MoE orchestration (sort by expert -> per-active-expert blockwise GEMM)
# ---------------------------------------------------------------------------


def _route_counts(topk_index, n_experts, device):
    """Host-sync-free routing metadata.

    ``argsort`` is stable, so sorting the flattened (token, slot) expert ids
    groups the slots by expert in *ascending expert order*: expert ``e`` owns
    sorted-slot rows ``[seg[e], seg[e] + counts[e])`` where ``seg`` is the
    exclusive cumsum of ``counts``. Counting with a device-side scatter-add
    keeps every shape static, which is what makes the dispatch capturable in a
    CUDA graph — ``torch.unique_consecutive`` must sync to learn its own output
    size, and reading ``int(toff[-1])`` for a launch grid is a host read of a
    device scalar. Both are avoided here. ROCm's ``torch.bincount`` is NOT
    capture-safe (``hipErrorStreamCaptureUnsupported``, beverin job 644282), so
    the counts come from ``scatter_add_`` on a zeroed int64 row instead.
    """
    import torch

    t, k = topk_index.shape
    flat = topk_index.reshape(-1)
    tokens = torch.arange(t, device=device).repeat_interleave(k)
    order = torch.argsort(flat, stable=True)
    counts = torch.zeros(n_experts, dtype=torch.int64, device=device).scatter_add_(
        0, flat, torch.ones_like(flat)
    )
    seg = torch.cumsum(counts, 0) - counts
    return order, tokens[order], counts, seg


def _tile_map_static(counts, seg, bm, device, cap):
    """Per-TILE metadata for a grouped launch of *static* grid size ``cap``.

    ``sum_e ceil(c_e / bm) <= min(slots / bm + E, slots)`` for any routing, so ``cap``
    be computed from tensor shapes alone — no device read, no grid sync. Tiles
    past the real tile count get ``m = 0`` and ``r0 = 0``: every load and store
    in the grouped kernel is masked by ``m`` and it returns early when
    ``m == 0``, so the padding costs only a bounded number of empty blocks.
    """
    import torch

    n_e = counts.numel()
    tiles = (counts + bm - 1) // bm
    ends = torch.cumsum(tiles, 0)  # inclusive end tile index per expert
    toff = ends - tiles
    n_tiles = ends[-1]  # device scalar — deliberately never read on the host
    tid = torch.arange(cap, device=device)
    valid = tid < n_tiles
    e = torch.searchsorted(ends, tid, right=True).clamp(max=n_e - 1)
    local = tid - toff[e]
    m = torch.clamp(counts[e] - local * bm, min=0, max=bm)
    r0 = seg[e] + local * bm
    zero = torch.zeros_like(tid)
    return (
        e.to(torch.int64),
        torch.where(valid, r0, zero),
        torch.where(valid, m, zero),
    )


def _route(topk_index, device):
    """Stable sort of the (token, slot) routing pairs by expert id.

    Returns ``(order, sorted_tokens, uniq_experts, counts, offsets)``:
    expert ``uniq_experts[j]`` owns sorted slots ``[offsets[j], offsets[j+1])``
    (``counts[j]`` of them); ``order``/``sorted_tokens`` are aligned to that
    sorted-slot space.
    """
    import torch

    t, k = topk_index.shape
    flat = topk_index.reshape(-1)
    tokens = torch.arange(t, device=device).repeat_interleave(k)
    order = torch.argsort(flat, stable=True)
    uniq, counts = torch.unique_consecutive(flat[order], return_counts=True)
    offsets = torch.zeros(len(uniq) + 1, device=device, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, 0)
    return order, tokens[order], uniq, counts, offsets


def glm_moe_grouped_gemm(
    x,
    gate_up_fp8,
    gate_up_scales,
    down_fp8,
    down_scales,
    topk_index,
    topk_weights,
    swiglu_limit=7.0,
    intermediate_dtype=None,
):
    """Routed expert MLP via blockwise fp8 GEMMs (API-stable; the internal
    per-expert loop is replaced by the fused grouped kernel later).

    x:            BF16 [T, H]
    gate_up_fp8:  [E, 2I, H] fp8, gate_up_scales [E, 2I/128, H/128]
    down_fp8:     [E, H, I] fp8,  down_scales   [E, H/128, I/128]
    topk_index:   int64 [T, K];  topk_weights: [T, K]
    Returns BF16 [T, H] (weighted sum of the K routed experts per token).
    """
    import torch

    e_gate, two_i, h = gate_up_fp8.shape
    e_down, h2, i = down_fp8.shape
    t, k = topk_index.shape
    if two_i != 2 * i or h != h2 or e_gate != e_down:
        raise ValueError("expert stack shapes inconsistent")
    if x.shape != (t, h):
        raise ValueError(
            f"x {tuple(x.shape)} does not match routing [{t}, {k}] and hidden {h}"
        )

    sort_idx, token_sorted, uniq, _, offsets = _route(topk_index, x.device)

    y = torch.zeros((t, h), device=x.device, dtype=torch.float32)
    for j in range(len(uniq)):
        expert = int(uniq[j])
        rows = sort_idx[offsets[j] : offsets[j + 1]]
        toks = token_sorted[offsets[j] : offsets[j + 1]]
        xa = x[toks]  # [m_j, H]
        a8, asc = quantize_activations_fp8_blockwise(xa)
        gu = fp8_blockwise_gemm(a8, asc, gate_up_fp8[expert], gate_up_scales[expert])
        gate, up = gu[:, :i].float(), gu[:, i:].float()
        # SwiGLU with the GLM limit (mirrors floe's _swiglu)
        act = (
            (gate / swiglu_limit).sigmoid() * up * swiglu_limit
            if swiglu_limit
            else torch.nn.functional.silu(gate) * up
        )
        a8d, ascd = quantize_activations_fp8_blockwise(act.to(torch.bfloat16))
        dn = fp8_blockwise_gemm(
            a8d, ascd, down_fp8[expert], down_scales[expert]
        ).float()
        w = topk_weights.reshape(-1)[rows].float().unsqueeze(-1)
        y.index_add_(0, toks, dn * w)
    return y.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Triton backend (gfx942/MI300A + portable): per-128-K-block scaled fp8 GEMM
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _triton_kernel():
    """Decode-fused blockwise fp8 GEMM: per-K-block tl.dot with the block
    scales applied to the block partial sums (identical math to the torch
    oracle). The raw e4m3FN bytes are loaded as uint8 and cast through
    tl.float8e4nv — Triton's e4m3 decode — which is exact for the FN byte
    layout on both gfx942 and sm_90 backends."""
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _e4m3fn_to_f32(raw):
        # Manual E4M3FN bit-decode: gfx942's native fp8 conversions/convert
        # instructions are FNUZ (different bias/NaN encodings), so both the
        # bitcast-to-fp8-dtype path AND hardware fp8 MMA produce wrong
        # values for checkpoint e4m3fn bytes. Decode: exponent bits +120
        # into the f32 exponent, mantissa <<20, subnormals = m/512, the
        # 0x7F/0xFF encodings are NaN (never present in weights).
        r = raw.to(tl.int32)  # promote: shifts below exceed 8 bits
        e = (r >> 3) & 15
        m = r & 7
        bits = ((e + 120) << 23) | (m << 20)
        v = bits.to(tl.float32, bitcast=True)
        v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, v)
        sign = tl.where((r & 128) != 0, -1.0, 1.0)
        return v * sign

    @triton.jit
    def _blockwise_gemm(
        A,
        SA,
        B,
        SB,
        D,
        M,
        N,
        K,
        sam,
        sakb,
        sbn,
        sbkb,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // 128):
            rk = kb * 128 + tl.arange(0, 128)
            # decode to fp32, round ONCE to bf16 for the tensor-core dot:
            # e4m3fn -> bf16 is EXACT (3 mantissa bits fit in 8), so the
            # dot sees the same values the fp32 oracle does.
            a = _e4m3fn_to_f32(
                tl.load(
                    A + rm[:, None] * K + rk[None, :],
                    mask=rm[:, None] < M,
                    other=0,
                )
            ).to(tl.bfloat16)
            b = _e4m3fn_to_f32(
                tl.load(
                    B + rn[:, None] * K + rk[None, :],
                    mask=rn[:, None] < N,
                    other=0,
                )
            ).to(tl.bfloat16)
            # row stride * row + col stride * col (a row/col stride swap
            # here reads scale[kb] of the WRONG row — row 0 still looks
            # correct, which is why small probes passed)
            sa = tl.load(SA + rm * sam + kb * sakb, mask=rm < M, other=0.0)
            sb = tl.load(SB + (rn // 128) * sbn + kb * sbkb, mask=rn < N, other=0.0)
            blk = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            acc += blk * (sa[:, None] * sb[None, :])
        tl.store(
            D + rm[:, None] * N + rn[None, :],
            acc.to(D.dtype.element_ty),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

    return _blockwise_gemm


def fnuz_required() -> bool:
    """Whether this hardware's fp8 tensor cores need the e4m3fnuz flavour.

    CDNA3 (gfx942) has no e4m3fn matrix unit: expert stacks must be rewritten
    to ``e4m3fnuz`` (halved payloads, doubled scales — ``e4m3fn_to_fnuz``,
    once per weight version) before any fp8 kernel can consume them. NVIDIA
    hardware fp8 IS e4m3fn, where that rewrite is pure overhead. This is the
    arch predicate behind that decision — owned here next to the conversion
    it drives, so model code asks instead of re-deriving it from
    ``torch.version.hip``.
    """
    import torch

    return torch.version.hip is not None


def e4m3fn_to_fnuz(w_fp8, scales):
    """e4m3fn -> e4m3fnuz with the payload-halving trick (one-time).

    gfx942's fp8 tensor cores and fp8->fp32 conversions are FNUZ-semantics
    (bias 4, max finite 240); checkpoint e4m3fn has bias 7 / max 448, so a
    VALUE-preserving cast overflows to NaN for payloads in (240, 448].
    Instead: halve every payload (EXACT for fp8 normals — one exponent
    decrement, mantissa untouched; subnormals round to the coarser fnuz
    subnormal grid) and DOUBLE the per-block scales, so payload x scale is
    unchanged. Returns (w_fnuz, scales_x2)."""
    import torch

    if w_fp8.dtype == torch.float8_e4m3fnuz:
        return w_fp8, scales
    half = (w_fp8.to(torch.float32) * 0.5).to(torch.float8_e4m3fnuz)
    return half, scales * 2.0


@lru_cache(maxsize=16)
def _fnuz_byte_lut(device):
    """256-entry e4m3fn -> e4m3fnuz byte map on ``device``.

    The halving conversion is a PURE per-byte function — halving an fp8
    value is an exact exponent decrement, subnormals land exactly on the
    fnuz subnormal grid, NaN (0x7f) maps to fnuz NaN (0x80) and fn -0
    (0x80) maps to fnuz +0 (0x00); verified exhaustive over all 256 bytes
    against this module's fp32-roundtrip reference. So the conversion needs
    no fp32 materialization of the weight stack at all: it is a byte gather
    through this table."""
    import torch

    raw = torch.arange(256, dtype=torch.uint8)
    fn = raw.view(torch.float8_e4m3fn)
    fz = (fn.to(torch.float32) * 0.5).to(torch.float8_e4m3fnuz)
    return fz.view(torch.uint8).to(device, non_blocking=True)


def e4m3fn_to_fnuz_inplace(w_fp8, scales, chunk_bytes=1 << 26):
    """In-place e4m3fn -> e4m3fnuz (issue #71): rewrite ``w_fp8``'s bytes to
    the halved-payload fnuz encoding and double ``scales`` in place, so the
    converted stack SHARES the checkpoint's storage.

    This removes both resident-memory costs of the copy-based conversion
    (``e4m3fn_to_fnuz``): the fnuz copy is no longer a second ~1 B/element
    resident copy of every expert weight (the copy-cache roughly doubled
    expert memory and capped grouped layers under a fixed HBM floor), and
    the ~4 B/element fp32 conversion transient is gone — the rewrite is a
    chunked gather through ``_fnuz_byte_lut`` whose peak extra memory is
    ~9x ``chunk_bytes`` (a uint8 result plus the int64 index temp).

    Returns ``(w_fnuz_view, scales)`` — a fnuz-dtype reinterpret of the SAME
    storage, bit-identical to ``e4m3fn_to_fnuz``'s payload bytes, drop-in
    for ``glm_moe_grouped_gemm_native`` and the fnuz-storage decode paths
    (``expert_gemv`` / ``gather_dequant`` with ``storage="e4m3fnuz"``).

    Contract (mirrors the #69 invalidation pattern):
    - raises TypeError on an fnuz-dtype input — the byte map is not
      idempotent and a second application would halve payloads again;
    - the caller keys a converted-marker on
      ``(data_ptr, _version, device)`` of BOTH tensors AFTER this call and
      re-runs it only when the key changes (checkpoint reload copies fresh
      e4m3fn bytes and bumps ``_version``);
    - run eagerly, not under CUDA-graph capture: the one-time rewrite is
      not replay-stable work.
    """
    import torch

    if w_fp8.dtype == torch.float8_e4m3fnuz:
        raise TypeError(
            "weights are already e4m3fnuz — refusing to halve payloads twice"
        )
    if w_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"expected e4m3fn weights, got {w_fp8.dtype}")
    if scales.device != w_fp8.device:
        raise ValueError("scales must share the weight device")
    if not w_fp8.is_contiguous():
        raise ValueError("weights must be contiguous for in-place byte conversion")
    lut = _fnuz_byte_lut(w_fp8.device)
    flat = w_fp8.view(torch.uint8).reshape(-1)
    step = max(int(chunk_bytes), 1)
    for begin in range(0, flat.numel(), step):
        seg = flat[begin : begin + step]
        seg.copy_(lut[seg.to(torch.int64)])
    scales.mul_(2.0)
    return w_fp8.view(torch.float8_e4m3fnuz), scales


@lru_cache(maxsize=1)
def _triton_native_kernel():
    """Native-fp8 blockwise GEMM: operands already e4m3fnuz; hardware
    fp8 tl.dot (CDNA3 v_mfma), per-128-K-block fp32 scale application —
    the same math as the oracle with fnuz-rounded operands."""
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _blockwise_gemm_native(
        A,
        SA,
        B,
        SB,
        D,
        M,
        N,
        K,
        sam,
        sakb,
        sbn,
        sbkb,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for kb in range(0, K // 128):
            rk = kb * 128 + tl.arange(0, 128)
            a = tl.load(
                A + rm[:, None] * K + rk[None, :],
                mask=rm[:, None] < M,
                other=0.0,
            )
            b = tl.load(
                B + rn[:, None] * K + rk[None, :],
                mask=rn[:, None] < N,
                other=0.0,
            )
            sa = tl.load(SA + rm * sam + kb * sakb, mask=rm < M, other=0.0)
            sb = tl.load(SB + (rn // 128) * sbn + kb * sbkb, mask=rn < N, other=0.0)
            blk = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            acc += blk * (sa[:, None] * sb[None, :])
        tl.store(
            D + rm[:, None] * N + rn[None, :],
            acc.to(D.dtype.element_ty),
            mask=(rm[:, None] < M) & (rn[None, :] < N),
        )

    return _blockwise_gemm_native


def _triton_native_backend(a_fnuz, a_scales, b_fnuz, b_scales, out):
    kfn = _triton_native_kernel()
    m, k = a_fnuz.shape
    n = b_fnuz.shape[0]
    import triton

    bm, bn, warps, stages = 128, 128, 4, 3
    cfg = os.environ.get("VK_FP8GEMM_TILES", "")
    if cfg:
        bm, bn, warps, stages = (int(x) for x in cfg.split(","))
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    kfn[grid](
        a_fnuz,
        a_scales,
        b_fnuz,
        b_scales,
        out,
        m,
        n,
        k,
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BM=bm,
        BN=bn,
        BK=128,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def quantize_activations_fnuz(x, group_size: int = 128):
    """BF16 [M, K] -> (fnuz payload, DOUBLED scales) in one step (the
    native-backend convention: halved payload x doubled scale = value)."""

    q, sc = quantize_activations_fp8(x, group_size)
    return e4m3fn_to_fnuz(q, sc)


def quantize_activations_native(x, group_size: int = 128):
    """BF16 [M, K] -> (e4m3fn payload, plain scales) for e4m3fn tensor cores.

    The CUDA counterpart of :func:`quantize_activations_fnuz`. NVIDIA's fp8
    tensor-core type *is* e4m3fn (bias 7), so the payload and the checkpoint's
    weight scales are used verbatim: the halve-payload/double-scale round trip
    that CDNA3 needs to reach ``e4m3fnuz`` would buy nothing here and costs a
    second resident copy of every expert stack. Same per-token per-128-group
    quantization as the fnuz form; only the storage flavour (and therefore the
    scale convention) differs.
    """
    return quantize_activations_fp8(x, group_size)


def _activation_quantizer(weight_dtype):
    """The activation quantizer matching the weight storage flavour.

    The grouped kernel is flavour-agnostic -- it loads fp8 operands and dots
    them -- but payload and scales must come from the *same* convention:
    e4m3fnuz weights carry halved payloads with doubled scales, e4m3fn weights
    carry the checkpoint's own bytes and scales.
    """
    import torch

    if weight_dtype == torch.float8_e4m3fnuz:
        return quantize_activations_fnuz
    if weight_dtype == torch.float8_e4m3fn:
        return quantize_activations_native
    raise TypeError(
        "native-fp8 MoE weights must be e4m3fn or e4m3fnuz, got "
        f"{weight_dtype}"
    )


@lru_cache(maxsize=1)
def _grouped_kernel():
    """Single-launch grouped blockwise fp8 GEMM (MoE): one grid covers
    every active expert's (row-tile, N-tile) work. A rows are GATHERED
    through the sort order (token index per slot); B and scales are
    selected per tile from the expert stacks. Operands are fnuz with the
    doubled-scale convention."""
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _grouped_gemm(
        A,
        SA,
        W,
        SW,
        D,
        TOK,
        TILE_EXP,
        TILE_R0,
        TILE_M,
        N,
        K,
        san,
        sakb,
        swn,
        swkb,
        BM: tl.constexpr,
        BN: tl.constexpr,
        A_BY_TOKEN: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        e = tl.load(TILE_EXP + pid_m).to(tl.int64)
        r0 = tl.load(TILE_R0 + pid_m)
        me = tl.load(TILE_M + pid_m)
        if me == 0:
            # Padding tile from a statically sized launch: nothing to do.
            # Exiting before the K loop keeps the padding cost to a prologue
            # instead of a full pass of masked (but still executed) dots.
            return
        ri = r0 + tl.arange(0, BM)  # sorted-slot space
        rmask = tl.arange(0, BM) < me
        # A rows: gate_up gathers the shared per-TOKEN activation; the down
        # GEMM's post-swiglu activations are per-SLOT (one per expert
        # assignment) and already live in sorted order — index directly.
        tok = tl.load(TOK + ri, mask=rmask, other=0).to(tl.int64) if A_BY_TOKEN else ri
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        wbase = W + e * N * K  # expert's [N, K] panel
        swbase = SW + e * (N // 128) * (K // 128)
        for kb in range(0, K // 128):
            rk = kb * 128 + tl.arange(0, 128)
            a = tl.load(
                A + tok[:, None] * K + rk[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            b = tl.load(
                wbase + rn[:, None] * K + rk[None, :],
                mask=rn[:, None] < N,
                other=0.0,
            )
            sa = tl.load(SA + tok * san + kb * sakb, mask=rmask, other=0.0)
            sb = tl.load(swbase + (rn // 128) * swn + kb * swkb, mask=rn < N, other=0.0)
            blk = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            acc += blk * (sa[:, None] * sb[None, :])
        tl.store(
            D + ri[:, None] * N + rn[None, :],
            acc.to(D.dtype.element_ty),
            mask=rmask[:, None] & (rn[None, :] < N),
        )

    return _grouped_gemm


def _grouped_backend(
    a_nz, asc2, w_nz, wsc2, out, tok, tile_exp, tile_r0, tile_m, n, k, a_by_token=True
):
    """One grouped launch. With a_by_token, a_nz rows are gathered by the
    TOKEN id in `tok` (shared activations, gate_up); otherwise rows are
    indexed directly in SORTED-slot order (per-slot activations, down).
    `out` rows are always in sorted-slot order ([n_slots, n])."""

    kfn = _grouped_kernel()
    bm, bn, warps, stages = 64, 128, 4, 3
    cfg = os.environ.get("VK_FP8GEMM_TILES", "")
    if cfg:
        bm, bn, warps, stages = (int(x) for x in cfg.split(","))
    import triton

    grid = (triton.cdiv(n, bn), len(tile_m))
    kfn[grid](
        a_nz,
        asc2,
        w_nz,
        wsc2,
        out,
        tok,
        tile_exp,
        tile_r0,
        tile_m,
        n,
        k,
        asc2.stride(0),
        asc2.stride(1),
        wsc2.stride(1),
        wsc2.stride(2),
        BM=bm,
        BN=bn,
        A_BY_TOKEN=a_by_token,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def glm_moe_grouped_gemm_native(
    x,
    gate_up_nz,
    gate_up_sc2,
    down_nz,
    down_sc2,
    topk_index,
    topk_weights,
    swiglu_limit=7.0,
):
    """Routed expert MLP via TWO single-launch grouped fp8 GEMMs.

    Inputs: x bf16 [T, H]; gate_up_nz/down_nz the expert stacks as fp8 with
    their scales; routing int64 [T, K] + fp weights [T, K]. Returns bf16 [T, H].

    Two storage flavours are accepted, discriminated by the weight dtype and
    kept consistent end to end: ``e4m3fnuz`` (ROCm CDNA3: payloads rewritten
    once by :func:`e4m3fn_to_fnuz`, scales doubled) and ``e4m3fn`` (NVIDIA
    tensor cores: checkpoint bytes and scales as they are, no conversion and no
    second resident weight copy).

    Host work per call: one sort, a handful of tiny torch ops for the
    tile map, two grouped launches, one swiglu + per-row quantize — no
    per-expert Python loop (that loop measured host-bound at ~200 ms on
    MI300A; this replaces it) and, with the static tile map, no host sync:
    the routing counts come from a static-shape scatter-add and the launch is
    sized from tensor shapes, so the whole MoE dispatch is CUDA-graph
    capturable on both NVIDIA and ROCm (ROCm's ``torch.bincount`` is not)."""
    import torch

    # Up-front contract check (the torch_ops dispatch convention): the floe
    # ladder gates only on policy (knob + token regime + capture) and relies
    # on OpNotEligible here for everything device/dtype/shape.
    if (
        x.dim() != 2 or not x.is_cuda or x.dtype != torch.bfloat16
        or gate_up_nz.dim() != 3 or down_nz.dim() != 3
        or gate_up_nz.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        or down_nz.dtype != gate_up_nz.dtype
        or gate_up_sc2.dtype != torch.float32 or down_sc2.dtype != torch.float32
        or topk_index.dtype != torch.int64
        or topk_index.shape != topk_weights.shape
        or not topk_weights.is_floating_point()
        or gate_up_nz.device != x.device or down_nz.device != x.device
    ):
        raise OpNotEligible(
            "grouped native-fp8 MoE wants CUDA bf16 x [T,H], matching fp8 "
            "(e4m3fn/fnuz) expert stacks with fp32 scales, int64 routing and "
            "float routing weights on one device")
    t, k = topk_index.shape
    h = x.shape[1]
    n_experts = gate_up_nz.shape[0]
    two_i = gate_up_nz.shape[1]
    i = two_i // 2
    if (
        down_nz.shape[:1] != gate_up_nz.shape[:1]
        or down_nz.shape[1] != h or down_nz.shape[2] != i
        or two_i % 2 or h % 128 or i % 128
        or gate_up_sc2.shape != (n_experts, two_i // 128, h // 128)
        or down_sc2.shape != (n_experts, h // 128, i // 128)
    ):
        raise OpNotEligible(
            "expert stacks must be [E,2I,H]/[E,H,I] block-128 with matching "
            "[E,O/128,I/128] scales")
    dev = x.device

    order, sorted_tok, counts, seg = _route_counts(topk_index, n_experts, dev)

    quantize = _activation_quantizer(gate_up_nz.dtype)
    a_nz, asc2 = quantize(x)

    bm = int(os.environ.get("VK_FP8GEMM_TILES", "64,128,4,3").split(",")[0])
    slots = t * k
    # Static upper bound on the row-tile count (see _tile_map_static). Two
    # bounds hold for any routing: sum_e ceil(c_e/bm) <= slots/bm + E, and no
    # tile is empty so tiles <= slots. The second one matters in DECODE, where
    # E (288) dwarfs the slot count (64 at t=8,k=8): without it the launch pads
    # to ~290 mostly-empty blocks for ~50 real ones, which measured as a 2.6%
    # decode regression. Taking the min keeps prefill's tight bound and stops
    # decode from paying for experts it cannot have routed to.
    cap = min(slots, slots // bm + n_experts + 1)
    tile_exp, tile_r0, tile_m = _tile_map_static(counts, seg, bm, dev, cap)

    gu = torch.empty((slots, two_i), device=dev, dtype=torch.bfloat16)
    _grouped_backend(
        a_nz,
        asc2,
        gate_up_nz,
        gate_up_sc2,
        gu,
        sorted_tok,
        tile_exp,
        tile_r0,
        tile_m,
        two_i,
        h,
    )
    gate = gu[:, :i].float().clamp(max=swiglu_limit)
    up = gu[:, i:].float().clamp(min=-swiglu_limit, max=swiglu_limit)
    act = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)

    # act rows are per-SLOT in sorted order (K expert assignments per
    # token) — the down GEMM indexes A directly, no gather.
    a2_nz, a2sc2 = quantize(act)
    dn = torch.empty((slots, h), device=dev, dtype=torch.bfloat16)
    _grouped_backend(
        a2_nz,
        a2sc2,
        down_nz,
        down_sc2,
        dn,
        sorted_tok,
        tile_exp,
        tile_r0,
        tile_m,
        h,
        i,
        a_by_token=False,
    )

    w = topk_weights.reshape(-1)[order].float().unsqueeze(-1)
    y = torch.zeros((t, h), device=dev, dtype=torch.float32)
    y.index_add_(0, sorted_tok, dn.float() * w)
    return y.to(torch.bfloat16)


def _triton_backend(a_fp8, a_scales, b_fp8, b_scales, out):
    """VK_FP8GEMM_TILES='BM,BN,warps,stages' overrides the defaults (tuning)."""
    import torch

    kfn = _triton_kernel()
    m, k = a_fp8.shape
    n = b_fp8.shape[0]
    a8 = a_fp8.view(torch.uint8)
    b8 = b_fp8.view(torch.uint8)
    import triton

    bm, bn, warps, stages = 128, 128, 8, 3
    cfg = os.environ.get("VK_FP8GEMM_TILES", "")
    if cfg:
        bm, bn, warps, stages = (int(x) for x in cfg.split(","))
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    kfn[grid](
        a8,
        a_scales,
        b8,
        b_scales,
        out,
        m,
        n,
        k,
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BM=bm,
        BN=bn,
        BK=128,
        num_warps=warps,
        num_stages=stages,
    )
    return out


# ---------------------------------------------------------------------------
# CuTe DSL kernel (sm_90) — lazy import; None when the DSL is unavailable.
# ---------------------------------------------------------------------------

_CUTE_KERNEL = None
_CUTE_TRIED = False


def _cute_kernel():
    global _CUTE_KERNEL, _CUTE_TRIED
    if _CUTE_TRIED:
        return _CUTE_KERNEL
    _CUTE_TRIED = True
    if os.environ.get("VKERNELS_DISABLE_CUTE", "") not in ("", "0"):
        return None
    try:
        _CUTE_KERNEL = _build_cute_kernel()
    except Exception as exc:  # noqa: BLE001 - optional accelerator path
        print(
            f"[glm_fp8_blockwise_gemm] CuTe DSL unavailable ({type(exc).__name__}: {exc}); torch reference path active"
        )
        _CUTE_KERNEL = None
    return _CUTE_KERNEL


def _build_cute_kernel():
    """Lazy-import the Hopper blockwise kernel's launch entry point.

    Compilation happens on the first GPU call (cute.compile inside launch);
    the pure-torch reference path is the oracle. launch's signature is
    (a_fp8, a_scales, b_fp8, b_scales, out) — exactly the dispatcher's."""
    from vkernels.torch_ops import _glm_fp8_sm90_gemm

    return _glm_fp8_sm90_gemm.launch
