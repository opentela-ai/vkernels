"""TokenSpeed Triton MLA kernels, vendored for the portable attention path.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel`` subpackage, ``main``
branch, fetched 2026-11 from
``tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/mla/_triton/``
(MIT, Copyright (c) 2026 LightSeek Foundation).

Ported (verbatim kernel bodies; see the per-section markers):
  ``page_table.py``  — ``resolve_group_slot`` / ``bounded_group_slots`` /
                       ``copy_page_table`` (group page-table resolution)
  ``prefill.py``     — ``_mla_prefill_kernel`` + ``mla_prefill_fwd`` /
                       ``triton_mla_prefill`` (varlen, online softmax,
                       split nope/rope head dims, optional LSE)
  ``decode.py``      — ``_mla_decode_kernel`` + ``mla_decode_fwd`` /
                       ``triton_mla_decode_with_kvcache`` (paged latent KV,
                       windowed/non-causal block mode, optional LSE)

Adapted (import point only, plus registration):
  ``tokenspeed_kernel._triton`` imports became stock ``triton`` /
  ``triton.language`` (this repo uses upstream Triton, not the fork), and
  the kernels register through :mod:`vkernels.torch_ops.mla_registry`
  (priority bands + format signatures) instead of tokenspeed's decorator.

Layout notes kept from the source: decode treats the KV cache as
``[pages, page_size, 1, lora+rope]`` with one KV head; prefill is dense
varlen with ``cu_seqlens``; the page-table helpers are the shared mapping
both use on the serving side.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = [
    "bounded_group_slots",
    "copy_page_table",
    "mla_prefill_fwd",
    "triton_mla_prefill",
    "mla_decode_fwd",
    "triton_mla_decode_with_kvcache",
]

_FP8_DTYPES = frozenset({torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e4m3fnuz})


# ---------------------------------------------------------------------------
# page_table.py (verbatim)
# ---------------------------------------------------------------------------


@triton.jit
def resolve_group_slot(
    Table,
    position,
    request,
    table_rows,
    table_cols,
    row_stride,
    col_stride,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    first_page,
    page_count,
):
    """The shared raw-position -> group-slot mapping, with explicit bounds."""
    logical = position // STRIDE
    column = logical // ROWS
    valid = (
        (position >= 0)
        & (request >= 0)
        & (request < table_rows)
        & (column >= 0)
        & (column < table_cols)
    )
    page = tl.load(
        Table + request * row_stride + column * col_stride, valid, other=-1
    ).to(tl.int64)
    valid &= (page >= first_page) & (page < page_count)
    return tl.where(valid, page * ROWS + logical % ROWS, -1)


# Table geometry follows the batch (rows) and the longest request (columns);
# left specialized, Triton recompiles when either hits 1 or a multiple of 16.
@triton.jit(do_not_specialize=["N", "TR", "TC", "TS0", "TS1"])
def _group_slots_kernel(
    P,
    R,
    Table,
    Out,
    N,
    W: tl.constexpr,
    PS0: tl.constexpr,
    PS1: tl.constexpr,
    RS0: tl.constexpr,
    RS1: tl.constexpr,
    TR,
    TC,
    TS0,
    TS1,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    first_page,
    page_count,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pos = tl.load(P + i // W * PS0 + i % W * PS1, i < N, other=-1)
    req = tl.load(R + i // W * RS0 + i % W * RS1, i < N, other=-1)
    slot = resolve_group_slot(
        Table, pos, req, TR, TC, TS0, TS1, ROWS, STRIDE, first_page, page_count
    )
    tl.store(Out + i, slot, i < N)


def bounded_group_slots(
    positions: torch.Tensor,
    requests: torch.Tensor,
    table: torch.Tensor,
    rows_per_page: int,
    entry_stride: int,
    first_page: int,
    page_count: int,
) -> torch.Tensor:
    """Map equal-shaped 1-D/2-D positions and requests into int64 group slots.

    Supports strided inputs, including broadcast request rows. Table entries
    outside [first_page, page_count), invalid coordinates, and negative positions
    resolve to -1. The output has positions.shape; no cache bytes are accessed.
    """
    if positions.shape != requests.shape or positions.ndim not in (1, 2):
        raise ValueError("positions and requests require equal 1-D/2-D shapes")
    if rows_per_page <= 0 or entry_stride <= 0:
        raise ValueError("group row geometry must be positive")
    if not positions.is_cuda:
        logical = positions.to(torch.int64) // entry_stride
        col = logical // rows_per_page
        if not table.numel():
            return torch.full_like(positions, -1, dtype=torch.int64)
        page = table[
            requests.clamp(0, table.shape[0] - 1), col.clamp(0, table.shape[1] - 1)
        ].to(torch.int64)
        valid = (
            (positions >= 0)
            & (requests >= 0)
            & (requests < table.shape[0])
            & (col < table.shape[1])
            & (page >= first_page)
            & (page < page_count)
        )
        return (page * rows_per_page + logical % rows_per_page).masked_fill(~valid, -1)
    out = torch.empty(positions.shape, dtype=torch.int64, device=positions.device)
    if positions.numel():
        p = positions[:, None] if positions.ndim == 1 else positions
        r = requests[:, None] if requests.ndim == 1 else requests
        _group_slots_kernel[(triton.cdiv(positions.numel(), 256),)](
            p,
            r,
            table,
            out,
            positions.numel(),
            p.shape[1],
            *p.stride(),
            *r.stride(),
            *table.shape,
            *table.stride(),
            rows_per_page,
            entry_stride,
            first_page,
            page_count,
            BLOCK=256,
        )
    return out


@triton.jit(
    do_not_specialize=[
        "live_rows",
        "source_cols",
        "source_stride",
        "out_cols",
        "out_stride",
    ]
)
def _copy_page_table_kernel(
    Source,
    Out,
    live_rows,
    source_cols,
    source_stride,
    out_cols,
    out_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        Source + row * source_stride + col,
        (row < live_rows) & (col < source_cols),
        other=0,
    )
    tl.store(Out + row * out_stride + col, value, col < out_cols)


def copy_page_table(source: torch.Tensor, out: torch.Tensor, live_rows: int) -> None:
    """Copy live rows into a persistent table, clearing padding and unused columns.

    source/out are int32 matrices with unit column stride. out may have a larger
    capacity; its entire extent is overwritten so stale cache pages cannot leak
    into a subsequent batch. The source is never changed and may not alias out.
    """
    if source.dtype != torch.int32 or out.dtype != torch.int32:
        raise ValueError("page tables must be int32")
    if source.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("page table columns must be contiguous")
    if (
        not 0 <= live_rows <= min(source.shape[0], out.shape[0])
        or source.shape[1] > out.shape[1]
    ):
        raise ValueError("page table copy exceeds capacity")
    if not out.is_cuda:
        out.zero_()
        out[:live_rows, : source.shape[1]].copy_(source[:live_rows])
        return
    if out.numel():
        _copy_page_table_kernel[(out.shape[0], triton.cdiv(out.shape[1], 256))](
            source,
            out,
            live_rows,
            source.shape[1],
            source.stride(0),
            out.shape[1],
            out.stride(0),
            BLOCK=256,
        )


# ---------------------------------------------------------------------------
# prefill.py (verbatim)
# ---------------------------------------------------------------------------


@triton.jit
def _mla_prefill_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    cu_seqlens_q,
    cu_seqlens_kv,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    stride_lse_h,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_LSE: tl.constexpr,
):
    cur_seq = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    q_start = tl.load(cu_seqlens_q + cur_seq)
    q_len = tl.load(cu_seqlens_q + cur_seq + 1) - q_start
    kv_start = tl.load(cu_seqlens_kv + cur_seq)
    kv_len = tl.load(cu_seqlens_kv + cur_seq + 1) - kv_start
    q_causal_start = tl.maximum(kv_len - q_len, 0)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    q_offsets_m = cur_block_m * BLOCK_M + offs_m
    mask_m = q_offsets_m < q_len
    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (q_start + q_offsets_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(Q + offs_q, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lq
        offs_qpe = (
            (q_start + q_offsets_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_m[:, None] & mask_dpe[None, :], other=0.0)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, kv_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_offsets_n = start_n + offs_n
        mask_n = kv_offsets_n < kv_len
        final_mask = mask_m[:, None] & mask_n[None, :]

        if IS_CAUSAL:
            query_positions = q_causal_start + q_offsets_m[:, None]
            key_positions = kv_offsets_n[None, :]
            final_mask &= query_positions >= key_positions

        offs_k = (
            (kv_start + kv_offsets_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(K + offs_k, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        qk = tl.dot(q.to(k.dtype), k)

        if BLOCK_DPE > 0:
            offs_kpe = (
                (kv_start + kv_offsets_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K + offs_kpe, mask=mask_n[None, :] & mask_dpe[:, None], other=0.0
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * libdevice.tanh(qk / logit_cap)

        qk = tl.where(final_mask, qk, float("-inf"))

        row_max = tl.max(qk, 1)
        row_max_fixed = tl.where(row_max == float("-inf"), -1e20, row_max)
        n_e_max = tl.maximum(row_max_fixed, e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (kv_start + kv_offsets_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(V + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)
        e_max = n_e_max

    safe_deno = tl.where(deno > 0.0, deno, 1.0)
    offs_o = (
        (q_start + q_offsets_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O + offs_o,
        acc / safe_deno[:, None],
        mask=mask_m[:, None] & mask_dv[None, :],
    )

    if HAS_LSE:
        offs_lse = (q_start + q_offsets_m) * stride_lse_bs + cur_head * stride_lse_h
        lse = tl.where(deno > 0.0, tl.log(deno) + e_max, float("-inf"))
        tl.store(LSE + offs_lse, lse, mask=mask_m)


def mla_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool,
    logit_cap: float = 0.0,
    lse: torch.Tensor | None = None,
) -> None:
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(
            f"q/k head dims must match, got {q.shape[-1]} and {k.shape[-1]}"
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads, "
            f"got {q.shape[1]} and {k.shape[1]}"
        )
    if out.shape != (q.shape[0], q.shape[1], v.shape[-1]):
        raise ValueError(
            f"out shape must be {(q.shape[0], q.shape[1], v.shape[-1])}, "
            f"got {tuple(out.shape)}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have contiguous last dimension")
    if lse is not None and lse.shape != (q.shape[0], q.shape[1]):
        raise ValueError(
            f"lse shape must be {(q.shape[0], q.shape[1])}, got {tuple(lse.shape)}"
        )

    q_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]

    if q_head_dim == 576:
        block_dmodel = 512
        block_dpe = 64
    elif q_head_dim == 288:
        block_dmodel = 256
        block_dpe = 32
    elif q_head_dim == 192:
        block_dmodel = 128
        block_dpe = 64
    else:
        block_dmodel = triton.next_power_of_2(q_head_dim)
        block_dpe = 0
    block_dv = triton.next_power_of_2(v_head_dim)
    block_m, block_n = (64, 64)
    num_warps = 4

    lse_arg = lse if lse is not None else out
    grid = (cu_seqlens_q.shape[0] - 1, q.shape[1], triton.cdiv(max_seqlen_q, block_m))

    _mla_prefill_kernel[grid](
        q,
        k,
        v,
        out,
        lse_arg,
        cu_seqlens_q,
        cu_seqlens_kv,
        softmax_scale,
        q.shape[1] // k.shape[1],
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        out.stride(0),
        out.stride(1),
        lse_arg.stride(0),
        lse_arg.stride(1),
        logit_cap=logit_cap,
        Lq=q_head_dim,
        Lv=v_head_dim,
        BLOCK_DMODEL=block_dmodel,
        BLOCK_DPE=block_dpe,
        BLOCK_DV=block_dv,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        IS_CAUSAL=is_causal,
        HAS_LSE=lse is not None,
        num_warps=num_warps,
        num_stages=1,
    )


def triton_mla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    is_causal: bool = True,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    seq_lens_kv: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Varlen MLA prefill (absorbed MQA form): dense K/V, optional LSE.

    ``q`` is ``[total_tokens, H, qk_head_dim]`` (nope+rope fused), ``k`` is
    ``[total_kv_tokens, 1, qk_head_dim]``, ``v`` is
    ``[total_kv_tokens, 1, v_head_dim]`` with ``cu_seqlens_*`` boundaries;
    returns ``[total_tokens, H, v_head_dim]`` (plus fp32 LSE on request).
    """
    if out is None:
        out_dtype = torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype
        out = torch.empty(
            (q.shape[0], q.shape[1], v.shape[-1]),
            dtype=out_dtype,
            device=q.device,
        )

    lse = (
        torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    mla_prefill_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        softmax_scale,
        is_causal=is_causal,
        logit_cap=logit_cap,
        lse=lse,
    )
    if return_lse:
        return out, lse
    return out


# ---------------------------------------------------------------------------
# decode.py (verbatim)
# ---------------------------------------------------------------------------


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _mla_decode_kernel(
    Q,
    KV_Cache,
    O,
    LSE,
    page_table,
    cache_seqlens,
    sm_scale,
    stride_qb,
    stride_qq,
    stride_qh,
    stride_kv_page,
    stride_kv_token,
    stride_kv_head,
    stride_ob,
    stride_oq,
    stride_oh,
    stride_lse_b,
    stride_lse_q,
    stride_lse_h,
    # Page-table width follows the batch; runtime so every batch shape
    # shares one binary.
    page_table_stride_b,
    PAGE_SIZE: tl.constexpr,
    # Unused by this kernel; the per-batch longest context must not key its
    # compilation.
    MAX_SEQLEN_K,
    logit_cap: tl.constexpr,
    KV_LORA_RANK: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_LSE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,
    NONCAUSAL_BLOCK_SIZE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_q = tl.program_id(1)
    cur_head = tl.program_id(2)

    offs_r = tl.arange(0, BLOCK_R)
    offs_rope = tl.arange(0, BLOCK_ROPE)
    mask_r = offs_r < KV_LORA_RANK
    mask_rope = offs_rope < QK_ROPE_HEAD_DIM

    q_base = cur_batch * stride_qb + cur_q * stride_qq + cur_head * stride_qh
    q_latent = tl.load(Q + q_base + offs_r, mask=mask_r, other=0.0)
    q_rope = tl.load(
        Q + q_base + KV_LORA_RANK + offs_rope,
        mask=mask_rope,
        other=0.0,
    )

    cache_len = tl.load(cache_seqlens + cur_batch)
    if WINDOW_LEFT >= 0:
        # DFlash2 flattens each non-causal proposal block into one decode row
        # per position. Every row sees the whole proposal block, while its
        # historical context follows the reference mask
        #     query_position - key_position <= window_left.
        block_position = cur_batch % NONCAUSAL_BLOCK_SIZE
        context_len = cache_len - NONCAUSAL_BLOCK_SIZE
        window_start = tl.maximum(
            0,
            context_len - WINDOW_LEFT + block_position,
        )
        loop_start = (window_start // BLOCK_N) * BLOCK_N
    else:
        window_start = 0
        loop_start = 0
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    e_sum = 0.0
    e_max = -float("inf")

    for start_n in tl.range(loop_start, cache_len, BLOCK_N, num_stages=2):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        token_offsets = start_n + offs_n
        mask_n = (token_offsets >= window_start) & (token_offsets < cache_len)
        page_indices = token_offsets // PAGE_SIZE
        page_offsets = token_offsets - page_indices * PAGE_SIZE
        physical_pages = tl.load(
            page_table + cur_batch * page_table_stride_b + page_indices,
            mask=mask_n,
            other=0,
        ).to(tl.int64)
        cache_base = (
            physical_pages * stride_kv_page
            + page_offsets * stride_kv_token
            + 0 * stride_kv_head
        )

        k_latent = tl.load(
            KV_Cache + cache_base[:, None] + offs_r[None, :],
            mask=mask_n[:, None] & mask_r[None, :],
            other=0.0,
        )
        k_rope = tl.load(
            KV_Cache + cache_base[:, None] + KV_LORA_RANK + offs_rope[None, :],
            mask=mask_n[:, None] & mask_rope[None, :],
            other=0.0,
        )

        qk = tl.sum(k_latent.to(tl.float32) * q_latent[None, :].to(tl.float32), axis=1)
        qk += tl.sum(k_rope.to(tl.float32) * q_rope[None, :].to(tl.float32), axis=1)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        qk = tl.where(mask_n, qk, float("-inf"))
        block_max = tl.max(qk, axis=0)
        block_max_fixed = tl.where(block_max == float("-inf"), -1e20, block_max)
        n_e_max = tl.maximum(block_max_fixed, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        acc = acc * old_scale + tl.sum(p[:, None] * k_latent.to(tl.float32), axis=0)
        e_sum = e_sum * old_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out_base = cur_batch * stride_ob + cur_q * stride_oq + cur_head * stride_oh
    tl.store(O + out_base + offs_r, acc / safe_sum, mask=mask_r)

    if HAS_LSE:
        lse = tl.where(e_sum > 0.0, tl.log(e_sum) + e_max, float("-inf"))
        tl.store(
            LSE
            + cur_batch * stride_lse_b
            + cur_q * stride_lse_q
            + cur_head * stride_lse_h,
            lse,
        )


def _normalize_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 3:
        return kv_cache.unsqueeze(2)
    if kv_cache.dim() != 4:
        raise ValueError(f"kv_cache must be 3D or 4D, got {kv_cache.dim()}D")
    if kv_cache.shape[2] == 1:
        return kv_cache
    if kv_cache.shape[1] == 1:
        return kv_cache[:, 0].unsqueeze(2)
    raise ValueError(f"unsupported kv_cache shape {tuple(kv_cache.shape)}")


def mla_decode_fwd(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    out: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    lse: torch.Tensor | None = None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
) -> None:
    if q.dim() != 4:
        raise ValueError(
            f"q must have shape [B, q_len, H, R + rope], got {tuple(q.shape)}"
        )
    if q.shape[1] != 1:
        raise NotImplementedError("Triton MLA decode currently supports q_len == 1")
    if q.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"q head dim must be {kv_lora_rank + qk_rope_head_dim}, "
            f"got {q.shape[-1]}"
        )
    if out.shape != q.shape[:-1] + (kv_lora_rank,):
        raise ValueError(
            f"out shape must be {q.shape[:-1] + (kv_lora_rank,)}, "
            f"got {tuple(out.shape)}"
        )
    if q.stride(-1) != 1 or out.stride(-1) != 1:
        raise ValueError("q and out must have contiguous last dimension")
    if lse is not None and lse.shape != q.shape[:-1]:
        raise ValueError(f"lse shape must be {q.shape[:-1]}, got {tuple(lse.shape)}")
    if window_left < -1:
        raise ValueError(f"window_left must be -1 or non-negative, got {window_left}")
    if noncausal_block_size <= 0:
        raise ValueError(
            f"noncausal_block_size must be positive, got {noncausal_block_size}"
        )
    if 0 <= window_left < noncausal_block_size - 1:
        raise ValueError(
            "window_left must cover the complete non-causal block; got "
            f"window_left={window_left}, block_size={noncausal_block_size}"
        )
    if window_left >= 0 and q.shape[0] % noncausal_block_size:
        raise ValueError(
            "sliding MLA decode rows must contain complete non-causal blocks: "
            f"batch={q.shape[0]}, block_size={noncausal_block_size}"
        )

    kv_cache = _normalize_kv_cache(kv_cache)
    if kv_cache.shape[2] != 1:
        raise ValueError(f"MLA kv_cache must have one KV head, got {kv_cache.shape[2]}")
    if kv_cache.shape[-1] != kv_lora_rank + qk_rope_head_dim:
        raise ValueError(
            f"kv_cache head dim must be {kv_lora_rank + qk_rope_head_dim}, "
            f"got {kv_cache.shape[-1]}"
        )
    if kv_cache.stride(-1) != 1:
        raise ValueError("kv_cache must have contiguous last dimension")

    block_n = 16
    block_r = triton.next_power_of_2(kv_lora_rank)
    block_rope = triton.next_power_of_2(qk_rope_head_dim)
    grid = (q.shape[0], q.shape[1], q.shape[2])
    lse_arg = lse if lse is not None else out

    _mla_decode_kernel[grid](
        q,
        kv_cache,
        out,
        lse_arg,
        page_table,
        cache_seqlens,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse_arg.stride(0),
        lse_arg.stride(1),
        lse_arg.stride(2),
        page_table.stride(0),
        kv_cache.shape[1],
        max_seqlen_k,
        logit_cap=logit_cap,
        KV_LORA_RANK=kv_lora_rank,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        BLOCK_R=block_r,
        BLOCK_ROPE=block_rope,
        BLOCK_N=block_n,
        HAS_LSE=lse is not None,
        WINDOW_LEFT=window_left,
        NONCAUSAL_BLOCK_SIZE=noncausal_block_size,
        num_warps=8,
        num_stages=2,
    )


def triton_mla_decode_with_kvcache(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    logit_cap: float = 0.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    window_left: int = -1,
    noncausal_block_size: int = 1,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Paged MLA decode (absorbed MQA form) over the latent KV cache.

    ``q`` is ``[B, 1, H, kv_lora_rank + qk_rope_head_dim]``; ``kv_cache``
    is ``[pages, page_size, 1, kv_lora_rank + qk_rope_head_dim]`` addressed
    through the int32 ``page_table``; returns ``[B, 1, H, kv_lora_rank]``
    (plus fp32 LSE on request). ``window_left``/``noncausal_block_size``
    enable the flattened non-causal proposal-block decode mode.
    """
    if out is None:
        out_dtype = torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype
        out = torch.empty(
            q.shape[:-1] + (kv_lora_rank,), dtype=out_dtype, device=q.device
        )

    lse = (
        torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    mla_decode_fwd(
        q,
        kv_cache,
        out,
        page_table,
        cache_seqlens,
        max_seqlen_k,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        softmax_scale,
        logit_cap=logit_cap,
        lse=lse,
        window_left=window_left,
        noncausal_block_size=noncausal_block_size,
    )
    if return_lse:
        return out, lse
    return out
