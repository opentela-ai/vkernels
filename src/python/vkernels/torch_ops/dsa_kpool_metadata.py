"""Pool-aware DSA (sparse-attention) decode metadata — one Triton launch.

Every decode step of a paged DSA layer needs the same metadata block the
indexer / sparse-attention kernels consume: the per-request kv lengths
(``cache_seqlens``), their zero-prefixed cumsums (``cu_seqlens_k``), the
indexer-visible lengths after pool-aware top-k capping
(``dsa_cache_seqlens`` / ``dsa_cu_seqlens_k``), and the page tables — the
wide page_size=1 table gathered row-wise from ``req_to_token`` plus the
compact real-page table. SGLang's eager fallback for that block
(``dsa_backend.py`` decode replay path) is a dtype cast, two ``cumsum``
copies, an advanced-index gather ``req_to_token[req_pool_indices, :max_len]``
plus a wide ``copy_`` per table, and ``compute_dsa_seqlens`` — a dozen
small kernels whose width is the FULL table width even when every live
sequence is short. This module vendors SGLang's fusion of that whole
block into ONE launch whose grid is shape-derived (CUDA-graph capturable)
and whose column scan is bounded by each row's LIVE kv length, read from
device memory at run time.

Vendored from SGLang v0.5.20
``sglang/kernels/ops/attention/dsa_kpool_metadata/`` (Apache-2.0):
``decode.py`` (the fused decode metadata kernel + wrapper),
``scan.py`` (``bounded_scan_num_splits``), ``verify.py`` (the
target-verify variant for speculative decoding). The kernels are
verbatim; only the Python wrappers were adapted (see DEVIATIONS).

CUT LIST
--------
Kept (verbatim unless noted):
  ``_fused_dsa_decode_metadata_kernel`` + ``fused_dsa_decode_metadata``
  ``_fused_dsa_target_verify_metadata_kernel`` +
    ``fused_dsa_target_verify_metadata`` (the ``_prep_..._launch`` helper
    folded into the wrapper — it was the kernel's only consumer)
  ``bounded_scan_num_splits`` / ``_TILE_PROGRAM_TARGET``
Dropped (reason):
  ``draft_extend.py`` — the DRAFT_EXTEND forward mode exists only under
    speculative decoding with a draft model (MTP/EAGLE extension); floe's
    glm53flash runner has no draft-extend counterpart, and its dynamic
    per-row prefix loop (non-static extend lens) is the largest
    unexercised surface in the set. The file is self-contained (imports
    only scan.py) — re-vendor it verbatim if floe grows spec-decode.
  the engine machinery around the kernels — ``dsa_metadata_manager.py``
    (env-gated fusion selection, the ``kpool_metadata_fusion_supported``
    page_size==64 gate), ``fused_metadata_copy`` (MTP replay-buffer
    reuse), FlashMLA ``num_splits`` plumbing, ``index_buf_accessor``.
    None of it is needed by the kernels; the geometry it enforced is
    documented below instead.

FLOE CONTRACT MAPPING (GLM-5.3-Flash)
-------------------------------------
  floe ``Glm53Indexer`` selects ``index_topk // index_kpool`` pools per
      query, flattens them back to token ids and appends the visible
      partial-pool tail — emitting int32 ``[B, S_q, W]`` ids with ``-1``
      masking, ``W = index_topk + index_kpool - 1`` (2048 + 4 - 1 = 2051).
      ``dsa_cache_seqlens`` is exactly the kv width that selection can
      cover: ``min(index_topk, pool-aligned history) + live tail`` — the
      kernel's ``index_kpool`` branch is SGLang's ``compute_dsa_seqlens``
      verbatim (the pool-capping oracle), and ``index_kpool <= 1``
      degenerates to plain DSA ``min(seq, index_topk)``.
  floe config (glm53flash/config.py): ``index_topk = 2048``,
      ``index_kpool = 4`` (config validates divisibility), so the
      pool-aware branch is the live one. ``real_page_size`` is the kv
      pool's page size; SGLang gates 64 with ``page_size % kpool == 0``
      (page-aligned pools) — floe should keep the same invariant.
  pool tensors (SGLang ``ReqToTokenPool`` shapes — what a floe paged DSA
      pool hands over): ``seq_lens`` ``[bs]`` int32/int64,
      ``req_pool_indices`` ``[bs]`` int32/int64 (rows into
      ``req_to_token``), ``req_to_token`` ``[pool_rows, >= max_len]``
      int32/int64 (logical token -> physical kv slot).
  outputs (int32, CALLER-allocated so a decode graph replays into them):
      ``cache_seqlens`` ``[bs]``, ``cu_seqlens_k`` ``[bs+1]``,
      ``dsa_cache_seqlens`` ``[bs]``, ``dsa_cu_seqlens_k`` ``[bs+1]``,
      ``page_table_1`` ``[bs, max_len]`` (OPTIONAL — pass ``None`` when
      only the compact table is consumed), ``real_page_table``
      ``[bs, ceil(max_len / real_page_size)]`` (required iff
      ``real_page_size > 1``; with ``real_page_size == 1`` the wide table
      IS the real table). No re-stride/re-dtype is ever performed: the
      kernels take strides as arguments (1-D non-contiguous index inputs
      and 2-D strided tables work as-is); only the OUTPUT dtypes are
      pinned to int32 (the values the kernels store).

CAPTURE SAFETY (why ``do_not_specialize`` matters)
--------------------------------------------------
``max_len``, ``num_splits`` and the page-table row strides are
``do_not_specialize``: one compiled kernel serves every decode bucket
(no recompile when ``max_len`` changes), and a captured launch bakes
only host-known values (shapes, strides, ``num_splits``). Every
data-dependent bound — each row's live kv length — is read from DEVICE
MEMORY inside the kernel (``seq_lens``), so a captured graph replays
correctly after the lengths change under static shapes, with zero host
syncs (the wrappers allocate nothing and read nothing back). Each
page-table row is written only over its live prefix (rounded up to the
kernel's 128-column tile); the tail keeps stale values across replays —
consumers must bound reads by ``cache_seqlens``.

DEVIATIONS FROM THE SOURCE (all documented, none silent)
--------------------------------------------------------
1. The donor's ``assert``-based contract checks became eligibility:
   :func:`fused_dsa_decode_metadata_eligible` /
   :func:`fused_dsa_target_verify_metadata_eligible` plus
   :class:`~vkernels.torch_ops._dispatch.OpNotEligible` raises (the
   vkernels torch_ops dispatch contract — floe catches and falls back).
2. ``bs`` / ``max_len`` (decode) and ``bs`` / ``max_seqlen_k`` (verify)
   are now optional keyword arguments derived from tensor shapes when
   omitted (``seq_lens.numel()``; the narrowest table width). Passed
   explicitly they override — the CUDA-graph bucket pattern hands over
   buffers larger than the live batch.
3. Kernel-side: none. The Triton sources (including the
   ``do_not_specialize`` lists and the int64 output-row offsets) are
   byte-identical to the donor.

Numerics: pure integer metadata (int32 lengths, cumsums, table copies) —
parity with the eager oracle is exact, not tolerance-gated.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from ._dispatch import OpNotEligible

__all__ = [
    "bounded_scan_num_splits",
    "fused_dsa_decode_metadata",
    "fused_dsa_decode_metadata_eligible",
    "fused_dsa_decode_metadata_reference",
    "fused_dsa_target_verify_metadata",
    "fused_dsa_target_verify_metadata_eligible",
    "fused_dsa_target_verify_metadata_reference",
]

# The page-table column tile (donor constant, decode.py / verify.py).
_BLOCK_N = 128
# The split-scan program budget (donor constant, scan.py).
_TILE_PROGRAM_TARGET = 8192


def bounded_scan_num_splits(rows: int, num_col_blocks: int) -> int:
    """Keep the grid capture-safe while bounding traversal by replay-time data.

    Verbatim from the donor ``scan.py``: cap the per-row split count so the
    grid stays at ~``_TILE_PROGRAM_TARGET`` programs while each row's scan
    is still split across ``num_splits`` programs (each bounded by its
    row's live kv length at run time).
    """
    assert rows > 0
    return max(1, min(num_col_blocks, _TILE_PROGRAM_TARGET // rows))


def _pool_visible_tokens(seq_i32: torch.Tensor, dsa_index_topk: int, index_kpool: int):
    """The indexer-visible token count for per-row kv lengths (eager torch).

    SGLang's ``compute_dsa_seqlens`` verbatim — the oracle the kernel's
    ``index_kpool`` branch implements. ``index_kpool <= 1`` (plain DSA):
    ``min(seq, index_topk)``. Pool-aware: the top-k covers whole pools, so
    the visible history is the pool-aligned prefix capped at ``index_topk``
    tokens, PLUS the live partial pool at the end (the
    ``index_kpool_always_select_tail`` tail).
    """
    if index_kpool <= 1:
        return torch.minimum(seq_i32, torch.full_like(seq_i32, dsa_index_topk))
    full_pool_tokens = (
        torch.div(seq_i32, index_kpool, rounding_mode="floor") * index_kpool
    )
    selected_history_tokens = full_pool_tokens.clamp(max=dsa_index_topk)
    tail_tokens = seq_i32 - full_pool_tokens
    return selected_history_tokens + tail_tokens


def _live_cols(kv_lens: torch.Tensor, max_len: int, block_n: int = _BLOCK_N):
    """Columns of a page-table row the kernel writes: the row's live kv
    length rounded UP to the kernel's column tile, clamped to ``max_len``.

    (The donor bounds the split scan by ``cdiv(kv_len, BLOCK_N)`` but masks
    stores only by ``offs_n < max_len``, so a partial trailing tile IS
    written — stale ``req_to_token`` slots beyond ``kv_len`` land there.)
    """
    blocks = torch.div(kv_lens, block_n, rounding_mode="floor") + (
        (kv_lens % block_n) > 0
    ).to(kv_lens.dtype)
    return torch.minimum(blocks * block_n, torch.full_like(blocks, max_len))


def _table_max_len(
    req_to_token: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    real_page_table: Optional[torch.Tensor],
    real_page_size: int,
) -> int:
    """Default ``max_len``: the narrowest table width the outputs allow."""
    max_len = int(req_to_token.shape[1])
    if page_table_1 is not None:
        max_len = min(max_len, int(page_table_1.shape[1]))
    if real_page_size > 1 and real_page_table is not None:
        max_len = min(max_len, int(real_page_table.shape[1]) * real_page_size)
    return max_len


# ---------------------------------------------------------------------------
# kernels (verbatim from the donor; triton imported lazily, sgl_moe pattern)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _decode_metadata_kernel():
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=[
            "page_table_stride_0",
            "real_page_table_stride_0",
            "max_len",
            "num_splits",
        ]
    )
    def _fused_dsa_decode_metadata_kernel(
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens_stride: tl.constexpr,
        req_pool_indices_stride: tl.constexpr,
        req_to_token_stride_0: tl.constexpr,
        req_to_token_stride_1: tl.constexpr,
        page_table_stride_0,
        page_table_stride_1: tl.constexpr,
        real_page_table_stride_0,
        real_page_table_stride_1: tl.constexpr,
        bs: tl.constexpr,
        max_len,
        num_splits,
        dsa_index_topk: tl.constexpr,
        index_kpool: tl.constexpr,
        real_page_size: tl.constexpr,
        HAS_REAL_PAGE_TABLE: tl.constexpr,
        HAS_PAGE_TABLE_1: tl.constexpr,
        BLOCK_BS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)

        if pid == 0:
            offs_b = tl.arange(0, BLOCK_BS)
            mask_b = offs_b < bs
            seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
            seq_i32 = seq.to(tl.int32)
            if index_kpool <= 1:
                dsa_seq = tl.minimum(seq_i32, dsa_index_topk)
            else:
                # Preserve the live partial pool after selecting pool-aligned history.
                full_pool_tokens = (seq_i32 // index_kpool) * index_kpool
                selected_history_tokens = tl.minimum(full_pool_tokens, dsa_index_topk)
                tail_tokens = seq_i32 - full_pool_tokens
                dsa_seq = selected_history_tokens + tail_tokens

            cu = tl.cumsum(seq_i32, 0)
            dsa_cu = tl.cumsum(dsa_seq, 0)

            tl.store(cache_seqlens + offs_b, seq_i32, mask=mask_b)
            tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
            tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)
            tl.store(dsa_cache_seqlens + offs_b, dsa_seq, mask=mask_b)
            tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
            tl.store(dsa_cu_seqlens_k + 1 + offs_b, dsa_cu, mask=mask_b)
            return

        page_pid = pid - 1
        row = page_pid // num_splits
        split_id = page_pid - row * num_splits

        req_idx = tl.load(
            req_pool_indices + row * req_pool_indices_stride,
            mask=row < bs,
            other=0,
        )
        kv_len = tl.load(
            seq_lens + row * seq_lens_stride,
            mask=row < bs,
            other=0,
        ).to(tl.int32)
        # Page-table row offsets can overflow int32 at 1M context.
        row_i64 = row.to(tl.int64)
        num_live_blocks = tl.minimum(tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_len, BLOCK_N))
        # Three stages hide latency across strided copy iterations.
        for col_block in tl.range(split_id, num_live_blocks, num_splits, num_stages=3):
            offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (row < bs) & (offs_n < max_len)
            vals = tl.load(
                req_to_token
                + req_idx * req_to_token_stride_0
                + offs_n * req_to_token_stride_1,
                mask=mask,
                other=0,
            ).to(tl.int32)
            if HAS_PAGE_TABLE_1:
                tl.store(
                    page_table_1
                    + row_i64 * page_table_stride_0
                    + offs_n * page_table_stride_1,
                    vals,
                    mask=mask,
                )

            if HAS_REAL_PAGE_TABLE:
                real_mask = mask & ((offs_n % real_page_size) == 0)
                real_cols = offs_n // real_page_size
                tl.store(
                    real_page_table
                    + row_i64 * real_page_table_stride_0
                    + real_cols * real_page_table_stride_1,
                    vals // real_page_size,
                    mask=real_mask,
                )

    return _fused_dsa_decode_metadata_kernel


@functools.lru_cache(maxsize=1)
def _verify_metadata_kernel():
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=[
            "page_table_stride_0",
            "real_page_table_stride_0",
            "max_seqlen_k",
            "num_splits",
        ]
    )
    def _fused_dsa_target_verify_metadata_kernel(
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        paged_mqa_ctx_lens_2d,
        seq_lens_stride: tl.constexpr,
        req_pool_indices_stride: tl.constexpr,
        req_to_token_stride_0: tl.constexpr,
        req_to_token_stride_1: tl.constexpr,
        page_table_stride_0,
        page_table_stride_1: tl.constexpr,
        real_page_table_stride_0,
        real_page_table_stride_1: tl.constexpr,
        paged_mqa_ctx_lens_stride_0: tl.constexpr,
        paged_mqa_ctx_lens_stride_1: tl.constexpr,
        bs: tl.constexpr,
        max_seqlen_k,
        num_splits,
        dsa_index_topk: tl.constexpr,
        index_kpool: tl.constexpr,
        real_page_size: tl.constexpr,
        next_n: tl.constexpr,
        HAS_REAL_PAGE_TABLE: tl.constexpr,
        HAS_PAGED_MQA_CTX_LENS: tl.constexpr,
        HAS_PAGE_TABLE_1: tl.constexpr,
        BLOCK_BS: tl.constexpr,
        BLOCK_EXPANDED: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        expanded_size: tl.constexpr = bs * next_n

        if pid == 0:
            offs_b = tl.arange(0, BLOCK_BS)
            mask_b = offs_b < bs
            seq = tl.load(seq_lens + offs_b * seq_lens_stride, mask=mask_b, other=0)
            cache_seq = seq.to(tl.int32) + next_n
            cu = tl.cumsum(cache_seq, 0)

            tl.store(cache_seqlens + offs_b, cache_seq, mask=mask_b)
            tl.store(cu_seqlens_k, tl.full((), 0, tl.int32))
            tl.store(cu_seqlens_k + 1 + offs_b, cu, mask=mask_b)

            offs_e = tl.arange(0, BLOCK_EXPANDED)
            mask_e = offs_e < expanded_size
            req_row = offs_e // next_n
            draft_off = offs_e - req_row * next_n
            base_seq = tl.load(
                seq_lens + req_row * seq_lens_stride,
                mask=mask_e,
                other=0,
            ).to(tl.int32)
            expanded_seq = base_seq + draft_off + 1
            expanded_seq = tl.where(mask_e, expanded_seq, 0)
            if index_kpool <= 1:
                dsa_seq = tl.minimum(expanded_seq, dsa_index_topk)
            else:
                # Preserve the live partial pool after selecting pool-aligned history.
                full_pool_tokens = (expanded_seq // index_kpool) * index_kpool
                selected_history_tokens = tl.minimum(full_pool_tokens, dsa_index_topk)
                tail_tokens = expanded_seq - full_pool_tokens
                dsa_seq = selected_history_tokens + tail_tokens
            dsa_cu = tl.cumsum(dsa_seq, 0)

            tl.store(seqlens_expanded + offs_e, expanded_seq, mask=mask_e)
            tl.store(dsa_cache_seqlens + offs_e, dsa_seq, mask=mask_e)
            tl.store(dsa_cu_seqlens_k, tl.full((), 0, tl.int32))
            tl.store(dsa_cu_seqlens_k + 1 + offs_e, dsa_cu, mask=mask_e)

            if HAS_PAGED_MQA_CTX_LENS:
                tl.store(
                    paged_mqa_ctx_lens_2d
                    + req_row * paged_mqa_ctx_lens_stride_0
                    + draft_off * paged_mqa_ctx_lens_stride_1,
                    base_seq + next_n,
                    mask=mask_e,
                )
            return

        page_pid = pid - 1
        out_row = page_pid // num_splits
        split_id = page_pid - out_row * num_splits

        req_row = out_row // next_n
        req_idx = tl.load(
            req_pool_indices + req_row * req_pool_indices_stride,
            mask=out_row < expanded_size,
            other=0,
        )
        kv_len = (
            tl.load(
                seq_lens + req_row * seq_lens_stride,
                mask=out_row < expanded_size,
                other=0,
            ).to(tl.int32)
            + next_n
        )
        # Output-row offsets can overflow int32 at 1M context.
        out_row_i64 = out_row.to(tl.int64)
        num_live_blocks = tl.minimum(
            tl.cdiv(kv_len, BLOCK_N), tl.cdiv(max_seqlen_k, BLOCK_N)
        )
        for col_block in tl.range(split_id, num_live_blocks, num_splits, num_stages=3):
            offs_n = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (out_row < expanded_size) & (offs_n < max_seqlen_k)
            vals = tl.load(
                req_to_token
                + req_idx * req_to_token_stride_0
                + offs_n * req_to_token_stride_1,
                mask=mask,
                other=0,
            ).to(tl.int32)
            if HAS_PAGE_TABLE_1:
                tl.store(
                    page_table_1
                    + out_row_i64 * page_table_stride_0
                    + offs_n * page_table_stride_1,
                    vals,
                    mask=mask,
                )

            if HAS_REAL_PAGE_TABLE:
                real_mask = mask & ((offs_n % real_page_size) == 0)
                real_cols = offs_n // real_page_size
                tl.store(
                    real_page_table
                    + out_row_i64 * real_page_table_stride_0
                    + real_cols * real_page_table_stride_1,
                    vals // real_page_size,
                    mask=real_mask,
                )

    return _fused_dsa_target_verify_metadata_kernel


# ---------------------------------------------------------------------------
# decode wrapper + eligibility + reference
# ---------------------------------------------------------------------------


def _decode_violations(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    bs: int,
    max_len: int,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int,
) -> list:
    named = [
        ("seq_lens", seq_lens),
        ("req_pool_indices", req_pool_indices),
        ("req_to_token", req_to_token),
        ("cache_seqlens", cache_seqlens),
        ("cu_seqlens_k", cu_seqlens_k),
        ("dsa_cache_seqlens", dsa_cache_seqlens),
        ("dsa_cu_seqlens_k", dsa_cu_seqlens_k),
    ]
    if page_table_1 is not None:
        named.append(("page_table_1", page_table_1))
    if real_page_table is not None:
        named.append(("real_page_table", real_page_table))
    v: list = []
    if not all(t.is_cuda for _, t in named):
        v.append(
            "all metadata tensors must be CUDA-resident (got "
            + ", ".join(f"{n}={t.device}" for n, t in named)
            + ")"
        )
        return v
    dev = seq_lens.device
    for name, t in named[1:]:
        if t.device != dev:
            v.append(f"{name} on {t.device}, expected {dev}")
    # Index inputs accept int32/int64 (the kernel loads and converts).
    for name, t, ndim, minlen in (
        ("seq_lens", seq_lens, 1, bs),
        ("req_pool_indices", req_pool_indices, 1, bs),
        ("req_to_token", req_to_token, 2, 0),
    ):
        if t.dim() != ndim:
            v.append(f"{name} must be {ndim}-D, got {tuple(t.shape)}")
        if t.dtype not in (torch.int32, torch.int64):
            v.append(f"{name} must be int32/int64, got {t.dtype}")
        if t.dim() == 1 and t.numel() < minlen:
            v.append(f"{name} needs >= {minlen} entries, got {t.numel()}")
    # Outputs are pinned to int32 (the values the kernel stores).
    for name, t, minlen in (
        ("cache_seqlens", cache_seqlens, bs),
        ("cu_seqlens_k", cu_seqlens_k, bs + 1),
        ("dsa_cache_seqlens", dsa_cache_seqlens, bs),
        ("dsa_cu_seqlens_k", dsa_cu_seqlens_k, bs + 1),
    ):
        if t.dtype != torch.int32:
            v.append(f"{name} must be int32 (kernel stores int32), got {t.dtype}")
        if t.dim() != 1:
            v.append(f"{name} must be 1-D, got {tuple(t.shape)}")
        elif t.numel() < minlen:
            v.append(f"{name} needs >= {minlen} entries, got {t.numel()}")
    if req_to_token.dim() == 2 and req_to_token.shape[1] < max_len:
        v.append(
            f"req_to_token width {req_to_token.shape[1]} < max_len {max_len}"
        )
    # Table presence/size requirements apply to the live (bs > 0) path only:
    # the donor returns from its bs == 0 early-out before touching tables.
    if bs > 0:
        if page_table_1 is not None:
            if page_table_1.dim() != 2:
                v.append(f"page_table_1 must be 2-D, got {tuple(page_table_1.shape)}")
            elif page_table_1.shape[0] < bs or page_table_1.shape[1] < max_len:
                v.append(
                    f"page_table_1 {tuple(page_table_1.shape)} too small for "
                    f"[{bs}, {max_len}]"
                )
            if page_table_1.dtype != torch.int32:
                v.append(f"page_table_1 must be int32, got {page_table_1.dtype}")
        if real_page_size == 1:
            if page_table_1 is None:
                v.append("real_page_size == 1 requires page_table_1 (the wide table IS the real table)")
        else:
            if real_page_table is None:
                v.append(f"real_page_table is required when real_page_size ({real_page_size}) > 1")
            else:
                if real_page_table.dim() != 2:
                    v.append(f"real_page_table must be 2-D, got {tuple(real_page_table.shape)}")
                elif (
                    real_page_table.shape[0] < bs
                    or real_page_table.shape[1] < (max_len + real_page_size - 1) // real_page_size
                ):
                    v.append(
                        f"real_page_table {tuple(real_page_table.shape)} too small for "
                        f"[{bs}, {(max_len + real_page_size - 1) // real_page_size}]"
                    )
                if real_page_table.dtype != torch.int32:
                    v.append(f"real_page_table must be int32, got {real_page_table.dtype}")
    if bs < 0:
        v.append(f"bs must be >= 0, got {bs}")
    if max_len < 0:
        v.append(f"max_len must be >= 0, got {max_len}")
    if dsa_index_topk < 1:
        v.append(f"dsa_index_topk must be >= 1, got {dsa_index_topk}")
    if index_kpool < 1:
        v.append(f"index_kpool must be >= 1, got {index_kpool}")
    if real_page_size < 1:
        v.append(f"real_page_size must be >= 1, got {real_page_size}")
    return v


def fused_dsa_decode_metadata_eligible(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    *,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int = 1,
    bs: Optional[int] = None,
    max_len: Optional[int] = None,
) -> bool:
    """Contract check for :func:`fused_dsa_decode_metadata` (no device sync)."""
    if bs is None:
        bs = int(seq_lens.numel())
    if max_len is None:
        max_len = _table_max_len(req_to_token, page_table_1, real_page_table, real_page_size)
    return not _decode_violations(
        seq_lens, req_pool_indices, req_to_token, cache_seqlens, cu_seqlens_k,
        page_table_1, dsa_cache_seqlens, dsa_cu_seqlens_k, real_page_table,
        bs, max_len, dsa_index_topk, real_page_size, index_kpool,
    )


def fused_dsa_decode_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    *,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int = 1,
    bs: Optional[int] = None,
    max_len: Optional[int] = None,
) -> None:
    """Fill decode DSA metadata (seqlens + cu_seqlens + page tables) in one launch.

    Writes, into CALLER-allocated int32 buffers: ``cache_seqlens`` ``[bs]``
    (= ``seq_lens`` as int32), ``cu_seqlens_k`` ``[bs+1]`` (zero-prefixed
    cumsum), ``dsa_cache_seqlens`` ``[bs]`` / ``dsa_cu_seqlens_k``
    ``[bs+1]`` (the pool-aware indexer-visible lengths and their cumsum),
    ``page_table_1`` ``[bs, max_len]`` (wide table; ``None`` skips it) and
    ``real_page_table`` ``[bs, ceil(max_len/real_page_size)]`` (compact
    table; with ``real_page_size == 1`` the wide table serves as both).

    ``bs`` / ``max_len`` default to ``seq_lens.numel()`` and the narrowest
    table width — pass them explicitly for CUDA-graph buckets whose
    buffers exceed the live batch. Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` on a contract
    miss (CPU tensors, wrong dtypes, undersized buffers) — callers gate on
    :func:`fused_dsa_decode_metadata_eligible` and keep SGLang's eager
    chain (the oracle, :func:`fused_dsa_decode_metadata_reference`) as the
    fallback. ``bs == 0`` just zeroes the two cu_seqlens heads (capture-
    safe, no launch). Sync-free and allocation-free: CUDA-graph safe once
    compiled (warm the shapes outside capture).
    """
    if bs is None:
        bs = int(seq_lens.numel())
    if max_len is None:
        max_len = _table_max_len(req_to_token, page_table_1, real_page_table, real_page_size)
    violations = _decode_violations(
        seq_lens, req_pool_indices, req_to_token, cache_seqlens, cu_seqlens_k,
        page_table_1, dsa_cache_seqlens, dsa_cu_seqlens_k, real_page_table,
        bs, max_len, dsa_index_topk, real_page_size, index_kpool,
    )
    if violations:
        raise OpNotEligible(
            "fused_dsa_decode_metadata contract: " + "; ".join(violations)
        )
    if bs == 0:
        cu_seqlens_k[:1].zero_()
        dsa_cu_seqlens_k[:1].zero_()
        return

    import triton

    has_real_page_table = real_page_size > 1
    # page_size==1: real IS page_table_1, so page_table_1 must be present.
    if not has_real_page_table:
        real_page_table = page_table_1
    # page_table_1 (the wide page_size=1 table) may be dropped for the fused
    # decode CUDA graph; the kernel then writes only real_page_table.
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        page_table_1 = real_page_table  # dummy pointer for stride args

    block_bs = triton.next_power_of_2(bs)
    num_col_blocks = triton.cdiv(max_len, _BLOCK_N)
    num_splits = bounded_scan_num_splits(bs, num_col_blocks)
    grid = (1 + bs * num_splits,)

    _decode_metadata_kernel()[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        bs,
        max_len,
        num_splits,
        dsa_index_topk,
        index_kpool,
        real_page_size,
        has_real_page_table,
        has_page_table_1,
        BLOCK_BS=block_bs,
        BLOCK_N=_BLOCK_N,
    )


def fused_dsa_decode_metadata_reference(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int = 1,
    page_table_1: Optional[torch.Tensor] = None,
    real_page_table: Optional[torch.Tensor] = None,
    max_len: Optional[int] = None,
    block_n: int = _BLOCK_N,
) -> dict:
    """Eager torch oracle for :func:`fused_dsa_decode_metadata`.

    Recomputes SGLang's decode metadata chain (the ``dsa_backend.py``
    eager fallback) with plain torch ops and returns FRESH tensors in a
    dict: ``cache_seqlens``, ``cu_seqlens_k``, ``dsa_cache_seqlens``,
    ``dsa_cu_seqlens_k``, ``page_table_1`` (None when the input is),
    ``real_page_table``, ``num_splits`` (the launch's split count). Runs
    on any device (CPU included). Page-table entries the kernel does NOT
    write (beyond each row's live prefix — see module CAPTURE SAFETY) are
    ``-1`` here; the kernel leaves those slots untouched.
    """
    bs = int(seq_lens.numel())
    if max_len is None:
        max_len = _table_max_len(req_to_token, page_table_1, real_page_table, real_page_size)
    dev = seq_lens.device

    seq = seq_lens.to(torch.int32)
    cache_seqlens = seq.clone()
    cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    cu_seqlens_k[1:] = torch.cumsum(seq, 0, dtype=torch.int32)
    dsa_cache_seqlens = _pool_visible_tokens(seq, dsa_index_topk, index_kpool)
    dsa_cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    dsa_cu_seqlens_k[1:] = torch.cumsum(dsa_cache_seqlens, 0, dtype=torch.int32)

    rpi = req_pool_indices.to(torch.int64)
    src = req_to_token.to(torch.int64)[rpi][:, :max_len]
    live = _live_cols(seq.to(torch.int64), max_len, block_n)  # [bs]
    cols = torch.arange(max_len, device=dev)
    live_mask = cols[None, :] < live[:, None]

    pt1 = None
    if page_table_1 is not None:
        pt1 = torch.where(live_mask, src.to(torch.int32), torch.full_like(src.to(torch.int32), -1))

    real = None
    if real_page_size == 1:
        real = pt1  # the wide table IS the real table (kernel contract)
    elif real_page_table is not None:
        pages = (max_len + real_page_size - 1) // real_page_size
        page_cols = torch.arange(pages, device=dev) * real_page_size
        real_live = page_cols[None, :] < live[:, None]
        rsrc = (src[:, ::real_page_size]).to(torch.int64)
        rsrc = rsrc[:, :pages] // real_page_size
        real = torch.where(
            real_live, rsrc.to(torch.int32), torch.full_like(rsrc.to(torch.int32), -1)
        )

    num_col_blocks = -(-max_len // block_n)  # ceil
    return {
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_k": cu_seqlens_k,
        "dsa_cache_seqlens": dsa_cache_seqlens,
        "dsa_cu_seqlens_k": dsa_cu_seqlens_k,
        "page_table_1": pt1,
        "real_page_table": real,
        "num_splits": bounded_scan_num_splits(bs, num_col_blocks) if bs else 1,
    }


# ---------------------------------------------------------------------------
# target-verify wrapper + eligibility + reference
# ---------------------------------------------------------------------------


def _verify_violations(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor],
    bs: int,
    max_seqlen_k: int,
    dsa_index_topk: int,
    real_page_size: int,
    next_n: int,
    index_kpool: int,
) -> list:
    expanded = bs * next_n
    named = [
        ("seq_lens", seq_lens),
        ("req_pool_indices", req_pool_indices),
        ("req_to_token", req_to_token),
        ("cache_seqlens", cache_seqlens),
        ("cu_seqlens_k", cu_seqlens_k),
        ("seqlens_expanded", seqlens_expanded),
        ("dsa_cache_seqlens", dsa_cache_seqlens),
        ("dsa_cu_seqlens_k", dsa_cu_seqlens_k),
    ]
    if page_table_1 is not None:
        named.append(("page_table_1", page_table_1))
    if real_page_table is not None:
        named.append(("real_page_table", real_page_table))
    if paged_mqa_ctx_lens_2d is not None:
        named.append(("paged_mqa_ctx_lens_2d", paged_mqa_ctx_lens_2d))
    v: list = []
    if not all(t.is_cuda for _, t in named):
        v.append(
            "all metadata tensors must be CUDA-resident (got "
            + ", ".join(f"{n}={t.device}" for n, t in named)
            + ")"
        )
        return v
    dev = seq_lens.device
    for name, t in named[1:]:
        if t.device != dev:
            v.append(f"{name} on {t.device}, expected {dev}")
    for name, t, ndim, minlen in (
        ("seq_lens", seq_lens, 1, bs),
        ("req_pool_indices", req_pool_indices, 1, bs),
        ("req_to_token", req_to_token, 2, 0),
    ):
        if t.dim() != ndim:
            v.append(f"{name} must be {ndim}-D, got {tuple(t.shape)}")
        if t.dtype not in (torch.int32, torch.int64):
            v.append(f"{name} must be int32/int64, got {t.dtype}")
        if t.dim() == 1 and t.numel() < minlen:
            v.append(f"{name} needs >= {minlen} entries, got {t.numel()}")
    for name, t, minlen in (
        ("cache_seqlens", cache_seqlens, bs),
        ("cu_seqlens_k", cu_seqlens_k, bs + 1),
        ("seqlens_expanded", seqlens_expanded, expanded),
        ("dsa_cache_seqlens", dsa_cache_seqlens, expanded),
        ("dsa_cu_seqlens_k", dsa_cu_seqlens_k, expanded + 1),
    ):
        if t.dtype != torch.int32:
            v.append(f"{name} must be int32 (kernel stores int32), got {t.dtype}")
        if t.dim() != 1:
            v.append(f"{name} must be 1-D, got {tuple(t.shape)}")
        elif t.numel() < minlen:
            v.append(f"{name} needs >= {minlen} entries, got {t.numel()}")
    if req_to_token.dim() == 2 and req_to_token.shape[1] < max_seqlen_k:
        v.append(f"req_to_token width {req_to_token.shape[1]} < max_seqlen_k {max_seqlen_k}")
    # The verify page tables are indexed by EXPANDED rows (bs * next_n).
    for name, t in (("page_table_1", page_table_1), ("real_page_table", real_page_table)):
        if t is None:
            continue
        if t.dim() != 2:
            v.append(f"{name} must be 2-D, got {tuple(t.shape)}")
        elif t.shape[0] < expanded:
            v.append(f"{name} has {t.shape[0]} rows < expanded rows {expanded}")
        if t.dtype != torch.int32:
            v.append(f"{name} must be int32, got {t.dtype}")
    if page_table_1 is not None and page_table_1.dim() == 2 and page_table_1.shape[1] < max_seqlen_k:
        v.append(f"page_table_1 width {page_table_1.shape[1]} < max_seqlen_k {max_seqlen_k}")
    if real_page_size == 1:
        if page_table_1 is None:
            v.append("real_page_size == 1 requires page_table_1 (the wide table IS the real table)")
    else:
        pages = (max_seqlen_k + real_page_size - 1) // real_page_size
        if real_page_table is None:
            v.append(f"real_page_table is required when real_page_size ({real_page_size}) > 1")
        elif real_page_table.dim() == 2 and (
            real_page_table.shape[0] < expanded or real_page_table.shape[1] < pages
        ):
            v.append(
                f"real_page_table {tuple(real_page_table.shape)} too small for "
                f"[{expanded}, {pages}]"
            )
    if paged_mqa_ctx_lens_2d is not None:
        if paged_mqa_ctx_lens_2d.dtype != torch.int32:
            v.append(f"paged_mqa_ctx_lens_2d must be int32, got {paged_mqa_ctx_lens_2d.dtype}")
        if paged_mqa_ctx_lens_2d.dim() != 2 or tuple(paged_mqa_ctx_lens_2d.shape) != (bs, next_n):
            v.append(
                f"paged_mqa_ctx_lens_2d must be [{bs}, {next_n}], got "
                f"{tuple(paged_mqa_ctx_lens_2d.shape)}"
            )
    if bs < 1:
        v.append(f"bs must be >= 1 for target-verify metadata, got {bs}")
    if next_n < 1:
        v.append(f"next_n must be >= 1, got {next_n}")
    if max_seqlen_k < 0:
        v.append(f"max_seqlen_k must be >= 0, got {max_seqlen_k}")
    if dsa_index_topk < 1:
        v.append(f"dsa_index_topk must be >= 1, got {dsa_index_topk}")
    if index_kpool < 1:
        v.append(f"index_kpool must be >= 1, got {index_kpool}")
    if real_page_size < 1:
        v.append(f"real_page_size must be >= 1, got {real_page_size}")
    return v


def fused_dsa_target_verify_metadata_eligible(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    *,
    next_n: int,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int = 1,
    bs: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None,
) -> bool:
    """Contract check for :func:`fused_dsa_target_verify_metadata` (no sync)."""
    if bs is None:
        bs = int(seq_lens.numel())
    if max_seqlen_k is None:
        max_seqlen_k = _table_max_len(req_to_token, page_table_1, real_page_table, real_page_size)
    return not _verify_violations(
        seq_lens, req_pool_indices, req_to_token, cache_seqlens, cu_seqlens_k,
        page_table_1, seqlens_expanded, dsa_cache_seqlens, dsa_cu_seqlens_k,
        real_page_table, paged_mqa_ctx_lens_2d, bs, max_seqlen_k,
        dsa_index_topk, real_page_size, next_n, index_kpool,
    )


def fused_dsa_target_verify_metadata(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table_1: Optional[torch.Tensor],
    seqlens_expanded: torch.Tensor,
    dsa_cache_seqlens: torch.Tensor,
    dsa_cu_seqlens_k: torch.Tensor,
    real_page_table: Optional[torch.Tensor],
    *,
    next_n: int,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int = 1,
    bs: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None,
) -> None:
    """Fill target-verify (speculative decoding) DSA metadata in one launch.

    Same contract as :func:`fused_dsa_decode_metadata`, but each request
    carries ``next_n`` draft tokens: ``cache_seqlens`` = ``seq_lens +
    next_n``; the per-draft-token expanded rows (``bs * next_n`` of them)
    get ``seqlens_expanded`` = ``seq_lens + draft_off + 1``, their own
    pool-aware ``dsa_cache_seqlens`` / ``dsa_cu_seqlens_k`` and page-table
    rows, and (optionally) ``paged_mqa_ctx_lens_2d`` ``[bs, next_n]`` =
    ``seq_lens + next_n``. ``bs >= 1`` here (a verify batch is never
    empty; ``bs == 0`` is a decode-graph concern only).
    """
    if bs is None:
        bs = int(seq_lens.numel())
    if max_seqlen_k is None:
        max_seqlen_k = _table_max_len(req_to_token, page_table_1, real_page_table, real_page_size)
    violations = _verify_violations(
        seq_lens, req_pool_indices, req_to_token, cache_seqlens, cu_seqlens_k,
        page_table_1, seqlens_expanded, dsa_cache_seqlens, dsa_cu_seqlens_k,
        real_page_table, paged_mqa_ctx_lens_2d, bs, max_seqlen_k,
        dsa_index_topk, real_page_size, next_n, index_kpool,
    )
    if violations:
        raise OpNotEligible(
            "fused_dsa_target_verify_metadata contract: " + "; ".join(violations)
        )

    import triton

    expanded_size = bs * next_n
    has_real_page_table = real_page_size > 1
    if not has_real_page_table:
        real_page_table = page_table_1
    has_page_table_1 = page_table_1 is not None
    if not has_page_table_1:
        page_table_1 = real_page_table  # dummy pointer for stride args

    has_paged_mqa_ctx_lens = paged_mqa_ctx_lens_2d is not None
    if not has_paged_mqa_ctx_lens:
        paged_mqa_ctx_lens_2d = page_table_1  # dummy pointer for stride args

    block_bs = triton.next_power_of_2(bs)
    block_expanded = triton.next_power_of_2(expanded_size)
    num_col_blocks = triton.cdiv(max_seqlen_k, _BLOCK_N)
    num_splits = bounded_scan_num_splits(expanded_size, num_col_blocks)
    grid = (1 + expanded_size * num_splits,)

    _verify_metadata_kernel()[grid](
        seq_lens,
        req_pool_indices,
        req_to_token,
        cache_seqlens,
        cu_seqlens_k,
        page_table_1,
        seqlens_expanded,
        dsa_cache_seqlens,
        dsa_cu_seqlens_k,
        real_page_table,
        paged_mqa_ctx_lens_2d,
        seq_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        req_to_token.stride(1),
        page_table_1.stride(0),
        page_table_1.stride(1),
        real_page_table.stride(0) if has_real_page_table else 0,
        real_page_table.stride(1) if has_real_page_table else 0,
        paged_mqa_ctx_lens_2d.stride(0) if has_paged_mqa_ctx_lens else 0,
        paged_mqa_ctx_lens_2d.stride(1) if has_paged_mqa_ctx_lens else 0,
        bs,
        max_seqlen_k,
        num_splits,
        dsa_index_topk,
        index_kpool,
        real_page_size,
        next_n,
        has_real_page_table,
        has_paged_mqa_ctx_lens,
        has_page_table_1,
        BLOCK_BS=block_bs,
        BLOCK_EXPANDED=block_expanded,
        BLOCK_N=_BLOCK_N,
    )


def fused_dsa_target_verify_metadata_reference(
    seq_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    *,
    next_n: int,
    dsa_index_topk: int,
    real_page_size: int,
    index_kpool: int = 1,
    page_table_1: Optional[torch.Tensor] = None,
    real_page_table: Optional[torch.Tensor] = None,
    paged_mqa_ctx_lens_2d: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    block_n: int = _BLOCK_N,
) -> dict:
    """Eager torch oracle for :func:`fused_dsa_target_verify_metadata`.

    Same return convention as
    :func:`fused_dsa_decode_metadata_reference` (fresh tensors in a dict;
    unwritten page-table slots are ``-1``), plus ``seqlens_expanded`` and
    ``paged_mqa_ctx_lens_2d`` (None when the input is). Runs on any device.
    """
    bs = int(seq_lens.numel())
    expanded = bs * next_n
    if max_seqlen_k is None:
        max_seqlen_k = _table_max_len(req_to_token, page_table_1, real_page_table, real_page_size)
    dev = seq_lens.device

    seq = seq_lens.to(torch.int32)
    cache_seqlens = seq + next_n
    cu_seqlens_k = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    cu_seqlens_k[1:] = torch.cumsum(cache_seqlens, 0, dtype=torch.int32)

    e = torch.arange(expanded, device=dev)
    req_row = e // next_n
    draft_off = e - req_row * next_n
    expanded_seq = seq[req_row] + draft_off.to(torch.int32) + 1
    seqlens_expanded = expanded_seq.clone()
    dsa_cache_seqlens = _pool_visible_tokens(expanded_seq, dsa_index_topk, index_kpool)
    dsa_cu_seqlens_k = torch.zeros(expanded + 1, dtype=torch.int32, device=dev)
    dsa_cu_seqlens_k[1:] = torch.cumsum(dsa_cache_seqlens, 0, dtype=torch.int32)

    pmqa = None
    if paged_mqa_ctx_lens_2d is not None:
        pmqa = (seq[req_row] + next_n).view(bs, next_n).clone()

    # Page tables over EXPANDED rows; each expanded row's kv length is its
    # request's seq_len + next_n (the kernel writes the full draft window).
    rpi = req_pool_indices.to(torch.int64)[req_row]
    src = req_to_token.to(torch.int64)[rpi][:, :max_seqlen_k]
    kv = seq[req_row].to(torch.int64) + next_n
    live = _live_cols(kv, max_seqlen_k, block_n)  # [expanded]
    cols = torch.arange(max_seqlen_k, device=dev)
    live_mask = cols[None, :] < live[:, None]

    pt1 = None
    if page_table_1 is not None:
        pt1 = torch.where(live_mask, src.to(torch.int32), torch.full_like(src.to(torch.int32), -1))

    real = None
    if real_page_size == 1:
        real = pt1
    elif real_page_table is not None:
        pages = (max_seqlen_k + real_page_size - 1) // real_page_size
        page_cols = torch.arange(pages, device=dev) * real_page_size
        real_live = page_cols[None, :] < live[:, None]
        rsrc = src[:, ::real_page_size].to(torch.int64)
        rsrc = rsrc[:, :pages] // real_page_size
        real = torch.where(
            real_live, rsrc.to(torch.int32), torch.full_like(rsrc.to(torch.int32), -1)
        )

    num_col_blocks = -(-max_seqlen_k // block_n)  # ceil
    return {
        "cache_seqlens": cache_seqlens,
        "cu_seqlens_k": cu_seqlens_k,
        "seqlens_expanded": seqlens_expanded,
        "dsa_cache_seqlens": dsa_cache_seqlens,
        "dsa_cu_seqlens_k": dsa_cu_seqlens_k,
        "paged_mqa_ctx_lens_2d": pmqa,
        "page_table_1": pt1,
        "real_page_table": real,
        "num_splits": bounded_scan_num_splits(expanded, num_col_blocks) if expanded else 1,
    }
