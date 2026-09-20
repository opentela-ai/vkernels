"""vLLM Triton sparse-MLA (DSA) attention kernels, vendored for floe GLM-5.3-Flash.

Source: vllm-project/vllm ``main`` branch, fetched 2026-01 from
https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
(exact commit pin not recorded at fetch time; re-diff before upgrading).

Self-contained: imports only ``math``, ``functools``, ``torch``, ``triton``.
All vllm-v1 machinery (CustomOp, platform detection, capture decorators) and
the aiter top-k indexer / fp8 paged MQA logits / GPTJ inverse-rope helpers
are dropped — floe's ``Glm53Indexer`` produces the top-k selection and floe
forms its own absorbed queries.

CUT LIST
--------
Kept (verbatim unless noted):
  ``_pack_dense_prefix_to_ragged_kernel`` / ``build_ragged_indices_from_dense``
  ``_as_int32_contiguous_1d`` / ``_sparse_kv_row_offset``
  ``_sparse_attn_prefill_ragged_kernel``            (verbatim)
  ``_sparse_attn_decode_reduce_kernel``             (verbatim)
  ``_decode_cu_count`` / ``_decode_partial_iters`` / ``_decode_num_splits``
  prefill wrappers ``sparse_attn_prefill{,_ragged}`` (adapted signature only)
Dropped (reason):
  all ``_gfx950*`` kernels + ``_decode_gfx950_num_splits`` + the
    ``extra_cache_nan_free`` machinery  — MI300A is gfx942/CDNA3; the gfx950
    paths are dead code here.
  aiter top-k indexer, ``fp8_paged_mqa_logits*``, ``rocm_aiter_sparse_attn_indexer``
    — floe runs its own lightning indexer (``Glm53Indexer``) emitting int32
    ``[B, S_q, W]`` token ids with ``-1`` masking.
  ``_inverse_rope_gptj_kernel`` / ``_fused_inverse_rope_gptj`` /
    ``_get_cached_wo_a_bf16`` / ``rocm_inv_rope_einsum`` — floe computes
    absorbed queries its own way (``q~[h] = q[h] . W_k[h]^T``).
  ``_indexer_k_quant_and_cache_kernel`` / ``cp_gather_indexer_*`` — indexer
    fp8 cache plumbing, not needed for the attention core.
  ``_sparse_attn_decode_ragged_kernel`` (non-split generic) — the split-K
    partial+reduce pair covers every decode shape (num_splits collapses to 1
    at large batch); keeping one decode path halves the adapted surface.
  DSV4 dim validation (``_validate_dsv4_sparse_dims``, 448/64) — floe dims
    are validated in the wrappers instead.

FLOE CONTRACT MAPPING (GLM-5.3-Flash real dims)
-----------------------------------------------
  floe ``q_abs``  [B, S_q, H, K] bf16   (absorbed queries q~[h]=q[h].W_k[h]^T)
      -> kernel q [B*S_q, H, K], K = ``kv_lora_rank`` = 512 (NoPE, rope = 0)
  floe latent cache [B, S, K] bf16      (post ``kv_a_layernorm`` compressed kv)
      -> kernel cache rows [B*S, K]; per-batch rows are folded into one global
      row space by offsetting the indices (``idx' = idx + b*S`` for idx >= 0),
      which removes vLLM's paged fp8 cache AND the per-request loop the HIP
      path needs for B > 1.
  floe indexer output [B, S_q, W] int32, ``-1`` = masked,
      W = ``index_topk + index_kpool - 1`` = 2051
      -> kernel dense indices [B*S_q, W] (ragged conversion in ``build_ragged_
      indices_from_dense``); ``-1`` slots are masked exactly like floe's
      reference.
  floe ``scaling`` = ``qk_head_dim**-0.5`` = 256**-0.5 (absorbed form — NOT
      ``1/sqrt(head_dim=512)``) -> kernel ``scale`` in natural-exp units
      (the Triton kernels use ``tl.exp``; no log2(e) prefactor is needed —
      the log2 prefactor in floe's HIP ABI is handled inside
      :func:`dsa_sparse_fwd`).
  kernel out [B*S_q, H, K] bf16 = floe ``out_latent`` (un-absorb by the
      ``W_v`` slice happens in floe, outside the kernel).
  ``attn_sink``: floe has none; the sink branches are kept (verbatim) and
      pass ``None``.

DEVIATIONS FROM THE SOURCE (all documented, none silent)
--------------------------------------------------------
1. Cache load path: the DSV4 decode kernels read a uint8 ``fp8_ds_mla``
   cache (fp8 payload + e8m0 per-64 scales at hardcoded byte offsets 576/8).
   floe's latent cache is plain bf16 ``[rows, K]`` — the decode partial
   kernel loads bf16 rows slot-indexed directly (same addressing as the
   prefill kernel). The fp8/e8m0 decode, IS_FNUZ switches, and the NaN guard
   on decoded fp8 payloads are therefore gone (bf16 cache holds no decode
   artifacts); nope/rope are fused into one head_dim (floe DSA is NoPE,
   ``tail_dim = 0`` — asserted in :func:`dsa_sparse_fwd`).
2. SWA main/extra cache split: floe has a single latent cache per layer, so
   the partial kernel drops the ``HAS_EXTRA`` segment (the reduce kernel's
   split-merge math is unchanged).
3. Batch folding (index offset) as described above — pure index arithmetic,
   no numerics change.
4. ``adaptive_splits`` is kept as a verbatim reduce-kernel constexpr but is
   always launched ``False`` (it only ever fires on gfx950).

Parity gate: ``bench`` harness (job-side) compares against floe's exact
eager reference (``Glm53Attention._sparse_attention_torch`` math) in fp32
and fp64; the kernel pays bf16 P·V rounding the fp32 reference does not.
"""

from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl

__all__ = [
    "build_ragged_indices_from_dense",
    "sparse_attn_prefill",
    "sparse_attn_decode",
    "dsa_sparse_fwd",
]


# ---------------------------------------------------------------------------
# ragged index helpers (verbatim from the source)
# ---------------------------------------------------------------------------


@triton.jit
def _pack_dense_prefix_to_ragged_kernel(
    indices_ptr,
    lengths_ptr,
    indptr_ptr,
    out_ptr,
    indices_stride0,
    num_rows_limit,
    row_width,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    row_len = tl.load(lengths_ptr + row_idx)
    if block_idx * BLOCK_SIZE >= row_len:
        return

    mask = offsets < row_len
    safe_offsets = tl.where(offsets < row_width, offsets, 0)
    vals = tl.load(
        indices_ptr + row_idx * indices_stride0 + safe_offsets,
        mask=mask & (offsets < row_width),
        other=-1,
    ).to(tl.int32)
    if num_rows_limit >= 0:
        vals = tl.where((vals >= 0) & (vals < num_rows_limit), vals, -1)

    out_start = tl.load(indptr_ptr + row_idx)
    tl.store(out_ptr + out_start + offsets, vals, mask=mask)


def build_ragged_indices_from_dense(
    indices: torch.Tensor,
    lengths: torch.Tensor,
    num_rows: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = indices.reshape(indices.shape[0], -1)
    lengths = lengths.to(device=indices.device, dtype=torch.int32).reshape(-1)
    assert lengths.numel() == indices.shape[0], (
        f"Expected one length per row, got {lengths.shape} for indices {indices.shape}"
    )

    max_width = indices.shape[1] if indices.ndim == 2 else 0
    lengths = lengths.clamp(min=0, max=max_width).contiguous()

    indptr = torch.zeros(indices.shape[0] + 1, dtype=torch.int32, device=indices.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])

    if indices.numel() == 0:
        flat = torch.empty(0, dtype=torch.int32, device=indices.device)
    else:
        flat = torch.empty(
            indices.shape[0] * max_width,
            dtype=torch.int32,
            device=indices.device,
        )
        if flat.numel() > 0:
            block_size = 128
            _pack_dense_prefix_to_ragged_kernel[
                (indices.shape[0], triton.cdiv(max_width, block_size))
            ](
                indices,
                lengths,
                indptr,
                flat,
                indices.stride(0),
                int(num_rows),
                max_width,
                BLOCK_SIZE=block_size,
            )

    return flat, indptr


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


@triton.jit
def _sparse_kv_row_offset(slot, stride):
    # A global token slot fits in int32, but its byte/element offset may not.
    return slot.to(tl.int64) * stride


# ---------------------------------------------------------------------------
# sparse prefill (verbatim kernel; wrapper adapted to the flat cache layout)
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_prefill_ragged_kernel(
    q_ptr,
    kv_ptr,
    kv_indices_ptr,
    kv_indptr_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    kv_stride_n,
    kv_stride_d,
    out_stride_t,
    out_stride_h,
    out_stride_d,
    num_heads,
    head_dim,
    num_kv,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :] * q_stride_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    kv_start = tl.load(kv_indptr_ptr + query_idx)
    kv_end = tl.load(kv_indptr_ptr + query_idx + 1)
    kv_len = kv_end - kv_start

    k_offsets = tl.arange(0, BLOCK_K)
    slot = tl.load(
        kv_indices_ptr + kv_start + k_offsets, mask=k_offsets < kv_len, other=-1
    )
    for k_start in tl.range(0, kv_len, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < kv_len
        valid = in_range & (slot >= 0) & (slot < num_kv)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            kv_ptr
            + _sparse_kv_row_offset(safe_slot[:, None], kv_stride_n)
            + dim_offsets[None, :] * kv_stride_d,
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        next_k_pos = k_start + BLOCK_K + k_offsets
        slot = tl.load(
            kv_indices_ptr + kv_start + next_k_pos, mask=next_k_pos < kv_len, other=-1
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_final)
        l_final = l_i * alpha + tl.exp(sink - m_final)
        denom = tl.maximum(l_final, 1.0e-30)
        out = tl.where(
            l_final[:, None] > 0.0,
            (acc * alpha[:, None]) / denom[:, None],
            0.0,
        )
    else:
        denom = tl.maximum(l_i, 1.0e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :] * out_stride_d,
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def _validate_floe_sparse_dims(head_dim: int, op_name: str) -> None:
    """Floe DSA: NoPE absorbed latent form — one dense head dim, no rope."""
    assert head_dim > 0, f"{op_name} expected a positive head_dim, got {head_dim}"
    assert head_dim % 16 == 0, (
        f"{op_name} expected head_dim to be a multiple of 16 (tl.dot), got {head_dim}"
    )


def sparse_attn_prefill_ragged(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    indptr: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged prefill over selected kv rows.

    ``q`` [sq, h, d] (bf16), ``kv`` [skv, d] flat bf16 rows, ``indices``
    [nnz] int32 + ``indptr`` [sq+1] int32 ragged selection (``-1`` masked),
    ``scale`` natural-exp units. Returns bf16 [sq, h, d].
    """
    assert q.ndim == 3, f"expected q=[sq,h,d], got {q.shape}"
    assert kv.ndim == 2, f"expected kv=[skv,d], got {kv.shape}"
    assert indices.ndim == 1, f"expected indices=[nnz], got {indices.shape}"
    assert indptr.ndim == 1, f"expected indptr=[sq+1], got {indptr.shape}"
    assert not q.is_cpu and not kv.is_cpu and not indices.is_cpu and not indptr.is_cpu

    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {indptr.shape}"
    )
    _validate_floe_sparse_dims(head_dim, "sparse_attn_prefill_ragged")

    block_h = 16
    block_d = triton.next_power_of_2(head_dim)
    block_k = 16 if head_dim >= 256 else 32
    num_warps = 4
    out = torch.empty_like(q, dtype=torch.bfloat16)
    _sparse_attn_prefill_ragged_kernel[(num_queries, triton.cdiv(num_heads, block_h))](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        float(scale),
        HAS_ATTN_SINK=has_attn_sink,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return out


def sparse_attn_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense-indices prefill: ``q`` [sq, h, d], ``kv`` [skv, d], ``indices``
    [sq, w] int32 with ``-1`` masked (``topk_length`` [sq] overrides the
    per-row valid count when given). Returns bf16 [sq, h, d]."""
    ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
        indices,
        topk_length
        if topk_length is not None
        else (indices >= 0).sum(dim=-1, dtype=torch.int32),
        num_rows=kv.shape[0],
    )
    return sparse_attn_prefill_ragged(
        q=q,
        kv=kv,
        indices=ragged_indices,
        indptr=ragged_indptr,
        scale=scale,
        attn_sink=attn_sink,
    )


# ---------------------------------------------------------------------------
# split-K decode — partial kernel is a bf16-cache ADAPTATION of the source's
# ``_sparse_attn_decode_partial_kernel`` (fp8_ds_mla load path replaced by the
# prefill kernel's bf16 row load; nope/rope fused; SWA extra segment dropped).
# The split slicing, online-softmax partial state and store layout are kept.
# ---------------------------------------------------------------------------


@triton.jit
def _sparse_attn_decode_partial_bf16_kernel(
    q_ptr,
    cache_ptr,
    indices_ptr,
    indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    q_stride0,
    q_stride1,
    cache_stride0,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_rows,
    scale,
    num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    query_idx = tl.program_id(0)
    split_id = tl.program_id(1)
    pid_h = tl.program_id(2)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    dim_offsets = tl.arange(0, HEAD_DIM)

    q = tl.load(
        q_ptr
        + query_idx * q_stride0
        + head_offsets[:, None] * q_stride1
        + dim_offsets[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, HEAD_DIM), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)

    seg_start = tl.load(indptr_ptr + query_idx)
    seg_end = tl.load(indptr_ptr + query_idx + 1)
    seg_len = seg_end - seg_start
    seg_chunk = (seg_len + NUM_SPLITS - 1) // NUM_SPLITS
    lo = split_id * seg_chunk
    hi = tl.minimum(lo + seg_chunk, seg_len)

    for k_start in tl.range(lo, hi, BLOCK_K, num_stages=NUM_STAGES):
        k_pos = k_start + k_offsets
        in_range = k_pos < hi
        slot = tl.load(indices_ptr + seg_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_rows)
        safe_slot = tl.where(valid, slot, 0)

        kv = tl.load(
            cache_ptr
            + safe_slot.to(tl.int64)[:, None] * cache_stride0
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # Store raw (un-normalized) partial state for this split. Softmax sink and
    # final normalization happen in the reduce kernel.
    pm_base = query_idx * pm_stride0 + split_id * pm_stride_s + head_offsets
    tl.store(part_m_ptr + pm_base, m_i, mask=head_mask)
    tl.store(part_l_ptr + pm_base, l_i, mask=head_mask)
    acc_base = (
        part_acc_ptr
        + query_idx * pa_stride0
        + split_id * pa_stride_s
        + head_offsets[:, None] * pa_stride_h
    )
    tl.store(
        acc_base + dim_offsets[None, :],
        acc,
        mask=head_mask[:, None],
    )


@triton.jit
def _sparse_attn_decode_reduce_kernel(
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    attn_sink_ptr,
    out_ptr,
    out_stride0,
    out_stride1,
    pm_stride0,
    pm_stride_s,
    pa_stride0,
    pa_stride_s,
    pa_stride_h,
    num_heads,
    HAS_ATTN_SINK: tl.constexpr,
    ADAPTIVE_SPLITS: tl.constexpr,
    COMB_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLITS_PAD: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    head_mask = head_offsets < num_heads
    comb_offsets = tl.arange(0, COMB_DIM)
    # SPLITS_PAD is NUM_SPLITS rounded up to a power of two so the parallel
    # split-axis load is a legal arange for any split count; padding lanes are
    # masked off.
    split_offsets = tl.arange(0, SPLITS_PAD)
    split_mask = split_offsets < NUM_SPLITS

    neg_large = -3.4028234663852886e38

    # Phase 1: load every split's running max/sum at once and reduce the max
    # in parallel (tl.max over the split axis) instead of walking the splits
    # serially. This breaks the long online-softmax dependency chain that made
    # the reduce latency-bound.
    load_mask = split_mask[:, None] & head_mask[None, :]
    pm_split = (
        part_m_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :]
    )
    m_all = tl.load(pm_split, mask=load_mask, other=neg_large)  # [S, H]
    l_all = tl.load(
        part_l_ptr
        + query_idx * pm_stride0
        + split_offsets[:, None] * pm_stride_s
        + head_offsets[None, :],
        mask=load_mask,
        other=0.0,
    )

    m_comb = tl.max(m_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=neg_large
        ).to(tl.float32)
        m_final = tl.maximum(m_comb, sink)
    else:
        m_final = m_comb

    w_all = tl.exp(m_all - m_final[None, :])  # [S, H]
    w_all = tl.where(load_mask, w_all, 0.0)
    l_final = tl.sum(w_all * l_all, axis=0)  # [H]
    if HAS_ATTN_SINK:
        l_final = l_final + tl.exp(sink - m_final)
    denom = tl.maximum(l_final, 1.0e-30)

    # Phase 2: weighted sum of the per-split accumulators. The combine weight
    # for each split only depends on the (already known) global max, so the
    # acc loads carry no cross-split dependency and the compiler can pipeline
    # them; only the cheap FMA into `acc` is loop-carried.
    acc = tl.zeros((BLOCK_H, COMB_DIM), dtype=tl.float32)
    for s in tl.static_range(NUM_SPLITS):
        m_s = tl.load(
            part_m_ptr + query_idx * pm_stride0 + s * pm_stride_s + head_offsets,
            mask=head_mask,
            other=neg_large,
        )
        w_s = tl.exp(m_s - m_final)
        if ADAPTIVE_SPLITS:
            active_split = m_s > neg_large
            w_s = tl.where(head_mask & active_split, w_s, 0.0)
        acc_base = (
            part_acc_ptr
            + query_idx * pa_stride0
            + s * pa_stride_s
            + head_offsets[:, None] * pa_stride_h
        )
        if ADAPTIVE_SPLITS:
            acc_s = tl.load(
                acc_base + comb_offsets[None, :],
                mask=head_mask[:, None] & active_split[:, None],
                other=0.0,
            )
        else:
            acc_s = tl.load(
                acc_base + comb_offsets[None, :],
                mask=head_mask[:, None],
                other=0.0,
            )
        acc += w_s[:, None] * acc_s

    out = tl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)

    out_row_ptr = (
        out_ptr + query_idx * out_stride0 + head_offsets[:, None] * out_stride1
    )
    tl.store(
        out_row_ptr + comb_offsets[None, :],
        out,
        mask=head_mask[:, None],
    )


# ---------------------------------------------------------------------------
# split-count heuristic (verbatim from the source, gfx942 branch)
# ---------------------------------------------------------------------------


@functools.lru_cache
def _decode_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return 256  # For gfx950 arch, gated behind a fallback path for other archs.


def _decode_partial_iters(avg_len: float, splits: int, block_k: int) -> int:
    """BLOCK_K iterations one partial workgroup walks for ``splits`` splits.

    (Source ``_decode_partial_iters`` with the SWA/extra pair collapsed to
    floe's single segment.)
    """
    return (
        math.ceil(math.ceil(avg_len / splits) / block_k) if avg_len > 0 else 0
    )


def _decode_num_splits(
    num_queries: int,
    heads_blocks: int,
    avg_len: float = 0.0,
    block_k: int = 32,
) -> int:
    """Pick a flash-decode split count to keep the GPU busy across batch sizes.

    We model the relative partial-kernel latency for a given split count ``s``
    as ``waves * (1/s + mu)`` where ``waves = ceil(base * s / CU)`` and ``mu``
    is a small per-wave overhead penalty, then snap down to the smallest split
    count with the same wave count and per-workgroup iteration count.
    (Verbatim model from the source; single-segment form.)
    """
    base = max(1, num_queries * heads_blocks)
    # Target ~1 workgroup per CU: enough to fill the device while keeping the
    # reduce cost (which grows with split count) small. Tuned on gfx950.
    cu = max(1, _decode_cu_count())
    # Per-wave overhead penalty: higher values discourage split counts that
    # spill into extra GPU waves. Tuned on gfx950.
    mu = 0.04
    best_splits = 1
    best_cost = None
    # Search up to 16 splits; beyond that the reduce/HBM overhead dominates.
    for splits in range(1, 17):
        waves = (base * splits + cu - 1) // cu
        cost = waves * (1.0 / splits + mu)
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits = splits
            best_cost = cost

    if best_splits > 1 and avg_len > 0:
        target_waves = (base * best_splits + cu - 1) // cu
        target_iters = _decode_partial_iters(avg_len, best_splits, block_k)
        for splits in range(1, best_splits):
            waves = (base * splits + cu - 1) // cu
            iters = _decode_partial_iters(avg_len, splits, block_k)
            if waves == target_waves and iters == target_iters:
                best_splits = splits
                break
    return best_splits


# ---------------------------------------------------------------------------
# decode wrapper (adapted from ``_rocm_sparse_attn_decode_ragged_triton``)
# ---------------------------------------------------------------------------


def sparse_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_k: int = 32,
) -> torch.Tensor:
    """Split-K sparse decode over selected kv rows.

    ``q`` [t, h, d] bf16 (t = batch * s_q, s_q == 1 for decode), ``kv_cache``
    [rows, d] bf16 flat rows (batch pre-folded into the row space),
    ``indices`` [t, w] int32 dense with ``-1`` masked (or a prebuilt
    ``(ragged_indices, ragged_indptr)`` tuple). ``scale`` natural-exp units.
    Returns bf16 [t, h, d].
    """
    assert q.ndim == 3, f"expected q=[t,h,d], got {q.shape}"
    assert kv_cache.ndim == 2, f"expected kv_cache=[rows,d], got {kv_cache.shape}"
    assert not q.is_cpu and not kv_cache.is_cpu
    _validate_floe_sparse_dims(q.shape[-1], "sparse_attn_decode")
    # the partial kernel indexes dims with an unmasked arange(HEAD_DIM)
    head_dim_p2 = triton.next_power_of_2(q.shape[-1])
    assert head_dim_p2 == q.shape[-1], (
        f"sparse_attn_decode expects a power-of-2 head_dim, got {q.shape[-1]}"
    )

    if isinstance(indices, tuple):
        ragged_indices, ragged_indptr = indices
    else:
        ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
            indices,
            (indices >= 0).sum(dim=-1, dtype=torch.int32),
            num_rows=kv_cache.shape[0],
        )
    ragged_indices = _as_int32_contiguous_1d(ragged_indices)
    ragged_indptr = _as_int32_contiguous_1d(ragged_indptr)

    has_attn_sink = attn_sink is not None
    if attn_sink is None:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    else:
        attn_sink = attn_sink.contiguous()

    num_queries, num_heads, head_dim = q.shape
    assert ragged_indptr.numel() == num_queries + 1, (
        f"expected indptr shape [{num_queries + 1}], got {ragged_indptr.shape}"
    )

    block_h = 16
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    else:
        assert out.shape == q.shape and out.dtype == torch.bfloat16
    heads_blocks = triton.cdiv(num_heads, block_h)

    # Average per-query segment length, read sync-free from the ragged index
    # size, lets the split heuristic avoid over-splitting (as in the source).
    inv_q = 1.0 / max(1, num_queries)
    avg_len = ragged_indices.numel() * inv_q
    num_splits = _decode_num_splits(num_queries, heads_blocks, avg_len, block_k)

    part_m = torch.empty(
        (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
    )
    part_l = torch.empty_like(part_m)
    part_acc = torch.empty(
        (num_queries, num_splits, num_heads, head_dim),
        dtype=torch.float32,
        device=q.device,
    )

    _sparse_attn_decode_partial_bf16_kernel[(num_queries, num_splits, heads_blocks)](
        q,
        kv_cache,
        ragged_indices,
        ragged_indptr,
        part_m,
        part_l,
        part_acc,
        q.stride(0),
        q.stride(1),
        kv_cache.stride(0),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        kv_cache.shape[0],
        float(scale),
        num_heads,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        NUM_STAGES=1,
        num_warps=4,
    )

    _sparse_attn_decode_reduce_kernel[(num_queries, num_heads)](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        part_m.stride(0),
        part_m.stride(1),
        part_acc.stride(0),
        part_acc.stride(1),
        part_acc.stride(2),
        num_heads,
        HAS_ATTN_SINK=has_attn_sink,
        ADAPTIVE_SPLITS=False,  # gfx950-only in the source; floe is gfx942
        COMB_DIM=head_dim,
        BLOCK_H=1,
        NUM_SPLITS=num_splits,
        SPLITS_PAD=triton.next_power_of_2(num_splits),
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# floe-facing drop-in (vkernels ``dsa_sparse_fwd`` ABI)
# ---------------------------------------------------------------------------


def _fold_batch_indices(
    indices: torch.Tensor, batch_size: int, cache_rows_per_batch: int
) -> torch.Tensor:
    """Offset per-batch selected rows into the flat [B*S, K] row space.

    ``indices`` [..., B, ..., W] int32 with ``-1`` masked -> same shape with
    valid slots shifted by ``b * cache_rows_per_batch``. Pure index
    arithmetic; ``-1`` stays ``-1``.
    """
    if batch_size == 1:
        return indices
    shape = indices.shape
    # indices arrive [B, ..., W]; the batch axis is axis 0
    idx = indices.reshape(batch_size, -1, shape[-1])
    offsets = (
        torch.arange(batch_size, device=indices.device, dtype=torch.int32)
        * cache_rows_per_batch
    ).view(batch_size, 1, 1)
    idx = torch.where(idx >= 0, idx + offsets, idx)
    return idx.reshape(shape)


def dsa_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    dim: int,
    tail_dim: int,
    topk: int,
    sm_scale: float,
) -> torch.Tensor:
    """Floe/vkernels ``dsa_sparse_fwd`` ABI over the vendored Triton kernels.

    ``q`` [B, S_q, H, K] bf16 absorbed latent queries; ``kv`` [B, S, 1, K]
    bf16 latent cache; ``indices`` [B, S_q, 1, W] int32 with ``-1`` masked;
    ``dim = tail_dim + head`` with ``tail_dim == 0`` (NoPE absorbed form);
    ``sm_scale`` in the vkernels ABI's LOG2 units (floe passes
    ``scaling * log2(e)``) — converted to the kernels' natural-exp units
    here. Returns bf16 [B, S_q, H, K] (floe ``out_latent``).

    Dispatch mirrors floe's policy: S_q == 1 (decode regime) takes the
    split-K partial+reduce path; larger S_q takes the prefill path (the
    source's split heuristic targets the low-concurrency decode regime).
    """
    assert tail_dim == 0, f"absorbed NoPE form expects tail_dim == 0, got {tail_dim}"
    assert q.dim() == 4 and kv.dim() == 4, (q.shape, kv.shape)
    bsz, s_q, num_heads, k_dim = q.shape
    assert k_dim == dim == kv.shape[-1], (k_dim, dim, kv.shape[-1])
    assert kv.shape[0] == bsz and kv.shape[2] == 1
    assert indices.dtype == torch.int32, indices.dtype
    idx = indices.reshape(bsz, s_q, topk)
    assert idx.shape[-1] == topk

    qf = q.reshape(bsz * s_q, num_heads, k_dim).contiguous()
    cache = kv.reshape(bsz * kv.shape[1], k_dim).contiguous()
    idx = _fold_batch_indices(idx, bsz, kv.shape[1]).reshape(bsz * s_q, topk)

    scale = float(sm_scale) * math.log(2.0)  # log2 units -> natural units
    if s_q == 1:
        out = sparse_attn_decode(qf, cache, idx, scale)
    else:
        out = sparse_attn_prefill(qf, cache, idx, scale)
    return out.view(bsz, s_q, num_heads, k_dim)
