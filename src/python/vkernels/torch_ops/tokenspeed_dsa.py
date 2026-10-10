"""TokenSpeed Triton DSA (sparse MLA) attention kernels, vendored portable path.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel`` subpackage, ``main``
branch, fetched 2026-11 from
``tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/dsa/triton.py``
(MIT, Copyright (c) 2026 LightSeek Foundation).

Ported (verbatim kernel bodies and launchers):
  ``_dsa_packed_kv_kernel`` / ``_run_packed_kv`` — sparse attention over a
      byte-packed FP8 KV cache (kv bytes + fp32 block scales + bf16 rope in
      one row buffer, addressed by global slot ids from the indexer)
  ``_dsa_dense_kv_kernel`` / ``_run_dense_kv`` — the same attention over a
      plain dense bf16 KV cache
  ``_flatten_*`` helpers and the shared ``_run_dsa`` dispatch
  ``triton_dsa_decode`` / ``triton_dsa_prefill`` entrypoints (registration
      decorator dropped; see :mod:`vkernels.torch_ops.dsa_registry`)

Not ported (covered by this repo's own machinery):
  the ``_triton/topk.py`` fp8 scoring/indexer kernels — this tree's
  ``dsa_topk`` / ``dsa_kpool`` C++ kernels and the vLLM sparse-MLA
  integration own that stage
  ``triton_dsa_{decode,prefill}_topk_fp8`` / ``index_candidates`` — depend
  on tokenspeed's ``ops.quantization.quantize_fp8``
Adapted: ``tokenspeed_kernel._triton`` imports became stock ``triton`` /
``triton.language`` (upstream Triton, not their fork).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "triton_dsa_decode",
    "triton_dsa_prefill",
    "flatten_packed_kv_cache",
    "flatten_dense_kv_cache",
]


@triton.jit
def _dsa_packed_kv_kernel(
    q,
    kv_fp8,
    kv_scale,
    kv_rope,
    topk_indices,
    topk_lens,
    out,
    lse,
    RETURN_LSE: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    row_bytes: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)

    topk_offsets = tl.arange(0, BLOCK_TOPK)
    k_offsets = tl.arange(0, BLOCK_K)
    rope_offsets = tl.arange(0, 64)
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_base = (token * num_heads + head) * head_dim
    q_nope_base = q_base
    q_rope_base = q_base + kv_lora_rank

    q_rope = tl.load(
        q + q_rope_base + rope_offsets,
        mask=rope_offsets < qk_rope_head_dim,
        other=0.0,
    ).to(tl.float32)

    valid_len = tl.load(topk_lens + token).to(tl.int32)
    max_score = tl.full((), -float("inf"), tl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv_fp8 + slots[:, None] * row_bytes + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                kv_scale
                + (slots * row_bytes + kv_lora_rank + (k_start // 128) * 4) // 4,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv_rope
            + (slots[:, None] * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4)
            // 2
            + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        max_score = tl.maximum(max_score, tl.max(score, axis=0))

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv_fp8 + slots[:, None] * row_bytes + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            k_scale = tl.load(
                kv_scale
                + (slots * row_bytes + kv_lora_rank + (k_start // 128) * 4) // 4,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * k_scale[:, None] * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv_rope
            + (slots[:, None] * row_bytes + kv_lora_rank + (kv_lora_rank // 128) * 4)
            // 2
            + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        probs = tl.exp(score - max_score)
        probs = tl.where(valid, probs, 0.0)
        denom += tl.sum(probs, axis=0)

        v_vals = tl.load(
            kv_fp8 + slots[:, None] * row_bytes + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            kv_scale
            + (
                slots[:, None] * row_bytes
                + kv_lora_rank
                + (v_offsets[None, :] // 128) * 4
            )
            // 4,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v_vals * v_scale, axis=0)

    result = acc / denom
    result = tl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)
    if RETURN_LSE:
        if v_block == 0:
            tl.store(
                lse + token * num_heads + head,
                tl.where(denom > 0.0, max_score + tl.log(denom), -float("inf")),
            )


@triton.jit
def _dsa_dense_kv_kernel(
    q,
    kv,
    topk_indices,
    topk_lens,
    out,
    lse,
    RETURN_LSE: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    qk_rope_head_dim: tl.constexpr,
    kv_dim: tl.constexpr,
    topk: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    v_block = tl.program_id(2)

    topk_offsets = tl.arange(0, BLOCK_TOPK)
    k_offsets = tl.arange(0, BLOCK_K)
    rope_offsets = tl.arange(0, 64)
    v_offsets = v_block * BLOCK_V + tl.arange(0, BLOCK_V)

    q_base = (token * num_heads + head) * head_dim
    q_nope_base = q_base
    q_rope_base = q_base + kv_lora_rank

    q_rope = tl.load(
        q + q_rope_base + rope_offsets,
        mask=rope_offsets < qk_rope_head_dim,
        other=0.0,
    ).to(tl.float32)

    valid_len = tl.load(topk_lens + token).to(tl.int32)
    max_score = tl.full((), -float("inf"), tl.float32)

    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv + slots[:, None] * kv_dim + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv + slots[:, None] * kv_dim + kv_lora_rank + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        max_score = tl.maximum(max_score, tl.max(score, axis=0))

    denom = tl.full((), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_V,), tl.float32)
    v_mask = v_offsets < kv_lora_rank
    for start in range(0, topk, BLOCK_TOPK):
        cols = start + topk_offsets
        valid = cols < valid_len
        slots = tl.load(
            topk_indices + token * topk + cols,
            mask=valid,
            other=0,
        ).to(tl.int64)
        valid = valid & (slots >= 0)
        score = tl.zeros((BLOCK_TOPK,), tl.float32)

        for k_start in range(0, kv_lora_rank, BLOCK_K):
            ks = k_start + k_offsets
            q_vals = tl.load(q + q_nope_base + ks).to(tl.float32)
            k_vals = tl.load(
                kv + slots[:, None] * kv_dim + ks[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            score += tl.sum(k_vals * q_vals[None, :], axis=1)

        k_rope = tl.load(
            kv + slots[:, None] * kv_dim + kv_lora_rank + rope_offsets[None, :],
            mask=valid[:, None] & (rope_offsets[None, :] < qk_rope_head_dim),
            other=0.0,
        ).to(tl.float32)
        score += tl.sum(k_rope * q_rope[None, :], axis=1)
        score *= softmax_scale
        score = tl.where(valid, score, -float("inf"))
        probs = tl.exp(score - max_score)
        probs = tl.where(valid, probs, 0.0)
        denom += tl.sum(probs, axis=0)

        v_vals = tl.load(
            kv + slots[:, None] * kv_dim + v_offsets[None, :],
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(probs[:, None] * v_vals, axis=0)

    result = acc / denom
    result = tl.where(denom > 0.0, result, 0.0)
    out_base = (token * num_heads + head) * kv_lora_rank
    tl.store(out + out_base + v_offsets, result, mask=v_mask)
    if RETURN_LSE:
        if v_block == 0:
            tl.store(
                lse + token * num_heads + head,
                tl.where(denom > 0.0, max_score + tl.log(denom), -float("inf")),
            )


def _run_packed_kv(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    row_bytes = int(packed_kv.shape[1])
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_packed_kv_kernel[grid](
        q,
        packed_kv.view(torch.float8_e4m3fn),
        packed_kv.view(torch.float32),
        packed_kv.view(torch.bfloat16),
        topk_indices,
        topk_lens,
        out,
        lse,
        return_lse,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        row_bytes,
        topk_indices.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_K=64,
        BLOCK_V=64,
        num_warps=4,
        num_stages=1,
    )
    return out, lse


def _run_dense_kv(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_lens: torch.Tensor,
    *,
    softmax_scale: float,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    kv_dim = int(kv_lora_rank) + int(qk_rope_head_dim)
    out = torch.empty(
        (q.shape[0], q.shape[1], kv_lora_rank),
        dtype=torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype,
        device=q.device,
    )
    lse = (
        torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    grid = (q.shape[0], q.shape[1], triton.cdiv(kv_lora_rank, 64))
    _dsa_dense_kv_kernel[grid](
        q,
        kv_cache,
        topk_indices,
        topk_lens,
        out,
        lse,
        return_lse,
        q.shape[1],
        q.shape[2],
        kv_lora_rank,
        qk_rope_head_dim,
        kv_dim,
        topk_indices.shape[1],
        float(softmax_scale),
        BLOCK_TOPK=32,
        BLOCK_K=64,
        BLOCK_V=64,
        num_warps=4,
        num_stages=1,
    )
    return out, lse


def flatten_packed_kv_cache(packed_kv_cache: torch.Tensor) -> torch.Tensor:
    if packed_kv_cache.dim() == 2:
        return packed_kv_cache
    return packed_kv_cache.reshape(-1, packed_kv_cache.shape[-1])


def flatten_dense_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    if kv_cache.dim() == 2:
        return kv_cache
    if kv_cache.dim() == 3:
        return kv_cache.squeeze(1)
    if kv_cache.shape[1] == 1:
        kv_cache = kv_cache.permute(0, 2, 1, 3)
    return kv_cache.reshape(-1, kv_cache.shape[-1])


def _flatten_query(q: torch.Tensor) -> torch.Tensor:
    if q.dim() == 3:
        return q
    return q.reshape(-1, q.shape[-2], q.shape[-1])


def _run_dsa(
    *,
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    packed_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    k_scale: float,
    out: torch.Tensor | None,
    return_lse: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    q = _flatten_query(q).contiguous()
    topk_slots = topk_slots.contiguous()
    topk_lens = (
        torch.full(
            (q.shape[0],), topk_slots.shape[-1], dtype=torch.int32, device=q.device
        )
        if topk_lens is None
        else topk_lens.contiguous()
    )
    softmax_scale = float(softmax_scale) * float(k_scale)

    if packed_kv_cache is not None:
        result, lse = _run_packed_kv(
            q,
            flatten_packed_kv_cache(packed_kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            return_lse=return_lse,
        )
    else:
        result, lse = _run_dense_kv(
            q,
            flatten_dense_kv_cache(kv_cache).contiguous(),
            topk_slots,
            topk_lens,
            softmax_scale=softmax_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            return_lse=return_lse,
        )

    if out is not None:
        out.reshape_as(result).copy_(result)
        result = out
    return (result, lse) if return_lse else result


def triton_dsa_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    kv_seq_lens: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse-MLA decode over indexer-selected top-k slots (portable Triton).

    ``q`` is ``[tokens, H, kv_lora_rank + qk_rope_head_dim]``;
    ``kv_cache`` is a flattened dense ``[slots, kv_dim]`` latent cache or
    ``sparse_kv_cache`` a byte-packed FP8 row buffer (kv + scales + rope),
    with ``topk_slots`` holding global slot ids (``-1`` masked) and
    ``topk_lens`` per-token valid counts. Returns
    ``[tokens, H, kv_lora_rank]`` (plus fp32 LSE on request).
    """
    del kv_seq_lens
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
        return_lse=return_lse,
    )


def triton_dsa_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor | None,
    sparse_kv_cache: torch.Tensor | None,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor | None,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    page_size: int,
    q_len_per_req: int = 1,
    kv_seq_lens: torch.Tensor | None = None,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Sparse-MLA prefill; same contract as :func:`triton_dsa_decode`."""
    del kv_seq_lens
    return _run_dsa(
        q=q,
        kv_cache=kv_cache,
        packed_kv_cache=sparse_kv_cache,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        k_scale=k_scale,
        out=out,
        return_lse=return_lse,
    )
