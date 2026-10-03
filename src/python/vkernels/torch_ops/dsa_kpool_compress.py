"""vLLM kpool compression kernels (GLM-5.3-Flash DSA indexer) — Triton tier.

The donor caches the indexer's K in POOLS: one fp8 entry per ``index_kpool``
consecutive tokens (the softmax(gate+ape)-weighted average over the pool,
Hadamard-128-rotated and absmax-quantized), plus a small paged bf16 *tail*
ring holding each request's in-progress pool. Decode then costs O(1) per
token: stash the new K+gate into the ring, and when a pool completes,
compress ring+current-token and write one fp8 row — no full-stream pool
re-formation, no O(S) scan to keep the pooled cache current. This module
vendors vLLM's whole Triton implementation of that write path plus the
pool-level top-k transform.

Vendored from vLLM PR #53906 "[Model] add GLM-5.3-Flash support" (merged
2026-09-03, merge commit 98ed0856), file ``vllm/models/glm5next/nvidia/ops/
kpool_compress.py`` @ main (Apache-2.0; local copy ``/tmp/vllm-kpool/``):
the four Triton kernels + jit helpers are verbatim (modulo the one indent
level of the lazy-import factory — see DEVIATIONS); only the Python
wrappers were adapted. The donor's caller
(``vllm/models/glm5next/{sparse_indexer,nvidia/sparse_indexer}.py``) was
read for the contract mapping but NOT vendored.

This is the portable/any-GPU Triton tier of the kpool write path — the
CUDA twin of the HIP ``vkernels.kernels.dsa_kpool_{assemble,decode_update}``
(+ ``_fp8``) kernels and the fp8-cache counterpart of SGLang's
``kpool_fp8_index.py`` lineage (round-8 borrow doc §3.1).

CUT LIST
--------
Kept (verbatim unless noted):
  ``fwht128_quant_fp8`` + ``_fwht_quant_kernel`` (+ ``_fwht_stage``) —
    fused Hadamard-128 rotation + per-row absmax fp8 quant of the QUERY
    side (block-128, power-of-two scales).
  ``kpool_compress_and_write_cache`` + kernel (+ ``_hadamard128`` helpers) —
    the PREFILL compress-write: one program per pool.
  ``kpool_seed_tail_cache`` + kernel — persist each request's incomplete
    prefill pool into its paged tail ring (so decode can finish it).
  ``kpool_decode_update_and_maybe_write_cache_batched`` + kernel — the
    DECODE-step tail update + pool-completion write, batched over
    ``[num_requests, next_n]`` (spec-verify safe; plain decode is
    ``next_n == 1``).
  ``history_group_budget_for_topk`` / ``expand_pools_to_tokens`` /
    ``append_tail_to_topk`` — the donor's pure-torch pool-topk helpers
    (verbatim, any device; they double as the fused op's oracle).
  ``expand_pools_and_append_tail`` + kernel — the fused
    expand+append-tail transform (the identity path — no ``page_table`` /
    ``topk_offsets`` — which is the only path the GLM-5.3-Flash indexer
    uses; one launch replacing ~25 elementwise kernels).
Dropped (reason):
  the AMD twin (``vllm/models/glm5next/amd/ops/kpool_compress.py``) —
    platform-derived fp8 dtype/FP8_MAX and the ROCm "preshuffle" 16-token
    tile K layout for deep_gemm-ROCm consumption. This port pins
    ``float8_e4m3fn`` / 448.0 (the NVIDIA-canonical values, what GB10
    sm_121 / H100 see). Re-vendor the amd twin if a ROCm preshuffle cache
    layout is ever needed.
  ``expand_pools_to_tokens``'s ``page_table`` / ``topk_offsets`` variants
    in the fused kernel — not needed; the torch twins keep them.
  the caller machinery (workspace manager, deep_gemm
    ``fp8_fp4_{,paged_}mqa_logits``, ``top_k_per_row_prefill`` /
    ``get_indexer_topk``, decode scatter/packing helpers,
    ``eager_break_during_capture``, FP4 cache) — floe has its own indexer
    scan and top-k; only the cache-write kernels and the token-expand
    transform are the seam.

CACHE LAYOUT (what the kernels assume; both caches are caller-allocated)
------------------------------------------------------------------------
  indexer K cache ``kv_cache``: ``[num_blocks, block_size, head_dim+4]``
    uint8. Each page (block) row region is ``block_size * head_dim`` bytes
    of fp8 K (token-major) followed by ``block_size * 4`` bytes of fp32
    per-token scales; a pool's fp8 row lands at
    ``page*buf_numel_per_page + tok*head_dim`` and its scale at
    ``page*buf_numel_per_page + block_size*head_dim + 4*tok`` where
    ``loc = page*block_size + tok`` is the pool-granular flat slot.
    ``loc``, like every length/index here, is read from DEVICE memory.
  tail cache ``tail_kv_cache``: ``[num_blocks, 2, ring, head_dim]`` bf16
    strided view — half 0 raw K, half 1 gate score, ``ring >= pool_size``
    tokens per request block. It may alias a padded allocation via
    ``stride(0)``/``stride(1)`` (production does exactly that); the kernels
    address through those strides, never as a dense array. ``ring`` must be
    a multiple of ``pool_size``; the donor's rejected-draft test shows a
    one-pool ring corrupts redo pools — size it >= 2 pools under spec
    verify.

FLOE CONTRACT MAPPING (GLM-5.3-Flash, ``glm53flash``)
-----------------------------------------------------
  ``index_kpool = 4`` (config validates ``index_topk`` divisibility),
  indexer ``head_dim = 128`` == ``INDEX_HEAD_DIM`` (kernel constant);
  ``index_kpool_always_select_tail`` is the donor's append-tail step;
  ``index_kpool_compress_{ape,gate}`` feed ``ape`` / ``slot_score``.
  The donor's write path replaces floe's per-step pool re-formation
  (``Glm53Indexer`` pools the whole stream every forward; the pooled memo
  is O(S) per step) with: prefill ``kpool_compress_and_write_cache`` +
  ``kpool_seed_tail_cache`` once per chunk, decode
  ``kpool_decode_update_and_maybe_write_cache_batched`` once per step,
  top-k over POOL logits (floated to floe's ``select_k =
  index_topk // index_kpool`` pools), then ``expand_pools_and_append_tail``
  to the ``[B, S_q, index_topk + index_kpool - 1]`` (2051) token-id
  contract the attention paths already consume. ``round_scale=True``
  matches vLLM's ue8m0/power-of-two ``scale_fmt``; keep it True — the
  scales are read back as raw fp32 bit patterns downstream.
  The selection-fidelity risk (bf16 gate vs fp8 pooled scorer) and the
  A/B gates live in the round-8 borrow doc §3.1/§5; this module only
  supplies the kernels.

CAPTURE SAFETY
--------------
``kpool_decode_update_...`` is designed as the decode-graph op: the grid is
the host-static ``num_requests``; ``NEXT_N`` is a RUNTIME argument (no
``.item()``); positions, slot mappings, tail slots and every bound are read
from device memory at replay time; no output allocation (both caches are
caller-owned, written in place). Inputs must arrive contiguous so the
wrapper's ``.contiguous()`` no-ops — an actual copy inside capture is an
allocator pitfall, not a correctness one (torch's capture-aware pool
handles it).

**Replay caveat (GB10 laptop, torch 2.14+cu130, observed 2026-09):** under
``torch.cuda.CUDAGraph`` REPLAY this loop-bearing kernel intermittently
diverged (identical hard-synced pre-state, same cubin; ~1/12 replays wrote
a spurious pool) while EAGER launches were byte-exact vs the reference in
80/80 varied rounds, and the loop-free kernels (compress / expand / the
vendored metadata kernel) replayed deterministically 12/12. Not fixed by
``num_stages=1``. Could not be reproduced as a logic error — treat replay
validation as OPEN and check on the rig (H100) before ever capturing this
launch; note the donor caller itself runs the whole indexer
eager-break-during-capture (``sparse_attn_indexer_kpool`` is decorated
``@eager_break_during_capture``), so vLLM never replays it in a graph
either. Floe's first consumer (long-context eager decode) needs no graph.
``kpool_compress_and_write_cache`` / ``kpool_seed_tail_cache`` /
``fwht128_quant_fp8`` / ``expand_pools_and_append_tail`` allocate their
outputs (or dummies) per call — fine eager (prefill / the indexer's
eager-break path), warm them outside capture if ever captured.
``expand_pools_and_append_tail`` passes ``topk``/``out_cols`` as runtime
args (no recompile across ``seq_lens``); Triton still specializes on int
values by default, so warm each (topk bucket, BLOCK_COLS) shape once.

DEVIATIONS FROM THE SOURCE (all documented, none silent)
--------------------------------------------------------
1. Donor ``assert``-based contract checks became eligibility:
   ``*_eligible`` predicates + :class:`~vkernels.torch_ops._dispatch.OpNotEligible`
   raises (the vkernels torch_ops dispatch contract — floe catches and
   falls back to the ``*_reference`` oracle). The pure-torch helpers
   (``expand_pools_to_tokens`` / ``append_tail_to_topk`` /
   ``history_group_budget_for_topk``) keep their donor ``assert``s — they
   ARE the any-device fallback, so "not eligible" would be meaningless.
2. ``from vllm.triton_utils import tl, triton`` became one lazily-imported
   ``_kernels()`` factory (the sgl_moe/metadata pattern): importing this
   module pulls in neither triton nor a GPU. The jit bodies are verbatim
   modulo that indent level; the factory exists only so the module imports
   without triton installed.
3. Eligibility adds what the donor assumed from its runtime: CUDA
   residency + one device for all tensors, ``kv_cache`` exactly 3-D with
   last dim ``head_dim + 4`` (the page layout the byte offsets encode),
   int32 for the decode-update mapping/position tensors, and ape
   fp32/bf16 K dtypes as asserted. Nothing kernel-side changed.

Numerics: fp32 softmax/rotation accumulation with explicit bf16 rounding
before quantization (``.to(bf16).to(fp32)``), absmax-clamped (1e-4)
power-of-two (``round_scale=True``) or linear scales, e4m3 clamp at 448.
Parity with the eager oracle is exact at the fp8 byte level on every seed
tested (the quantization absorbs the last-ulp exp/log2 differences); the
tests pin ``torch.equal`` on the written cache bytes.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Optional

import torch

from ._dispatch import OpNotEligible

__all__ = [
    "INDEX_HEAD_DIM",
    "FP8_DTYPE",
    "FP8_MAX",
    "fwht128_quant_fp8",
    "fwht128_quant_fp8_eligible",
    "fwht128_quant_fp8_reference",
    "kpool_compress_and_write_cache",
    "kpool_compress_and_write_cache_eligible",
    "kpool_compress_and_write_cache_reference",
    "kpool_seed_tail_cache",
    "kpool_seed_tail_cache_eligible",
    "kpool_seed_tail_cache_reference",
    "kpool_decode_update_and_maybe_write_cache_batched",
    "kpool_decode_update_eligible",
    "kpool_decode_update_reference",
    "history_group_budget_for_topk",
    "expand_pools_to_tokens",
    "append_tail_to_topk",
    "expand_pools_and_append_tail",
    "expand_pools_and_append_tail_eligible",
]

# The GLM-5.3-Flash indexer head dimension is fixed at 128 (donor constant).
INDEX_HEAD_DIM = 128
# The NVIDIA-canonical fp8 flavour (donor values; the AMD twin derives these
# from the platform — see CUT LIST).
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# kernels (verbatim from the donor; one lazily-imported factory, sgl_moe
# pattern — triton is imported only on first launch)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def _hadamard128_stage(x, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
        x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
        x3 = tl.trans(x3, 0, 2, 1)
        a, b = tl.split(x3)
        x3 = tl.join(a + b, a - b)
        x3 = tl.trans(x3, 0, 2, 1)
        return tl.reshape(x3, (128,))

    @triton.jit
    def _hadamard128(x):
        x = _hadamard128_stage(x, 64, 1)
        x = _hadamard128_stage(x, 32, 2)
        x = _hadamard128_stage(x, 16, 4)
        x = _hadamard128_stage(x, 8, 8)
        x = _hadamard128_stage(x, 4, 16)
        x = _hadamard128_stage(x, 2, 32)
        x = _hadamard128_stage(x, 1, 64)
        return x * 0.08838834764831845  # 1/sqrt(128)

    @triton.jit
    def _fwht_stage(x, N: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
        # One FWHT butterfly stage on a flat tensor of N = GROUPS*2*STRIDE elems;
        # same construction as _hadamard128_stage but with a parametric N so it can
        # process BLOCK_R rows at once.
        x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
        x3 = tl.trans(x3, 0, 2, 1)
        a, b = tl.split(x3)
        x3 = tl.join(a + b, a - b)
        x3 = tl.trans(x3, 0, 2, 1)
        return tl.reshape(x3, (N,))

    @triton.jit
    def _fwht_quant_kernel(
        q_ptr,
        qout_ptr,
        sout_ptr,
        n_rows,
        BLOCK_R: tl.constexpr,
    ):
        """Fused Hadamard-128 rotation + per-row absmax FP8 (ue8m0) quant.

        Each row uses fp32 butterflies and scaling, rounds to bf16, then applies
        absmax quantization with a power-of-two scale.
        """
        pid = tl.program_id(0)
        rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
        rmask = rows < n_rows
        offs = tl.arange(0, 128)
        x = tl.load(
            q_ptr + rows[:, None] * 128 + offs[None, :], mask=rmask[:, None], other=0.0
        ).to(tl.float32)

        # Flatten so each row's 128 lanes stay contiguous: every stage's
        # (GROUPS, 2, STRIDE) tiling has 2*STRIDE dividing 128, so pairs never
        # straddle a row boundary. GROUPS of each stage scales by BLOCK_R.
        N: tl.constexpr = BLOCK_R * 128
        x = tl.reshape(x, (N,))
        x = _fwht_stage(x, N, BLOCK_R * 64, 1)
        x = _fwht_stage(x, N, BLOCK_R * 32, 2)
        x = _fwht_stage(x, N, BLOCK_R * 16, 4)
        x = _fwht_stage(x, N, BLOCK_R * 8, 8)
        x = _fwht_stage(x, N, BLOCK_R * 4, 16)
        x = _fwht_stage(x, N, BLOCK_R * 2, 32)
        x = _fwht_stage(x, N, BLOCK_R, 64)
        x = x * 0.08838834764831845  # 1/sqrt(128), exact in fp32

        # Match the unfused path's bf16 materialization before quantizing.
        x = x.to(tl.bfloat16).to(tl.float32)
        x = tl.reshape(x, (BLOCK_R, 128))

        absmax = tl.maximum(tl.max(tl.abs(x), axis=1), 1e-4)
        scale = tl.exp2(tl.ceil(tl.log2(absmax * (1.0 / 448.0))))
        y = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)

        tl.store(qout_ptr + rows[:, None] * 128 + offs[None, :], y, mask=rmask[:, None])
        tl.store(sout_ptr + rows, scale, mask=rmask)

    @triton.jit
    def _kpool_softmax_rotate_write_cache_kernel(
        buf_fp8_ptr,
        buf_fp32_ptr,
        slot_k_ptr,
        slot_score_ptr,
        ape_ptr,
        loc_ptr,
        write_mask_ptr,
        compressed_k_ptr,
        compressed_scale_ptr,
        slot_k_stride_0,
        slot_k_stride_1,
        slot_score_stride_0,
        slot_score_stride_1,
        ape_stride_0,
        PAGE_SIZE: tl.constexpr,
        BUF_NUMEL_PER_PAGE: tl.constexpr,
        POOL_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
        ROUND_SCALE: tl.constexpr,
        HAS_WRITE_MASK: tl.constexpr,
        RETURN_COMPRESSED: tl.constexpr,
        WRITE_CACHE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per pool. softmax(slot_score+ape)-weighted sum of slot_k ->
        Hadamard-128 -> per-vector fp8 absmax quant -> write to cache at ``loc``."""
        row = tl.program_id(0)
        do_write = True
        if HAS_WRITE_MASK:
            do_write = tl.load(write_mask_ptr + row)

        offs = tl.arange(0, BLOCK_D)
        mask = (offs < HEAD_DIM) & do_write

        # --- Pass 1: per-dim max over the pool (softmax numerical stability) ---
        max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            score = tl.load(
                slot_score_ptr
                + row * slot_score_stride_0
                + slot * slot_score_stride_1
                + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            max_score = tl.maximum(max_score, score)

        # --- Pass 2: softmax-weighted sum of K ---
        acc = tl.full((BLOCK_D,), 0.0, tl.float32)
        denom = tl.full((BLOCK_D,), 0.0, tl.float32)
        for slot in tl.static_range(0, POOL_SIZE):
            score = tl.load(
                slot_score_ptr
                + row * slot_score_stride_0
                + slot * slot_score_stride_1
                + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            score += tl.load(ape_ptr + slot * ape_stride_0 + offs, mask=mask, other=0.0).to(
                tl.float32
            )
            prob = tl.exp(score - max_score)
            denom += prob
            k = tl.load(
                slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            acc += k * prob

        x = acc / denom
        x = tl.where(do_write, x, 0.0).to(tl.bfloat16).to(tl.float32)

        # Match the unfused pooled-K path's bf16 precision before quantization.
        x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

        # --- per-vector absmax fp8 quant ---
        fp8_max = 448.0
        fp8_max_inv = 1.0 / fp8_max
        absmax = tl.max(tl.abs(x), axis=0)
        absmax = tl.maximum(absmax, 1e-4)
        if ROUND_SCALE:
            scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
        else:
            scale = absmax * fp8_max_inv
        quantized = x / scale
        quantized = tl.minimum(tl.maximum(quantized, -fp8_max), fp8_max)

        if WRITE_CACHE:
            loc = tl.load(loc_ptr + row, mask=do_write, other=0)
            loc_page_index = loc // PAGE_SIZE
            loc_token_offset_in_page = loc % PAGE_SIZE
            out_k_offsets = (
                loc_page_index * BUF_NUMEL_PER_PAGE
                + loc_token_offset_in_page * HEAD_DIM
                + offs
            )
            out_s_offset = (
                loc_page_index * BUF_NUMEL_PER_PAGE // 4
                + S_OFFSET_NBYTES_IN_PAGE // 4
                + loc_token_offset_in_page
            )
            tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=mask)
            tl.store(buf_fp32_ptr + out_s_offset, scale, mask=do_write)

        if RETURN_COMPRESSED:
            tl.store(
                compressed_k_ptr + row * HEAD_DIM + offs,
                quantized,
                mask=offs < HEAD_DIM,
            )
            tl.store(compressed_scale_ptr + row, scale)

    @triton.jit
    def _kpool_tail_seed_kernel(
        key_ptr,
        score_ptr,
        tslot_ptr,
        tail_ptr,
        n_tokens,
        TAIL_BLOCK_ELEMS: tl.constexpr,
        KPOOL_HEAD: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        KPOOL: tl.constexpr,
        RING: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Copy token ``i``'s raw K + gate into its request's tail ring.

        Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
        ahead belongs to a different tail block (or is past the batch / padding,
        slot < 0). ``tslot = block * RING + pos % RING``; the destination is
        ``tail[block, {0:K, 1:score}, pos % RING, :]``.

        The tail cache aliases the indexer cache with the indexer's (padded) block
        stride, so blocks are addressed through ``TAIL_BLOCK_ELEMS`` /
        ``KPOOL_HEAD`` (``tail.stride(0)`` / ``tail.stride(1)``), never as a dense
        ``[num_blocks, 2, KPOOL, HEAD_DIM]`` array.
        """
        i = tl.program_id(0)
        t = tl.load(tslot_ptr + i).to(tl.int64)
        if t < 0:
            return
        blk = t // RING  # t >= 0 here, so trunc == floor
        ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
            tl.int64
        )
        # Match the torch semantics exactly: a negative ahead slot floors to a
        # block id that differs from every real block -> token is in the tail.
        # Only divide non-negative slots (Triton int div truncates, torch floors).
        if ahead >= 0 and ahead // RING == blk:
            return
        offs = tl.arange(0, BLOCK_D)
        m = offs < HEAD_DIM
        base = blk * TAIL_BLOCK_ELEMS + (t % RING) * HEAD_DIM
        k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
        s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
        tl.store(tail_ptr + base + offs, k, mask=m)
        tl.store(tail_ptr + base + KPOOL_HEAD + offs, s, mask=m)

    @triton.jit
    def _kpool_decode_update_batched_kernel(
        buf_fp8_ptr,
        buf_fp32_ptr,
        tail_kv_ptr,
        tail_slot_mapping_ptr,  # [B, NEXT_N] int32
        key_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
        key_stride_b,
        key_stride_t,
        slot_score_ptr,  # [B, NEXT_N, HEAD_DIM] bf16
        ss_stride_b,
        ss_stride_t,
        ape_ptr,
        ape_stride_0,
        slot_mapping_ptr,  # [B, NEXT_N] int32
        positions_ptr,  # [B, NEXT_N] int32
        NEXT_N,  # runtime token count per request (no .item() needed)
        PAGE_SIZE: tl.constexpr,
        BUF_NUMEL_PER_PAGE: tl.constexpr,
        POOL_SIZE: tl.constexpr,
        RING: tl.constexpr,
        TAIL_BLOCK_ELEMS: tl.constexpr,
        KPOOL_HEAD: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,
        ROUND_SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program per request; iterates its NEXT_N verify tokens in order.

        Replaces the caller's per-token sequential launch loop. The intra-request
        iteration MUST stay in position order: a pool-completion at token t* reads
        the tail-ring slots that tokens t < t* (same request) just stashed in this
        same invocation. ``tl.range`` iterates sequentially within the program, so
        those stashes are visible to the later completion read. Cross-request
        programs are independent (distinct tail blocks). RING >= POOL_SIZE.
        """
        req = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        dim_mask = offs < HEAD_DIM

        for t in tl.range(0, NEXT_N):
            idx = req * NEXT_N + t
            cache_loc = tl.load(slot_mapping_ptr + idx)
            pos = tl.load(positions_ptr + idx)
            safe_pos = tl.maximum(pos, 0)
            pos_valid = (cache_loc >= 0) & (pos >= 0)

            slot = safe_pos % POOL_SIZE
            phys_slot = safe_pos % RING

            # Derive the tail block from THIS token's tail_slot (the request's block
            # is constant across a pool, but a padded / invalid entry carries a
            # negative sentinel -- reading it from token 0 would poison every
            # token's base address). Clamp so an invalid entry can never form an
            # out-of-bounds base; the accesses below are gated on pos_valid anyway.
            tail_slot = tl.load(tail_slot_mapping_ptr + idx)
            block = tl.maximum(tail_slot, 0).to(tl.int64) // RING
            block_base = block * TAIL_BLOCK_ELEMS

            # The tail-ring stash must run for EVERY real token, so it is gated on
            # the token-granular tail slot -- not on `pos_valid`, which keys off the
            # POOL-granular `slot_mapping` and is therefore only true on the pool's
            # last token. Gating the stash on pos_valid dropped every intra-pool
            # token, so a decode-built pool compressed 3 stale ring entries (the
            # prefill-seeded prompt tail, frozen forever) plus the current token.
            stash_valid = (pos >= 0) & (tail_slot >= 0)

            key = tl.load(
                key_ptr + req * key_stride_b + t * key_stride_t + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            score_current = tl.load(
                slot_score_ptr + req * ss_stride_b + t * ss_stride_t + offs,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)

            if pos_valid & (slot == POOL_SIZE - 1):
                pool_logical_start = safe_pos - slot

                max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
                for pool_slot in tl.static_range(0, POOL_SIZE):
                    is_current = pool_slot == slot
                    phys = (pool_logical_start + pool_slot) % RING
                    score_buf = tl.load(
                        tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                        mask=dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    score = tl.where(is_current, score_current, score_buf)
                    score += tl.load(
                        ape_ptr + pool_slot * ape_stride_0 + offs,
                        mask=dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    max_score = tl.maximum(max_score, score)

                acc = tl.full((BLOCK_D,), 0.0, tl.float32)
                denom = tl.full((BLOCK_D,), 0.0, tl.float32)
                for pool_slot in tl.static_range(0, POOL_SIZE):
                    is_current = pool_slot == slot
                    phys = (pool_logical_start + pool_slot) % RING
                    score_buf = tl.load(
                        tail_kv_ptr + block_base + KPOOL_HEAD + phys * HEAD_DIM + offs,
                        mask=dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    score = tl.where(is_current, score_current, score_buf)
                    score += tl.load(
                        ape_ptr + pool_slot * ape_stride_0 + offs,
                        mask=dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    prob = tl.exp(score - max_score)
                    denom += prob
                    k_buf = tl.load(
                        tail_kv_ptr + block_base + phys * HEAD_DIM + offs,
                        mask=dim_mask,
                        other=0.0,
                    ).to(tl.float32)
                    k = tl.where(is_current, key, k_buf)
                    acc += k * prob

                x = (acc / denom).to(tl.bfloat16).to(tl.float32)
                x = _hadamard128(x).to(tl.bfloat16).to(tl.float32)

                fp8_max = 448.0
                fp8_max_inv = 1.0 / fp8_max
                absmax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-4)
                if ROUND_SCALE:
                    scale = tl.exp2(tl.ceil(tl.log2(absmax * fp8_max_inv)))
                else:
                    scale = absmax * fp8_max_inv
                quantized = tl.minimum(tl.maximum(x / scale, -fp8_max), fp8_max)

                loc = cache_loc.to(tl.int64)
                loc_page_index = loc // PAGE_SIZE
                loc_token_offset_in_page = loc % PAGE_SIZE
                out_k_offsets = (
                    loc_page_index * BUF_NUMEL_PER_PAGE
                    + loc_token_offset_in_page * HEAD_DIM
                    + offs
                )
                out_s_offset = (
                    loc_page_index * BUF_NUMEL_PER_PAGE // 4
                    + S_OFFSET_NBYTES_IN_PAGE // 4
                    + loc_token_offset_in_page
                )
                tl.store(buf_fp8_ptr + out_k_offsets, quantized, mask=dim_mask)
                tl.store(buf_fp32_ptr + out_s_offset, scale)

            # Stash the current token AFTER any completion read so the completion
            # uses prior stashes (and the current token's own key/score via
            # is_current), then leaves this token for future pools. Order matches
            # the per-token kernel: completion read first, stash second.
            update_mask = dim_mask & stash_valid
            tl.store(
                tail_kv_ptr + block_base + phys_slot * HEAD_DIM + offs,
                key,
                mask=update_mask,
            )
            tl.store(
                tail_kv_ptr + block_base + KPOOL_HEAD + phys_slot * HEAD_DIM + offs,
                score_current,
                mask=update_mask,
            )

    @triton.jit
    def _expand_pools_and_append_tail_kernel(
        pool_ids_ptr,  # [rows, n_groups], int (any int dtype)
        seq_lens_ptr,  # [rows], int32 (token-granular seq_len)
        out_ptr,  # [rows, out_cols], int32
        topk,  # n_groups * pool_size
        out_cols,  # topk + pool_size - 1
        POOL_SIZE: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
        pid_s0,
        out_s0,
    ):
        # Fuses expand_pools_to_tokens + append_tail_to_topk (identity path) into a
        # single kernel. Each program writes one (row, column-tile) of the output.
        row = tl.program_id(0)
        tile = tl.program_id(1)
        cols = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = cols < out_cols

        seq_len = tl.load(seq_lens_ptr + row)
        pool_len = seq_len // POOL_SIZE
        tail_start = pool_len * POOL_SIZE
        tail_count = seq_len - tail_start  # in [0, POOL_SIZE)

        # History region [0, topk): expand selected pool g = cols // POOL_SIZE.
        is_history = cols < topk
        g = cols // POOL_SIZE
        o = cols % POOL_SIZE
        pid = tl.load(pool_ids_ptr + row * pid_s0 + g, mask=mask & is_history, other=-1)
        hist_val = (pid * POOL_SIZE + o).to(tl.int32)
        hist_out = tl.where(pid >= 0, hist_val, -1)

        # Tail region [topk, out_cols): the request's trailing incomplete pool.
        tail_off = cols - topk
        is_tail = (tail_off >= 0) & (tail_off < tail_count)
        tail_val = (tail_start + tail_off).to(tl.int32)
        tail_out = tl.where(is_tail, tail_val, -1)

        result = tl.where(is_history, hist_out, tail_out)
        tl.store(out_ptr + row * out_s0 + cols, result, mask=mask)

    return SimpleNamespace(
        fwht_quant=_fwht_quant_kernel,
        compress_write=_kpool_softmax_rotate_write_cache_kernel,
        tail_seed=_kpool_tail_seed_kernel,
        decode_update=_kpool_decode_update_batched_kernel,
        expand_tail=_expand_pools_and_append_tail_kernel,
    )


# ---------------------------------------------------------------------------
# shared reference helpers (pure torch, any device)
# ---------------------------------------------------------------------------


def _fwht128_rows(x: torch.Tensor) -> torch.Tensor:
    """Hadamard-128 on each row of ``[..., 128]`` fp32 — the donor kernel's
    exact butterfly sequence (same fp32 op order, same ``1/sqrt(128)`` scale),
    so the reference is bitwise against the jit butterflies. Each stage
    writes even-index-half = a+b, odd-index-half = a-b within every block
    (the donor's ``trans(0,2,1)`` after ``join`` — NOT an interleave)."""
    stages = ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64))
    lead = x.shape[:-1]
    for groups, stride in stages:
        x3 = x.reshape(*lead, groups, 2, stride)
        a, b = x3[..., 0, :], x3[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*lead, 128)
    return x * 0.08838834764831845  # 1/sqrt(128), exact in fp32


def _absmax_quant(x: torch.Tensor, round_scale: bool, fp8_max: float = FP8_MAX):
    """Per-row (last dim) absmax fp8 quant — the donor kernels' epilogue.

    Returns ``(quantized fp8, scale fp32)``; ``x`` is rounded to bf16 first
    (the kernels materialize bf16 before quantizing). The scale uses the
    kernel's exact arithmetic: multiply by the precomputed ``1/448``
    constant, never divide (a 1-ulp scale difference flips fp8 codes)."""
    x = x.to(torch.bfloat16).to(torch.float32)
    absmax = torch.clamp(x.abs().amax(dim=-1), min=1e-4)
    fp8_max_inv = 1.0 / fp8_max
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax * fp8_max_inv)))
    else:
        scale = absmax * fp8_max_inv
    q = torch.clamp(x / scale.unsqueeze(-1), -fp8_max, fp8_max).to(FP8_DTYPE)
    return q, scale


def _page_offsets(loc: int, page_size: int, head_dim: int):
    """(k_byte_offset, scale_byte_offset) for flat cache slot ``loc``.

    The page layout the kernels encode: ``page * (block_size*head_dim)`` bytes
    of token-major fp8 K, then ``block_size*4`` bytes of fp32 scales."""
    page, tok = loc // page_size, loc % page_size
    return page * page_size * (head_dim + 4) + tok * head_dim, (
        page * page_size * (head_dim + 4) + page_size * head_dim + 4 * tok
    )


def _softmax_pool(slot_k, slot_score, ape, write_mask=None):
    """The shared pool math: softmax(score+ape)-weighted sum of K, fp32,
    then the kernel's bf16 materialization of the pooled sum (the kernel
    rounds to bf16 BEFORE the Hadamard — skipping it shifts fp8 codes)."""
    scores = slot_score.float() + ape.float().unsqueeze(0)  # [n, pool, d]
    max_score = scores.max(dim=1).values
    prob = torch.exp(scores - max_score.unsqueeze(1))
    denom = prob.sum(dim=1)
    acc = (slot_k.float() * prob).sum(dim=1)
    x = acc / denom
    if write_mask is not None:
        x = torch.where(write_mask.unsqueeze(-1), x, torch.zeros_like(x))
    return x.to(torch.bfloat16).to(torch.float32)


# ---------------------------------------------------------------------------
# fwht128_quant_fp8 (query-side rotate + quant)
# ---------------------------------------------------------------------------


def fwht128_quant_fp8_eligible(q: torch.Tensor) -> bool:
    """Contract check for :func:`fwht128_quant_fp8` (no device sync)."""
    return not _fwht_violations(q)


def fwht128_quant_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each 128-wide row by the Hadamard-128 transform, then FP8-quant.

    Donor contract verbatim: ``q`` is ``[rows, 128]`` bf16 contiguous CUDA;
    returns fresh ``(q_fp8 [rows, 128] float8_e4m3fn, scale [rows, 1] fp32)``
    with power-of-two (ue8m0-style) scales. Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` on a contract miss —
    CPU callers take :func:`fwht128_quant_fp8_reference`. Allocates both
    outputs per call (eager path; warm outside capture).
    """
    violations = _fwht_violations(q)
    if violations:
        raise OpNotEligible("fwht128_quant_fp8 contract: " + "; ".join(violations))

    import triton

    n_rows = q.shape[0]
    q_fp8 = torch.empty((n_rows, 128), dtype=FP8_DTYPE, device=q.device)
    q_scale = torch.empty((n_rows, 1), dtype=torch.float32, device=q.device)
    if n_rows == 0:
        return q_fp8, q_scale
    BLOCK_R = 32
    grid = (triton.cdiv(n_rows, BLOCK_R),)
    _kernels().fwht_quant[grid](q, q_fp8, q_scale, n_rows, BLOCK_R=BLOCK_R, num_warps=2)
    return q_fp8, q_scale


def _fwht_violations(q: torch.Tensor) -> list:
    v: list = []
    if not q.is_cuda:
        v.append(f"q must be CUDA-resident, got {q.device}")
    if q.dim() != 2 or q.shape[1] != INDEX_HEAD_DIM:
        v.append(f"q must be 2-D [rows, {INDEX_HEAD_DIM}], got {tuple(q.shape)}")
    if q.dtype != torch.bfloat16:
        v.append(f"q must be bfloat16, got {q.dtype}")
    if not q.is_contiguous():
        v.append("q must be contiguous")
    return v


def fwht128_quant_fp8_reference(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager torch oracle for :func:`fwht128_quant_fp8` (any device, any dtype
    that round-trips through fp32). Same butterfly order, same bf16
    materialization, same power-of-two scales."""
    x = _fwht128_rows(q.float())
    return _absmax_quant(x, round_scale=True)


# ---------------------------------------------------------------------------
# kpool_compress_and_write_cache (prefill compress-write)
# ---------------------------------------------------------------------------


def _compress_violations(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int,
    write_mask: Optional[torch.Tensor],
    write_cache: bool,
    return_compressed: bool,
) -> list:
    named = [("kv_cache", kv_cache), ("slot_k", slot_k), ("slot_score", slot_score),
             ("ape", ape), ("loc", loc)]
    if write_mask is not None:
        named.append(("write_mask", write_mask))
    v: list = []
    if not all(t.is_cuda for _, t in named):
        v.append(
            "all tensors must be CUDA-resident (got "
            + ", ".join(f"{n}={t.device}" for n, t in named)
            + ")"
        )
        return v
    dev = kv_cache.device
    for name, t in named[1:]:
        if t.device != dev:
            v.append(f"{name} on {t.device}, expected {dev}")
    # the page layout the byte offsets encode (donor-implied, now checked)
    if kv_cache.dim() != 3 or kv_cache.shape[-1] != head_dim + 4:
        v.append(
            f"kv_cache must be 3-D [blocks, page, {head_dim}+4], got "
            f"{tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype != torch.uint8:
        v.append(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if slot_k.dim() != 3:
        v.append(f"slot_k must be 3-D [n_pools, pool, dim], got {tuple(slot_k.shape)}")
    if slot_score.shape != slot_k.shape:
        v.append(f"slot_score shape {tuple(slot_score.shape)} != slot_k {tuple(slot_k.shape)}")
    if slot_k.dim() == 3 and ape.shape != slot_k.shape[1:]:
        v.append(f"ape shape {tuple(ape.shape)} != [pool, dim] {tuple(slot_k.shape[1:])}")
    if slot_k.dim() == 3 and slot_k.shape[2] != head_dim:
        v.append(f"slot_k last dim {slot_k.shape[2]} != head_dim {head_dim}")
    if slot_k.dtype != torch.bfloat16:
        v.append(f"slot_k must be bfloat16, got {slot_k.dtype}")
    if ape.dtype != torch.float32:
        v.append(f"ape must be float32, got {ape.dtype}")
    if loc.dtype != torch.int64:
        v.append(f"loc must be int64, got {loc.dtype}")
    if write_mask is not None:
        if write_mask.shape != (slot_k.shape[0],):
            v.append(f"write_mask shape {tuple(write_mask.shape)} != ({slot_k.shape[0]},)")
        if return_compressed:
            v.append("write_mask and return_compressed are mutually exclusive")
    if not (write_cache or return_compressed):
        v.append("one of write_cache / return_compressed must be true")
    if pool_size < 1:
        v.append(f"pool_size must be >= 1, got {pool_size}")
    if head_dim < 1:
        v.append(f"head_dim must be >= 1, got {head_dim}")
    return v


def kpool_compress_and_write_cache_eligible(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    *,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: Optional[torch.Tensor] = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
) -> bool:
    """Contract check for :func:`kpool_compress_and_write_cache`."""
    return not _compress_violations(
        kv_cache, slot_k, slot_score, ape, loc, pool_size, head_dim,
        write_mask, write_cache, return_compressed,
    )


def kpool_compress_and_write_cache(
    kv_cache: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    write_mask: torch.Tensor | None = None,
    round_scale: bool = True,
    return_compressed: bool = False,
    write_cache: bool = True,
):
    """Compress ``pool_size`` tokens into one fp8 K and write at ``loc``.

    Donor contract (signature and semantics verbatim): ``kv_cache`` is the
    indexer K cache ``[num_blocks, block_size, head_dim+4]`` uint8;
    ``slot_k`` / ``slot_score`` are ``[n_pools, pool_size, head_dim]`` bf16;
    ``ape`` is ``[pool_size, head_dim]`` fp32; ``loc`` is ``[n_pools]`` int64
    flat pool slots; ``write_mask`` ``[n_pools]`` bool selects pools to write
    (``None`` = all); ``round_scale`` picks power-of-two vs linear fp8 scales;
    ``return_compressed`` additionally returns ``(fp8 K, fp32 scales)`` fresh
    tensors (mutually exclusive with ``write_mask``); ``write_cache=False``
    skips the cache write (compute-only, with ``return_compressed=True``).
    ``n_pools == 0`` returns empty tensors. Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` on a contract miss
    (CPU tensors, wrong dtypes/shapes) — callers gate on
    :func:`kpool_compress_and_write_cache_eligible` and fall back to
    :func:`kpool_compress_and_write_cache_reference`. Inputs are normalized
    with ``.contiguous()`` (no-ops on contiguous inputs — keep them
    contiguous under capture).
    """
    violations = _compress_violations(
        kv_cache, slot_k, slot_score, ape, loc, pool_size, head_dim,
        write_mask, write_cache, return_compressed,
    )
    if violations:
        raise OpNotEligible(
            "kpool_compress_and_write_cache contract: " + "; ".join(violations)
        )

    import triton

    page_size = kv_cache.shape[1]
    buf = kv_cache
    slot_k = slot_k.contiguous()
    slot_score = slot_score.contiguous()
    ape = ape.contiguous()
    loc = loc.contiguous()
    if write_mask is None:
        write_mask = torch.empty((1,), dtype=torch.bool, device=slot_k.device)
        has_write_mask = False
    else:
        write_mask = write_mask.contiguous()
        has_write_mask = True

    if slot_k.shape[0] == 0:
        if return_compressed:
            return (
                torch.empty((0, head_dim), dtype=FP8_DTYPE, device=slot_k.device),
                torch.empty((0,), dtype=torch.float32, device=slot_k.device),
            )
        return None

    buf_fp8 = buf.view(FP8_DTYPE)
    buf_fp32 = buf.view(torch.float32)
    # bytes per page (last dim of kv_cache) viewed as uint8
    buf_numel_per_page = buf.stride(0)
    s_offset_nbytes_in_page = page_size * head_dim

    if return_compressed:
        compressed_k = torch.empty(
            (slot_k.shape[0], head_dim), dtype=FP8_DTYPE, device=slot_k.device
        )
        compressed_scale = torch.empty(
            (slot_k.shape[0],), dtype=torch.float32, device=slot_k.device
        )
    else:
        compressed_k = buf_fp8
        compressed_scale = buf_fp32

    _kernels().compress_write[(slot_k.shape[0],)](
        buf_fp8,
        buf_fp32,
        slot_k,
        slot_score,
        ape,
        loc,
        write_mask,
        compressed_k,
        compressed_scale,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        POOL_SIZE=slot_k.shape[1],
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=s_offset_nbytes_in_page,
        ROUND_SCALE=round_scale,
        HAS_WRITE_MASK=has_write_mask,
        RETURN_COMPRESSED=return_compressed,
        WRITE_CACHE=write_cache,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )

    if return_compressed:
        return compressed_k, compressed_scale
    return None


def kpool_compress_and_write_cache_reference(
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    *,
    head_dim: int = INDEX_HEAD_DIM,
    kv_cache: Optional[torch.Tensor] = None,
    write_mask: Optional[torch.Tensor] = None,
    round_scale: bool = True,
    return_compressed: bool = True,
) -> dict:
    """Eager torch oracle for :func:`kpool_compress_and_write_cache`.

    Returns tensors in a dict: ``compressed_k`` (fp8, ``[n_pools,
    head_dim]``), ``compressed_scale`` (fp32, ``[n_pools]``), and
    ``kv_cache`` — a byte-level CLONE of the input cache with the same
    page-offset writes applied (absent when the input is ``None``). The
    float math always runs on CPU copies (deterministic reduction order,
    mirroring the donor test oracle); every output comes back on CPU —
    move it before comparing device-side kernels. Unwritten (masked-off)
    pools leave the clone untouched.
    """
    slot_k, slot_score = slot_k.cpu(), slot_score.cpu()
    ape, loc = ape.cpu(), loc.cpu()
    if write_mask is not None:
        write_mask = write_mask.cpu()
    kv_cpu = kv_cache.cpu() if kv_cache is not None else None

    n_pools = int(slot_k.shape[0])
    x = _softmax_pool(slot_k, slot_score, ape, write_mask)
    x = _fwht128_rows(x)
    compressed_k, compressed_scale = _absmax_quant(x, round_scale)

    out = {"compressed_k": compressed_k, "compressed_scale": compressed_scale}
    if kv_cpu is not None:
        clone = kv_cpu.clone()
        flat = clone.view(-1)
        page_size = int(kv_cpu.shape[1])
        wm = write_mask if write_mask is not None else torch.ones(
            n_pools, dtype=torch.bool, device=slot_k.device
        )
        for r in range(n_pools):
            if not bool(wm[r]):
                continue
            k_off, s_off = _page_offsets(int(loc[r]), page_size, head_dim)
            flat[k_off : k_off + head_dim] = compressed_k[r].view(torch.uint8)
            flat[s_off : s_off + 4] = compressed_scale[r].reshape(1).view(torch.uint8)
        out["kv_cache"] = clone
    return out


# ---------------------------------------------------------------------------
# kpool_seed_tail_cache (prefill tail seeding)
# ---------------------------------------------------------------------------


def _seed_violations(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int,
) -> list:
    named = [("tail_kv_cache", tail_kv_cache), ("key", key),
             ("gate_score", gate_score), ("tslot", tslot)]
    v: list = []
    if not all(t.is_cuda for _, t in named):
        v.append(
            "all tensors must be CUDA-resident (got "
            + ", ".join(f"{n}={t.device}" for n, t in named)
            + ")"
        )
        return v
    dev = tail_kv_cache.device
    for name, t in named[1:]:
        if t.device != dev:
            v.append(f"{name} on {t.device}, expected {dev}")
    if tail_kv_cache.dtype != torch.bfloat16:
        v.append(f"tail_kv_cache must be bfloat16, got {tail_kv_cache.dtype}")
    if tail_kv_cache.dim() != 4 or tail_kv_cache.shape[1] != 2:
        v.append(
            f"tail_kv_cache must be 4-D [blocks, 2, ring, dim], got "
            f"{tuple(tail_kv_cache.shape)}"
        )
    if tail_kv_cache.stride(3) != 1 or tail_kv_cache.stride(2) != head_dim:
        v.append(
            f"tail_kv_cache must be block-contiguous (stride(2)=={head_dim}, "
            f"stride(3)==1), got {tail_kv_cache.stride()}"
        )
    if key.dtype != torch.bfloat16:
        v.append(f"key must be bfloat16, got {key.dtype}")
    if key.dim() != 2 or key.shape[1] != head_dim:
        v.append(f"key must be [n, {head_dim}], got {tuple(key.shape)}")
    if gate_score.shape != key.shape:
        v.append(f"gate_score shape {tuple(gate_score.shape)} != key {tuple(key.shape)}")
    if tslot.shape != (key.shape[0],):
        v.append(f"tslot shape {tuple(tslot.shape)} != ({key.shape[0]},)")
    if kpool < 1:
        v.append(f"kpool must be >= 1, got {kpool}")
    if head_dim < 1:
        v.append(f"head_dim must be >= 1, got {head_dim}")
    return v


def kpool_seed_tail_cache_eligible(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    *,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> bool:
    """Contract check for :func:`kpool_seed_tail_cache`."""
    return not _seed_violations(tail_kv_cache, key, gate_score, tslot, kpool, head_dim)


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel).

    Donor contract verbatim: token ``i``'s raw K + gate land in its
    request's tail ring at ``tail[block, {0,1}, tslot % ring, :]`` where
    ``tslot = block * ring + pos % ring`` — only for the batch's LAST
    ``kpool`` tokens of each request (a token whose ``kpool``-ahead slot is
    in the same block is mid-pool and skipped). ``tail_kv_cache`` may be a
    strided alias onto a padded allocation (addressed via stride(0)/stride(1)).
    Raises :class:`~vkernels.torch_ops._dispatch.OpNotEligible` on a contract
    miss; ``n == 0`` is a no-op. In-place, no allocation.
    """
    violations = _seed_violations(tail_kv_cache, key, gate_score, tslot, kpool, head_dim)
    if violations:
        raise OpNotEligible("kpool_seed_tail_cache contract: " + "; ".join(violations))
    n = tslot.shape[0]
    if n == 0:
        return

    import triton

    _kernels().tail_seed[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        RING=tail_kv_cache.shape[2],
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def kpool_seed_tail_cache_reference(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
) -> torch.Tensor:
    """Eager torch oracle for :func:`kpool_seed_tail_cache`: returns a CLONE
    of ``tail_kv_cache`` (any device, any strides) with the seed writes
    applied. ``tslot[i] < 0`` (padding) skips token ``i``."""
    tail = tail_kv_cache.clone()
    ring = int(tail_kv_cache.shape[2])
    ts = tslot.to(torch.int64).tolist()
    for i, t in enumerate(ts):
        if t < 0:
            continue
        blk, pos = t // ring, t % ring
        ahead = ts[i + kpool] if i + kpool < len(ts) else -1
        if ahead >= 0 and ahead // ring == blk:
            continue  # mid-pool: a later seed overwrites this ring slot
        tail[blk, 0, pos] = key[i]
        tail[blk, 1, pos] = gate_score[i]
    return tail


# ---------------------------------------------------------------------------
# kpool_decode_update_and_maybe_write_cache_batched (decode tail update)
# ---------------------------------------------------------------------------


def _decode_update_violations(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int,
) -> list:
    num_requests, next_n = key.shape[0], key.shape[1]
    named = [("kv_cache", kv_cache), ("tail_kv_cache", tail_kv_cache),
             ("tail_slot_mapping", tail_slot_mapping), ("key", key),
             ("slot_score", slot_score), ("ape", ape),
             ("slot_mapping", slot_mapping), ("positions", positions)]
    v: list = []
    if not all(t.is_cuda for _, t in named):
        v.append(
            "all tensors must be CUDA-resident (got "
            + ", ".join(f"{n}={t.device}" for n, t in named)
            + ")"
        )
        return v
    dev = kv_cache.device
    for name, t in named[1:]:
        if t.device != dev:
            v.append(f"{name} on {t.device}, expected {dev}")
    if kv_cache.dim() != 3 or kv_cache.shape[-1] != head_dim + 4:
        v.append(
            f"kv_cache must be 3-D [blocks, page, {head_dim}+4], got "
            f"{tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype != torch.uint8:
        v.append(f"kv_cache must be uint8, got {kv_cache.dtype}")
    if tail_kv_cache.dim() != 4 or tail_kv_cache.shape[1] != 2:
        v.append(
            f"tail_kv_cache must be 4-D [blocks, 2, ring, dim], got "
            f"{tuple(tail_kv_cache.shape)}"
        )
    ring = int(tail_kv_cache.shape[2]) if tail_kv_cache.dim() == 4 else -1
    if tail_kv_cache.shape[-1] != head_dim:
        v.append(f"tail_kv_cache last dim {tail_kv_cache.shape[-1]} != head_dim {head_dim}")
    if not (ring >= pool_size and ring % pool_size == 0):
        v.append(f"ring ({ring}) must be >= pool_size ({pool_size}) and a multiple of it")
    if tail_kv_cache.dtype != torch.bfloat16:
        v.append(f"tail_kv_cache must be bfloat16, got {tail_kv_cache.dtype}")
    if key.dim() != 3 or key.shape[2] != head_dim:
        v.append(f"key must be [num_requests, next_n, {head_dim}], got {tuple(key.shape)}")
    if slot_score.shape != key.shape:
        v.append(f"slot_score shape {tuple(slot_score.shape)} != key {tuple(key.shape)}")
    if ape.shape != (pool_size, head_dim):
        v.append(f"ape shape {tuple(ape.shape)} != ({pool_size}, {head_dim})")
    if tail_slot_mapping.shape != (num_requests, next_n):
        v.append(
            f"tail_slot_mapping shape {tuple(tail_slot_mapping.shape)} != "
            f"({num_requests}, {next_n})"
        )
    if slot_mapping.shape != (num_requests, next_n):
        v.append(
            f"slot_mapping shape {tuple(slot_mapping.shape)} != "
            f"({num_requests}, {next_n})"
        )
    if positions.shape != (num_requests, next_n):
        v.append(
            f"positions shape {tuple(positions.shape)} != ({num_requests}, {next_n})"
        )
    if key.dtype != torch.bfloat16:
        v.append(f"key must be bfloat16, got {key.dtype}")
    if slot_score.dtype != torch.bfloat16:
        v.append(f"slot_score must be bfloat16, got {slot_score.dtype}")
    if ape.dtype != torch.float32:
        v.append(f"ape must be float32, got {ape.dtype}")
    for name, t in (("tail_slot_mapping", tail_slot_mapping),
                    ("slot_mapping", slot_mapping), ("positions", positions)):
        if t.dtype != torch.int32:
            v.append(f"{name} must be int32, got {t.dtype}")
    if pool_size < 1:
        v.append(f"pool_size must be >= 1, got {pool_size}")
    if head_dim < 1:
        v.append(f"head_dim must be >= 1, got {head_dim}")
    return v


def kpool_decode_update_eligible(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    *,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> bool:
    """Contract check for :func:`kpool_decode_update_and_maybe_write_cache_batched`."""
    return not _decode_update_violations(
        kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape,
        slot_mapping, positions, pool_size, head_dim,
    )


def kpool_decode_update_and_maybe_write_cache_batched(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int = INDEX_HEAD_DIM,
    round_scale: bool = True,
) -> None:
    """Batched decode-step kpool update for spec verify (``next_n > 1``).

    Donor contract (signature and semantics verbatim): one launch over
    ``[num_requests, next_n]`` grouped inputs replaces the per-token loop.
    Each program stashes its request's tokens into the tail ring in position
    order and, on pool completion (``pos % pool_size == pool_size - 1`` with
    a valid pool-granular ``slot_mapping`` entry), compresses
    ring+current-token into one fp8 K and writes it. ``tail_slot_mapping``
    is TOKEN-granular (every real token carries a slot; padding is ``-1``);
    ``slot_mapping`` is POOL-granular (only the last token of a completing
    pool carries the fp8 cache slot); ``positions`` stay token-granular.
    Plain decode is ``next_n == 1``. ``num_requests == 0`` or ``next_n == 0``
    returns without launching. Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` on a contract miss.
    In-place, allocation-free (the three int tensors are made contiguous —
    pass them contiguous to keep it a no-op under capture).
    """
    violations = _decode_update_violations(
        kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape,
        slot_mapping, positions, pool_size, head_dim,
    )
    if violations:
        raise OpNotEligible(
            "kpool_decode_update_and_maybe_write_cache_batched contract: "
            + "; ".join(violations)
        )
    num_requests, next_n = key.shape[0], key.shape[1]
    if num_requests == 0 or next_n == 0:
        return

    import triton

    ring = tail_kv_cache.shape[2]
    page_size = kv_cache.shape[1]
    buf = kv_cache
    buf_fp8 = buf.view(FP8_DTYPE)
    buf_fp32 = buf.view(torch.float32)

    # The kernel indexes the int tensors as ``req * next_n + t`` (row-major),
    # so they must be contiguous. Callers pass either a view of a contiguous
    # slice or a freshly scattered tensor, making these no-ops; the calls guard
    # against a future caller handing over a strided view.
    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()

    _kernels().decode_update[(num_requests,)](
        buf_fp8,
        buf_fp32,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        next_n,
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf.stride(0),
        POOL_SIZE=pool_size,
        RING=ring,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        S_OFFSET_NBYTES_IN_PAGE=page_size * head_dim,
        ROUND_SCALE=round_scale,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def kpool_decode_update_reference(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    *,
    pool_size: int,
    round_scale: bool = True,
) -> dict:
    """Eager torch oracle for
    :func:`kpool_decode_update_and_maybe_write_cache_batched`.

    Mirrors the kernel's per-request, in-position-order semantics (the
    read-after-stash dependency) and returns a dict ``{"kv_cache",
    "tail_kv_cache"}`` of CLONES. The float math always runs on CPU copies
    (deterministic reduction order, mirroring the donor test oracle); every
    output comes back on CPU — move it before comparing device-side kernels.
    """
    kv = kv_cache.cpu().clone()
    tail = tail_kv_cache.cpu().clone()
    tail_slot_mapping = tail_slot_mapping.cpu()
    key, slot_score = key.cpu(), slot_score.cpu()
    ape, slot_mapping, positions = ape.cpu(), slot_mapping.cpu(), positions.cpu()

    B, next_n = positions.shape
    head_dim = int(kv_cache.shape[-1]) - 4
    page_size = int(kv_cache.shape[1])
    ring = int(tail_kv_cache.shape[2])
    flat = kv.view(-1)

    ts = tail_slot_mapping.to(torch.int64).tolist()
    sm = slot_mapping.to(torch.int64).tolist()
    pos = positions.to(torch.int64).tolist()
    key_f = key.float()
    score_f = slot_score.float()
    ape_f = ape.float()

    for b in range(B):
        for t in range(next_n):
            cache_loc = sm[b][t]
            p = pos[b][t]
            pos_valid = cache_loc >= 0 and p >= 0
            safe_pos = max(p, 0)
            slot = safe_pos % pool_size
            phys_slot = safe_pos % ring
            # per-token block derivation (a leading invalid sentinel must not
            # poison the base); clamped like the kernel.
            block = max(ts[b][t], 0) // ring

            cur_key = key_f[b, t]
            cur_score = score_f[b, t]

            if pos_valid and slot == pool_size - 1:
                pool_logical_start = safe_pos - slot
                pool_scores, pool_ks = [], []
                for ps in range(pool_size):
                    is_current = ps == slot
                    phys = (pool_logical_start + ps) % ring
                    if is_current:
                        s, k = cur_score, cur_key
                    else:
                        s = tail[block, 1, phys].float()
                        k = tail[block, 0, phys].float()
                    pool_scores.append(s + ape_f[ps])
                    pool_ks.append(k)
                pool_scores = torch.stack(pool_scores)  # [pool, d]
                pool_ks = torch.stack(pool_ks)
                max_score = pool_scores.max(dim=0).values
                prob = torch.exp(pool_scores - max_score)
                denom = prob.sum(dim=0)
                x = (pool_ks * prob).sum(dim=0) / denom
                x = x.to(torch.bfloat16).to(torch.float32)  # kernel rounds pre-Hadamard
                x = _fwht128_rows(x)
                q, scale = _absmax_quant(x, round_scale)
                k_off, s_off = _page_offsets(cache_loc, page_size, head_dim)
                flat[k_off : k_off + head_dim] = q.view(torch.uint8)
                flat[s_off : s_off + 4] = scale.reshape(1).view(torch.uint8)

            # stash — gated on the TOKEN-granular tail slot, not pos_valid.
            if p >= 0 and ts[b][t] >= 0:
                tail[block, 0, phys_slot] = cur_key.to(tail.dtype)
                tail[block, 1, phys_slot] = cur_score.to(tail.dtype)

    return {"kv_cache": kv, "tail_kv_cache": tail}


# ---------------------------------------------------------------------------
# pool-level top-k helpers (donor torch helpers verbatim + the fused twin)
# ---------------------------------------------------------------------------


def history_group_budget_for_topk(topk: int, pool_size: int) -> int:
    """Number of pools to select so that expanding yields ``topk`` tokens."""
    assert topk % pool_size == 0
    return topk // pool_size


def expand_pools_to_tokens(
    group_ids: torch.Tensor,
    group_valid: torch.Tensor,
    topk: int,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand selected full-pool ids to a strict-width token topk tensor.

    (Donor verbatim — pure torch, any device, asserts kept; this IS the
    oracle for :func:`expand_pools_and_append_tail`.)
    """
    assert group_ids.ndim == 2
    assert group_valid.shape == group_ids.shape
    assert topk % pool_size == 0
    assert group_ids.shape[1] == history_group_budget_for_topk(topk, pool_size)
    assert page_table is None or topk_offsets is None

    device = group_ids.device
    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = (
        group_valid.unsqueeze(-1)
        .expand(-1, -1, pool_size)
        .reshape(group_ids.shape[0], topk)
    )

    if page_table is not None:
        assert page_table.ndim == 2
        safe_ids = token_ids.clamp(min=0, max=page_table.shape[1] - 1)
        output = torch.gather(page_table, dim=1, index=safe_ids).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2:
            assert topk_offsets.shape[1] == 1
            topk_offsets = topk_offsets.squeeze(1)
        output = (token_ids + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = token_ids.to(torch.int32)

    return torch.where(valid, output, torch.full_like(output, -1))


def append_tail_to_topk(
    topk_result: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_lens: torch.Tensor,
    pool_size: int,
    page_table: torch.Tensor | None = None,
    topk_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append non-pooled tail tokens after expanded history tokens.

    ``index_kpool_always_select_tail`` keeps the (incomplete) trailing pool so
    the most recent tokens are always attended to.

    (Donor verbatim — pure torch, any device, asserts kept.)
    """
    assert topk_result.dtype == torch.int32
    assert seq_lens.ndim == 1
    assert pool_lens.ndim == 1

    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result

    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    out = torch.empty(
        (rows, out_cols), dtype=topk_result.dtype, device=topk_result.device
    )

    # tail tokens: [pool_len*pool_size, seq_len) for each row.
    pool_len = pool_lens.to(torch.int32)
    tail_start = pool_len * pool_size
    seq_len = seq_lens.to(torch.int32)
    tail_count = seq_len - tail_start  # in [0, pool_size)

    cols = torch.arange(out_cols, device=topk_result.device)[None, :]
    history_len = n_cols
    is_history = cols < history_len
    tail_off = cols - history_len
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])

    # safe_hist must be per-row [rows, out_cols] so the gather reads each row's
    # OWN history. cols is [1, out_cols]; if used directly, gather (which does
    # NOT broadcast the index) would read only row 0 of topk_result, making every
    # query inherit row 0's history (empty for the first token) and lose all its
    # selected tokens — only the per-row tail would survive. This only manifests
    # for multi-row sparse PREFILL (decode has 1 row, so it reads its own row 0).
    safe_hist = torch.minimum(cols, torch.full_like(cols, n_cols - 1)).expand(
        rows, out_cols
    )
    history_val = torch.gather(topk_result, 1, safe_hist)

    tail_raw = tail_start[:, None] + tail_off
    tail_val = tail_raw.to(torch.int32)
    if page_table is not None:
        safe_tail = tail_raw.clamp(min=0, max=page_table.shape[1] - 1)
        tail_val = torch.gather(page_table, 1, safe_tail).to(torch.int32)
    elif topk_offsets is not None:
        tail_val = (tail_raw + topk_offsets.to(torch.int64).unsqueeze(1)).to(
            torch.int32
        )

    out = torch.where(is_history, history_val, -1)
    out = torch.where(is_tail, tail_val, out)
    return out


def _expand_violations(pool_ids: torch.Tensor, seq_lens: torch.Tensor) -> list:
    v: list = []
    if not (pool_ids.is_cuda and seq_lens.is_cuda):
        v.append(
            f"pool_ids/seq_lens must be CUDA-resident (got {pool_ids.device} / "
            f"{seq_lens.device})"
        )
        return v
    if pool_ids.device != seq_lens.device:
        v.append(f"pool_ids on {pool_ids.device}, seq_lens on {seq_lens.device}")
    if pool_ids.dim() != 2:
        v.append(f"pool_ids must be 2-D [rows, n_groups], got {tuple(pool_ids.shape)}")
    if seq_lens.shape != (pool_ids.shape[0],):
        v.append(
            f"seq_lens shape {tuple(seq_lens.shape)} != ({pool_ids.shape[0]},)"
        )
    if seq_lens.dtype != torch.int32:
        v.append(f"seq_lens must be int32 (kernel loads int32), got {seq_lens.dtype}")
    if pool_ids.dim() == 2 and pool_ids.shape[1] < 1:
        v.append(f"pool_ids needs >= 1 selected pool per row, got {pool_ids.shape[1]}")
    return v


def expand_pools_and_append_tail_eligible(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
) -> bool:
    """Contract check for :func:`expand_pools_and_append_tail`."""
    return not _expand_violations(pool_ids, seq_lens)


def expand_pools_and_append_tail(
    pool_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    pool_size: int,
) -> torch.Tensor:
    """Fuse ``expand_pools_to_tokens`` + ``append_tail_to_topk`` (identity path).

    Donor contract verbatim: produces the ``[rows, n_groups*pool_size +
    pool_size - 1]`` int32 output (selected pools expanded to token ids, then
    the request's trailing incomplete pool appended, ``-1`` masking) when
    neither ``page_table`` nor ``topk_offsets`` applies — the only path the
    GLM-5.3-Flash indexer uses. ``pool_len = seq_len // pool_size`` is
    derived inside the kernel. Replaces ~25 elementwise kernels with one
    Triton launch. ``seq_lens`` is int32 and TOKEN-granular. Allocates the
    output per call. Raises
    :class:`~vkernels.torch_ops._dispatch.OpNotEligible` on a contract miss —
    CPU callers compose the torch twins
    (``expand_pools_to_tokens`` + ``append_tail_to_topk``) instead.
    """
    violations = _expand_violations(pool_ids, seq_lens)
    if violations:
        raise OpNotEligible(
            "expand_pools_and_append_tail contract: " + "; ".join(violations)
        )

    import triton

    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    out = torch.empty((rows, out_cols), dtype=torch.int32, device=pool_ids.device)
    BLOCK_COLS = 128
    n_tiles = triton.cdiv(out_cols, BLOCK_COLS)
    _kernels().expand_tail[(rows, n_tiles)](
        pool_ids,
        seq_lens,
        out,
        topk,
        out_cols,
        POOL_SIZE=pool_size,
        BLOCK_COLS=BLOCK_COLS,
        pid_s0=pool_ids.stride(0),
        out_s0=out.stride(0),
    )
    return out
