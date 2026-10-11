"""CDNA3 (gfx942/MI300A) port of TokenSpeed's GFX950 MHA decode kernels.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel-amd`` ``main`` (MIT,
fetched 2026-11): ``ops/gfx950/attention/mha/decode.py`` + the pieces of
``attention/_common.py`` this closure needs. This is a PORT, not a verbatim
vendor — the deltas, each proven on MI300A by
``tests/python/test_tokenspeed_amd.py::test_gpu_gluon_cdna3_mfma_tile``:

* ``cdna4.buffer_load/store`` → the CDNA3-first ``cdna4`` alias (identical
  buffer-op API; see ``kda_decode.py``).
* ``cdna4.mfma`` → ``amd.cdna3.mfma`` with ``AMDMFMALayout(version=2,
  instr_shape=[16,16,16])`` (upstream decode uses the v4 [16,16,32] shape,
  a CDNA4-only K-width; CDNA3 bf16 MFMA is [16,16,16]/[32,32,8]).
* ``cdna4.async_copy.global_load_to_shared`` K/V tile loads → synchronous
  global loads staged through ``SwizzledSharedLayout`` memory (the
  DMA-legal padded layouts upstream derives from
  ``cdna4.compute_efficient_padded_shared_layout`` have no CDNA3
  counterpart, and CDNA3 has no TMA anyway). First cut is unpipelined —
  one smem slot, no commit/wait groups; the ``warp_pipeline_stage``
  scheduler hints are kept.
* FP8 KV paths are dropped (they need ``dot_scaled`` MFMA); bf16/fp16 KV
  only — the launcher raises on fp8 inputs.
* ``_GFX950_SM_COUNT`` hardcode (256) → the launching device's
  ``multi_processor_count`` (228 on MI300A).

Contract (unchanged from upstream): q [total_q, H_q, D], paged K/V caches
[pages, page_size, H_kv, D], page_table [batch, pages_per_seq],
cache_seqlens [batch] (q token at position cache_seqlens-1), split-K
decode + reduce, optional sliding window and sinks, output q-shaped.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from vkernels.torch_ops.tokenspeed_amd._gluon import gl, gluon, tl

from triton.experimental.gluon.language import amd

# Upstream binds ``gl.amd.cdna4``; CDNA3 exposes identical buffer ops.
_amd_ns = getattr(gl, "amd", None)
cdna4 = getattr(_amd_ns, "cdna3", None) or getattr(_amd_ns, "cdna4", None)

cdiv = gl.cdiv

_INV_LN2 = 1.4426950408889634
_INV_LN2_VALUE = 1.4426950408889634


@gluon.jit
def maximum(a, b, propagate_nan: gl.constexpr = tl.PropagateNan.ALL):
    return gl.maximum(a, b, propagate_nan=propagate_nan)


@gluon.jit
def max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, maximum, keep_dims=keep_dims)


MAX_KV_SPLITS = 32


def select_kv_splits(*, base_ctas: int, num_pages: int, sm_count: int) -> int:
    """Upstream verbatim: balance occupancy against reduce overhead."""
    wave_target = 2
    min_pages_per_split = 2
    min_page_splits = 8
    max_page_splits = MAX_KV_SPLITS

    target_ctas = sm_count * wave_target
    splits_for_occupancy = (target_ctas + base_ctas - 1) // base_ctas

    splits_for_pages = num_pages // min_pages_per_split
    min_page_splits = min(min_page_splits, num_pages)
    if splits_for_pages < min_page_splits:
        splits_for_pages = min_page_splits
    if splits_for_pages > max_page_splits:
        splits_for_pages = max_page_splits
    return min(splits_for_occupancy, splits_for_pages)


@gluon.constexpr_function
def attention_layouts(head_dim, block_n, dtype, num_warps):
    """CDNA3 layouts (replaces upstream's v4/async_copy-derived set).

    bf16/fp16 only. MFMA v2 instr [16, 16, 16] with 1-warp tiling matches
    the decode BLOCK_M=16 tiles; operand k-widths follow the upstream
    128-bit-load rule (8 elems at 16-bit); K/V tiles stage through plain
    ``SwizzledSharedLayout`` shared memory (probe-proven) instead of the
    CDNA4 DMA-legal padded layouts.
    """
    mfma = gl.amd.AMDMFMALayout(
        version=2,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[num_warps, 1],
    )
    qk_layout = mfma
    pv_layout = mfma
    qk_kw = 8
    pv_kw = 4
    q_layout = gl.DotOperandLayout(0, qk_layout, k_width=qk_kw)
    k_layout = gl.DotOperandLayout(1, qk_layout, k_width=qk_kw)
    p_layout = gl.DotOperandLayout(0, pv_layout, k_width=pv_kw)
    v_layout = gl.DotOperandLayout(1, pv_layout, k_width=pv_kw)
    load_vec = 8
    load_threads = head_dim // load_vec
    load_layout = gl.BlockedLayout(
        [1, load_vec], [64 // load_threads, load_threads], [num_warps, 1], [1, 0]
    )
    store_vec = 8
    store_threads = head_dim // store_vec
    store_layout = gl.BlockedLayout(
        [1, store_vec], [64 // store_threads, store_threads], [num_warps, 1], [1, 0]
    )
    k_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    v_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    return (
        qk_layout,
        pv_layout,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        load_layout,
        store_layout,
        k_smem_layout,
        v_smem_layout,
    )


@gluon.aggregate
class InputStrides:
    stride_t: gl.constexpr
    stride_h: gl.constexpr
    stride_d: gl.constexpr

    @gluon.jit
    def offsets(self, token, head, dim):
        return (token * self.stride_t + head * self.stride_h + dim * self.stride_d).to(
            gl.int32
        )


# ===-----------------------------------------------------------------------===#
# Kernel Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionConfig:
    SM_SCALE: gl.constexpr
    PAGE_TABLE_STRIDE: gl.constexpr
    PAGE_SIZE: gl.constexpr
    NUM_KV_SPLITS: gl.constexpr
    MAX_SEQLEN_Q: gl.constexpr
    NUM_Q_HEADS: gl.constexpr
    NUM_KV_HEADS: gl.constexpr
    HEAD_DIM: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    IS_SLIDING: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    GROUP_SIZE: gl.constexpr
    NUM_GROUPS: gl.constexpr
    q_strides: InputStrides
    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    store_layout: gl.constexpr
    reduce_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        KV_DTYPE,
        q_strides,
    ):
        assert NUM_Q_HEADS % NUM_KV_HEADS == 0
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE
        if IS_SLIDING:
            assert WINDOW_LEFT >= 0
        else:
            assert WINDOW_LEFT == -1

        (
            qk_layout,
            pv_layout,
            q_layout,
            k_layout,
            p_layout,
            v_layout,
            load_layout,
            store_layout,
            k_smem_layout,
            v_smem_layout,
        ) = attention_layouts(
            HEAD_DIM,
            BLOCK_N,
            KV_DTYPE,
            num_warps=1,
        )
        reduce_layout = gl.BlockedLayout([1, HEAD_DIM // 64], [1, 64], [1, 1], [1, 0])

        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.PAGE_TABLE_STRIDE = gl.constexpr(PAGE_TABLE_STRIDE)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.NUM_KV_SPLITS = gl.constexpr(NUM_KV_SPLITS)
        self.MAX_SEQLEN_Q = gl.constexpr(MAX_SEQLEN_Q)
        self.NUM_Q_HEADS = gl.constexpr(NUM_Q_HEADS)
        self.NUM_KV_HEADS = gl.constexpr(NUM_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.IS_SLIDING = gl.constexpr(IS_SLIDING)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.GROUP_SIZE = gl.constexpr(NUM_Q_HEADS // NUM_KV_HEADS)
        self.NUM_GROUPS = gl.constexpr((self.GROUP_SIZE + BLOCK_M - 1) // BLOCK_M)
        self.q_strides = q_strides
        self.qk_layout = gl.constexpr(qk_layout)
        self.pv_layout = gl.constexpr(pv_layout)
        self.q_layout = gl.constexpr(q_layout)
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(p_layout)
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.store_layout = gl.constexpr(store_layout)
        self.reduce_layout = gl.constexpr(reduce_layout)
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


# ===-----------------------------------------------------------------------===#
# Kernel Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    k_cache_ptr: gl.tensor
    v_cache_ptr: gl.tensor
    page_table_ptr: gl.tensor
    cache_seqlens_ptr: gl.tensor
    mid_o_ptr: gl.tensor
    mid_lse_ptr: gl.tensor
    q_index: gl.tensor
    batch: gl.tensor
    kv_head: gl.tensor
    group_start: gl.tensor
    split_id: gl.tensor
    cache_len: gl.tensor
    kv_start: gl.tensor
    split_start: gl.tensor
    split_end: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
        q_index,
        batch,
        kv_head,
        group_start,
        split_id,
        cache_len,
        kv_start,
        split_start,
        split_end,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.k_cache_ptr = k_cache_ptr
        self.v_cache_ptr = v_cache_ptr
        self.page_table_ptr = page_table_ptr
        self.cache_seqlens_ptr = cache_seqlens_ptr
        self.mid_o_ptr = mid_o_ptr
        self.mid_lse_ptr = mid_lse_ptr
        self.q_index = q_index
        self.batch = batch
        self.kv_head = kv_head
        self.group_start = group_start
        self.split_id = split_id
        self.cache_len = cache_len
        self.kv_start = kv_start
        self.split_start = split_start
        self.split_end = split_end

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
    ):
        q_index = gl.program_id(0)
        batch = q_index // cfg.MAX_SEQLEN_Q
        q_pos = q_index - batch * cfg.MAX_SEQLEN_Q
        head_block = gl.program_id(1)
        kv_head = head_block // cfg.NUM_GROUPS
        group_block = head_block - kv_head * cfg.NUM_GROUPS
        group_start = group_block * cfg.BLOCK_M
        split_id = gl.program_id(2)
        cache_len = gl.load(cache_seqlens_ptr + batch)
        cache_len = cache_len - (cfg.MAX_SEQLEN_Q - 1 - q_pos)
        cache_len = maximum(cache_len, 0)
        if cfg.IS_SLIDING:
            kv_start = cache_len - min(cache_len, cfg.WINDOW_LEFT + 1)
        else:
            kv_start = cache_len - cache_len
        first_page = kv_start // cfg.PAGE_SIZE
        end_page = cdiv(cache_len, cfg.PAGE_SIZE)
        num_pages = end_page - first_page
        pages_per_split = cdiv(num_pages, cfg.NUM_KV_SPLITS)
        split_start_page = first_page + split_id * pages_per_split
        split_end_page = min(split_start_page + pages_per_split, end_page)
        split_start = split_start_page * cfg.PAGE_SIZE
        split_end = min(split_end_page * cfg.PAGE_SIZE, cache_len)
        return AttentionProgram(
            gl.constexpr(cfg),
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            cache_seqlens_ptr,
            mid_o_ptr,
            mid_lse_ptr,
            q_index,
            batch,
            kv_head,
            group_start,
            split_id,
            cache_len,
            kv_start,
            split_start,
            split_end,
        )

    @gluon.jit
    def load_q(self, q_smem):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        # PORT: upstream's cdna4 buffer_load yields an operand-ready register
        # layout; on CDNA3 the q operand stages through shared memory like
        # the probe tile (see test_gpu_gluon_cdna3_mfma_tile).
        tile = gl.load(
            self.q_ptr + self.q_index * cfg.q_strides.stride_t
            + q_heads[:, None] * cfg.q_strides.stride_h
            + offs_d[None, :] * cfg.q_strides.stride_d,
            mask=valid[:, None],
            other=0.0,
        )
        q_smem.store(tile)
        return q_smem.load(cfg.q_layout)

    @gluon.jit
    def init_state(self, sink_ptr, has_sink: gl.constexpr):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        if has_sink:
            sink_log2 = (
                gl.load(sink_ptr + q_heads, mask=valid, other=0.0).to(gl.float32)
                * _INV_LN2
            )
            m_i = sink_log2 / cfg.SM_SCALE
        else:
            sink_log2 = gl.full(
                [cfg.BLOCK_M],
                value=0.0,
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout),
            )
            m_i = gl.full(
                [cfg.BLOCK_M],
                value=-float("inf"),
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout),
            )
        l_i = gl.full(
            [cfg.BLOCK_M],
            value=0.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout
        )
        return m_i, l_i, acc, sink_log2

    @gluon.jit
    def apply_sinks(self, l_i, m_i, sink_log2, has_sink: gl.constexpr):
        cfg = self.cfg
        if has_sink:
            l_i += gl.exp2(sink_log2 - m_i * cfg.SM_SCALE)
        return l_i

    @gluon.jit
    def load_page(self, start_n):
        cfg = self.cfg
        page_index = start_n // cfg.PAGE_SIZE
        valid = start_n < self.split_end
        return gl.load(
            self.page_table_ptr + self.batch * cfg.PAGE_TABLE_STRIDE + page_index,
            mask=valid,
            other=0,
        )

    # --- PORT: the async_copy issue/wait pairs become synchronous loads. The
    # global load lands in registers with the blocked load_layout, stages
    # through shared memory, and re-loads with the dot-operand layout (the
    # probe-proven CDNA3 path; see test_gpu_gluon_cdna3_mfma_tile).

    @gluon.jit
    def load_k(self, physical_page, k_smem):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        token_loc = physical_page.to(gl.int64) * cfg.PAGE_SIZE + offs_n.to(gl.int64)
        offsets = (
            token_loc[:, None] * cfg.NUM_KV_HEADS * cfg.HEAD_DIM
            + self.kv_head * cfg.HEAD_DIM
            + offs_d[None, :]
        )
        valid = (offs_n < cfg.PAGE_SIZE)[:, None] & (offs_d < cfg.HEAD_DIM)[None, :]
        tile = gl.load(self.k_cache_ptr + offsets, mask=valid, other=0.0)
        # PORT: upstream async-copies the [N, D] tile and reads a permuted
        # view (DMA-legal on CDNA4). Here the tile is staged genuinely
        # transposed: the register permute happens before the shared store,
        # and the operand load reads the physical [D, N] buffer directly.
        k_smem.store(tile.permute([1, 0]))
        return k_smem.load(cfg.k_layout)

    @gluon.jit
    def load_v(self, physical_page, v_smem):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        token_loc = physical_page.to(gl.int64) * cfg.PAGE_SIZE + offs_n.to(gl.int64)
        offsets = (
            token_loc[:, None] * cfg.NUM_KV_HEADS * cfg.HEAD_DIM
            + self.kv_head * cfg.HEAD_DIM
            + offs_d[None, :]
        )
        valid = (offs_n < cfg.PAGE_SIZE)[:, None] & (offs_d < cfg.HEAD_DIM)[None, :]
        tile = gl.load(self.v_cache_ptr + offsets, mask=valid, other=0.0)
        v_smem.store(tile)
        return v_smem.load(cfg.v_layout)

    @gluon.jit
    def compute_qk(self, q, k):
        cfg = self.cfg
        qk = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout
        )
        return amd.cdna3.mfma(q, k, qk)

    @gluon.jit
    def apply_kv_mask(self, qk, start_n):
        cfg = self.cfg
        offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
        tokens = start_n + offs_n[None, :]
        mask = (tokens >= self.kv_start) & (tokens < self.split_end)
        return gl.where(mask, qk, -float("inf"))

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        row_max = max(qk, axis=1)
        row_max = gl.convert_layout(row_max, gl.SliceLayout(1, cfg.pv_layout))
        m_new = maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return amd.cdna3.mfma(p, v, acc)

    @gluon.jit
    def store_split(self, acc, l_i, m_i):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = ((self.group_start + offs_m) < cfg.GROUP_SIZE) & (
            self.split_start < self.split_end
        )
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        m_i = gl.convert_layout(m_i, gl.SliceLayout(1, cfg.store_layout))
        recip_l_i = 1.0 / l_i
        part_o = acc * recip_l_i[:, None]
        part_lse = m_i * cfg.SM_SCALE + gl.log2(l_i)
        mid_o_offsets = (
            (self.q_index * cfg.NUM_Q_HEADS + q_heads[:, None]) * cfg.NUM_KV_SPLITS
            + self.split_id
        ) * cfg.HEAD_DIM + offs_d[None, :]
        mid_lse_offsets = (
            self.q_index * cfg.NUM_Q_HEADS + q_heads
        ) * cfg.NUM_KV_SPLITS + self.split_id
        cdna4.buffer_store(part_o, self.mid_o_ptr, mid_o_offsets, mask=valid[:, None])
        cdna4.buffer_store(part_lse, self.mid_lse_ptr, mid_lse_offsets, mask=valid)

    @gluon.jit
    def store_output(self, acc, l_i):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout))
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        acc = gl.convert_layout(acc, cfg.store_layout)
        l_i = gl.convert_layout(l_i, gl.SliceLayout(1, cfg.store_layout))
        output = acc * (1.0 / l_i)[:, None]
        output = output.to(self.mid_o_ptr.dtype.element_ty)
        offsets = (self.q_index * cfg.NUM_Q_HEADS + q_heads[:, None]) * cfg.HEAD_DIM
        offsets += offs_d[None, :]
        cdna4.buffer_store(output, self.mid_o_ptr, offsets, mask=valid[:, None])


# ===-----------------------------------------------------------------------===#
# Entry Points
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_mha_decode_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    mid_o_ptr,
    mid_lse_ptr,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
    )
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.HEAD_DIM, cfg.BLOCK_N], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )
    q_smem = gl.allocate_shared_memory(
        q_ptr.dtype.element_ty, [cfg.BLOCK_M, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q(q_smem)
    m_i, l_i, acc, sink_log2 = program.init_state(q_ptr, False)

    physical_page = program.load_page(program.split_start)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            k = program.load_k(physical_page, k_smem)
            v = program.load_v(physical_page, v_smem)

        with gl.amd.warp_pipeline_stage("qk_softmax", priority=0):
            qk = program.compute_qk(q, k)
            qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        with gl.amd.warp_pipeline_stage("pv", priority=1):
            acc = program.compute_pv(p, v, acc)
            physical_page = program.load_page(start_n + cfg.BLOCK_N)

    program.store_split(acc, l_i, m_i)


@gluon.jit
def gluon_mha_decode_sliding_gfx950(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    out_ptr,
    sink_ptr,
    Q_STRIDE_B: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    IS_SLIDING: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    HAS_SINK: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        1,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        IS_SLIDING,
        WINDOW_LEFT,
        k_cache_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        out_ptr,
        out_ptr,
    )
    k_smem = gl.allocate_shared_memory(
        k_cache_ptr.dtype.element_ty, [cfg.HEAD_DIM, cfg.BLOCK_N], cfg.k_smem_layout
    )
    v_smem = gl.allocate_shared_memory(
        v_cache_ptr.dtype.element_ty, [cfg.BLOCK_N, cfg.HEAD_DIM], cfg.v_smem_layout
    )
    q_smem = gl.allocate_shared_memory(
        q_ptr.dtype.element_ty, [cfg.BLOCK_M, cfg.HEAD_DIM], cfg.v_smem_layout
    )

    q = program.load_q(q_smem)
    m_i, l_i, acc, sink_log2 = program.init_state(sink_ptr, HAS_SINK)

    physical_page = program.load_page(program.split_start)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            k = program.load_k(physical_page, k_smem)
            v = program.load_v(physical_page, v_smem)

        with gl.amd.warp_pipeline_stage("qk_softmax", priority=0):
            qk = program.compute_qk(q, k)
            qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        with gl.amd.warp_pipeline_stage("pv", priority=1):
            acc = program.compute_pv(p, v, acc)
            physical_page = program.load_page(start_n + cfg.BLOCK_N)

    l_i = program.apply_sinks(l_i, m_i, sink_log2, HAS_SINK)
    program.store_output(acc, l_i)


@gluon.jit
def gluon_mha_decode_reduce_gfx950(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    cache_seqlens_ptr,
    sink_ptr,
    k_cache_ptr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    NUM_KV_SPLITS: gl.constexpr,
    MAX_SEQLEN_Q: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HAS_SINK: gl.constexpr,
):
    cfg = AttentionConfig(
        SM_SCALE,
        PAGE_TABLE_STRIDE,
        PAGE_SIZE,
        NUM_KV_SPLITS,
        MAX_SEQLEN_Q,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_N,
        False,
        -1,
        k_cache_ptr.dtype.element_ty,
        InputStrides(1, 1, 1),
    )
    q_index = gl.program_id(0)
    batch = q_index // MAX_SEQLEN_Q
    q_pos = q_index - batch * MAX_SEQLEN_Q
    q_head = gl.program_id(1)
    cache_len = gl.load(cache_seqlens_ptr + batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = maximum(cache_len, 0)
    first_page = 0
    end_page = cdiv(cache_len, cfg.PAGE_SIZE)
    num_pages = end_page - first_page
    pages_per_split = cdiv(num_pages, cfg.NUM_KV_SPLITS)

    SPLIT_TILE: gl.constexpr = 1 << (NUM_KV_SPLITS - 1).bit_length()
    offs_s = gl.arange(0, SPLIT_TILE, layout=gl.SliceLayout(1, cfg.reduce_layout))
    offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.reduce_layout))
    split_start_page = first_page + offs_s * pages_per_split
    split_end_page_raw = split_start_page + pages_per_split
    split_end_page = gl.where(
        split_end_page_raw < end_page, split_end_page_raw, end_page
    )
    split_start_tok = split_start_page * cfg.PAGE_SIZE
    split_end_raw = split_end_page * cfg.PAGE_SIZE
    split_end_tok = gl.where(split_end_raw < cache_len, split_end_raw, cache_len)
    split_valid = (split_start_tok < split_end_tok) & (offs_s < cfg.NUM_KV_SPLITS)
    base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.NUM_KV_SPLITS + offs_s
    part_lse = gl.load(mid_lse_ptr + base, mask=split_valid, other=-float("inf"))
    o_off = base[:, None] * cfg.HEAD_DIM + offs_d[None, :]
    part_o = cdna4.buffer_load(mid_o_ptr, o_off, mask=split_valid[:, None], other=0.0)

    m_i = max(part_lse, axis=0)
    if HAS_SINK:
        sink = gl.load(sink_ptr + q_head).to(gl.float32) * _INV_LN2
        m_i = maximum(m_i, sink)
    beta = gl.exp2(part_lse - m_i)
    l_i = gl.sum(beta, axis=0)
    if HAS_SINK:
        l_i = l_i + gl.exp2(sink - m_i)
    acc = gl.sum(part_o * beta[:, None], axis=0)

    out_base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.HEAD_DIM
    output = acc * (1.0 / l_i)
    output = output.to(out_ptr.dtype.element_ty)
    cdna4.buffer_store(output, out_ptr, out_base + offs_d)


# ===-----------------------------------------------------------------------===#
# Host
# ===-----------------------------------------------------------------------===#


class LaunchConfig(NamedTuple):
    num_q_heads: int
    num_kv_heads: int
    num_groups: int
    head_dim: int
    page_size: int
    num_kv_splits: int
    block_m: int
    block_n: int
    sm_scale: float
    is_sliding: bool
    window_left: int


def get_config(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    max_seqlen_k: int,
    window_left: int,
    softmax_scale: float | None,
) -> LaunchConfig:
    head_dim = q.shape[2]
    page_size = k_cache.shape[1]
    block_m = 16
    block_n = 64
    group_size = q.shape[1] // k_cache.shape[2]
    num_groups = math.ceil(group_size / block_m)
    is_sliding = window_left >= 0
    window_left = window_left if is_sliding else -1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    effective_seqlen_k = min(max_seqlen_k, window_left) if is_sliding else max_seqlen_k
    num_pages = (effective_seqlen_k + page_size - 1) // page_size
    # PORT: upstream hardcodes 256 (MI355X); MI300A has 228 CUs.
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_kv_splits = select_kv_splits(
        base_ctas=q.shape[0] * k_cache.shape[2] * num_groups,
        num_pages=num_pages,
        sm_count=sm_count,
    )
    return LaunchConfig(
        num_q_heads=q.shape[1],
        num_kv_heads=k_cache.shape[2],
        num_groups=num_groups,
        head_dim=head_dim,
        page_size=page_size,
        num_kv_splits=num_kv_splits,
        block_m=block_m,
        block_n=block_n,
        sm_scale=softmax_scale * _INV_LN2_VALUE,
        is_sliding=is_sliding,
        window_left=window_left,
    )


def launch_gluon_mha_decode_gfx942(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    sinks: torch.Tensor | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Paged MHA decode (CDNA3 port). bf16/fp16 KV only; fp8 raises."""
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or k_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError(
            "the CDNA3 MHA decode port supports bf16/fp16 KV only (fp8 needs "
            "dot_scaled MFMA — not ported)"
        )
    total_q = q.shape[0]

    has_sink = sinks is not None
    sink_arg = sinks if sinks is not None else q
    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=max_seqlen_k,
        window_left=window_left,
        softmax_scale=softmax_scale,
    )

    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    if config.is_sliding:
        grid = (total_q, config.num_kv_heads * config.num_groups)
        gluon_mha_decode_sliding_gfx950[grid](
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            output,
            sink_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            config.sm_scale,
            page_table.stride(0),
            config.page_size,
            max_seqlen_q,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            config.is_sliding,
            config.window_left,
            has_sink,
            num_warps=1,
        )
    else:
        mid_o = torch.empty(
            (total_q, config.num_q_heads, config.num_kv_splits, config.head_dim),
            device=q.device,
            dtype=torch.float32,
        )
        mid_lse = torch.empty(
            (total_q, config.num_q_heads, config.num_kv_splits),
            device=q.device,
            dtype=torch.float32,
        )

        grid = (
            total_q,
            config.num_kv_heads * config.num_groups,
            config.num_kv_splits,
        )
        gluon_mha_decode_gfx950[grid](
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            mid_o,
            mid_lse,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            config.sm_scale,
            page_table.stride(0),
            config.page_size,
            config.num_kv_splits,
            max_seqlen_q,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            config.is_sliding,
            config.window_left,
            num_warps=1,
        )

        grid = (total_q, config.num_q_heads)
        gluon_mha_decode_reduce_gfx950[grid](
            mid_o,
            mid_lse,
            output,
            cache_seqlens,
            sink_arg,
            k_cache,
            config.sm_scale,
            page_table.stride(0),
            config.num_kv_splits,
            max_seqlen_q,
            config.page_size,
            config.num_q_heads,
            config.num_kv_heads,
            config.head_dim,
            config.block_m,
            config.block_n,
            has_sink,
            num_warps=1,
        )
    return output


__all__ = ["launch_gluon_mha_decode_gfx942"]
