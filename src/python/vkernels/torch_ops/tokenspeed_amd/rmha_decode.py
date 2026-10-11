"""CDNA3 (gfx942/MI300A) port of TokenSpeed's GFX950 relative-position MHA
decode kernel.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel-amd`` ``main`` (MIT,
fetched 2026-11): ``ops/gfx950/attention/rmha/decode.py``. PORT (not a
verbatim vendor), following the proven ``mha_decode.py`` recipe:

* ``AMDMFMALayout(version=2, instr_shape=[16,16,16], transposed=False)``
  (upstream: v4 [16,16,32] transposed), 1-warp tiling as upstream.
* q stages through shared memory into the operand layout; the K/V
  ``async_copy.global_load_to_shared`` pairs become synchronous loads into
  dual physical buffers (K pre-permuted ``[HEAD_DIM, BLOCK_N]``, V natural
  ``[BLOCK_N, HEAD_DIM]``).
* The rel-bias add (``apply_rel_bias``) and the split-K reduce kernel are
  upstream verbatim over the CDNA3-first buffer-op alias.
* FP8 dropped (bf16/fp16 only; the launcher raises). Occupancy uses the
  device CU count (228 on MI300A; upstream hardcodes 256).

Contract (unchanged): paged split-K decode with a per-head relative-
position bias table ``rel_logits [total_q, H, REL_EXTENT]`` added to QK^T
before masking (scaled by ``rel_bias_qk_scale``), optional sliding window.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from vkernels.torch_ops.tokenspeed_amd._gluon import gl, gluon

from triton.experimental.gluon.language import amd

# Upstream binds ``gl.amd.cdna4``; CDNA3 exposes identical buffer ops.
_amd_ns = getattr(gl, "amd", None)
cdna4 = getattr(_amd_ns, "cdna3", None) or getattr(_amd_ns, "cdna4", None)

cdiv = gl.cdiv

_INV_LN2_VALUE = 1.4426950408889634


@gluon.jit
def _maximum(a, b):
    return gl.maximum(a, b)


@gluon.jit
def _max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, _maximum, keep_dims=keep_dims)


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
    REL_EXTENT: gl.constexpr
    REL_BIAS_QK_SCALE: gl.constexpr
    GROUP_SIZE: gl.constexpr
    NUM_GROUPS: gl.constexpr
    q_strides: InputStrides
    rel_strides: InputStrides
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
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        q_strides,
        rel_strides,
    ):
        assert NUM_Q_HEADS % NUM_KV_HEADS == 0
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE
        if IS_SLIDING:
            assert WINDOW_LEFT >= 0
        else:
            assert WINDOW_LEFT == -1

        # PORT: v2 [16,16,16] non-transposed (upstream: v4 [16,16,32]
        # transposed). Operand k-widths follow the 128-bit-load rule.
        mfma_layout = gl.amd.AMDMFMALayout(
            version=2,
            instr_shape=[16, 16, 16],
            transposed=False,
            warps_per_cta=[1, 1],
        )
        qk_layout = mfma_layout
        pv_layout = mfma_layout
        qk_kw = 8
        pv_kw = 4
        q_layout = gl.DotOperandLayout(0, qk_layout, k_width=qk_kw)
        k_layout = gl.DotOperandLayout(1, qk_layout, k_width=qk_kw)
        p_layout = gl.DotOperandLayout(0, pv_layout, k_width=pv_kw)
        v_layout = gl.DotOperandLayout(1, pv_layout, k_width=pv_kw)
        load_vec = 8
        load_threads = HEAD_DIM // load_vec
        load_layout = gl.BlockedLayout(
            [1, load_vec], [64 // load_threads, load_threads], [1, 1], [1, 0]
        )
        store_vec = 8
        store_threads = HEAD_DIM // store_vec
        store_layout = gl.BlockedLayout(
            [1, store_vec], [64 // store_threads, store_threads], [1, 1], [1, 0]
        )
        reduce_layout = gl.BlockedLayout([1, HEAD_DIM // 64], [1, 64], [1, 1], [1, 0])
        k_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
        v_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

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
        self.REL_EXTENT = gl.constexpr(REL_EXTENT)
        self.REL_BIAS_QK_SCALE = gl.constexpr(REL_BIAS_QK_SCALE)
        self.GROUP_SIZE = gl.constexpr(NUM_Q_HEADS // NUM_KV_HEADS)
        self.NUM_GROUPS = gl.constexpr((self.GROUP_SIZE + BLOCK_M - 1) // BLOCK_M)
        self.q_strides = q_strides
        self.rel_strides = rel_strides
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
    rel_logits_ptr: gl.tensor
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
        rel_logits_ptr,
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
        self.rel_logits_ptr = rel_logits_ptr
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
        rel_logits_ptr,
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
        cache_len = _maximum(cache_len, 0)
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
            rel_logits_ptr,
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
        # PORT: q stages through shared memory (see mha_decode.py).
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
    def init_state(self):
        cfg = self.cfg
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
        return m_i, l_i, acc

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

    # --- PORT: async issue/wait becomes synchronous dual-orientation staging.

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
    def apply_rel_bias(self, qk, start_n):
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
        offs_n = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)
        )
        q_heads = self.kv_head * cfg.GROUP_SIZE + self.group_start + offs_m
        rel_dist = (self.cache_len - 1) - offs_n
        rel_valid = (rel_dist >= 0) & (rel_dist < cfg.REL_EXTENT)
        rel_idx = gl.where(rel_dist >= 0, rel_dist, 0)
        rel_idx = gl.where(rel_idx < cfg.REL_EXTENT, rel_idx, cfg.REL_EXTENT - 1)
        offsets = cfg.rel_strides.offsets(
            self.q_index, q_heads[:, None], rel_idx[None, :]
        )
        head_valid = (self.group_start + offs_m) < cfg.GROUP_SIZE
        rel_bias = cdna4.buffer_load(
            self.rel_logits_ptr,
            offsets,
            mask=head_valid[:, None] & rel_valid[None, :],
            other=0.0,
        ).to(gl.float32)
        return qk + rel_bias * cfg.REL_BIAS_QK_SCALE

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
        row_max = _max(qk, axis=1)
        row_max = gl.convert_layout(row_max, gl.SliceLayout(1, cfg.pv_layout))
        m_new = _maximum(m_i, row_max)
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


# ===-----------------------------------------------------------------------===#
# Entry Points
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_rel_mha_decode_gfx950(
    q_ptr,
    rel_logits_ptr,
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
    REL_STRIDE_T: gl.constexpr,
    REL_STRIDE_H: gl.constexpr,
    REL_STRIDE_E: gl.constexpr,
    REL_EXTENT: gl.constexpr,
    REL_BIAS_QK_SCALE: gl.constexpr,
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
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        InputStrides(Q_STRIDE_B, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(REL_STRIDE_T, REL_STRIDE_H, REL_STRIDE_E),
    )
    program = AttentionProgram.create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        cache_seqlens_ptr,
        mid_o_ptr,
        mid_lse_ptr,
    )
    # PORT: K stages pre-permuted [HEAD_DIM, BLOCK_N] (dual-orientation).
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
    m_i, l_i, acc = program.init_state()

    physical_page = program.load_page(program.split_start)

    for start_n in range(program.split_start, program.split_end, cfg.BLOCK_N):
        with gl.amd.warp_pipeline_stage("load", priority=1):
            k = program.load_k(physical_page, k_smem)
            v = program.load_v(physical_page, v_smem)

        with gl.amd.warp_pipeline_stage("qk_softmax", priority=0):
            qk = program.compute_qk(q, k)
            qk = program.apply_rel_bias(qk, start_n)
            qk = program.apply_kv_mask(qk, start_n)
            p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)

        with gl.amd.warp_pipeline_stage("pv", priority=1):
            acc = program.compute_pv(p, v, acc)
            physical_page = program.load_page(start_n + cfg.BLOCK_N)

    program.store_split(acc, l_i, m_i)


@gluon.jit
def gluon_rel_mha_decode_reduce_gfx950(
    mid_o_ptr,
    mid_lse_ptr,
    out_ptr,
    cache_seqlens_ptr,
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
        1,
        1.0,
        InputStrides(1, 1, 1),
        InputStrides(1, 1, 1),
    )
    q_index = gl.program_id(0)
    batch = q_index // MAX_SEQLEN_Q
    q_pos = q_index - batch * MAX_SEQLEN_Q
    q_head = gl.program_id(1)
    cache_len = gl.load(cache_seqlens_ptr + batch)
    cache_len = cache_len - (MAX_SEQLEN_Q - 1 - q_pos)
    cache_len = _maximum(cache_len, 0)
    if cfg.IS_SLIDING:
        kv_start = cache_len - min(cache_len, cfg.WINDOW_LEFT + 1)
    else:
        kv_start = cache_len - cache_len
    first_page = kv_start // cfg.PAGE_SIZE
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

    m_i = _max(part_lse, axis=0)
    beta = gl.exp2(part_lse - m_i)
    l_i = gl.sum(beta, axis=0)
    acc = gl.sum(part_o * beta[:, None], axis=0)

    out_base = (q_index * cfg.NUM_Q_HEADS + q_head) * cfg.HEAD_DIM
    output = acc * (1.0 / l_i)
    output = output.to(out_ptr.dtype.element_ty)
    cdna4.buffer_store(output, out_ptr, out_base + offs_d)


# ===-----------------------------------------------------------------------===#
# Host
# ===-----------------------------------------------------------------------===#


def _select_num_kv_splits(
    *, base_ctas: int, num_pages: int, sm_count: int
) -> int:
    """Upstream verbatim: balance occupancy against reduce overhead."""
    wave_target = 2
    min_pages_per_split = 2
    min_page_splits = 8
    max_page_splits = 32

    target_ctas = sm_count * wave_target
    splits_for_occupancy = (target_ctas + base_ctas - 1) // base_ctas

    splits_for_pages = num_pages // min_pages_per_split
    min_page_splits = min(min_page_splits, num_pages)
    if splits_for_pages < min_page_splits:
        splits_for_pages = min_page_splits
    if splits_for_pages > max_page_splits:
        splits_for_pages = max_page_splits
    return min(splits_for_occupancy, splits_for_pages)


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
    rel_bias_qk_scale: float


def get_config(
    *,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    max_seqlen_k: int,
    window_left: int,
    softmax_scale: float | None,
    rel_bias_qk_scale: float,
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
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    num_kv_splits = _select_num_kv_splits(
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
        rel_bias_qk_scale=rel_bias_qk_scale,
    )


def launch_gluon_rel_mha_decode_gfx942(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_k: int,
    rel_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int = 1,
    window_left: int = -1,
    softmax_scale: float | None = None,
    rel_bias_qk_scale: float = 1.0,
) -> torch.Tensor:
    """Paged rel-MHA decode with a relative-position bias (CDNA3 port)."""
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or k_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError(
            "the CDNA3 rel-MHA decode port supports bf16/fp16 only (fp8 needs "
            "dot_scaled MFMA — not ported)"
        )
    total_q = q.shape[0]
    config = get_config(
        q=q,
        k_cache=k_cache,
        max_seqlen_k=max_seqlen_k,
        window_left=window_left,
        softmax_scale=softmax_scale,
        rel_bias_qk_scale=rel_bias_qk_scale,
    )

    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

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
    gluon_rel_mha_decode_gfx950[grid](
        q,
        rel_logits,
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
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        rel_logits.shape[2],
        config.rel_bias_qk_scale,
        num_warps=1,
    )

    grid = (total_q, config.num_q_heads)
    gluon_rel_mha_decode_reduce_gfx950[grid](
        mid_o,
        mid_lse,
        output,
        cache_seqlens,
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
        config.is_sliding,
        config.window_left,
        num_warps=1,
    )
    return output


__all__ = ["launch_gluon_rel_mha_decode_gfx942"]
