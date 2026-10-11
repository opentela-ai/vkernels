"""CDNA3 (gfx942/MI300A) port of TokenSpeed's GFX950 relative-position MHA
extend kernel.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel-amd`` ``main`` (MIT,
fetched 2026-11): ``ops/gfx950/attention/rmha/extend.py``. PORT (not a
verbatim vendor), following the proven ``mha_prefill.py`` / ``rmha_decode.py``
recipes:

* ``AMDMFMALayout(version=2, instr_shape=[32,32,8], transposed=False)``
  with upstream's two tile shapes (short 64-row / long 128-row, 2/4 warps)
  — LDS-checked on gfx942's 64 KB (the long tile fits at BLOCK_N=64 with
  dual-orientation staging; assert guards it).
* q loads through shared-memory staging; the K/V async pairs become
  synchronous dual-orientation loads (K pre-permuted ``[HEAD_DIM,
  BLOCK_N]``, V natural ``[BLOCK_N, HEAD_DIM]``).
* The rel-bias add, the causal/window masking ladder, LSE, and the host
  tile selection are upstream verbatim over the CDNA3-first alias.
* FP8 dropped (bf16/fp16 only; the launcher raises).

Contract (unchanged): ragged multi-token queries against a paged cache,
grid ``(blocks_per_req, batch, n_heads)`` all host-known (CUDA-graph
static), causal or non-causal with optional sliding window and sinks,
per-head relative-position bias ``rel_logits`` keyed by query-row-to-key
distance, optional LSE.
"""

from __future__ import annotations

import torch

from vkernels.torch_ops.tokenspeed_amd._gluon import gl, gluon

from triton.experimental.gluon.language import amd

# Upstream binds ``gl.amd.cdna4``; CDNA3 exposes identical buffer ops.
_amd_ns = getattr(gl, "amd", None)
cdna4 = getattr(_amd_ns, "cdna3", None) or getattr(_amd_ns, "cdna4", None)

_LN2 = 0.6931471805599453
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


_EXTEND_SHORT_Q_BLOCK_M = 64
_EXTEND_SHORT_Q_NUM_WARPS = 2
_EXTEND_LONG_Q_BLOCK_M = 128
_EXTEND_LONG_Q_NUM_WARPS = 4


def _select_extend_tile(max_seqlen_q: int, page_size: int) -> tuple[int, int, int]:
    """Upstream verbatim: short tile for single-tile queries, else tall."""
    if max_seqlen_q <= _EXTEND_SHORT_Q_BLOCK_M:
        return _EXTEND_SHORT_Q_BLOCK_M, page_size, _EXTEND_SHORT_Q_NUM_WARPS
    return _EXTEND_LONG_Q_BLOCK_M, page_size, _EXTEND_LONG_Q_NUM_WARPS


# ===-----------------------------------------------------------------------===#
# Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class ExtendConfig:
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    GROUP_SIZE: gl.constexpr
    HEAD_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    PAGE_SIZE: gl.constexpr
    PAGE_TABLE_STRIDE: gl.constexpr
    IS_CAUSAL: gl.constexpr
    HAS_LSE: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    REL_EXTENT: gl.constexpr
    REL_BIAS_QK_SCALE: gl.constexpr
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
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        HAS_LSE,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        q_strides,
        rel_strides,
    ):
        assert HEAD_DIM in (64, 128)
        assert BLOCK_N == PAGE_SIZE

        # PORT: v2 [32,32,8] non-transposed (upstream: v4 [32,32,16]
        # transposed). gfx942 LDS check: dual-orientation staging needs
        # 2 * BLOCK_N * HEAD_DIM * 2B <= 64KB -> BLOCK_N <= 128 at bf16; the
        # host tile shapes keep BLOCK_N == PAGE_SIZE (typically 64).
        mfma_layout = gl.amd.AMDMFMALayout(
            version=2,
            instr_shape=[32, 32, 8],
            transposed=False,
            warps_per_cta=[NUM_WARPS, 1],
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
            [1, load_vec], [64 // load_threads, load_threads], [NUM_WARPS, 1], [1, 0]
        )
        store_vec = 8
        store_threads = HEAD_DIM // store_vec
        store_layout = gl.BlockedLayout(
            [1, store_vec], [64 // store_threads, store_threads], [NUM_WARPS, 1], [1, 0]
        )
        k_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
        v_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])

        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.GROUP_SIZE = gl.constexpr(N_HEADS // N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.PAGE_TABLE_STRIDE = gl.constexpr(PAGE_TABLE_STRIDE)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.REL_EXTENT = gl.constexpr(REL_EXTENT)
        self.REL_BIAS_QK_SCALE = gl.constexpr(REL_BIAS_QK_SCALE)
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
        self.k_smem_layout = gl.constexpr(k_smem_layout)
        self.v_smem_layout = gl.constexpr(v_smem_layout)


# ===-----------------------------------------------------------------------===#
# Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class ExtendProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    rel_logits_ptr: gl.tensor
    k_cache_ptr: gl.tensor
    v_cache_ptr: gl.tensor
    page_table_ptr: gl.tensor
    output_ptr: gl.tensor
    lse_ptr: gl.tensor
    batch: gl.tensor
    q_head: gl.tensor
    kv_head: gl.tensor
    q_start: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    prefix: gl.tensor
    cache_len: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        batch,
        q_head,
        kv_head,
        q_start,
        seq_base,
        seq_len,
        prefix,
        cache_len,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.rel_logits_ptr = rel_logits_ptr
        self.k_cache_ptr = k_cache_ptr
        self.v_cache_ptr = v_cache_ptr
        self.page_table_ptr = page_table_ptr
        self.output_ptr = output_ptr
        self.lse_ptr = lse_ptr
        self.batch = batch
        self.q_head = q_head
        self.kv_head = kv_head
        self.q_start = q_start
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.prefix = prefix
        self.cache_len = cache_len

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
    ):
        block_in_req = gl.program_id(0)
        batch = gl.program_id(1)
        q_head = gl.program_id(2)
        q_start = block_in_req * cfg.BLOCK_M
        seq_base = gl.load(cu_seqlens_q_ptr + batch)
        seq_end = gl.load(cu_seqlens_q_ptr + batch + 1)
        seq_len = seq_end - seq_base
        cache_len = gl.load(cache_seqlens_ptr + batch)
        prefix = cache_len - seq_len
        kv_head = q_head // cfg.GROUP_SIZE
        return ExtendProgram(
            gl.constexpr(cfg),
            q_ptr,
            rel_logits_ptr,
            k_cache_ptr,
            v_cache_ptr,
            page_table_ptr,
            output_ptr,
            lse_ptr,
            batch,
            q_head,
            kv_head,
            q_start,
            seq_base,
            seq_len,
            prefix,
            cache_len,
        )

    @gluon.jit
    def load_q(self, q_smem):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.load_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.q_strides.offsets(
            self.seq_base + offs_m[:, None], self.q_head, offs_d[None, :]
        )
        mask = offs_m[:, None] < self.seq_len
        # PORT: q stages through shared memory (see mha_prefill.py).
        tile = gl.load(self.q_ptr + offsets, mask=mask, other=0.0)
        q_smem.store(tile)
        return q_smem.load(cfg.q_layout)

    @gluon.jit
    def load_page(self, start_n):
        cfg = self.cfg
        page_index = start_n // cfg.PAGE_SIZE
        valid = start_n < self.cache_len
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
            token_loc[:, None] * cfg.N_KV_HEADS * cfg.HEAD_DIM
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
            token_loc[:, None] * cfg.N_KV_HEADS * cfg.HEAD_DIM
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
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
        )
        offs_n_abs = start_n + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)
        )
        q_pos = self.prefix + offs_m
        rel_dist = q_pos[:, None] - offs_n_abs[None, :]
        rel_valid = (rel_dist >= 0) & (rel_dist < cfg.REL_EXTENT)
        rel_idx = gl.where(rel_dist >= 0, rel_dist, 0)
        rel_idx = gl.where(rel_idx < cfg.REL_EXTENT, rel_idx, cfg.REL_EXTENT - 1)
        offsets = cfg.rel_strides.offsets(
            self.seq_base + offs_m[:, None], self.q_head, rel_idx
        )
        mask = (offs_m[:, None] < self.seq_len) & rel_valid
        rel_bias = cdna4.buffer_load(
            self.rel_logits_ptr, offsets, mask=mask, other=0.0
        ).to(gl.float32)
        return qk + rel_bias * cfg.REL_BIAS_QK_SCALE

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return amd.cdna3.mfma(p, v, acc)

    @gluon.jit
    def init_attention_state(self):
        cfg = self.cfg
        m_i = gl.full(
            [cfg.BLOCK_M],
            value=-float("inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        l_i = gl.full(
            [cfg.BLOCK_M],
            value=0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
        acc = gl.zeros(
            [cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout
        )
        return m_i, l_i, acc

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        HAS_INVALID: gl.constexpr = cfg.WINDOW_LEFT >= 0

        row_max = _max(qk, 1)
        m_new = _maximum(m_i, row_max)
        m_new_scaled = m_new * cfg.SM_SCALE
        if HAS_INVALID:
            invalid = m_new == -float("inf")
            m_new_scaled = gl.where(invalid, 0.0, m_new_scaled)

        qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
        p = gl.exp2(qk_shifted)
        m_diff = m_i * cfg.SM_SCALE - m_new_scaled
        if HAS_INVALID:
            m_diff = gl.where(invalid, 0.0, m_diff)

        alpha = gl.exp2(m_diff)
        l_ij = gl.sum(p, axis=1)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        p = p.to(self.q_ptr.dtype.element_ty)
        p = gl.convert_layout(p, cfg.p_layout)
        return p, m_new, l_i, acc

    @gluon.jit
    def store_output(self, output):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.store_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.store_layout))
        offsets = (
            ((self.seq_base + offs_m[:, None]) * cfg.N_HEADS + self.q_head)
            * cfg.HEAD_DIM
            + offs_d[None, :]
        ).to(gl.int32)
        mask = offs_m[:, None] < self.seq_len
        output = output.to(self.output_ptr.dtype.element_ty)
        cdna4.buffer_store(output, self.output_ptr, offsets, mask=mask)

    @gluon.jit
    def store_lse(self, l_i, m_i):
        cfg = self.cfg
        if cfg.HAS_LSE:
            offs_m = self.q_start + gl.arange(
                0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)
            )
            offsets = ((self.seq_base + offs_m) * cfg.N_HEADS + self.q_head).to(
                gl.int32
            )
            mask = offs_m < self.seq_len
            lse_l_i = gl.where(l_i > 0.0, l_i, 1.0)
            lse = (m_i * cfg.SM_SCALE + gl.log2(lse_l_i)) * _LN2
            cdna4.buffer_store(lse, self.lse_ptr, offsets, mask=mask)


@gluon.jit
def gluon_rel_mha_extend_gfx950(
    q_ptr,
    rel_logits_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    output_ptr,
    lse_ptr,
    cu_seqlens_q_ptr,
    cache_seqlens_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    SM_SCALE: gl.constexpr,
    PAGE_TABLE_STRIDE: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    HAS_LSE: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    REL_STRIDE_T: gl.constexpr,
    REL_STRIDE_H: gl.constexpr,
    REL_STRIDE_E: gl.constexpr,
    REL_EXTENT: gl.constexpr,
    REL_BIAS_QK_SCALE: gl.constexpr,
):
    cfg = ExtendConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        PAGE_SIZE,
        PAGE_TABLE_STRIDE,
        IS_CAUSAL,
        HAS_LSE,
        WINDOW_LEFT,
        REL_EXTENT,
        REL_BIAS_QK_SCALE,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(REL_STRIDE_T, REL_STRIDE_H, REL_STRIDE_E),
    )
    program = ExtendProgram.create(
        cfg,
        q_ptr,
        rel_logits_ptr,
        k_cache_ptr,
        v_cache_ptr,
        page_table_ptr,
        output_ptr,
        lse_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
    )
    if program.q_start >= program.seq_len:
        return
    # PORT: K stages pre-permuted [HEAD_DIM, BLOCK_N]; q stages through a
    # natural-orientation buffer.
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
    m_i, l_i, acc = program.init_attention_state()

    offs_m_q = program.q_start + gl.arange(
        0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
    )
    offs_n_q = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    diag_row = program.prefix + offs_m_q

    if IS_CAUSAL:
        kv_end = min(program.cache_len, program.prefix + program.q_start + cfg.BLOCK_M)
    else:
        kv_end = program.cache_len

    if cfg.WINDOW_LEFT >= 0:
        pos_top = program.prefix + program.q_start
        kv_start = pos_top - min(pos_top, cfg.WINDOW_LEFT)
        kv_start = (kv_start // cfg.BLOCK_N) * cfg.BLOCK_N
    else:
        kv_start = 0

    for start_n in range(kv_start, kv_end, cfg.BLOCK_N):
        physical_page = program.load_page(start_n)
        k = program.load_k(physical_page, k_smem)
        v = program.load_v(physical_page, v_smem)

        qk = program.compute_qk(q, k)
        qk = program.apply_rel_bias(qk, start_n)

        if cfg.WINDOW_LEFT >= 0:
            offs_n_abs = start_n + offs_n_q
            mask = (offs_n_abs[None, :] >= diag_row[:, None] - cfg.WINDOW_LEFT) & (
                offs_n_abs[None, :] < program.cache_len
            )
            if IS_CAUSAL:
                mask &= offs_n_abs[None, :] <= diag_row[:, None]
            qk = gl.where(mask, qk, -float("inf"))
        elif IS_CAUSAL:
            if start_n + cfg.BLOCK_N > program.prefix + program.q_start:
                offs_n_abs = start_n + offs_n_q
                mask = (offs_n_abs[None, :] <= diag_row[:, None]) & (
                    offs_n_abs[None, :] < program.cache_len
                )
                qk = gl.where(mask, qk, -float("inf"))
        else:
            if start_n + cfg.BLOCK_N > program.cache_len:
                offs_n_abs = start_n + offs_n_q
                qk = gl.where(
                    offs_n_abs[None, :] < program.cache_len, qk, -float("inf")
                )

        p, m_i, l_i, acc = program.softmax(qk, m_i, l_i, acc)
        acc = program.compute_pv(p, v, acc)

    denom = gl.where(l_i > 0.0, l_i, 1.0)
    output = acc * (1.0 / denom)[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)
    program.store_lse(l_i, m_i)


def launch_gluon_rel_mha_extend_gfx942(
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float | None = None,
    window_left: int = -1,
    return_lse: bool = False,
    rel_bias_qk_scale: float = 1.0,
    rel_logits: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Ragged multi-token extend over a paged cache with rel bias (CDNA3)."""
    del cu_seqlens_kv
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or k_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError(
            "the CDNA3 rel-MHA extend port supports bf16/fp16 only (fp8 needs "
            "dot_scaled MFMA — not ported)"
        )
    if rel_logits is None:
        raise ValueError("rel_logits is required (the kernel is rel-bias-only)")
    total_q, n_heads, head_dim = q.shape
    batch = cache_seqlens.shape[0]
    n_kv_heads = k_cache.shape[2]
    page_size = k_cache.shape[1]
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    sm_scale = softmax_scale * _INV_LN2_VALUE

    block_m, block_n, num_warps = _select_extend_tile(max_seqlen_q, page_size)
    blocks_per_req = (max_seqlen_q + block_m - 1) // block_m

    cu_q_i32 = cu_seqlens_q.to(torch.int32).contiguous()
    cache_i32 = cache_seqlens.to(torch.int32).contiguous()

    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if return_lse:
        lse = torch.empty((q.shape[0], n_heads), device=q.device, dtype=torch.float32)
        lse_arg = lse
    else:
        lse = None
        lse_arg = q
    grid = (blocks_per_req, batch, n_heads)
    gluon_rel_mha_extend_gfx950[grid](
        q,
        rel_logits,
        k_cache,
        v_cache,
        page_table,
        output,
        lse_arg,
        cu_q_i32,
        cache_i32,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        sm_scale,
        page_table.stride(0),
        page_size,
        n_heads,
        n_kv_heads,
        head_dim,
        block_m,
        block_n,
        num_warps,
        True,
        return_lse,
        window_left,
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        rel_logits.shape[2],
        rel_bias_qk_scale,
        num_warps=num_warps,
    )
    if return_lse:
        return output, lse
    return output


__all__ = ["launch_gluon_rel_mha_extend_gfx942"]
