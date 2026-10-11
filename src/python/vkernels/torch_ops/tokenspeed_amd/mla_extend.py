"""CDNA3 (gfx942/MI300A) port of TokenSpeed's GFX950 MLA extend kernel.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel-amd`` ``main`` (MIT,
fetched 2026-11): ``ops/gfx950/attention/mla/extend.py`` + the
``_find_seq_idx`` ragged-grid helper from ``mha/extend.py``. This is a
PORT, not a verbatim vendor — the deltas, each following the recipe proven
by ``mha_decode.py`` and ``test_gpu_gluon_cdna3_mfma_tile``:

* Layouts: ``AMDMFMALayout(version=2, instr_shape=[32,32,8],
  transposed=False)`` (upstream: v4 [32,32,16] transposed). The 32-row
  per-warp M packing and ``BLOCK_M = 32 * NUM_WARPS`` are kept.
* LDS budget: MI350-class parts have 128 KB shared memory; MI300A has
  64 KB. Upstream's 4-stage ``BLOCK_N=32`` ring (144 KB) cannot fit, so
  this port runs ``BLOCK_N=16`` with a single stage (36 KB): 4 tiles per
  64-row page, unpipelined correctness-first cut.
* KV staging: upstream's ``async_copy.buffer_load_to_shared`` +
  ``load_shared_relaxed`` pair (which reads both the K-transposed and V
  orientations from one DMA'd buffer) becomes synchronous loads into TWO
  physically-distinct staging buffers — ``[CHUNK, BLOCK_N]`` for the K
  operands (stored pre-permuted) and ``[BLOCK_N, CHUNK]`` for V — the
  MHA-proven pattern; a permuted read of a physically [N, D] buffer
  produced wrong values there.
* The manual software scheduler (``sched_barrier``/``sched_group`` +
  ``wave_uniform_i32`` from the ``sched_barrier.ll`` library) is dropped
  — stock triton's dependency tracking inserts its own barriers; the
  relaxed-load semantics those barriers guarded no longer exist with
  plain ``smem.load``.
* FP8 KV is dropped (needs ``dot_scaled`` MFMA); bf16 q + bf16 cache
  only — the launcher raises otherwise.

Contract (unchanged): absorbed MLA extend — q ``[total_q, H, 512+64]``
already in latent space, paged cache ``[pages, 64, 1, 576]`` already
holding the new tokens, ragged ``cu_seqlens_q``, causal, optional KV
split + fp32 partials + reduce. ``kv_lora_rank=512``, ``rope=64``,
``page_size=64`` fixed.
"""

from __future__ import annotations

import torch

from vkernels.torch_ops.tokenspeed_amd._gluon import gl, gluon

from triton.experimental.gluon.language import amd

# Upstream binds ``gl.amd.cdna4``; CDNA3 exposes identical buffer ops.
_amd_ns = getattr(gl, "amd", None)
cdna4 = getattr(_amd_ns, "cdna3", None) or getattr(_amd_ns, "cdna4", None)

_NUM_WARPS = 4
_BLOCK_M = 128
_PAGE_SIZE = 64
# PORT: gfx942 LDS is 64 KB (MI350: 128 KB) — 16-wide tiles, one stage.
_BLOCK_N = 16
_NUM_STAGES = 1
_CHUNK = 128
_REDUCE_SPLITS = 16
_INV_LN2_VALUE = 1.4426950408889634


@gluon.jit
def _maximum(a, b):
    return gl.maximum(a, b)


@gluon.jit
def _max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, _maximum, keep_dims=keep_dims)


@gluon.jit
def _find_seq_idx(cu_seqlens_q_ptr, target_block, num_seqs, BLOCK_Q: gl.constexpr):
    # Map a ragged global q-block index to its request via binary search
    # (vendored from upstream mha/extend.py).
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(cu_seqlens_q_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_block:
            left = mid + 1
        else:
            right = mid
    return left - 1


# ===-----------------------------------------------------------------------===#
# Config
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class ExtendConfig:
    N_HEADS: gl.constexpr
    HEAD_GROUP: gl.constexpr
    BLOCK_Q: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    PAGE_SIZE: gl.constexpr
    TILES_PER_PAGE: gl.constexpr
    NUM_STAGES: gl.constexpr
    KV_LORA_RANK: gl.constexpr
    ROPE_DIM: gl.constexpr
    CHUNK: gl.constexpr
    NUM_CHUNKS: gl.constexpr
    K_PIECE: gl.constexpr
    NUM_K_PIECES: gl.constexpr
    SM_SCALE: gl.constexpr
    IS_CAUSAL: gl.constexpr
    SPLIT: gl.constexpr
    Q_STRIDE_T: gl.constexpr
    Q_STRIDE_H: gl.constexpr
    O_STRIDE_T: gl.constexpr
    O_STRIDE_H: gl.constexpr
    KV_PAGE_STRIDE: gl.constexpr
    KV_TOKEN_STRIDE: gl.constexpr
    mma_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    p_layout: gl.constexpr
    v_layout: gl.constexpr
    load_layout: gl.constexpr
    load_pe_layout: gl.constexpr
    store_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        N_HEADS,
        HEAD_GROUP,
        BLOCK_Q,
        BLOCK_M,
        BLOCK_N,
        PAGE_SIZE,
        KV_LORA_RANK,
        ROPE_DIM,
        SM_SCALE,
        IS_CAUSAL,
        SPLIT,
        NUM_WARPS,
        KV_DTYPE,
        Q_STRIDE_T,
        Q_STRIDE_H,
        O_STRIDE_T,
        O_STRIDE_H,
        KV_PAGE_STRIDE,
        KV_TOKEN_STRIDE,
    ):
        assert BLOCK_M == 32 * NUM_WARPS
        assert BLOCK_Q * HEAD_GROUP <= BLOCK_M
        assert KV_LORA_RANK == 4 * _CHUNK
        assert PAGE_SIZE % BLOCK_N == 0

        # PORT: v2 [32, 32, 8] non-transposed (upstream: v4 [32, 32, 16]
        # transposed). Operand k-widths follow the 128-bit-load rule.
        mma_layout = gl.amd.AMDMFMALayout(
            version=2,
            instr_shape=[32, 32, 8],
            transposed=False,
            warps_per_cta=[NUM_WARPS, 1],
        )
        qk_kw = 8
        pv_kw = 4
        k_layout = gl.DotOperandLayout(1, mma_layout, k_width=qk_kw)
        v_layout = gl.DotOperandLayout(1, mma_layout, k_width=pv_kw)
        vec = 8
        load_layout = gl.BlockedLayout(
            [1, vec], [64 // (_CHUNK // vec), _CHUNK // vec], [NUM_WARPS, 1], [1, 0]
        )
        load_pe_layout = gl.BlockedLayout(
            [1, vec],
            [64 // (ROPE_DIM // vec), ROPE_DIM // vec],
            [NUM_WARPS, 1],
            [1, 0],
        )

        self.N_HEADS = gl.constexpr(N_HEADS)
        self.HEAD_GROUP = gl.constexpr(HEAD_GROUP)
        self.BLOCK_Q = gl.constexpr(BLOCK_Q)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.PAGE_SIZE = gl.constexpr(PAGE_SIZE)
        self.TILES_PER_PAGE = gl.constexpr(PAGE_SIZE // BLOCK_N)
        self.NUM_STAGES = gl.constexpr(_NUM_STAGES)
        self.KV_LORA_RANK = gl.constexpr(KV_LORA_RANK)
        self.ROPE_DIM = gl.constexpr(ROPE_DIM)
        self.CHUNK = gl.constexpr(_CHUNK)
        self.NUM_CHUNKS = gl.constexpr(KV_LORA_RANK // _CHUNK)
        k_piece = _CHUNK
        self.K_PIECE = gl.constexpr(k_piece)
        self.NUM_K_PIECES = gl.constexpr(KV_LORA_RANK // k_piece + 1)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.IS_CAUSAL = gl.constexpr(IS_CAUSAL)
        self.SPLIT = gl.constexpr(SPLIT)
        self.Q_STRIDE_T = gl.constexpr(Q_STRIDE_T)
        self.Q_STRIDE_H = gl.constexpr(Q_STRIDE_H)
        self.O_STRIDE_T = gl.constexpr(O_STRIDE_T)
        self.O_STRIDE_H = gl.constexpr(O_STRIDE_H)
        self.KV_PAGE_STRIDE = gl.constexpr(KV_PAGE_STRIDE)
        self.KV_TOKEN_STRIDE = gl.constexpr(KV_TOKEN_STRIDE)
        self.mma_layout = gl.constexpr(mma_layout)
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, mma_layout, k_width=qk_kw))
        self.k_layout = gl.constexpr(k_layout)
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, mma_layout, k_width=pv_kw))
        self.v_layout = gl.constexpr(v_layout)
        self.load_layout = gl.constexpr(load_layout)
        self.load_pe_layout = gl.constexpr(load_pe_layout)
        self.store_layout = gl.constexpr(
            gl.BlockedLayout([1, 8], [32, 2], [NUM_WARPS, 1], [0, 1])
        )


# ===-----------------------------------------------------------------------===#
# Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class ExtendProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    kv_ptr: gl.tensor
    page_table_ptr: gl.tensor
    batch: gl.tensor
    page_table_stride: gl.constexpr
    head_base: gl.tensor
    q_pos_base: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    prefix: gl.tensor
    cache_len: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        kv_ptr,
        page_table_ptr,
        batch,
        page_table_stride,
        head_base,
        q_pos_base,
        seq_base,
        seq_len,
        prefix,
        cache_len,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.kv_ptr = kv_ptr
        self.page_table_ptr = page_table_ptr
        self.batch = batch
        self.page_table_stride = gl.constexpr(page_table_stride)
        self.head_base = head_base
        self.q_pos_base = q_pos_base
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.prefix = prefix
        self.cache_len = cache_len

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        kv_ptr,
        page_table_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        num_seqs,
        PAGE_TABLE_STRIDE,
    ):
        pid_q = gl.program_id(0)
        batch = _find_seq_idx(cu_seqlens_q_ptr, pid_q, num_seqs, cfg.BLOCK_Q)
        seq_base = gl.load(cu_seqlens_q_ptr + batch)
        seq_len = gl.load(cu_seqlens_q_ptr + batch + 1) - seq_base
        q_pos_base = (pid_q - (seq_base // cfg.BLOCK_Q + batch)) * cfg.BLOCK_Q
        cache_len = gl.load(cache_seqlens_ptr + batch)
        return ExtendProgram(
            cfg,
            q_ptr,
            kv_ptr,
            page_table_ptr,
            batch,
            PAGE_TABLE_STRIDE,
            gl.program_id(1) * cfg.HEAD_GROUP,
            q_pos_base,
            seq_base,
            seq_len,
            cache_len - seq_len,
            cache_len,
        )

    @gluon.jit
    def rows(self, layout: gl.constexpr):
        # Row m packs (query position m // HEAD_GROUP, head m % HEAD_GROUP).
        cfg = self.cfg
        offs_m = gl.arange(0, cfg.BLOCK_M, layout=layout)
        q_idx = offs_m // cfg.HEAD_GROUP
        q_pos = self.q_pos_base + q_idx
        head = self.head_base + offs_m % cfg.HEAD_GROUP
        valid = (q_idx < cfg.BLOCK_Q) & (q_pos < self.seq_len)
        return q_pos, head, valid

    @gluon.jit
    def load_q(self, col: gl.constexpr, width: gl.constexpr):
        # PORT: upstream loads q straight into the MFMA operand layout via a
        # cdna4 buffer op; the CDNA3 alias keeps that pattern (q is too large
        # to stage through LDS at BLOCK_M x CHUNK).
        cfg = self.cfg
        q_pos, head, valid = self.rows(gl.SliceLayout(1, cfg.q_layout))
        offs_d = col + gl.arange(0, width, layout=gl.SliceLayout(0, cfg.q_layout))
        offsets = (
            (self.seq_base + q_pos)[:, None] * cfg.Q_STRIDE_T
            + head[:, None] * cfg.Q_STRIDE_H
            + offs_d[None, :]
        ).to(gl.int32)
        return cdna4.buffer_load(self.q_ptr, offsets, mask=valid[:, None], other=0.0)

    @gluon.jit
    def load_q_pieces(self):
        cfg = self.cfg
        return (
            self.load_q(0 * cfg.CHUNK, cfg.CHUNK),
            self.load_q(1 * cfg.CHUNK, cfg.CHUNK),
            self.load_q(2 * cfg.CHUNK, cfg.CHUNK),
            self.load_q(3 * cfg.CHUNK, cfg.CHUNK),
            self.load_q(cfg.KV_LORA_RANK, cfg.ROPE_DIM),
        )

    @gluon.jit
    def load_page(self, tile, tile_end):
        return gl.load(
            self.page_table_ptr
            + self.batch * self.page_table_stride
            + tile // self.cfg.TILES_PER_PAGE,
            mask=tile < tile_end,
            other=0,
        )

    # --- PORT: the async_copy issue/wait pipeline becomes synchronous dual-
    # orientation staging. K reads need [CHUNK, N]-oriented operands, V reads
    # need [N, CHUNK]; the two live in physically distinct buffers (a
    # permuted read of a physical [N, D] buffer gave wrong values in the MHA
    # port). Each tile's registers load once and store twice.

    @gluon.jit
    def issue_tile(self, k_smem, v_smem, pe_smem, page, tile, tile_end):
        cfg = self.cfg
        num_valid = gl.where(tile < tile_end, self.cache_len - tile * cfg.BLOCK_N, 0)
        row = (tile % cfg.TILES_PER_PAGE) * cfg.BLOCK_N
        base = (
            self.kv_ptr
            + page.to(gl.int64) * cfg.KV_PAGE_STRIDE
            + row * cfg.KV_TOKEN_STRIDE
        )
        for c in gl.static_range(cfg.NUM_CHUNKS):
            offs_n = gl.arange(
                0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout)
            )
            offs_d = gl.arange(
                0, cfg.CHUNK, layout=gl.SliceLayout(0, cfg.load_layout)
            )
            offsets = offs_n[:, None] * cfg.KV_TOKEN_STRIDE + (
                c * cfg.CHUNK + offs_d
            )[None, :]
            tile_v = gl.load(
                base + offsets, mask=(offs_n < num_valid)[:, None], other=0.0
            )
            v_smem.index(c).store(tile_v)
            k_smem.index(c).store(tile_v.permute([1, 0]))
        offs_n_pe = gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_pe_layout)
        )
        offs_d_pe = gl.arange(
            0, cfg.ROPE_DIM, layout=gl.SliceLayout(0, cfg.load_pe_layout)
        )
        offsets_pe = offs_n_pe[:, None] * cfg.KV_TOKEN_STRIDE + (
            cfg.KV_LORA_RANK + offs_d_pe
        )[None, :]
        tile_pe = gl.load(
            base + offsets_pe, mask=(offs_n_pe < num_valid)[:, None], other=0.0
        )
        pe_smem.store(tile_pe.permute([1, 0]))

    @gluon.jit
    def dot(self, a, b, acc):
        return amd.cdna3.mfma(a, b, acc)

    @gluon.jit
    def load_k(self, k_smem, pe_smem, piece: gl.constexpr):
        cfg = self.cfg
        if piece + 1 < cfg.NUM_K_PIECES:
            return k_smem.index(piece).load(cfg.k_layout)
        else:
            return pe_smem.load(cfg.k_layout)

    @gluon.jit
    def load_v(self, v_smem, chunk: gl.constexpr):
        return v_smem.index(chunk).load(self.cfg.v_layout)

    @gluon.jit
    def qk(self, q, k_smem, pe_smem):
        cfg = self.cfg
        qk = gl.zeros([cfg.BLOCK_M, cfg.BLOCK_N], gl.float32, layout=cfg.mma_layout)
        k = self.load_k(k_smem, pe_smem, 0)
        for j in gl.static_range(cfg.NUM_K_PIECES):
            if j + 1 < cfg.NUM_K_PIECES:
                k_next = self.load_k(k_smem, pe_smem, j + 1)
            qk = self.dot(q[j], k, qk)
            if j + 1 < cfg.NUM_K_PIECES:
                k = k_next
        return qk

    @gluon.jit
    def mask_scores(self, qk, tile, mask_from, diag):
        cfg = self.cfg
        start_n = tile * cfg.BLOCK_N
        if start_n + cfg.BLOCK_N > mask_from:
            col = start_n + gl.arange(
                0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.mma_layout)
            )
            visible = col[None, :] < self.cache_len
            if cfg.IS_CAUSAL:
                visible = visible & (col[None, :] <= diag[:, None])
            qk = gl.where(visible, qk, -float("inf"))
        return qk

    @gluon.jit
    def softmax(self, qk, m_i, l_i):
        cfg = self.cfg
        m_tile = _maximum(m_i, _max(qk, 1) * cfg.SM_SCALE)
        m_new = gl.where(m_tile - m_i > 8.0, m_tile, m_i)
        m_use = gl.where(m_new == -float("inf"), 0.0, m_new)
        p = gl.exp2(qk * cfg.SM_SCALE - m_use[:, None])
        alpha = gl.exp2(m_i - m_use)
        l_i = l_i * alpha + gl.sum(p, axis=1)
        p = gl.convert_layout(p.to(self.kv_ptr.dtype.element_ty), cfg.p_layout)
        return p, alpha, m_new, l_i

    @gluon.jit
    def pv(self, p, v, v_smem, acc0, acc1, acc2, acc3):
        acc0 = self.dot(p, v, acc0)
        v1 = self.load_v(v_smem, 1)
        acc1 = self.dot(p, v1, acc1)
        v2 = self.load_v(v_smem, 2)
        acc2 = self.dot(p, v2, acc2)
        v3 = self.load_v(v_smem, 3)
        acc3 = self.dot(p, v3, acc3)
        return acc0, acc1, acc2, acc3


@gluon.jit
def _rescale(acc, alpha):
    # PORT: upstream's pack=64 wave-branch rescale is a perf optimization;
    # the plain row broadcast is semantically identical.
    return acc * alpha[:, None]


@gluon.jit
def _store_chunk(out_ptr, o, program, col: gl.constexpr, stride_t, stride_h):
    cfg = program.cfg
    layout: gl.constexpr = o.type.layout
    q_pos, head, valid = program.rows(gl.SliceLayout(1, layout))
    offs_d = col + gl.arange(0, cfg.CHUNK, layout=gl.SliceLayout(0, layout))
    offsets = (
        (program.seq_base + q_pos)[:, None] * stride_t
        + head[:, None] * stride_h
        + offs_d[None, :]
    ).to(gl.int32)
    cdna4.buffer_store(o, out_ptr, offsets, mask=valid[:, None])


@gluon.jit
def _finish_chunk(out_ptr, acc, inv_l, program, col: gl.constexpr):
    cfg = program.cfg
    o = acc * inv_l[:, None]
    if cfg.SPLIT:
        _store_chunk(
            out_ptr,
            o,
            program,
            col,
            cfg.N_HEADS * cfg.KV_LORA_RANK,
            cfg.KV_LORA_RANK,
        )
    else:
        o = gl.convert_layout(o.to(out_ptr.dtype.element_ty), cfg.store_layout)
        _store_chunk(out_ptr, o, program, col, cfg.O_STRIDE_T, cfg.O_STRIDE_H)


# ===-----------------------------------------------------------------------===#
# Kernels
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_mla_extend_gfx950(
    q_ptr,
    kv_ptr,
    page_table_ptr,
    out_ptr,
    lse_ptr,
    cu_seqlens_q_ptr,
    cache_seqlens_ptr,
    num_seqs,
    split_stride,
    PAGE_TABLE_STRIDE: gl.constexpr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    O_STRIDE_T: gl.constexpr,
    O_STRIDE_H: gl.constexpr,
    KV_PAGE_STRIDE: gl.constexpr,
    KV_TOKEN_STRIDE: gl.constexpr,
    SM_SCALE: gl.constexpr,
    N_HEADS: gl.constexpr,
    HEAD_GROUP: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    KV_LORA_RANK: gl.constexpr,
    ROPE_DIM: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    SPLIT: gl.constexpr,
):
    cfg = ExtendConfig(
        N_HEADS,
        HEAD_GROUP,
        BLOCK_Q,
        BLOCK_M,
        BLOCK_N,
        PAGE_SIZE,
        KV_LORA_RANK,
        ROPE_DIM,
        SM_SCALE,
        IS_CAUSAL,
        SPLIT,
        NUM_WARPS,
        kv_ptr.dtype.element_ty,
        Q_STRIDE_T,
        Q_STRIDE_H,
        O_STRIDE_T,
        O_STRIDE_H,
        KV_PAGE_STRIDE,
        KV_TOKEN_STRIDE,
    )
    program = ExtendProgram.create(
        cfg,
        q_ptr,
        kv_ptr,
        page_table_ptr,
        cu_seqlens_q_ptr,
        cache_seqlens_ptr,
        num_seqs,
        PAGE_TABLE_STRIDE,
    )
    if program.q_pos_base >= program.seq_len:
        return

    if cfg.IS_CAUSAL:
        kv_end = min(
            program.cache_len, program.prefix + program.q_pos_base + cfg.BLOCK_Q
        )
    else:
        kv_end = program.cache_len
    num_tiles = gl.cdiv(kv_end, cfg.BLOCK_N)
    if cfg.SPLIT:
        split = gl.program_id(2)
        tiles_per_split = gl.cdiv(num_tiles, gl.num_programs(2))
        tile_begin = split * tiles_per_split
        tile_end = min(tile_begin + tiles_per_split, num_tiles)
    else:
        tile_begin = 0
        tile_end = num_tiles

    kv_dtype: gl.constexpr = kv_ptr.dtype.element_ty
    # PORT: dual-orientation staging, single stage (see module docstring).
    k_smem = gl.allocate_shared_memory(
        kv_dtype,
        [cfg.NUM_CHUNKS, cfg.CHUNK, cfg.BLOCK_N],
        gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1]),
    )
    v_smem = gl.allocate_shared_memory(
        kv_dtype,
        [cfg.NUM_CHUNKS, cfg.BLOCK_N, cfg.CHUNK],
        gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1]),
    )
    pe_smem = gl.allocate_shared_memory(
        kv_dtype,
        [cfg.ROPE_DIM, cfg.BLOCK_N],
        gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1]),
    )

    row_layout: gl.constexpr = gl.SliceLayout(1, cfg.mma_layout)
    m_i = gl.full([cfg.BLOCK_M], -float("inf"), gl.float32, layout=row_layout)
    l_i = gl.zeros([cfg.BLOCK_M], gl.float32, layout=row_layout)
    acc0 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    acc1 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    acc2 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    acc3 = gl.zeros([cfg.BLOCK_M, cfg.CHUNK], gl.float32, layout=cfg.mma_layout)
    q_pos, _, _ = program.rows(row_layout)
    diag = program.prefix + q_pos
    if cfg.IS_CAUSAL:
        mask_from = min(program.prefix + program.q_pos_base, program.cache_len)
    else:
        mask_from = program.cache_len

    q = program.load_q_pieces()

    # PORT: single-stage synchronous loop — load tile, Q@K^T, softmax, P@V;
    # the upstream 4-stage DMA pipeline degenerates to this on 64 KB LDS.
    for tile in range(tile_begin, tile_end):
        page = program.load_page(tile, tile_end)
        program.issue_tile(k_smem, v_smem, pe_smem, page, tile, tile_end)
        qk = program.qk(q, k_smem, pe_smem)
        qk = program.mask_scores(qk, tile, mask_from, diag)
        p, alpha, m_i, l_i = program.softmax(qk, m_i, l_i)
        acc0 = _rescale(acc0, alpha)
        acc1 = _rescale(acc1, alpha)
        acc2 = _rescale(acc2, alpha)
        acc3 = _rescale(acc3, alpha)
        v = program.load_v(v_smem, 0)
        acc0, acc1, acc2, acc3 = program.pv(p, v, v_smem, acc0, acc1, acc2, acc3)

    has_kv = l_i > 0.0
    inv_l = gl.where(has_kv, 1.0 / gl.where(has_kv, l_i, 1.0), 0.0)
    if cfg.SPLIT:
        split_offset = gl.program_id(2).to(gl.int64) * split_stride
        out_ptr = out_ptr + split_offset * cfg.KV_LORA_RANK
        lse_ptr = lse_ptr + split_offset
    _finish_chunk(out_ptr, acc0, inv_l, program, 0 * cfg.CHUNK)
    _finish_chunk(out_ptr, acc1, inv_l, program, 1 * cfg.CHUNK)
    _finish_chunk(out_ptr, acc2, inv_l, program, 2 * cfg.CHUNK)
    _finish_chunk(out_ptr, acc3, inv_l, program, 3 * cfg.CHUNK)
    if cfg.SPLIT:
        _, head, valid = program.rows(row_layout)
        lse = gl.where(has_kv, m_i + gl.log2(gl.where(has_kv, l_i, 1.0)), -float("inf"))
        offsets = ((program.seq_base + q_pos) * cfg.N_HEADS + head).to(gl.int32)
        cdna4.buffer_store(lse, lse_ptr, offsets, mask=valid)


@gluon.jit
def gluon_mla_extend_reduce_gfx950(
    part_o_ptr,
    part_lse_ptr,
    out_ptr,
    num_splits,
    split_stride,
    N_HEADS: gl.constexpr,
    O_STRIDE_T: gl.constexpr,
    O_STRIDE_H: gl.constexpr,
    KV_LORA_RANK: gl.constexpr,
    REDUCE_SPLITS: gl.constexpr,
):
    # Merge the per-split partials of one (token, head) row (verbatim).
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
    split_layout: gl.constexpr = gl.SliceLayout(1, layout)
    token = gl.program_id(0)
    head = gl.program_id(1)
    row = token * N_HEADS + head
    offs_s = gl.arange(0, REDUCE_SPLITS, layout=split_layout)
    offs_d = gl.arange(0, KV_LORA_RANK, layout=gl.SliceLayout(0, layout))
    m = gl.full([REDUCE_SPLITS], -float("inf"), gl.float32, layout=split_layout)
    for start in range(0, num_splits, REDUCE_SPLITS):
        split = start + offs_s
        lse = gl.load(
            part_lse_ptr + split * split_stride + row,
            mask=split < num_splits,
            other=-float("inf"),
        )
        m = gl.maximum(m, lse)
    m_max = gl.max(m, 0)
    m_max = gl.where(m_max == -float("inf"), 0.0, m_max)
    acc = gl.zeros([KV_LORA_RANK], gl.float32, layout=gl.SliceLayout(0, layout))
    l_i = gl.zeros([REDUCE_SPLITS], gl.float32, layout=split_layout)
    for start in range(0, num_splits, REDUCE_SPLITS):
        split = start + offs_s
        lse = gl.load(
            part_lse_ptr + split * split_stride + row,
            mask=split < num_splits,
            other=-float("inf"),
        )
        weight = gl.exp2(lse - m_max)
        offsets = (split * split_stride + row).to(gl.int64) * KV_LORA_RANK
        part = gl.load(
            part_o_ptr + offsets[:, None] + offs_d[None, :],
            mask=(lse != -float("inf"))[:, None],
            other=0.0,
        )
        acc += gl.sum(part * weight[:, None], 0)
        l_i += weight
    l_sum = gl.sum(l_i, 0)
    out = acc / gl.where(l_sum > 0.0, l_sum, 1.0)
    gl.store(
        out_ptr + token * O_STRIDE_T + head * O_STRIDE_H + offs_d,
        out.to(out_ptr.dtype.element_ty),
    )


# ===-----------------------------------------------------------------------===#
# Host
# ===-----------------------------------------------------------------------===#


def _select_head_group(num_heads: int) -> int:
    for group in (16, 8, 4, 2):
        if num_heads % group == 0 and group * 8 <= _BLOCK_M:
            return group
    return 1


def _select_kv_splits(base_ctas: int, num_tiles: int) -> int:
    sm_count = 228  # MI300A; queried lazily in the launcher path anyway
    wave = sm_count * 2
    splits = (wave + base_ctas - 1) // base_ctas
    splits = min(splits, max(1, num_tiles // 4), 8)
    return splits if splits > 1 else 1


def launch_gluon_mla_extend_gfx942(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    softmax_scale: float,
    *,
    is_causal: bool,
    logit_cap: float,
    return_lse: bool,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Absorbed MLA extend over a paged latent cache (CDNA3 port).

    bf16 q + bf16 cache only (the FP8 KV paths need dot_scaled MFMA — not
    ported); see the module docstring for the full contract.
    """
    del cu_seqlens_kv, max_seqlen_q, qk_nope_head_dim
    if not is_causal:
        raise NotImplementedError("the CDNA3 MLA extend port requires causal attention")
    if logit_cap != 0.0:
        raise NotImplementedError("the CDNA3 MLA extend port does not support logit_cap")
    if return_lse:
        raise NotImplementedError("the CDNA3 MLA extend port does not return LSE")
    if kv_lora_rank != 512 or qk_rope_head_dim != 64:
        raise NotImplementedError(
            f"requires kv_lora_rank=512 and qk_rope_head_dim=64, got "
            f"{kv_lora_rank} and {qk_rope_head_dim}"
        )
    if q.dtype is not torch.bfloat16 or kv_cache.dtype is not torch.bfloat16:
        raise TypeError(
            "the CDNA3 MLA extend port supports bf16 q + bf16 cache only, got "
            f"q={q.dtype}, kv_cache={kv_cache.dtype}"
        )
    head_dim = kv_lora_rank + qk_rope_head_dim
    if q.ndim != 3 or q.shape[2] != head_dim or q.stride(2) != 1:
        raise ValueError(f"q must be [total_q, num_heads, {head_dim}], got {q.shape}")
    if kv_cache.ndim != 4 or kv_cache.shape[2:] != (1, head_dim):
        raise ValueError(
            f"kv_cache must be [num_pages, page_size, 1, {head_dim}], got "
            f"{tuple(kv_cache.shape)}"
        )
    if not kv_cache.is_contiguous():
        raise ValueError("kv_cache must be contiguous")
    page_size = kv_cache.shape[1]
    if page_size != _PAGE_SIZE:
        raise NotImplementedError(f"supports page size {_PAGE_SIZE}, got {page_size}")
    for name, t in (
        ("page_table", page_table),
        ("cache_seqlens", cache_seqlens),
        ("cu_seqlens_q", cu_seqlens_q),
    ):
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")

    total_q, num_heads, _ = q.shape
    batch = cache_seqlens.shape[0]
    out_shape = (total_q, num_heads, kv_lora_rank)
    out_dtype = torch.bfloat16
    if out is None:
        out = torch.empty(out_shape, dtype=out_dtype, device=q.device)
    elif out.shape != out_shape or out.dtype != out_dtype or out.stride(2) != 1:
        raise ValueError(
            f"out must be {out_shape} {out_dtype} with unit last stride, got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    if total_q == 0:
        return out

    head_group = _select_head_group(num_heads)
    block_q = _BLOCK_M // head_group
    num_head_groups = num_heads // head_group
    num_q_blocks = (total_q - 1) // block_q + batch
    block_n = _BLOCK_N
    num_tiles = (max_seqlen_k + block_n - 1) // block_n
    num_splits = _select_kv_splits(num_q_blocks * num_head_groups, num_tiles)
    split = num_splits > 1
    if split:
        split_stride = total_q * num_heads
        part_o = torch.empty(
            (num_splits, total_q, num_heads, kv_lora_rank),
            dtype=torch.float32,
            device=q.device,
        )
        part_lse = torch.empty(
            (num_splits, total_q, num_heads), dtype=torch.float32, device=q.device
        )
        attn_out, attn_lse = part_o, part_lse
    else:
        split_stride = 0
        attn_out, attn_lse = out, out

    gluon_mla_extend_gfx950[(num_q_blocks, num_head_groups, num_splits)](
        q,
        kv_cache,
        page_table,
        attn_out,
        attn_lse,
        cu_seqlens_q,
        cache_seqlens,
        batch,
        split_stride,
        PAGE_TABLE_STRIDE=page_table.stride(0),
        Q_STRIDE_T=q.stride(0),
        Q_STRIDE_H=q.stride(1),
        O_STRIDE_T=out.stride(0),
        O_STRIDE_H=out.stride(1),
        KV_PAGE_STRIDE=kv_cache.stride(0),
        KV_TOKEN_STRIDE=kv_cache.stride(1),
        SM_SCALE=softmax_scale * _INV_LN2_VALUE,
        N_HEADS=num_heads,
        HEAD_GROUP=head_group,
        BLOCK_Q=block_q,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=block_n,
        PAGE_SIZE=_PAGE_SIZE,
        KV_LORA_RANK=kv_lora_rank,
        ROPE_DIM=qk_rope_head_dim,
        NUM_WARPS=_NUM_WARPS,
        IS_CAUSAL=is_causal,
        SPLIT=split,
        num_warps=_NUM_WARPS,
    )
    if split:
        gluon_mla_extend_reduce_gfx950[(total_q, num_heads)](
            part_o,
            part_lse,
            out,
            num_splits,
            split_stride,
            N_HEADS=num_heads,
            O_STRIDE_T=out.stride(0),
            O_STRIDE_H=out.stride(1),
            KV_LORA_RANK=kv_lora_rank,
            REDUCE_SPLITS=_REDUCE_SPLITS,
            num_warps=4,
        )
    return out


__all__ = ["launch_gluon_mla_extend_gfx942"]
