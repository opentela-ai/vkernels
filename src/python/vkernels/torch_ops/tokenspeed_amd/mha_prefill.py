"""CDNA3 (gfx942/MI300A) port of TokenSpeed's GFX950 MHA prefill kernel.

Source: lightseekorg/tokenspeed ``tokenspeed-kernel-amd`` ``main`` (MIT,
fetched 2026-11): ``ops/gfx950/attention/mha/prefill.py``. This is a PORT,
not a verbatim vendor — the deltas, following the recipes proven by
``mha_decode.py`` / ``mla_extend.py`` / ``test_gpu_gluon_cdna3_mfma_tile``:

* Layouts: ``AMDMFMALayout(version=2, instr_shape=[32,32,8],
  transposed=False)`` with NUM_WARPS=4 tiling (upstream: v4 [32,32,16]
  transposed). ``BLOCK_M=128`` and the persistent ``ProgramScheduler``
  (atomic work-stealing with q-cycle swizzle + dynamic skip-softmax
  scheduling) are kept verbatim.
* KV staging: the ``async_copy.buffer_load_to_shared`` + permuted shared
  read pair becomes synchronous loads into dual physical buffers — V in
  natural ``[BLOCK_N, HEAD_DIM]``, K pre-permuted into
  ``[HEAD_DIM, BLOCK_N]`` — the orientation lesson from the earlier ports
  (a permuted read of a physical [N, D] buffer produced wrong values).
* The K/V load pipelining (wait_group(0/1) interleavings, DEFER_V_LOAD)
  degenerates to synchronous order; DEFER_V_LOAD still gates whether V
  stages before or after the skip decision, so the skip-softmax traffic
  saving survives.
* FP8 dropped (bf16/fp16 only; the launcher raises). ``NUM_XCDS`` stays 8
  — it only spreads persistent pids, correctness-neutral on MI300A's 6.

Contract (unchanged): varlen causal MHA prefill over cu_seqlens with
optional sliding window, sinks, LSE, and skip-softmax sparsity
(``skip_softmax_threshold``); q/k/v [total_tokens, H, D] dense.
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

_INV_LN2 = 1.4426950408889634
_INV_LN2_VALUE = 1.4426950408889634
_LN2 = 0.6931471805599453


@gluon.jit
def _maximum(a, b):
    return gl.maximum(a, b)


@gluon.jit
def _max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, _maximum, keep_dims=keep_dims)


@gluon.constexpr_function
def _attention_layouts(head_dim, block_n, dtype, num_warps):
    """CDNA3 layouts (replaces upstream's v4/DMA-derived set)."""
    mfma = gl.amd.AMDMFMALayout(
        version=2,
        instr_shape=[32, 32, 8],
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
    N_HEADS: gl.constexpr
    N_KV_HEADS: gl.constexpr
    HEAD_DIM: gl.constexpr
    SM_SCALE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_WARPS: gl.constexpr
    BATCH_SIZE: gl.constexpr
    HAS_SINK: gl.constexpr
    HAS_LSE: gl.constexpr
    WINDOW_LEFT: gl.constexpr
    NUM_XCDS: gl.constexpr
    NUM_BLOCKS: gl.constexpr
    IS_FP8: gl.constexpr
    ENABLE_SKIP_SOFTMAX: gl.constexpr
    DEFER_V_LOAD: gl.constexpr
    DYNAMIC_SCHED: gl.constexpr
    q_strides: InputStrides
    k_strides: InputStrides
    v_strides: InputStrides
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
        BATCH_SIZE,
        HAS_SINK,
        HAS_LSE,
        WINDOW_LEFT,
        IS_FP8,
        ENABLE_SKIP_SOFTMAX,
        DEFER_V_LOAD,
        DYNAMIC_SCHED,
        KV_DTYPE,
        q_strides,
        k_strides,
        v_strides,
    ):
        assert HEAD_DIM in (64, 128)
        assert NUM_WARPS == 2

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
        ) = _attention_layouts(
            HEAD_DIM,
            BLOCK_N,
            KV_DTYPE,
            num_warps=NUM_WARPS,
        )
        self.N_HEADS = gl.constexpr(N_HEADS)
        self.N_KV_HEADS = gl.constexpr(N_KV_HEADS)
        self.HEAD_DIM = gl.constexpr(HEAD_DIM)
        self.SM_SCALE = gl.constexpr(SM_SCALE)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.NUM_WARPS = gl.constexpr(NUM_WARPS)
        self.BATCH_SIZE = gl.constexpr(BATCH_SIZE)
        self.HAS_SINK = gl.constexpr(HAS_SINK)
        self.HAS_LSE = gl.constexpr(HAS_LSE)
        self.WINDOW_LEFT = gl.constexpr(WINDOW_LEFT)
        self.NUM_XCDS = gl.constexpr(8)
        self.NUM_BLOCKS = gl.constexpr(512)
        self.IS_FP8 = gl.constexpr(IS_FP8)
        self.ENABLE_SKIP_SOFTMAX = gl.constexpr(ENABLE_SKIP_SOFTMAX)
        self.DEFER_V_LOAD = gl.constexpr(DEFER_V_LOAD)
        self.DYNAMIC_SCHED = gl.constexpr(DYNAMIC_SCHED)
        self.q_strides = q_strides
        self.k_strides = k_strides
        self.v_strides = v_strides
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
# Kernel Program
# ===-----------------------------------------------------------------------===#


@gluon.aggregate
class AttentionProgram:
    cfg: gl.constexpr
    q_ptr: gl.tensor
    k_ptr: gl.tensor
    v_ptr: gl.tensor
    output_ptr: gl.tensor
    sink_ptr: gl.tensor
    lse_ptr: gl.tensor
    seq_base: gl.tensor
    seq_len: gl.tensor
    q_start: gl.tensor
    q_head: gl.tensor
    kv_head: gl.tensor
    log2_threshold: gl.tensor

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        seq_base,
        seq_len,
        q_start,
        q_head,
        kv_head,
        log2_threshold,
    ):
        self.cfg = gl.constexpr(cfg)
        self.q_ptr = q_ptr
        self.k_ptr = k_ptr
        self.v_ptr = v_ptr
        self.output_ptr = output_ptr
        self.sink_ptr = sink_ptr
        self.lse_ptr = lse_ptr
        self.seq_base = seq_base
        self.seq_len = seq_len
        self.q_start = q_start
        self.q_head = q_head
        self.kv_head = kv_head
        self.log2_threshold = log2_threshold

    @gluon.jit
    def initialize_from_state(
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        seq_base,
        seq_len,
        query_block,
        q_head,
        log2_threshold,
    ):
        kv_head = q_head // (cfg.N_HEADS // cfg.N_KV_HEADS)
        q_start = query_block * cfg.BLOCK_M
        return AttentionProgram(
            cfg,
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            sink_ptr,
            lse_ptr,
            seq_base,
            seq_len,
            q_start,
            q_head,
            kv_head,
            log2_threshold,
        )

    @gluon.jit
    def load_q(self, other=None):
        cfg = self.cfg
        offs_m = self.q_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.q_layout))
        offsets = cfg.q_strides.offsets(
            self.seq_base + offs_m[:, None], self.q_head, offs_d[None, :]
        )
        mask = offs_m[:, None] < self.seq_len
        if other is None:
            return cdna4.buffer_load(self.q_ptr, offsets, mask=mask)
        return cdna4.buffer_load(self.q_ptr, offsets, mask=mask, other=other)

    @gluon.jit
    def make_k_offsets(self, kv_start):
        cfg = self.cfg
        offs_n = kv_start + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.k_strides.offsets(
            self.seq_base + offs_n[:, None], self.kv_head, offs_d[None, :]
        )
        return offsets, offs_n

    @gluon.jit
    def make_v_offsets(self, kv_start):
        cfg = self.cfg
        offs_n = kv_start + gl.arange(
            0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.load_layout)
        )
        offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.load_layout))
        offsets = cfg.v_strides.offsets(
            self.seq_base + offs_n[:, None], self.kv_head, offs_d[None, :]
        )
        return offsets

    @gluon.jit
    def update_k_offsets(self, offsets):
        cfg = self.cfg
        return offsets + cfg.BLOCK_N * cfg.k_strides.stride_t

    @gluon.jit
    def update_v_offsets(self, offsets):
        cfg = self.cfg
        return offsets + cfg.BLOCK_N * cfg.v_strides.stride_t

    # --- PORT: async issue/wait becomes synchronous dual-orientation staging.
    # The tile loads once into registers and stores twice: natural [N, D] for
    # the V operand and pre-permuted [D, N] for the K operand.

    @gluon.jit
    def load_k(self, offsets, k_smem, mask=None, other=None):
        if mask is None:
            tile = gl.load(self.k_ptr + offsets)
        elif other is None:
            tile = gl.load(self.k_ptr + offsets, mask=mask)
        else:
            tile = gl.load(self.k_ptr + offsets, mask=mask, other=other)
        k_smem.store(tile.permute([1, 0]))

    @gluon.jit
    def load_v(self, offsets, v_smem, mask=None, other=None):
        if mask is None:
            tile = gl.load(self.v_ptr + offsets)
        elif other is None:
            tile = gl.load(self.v_ptr + offsets, mask=mask)
        else:
            tile = gl.load(self.v_ptr + offsets, mask=mask, other=other)
        v_smem.store(tile)

    @gluon.jit
    def shared_load_k(self, k_smem):
        cfg = self.cfg
        return k_smem.load(cfg.k_layout)

    @gluon.jit
    def shared_load_v(self, v_smem):
        cfg = self.cfg
        return v_smem.load(cfg.v_layout)

    @gluon.jit
    def compute_qk(self, q, k):
        cfg = self.cfg
        qk = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_N], dtype=gl.float32, layout=cfg.qk_layout
        )
        return amd.cdna3.mfma(q, k, qk)

    @gluon.jit
    def compute_pv(self, p, v, acc):
        return amd.cdna3.mfma(p, v, acc)

    @gluon.jit
    def init_attention_state(self):
        cfg = self.cfg
        if cfg.HAS_SINK:
            sink_log2 = gl.load(self.sink_ptr + self.q_head).to(gl.float32) * _INV_LN2
            sink_unscaled = sink_log2 / cfg.SM_SCALE
            m_i = gl.full(
                [cfg.BLOCK_M],
                value=0,
                dtype=gl.float32,
                layout=gl.SliceLayout(1, cfg.pv_layout),
            )
            m_i += sink_unscaled
        else:
            sink_log2 = 0.0
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
        return m_i, l_i, acc, sink_log2

    @gluon.jit
    def softmax(self, qk, m_i, l_i, acc):
        cfg = self.cfg
        HAS_INVALID: gl.constexpr = cfg.WINDOW_LEFT >= 0 and not cfg.HAS_SINK

        row_max = _max(qk, 1)
        m_new = _maximum(m_i, row_max)

        if cfg.ENABLE_SKIP_SOFTMAX:
            skip = (row_max - m_i) * cfg.SM_SCALE < self.log2_threshold
            offs_m = self.q_start + gl.arange(
                0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)
            )
            skip |= offs_m >= self.seq_len
            all_skip = gl.sum(skip.to(gl.int32), axis=0) == cfg.BLOCK_M
            if all_skip:
                m_new = m_i
        else:
            all_skip = False

        p = gl.zeros(
            [cfg.BLOCK_M, cfg.BLOCK_N],
            dtype=self.q_ptr.dtype.element_ty,
            layout=cfg.p_layout,
        )
        if not all_skip:
            m_new_scaled = m_new * cfg.SM_SCALE
            if HAS_INVALID:
                invalid = m_new == -float("inf")
                m_new_scaled = gl.where(invalid, 0.0, m_new_scaled)

            qk_shifted = qk * cfg.SM_SCALE - m_new_scaled[:, None]
            p_f32 = gl.exp2(qk_shifted)
            m_diff = m_i * cfg.SM_SCALE - m_new_scaled
            if HAS_INVALID:
                m_diff = gl.where(invalid, 0.0, m_diff)

            alpha = gl.exp2(m_diff)
            l_ij = gl.sum(p_f32, axis=1)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            p = gl.convert_layout(p_f32.to(self.q_ptr.dtype.element_ty), cfg.p_layout)
        return p, m_new, l_i, acc, all_skip

    @gluon.jit
    def apply_sinks(self, l_i, m_i, sink_log2):
        cfg = self.cfg
        if cfg.HAS_SINK:
            l_i += gl.exp2(sink_log2 - m_i * cfg.SM_SCALE)
        return l_i

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


@gluon.aggregate
class ProgramScheduler:
    # ProgramScheduler only controls the persistent work order. Attention
    # semantics such as sliding-window masking remain in AttentionConfig.
    cfg: gl.constexpr
    swizzled_order: gl.constexpr
    work: gl.tensor
    total_work: gl.tensor
    num_q_blocks: gl.tensor
    slot_valid: gl.tensor
    batch_slot: gl.tensor
    q_head: gl.tensor
    q_slot: gl.tensor
    q_cycles_per_batch_group: gl.tensor
    counter_ptr: gl.tensor
    batch_slots: gl.constexpr
    q_slots: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        swizzled_order,
        work,
        total_work,
        num_q_blocks,
        slot_valid,
        batch_slot,
        q_head,
        q_slot,
        q_cycles_per_batch_group,
        counter_ptr,
        batch_slots,
        q_slots,
    ):
        self.cfg = gl.constexpr(cfg)
        self.swizzled_order = gl.constexpr(swizzled_order)
        self.work = work
        self.total_work = total_work
        self.num_q_blocks = num_q_blocks
        self.slot_valid = slot_valid
        self.batch_slot = batch_slot
        self.q_head = q_head
        self.q_slot = q_slot
        self.q_cycles_per_batch_group = q_cycles_per_batch_group
        self.counter_ptr = counter_ptr
        self.batch_slots = gl.constexpr(batch_slots)
        self.q_slots = gl.constexpr(q_slots)

    @gluon.jit
    def create(
        cfg, batch_size, max_seqlen_q, counter_ptr, swizzled_order: gl.constexpr
    ):
        # Upstream verbatim (plain gluon, no CDNA4 ops), with one branch made
        # constexpr (``cfg.BATCH_SIZE`` instead of the runtime ``batch_size``)
        # so stock triton accepts the aggregate return types across branches.
        num_q_blocks = (max_seqlen_q + cfg.BLOCK_M - 1) // cfg.BLOCK_M

        start_pid = gl.program_id(axis=0)
        pids_per_xcd: gl.constexpr = cfg.NUM_BLOCKS // cfg.NUM_XCDS
        xcd = start_pid % cfg.NUM_XCDS
        local_pid = start_pid // cfg.NUM_XCDS
        logical_pid = xcd * pids_per_xcd + local_pid

        if swizzled_order:
            max_batch_slots: gl.constexpr = cfg.NUM_BLOCKS // cfg.N_HEADS
            if cfg.BATCH_SIZE < max_batch_slots:
                batch_slots: gl.constexpr = cfg.BATCH_SIZE
            else:
                batch_slots: gl.constexpr = max_batch_slots
            q_slots: gl.constexpr = cfg.NUM_BLOCKS // (batch_slots * cfg.N_HEADS)

            q_cycles_per_batch_group = (num_q_blocks + q_slots - 1) // q_slots
            num_batch_groups: gl.constexpr = (
                cfg.BATCH_SIZE + batch_slots - 1
            ) // batch_slots

            active_slots: gl.constexpr = batch_slots * cfg.N_HEADS * q_slots

            if cfg.DYNAMIC_SCHED:
                group_size: gl.constexpr = cfg.DYNAMIC_SCHED
                n_groups: gl.constexpr = (
                    cfg.BATCH_SIZE * cfg.N_HEADS + group_size - 1
                ) // group_size
                total_work = n_groups * group_size * num_q_blocks
                work = gl.atomic_add(counter_ptr, 1, sem="relaxed", scope="gpu")
                slot_valid = logical_pid >= 0
            else:
                total_work = num_batch_groups * q_cycles_per_batch_group
                zero = logical_pid - logical_pid
                work = zero
                slot_valid = logical_pid < active_slots

            safe_pid = gl.where(slot_valid, logical_pid, 0)
            q_slot = safe_pid % q_slots
            head_batch_slot = safe_pid // q_slots
            q_head = head_batch_slot % cfg.N_HEADS
            batch_slot = head_batch_slot // cfg.N_HEADS
        else:
            total_work = batch_size * cfg.N_HEADS * num_q_blocks
            zero = logical_pid - logical_pid
            slot_valid = logical_pid >= 0
            batch_slot = zero
            q_head = zero
            q_slot = zero
            q_cycles_per_batch_group = num_q_blocks
            work = logical_pid

        # Normalize the constexpr slot fields across branches: upstream's
        # same-named locals are fine on their triton; stock triton requires
        # the two branches' aggregates to match, so bind them through the
        # constants selected above.
        if swizzled_order:
            bs: gl.constexpr = cfg.BATCH_SIZE if cfg.BATCH_SIZE < (cfg.NUM_BLOCKS // cfg.N_HEADS) else (cfg.NUM_BLOCKS // cfg.N_HEADS)
            qs: gl.constexpr = cfg.NUM_BLOCKS // (bs * cfg.N_HEADS)
        else:
            bs: gl.constexpr = 1
            qs: gl.constexpr = 1

        return ProgramScheduler(
            gl.constexpr(cfg),
            swizzled_order,
            work,
            total_work,
            num_q_blocks,
            slot_valid,
            batch_slot,
            q_head,
            q_slot,
            q_cycles_per_batch_group,
            counter_ptr,
            bs,
            qs,
        )

    @gluon.jit
    def has_work(self):
        return self.work < self.total_work

    @gluon.jit
    def advance(self):
        cfg = self.cfg
        if self.swizzled_order:
            if cfg.DYNAMIC_SCHED:
                next_work = gl.atomic_add(
                    self.counter_ptr, 1, sem="relaxed", scope="gpu"
                )
            else:
                next_work = self.work + 1
        else:
            next_work = self.work + cfg.NUM_BLOCKS
        return ProgramScheduler(
            gl.constexpr(self.cfg),
            self.swizzled_order,
            next_work,
            self.total_work,
            self.num_q_blocks,
            self.slot_valid,
            self.batch_slot,
            self.q_head,
            self.q_slot,
            self.q_cycles_per_batch_group,
            self.counter_ptr,
            self.batch_slots,
            self.q_slots,
        )

    @gluon.jit
    def get_program(
        self,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        sink_ptr,
        lse_ptr,
        cu_seqlens_ptr,
        log2_threshold,
    ):
        cfg = self.cfg
        if self.swizzled_order and cfg.DYNAMIC_SCHED:
            GROUP: gl.constexpr = cfg.DYNAMIC_SCHED
            per_group = GROUP * self.num_q_blocks
            ticket = self.work
            group = ticket // per_group
            within = ticket - group * per_group
            qb_index = within // GROUP
            head_in_group = within - qb_index * GROUP
            query_block = self.num_q_blocks - 1 - qb_index
            flat_head = group * GROUP + head_in_group
            q_head = flat_head % cfg.N_HEADS
            batch = flat_head // cfg.N_HEADS
            valid = (
                (flat_head < cfg.BATCH_SIZE * cfg.N_HEADS)
                & (query_block >= 0)
                & (query_block < self.num_q_blocks)
            )

            safe_batch = gl.where(valid, batch, 0)
            seq_base = gl.load(cu_seqlens_ptr + safe_batch)
            seq_end = gl.load(cu_seqlens_ptr + safe_batch + 1)
            seq_len = seq_end - seq_base
            program = AttentionProgram.initialize_from_state(
                cfg,
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sink_ptr,
                lse_ptr,
                seq_base,
                seq_len,
                gl.where(valid, query_block, 0),
                q_head,
                log2_threshold,
            )
            return program, valid & (program.q_start < program.seq_len)

        elif self.swizzled_order:
            q_cycle_global = self.work
            batch_group = q_cycle_global // self.q_cycles_per_batch_group
            q_cycle = q_cycle_global - batch_group * self.q_cycles_per_batch_group

            query_block_inc = q_cycle * self.q_slots + self.q_slot
            query_block_dec = q_cycle * self.q_slots + (self.q_slots - 1 - self.q_slot)
            query_block = gl.where(q_cycle % 2 == 0, query_block_inc, query_block_dec)
            batch = batch_group * self.batch_slots + self.batch_slot
            valid = self.slot_valid & (query_block < self.num_q_blocks)

            safe_batch = gl.where(valid, batch, 0)
            seq_base = gl.load(cu_seqlens_ptr + safe_batch)
            seq_end = gl.load(cu_seqlens_ptr + safe_batch + 1)
            seq_len = seq_end - seq_base
            program = AttentionProgram.initialize_from_state(
                cfg,
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sink_ptr,
                lse_ptr,
                seq_base,
                seq_len,
                query_block,
                self.q_head,
                log2_threshold,
            )
            return program, valid & (program.q_start < program.seq_len)

        else:
            query_block = self.work % self.num_q_blocks
            head_batch = self.work // self.num_q_blocks
            q_head = head_batch % cfg.N_HEADS
            batch = head_batch // cfg.N_HEADS
            seq_base = gl.load(cu_seqlens_ptr + batch)
            seq_end = gl.load(cu_seqlens_ptr + batch + 1)
            seq_len = seq_end - seq_base
            program = AttentionProgram.initialize_from_state(
                cfg,
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                sink_ptr,
                lse_ptr,
                seq_base,
                seq_len,
                query_block,
                q_head,
                log2_threshold,
            )
            return program, program.q_start < program.seq_len


# ===-----------------------------------------------------------------------===#
# Tile Processing
# ===-----------------------------------------------------------------------===#


@gluon.jit
def process_single_attention_tile(program: AttentionProgram):
    cfg = program.cfg
    q = program.load_q(other=0.0)

    k_offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(1, cfg.k_layout))
    k_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.k_layout))
    k_offsets = cfg.k_strides.offsets(
        program.seq_base + k_offs_n[None, :], program.kv_head, k_offs_d[:, None]
    )
    k_mask = k_offs_n[None, :] < program.seq_len
    k = cdna4.buffer_load(program.k_ptr, k_offsets, mask=k_mask, other=0.0)

    v_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(1, cfg.v_layout))
    v_offs_d = gl.arange(0, cfg.HEAD_DIM, layout=gl.SliceLayout(0, cfg.v_layout))
    v_offsets = cfg.v_strides.offsets(
        program.seq_base + v_offs_n[:, None], program.kv_head, v_offs_d[None, :]
    )
    v_mask = v_offs_n[:, None] < program.seq_len
    v = cdna4.buffer_load(program.v_ptr, v_offsets, mask=v_mask, other=0.0)

    qk = program.compute_qk(q, k)

    mask_offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
    mask_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    valid = mask_offs_m[:, None] < program.seq_len
    valid &= mask_offs_n[None, :] < program.seq_len
    valid &= mask_offs_n[None, :] <= mask_offs_m[:, None]
    if cfg.WINDOW_LEFT >= 0:
        valid &= mask_offs_m[:, None] <= mask_offs_n[None, :] + cfg.WINDOW_LEFT

    qk = gl.where(valid, qk, -1.0e20)
    row_has_valid = gl.sum(valid.to(gl.int32), axis=1) > 0
    row_max = _max(qk, 1)
    m_i = gl.where(row_has_valid, row_max, 0.0)
    m_i_scaled = m_i * cfg.SM_SCALE
    p = gl.where(valid, gl.exp2(qk * cfg.SM_SCALE - m_i_scaled[:, None]), 0.0)
    l_i = gl.sum(p, axis=1)
    if cfg.HAS_SINK:
        sink_log2 = gl.load(program.sink_ptr + program.q_head).to(gl.float32) * _INV_LN2
        l_i += gl.exp2(sink_log2 - m_i_scaled)

    acc = gl.zeros([cfg.BLOCK_M, cfg.HEAD_DIM], dtype=gl.float32, layout=cfg.pv_layout)
    p = p.to(program.q_ptr.dtype.element_ty)
    p = gl.convert_layout(p, cfg.p_layout)
    acc = program.compute_pv(p, v, acc)

    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


@gluon.jit
def process_attention_tile(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
    boundary_mask0=None,
    boundary_mask1=None,
):
    cfg = program.cfg
    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_attention_state()

    main_end = program.q_start // cfg.BLOCK_N
    base_k_offsets, base_offs_n = program.make_k_offsets(0)
    base_v_offsets = program.make_v_offsets(0)

    k_offsets = base_k_offsets
    v_offsets = base_v_offsets
    offs_n = base_offs_n

    for _ in range(0, main_end):
        program.load_k(k_offsets, k_smem)
        if cfg.DEFER_V_LOAD:
            k = program.shared_load_k(k_smem)
            qk = program.compute_qk(q, k)
            p, m_i, l_i, acc, all_skip = program.softmax(qk, m_i, l_i, acc)

            if not (cfg.ENABLE_SKIP_SOFTMAX and all_skip):
                program.load_v(v_offsets, v_smem)
                v = program.shared_load_v(v_smem)
                acc = program.compute_pv(p, v, acc)
        else:
            program.load_v(v_offsets, v_smem)
            k = program.shared_load_k(k_smem)
            qk = program.compute_qk(q, k)
            p, m_i, l_i, acc, all_skip = program.softmax(qk, m_i, l_i, acc)

            v = program.shared_load_v(v_smem)
            if not (cfg.ENABLE_SKIP_SOFTMAX and all_skip):
                acc = program.compute_pv(p, v, acc)

        k_offsets = program.update_k_offsets(k_offsets)
        v_offsets = program.update_v_offsets(v_offsets)
        offs_n = offs_n + cfg.BLOCK_N

    boundary_start = main_end * cfg.BLOCK_N
    k_offsets, offs_n = program.make_k_offsets(boundary_start)
    v_offsets = program.make_v_offsets(boundary_start)
    mask = offs_n[:, None] < program.seq_len
    program.load_k(k_offsets, k_smem, mask=mask, other=0.0)
    program.load_v(v_offsets, v_smem, mask=mask, other=0.0)

    k = program.shared_load_k(k_smem)
    qk = program.compute_qk(q, k)
    qk = gl.where(boundary_mask0, qk, -float("inf"))
    p, m_i, l_i, acc, all_skip = program.softmax(qk, m_i, l_i, acc)

    v = program.shared_load_v(v_smem)
    if not (cfg.ENABLE_SKIP_SOFTMAX and all_skip):
        acc = program.compute_pv(p, v, acc)

    boundary_start = boundary_start + cfg.BLOCK_N
    k_offsets, offs_n = program.make_k_offsets(boundary_start)
    v_offsets = program.make_v_offsets(boundary_start)
    mask = offs_n[:, None] < program.seq_len
    program.load_k(k_offsets, k_smem, mask=mask, other=0.0)
    program.load_v(v_offsets, v_smem, mask=mask, other=0.0)

    k = program.shared_load_k(k_smem)
    qk = program.compute_qk(q, k)
    qk = gl.where(boundary_mask1, qk, -float("inf"))
    p, m_i, l_i, acc, all_skip = program.softmax(qk, m_i, l_i, acc)

    v = program.shared_load_v(v_smem)
    if not (cfg.ENABLE_SKIP_SOFTMAX and all_skip):
        acc = program.compute_pv(p, v, acc)

    l_i = program.apply_sinks(l_i, m_i, sink_log2)
    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


@gluon.jit
def process_sliding_attention_tile(
    program: AttentionProgram,
    k_smem: gl.shared_memory_descriptor,
    v_smem: gl.shared_memory_descriptor,
):
    cfg = program.cfg
    q = program.load_q()
    m_i, l_i, acc, sink_log2 = program.init_attention_state()

    kv_start = program.q_start - cfg.WINDOW_LEFT
    kv_start = gl.where(kv_start > 0, (kv_start // cfg.BLOCK_N) * cfg.BLOCK_N, 0)
    num_kv_tiles: gl.constexpr = (
        cfg.BLOCK_M + cfg.WINDOW_LEFT + cfg.BLOCK_N - 1
    ) // cfg.BLOCK_N
    offs_m = program.q_start + gl.arange(
        0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout)
    )
    mask_n = kv_start + gl.arange(
        0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout)
    )
    mask_diff = offs_m[:, None] - mask_n[None, :]

    for _ in gl.static_range(num_kv_tiles):
        k_offsets, offs_n = program.make_k_offsets(kv_start)
        v_offsets = program.make_v_offsets(kv_start)
        mask = offs_n[:, None] < program.seq_len
        program.load_k(k_offsets, k_smem, mask=mask)
        program.load_v(v_offsets, v_smem, mask=mask, other=0.0)

        valid = mask_diff.to(gl.uint32) <= cfg.WINDOW_LEFT
        if kv_start + cfg.BLOCK_N > program.seq_len:
            valid &= mask_n[None, :] < program.seq_len

        k = program.shared_load_k(k_smem)
        qk = program.compute_qk(q, k)
        qk = gl.where(valid, qk, -float("inf"))
        p, m_i, l_i, acc, all_skip = program.softmax(qk, m_i, l_i, acc)

        v = program.shared_load_v(v_smem)
        if not (cfg.ENABLE_SKIP_SOFTMAX and all_skip):
            acc = program.compute_pv(p, v, acc)
        kv_start = kv_start + cfg.BLOCK_N

        mask_n = mask_n + cfg.BLOCK_N
        mask_diff = mask_diff - cfg.BLOCK_N

    l_i = program.apply_sinks(l_i, m_i, sink_log2)
    program.store_lse(l_i, m_i)
    denom = gl.where(l_i > 0.0, l_i, 1.0)
    recip_denom = 1.0 / denom
    output = acc * recip_denom[:, None]
    output = gl.convert_layout(output, cfg.store_layout)
    program.store_output(output)


# ===-----------------------------------------------------------------------===#
# Entry Points
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_mha_prefill_gfx950(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_ptr,
    output_ptr,
    sink_ptr,
    lse_ptr,
    sched_counter_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    K_STRIDE_T: gl.constexpr,
    K_STRIDE_H: gl.constexpr,
    K_STRIDE_D: gl.constexpr,
    V_STRIDE_T: gl.constexpr,
    V_STRIDE_H: gl.constexpr,
    V_STRIDE_D: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SM_SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    max_seqlen_q,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    ENABLE_SKIP_SOFTMAX: gl.constexpr,
    DEFER_V_LOAD: gl.constexpr,
    DYNAMIC_SCHED: gl.constexpr,
    log2_threshold,
):
    cfg = AttentionConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        BATCH_SIZE,
        HAS_SINK,
        HAS_LSE,
        -1,
        IS_FP8,
        ENABLE_SKIP_SOFTMAX,
        DEFER_V_LOAD,
        DYNAMIC_SCHED,
        k_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(K_STRIDE_T, K_STRIDE_H, K_STRIDE_D),
        InputStrides(V_STRIDE_T, V_STRIDE_H, V_STRIDE_D),
    )
    # PORT: K stages pre-permuted [HEAD_DIM, BLOCK_N] (see module docstring).
    k_smem = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [cfg.HEAD_DIM, cfg.BLOCK_N],
        cfg.k_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.v_smem_layout,
    )

    scheduler = ProgramScheduler.create(
        cfg, BATCH_SIZE, max_seqlen_q, sched_counter_ptr, True
    )
    mask_offs_m = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.qk_layout))
    mask_offs_n = gl.arange(0, cfg.BLOCK_N, layout=gl.SliceLayout(0, cfg.qk_layout))
    boundary_mask0 = mask_offs_n[None, :] <= mask_offs_m[:, None]
    boundary_mask1 = (mask_offs_n[None, :] + cfg.BLOCK_N) <= mask_offs_m[:, None]

    while scheduler.has_work():
        program, active = scheduler.get_program(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            sink_ptr,
            lse_ptr,
            cu_seqlens_ptr,
            log2_threshold,
        )
        if active:
            if program.seq_len < cfg.BLOCK_N:
                if program.q_start == 0:
                    process_single_attention_tile(program)
            else:
                process_attention_tile(
                    program, k_smem, v_smem, boundary_mask0, boundary_mask1
                )
        scheduler = scheduler.advance()


@gluon.jit
def gluon_mha_prefill_sliding_gfx950(
    q_ptr,
    k_ptr,
    v_ptr,
    cu_seqlens_ptr,
    output_ptr,
    sink_ptr,
    lse_ptr,
    sched_counter_ptr,
    Q_STRIDE_T: gl.constexpr,
    Q_STRIDE_H: gl.constexpr,
    Q_STRIDE_D: gl.constexpr,
    K_STRIDE_T: gl.constexpr,
    K_STRIDE_H: gl.constexpr,
    K_STRIDE_D: gl.constexpr,
    V_STRIDE_T: gl.constexpr,
    V_STRIDE_H: gl.constexpr,
    V_STRIDE_D: gl.constexpr,
    N_HEADS: gl.constexpr,
    N_KV_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    SM_SCALE: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    BATCH_SIZE: gl.constexpr,
    max_seqlen_q,
    HAS_SINK: gl.constexpr,
    HAS_LSE: gl.constexpr,
    WINDOW_LEFT: gl.constexpr,
    IS_FP8: gl.constexpr,
    ENABLE_SKIP_SOFTMAX: gl.constexpr,
    DEFER_V_LOAD: gl.constexpr,
    DYNAMIC_SCHED: gl.constexpr,
    log2_threshold,
):
    cfg = AttentionConfig(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        SM_SCALE,
        BLOCK_M,
        BLOCK_N,
        NUM_WARPS,
        BATCH_SIZE,
        HAS_SINK,
        HAS_LSE,
        WINDOW_LEFT,
        IS_FP8,
        ENABLE_SKIP_SOFTMAX,
        False,
        0,
        k_ptr.dtype.element_ty,
        InputStrides(Q_STRIDE_T, Q_STRIDE_H, Q_STRIDE_D),
        InputStrides(K_STRIDE_T, K_STRIDE_H, K_STRIDE_D),
        InputStrides(V_STRIDE_T, V_STRIDE_H, V_STRIDE_D),
    )
    k_smem = gl.allocate_shared_memory(
        k_ptr.dtype.element_ty,
        [cfg.HEAD_DIM, cfg.BLOCK_N],
        cfg.k_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        v_ptr.dtype.element_ty,
        [cfg.BLOCK_N, cfg.HEAD_DIM],
        cfg.v_smem_layout,
    )

    scheduler = ProgramScheduler.create(
        cfg, BATCH_SIZE, max_seqlen_q, sched_counter_ptr, False
    )

    while scheduler.has_work():
        program, active = scheduler.get_program(
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr,
            sink_ptr,
            lse_ptr,
            cu_seqlens_ptr,
            log2_threshold,
        )
        if active:
            process_sliding_attention_tile(program, k_smem, v_smem)
        scheduler = scheduler.advance()


# ===-----------------------------------------------------------------------===#
# Host
# ===-----------------------------------------------------------------------===#


class LaunchConfig(NamedTuple):
    n_heads: int
    n_kv_heads: int
    head_dim: int
    sm_scale: float
    block_m: int
    block_n: int
    num_warps: int
    batch_size: int
    max_seqlen: int
    window_left: int
    grid: tuple[int, ...]
    log2_threshold: float


def get_config(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen: int,
    window_left: int,
    softmax_scale: float | None,
    skip_softmax_threshold: float,
) -> LaunchConfig:
    n_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    block_m = 64
    block_n = 32  # PORT: gfx942 64KB LDS (upstream 64 needs MI350's 128KB)
    num_warps = 2
    batch_size = cu_seqlens_q.numel() - 1
    window_left = window_left if window_left >= 0 else -1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    sm_scale = softmax_scale * _INV_LN2_VALUE
    log2_threshold = (
        math.log(skip_softmax_threshold) * _INV_LN2_VALUE
        if skip_softmax_threshold > 0.0
        else 0.0
    )
    return LaunchConfig(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        sm_scale=sm_scale,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        batch_size=batch_size,
        max_seqlen=max_seqlen,
        window_left=window_left,
        grid=(512,),
        log2_threshold=log2_threshold,
    )


def launch_gluon_mha_prefill_gfx942(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: list[int],
    max_seqlen: int,
    window_left: int = -1,
    logit_cap: float = 0.0,
    sinks: torch.Tensor | None = None,
    return_lse: bool = False,
    softmax_scale: float | None = None,
    skip_softmax_threshold: float = 0.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Varlen causal MHA prefill (CDNA3 port). bf16/fp16 only; fp8 raises."""
    del cu_seqlens_cpu
    if logit_cap != 0.0:
        raise NotImplementedError("the CDNA3 MHA prefill port does not support logit_cap")
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) or k.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        raise ValueError(
            "the CDNA3 MHA prefill port supports bf16/fp16 only (fp8 needs "
            "dot_scaled MFMA — not ported)"
        )
    total_tokens, n_heads, _ = q.shape
    config = get_config(
        q=q,
        k=k,
        cu_seqlens_q=cu_seqlens,
        max_seqlen=max_seqlen,
        window_left=window_left,
        softmax_scale=softmax_scale,
        skip_softmax_threshold=skip_softmax_threshold,
    )
    enable_skip_softmax = skip_softmax_threshold > 0.0
    is_sliding = config.window_left >= 0
    has_sink = sinks is not None
    sink_arg = sinks if sinks is not None else q
    if return_lse:
        lse = torch.empty(
            (total_tokens, n_heads), dtype=torch.float32, device=q.device
        )
    else:
        lse = q
    has_lse = return_lse
    output = torch.empty(q.shape, device=q.device, dtype=q.dtype)

    defer_v_load = enable_skip_softmax and not is_sliding
    dynamic_sched = (
        config.n_heads // config.n_kv_heads
        if enable_skip_softmax and not is_sliding
        else 0
    )
    sched_counter = (
        torch.zeros(1, device=q.device, dtype=torch.int32) if dynamic_sched else q
    )

    kernel = (
        gluon_mha_prefill_sliding_gfx950 if is_sliding else gluon_mha_prefill_gfx950
    )
    kernel[config.grid](
        q,
        k,
        v,
        cu_seqlens,
        output,
        sink_arg,
        lse,
        sched_counter,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        config.n_heads,
        config.n_kv_heads,
        config.head_dim,
        config.sm_scale,
        config.block_m,
        config.block_n,
        config.num_warps,
        config.batch_size,
        config.max_seqlen,
        has_sink,
        has_lse,
        config.window_left,
        False,
        enable_skip_softmax,
        defer_v_load,
        dynamic_sched,
        config.log2_threshold,
        num_warps=config.num_warps,
    )
    if return_lse:
        return output, lse
    return output


__all__ = ["launch_gluon_mha_prefill_gfx942"]
