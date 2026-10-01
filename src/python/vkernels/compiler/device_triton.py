"""Triton device backend: the phase-synchronous megakernel on a real GPU.

This closes the loop the design doc leaves open at §8.4: a *verified* grid
synchronization path on the actual stack. Triton 3.7 on this GB10 provides
GPU-scope acquire/release atomics (``tl.atomic_add(..., sem=..., scope="gpu")``)
and data-dependent control flow, which is exactly what the §8.1 device
algorithm needs:

    for phase in compiled_phases:
        task_id = worker
        while task_id < phase.task_count:
            run_task(...)
            task_id += workers
        grid_sync()          # outside the tile loop: idle workers participate

The barrier is a monotonic-counter spin barrier: each block publishes its
prior writes with an ``acq_rel`` arrival and re-acquires with an atomic
poll; the host passes a per-invocation counter base (deterministic: the
base grows by ``num_barriers * P`` per step), so no reset kernel and no
extra launch is needed between invocations — **one kernel launch per
decode step**, verifiable by trace (§15.2).

Batched decode (B>1, the §1.1 "each sequence supplies one token" contract,
one launch serving B sequences): task domains grow by B where per-row or
per-head (RMSNorm per row, QK-norm/RoPE/attention per (batch, head),
append per (batch, kv-head)); projection tiles compute B rows per task
(shared weight tile, one row at a time — B=1 compiles to the original
single-row path). Every sequence carries its **own runtime position** and
its **own slot-table row**, so a batch is naturally ragged: sequences at
different cache lengths, some continuing committed kvaas prefixes, some
starting cold — all in the same persistent launch.

The KV cache is addressed through the kvaas data plane: token-slot-major
pool buffers (one slot = one token's K or V per layer,
``n_kv_heads * head_dim * dtype`` bytes) indexed by ``table[b, t]`` — the
identity table locally, ``Admission.block_table()`` rows under a lease.

The CPU reference executor remains the schedule-logic oracle; this backend
is validated against the HF-checked NumPy oracle (fp32 ~1e-6, bf16 ~5e-3
through 28 layers).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from .triton_dense import _t_embed as _t_embed, _t_rms2d as _t_rms2d, _t_rms_heads as _t_rms_heads, _t_rms2d_gated as _t_rms2d_gated, _t_rms_heads_gated as _t_rms_heads_gated, _t_gemv as _t_gemv, _t_gemv_transposed as _t_gemv_transposed, _t_swiglu as _t_swiglu, _t_add as _t_add, _t_linear_grouped as _t_linear_grouped, _t_gemv_fp8 as _t_gemv_fp8, _t_moe_expert as _t_moe_expert, _t_moe_combine as _t_moe_combine, _t_moe_route as _t_moe_route
from .triton_attention import _t_rope as _t_rope, _t_append as _t_append, _t_scores as _t_scores, _t_softmax as _t_softmax, _t_values as _t_values, _t_values_gated as _t_values_gated, _t_rope_interleaved as _t_rope_interleaved, _t_mla_scores as _t_mla_scores, _t_mla_values as _t_mla_values, _t_indexer_scores as _t_indexer_scores, _t_index_topk as _t_index_topk, _t_compressor_append as _t_compressor_append
from .triton_recurrent import _t_gdn_conv as _t_gdn_conv, _t_gdn_conv_tiled as _t_gdn_conv_tiled, _t_gdn_heads as _t_gdn_heads, _t_mhc_pre as _t_mhc_pre, _t_mhc_post as _t_mhc_post, _t_gdn_heads_batched as _t_gdn_heads_batched, _t_kda_heads_batched as _t_kda_heads_batched

__all__ = [
    "grid_barrier",
    "qwen3_megakernel",
    "TritonMegakernel",
    "triton_available",
    "milestone0_barrier_test",
    "attach_megakernel_pool",
]


def triton_available() -> bool:
    # torch and triton are imported unconditionally at module scope, so this
    # is a pure device probe: the old inner 'import triton' try/except was
    # dead code. torch.cuda.is_available() returns False on healthy GPU-less
    # hosts; a broken CUDA runtime raises here (surfaced) instead of being
    # masked as "triton unavailable".
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Milestone-0 primitive: the grid barrier (§8.4)
# ---------------------------------------------------------------------------


@triton.jit
def grid_barrier(bar_ptr, target: tl.int64):
    """Monotonic spin barrier over all resident blocks.

    ``bar_ptr`` is a single int64 counter that never resets within (or
    across) invocations; ``target`` is the absolute arrival count that
    releases this barrier instance (``base + (k+1) * P`` for barrier k).
    The block-level fence first orders every thread's prior global stores;
    arrival uses acq_rel at gpu scope (publishing them, acquiring the
    counter); the poll uses acquire, ordering all post-barrier loads after
    the release wave.
    """
    tl.debug_barrier()  # all threads' stores issued before the arrival
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def milestone0_barrier_test(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64, IDLE: tl.constexpr):
    """§8.4 producer-barrier-consumer capability microprogram.

    Every round: each block writes a round-tagged value, barrier, then
    reads a neighbour's value (cross-block visibility through the
    barrier). With ``IDLE`` set, odd-numbered blocks skip the write work
    but still arrive at every barrier (§8.3 idle-worker participation).
    Accumulated neighbour values are checksummed per block.
    """
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        if (not IDLE) or (pid % 2 == 0):
            tl.store(out_ptr + pid, pid * 1000.0 + r)
        grid_barrier(bar_ptr, base + (r + 1) * P)
        nb = (pid + 1) % P
        v = tl.load(out_ptr + nb)
        if (not IDLE) or (nb % 2 == 0):
            acc += v
        r += 1
    tl.store(out_ptr + P + pid, acc)


# ---------------------------------------------------------------------------
# Task bodies (§6), batched: each runs tasks worker, worker+P, ... of its
# phase. B is constexpr; B == 1 compiles the original single-row paths.
# Row/head tasks decode (b, h) from the flat task id; every sequence has
# its own runtime position (pos_ptr[b]) and slot-table row (table[b, t]).
# ---------------------------------------------------------------------------
































# ---------------------------------------------------------------------------
# The megakernel (§8.1): embedding | L x 17 phases | final norm + tied head
# ---------------------------------------------------------------------------


@triton.jit(do_not_specialize=["bar_base", "L", "WQKV_L", "WOP_L", "WGU_L", "WDOWN_L", "LNLN_L", "KCBASE_L"])
def qwen3_megakernel(
    ids_ptr,
    pos_ptr,
    table_ptr,
    tok_ptr,
    final_g_ptr,
    cos_ptr,
    sin_ptr,
    ln1_ptr,
    qkv_w_ptr,
    qn_ptr,
    kn_ptr,
    op_ptr,
    ln2_ptr,
    gu_ptr,
    down_ptr,
    k_cache_ptr,
    v_cache_ptr,
    ws_ptr,
    logits_ptr,
    bar_ptr,
    bar_base_ptr,
    L,
    WQKV_L,
    WOP_L,
    WGU_L,
    WDOWN_L,
    LNLN_L,
    KCBASE_L,
    # dims
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    F: tl.constexpr,
    V: tl.constexpr,
    SCAP: tl.constexpr,
    QKVW: tl.constexpr,
    HD: tl.constexpr,
    F2: tl.constexpr,
    layout_ptr,
    O_EMBED: tl.constexpr,
    O_FINAL_INPUT: tl.constexpr,
    O_FINAL: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    TILE: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    BTR: tl.constexpr,
    ELEM: tl.constexpr,
):
    bar_base = tl.load(bar_base_ptr)
    worker = tl.program_id(0)
    P = tl.num_programs(0)
    BC: tl.constexpr = B * C

    # ---- phase 0: embedding -------------------------------------------
    _t_embed(worker, P, ids_ptr, tok_ptr, ws_ptr + O_EMBED, B, C, C)
    grid_barrier(bar_ptr, bar_base + P)

    for l in range(L):
        li = l.to(tl.int64)
        base = ws_ptr
        o_hidden_a = tl.load(layout_ptr + li * 17 + 0)
        o_rms1 = tl.load(layout_ptr + li * 17 + 1)
        o_qkv = tl.load(layout_ptr + li * 17 + 2)
        o_qn = tl.load(layout_ptr + li * 17 + 3)
        o_kn = tl.load(layout_ptr + li * 17 + 4)
        o_rq = tl.load(layout_ptr + li * 17 + 5)
        o_rk = tl.load(layout_ptr + li * 17 + 6)
        o_scores = tl.load(layout_ptr + li * 17 + 7)
        o_probs = tl.load(layout_ptr + li * 17 + 8)
        o_ctx = tl.load(layout_ptr + li * 17 + 9)
        o_attn = tl.load(layout_ptr + li * 17 + 10)
        o_hidden_b = tl.load(layout_ptr + li * 17 + 11)
        o_rms2 = tl.load(layout_ptr + li * 17 + 12)
        o_gu = tl.load(layout_ptr + li * 17 + 13)
        o_act = tl.load(layout_ptr + li * 17 + 14)
        o_down = tl.load(layout_ptr + li * 17 + 15)
        o_hidden_out = tl.load(layout_ptr + li * 17 + 16)
        # phase 1: rms1
        _t_rms2d(worker, P, ws_ptr + o_hidden_a, ln1_ptr + li * LNLN_L, base + o_rms1, B, C, C, EPS)
        grid_barrier(bar_ptr, bar_base + (2 + 17 * l) * P)
        # phase 2: qkv projection [C -> QKVW]
        _t_gemv(worker, P, base + o_rms1, qkv_w_ptr + li * WQKV_L, base + o_qkv, B, C, QKVW, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (3 + 17 * l) * P)
        # phases 3-4: q/k RMSNorm per head (v needs no processing); the q/k
        # slices are interleaved inside each row's packed QKV projection.
        _t_rms_heads(worker, P, base + o_qkv, qn_ptr + li * D, base + o_qn, B, H, D, QKVW, EPS)
        grid_barrier(bar_ptr, bar_base + (4 + 17 * l) * P)
        _t_rms_heads(worker, P, base + o_qkv + HD, kn_ptr + li * D, base + o_kn, B, KVH, D, QKVW, EPS)
        grid_barrier(bar_ptr, bar_base + (5 + 17 * l) * P)
        # phases 5-6: rope q/k at each row's runtime position
        # (ROT=D, TSTRIDE=D: full-width rotate_half via the partial template)
        _t_rope(worker, P, base + o_qn, cos_ptr, sin_ptr, pos_ptr, base + o_rq, B, H, D, D, D)
        grid_barrier(bar_ptr, bar_base + (6 + 17 * l) * P)
        _t_rope(worker, P, base + o_kn, cos_ptr, sin_ptr, pos_ptr, base + o_rk, B, KVH, D, D, D)
        grid_barrier(bar_ptr, bar_base + (7 + 17 * l) * P)
        # phase 7: cache append (k roped; v straight from the qkv buffer)
        kcl = k_cache_ptr + li * KCBASE_L
        vcl = v_cache_ptr + li * KCBASE_L
        _t_append(worker, P, kcl, vcl, table_ptr, pos_ptr, base + o_rk, base + o_qkv + HD + KVH * D, B, KVH, D, QKVW, SCAP)
        grid_barrier(bar_ptr, bar_base + (8 + 17 * l) * P)
        # phases 8-10: attention (GQA, per-row valid lengths)
        _t_scores(worker, P, base + o_rq, kcl, table_ptr, pos_ptr, base + o_scores, B, H, KVH, D, SCAP, BT, SCALE)
        grid_barrier(bar_ptr, bar_base + (9 + 17 * l) * P)
        _t_softmax(worker, P, base + o_scores, pos_ptr, base + o_probs, B, H, SCAP, BTR)
        grid_barrier(bar_ptr, bar_base + (10 + 17 * l) * P)
        _t_values(worker, P, base + o_probs, vcl, table_ptr, pos_ptr, base + o_ctx, B, H, KVH, D, SCAP, BT)
        grid_barrier(bar_ptr, bar_base + (11 + 17 * l) * P)
        # phase 11: o_proj [H*D -> C], phase 12: residual add
        _t_gemv(worker, P, base + o_ctx, op_ptr + li * WOP_L, base + o_attn, B, HD, C, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (12 + 17 * l) * P)
        _t_add(worker, P, ws_ptr + o_hidden_a, base + o_attn, ws_ptr + o_hidden_b, BC, ELEM)
        grid_barrier(bar_ptr, bar_base + (13 + 17 * l) * P)
        # phase 13: rms2, phase 14: gate_up [C -> 2F]
        _t_rms2d(worker, P, ws_ptr + o_hidden_b, ln2_ptr + li * LNLN_L, base + o_rms2, B, C, C, EPS)
        grid_barrier(bar_ptr, bar_base + (14 + 17 * l) * P)
        _t_gemv(worker, P, base + o_rms2, gu_ptr + li * WGU_L, base + o_gu, B, C, F2, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (15 + 17 * l) * P)
        # phase 15: swiglu, phase 16: down [F -> C], phase 17: residual add
        _t_swiglu(worker, P, base + o_gu, base + o_act, B, F, F2, ELEM)
        grid_barrier(bar_ptr, bar_base + (16 + 17 * l) * P)
        _t_gemv(worker, P, base + o_act, down_ptr + li * WDOWN_L, base + o_down, B, F, C, TILE, BK)
        grid_barrier(bar_ptr, bar_base + (17 + 17 * l) * P)
        _t_add(worker, P, ws_ptr + o_hidden_b, base + o_down, ws_ptr + o_hidden_out, BC, ELEM)
        grid_barrier(bar_ptr, bar_base + (18 + 17 * l) * P)

    # ---- final phases: norm + tied head (no barrier after the last) ------
    _t_rms2d(worker, P, ws_ptr + O_FINAL_INPUT, final_g_ptr, ws_ptr + O_FINAL, B, C, C, EPS)
    grid_barrier(bar_ptr, bar_base + (17 * L + 2) * P)
    _t_gemv_transposed(worker, P, ws_ptr + O_FINAL, tok_ptr, logits_ptr, B, C, V, TILE, BK)
    if worker == 0:
        tl.store(bar_base_ptr, bar_base + (17 * L + 2) * P)











# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    SCAP: int
    ws_total: int
    num_phases: int
    num_barriers: int
    embedding: int
    final_input: int
    final: int
    logits: int
    layer_offsets: list[list[int]]


def _plan(executable) -> _Plan:
    """Adapt the canonical schedule and packed workspace to the Qwen3 template."""
    graph, families = executable.graph, executable.families
    offsets = {b.storage_id: b.offset for b in executable.workspace_plan.buffers}

    def output(family):
        value = graph.tensor(family.op.outputs[0])
        return offsets[value.storage_id] + value.offset

    expected = ("rms_norm", "linear", "rms_norm", "rms_norm", "rope", "rope",
                "cache_append", "attention_scores", "softmax", "attention_values",
                "linear", "add", "rms_norm", "linear", "swiglu", "linear", "add")
    if len(families) != 3 + 17 * executable.config.layers:
        raise ValueError("Qwen3 backend does not support this schedule")
    layers = []
    hidden = output(families[0])
    embedding = hidden
    for layer in range(executable.config.layers):
        fs = families[1+17*layer:1+17*(layer+1)]
        if tuple(f.op.kind for f in fs) != expected:
            raise ValueError("Qwen3 backend phase contract mismatch")
        values = [output(f) for i, f in enumerate(fs) if i != 6]
        layers.append([hidden, *values])
        hidden = values[-1]
    return _Plan(executable.config.cache_capacity, executable.workspace_plan.total_elements,
                 len(families), len(families)-1, embedding, hidden,
                 output(families[-2]), output(families[-1]), layers)


def _pow2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _elem_tile(C: int, F: int) -> int:
    """Largest power-of-two elementwise tile dividing both C and F (<=1024)."""
    t = 1024
    while t > 1 and (C % t or F % t):
        t //= 2
    return t


class TritonMegakernel:
    """Host runner: one persistent launch per step, B sequences per launch.

    ``run(tokens, positions)`` executes one decode step for every batch row:
    sequence ``b`` appends token ``tokens[b]`` at its own runtime position
    ``positions[b]``, addressed through slot-table row ``b``. Rows may be at
    different cache lengths (ragged batch) and may mix cold sequences with
    ones continuing committed kvaas prefixes.
    """

    def __init__(
        self,
        config,
        weights,
        *,
        capacity: int = 256,
        workers: int = 48,
        dtype=torch.bfloat16,
        device="cuda",
        batch: int = 1,
        k_cache=None,
        v_cache=None,
        executable=None,
    ):
        assert config.heads % config.kv_heads == 0
        assert 1 <= batch
        self.config = config
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.device = device
        self.dtype = dtype
        self.batch = batch
        from dataclasses import replace
        from .compile import compile_model
        planned_config = replace(config, batch=batch, cache_capacity=capacity)
        self.executable = executable or compile_model(model_config=planned_config, weights=weights, workers=workers)
        if self.executable.config != planned_config:
            raise ValueError("executable configuration does not match the runner")
        self.plan = _plan(self.executable)
        p = self.plan
        self.elem = _elem_tile(config.hidden, config.intermediate)
        self.btcap = 64  # attention chunk; [64, D] fp32 tiles stay in registers
        assert capacity <= 1024, "loop-free softmax needs the row in one block"
        assert config.hidden % self.elem == 0 and config.intermediate % self.elem == 0

        def stack(getfn):
            return torch.stack([torch.from_numpy(getfn(l)) for l in range(config.layers)]).to(device, dtype).contiguous()

        w = weights
        self.tok = torch.from_numpy(w.token_emb).to(device, dtype).contiguous()
        self.final_g = torch.from_numpy(w.final_gamma).to(device, dtype).contiguous()
        # RoPE tables stay fp32 regardless of weight dtype (fp32 math inside).
        self.cos = torch.from_numpy(w.cos[:capacity]).to(device, torch.float32).contiguous()
        self.sin = torch.from_numpy(w.sin[:capacity]).to(device, torch.float32).contiguous()
        self.ln1 = stack(lambda l: w.layers[l].ln1_gamma)
        self.qkv_w = stack(lambda l: w.layers[l].qkv_w)
        self.qn = stack(lambda l: w.layers[l].q_norm_gamma)
        self.kn = stack(lambda l: w.layers[l].k_norm_gamma)
        self.op_w = stack(lambda l: w.layers[l].o_proj_w)
        self.ln2 = stack(lambda l: w.layers[l].ln2_gamma)
        self.gu_w = stack(lambda l: w.layers[l].gate_up_w)
        self.down_w = stack(lambda l: w.layers[l].down_w)
        self.ws = torch.zeros(p.ws_total, device=device, dtype=torch.float32)
        self._layout = torch.tensor(p.layer_offsets, device=device, dtype=torch.int64)
        # KV cache: token-slot-major [layers, max_tokens, KVH, D] — the kvaas
        # pool granularity — either locally allocated or caller-provided
        # (daemon-owned, imported via CUDA IPC). Addressing always goes
        # through the per-row slot table; the DEFAULT table gives each batch
        # row a disjoint slot range (row b -> slots [b*capacity, (b+1)*capacity))
        # so independent sequences never alias. A per-layer buffer tuple is
        # accepted when the layer stride is uniform (attach_megakernel_pool's
        # layout); the kernel then gets layer 0's pointer + the stride.
        need_slots = capacity * batch
        shape = (config.layers, need_slots, config.kv_heads, config.head_dim)
        if k_cache is None or v_cache is None:
            self.k_cache = torch.zeros(shape, device=device, dtype=torch.float32)
            self.v_cache = torch.zeros(shape, device=device, dtype=torch.float32)
            self.pool_slots = need_slots
            self._k_launch, self._v_launch = self.k_cache, self.v_cache
            self._layer_stride = need_slots * config.kv_heads * config.head_dim
        elif isinstance(k_cache, torch.Tensor):
            if tuple(k_cache.shape[0:1]) + tuple(k_cache.shape[2:]) != (config.layers, config.kv_heads, config.head_dim):
                raise ValueError(f"pool shape {tuple(k_cache.shape)} incompatible with ([layers, >= {need_slots}, kv_heads, head_dim])")
            if k_cache.shape[1] < need_slots or v_cache.shape[1] < need_slots:
                raise ValueError(f"pool has {min(k_cache.shape[1], v_cache.shape[1])} slots < {need_slots} (batch {batch} x capacity {capacity})")
            self.k_cache, self.v_cache = k_cache, v_cache
            self.pool_slots = k_cache.shape[1]
            self._k_launch, self._v_launch = k_cache, v_cache
            self._layer_stride = k_cache.shape[1] * config.kv_heads * config.head_dim
        else:
            kl, vl = tuple(k_cache), tuple(v_cache)
            if len(kl) != config.layers or len(vl) != config.layers:
                raise ValueError(f"pool layer counts {(len(kl), len(vl))} != {(config.layers, config.layers)}")
            k0 = kl[0]
            stride_bytes = k0.numel() * k0.element_size()
            for bufs in (kl, vl):
                for l in range(len(bufs) - 1):
                    if bufs[l + 1].data_ptr() - bufs[l].data_ptr() != stride_bytes:
                        raise ValueError("non-uniform pool layer stride (daemon-placed buffers); use attach_megakernel_pool()")
            self.k_cache, self.v_cache = kl, vl
            self.pool_slots = k0.shape[0]
            self._k_launch, self._v_launch = k0, vl[0]
            self._layer_stride = k0.numel()
        self.table = torch.arange(need_slots, device=device, dtype=torch.int64).view(batch, capacity).contiguous()
        self._table_row_lens = [capacity] * batch
        # Persistent PINNED staging for the per-step tokens/positions: the H2D
        # copies stay non_blocking without the pageable-temporary lifetime
        # hazard (a freed CPU source read by a still-queued async copy).
        self._ids_host = torch.zeros(batch, dtype=torch.int64, pin_memory=True)
        self._pos_host = torch.zeros(batch, dtype=torch.int32, pin_memory=True)
        self.ids = torch.zeros(batch, device=device, dtype=torch.int64)
        self.pos = torch.zeros(batch, device=device, dtype=torch.int32)
        self.logits = self.ws[p.logits:p.logits + batch*config.vocab].view(batch, config.vocab)
        self.barrier_counter = torch.zeros(1, device=device, dtype=torch.int64)
        self._bar_base_device = torch.zeros(1, device=device, dtype=torch.int64)

        from .runtime.residency import validate_residency
        self._stream = torch.cuda.current_stream(self.ids.device)
        with torch.cuda.device(self.ids.device):
            self._compiled = self._launch(self.ids, self.pos, warmup=True)
            self.residency_bound = validate_residency(self._compiled, self.workers, self.ids.device)

    # -- kvaas integration --------------------------------------------------

    def set_slot_table(self, slots):
        """Install per-row slot tables (e.g. ``Admission.block_table`` rows).

        Accepts a 1-D table (batch size must be 1) or ``[B, n]`` rows where
        ``slots[b, i]`` is the pool slot holding sequence ``b``'s logical
        position ``i``. Rows are zero-padded to the cache capacity (the
        kernel's row stride is a compile-time constant); padding is never
        read because each row's accesses are masked to its runtime position,
        and :meth:`run` bounds-checks positions against the *logical* row
        lengths recorded here.
        """
        t = slots if torch.is_tensor(slots) else torch.as_tensor(slots, dtype=torch.int64)
        t = t.to(torch.int64) if torch.is_tensor(slots) else t
        if t.dim() == 1:
            if self.batch != 1:
                raise ValueError(f"1-D slot table given for batch {self.batch}; pass [B, n] rows")
            t = t.unsqueeze(0)
        if t.shape[0] != self.batch:
            raise ValueError(f"slot table has {t.shape[0]} rows, batch is {self.batch}")
        if t.numel() == 0:
            raise ValueError("slot table is empty")
        if int(t.max()) >= self.pool_slots:
            raise ValueError(f"slot {int(t.max())} outside pool of {self.pool_slots} slots")
        self._table_row_lens = [t.shape[1]] * self.batch
        cap = self.plan.SCAP
        if t.shape[1] < cap:
            pad = torch.zeros((self.batch, cap - t.shape[1]), dtype=torch.int64)
            t = torch.cat([t, pad], dim=1)
        elif t.shape[1] > cap:
            raise ValueError(f"slot table covers {t.shape[1]} positions > capacity {cap}")
        self.table = t.to(device=self.device, dtype=torch.int64).contiguous()
        return self

    @classmethod
    def from_pool_view(cls, config, view, weights, *, capacity: int, workers: int = 48, dtype=torch.bfloat16, device="cuda", batch: int = 1):
        """Build the megakernel over a kvaas :class:`PoolView`'s buffers.

        The pool must expose per-layer buffers with a uniform layer stride
        (:func:`attach_megakernel_pool` requests exactly that layout: one
        packed buffer per side). Daemon placement of individually planned
        buffers is not uniform, so those are rejected with a clear
        diagnostic rather than silently misaddressed.
        """
        if len(view.k_buffers) != config.layers or len(view.v_buffers) != config.layers:
            raise ValueError(f"pool has {len(view.k_buffers)} layers, config wants {config.layers}")
        if view.n_kv_heads != config.kv_heads or view.head_dim != config.head_dim:
            raise ValueError("pool geometry (kv_heads/head_dim) does not match the config")
        k0 = view.k_buffers[0]
        expect = k0.numel() * k0.element_size()
        for bufs, name in ((view.k_buffers, "k"), (view.v_buffers, "v")):
            for l in range(len(bufs) - 1):
                if bufs[l + 1].data_ptr() - bufs[l].data_ptr() != expect:
                    raise ValueError(f"pool {name} layer strides are not uniform (daemon-placed buffers); use attach_megakernel_pool(), which requests one packed buffer per side")
        return cls(
            config,
            weights,
            capacity=capacity,
            workers=workers,
            dtype=dtype,
            device=device,
            batch=batch,
            k_cache=tuple(view.k_buffers),
            v_cache=tuple(view.v_buffers),
        )

    # -- schedule facts (cross-checkable against the compiled program) ------

    @property
    def num_phases(self) -> int:
        return self.plan.num_phases

    @property
    def num_barriers(self) -> int:
        return self.plan.num_barriers

    @property
    def barrier_base(self) -> int:
        return int(self._bar_base_device[0])

    def task_counts(self) -> dict[str, int]:
        """Task counts from the canonical compiled schedule."""
        aliases = {"embedding lookup": "embedding"}
        for layer in range(self.config.layers):
            for source, short in (("kv cache append", "append"), ("attention scores", "scores"),
                                  ("softmax", "softmax"), ("attention values", "values"),
                                  ("rope q", "rope_q"), ("rope k", "rope_k")):
                aliases[f"layer {layer} {source}"] = f"l{layer}_{short}"
        return {aliases.get(f.op.source_location, f.op.source_location): f.task_count
                for f in self.executable.families}

    # -- launch -------------------------------------------------------------

    def run(self, token_ids, positions, *, check_counter: bool = False) -> torch.Tensor:
        """One batched decode step = exactly one kernel launch (§15.2).

        ``token_ids``/``positions`` are per-row (int/list/tensor); sequence
        ``b`` appends ``token_ids[b]`` at ``positions[b]``. Returns the
        ``[B, V]`` logits. Two tiny H2D copies carry the step's tokens and
        positions (disclosed: one kernel launch + two ≤8·B-byte copies).
        """
        self._check_stream()
        p = self.plan
        ids = self._as_row(token_ids, self.ids)
        pos = self._as_row(positions, self.pos)
        for b in range(self.batch):
            if not (0 <= int(pos[b]) < p.SCAP):
                raise ValueError(f"row {b}: position {int(pos[b])} out of [0, {p.SCAP})")
            if int(pos[b]) >= self._table_row_lens[b]:
                raise ValueError(f"row {b}: position {int(pos[b])} beyond its slot table ({self._table_row_lens[b]} entries)")
        self._ids_host.copy_(ids)
        self._pos_host.copy_(pos)
        # BLOCKING device copies (deliberate): non_blocking=True from the
        # pinned staging raced the kernel queue on this stack and corrupted
        # per-step tokens/positions under deep async pipelines (diagnosed:
        # appends landing in wild slots / K rows zeroed, fully reproducible
        # with sync-per-step). The copies are two <=8*B-byte transfers; the
        # per-step block drains a queue that decode steps sync anyway.
        self.ids.copy_(self._ids_host, non_blocking=False)
        self.pos.copy_(self._pos_host, non_blocking=False)
        self._launch(self.ids, self.pos)
        if check_counter:
            torch.cuda.synchronize()
            got = int(self.barrier_counter[0])
            expected = self.barrier_base
            if got != expected:
                raise RuntimeError(f"barrier counter {got} != expected {expected} (progress/protocol failure)")
        return self.logits

    def _launch(self, ids, pos, *, warmup=False):
        cfg, p = self.config, self.plan
        launch = qwen3_megakernel.warmup if warmup else qwen3_megakernel[(self.workers,)]
        extra = {"grid": (self.workers,)} if warmup else {}
        return launch(
            ids,
            pos,
            self.table,
            self.tok,
            self.final_g,
            self.cos,
            self.sin,
            self.ln1,
            self.qkv_w,
            self.qn,
            self.kn,
            self.op_w,
            self.ln2,
            self.gu_w,
            self.down_w,
            self._k_launch,
            self._v_launch,
            self.ws,
            self.logits,
            self.barrier_counter,
            self._bar_base_device,
            cfg.layers,
            cfg.hidden * ((cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim),
            (cfg.heads * cfg.head_dim) * cfg.hidden,
            cfg.hidden * (2 * cfg.intermediate),
            cfg.intermediate * cfg.hidden,
            cfg.hidden,
            self._layer_stride,
            B=self.batch,
            C=cfg.hidden,
            H=cfg.heads,
            KVH=cfg.kv_heads,
            D=cfg.head_dim,
            F=cfg.intermediate,
            V=cfg.vocab,
            SCAP=p.SCAP,
            QKVW=(cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim,
            HD=cfg.heads * cfg.head_dim,
            F2=2 * cfg.intermediate,
            layout_ptr=self._layout,
            O_EMBED=p.embedding,
            O_FINAL_INPUT=p.final_input,
            O_FINAL=p.final,
            EPS=cfg.rms_eps,
            SCALE=cfg.attention_scale,
            TILE=16,
            BK=256,
            BT=self.btcap,
            BTR=_pow2(p.SCAP),
            ELEM=self.elem,
            num_warps=8,  # §7.1: T = 256 threads per persistent worker
            **extra,
        )

    def run_device(self, token_ids, positions) -> torch.Tensor:
        """Prepared, device-resident decode. No host copies or scalar reads.

        Caller guarantees token IDs are in vocabulary, positions address the
        installed slot table, and all inputs are ready on the current stream.
        Calls on one runner must use its preparation stream; outputs alias its
        workspace and must be consumed before the next step.
        """
        for name, value, expected in (("token_ids", token_ids, self.ids), ("positions", positions, self.pos)):
            if (value.device != self.ids.device or value.dtype != expected.dtype
                    or value.shape != (self.batch,) or not value.is_contiguous()):
                raise ValueError(f"{name} must be contiguous {expected.dtype} [{self.batch}] on {self.ids.device}")
        self._check_stream()
        self._launch(token_ids, positions)
        return self.logits

    def _check_stream(self):
        if torch.cuda.current_stream(self.ids.device) != self._stream:
            raise ValueError("runner belongs to its preparation stream; create a separate runner for concurrency")

    def _as_row(self, val, like):
        t = torch.as_tensor(val)
        t = t.reshape(-1)
        if t.numel() != self.batch:
            raise ValueError(f"expected {self.batch} rows, got {t.numel()}")
        return t.to(like.dtype)


def attach_megakernel_pool(
    socket_path: str,
    engine_id: str,
    gpu: int,
    *,
    layers: int,
    max_tokens: int,
    kv_heads: int,
    head_dim: int,
    dtype_label: str = "torch.bfloat16",
    pool=None,
):
    """Attach a daemon-owned KV pool in the megakernel's packed layout.

    Requests **one packed buffer per side** (``[layers * max_tokens,
    kv_heads, head_dim]``) instead of the floe-canonical per-layer buffer
    plans: the daemon's VMM placement of individually planned buffers is
    not uniformly strided, while a single planned buffer per side gives the
    kernel one base pointer + a uniform layer stride by construction. The
    slot granularity is unchanged (one token's K/V per slot), so
    ``Admission`` slot tables address it identically.

    ``pool`` optionally carries a floe-side ``DensePoolLeases``
    session (the unified BlockTable's dense view). When given, the physical
    attach is unchanged but the session's geometry is validated against the
    pool being attached, so lease extents can never describe a different
    pool than the tensors the megakernel writes. Ships dark: ``None``
    (the default) keeps today's behavior exactly.

    Returns ``(k_tensor, v_tensor)``: non-owning torch views over
    daemon-owned device memory (CUDA IPC import), shape
    ``[layers, max_tokens, kv_heads, head_dim]``.
    """
    import os

    import torch

    from kvaas_runtime import CudaVmm, allocate_device_pool
    from kvaas_runtime import kv_pool_import as kpi
    from kvaas_runtime.kv_pool_import import import_pool_buffers

    if pool is not None:
        if pool.kv_width != kv_heads * head_dim:
            raise ValueError(
                f"lease session kv_width {pool.kv_width} != pool row width {kv_heads * head_dim}"
            )
        if max_tokens % pool.page_tokens:
            raise ValueError(
                f"--max-total ({max_tokens}) must be a whole multiple of the "
                f"{pool.page_tokens}-token dense page"
            )
        if pool.gpu != gpu:
            raise ValueError(f"lease session gpu {pool.gpu} != attach gpu {gpu}")

    itemsize = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}[dtype_label]
    dims = (layers * max_tokens, kv_heads, head_dim)
    plans = [
        kpi.BufferPlan("k.all", dims, itemsize, dtype_label),
        kpi.BufferPlan("v.all", dims, itemsize, dtype_label),
    ]

    class _Deps:
        pass

    deps = _Deps()
    deps.torch = torch
    deps.cuda = CudaVmm(libpath=os.environ.get("KVAAS_CUDA_DRV_PATH"))

    response = allocate_device_pool(
        socket_path,
        engine_id=engine_id,
        gpu=gpu,
        buffers=[(p.name, p.nbytes) for p in plans],
    )
    records = import_pool_buffers(deps, gpu, plans, response.get("buffers") or [])
    by_name = {rec["plan"].name: rec for rec in records}
    if set(by_name) != {"k.all", "v.all"}:
        raise RuntimeError(f"attach_megakernel_pool: unexpected buffers {sorted(by_name)}")
    k = by_name["k.all"]["tensor"].view(layers, max_tokens, kv_heads, head_dim)
    v = by_name["v.all"]["tensor"].view(layers, max_tokens, kv_heads, head_dim)
    # Retain the import records on the tensors so the IPC mappings stay open
    # for the tensors' lifetime (the records own the handles).
    k._kvaas_records, v._kvaas_records = records, records
    return k, v


# ---------------------------------------------------------------------------
# Qwen3.8-27B feasibility building blocks (hybrid GDN target, FP8 weights).
#
# These task bodies implement the pieces the dense-Qwen3 megakernel lacks
# for the 27B target; they are validated standalone (tests/test_megakernel_27b.py)
# against the real checkpoint tensors and the repo's GatedDeltaNet reference:
#
# * ``_t_gemv_fp8`` — GEMV over DeepSeek-style block-quantized weights
#   (F8_E4M3 [N, K] + bf16 ``weight_scale_inv`` [N/128, K/128]; BK == the
#   128-wide quant block, 16-column output tiles as in §6.4);
# * ``_t_gdn_conv`` — the GDN short-conv state update (silu FIR over a
#   [3, C] channel state + the new mixed qkv row);
# * ``_t_gdn_heads`` — the per-value-head gated delta rule (normalize q/k,
#   decay/outer-product state update, per-head RMSNorm + z-gate), one task
#   per value head over its [hv, hk] fp32 state slice.
# ---------------------------------------------------------------------------


















# ---------------------------------------------------------------------------
# MoE decode task templates (issue #98)
#
# * ``_t_moe_expert`` — one task per (row, slot): the expert weight base is
#   loaded from the routing table at run time (the #94 slot-table indirection
#   applied to a read-only weight pool); gate/up row GEMVs as [I, H] block
#   reductions, swiglu_limit clamp folded into the activation, down GEMV as an
#   [H, I] block reduction;
# * ``_t_moe_combine`` — per-row weighted scatter-add over the k slots in slot
#   order (+ the dense shared-expert path when wired);
# * ``_t_moe_route`` — per-row router for the degenerate-group and hash paths
#   (n_group == 1 noaux_tc, sqrtsoftplus global top-k, frozen tid2eid gather):
#   E-wide fp32 logits GEMV, score fn, K rounds of masked ``tl.argmax`` whose
#   first-maximal-index semantics realize the documented tie rule (ties to the
#   lower expert index). The general n_group > 1 noaux_tc group restriction is
#   reference-executor-only in this slice (CUDA-gated gap, flagged in the PR).
#
# ``I``/``H``/``EP`` must be powers of two (``EP >= E``, masked). CPU mirrors
# of these exact task decompositions live in
# ``tests/python/test_moe_decode_compiler.py`` (triton is not importable in
# the bare test environment — §15.1 division of labor).
# ---------------------------------------------------------------------------
# mHC hyper-connection mixing (issue #99): per-token data-dependent
# pre/post/comb weights and stream collapse (pre), then the composed stream
# update (post). Mirrors floe DeepseekV4HyperConnection /
# Glm53HyperConnection exactly; all mixing math fp32 (Sinkhorn numerics
# demand it), no tl.dot — the [mix, hc·C] fn projection is a per-row
# K-reduction GEMV folded into the pre task, and the hc×hc Sinkhorn chain
# is register-resident elementwise work.
# ---------------------------------------------------------------------------
