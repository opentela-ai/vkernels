"""Prepared, single-launch block-FP8 routed MLP for bounded CUDA cohorts.

This is a compute region, not a serving scheduler. The owner retains its
weights and workspace until physical completion, submits on one stream, and
consumes the borrowed output before the next invocation. No KV, collective,
admission or publication state is owned here.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from ._dispatch import OpNotEligible


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice

    from vkernels.compiler.device_triton import grid_barrier

    @triton.jit
    def routing(IDS, RW, TOK, INV, SO, TE, TR, TM, LIVE,
                S: tl.constexpr, KTOP: tl.constexpr, E: tl.constexpr,
                BS: tl.constexpr, BE: tl.constexpr, CAP: tl.constexpr,
                BC: tl.constexpr, BM: tl.constexpr, SKIP: tl.constexpr,
                STEPS: tl.constexpr):
        original = tl.arange(0, BS)
        expert = tl.load(IDS + original, original < S, E).to(tl.int32)
        if SKIP:
            weight = tl.load(RW + original, original < S, 0)
            expert = tl.where((expert >= 0) & (weight != 0), expert, E)
        # The original slot is an explicit tie-break, matching stable argsort.
        key = tl.where(original < S, expert * (S + 1) + original, 2147483647)
        key = tl.sort(key, descending=False)
        sorted_original = key % (S + 1)
        sorted_expert = key // (S + 1)
        tl.store(TOK + original, sorted_original // KTOP, original < S)
        tl.store(INV + sorted_original, original, original < S)
        # Sorted position -> original slot: the SGL down GEMM weights each
        # expert-sorted row by the route weight of its original slot.
        tl.store(SO + original, sorted_original, original < S)
        histogram = tl.histogram(sorted_expert, BE, mask=(original < S) & (sorted_expert < E))
        ex = tl.arange(0, BE)
        counts = tl.where(ex < E, histogram, 0)
        segments = tl.cumsum(counts) - counts
        tiles = (counts + BM - 1) // BM
        ends = tl.cumsum(tiles)
        offsets = ends - tiles
        tl.store(LIVE, tl.sum(counts))
        tid = tl.arange(0, BC)
        lo = tl.full((BC,), 0, tl.int32)
        hi = tl.full((BC,), E, tl.int32)
        for _ in range(STEPS):
            mid = (lo + hi) // 2
            end = tl.gather(ends, tl.minimum(mid, E - 1), 0)
            right = (mid < E) & (end <= tid)
            active = lo < hi
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
        expert = tl.minimum(lo, E - 1)
        local = tid - tl.gather(offsets, expert, 0)
        count = tl.gather(counts, expert, 0)
        start = tl.gather(segments, expert, 0) + local * BM
        size = tl.minimum(tl.maximum(count - local * BM, 0), BM)
        valid = tid < tl.sum(tiles)
        tl.store(TE + tid, expert, tid < CAP)
        tl.store(TR + tid, tl.where(valid, start, 0), tid < CAP)
        tl.store(TM + tid, tl.where(valid, size, 0), tid < CAP)

    @triton.jit
    def quantize(task, X, Q, QS, WIDTH: tl.constexpr, SGL: tl.constexpr):
        row = task // (WIDTH // 128)
        group = task % (WIDTH // 128)
        col = group * 128 + tl.arange(0, 128)
        x = tl.load(X + row * WIDTH + col).to(tl.float32)
        if SGL:
            # SGL donor: amax floor 1e-10, reciprocal multiply, no scale floor.
            maximum = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-10)
            scale = maximum / 448.0
            q = tl.minimum(tl.maximum(x * (1.0 / scale), -448.0), 448.0)
        else:
            maximum = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-4)
            scale = tl.maximum(libdevice.mul_rn(maximum, 1.0 / 448.0), 1.0e-12)
            q = tl.minimum(tl.maximum(tl.div_rn(x, scale), -448.0), 448.0)
        tl.store(Q + row * WIDTH + col, q)
        tl.store(QS + task, scale)

    @triton.jit
    def gemm(task, A, SA, W, SW, OUT, TOK, TE, TR, TM, RW, INV, SO,
             N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BY_TOKEN: tl.constexpr, WEIGHT: tl.constexpr):
        tile = task // tl.cdiv(N, BN)
        column = task % tl.cdiv(N, BN)
        expert = tl.load(TE + tile, cache_modifier=".cg").to(tl.int64)
        start = tl.load(TR + tile, cache_modifier=".cg")
        count = tl.load(TM + tile, cache_modifier=".cg")
        if count > 0:
            row = start + tl.arange(0, BM)
            mask = tl.arange(0, BM) < count
            source = tl.load(TOK + row, mask, 0, cache_modifier=".cg").to(tl.int64) if BY_TOKEN else row
            col = column * BN + tl.arange(0, BN)
            acc = tl.zeros((BM, BN), tl.float32)
            for group in range(K // 128):
                kk = group * 128 + tl.arange(0, 128)
                a = tl.load(A + source[:, None] * K + kk[None, :], mask[:, None], 0.0,
                            cache_modifier=".cg")
                b = tl.load(W + expert * N * K + col[:, None] * K + kk[None, :],
                            col[:, None] < N, 0.0)
                sa = tl.load(SA + source * (K // 128) + group, mask, 0, cache_modifier=".cg")
                sb = tl.load(SW + expert * (N // 128) * (K // 128)
                             + (col // 128) * (K // 128) + group, col < N, 0)
                block = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
                # Same block order and fused accumulation as grouped GEMM.
                acc += block * (sa[:, None] * sb[None, :])
            if WEIGHT:
                # SGL donor applies the route weight in fp32 before the BF16
                # store; rows are expert-sorted, so map back to the original
                # slot to fetch the right route weight.
                original = tl.load(SO + row, mask, 0, cache_modifier=".cg")
                weight = tl.load(RW + original, mask, 0.0).to(tl.float32)
                acc = acc * weight[:, None]
            tl.store(OUT + row[:, None] * N + col[None, :], acc.to(tl.bfloat16),
                     mask[:, None] & (col[None, :] < N))

    @triton.jit
    def activate_quantize(task, GU, Q, QS, LIVE, I: tl.constexpr, LIMIT: tl.constexpr,
                          SGL: tl.constexpr):
        # Four independent groups per task; the maximum still reduces only
        # the original 128 elements of each group.
        flat_group = task * 4 + tl.arange(0, 4)
        row = flat_group // (I // 128)
        group = flat_group % (I // 128)
        col = group[:, None] * 128 + tl.arange(0, 128)[None, :]
        valid = row < tl.load(LIVE, cache_modifier=".cg")
        gate = tl.load(GU + row[:, None] * (2 * I) + col, valid[:, None], 0,
                       cache_modifier=".cg").to(tl.float32)
        up = tl.load(GU + row[:, None] * (2 * I) + I + col, valid[:, None], 0,
                     cache_modifier=".cg").to(tl.float32)
        if SGL:
            # SGL donor rounds every clamp/SiLU/product boundary through BF16.
            gate = tl.minimum(gate, LIMIT, propagate_nan=tl.PropagateNan.ALL).to(tl.bfloat16).to(tl.float32)
            up = tl.minimum(tl.maximum(up, -LIMIT, propagate_nan=tl.PropagateNan.ALL),
                            LIMIT, propagate_nan=tl.PropagateNan.ALL).to(tl.bfloat16).to(tl.float32)
            silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
            x = (silu * up).to(tl.bfloat16).to(tl.float32)
            maximum = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
            scale = maximum / 448.0
            q = tl.minimum(tl.maximum(x * (1.0 / scale[:, None]), -448.0), 448.0)
        else:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
            silu = tl.div_rn(gate, libdevice.add_rn(1.0, libdevice.exp(-gate)))
            # Virtual eager activation store: BF16 round precedes FP8 amax.
            x = libdevice.mul_rn(silu, up).to(tl.bfloat16).to(tl.float32)
            maximum = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-4)
            scale = tl.maximum(libdevice.mul_rn(maximum, 1.0 / 448.0), 1.0e-12)
            q = tl.minimum(tl.maximum(tl.div_rn(x, scale[:, None]), -448.0), 448.0)
        tl.store(Q + row[:, None] * I + col, q, valid[:, None])
        tl.store(QS + flat_group, scale, valid)

    @triton.jit
    def combine(task, DN, INV, IDS, RW, OUT, H: tl.constexpr,
                KTOP: tl.constexpr, SKIP: tl.constexpr, ROUND: tl.constexpr,
                WEIGHTED: tl.constexpr):
        token = task // tl.cdiv(H, 256)
        col = task % tl.cdiv(H, 256) * 256 + tl.arange(0, 256)
        acc = tl.zeros((256,), tl.float32)
        for slot in range(KTOP):
            original = token * KTOP + slot
            row = tl.load(INV + original, cache_modifier=".cg")
            weight = tl.load(RW + original).to(tl.float32)
            live = tl.full((), True, tl.int1)
            if SKIP:
                live = (tl.load(IDS + original) >= 0) & (weight != 0)
            value = tl.load(DN + row * H + col, (col < H) & live, 0, cache_modifier=".cg").to(tl.float32)
            if WEIGHTED:
                product = libdevice.mul_rn(value, weight)
                if ROUND:
                    product = product.to(tl.bfloat16).to(tl.float32)
                acc = libdevice.add_rn(acc, product)
            else:
                # SGL donor: the route weight already lives in the down store,
                # so the final reduction is a plain fp32 sum over k.
                acc = libdevice.add_rn(acc, value)
        tl.store(OUT + token * H + col, acc, col < H)

    @triton.jit
    def region(X, GUW, GUS, DNW, DNS, IDS, RW, TOK, INV, SO, TE, TR, TM,
               LIVE, Q1, S1, GU, Q2, S2, DN, OUT, COUNTER, BASE,
               T: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
               E: tl.constexpr, KTOP: tl.constexpr, P: tl.constexpr,
               CAP: tl.constexpr, BS: tl.constexpr, BE: tl.constexpr,
               BC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
               STEPS: tl.constexpr, LIMIT: tl.constexpr,
               SKIP: tl.constexpr, ROUND: tl.constexpr, SGL: tl.constexpr,
               WEIGHT_DOWN: tl.constexpr, WEIGHT_COMBINE: tl.constexpr):
        worker = tl.program_id(0)
        base = tl.load(BASE)
        task = worker
        while task < 1 + T * (H // 128):
            if task == 0:
                routing(IDS, RW, TOK, INV, SO, TE, TR, TM, LIVE, T * KTOP,
                        KTOP, E, BS, BE, CAP, BC, BM, SKIP, STEPS)
            else:
                quantize(task - 1, X, Q1, S1, H, SGL)
            task += P
        grid_barrier(COUNTER, base + P)
        task = worker
        while task < CAP * tl.cdiv(2 * I, BN):
            gemm(task, Q1, S1, GUW, GUS, GU, TOK, TE, TR, TM, RW, INV, SO,
                 2 * I, H, BM, BN, True, False)
            task += P
        grid_barrier(COUNTER, base + 2 * P)
        task = worker
        # Only expert-sorted live rows feed the down GEMM. The tail is never
        # read by down/combine, so its quantization needs no work or fill.
        activation_tasks = tl.cdiv(tl.load(LIVE, cache_modifier=".cg") * (I // 128), 4)
        while task < activation_tasks:
            activate_quantize(task, GU, Q2, S2, LIVE, I, LIMIT, SGL)
            task += P
        grid_barrier(COUNTER, base + 3 * P)
        task = worker
        while task < CAP * tl.cdiv(H, BN):
            gemm(task, Q2, S2, DNW, DNS, DN, TOK, TE, TR, TM, RW, INV, SO,
                 H, I, BM, BN, False, WEIGHT_DOWN)
            task += P
        grid_barrier(COUNTER, base + 4 * P)
        task = worker
        while task < T * tl.cdiv(H, 256):
            combine(task, DN, INV, IDS, RW, OUT, H, KTOP, SKIP, ROUND, WEIGHT_COMBINE)
            task += P
        if worker == 0:
            tl.store(BASE, base + 4 * P)

    return triton, region


class PersistentMoE:
    """Prepare one fixed-shape routed MLP; ``run`` issues one compute kernel.

    Finite inputs/scales/weights and valid local expert ids are the caller's
    contract. Negative sentinels are accepted only with ``skip_zero_weights``.
    All tensors must be contiguous. Weights are borrowed immutable storage.
    Preparation compiles and checks *actual* kernel occupancy before launch;
    it must happen before capture. Every run/replay must use the preparation
    stream and retain this object through completion. Output is overwritten
    on the next run. Separate concurrent regions need separate instances.

    ``numerics='native'`` (default) keeps the block-FP8 grouped-prefill
    boundaries (amax floor ``1e-4``) and rounds the route weight in the
    combine. ``numerics='sgl_decode'`` keeps the SGLang decode donor's
    boundaries instead (amax floor ``1e-10``, reciprocal multiply, BF16
    round-trips at every clamp/SiLU/product stage, route weight applied in
    fp32 by the down GEMM before its BF16 store, and a plain fp32 reduction),
    with a small-row tile and EP zero-route skipping for decode geometry.
    """

    def __init__(self, x, gate_up, gate_scales, down, down_scales, indices, weights,
                 *, swiglu_limit=7.0, skip_zero_weights=False,
                 round_products_to_bf16=False, workers=None, blocks_per_sm=1,
                 numerics="native", block_m=None):
        if numerics not in ("native", "sgl_decode"):
            raise ValueError("numerics must be 'native' or 'sgl_decode'")
        if numerics == "sgl_decode" and round_products_to_bf16:
            raise ValueError(
                "sgl_decode already keeps the route weight in the down store; "
                "round_products_to_bf16 would double-round")
        tensors = x, gate_up, gate_scales, down, down_scales, indices, weights
        if not (
            x.is_cuda and torch.version.hip is None
            and all(t.device == x.device and t.is_contiguous() for t in tensors)
            and x.ndim == indices.ndim == weights.ndim == 2
            and x.dtype == torch.bfloat16 and indices.dtype == torch.int64
            and weights.dtype == torch.float32 and indices.shape == weights.shape
            and x.shape[0] == indices.shape[0]
            and gate_up.ndim == down.ndim == 3
            and gate_up.dtype == down.dtype == torch.float8_e4m3fn
            and gate_scales.dtype == down_scales.dtype == torch.float32
            and all(not (torch.is_grad_enabled() and t.requires_grad) for t in tensors)
        ):
            raise OpNotEligible("persistent MoE requires same-device contiguous CUDA BF16/FP8 inference tensors")
        t, h = x.shape
        e, two_i, wh = gate_up.shape
        i = two_i // 2
        top_k = indices.shape[1]
        slots = t * top_k
        if not (
            0 < slots <= 4096 and 0 < e <= 1024 and 0 < top_k <= 32
            and h > 0 and i > 0 and h % 128 == i % 128 == two_i % 2 == 0
            and wh == h and down.shape == (e, h, i)
            and gate_scales.shape == (e, two_i // 128, h // 128)
            and down_scales.shape == (e, h // 128, i // 128)
            and math.isfinite(swiglu_limit) and swiglu_limit > 0
        ):
            raise OpNotEligible("persistent MoE requires block-128 stacks and 1..4096 route slots")
        with torch.cuda.device(x.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare persistent MoE before CUDA graph capture")
        self.device = x.device
        self.stream = torch.cuda.current_stream(x.device)
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        if blocks_per_sm not in (1, 2):
            raise ValueError("blocks_per_sm must be 1/2")
        self.workers = sm_count * blocks_per_sm if workers is None else workers
        if not isinstance(self.workers, int) or not 1 <= self.workers <= sm_count * blocks_per_sm:
            raise ValueError("persistent workers exceed the requested per-SM limit")
        self.weights = gate_up, gate_scales, down, down_scales
        self._weight_spec = tuple((tuple(v.shape), v.dtype) for v in self.weights)
        self._spec = tuple((tuple(v.shape), v.dtype) for v in (x, indices, weights))
        self._input_alignment = tuple(v.data_ptr() % 16 for v in (x, indices, weights))
        triton, self._kernel = _kernels()
        sgl = numerics == "sgl_decode"
        self.numerics = numerics
        # Prefill keeps the qualified BM64 tile. Decode keeps the SGL donor's
        # small-row tile so the block accumulation order is unchanged.
        bm = (16 if sgl else 64) if block_m is None else int(block_m)
        bn = 128
        cap = min(slots, slots // bm + e + 1)
        self._constants = dict(T=t, H=h, I=i, E=e, KTOP=top_k, P=self.workers,
                               CAP=cap, BS=triton.next_power_of_2(slots),
                               BE=triton.next_power_of_2(e + 1), BC=triton.next_power_of_2(cap),
                               BM=bm, BN=bn, STEPS=e.bit_length() + 1,
                               LIMIT=float(swiglu_limit), SKIP=bool(skip_zero_weights),
                               ROUND=bool(round_products_to_bf16),
                               SGL=sgl, WEIGHT_DOWN=sgl, WEIGHT_COMBINE=not sgl,
                               num_warps=4, num_stages=3,
                               launch_cooperative_grid=True)
        def empty(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=x.device)
        self.tokens = empty(slots, torch.int64)
        self.inverse = empty(slots, torch.int64)
        self.order = empty(slots, torch.int64)
        self.tile_expert, self.tile_start, self.tile_count = (empty(cap, torch.int64) for _ in range(3))
        self.live = empty((), torch.int32)
        self.q1 = empty((t, h), torch.float8_e4m3fn)
        self.s1 = empty((t, h // 128), torch.float32)
        self.gate_up = empty((slots, two_i), torch.bfloat16)
        self.q2 = empty((slots, i), torch.float8_e4m3fn)
        self.s2 = empty((slots, i // 128), torch.float32)
        self.down = empty((slots, h), torch.bfloat16)
        self.output = empty((t, h), torch.bfloat16)
        self.counter = torch.zeros((), dtype=torch.int64, device=x.device)
        self.base = torch.zeros_like(self.counter)
        with torch.cuda.device(x.device):
            self.compiled = self._kernel.warmup(*self._args(x, indices, weights),
                                                **self._constants, grid=(self.workers,))
            from vkernels.compiler.runtime.residency import validate_residency
            self.residency_bound = validate_residency(self.compiled, self.workers, x.device,
                                                     blocks_per_sm=blocks_per_sm)

    def _args(self, x, indices, weights, expert_weights=None):
        return (x, *(self.weights if expert_weights is None else expert_weights),
                indices, weights, self.tokens, self.inverse, self.order,
                self.tile_expert, self.tile_start, self.tile_count, self.live,
                self.q1, self.s1, self.gate_up, self.q2, self.s2, self.down,
                self.output, self.counter, self.base)

    def run(self, x, indices, weights, *, expert_weights=None):
        """Submit on the preparation stream; return borrowed BF16 output.

        Optional expert weights are runtime inputs with the prepared geometry.
        Serialized layers can share this workspace; callers retain their
        current weights through completion and consume output before reuse.
        """
        for tensor, (shape, dtype), alignment in zip((x, indices, weights), self._spec, self._input_alignment):
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                    or tensor.device != self.device or not tensor.is_contiguous()
                    or tensor.data_ptr() % 16 != alignment
                    or (torch.is_grad_enabled() and tensor.requires_grad)):
                raise OpNotEligible("persistent region input differs from its prepared contract")
        if expert_weights is not None:
            if len(expert_weights) != len(self._weight_spec):
                raise OpNotEligible("persistent region requires four expert weight tensors")
            for tensor, initial, (shape, dtype) in zip(expert_weights, self.weights, self._weight_spec):
                if (tuple(tensor.shape) != shape or tensor.dtype != dtype
                        or tensor.device != self.device or not tensor.is_contiguous()
                        or tensor.data_ptr() % 16 != initial.data_ptr() % 16
                        or (torch.is_grad_enabled() and tensor.requires_grad)):
                    raise OpNotEligible("persistent region weights differ from its prepared contract")
        if torch.cuda.current_stream(self.device) != self.stream:
            raise RuntimeError("persistent MoE workspace requires its preparation stream")
        with torch.cuda.device(self.device):
            self._kernel[(self.workers,)](*self._args(x, indices, weights, expert_weights), **self._constants)
        return self.output
