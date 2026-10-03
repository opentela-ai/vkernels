"""GLM-5.3 DSA indexer top-k selection (one launch, exact, deterministic).

Replaces the per-step ``index_scores.topk(select_k)`` of the GLM-5.3-Flash
DSA indexer (11 layers/step; trace kin-3498744: the radixSelect +
radixSortKVInPlace pair costs ~19 us/layer, most of it the k-wide sort we
do not need). Algorithm shape ported from the SGLang DSA top-k kernels
(``kernels/aot/include/hip/dsa_topk_coop.cuh``, the exact sibling of
``csrc/elementwise/topk.cu``'s tilelang radix, whose shape in turn follows
vLLM's ``csrc/libtorch_stable/cooperative_topk.cuh`` / sgl-project#23600),
re-expressed in Triton: per row, histogram the top byte of the ordered
fp32 key, find the bin where the running count from the top crosses k,
then refine that bin on the next byte while it still holds more members
than unfilled slots. Elements above the resolved prefix are winners;
members of the final prefix break exactly by lowest index (the register
form of the donor's unbounded-tie fallback — histograms are fixed [256]
regardless of membership, so no tie buffer can overflow). Typical rows
resolve in one histogram round: two passes over the row, against the
three-to-five the donor needs and torch.topk's select+sort pair.

Selection is exact vs ``torch.topk`` for distinct values; exact ties
(which the floe decode path does hit — invalid pools are masked to
``finfo.min``) break deterministically to the lowest index instead of
``torch.topk``'s unspecified tie order. ``sorted=True`` returns values
descending with ties lowest-index-first (stable-descending, the practical
CUDA ``torch.topk`` order); ``sorted=False`` mirrors the free
``indexer_topk_unsorted`` knob — same selection set, winners in row-scan
order then ties in row-scan order, no winner sort.

Sign-bit edge: ``-0.0`` is folded onto ``+0.0`` (exact tie); NaN ranks by
sign bit (positive NaN above +inf, negative NaN below -inf), where
``torch.topk`` ranks all NaN above everything — floe ``nan_to_num``s the
logits, so this never fires in the wired path.

DTYPE: fp32, bf16 and fp16 scores are accepted. Sub-fp32 inputs widen to
fp32 IN-KERNEL (``x.to(tl.float32)`` immediately after each load), not at
the wrapper boundary. Why in-kernel: (1) Exactness — bf16/fp16 -> fp32
widening is exact (a mantissa bit-append, every bf16/fp16 value IS an fp32
value) and monotone, so the fp32 radix key of the widened value is the
ordered key of the INPUT value: bf16 quantization-grid tie classes (the
deployed ``--model-opt gemm_dtype=bfloat16`` recipe quantizes floe's
index_scores to bf16, where ties are MORE likely than fp32) widen to
identical fp32 bit patterns and the lowest-index tie-break applies
unchanged — selection on bf16 is by construction identical to selection
on ``scores.float()``. (2) Cost — a wrapper-boundary ``.float()`` would
add an elementwise cast launch plus an fp32 [T, N] materialization (an
extra launch per indexer layer is a real fraction of the 10-12 us/layer
decode win), while in-kernel widening halves score load traffic (2
bytes/elem vs 4). ``values`` come back in the INPUT dtype — the widened
bits narrow back exactly for every finite value (they came from that
grid); NaN ORDER keeps the sign-carrying key (positive NaN above +inf,
negative below -inf) but the returned NaN VALUE bits may canonicalize
(the narrowing cvt applies PTX NaN canonicalization) — floe never feeds
NaN (nan_to_num upstream), so this is cosmetic. The fp32 path is
bit-identical to before (identity casts are elided at Triton's semantic
layer).

PERF GATE: this op is only worth wiring if it beats BOTH incumbents —
``torch.topk(sorted=True)`` (radixSelect + radixSortKVInPlace) AND the
zero-code ``indexer_topk_unsorted`` flip (``torch.topk(sorted=False)``,
select only). Measure with ``meta/benchmarks/bench_glm_indexer_topk.py``
before adopting; do not ship a kernel that loses to the free knob.
GB10 baseline (CUDA-graph medians, k=512, fp32 scores): 2.0-2.7x vs BOTH
incumbents at bs=1-64 decode and prefill shapes (e.g. bs=1/N=4096: 10-12
us vs 18-30; prefill 2048x32768: 2.2 ms vs 5.8-6.0 ms). GB10 bf16 (the
deployed gemm_dtype recipe, in-kernel widen): the win HOLDS at bs=1
(N=4096: 9.8-11.5 us vs 15.2-23.3; N=16384: 25.2-26.0 vs 41.9-44.8), bs=64
decode (26.2-30.3 vs 34.4-42.2) and 2048-row prefill (1.9-2.0 ms vs
2.06-2.14); the gate LOSES at bs=8/N=32768 (both dtypes — the known
limit below), and bf16-only at bs=32/N=32768 (44.3 vs 42.6 us) and
prefill 512x8192 (130.3 vs 102.1 us): torch's multi-pass radix select is
bandwidth-bound and halves under bf16, while this kernel's per-row
histogram/emit chain is latency/compute-bound (num_warps 4-32 sweeps do
not close it) — the margin narrows as rows shrink. Wire per shape (the
dispatch seam falls back to the incumbent where the gate fails).
KNOWN LIMIT: the one-CTA-per-row design is latency-bound at small batch
x long context (T=8, N=32768: ~50 us — ties torch-sorted, loses to
torch-unsorted's 41 us on GB10, both dtypes); if the serving-GPU bench
confirms that shape matters, the follow-up is a segment-parallel
histogram/emit (multi-CTA per row), not a tuning knob.

Capture-safe contract: static shapes per call, plain current-stream
launch, no host syncs (threshold search is in-kernel), outputs and the
``sorted`` winner-scratch allocated per call through the caching
allocator (graph-pool capture safe). JIT during eager warmup.
Torch/Triton load lazily. Inference-only, no autograd backward.
"""

from functools import lru_cache

from ._dispatch import OpNotEligible


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _indexer_topk(
        SCORES,
        VALUES,
        INDICES,
        SCR_KEY,
        SCR_IDX,
        N,
        K,
        SORTED: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        src = SCORES + row * N
        val = VALUES + row * K
        idx = INDICES + row * K

        # Ordered 32-bit key as a SIGNED int32: positive floats keep their
        # bits (>= 0), negative floats map to ~bits + INT32_MIN (monotone:
        # more negative float -> smaller key). -0.0 folds onto +0.0 so the
        # two zeros tie and the index tie-break applies.
        #
        # PREFIX DOMAIN: every partial-key compare (binning, member test,
        # emit) extracts the top bits of ``ukey = skey ^ INT32_MIN`` — the
        # UNSIGNED radix key — because an extracted prefix of the signed
        # key orders negative floats above positive ones (arithmetic >>
        # sign-extends, and masking alone flips the sign classes). The
        # full-width sort composite keeps the plain signed ``skey``; a
        # signed compare over all 32 bits is already the correct order.
        bins = tl.arange(0, 256)
        one64 = tl.full([], 1, tl.int64)

        a = N * 0  # winners already resolved (strictly above the prefix)
        m = N  # members of the current prefix bin (invariant: a < K <= a + m)
        p = N * 0  # resolved key prefix (8 bits per executed round)
        shift = N * 0 + 32  # unresolved low bits; emit compares key >> shift
        act = N * 0 + 1  # this round still refines (branch-free: sequential
        # runtime ifs inside tl.static_range miscompile loop-carried
        # scalars on Triton 3.8, so rounds are selected with tl.where and
        # the member mask carries the flag — inactive rounds issue no loads)
        for r in tl.static_range(4):
            shift_r = 24 - 8 * r
            pmask = (1 << (8 * r)) - 1  # trace-time: resolved prefix width
            want = K - a
            h = tl.zeros([256], dtype=tl.int32)
            for n0 in tl.range(0, N, BLOCK_N):
                n = n0 + tl.arange(0, BLOCK_N)
                inb = n < N
                x = tl.load(src + n, mask=inb & (act != 0), other=float("-inf"))
                # bf16/fp16 widen to fp32 in-register — exact and
                # monotone, so the key below is the ordered key of the
                # INPUT value; fp32 loads take the identity-cast fast
                # path (see the DTYPE paragraph in the module docstring)
                x = x.to(tl.float32)
                b = x.to(tl.int32, bitcast=True)
                b = tl.where(x == 0.0, 0, b)
                skey = tl.where(b >= 0, b, (~b) + (-2147483648))
                ukey = skey ^ (-2147483648)
                byte = (ukey >> shift_r) & 0xFF
                if r == 0:
                    keep = inb & (act != 0)
                else:
                    member = ((ukey >> (shift_r + 8)) & pmask) == p
                    keep = inb & member & (act != 0)
                h += tl.histogram(byte, 256, mask=keep)
            incl = tl.cumsum(h, 0)
            excl = incl - h  # elements in bins strictly below i
            ge = tl.sum(h, 0) - excl  # elements in bin i or above
            # threshold bin: the LARGEST bin whose at-or-above count
            # still covers the slots (above(b) < want <= above(b)+h[b])
            thr = tl.max(tl.where(ge >= want, bins, -1), 0)
            sel = act != 0
            a = tl.where(sel, a + tl.sum(tl.where(bins == thr, ge - h, 0), 0), a)
            m = tl.where(sel, tl.sum(tl.where(bins == thr, h, 0), 0), m)
            p = tl.where(sel, (p << 8) | thr, p)
            shift = tl.where(sel, shift_r, shift)
            # refine again only while the threshold bin overflows the
            # remaining slots (m == K - a means it fits exactly: stop)
            act = tl.where(sel & (m > K - a), 1, 0)

        # Emit: one scan. Winners (ukey prefix > p) take slots [0, a);
        # prefix members take [a, K) in row-scan (lowest-index) order,
        # which by the stop invariant m == K - a fills them exactly.
        # The sorted variant re-sorts the whole [0, K) block afterwards:
        # early-stopped "ties" are only prefix-equal, so their relative
        # order is still decided by the full key.
        #
        # The prefix compare must be UNSIGNED (at shift == 0 the full
        # ukey has its top bit set for positive floats, and a signed
        # compare would rank every negative float above the threshold).
        # Triton int32 compares are signed, so re-bias both sides by the
        # prefix's top bit: signed compare of the biased values then
        # matches the unsigned compare of the raw prefixes.
        want = K - a
        wbits = (32 - shift).to(tl.int64)
        segmask = ((one64 << wbits) - 1).to(tl.int32)
        bias = (one64 << (wbits - 1)).to(tl.int32)  # the prefix's top bit
        p = p - bias
        wcarry = N * 0
        tcarry = N * 0
        for n0 in tl.range(0, N, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            inb = n < N
            x = tl.load(src + n, mask=inb, other=float("-inf"))
            x = x.to(tl.float32)  # bf16/fp16 widen (exact; see histogram)
            b = x.to(tl.int32, bitcast=True)
            b = tl.where(x == 0.0, 0, b)
            skey = tl.where(b >= 0, b, (~b) + (-2147483648))
            ukey = skey ^ (-2147483648)
            seg = ((ukey >> shift) & segmask) - bias
            is_win = inb & (seg > p)
            is_tie = inb & (seg == p)
            w32 = is_win.to(tl.int32)
            t32 = is_tie.to(tl.int32)
            wslot = wcarry + tl.cumsum(w32, 0) - w32  # exclusive rank
            tslot = tcarry + tl.cumsum(t32, 0) - t32
            take = is_tie & (tslot < want)
            if SORTED:
                tl.store(SCR_KEY + row * BLOCK_K + wslot, skey, mask=is_win)
                tl.store(SCR_IDX + row * BLOCK_K + wslot, n.to(tl.int32), mask=is_win)
                tl.store(SCR_KEY + row * BLOCK_K + a + tslot, skey, mask=take)
                tl.store(SCR_IDX + row * BLOCK_K + a + tslot, n.to(tl.int32), mask=take)
            else:
                # values narrow fp32 -> input dtype on store: exact for
                # bf16/fp16 (the widened bits came from that grid)
                tl.store(idx + wslot, n.to(tl.int32), mask=is_win)
                tl.store(val + wslot, x.to(VALUES.dtype.element_ty), mask=is_win)
                tl.store(idx + a + tslot, n.to(tl.int32), mask=take)
                tl.store(val + a + tslot, x.to(VALUES.dtype.element_ty), mask=take)
            wcarry += tl.sum(w32, 0)
            tcarry += tl.sum(t32, 0)

        if SORTED:
            # The whole selection, descending value with lowest-index
            # tie-break: sort the (key, ~index) composite. Masked lanes load
            # key = INT32_MIN so their composite sorts below every finite
            # key and lands after the K real entries (key 0 would land
            # mid-array: above all negative values).
            kk = tl.arange(0, BLOCK_K)
            sel = kk < K
            sk = tl.load(SCR_KEY + row * BLOCK_K + kk, mask=sel, other=-2147483648)
            si = tl.load(SCR_IDX + row * BLOCK_K + kk, mask=sel, other=0)
            comp = (sk.to(tl.int64) << 32) | (0xFFFFFFFF - si.to(tl.int64))
            comp = tl.sort(comp, descending=True)
            sk2 = (comp >> 32).to(tl.int32)
            si2 = (0xFFFFFFFF - (comp & 0xFFFFFFFF)).to(tl.int32)
            t = sk2 ^ (-2147483648)  # signed key -> ordered unsigned key
            bits = tl.where(t < 0, t ^ (-2147483648), ~t)
            out = bits.to(tl.float32, bitcast=True)
            tl.store(val + kk, out.to(VALUES.dtype.element_ty), mask=sel)
            tl.store(idx + kk, si2, mask=sel)

    return _indexer_topk


def glm_indexer_topk(scores, k, sorted=True):
    """Return ``(values [T, k] in the scores dtype, indices [T, k] int32)``:
    the top-``k`` of the indexer scores ``scores`` ``[T, N]`` along the
    last axis.

    fp32, bf16 and fp16 inputs are accepted; sub-fp32 inputs widen to fp32
    in-kernel (exact, monotone — see the module docstring's DTYPE
    paragraph), so selection on a bf16 row is by construction identical to
    selection on its fp32 widening. ``values`` are returned in the input
    dtype (the exact bf16/fp16 round-trip of the widened bits), matching
    ``torch.topk``'s dtype contract for drop-in wiring.

    ``sorted=True`` (default) returns values descending, ties broken to the
    lowest index — the practical CUDA ``torch.topk`` order, so the
    selection SET and, absent exact ties, the index ORDER match
    ``index_scores.topk(k)``. ``sorted=False`` returns the same selection
    set in scan order (the ``indexer_topk_unsorted`` knob's contract).

    Indices are int32 (house convention; the floe seam's ``gather`` wants
    int64, so wiring adds one ``.long()`` on a ``[T, 512]`` tensor).

    Eligibility (checked here; callers fall back on :class:`OpNotEligible`):
    CUDA fp32/bf16/fp16 contiguous 2-D ``[T, N]`` scores with ``1 <= k <= N``
    (other dtypes — e.g. fp64 — are rejected; bf16 is the deployed
    ``gemm_dtype=bfloat16`` recipe's index_scores dtype, fp16 is defensive).
    """
    import torch

    if scores.ndim != 2:
        raise OpNotEligible("index scores must be 2-D [tokens, pools]")
    tokens, pools = scores.shape
    if not (1 <= k <= pools):
        raise OpNotEligible(f"k must be within [1, pools]; got k={k}, pools={pools}")
    if scores.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise OpNotEligible(
            f"index scores must be FP32, BF16 or FP16; got {scores.dtype}")
    if not scores.is_cuda:
        raise OpNotEligible(
            "index scores must be CUDA-resident (CPU callers take the "
            "*_reference path)")
    if not scores.is_contiguous():
        raise OpNotEligible("index scores must be contiguous (caller owns the view)")
    values = torch.empty(tokens, k, device=scores.device, dtype=scores.dtype)
    indices = torch.empty(tokens, k, device=scores.device, dtype=torch.int32)
    if tokens:
        import triton

        block_n = min(4096, triton.next_power_of_2(pools))
        block_k = max(2, triton.next_power_of_2(k))
        if sorted:
            scr_key = torch.empty(tokens, block_k, device=scores.device, dtype=torch.int32)
            scr_idx = torch.empty(tokens, block_k, device=scores.device, dtype=torch.int32)
        else:
            scr_key = scr_idx = indices  # dead args; never stored to
        _kernel()[(tokens,)](
            scores,
            values,
            indices,
            scr_key,
            scr_idx,
            pools,
            k,
            sorted,
            block_n,
            block_k,
            # GB10-tuned: deep scans (N > 16k) want the extra warps for
            # latency; small rows prefer 4-8. Re-tune on the target GPU.
            num_warps=16 if pools > 16384 else (8 if block_n >= 2048 else 4),
        )
    return values, indices


def glm_indexer_topk_reference(scores, k):
    """Eager oracle: stable-descending top-k (values descending, ties to
    the lowest index) — the kernel's ``sorted=True`` contract exactly."""
    import torch

    order = torch.argsort(scores.float(), dim=-1, descending=True, stable=True)
    indices = order[:, :k]
    return scores.float().gather(1, indices), indices
