"""glm_indexer_topk: exact top-k selection vs torch.topk and the stable
oracle.

The op replaces the DSA indexer's ``index_scores.topk(select_k)`` (floe
glm53flash forward.py, ``_index_rows`` / ``_decode_glue_rows`` seams) with
one radix-select launch (histogram the top key byte, find the threshold
bin, refine while the bin overflows the remaining slots, one emit scan).

Parity bar, in order of strictness:

* **Selection SET** must match ``torch.topk`` exactly for distinct values;
  exact ties break deterministically to the lowest index (torch.topk's
  tie order is unspecified — the floe decode path does hit tie clusters:
  invalid pools are masked to ``finfo.min``).
* **Order**: ``sorted=True`` returns values descending with ties
  lowest-index-first (stable descending — the practical CUDA torch.topk
  order), pinned exactly against the ``stable=True`` reference; absent
  exact ties this is index-exact vs ``torch.topk`` too.
* ``sorted=False`` mirrors the ``indexer_topk_unsorted`` knob: same
  selection set, winners then ties in row-scan order.
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_indexer_topk; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# CPU oracle contracts
# ---------------------------------------------------------------------------

def test_reference_matches_torch_on_distinct_values(torch):
    """Distinct values: the stable oracle is index-exact vs torch.topk."""
    from vkernels.torch_ops.glm_indexer_topk import glm_indexer_topk_reference

    torch.manual_seed(3)
    for tokens, pools, k in [(4, 2048, 512), (7, 257, 64), (1, 33, 33), (5, 64, 1)]:
        scores = torch.randn(tokens, pools)  # randn: no exact fp32 ties
        values, indices = glm_indexer_topk_reference(scores, k)
        tv, ti = torch.topk(scores, k, dim=-1)
        assert torch.equal(values, tv)
        assert torch.equal(indices, ti)


def test_reference_tie_breaks_lowest_index(torch):
    """All-equal row: the top-k is exactly {0..k-1}, ties to the lowest index."""
    from vkernels.torch_ops.glm_indexer_topk import glm_indexer_topk_reference

    scores = torch.zeros(2, 16)
    values, indices = glm_indexer_topk_reference(scores, 5)
    assert all(row.tolist() == [0, 1, 2, 3, 4] for row in indices)
    assert torch.equal(values, torch.zeros_like(values))

    # tie cluster straddling the k boundary: k=3 over {10, 10, 9, 9, 9}
    scores = torch.tensor([[10.0, 10.0, 9.0, 9.0, 9.0, 0.0]])
    values, indices = glm_indexer_topk_reference(scores, 3)
    assert indices[0].tolist() == [0, 1, 2]  # two 10s + lowest-index 9

    # +/- 0.0 are one tie class (index tie-break, not -0.0 > +0.0)
    scores = torch.tensor([[+0.0, -0.0, -1.0, -2.0, -3.0]])
    _, indices = glm_indexer_topk_reference(scores, 2)
    assert indices[0].tolist() == [0, 1]


def test_contract(torch):
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_indexer_topk import glm_indexer_topk

    scores = torch.randn(3, 1024)
    with pytest.raises(OpNotEligible, match="2-D"):
        glm_indexer_topk(scores[0], 8)
    with pytest.raises(OpNotEligible, match="k must be within"):
        glm_indexer_topk(scores, 0)
    with pytest.raises(OpNotEligible, match="k must be within"):
        glm_indexer_topk(scores, 1025)
    with pytest.raises(OpNotEligible, match="FP32, BF16 or FP16"):
        glm_indexer_topk(scores.double(), 8)
    with pytest.raises(OpNotEligible, match="CUDA"):
        glm_indexer_topk(scores, 8)  # CPU tensors rejected
    with pytest.raises(OpNotEligible, match="CUDA"):
        glm_indexer_topk(scores.bfloat16(), 8)  # bf16 accepted, CPU not
    # zero tokens: no launch, empty outputs (not an eligibility miss)
    values, indices = glm_indexer_topk(torch.randn(0, 64, device="cuda"), 8) \
        if torch.cuda.is_available() else (torch.empty(0, 8), torch.empty(0, 8, dtype=torch.int32))
    assert values.shape == (0, 8) and indices.shape == (0, 8)


# ---------------------------------------------------------------------------
# GPU parity
# ---------------------------------------------------------------------------

def _gpu(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_indexer_topk import glm_indexer_topk, glm_indexer_topk_reference
    return glm_indexer_topk, glm_indexer_topk_reference


SHAPES = [  # (T, N, k) — the wired envelopes plus degenerate corners
    (4, 2048, 512),    # prefill chunk @ 8k context
    (1, 4096, 512),    # decode, short context
    (16, 32768, 512),  # decode, 128k context
    (7, 8192, 512),    # odd batch, two-scan chunking
    (64, 4096, 512),   # wide decode batch
    (1, 1, 1),         # single element
    (5, 64, 1),        # k = 1
    (2, 33, 33),       # k == N (select everything)
    (4, 128, 127),     # k == N - 1, non-power-of-two BLOCK_K padding
    (3, 257, 64),      # N not a multiple of BLOCK_N
]


def test_gpu_sorted_parity_vs_reference_and_torch(torch):
    """sorted=True: exact values+indices vs the stable oracle AND vs
    torch.topk (randn rows carry no exact ties, so index order is
    well-defined and must match bit-for-bit)."""
    op, ref = _gpu(torch)
    torch.manual_seed(17)
    for tokens, pools, k in SHAPES:
        scores = torch.randn(tokens, pools, device="cuda")
        values, indices = op(scores, k)
        rv, ri = ref(scores, k)
        assert indices.dtype == torch.int32 and values.dtype == torch.float32
        assert torch.equal(values, rv), f"values {tokens}x{pools} k={k}"
        assert torch.equal(indices, ri.long()), f"indices(ref) {tokens}x{pools} k={k}"
        tv, ti = torch.topk(scores, k, dim=-1)
        assert torch.equal(values, tv), f"values vs torch {tokens}x{pools} k={k}"
        assert torch.equal(indices.long(), ti), f"indices vs torch {tokens}x{pools} k={k}"


def test_gpu_sorted_adversarial_ties(torch):
    """Tie-heavy rows: selection sets equal to the oracle's, order exactly
    the oracle's (descending value, ties lowest-index-first)."""
    op, ref = _gpu(torch)
    torch.manual_seed(19)
    cases = []
    # all-equal row: everything ties
    cases.append(torch.full((2, 300), 0.25))
    # quantized scores: a handful of distinct levels, huge tie clusters
    grid = torch.randn(2, 4096).round() * 0.5  # ~9 distinct levels
    cases.append(grid)
    # relu + finfo.min masking with FEWER valid pools than k (the floe
    # short-context decode shape: select_k == n_pools, tail all masked)
    masked = torch.randn(3, 600).relu()
    masked[:, 400:] = torch.finfo(torch.float32).min
    cases.append(masked)
    # +/- 0.0 mixed with distinct values
    zeros = torch.randn(2, 257)
    zeros[0, ::3] = 0.0
    zeros[0, 1::6] = -0.0
    cases.append(zeros)
    for scores in cases:
        k = min(64, scores.shape[1] - 1)
        scores = scores.cuda()
        values, indices = op(scores, k)
        rv, ri = ref(scores, k)
        assert torch.equal(values, rv)
        assert torch.equal(indices, ri.long()), (
            f"tie row mismatch: got {indices[0][:8].tolist()} "
            f"want {ri[0][:8].tolist()}")
        # torch.topk set parity (ties aside for the set itself)
        _, ti = torch.topk(scores, k, dim=-1)
        assert torch.equal(indices.long().sort(-1).values, ti.sort(-1).values)


def test_gpu_unsorted_selection_set(torch):
    """sorted=False (the indexer_topk_unsorted knob's contract): same
    selection set as torch.topk, k distinct in-range indices per row."""
    op, _ = _gpu(torch)
    torch.manual_seed(23)
    for tokens, pools, k in SHAPES:
        scores = torch.randn(tokens, pools, device="cuda")
        values, indices = op(scores, k, sorted=False)
        _, ti = torch.topk(scores, k, dim=-1, sorted=False)
        assert torch.equal(indices.long().sort(-1).values, ti.sort(-1).values)
        assert indices.min() >= 0 and indices.max() < pools
        # per row: k distinct indices
        for row in range(tokens):
            assert len(set(indices[row].tolist())) == k
        # values pair with indices
        assert torch.equal(values, scores.gather(1, indices.long()))


def test_gpu_deterministic_replay(torch):
    """Same inputs -> bit-identical outputs across repeats (the radix
    rounds, emit ranks and bitonic sort are all deterministic)."""
    op, _ = _gpu(torch)
    torch.manual_seed(29)
    scores = torch.randn(7, 8192, device="cuda")
    v0, i0 = op(scores, 512)
    for _ in range(10):
        v, i = op(scores, 512)
        assert torch.equal(v, v0) and torch.equal(i, i0)


def test_gpu_graph_capture_replay(torch):
    """Capture-safe: plain current-stream launch, no host syncs; replay
    reproduces the eager result bit-for-bit and tracks input mutation."""
    op, ref = _gpu(torch)
    torch.manual_seed(31)
    scores = torch.randn(4, 2048, device="cuda")
    eager_v, eager_i = op(scores, 512)  # eager warmup JITs the kernel
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        op(scores, 512)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        captured_v, captured_i = op(scores, 512)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured_v, eager_v)
    assert torch.equal(captured_i, eager_i)
    # replay reads live inputs
    scores.mul_(2.0)
    graph.replay()
    torch.cuda.synchronize()
    rv, ri = ref(scores, 512)
    assert torch.equal(captured_v, rv)
    assert torch.equal(captured_i, ri.long())
    scores.div_(2.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured_v, eager_v)
    assert torch.equal(captured_i, eager_i)


def test_gpu_non_contiguous_rejected(torch):
    from vkernels.torch_ops._dispatch import OpNotEligible

    op, _ = _gpu(torch)
    wide = torch.randn(5, 4096, device="cuda")[:, :2048]
    assert not wide.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        op(wide, 8)


# ---------------------------------------------------------------------------
# bf16 / fp16 (the deployed gemm_dtype=bfloat16 recipe feeds bf16
# index_scores at the floe seam; fp16 is defensive). Both widen to fp32
# in-kernel — exact and monotone — so parity is by construction.
# ---------------------------------------------------------------------------

def test_gpu_bf16_sorted_parity_vs_reference_and_torch(torch):
    """bf16 scores: values+indices EXACTLY the stable oracle run on the
    widened fp32 (widening is lossless, ties included — bf16 randn rows
    DO carry exact ties, the 8-bit mantissa grid), values exactly
    torch.topk's bf16 values, selection sets exactly torch.topk's."""
    op, ref = _gpu(torch)
    torch.manual_seed(37)
    for tokens, pools, k in SHAPES:
        scores = torch.randn(tokens, pools, device="cuda").to(torch.bfloat16)
        values, indices = op(scores, k)
        assert values.dtype == torch.bfloat16 and indices.dtype == torch.int32
        rv, ri = ref(scores, k)  # the oracle widens to fp32 internally
        assert torch.equal(values.float(), rv), f"values {tokens}x{pools} k={k}"
        assert torch.equal(indices, ri.long()), f"indices(ref) {tokens}x{pools} k={k}"
        tv, ti = torch.topk(scores, k, dim=-1)
        assert torch.equal(values, tv), f"values vs torch {tokens}x{pools} k={k}"
        # bf16 randn rows hit exact ties -> torch.topk's tie order is
        # unspecified; only the SET is pinned against it
        assert torch.equal(indices.long().sort(-1).values, ti.sort(-1).values)


def test_gpu_bf16_adversarial_ties(torch):
    """Tie-heavy bf16 rows (the quantization grid makes ties structural:
    coarse mantissa, masked tails are one giant finfo.min cluster).
    Order is pinned exactly against the stable reference."""
    op, ref = _gpu(torch)
    torch.manual_seed(41)
    cases = []
    # all-equal row: everything ties
    cases.append(torch.full((2, 300), 0.25, dtype=torch.bfloat16))
    # coarse grid: quarter-steps -> few levels, huge clusters
    cases.append((torch.randn(2, 4096) * 4).round() / 4)
    # small magnitudes: mantissa steps large relative to the spread
    cases.append(torch.randn(2, 1024) * 1e-3)
    # the floe short-context decode shape: FEWER valid pools than k,
    # invalid tail masked to bf16 finfo.min (floe masks in the scores'
    # own dtype once gemm_dtype=bfloat16 flows through)
    masked = torch.randn(3, 600).relu().to(torch.bfloat16)
    masked[:, 400:] = torch.finfo(torch.bfloat16).min
    cases.append(masked)
    # +/- 0.0 mixed with distinct values
    zeros = torch.randn(2, 257)
    zeros[0, ::3] = 0.0
    zeros[0, 1::6] = -0.0
    cases.append(zeros)
    for scores in cases:
        k = min(64, scores.shape[1] - 1)
        scores = scores.cuda().to(torch.bfloat16)
        values, indices = op(scores, k)
        rv, ri = ref(scores, k)
        assert torch.equal(values.float(), rv)
        assert torch.equal(indices, ri.long()), (
            f"tie row mismatch: got {indices[0][:8].tolist()} "
            f"want {ri[0][:8].tolist()}")
        _, ti = torch.topk(scores, k, dim=-1)
        # Set parity vs torch holds for every class EXCEPT the zero tie
        # class straddling the k boundary: the kernel folds -0.0 onto
        # +0.0 (documented fp32 contract — one tie class, lowest-index
        # break) while torch.topk's radix ranks +0.0 strictly above -0.0
        # by bits. floe's relu'd scores never produce -0.0, so this
        # divergence cannot fire in the wired path; assert the sets
        # differ only within the zero class.
        s_i, s_t = indices.long().sort(-1).values, ti.sort(-1).values
        same = torch.equal(s_i, s_t)
        if not same:
            for row in range(scores.shape[0]):
                only_vk = set(s_i[row].tolist()) - set(s_t[row].tolist())
                only_t = set(s_t[row].tolist()) - set(s_i[row].tolist())
                for j in only_vk | only_t:
                    assert scores[row, j] == 0.0, (
                        f"set diff escapes the zero class: idx {j} "
                        f"val {scores[row, j].item()}")


def test_gpu_bf16_k_edges(torch):
    """k edges on the bf16 grid: k=1, k==N (all-equal row exercises the
    tie-fill invariant m == K - a), k==N-1, N not a multiple of BLOCK_N."""
    op, ref = _gpu(torch)
    torch.manual_seed(43)
    for tokens, pools, k in [(1, 1, 1), (5, 64, 1), (2, 33, 33),
                             (4, 128, 127), (3, 257, 64)]:
        scores = torch.randn(tokens, pools, device="cuda").to(torch.bfloat16)
        values, indices = op(scores, k)
        rv, ri = ref(scores, k)
        assert torch.equal(values.float(), rv)
        assert torch.equal(indices, ri.long())
    # all-equal row, k == N: every slot is a tie, filled lowest-index-first
    scores = torch.full((2, 33), 0.5, device="cuda", dtype=torch.bfloat16)
    values, indices = op(scores, 33)
    assert all(row.tolist() == list(range(33)) for row in indices)
    assert torch.equal(values, scores)


def test_gpu_bf16_nan_documented(torch):
    """NaN on the widened path keeps the fp32 ORDER contract: NaN ranks by
    SIGN BIT in the key (positive NaN above +inf, negative NaN below
    -inf — the widen preserves the sign, as the ranking proves), whereas
    torch.topk ranks every NaN above everything. The returned NaN VALUE
    bits may canonicalize (the fp32->input-dtype narrowing cvt applies
    PTX NaN canonicalization: 0xFFC0 came back as 0x7FFF — cosmetic,
    NaN != NaN for any value-level comparison). floe never feeds NaN
    here (pool logits are nan_to_num'd upstream, invalid pools are
    masked to finfo.min) — this pins the documented divergence so any
    change is a conscious one."""
    op, _ = _gpu(torch)
    scores = torch.tensor(
        [[1.0, float("nan"), float("inf"), float("-inf"),
          float("nan"), 0.5, -0.0]], device="cuda").to(torch.bfloat16)
    scores.view(torch.int16)[0, 4] = 0xFFC0 - (1 << 16)  # negative quiet NaN (bf16 bits 0xFFC0)
    # expected order: +NaN(1) > +inf(2) > 1.0(0) > 0.5(5) > -0.0(6)
    #                 > -inf(3) > -NaN(4)
    values, indices = op(scores, 7)
    assert indices[0].tolist() == [1, 2, 0, 5, 6, 3, 4]
    vbits = values.view(torch.int16)[0]
    assert vbits[0] > 0 and (vbits[0] & 0x7F80) == 0x7F80  # positive NaN on top
    assert (vbits[6] & 0x7F80) == 0x7F80                    # NaN at bottom (payload canonicalized)
    assert vbits[5] == -128  # -inf bits (0xFF80 as int16)
    assert vbits[4] == 0     # -0.0 folded to +0.0 (the zero tie class)
    # deterministic replay (NaN != NaN, compare raw bits)
    v2, i2 = op(scores, 7)
    assert torch.equal(i2, indices) and torch.equal(v2.view(torch.int16)[0], vbits)


def test_gpu_fp16_defensive_parity(torch):
    """fp16 (defensive acceptance; same exact-widen argument as bf16):
    exact vs the stable reference, values and sets vs torch.topk."""
    op, ref = _gpu(torch)
    torch.manual_seed(45)
    for tokens, pools, k in SHAPES:
        scores = torch.randn(tokens, pools, device="cuda").to(torch.float16)
        values, indices = op(scores, k)
        assert values.dtype == torch.float16
        rv, ri = ref(scores, k)
        assert torch.equal(values.float(), rv), f"values {tokens}x{pools} k={k}"
        assert torch.equal(indices, ri.long()), f"indices(ref) {tokens}x{pools} k={k}"
        tv, ti = torch.topk(scores, k, dim=-1)
        assert torch.equal(values, tv)
        assert torch.equal(indices.long().sort(-1).values, ti.sort(-1).values)


def test_gpu_bf16_unsorted_selection_set(torch):
    """sorted=False on bf16: same selection set as torch.topk, k distinct
    in-range indices, values pair with indices in the input dtype."""
    op, _ = _gpu(torch)
    torch.manual_seed(47)
    for tokens, pools, k in SHAPES:
        scores = torch.randn(tokens, pools, device="cuda").to(torch.bfloat16)
        values, indices = op(scores, k, sorted=False)
        assert values.dtype == torch.bfloat16
        _, ti = torch.topk(scores, k, dim=-1, sorted=False)
        assert torch.equal(indices.long().sort(-1).values, ti.sort(-1).values)
        assert indices.min() >= 0 and indices.max() < pools
        for row in range(tokens):
            assert len(set(indices[row].tolist())) == k
        assert torch.equal(values, scores.gather(1, indices.long()))
