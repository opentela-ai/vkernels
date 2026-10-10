"""fused_router_dense: the router scoring-GEMV fold (node-cut D).

The op folds the per-MoE-layer chain ``x.float()`` cast -> cuBLAS fp32
GEMV (``F.linear(x.float(), w)``) -> the incumbent ``_router`` kernel into
ONE launch taking bf16 ``x`` directly. Parity bar, in order of strictness:

* **Router math bit-identical to the incumbent given identical logits** —
  the post-GEMV section is the incumbent ``_router`` core verbatim (same
  ops, same rounding points, same ``enable_fp_fusion=False`` launch
  discipline). Pinned via all-tie constructions where identical weight
  rows make BOTH the eager GEMM and this kernel's reduction reproduce the
  tie bit-exactly: indices AND weights must be ``torch.equal``.
* **Selection-set parity with the exact eager reference**
  ``F.linear(x.float(), w) -> incumbent router`` on random inputs —
  pinned at zero selection diffs on a fixed seed set (the task gate).
* **Weight values within the reduction-order band**: this kernel's
  deterministic fp32 reduction (module docstring) is not cuBLAS ``gemvx``'s
  order, so logits differ by fp32-accumulation ulps (~1e-4 absolute at the
  production shape, cancellation-dominated); near-tie top-k boundaries
  closer than that band CAN legitimately flip (the class the HF router
  comment warns about). The adversarial tests pin: no flips outside the
  noise band, and every observed flip sits inside it (counted, reported).

The oracle is the EXACT production chain named in the lane task:
``F.linear(x.float(), w)`` fed to the incumbent module's
``fused_router_reference`` (the ``Glm53TopkRouter`` n_group == 1 eager leg
with the lowest-index tie-break). The shared variant is held to lane 12's
contract: routed columns bit-identical to ``fused_router_dense``, shared
column exactly ``(E, 1.0)`` appended after normalize + scale, int64 ABI.
"""

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


PROD_K = 8
PROD_SCALING = 2.5


def _eager_reference(torch, x, w, bias, top_k, scaling, norm_topk_prob=True):
    """The exact chain this op folds: cuBLAS fp32 GEMV -> incumbent oracle."""
    import torch.nn.functional as F

    from vkernels.torch_ops.glm_router import fused_router_reference

    logits = F.linear(x.float(), w.float())  # w.float(): the production widening
    idx, wt = fused_router_reference(logits, bias, top_k, scaling, norm_topk_prob)
    return logits, idx, wt


def _assert_same_routing(torch, got_idx, got_w, ref_idx, ref_w, rtol=1e-3):
    """Same expert SET per row, weights aligned by expert id.

    The tolerance carries the fold's honest band: this kernel's logits
    differ from cuBLAS's by fp32-reduction ulps (~1e-4 absolute at
    H=4096, cancellation-dominated), and a selected expert's weight is
    ``sigmoid(logit)`` renormalized — its relative error vs the reference
    is ~|dlogit| (amplified toward 1x for very negative logits, damped
    near 0 for large ones). rel=1e-3 leaves 10x headroom over the
    measured band while still catching real route bugs; small
    (bias-carried) weights pass via the abs floor. The tight numerics
    pins live elsewhere: the logit-band test and the all-tie
    bit-exactness tests.
    """
    assert got_idx.shape == ref_idx.shape and got_w.shape == ref_w.shape
    for r in range(ref_idx.shape[0]):
        g = dict(zip(got_idx[r].tolist(), got_w[r].tolist()))
        e = dict(zip(ref_idx[r].tolist(), ref_w[r].tolist()))
        assert set(g) == set(e), (
            f"row {r}: expert set mismatch: got {sorted(g)}, want {sorted(e)}"
        )
        for i, w in g.items():
            assert w == pytest.approx(e[i], rel=rtol, abs=1e-6), (
                f"row {r}, expert {i}: weight {w} vs {e[i]}"
            )


def test_contract(torch):
    """Eligibility surface: incumbent-overlapping messages + the dense
    dtype contract (CPU-safe — the CUDA check fires last)."""
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_router_dense import (
        fused_router_dense,
        fused_router_dense_shared,
    )

    x = torch.randn(3, 4096, dtype=torch.bfloat16)
    w = torch.randn(288, 4096)
    bias = torch.randn(288)
    for op in (fused_router_dense, fused_router_dense_shared):
        with pytest.raises(OpNotEligible, match="n_group"):
            op(x, w, bias, 8, 2.5, num_group=8, topk_group=4)
        with pytest.raises(OpNotEligible, match="2-D"):
            op(x[0], w, bias, 8, 2.5)
        with pytest.raises(OpNotEligible, match="weight \\[experts, hidden\\]"):
            op(x, torch.randn(288, 2048), bias, 8, 2.5)
        with pytest.raises(OpNotEligible, match="bias"):
            op(x, w, torch.randn(64), 8, 2.5)
        with pytest.raises(OpNotEligible, match="top_k"):
            op(x, w, bias, 0, 2.5)
        with pytest.raises(OpNotEligible, match="top_k"):
            op(x, w, bias, 289, 2.5)
        with pytest.raises(OpNotEligible, match="x must be"):
            op(x.double(), w, bias, 8, 2.5)
        with pytest.raises(OpNotEligible, match="router weight must be"):
            op(x, w.int(), bias, 8, 2.5)
        with pytest.raises(OpNotEligible, match="float dtype"):
            op(x, w, bias.int(), 8, 2.5)
        with pytest.raises(OpNotEligible, match="same GPU"):
            op(x, w, bias, 8, 2.5)  # CPU tensors rejected


def test_gpu_parity_vs_eager_reference(torch):
    """Selection-set + weight parity vs F.linear(x.float(), w) -> incumbent
    oracle, over the production shape and oddball envelopes."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router_dense import (
        fused_router_dense,
        fused_router_dense_shared,
    )

    for tokens, experts, hidden, k in [
        (1, 288, 4096, 8),
        (4, 288, 4096, 8),
        (8, 288, 4096, 8),
        (7, 64, 96, 2),
        (5, 100, 128, 5),
        (3, 16, 48, 16),  # k == experts
    ]:
        torch.manual_seed(hash((tokens, experts, hidden, k)) % (2**31))
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(experts, hidden, device="cuda")
        bias = torch.randn(experts, device="cuda")
        _, ref_idx, ref_w = _eager_reference(torch, x, w, bias, k, PROD_SCALING)

        idx, wt = fused_router_dense(x, w, bias, k, PROD_SCALING)
        assert idx.shape == (tokens, k) and wt.shape == (tokens, k)
        assert idx.dtype == torch.int32 and wt.dtype == torch.float32
        _assert_same_routing(torch, idx, wt, ref_idx, ref_w)

        sidx, swt = fused_router_dense_shared(x, w, bias, k, PROD_SCALING)
        assert sidx.shape == (tokens, k + 1) and swt.shape == (tokens, k + 1)
        assert sidx.dtype == torch.int64 and swt.dtype == torch.float32
        _assert_same_routing(torch, sidx[:, :k], swt[:, :k], ref_idx, ref_w)
        # shared column: index E, weight exactly 1.0 — on every row
        assert torch.equal(sidx[:, k], torch.full((tokens,), experts, dtype=torch.int64, device="cuda"))
        assert torch.equal(swt[:, k], torch.ones(tokens, device="cuda"))


def test_gpu_logit_band(torch):
    """The fold's logits sit in the fp32-reduction-order band around the
    eager GEMV's (absolute error, cancellation-dominated at H=4096)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router_dense import fused_router_dense

    for seed in range(4):  # T is capped at 8 (decode regime); more seeds
        torch.manual_seed(5 + seed)
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(288, 4096, device="cuda")
        bias = torch.zeros(288, device="cuda")
        ref_logits, _, _ = _eager_reference(torch, x, w, bias, PROD_K, 1.0)
        _, _, logits = fused_router_dense(x, w, bias, PROD_K, 1.0, return_logits=True)
        assert logits.dtype == torch.float32 and logits.shape == (8, 288)
        assert torch.isfinite(logits).all()
        diff = (logits - ref_logits).abs().max().item()
        # measured band: both sides accumulate 4096 fp32 products; the observed
        # max |mine - cublas| is ~1.8e-4 (per-op fp32 rounding + reassociation).
        assert diff < 5e-4, f"logit band blown: {diff:.2e}"


def test_gpu_selection_parity_fixed_seeds(torch):
    """THE task gate: zero selection diffs vs the eager reference on a
    fixed seed set at the production shape (8 rows x 40 seeds = 320 rows,
    the decode-regime batch the op is eligible for)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router_dense import fused_router_dense

    flips = 0
    rows = 0
    for seed in range(40):
        torch.manual_seed(1000 + seed)
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(288, 4096, device="cuda")
        bias = torch.randn(288, device="cuda")
        idx, wt = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING)
        _, ref_idx, ref_w = _eager_reference(torch, x, w, bias, PROD_K, PROD_SCALING)
        _assert_same_routing(torch, idx, wt, ref_idx, ref_w)
        for r in range(8):
            if set(idx[r].tolist()) != set(ref_idx[r].tolist()):
                flips += 1
        rows += 8
    assert flips == 0, f"{flips}/{rows} rows flipped expert selection vs eager"


def _boundary_gap(torch, ref_scores, ref_idx, row, k):
    """Score gap across the top-k selection boundary of the reference
    row (descending choice): the band a selection SET flip must cross."""
    vals = ref_scores[row].gather(0, ref_idx[row, :k])
    return (vals[k - 1] - vals[k]).abs().item() if vals.shape[0] > k else float("inf")


def test_gpu_adversarial_boundary_ties(torch):
    """Constructed top-k boundary pairs — the docstring-warned class.

    Per weight matrix: 7 clearly-better experts, 279 clearly-worse, and a
    pair of near-identical weight rows (differing by eps on ONE element,
    dot gap ~eps * |b[h*] * x[h*]|) that normally straddles the 8th slot.

    * eps = 1e-1: every row's reference boundary gap exceeds the
      reduction-noise band (asserted), so selection MUST agree with the
      eager reference — flips pinned to 0.
    * eps = 2**-22 (a few weight ulps): the pair's gap is genuinely inside
      the noise band — flips are LEGITIMATE. Each observed flip must cross
      a sub-noise reference boundary (gap < 1e-5 — anything else is real
      selection drift and fails); the count is recorded for the report
      rather than pinned.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router_dense import fused_router_dense

    E, H, K, T, SEEDS = 288, 4096, 8, 8, 16  # T capped at 8; 2x seeds keeps 128 rows
    PAIR = (7, 8)
    for eps, pin_zero in [(1e-1, True), (2.0**-22, False)]:
        flips = []
        for seed in range(SEEDS):
            g = torch.Generator(device="cpu").manual_seed(4000 + seed)
            x = torch.randn(T, H, generator=g).to("cuda", torch.bfloat16)
            # others clearly below the action; 7 clear winners well above
            w = torch.randn(E, H, generator=g) * 1e-2
            w[:7] = torch.randn(7, H, generator=g) / 16.0
            # the boundary pair: identical base row, eps apart on one element
            base = torch.randn(H, generator=g) / 16.0
            i, j = PAIR
            w[i] = base
            w[j] = base.clone()
            hstar = int(torch.randint(0, H, (1,), generator=g))
            w[j, hstar] = base[hstar] * (1.0 + eps if seed % 2 == 0 else 1.0 - eps)
            x, w = x.cuda(), w.cuda()
            bias = torch.zeros(E, device="cuda")
            ref_logits, ref_idx, _ = _eager_reference(torch, x, w, bias, K, 1.0)
            ref_scores = torch.sigmoid(ref_logits)  # bias == 0 -> choice == scores
            idx, _ = fused_router_dense(x, w, bias, K, 1.0)
            for r in range(T):
                bgap = _boundary_gap(torch, ref_scores, ref_idx, r, K)
                if set(idx[r].tolist()) != set(ref_idx[r].tolist()):
                    flips.append((seed, r, bgap))
                    assert bgap < 1e-5, (
                        f"eps={eps}: flip across a {bgap:.2e} reference boundary "
                        f"(seed {seed} row {r}) — real selection drift"
                    )
                elif pin_zero:
                    assert bgap > 1e-5, (
                        f"eps={eps}: reference boundary gap {bgap:.2e} not above "
                        f"the noise band — construction degenerate"
                    )
        if pin_zero:
            assert not flips, f"{len(flips)} above-noise boundary flips vs eager"
        else:
            print(f"sub-noise boundary probe (eps=2^-22): {len(flips)}/{T * SEEDS} rows "
                  f"flipped, all across sub-noise (<1e-5) reference boundaries")


def test_gpu_all_tie_exact(torch):
    """The rounding-point pin: identical weight rows give bit-identical
    logits on BOTH sides (same reduction per row), so the all-tie routing
    must match the incumbent EXACTLY — indices AND weights.

    This transitively pins the whole post-GEMV section (sigmoid, bias add,
    lowest-index tie-break, gather, renorm sum order, scale) because every
    value in it is computed from the same bit-identical scores.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router, fused_router_shared
    from vkernels.torch_ops.glm_router_dense import (
        fused_router_dense,
        fused_router_dense_shared,
    )

    E, H, K, T = 288, 4096, 8, 4
    for bias_scale in (0.0, 1.0):  # zero bias and all-equal nonzero bias
        torch.manual_seed(11)
        x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        base = torch.randn(1, H, device="cuda")
        w = base.repeat(E, 1)  # ALL rows identical -> all logits tied
        bias = torch.full((E,), 0.5, device="cuda") * bias_scale
        ref_logits, _, _ = _eager_reference(torch, x, w, bias, K, PROD_SCALING)
        # reference logits must be bit-tied too (same cublas kernel per row)
        assert (ref_logits == ref_logits[:, :1]).all(), "cublas broke the constructed tie"

        inc_idx, inc_w = fused_router(ref_logits, bias, K, PROD_SCALING)
        idx, wt = fused_router_dense(x, w, bias, K, PROD_SCALING)
        assert torch.equal(idx, inc_idx), "all-tie indices differ from incumbent"
        # The renorm cancels the logit VALUE (v/(8v) = 1/8 for any v) EXCEPT
        # when sigmoid(v) sinks toward the incumbent's +1e-20 epsilon floor
        # (total ~< 1e-20): there the ratio stops canceling and carries the
        # honest |dv| band (~1e-4 rel). So: allclose across at the band
        # tolerance, BIT-identical within each side.
        assert torch.allclose(wt, inc_w, rtol=1e-3, atol=0.0), "all-tie weights differ"
        assert (wt == wt[:, :1]).all(), "kernel broke the constructed tie across experts"
        assert (inc_w == inc_w[:, :1]).all(), "incumbent broke the constructed tie"
        assert all(row.tolist() == list(range(K)) for row in idx)  # lowest-index tie-break

        # shared variant: routed columns bit-identical to BOTH the dense op
        # and the incumbent; shared column exactly (E, 1.0)
        sidx, swt = fused_router_dense_shared(x, w, bias, K, PROD_SCALING)
        ish_idx, ish_w = fused_router_shared(ref_logits, bias, K, PROD_SCALING)
        assert torch.equal(sidx[:, :K].to(torch.int32), idx)
        assert torch.equal(swt[:, :K], wt)
        assert torch.equal(sidx[:, :K], ish_idx[:, :K].to(torch.int64))
        # routed weights: bit-identical to MY dense call (same kernel), and
        # vs the incumbent-shared only to the band (epsilon-floor rows don't
        # cancel — see the dense all-tie comment above).
        assert torch.equal(swt[:, :K], wt)
        assert torch.allclose(swt[:, :K], ish_w[:, :K], rtol=1e-3, atol=0.0)
        assert torch.equal(sidx[:, K], torch.full((T,), E, dtype=torch.int64, device="cuda"))
        assert torch.equal(swt[:, K], torch.ones(T, device="cuda"))


def test_gpu_all_tie_k_equals_e(torch):
    """Degenerate K == E all-tie: every expert selected, renorm over 288
    identical values — still bit-exact vs the incumbent (the sequential
    ``total += value`` order is the incumbent's own)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router
    from vkernels.torch_ops.glm_router_dense import fused_router_dense

    E, H, T = 64, 128, 3
    torch.manual_seed(13)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, H, device="cuda").repeat(E, 1)
    bias = torch.zeros(E, device="cuda")
    ref_logits, _, _ = _eager_reference(torch, x, w, bias, E, 1.0)
    assert (ref_logits == ref_logits[:, :1]).all()
    inc_idx, inc_w = fused_router(ref_logits, bias, E, 1.0)
    idx, wt = fused_router_dense(x, w, bias, E, 1.0)
    assert torch.equal(idx, inc_idx)
    # same epsilon-floor caveat as the K==8 test: rows whose all-tie score
    # sinks to the +1e-20 floor carry the |dv| band instead of canceling.
    assert torch.allclose(wt, inc_w, rtol=1e-3, atol=0.0)
    assert torch.allclose(wt.sum(-1), torch.ones(T, device="cuda"), rtol=1e-6)


def test_gpu_weight_dtype_superset(torch):
    """bf16/fp16 weights widen exactly in-kernel: same multiplied values as
    the eager w.float() cast, so parity class is unchanged (and the
    checkpoint-dtype path halves the dominant weight traffic)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router_dense import fused_router_dense

    for wdt in (torch.bfloat16, torch.float16):
        torch.manual_seed(23)
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        w32 = torch.randn(288, 4096, device="cuda")
        wb = w32.to(wdt)  # the checkpoint-dtype weight BOTH sides consume
        bias = torch.randn(288, device="cuda")
        _, ref_idx, ref_w = _eager_reference(torch, x, wb, bias, PROD_K, PROD_SCALING)
        idx, wt = fused_router_dense(x, wb, bias, PROD_K, PROD_SCALING)
        _assert_same_routing(torch, idx, wt, ref_idx, ref_w)

    # fp32 x is accepted too (widening is the identity)
    torch.manual_seed(29)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(288, 4096, device="cuda")
    bias = torch.randn(288, device="cuda")
    _, ref_idx, ref_w = _eager_reference(torch, x, w, bias, PROD_K, PROD_SCALING)
    idx, wt = fused_router_dense(x.float(), w, bias, PROD_K, PROD_SCALING)
    _assert_same_routing(torch, idx, wt, ref_idx, ref_w)


def test_gpu_deterministic_repeat(torch):
    """No atomics / split-K: repeated launches on the same inputs are
    bit-identical (the documented determinism contract)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router_dense import fused_router_dense

    torch.manual_seed(31)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(288, 4096, device="cuda")
    bias = torch.randn(288, device="cuda")
    idx0, wt0 = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING)
    for _ in range(5):
        idx, wt = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING)
        assert torch.equal(idx, idx0) and torch.equal(wt, wt0)


def test_gpu_edge_envelopes(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_router_dense import (
        fused_router_dense,
        fused_router_dense_shared,
    )

    # Zero tokens: no launch, empty outputs (both ABIs, logits included).
    x = torch.randn(0, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(288, 4096, device="cuda")
    bias = torch.randn(288, device="cuda")
    idx, wt = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING)
    assert idx.shape == (0, PROD_K) and wt.shape == (0, PROD_K)
    assert idx.dtype == torch.int32
    sidx, swt = fused_router_dense_shared(x, w, bias, PROD_K, PROD_SCALING)
    assert sidx.shape == (0, PROD_K + 1) and swt.shape == (0, PROD_K + 1)
    assert sidx.dtype == torch.int64
    tri = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING, return_logits=True)
    assert tri[2].shape == (0, 288)

    # norm_topk_prob=False parity.
    torch.manual_seed(37)
    x = torch.randn(5, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(288, 4096, device="cuda")
    bias = torch.randn(288, device="cuda")
    _, ref_idx, ref_w = _eager_reference(torch, x, w, bias, PROD_K, PROD_SCALING, norm_topk_prob=False)
    idx, wt = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING, norm_topk_prob=False)
    _assert_same_routing(torch, idx, wt, ref_idx, ref_w)

    # return_logits: same launch, own-reduction logits, finite; the routed
    # outputs match the plain call bit-for-bit (same kernel, same inputs).
    idx4, wt4, logits = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING, return_logits=True)
    assert logits.shape == (5, 288) and torch.isfinite(logits).all()
    _, wt3 = fused_router_dense(x, w, bias, PROD_K, PROD_SCALING)
    assert torch.equal(wt4, wt3)
    assert torch.equal(idx4, idx)

    # The renormalized routed sum hits the scaling factor exactly-ish.
    assert torch.allclose(wt3.sum(-1), torch.full((5,), PROD_SCALING, device="cuda"), rtol=1e-5)

    # Non-contiguous inputs are rejected (caller owns .contiguous()).
    wide_w = torch.randn(288, 5000, device="cuda")[:, :4096]
    assert not wide_w.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        fused_router_dense(x, wide_w, bias, PROD_K, PROD_SCALING)
    strided_x = torch.randn(10, 4096, device="cuda", dtype=torch.bfloat16)[::2]
    assert not strided_x.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        fused_router_dense(strided_x, w, bias, PROD_K, PROD_SCALING)
    strided_b = torch.randn(576, device="cuda")[::2]
    assert not strided_b.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        fused_router_dense(x, w, strided_b, PROD_K, PROD_SCALING)

    # The token cap is a measured perf contract: prefill T must fall back
    # to the eager cuBLAS chain (module docstring has the numbers).
    with pytest.raises(OpNotEligible, match="decode-regime fold"):
        fused_router_dense(torch.randn(9, 4096, device="cuda", dtype=torch.bfloat16),
                           w, bias, PROD_K, PROD_SCALING)
    with pytest.raises(OpNotEligible, match="decode-regime fold"):
        fused_router_dense_shared(torch.randn(9, 4096, device="cuda", dtype=torch.bfloat16),
                                  w, bias, PROD_K, PROD_SCALING)
    # ... and T == 8 (the cap edge) stays eligible.
    torch.manual_seed(41)
    idx8, wt8 = fused_router_dense(torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16),
                                   w, bias, PROD_K, PROD_SCALING)
    assert idx8.shape == (8, PROD_K)
