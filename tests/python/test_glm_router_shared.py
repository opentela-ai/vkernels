"""fused_router_shared: the shared-expert append companion (node-cut A).

The sibling op extends the fused router with floe's
``Glm53TopkRouter._append_shared_slot`` tail in-kernel (the required
companion of stacking the shared expert as slot ``E``: without it the
``full``+``cat``+``cat`` appends and the ``to(int64)`` cast add back the
graph nodes node-cut A removes). Parity bar, in order of strictness:

* **Routed columns bit-identical to the incumbent ``fused_router``** —
  the append happens AFTER the normalize + routed scaling and the shared
  slot never joins the renormalization sum, so selection and weights of
  the routed K columns cannot move. A renormalize-over-K+1 or append-pre-
  scale variant would fail the ``torch.equal`` checks here.
* **Shared column exact**: index ``E`` (the stacked shared slot) with
  weight EXACTLY ``1.0`` — floe's donor semantics, bit-exact.
* **int64 indices emitted in-kernel** (``torch.topk``'s ABI): the eager
  oracle below already produces int64, so dtype parity pins it.

The oracle is floe's ACTUAL production code path, copied verbatim from
``floe/engine/runner/models/glm53flash/forward.py``: the
``Glm53TopkRouter.forward`` eager ``n_group == 1`` leg plus
``_append_shared_slot`` (source-named below). If floe's semantics move,
this file must move with them.
"""

import pytest


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _floe_router_eager_leg(logits, bias, top_k, scaling, norm_topk_prob=True):
    """VERBATIM copy of ``Glm53TopkRouter.forward``'s eager n_group == 1
    leg (forward.py, the ``scores = router_logits.float().sigmoid()`` block
    with the degenerate group mask) — returns int64 [T, K] indices and
    [T, K] weights, k-axis order unspecified (``sorted=False``)."""
    import torch

    scores = logits.float().sigmoid()
    scores_for_choice = scores + bias
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    return topk_indices, topk_weights * scaling


def _floe_append_shared_slot(indices, weights, num_experts):
    """VERBATIM copy of ``Glm53TopkRouter._append_shared_slot`` (forward.py):
    append index ``num_experts`` with weight EXACTLY 1.0, AFTER the
    normalize + scale (the caller runs this on the finished tensors)."""
    import torch

    shared_idx = torch.full((indices.shape[0], 1), num_experts, dtype=indices.dtype, device=indices.device)
    shared_w = torch.ones((weights.shape[0], 1), dtype=weights.dtype, device=weights.device)
    return torch.cat([indices, shared_idx], dim=1), torch.cat([weights, shared_w], dim=1)


def _floe_reference(logits, bias, top_k, scaling, norm_topk_prob=True):
    """floe's production path: eager routing leg + shared-slot append."""
    experts = logits.shape[1]
    idx, w = _floe_router_eager_leg(logits, bias, top_k, scaling, norm_topk_prob)
    return _floe_append_shared_slot(idx, w, experts)


def _assert_same_routing(got_idx, got_w, ref_idx, ref_w):
    """Same expert SET per row, with weights aligned by expert id (the
    k-axis order is free on both sides)."""
    assert got_idx.shape == ref_idx.shape and got_w.shape == ref_w.shape
    for r in range(ref_idx.shape[0]):
        g = dict(zip(got_idx[r].tolist(), got_w[r].tolist()))
        e = dict(zip(ref_idx[r].tolist(), ref_w[r].tolist()))
        assert set(g) == set(e), (
            f"row {r}: expert set mismatch: got {sorted(g)}, want {sorted(e)}"
        )
        for i, w in g.items():
            assert w == pytest.approx(e[i], rel=1e-5, abs=1e-6), (
                f"row {r}, expert {i}: weight {w} vs {e[i]}"
            )


def test_floe_reference_semantics(torch):
    """The copied oracle itself: K+1 columns, int64, shared col == E, 1.0."""
    logits = torch.randn(6, 288)
    bias = torch.randn(288)
    idx, w = _floe_reference(logits, bias, 8, 2.5)
    assert idx.shape == (6, 9) and w.shape == (6, 9)
    assert idx.dtype == torch.int64
    assert torch.equal(idx[:, 8], torch.full((6,), 288, dtype=torch.int64))
    assert torch.equal(w[:, 8], torch.ones(6))


def test_contract(torch):
    """Eligibility mirrors the incumbent op (same messages, CPU-safe)."""
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_router import fused_router_shared

    logits = torch.randn(3, 288)
    bias = torch.randn(288)
    with pytest.raises(OpNotEligible, match="n_group"):
        fused_router_shared(logits, bias, 8, 2.5, num_group=8, topk_group=4)
    with pytest.raises(OpNotEligible, match="2-D"):
        fused_router_shared(logits[0], bias, 8, 2.5)
    with pytest.raises(OpNotEligible, match="bias"):
        fused_router_shared(logits, torch.randn(64), 8, 2.5)
    with pytest.raises(OpNotEligible, match="top_k"):
        fused_router_shared(logits, bias, 0, 2.5)
    with pytest.raises(OpNotEligible, match="top_k"):
        fused_router_shared(logits, bias, 289, 2.5)
    with pytest.raises(OpNotEligible, match="FP32"):
        fused_router_shared(logits.half(), bias.half(), 8, 2.5)
    with pytest.raises(OpNotEligible, match="float dtype"):
        fused_router_shared(logits, bias.int(), 8, 2.5)
    with pytest.raises(OpNotEligible, match="GPU"):
        fused_router_shared(logits, bias, 8, 2.5)  # CPU tensors rejected


def test_gpu_parity_vs_floe_reference(torch):
    """Routing parity with floe's actual eager+append path, K+1 wide."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router_shared

    torch.manual_seed(17)
    for tokens, experts, k in [(1, 288, 8), (33, 288, 8), (7, 64, 2), (4, 100, 5)]:
        logits = torch.randn(tokens, experts, device="cuda")
        bias = torch.randn(experts, device="cuda")
        idx, w = fused_router_shared(logits, bias, k, 2.5)
        assert idx.shape == (tokens, k + 1) and w.shape == (tokens, k + 1)
        assert idx.dtype == torch.int64 and w.dtype == torch.float32
        ref_idx, ref_w = _floe_reference(logits, bias, k, 2.5)
        _assert_same_routing(idx, w, ref_idx, ref_w)
        # shared column: index E, weight exactly 1.0 — on every row
        assert torch.equal(idx[:, k], torch.full((tokens,), experts, dtype=torch.int64, device="cuda"))
        assert torch.equal(w[:, k], torch.ones(tokens, device="cuda"))


def test_gpu_bit_identical_to_incumbent(torch):
    """The ordering contract: append AFTER normalize + scale.

    The routed K columns must be BIT-identical to the incumbent
    ``fused_router`` on the same inputs — anything else (renormalizing
    over K+1, appending pre-scale) moves weights and fails here.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router, fused_router_shared

    torch.manual_seed(19)
    for norm in (True, False):
        for tokens, experts, k in [(1, 288, 8), (33, 288, 8), (7, 64, 2), (4, 100, 5)]:
            logits = torch.randn(tokens, experts, device="cuda")
            bias = torch.randn(experts, device="cuda")
            inc_idx, inc_w = fused_router(logits, bias, k, 2.5, norm)
            sh_idx, sh_w = fused_router_shared(logits, bias, k, 2.5, norm)
            assert torch.equal(sh_idx[:, :k], inc_idx.to(torch.int64))
            assert torch.equal(sh_w[:, :k], inc_w)

    # bf16 bias (checkpoint dtype) widens identically to the incumbent.
    logits = torch.randn(5, 288, device="cuda")
    bias32 = torch.randn(288, device="cuda")
    inc_idx, inc_w = fused_router(logits, bias32.bfloat16(), 8, 2.5)
    sh_idx, sh_w = fused_router_shared(logits, bias32.bfloat16(), 8, 2.5)
    assert torch.equal(sh_idx[:, :8], inc_idx.to(torch.int64))
    assert torch.equal(sh_w[:, :8], inc_w)


def test_gpu_renorm_excludes_shared_slot(torch):
    """The shared 1.0 never joins the renormalization: routed weights sum
    to scaling (incumbent contract), not to 1.0-ish + shared term."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router_shared

    logits = torch.randn(9, 288, device="cuda")
    bias = torch.randn(288, device="cuda")
    _, w = fused_router_shared(logits, bias, 8, 2.5)
    assert torch.allclose(
        w[:, :8].sum(-1), torch.full((9,), 2.5, device="cuda"), rtol=1e-5
    )
    # and the appended column is untouched by both renorm and scale
    assert torch.equal(w[:, 8], torch.ones(9, device="cuda"))


def test_gpu_selection_no_drift_vs_eager(torch):
    """Selection must match the eager leg EXACTLY across trials (the
    near-tie sensitivity probe from the incumbent suite, shared tail)."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router_shared

    torch.manual_seed(23)
    flips = 0
    for _ in range(20):
        logits = torch.randn(64, 288, device="cuda")
        bias = torch.randn(288, device="cuda")
        idx, w = fused_router_shared(logits, bias, 8, 2.5)
        ref_idx, ref_w = _floe_reference(logits, bias, 8, 2.5)
        flips += sum(
            1
            for r in range(64)
            if set(idx[r, :8].tolist()) != set(ref_idx[r, :8].tolist())
        )
        _assert_same_routing(idx, w, ref_idx, ref_w)
    assert flips == 0, f"{flips} rows flipped expert selection vs eager"


def test_gpu_tie_breaks_lowest_shared_last(torch):
    """All-tie rows: routed slots break to the lowest indices; the shared
    column stays exactly (E, 1.0) even in the degenerate tie."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router_shared

    logits = torch.zeros(2, 288, device="cuda")
    bias = torch.zeros(288, device="cuda")
    idx, w = fused_router_shared(logits, bias, 8, 1.0)
    assert all(set(row[:8].tolist()) == set(range(8)) for row in idx)
    assert torch.equal(idx[:, 8], torch.full((2,), 288, dtype=torch.int64, device="cuda"))
    assert torch.equal(w[:, :8], torch.full_like(w[:, :8], 1.0 / 8))
    assert torch.equal(w[:, 8], torch.ones(2, device="cuda"))


def test_gpu_edge_envelopes(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_router import fused_router_shared

    # Zero tokens: no launch, empty K+1 outputs.
    logits = torch.randn(0, 288, device="cuda")
    bias = torch.randn(288, device="cuda")
    idx, w = fused_router_shared(logits, bias, 8, 2.5)
    assert idx.shape == (0, 9) and w.shape == (0, 9)
    assert idx.dtype == torch.int64

    # k == experts: every expert selected, shared column still appended.
    logits = torch.randn(3, 16, device="cuda")
    bias = torch.randn(16, device="cuda")
    idx, w = fused_router_shared(logits, bias, 16, 2.5)
    assert idx.shape == (3, 17) and w.shape == (3, 17)
    assert torch.equal(idx[:, 16], torch.full((3,), 16, dtype=torch.int64, device="cuda"))
    assert torch.allclose(w[:, :16].sum(-1), torch.full((3,), 2.5, device="cuda"), rtol=1e-5)
    assert torch.equal(w[:, 16], torch.ones(3, device="cuda"))

    # norm_topk_prob=False: raw scaled scores, shared column still 1.0.
    logits = torch.randn(5, 288, device="cuda")
    bias = torch.randn(288, device="cuda")
    idx, w = fused_router_shared(logits, bias, 8, 2.5, norm_topk_prob=False)
    ref_idx, ref_w = _floe_reference(logits, bias, 8, 2.5, norm_topk_prob=False)
    _assert_same_routing(idx, w, ref_idx, ref_w)

    # Non-contiguous logits are rejected (caller owns .contiguous()).
    wide = torch.randn(5, 300, device="cuda")[:, :288]
    assert not wide.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        fused_router_shared(wide, torch.randn(288, device="cuda"), 8, 2.5)
