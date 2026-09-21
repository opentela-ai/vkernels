"""GLM-5 sigmoid top-k router fusion: contract checks and GPU parity.

The kernel replaces ``Glm53TopkRouter.forward``'s post-GEMM glue for the
shipped ``n_group == 1`` configuration. Parity bar, in order of strictness:

* **Selection** (the top-k expert SET) must match the eager reference
  exactly — a flipped expert changes the model output materially. Ties
  break to the lowest expert index, matching the documented kernel
  contract (and the floe ``_fused_router`` docstring).
* **Weights** are gathered raw sigmoid scores, renormalized with floe's
  ``1e-20`` epsilon (note: sglang's ``biased_topk_impl`` renormalizes
  without epsilon — floe is the parity bar here) and scaled.
* **k-axis order** is free: every consumer pairs (id, weight) element-wise
  (``Glm53Experts`` gathers/scatters by id), so the kernel's
  descending-selection-value order is interchangeable with
  ``torch.topk(sorted=False)``'s unspecified order.
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_router; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def _eager_reference(logits, bias, top_k, scaling, norm_topk_prob=True):
    """Mirror of ``Glm53TopkRouter.forward``'s n_group == 1 eager leg.

    Returns (indices, weights) with ``torch.topk(sorted=False)``'s
    unspecified k-axis order — the comparison is set- and pair-based.
    """
    import torch  # function-local: collection must not require torch

    scores = logits.float().sigmoid()
    scores_for_choice = scores + bias
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1.0e-20)
    return topk_indices, topk_weights * scaling


def _assert_same_routing(got_idx, got_w, ref_idx, ref_w):
    """Same expert SET per row, with weights aligned by expert id."""
    assert got_idx.shape == ref_idx.shape and got_w.shape == ref_w.shape
    rows = ref_idx.shape[0]
    for r in range(rows):
        g = dict(zip(got_idx[r].tolist(), got_w[r].tolist()))
        e = dict(zip(ref_idx[r].tolist(), ref_w[r].tolist()))
        assert set(g) == set(e), (
            f"row {r}: expert set mismatch: got {sorted(g)}, want {sorted(e)}"
        )
        for i, w in g.items():
            assert w == pytest.approx(e[i], rel=1e-5, abs=1e-6), (
                f"row {r}, expert {i}: weight {w} vs {e[i]}"
            )


def test_reference_matches_torch_eager(torch):
    """The CPU oracle agrees with the eager topk leg (selection exact)."""
    from vkernels.torch_ops.glm_router import fused_router_reference

    torch.manual_seed(11)
    for tokens, experts, k in [(7, 288, 8), (1, 64, 1), (5, 32, 4), (3, 257, 8)]:
        logits = torch.randn(tokens, experts)
        bias = torch.randn(experts)
        idx, w = fused_router_reference(logits, bias, k, 2.5)
        ref_idx, ref_w = _eager_reference(logits, bias, k, 2.5)
        _assert_same_routing(idx, w, ref_idx, ref_w)


def test_reference_tie_breaks_lowest(torch):
    """Ties in ``scores + bias`` resolve to the lowest expert index."""
    from vkernels.torch_ops.glm_router import fused_router_reference

    # Identical logits + identical bias: every expert ties; the top-k must
    # be exactly {0..k-1}.
    logits = torch.zeros(2, 16)
    bias = torch.zeros(16)
    idx, _ = fused_router_reference(logits, bias, 4, 1.0)
    assert all(set(row.tolist()) == {0, 1, 2, 3} for row in idx)

    # Tie in score+bias across an expert pair, third slot contested.
    logits = torch.tensor([[2.0, 2.0, 2.0, 0.0, 0.0, -1.0, -2.0, -3.0]])
    bias = torch.tensor([0.1, 0.1, 0.1, 0.9, 0.0, 0.0, 0.0, 0.0])
    # choice = sigmoid+0.9 @ e3 wins; e0/e1/e2 tie at sigmoid(2)+0.1; k=3 -> {3,0,1}
    idx, _ = fused_router_reference(logits, bias, 3, 1.0)
    assert set(idx[0].tolist()) == {0, 1, 3}


def test_contract(torch):
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_router import fused_router

    logits = torch.randn(3, 288)
    bias = torch.randn(288)
    # Non-degenerate group config must stay on the caller's grouped path.
    with pytest.raises(OpNotEligible, match="n_group"):
        fused_router(logits, bias, 8, 2.5, num_group=8, topk_group=4)
    with pytest.raises(OpNotEligible, match="n_group"):
        fused_router(logits, bias, 8, 2.5, num_group=1, topk_group=2)
    with pytest.raises(OpNotEligible, match="2-D"):
        fused_router(logits[0], bias, 8, 2.5)
    with pytest.raises(OpNotEligible, match="bias"):
        fused_router(logits, torch.randn(64), 8, 2.5)
    with pytest.raises(OpNotEligible, match="top_k"):
        fused_router(logits, bias, 0, 2.5)
    with pytest.raises(OpNotEligible, match="top_k"):
        fused_router(logits, bias, 289, 2.5)
    with pytest.raises(OpNotEligible, match="FP32"):
        fused_router(logits.half(), bias.half(), 8, 2.5)
    with pytest.raises(OpNotEligible, match="float dtype"):
        fused_router(logits, bias.int(), 8, 2.5)
    with pytest.raises(OpNotEligible, match="GPU"):
        fused_router(logits, bias, 8, 2.5)  # CPU tensors rejected


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router, fused_router_reference

    torch.manual_seed(13)
    # The shipped shape plus oddball envelopes.
    for tokens, experts, k in [(1, 288, 8), (33, 288, 8), (7, 64, 2), (4, 100, 5)]:
        logits = torch.randn(tokens, experts, device="cuda")
        bias = torch.randn(experts, device="cuda")
        idx, w = fused_router(logits, bias, k, 2.5)
        assert idx.dtype == torch.int32 and w.dtype == torch.float32
        ref_idx, ref_w = fused_router_reference(logits, bias, k, 2.5)
        _assert_same_routing(idx, w, ref_idx, ref_w)


def test_gpu_matches_torch_eager_selection(torch):
    """Selection must match the eager torch.topk leg EXACTLY (no drift).

    Near-tie sensitivity probe: sigmoid is recomputed in Triton, so any
    1-ulp drift vs torch could flip an expert at a selection boundary.
    A flipped expert here means the A/B shadow gate would fail in the
    serving rig — this test is the cheap guard for that.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router

    torch.manual_seed(29)
    total_flips = 0
    for trial in range(50):
        logits = torch.randn(64, 288, device="cuda")
        bias = torch.randn(288, device="cuda")
        idx, w = fused_router(logits, bias, 8, 2.5)
        ref_idx, ref_w = _eager_reference(logits, bias, 8, 2.5)
        total_flips += sum(
            1 for r in range(64) if set(idx[r].tolist()) != set(ref_idx[r].tolist())
        )
        _assert_same_routing(idx, w, ref_idx, ref_w)
    assert total_flips == 0, f"{total_flips} rows flipped expert selection vs eager"


def test_gpu_tie_breaks_lowest(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router

    # All-equal logits and bias: every expert ties; lowest indices win.
    logits = torch.zeros(2, 288, device="cuda")
    bias = torch.zeros(288, device="cuda")
    idx, w = fused_router(logits, bias, 8, 1.0)
    assert all(set(row.tolist()) == set(range(8)) for row in idx)
    assert torch.allclose(w, torch.full_like(w, 1.0 / 8))


def test_gpu_edge_envelopes(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_router import fused_router

    # Zero tokens: no launch, empty outputs.
    logits = torch.randn(0, 288, device="cuda")
    bias = torch.randn(288, device="cuda")
    idx, w = fused_router(logits, bias, 8, 2.5)
    assert idx.shape == (0, 8) and w.shape == (0, 8)

    # k == experts: everything is selected, renorm is a no-op pre-scale.
    logits = torch.randn(3, 16, device="cuda")
    bias = torch.randn(16, device="cuda")
    idx, w = fused_router(logits, bias, 16, 2.5, norm_topk_prob=True)
    scores = logits.sigmoid()
    ref = scores / (scores.sum(-1, keepdim=True) + 1.0e-20) * 2.5
    # renormalized weights sum to 1, scaled by 2.5
    assert torch.allclose(w.sum(-1), torch.full((3,), 2.5, device="cuda"), rtol=1e-5)
    assert torch.allclose(w.sort(-1).values, ref.sort(-1).values, atol=1e-6, rtol=1e-6)

    # bf16 bias (checkpoint dtype) is widened in-kernel.
    logits = torch.randn(5, 288, device="cuda")
    bias32 = torch.randn(288, device="cuda")
    idx_bf, w_bf = fused_router(logits, bias32.bfloat16(), 8, 2.5)
    ref_idx, ref_w = _eager_reference(logits, bias32.bfloat16().float(), 8, 2.5)
    _assert_same_routing(idx_bf, w_bf, ref_idx, ref_w)

    # Non-contiguous logits are rejected (caller owns the .contiguous()).
    from vkernels.torch_ops._dispatch import OpNotEligible

    wide = torch.randn(5, 300, device="cuda")[:, :288]
    assert not wide.is_contiguous()
    with pytest.raises(OpNotEligible, match="contiguous"):
        fused_router(wide, torch.randn(288, device="cuda"), 8, 2.5)
