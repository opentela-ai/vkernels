"""Static-capacity routes preserve the compacted weighted-BF16 contract."""
import pytest
import torch

from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
    _route_counts,
    glm_moe_grouped_gemm_native,
)
from vkernels.torch_ops.moe_deterministic_combine import (
    deterministic_compacted_combine,
    deterministic_route_combine,
)


def test_dropped_routes_have_no_expert_tiles_or_dynamic_shape(monkeypatch):
    ids = torch.tensor([[0, -1, 2], [2, 1, 1]])
    weights = torch.tensor([[1., .9, 0.], [.5, 0., 1.]])

    def forbidden(*args, **kwargs):
        raise AssertionError('route planning must not read a live count')

    monkeypatch.setattr(torch, 'nonzero', forbidden)
    order, tokens, counts, starts = _route_counts(ids, 3, 'cpu', weights)
    assert order.tolist() == [0, 5, 3, 1, 2, 4]
    assert tokens.tolist() == [0, 1, 1, 0, 0, 1]
    assert counts.tolist() == [1, 1, 1]
    assert starts.tolist() == [0, 1, 2]
    _, _, counts, starts = _route_counts(ids, 3, 'cpu', weights.zero_())
    assert counts.tolist() == starts.tolist() == [0, 0, 0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_weighted_bf16_boundary_and_original_slot_order(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('GPU required')
    torch.manual_seed(912)
    original = torch.randn(13, 8, 129, device=device).bfloat16()
    weights = torch.randn(13, 8, device=device)
    order = torch.randperm(104, device=device)
    products = (original.float() * weights[..., None]).bfloat16().float()
    expected = torch.zeros(13, 129, device=device)
    for slot in range(8):
        expected.add_(products[:, slot])
    actual = deterministic_route_combine(
        original.reshape(104, 129)[order], order, weights,
        round_products_to_bf16=True)
    assert torch.equal(actual, expected.bfloat16())
    ordinary = deterministic_route_combine(
        original.reshape(104, 129)[order], order, weights)
    assert not torch.equal(actual, ordinary)


def _case(tokens, top_k, device='cuda'):
    torch.manual_seed(543 + tokens)
    experts, hidden, intermediate = 8, 256, 256
    x = torch.randn(tokens, hidden, device=device).bfloat16()
    gu = (torch.randn(experts, 2*intermediate, hidden, device=device)*.1).to(torch.float8_e4m3fn)
    dn = (torch.randn(experts, hidden, intermediate, device=device)*.1).to(torch.float8_e4m3fn)
    gus = torch.rand(experts, 4, 2, device=device)+.5
    dns = torch.rand(experts, 2, 2, device=device)+.5
    ids = torch.randint(experts, (tokens, top_k), device=device)
    # Skew produces experts with multiple row tiles.
    ids[:, 0] = 0
    weights = torch.rand(tokens, top_k, device=device)
    return x, gu, gus, dn, dns, ids, weights


def _compact_reference(case):
    x, gu, gus, dn, dns, ids, weights = case
    weights = torch.where(ids >= 0, weights, 0.)
    live = (weights != 0).reshape(-1).nonzero(as_tuple=True)[0]
    if live.numel() == 0:
        return torch.zeros_like(x)
    rows = glm_moe_grouped_gemm_native(
        x[live//ids.shape[1]], gu, gus, dn, dns,
        ids.reshape(-1)[live, None], weights.reshape(-1)[live, None],
        deterministic_combine=True)
    return deterministic_compacted_combine(
        rows, live, tokens=x.shape[0], top_k=ids.shape[1])


def _device_result(case):
    return glm_moe_grouped_gemm_native(
        *case, deterministic_combine=True, skip_zero_weights=True,
        round_products_to_bf16=True)


@pytest.mark.gpu
@pytest.mark.parametrize('tokens,top_k', [(1, 1), (17, 8), (128, 8), (384, 8)])
def test_device_routes_match_compaction_with_changed_graph_routes(tokens, top_k):
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    case = _case(tokens, top_k)
    x, _, _, _, _, ids, weights = case
    weights[:, ::2] = 0
    for _ in range(3):
        result = _device_result(case)
    assert torch.equal(result, _compact_reference(case))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _device_result(case)
    for pattern in ['complement', 'all', 'none', 'mixed', 'negative']:
        x.copy_(torch.randn_like(x))
        ids.copy_(torch.randint(8, ids.shape, device='cuda'))
        ids[:, 0] = 0
        weights.copy_(torch.rand_like(weights))
        if pattern == 'complement':
            weights[:, 1::2] = 0
        elif pattern == 'none':
            weights.zero_()
        elif pattern == 'mixed':
            weights[:, ::2] = 0
        elif pattern == 'negative':
            ids[:, ::2] = -1
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(captured).all()
        assert torch.equal(captured, _compact_reference(case))
