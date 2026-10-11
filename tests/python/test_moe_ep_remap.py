"""Fused EP masking matches the original chain under changing graph routes."""
import pytest
torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.moe_ep_remap import ep_remap, ep_remap_eligible


def test_cpu_contract_declines():
    ids = torch.zeros(1, 8, dtype=torch.int64)
    weights = torch.ones(1, 8)
    assert not ep_remap_eligible(ids, weights, rank=0, local_experts=72)
    with pytest.raises(OpNotEligible):
        ep_remap(ids, weights, rank=0, local_experts=72)


def _oracle(ids, weights, rank, experts):
    local = ids-rank*experts
    valid = (local >= 0) & (local < experts)
    return local.clamp(0, experts-1), weights*valid.to(weights.dtype)


@pytest.mark.gpu
@pytest.mark.parametrize('tokens', [0, 1, 4, 33])
@pytest.mark.parametrize('rank', [0, 1, 2, 3])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_fused_remap_graph_parity(tokens, rank, dtype):
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    ids = torch.randint(-1, 289, (tokens, 8), device='cuda')
    weights = torch.randn(tokens, 8, device='cuda').to(dtype)
    for _ in range(3):
        actual = ep_remap(ids, weights, rank=rank, local_experts=72)
    expected = _oracle(ids, weights, rank, 72)
    assert all(torch.equal(a,b) for a,b in zip(actual,expected))
    if not tokens:
        return
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ep_remap(ids, weights, rank=rank, local_experts=72)
    for _ in range(3):
        ids.copy_(torch.randint(-1, 289, ids.shape, device='cuda'))
        weights.copy_(torch.randn_like(weights))
        graph.replay()
        expected = _oracle(ids, weights, rank, 72)
        assert all(torch.equal(a,b) for a,b in zip(captured,expected))
    # Preserve IEEE special values and signs at zero-masked routes.
    ids.fill_(rank*72+72)
    weights[:, :4] = torch.tensor([float('inf'),float('nan'),-0.,-1.],device='cuda',dtype=dtype)
    graph.replay()
    expected = _oracle(ids, weights, rank, 72)
    torch.testing.assert_close(captured[1],expected[1],atol=0,rtol=0,equal_nan=True)
    assert torch.equal(torch.signbit(captured[1][:, 2:4]),torch.signbit(expected[1][:, 2:4]))
