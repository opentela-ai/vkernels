"""Fixed-order GPU combine must match the independent FP32 product oracle."""
import pytest
import torch

from vkernels.torch_ops.moe_deterministic_combine import deterministic_route_combine


@pytest.mark.gpu
@pytest.mark.parametrize('tokens,top_k,hidden', [(1, 1, 1), (13, 8, 129), (384, 8, 4096), (32, 3, 511), (0, 8, 64)])
def test_gpu_original_route_order(tokens, top_k, hidden):
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    torch.manual_seed(71)
    original = torch.randn(tokens * top_k, hidden, device='cuda').bfloat16()
    weights = torch.randn(tokens, top_k, device='cuda')
    order = torch.randperm(tokens * top_k, device='cuda')
    # These products deliberately round before summation, independently of
    # the optimized inverse-permutation and Triton implementation.
    products = original.float().reshape(tokens, top_k, hidden) * weights[:, :, None]
    expected = torch.zeros(tokens, hidden, device='cuda')
    for slot in range(top_k):
        expected.add_(products[:, slot])
    expected = expected.bfloat16()
    for _ in range(3):
        actual = deterministic_route_combine(original[order], order, weights)
        assert torch.equal(actual, expected)


@pytest.mark.gpu
def test_gpu_cancellation_preserves_product_rounding_and_add_order():
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    down = torch.tensor([[1.], [-1.], [2.**-25]], device='cuda').bfloat16()
    weights = torch.full((1, 3), .5, device='cuda')
    order = torch.tensor([2, 0, 1], device='cuda')
    actual = deterministic_route_combine(down[order], order, weights)
    assert actual.item() == 2.**-26
