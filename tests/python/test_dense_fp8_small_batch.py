"""Dequantization, split-K tails, graphs and fallback eligibility."""
import pytest
torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.dense_fp8_small_batch import _run, dense_fp8_small_batch

CUDA = (torch.cuda.is_available() and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 9)


def operands(m, o, k, device):
    torch.manual_seed(38)
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * 4
    weight = torch.randn(o, k, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(o // 128, k // 128, device=device) * .03
    return x, weight, scales


def reference(x, weight, scales):
    o, k = weight.shape
    bf = (weight.float().reshape(o // 128, 128, k // 128, 128)
          * scales[:, None, :, None]).reshape(o, k).bfloat16()
    return torch.nn.functional.linear(x, bf)


@pytest.mark.skipif(not CUDA, reason="CUDA tensor cores required")
@pytest.mark.parametrize("m,o,k,config", [
    (2, 128, 128, (32, 1, 4)), (3, 256, 384, (64, 4, 4)),
    (4, 512, 512, (128, 8, 8)), (4, 4096, 4096, (64, 16, 4))])
def test_dot_matches_existing_dense_tolerance_without_mutating_inputs(m, o, k, config):
    args = operands(m, o, k, "cuda")
    saved = [t.clone() for t in args]
    actual = _run(*args, config)
    torch.testing.assert_close(actual, reference(*args), rtol=.01, atol=.01)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(args, saved))


@pytest.mark.skipif(not CUDA, reason="CUDA graph required")
def test_graph_replay_reads_updated_inputs_weights_and_scales():
    args = operands(3, 256, 384, "cuda")
    _run(*args, (64, 4, 4))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run(*args, (64, 4, 4))
    with torch.no_grad():
        args[0].mul_(.5)
        args[1].copy_(args[1].float().neg().to(args[1].dtype))
        args[2].mul_(2)
    graph.replay()
    torch.testing.assert_close(actual, reference(*args), rtol=.01, atol=.01)


def test_cpu_declines_explicitly():
    with pytest.raises(OpNotEligible):
        _run(*operands(4, 256, 384, "cpu"), (64, 4, 4))


@pytest.mark.skipif(not CUDA, reason="CUDA eligibility tests")
def test_unsupported_shape_grad_and_public_unqualified_dispatch_decline():
    args = operands(4, 256, 384, "cuda")
    with pytest.raises(OpNotEligible):
        _run(args[0][:1], args[1], args[2], (64, 4, 4))
    args[0].requires_grad_(True)
    with pytest.raises(OpNotEligible):
        _run(*args, (64, 4, 4))
    args[0].requires_grad_(False)
    with pytest.raises(OpNotEligible):
        dense_fp8_small_batch(*args)  # Fixture is outside the qualified shape table.
