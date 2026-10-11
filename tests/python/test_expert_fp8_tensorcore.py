"""Mechanics and bounded operator error; not legacy bitwise MoE qualification."""
import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.expert_fp8_tensorcore import expert_fp8_tensorcore  # noqa: E402


def test_cpu_declines():
    with pytest.raises(OpNotEligible):
        expert_fp8_tensorcore(torch.zeros(1, 128, dtype=torch.bfloat16),
                             torch.zeros(2, 128, 128, dtype=torch.float8_e4m3fn),
                             torch.ones(2, 1, 1), torch.zeros(1, 2, dtype=torch.int64))


def reference(x, weight, scales, indices):
    # Small fixtures only: an independent, explicit BF16 weight boundary.
    expanded = scales.repeat_interleave(128, -2).repeat_interleave(128, -1)
    dequantized = (weight.float() * expanded).to(torch.bfloat16).float()
    selected = dequantized[indices]
    activations = x[:, None, :] if x.ndim == 2 else x
    return (selected * activations.float().unsqueeze(-2)).sum(-1).to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip,
                    reason="CUDA BF16 tensor cores")
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("tile", [32, 64, 128])
def test_changed_input_graph_and_owned_outputs(routed, tile):
    torch.manual_seed(982)
    t, k, e, o, i = 3, 4, 5, 256, 256
    x = torch.randn((t, k, i) if routed else (t, i), device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(e, o, i, device="cuda") * .3).to(torch.float8_e4m3fn)
    scales = torch.rand(e, o // 128, i // 128, device="cuda") * .02
    indices = torch.tensor([[0, 0, 4, 2], [1, 4, 2, 0], [4, 3, 1, 1]], device="cuda")
    values = x, weight, scales, indices

    def check(got):
        want = reference(*values)
        error = (want.float() - got.float()).abs()
        assert got.shape == (t, k, o) and got.dtype == torch.bfloat16
        assert torch.isfinite(got).all()
        assert error.max() <= .01 and error.square().mean().sqrt() <= .01

    first = expert_fp8_tensorcore(*values, tile=tile)
    check(first)
    owned = first.clone()
    snapshots = [v.clone() for v in values]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = expert_fp8_tensorcore(*values, tile=tile)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(v.view(torch.uint8), s.view(torch.uint8))
               for v, s in zip(values, snapshots))
    x.mul_(-.75)
    indices.copy_(indices.roll(1, 1))
    graph.replay()
    torch.cuda.synchronize()
    check(out)
    assert torch.equal(first, owned), "Later invocations must not alias a previous output"
    with pytest.raises(OpNotEligible):
        expert_fp8_tensorcore(*values, tile=16)
