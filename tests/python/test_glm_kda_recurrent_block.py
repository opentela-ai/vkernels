"""Short blocks retain the incumbent decode operation and every prefix state."""
import subprocess
import sys
import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, '-c', "import sys; import vkernels.torch_ops.glm_kda_recurrent_block; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.mark.parametrize('change', ['cpu', 'ndim', 'batch', 'width', 'heads', 'dim', 'key', 'beta', 'state', 'dtype', 'grad'])
def test_unsupported_contract_declines_without_mutation(change):
    import torch
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_kda_recurrent_block import kda_recurrent_block
    q = torch.zeros(1, 2, 16, 128, dtype=torch.bfloat16)
    k, v, g = [q.clone() for _ in range(3)]
    beta = torch.zeros(1, 2, 16, dtype=q.dtype)
    state = torch.zeros(1, 16, 128, 128)
    if change == 'ndim': q = q[0]
    elif change == 'batch': q = q.repeat(5, 1, 1, 1)
    elif change == 'width': q = q[:, :1]
    elif change == 'heads': q = q[:, :, :8]
    elif change == 'dim': q = q[..., :64]
    elif change == 'key': k = k[..., :64]
    elif change == 'beta': beta = beta[..., :8]
    elif change == 'state': state = state.bfloat16()
    elif change == 'dtype': q = q.float()
    elif change == 'grad': state.requires_grad_(True)
    tensors = (q, k, v, g, beta, state)
    before = [x.detach().clone() for x in tensors]
    with pytest.raises(OpNotEligible): kda_recurrent_block(*tensors)
    assert all(torch.equal(a, b) for a, b in zip(tensors, before))


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
def test_every_prefix_matches_decode_bits_on_qualified_stack(batch):
    import torch
    if str(torch.__version__) != '2.9.0a0+git7bcbafe' or torch.version.hip != '7.0.51831-a3e329ad8':
        pytest.skip('pinned Beverin stack required')
    import triton
    if not torch.cuda.is_available() or triton.__version__ != '3.4.0' or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx942':
        pytest.skip('pinned gfx942 required')
    from vkernels.torch_ops.glm_kda_decode import kda_decode
    from vkernels.torch_ops.glm_kda_recurrent_block import kda_recurrent_block
    def bits(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    with torch.no_grad():
        for width in (2, 3, 4):
            torch.manual_seed(batch * 37 + width)
            shape = (batch, width, 16, 128)
            values = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
            values[3] = -values[3].abs()
            beta = torch.rand(batch, width, 16, device='cuda', dtype=torch.bfloat16)
            initial = torch.randn(batch, 16, 128, 128, device='cuda')
            inputs = (*values, beta, initial)
            before = [x.clone() for x in inputs]
            output, prefixes = kda_recurrent_block(*inputs)
            state = initial
            for token in range(width):
                vectors = [x[:, token:token + 1].contiguous() for x in (*values, beta)]
                expected, state = kda_decode(*vectors, state, out_fp32=True)
                assert bits(output[:, token:token + 1], expected)
                assert bits(prefixes[:, token], state)
            assert all(bits(x, y) for x, y in zip(inputs, before))
            torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): actual, states = kda_recurrent_block(*inputs)
            graph.replay(); torch.cuda.synchronize()
            assert bits(actual, output) and bits(states, prefixes)
