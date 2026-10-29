"""Short convolution retains native decode activation and raw-history bits."""
import subprocess
import sys
import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, '-c', "import sys; import vkernels.torch_ops.glm_kda_conv_block; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.mark.parametrize('change', ['cpu', 'ndim', 'batch', 'width', 'channels', 'dtype', 'history', 'weight', 'grad'])
def test_unsupported_contract_declines(change):
    import torch
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_kda_conv_block import kda_conv_block
    raw = torch.zeros(1, 6144, 2, dtype=torch.bfloat16)
    history = torch.zeros(1, 6144, 3, dtype=raw.dtype)
    weight = torch.zeros(6144, 1, 4, dtype=raw.dtype)
    if change == 'ndim': raw = raw[0]
    elif change == 'batch': raw = raw.repeat(5, 1, 1)
    elif change == 'width': raw = raw[:, :, :1]
    elif change == 'channels': raw = raw[:, :128]
    elif change == 'dtype': raw = raw.float()
    elif change == 'history': history = history[:, :, :2]
    elif change == 'weight': weight = weight[:, :, :3]
    elif change == 'grad': weight.requires_grad_(True)
    inputs = (raw, history, weight); before = [x.detach().clone() for x in inputs]
    with pytest.raises(OpNotEligible): kda_conv_block(*inputs)
    assert all(torch.equal(x, y) for x, y in zip(inputs, before))


@pytest.mark.parametrize('batch', [1, 2, 3, 4])
def test_decode_activations_and_every_history_bit_on_qualified_stack(batch):
    import torch
    if str(torch.__version__) != '2.9.0a0+git7bcbafe' or torch.version.hip != '7.0.51831-a3e329ad8':
        pytest.skip('pinned Beverin stack required')
    import triton
    if not torch.cuda.is_available() or triton.__version__ != '3.4.0' or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx942':
        pytest.skip('pinned gfx942 required')
    import torch.nn.functional as F
    from vkernels.torch_ops.glm_kda_conv_block import kda_conv_block
    def bits(a, b):
        return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    with torch.no_grad():
        for width in (2, 3, 4):
            torch.manual_seed(batch * 19 + width)
            for zero in (False, True):
                raw = torch.randn(batch, 6144, width, device='cuda', dtype=torch.bfloat16)
                initial = torch.randn(batch, 6144, 3, device='cuda', dtype=raw.dtype)
                weight = torch.randn(6144, 1, 4, device='cuda', dtype=raw.dtype)
                if zero: raw.fill_(-0.0); initial.fill_(-0.0)
                before = [x.clone() for x in (raw, initial, weight)]
                actual, histories = kda_conv_block(raw, initial, weight)
                state = initial; outputs = []; expected = []
                for token in range(width):
                    window = torch.cat([state, raw[:, :, token:token + 1]], -1)
                    outputs.append(F.silu((window * weight.squeeze(1)).sum(-1)))
                    state = window[:, :, -3:].contiguous(); expected.append(state)
                assert bits(actual, torch.stack(outputs, -1))
                assert bits(histories, torch.stack(expected, 1))
                assert all(bits(x, y) for x, y in zip((raw, initial, weight), before))
                torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): gout, ghist = kda_conv_block(raw, initial, weight)
                graph.replay(); torch.cuda.synchronize()
                assert bits(gout, actual) and bits(ghist, histories)
