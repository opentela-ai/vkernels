"""Check epilogue weighting against the same FP32 down-dot accumulator."""
import inspect

import pytest
import torch

import vkernels.torch_ops.sgl_moe as module


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('tokens,skip', [(1, False), (4, False), (22, False), (4, True)])
def test_weighted_down_uses_fp32_accumulator_before_bf16_store(monkeypatch, tokens, skip):
    generator = torch.Generator(device='cuda').manual_seed(61)
    def random(shape):
        return torch.randn(shape, device='cuda', generator=generator)
    args = [random((tokens, 256)).to(torch.bfloat16),
            (random((16, 256, 256)) * .05).to(torch.float8_e4m3fn),
            torch.ones(16, 2, 2, device='cuda'),
            (random((16, 256, 128)) * .05).to(torch.float8_e4m3fn),
            torch.ones(16, 2, 1, device='cuda'),
            torch.stack([torch.randperm(16, device='cuda', generator=generator)[:4]
                         for _ in range(tokens)]),
            random((tokens, 4)).softmax(dim=-1)]
    if skip:
        args[-1][:, 1::2] = 0
    original = module._invoke_fused_moe_kernel
    source = inspect.getsource(original)
    assert source.count('compute_type=tl.bfloat16') == 1
    source = source.replace('def _invoke_fused_moe_kernel(', 'def fp32_invoke(')
    namespace = dict(module.__dict__)
    exec(source.replace('compute_type=tl.bfloat16', 'compute_type=tl.float32'), namespace)
    observed = {}

    def record(*inputs, **kwargs):
        original(*inputs, **kwargs)
        if kwargs['top_k'] == 1:
            assert kwargs['mul_routed_weight'] is True
            inputs = list(inputs)
            weighted = inputs[2].clone()
            fp32 = torch.zeros_like(weighted, dtype=torch.float32)
            inputs[2] = fp32
            namespace['fp32_invoke'](*inputs, **dict(kwargs, mul_routed_weight=False))
            expected = (fp32 * args[-1][:, :, None]).to(torch.bfloat16)
            assert torch.equal(weighted, expected)
            observed['expected'] = expected.sum(dim=1)

    monkeypatch.setattr(module, '_invoke_fused_moe_kernel', record)
    actual = module.sgl_fused_moe(*args, swiglu_limit=7., skip_zero_weights=skip,
                                 route_weight_in_down=True)
    assert torch.equal(actual, observed['expected'])
