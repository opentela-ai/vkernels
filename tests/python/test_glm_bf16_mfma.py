"""Admission and launch contracts; AMD arithmetic is qualified separately."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vkernels.torch_ops import glm_bf16_mfma as op
from vkernels.torch_ops._dispatch import OpNotEligible


def test_cpu_declines_before_compiling(monkeypatch):
    monkeypatch.setattr(op, '_kernels', lambda: pytest.fail('compiled on CPU'))
    with pytest.raises(OpNotEligible, match='HIP device'):
        op.bf16_mfma_linear(torch.zeros(1, 4096, dtype=torch.bfloat16),
                            torch.zeros(16, 4096, dtype=torch.bfloat16))


@pytest.fixture
def gpu_stack(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('CUDA needed for wrapper contract')
    x = torch.randn(1, 2, 4096, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(16, 4096, device='cuda', dtype=torch.bfloat16)
    monkeypatch.setattr(torch, '__version__', op._QUALIFIED_TORCH)
    monkeypatch.setattr(torch.version, 'hip', op._QUALIFIED_HIP)
    monkeypatch.setattr(torch.backends.cuda, 'preferred_blas_library', lambda: '_BlasBackend.Cublaslt')
    monkeypatch.setattr(torch.cuda.tunable, 'is_enabled', lambda: False)
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(gcnArchName='gfx942:sramecc+:xnack-'))
    return x, w


def test_wrapper_restores_shape_and_selects_operand_order(gpu_stack, monkeypatch):
    x, w = gpu_stack
    calls = []
    class Kernel:
        def __init__(self, order): self.order = order
        def __getitem__(self, grid):
            def launch(value, weight, output, rows, n, k, block_k, **kwargs):
                calls.append((self.order, grid, rows, n, k, block_k, kwargs))
                output.copy_(F.linear(value, weight))
            return launch
    monkeypatch.setattr(op, '_kernels', lambda: (SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b), Kernel('input'), Kernel('weight')))
    original_x, original_w = x.clone(), w.clone()
    output = op.bf16_mfma_linear(x, w)
    assert output.shape == (1, 2, 16)
    assert torch.equal(output, F.linear(x,w))
    assert torch.equal(x, original_x) and torch.equal(w, original_w)
    assert calls == [('weight', (1,), 2, 16, 4096, 512,
                       dict(num_warps=2,num_stages=1,matrix_instr_nonkdim=16,kpack=1))]
    w2 = torch.randn(512, 4096, device=x.device, dtype=x.dtype)
    assert op.bf16_mfma_linear(x,w2).shape == (1,2,512)
    assert calls[-1][0] == 'input' and calls[-1][-1]['kpack'] == 2


@pytest.mark.parametrize('policy', ['torch', 'hip', 'arch', 'blas', 'tunable', 'geometry', 'layout', 'gradient'])
def test_unqualified_policy_declines(gpu_stack, monkeypatch, policy):
    x, w = gpu_stack
    monkeypatch.setattr(op, '_kernels', lambda: pytest.fail('compiled an ineligible case'))
    if policy == 'torch': monkeypatch.setattr(torch, '__version__', 'other')
    elif policy == 'hip': monkeypatch.setattr(torch.version, 'hip', 'other')
    elif policy == 'arch': monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: SimpleNamespace(gcnArchName='gfx950'))
    elif policy == 'blas': monkeypatch.setattr(torch.backends.cuda, 'preferred_blas_library', lambda: '_BlasBackend.Cublas')
    elif policy == 'tunable': monkeypatch.setattr(torch.cuda.tunable, 'is_enabled', lambda: True)
    elif policy == 'geometry': w = w[:15]
    elif policy == 'layout': x = x[..., ::2]; w = w[:, ::2]
    elif policy == 'gradient': x.requires_grad_()
    with pytest.raises(OpNotEligible): op.bf16_mfma_linear(x,w)
