"""Native norm fusion declines unsupported contracts without eager GPU imports."""
import subprocess
import sys
from types import SimpleNamespace
import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable,'-c',"import sys; import vkernels.torch_ops.glm_norm_native; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],check=True)


@pytest.mark.parametrize('change',['cpu','dtype','ndim','dimension','rows','layout'])
def test_unsupported_input_declines(change):
    import torch
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_norm_native import rms_norm_native
    x = torch.zeros(2,512,dtype=torch.bfloat16)
    if change == 'dtype': x = x.float()
    if change == 'ndim': x = x[0]
    if change == 'dimension': x = x[:,:32].contiguous()
    if change == 'rows': x = torch.zeros(257,512,dtype=torch.bfloat16)
    if change == 'layout': x = torch.zeros(2,1024,dtype=torch.bfloat16)[:,::2]
    module = SimpleNamespace(weight=torch.ones(x.shape[-1],dtype=torch.bfloat16),variance_epsilon=1e-6)
    before=x.clone()
    with pytest.raises(OpNotEligible): rms_norm_native(x,module)
    assert torch.equal(x,before)


@pytest.mark.parametrize('kind',['plain','gated','unweighted'])
def test_exact_output_on_qualified_stack(kind):
    import torch
    if str(torch.__version__) != '2.9.0a0+git7bcbafe' or torch.version.hip != '7.0.51831-a3e329ad8':
        pytest.skip('pinned gfx942 stack required')
    import triton
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx942' or triton.__version__ != '3.4.0':
        pytest.skip('pinned gfx942 stack required')
    from vkernels.torch_ops.glm_norm_native import rms_norm_native
    d = 128 if kind == 'gated' else 4096
    module=SimpleNamespace(eps=1e-6,variance_epsilon=1e-6)
    if kind != 'unweighted': module.weight=torch.randn(d,device='cuda',dtype=torch.bfloat16)
    with torch.no_grad():
        for sample in range(16):
            x=torch.randn(2,4,d,16,device='cuda',dtype=torch.bfloat16).transpose(-1,-2)
            gate=torch.randn_like(x)*sample if kind == 'gated' else None
            inv=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)
            if kind == 'plain': expected=module.weight*(x.float()*inv).bfloat16()
            elif kind == 'gated': expected=((module.weight.float()*(x.float()*inv))*gate.float().sigmoid()).bfloat16()
            else: expected=x*inv.bfloat16()
            actual=rms_norm_native(x,module,gate)
            assert torch.equal(actual.contiguous().view(torch.int16),expected.contiguous().view(torch.int16))
