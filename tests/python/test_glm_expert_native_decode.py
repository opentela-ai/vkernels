"""FN native conversion preserves the public inference ABI and exact output."""
import subprocess
import sys
import pytest


def test_import_remains_lazy():
    subprocess.run([sys.executable,'-c',"import sys; import vkernels.torch_ops.glm_expert_gemv; assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"],check=True)


@pytest.mark.parametrize('enabled',[False,True])
def test_cpu_declines_without_mutating_inputs(enabled):
    import torch
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv
    from vkernels.torch_ops._dispatch import OpNotEligible
    x=torch.ones(1,128,dtype=torch.bfloat16)
    w=torch.zeros(8,128,128,dtype=torch.uint8).view(torch.float8_e4m3fn)
    s=torch.ones(8,1,1);idx=torch.arange(8).reshape(1,8)
    with pytest.raises(OpNotEligible): expert_gemv(x,w,s,idx,native_decode=enabled)
    assert torch.equal(x,torch.ones_like(x)) and not torch.count_nonzero(w.view(torch.uint8))


@pytest.mark.parametrize('rows',[1,2,3,4])
def test_exact_native_or_fallback_on_target(rows):
    import torch
    if torch.__version__!='2.9.0a0+git7bcbafe' or torch.version.hip!='7.0.51831-a3e329ad8':
        pytest.skip('pinned gfx942 stack required')
    import triton
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]!='gfx942' or triton.__version__!='3.4.0':
        pytest.skip('pinned gfx942 stack required')
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv
    with torch.no_grad():
        for outputs,inputs in [(1024,4096),(4096,512),(128,128)]:
            raw=torch.randint(0,256,(8,outputs,inputs),device='cuda',dtype=torch.uint8)
            # Keep FN negative zero, but avoid its two NaN bytes here.
            raw.masked_fill_((raw & 127)==127,0)
            w=raw.view(torch.float8_e4m3fn);s=torch.rand(8,outputs//128,inputs//128,device='cuda')*.01
            idx=torch.arange(8,device='cuda').repeat(rows,1)
            for sample in range(8):
                x=torch.randn(rows,inputs,device='cuda',dtype=torch.bfloat16)*sample
                a=expert_gemv(x,w,s,idx,t_cap=8)
                b=expert_gemv(x,w,s,idx,t_cap=8,native_decode=True)
                assert torch.equal(a.view(torch.int16),b.view(torch.int16))
