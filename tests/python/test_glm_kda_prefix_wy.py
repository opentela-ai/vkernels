"""WY prefix matches independent truncated forwards, preserving replay inputs."""
import subprocess
import sys
import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, '-c',
        "import sys; import vkernels.torch_ops.glm_kda_prefix_wy; "
        "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


def inputs(torch, device, tokens=4, dtype=None):
    generator=torch.Generator(device=device).manual_seed(941)
    key=torch.randn(6,16,tokens,128,device=device,generator=generator)
    key=torch.nn.functional.normalize(key,dim=-1)
    value=torch.randn(key.shape,device=device,generator=generator)
    gate=-torch.rand(key.shape,device=device,generator=generator)
    beta=torch.rand(6,16,tokens,device=device,generator=generator)
    state=torch.randn(6,16,128,128,device=device,generator=generator)*0.1
    state[0].fill_(-0.0)
    lengths=torch.tensor([-1,0,1,2,3,99],device=device,dtype=dtype or torch.int32)
    return key,value,gate,beta,state,lengths


def oracle(torch, args):
    from torch.nn import functional as F
    from vkernels.torch_ops.vllm_kda import kda_chunk_floe
    key,value,gate,beta,state,lengths=args
    outputs=[]
    for row,count in enumerate(lengths.tolist()):
        count=max(0,min(key.shape[2],count))
        if count==0:
            outputs.append(state[row:row+1].clone())
        else:
            k,v,g=[F.pad(t[row:row+1,:,:count],(0,0,0,32-count)).contiguous() for t in (key,value,gate)]
            b=F.pad(beta[row:row+1,:,:count],(0,32-count)).contiguous()
            _,last=kda_chunk_floe(k,k,v,g,b,chunk_size=32,initial_state=state[row:row+1],output_final_state=True)
            outputs.append(last)
    return torch.cat(outputs)


def test_cpu_declines_before_backend_import():
    torch=pytest.importorskip('torch')
    from vkernels.torch_ops._dispatch import OpNotEligible
    from vkernels.torch_ops.glm_kda_prefix_wy import kda_prefix_state_wy
    with pytest.raises(OpNotEligible,match='CUDA'):
        kda_prefix_state_wy(*inputs(torch,'cpu'))


@pytest.mark.parametrize('tokens',[3,4])
@pytest.mark.parametrize('length_dtype',['int32','int64'])
def test_truncated_forward_parity_and_changed_input_replay(tokens,length_dtype):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('GPU required')
    from vkernels.torch_ops.glm_kda_prefix_wy import kda_prefix_state_wy
    args=inputs(torch,'cuda',tokens,getattr(torch,length_dtype));before=[v.clone() for v in args]
    with torch.no_grad():
        actual=kda_prefix_state_wy(*args)
        torch.testing.assert_close(actual,oracle(torch,args),atol=1e-5,rtol=1e-4)
        assert torch.equal(actual[0].view(torch.int32),args[4][0].view(torch.int32))
        assert all(torch.equal(v,old) for v,old in zip(args,before))
        kda_prefix_state_wy(*args)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=kda_prefix_state_wy(*args)
        args[0].neg_();args[1].mul_(0.5);args[2].mul_(0.9)
        args[-1].copy_(torch.tensor([tokens,3,0,-1,2,1],device='cuda',dtype=args[-1].dtype))
        graph.replay()
        torch.testing.assert_close(captured,oracle(torch,args),atol=1e-5,rtol=1e-4)
        for row in (2,3):
            assert torch.equal(captured[row].view(torch.int32),args[4][row].view(torch.int32))
