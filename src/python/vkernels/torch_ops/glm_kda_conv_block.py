"""Decode-rounded short KDA convolution with every raw prefix history.

Experimental pinned Beverin component. Retains BF16 product/store rounding,
the native adjacent-pair FP32 sum with +0 identity, and native Torch SiLU.
Torch and Triton remain lazy; no model dispatch or defaults change.
"""
from functools import lru_cache
from ._dispatch import OpNotEligible, same_gpu_contiguous


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def conv_pre(S,X,W,Y,P,C:tl.constexpr,T:tl.constexpr,BLOCK:tl.constexpr):
     b=tl.program_id(0);c=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK);mask=c<C
     s0=tl.load(S+b*C*3+c*3,mask,0).to(tl.float32)
     s1=tl.load(S+b*C*3+c*3+1,mask,0).to(tl.float32)
     s2=tl.load(S+b*C*3+c*3+2,mask,0).to(tl.float32)
     w0=tl.load(W+c*4,mask,0).to(tl.float32);w1=tl.load(W+c*4+1,mask,0).to(tl.float32)
     w2=tl.load(W+c*4+2,mask,0).to(tl.float32);w3=tl.load(W+c*4+3,mask,0).to(tl.float32)
     for token in range(T):
      value=tl.load(X+b*C*T+c*T+token,mask,0).to(tl.float32)
      p0=(s0*w0).to(tl.bfloat16).to(tl.float32)
      p1=(s1*w1).to(tl.bfloat16).to(tl.float32)
      p2=(s2*w2).to(tl.bfloat16).to(tl.float32)
      p3=(value*w3).to(tl.bfloat16).to(tl.float32)
      acc=tl.full((BLOCK,),0.0,tl.float32)+((p0+p1)+(p2+p3))
      tl.store(Y+b*C*T+c*T+token,acc.to(tl.bfloat16),mask)
      s0=s1;s1=s2;s2=value
      offset=(b*T+token)*C*3+c*3
      tl.store(P+offset,s0.to(tl.bfloat16),mask);tl.store(P+offset+1,s1.to(tl.bfloat16),mask);tl.store(P+offset+2,s2.to(tl.bfloat16),mask)

    return triton, conv_pre


def kda_conv_block(raw, initial_state, weight):
    """Return BF16 activations [B,C,T] and prefix histories [B,T,C,3].

    Eligible cells: contiguous BF16 B1-4/C6144/T2-4 raw projections,
    initial history [B,C,3], weight [C,1,4], all on one pinned gfx942 GPU.
    Every prefix is the raw three-column history after consuming that
    token. Inputs are read-only; no autograd or process-global changes.
    Unsupported contracts raise OpNotEligible. Warm eagerly before capture.
    """
    import torch
    import torch.nn.functional as F

    if raw.ndim != 3:
        raise OpNotEligible("expected raw projections [B,6144,T]")
    batch, channels, width = raw.shape
    if not (1 <= batch <= 4 and channels == 6144 and 2 <= width <= 4):
        raise OpNotEligible("unmeasured short convolution geometry")
    if initial_state.shape != (batch, channels, 3) or weight.shape != (channels, 1, 4):
        raise OpNotEligible("expected history [B,C,3] and filter [C,1,4]")
    tensors = (raw, initial_state, weight)
    if any(x.dtype != torch.bfloat16 for x in tensors):
        raise OpNotEligible("requires BF16 projections, history and filters")
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise OpNotEligible("inference only")
    same_gpu_contiguous(*tensors)
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        raise OpNotEligible("unqualified Torch/HIP stack")
    if torch.cuda.get_device_properties(raw.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("qualified on gfx942 only")
    import triton
    if triton.__version__ != "3.4.0":
        raise OpNotEligible("unqualified Triton version")
    triton, kernel = _kernels()
    preactivation = torch.empty_like(raw)
    histories = torch.empty((batch, width, channels, 3), device=raw.device, dtype=raw.dtype)
    with torch.cuda.device(raw.device):
        kernel[(batch, triton.cdiv(channels, 256))](
            initial_state, raw, weight, preactivation, histories,
            channels, width, 256, num_warps=4, enable_fp_fusion=False)
        output = F.silu(preactivation)
    return output, histories
