"""Decode-order short KDA blocks for the pinned Beverin inference stack.

For BF16 [B,T,16,128] raw vectors and FP32 initial state, one launch repeats
incumbent kda_decode arithmetic and returns FP32 outputs plus every prefix
state. This is an experimental component, not a model equivalence claim:
callers must separately preserve convolution and collective numerics.
"""
from functools import lru_cache
from ._dispatch import OpNotEligible, same_gpu_contiguous


@lru_cache(maxsize=1)
def _kernels():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def recurrent_block(Q,K,V,G,BETA,STATE,OUT,PREFIX,T:tl.constexpr,H:tl.constexpr,D:tl.constexpr=128,BV:tl.constexpr=32):
     head=tl.program_id(0);block=tl.program_id(1)
     batch=head//H;h=head%H;rows=tl.arange(0,D);cols=block*BV+tl.arange(0,BV)
     so=head*D*D+rows[:,None]*D+cols[None,:]
     state=tl.load(STATE+so,cols[None,:]<D,0)
     for token in range(T):
      offset=((batch*T+token)*H+h)*D
      q=tl.load(Q+offset+rows).to(tl.float32);k=tl.load(K+offset+rows).to(tl.float32)
      q=tl.div_rn(q,tl.sqrt(tl.sum(q*q,0)+1e-6));k=tl.div_rn(k,tl.sqrt(tl.sum(k*k,0)+1e-6));q=q*(D**-0.5)
      decay=tl.exp(tl.load(G+offset+rows).to(tl.float32))
      state=state*decay[:,None]
      memory=tl.sum(state*k[:,None],0)
      value=tl.load(V+offset+cols,cols<D,0).to(tl.float32)
      delta=(value-memory)*tl.load(BETA+(batch*T+token)*H+h).to(tl.float32)
      state=state+k[:,None]*delta[None,:]
      output=tl.sum(state*q[:,None],0)
      tl.store(OUT+offset+cols,output,cols<D)
      po=((batch*T+token)*H+h)*D*D+rows[:,None]*D+cols[None,:]
      tl.store(PREFIX+po,state,cols[None,:]<D)

    return triton, recurrent_block


def kda_recurrent_block(query, key, value, gate, beta, initial_state):
    """Return (FP32 [B,T,H,D], FP32 [B,T,H,D,D]) without input mutation.

    Pinned gfx942/Torch2.9/HIP7/Triton3.4 inference cells: B1-4, T2-4,
    H16, D128. Vectors/gate/beta are contiguous BF16 tensors on one GPU;
    state is contiguous FP32. Unsupported contracts raise OpNotEligible.
    Prefix state t is the state after consuming token t, in K-major order.
    Eager warmup must precede graph capture. No backend settings change.
    """
    import torch

    if query.ndim != 4:
        raise OpNotEligible("expected raw query [B,T,16,128]")
    batch, width, heads, dim = query.shape
    if not (1 <= batch <= 4 and 2 <= width <= 4 and heads == 16 and dim == 128):
        raise OpNotEligible("unmeasured short KDA geometry")
    if any(x.shape != query.shape for x in (key, value, gate)):
        raise OpNotEligible("query/key/value/gate shapes must match")
    if beta.shape != (batch, width, heads) or initial_state.shape != (batch, heads, dim, dim):
        raise OpNotEligible("incorrect beta or recurrent state shape")
    vectors = (query, key, value, gate, beta)
    if any(x.dtype != torch.bfloat16 for x in vectors) or initial_state.dtype != torch.float32:
        raise OpNotEligible("requires BF16 vectors and FP32 recurrent state")
    tensors = (*vectors, initial_state)
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise OpNotEligible("inference only")
    same_gpu_contiguous(*tensors)
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        raise OpNotEligible("unqualified Torch/HIP stack")
    if torch.cuda.get_device_properties(query.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("qualified on gfx942 only")
    import triton
    if triton.__version__ != "3.4.0":
        raise OpNotEligible("unqualified Triton version")
    triton, kernel = _kernels()
    output = torch.empty(query.shape, device=query.device, dtype=torch.float32)
    prefixes = torch.empty((batch, width, heads, dim, dim), device=query.device, dtype=torch.float32)
    with torch.cuda.device(query.device):
        kernel[(batch * heads, triton.cdiv(dim, 32))](
            query, key, value, gate, beta, initial_state, output, prefixes,
            width, heads, num_warps=4, enable_fp_fusion=False)
    return output, prefixes
