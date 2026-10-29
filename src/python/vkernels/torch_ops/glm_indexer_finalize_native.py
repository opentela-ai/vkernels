"""Integer-only GLM indexer emission, preserving native selection order."""
from functools import lru_cache
from ._dispatch import OpNotEligible


@lru_cache(maxsize=2)
def _kernel(dynamic_geometry=False):
    global tl
    import triton
    import triton.language as tl

    dynamic = ["K", "L", "IB", "IK", "IS", "SB", "SK", "VB", "VL", "KB", "KL", "MB"]
    def _emit(Indices,SelectedValid,Visible,KeyValid,Mask,Out,
              K,L,W:tl.constexpr,
              IB,IK,IS,
              SB,SK,VB,VL,
              KB,KL,MB,
              BLOCK_L:tl.constexpr,BLOCK_W:tl.constexpr):
        row=tl.program_id(0)
        pos=tl.arange(0,BLOCK_L)
        keys=tl.load(KeyValid+row*KB+pos*KL,pos<L,other=0).to(tl.int1)
        visible=tl.load(Visible+row*VB+pos*VL,pos<L,other=0).to(tl.int1)
        first=tl.min(tl.where(keys,pos,L),0)
        count=tl.sum(visible.to(tl.int32),0)
        tail_count=count%4
        start=first+count-tail_count
        out_pos=tl.arange(0,BLOCK_W)
        in_picks=out_pos<K*4
        pool=out_pos//4
        lane=out_pos%4
        pick=tl.load(Indices+row*IB+pool*IK+lane*IS,in_picks,other=-1).to(tl.int32)
        good=tl.load(SelectedValid+row*SB+pool*SK,in_picks,other=0).to(tl.int1)
        pick=tl.where(good,pick,-1)
        tail_offset=out_pos-K*4
        tail_id=start+tail_offset
        tail_slot=(tail_offset>=0)&(tail_offset<3)
        safe=tl.minimum(tl.maximum(tail_id,0),L-1)
        tail_visible=tl.load(Visible+row*VB+safe*VL,tail_slot,other=0).to(tl.int1)
        tail_good=tail_slot&(tail_offset<tail_count)&(tail_id<L)&tail_visible
        value=tl.where(in_picks,pick,tl.where(tail_good,tail_id,-1))
        enabled=tl.load(Mask+row*MB).to(tl.int1)
        value=tl.where(enabled,value,-1).to(tl.int32)
        tl.store(Out+row*W+out_pos,value,out_pos<W)

    # Triton 3.4 resolves source annotations during compilation; local type
    # aliases are not supported. Bind static scalar annotations before JIT.
    if not dynamic_geometry:
        _emit.__annotations__.update({name: tl.constexpr for name in dynamic})
    return triton.jit(_emit, do_not_specialize=dynamic if dynamic_geometry else [])


def glm_indexer_native_finalize(selected_indices, selected_valid, visible,
                                valid_keys, mask, *, dynamic_geometry=False):
    """Emit native picked tokens and visible tail, or decline unsupported inputs.

    Pool selection and its order are supplied by the unchanged Torch top-k.
    Private MTP uses dynamic_geometry to avoid compiling each growing context
    length and pool count. Its integer/bool math and reduction tiles are unchanged.
    Only integer/bool operations are fused; the selected prefix, tail placement,
    invalid picks, output padding and int32 conversion retain their native order.
    """
    import torch

    if not isinstance(dynamic_geometry, bool):
        raise OpNotEligible("dynamic geometry selection must be boolean")
    if selected_indices.ndim != 4 or selected_indices.dtype != torch.int64:
        raise OpNotEligible("requires int64 selected pools with four dimensions")
    batch, width, k, pool = selected_indices.shape
    if visible.ndim != 3:
        raise OpNotEligible("requires a three-dimensional visibility mask")
    length = visible.shape[-1]
    if not (1 <= batch <= 4 and width == 1 and pool == 4 and 0 <= k <= 512
            and 1 <= length <= 131072):
        raise OpNotEligible("unqualified native finalizer geometry")
    for tensor, shape in [(selected_valid, (batch, 1, k)),
                          (visible, (batch, 1, length)),
                          (valid_keys, (batch, length)), (mask, (batch, 1))]:
        if tensor.shape != shape or tensor.dtype != torch.bool:
            raise OpNotEligible("requires matching boolean selection/visibility masks")
    tensors = (selected_indices, selected_valid, visible, valid_keys, mask)
    if not selected_indices.is_cuda or any(t.device != selected_indices.device
                                           or any(s < 0 for s in t.stride())
                                           for t in tensors):
        raise OpNotEligible("requires nonnegative strides on one HIP device")
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        raise OpNotEligible("unqualified Torch/HIP stack")
    if torch.cuda.get_device_properties(selected_indices.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("unqualified GPU architecture")
    import triton
    if triton.__version__ != "3.4.0":
        raise OpNotEligible("unqualified Triton stack")
    output = torch.empty((batch, 1, 2051), device=selected_indices.device, dtype=torch.int32)
    _kernel(dynamic_geometry)[(batch,)](selected_indices, selected_valid, visible, valid_keys, mask, output,
        k, length, 2051, selected_indices.stride(0), selected_indices.stride(2), selected_indices.stride(3),
        selected_valid.stride(0), selected_valid.stride(2), visible.stride(0), visible.stride(2),
        valid_keys.stride(0), valid_keys.stride(1), mask.stride(0), triton.next_power_of_2(length), 4096,
        num_warps=4 if length <= 8192 else 8)
    return output


@lru_cache(maxsize=1)
def _gather_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _emit(Indices,Selected,Candidates,Visible,KeyValid,Mask,Out,
              K:tl.constexpr,P:tl.constexpr,C:tl.constexpr,L:tl.constexpr,W:tl.constexpr,
              IB:tl.constexpr,IK:tl.constexpr,IS:tl.constexpr,
              SB:tl.constexpr,SK:tl.constexpr,CB:tl.constexpr,CC:tl.constexpr,VB:tl.constexpr,VL:tl.constexpr,
              KB:tl.constexpr,KL:tl.constexpr,MB:tl.constexpr,
              BLOCK_L:tl.constexpr,BLOCK_W:tl.constexpr):
        row=tl.program_id(0)
        pos=tl.arange(0,BLOCK_L)
        keys=tl.load(KeyValid+row*KB+pos*KL,pos<L,other=0).to(tl.int1)
        visible=tl.load(Visible+row*VB+pos*VL,pos<L,other=0).to(tl.int1)
        first=tl.min(tl.where(keys,pos,L),0)
        count=tl.sum(visible.to(tl.int32),0)
        tail_count=count%4
        start=first+count-tail_count
        out_pos=tl.arange(0,BLOCK_W)
        in_picks=out_pos<K*4
        pool=out_pos//4
        lane=out_pos%4
        chosen_pool=tl.load(Selected+row*SB+pool*SK,in_picks,other=0)
        pool_present=(chosen_pool>=0)&(chosen_pool<P)
        pick=tl.load(Indices+row*IB+chosen_pool*IK+lane*IS,in_picks&pool_present,other=0).to(tl.int32)
        good=tl.load(Candidates+row*CB+chosen_pool*CC,in_picks&(chosen_pool>=0)&(chosen_pool<C),other=0).to(tl.int1)
        pick=tl.where(good,pick,-1)
        tail_offset=out_pos-K*4
        tail_id=start+tail_offset
        tail_slot=(tail_offset>=0)&(tail_offset<3)
        safe=tl.minimum(tl.maximum(tail_id,0),L-1)
        tail_visible=tl.load(Visible+row*VB+safe*VL,tail_slot,other=0).to(tl.int1)
        tail_good=tail_slot&(tail_offset<tail_count)&(tail_id<L)&tail_visible
        value=tl.where(in_picks,pick,tl.where(tail_good,tail_id,-1))
        enabled=tl.load(Mask+row*MB).to(tl.int1)
        value=tl.where(enabled,value,-1).to(tl.int32)
        tl.store(Out+row*W+out_pos,value,out_pos<W)

    return _emit


def glm_indexer_native_gather_finalize(pool_indices, selected, candidates,
                                      visible, valid_keys, mask):
    """Gather native Torch top-k pool picks and emit the unchanged token order.

    ``selected`` must contain pool offsets in ``[0, candidates.shape[-1])``
    from the native top-k. Physical pools missing beyond the cached table
    retain the original zero-padding semantics. No floating math is fused.
    """
    import torch

    if selected.ndim != 3 or selected.dtype != torch.int64 or pool_indices.ndim != 3:
        raise OpNotEligible("requires int64 top-k offsets and three-dimensional pool ids")
    batch, width, k = selected.shape
    if candidates.ndim != 3 or visible.ndim != 3:
        raise OpNotEligible("requires three-dimensional candidate/visibility masks")
    pools, columns, length = pool_indices.shape[1], candidates.shape[-1], visible.shape[-1]
    if not (1 <= batch <= 4 and width == 1 and 0 <= k <= 512
            and 0 <= pools <= 32768 and 0 <= columns <= 32768
            and 1 <= length <= 131072):
        raise OpNotEligible("unqualified native gather/emit geometry")
    if pool_indices.shape != (batch, pools, 4) or pool_indices.dtype != torch.int64:
        raise OpNotEligible("requires native int64 four-token pools")
    for tensor, shape in [(candidates, (batch, 1, columns)),
                          (visible, (batch, 1, length)),
                          (valid_keys, (batch, length)), (mask, (batch, 1))]:
        if tensor.shape != shape or tensor.dtype != torch.bool:
            raise OpNotEligible("requires matching boolean candidate/visibility masks")
    tensors = (pool_indices, selected, candidates, visible, valid_keys, mask)
    if not pool_indices.is_cuda or any(t.device != pool_indices.device
                                      or any(s < 0 for s in t.stride()) for t in tensors):
        raise OpNotEligible("requires nonnegative strides on one HIP device")
    if str(torch.__version__) != "2.9.0a0+git7bcbafe" or torch.version.hip != "7.0.51831-a3e329ad8":
        raise OpNotEligible("unqualified Torch/HIP stack")
    if torch.cuda.get_device_properties(pool_indices.device).gcnArchName.split(":", 1)[0] != "gfx942":
        raise OpNotEligible("unqualified GPU architecture")
    import triton
    if triton.__version__ != "3.4.0":
        raise OpNotEligible("unqualified Triton stack")
    output = torch.empty((batch, 1, 2051), device=pool_indices.device, dtype=torch.int32)
    _gather_kernel()[(batch,)](pool_indices, selected, candidates, visible, valid_keys, mask, output,
        k, pools, columns, length, 2051, pool_indices.stride(0), pool_indices.stride(1), pool_indices.stride(2),
        selected.stride(0), selected.stride(2), candidates.stride(0), candidates.stride(2), visible.stride(0), visible.stride(2),
        valid_keys.stride(0), valid_keys.stride(1), mask.stride(0), triton.next_power_of_2(length), 4096,
        num_warps=4 if length <= 8192 else 8)
    return output
