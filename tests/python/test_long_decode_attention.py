"""Long decode fresh-store ownership, dynamic graph inputs and FP32 parity."""

import subprocess
import sys
import pytest
import torch
from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_gemv import _is_gfx942
from vkernels.torch_ops.long_decode_attention import decode_attention_long

GFX942 = torch.cuda.is_available() and _is_gfx942(torch.device("cuda"))


def test_lazy_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.long_decode_attention; "
            "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules",
        ],
        check=True,
    )


def test_cpu_is_ineligible():
    q = torch.zeros(1, 8, 128, dtype=torch.bfloat16)
    cache = torch.zeros(8193, 2, 128, dtype=q.dtype)
    with pytest.raises(OpNotEligible, match="CUDA"):
        decode_attention_long(
            q,
            cache,
            cache,
            torch.zeros(1, 8193, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int64),
            max_len_hint=8193,
        )


def reference(q, k, v, table, lengths, kn, vn):
    # Keep one full FP32 softmax, independent of the kernel's online merge.
    # Chunk only the products: this image's FP32 BLAS path for a 262K
    # reduction stalls/aborts, and full gathered KV needlessly consumes GiB.
    out = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
    hk, group, step = k.shape[1], q.shape[1] // k.shape[1], 4096
    for row in range(q.shape[0]):
        n = int(lengths[row])
        if not n:
            continue
        query = q[row].float().reshape(hk, group, 128).contiguous()
        scores = torch.empty(hk, group, n, device=q.device, dtype=torch.float32)
        for start in range(0, n, step):
            end = min(start + step, n)
            keys = k[table[row, start:end].long()].float()
            if kn is not None and end == n:
                keys[-1] = kn[row].float()
            scores[..., start:end] = torch.bmm(query, keys.permute(1, 2, 0).contiguous()) * (128**-.5)
        probability = scores.softmax(-1)
        accumulator = torch.zeros_like(query)
        for start in range(0, n, step):
            end = min(start + step, n)
            values = v[table[row, start:end].long()].float()
            if vn is not None and end == n:
                values[-1] = vn[row].float()
            accumulator += torch.bmm(probability[..., start:end].contiguous(), values.permute(1, 0, 2).contiguous())
        out[row] = accumulator.reshape(q.shape[1], 128)
    return out


def test_chunked_oracle_matches_dense_fp32_sdpa():
    torch.manual_seed(929)
    width = 8193
    q = torch.randn(1,8,128,dtype=torch.bfloat16)
    k = torch.randn(width+1,2,128,dtype=q.dtype);v = torch.randn_like(k)
    table = torch.arange(1,width+1,dtype=torch.int32)[None]
    lengths = torch.tensor([width],dtype=torch.int32)
    kn = torch.randn(1,2,128,dtype=q.dtype);vn = torch.randn_like(kn)
    keys = k[table.long()].float();values = v[table.long()].float()
    keys[0,-1]=kn[0].float();values[0,-1]=vn[0].float()
    expected = torch.nn.functional.scaled_dot_product_attention(q.float()[:,:,None],
        keys.repeat_interleave(4,dim=2).permute(0,2,1,3),
        values.repeat_interleave(4,dim=2).permute(0,2,1,3))[:,:,0]
    actual = reference(q,k,v,table,lengths,kn,vn)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason="qualified on gfx942")
@pytest.mark.parametrize("heads,kvheads", [(32, 8), (16, 2), (8, 2)])
@pytest.mark.parametrize("batch,width", [(1,8193),(4,8193),(1,262144),(4,262144),(8,8193),(16,8193),(32,8193),(32,262144)])
@pytest.mark.parametrize("fresh", [False, True])
def test_fp32_oracle_and_slot_ownership(heads, kvheads, batch, width, fresh):
    torch.manual_seed(329)
    torch.backends.cuda.matmul.allow_tf32 = False
    # Model KV budgets cannot hold B independent native contexts at B32.
    # Share immutable history through a bounded physical pool; reserve a
    # distinct fresh slot for every row so write ownership stays testable.
    pool = min(batch * width + 1, max(width + batch + 1, 1 << 19))
    q = torch.randn(batch, heads * 3, 128, device="cuda", dtype=torch.bfloat16)[:, ::3]
    k = torch.randn(pool, kvheads, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    if pool == batch * width + 1:
        table = (torch.randperm(pool - 1, device="cuda").reshape(batch, width) + 1).int()
    else:
        table = torch.stack([torch.randperm(pool - batch - 1, device="cuda")[:width] + 1 for _ in range(batch)]).int()
    lengths = torch.tensor(([width, width - 17, 0, 511] * ((batch + 3) // 4))[:batch], device="cuda", dtype=torch.int32)
    if pool != batch * width + 1:
        for row in range(batch):
            if int(lengths[row]):
                table[row, int(lengths[row])-1] = pool - batch + row
    if batch >= 4:
        table[1, int(lengths[1]) - 1] = 0
    kn = torch.randn(batch, kvheads * 2, 128, device="cuda", dtype=q.dtype)[:, ::2] if fresh else None
    vn = torch.randn_like(kn) if fresh else None
    before_k, before_v = k.clone(), v.clone()
    with torch.inference_mode():
        expected = reference(q, k, v, table, lengths, kn, vn)
        actual = decode_attention_long(q, k, v, table, lengths, max_len_hint=width, k_new=kn, v_new=vn, scratch_slot=0)
    torch.testing.assert_close(actual.float(), expected, rtol=1e-3, atol=1e-3)
    if fresh:
        for row in range(batch):
            n = int(lengths[row])
            slot = int(table[row, n - 1]) if n else 0
            if slot:
                torch.testing.assert_close(k[slot], kn[row], rtol=0, atol=0)
                torch.testing.assert_close(v[slot], vn[row], rtol=0, atol=0)
                before_k[slot] = kn[row]
                before_v[slot] = vn[row]
    assert torch.equal(k, before_k) and torch.equal(v, before_v)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason="qualified on gfx942")
@pytest.mark.parametrize("batch", [4,8,16,32])
def test_graph_reads_current_inputs_and_zero_rows(batch):
    torch.manual_seed(553)
    torch.backends.cuda.matmul.allow_tf32 = False
    width = 32768
    q = torch.randn(batch, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * width + 1, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    table = torch.arange(1, batch * width + 1, device="cuda").reshape(batch, width).int()
    lengths = torch.tensor(([width,8193,1,0] * ((batch+3)//4))[:batch], device="cuda", dtype=torch.int64)
    kn = torch.randn(batch, 2, 128, device="cuda", dtype=q.dtype)
    vn = torch.randn_like(kn)

    def run():
        return decode_attention_long(q, k, v, table, lengths, max_len_hint=width, k_new=kn, v_new=vn, scratch_slot=0)

    with torch.inference_mode():
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = run()
        for _ in range(3):
            q.normal_()
            kn.normal_()
            vn.normal_()
            table.copy_(table.roll(17, 1))
            lengths[1] += 31
            expected = reference(q, k, v, table, lengths, kn, vn)
            graph.replay()
            torch.testing.assert_close(result.float(), expected, rtol=1e-3, atol=1e-3)
            assert not torch.count_nonzero(result[3])



@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason="gfx942 large-plane address qualification")
@pytest.mark.parametrize("heads,kvheads",[(32,8),(16,2),(8,2)])
@pytest.mark.parametrize("fresh",[False,True])
def test_large_plane_sparse_addresses_and_graph_inputs(heads,kvheads,fresh):
    batch,width=32,262144
    pool=(1<<33)//(kvheads*128*2)+1
    q=torch.ones(batch,heads,128,device="cuda",dtype=torch.bfloat16)
    k=torch.empty(pool,kvheads,128,device="cuda",dtype=q.dtype)
    v=torch.empty_like(k)
    k[-1].fill_(.5);v[-1].fill_(.25)
    k[-66].fill_(.5);v[-66].fill_(.125)
    k[-batch-1:-1].zero_();v[-batch-1:-1].zero_()
    k[0].fill_(.125);v[0].fill_(.375)
    table=torch.full((batch,width),pool-1,device="cuda",dtype=torch.int32)
    lengths=torch.full((batch,),2 if fresh else 1,device="cuda",dtype=torch.int32)
    lengths[3]=0
    if fresh:
        table[:,1]=torch.arange(pool-2,pool-batch-2,-1,device="cuda",dtype=torch.int32)
        table[1,1]=0
    kn=torch.full((batch,kvheads,128),.5,device="cuda",dtype=q.dtype) if fresh else None
    vn=torch.full_like(kn,.75) if fresh else None
    def run():
        return decode_attention_long(q,k,v,table,lengths,max_len_hint=width,k_new=kn,v_new=vn,scratch_slot=0)
    with torch.inference_mode():
        expected=torch.full_like(q,.5 if fresh else .25);expected[3]=0
        torch.testing.assert_close(run(),expected,rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): replayed=run()
        q.normal_();table[:,0]=pool-66
        lengths[5]=0
        graph.replay()
        expected.fill_(.4375 if fresh else .125);expected[3]=0;expected[5]=0
        torch.testing.assert_close(replayed,expected,rtol=0,atol=0)
    torch.testing.assert_close(k[-1].cpu(),torch.full((kvheads,128),.5,dtype=q.dtype),rtol=0,atol=0)
    torch.testing.assert_close(v[-1].cpu(),torch.full((kvheads,128),.25,dtype=q.dtype),rtol=0,atol=0)
    assert torch.equal(k[0].cpu(),torch.full((kvheads,128),.125,dtype=q.dtype))
    assert torch.equal(v[0].cpu(),torch.full((kvheads,128),.375,dtype=q.dtype))
    if fresh:
        for row in range(batch):
            if row in (1,3):continue
            torch.testing.assert_close(k[pool-2-row].cpu(),kn[row].cpu(),rtol=0,atol=0)
            torch.testing.assert_close(v[pool-2-row].cpu(),vn[row].cpu(),rtol=0,atol=0)
