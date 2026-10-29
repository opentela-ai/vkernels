"""Paged prefill positions, FP32 oracle, read-only storage and graph lifetimes."""
import subprocess
import sys

import pytest
import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_gemv import _is_gfx942
from vkernels.torch_ops.prefill_attention import prefill_attention

GFX942 = torch.cuda.is_available() and _is_gfx942(torch.device('cuda'))


def test_import_is_lazy():
    subprocess.run([sys.executable, '-c', 'import sys; import vkernels.torch_ops.prefill_attention; '
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


def _inputs(device, *, n=127, batch=1, heads=32, kv_heads=8, width=None, prefix=5):
    width = width or n + prefix
    pool = batch * width + 19
    # Packed QKV/normalization can leave a noncontiguous query view.
    q = torch.randn(batch, n, heads * 3, 128, device=device, dtype=torch.bfloat16)[:, :, ::3]
    k = torch.randn(pool, kv_heads, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randperm(pool, device=device)[:batch * width].reshape(batch, width).int()
    lengths = torch.full((batch,), width, device=device, dtype=torch.int32)
    positions = torch.arange(prefix, prefix+n, device=device)[None].expand(batch, -1).contiguous()
    return q, k, v, table, lengths, positions


def _oracle(q, k, v, table, lengths, positions):
    keys = k[table.long()].repeat_interleave(q.shape[2] // k.shape[1], dim=2).permute(0, 2, 1, 3).float()
    values = v[table.long()].repeat_interleave(q.shape[2] // v.shape[1], dim=2).permute(0, 2, 1, 3).float()
    columns = torch.arange(table.shape[1], device=q.device)[None, None, :]
    mask = (columns <= positions[:, :, None]) & (columns < lengths[:, None, None])
    return torch.nn.functional.scaled_dot_product_attention(q.permute(0, 2, 1, 3).float(), keys, values,
        attn_mask=mask[:, None]).permute(0, 2, 1, 3)


def test_cpu_contract_falls_back():
    with pytest.raises(OpNotEligible, match='CUDA'):
        prefill_attention(*_inputs('cpu'))


@pytest.mark.parametrize('n', [0, 262145])
def test_unqualified_query_width_falls_back(n):
    q = torch.empty(1, n, 32, 128, device='meta', dtype=torch.bfloat16)
    k = torch.empty(5000, 8, 128, device='meta', dtype=q.dtype)
    with pytest.raises(OpNotEligible, match='range'):
        prefill_attention(q, k, k, torch.empty(1, 262144, device='meta', dtype=torch.int32),
            torch.empty(1, device='meta', dtype=torch.int32), torch.empty(1, n, device='meta', dtype=torch.int64))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available() or GFX942, reason='requires an unsupported GPU architecture')
def test_other_gpu_architecture_falls_back():
    with pytest.raises(OpNotEligible, match='gfx942'):
        prefill_attention(*_inputs('cuda'))


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
@pytest.mark.parametrize('heads,kv_heads', [(32, 8), (16, 2), (8, 2)])
@pytest.mark.parametrize('batch', [1, 2, 3, 4, 8, 16, 32])
@pytest.mark.parametrize('n,prefix', [(32, 0), (127, 5), (129, 17), (511, 0), (2043, 5)])
def test_fp32_oracle_and_read_only_operands(heads, kv_heads, batch, n, prefix):
    torch.manual_seed(541)
    args = _inputs('cuda', n=n, batch=batch, heads=heads, kv_heads=kv_heads, prefix=prefix)
    saved = [t.clone() for t in args]
    with torch.inference_mode():
        actual = prefill_attention(*args)
        expected = _oracle(*args)
    torch.testing.assert_close(actual.float(), expected, rtol=1/128, atol=2e-3)
    assert all(torch.equal(a, b) for a, b in zip(args, saved))


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
@pytest.mark.parametrize('heads,kv_heads', [(32, 8), (16, 2), (8, 2)])
@pytest.mark.parametrize("batch", [4,8,16,32])
def test_ragged_positions_fully_masked_rows_and_dynamic_graph_inputs(heads, kv_heads, batch):
    torch.manual_seed(811)
    args = _inputs('cuda', n=129, batch=batch, heads=heads, kv_heads=kv_heads, prefix=33)
    q, k, v, table, lengths, positions = args
    positions[0, 0] = -1
    positions[1, 70:] = 0
    positions[2, 1:] = 0
    lengths[3] = 0
    with torch.inference_mode():
        prefill_attention(*args)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = prefill_attention(*args)
        for change in ('query', 'kv', 'slots', 'lengths_positions'):
            if change == 'query':
                q.normal_()
            elif change == 'kv':
                k.normal_(); v.normal_()
            elif change == 'slots':
                table.copy_(table.roll(7, dims=1))
            else:
                lengths.sub_(17).clamp_(min=0)
                positions.sub_(9)
            saved = [t.clone() for t in args]
            graph.replay()
            torch.testing.assert_close(result.float(), _oracle(*args), rtol=1/128, atol=2e-3)
            assert all(torch.equal(a, b) for a, b in zip(args, saved))
        assert torch.count_nonzero(result[3]) == 0


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
@pytest.mark.parametrize('heads,kv_heads', [(32, 8), (16, 2), (8, 2)])
def test_maximum_context_and_uniform_scores(heads, kv_heads):
    args = _inputs('cuda', n=4096, batch=1, heads=heads, kv_heads=kv_heads, prefix=0)
    q, k, v, table, lengths, positions = args
    k.zero_()
    with torch.inference_mode():
        torch.testing.assert_close(prefill_attention(*args).float(), _oracle(*args), rtol=1/128, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
@pytest.mark.parametrize('miss', ['dtype', 'metadata', 'grad', 'geometry', 'layout'])
def test_unsupported_contracts_raise_for_sdpa_fallback(miss):
    args = list(_inputs('cuda'))
    if miss == 'dtype':
        args[0] = args[0].float()
    elif miss == 'metadata':
        args[3] = args[3].long()
    elif miss == 'grad':
        args[1].requires_grad_()
    elif miss == 'geometry':
        args[0] = args[0][:, :, :16]
    else:
        args[2] = args[2].transpose(1, 2).contiguous().transpose(1, 2)
    with torch.enable_grad(), pytest.raises(OpNotEligible):
        prefill_attention(*args)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
def test_varying_suffix_lengths_reuse_one_compilation():
    from vkernels.torch_ops.prefill_attention import _kernel

    # A fresh JIT instance also detects extra specializations loaded from the
    # disk cache: a Python compiler-call counter misses those runtime variants.
    _kernel.cache_clear()
    _, kernel = _kernel()
    with torch.inference_mode():
        for n in (64, 65, 73, 75, 80, 121, 125, 128):
            args = _inputs('cuda', n=n)
            torch.testing.assert_close(prefill_attention(*args).float(), _oracle(*args), rtol=1/128, atol=2e-3)
    cache = kernel.device_caches[torch.cuda.current_device()][0]
    assert len(cache) == 1, f'varying suffix lengths created {len(cache)} kernel variants'


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
@pytest.mark.parametrize('heads,kv_heads', [(32, 8), (16, 2), (8, 2)])
@pytest.mark.parametrize('n,width', [(2, 8192), (7, 262144), (31, 32768)])
def test_long_prefix_and_short_chunk_tail(heads, kv_heads, n, width):
    torch.manual_seed(819)
    torch.backends.cuda.matmul.allow_tf32 = False
    args = _inputs('cuda', n=n, width=width, prefix=width-n, heads=heads, kv_heads=kv_heads)
    q, k, v, table, lengths, positions = args
    with torch.inference_mode():
        actual = prefill_attention(*args)
        for kv in range(kv_heads):
            group = heads // kv_heads
            query = q[:, :, kv*group:(kv+1)*group].transpose(1, 2).float()
            keys = k[table.long(), kv].unsqueeze(1).float()
            values = v[table.long(), kv].unsqueeze(1).float()
            mask = torch.arange(width, device=q.device)[None, None, :] <= positions[:, :, None]
            expected = torch.nn.functional.scaled_dot_product_attention(query, keys, values, attn_mask=mask[:, None])
            torch.testing.assert_close(actual[:, :, kv*group:(kv+1)*group].transpose(1, 2).float(), expected, rtol=1/128, atol=2e-3)


@pytest.mark.gpu
@pytest.mark.skipif(not GFX942, reason='paged prefill is qualified on gfx942')
def test_ragged_scratch_padding_can_exceed_model_context():
    torch.manual_seed(629)
    args = _inputs('cuda', n=2, width=262151, prefix=262142, heads=8, kv_heads=2)
    with torch.inference_mode():
        torch.testing.assert_close(prefill_attention(*args).float(), _oracle(*args), rtol=1/128, atol=2e-3)




@pytest.mark.gpu
@pytest.mark.skipif(not GFX942,reason="gfx942 large-plane prefill qualification")
@pytest.mark.parametrize("heads,kvheads",[(32,8),(16,2),(8,2)])
def test_large_plane_sparse_prefill_is_read_only(heads,kvheads):
    batch,n,width=8,2,262144
    pool=(1<<33)//(kvheads*128*2)+1
    q=torch.ones(batch,n,heads,128,device="cuda",dtype=torch.bfloat16)
    k=torch.empty(pool,kvheads,128,device="cuda",dtype=q.dtype);v=torch.empty_like(k)
    k[-1].fill_(.5);v[-1].fill_(.25)
    table=torch.full((batch,width),pool-1,device="cuda",dtype=torch.int32)
    lengths=torch.full((batch,),width,device="cuda",dtype=torch.int32);lengths[3]=0
    positions=torch.tensor([width-2,width-1],device="cuda").expand(batch,-1).contiguous()
    with torch.inference_mode():actual=prefill_attention(q,k,v,table,lengths,positions)
    expected=torch.full_like(q,.25);expected[3]=0
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    torch.testing.assert_close(k[-1].cpu(),torch.full((kvheads,128),.5,dtype=q.dtype),rtol=0,atol=0)
    torch.testing.assert_close(v[-1].cpu(),torch.full((kvheads,128),.25,dtype=q.dtype),rtol=0,atol=0)
