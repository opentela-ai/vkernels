"""Selected attention preserves request isolation and selection multiplicity."""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.selected_attention_sdpa import selected_attention_sdpa


def oracle(q, kv, indices, scale):
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    valid = (indices >= 0) & (indices < kv.shape[1])
    gathered = kv[batch, indices.clamp(0, kv.shape[1] - 1).long()].double()
    scores = torch.einsum("bshd,bswd->bshw", q.double(), gathered) * scale
    scores = scores.masked_fill(~valid[:, :, None], -torch.inf)
    p = torch.softmax(scores, -1).nan_to_num()
    return torch.einsum("bshw,bswd->bshd", p, gathered).to(q.dtype)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
@pytest.mark.parametrize("batch,seq,heads,dim,count,width", [(1, 17, 2, 32, 31, 47), (4, 128, 16, 512, 128, 131)])
@pytest.mark.parametrize("strided", [False, True])
def test_multiset_selection_and_request_isolation(batch, seq, heads, dim, count, width, strided):
    torch.manual_seed(821)
    q = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16) * 0.5
    kv = torch.randn(batch, count, dim, device="cuda", dtype=q.dtype)
    idx = torch.randint(0, count, (batch, seq, width), device="cuda", dtype=torch.int64)
    idx[..., ::3] = -1
    idx[..., 1:3] = 7  # multiplicity must survive; a boolean set mask is incorrect.
    idx[:, 0] = -1
    idx[:, 1] = count  # never read a different request's cache row.
    if strided:
        q = q.transpose(1, 2).contiguous().transpose(1, 2)
        kv = kv.transpose(1, 2).contiguous().transpose(1, 2)
        idx = idx.transpose(1, 2).contiguous().transpose(1, 2)
    originals = [x.clone() for x in (q, kv, idx)]
    with torch.inference_mode():
        actual = selected_attention_sdpa(q, kv, idx, 0.0625)
        expected = oracle(q, kv, idx, 0.0625)
    assert torch.isfinite(actual).all()
    assert not actual[:, :2].any()
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.01)
    for x, before in zip((q, kv, idx), originals):
        assert torch.equal(x, before)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
def test_replay_observes_changed_queries_cache_and_selections():
    torch.manual_seed(97)
    q = torch.randn(2, 9, 4, 32, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 19, 32, device=q.device, dtype=q.dtype)
    idx = torch.randint(-1, 19, (2, 9, 23), device=q.device, dtype=torch.int32)
    with torch.inference_mode():
        for _ in range(3):
            selected_attention_sdpa(q, kv, idx, 0.125)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = selected_attention_sdpa(q, kv, idx, 0.125)
        for _ in range(3):
            q.normal_()
            kv.normal_()
            idx.random_(-1, 19)
            idx[:, 2] = -1
            graph.replay()
            torch.testing.assert_close(output, oracle(q, kv, idx, 0.125), atol=0.008, rtol=0.01)
            assert not output[:, 2].any()


def test_cpu_declines_before_execution():
    with pytest.raises(OpNotEligible):
        selected_attention_sdpa(torch.zeros(1, 3, 2, 32), torch.zeros(1, 7, 32), torch.zeros(1, 3, 9), 0.1)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA")
def test_bounds_and_empty_shapes():
    q = torch.zeros(1, 3, 2, 32, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(1, 7, 32, device=q.device, dtype=q.dtype)
    idx = torch.zeros(1, 3, 9, device=q.device, dtype=torch.int32)
    for args in ((q, kv, idx.float()), (q, kv.expand(2, -1, -1), idx), (q, kv, idx[:, :2]), (q.float(), kv, idx)):
        with pytest.raises(OpNotEligible):
            selected_attention_sdpa(*args, 0.1)
    with pytest.raises(OpNotEligible):
        selected_attention_sdpa(q, kv, idx, float("nan"))
    with pytest.raises(OpNotEligible):
        selected_attention_sdpa(q.requires_grad_(), kv, idx, 0.1)
    with torch.inference_mode():
        for qs, ks, ids in ((q[:, :0], kv, idx[:, :0]), (q, kv[:, :0], idx), (q, kv, idx[:, :, :0])):
            result = selected_attention_sdpa(qs, ks, ids, 0.1)
            assert result.shape == qs.shape and not result.any()
        with pytest.raises(OpNotEligible):
            selected_attention_sdpa(q, torch.zeros(1, 513, 32, device=q.device, dtype=q.dtype), idx, 0.1)
