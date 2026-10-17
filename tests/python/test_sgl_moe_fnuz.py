"""Explicit gfx942 donor arithmetic, each linear boundary and changed replay."""
import pytest
import torch

from vkernels.torch_ops import sgl_moe
from vkernels.torch_ops._dispatch import OpNotEligible


def gfx942():
    return (torch.cuda.is_available() and torch.version.hip is not None
            and getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '').split(':')[0] == 'gfx942')


amd = pytest.mark.skipif(not gfx942(), reason='gfx942 component qualification')


def inputs(rows=4, experts=17, hidden=256, intermediate=128, topk=4, device='cpu'):
    generator = torch.Generator(device=device).manual_seed(1720)
    def random(shape):
        return torch.randn(shape, generator=generator, device=device)
    x = random((rows, hidden)).bfloat16()
    gu = (random((experts, 2 * intermediate, hidden)) * 4).to(torch.float8_e4m3fnuz)
    gs = torch.rand(experts, 2 * intermediate // 128, hidden // 128, generator=generator, device=device) * .04
    down = (random((experts, hidden, intermediate)) * 4).to(torch.float8_e4m3fnuz)
    ds = torch.rand(experts, hidden // 128, intermediate // 128, generator=generator, device=device) * .04
    ids = torch.stack([torch.randperm(experts, generator=generator, device=device)[:topk] for _ in range(rows)])
    # Include a zero route and a fused-shared-style final expert in every row.
    ids[:, -1] = experts - 1
    weights = random((rows, topk)).softmax(-1)
    weights[:, 0] = 0
    return [x, gu, gs, down, ds, ids, weights]


def test_cpu_fnuz_contract_declines():
    args = inputs()
    assert not sgl_moe.sgl_fused_moe_eligible(*args, storage='e4m3fnuz')
    with pytest.raises(OpNotEligible):
        sgl_moe.sgl_fused_moe(*args, storage='e4m3fnuz')
    assert not sgl_moe.sgl_fused_moe_eligible(*args)
    assert not sgl_moe.sgl_fused_moe_eligible(*args, storage='unknown')


def test_hip_never_uses_sm90_transposed_tile(monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'test')
    sgl_moe._should_enable_swap_ab.cache_clear()
    try:
        assert not sgl_moe._should_enable_swap_ab(16, 128)
    finally:
        sgl_moe._should_enable_swap_ab.cache_clear()


@amd
@pytest.mark.parametrize('rows,experts,hidden,intermediate,topk', [
    (1, 17, 256, 128, 4), (4, 17, 256, 128, 4), (16, 17, 256, 128, 9),
    (1, 289, 4096, 512, 9), (4, 289, 4096, 512, 9), (16, 289, 4096, 512, 9),
])
@pytest.mark.parametrize('weighted', [False, True])
def test_each_dot_matches_float64_quantized_oracle(monkeypatch, rows, experts, hidden, intermediate, topk, weighted):
    args = inputs(rows, experts, hidden, intermediate, topk, device='cuda')
    assert sgl_moe.sgl_fused_moe_eligible(*args, storage='e4m3fnuz')
    assert not sgl_moe.sgl_fused_moe_eligible(*args)  # Explicit storage is required.
    original = sgl_moe._invoke_fused_moe_kernel
    observed = []

    def check(a, b, c, sa, sb, route_w, route_ids, sorted_ids, experts_ids, padded, **kwargs):
        original(a, b, c, sa, sb, route_w, route_ids, sorted_ids, experts_ids, padded, **kwargs)
        aq = a.cpu().double() * sa.cpu().double().repeat_interleave(128, -1)
        ids = route_ids.cpu().reshape(-1)
        # Only selected matrices are copied to the CPU oracle.
        selected = b.view(torch.uint8).index_select(0, ids).view(torch.float8_e4m3fnuz).cpu().double()
        scales = sb.index_select(0, ids).cpu().double().repeat_interleave(128, -2).repeat_interleave(128, -1)
        weights = selected * scales
        positions = torch.arange(ids.numel()) // kwargs['top_k']
        expected = torch.bmm(weights, aq[positions, :, None]).squeeze(-1)
        if kwargs['mul_routed_weight']:
            expected *= route_w.cpu().double().reshape(-1, 1)
        # Same FP64 dot gate used by the established dense projection.
        torch.testing.assert_close(c.reshape_as(expected).cpu(), expected.bfloat16(), rtol=.0078125, atol=.00390625)
        observed.append(c.clone())

    monkeypatch.setattr(sgl_moe, '_invoke_fused_moe_kernel', check)
    actual = sgl_moe.sgl_fused_moe(*args, storage='e4m3fnuz', swiglu_limit=10, route_weight_in_down=weighted)
    assert len(observed) == 2
    if weighted:
        assert torch.equal(actual, observed[-1].sum(dim=1))
    assert torch.isfinite(actual).all()


@amd
@pytest.mark.parametrize('rows', [1, 4, 16])
def test_replay_changes_inputs_routes_and_weighted_outputs(rows):
    args = inputs(rows=rows, topk=9, device='cuda')
    def run():
        return sgl_moe.sgl_fused_moe(*args, storage='e4m3fnuz', swiglu_limit=10, route_weight_in_down=True)
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for zero in (True, False):
        args[0].normal_()
        args[-2].copy_((args[-2] + 3) % args[1].shape[0])
        if zero:
            args[-1].zero_()
        else:
            args[-1].uniform_(0, 1)
        captured.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, run(), rtol=0, atol=0)
        if zero:
            assert torch.count_nonzero(captured) == 0
