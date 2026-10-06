"""Strict serving-rounding checks for the three-launch decode backend."""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.glm_moe_decode import moe_decode, moe_decode_eligible  # noqa: E402


def case(t, width=256, ia=128):
    gen = torch.Generator(device="cuda").manual_seed(123 + t)
    e, k = 16, 8
    gu = (torch.randn(e, 2 * ia, width, generator=gen, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    dw = (torch.randn(e, width, ia, generator=gen, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    gs = torch.exp(torch.randn(e, 2 * ia // 128, width // 128, generator=gen, device="cuda")) * 0.02
    ds = torch.exp(torch.randn(e, width // 128, ia // 128, generator=gen, device="cuda")) * 0.02
    idx = torch.randint(e, (t, k), generator=gen, device="cuda")
    routes = torch.randn(t, k, generator=gen, device="cuda")
    routes[:, 0] = 0  # Zero routes and repeated experts remain valid assignments.
    x = torch.randn(t, width, generator=gen, device="cuda", dtype=torch.bfloat16)
    return x, gu, gs, dw, ds, idx, routes


def oracle(values, limit):
    from vkernels.torch_ops.elementwise import swiglu_limit
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv

    x, gu, gs, dw, ds, idx, routes = values
    projected = expert_gemv(x, gu, gs, idx, t_cap=8)
    gate, up = projected.chunk(2, dim=-1)
    act = swiglu_limit(gate, up, limit)
    out = expert_gemv(act, dw, ds, idx, t_cap=8)
    return (out * routes.to(out.dtype).unsqueeze(-1)).sum(dim=1)


def test_cpu_declines_without_importing_triton():
    x = torch.zeros(2, 256, dtype=torch.bfloat16)
    gu = torch.zeros(8, 256, 256, dtype=torch.float8_e4m3fn)
    dw = torch.zeros(8, 256, 128, dtype=torch.float8_e4m3fn)
    values = x, gu, torch.ones(8, 2, 2), dw, torch.ones(8, 2, 1), torch.zeros(2, 8, dtype=torch.int64), torch.ones(2, 8)
    assert not moe_decode_eligible(*values)
    with pytest.raises(OpNotEligible):
        moe_decode(*values, 7.0)
    assert not moe_decode_eligible(*values[:-1], torch.ones(2, 7))
    assert not moe_decode_eligible(values[0].flatten(), *values[1:])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="CUDA E4M3FN kernel")
@pytest.mark.parametrize("t", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("limit", [0.01, 7.0, float("inf")])
def test_exact_rounding_and_changed_input_graph(t, limit):
    values = case(t)
    want = oracle(values, limit)
    got = moe_decode(*values, limit)
    assert torch.equal(got, want)
    snapshots = [v.clone() for v in values]
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        captured = moe_decode(*values, limit)
    torch.cuda.current_stream().wait_stream(stream)
    assert all(torch.equal(a, b) for a, b in zip(values, snapshots))
    values[0].mul_(-0.75)
    values[-1].copy_(values[-1].roll(1, dims=1))
    values[-2].copy_(values[-2].roll(1, dims=1))
    graph.replay()
    assert torch.equal(captured, oracle(values, limit))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="CUDA E4M3FN kernel")
@pytest.mark.parametrize("gate_rows,gate_warps,down_rows", [(1, 4, 4), (2, 4, 16), (4, 4, 16), (8, 4, 32)])
def test_serving_width_geometry(gate_rows, gate_warps, down_rows):
    values = case(4, width=4096, ia=512)
    got = moe_decode(*values, 7.0, gate_rows=gate_rows, gate_warps=gate_warps, down_rows=down_rows)
    assert torch.equal(got, oracle(values, 7.0))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="CUDA E4M3FN kernel")
def test_changed_reduction_order_declines():
    values = case(4)
    with pytest.raises(OpNotEligible, match="four-warp"):
        moe_decode(*values, 7.0, gate_warps=8)
