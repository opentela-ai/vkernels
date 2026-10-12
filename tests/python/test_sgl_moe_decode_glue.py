"""Decode launch fusion must preserve the incumbent W8A8 outputs exactly."""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.sgl_moe import sgl_fused_moe
from vkernels.torch_ops.sgl_moe_decode_glue import sgl_fused_moe_decode_glue


def test_cpu_declines():
    with pytest.raises(OpNotEligible):
        sgl_fused_moe_decode_glue(*(torch.zeros(1) for _ in range(7)))


def make_case(t, h=256, i=128, e=12):
    g = torch.Generator(device="cuda").manual_seed(417 + t)
    k = 8
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16, generator=g)
    gu = torch.randn(e, 2 * i, h, device="cuda", generator=g).to(torch.float8_e4m3fn)
    dn = torch.randn(e, h, i, device="cuda", generator=g).to(torch.float8_e4m3fn)
    gs = torch.rand(e, 2 * i // 128, h // 128, device="cuda", generator=g) * .05
    ds = torch.rand(e, h // 128, i // 128, device="cuda", generator=g) * .05
    ids = torch.randint(e, (t, k), device="cuda", generator=g)
    routes = torch.rand(t, k, device="cuda", generator=g)
    routes[:, ::2] = 0
    return x, gu, gs, dn, ds, ids, routes


cuda = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip,
                          reason="CUDA E4M3FN decode fusion")


@cuda
@pytest.mark.parametrize("t", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize("limit", [0.01, 7.0, float("inf")])
def test_bitwise_graph_replay_with_zero_and_revived_routes(t, skip, limit):
    values = make_case(t)
    kwargs = dict(swiglu_limit=limit, skip_zero_weights=skip)
    def oracle():
        return sgl_fused_moe(*values, route_weight_in_down=True, **kwargs)
    def fused():
        return sgl_fused_moe_decode_glue(*values, **kwargs)
    assert torch.equal(fused(), oracle())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused()
    torch.cuda.current_stream().wait_stream(stream)
    for state in ("zero", "revive", "change"):
        if state == "zero":
            values[-1].zero_()
        else:
            values[-1].fill_(0.125)
            values[-2].copy_(values[-2].roll(1, dims=1))
            values[0].mul_(-.75)
        graph.replay()
        assert torch.equal(actual, oracle()), state


@cuda
@pytest.mark.parametrize("t", [1, 4])
def test_serving_geometry_and_five_launches(t):
    values = make_case(t, h=4096, i=2048)
    kwargs = dict(swiglu_limit=7.0, skip_zero_weights=True)
    expected = sgl_fused_moe(*values, route_weight_in_down=True, **kwargs)
    assert torch.equal(sgl_fused_moe_decode_glue(*values, **kwargs), expected)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        actual = sgl_fused_moe_decode_glue(*values, **kwargs)
        torch.cuda.synchronize()
    kernels = [v for v in prof.events() if v.device_type == torch.autograd.DeviceType.CUDA]
    assert len(kernels) == 5, [v.name for v in kernels]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("change", [{}, {"tokens": 2}, {"tokens": 8},
                                   {"capability": (12, 1)}, {"device_name": "NVIDIA H100"},
                                   {"experts": 288}, {"skip": False}, {"tuned": False}])
def test_tuned_geometry_qualification_and_config_ownership(change):
    from vkernels.torch_ops.sgl_moe import _get_default_config
    from vkernels.torch_ops.sgl_moe_decode_glue import _gemm_configs

    config = _get_default_config(4)
    kwargs = dict(tuned=True, capability=(9, 0), device_name="NVIDIA GH200 120GB",
                  tokens=4, experts=72, hidden=4096, intermediate=2048, top_k=8, skip=True)
    gate, down = _gemm_configs(config, **(kwargs | change))
    assert config == _get_default_config(4)
    assert gate is not down and gate is not config and down is not config
    if change:
        assert gate == down == config
    else:
        assert gate == config | {"BLOCK_SIZE_N": 64, "GROUP_SIZE_M": 4}
        assert down == config | {"BLOCK_SIZE_N": 64, "num_stages": 4}
    config["num_stages"] = 2
    assert _gemm_configs(config, **kwargs) == (config, config)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip
                    or not torch.cuda.get_device_name().startswith("NVIDIA GH200"),
                    reason="GH200-only measured tile policy")
@pytest.mark.parametrize("t", [1, 4])
def test_qualified_tiles_bitwise_graph_replay(t, monkeypatch):
    import vkernels.torch_ops.sgl_moe as donor

    values = make_case(t, h=4096, i=2048, e=72)
    kwargs = dict(swiglu_limit=7., skip_zero_weights=True)
    expected = sgl_fused_moe(*values, route_weight_in_down=True, **kwargs)
    calls = []
    original = donor._invoke_fused_moe_kernel
    def invoke(*args, **options):
        calls.append(dict(options["config"] if "config" in options else args[-1]))
        return original(*args, **options)
    monkeypatch.setattr(donor, "_invoke_fused_moe_kernel", invoke)
    actual = sgl_fused_moe_decode_glue(*values, tuned_gemm=True, **kwargs)
    assert torch.equal(actual, expected)
    assert [v["BLOCK_SIZE_N"] for v in calls] == [64, 64]
    assert [v["num_stages"] for v in calls] == [3, 4]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = sgl_fused_moe_decode_glue(*values, tuned_gemm=True, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    for live in (False, True):
        values[-1].fill_(.125 if live else 0.)
        values[-2].copy_(values[-2].roll(1, dims=1))
        values[0].mul_(-.75)
        graph.replay()
        assert torch.equal(actual, sgl_fused_moe(*values, route_weight_in_down=True, **kwargs))
