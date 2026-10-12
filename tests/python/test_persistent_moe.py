"""Persistent region arithmetic, replay, ownership and one-launch contract."""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
    _route_counts,
    _tile_map_static,
    glm_moe_grouped_gemm_native,
    quantize_activations_fp8,
)
from vkernels.torch_ops.persistent_moe import PersistentMoE

gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                         reason="CUDA required")


def inputs(tokens=17, hidden=256, inner=128, experts=4, top_k=3, skip=False):
    torch.manual_seed(51)
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    gu = torch.randn(experts, 2 * inner, hidden, device="cuda").to(torch.float8_e4m3fn)
    dn = torch.randn(experts, hidden, inner, device="cuda").to(torch.float8_e4m3fn)
    gs = torch.rand(experts, 2 * inner // 128, hidden // 128, device="cuda") * .1
    ds = torch.rand(experts, hidden // 128, inner // 128, device="cuda") * .1
    ids = torch.randint(experts, (tokens, top_k), device="cuda")
    weights = torch.rand(tokens, top_k, device="cuda")
    if skip:
        ids[:, 1::2] = -1
        weights[:, 1::3] = 0
    return x, gu, gs, dn, ds, ids, weights


def reference(args, skip, rounded):
    return glm_moe_grouped_gemm_native(*args, deterministic_combine=True,
                                      skip_zero_weights=skip,
                                      round_products_to_bf16=rounded,
                                      fused_prefill_glue=True)


def assert_intermediates(region, args, skip):
    x, gu, gs, dn, ds, ids, weights = args
    order, tok, counts, segments = _route_counts(ids, gu.shape[0], x.device, weights if skip else None)
    assert torch.equal(region.tokens, tok)
    assert torch.equal(region.inverse[order], torch.arange(order.numel(), device=x.device))
    tiles = _tile_map_static(counts, segments, region._constants["BM"], x.device, len(region.tile_count))
    assert all(torch.equal(a, b) for a, b in zip(
        (region.tile_expert, region.tile_start, region.tile_count), tiles))
    q, s = quantize_activations_fp8(x)
    assert torch.equal(region.q1.view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(region.s1, s)
    inner = gu.shape[1] // 2
    active = int(counts.sum())
    gate = region.gate_up[:active, :inner].float().clamp(max=7)
    up = region.gate_up[:active, inner:].float().clamp(min=-7, max=7)
    activation = (torch.nn.functional.silu(gate) * up).bfloat16()
    q, s = quantize_activations_fp8(activation)
    assert torch.equal(region.q2[:active].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(region.s2[:active], s)


@gpu
@pytest.mark.parametrize("skip,rounded", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("tokens,experts,top_k,workers", [(1, 1, 1, 1), (17, 7, 3, 4), (128, 72, 8, None)])
def test_complete_region_bitwise_and_metadata(skip, rounded, tokens, experts, top_k, workers):
    args = inputs(tokens=tokens, experts=experts, top_k=top_k, skip=skip)
    before = [v.clone() for v in args]
    region = PersistentMoE(*args, workers=workers, skip_zero_weights=skip,
                           round_products_to_bf16=rounded)
    expected = reference(args, skip, rounded)
    for _ in range(3):
        actual = region.run(args[0], args[-2], args[-1])
        assert_intermediates(region, args, skip)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(torch.equal(v.view(torch.uint8), old.view(torch.uint8)) for v, old in zip(args, before))
    assert int(region.base) == int(region.counter) == 3 * 4 * region.workers


@gpu
def test_graph_replay_changed_routes_values_and_all_dropped():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        args = inputs(skip=True)
        region = PersistentMoE(*args, skip_zero_weights=True, round_products_to_bf16=True, workers=4)
        region.run(args[0], args[-2], args[-1])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = region.run(args[0], args[-2], args[-1])
        for all_dropped in (False, True, False):
            args[0].normal_()
            args[-2].random_(0, args[1].shape[0])
            args[-1].uniform_()
            if all_dropped:
                args[-2].fill_(-1)
                args[-1].zero_()
            expected = reference(args, True, True)
            for _ in range(4):
                graph.replay()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert int(region.base) == int(region.counter)

@gpu
def test_run_issues_exactly_one_cuda_kernel(tmp_path):
    import json

    args = inputs()
    region = PersistentMoE(*args, workers=4)
    region.run(args[0], args[-2], args[-1])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        region.run(args[0], args[-2], args[-1])
        torch.cuda.synchronize()
    path = tmp_path / "trace.json"
    profile.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    assert len(kernels) == 1
    assert kernels[0]["name"].startswith("region")


@gpu
def test_preparation_bounds_and_stream_ownership():
    args = inputs()
    with pytest.raises(ValueError, match="workers"):
        PersistentMoE(*args, workers=torch.cuda.get_device_properties("cuda").multi_processor_count + 1)
    region = PersistentMoE(*args, workers=4)
    region.run(args[0], args[-2], args[-1])
    other = torch.cuda.Stream()
    with torch.cuda.stream(other), pytest.raises(RuntimeError, match="preparation stream"):
        region.run(args[0], args[-2], args[-1])
    with pytest.raises(OpNotEligible):
        region.run(args[0].float(), args[-2], args[-1])


def test_cpu_contract_rejected():
    x = torch.zeros(1, 128, dtype=torch.bfloat16)
    with pytest.raises(OpNotEligible):
        PersistentMoE(x, x, x, x, x, torch.zeros(1, 1, dtype=torch.int64), x.float())


@gpu
def test_two_blocks_per_sm_or_compiled_occupancy_rejection():
    args = inputs(tokens=128, hidden=512, inner=256, experts=72, top_k=8, skip=True)
    try:
        region = PersistentMoE(*args, skip_zero_weights=True, round_products_to_bf16=True,
                               blocks_per_sm=2)
    except ValueError as error:
        # A target with less than two resident CTAs must refuse before launch.
        assert "compiled occupancy" in str(error)
        return
    expected = reference(args, True, True)
    for _ in range(3):
        result = region.run(args[0], args[-2], args[-1])
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
    assert int(region.base) == int(region.counter) == 12 * region.workers


@gpu
def test_cooperative_region_with_independent_stream_work():
    args = inputs(tokens=128, hidden=512, inner=256, experts=72, top_k=8, skip=True)
    region = PersistentMoE(*args, skip_zero_weights=True, round_products_to_bf16=True)
    assert region.compiled.metadata.launch_cooperative_grid
    expected = reference(args, True, True)
    main = torch.cuda.current_stream()
    sibling = torch.cuda.Stream()
    sibling.wait_stream(main)
    with torch.cuda.stream(sibling):
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        for _ in range(8):
            b = a @ b
    result = region.run(args[0], args[-2], args[-1])
    main.wait_stream(sibling)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@gpu
def test_serialized_layers_share_workspace_with_runtime_weights():
    args = inputs(skip=True)
    region = PersistentMoE(*args, skip_zero_weights=True, round_products_to_bf16=True)
    alternate = list(args)
    alternate[2] = args[2] * 2
    alternate[4] = args[4] * .5
    for current in (args, alternate, args):
        expected = reference(current, True, True)
        actual = region.run(current[0], current[-2], current[-1], expert_weights=current[1:5])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(OpNotEligible, match="weight"):
        region.run(args[0], args[-2], args[-1], expert_weights=alternate[1:4])
