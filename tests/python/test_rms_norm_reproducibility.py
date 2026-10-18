"""Operator-cache choices must not change a normalization's reduction order."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def same_bits(a, b):
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def fixed_layout_control(elementwise, x, module, residual, rounded):
    """Unchanged kernel with the original cache-off four-warp launch.

    This control bypasses the selection code being tested. Numerical parity
    with the eager formula remains covered by the existing elementwise tests;
    this regression requires exact preservation of the original kernel bits.
    """
    import triton

    output = torch.empty_like(x)
    summed = x if residual is None else torch.empty_like(x)
    elementwise._kernels().norm[(x.numel() // x.shape[-1],)](
        x, module.weight, x if residual is None else residual, output, summed,
        x.shape[-1], module.variance_epsilon, residual is not None,
        triton.next_power_of_2(x.shape[-1]), ROUND_INTERMEDIATES=rounded,
        num_warps=4, enable_fp_fusion=False,
    )
    return output, summed


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("rounded", [False, True])
def test_cache_independence_and_changed_input_graph_replay(monkeypatch, tmp_path, dtype, residual, rounded):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    pytest.importorskip("triton")
    from vkernels.torch_ops import elementwise
    from vkernels.tuning import cache

    generator = torch.Generator().manual_seed(163)
    x = torch.randn(1024, 2048, generator=generator).to(dtype).cuda()
    r = torch.randn(x.shape, generator=generator).to(dtype).cuda() if residual else None
    module = SimpleNamespace(weight=torch.randn(2048, generator=generator).to(dtype).cuda(), variance_epsilon=1e-6)
    original = [t.clone() for t in (x, module.weight, r) if t is not None]
    monkeypatch.setenv("VKERNELS_CACHE", "off")
    monkeypatch.setattr(elementwise, "_NORM_CFG", {})
    baseline = fixed_layout_control(elementwise, x, module, r, rounded)
    cache_off = elementwise.rms_norm(x, module, r, round_intermediates=rounded)
    assert all(same_bits(a, b) for a, b in zip(baseline, cache_off, strict=True))
    for proposed in (2, 8):
        monkeypatch.setenv("VKERNELS_CACHE", str(tmp_path))
        monkeypatch.setattr(cache, "op_config", lambda *a, **k: ({"num_warps": proposed}, "tuned"))
        elementwise._NORM_CFG.clear()
        actual = elementwise.rms_norm(x, module, r, round_intermediates=rounded)
        assert all(same_bits(a, b) for a, b in zip(baseline, actual, strict=True))
        assert all(same_bits(a, b) for a, b in zip(original, (t for t in (x, module.weight, r) if t is not None), strict=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = elementwise.rms_norm(x, module, r, round_intermediates=rounded)
    x.copy_(x.roll(1, 0) * .75)
    expected = elementwise.rms_norm(x, module, r, round_intermediates=rounded)
    graph.replay()
    assert all(same_bits(a, b) for a, b in zip(expected, captured, strict=True))
