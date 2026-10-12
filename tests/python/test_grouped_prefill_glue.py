"""Exact parity against the frozen eager metadata/quantization and full MoE."""

import pytest
import torch

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
    _tile_map_static,
    glm_moe_grouped_gemm_native,
    quantize_activations_fp8,
)
from vkernels.torch_ops.grouped_prefill_glue import grouped_tile_map, quantize_per_token_fp8_exact

gpu = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="CUDA required")


def assert_quantized_equal(x, group=128):
    before = x.clone()
    q, scale = quantize_per_token_fp8_exact(x, group)
    ref_q, ref_scale = quantize_activations_fp8(x, group)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scale, ref_scale)
    assert torch.equal(x, before)


@gpu
@pytest.mark.parametrize("experts,rows,cap", [(1, 1, 0), (1, 16, 19), (7, 4, 57), (72, 16, 137), (288, 64, 1311), (1024, 128, 4097)])
def test_tile_map_empty_experts_padding_strides(experts, rows, cap):
    torch.manual_seed(127)
    storage = torch.randint(0, 129, (experts, 2), device="cuda", dtype=torch.int64)
    counts = storage[:, 0]
    counts[::3] = 0
    segments = (counts.cumsum(0) - counts).repeat_interleave(2)[::2]
    before = counts.clone(), segments.clone()
    for c in (counts, torch.zeros_like(counts)):
        s = segments if c is counts else torch.zeros_like(segments)
        actual = grouped_tile_map(c, s, rows, cap)
        expected = _tile_map_static(c, s, rows, c.device, cap)
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert torch.equal(counts, before[0]) and torch.equal(segments, before[1])


@gpu
@pytest.mark.parametrize("rows,cols,group", [(0, 128, 128), (17, 2048, 128), (128, 4096, 128), (8, 1024, 32), (1, 1024, 1024)])
def test_quantizer_bytes_scales_and_strides(rows, cols, group):
    torch.manual_seed(118)
    x = torch.randn(rows, cols * 2, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert_quantized_equal(x, group)
    if rows:
        x.zero_()
        assert_quantized_equal(x, group)


@gpu
def test_quantizer_all_finite_bf16_values_and_scale_exponents():
    # Exercise every finite input bit pattern, including subnormals, signs,
    # signed zero, and values close to the largest finite BF16 number.
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    assert_quantized_equal(values.to("cuda").reshape(-1, 128))
    # Each positive finite BF16 value also becomes its own group's amax.
    maxima = values[values > 0].to("cuda")
    x = maxima[:, None] * torch.linspace(-1, 1, 128, device="cuda")[None, :]
    assert_quantized_equal(x.to(torch.bfloat16))


@gpu
def test_changed_inputs_cuda_replay():
    counts = torch.tensor([0, 12, 1, 0, 33, 4, 0], dtype=torch.int64, device="cuda")
    segments = counts.cumsum(0) - counts
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        grouped_tile_map(counts, segments, 16, 71)
        quantize_per_token_fp8_exact(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tiles = grouped_tile_map(counts, segments, 16, 71)
        q, scale = quantize_per_token_fp8_exact(x)
    counts.copy_(torch.tensor([16, 0, 0, 17, 0, 0, 8], device="cuda"))
    segments.copy_(counts.cumsum(0) - counts)
    x.normal_()
    graph.replay()
    expected = _tile_map_static(counts, segments, 16, x.device, 71)
    assert all(torch.equal(a, b) for a, b in zip(tiles, expected))
    ref_q, ref_scale = quantize_activations_fp8(x)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(scale, ref_scale)


@gpu
@pytest.mark.parametrize("skip", [False, True])
def test_full_grouped_moe_is_bitwise_unchanged(skip):
    torch.manual_seed(5)
    x = torch.randn(17, 256, device="cuda", dtype=torch.bfloat16)
    gu = torch.randn(4, 256, 256, device="cuda").to(torch.float8_e4m3fn)
    dn = torch.randn(4, 256, 128, device="cuda").to(torch.float8_e4m3fn)
    gs = torch.full((4, 2, 2), 0.03125, device="cuda")
    ds = torch.full((4, 2, 1), 0.03125, device="cuda")
    indices = torch.randint(0, 4, (17, 3), device="cuda")
    weights = torch.rand(17, 3, device="cuda")
    weights[:, 1] = 0
    if skip:
        indices[:, 1] = -1
    args = x, gu, gs, dn, ds, indices, weights
    kw = dict(deterministic_combine=True, skip_zero_weights=skip, round_products_to_bf16=skip)
    expected = glm_moe_grouped_gemm_native(*args, **kw)
    actual = glm_moe_grouped_gemm_native(*args, **kw, fused_prefill_glue=True)
    assert torch.equal(actual, expected)


def test_contract_misses():
    for x in (torch.zeros(3, 128), torch.zeros(3, 127, dtype=torch.bfloat16)):
        with pytest.raises(OpNotEligible):
            quantize_per_token_fp8_exact(x)
    with pytest.raises(OpNotEligible):
        grouped_tile_map(torch.zeros(7, dtype=torch.int64), torch.zeros(7, dtype=torch.int64), 16, 71)


@gpu
def test_inference_and_bounds():
    x = torch.zeros(3, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    with pytest.raises(OpNotEligible):
        quantize_per_token_fp8_exact(x)
    with pytest.raises(OpNotEligible):
        quantize_per_token_fp8_exact(x.detach(), 3)
    c = torch.zeros(1025, dtype=torch.int64, device="cuda")
    with pytest.raises(OpNotEligible):
        grouped_tile_map(c, c, 16, 71)
