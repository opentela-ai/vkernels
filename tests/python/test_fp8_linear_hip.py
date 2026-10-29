"""CDNA3 dense projections preserve the explicit FN quantization contract."""
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl

from vkernels.torch_ops.fp8_linear import _fn_weight_as_fnuz, fp8_linear
from vkernels.torch_ops.fp8_linear_hip import quantize_fn_as_fnuz


def _gfx942():
    return (torch.cuda.is_available() and torch.version.hip is not None
            and getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0] == "gfx942")


amd = pytest.mark.skipif(not _gfx942(), reason="gfx942 numerical and graph gate")


@triton.jit
def _convert_bytes(X, Y):
    offsets = tl.arange(0, 256)
    tl.store(Y + offsets, _fn_weight_as_fnuz(tl.load(X + offsets)))


@amd
def test_all_weight_bytes_match_independent_conversion():
    raw = torch.arange(256, dtype=torch.uint8)
    expected = (raw.view(torch.float8_e4m3fn).float() * .5).to(torch.float8_e4m3fnuz)
    actual = torch.empty(256, device="cuda", dtype=torch.float8_e4m3fnuz)
    _convert_bytes[(1,)](raw.cuda(), actual, num_warps=1)
    assert torch.equal(actual.view(torch.uint8).cpu(), expected.view(torch.uint8))


@triton.jit
def _donor_normalized(X, Y, S, eps, fp8_min, fp8_max, BLOCK: tl.constexpr):
    # Independent original FN donor math, before its unsupported HIP FN cast.
    # Store FP32 normalized values; CPU performs the reference FN rounding.
    group = tl.program_id(0)
    offsets = group * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), eps) / fp8_max
    inverse = 1.0 / scale
    tl.store(Y + offsets, tl.clamp(x * inverse, fp8_min, fp8_max))
    tl.store(S + group, scale)


def _donor_quantize(x):
    x = x.cuda()
    normalized = torch.empty_like(x, dtype=torch.float32)
    scales = torch.empty(x.shape[0], x.shape[1] // 128, device=x.device)
    _donor_normalized[(x.numel() // 128,)](x, normalized, scales,
                                          1e-10, -448.0, 448.0, BLOCK=128,
                                          num_warps=1, num_stages=1)
    return normalized.cpu().to(torch.float8_e4m3fn), scales.cpu()


@amd
def test_quantizer_zero_tiny_and_rounding_groups():
    generator = torch.Generator().manual_seed(1847)
    x = torch.randn(8, 1024, generator=generator).bfloat16()
    x[0].zero_()
    for row, magnitude in enumerate((1e-20, 1e-12, 1e-10, 1e-7, 1e-4), 1):
        x[row] *= magnitude
    # Values at and beside FP8 halfway points, with a fixed unit scale.
    x[6, :8] = torch.tensor([448, -448, 1.0625, 1.1875, -1.0625, -1.1875, 0, -0.0])
    expected, scales = _donor_quantize(x)
    actual, doubled = quantize_fn_as_fnuz(x.cuda())
    expected_bytes = (expected.float() * .5).to(torch.float8_e4m3fnuz).view(torch.uint8)
    assert torch.equal(actual.view(torch.uint8).cpu(), expected_bytes)
    torch.testing.assert_close(doubled.cpu(), scales * 2, rtol=0, atol=0)


@amd
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 37, 128])
@pytest.mark.parametrize("outputs,inputs", [(512, 4096), (1536, 4096), (3072, 4096),
                                            (4096, 1536), (4096, 3072), (4096, 4096),
                                            (129, 256)])
def test_projection_matches_float64_quantized_oracle(rows, outputs, inputs):
    generator = torch.Generator().manual_seed(1848)
    x = torch.randn(rows, inputs, generator=generator).bfloat16()
    x[0, :128] = 0
    weight = torch.randn(outputs, inputs, generator=generator).to(torch.float8_e4m3fn)
    scales = torch.rand((outputs + 127) // 128, inputs // 128, generator=generator) * .03
    q, sa = _donor_quantize(x)
    a = q.double() * sa.double().repeat_interleave(128, -1)
    w = weight.double() * scales.double().repeat_interleave(128, 0)[:outputs].repeat_interleave(128, 1)
    expected = (a @ w.T).bfloat16()
    x, weight, scales = x.cuda(), weight.cuda(), scales.cuda()
    actual = fp8_linear(x, weight, scales)
    assert actual.shape == (rows, outputs) and actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    # Identical to the incumbent NVIDIA projection's FP64 oracle gate.
    torch.testing.assert_close(actual.cpu(), expected, rtol=.0078125, atol=.00390625)
    bias = torch.randn(outputs, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(fp8_linear(x, weight, scales, bias), actual + bias, rtol=0, atol=0)


@amd
@pytest.mark.parametrize("rows", [1, 4, 16])
def test_graph_replay_updates_inputs_weights_and_scales(rows):
    torch.manual_seed(1849)
    x = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(512, 1536, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(4, 12, device="cuda") * .03
    for _ in range(3):
        fp8_linear(x, weight, scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8_linear(x, weight, scales)
    for zero in (True, False):
        if zero:
            x.zero_()
        else:
            x.normal_()
            weight.copy_(torch.randn(weight.shape, device="cuda").to(weight.dtype))
            scales.uniform_(.01, .03)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, fp8_linear(x, weight, scales), rtol=0, atol=0)
        if zero:
            assert torch.count_nonzero(captured) == 0
