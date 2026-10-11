"""Explicit prefill eligibility, quantization parity and replay of the SM90 backend."""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.deepgemm_linear import deepgemm_linear, quantize_tma  # noqa: E402


def test_cpu_rejected_before_optional_backend_import():
    with pytest.raises(ValueError, match="SM90 CUDA"):
        deepgemm_linear(torch.zeros(128, 128, dtype=torch.bfloat16),
                        torch.zeros(128, 128, dtype=torch.float8_e4m3fn),
                        torch.ones(1, 1), allow_prefill=True)


_SM90 = (torch.cuda.is_available() and torch.version.hip is None
         and torch.cuda.get_device_capability()[0] == 9)


@pytest.mark.skipif(not _SM90, reason="SM90 DeepGEMM backend required")
@pytest.mark.parametrize("rows", [128, 384, 512])
def test_prefill_projection_quantization_and_replay(rows):
    from vkernels.torch_ops.sgl_moe import per_token_group_quant_fp8

    torch.manual_seed(915)
    x = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 256, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(2, 2, device="cuda") * .03
    qa, sa = per_token_group_quant_fp8(x, 128)
    qb, sb = quantize_tma(x)
    assert torch.equal(qa.view(torch.uint8), qb.view(torch.uint8))
    assert torch.equal(sa, sb)
    with pytest.raises(ValueError, match="explicit prefill"):
        deepgemm_linear(x, weight, scales)
    actual = deepgemm_linear(x, weight, scales, allow_prefill=True)
    expected = ((qa.double() * sa.double().repeat_interleave(128, 1)) @
                (weight.double() * scales.double().repeat_interleave(128, 0)
                 .repeat_interleave(128, 1)).T).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=.0078125, atol=.00390625)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(deepgemm_linear(x, weight, scales, bias, allow_prefill=True), actual + bias)
    for _ in range(3):
        deepgemm_linear(x, weight, scales, allow_prefill=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = deepgemm_linear(x, weight, scales, allow_prefill=True)
    for zero in (True, False):
        x.zero_() if zero else x.normal_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, deepgemm_linear(x, weight, scales, allow_prefill=True))
