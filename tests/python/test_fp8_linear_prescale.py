"""Exact preservation through split projections, bias and changed graph inputs."""
import pytest
import torch

pytest.importorskip('triton')
from vkernels.torch_ops.fp8_linear import fp8_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None,
                                reason='NVIDIA GPU required')


@pytest.mark.parametrize('rows,outputs,inputs', [
    (1, 512, 4096), (2, 1536, 4096), (4, 4096, 1536), (8, 3072, 4096),
    (1, 4096, 3072), (4, 4096, 4096), (3, 1537, 1152), (2, 129, 640),
    (1, 1, 768), (4, 127, 896), (1, 4096, 512), (9, 129, 768),
])
@pytest.mark.parametrize('seed', [181, 182])
@pytest.mark.parametrize('bias_enabled', [False, True])
def test_prescale_preserves_projection_and_changed_graph(rows, outputs, inputs, seed, bias_enabled):
    torch.manual_seed(seed)
    # A strided input also exercises the common reshape/contiguous boundary.
    backing = torch.randn(rows, inputs * 2, device='cuda', dtype=torch.bfloat16)
    x = backing[:, ::2]
    w = torch.randn(outputs, inputs, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.rand((outputs + 127) // 128, inputs // 128, device='cuda') * .03
    bias = torch.randn(outputs, device='cuda', dtype=torch.bfloat16) if bias_enabled else None

    def compare():
        expected = fp8_linear(x, w, scales, bias)
        actual = fp8_linear(x, w, scales, bias, prescale_activation=True)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))

    compare()
    for _ in range(3):
        fp8_linear(x, w, scales, bias, prescale_activation=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fp8_linear(x, w, scales, bias, prescale_activation=True)
    for phase in range(3):
        if phase == 0:
            x.zero_()
        elif phase == 1:
            x.normal_(std=.5)
        else:
            w.copy_(torch.randn(outputs, inputs, device='cuda').to(w.dtype))
            scales.mul_(.875)
        graph.replay()
        expected = fp8_linear(x, w, scales, bias)
        torch.cuda.synchronize()
        assert torch.equal(captured.view(torch.uint8), expected.view(torch.uint8))
        compare()
