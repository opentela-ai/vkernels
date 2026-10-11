"""Opt-in row independence preserves M=1 arithmetic across batch widths."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from vkernels.torch_ops.glm_gemv import dense_gemv
from vkernels.torch_ops.glm_dense_fp8_gemv import dense_gemv_fp8


@pytest.mark.gpu
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("batch,outputs,width", [(1, 128, 128), (3, 512, 384), (4, 512, 4096), (8, 128, 512)])
def test_rows_match_singletons_and_read_changed_graph_inputs(fp8, batch, outputs, width):
    if not torch.cuda.is_available() or (fp8 and torch.version.hip):
        pytest.skip("FP8 native conversion requires NVIDIA; BF16 requires GPU")
    gen = torch.Generator(device="cuda").manual_seed(427)
    x = torch.randn(batch, width, generator=gen, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(outputs, width, generator=gen, device="cuda")
    scales = torch.rand(outputs // 128, width // 128, generator=gen, device="cuda") * .05
    weight = weight.to(torch.float8_e4m3fn if fp8 else torch.bfloat16)

    def run(a, independent=False):
        if fp8:
            return dense_gemv_fp8(a, weight, scales, m_cap=batch, independent_rows=independent)
        return dense_gemv(a, weight, m_cap=batch, independent_rows=independent)

    def reference():
        return torch.cat([run(x[i:i + 1]) for i in range(batch)])

    saved = [v.view(torch.uint8).clone() for v in (x, weight, scales)]
    expected = reference()
    actual = run(x, True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert all(torch.equal(a.view(torch.uint8), b) for a, b in zip((x, weight, scales), saved))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = run(x, True)
    x.mul_(.75)
    weight.copy_((-weight.float()).to(weight.dtype))
    scales.mul_(1.25)
    graph.replay()
    assert not torch.equal(actual, replayed)
    assert torch.equal(replayed.view(torch.int16), reference().view(torch.int16))
