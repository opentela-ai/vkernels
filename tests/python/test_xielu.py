"""Activation parity at every eager low-precision rounding boundary."""
import pytest

torch = pytest.importorskip("torch")
from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.xielu import xielu

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def eager(x, ap, an, beta, eps):
    ap = torch.nn.functional.softplus(ap)
    an = beta + torch.nn.functional.softplus(an)
    return torch.where(x > 0, ap * x * x + beta * x,
                       (torch.expm1(torch.minimum(x, eps)) - x) * an + beta * x)


def test_cpu_requires_fallback():
    with pytest.raises(OpNotEligible, match="share a GPU"):
        xielu(*[torch.ones(1) for _ in range(5)])


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1, 16384), (4, 1, 28672), (2, 17, 777), (2, 129, 8192)])
def test_matches_eager_and_keeps_inputs(shape, dtype):
    torch.manual_seed(71)
    x = torch.randn(shape, device="cuda", dtype=dtype) * 7
    before = x.clone()
    ap, an = [torch.tensor([v], device="cuda", dtype=dtype) for v in (-0.3, -1.1)]
    beta, eps = [torch.tensor(v, device="cuda", dtype=dtype) for v in (0.5, -1e-6)]
    with torch.inference_mode():
        torch.testing.assert_close(xielu(x, ap, an, beta, eps), eager(x, ap, an, beta, eps), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_low_precision_inputs(dtype):
    x = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16).view(dtype)
    beta, eps = [torch.tensor(v, device="cuda", dtype=dtype) for v in (0.5, -1e-6)]
    with torch.inference_mode():
        for p, n in [(-0.3, -1.1), (-21.0, 21.0), (5.2, -4.8)]:
            ap, an = [torch.tensor([v], device="cuda", dtype=dtype) for v in (p, n)]
            torch.testing.assert_close(xielu(x, ap, an, beta, eps), eager(x, ap, an, beta, eps), rtol=0, atol=0, equal_nan=True)


@gpu
def test_graph_replay_reads_updated_parameters():
    x = torch.linspace(-5, 5, 4096, device="cuda", dtype=torch.bfloat16)
    ap, an, beta, eps = [torch.tensor([v], device="cuda", dtype=x.dtype) for v in (-0.3, -1.1, 0.5, -1e-6)]
    with torch.inference_mode():
        xielu(x, ap, an, beta, eps)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            xielu(x, ap, an, beta, eps)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = xielu(x, ap, an, beta, eps)
        torch.cuda.current_stream().wait_stream(stream)
        ap.fill_(0.8)
        an.fill_(-0.5)
        graph.replay()
        torch.testing.assert_close(output, eager(x, ap, an, beta, eps), rtol=0, atol=0)
