"""GLM projection contract checks and optional GPU parity/capture checks."""

import importlib
import subprocess
import sys

import pytest

# The GLM shapes from issues #66/#68: (N, K) with M in {1, 2}.
GLM_SHAPES = [(64, 4096), (128, 4096), (512, 4096), (1536, 4096)]


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import vkernels.torch_ops.glm_projection; "
                    "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules"], check=True)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


@pytest.mark.parametrize("shape", [(4096,), (1, 4096), (1, 2, 4096)])
def test_reference_shape_dtype(torch, shape):
    from vkernels.torch_ops.glm_projection import glm_projection_reference

    x = torch.ones(shape, dtype=torch.bfloat16)
    for n in (64, 128):
        w = torch.full((n, 4096), 1 / 4096, dtype=torch.bfloat16)
        actual = glm_projection_reference(x, w)
        assert actual.shape == (*shape[:-1], n)
        assert actual.dtype == torch.bfloat16
        assert actual.is_contiguous()
        assert torch.allclose(actual.float(), torch.ones_like(actual.float()))


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("n,k", GLM_SHAPES)
def test_gpu_parity_graph_and_readonly(torch, tokens, n, k):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_projection import glm_projection, glm_projection_reference

    torch.manual_seed(66 + n)
    x = torch.randn((1, tokens, k), device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 64
    before = [t.clone() for t in (x, w)]
    eager = glm_projection(x, w)
    torch.testing.assert_close(eager, glm_projection_reference(x, w),
                               rtol=0.008, atol=0.008)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = glm_projection(x, w)
    graph.replay()
    torch.testing.assert_close(captured, eager, rtol=0, atol=0)
    for original, t in zip(before, (x, w)):
        assert torch.equal(original, t)
    x.copy_(before[0] * 0.5)
    graph.replay()
    torch.testing.assert_close(captured, glm_projection_reference(x, w),
                               rtol=0.008, atol=0.008)


def test_gpu_zero_and_known_values(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_projection import glm_projection

    x = torch.zeros((2, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((128, 4096), device="cuda", dtype=torch.bfloat16)
    assert torch.count_nonzero(glm_projection(x, w)).item() == 0
    x.fill_(1.0)
    w[:, 1::2] = -1
    w[:, -1] = 0.5
    # 2048 even cols at +1, 2047 odd cols at -1 (the last overridden): 1.5.
    result = glm_projection(x, w)
    torch.testing.assert_close(result, torch.full_like(result, 1.5), rtol=0, atol=0)


def test_contract(torch):
    from vkernels.torch_ops.glm_projection import glm_projection, glm_projection_reference

    x = torch.ones((1, 4096), dtype=torch.bfloat16)
    w = torch.ones((64, 4096), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GPU"):
        glm_projection(x, w)
    with pytest.raises(TypeError, match="BF16"):
        glm_projection_reference(x.float(), w)
    with pytest.raises(ValueError, match="one or two"):
        glm_projection_reference(x.expand(3, -1), w)
    with pytest.raises(ValueError, match="contiguous"):
        glm_projection_reference(x, w.T.contiguous().T)
    with pytest.raises(ValueError, match="4096"):
        glm_projection_reference(torch.ones((1, 2048), dtype=torch.bfloat16),
                                 torch.ones((64, 2048), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="matching"):
        glm_projection_reference(x, torch.ones((64, 8192), dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="share a device"):
        glm_projection_reference(x, w.to("meta"))


def test_gpu_cold_capture_guard(torch, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    module = importlib.import_module("vkernels.torch_ops.glm_projection")
    x = torch.ones((1, 4096), device="cuda", dtype=torch.bfloat16)
    w = torch.ones((64, 4096), device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr(module, "_WARMED", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="warm up.*eagerly"):
        module.glm_projection(x, w)
    assert module._WARMED == set()


def test_gpu_device_guard(torch):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_projection import glm_projection

    original = torch.cuda.current_device()
    x = torch.ones((1, 4096), device="cuda:1", dtype=torch.bfloat16)
    w = torch.full((64, 4096), 1 / 4096, device="cuda:1", dtype=torch.bfloat16)
    with torch.cuda.device(0):
        result = glm_projection(x, w)
        assert result.device == torch.device("cuda:1")
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(result, torch.ones_like(result), rtol=0, atol=0)
    assert torch.cuda.current_device() == original
