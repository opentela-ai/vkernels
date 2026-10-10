"""Host guards for the optional TP4 push plan; TP4 replay lives in campaign screen."""
import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.sgl_push import SglPushPlan, eligible


def test_cpu_declines_without_loading_optional_ffi():
    assert not eligible(torch.ones(4096, dtype=torch.bfloat16), 4)


def test_cold_plan_rejects_capture_before_compilation_or_rendezvous(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="rendezvous must be warmed"):
        SglPushPlan(0, 4, torch.device("cuda", 0), None)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_alignment_size_and_shape_fallback_guards(monkeypatch, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA tensor metadata checks need a GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    x = torch.empty(4096, device="cuda", dtype=dtype)
    assert eligible(x, 4)
    assert not eligible(x, 2)
    assert not eligible(x[:0], 4)
    assert not eligible(x[1:], 4)
    assert not eligible(x[:-1], 4)
    assert not eligible(x[::2], 4)
    assert not eligible(torch.empty(65536, device="cuda", dtype=dtype), 4)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 1))
    assert not eligible(x, 4)
