"""Selected-expert FP8 GEMV contract checks and optional GPU parity."""

import importlib
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run([sys.executable, "-c",
                    "import sys; import vkernels.torch_ops.glm_expert_gemv; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'triton' not in sys.modules"], check=True)


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def make_inputs(torch, e=32, o=256, i=256, t=1, k=8, seed=7, device="cpu"):
    torch.manual_seed(seed)
    raw = torch.randint(0, 256, (e, o, i), dtype=torch.uint8)
    raw.masked_fill_((raw & 127) == 127, 0)          # drop NaN encodings
    weights = raw.view(torch.float8_e4m3fn)
    scales = 0.0005 + torch.rand((e, o // 128, i // 128)) * 0.001
    indices = torch.rand((t, e)).topk(k, dim=-1).indices.to(torch.int64)
    return weights, scales, indices


@pytest.mark.parametrize("shape", [(t, k) for t in (1, 2) for k in (1, 8)])
def test_gpu_parity_vs_reference(torch, shape):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gemv import (
        expert_gemv, expert_gemv_reference)

    t, k = shape
    weights, scales, indices = make_inputs(torch, t=t, k=k)
    for broadcast in (True, False):
        weights = weights.to("cuda")
        scales = scales.to("cuda")
        indices = indices.to("cuda")
        x = torch.randn((t, 256 if broadcast else k, 256),
                        device="cuda", dtype=torch.bfloat16)
        if broadcast:
            x = x[:, 0, :]                    # x[T, I]
        actual = expert_gemv(x, weights, scales, indices)
        expected = expert_gemv_reference(x, weights, scales, indices)
        assert actual.shape == (t, k, 256)
        assert actual.dtype == torch.bfloat16
        diff = actual.float() - expected.float()
        rel = (diff.norm() / expected.float().norm().clamp_min(1e-12)).item()
        assert torch.isfinite(actual).all()
        assert rel < 3e-3, rel


def test_contract(torch):
    from vkernels.torch_ops.glm_expert_gemv import expert_gemv

    weights, scales, indices = make_inputs(torch, t=1, k=8)
    x = torch.randn(1, 8, 256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="GPU"):
        expert_gemv(x, weights, scales, indices)
    with pytest.raises(ValueError, match="T<=2"):
        expert_gemv(x[:3].expand(3, 8, 256), weights, scales,
                    torch.randint(0, 32, (3, 8), dtype=torch.int64))
    with pytest.raises(TypeError, match="BF16"):
        expert_gemv(x.float(), weights, scales, indices)
    with pytest.raises(ValueError, match="expected scales"):
        expert_gemv(x, weights, scales[:, :, :-1], indices)
    with pytest.raises(ValueError, match="x\\[T,I\\]"):
        expert_gemv(torch.randn(1, 128, dtype=torch.bfloat16),
                    weights, scales, indices)
