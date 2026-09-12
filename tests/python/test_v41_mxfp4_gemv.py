"""MXFP4 expert decode GEMV: reference oracle + optional GPU parity."""

import importlib.util
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.v41_mxfp4_gemv as m; "
            "assert 'torch' not in sys.modules; assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_concrete_ones(torch):
    from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv_reference

    E, O, I, group = 2, 4, 8, 4
    # byte 0x22 -> nibbles (2, 2) -> E2M1 value 1.0 both; scale 1 -> weight all ones
    weights = torch.full((E, O, I // 2), 0x22, dtype=torch.uint8)
    scales = torch.ones(E, O, I // group)
    x = torch.ones(1, I, dtype=torch.bfloat16)  # T=1
    indices = torch.tensor([[0, 1]], dtype=torch.int64)  # K=2
    out = mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group)
    assert out.shape == (1, 2, O)
    assert torch.allclose(out.float(), torch.full((1, 2, O), float(I)))


def test_reference_matches_dequant_then_matmul(torch):
    from vkernels.torch_ops.v41_mxfp4_dequant import mxfp4_dequant_reference
    from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv_reference

    torch.manual_seed(0)
    E, O, I, group = 4, 8, 32, 32
    weights = torch.randint(0, 256, (E, O, I // 2), dtype=torch.uint8)
    scales = torch.rand(E, O, I // group) + 0.5
    x = torch.randn(2, 2, I, dtype=torch.bfloat16)  # explicit [T,K,I]
    indices = torch.tensor([[0, 3], [1, 2]], dtype=torch.int64)

    out = mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group)
    W = mxfp4_dequant_reference(weights, scales, group=group, dtype=torch.bfloat16)  # [E,O,I]
    sel = W[indices]  # [T,K,O,I]
    manual = torch.einsum("tki,tkoi->tko", x.float(), sel.float()).to(torch.bfloat16)
    assert torch.equal(out, manual)


def test_broadcast_x_matches_explicit(torch):
    from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv_reference

    torch.manual_seed(1)
    E, O, I, group = 3, 8, 16, 4
    weights = torch.randint(0, 256, (E, O, I // 2), dtype=torch.uint8)
    scales = torch.rand(E, O, I // group) + 0.5
    x = torch.randn(2, I, dtype=torch.bfloat16)  # [T,I] broadcast over K
    indices = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64)
    bcast = mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group)
    expl = mxfp4_expert_gemv_reference(x[:, None, :].expand(2, 2, I).contiguous(), weights, scales, indices, group=group)
    assert torch.equal(bcast, expl)


def test_wrapper_falls_back_on_cpu(torch):
    from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv, mxfp4_expert_gemv_reference

    torch.manual_seed(4)
    E, O, I, group = 3, 8, 16, 4
    weights = torch.randint(0, 256, (E, O, I // 2), dtype=torch.uint8)
    scales = torch.rand(E, O, I // group) + 0.5
    x = torch.randn(2, I, dtype=torch.bfloat16)
    indices = torch.tensor([[1, 2], [0, 2]], dtype=torch.int64)
    got = mxfp4_expert_gemv(x, weights, scales, indices, group=group)  # CPU -> reference
    assert torch.equal(got, mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group))


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_matches_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv, mxfp4_expert_gemv_reference

    torch.manual_seed(2)
    E, O, I, group = 6, 16, 128, 32
    weights = torch.randint(0, 256, (E, O, I // 2), dtype=torch.uint8)
    scales = torch.rand(E, O, I // group) + 0.5
    x = torch.randn(2, I, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 5], [3, 1]], dtype=torch.int64)
    ref = mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group)
    got = mxfp4_expert_gemv(x.cuda(), weights.cuda(), scales.cuda(), indices.cuda(), group=group).cpu()
    # Triton and torch einsum reduce in different fp32 orders: bf16 rounding
    # usually absorbs the last-ulp difference, but bitwise equality is not
    # guaranteed (unlike the elementwise dequant above) — use a tolerance
    # loose enough to absorb the ulp noise, tight enough for any real error.
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton required")
def test_gpu_uint8_e8m0_scales(torch):
    """Regression: the device path must decode E8M0 codes like the reference
    (raw uint8 codes are a documented input; a bare .float() would scale by
    the code value instead of 2**(code-127))."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.v41_mxfp4_gemv import mxfp4_expert_gemv, mxfp4_expert_gemv_reference

    torch.manual_seed(3)
    E, O, I, group = 4, 16, 64, 32
    weights = torch.randint(0, 256, (E, O, I // 2), dtype=torch.uint8)
    # raw E8M0 codes in 120..129 -> scales 2**-7 .. 2**2
    scales = torch.randint(120, 130, (E, O, I // group), dtype=torch.uint8)
    x = torch.randn(2, I, dtype=torch.bfloat16)
    indices = torch.tensor([[0, 3], [2, 1]], dtype=torch.int64)
    ref = mxfp4_expert_gemv_reference(x, weights, scales, indices, group=group)
    got = mxfp4_expert_gemv(x.cuda(), weights.cuda(), scales.cuda(), indices.cuda(), group=group).cpu()
    torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)
