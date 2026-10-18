"""FP32 fused intermediates are explicit; default HF storage roundings persist."""
from types import SimpleNamespace

import pytest
import torch

from vkernels.torch_ops import elementwise as ops


def norm(width, device, dtype):
    return SimpleNamespace(weight=(1 + torch.randn(width, device=device) * .2).to(dtype),
                           variance_epsilon=1e-6)


@pytest.mark.parametrize("residual", [False, True])
def test_reference_fused_norm_keeps_fp32_sum_and_weight_product(residual):
    torch.manual_seed(71)
    x = torch.randn(17, 128).bfloat16()
    r = torch.randn_like(x) if residual else None
    module = norm(128, "cpu", torch.bfloat16)
    summed = x.float() if r is None else x.float() + r.float()
    expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + module.variance_epsilon)
                * module.weight.float()).to(x.dtype)
    actual, stored = ops.rms_norm_reference(x, module, r, round_intermediates=False)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(stored, summed.to(x.dtype), atol=0, rtol=0)
    hf, _ = ops.rms_norm_reference(x, module, r)
    assert torch.count_nonzero(hf != actual) > 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("residual", [False, True])
@pytest.mark.parametrize("width", [64, 128, 2048])
def test_gpu_fused_norm_matches_explicit_fp32_reference(dtype, residual, width):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(71)
    x = torch.randn(17, width, device="cuda", dtype=dtype)
    r = torch.randn_like(x) if residual else None
    module = norm(width, "cuda", dtype)
    saved = x.clone(), None if r is None else r.clone()
    expected, expected_sum = ops.rms_norm_reference(x, module, r, round_intermediates=False)
    actual, stored = ops.rms_norm(x, module, r, round_intermediates=False)
    torch.testing.assert_close(actual, expected, atol=.008 if dtype == torch.bfloat16 else .001, rtol=.008)
    torch.testing.assert_close(stored, expected_sum, atol=0, rtol=0)
    assert torch.equal(x, saved[0])
    if r is not None:
        assert torch.equal(r, saved[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_rope_preserves_norm_output_cast_and_fp32_rotary_products(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(19)
    q = torch.randn(17, 4, 64, device="cuda", dtype=dtype)
    k = torch.randn(17, 2, 64, device="cuda", dtype=dtype)
    qn, kn = norm(64, "cuda", dtype), norm(64, "cuda", dtype)
    angles = torch.randn(17, 64, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    actual = ops.qk_norm_rope(q, k, qn, kn, cos, sin, round_intermediates=False)
    nq, nk = ops.qk_norm(q, k, qn, kn, round_intermediates=False)
    split = ops.rotary(nq, nk, cos, sin, round_intermediates=False)
    ref_q, ref_k = ops.qk_norm_reference(q, k, qn, kn, round_intermediates=False)
    expected = ops.rotary_reference(ref_q, ref_k, cos, sin, round_intermediates=False)
    for a, b, c in zip(actual, split, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a, c, atol=.008 if dtype == torch.bfloat16 else .001, rtol=.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_silu_keeps_product_in_fp32(dtype):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(17)
    gate, up = [torch.randn(17, 128, device="cuda", dtype=dtype) for _ in range(2)]
    expected = ops.silu_mul_reference(gate, up, round_intermediates=False)
    actual = ops.silu_mul(gate, up, round_intermediates=False)
    torch.testing.assert_close(actual, expected, atol=.008 if dtype == torch.bfloat16 else .001, rtol=.01)


def test_precision_adapter_binds_once_and_retains_native_slot_store():
    bound = ops.with_rounding(round_intermediates=False)
    for name in ["rms_norm", "qk_norm", "qk_norm_rope", "rotary", "silu_mul"]:
        assert getattr(bound, name).func is getattr(ops, name)
        assert getattr(bound, name).keywords == {"round_intermediates": False}
    assert bound.store_kv is ops.store_kv
