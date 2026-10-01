"""moe_weighted_sum: the vendored weighted top-k reduce vs the eager oracle.

The Triton kernel (torch_ops/moe_combine.py, adapted from SGLang's
_moe_sum_reduce_kernel) replaces ``(out * w.unsqueeze(-1)).sum(dim=1)``
on the routed-expert decode paths. Contract: fp32 multiply-accumulate
over K, ONE rounding at the store dtype — the eager pair rounds ``w`` and
each product at bf16 first. Parity is therefore tolerance-gated (bf16
ulp scale), the same gate the grouped path ships.

CPU environments exercise the wrapper's fallback/contract paths only;
the kernel itself needs CUDA (skipped otherwise, matching the house
pattern for GPU-kernel tests). Ported from floe's
test_moe_combine_kernel.py (the arch-knob forward test stays in floe).
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.moe_combine import (  # noqa: E402
    moe_weighted_sum,
    moe_weighted_sum_eligible,
)
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402


def _case(device, dtype, t, k, h, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    out = torch.randn(t, k, h, device=device, dtype=dtype, generator=gen)
    w = torch.softmax(torch.randn(t, k, device=device, generator=gen), dim=-1)
    return out, w


def _eager(out, w):
    return (out * w.to(out.dtype).unsqueeze(-1)).sum(dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 8, 4096), (4, 8, 4096), (2, 3, 512), (8, 8, 1536)])
def test_weighted_sum_matches_eager(dtype, shape):
    out, w = _case("cuda", dtype, *shape)
    fused = moe_weighted_sum(out, w)
    oracle = _eager(out, w)
    torch.testing.assert_close(fused, oracle, rtol=2e-2, atol=2e-2 if dtype != torch.float32 else 1e-5)
    assert fused.shape == (shape[0], shape[2]) and fused.dtype == out.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_weighted_sum_preallocated_result_and_strides():
    out, w = _case("cuda", torch.bfloat16, 4, 8, 1024)
    result = torch.empty(4, 1024, device="cuda", dtype=torch.bfloat16)
    fused = moe_weighted_sum(out, w, result=result)
    assert fused is result
    torch.testing.assert_close(fused, _eager(out, w), rtol=2e-2, atol=2e-2)
    # strided (non-contiguous) input rows: the kernel strides, does not assume contiguity
    wide = torch.randn(4, 8, 2048, device="cuda", dtype=torch.bfloat16)
    view = wide[:, :, ::2]
    assert not view.is_contiguous()
    wv = torch.softmax(torch.randn(4, 8, device="cuda"), dim=-1)
    torch.testing.assert_close(
        moe_weighted_sum(view, wv), _eager(view, wv), rtol=2e-2, atol=2e-2)


def test_eligibility_contract():
    if torch.cuda.is_available():
        out, w = _case("cuda", torch.bfloat16, 2, 4, 128)
        assert moe_weighted_sum_eligible(out, w)
        assert not moe_weighted_sum_eligible(out, w.to(torch.bfloat16))  # fp32 weights required
        assert not moe_weighted_sum_eligible(out[0], w)  # ndim
        assert not moe_weighted_sum_eligible(out, w[:, :2])  # K mismatch
    cpu_out = torch.randn(2, 4, 8)
    cpu_w = torch.ones(2, 4)
    assert not moe_weighted_sum_eligible(cpu_out, cpu_w)  # CUDA required
    with pytest.raises(OpNotEligible):
        moe_weighted_sum(cpu_out, cpu_w)
