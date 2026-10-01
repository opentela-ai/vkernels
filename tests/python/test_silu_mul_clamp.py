"""silu_mul_clamp: the vendored grouped-MoE activation kernel.

The Triton kernel (torch_ops/silu_mul_clamp.py, the SGLang sgl-kernel
``silu_and_mul_with_thresh`` design) replaces the between-stages eager
chain of the grouped fp8 MoE — fp32 casts + clamps + silu + mul + bf16
cast, five-to-seven launches per MoE layer — with ONE launch (injected
through the ``swiglu_fn`` seam of ``glm_moe_grouped_gemm_native`` at the
call sites). Contract: the eager chain's exact fp32 op sequence with ONE
bf16 round at the store — bit-identical except ~1e-6 of elements on exp
rounding boundaries (sub-ulp; measured 4/4.2M elements, worst 1.2e-07,
on GB200-class local hardware); the eager chain stays the oracle and the
fallback.

CPU environments exercise the fallback/contract paths only; the kernel
itself needs CUDA (skipped otherwise, matching the house pattern).
Ported from floe's test_silu_mul_clamp_kernel.py (the knob + seam tests
stay in floe).
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.silu_mul_clamp import (  # noqa: E402
    silu_mul_clamp,
    silu_mul_clamp_eligible,
)
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# ---------------------------------------------------------------------------
# the eager oracle — the in-op chain, also the fallback contract
# ---------------------------------------------------------------------------
def _oracle(gate_up, limit):
    gate = gate_up[:, : gate_up.shape[1] // 2].float().clamp(max=limit)
    up = gate_up[:, gate_up.shape[1] // 2 :].float().clamp(
        min=-limit, max=limit
    )
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def _bf16_ulp(x32: torch.Tensor) -> torch.Tensor:
    """bf16 spacing at each element of a float32 tensor."""
    nxt = torch.nextafter(
        x32.to(torch.bfloat16).float(), torch.full_like(x32, float("inf"))
    )
    return (nxt - x32.to(torch.bfloat16).float()).abs()


# ---------------------------------------------------------------------------
# eligibility + contract
# ---------------------------------------------------------------------------
def test_eligibility():
    gu = torch.zeros(4, 64, dtype=torch.bfloat16)
    assert not silu_mul_clamp_eligible(gu)  # CPU
    if not torch.cuda.is_available():
        return
    gu = gu.cuda()
    assert silu_mul_clamp_eligible(gu)
    assert not silu_mul_clamp_eligible(gu[:, :32])  # non-contiguous view
    assert not silu_mul_clamp_eligible(gu.float())  # dtype
    assert not silu_mul_clamp_eligible(gu[:, :33])  # odd inner (also a view)


def test_cpu_rejects():
    with pytest.raises(OpNotEligible):
        silu_mul_clamp(torch.zeros(2, 8, dtype=torch.bfloat16), 7.0)


# ---------------------------------------------------------------------------
# parity vs the eager oracle
# ---------------------------------------------------------------------------
@gpu
@pytest.mark.parametrize("shape", [(1, 256), (3, 128), (32, 512), (256, 1536)])
def test_bitexact_vs_eager(shape):
    torch.manual_seed(sum(shape))
    gu = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 5.0
    assert torch.equal(silu_mul_clamp(gu, 7.0), _oracle(gu, 7.0))


@gpu
def test_clamp_saturation_exact():
    # both halves far beyond the limit: the clamps dominate and the chain
    # is exactly limit * silu(limit) territory — no boundary ambiguity.
    gu = torch.linspace(-40.0, 40.0, 2048, device="cuda").to(torch.bfloat16)
    gu = gu[None, :].expand(8, -1).contiguous()
    assert torch.equal(silu_mul_clamp(gu, 7.0), _oracle(gu, 7.0))


@gpu
def test_large_sweep_subulp_bound():
    # Adversarial large sweep: bit-identical except a ~1e-6 tail where the
    # fp32 exp rounding differs by ONE fp32 rounding step (<= 1.2e-07
    # absolute); the bf16 store preserves a difference only where the
    # pre-round value sits on a store rounding boundary (near zero, where
    # bf16 spacing collapses, it always survives — hence the absolute
    # floor next to the one-bf16-ulp bound).
    torch.manual_seed(11)
    gu = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16) * 6.0
    fused, ref = silu_mul_clamp(gu, 7.0), _oracle(gu, 7.0)
    mism = fused != ref
    rate = mism.float().mean().item()
    assert rate <= 1e-5, f"mismatch rate {rate:.2e} exceeds the contract"
    if mism.any():
        diff = (fused.float() - ref.float()).abs()
        bound = torch.maximum(
            _bf16_ulp(ref.float()),
            torch.full_like(ref.float(), 1.2e-07),
        )
        assert (diff[~mism] == 0).all()
        assert (diff[mism] <= bound[mism]).all(), "mismatch beyond one rounding step"


@gpu
def test_zero_rows():
    gu = torch.zeros(0, 512, device="cuda", dtype=torch.bfloat16)
    out = silu_mul_clamp(gu, 7.0)
    assert out.shape == (0, 256)
