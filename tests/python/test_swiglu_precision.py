"""The final-store activation contract is separate from eager BF16 rounding."""
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from vkernels.torch_ops._dispatch import OpNotEligible
from vkernels.torch_ops.elementwise import swiglu_limit


def test_final_store_rejects_cpu():
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    with pytest.raises(OpNotEligible):
        swiglu_limit(x, x, 10.0, round_intermediates=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("shape", [(1, 3072), (2, 1536), (128, 3072)])
@pytest.mark.parametrize("limit", [10.0, 1.234])
def test_single_store_matches_fp32_oracle_with_projection_views(shape, limit):
    torch.manual_seed(163)
    packed = (torch.randn(*shape[:-1], shape[-1] * 2, device="cuda") * 5).bfloat16()
    gate, up = packed.chunk(2, dim=-1)
    saved = packed.clone()
    expected = (F.silu(gate.float().clamp(max=limit)) * up.float().clamp(-limit, limit)).bfloat16()
    actual = swiglu_limit(gate, up, limit, round_intermediates=False)
    # Exp implementations can differ at ties; BF16 ULP tolerance is explicit.
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-6)
    eager = F.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
    assert not torch.equal(expected, eager)
    torch.testing.assert_close(swiglu_limit(gate, up, limit), eager, rtol=0, atol=0)
    assert torch.equal(packed, saved)
