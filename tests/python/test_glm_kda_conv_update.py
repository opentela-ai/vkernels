"""kda_conv_update: the vendored conv + SiLU + state-roll kernel.

The Triton kernel (torch_ops/glm_kda_conv_update.py, the SGLang
causal_conv1d_update design) replaces the KDA decode conv step — window
cat + broadcast-mul + sum(-1) + F.silu PLUS the cache state roll (cat +
D2D memcpy) — with ONE launch that consumes and rolls the state buffer in
place. Contract: the activation is BIT-IDENTICAL to the eager chain (the
``kda_conv_decode`` rounding contract: bf16 product rounding per tap,
fp32 tail accumulation, one round on the conv store, SiLU in fp32 on the
stored value) and the state roll is pure storage-dtype moves.

CPU environments exercise the fallback/contract paths only; the kernel
itself needs CUDA (skipped otherwise, matching the house pattern).
Ported from floe's test_kda_conv_update_kernel.py (the site-dispatch /
knob tests stay in floe).
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.glm_kda_conv_update import (  # noqa: E402
    kda_conv_update,
    kda_conv_update_eligible,
)
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _oracle_step(weight, state, x):
    """The exact eager decode step: conv+silu, then the cat state roll."""
    window = torch.cat([state, x], dim=-1)
    act = torch.nn.functional.silu((window * weight.squeeze(1)).sum(dim=-1))
    rolled = torch.cat([state[..., 1:], x.to(state.dtype)], dim=-1)
    return act, rolled


@gpu
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "batch,channels", [(1, 6144), (2, 512), (4, 3072), (8, 1024), (1, 100), (3, 1025)]
)
def test_conv_update_parity_and_roll(dtype, batch, channels):
    pytest.importorskip("triton")
    torch.manual_seed(29)
    state = torch.randn(batch, channels, 3, device="cuda", dtype=dtype)
    # the decode-site layout: x arrives as [B, 1, C].transpose(1, 2)
    x = (
        torch.randn(batch, 1, channels, device="cuda", dtype=dtype)
        .transpose(1, 2)
        .contiguous()
    )
    weight = torch.randn(channels, 1, 4, device="cuda", dtype=dtype)
    fused_state = state.clone()
    act = kda_conv_update(fused_state, x, weight)
    ref_act, ref_state = _oracle_step(weight, state, x)
    assert act.shape == (batch, channels) and act.dtype == dtype
    if dtype is torch.bfloat16:
        # bit-identical (the serving dtype): every rounding point matches
        torch.testing.assert_close(act, ref_act, rtol=0, atol=0)
    else:
        # tolerance-class, the landed silu_mul/swiglu_limit fp16 profile
        # (tl.exp vs CUDA expf lowering on the silu)
        torch.testing.assert_close(act, ref_act, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(fused_state, ref_state, rtol=0, atol=0)  # exact roll


@gpu
@pytest.mark.parametrize("kernel_size", [2, 3, 8])
def test_conv_update_kernel_widths(kernel_size):
    pytest.importorskip("triton")
    torch.manual_seed(31)
    channels = 256
    state = torch.randn(2, channels, kernel_size - 1, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, channels, 1, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, 1, kernel_size, device="cuda", dtype=torch.bfloat16)
    fused_state = state.clone()
    act = kda_conv_update(fused_state, x, weight)
    ref_act, ref_state = _oracle_step(weight, state, x)
    torch.testing.assert_close(act, ref_act, rtol=0, atol=0)
    torch.testing.assert_close(fused_state, ref_state, rtol=0, atol=0)


@gpu
def test_conv_update_three_step_stream():
    """The roll composes: a 3-token stream matches the oracle loop exactly."""
    pytest.importorskip("triton")
    torch.manual_seed(37)
    channels = 1536
    weight = torch.randn(channels, 1, 4, device="cuda", dtype=torch.bfloat16)
    fused_state = torch.randn(2, channels, 3, device="cuda", dtype=torch.bfloat16)
    eager_state = fused_state.clone()
    for _ in range(3):
        x = torch.randn(2, 1, channels, device="cuda", dtype=torch.bfloat16).transpose(1, 2).contiguous()
        act = kda_conv_update(fused_state, x, weight)
        ref_act, eager_state = _oracle_step(weight, eager_state, x)
        torch.testing.assert_close(act, ref_act, rtol=0, atol=0)
        torch.testing.assert_close(fused_state, eager_state, rtol=0, atol=0)


@gpu
def test_conv_update_contract():
    pytest.importorskip("triton")
    st = torch.randn(2, 96, 3, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(2, 96, 1, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, 1, 4, device="cuda", dtype=torch.bfloat16)
    assert kda_conv_update_eligible(st, x, w)
    with pytest.raises(OpNotEligible):
        kda_conv_update(st.cpu(), x.cpu(), w.cpu())  # CPU -> eager fallback
    with pytest.raises(OpNotEligible):
        kda_conv_update(st, x, w.float())  # mixed dtypes -> eager fallback
    with pytest.raises(OpNotEligible):
        kda_conv_update(st, x, torch.randn(96, 1, 5, device="cuda", dtype=torch.bfloat16))  # bad K
    bad_x = torch.randn(2, 96, 2, device="cuda", dtype=torch.bfloat16)
    assert not kda_conv_update_eligible(st, bad_x, w)  # x width != 1
    assert not kda_conv_update_eligible(st, x, w.float())  # dtype mix
    assert not kda_conv_update_eligible(
        st[:, :, ::2].contiguous(), x[:, :, ::2].contiguous(), w
    )  # state/x channel mismatch
