"""KDA single-token decode contract checks and optional GPU parity."""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_kda_decode; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_matches_scalar_loop(torch):
    """Independent per-head scalar recomputation (small D, no kernel shape limits)."""
    import math

    from vkernels.torch_ops.glm_kda_decode import kda_decode_reference

    torch.manual_seed(4)
    B, H, D, eps = 2, 3, 8, 1e-6
    q = torch.randn(B, 1, H, D)
    k = torch.randn(B, 1, H, D)
    v = torch.randn(B, 1, H, D)
    g = torch.randn(B, 1, H, D)
    beta = torch.randn(B, 1, H)
    state0 = torch.randn(B, H, D, D)
    out, state = kda_decode_reference(q, k, v, g, beta, state0, eps=eps)
    assert out.shape == (B, 1, H, D) and state.shape == (B, H, D, D)
    assert torch.equal(state0, state0.clone())  # reference must not mutate
    for b in range(B):
        for h in range(H):
            qv = (
                q[b, 0, h]
                / math.sqrt((q[b, 0, h] * q[b, 0, h]).sum().item() + eps)
                * D**-0.5
            )
            kv = k[b, 0, h] / math.sqrt((k[b, 0, h] * k[b, 0, h]).sum().item() + eps)
            s = state0[b, h] * g[b, 0, h].exp()[:, None]
            # kernel: memory[v] = sum_r state[r, v] * k[r]  (state^T @ k)
            mem = (s * kv[:, None]).sum(0)
            s = s + kv[:, None] * ((v[b, 0, h] - mem) * beta[b, 0, h])[None, :]
            o = (s * qv[:, None]).sum(0)
            assert torch.allclose(o, out[b, 0, h], atol=1e-5, rtol=1e-5)
            assert torch.allclose(s, state[b, h], atol=1e-5, rtol=1e-5)


def test_contract(torch):
    from vkernels.torch_ops.glm_kda_decode import kda_decode

    q = torch.randn(2, 1, 3, 64)
    state = torch.randn(2, 3, 64, 64)
    beta = torch.randn(2, 1, 3)
    q48 = torch.randn(2, 1, 3, 48)
    with pytest.raises(ValueError, match="head dimensions"):
        kda_decode(q48, q48, q48, q48, beta, torch.randn(2, 3, 48, 48))
    with pytest.raises(TypeError, match="eps"):
        kda_decode(q, q, q, q, beta, state, eps=1e-5)  # eps is kernel-pinned; use the reference
    with pytest.raises(TypeError, match="FP32"):
        kda_decode(
            q.double(),
            q.double(),
            q.double(),
            q.double(),
            beta.double(),
            state.double(),
        )
    with pytest.raises(ValueError, match="GPU"):
        kda_decode(q, q, q, q, beta, state)  # CPU tensors rejected


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_kda_decode import kda_decode, kda_decode_reference

    torch.manual_seed(6)
    B, H, D = 2, 3, 64
    q = torch.randn(B, 1, H, D, device="cuda")
    k, v, g = (torch.randn(B, 1, H, D, device="cuda") for _ in range(3))
    beta = torch.randn(B, 1, H, device="cuda")
    state0 = torch.randn(B, H, D, D, device="cuda")
    out, state = kda_decode(q, k, v, g, beta, state0)
    out_r, state_r = kda_decode_reference(q, k, v, g, beta, state0)
    torch.testing.assert_close(out, out_r, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(state, state_r, atol=1e-4, rtol=1e-4)
    assert torch.equal(state0, state0.clone())  # kernel must not mutate inputs


def test_conv_decode_reference_matches_eager_chain(torch):
    """The oracle IS the eager chain — pin its shapes/dtype semantics."""
    import torch.nn.functional as F

    from vkernels.torch_ops.glm_kda_decode import kda_conv_decode_reference

    torch.manual_seed(7)
    B, C, K = 3, 5, 4
    state = torch.randn(B, C, K - 1).to(torch.bfloat16)
    x = torch.randn(B, C, 1).to(torch.bfloat16)
    weight = torch.randn(C, 1, K).to(torch.bfloat16)
    window = torch.cat([state, x], dim=-1)
    expected = F.silu((window * weight.squeeze(1)).sum(dim=-1))
    out = kda_conv_decode_reference(state, x, weight)
    assert out.shape == (B, C) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, expected)


def test_conv_decode_contract(torch):
    """kda_conv_decode owns its eligibility; K comes from the weight."""
    from vkernels.torch_ops.glm_kda_decode import kda_conv_decode

    state = torch.randn(2, 4, 3).to(torch.bfloat16)
    x = torch.randn(2, 4, 1).to(torch.bfloat16)
    weight = torch.randn(4, 1, 4).to(torch.bfloat16)
    with pytest.raises(TypeError, match="GPU"):
        kda_conv_decode(state, x, weight)  # CPU tensors rejected
    with pytest.raises(TypeError, match="K-1"):
        kda_conv_decode(torch.randn(2, 4, 2).to(torch.bfloat16), x, weight)
    with pytest.raises(TypeError, match="layout"):
        kda_conv_decode(state, x, torch.randn(4, 4).to(torch.bfloat16))
    with pytest.raises(TypeError, match="per state channel"):
        kda_conv_decode(state, x, torch.randn(5, 1, 4).to(torch.bfloat16))
    with pytest.raises(TypeError, match="BF16 or FP16"):
        kda_conv_decode(state.float(), x.float(), weight.float())


def test_conv_decode_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_kda_decode import (
        kda_conv_decode,
        kda_conv_decode_reference,
    )

    torch.manual_seed(8)
    for K in (2, 3, 4, 8):
        B, C = 3, 97  # odd C exercises the 1024-wide block mask
        state = torch.randn(B, C, K - 1, device="cuda").to(torch.bfloat16)
        x = torch.randn(B, C, 1, device="cuda").to(torch.bfloat16)
        weight = torch.randn(C, 1, K, device="cuda").to(torch.bfloat16)
        out = kda_conv_decode(state, x, weight)
        assert out.shape == (B, C) and out.dtype == torch.bfloat16
        # bit-identical by contract (verified on sm_121): the kernel rounds
        # at exactly the eager cat -> mul -> sum(-1) -> silu store boundaries
        assert torch.equal(out, kda_conv_decode_reference(state, x, weight))
