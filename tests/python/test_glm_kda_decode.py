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
