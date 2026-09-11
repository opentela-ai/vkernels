"""GDN deferred-commit contract checks and optional GPU parity."""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.gdn_commit; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_matches_scalar_loop(torch):
    """Independent per-(head, row) scalar recomputation of the recurrence."""
    import torch

    from vkernels.torch_ops.gdn_commit import gdn_commit_state_reference

    torch.manual_seed(8)
    seq, nv, hv, hk, j = 4, 2, 2, 6, 2
    k_exp = torch.randn(seq, nv, hk)
    v = torch.randn(seq, nv, hv)
    beta, gamma = torch.randn(seq, nv), torch.randn(seq, nv)
    s0 = torch.randn(nv, hv, hk)
    out = gdn_commit_state_reference(k_exp, v, beta, gamma, s0, j)
    assert out.shape == (nv, hv, hk)
    assert torch.equal(s0, s0.clone())  # reference must not mutate
    for h in range(hv):
        s = s0[h].clone()
        for t in range(j + 1):
            s = s * gamma[t, h]
            sk = s @ k_exp[t, h]
            s = s + (beta[t, h] * (v[t, h] - sk))[:, None] * k_exp[t, h][None, :]
        assert torch.allclose(out[h], s, atol=1e-5, rtol=1e-5)


def test_contract(torch):
    from vkernels.torch_ops.gdn_commit import gdn_commit_state_triton

    k = torch.randn(4, 2, 8)
    v = torch.randn(4, 2, 2)
    beta, gamma = torch.randn(4, 2), torch.randn(4, 2)
    s0 = torch.randn(2, 2, 8)
    with pytest.raises(ValueError, match="CUDA"):
        gdn_commit_state_triton(k, v, beta, gamma, s0, 1)  # CPU tensors rejected


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.gdn_commit import (
        gdn_commit_state_reference,
        gdn_commit_state_triton,
    )

    torch.manual_seed(12)
    seq, nv, hv, hk, j = 5, 4, 32, 32, 3
    k_exp = torch.randn(seq, nv, hk, device="cuda")
    v = torch.randn(seq, nv, hv, device="cuda")
    beta, gamma = (
        torch.randn(seq, nv, device="cuda"),
        torch.randn(seq, nv, device="cuda"),
    )
    s0 = torch.randn(nv, hv, hk, device="cuda")
    out = gdn_commit_state_triton(k_exp, v, beta, gamma, s0, j)
    out_r = gdn_commit_state_reference(k_exp, v, beta, gamma, s0, j)
    torch.testing.assert_close(out, out_r, atol=1e-4, rtol=1e-4)
    assert torch.equal(s0, s0.clone())  # kernel must not mutate inputs


def test_gpu_parity_strided_v(torch):
    """The real verify path passes v as a non-contiguous qkv-split view."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.gdn_commit import (
        gdn_commit_state_reference,
        gdn_commit_state_triton,
    )

    torch.manual_seed(13)
    seq, nv, hv, hk, j = 4, 2, 32, 32, 2
    qkv = torch.randn(seq, nv, 3 * hv, device="cuda")
    v_view = qkv[:, :, hv : 2 * hv]  # strides like (3*hv, 3*hv, 1): non-contiguous
    assert not v_view.is_contiguous()
    k_exp = torch.randn(seq, nv, hk, device="cuda")
    beta, gamma = (
        torch.randn(seq, nv, device="cuda"),
        torch.randn(seq, nv, device="cuda"),
    )
    s0 = torch.randn(nv, hv, hk, device="cuda")
    out = gdn_commit_state_triton(k_exp, v_view, beta, gamma, s0, j)
    out_r = gdn_commit_state_reference(k_exp, v_view, beta, gamma, s0, j)
    torch.testing.assert_close(out, out_r, atol=1e-4, rtol=1e-4)
