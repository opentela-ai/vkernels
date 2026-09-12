"""mHC mix contract checks and optional GPU parity."""

import math
import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_mhc_mix; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_matches_scalar_loop(torch):
    """Independent per-element recomputation of the exact kernel order."""
    from vkernels.torch_ops.glm_mhc_mix import mhc_mix_reference

    hc, iters, eps = 2, 3, 1e-6
    torch.manual_seed(3)
    logits = torch.randn(5, hc * (hc + 2))
    base = torch.randn(hc * (hc + 2))
    scale = torch.randn(3)
    pre, post, comb = mhc_mix_reference(
        logits, base, scale, hc=hc, eps=eps, sinkhorn_iters=iters
    )
    assert pre.shape == (5, hc) and post.shape == (5, hc) and comb.shape == (5, hc, hc)
    for tok in range(5):
        for j in range(hc):
            assert pre[tok, j] == pytest.approx(
                1 / (1 + math.exp(-(logits[tok, j] * scale[0] + base[j]))) + eps,
                abs=1e-6,
            )
            assert post[tok, j] == pytest.approx(
                2 / (1 + math.exp(-(logits[tok, hc + j] * scale[1] + base[hc + j]))),
                abs=1e-6,
            )
        m = [
            [
                logits[tok, 2 * hc + a * hc + b] * scale[2] + base[2 * hc + a * hc + b]
                for b in range(hc)
            ]
            for a in range(hc)
        ]
        am = [max(row) for row in m]
        e = [[math.exp(m[a][b] - am[a]) for b in range(hc)] for a in range(hc)]
        rs = [sum(e[a]) for a in range(hc)]
        v = [[e[a][b] / rs[a] + eps for b in range(hc)] for a in range(hc)]
        cs = [sum(v[a][b] for a in range(hc)) for b in range(hc)]
        v = [[v[a][b] / (cs[b] + eps) for b in range(hc)] for a in range(hc)]
        for _ in range(iters - 1):
            rs = [sum(v[a]) for a in range(hc)]
            v = [[v[a][b] / (rs[a] + eps) for b in range(hc)] for a in range(hc)]
            cs = [sum(v[a][b] for a in range(hc)) for b in range(hc)]
            v = [[v[a][b] / (cs[b] + eps) for b in range(hc)] for a in range(hc)]
        for a in range(hc):
            for b in range(hc):
                assert comb[tok, a, b] == pytest.approx(v[a][b], rel=1e-5, abs=1e-7)


def test_contract(torch):
    from vkernels.torch_ops.glm_mhc_mix import mhc_mix

    torch.manual_seed(5)
    logits = torch.randn(2, 24)
    base, scale = torch.randn(24), torch.randn(3)
    with pytest.raises(ValueError, match="hc values"):
        mhc_mix(logits, base, scale, hc=3)
    with pytest.raises(ValueError, match="width"):
        mhc_mix(torch.randn(2, 23), base, scale)
    with pytest.raises(TypeError, match="FP32"):
        mhc_mix(logits.double(), base.double(), scale.double())
    with pytest.raises(ValueError, match="GPU"):
        mhc_mix(logits, base, scale)  # CPU tensors rejected


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_mhc_mix import mhc_mix, mhc_mix_reference

    torch.manual_seed(9)
    logits = torch.randn(7, 24, device="cuda")
    base, scale = torch.randn(24, device="cuda"), torch.randn(3, device="cuda")
    pre, post, comb = mhc_mix(logits, base, scale, hc=4, sinkhorn_iters=6)
    pre_r, post_r, comb_r = mhc_mix_reference(
        logits, base, scale, hc=4, sinkhorn_iters=6
    )
    torch.testing.assert_close(pre, pre_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(post, post_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(comb, comb_r, atol=1e-5, rtol=1e-5)
