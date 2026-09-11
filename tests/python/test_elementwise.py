"""Elementwise fusion contract checks and optional GPU parity.

The eager HF expressions (with their exact intermediate rounding) are the
oracle; the ``*_reference`` functions in the module restate them and the
tests cross-check both.
"""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.elementwise; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


class _Norm:
    """Minimal RMSNorm stub (HF attribute names: weight, variance_epsilon)."""

    def __init__(self, torch, d, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(d, generator=g).to(torch.bfloat16)
        self.variance_epsilon = 1e-5


def test_rms_norm_reference_matches_hf_eager(torch):
    from vkernels.torch_ops.elementwise import rms_norm_reference

    torch.manual_seed(21)
    x = torch.randn(6, 32, dtype=torch.bfloat16)
    r = torch.randn(6, 32, dtype=torch.bfloat16)
    module = _Norm(torch, 32, seed=1)
    out, summed = rms_norm_reference(x, module, residual=r)
    x32 = (x.float() + r.float()).to(torch.bfloat16).float()
    inv = torch.rsqrt(x32.pow(2).sum(-1, keepdim=True) / 32 + module.variance_epsilon)
    expected = ((x32 * inv).to(torch.bfloat16).float() * module.weight.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(summed, (x.float() + r.float()).to(torch.bfloat16))


def test_silu_mul_reference_matches_hf_eager(torch):
    from vkernels.torch_ops.elementwise import silu_mul_reference

    torch.manual_seed(22)
    gate = torch.randn(5, 16, dtype=torch.bfloat16)
    up = torch.randn(5, 16, dtype=torch.bfloat16)
    act = (gate.float() / (1.0 + torch.exp(-gate.float()))).to(torch.bfloat16)
    expected = (act.float() * up.float()).to(torch.bfloat16)
    torch.testing.assert_close(silu_mul_reference(gate, up), expected)


def test_store_kv_reference_skips_scratch(torch):
    import torch

    from vkernels.torch_ops.elementwise import store_kv_reference

    b, n, h, d, max_total = 2, 3, 2, 8, 12
    k = torch.randn(b, n, h, d)
    v = torch.randn(b, n, h, d)
    kc = torch.zeros(max_total, h, d)
    vc = torch.zeros(max_total, h, d)
    # Slot 0 is the shared scratch/null page: batch 1's padding targets it.
    block_table = torch.tensor([[1, 2, 3, 4, 5], [0, 0, 0, 6, 7]], dtype=torch.int32)
    seqlens = torch.tensor([0, 2], dtype=torch.int32)
    out_kc, out_vc = store_kv_reference(
        k, v, kc, vc, block_table, seqlens, scratch_slot=0
    )
    assert torch.equal(out_kc[0], torch.zeros(h, d))  # scratch never written
    for t in range(3):
        assert torch.allclose(out_kc[1 + t], k[0, t])
        assert torch.allclose(out_vc[1 + t], v[0, t])
    for t in range(3):
        slot = int(block_table[1, 2 + t])
        if slot == 0:
            continue  # shared scratch/null page is never written
        assert torch.allclose(out_kc[slot], k[1, t])
        assert torch.allclose(out_vc[slot], v[1, t])
    assert torch.equal(out_kc[0], torch.zeros(h, d))


def test_gpu_parity(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    import torch

    from vkernels.torch_ops.elementwise import (
        qk_norm,
        qk_norm_reference,
        rms_norm,
        rms_norm_reference,
        rotary,
        rotary_reference,
        silu_mul,
        silu_mul_reference,
        store_kv,
        store_kv_reference,
    )

    torch.manual_seed(23)
    dev, dt = "cuda", torch.bfloat16

    x = torch.randn(6, 32, device=dev, dtype=dt)
    r = torch.randn(6, 32, device=dev, dtype=dt)
    module = _Norm(torch, 32, seed=1)
    module.weight = module.weight.to(dev)
    out, summed = rms_norm(x, module, residual=r)
    out_r, summed_r = rms_norm_reference(x, module, residual=r)
    torch.testing.assert_close(out, out_r)
    torch.testing.assert_close(summed, summed_r)

    q = torch.randn(4, 6, 32, device=dev, dtype=dt)
    k = torch.randn(4, 2, 32, device=dev, dtype=dt)
    q_norm, k_norm = _Norm(torch, 32, seed=3), _Norm(torch, 32, seed=4)
    q_norm.weight, k_norm.weight = q_norm.weight.to(dev), k_norm.weight.to(dev)
    oq, ok = qk_norm(q, k, q_norm, k_norm)
    oq_r, ok_r = qk_norm_reference(q, k, q_norm, k_norm)
    torch.testing.assert_close(oq, oq_r)
    torch.testing.assert_close(ok, ok_r)

    cos = torch.randn(4, 32, device=dev, dtype=dt)
    sin = torch.randn(4, 32, device=dev, dtype=dt)
    ro_q, ro_k = rotary(q, k, cos, sin)
    ro_q_r, ro_k_r = rotary_reference(q, k, cos, sin)
    torch.testing.assert_close(ro_q, ro_q_r)
    torch.testing.assert_close(ro_k, ro_k_r)

    gate = torch.randn(4, 5, 16, device=dev, dtype=dt)
    up = torch.randn(4, 5, 16, device=dev, dtype=dt)
    torch.testing.assert_close(silu_mul(gate, up), silu_mul_reference(gate, up))

    b, n, h, d, max_total = 2, 3, 2, 32, 12
    kk = torch.randn(b, n, h, d, device=dev, dtype=dt)
    vv = torch.randn(b, n, h, d, device=dev, dtype=dt)
    kc = torch.zeros(max_total, h, d, device=dev, dtype=dt)
    vc = torch.zeros(max_total, h, d, device=dev, dtype=dt)
    block_table = torch.tensor(
        [[1, 2, 3, 4, 5], [0, 0, 0, 6, 7]], dtype=torch.int32, device=dev
    )
    seqlens = torch.tensor([0, 2], dtype=torch.int32, device=dev)
    kc0, vc0 = kc.clone(), vc.clone()  # pristine caches for the oracle
    store_kv(kk, vv, kc, vc, block_table, seqlens, scratch_slot=0)
    exp_kc, exp_vc = store_kv_reference(
        kk, vv, kc0, vc0, block_table, seqlens, scratch_slot=0
    )
    torch.testing.assert_close(kc, exp_kc)
    torch.testing.assert_close(vc, exp_vc)
