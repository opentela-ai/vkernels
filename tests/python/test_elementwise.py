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


def test_kernels_surface_is_named():
    """_kernels() returns a NamedTuple — consumers must use named access.

    Positional unpacks of the kernel set rebind silently when it grows or
    shuffles (the clariden job 3499983 incident); the named surface is the
    contract. Runs wherever triton imports (defining the kernels needs no
    GPU — only launching does).
    """
    pytest.importorskip("triton")
    from vkernels.torch_ops.elementwise import _kernels

    kernels = _kernels()
    for name in (
        "norm",
        "qk_norm",
        "rope",
        "store_kv",
        "qk_norm_rope",
        "norm_uw",
        "norm_gated",
        "swiglu_limit",
    ):
        assert callable(getattr(kernels, name)), name


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


def test_swiglu_limit_reference_matches_eager_chain(torch):
    from vkernels.torch_ops.elementwise import swiglu_limit_reference

    torch.manual_seed(23)
    gate = torch.randn(7, 16, dtype=torch.bfloat16) * 12
    up = torch.randn(7, 16, dtype=torch.bfloat16) * 12
    limit = 6.0
    # the clamps must bind, or this test proves nothing about them
    assert bool((gate > limit).any()) and bool((up.abs() > limit).any())
    expected = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(
        min=-limit, max=limit
    )
    torch.testing.assert_close(swiglu_limit_reference(gate, up, limit), expected)


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
        swiglu_limit,
        swiglu_limit_reference,
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

    # fused qk_norm+rope must match the two-kernel chain bit-for-bit
    from vkernels.torch_ops.elementwise import qk_norm_rope

    fq, fk = qk_norm_rope(q, k, q_norm, k_norm, cos, sin)
    nq, nk = qk_norm(q, k, q_norm, k_norm)
    rq, rk = rotary(nq, nk, cos, sin)
    torch.testing.assert_close(fq, rq)
    torch.testing.assert_close(fk, rk)

    gate = torch.randn(4, 5, 16, device=dev, dtype=dt) * 12
    up = torch.randn(4, 5, 16, device=dev, dtype=dt) * 12
    limit = 6.0
    torch.testing.assert_close(silu_mul(gate, up), silu_mul_reference(gate, up))
    # GLM swiglu is bit-identical to the eager chain (verified on sm_121);
    # silu_mul is the same kernel with the clamps disabled (limit=+inf).
    assert torch.equal(
        swiglu_limit(gate, up, limit), swiglu_limit_reference(gate, up, limit)
    )
    # NaN must propagate like torch.clamp (propagate_nan=ALL, not minnum —
    # minnum semantics would return LIMIT for a NaN gate).
    g_nan = torch.tensor([[float("nan"), 3.0]], device=dev, dtype=dt)
    u_nan = torch.ones_like(g_nan)
    assert torch.equal(
        swiglu_limit(g_nan, u_nan, limit).isnan(),
        swiglu_limit_reference(g_nan, u_nan, limit).isnan(),
    )

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


def test_rms_norm_gated_reference_matches_floe_eager(torch):
    """The oracle restates floe's ``Glm53RMSNormGated`` eager chain exactly:
    fp32 stats -> rsqrt -> x*inv -> *w -> *sigmoid(gate) -> one bf16 cast."""
    import torch

    from vkernels.torch_ops.elementwise import rms_norm_gated_reference

    torch.manual_seed(5)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    gate = torch.randn(4, 16, dtype=torch.bfloat16) * 8
    module = _Norm(torch, 16, seed=7)

    dtype = x.dtype
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
    x32 = module.weight.float() * x32
    expected = (x32 * torch.sigmoid(gate.float())).to(dtype)
    torch.testing.assert_close(rms_norm_gated_reference(x, gate, module), expected)


def test_gpu_rms_norm_gated_parity(torch):
    """``rms_norm_gated`` (sigmoid, GLM o_norm) — one launch, strict fp32,
    the eager left-to-right multiply order. This is the T5 target-2a
    variant: the kernel family already covers the sigmoid activation, so
    the parity bar is the eager chain itself, not a sibling kernel."""
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    import torch

    from vkernels.torch_ops.elementwise import rms_norm_gated, rms_norm_gated_reference

    torch.manual_seed(19)
    for rows, d in [(6, 128), (1, 128), (17, 64), (3, 96)]:
        x = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(rows, d, device="cuda", dtype=torch.bfloat16) * 12
        module = _Norm(torch, d, seed=3)
        module.weight = module.weight.to("cuda")
        out = rms_norm_gated(x, gate, module)
        ref = rms_norm_gated_reference(x, gate, module)
        torch.testing.assert_close(out, ref)
        # bf16 store contract: out rounds exactly where the reference does
        assert out.dtype == torch.bfloat16

    # fp64 oracle cross-check (stats-dominated eps region)
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    module = _Norm(torch, 128, seed=9)
    module.weight = module.weight.to("cuda")
    out = rms_norm_gated(x, gate, module).float()
    x64 = x.double()
    inv = torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + module.variance_epsilon)
    oracle = (((x64 * inv) * module.weight.double()) * torch.sigmoid(gate.double())).float()
    torch.testing.assert_close(out, oracle, atol=2e-2, rtol=2e-2)

    # gate -> -inf saturates fp32 sigmoid to exactly 0: the product is an
    # exact bf16 zero; a NaN gate propagates like the eager chain
    gate_inf = torch.full((2, 128), float("-inf"), device="cuda", dtype=torch.bfloat16)
    assert torch.equal(rms_norm_gated(x[:2], gate_inf, module), torch.zeros(2, 128, device="cuda", dtype=torch.bfloat16))
    g_nan = gate.clone()
    g_nan[0, 0] = float("nan")
    assert torch.isnan(rms_norm_gated(x, g_nan, module)[0, 0])
