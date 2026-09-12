"""Tests for the GLM fp8 blockwise grouped GEMM (torch_ops).

The torch paths (activation quantization, blockwise GEMM reference, MoE
orchestration) run everywhere and pin the numerics contract. The kernel
dispatch (native-cast Triton GEMV on CUDA; CuTe DSL sm90 blockwise GEMM when
the CuTe DSL is installed) is GPU-gated: parity vs the torch oracle at real
GLM MoE shapes is the acceptance gate.
"""

import pytest

torch = pytest.importorskip(
    "torch", reason="torch paths are lazily imported in production; tests need it"
)

from vkernels.torch_ops.glm_fp8_blockwise_gemm import (  # noqa: E402
    _torch_blockwise_gemm,
    fp8_blockwise_gemm,
    glm_moe_grouped_gemm,
    quantize_activations_fp8,
)


def _fake_expert_stack(e=4, n=256, k=256, device="cpu"):
    """Small fp8 expert weights + block scales with known values."""
    gen = torch.Generator(device=device).manual_seed(7)
    w = (torch.randn(e, n, k, generator=gen, device=device) * 0.05).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(e, n // 128, k // 128, generator=gen, device=device) + 0.5
    return w.contiguous(), scales.contiguous()


def test_quantize_activations_roundtrip_error():
    x = torch.randn(32, 256, generator=torch.Generator().manual_seed(0)) * 3
    q, s = quantize_activations_fp8(x)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (32, 2)
    deq = q.to(torch.float32).view(32, 2, 128) * s.unsqueeze(-1)
    err = (deq.view(32, 256) - x).abs() / x.abs().clamp_min(1e-3)
    assert err.median() < 0.05, (
        f"activation quant too lossy: median rel {err.median():.4f}"
    )


def test_quantize_activations_rejects_bad_shapes():
    with pytest.raises(ValueError):
        quantize_activations_fp8(torch.randn(4, 100))  # K not multiple of 128


def test_blockwise_gemm_torch_matches_manual():
    m, n, k = 16, 256, 256
    gen = torch.Generator().manual_seed(1)
    a = (torch.randn(m, k, generator=gen) * 0.5).to(torch.float8_e4m3fn)
    asc = torch.rand((m + 127) // 128, k // 128, generator=gen) + 0.5
    b, bsc = _fake_expert_stack(e=1, n=n, k=k)
    out = torch.empty(m, n, dtype=torch.bfloat16)
    _torch_blockwise_gemm(a, asc, b[0], bsc[0], out)

    ref = torch.zeros(m, n)
    for kb in range(k // 128):
        a_blk = a[:, kb * 128 : (kb + 1) * 128].to(torch.float32) * asc[:, kb : kb + 1]
        w_blk = b[0][:, kb * 128 : (kb + 1) * 128].to(torch.float32)
        w_blk = w_blk.view(n // 128, 128, 128) * bsc[0][:, kb].unsqueeze(-1).unsqueeze(
            -1
        )
        ref += a_blk @ w_blk.reshape(n, 128).T
    torch.testing.assert_close(
        out.float(), ref.to(torch.bfloat16).float(), rtol=5e-2, atol=5e-2
    )


def test_blockwise_gemm_wrapper_validates():
    a = torch.zeros(8, 256, dtype=torch.float8_e4m3fn)
    b, _ = _fake_expert_stack(e=1)
    with pytest.raises(ValueError):
        fp8_blockwise_gemm(
            a, torch.zeros(8, 2), b[0], torch.zeros(1, 1)
        )  # bad scale shape


def test_moe_grouped_gemm_matches_gather_reference():
    """The orchestration must equal the per-(token,slot) gather+swiglu path."""
    t, e, k_route, h, i = 6, 4, 2, 128, 128
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(t, h, generator=gen)
    gu = (
        (torch.randn(e, 2 * i, h, generator=gen) * 0.05)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    gus = (torch.rand(e, 2 * i // 128, h // 128, generator=gen) + 0.5).contiguous()
    dn = (
        (torch.randn(e, h, i, generator=gen) * 0.05)
        .to(torch.float8_e4m3fn)
        .contiguous()
    )
    dns = (torch.rand(e, h // 128, i // 128, generator=gen) + 0.5).contiguous()
    idx = torch.stack([torch.randperm(e, generator=gen)[:k_route] for _ in range(t)])
    w = torch.softmax(torch.randn(t, k_route, generator=gen), -1)

    out = glm_moe_grouped_gemm(x, gu, gus, dn, dns, idx, w, swiglu_limit=0)

    # Naive per-(token, slot) reference using the SAME primitive + quantizer
    # (isolates the sort/gather/index_add orchestration from fp8 noise).
    from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
        _torch_blockwise_gemm,
        quantize_activations_fp8 as quant,
    )

    ref = torch.zeros(t, h)
    for ti in range(t):
        for ki in range(k_route):
            ei = int(idx[ti, ki])
            a8, asc = quant(x[ti : ti + 1])
            gu_out = torch.empty(1, 2 * i, dtype=torch.bfloat16)
            _torch_blockwise_gemm(a8, asc, gu[ei], gus[ei], gu_out)
            gate, up = gu_out[0, :i].float(), gu_out[0, i:].float()
            act = torch.nn.functional.silu(gate) * up
            a8d, ascd = quant(act.to(torch.bfloat16).unsqueeze(0))
            dn_out = torch.empty(1, h, dtype=torch.bfloat16)
            _torch_blockwise_gemm(a8d, ascd, dn[ei], dns[ei], dn_out)
            ref[ti] += w[ti, ki] * dn_out[0].float()
    torch.testing.assert_close(out.float(), ref, rtol=5e-2, atol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestGroupedNativeMultiTile:
    """Regression for #58: experts with count > BM (hot routing) span
    several row tiles. The tile map used to repeat the segment start/count
    across an expert's tiles, leaving rows beyond the first BM unwritten
    (torch.empty garbage -> NaNs under a dirty allocator)."""

    def test_hot_experts_match_per_expert_path(self):
        import torch.nn.functional as F

        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            e4m3fn_to_fnuz,
            glm_moe_grouped_gemm_native,
            quantize_activations_fnuz,
        )

        t, e, k_route, h, i = 256, 64, 8, 256, 256
        gen = torch.Generator(device="cuda").manual_seed(7)
        x = torch.randn(t, h, generator=gen, device="cuda", dtype=torch.bfloat16)
        gu = (
            (torch.randn(e, 2 * i, h, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        gus = (torch.rand(e, 2 * i // 128, h // 128, generator=gen, device="cuda") + 0.5).contiguous()
        dn = (
            (torch.randn(e, h, i, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        dns = (torch.rand(e, h // 128, i // 128, generator=gen, device="cuda") + 0.5).contiguous()
        # Skewed routing: experts 0 and 1 get ~4x the average count, forcing
        # multi-tile experts at the default BM=64 (2048 slots / 64 experts).
        idx = torch.randint(2, e, (t, k_route), generator=gen, device="cuda", dtype=torch.int64)
        idx[:, :3] = 0
        idx[:, 3:5] = 1
        w = torch.softmax(torch.randn(t, k_route, generator=gen, device="cuda"), -1)

        gu_nz, gu_sc2 = e4m3fn_to_fnuz(gu, gus)
        dn_nz, dn_sc2 = e4m3fn_to_fnuz(dn, dns)
        out = glm_moe_grouped_gemm_native(x, gu_nz, gu_sc2, dn_nz, dn_sc2, idx, w)

        # Reference: per-expert fp32 matmuls on the SAME fnuz operands with
        # the runner's swiglu (silu gate, +/-limit clamps, mirroring floe
        # arch._swiglu) -- isolates the grouped tile map from fp8 noise.
        # (glm_moe_grouped_gemm is NOT a valid reference here: its
        # API-stable wrapper uses a sigmoid-form swiglu.)
        slots = idx.reshape(-1)
        toks = torch.arange(t, device="cuda").repeat_interleave(k_route)
        order = torch.argsort(slots, stable=True)
        sorted_tok, sorted_exp = toks[order], slots[order]
        flat_w = w.reshape(-1)[order]  # per-slot weights in SORTED order
        a_nz, asc2 = quantize_activations_fnuz(x)

        def deq_a(p, s):
            """[rows, K] fnuz payload x [rows, K//128] scales -> fp32 [rows, K]."""
            g = s.shape[-1]
            return (p.float().view(-1, g, 128) * s.float()[:, :, None]).view(-1, g * 128)

        def deq_w(p, s):
            """[N, K] fnuz payload x [N//128, K//128] scales -> fp32 [N, K]."""
            g0, g1 = s.shape
            return (p.float().view(g0, 128, g1, 128) * s.float()[:, None, :, None]).view(g0 * 128, g1 * 128)

        gu_v = torch.empty(slots.numel(), 2 * i, device="cuda", dtype=torch.float32)
        for e in sorted_exp.unique().tolist():
            m = sorted_exp == e
            av = deq_a(a_nz[sorted_tok[m]], asc2[sorted_tok[m]])
            bv = deq_w(gu_nz[e], gu_sc2[e])
            gu_v[m] = av @ bv.t()
        # the native path stores gu as bf16 BEFORE the swiglu; matching that
        # rounding is required or act re-quantization flips fp8 payload ULPs
        gu_v = gu_v.to(torch.bfloat16).float()
        gate = gu_v[:, :i].clamp(max=7.0)
        up = gu_v[:, i:].clamp(min=-7.0, max=7.0)
        act = (F.silu(gate) * up).to(torch.bfloat16)
        a2_nz, a2_sc2 = quantize_activations_fnuz(act)
        ref = torch.zeros(t, h, device="cuda", dtype=torch.float32)
        for e in sorted_exp.unique().tolist():
            m = sorted_exp == e
            av2 = deq_a(a2_nz[m], a2_sc2[m])
            dv = deq_w(dn_nz[e], dn_sc2[e])
            dn_out = (av2 @ dv.t()).to(torch.bfloat16).float()
            ref.index_add_(0, sorted_tok[m], dn_out * flat_w[m].unsqueeze(-1))
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestGpuKernelParity:
    """On-cluster acceptance: the kernel paths must match the torch oracle
    at the real GLM MoE shapes (E=288 collapsed to active experts, N=4096,
    K=4096 gate_up / N=4096 K=2048 down)."""

    def test_native_cast_gemv_matches_reference(self):
        from vkernels.torch_ops.glm_expert_gemv import (
            expert_gemv,
            expert_gemv_reference,
        )

        gen = torch.Generator(device="cuda").manual_seed(11)
        e, o, i, t, kk = 288, 4096, 4096, 8, 8
        w = (torch.randn(e, o, i, generator=gen, device="cuda") * 0.05).to(
            torch.float8_e4m3fn
        )
        s = torch.rand(e, o // 128, i // 128, generator=gen, device="cuda") + 0.5
        x = torch.randn(t, i, generator=gen, device="cuda", dtype=torch.bfloat16)
        idx = torch.stack(
            [torch.randperm(e, generator=gen, device="cuda")[:kk] for _ in range(t)]
        ).to(torch.int64)
        idx = idx.cpu().to(torch.int64).cuda()
        got = expert_gemv(x, w, s, idx)
        ref = expert_gemv_reference(x, w, s, idx)
        rel = (
            got.float() - ref.float()
        ).abs().max() / ref.float().abs().max().clamp_min(1e-6)
        assert rel < 0.02, f"native-cast GEMV parity rel={rel:.4f}"

    def test_blockwise_gemm_kernel_if_present(self):
        from vkernels.torch_ops import glm_fp8_blockwise_gemm as mod

        if mod._cute_kernel() is None:
            pytest.skip("CuTe DSL kernel not wired yet (torch oracle path active)")
        gen = torch.Generator(device="cuda").manual_seed(12)
        m, n, k = 64, 4096, 4096
        a8, asc = mod.quantize_activations_fp8_blockwise(
            torch.randn(m, k, generator=gen, device="cuda")
        )
        b = (
            (torch.randn(n, k, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        bsc = (
            torch.rand(n // 128, k // 128, generator=gen, device="cuda") + 0.5
        ).contiguous()
        out = fp8_blockwise_gemm(a8, asc, b, bsc)
        ref = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        _torch_blockwise_gemm(a8, asc, b, bsc, ref)
        rel = (
            out.float() - ref.float()
        ).abs().max() / ref.float().abs().max().clamp_min(1e-6)
        assert rel < 0.02, f"blockwise GEMM kernel parity rel={rel:.4f}"
