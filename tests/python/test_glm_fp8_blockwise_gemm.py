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


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="fnuz fp8 operands need fp8e4b8 tensor cores (CDNA3); NVIDIA Triton has no fp8e4b8",
)
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

    def _native_cast_gemv_parity(self, e, o, i, t, kk, t_cap=None):
        from vkernels.torch_ops.glm_expert_gemv import (
            expert_gemv,
            expert_gemv_reference,
        )

        gen = torch.Generator(device="cuda").manual_seed(11)
        w = (torch.randn(e, o, i, generator=gen, device="cuda") * 0.05).to(
            torch.float8_e4m3fn
        )
        s = torch.rand(e, o // 128, i // 128, generator=gen, device="cuda") + 0.5
        x = torch.randn(t, i, generator=gen, device="cuda", dtype=torch.bfloat16)
        idx = torch.stack(
            [torch.randperm(e, generator=gen, device="cuda")[:kk] for _ in range(t)]
        ).to(torch.int64)
        idx = idx.cpu().to(torch.int64).cuda()
        got = expert_gemv(x, w, s, idx, t_cap=t_cap)
        ref = expert_gemv_reference(x, w, s, idx, t_cap=t_cap)
        rel = (
            got.float() - ref.float()
        ).abs().max() / ref.float().abs().max().clamp_min(1e-6)
        assert rel < 0.02, f"native-cast GEMV parity rel={rel:.4f}"

    def test_native_cast_gemv_matches_reference(self):
        """T=8 is the DFlash2 verify/replay block, which the wrapper only
        admits when the caller widens the cap — passed here explicitly via
        ``t_cap``.

        Small shapes, so this parity check runs on every GPU: it still crosses
        block-scale boundaries (2x2 scale blocks) and the fp8 -> bf16 dequant.
        The plugin shapes are a separate, memory-gated acceptance case below.
        """
        t = 8
        self._native_cast_gemv_parity(24, 256, 256, t, 4, t_cap=t)

    def test_native_cast_gemv_matches_reference_at_plugin_shapes(self):
        """The same parity at the real MoE shapes (E=288, N=K=4096).

        This one needs a large-memory card to itself: the oracle builds fp32
        [E, O, I] weights (~19 GiB here) before casting to fp8, and the native
        kernel dequantizes the whole stack through fp32 temporaries of its own
        (~4x the fp32 stack in total).

        Two measured traps behind the guard below, both on clariden GH200:
        (a) ``torch.cuda.mem_get_info`` reports *unified* free memory there, so
        it can say 85 GiB while the device allocator has 4 GiB left (gate
        3456232); (b) on a node whose card a co-tenant job shares, the 19 GiB
        requests fail outright (gates 3453378/3455389/3454794). So: pre-check
        against the device-visible free memory, and if the allocation still
        fails, skip rather than red -- this is an acceptance case, and a solo
        large-memory GPU is the precondition it cannot verify itself.
        """
        e, o, i, t, kk = 288, 4096, 4096, 8, 8
        fp32_bytes = e * o * i * 4
        torch.cuda.empty_cache()
        total = torch.cuda.get_device_properties(0).total_memory
        used = torch.cuda.memory_allocated() + torch.cuda.memory_reserved()
        driver_free, _ = torch.cuda.mem_get_info()
        # GH200 unified memory: mem_get_info can include host pages, so take the
        # stricter of the driver's view and the device allocator's headroom.
        free_bytes = min(driver_free, max(0, total - used))
        need_bytes = 4 * fp32_bytes + 8 * 2**30
        if free_bytes < need_bytes:
            pytest.skip(
                f"needs ~{need_bytes / 2**30:.0f} GiB of device memory free for "
                f"the fp32 oracle weights + the kernel's own fp32 temporaries "
                f"+ workspace, have {free_bytes / 2**30:.1f} GiB of "
                f"{total / 2**30:.1f} GiB"
            )
        try:
            self._native_cast_gemv_parity(e, o, i, t, kk, t_cap=t)
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            pytest.skip(f"plugin-shape parity needs a solo large-memory GPU: {exc}")

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


def _oracle_e4m3nv(quantize, x, gu, gus, dn, dns, idx, w, swiglu_limit=7.0):
    """Per-expert fp32 matmuls on the SAME fp8 operands the kernel reads.

    Isolates the grouped tile map from fp8 noise: the only differences left
    between kernel and oracle are tile-map bugs and the kernel's own rounding
    points (bf16 gu before the swiglu, bf16 down output), which are mirrored
    here exactly as the runner's swiglu has them.
    """
    import torch.nn.functional as F

    t, k_route = idx.shape
    h, i = x.shape[1], gu.shape[1] // 2

    def deq_a(p, s):
        g = s.shape[-1]
        return (p.float().view(-1, g, 128) * s.float()[:, :, None]).view(-1, g * 128)

    def deq_w(p, s):
        g0, g1 = s.shape
        return (
            p.float().view(g0, 128, g1, 128) * s.float()[:, None, :, None]
        ).view(g0 * 128, g1 * 128)

    slots = idx.reshape(-1)
    toks = torch.arange(t, device=x.device).repeat_interleave(k_route)
    order = torch.argsort(slots, stable=True)
    sorted_tok, sorted_exp = toks[order], slots[order]
    flat_w = w.reshape(-1)[order]

    a_n, asc = quantize(x)
    gu_v = torch.empty(slots.numel(), 2 * i, device=x.device, dtype=torch.float32)
    for exp in sorted_exp.unique().tolist():
        m = sorted_exp == exp
        gu_v[m] = deq_a(a_n[sorted_tok[m]], asc[sorted_tok[m]]) @ deq_w(gu[exp], gus[exp]).t()
    gu_v = gu_v.to(torch.bfloat16).float()
    gate = gu_v[:, :i].clamp(max=swiglu_limit)
    up = gu_v[:, i:].clamp(min=-swiglu_limit, max=swiglu_limit)
    act = (F.silu(gate) * up).to(torch.bfloat16)
    a2_n, a2sc = quantize(act)
    ref = torch.zeros(t, h, device=x.device, dtype=torch.float32)
    for exp in sorted_exp.unique().tolist():
        m = sorted_exp == exp
        dn_out = (deq_a(a2_n[m], a2sc[m]) @ deq_w(dn[exp], dns[exp]).t()).to(torch.bfloat16).float()
        ref.index_add_(0, sorted_tok[m], dn_out * flat_w[m].unsqueeze(-1))
    return ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestGroupedNativeE4M3NV:
    """The NVIDIA flavour of the grouped native-fp8 MoE.

    CDNA3 has no e4m3fn matrix unit, so the ROCm path rewrites every expert
    stack to ``e4m3fnuz`` (halved payloads, doubled scales). NVIDIA tensor
    cores *are* e4m3fn, so the checkpoint's bytes and scales are used as they
    are. The kernel is flavour-agnostic; what must not be mixed is the payload
    bytes with the wrong scale convention, and the e4m3fn path has to reach the
    same per-expert result as an oracle run on the *same* operands.
    """

    def _inputs(self, t, e, k_route, h, i, seed=7):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(t, h, generator=gen, device="cuda", dtype=torch.bfloat16)
        gu = (
            (torch.randn(e, 2 * i, h, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        gus = (
            torch.rand(e, 2 * i // 128, h // 128, generator=gen, device="cuda") + 0.5
        ).contiguous()
        dn = (
            (torch.randn(e, h, i, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        dns = (
            torch.rand(e, h // 128, i // 128, generator=gen, device="cuda") + 0.5
        ).contiguous()
        idx = torch.randint(0, e, (t, k_route), generator=gen, device="cuda", dtype=torch.int64)
        w = torch.softmax(torch.randn(t, k_route, generator=gen, device="cuda"), -1)
        return x, gu, gus, dn, dns, idx, w

    def test_e4m3fn_weights_match_per_expert_oracle(self):
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            glm_moe_grouped_gemm_native,
            quantize_activations_native as q,
        )

        x, gu, gus, dn, dns, idx, w = self._inputs(256, 8, 4, 256, 256)
        # No e4m3fn_to_fnuz call: the checkpoint storage goes straight in.
        out = glm_moe_grouped_gemm_native(x, gu, gus, dn, dns, idx, w)
        ref = _oracle_e4m3nv(q, x, gu, gus, dn, dns, idx, w)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    def test_hot_experts_span_row_tiles(self):
        """Regression for #58, in the flavour NVIDIA actually runs.

        Experts with count > BM occupy several row tiles; the tile map used to
        repeat the segment start across an expert's tiles, leaving rows beyond
        the first BM unwritten (torch.empty garbage -> NaNs under a dirty
        allocator). The fnuz test covers this only on ROCm.
        """
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            glm_moe_grouped_gemm_native,
            quantize_activations_native as q,
        )

        t, e, k_route = 256, 64, 8  # 2048 slots / 64 experts, BM=64
        x, gu, gus, dn, dns, idx, w = self._inputs(t, e, k_route, 256, 256)
        idx = idx.clone()
        idx[:, :3], idx[:, 3:5] = 0, 1  # experts 0/1 take ~4x the average
        out = glm_moe_grouped_gemm_native(x, gu, gus, dn, dns, idx, w)
        assert torch.isfinite(out).all(), "multi-tile experts left rows unwritten"
        ref = _oracle_e4m3nv(q, x, gu, gus, dn, dns, idx, w)
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    def test_flavour_selector_keeps_bytes_and_scales_consistent(self):
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            _activation_quantizer,
            quantize_activations_fnuz,
            quantize_activations_native,
        )

        assert _activation_quantizer(torch.float8_e4m3fn) is quantize_activations_native
        assert _activation_quantizer(torch.float8_e4m3fnuz) is quantize_activations_fnuz
        with pytest.raises(TypeError, match="e4m3fn or e4m3fnuz"):
            _activation_quantizer(torch.bfloat16)

    def test_e4m3nv_is_at_least_as_accurate_as_fnuz(self):
        """e4m3fn reaches 448, e4m3fnuz only 240 at the same bit width.

        Not a correctness requirement, but it is why the CUDA path cannot be
        *worse* than the fnuz one on activations beyond fnuz's amax: the
        halve/double trick is exact only inside fnuz's representable range.
        """
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            quantize_activations_fnuz,
            quantize_activations_native,
        )

        gen = torch.Generator(device="cuda").manual_seed(3)
        x = torch.cat(
            [
                torch.randn(4, 128, generator=gen, device="cuda", dtype=torch.bfloat16),
                torch.full((4, 128), 13.0, device="cuda", dtype=torch.bfloat16),
            ],
            dim=0,
        )
        n_q, n_s = quantize_activations_native(x)
        f_q, f_s = quantize_activations_fnuz(x)

        def err(qq, ss):
            g = ss.shape[-1]
            deq = (qq.float().view(-1, g, 128) * ss.float()[:, :, None]).view(-1, g * 128)
            return (deq.float() - x.float()).abs().max().item()

        assert n_q.dtype == torch.float8_e4m3fn and f_q.dtype == torch.float8_e4m3fnuz
        assert err(n_q, n_s) <= err(f_q, f_s) + 1e-3, (err(n_q, n_s), err(f_q, f_s))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestGroupedNativeStaticTileMap:
    """⑨b: the grouped dispatch must launch from a *static* grid.

    Verifying the sync removal needs evidence that any host read would break,
    not just that timings improved, so the primary test is a CUDA graph
    capture: capture fails loudly the moment the dispatch synchronises or
    allocates shape-dependent grids from device values. The same operands are
    then replayed against eager output, including a mutation of the input to
    prove the captured graph reads live data.
    """

    @staticmethod
    def _flavour():
        """(to_native, quantize) for the fp8 type this device's tensor cores take.

        CDNA3 has no e4m3fn matrix unit, so ROCm rewrites the expert stack to
        e4m3fnuz (halved payloads, doubled scales) and uses the matching
        activation quantizer; NVIDIA uses the checkpoint bytes as they are. The
        dispatch under test here is flavour-independent.
        """
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            e4m3fn_to_fnuz,
            quantize_activations_fnuz,
            quantize_activations_native,
        )

        if torch.version.hip is not None:
            return e4m3fn_to_fnuz, quantize_activations_fnuz
        return lambda p, s: (p, s), quantize_activations_native

    def _operands(self, t, e, k_route, h, i, idx_mode="uniform", seed=7, idx_hi=None):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(t, h, generator=gen, device="cuda", dtype=torch.bfloat16)
        gu = (
            (torch.randn(e, 2 * i, h, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        gus = (
            torch.rand(e, 2 * i // 128, h // 128, generator=gen, device="cuda") + 0.5
        ).contiguous()
        dn = (
            (torch.randn(e, h, i, generator=gen, device="cuda") * 0.05)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        dns = (torch.rand(e, h // 128, i // 128, generator=gen, device="cuda") + 0.5).contiguous()
        if idx_mode == "uniform":
            idx = torch.randint(
                0, idx_hi or e, (t, k_route), generator=gen, device="cuda", dtype=torch.int64
            )
        else:  # "skewed": hot experts span row tiles at BM=64
            idx = torch.randint(2, e, (t, k_route), generator=gen, device="cuda", dtype=torch.int64)
            idx[:, :3] = 0
            idx[:, 3:5] = 1
        w = torch.softmax(torch.randn(t, k_route, generator=gen, device="cuda"), -1)
        to_native, quantize = self._flavour()
        gu_n, gu_s = to_native(gu, gus)
        dn_n, dn_s = to_native(dn, dns)
        return x, gu_n, gu_s, dn_n, dn_s, idx, w, quantize

    def test_dispatch_is_cuda_graph_capturable(self):
        """A single host read left in the dispatch makes this fail."""
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import glm_moe_grouped_gemm_native

        args = self._operands(256, 32, 4, 256, 256, idx_mode="skewed")[:7]
        x = args[0]

        eager = glm_moe_grouped_gemm_native(*args)

        # Warm up on a side stream: Triton's first-call JIT compile may not
        # happen inside a capture, and the allocator needs its pool warm.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                glm_moe_grouped_gemm_native(*args)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = glm_moe_grouped_gemm_native(*args)
        graph.replay()

        # Capture must not change the dispatch's semantics. Not bitwise: two
        # same-input runs of this kernel land ~1 ulp apart where accumulated
        # terms cancel (observed on MI300A: 2.4e-7 absolute at one element of
        # 65k), so the check uses the same tolerance class as the golden test
        # above -- a wrong tile map shows up at ~1e-2, not at 1 ulp.
        torch.testing.assert_close(captured.float(), eager.float(), rtol=2e-2, atol=2e-2)

        # Replay must consume *live* inputs, not the values frozen at capture.
        # Halving x moves the output far more than the re-association noise, so
        # a replay that reuses captured operands would be caught here.
        x.mul_(0.5)
        mutated = glm_moe_grouped_gemm_native(*args)
        graph.replay()
        torch.testing.assert_close(captured.float(), mutated.float(), rtol=2e-2, atol=2e-2)
        drift = (captured.float() - eager.float()).abs().max().item()
        assert drift > 1e-2, f"replay looks frozen at capture-time inputs (max|delta|={drift})"

    def test_zero_count_experts_and_padding_tiles_are_inert(self):
        """E far above the routing breadth => many empty experts + padding tiles.

        With E=288 (the real GLM expert count) and short routing, most experts
        have count 0 and the static capacity ``slots // BM + E + 1`` is well
        above the real tile count. Empty experts must contribute nothing and
        the padding tiles (``m = 0``) must neither write rows nor poison the
        sums with uninitialised memory.
        """
        from vkernels.torch_ops.glm_fp8_blockwise_gemm import glm_moe_grouped_gemm_native

        x, gu, gus, dn, dns, idx, w, quantize = self._operands(
            64, 288, 2, 256, 256, idx_mode="uniform", idx_hi=8
        )
        assert int(idx.unique().numel()) <= 8  # 280 of 288 experts stay idle
        n_tiles = int(((torch.bincount(idx.reshape(-1), minlength=288) + 63) // 64).sum())
        cap = min(64 * 2, 64 * 2 // 64 + 288 + 1)  # same bound the dispatch uses
        assert cap - n_tiles > 64, (cap, n_tiles)  # lots of padding exercised
        out = glm_moe_grouped_gemm_native(x, gu, gus, dn, dns, idx, w)
        ref = _oracle_e4m3nv(quantize, x, gu, gus, dn, dns, idx, w)
        assert torch.isfinite(out.float()).all()
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)

    def test_static_tile_map_matches_the_device_built_one(self):
        """The rewrite must be an optimisation, not a semantic change.

        The old tile map (unique_consecutive + int(toff[-1])) was correct but
        syncing; rebuilding both and comparing field by field pins the new
        one to it without needing a second GPU implementation.
        """
        import torch as _t

        from vkernels.torch_ops.glm_fp8_blockwise_gemm import (
            _tile_map_static,
            glm_moe_grouped_gemm_native,  # noqa: F401  (import check)
        )

        t, e, k_route, bm = 256, 64, 8, 64
        gen = _t.Generator(device="cuda").manual_seed(11)
        idx = _t.randint(0, e, (t, k_route), generator=gen, device="cuda", dtype=_t.int64)
        idx[:, :3] = 0
        idx[:, 3:5] = 1

        counts = _t.bincount(idx.reshape(-1), minlength=e)
        seg = _t.cumsum(counts, 0) - counts
        tiles = (counts + bm - 1) // bm
        ends = _t.cumsum(tiles, 0)
        toff = ends - tiles
        n_tiles = int(ends[-1])

        # Old construction, exactly as the pre-⑨b code did it.
        local_old = _t.arange(n_tiles, device="cuda", dtype=_t.int64) - _t.repeat_interleave(
            toff, tiles, output_size=n_tiles
        )
        exp_old = _t.repeat_interleave(_t.arange(e, device="cuda"), tiles, output_size=n_tiles)
        r0_old = _t.repeat_interleave(seg, tiles, output_size=n_tiles) + local_old * bm
        m_old = _t.clamp(
            _t.repeat_interleave(counts, tiles, output_size=n_tiles) - local_old * bm, max=bm
        )

        cap = min(t * k_route, t * k_route // bm + e + 1)
        exp_new, r0_new, m_new = _tile_map_static(counts, seg, bm, "cuda", cap)
        assert cap >= n_tiles, (cap, n_tiles)
        keep = m_new > 0
        assert bool(keep.sum() == n_tiles)
        assert _t.equal(exp_new[keep], exp_old)
        assert _t.equal(r0_new[keep], r0_old)
        assert _t.equal(m_new[keep], m_old)
        assert bool((m_new[~keep] == 0).all()) and bool((r0_new[~keep] == 0).all())


def test_triton_fallback_warns_once(monkeypatch, capsys):
    """The triton-blockwise -> torch-oracle fallback keeps its result contract
    and logs the first failure once per process (no silent downgrade)."""
    import threading
    import types

    from vkernels.torch_ops import glm_fp8_blockwise_gemm as m

    monkeypatch.setattr(m, "_TRITON_FALLBACK_WARNED", threading.Event())
    monkeypatch.setattr(m, "_CUTE_TRIED", True)  # skip the CuTe kernel probe
    monkeypatch.setattr(m, "_CUTE_KERNEL", None)
    monkeypatch.setattr(
        m, "_triton_backend", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    sentinel = object()
    monkeypatch.setattr(m, "_torch_blockwise_gemm", lambda *a, **k: sentinel)

    fake = types.SimpleNamespace(
        is_cuda=True, device=torch.device("cpu"), dtype=torch.float8_e4m3fn, shape=(128, 128)
    )
    scale = types.SimpleNamespace(shape=(1, 1))
    out = m.fp8_blockwise_gemm(fake, scale, fake, scale)
    assert out is sentinel  # fallback result contract unchanged
    first = capsys.readouterr().out
    assert "[glm_fp8_blockwise_gemm]" in first
    assert "unavailable or failed" in first and "torch reference path active" in first
    # second failure: same fallback result, no second warning (one-shot)
    assert m.fp8_blockwise_gemm(fake, scale, fake, scale) is sentinel
    assert "[glm_fp8_blockwise_gemm]" not in capsys.readouterr().out
