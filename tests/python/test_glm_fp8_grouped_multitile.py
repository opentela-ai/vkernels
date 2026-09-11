"""Regression test for the grouped native-fp8 MoE tile map (#58).

An expert whose routed count exceeds the row-tile size (BM=64) spans
several row tiles. The tile map used to repeat the expert's segment
start/count across ALL of its tiles, so every tile rewrote the expert's
first 64 rows and rows beyond the first 64 were left UNWRITTEN — the
torch.empty() garbage surfaced as NaNs under a dirty allocator (the
random-prompt NLL = NaN in the #58 real-checkpoint A/B, 5 hot experts
affected at T>=256, non-deterministic across allocator states).

The test forces multi-tile experts via skewed routing and compares the
grouped native path against a per-expert fp32 oracle on the same fnuz
operands (with the runner's silu-form swiglu and bf16 intermediate
rounding, mirroring floe arch._swiglu / glm_moe_grouped_gemm_native).
"""

import pytest

torch = pytest.importorskip(
    "torch", reason="torch paths are lazily imported in production; tests need it"
)

from vkernels.torch_ops.glm_fp8_blockwise_gemm import (  # noqa: E402
    e4m3fn_to_fnuz,
    glm_moe_grouped_gemm_native,
    quantize_activations_fnuz,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
class TestGroupedNativeMultiTile:
    def test_hot_experts_match_fnuz_oracle(self):
        import torch.nn.functional as F

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
        # arch._swiglu). glm_moe_grouped_gemm is NOT a valid reference here
        # (its API-stable wrapper uses a sigmoid-form swiglu).
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
        for e0 in sorted_exp.unique().tolist():
            m = sorted_exp == e0
            av = deq_a(a_nz[sorted_tok[m]], asc2[sorted_tok[m]])
            bv = deq_w(gu_nz[e0], gu_sc2[e0])
            gu_v[m] = av @ bv.t()
        # the native path stores gu as bf16 BEFORE the swiglu; matching that
        # rounding is required or act re-quantization flips fp8 payload ULPs
        gu_v = gu_v.to(torch.bfloat16).float()
        gate = gu_v[:, :i].clamp(max=7.0)
        up = gu_v[:, i:].clamp(min=-7.0, max=7.0)
        act = (F.silu(gate) * up).to(torch.bfloat16)
        a2_nz, a2_sc2 = quantize_activations_fnuz(act)
        ref = torch.zeros(t, h, device="cuda", dtype=torch.float32)
        for e0 in sorted_exp.unique().tolist():
            m = sorted_exp == e0
            av2 = deq_a(a2_nz[m], a2_sc2[m])
            dv = deq_w(dn_nz[e0], dn_sc2[e0])
            dn_out = (av2 @ dv.t()).to(torch.bfloat16).float()
            ref.index_add_(0, sorted_tok[m], dn_out * flat_w[m].unsqueeze(-1))
        torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)
