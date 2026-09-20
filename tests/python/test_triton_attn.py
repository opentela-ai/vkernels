"""Paged decode-attention contract checks and optional GPU parity."""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.triton_attn; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def test_reference_uniform_scores_averages_values(torch):
    """Zero keys => uniform softmax => out is the mean of the first s values."""
    import torch

    from vkernels.torch_ops.triton_attn import decode_attention_reference

    B, n_q, n_kv, D = 2, 4, 2, 8
    max_total = 16
    q = torch.randn(B, n_q, D)
    kc = torch.zeros(max_total, n_kv, D)
    vc = torch.randn(max_total, n_kv, D)
    block_table = (
        torch.arange(max_total).reshape(1, max_total).repeat(B, 1).to(torch.int32)
    )
    seq_lens = torch.tensor([3, 7], dtype=torch.int32)
    out = decode_attention_reference(q, kc, vc, block_table, seq_lens)
    for b, s in enumerate((3, 7)):
        for hkv in range(n_kv):
            for g in range(n_q // n_kv):
                h = hkv * (n_q // n_kv) + g
                expected = vc[:s, hkv].mean(dim=0)
                assert torch.allclose(out[b, h], expected, atol=1e-5), (b, h)


def test_contract(torch):
    from vkernels.torch_ops.triton_attn import decode_attention

    q = torch.randn(1, 3, 16)  # n_q not divisible by n_kv=2
    kc = torch.zeros(4, 2, 16)
    vc = torch.zeros(4, 2, 16)
    with pytest.raises(ValueError, match="multiple of n_kv"):
        decode_attention(
            q,
            kc,
            vc,
            torch.zeros(1, 2, dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        )


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.triton_attn import (
        decode_attention,
        decode_attention_reference,
    )

    torch.manual_seed(15)
    B, n_q, n_kv, D, max_total = 2, 4, 2, 64, 64
    q = torch.randn(B, n_q, D, device="cuda", dtype=torch.bfloat16)
    kc = torch.randn(max_total, n_kv, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(max_total, n_kv, D, device="cuda", dtype=torch.bfloat16)
    perm = torch.randperm(max_total, device="cuda")
    block_table = perm[: max_total // 2].reshape(1, -1).repeat(B, 1).to(torch.int32)
    seq_lens = torch.tensor([5, 29], dtype=torch.int32, device="cuda")
    out = decode_attention(q, kc, vc, block_table, seq_lens)
    out_r = decode_attention_reference(q, kc, vc, block_table, seq_lens)
    torch.testing.assert_close(out.float(), out_r.float(), atol=1e-3, rtol=1e-3)


def test_decode_attention_split_matches_reference():
    pytest.importorskip("torch")
    pytest.importorskip("torch.cuda")
    import torch as th
    if not th.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.triton_attn import (
        decode_attention_reference,
        decode_attention_split,
    )

    th.manual_seed(0)
    dev = "cuda"
    for B, n_q, n_kv, D, T in ((4, 16, 8, 128, 600), (1, 8, 8, 128, 64), (2, 4, 2, 64, 2048)):
        q = th.randn(B, n_q, D, device=dev, dtype=th.bfloat16)
        kc = th.randn(4096, n_kv, D, device=dev, dtype=th.bfloat16)
        vc = th.randn(4096, n_kv, D, device=dev, dtype=th.bfloat16)
        bt = th.stack([th.randperm(4000, device=dev)[:T].to(th.int32) for _ in range(B)])
        sl = th.full((B,), T, device=dev, dtype=th.int32)
        ref = decode_attention_reference(q, kc, vc, bt, sl)
        out = decode_attention_split(q, kc, vc, bt, sl, max_len_hint=T)
        err = (out.float() - ref).abs().max().item()
        assert err < 0.02, f"split-vs-ref err {err} @ B={B} T={T}"
        # ragged lens
        sl2 = sl.clone()
        sl2[0] = T // 3
        ref2 = decode_attention_reference(q, kc, vc, bt, sl2)
        out2 = decode_attention_split(q, kc, vc, bt, sl2, max_len_hint=T)
        err2 = (out2.float() - ref2).abs().max().item()
        assert err2 < 0.02, f"split-vs-ref ragged err {err2} @ B={B} T={T}"


def test_fused_kv_store_parity():
    """decode_attention{,_split} with k_new/v_new must match the two-step
    store_kv-then-attend sequence bit-for-bit, and must not touch the
    scratch page."""
    pytest.importorskip("torch")
    pytest.importorskip("torch.cuda")
    import torch as th
    if not th.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.triton_attn import (
        decode_attention,
        decode_attention_split,
    )

    th.manual_seed(0)
    dev = "cuda"
    for B, n_q, n_kv, D, T in ((4, 16, 8, 128, 600), (2, 4, 2, 64, 77), (3, 8, 8, 64, 1300)):
        q = th.randn(B, n_q, D, device=dev, dtype=th.bfloat16)
        pool = T + 16  # per-batch disjoint slot ranges -> no cross-batch collision
        kc = th.randn(B * pool, n_kv, D, device=dev, dtype=th.bfloat16)
        vc = th.randn(B * pool, n_kv, D, device=dev, dtype=th.bfloat16)
        bt = th.stack(
            [th.randperm(pool - 16, device=dev)[:T].to(th.int32) + b * pool for b in range(B)]
        )
        scratch = B * pool - 1
        # last batch's new token lands on the scratch page: never written
        bt[-1, T - 1] = scratch
        sl = th.full((B,), T, device=dev, dtype=th.int32)
        kn = th.randn(B, n_kv, D, device=dev, dtype=th.bfloat16)
        vn = th.randn(B, n_kv, D, device=dev, dtype=th.bfloat16)
        kc0, vc0 = kc.clone(), vc.clone()
        kcs, vcs = kc.clone(), vc.clone()
        # reference: two-step (store then attend). The scratch batch's
        # attention math still uses the fresh k/v (the kernel substitutes
        # from registers; only the pool write is suppressed), so model the
        # substitution in kc0 too -- only b=-1 hits scratch, and scratch
        # lies in the unused tail no live batch reads.
        slots = th.gather(bt, 1, ((sl - 1)[:, None]).to(th.int64))[:, 0]  # [B]
        live = slots != scratch
        kc0[slots[live].long()] = kn[live]
        vc0[slots[live].long()] = vn[live]
        kc0[scratch] = kn[-1]
        vc0[scratch] = vn[-1]
        ref = decode_attention(q, kc0, vc0, bt, sl)
        ref_s = decode_attention_split(q, kc0, vc0, bt, sl, max_len_hint=T)
        # fused: store folded into the attention kernel
        out = decode_attention(q, kc, vc, bt, sl, k_new=kn, v_new=vn, scratch_slot=scratch)
        out_s = decode_attention_split(q, kc, vc, bt, sl, max_len_hint=T, k_new=kn, v_new=vn, scratch_slot=scratch)
        th.testing.assert_close(out, ref)
        th.testing.assert_close(out_s, ref_s)
        # pool state: unchanged everywhere except the written slots
        # (scratch untouched)
        keep = th.ones(kc.shape[0], dtype=th.bool, device=dev)
        written = slots[live].long()
        keep[written] = False
        th.testing.assert_close(kcs[keep], kc[keep])
        th.testing.assert_close(vcs[keep], vc[keep])
        th.testing.assert_close(kc[written], kn[live])
        th.testing.assert_close(vc[written], vn[live])


def test_decode_attention_gqa_matches_reference():
    """GQA-grouped kernel (one program per KV head, tl.dot) vs the eager
    oracle, uniform and ragged lengths."""
    pytest.importorskip("torch")
    pytest.importorskip("torch.cuda")
    import torch as th
    if not th.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.triton_attn import (
        decode_attention_gqa,
        decode_attention_reference,
    )

    th.manual_seed(0)
    dev = "cuda"
    # G=2 (Qwen3-0.6B shape), G=2 D=64, G=4 and G=8 (larger models: the
    # grouping win grows with G), plus a G=8 kv=1 shape (MQA-style)
    for B, n_q, n_kv, D, T in (
        (4, 16, 8, 128, 600),
        (1, 8, 8, 128, 64),
        (2, 4, 2, 64, 2048),
        (3, 8, 8, 64, 1300),
        (2, 32, 8, 128, 777),
        (2, 64, 8, 128, 300),
        (2, 64, 1, 128, 500),
    ):
        q = th.randn(B, n_q, D, device=dev, dtype=th.bfloat16)
        kc = th.randn(4096, n_kv, D, device=dev, dtype=th.bfloat16)
        vc = th.randn(4096, n_kv, D, device=dev, dtype=th.bfloat16)
        bt = th.stack([th.randperm(4000, device=dev)[:T].to(th.int32) for _ in range(B)])
        sl = th.full((B,), T, device=dev, dtype=th.int32)
        ref = decode_attention_reference(q, kc, vc, bt, sl)
        out = decode_attention_gqa(q, kc, vc, bt, sl)
        err = (out.float() - ref).abs().max().item()
        assert err < 0.02, f"gqa-vs-ref err {err} @ B={B} q={n_q} kv={n_kv} T={T}"
        # ragged lens
        sl2 = sl.clone()
        sl2[0] = T // 3
        ref2 = decode_attention_reference(q, kc, vc, bt, sl2)
        out2 = decode_attention_gqa(q, kc, vc, bt, sl2)
        err2 = (out2.float() - ref2).abs().max().item()
        assert err2 < 0.02, f"gqa-vs-ref ragged err {err2} @ B={B} T={T}"


def test_gqa_fused_kv_store_parity():
    """decode_attention_gqa with k_new/v_new matches the two-step
    store-then-attend sequence and never writes the scratch page."""
    pytest.importorskip("torch")
    pytest.importorskip("torch.cuda")
    import torch as th
    if not th.cuda.is_available():
        pytest.skip("CUDA required")
    from vkernels.torch_ops.triton_attn import (
        decode_attention,
        decode_attention_gqa,
    )

    th.manual_seed(0)
    dev = "cuda"
    for B, n_q, n_kv, D, T in ((4, 16, 8, 128, 600), (2, 4, 2, 64, 77), (3, 8, 8, 64, 1300), (2, 64, 8, 128, 300)):
        pool = T + 16
        kc = th.randn(B * pool, n_kv, D, device=dev, dtype=th.bfloat16)
        vc = th.randn(B * pool, n_kv, D, device=dev, dtype=th.bfloat16)
        bt = th.stack(
            [th.randperm(pool - 16, device=dev)[:T].to(th.int32) + b * pool for b in range(B)]
        )
        scratch = B * pool - 1
        bt[-1, T - 1] = scratch
        sl = th.full((B,), T, device=dev, dtype=th.int32)
        kn = th.randn(B, n_kv, D, device=dev, dtype=th.bfloat16)
        vn = th.randn(B, n_kv, D, device=dev, dtype=th.bfloat16)
        q = th.randn(B, n_q, D, device=dev, dtype=th.bfloat16)
        kc0, vc0 = kc.clone(), vc.clone()
        slots = th.gather(bt, 1, ((sl - 1)[:, None]).to(th.int64))[:, 0]
        live = slots != scratch
        kc0[slots[live].long()] = kn[live]
        vc0[slots[live].long()] = vn[live]
        kc0[scratch] = kn[-1]
        vc0[scratch] = vn[-1]
        ref = decode_attention(q, kc0, vc0, bt, sl)  # per-q-head two-step oracle
        kcs, vcs = kc.clone(), vc.clone()
        out = decode_attention_gqa(q, kc, vc, bt, sl, k_new=kn, v_new=vn, scratch_slot=scratch)
        err = (out.float() - ref.float()).abs().max().item()
        assert err < 0.02, f"gqa fused-vs-two-step err {err} @ B={B} T={T}"
        keep = th.ones(kc.shape[0], dtype=th.bool, device=dev)
        written = slots[live].long()
        keep[written] = False
        assert th.equal(kcs[keep], kc[keep]) and th.equal(vcs[keep], vc[keep])
        assert th.equal(kc[written], kn[live]) and th.equal(vc[written], vn[live])
        assert th.equal(kcs[scratch], kc[scratch]) and th.equal(vcs[scratch], vc[scratch])
