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
