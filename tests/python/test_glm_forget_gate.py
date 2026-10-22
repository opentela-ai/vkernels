"""Batched gates must retain row ownership, BF16 rounding and graph inputs."""
import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.glm_forget_gate import (  # noqa: E402
    forget_gate,
    forget_gate_reference,
)


def inputs(batch, device, head_dim=128, heads=16, seq=1):
    generator = torch.Generator(device=device).manual_seed(781 + batch)
    x = torch.randn(batch, seq, 4096, generator=generator, device=device, dtype=torch.bfloat16) / 8
    fa = torch.randn(head_dim, 4096, generator=generator, device=device, dtype=torch.bfloat16) / 64
    fb = torch.randn(heads * head_dim, head_dim, generator=generator, device=device, dtype=torch.bfloat16) / 16
    dt = torch.randn(heads * head_dim, generator=generator, device=device)
    a = torch.randn(heads, generator=generator, device=device) / 4
    return x, fa, fb, dt, a


@pytest.mark.parametrize("lower", [-5.0, None])
def test_reference_preserves_batch_rows(lower):
    args = inputs(4, "cpu")
    result = forget_gate_reference(*args, lower)
    assert result.shape == (4, 1, 16, 128)
    for row in range(4):
        expected = forget_gate_reference(args[0][row:row + 1], *args[1:], lower)
        torch.testing.assert_close(result[row:row + 1], expected, rtol=0, atol=0)


def test_cpu_declines():
    with pytest.raises(OpNotEligible, match="GPU"):
        forget_gate(*inputs(4, "cpu"), -5.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 17])
@pytest.mark.parametrize("lower", [-5.0, None])
def test_batched_gate_matches_eager(batch, lower):
    args = inputs(batch, "cuda")
    result = forget_gate(*args, lower)
    expected = forget_gate_reference(*args, lower)
    # One BF16 ulp for the adopted projection reduction and libdevice math.
    torch.testing.assert_close(result, expected, rtol=0.008, atol=0.002)
    assert result.shape == (batch, 1, 16, 128)
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("seq", [2, 3, 8])
@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("lower", [-5.0, None])
def test_small_t_verify_block_matches_eager(seq, batch, lower):
    """The speculative-verify block's small-T shape (S rows flatten onto
    the batch axis; row-parallel math is unchanged): fused vs reference
    parity plus per-row ownership against the S == 1 path."""
    args = inputs(batch, "cuda", seq=seq)
    result = forget_gate(*args, lower)
    expected = forget_gate_reference(*args, lower)
    torch.testing.assert_close(result, expected, rtol=0.008, atol=0.002)
    assert result.shape == (batch, seq, 16, 128)
    assert torch.isfinite(result).all()
    single = inputs(1, "cuda", seq=1)
    # per-row equivalence: row (b, s) equals the S == 1 call on that row
    for b in range(batch):
        for s_i in range(seq):
            row = args[0][b, s_i:s_i + 1].unsqueeze(0).contiguous()
            one = forget_gate(row, *args[1:], lower)
            torch.testing.assert_close(result[b, s_i:s_i + 1], one[0], rtol=0.008, atol=0.002)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batch", [1, 4, 8])
def test_graph_replay_reads_changed_rows(batch):
    args = inputs(batch, "cuda")
    forget_gate(*args, -5.0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forget_gate(*args, -5.0)
    original = captured.clone()
    args[0].mul_(-2)
    expected = forget_gate(*args, -5.0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, expected)
    assert not torch.equal(captured, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_declines_prefill_and_non_power_of_two_head():
    args = inputs(4, "cuda")
    with pytest.raises(OpNotEligible, match="1 <= S <= 8"):
        forget_gate(args[0].expand(-1, 16, -1).contiguous(), *args[1:], -5.0)
    with pytest.raises(OpNotEligible, match="power-of-two"):
        forget_gate(*inputs(4, "cuda", head_dim=96), -5.0)
