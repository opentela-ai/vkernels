"""Shared-latent gathered attention, including learned gate derivatives."""

import pytest
import torch

from vkernels.torch_ops.indexed_mla_train import (
    indexed_mla_backward,
    indexed_mla_forward,
    indexed_mla_reference,
)


def inputs(device="cpu", dtype=torch.float64, seq=5, heads=2, rank=16, rope=8, count=3, batch=1):
    torch.manual_seed(17)
    q = torch.randn(batch, seq, heads, rank, device=device, dtype=dtype, requires_grad=True)
    qr = torch.randn(batch, seq, heads, rope, device=device, dtype=dtype, requires_grad=True)
    kv = torch.randn(batch, seq, rank, device=device, dtype=dtype, requires_grad=True)
    kr = torch.randn(batch, seq, rope, device=device, dtype=dtype, requires_grad=True)
    ix = torch.randint(seq, (batch, seq, count), device=device)
    ix[..., -1] = -1
    gate = torch.randn(batch, seq, count, device=device, dtype=dtype, requires_grad=True)
    return q, qr, kv, kr, ix, gate


def test_reference_gradcheck_and_invalid_keys():
    q, qr, kv, kr, ix, gate = inputs()
    fn = lambda q, qr, kv, kr, gate: indexed_mla_reference(q, qr, kv, kr, ix, gate, 0.2)
    assert torch.autograd.gradcheck(fn, (q, qr, kv, kr, gate), fast_mode=True)
    out = fn(q, qr, kv, kr, gate)
    grad = torch.autograd.grad(out.sum(), gate)[0]
    assert grad[..., -1].count_nonzero() == 0
    assert grad[..., :-1].abs().sum() > 0


def test_reference_all_invalid_row_is_zero_with_finite_gradients():
    q, qr, kv, kr, ix, gate = inputs(rope=0)
    ix[:, 1] = -1
    out = indexed_mla_reference(q, qr, kv, kr, ix, gate, 0.2)
    assert out[:, 1].count_nonzero() == 0
    grads = torch.autograd.grad(out.square().sum(), (q, qr, kv, kr, gate), allow_unused=True, materialize_grads=True)
    assert all(torch.isfinite(g).all() for g in grads)
    assert grads[-1][:, 1].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU Triton")
@pytest.mark.parametrize(
    "heads,rank,rope,seq,count,batch",
    [(4, 64, 64, 64, 17, 1), (16, 512, 64, 128, 49, 1), (4, 32, 0, 32, 11, 1), (4, 64, 32, 257, 17, 2)],
)
def test_device_forward_and_all_five_gradients(heads, rank, rope, seq, count, batch):
    q, qr, kv, kr, ix, gate = inputs("cuda", torch.bfloat16, seq, heads, rank, rope, count, batch)
    # Include an entirely empty row and an out-of-range positive index.
    ix[:, 1] = -1
    ix[:, 2, 0] = seq
    scale = (rank + max(rope, 1)) ** -0.5
    ref = indexed_mla_reference(q, qr, kv, kr, ix, gate, scale)
    dy = torch.randn_like(ref)
    expected = torch.autograd.grad(ref, (q, qr, kv, kr, gate), dy, allow_unused=True, materialize_grads=True)
    got, out32, lse = indexed_mla_forward(q, qr, kv, kr, ix, gate, scale)
    actual = indexed_mla_backward(dy, q, qr, kv, kr, ix, gate, out32, lse, scale)
    for want, have in [(ref, got), *zip(expected, actual)]:
        assert torch.isfinite(have).all()
        relative = (have.float() - want.float()).norm() / want.float().norm().clamp_min(1e-12)
        assert relative < 0.01
    assert actual[-1][..., -1].count_nonzero() == 0
