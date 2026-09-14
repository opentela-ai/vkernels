"""Checks for the FP64-direct BF16 oracle and argmax evidence (#67)."""

import math

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.bf16_oracle import (  # noqa: E402
    _fraction_rne_bf16, argmax_evidence, bf16_round_fp64, qkv_oracle_fp64)


def _fraction_reference(x):
    return torch.tensor([_fraction_rne_bf16(v) for v in x.tolist()],
                        dtype=torch.float64)


def _same(a, b):
    return a == b or (math.isnan(a) and math.isnan(b))


def test_matches_fraction_round_to_nearest_even():
    """Ground truth: RNE of the exact float64 value, via Fraction arithmetic."""
    generator = torch.Generator().manual_seed(7)
    values = [0.0, -0.0, float("inf"), float("-inf"), float("nan"),
              2.0 ** -126, 2.0 ** -127, 2.0 ** -128, 2.0 ** -133, 2.0 ** -134,
              1.5 * 2.0 ** -133, 2.0 ** -133 * (1 + 2.0 ** -8),
              2.0 ** -133 * 127.5, 2.0 ** -133 * 126.5,
              2.0 ** 128, (2 - 2.0 ** -7) * 2.0 ** 127, 2.0 ** 127,
              5e-324, 2.5e-323, 1.0, -1.0, 2.0 ** -7, 1.0 + 2.0 ** -8]
    values += (torch.rand(2048, generator=generator)
               * 2.0 ** -torch.randint(1, 300, (2048,), generator=generator)).tolist()
    values += (torch.randn(2048, generator=generator) * 10 ** 6).tolist()
    x = torch.tensor(values, dtype=torch.float64)
    oracle, reference = bf16_round_fp64(x), _fraction_reference(x)
    for index in range(len(values)):
        assert _same(oracle[index].item(), reference[index].item()), \
            f"value {values[index]!r}: oracle {oracle[index].item()!r} != {reference[index].item()!r}"


def test_double_rounding_hazard_is_real_and_avoided():
    """The FP64->FP32->BF16 hop rounds a just-below-midpoint value up across
    the midpoint; the direct oracle rounds down. This is the hazard #67 names,
    and why quality comparisons must round from FP64 directly."""
    below_midpoint = 1.0 + 3 * 2.0 ** -8 - 2.0 ** -24  # odd multiple of 2**-24
    x = torch.tensor([below_midpoint], dtype=torch.float64)
    direct = bf16_round_fp64(x).item()
    via_fp32 = x.to(torch.float32).to(torch.bfloat16).item()
    assert direct == 1.0 + 2.0 ** -7           # rounds down, correctly
    assert via_fp32 == 1.0 + 2 * 2.0 ** -7     # double-rounded two steps up
    assert direct != via_fp32

    found = 0
    for k in range(1, 400):
        witness = 1.0 + (2 * k + 1) * 2.0 ** -8 - 2.0 ** -24
        tensor = torch.tensor([witness], dtype=torch.float64)
        assert bf16_round_fp64(tensor).item() == _fraction_rne_bf16(witness)
        found += tensor.to(torch.float32).to(torch.bfloat16).item() != \
            bf16_round_fp64(tensor).item()
    assert found > 0


def test_requires_float64_input():
    with pytest.raises(TypeError, match="float64"):
        bf16_round_fp64(torch.ones(2, dtype=torch.float32))


def test_qkv_oracle_shape_and_dtype():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 4096, dtype=torch.bfloat16)
    weights = [torch.randn(8192, 4096, dtype=torch.bfloat16) * 0.01 for _ in range(3)]
    out = qkv_oracle_fp64(x, weights)
    assert out.dtype == torch.bfloat16 and out.shape == (1, 1, 24576)
    # The oracle must see exactly the caller's values: BF16 -> FP64 is exact,
    # so an all-equal input row and weights give that value times 4096.
    v = 0.125
    x0 = torch.full((1, 1, 4096), v, dtype=torch.bfloat16)
    w0 = torch.full((8192, 4096), v, dtype=torch.bfloat16)
    exact = bf16_round_fp64(torch.tensor([[v * v * 4096]], dtype=torch.float64)).item()
    assert qkv_oracle_fp64(x0, [w0, w0, w0])[0, 0, 0].item() == exact


def test_argmax_evidence_records_ids_and_ties():
    reference = torch.tensor([[1.0, 2.0, 2.0],    # exact tie -> margin 0
                              [3.0, 1.0, 0.0],
                              [0.0, 0.0, 5.0]])
    candidate = torch.tensor([[1.0, 2.0, 2.0],    # same values, same first-max id
                              [0.0, 3.5, 0.0],    # changed id 0 -> 1, margin 2.5
                              [0.0, 0.0, 5.0]])
    evidence = argmax_evidence(reference, candidate)
    assert evidence["rows"] == 3
    assert evidence["changed_count"] == 1
    assert evidence["agreement"] == pytest.approx(2 / 3)
    row = evidence["changed"][0]
    assert row["index"] == [1]
    assert row["reference_argmax_id"] == 0 and row["candidate_argmax_id"] == 1
    assert row["reference_margin"] == 2.0 and row["candidate_margin"] == 3.5
    assert row["reference_tie"] is False

    tied = argmax_evidence(torch.tensor([[2.0, 2.0]]), torch.tensor([[2.0, 2.0]]))
    assert tied["changed_count"] == 0  # tie recorded via margins, not order flips
    changed_tie = argmax_evidence(torch.tensor([[2.0, 2.0], [1.0, 0.0]]),
                                  torch.tensor([[2.0, 2.0], [0.0, 1.0]]))
    assert changed_tie["changed"][0]["reference_tie"] is False
    assert changed_tie["changed"][0]["reference_margin"] == 1.0
