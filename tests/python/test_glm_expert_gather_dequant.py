"""Gather-dequant contract checks and optional GPU parity."""

import subprocess
import sys

import pytest


def test_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import vkernels.torch_ops.glm_expert_gather_dequant; "
            "assert 'torch' not in sys.modules; "
            "assert 'triton' not in sys.modules",
        ],
        check=True,
    )


@pytest.fixture
def torch():
    return pytest.importorskip("torch")


def make_inputs(torch, e=8, o=256, i=256, t=2, k=3, seed=11):
    torch.manual_seed(seed)
    raw = torch.randint(0, 256, (e, o, i), dtype=torch.uint8)
    raw.masked_fill_((raw & 127) == 127, 0)  # drop NaN encodings
    weights = raw.view(torch.float8_e4m3fn)
    scales = 0.0005 + torch.rand((e, o // 128, i // 128)) * 0.001
    indices = torch.rand((t, e)).topk(k, dim=-1).indices.to(torch.int64)
    return weights, scales, indices


def test_reference_matches_native_fp8_decode(torch):
    """The bit-decode oracle must agree with torch's native E4M3FN cast."""
    from vkernels.torch_ops.glm_expert_gather_dequant import gather_dequant_reference

    weights, scales, indices = make_inputs(torch)
    expand = scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
    expected = (weights[indices].to(torch.float32) * expand[indices]).to(torch.bfloat16)
    actual = gather_dequant_reference(weights, scales, indices)
    assert actual.shape == (2, 3, 256, 256)
    torch.testing.assert_close(actual.float(), expected.float())


def test_contract(torch):
    from vkernels.torch_ops.glm_expert_gather_dequant import gather_dequant

    weights, scales, indices = make_inputs(torch, t=1, k=1)
    with pytest.raises(ValueError, match="GPU"):
        gather_dequant(weights, scales, indices)  # CPU tensors rejected
    with pytest.raises(ValueError, match="multiples of 128"):
        gather_dequant(weights[:, :-1, :], scales, indices)
    with pytest.raises(ValueError, match="expected scales"):
        gather_dequant(weights, scales[:, :, :-1], indices)
    with pytest.raises(TypeError, match="E4M3FN"):
        gather_dequant(weights.float(), scales, indices)


def test_gpu_parity_vs_reference(torch):
    if not torch.cuda.is_available():
        pytest.skip("requires GPU")
    pytest.importorskip("triton")
    from vkernels.torch_ops.glm_expert_gather_dequant import (
        gather_dequant,
        gather_dequant_reference,
    )

    weights, scales, indices = make_inputs(torch, t=2, k=8)
    weights, scales, indices = weights.to("cuda"), scales.to("cuda"), indices.to("cuda")
    actual = gather_dequant(weights, scales, indices)
    expected = gather_dequant_reference(weights, scales, indices)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), expected.float())
