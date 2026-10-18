"""Tuning the KDA triangular solve must preserve its FP32 reduction layout."""

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")


@pytest.mark.parametrize("batch,tokens,heads,seed", [(2, 64, 3, 71), (1, 1024, 16, 83)])
def test_every_solve_configuration_preserves_bits_and_inverse(monkeypatch, batch, tokens, heads, seed):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("NVIDIA GPU required")
    from vkernels.torch_ops import vllm_kda
    from vkernels.torch_ops.tuning_cache import device_fingerprint

    generator = torch.Generator().manual_seed(seed)
    lower = torch.randn(batch, tokens // 64, heads, 64, 64, generator=generator).tril(-1) * .01
    identity = torch.eye(64, dtype=torch.float64).expand_as(lower)
    oracle = torch.linalg.solve_triangular(identity + lower.double(), identity, upper=False)
    operand = lower.permute(0, 1, 3, 2, 4).reshape(batch, tokens, heads, 64).contiguous().cuda()
    original_operand = operand.clone()
    tuner = vllm_kda.merge_16x16_to_64x64_inverse_kernel.fn
    device = json.dumps(device_fingerprint(operand.device), sort_keys=True)
    monkeypatch.setenv("VKERNELS_TUNING_CACHE", "off")
    monkeypatch.setattr(tuner, "_stores", {})
    monkeypatch.setattr(tuner, "cache", {})
    baseline = None
    for selected in tuner.configs:
        # Seed the actual selection table, then use the original public solve
        # path; no benchmark noise or kernel reimplementation enters the test.
        monkeypatch.setattr(tuner, "_memory_configs", {device: {(heads, 64, False): selected}})
        for _ in range(2):
            actual = vllm_kda.solve_tril(operand)
            assert tuner.cache[(heads, 64, False)] is selected
            assert torch.equal(operand.view(torch.int32), original_operand.view(torch.int32))
            if baseline is None:
                baseline = actual.clone()
            else:
                assert torch.equal(actual.view(torch.int32), baseline.view(torch.int32))
            matrices = actual.cpu().reshape(batch, tokens // 64, 64, heads, 64).permute(0, 1, 3, 2, 4)
            torch.testing.assert_close(matrices.double(), oracle, atol=1e-6, rtol=1e-4)
