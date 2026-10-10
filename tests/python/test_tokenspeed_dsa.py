"""Tests for the vendored TokenSpeed Triton DSA kernels (torch_ops/tokenspeed_dsa)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops import tokenspeed_dsa as ts_dsa  # noqa: E402
from vkernels.torch_ops.dsa_registry import DSA_REGISTRY  # noqa: E402
from vkernels.registry import KernelRequest, TensorMetadata  # noqa: E402


def test_module_exposes_vendored_kernels():
    for name in ("_dsa_packed_kv_kernel", "_dsa_dense_kv_kernel"):
        assert hasattr(ts_dsa, name)
    for name in ("triton_dsa_decode", "triton_dsa_prefill"):
        assert callable(getattr(ts_dsa, name))
    import sys

    assert not any(m == "tokenspeed_kernel" for m in sys.modules)


def test_flatten_dense_kv_cache_variants():
    flat = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)
    assert ts_dsa.flatten_dense_kv_cache(flat) is flat

    three_d = flat.reshape(4, 1, 6)  # [pages, 1, dim] -> squeezed
    assert torch.equal(ts_dsa.flatten_dense_kv_cache(three_d), flat)

    four_d_single_head = flat.reshape(2, 2, 1, 6).permute(0, 2, 1, 3)  # [pages, 1, ps, dim]
    out = ts_dsa.flatten_dense_kv_cache(four_d_single_head)
    assert out.shape == (4, 6)
    assert out.is_contiguous()


def test_flatten_packed_kv_cache_variants():
    flat = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    assert ts_dsa.flatten_packed_kv_cache(flat) is flat
    nested = flat.reshape(2, 1, 8)
    assert ts_dsa.flatten_packed_kv_cache(nested).shape == (2, 8)


def _request(dtype="bfloat16"):
    return KernelRequest(
        "dsa_attention",
        (
            TensorMetadata((8, 4, 576), dtype, "cuda:0"),
            TensorMetadata((1024, 576), dtype, "cuda:0"),
        ),
        backends=frozenset({"triton"}),
    )


def test_registry_selects_and_gates():
    assert DSA_REGISTRY.select(_request()).name == "triton_dense_kv"
    with pytest.raises(Exception, match="q requires"):
        DSA_REGISTRY.select(_request(dtype="float32"))
    bad_shape = KernelRequest(
        "dsa_attention",
        (
            TensorMetadata((8, 4), "bfloat16", "cuda:0"),  # 2-D q
            TensorMetadata((1024, 576), "bfloat16", "cuda:0"),
        ),
        backends=frozenset({"triton"}),
    )
    with pytest.raises(Exception, match=r"\[tokens, H"):
        DSA_REGISTRY.select(bad_shape)


@pytest.mark.gpu
def test_dense_kv_matches_reference_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        import triton

        triton.runtime.driver.active  # noqa: B018
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"triton driver unavailable: {exc}")

    torch.manual_seed(5)
    tokens, heads, lora, rope = 3, 2, 128, 64
    kv_dim = lora + rope
    q = torch.randn(tokens, heads, kv_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(256, kv_dim, device="cuda", dtype=torch.bfloat16)
    topk = 64
    slots = torch.stack(
        [torch.randperm(256, device="cuda")[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    lens = torch.tensor([topk, topk - 8, 0], device="cuda", dtype=torch.int32)
    scale = kv_dim ** -0.5

    out = ts_dsa.triton_dsa_decode(
        q, kv, None, slots, lens, topk, lora, lora, rope, scale, page_size=64
    )

    qf, kvf = q.float(), kv.float()
    for t in range(tokens):
        valid = slots[t][slots[t] >= 0][: lens[t]]
        want = torch.zeros(heads, lora, device="cuda")
        if len(valid):
            sel = kvf[valid.long()]
            logits = qf[t] @ sel.T * scale          # [H, topk_valid]
            probs = torch.softmax(logits, dim=-1)
            # v part of each selected row is the first lora columns
            want = probs @ sel[:, :lora]
        torch.testing.assert_close(out[t].float(), want, rtol=2e-2, atol=2e-2)
