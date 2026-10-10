"""Tests for the vendored TokenSpeed Triton MLA kernels (torch_ops/tokenspeed_mla).

Host CI covers the import surface, the CPU fallback paths of the page-table
helpers, and the metadata registry. GPU parity against torch references is
marked ``gpu`` and runs only where a working Triton driver exists (this
repo's serving boxes), mirroring the vllm_sparse_mla test conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops import tokenspeed_mla as ts_mla  # noqa: E402
from vkernels.torch_ops.mla_registry import (  # noqa: E402
    MLA_DECODE_REGISTRY,
    MLA_PREFILL_REGISTRY,
)
from vkernels.registry import KernelRequest, TensorMetadata  # noqa: E402


# --- import surface -----------------------------------------------------------


def test_module_exposes_vendored_kernels():
    for name in (
        "_mla_prefill_kernel",
        "_mla_decode_kernel",
        "_group_slots_kernel",
        "_copy_page_table_kernel",
    ):
        kernel = getattr(ts_mla, name)
        assert hasattr(kernel, "run") or callable(kernel)  # triton JIT object


def test_no_tokenspeed_dependency():
    import sys

    assert not any(m == "tokenspeed_kernel" for m in sys.modules)


# --- page-table CPU paths -------------------------------------------------------


def _reference_slots(positions, requests, table, rows_per_page, entry_stride, first_page, page_count):
    out = np.full(positions.shape, -1, dtype=np.int64)
    for idx in np.ndindex(positions.shape):
        pos, req = int(positions[idx]), int(requests[idx])
        logical, column = pos // entry_stride, (pos // entry_stride) // rows_per_page
        if pos < 0 or req < 0 or req >= table.shape[0] or column >= table.shape[1]:
            continue
        page = int(table[req, column])
        if first_page <= page < page_count:
            out[idx] = page * rows_per_page + logical % rows_per_page
    return out


@pytest.mark.parametrize("shape", [(8,), (2, 6)])
def test_bounded_group_slots_cpu_matches_reference(shape):
    rng = np.random.default_rng(7)
    rows_per_page, entry_stride = 16, 1
    table = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    positions = rng.integers(-8, 300, size=shape).astype(np.int32)
    requests = rng.integers(-1, 3, size=shape).astype(np.int32)
    got = ts_mla.bounded_group_slots(
        torch.from_numpy(positions),
        torch.from_numpy(requests),
        torch.from_numpy(table),
        rows_per_page,
        entry_stride,
        first_page=0,
        page_count=8,
    )
    want = _reference_slots(positions, requests, table, rows_per_page, entry_stride, 0, 8)
    np.testing.assert_array_equal(got.numpy(), want)
    # out-of-range pages resolve to -1
    out_of_range = ts_mla.bounded_group_slots(
        torch.from_numpy(positions),
        torch.from_numpy(requests),
        torch.from_numpy(table),
        rows_per_page,
        entry_stride,
        first_page=0,
        page_count=4,  # pages 4..7 invalid
    )
    assert ((out_of_range.numpy() == -1) | (want < 4 * rows_per_page)).all()


def test_bounded_group_slots_input_validation():
    p = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(ValueError, match="equal 1-D/2-D"):
        ts_mla.bounded_group_slots(p, torch.zeros(3, dtype=torch.int32),
                                   torch.zeros((1, 1), dtype=torch.int32), 1, 1, 0, 1)
    with pytest.raises(ValueError, match="geometry"):
        ts_mla.bounded_group_slots(p, p, torch.zeros((1, 1), dtype=torch.int32), 0, 1, 0, 1)


def test_copy_page_table_cpu_clears_stale_columns():
    source = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    out = torch.full((5, 6), 99, dtype=torch.int32)
    ts_mla.copy_page_table(source, out, live_rows=2)
    want = torch.zeros((5, 6), dtype=torch.int32)
    want[:2, :4] = source[:2]
    assert torch.equal(out, want)
    with pytest.raises(ValueError, match="capacity"):
        ts_mla.copy_page_table(source, torch.zeros((3, 3), dtype=torch.int32), 3)


# --- metadata registry ----------------------------------------------------------


def _prefill_request(dtypes=("bfloat16", "bfloat16", "bfloat16")):
    return KernelRequest(
        "mla_prefill",
        (
            TensorMetadata((128, 16, 576), dtypes[0], "cuda:0"),
            TensorMetadata((96, 1, 576), dtypes[1], "cuda:0"),
            TensorMetadata((96, 1, 512), dtypes[2], "cuda:0"),
        ),
        backends=frozenset({"triton"}),
    )


def test_prefill_registry_selects_triton_and_gates_dtypes():
    assert MLA_PREFILL_REGISTRY.select(_prefill_request()).name == "triton"
    request = _prefill_request(dtypes=("float32", "bfloat16", "bfloat16"))
    with pytest.raises(Exception, match="q requires"):
        MLA_PREFILL_REGISTRY.select(request)


def test_prefill_kv_head_grouping_gate():
    request = KernelRequest(
        "mla_prefill",
        (
            TensorMetadata((128, 16, 576), "bfloat16", "cuda:0"),
            TensorMetadata((96, 3, 576), "bfloat16", "cuda:0"),  # 16 % 3 != 0
            TensorMetadata((96, 1, 512), "bfloat16", "cuda:0"),
        ),
        backends=frozenset({"triton"}),
    )
    with pytest.raises(Exception, match="divisible"):
        MLA_PREFILL_REGISTRY.select(request)


def test_decode_registry_shape_and_layout():
    ok = KernelRequest(
        "mla_decode",
        (
            TensorMetadata((4, 1, 8, 640), "bfloat16", "cuda:0"),
            TensorMetadata((16, 64, 1, 640), "bfloat16", "cuda:0"),
        ),
        backends=frozenset({"triton"}),
    )
    assert MLA_DECODE_REGISTRY.select(ok).name == "triton"
    bad_q = KernelRequest(
        "mla_decode",
        (
            TensorMetadata((4, 7, 8, 640), "bfloat16", "cuda:0"),  # q_len != 1
            TensorMetadata((16, 64, 1, 640), "bfloat16", "cuda:0"),
        ),
        backends=frozenset({"triton"}),
    )
    with pytest.raises(Exception, match="q_len == 1"):
        MLA_DECODE_REGISTRY.select(bad_q)


# --- GPU parity (skipped without a working triton driver) ------------------------


@pytest.mark.gpu
def test_prefill_matches_torch_reference_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        import triton

        triton.runtime.driver.active  # noqa: B018 — probes the runtime driver
    except Exception as exc:  # noqa: BLE001 — e.g. host lacks python headers for driver.c
        pytest.skip(f"triton driver unavailable: {exc}")
    torch.manual_seed(11)
    seqs = [(12, 12), (7, 7)]
    total_q = sum(q for q, _ in seqs)
    total_kv = sum(kv for _, kv in seqs)
    q = torch.randn(total_q, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total_kv, 1, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total_kv, 1, 96, device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 12, 19], device="cuda", dtype=torch.int32)
    cu_kv = torch.tensor([0, 12, 19], device="cuda", dtype=torch.int32)
    scale = 128 ** -0.5
    out = ts_mla.triton_mla_prefill(
        q, k, v, cu_q, cu_kv, 12, 12, scale, is_causal=True
    )
    # per-sequence torch reference
    want = torch.empty_like(out, dtype=torch.float32)
    start = 0
    for (q_len, kv_len) in seqs:
        qs = q[start:start + q_len].float()             # [q, heads, 128]
        ks = k[start:start + kv_len].float()[:, 0]       # [kv, 128]
        vs = v[start:start + kv_len].float()[:, 0]       # [kv, 96]
        logits = torch.einsum("qhd,kd->hqk", qs, ks) * scale
        pos_q = torch.arange(kv_len - q_len, kv_len, device="cuda")
        pos_k = torch.arange(kv_len, device="cuda")
        mask = pos_q[:, None] >= pos_k[None, :]
        logits = logits.masked_fill(~mask[None], float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        # MQA shares K/V, while each query head has its own attention scores.
        want[start:start + q_len] = torch.einsum("hqk,kd->qhd", probs, vs)
        start += q_len
    torch.testing.assert_close(out.float(), want.to(out.dtype).float(), rtol=2e-2, atol=2e-2)
