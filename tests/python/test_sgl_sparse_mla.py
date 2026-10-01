"""SGLang v0.5.20 DSA sparse-fwd donor wrapper (lane 32).

Pins the ``flash_mla_sparse_fwd`` boundary across all three backends:

* **torch oracle** — every device; the parity authority the other
  backends are gated against (fp32 math, exact -1 / OOB / topk_length
  masking semantics);
* **triton backend** (GPU) — the vendored vLLM kernels under the pinned
  op-config-cache configs; parity vs the oracle at the bf16 rounding
  class, store replay costs zero benching, and a CUDA-graph capture
  replays bit-identical to its eager recording after :func:`warmup`;
* **sgl_kernel backend** — the donor native kernel when the wheel is
  importable AND its probe passes on this device (parity vs the oracle);
  otherwise the explicit-backend request must fail LOUD, and the auto
  resolution must have skipped it (no silent backend swap).

The dense-prefix contract (padding at the row TAIL — floe's `-1`
convention) is pinned too: the ragged builder packs the dense prefix, so
interior padding is a caller-contract violation, not a kernel bug.
"""

from __future__ import annotations

import json

import pytest
import torch

from vkernels.torch_ops import sgl_sparse_mla as S

CUDA = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def _clean_memos():
    """No test inherits another's resolution memo (tuning-cache hygiene,
    mirroring the kernel-tier tests' convention)."""
    S._DECODE_CFG.clear()
    S._PREFILL_CFG.clear()
    try:
        from vkernels.tuning import cache as tcache

        tcache.reset_memo()
    except Exception:
        pass
    yield
    S._DECODE_CFG.clear()
    S._PREFILL_CFG.clear()


def _case(seed=0, rows=4096, w=2051, heads=8, dim=512, pad=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, dim, generator=g).to(device=device, dtype=torch.bfloat16)
    kv = torch.randn(rows, 1, dim, generator=g).to(device=device, dtype=torch.bfloat16)
    perm = torch.randperm(rows, generator=g)
    idx = perm[:w].to(torch.int32).view(1, 1, w).to(device)
    if pad:
        # tail padding: the dense-prefix contract
        idx[..., -pad:] = -1
    return q, kv, idx


# ---------------------------------------------------------------------------
# backend resolution + validation (CPU-safe)
# ---------------------------------------------------------------------------


def test_torch_backend_is_the_cpu_resolution():
    assert S.active_backend(torch.device("cpu")) == "torch"
    out, ml, lse = S.flash_mla_sparse_fwd(*_case(), sm_scale=0.04)
    assert out.shape == (1, 8, 512) and out.dtype == torch.bfloat16
    assert ml.numel() == 0 and lse.numel() == 0  # stats off by default


def test_backend_names_are_pinned():
    q, kv, idx = _case()
    S.flash_mla_sparse_fwd(q, kv, idx, sm_scale=0.04, backend="torch")
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="warp")
    if not S._sgl_kernel_available():
        # explicit donor request on a box without the wheel: fail LOUD,
        # never silently swap backends
        with pytest.raises(RuntimeError):
            S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="sgl_kernel")


def test_validation_rejects_contract_violations():
    q, kv, idx = _case()
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(q, kv, idx.to(torch.int64), sm_scale=0.04, backend="torch")
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(q, kv, idx.expand(2, -1, -1).contiguous(), sm_scale=0.04,
                               backend="torch")  # indices rows != s_q
    bad_dv = q.clone()
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(bad_dv, kv, idx, 0.04, d_v=513, backend="torch")
    tl = torch.zeros(3, dtype=torch.int32)  # wrong length
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(q, kv, idx, 0.04, topk_length=tl, backend="torch")
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(q, kv, idx, 0.04, d_v=513, backend="torch")


def test_shape_class_tiers():
    assert S.decode_shape_class(1, 2051) == "t1.topk3k"
    assert S.decode_shape_class(8, 2051) == "t8.topk3k"
    assert S.decode_shape_class(64, 2051) == "t64.topk3k"
    assert S.decode_shape_class(4, 2048) == "t8.topk2k"
    assert S.prefill_shape_class(512) == "prefill.d512"


def test_split_selector_is_host_arithmetic():
    n = S.sparse_decode_splits_for(1, 64, 2051)
    assert 1 <= n <= 16  # the heuristic's own bound


# ---------------------------------------------------------------------------
# torch-oracle masking semantics (CPU)
# ---------------------------------------------------------------------------


def test_oracle_tail_padding_equals_topk_length():
    q, kv, idx = _case(pad=10)
    out_pad, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch")
    idx_full = _case(pad=0)[2]
    tl = torch.full((1,), idx.shape[-1] - 10, dtype=torch.int32)
    out_len, _, _ = S.flash_mla_sparse_fwd(q, kv, idx_full, 0.04, backend="torch",
                                           topk_length=tl)
    assert torch.equal(out_pad, out_len)


def test_oracle_out_of_bounds_indices_mask_like_negative():
    q, kv, idx = _case(pad=5)
    idx_oob = idx.clone()
    idx_oob[..., -5:] = kv.shape[0] + 7  # >= s_kv is invalid per the donor
    out_a, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch")
    out_b, _, _ = S.flash_mla_sparse_fwd(q, kv, idx_oob, 0.04, backend="torch")
    assert torch.equal(out_a, out_b)


def test_oracle_all_invalid_row_yields_zero():
    q, kv, idx = _case()
    idx[..., :] = -1
    out, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch")
    assert (out == 0).all()


def test_oracle_stats_match_hand_computation():
    q, kv, idx = _case(pad=3)
    out, ml, lse = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch",
                                          return_stats=True)
    safe = idx.clamp(min=0).long()[0, 0]
    scores = torch.einsum("hd,wd->hw", q[0].float(), kv[:, 0, :][safe].float()) * 0.04
    scores[..., -3:] = float("-inf")
    assert torch.allclose(ml[0], scores.amax(-1), atol=1e-5)
    lse_ref = torch.logsumexp(scores, dim=-1) / torch.log(torch.tensor(2.0))
    assert torch.allclose(lse[0], lse_ref, atol=1e-4)


# ---------------------------------------------------------------------------
# triton backend (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA required")
@pytest.mark.parametrize("shape", [(1, 8, 512, 2051), (4, 64, 512, 2051)],
                         ids=["decode-t1", "decode-b4"])
def test_triton_decode_parity_vs_oracle(shape):
    t, heads, dim, w = shape
    q, kv, idx = _case(seed=1, rows=8192, w=w, heads=heads, dim=dim,
                       pad=17, device="cuda")
    q = q.expand(t, -1, -1).contiguous()
    idx = idx.expand(t, -1, -1).contiguous()
    out, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="triton")
    ref, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch")
    assert (out.float() - ref.float()).abs().max().item() <= 0.02


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_triton_prefill_parity_vs_oracle():
    rows, w, heads, dim = 8192, 2048, 8, 512
    g = torch.Generator().manual_seed(2)
    q = torch.randn(96, heads, dim, generator=g).to("cuda", torch.bfloat16)
    kv = torch.randn(rows, 1, dim, generator=g).to("cuda", torch.bfloat16)
    perm = torch.randperm(rows, generator=g)
    idx = perm[:w].to(torch.int32).view(1, 1, w).expand(96, -1, -1).contiguous().to("cuda")
    idx[:, :, -5:] = -1
    out, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="triton")
    ref, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch")
    assert (out.float() - ref.float()).abs().max().item() <= 0.02


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_triton_backend_declines_attn_sink():
    q, kv, idx = _case(device="cuda")
    sink = torch.zeros(8, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="triton", attn_sink=sink)


# ---------------------------------------------------------------------------
# pinned configs through the op-config cache (GPU: real sweep + replay)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_op_config_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VKERNELS_CACHE", str(tmp_path))
    from vkernels.tuning import cache as tcache

    tcache.reset_memo()
    q, kv, idx = _case(device="cuda")
    cfg1 = S._decode_config(1, 8, 2051, 32, q, kv, idx)
    assert cfg1 == S._decode_config(1, 8, 2051, 32, q, kv, idx)  # memo hit
    cap = torch.cuda.get_device_capability()
    store_file = tmp_path / f"sgl_sparse_mla.decode.sm{cap[0]}{cap[1]}.json"
    assert store_file.exists()
    doc = json.loads(store_file.read_text())
    assert doc["records"]["t1.topk3k"]["config"]["num_splits"] == cfg1[0]

    cfg2 = S._decode_config(1, 8, 2051, 32, q, kv, idx)
    assert cfg2 == cfg1
    assert torch.cuda.synchronize() is None


# ---------------------------------------------------------------------------
# capture safety (GPU)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.skipif(not CUDA, reason="CUDA required")
def test_warmup_then_capture_replays_eager():
    report = S.warmup("cuda", num_heads=8, head_dim=512, topk=2051)
    assert report["backend"] == "triton"
    assert "decode_t1" in report["classes"] and "prefill" in report["classes"]

    q, kv, idx = _case(device="cuda")
    eager, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04)

    static_q = q.clone()
    static_idx = idx.clone()
    static_kv = kv.clone()
    g = torch.cuda.CUDAGraph()
    S.flash_mla_sparse_fwd(static_q, static_kv, static_idx, 0.04)  # memo warm
    with torch.cuda.graph(g):
        captured, _, _ = S.flash_mla_sparse_fwd(static_q, static_kv, static_idx, 0.04)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


# ---------------------------------------------------------------------------
# donor boundary (integration: only when sgl_kernel is actually installed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_sgl_kernel_donor_parity_when_present():
    if not S._sgl_kernel_available():
        pytest.skip("sgl_kernel wheel not importable on this device")
    q, kv, idx = _case(device="cuda")
    donor, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="sgl_kernel")
    ref, _, _ = S.flash_mla_sparse_fwd(q, kv, idx, 0.04, backend="torch")
    assert (donor.float() - ref.float()).abs().max().item() <= 0.02
