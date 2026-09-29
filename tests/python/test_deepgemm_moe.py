"""deepgemm_moe: the DeepGEMM masked-grouped borrow — gate + contract tests.

Lane 31. The capability reality this suite encodes: DeepGEMM dispatches on
arch major 9 (sm90 wgmma fp8) / 10 (sm100 tcgen05) and is DG_HOST_UNREACHABLE
elsewhere; GB10 (capability (12,1)) rejects BOTH ISA families in ptxas, so on
this lane's bench box the op self-gates OFF and everything downstream
(eligibility, warmup, the tuning-cache seed) must degrade gracefully and
cheaply — these tests run everywhere and assert exactly that.

The numerics parity tests (masked-layout correctness, fp8 e4m3 tolerance
gates vs the eager oracle, the runs/moegrp-micro halfway-rounding class)
are skip-gated on :func:`deepgemm_available` — they run where DeepGEMM
actually works (the sglang-glm53 sm90 image), not on GB10. The routing
prep (sort/histogram/rank/inverse-gather) has its own always-on test
against a CPU oracle, because that glue is ours and must be right before
any donor kernel ever sees it.
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops.deepgemm_moe import (  # noqa: E402
    SM90_SEED,
    deepgemm_available,
    deepgemm_grouped_moe,
    deepgemm_moe_eligible,
    deepgemm_unavailable_reason,
    seed_deepgemm_defaults,
    warmup_deepgemm_moe,
)
from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402

SWIGLU_LIMIT = 7.0


def _make_case(device, e, h, i, topk, t, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(t, h, device=device, dtype=torch.bfloat16, generator=gen)
    w13 = (torch.randn(e, 2 * i, h, device=device, generator=gen) * 0.02).to(torch.bfloat16)
    w2 = (torch.randn(e, h, i, device=device, generator=gen) * 0.02).to(torch.bfloat16)
    wb = w13.float().view(e, 2 * i // 128, 128, h // 128, 128)
    s13 = (wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0)
    w13_8 = (wb / s13).clamp(-448, 448).to(torch.float8_e4m3fn).view(e, 2 * i, h)
    s13 = s13.squeeze(2).squeeze(3).contiguous()
    wb = w2.float().view(e, h // 128, 128, i // 128, 128)
    s2 = (wb.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0)
    w2_8 = (wb / s2).clamp(-448, 448).to(torch.float8_e4m3fn).view(e, h, i)
    s2 = s2.squeeze(2).squeeze(3).contiguous()
    top_k_index = torch.stack(
        [torch.randperm(e, device=device, generator=gen)[:topk] for _ in range(t)]
    ).to(torch.int64)
    top_k_weights = torch.softmax(
        torch.randn(t, topk, device=device, generator=gen), dim=-1)
    return x, w13_8, s13, w2_8, s2, top_k_index, top_k_weights


# ---------------------------------------------------------------------------
# capability gate (every device, no deep_gemm requirement)
# ---------------------------------------------------------------------------
def test_gate_matches_probe():
    reason = deepgemm_unavailable_reason()
    assert (reason is None) == deepgemm_available()
    if reason is not None:
        # one-line, actionable: names the arch family it needs
        assert "arch major" in reason or "import failed" in reason or "no CUDA" in reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_gate_arch_statement():
    """The lane-31 finding, as a device test: on sm_121 (GB10) the gate is
    OFF and says why; on sm90/sm100 it is ON."""
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) == (12, 1):
        reason = deepgemm_unavailable_reason()
        assert reason is not None and "12" in reason
        assert not deepgemm_available()
    elif major in (9, 10):
        # only when deep_gemm is actually importable in this env
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            pytest.skip("deep_gemm not installed in this env")
        assert deepgemm_available()


# ---------------------------------------------------------------------------
# contract (runs everywhere CUDA runs; shapes chosen to fit the contract)
# ---------------------------------------------------------------------------
def _eligibility_matrix(device):
    e, h, i, topk = 288, 4096, 512, 8
    ok = _make_case(device, e, h, i, topk, 4)
    assert deepgemm_moe_eligible(*ok) == deepgemm_available()
    # T beyond the decode band
    too_big = _make_case(device, e, h, i, topk, 9)
    assert not deepgemm_moe_eligible(*too_big)
    # slots beyond the 64 masked-layout cap
    wide = _make_case(device, e, h, i, 9, 8)
    assert not deepgemm_moe_eligible(*wide)
    # missing scales
    no_scale = ok[:4] + (None, None) + ok[6:]
    assert not deepgemm_moe_eligible(*no_scale)
    # dim-3 activations
    bad_x = _make_case(device, e, h, i, topk, 2)
    bad_x = (bad_x[0].unsqueeze(0),) + bad_x[1:]
    assert not deepgemm_moe_eligible(*bad_x)
    # mismatched expert counts
    mismatch = list(ok)
    mismatch[3] = mismatch[3][: e - 1].contiguous()
    assert not deepgemm_moe_eligible(*mismatch)


def test_eligibility_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only")
    _eligibility_matrix(torch.device("cuda"))


def test_op_raises_on_gate_miss():
    """On a gated device the op raises OpNotEligible (the caller's fallback
    trigger), never a deep_gemm assertion."""
    if deepgemm_available() and torch.cuda.is_available():
        pytest.skip("deep_gemm available here — the gate does not miss")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only")
    case = _make_case(torch.device("cuda"), 288, 4096, 512, 8, 4)
    with pytest.raises(OpNotEligible):
        deepgemm_grouped_moe(*case, swiglu_limit=SWIGLU_LIMIT)


# ---------------------------------------------------------------------------
# routing prep glue (always-on: CPU oracle vs the device tensors the op builds)
# ---------------------------------------------------------------------------
def test_routing_prep_matches_oracle():
    """The masked-layout build must reproduce: group j's live rows are the
    tokens routed to j, in stable slot order; the inverse gather restores
    the [T, topk, H] slot order exactly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only")
    dev = torch.device("cuda")
    t, topk, e, h = 4, 8, 288, 128
    gen = torch.Generator(device=dev).manual_seed(3)
    idx = torch.stack([torch.randperm(e, device=dev, generator=gen)[:topk]
                       for _ in range(t)]).to(torch.int64)
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    masked_m = torch.zeros(e, device=dev, dtype=torch.int32)
    masked_m.scatter_add_(0, sorted_experts,
                          torch.ones_like(sorted_experts, dtype=torch.int32))
    starts = torch.zeros(e, device=dev, dtype=torch.int64)
    starts[1:] = torch.cumsum(masked_m.to(torch.int64), 0)[:-1]
    rank = torch.arange(order.numel(), device=dev, dtype=torch.int64) \
        - starts[sorted_experts]
    src_token = order // topk

    # oracle on CPU
    host_idx = idx.cpu()
    counts = [0] * e
    for row in host_idx.reshape(-1).tolist():
        counts[row] += 1
    assert masked_m.cpu().tolist() == counts

    # scatter + inverse-gather round trip with marker values
    payload = torch.arange(order.numel(), device=dev, dtype=torch.float32) \
        .unsqueeze(1).expand(-1, h).contiguous()
    table = torch.zeros(e, t, h, device=dev, dtype=torch.float32)
    table[sorted_experts, rank] = payload
    slots = torch.empty(order.numel(), h, device=dev, dtype=torch.float32)
    slots[order] = table[sorted_experts, rank]
    restored = slots.view(t, topk, h)
    # slots[f] = marker of the sorted position f was sourced FROM, i.e. the
    # rank of flat slot f in the sorted order — the inverse permutation.
    expect = torch.argsort(order).to(torch.float32).unsqueeze(1).expand(-1, h)
    assert torch.equal(restored.view(-1, h), expect)
    # and the grouped table holds each expert's routed tokens in stable flat
    # slot order (the ordering the oracle GEMM would visit them in):
    host_idx = idx.cpu().tolist()
    flat_slots = list(range(t * topk))
    by_expert: dict[int, list[int]] = {}
    for f, j in enumerate(v for row in host_idx for v in row):
        by_expert.setdefault(j, []).append(flat_slots[f])
    table_host = table.cpu()
    ord_cpu = order.cpu().tolist()
    for j in range(e):
        slots_j = by_expert.get(j, [])
        for r, f in enumerate(slots_j):
            assert table_host[j, r, 0].item() == ord_cpu.index(f)
    # live rows carry exactly the src tokens' payloads (stable order)
    src_cpu = src_token.cpu().tolist()
    ord_cpu = order.cpu().tolist()
    for s, tok in enumerate(src_cpu):
        assert tok == ord_cpu[s] // topk


# ---------------------------------------------------------------------------
# tuning-cache seed (no GPU needed beyond the store)
# ---------------------------------------------------------------------------
def test_seed_lands_sm90_prior(tmp_path, monkeypatch):
    from vkernels.tuning.cache import stored_records

    monkeypatch.setenv("VKERNELS_CACHE", str(tmp_path))  # the session conftest defaults off
    seed_deepgemm_defaults(store_dir=tmp_path, force=True)
    recs = stored_records(
        "deepgemm_moe", store_dir=tmp_path,
        device={"capability": (9, 0), "sm_count": 0, "name": "sm90-seed"},
    )
    for bucket in ("t2", "t4", "t8"):
        assert bucket in recs
        rec: dict = recs[bucket]  # ty: record payloads are JSON objects
        assert rec["status"] == "seeded"
        cfg: dict = rec["config"]
        seed_entry: dict = SM90_SEED[bucket]
        assert cfg["expected_m_mult"] == seed_entry["config"]["expected_m_mult"]
    # idempotent: re-seeding without force keeps the records, no error
    seed_deepgemm_defaults(store_dir=tmp_path)


# ---------------------------------------------------------------------------
# warmup (gate-verified no-op here; the compile path runs on sm90/sm100 only)
# ---------------------------------------------------------------------------
def test_warmup_gated_noop():
    if deepgemm_available():
        pytest.skip("deep_gemm available — warmup would really compile")
    assert warmup_deepgemm_moe([(288, 1024, 4096), (288, 4096, 512)]) == 0


# ---------------------------------------------------------------------------
# numerics (DeepGEMM-working archs only — sm90 donor image / sm100)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not deepgemm_available(), reason="DeepGEMM unsupported on this arch")
@pytest.mark.parametrize("t", [2, 4, 8])
def test_parity_vs_eager_oracle(t):
    from vkernels.torch_ops.sgl_moe import per_token_group_quant_fp8

    dev = torch.device("cuda")
    e, h, i, topk = 288, 4096, 512, 8
    x, w13_8, s13, w2_8, s2, idx, w = _make_case(dev, e, h, i, topk, t, seed=t)
    out = deepgemm_grouped_moe(x, w13_8, s13, w2_8, s2, idx, w,
                               swiglu_limit=SWIGLU_LIMIT)

    # eager oracle: dequantized weights, fp8-quantized activations (the
    # w8a8 design's inherent noise), per-token-group 128 both stages
    def deq(w8, s):
        ee, o, ii = w8.shape
        return (w8.float().view(ee, o // 128, 128, ii // 128, 128)
                * s.float()[:, :, None, :, None]).reshape(ee, o, ii)

    w13 = deq(w13_8, s13)
    w2 = deq(w2_8, s2)
    acc = torch.zeros(t, h, device=dev, dtype=torch.float32)
    for tok in range(t):
        for kk in range(topk):
            j = int(idx[tok, kk])
            xq, xs = per_token_group_quant_fp8(x[tok: tok + 1])
            a = ((xq.float() * xs.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
                 @ w13[j].t().to(torch.bfloat16))
            g, u = a[0, :i], a[0, i:]
            act = (torch.nn.functional.silu(g.clamp(max=SWIGLU_LIMIT))
                   * u.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT))
            aq, as_ = per_token_group_quant_fp8(act.unsqueeze(0))
            b = ((aq.float() * as_.repeat_interleave(128, dim=-1)).to(torch.bfloat16)
                 @ w2[j].t().to(torch.bfloat16))
            acc[tok] += float(w[tok, kk]) * b[0].float()
    ref = acc.to(torch.bfloat16)

    # tolerance gate: the house bf16-ulp band PLUS the w8a8 activation-quant
    # noise both donors share; the known halfway-rounding class (runs/
    # moegrp-micro: ~2-in-540k elements at the 2^-3 ulp boundary) is inside
    # this band — do NOT tighten to bit-level here.
    diff = (out.float() - ref.float()).abs()
    rel = diff / ref.float().abs().clamp_min(1e-3)
    assert rel.median() < 2e-2, f"median rel {rel.median():.4f}"
    assert (diff.max() / ref.float().abs().max()) < 4e-2


@pytest.mark.skipif(not deepgemm_available(), reason="DeepGEMM unsupported on this arch")
def test_masked_layout_bit_determinism():
    """Same inputs twice -> bitwise-equal outputs (DeepGEMM masked kernels
    are deterministic at fixed config; the tuning memo freezes config)."""
    dev = torch.device("cuda")
    x, w13_8, s13, w2_8, s2, idx, w = _make_case(dev, 288, 4096, 512, 8, 4, seed=7)
    o1 = deepgemm_grouped_moe(x, w13_8, s13, w2_8, s2, idx, w, swiglu_limit=SWIGLU_LIMIT)
    o2 = deepgemm_grouped_moe(x, w13_8, s13, w2_8, s2, idx, w, swiglu_limit=SWIGLU_LIMIT)
    assert torch.equal(o1, o2)
