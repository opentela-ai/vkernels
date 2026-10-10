"""Tests for the vendored TokenSpeed-Kernel-AMD ops (torch_ops/tokenspeed_amd).

Host CI covers the import surface, the shim degradation behavior, the
pure-torch preprocessing (scale swizzle, N16 banks, weight preprocessing),
the torch fallbacks, and the metadata registry. GPU parity against torch
references is marked ``gpu`` and runs only where a working Gluon driver
exists (beverin / MI300A), mirroring the tokenspeed_mla test conventions.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops import tokenspeed_amd as ts_amd  # noqa: E402
from vkernels.torch_ops.tokenspeed_amd import _gluon as shim  # noqa: E402
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    argmax as ts_argmax,
)
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    moe_sorting as ts_sorting,
)
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    n16_weights as ts_n16,
)
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    routing as ts_routing,
)
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    scale_layout as ts_scale,
)
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    weight_preprocess as ts_wp,
)
from vkernels.torch_ops.tokenspeed_amd_registry import (  # noqa: E402
    HADAMARD_TRANSFORM_REGISTRY,
    MOE_SORTING_REGISTRY,
    MHC_PRE_REGISTRY,
    ROUTER_TOPK_REGISTRY,
    SAMPLING_ARGMAX_REGISTRY,
    TOKENSPEED_AMD_REGISTRY,
)
from vkernels.torch_ops.tokenspeed_amd import (  # noqa: E402
    dsa_sparse_mla as ts_dsa_topk,
)
from vkernels.registry import KernelRequest, TensorMetadata  # noqa: E402


# --- import surface / shim ----------------------------------------------------


def test_module_exposes_probes_and_no_upstream_dependency():
    assert isinstance(ts_amd.gluon_available(), bool)
    assert isinstance(ts_amd.cdna4_available(), bool)
    import sys

    upstream = [m for m in sys.modules if m.startswith("tokenspeed")]
    assert not upstream, f"upstream tokenspeed modules leaked: {upstream}"


def test_inert_stub_sinks_attributes_and_fails_loudly_on_launch():
    inert = shim._Inert()
    assert inert.anything.deeper is not None
    assert inert.constexpr(3) is not None  # module-level constants survive
    with pytest.raises(RuntimeError, match="gluon"):
        inert[(1,)]


def test_probes_consistent_with_host_triton():
    try:
        import triton

        triton.experimental.gluon  # noqa: B018
    except (ImportError, AttributeError):
        assert not ts_amd.gluon_available()
    else:
        assert ts_amd.gluon_available()
        # cdna4 may or may not exist (triton >= ~3.5); either is fine, but
        # the probe must agree with attribute truth.
        assert ts_amd.cdna4_available() == (
            getattr(getattr(triton.experimental.gluon.language, "amd", None), "cdna4", None)
            is not None
        )


# --- scale_layout (pure torch) -------------------------------------------------


def _unswizzle_via_identity(swizzled_fn, n: int, k_scale: int, scale: torch.Tensor):
    """Recover ``scale`` by inverting the swizzle through an identity map."""
    ids = torch.arange(n * k_scale, dtype=torch.int64).reshape(n, k_scale)
    perm = swizzled_fn(ids).reshape(-1)  # perm[p] = original flat index at p
    out = swizzled_fn(scale).reshape(-1)
    recovered = torch.empty_like(out)
    recovered[perm] = out
    return recovered.reshape(n, k_scale)


def test_scale_swizzle_roundtrip_and_signature():
    torch.manual_seed(0)
    n, k_scale = 64, 16
    scale = torch.randint(0, 255, (n, k_scale), dtype=torch.uint8)
    out = ts_scale.swizzle_cdna4_mxfp4_scale(scale)
    assert ts_scale.is_swizzled_cdna4_mxfp4_scale(out)
    assert not ts_scale.is_swizzled_cdna4_mxfp4_scale(scale)
    recovered = _unswizzle_via_identity(ts_scale.swizzle_cdna4_mxfp4_scale, n, k_scale, scale)
    assert torch.equal(recovered, scale)


def test_scale_swizzle_pads_partial_blocks():
    n, k_scale = 33, 9  # pads to 32-row/8-col multiples
    scale = torch.randint(0, 255, (n, k_scale), dtype=torch.uint8)
    out = ts_scale.swizzle_cdna4_mxfp4_scale(scale)
    assert out.shape == (1, 16 * 32, 2)  # k_pad=16 -> 512, n_pad=64 -> 2 blocks
    empty = ts_scale.empty_swizzled_cdna4_mxfp4_scale(
        n, k_scale, device=scale.device
    )
    assert empty.shape == out.shape[1:] and empty.stride() == out.stride()[1:]
    assert ts_scale.swizzle_cdna4_mxfp4_scale(scale.unsqueeze(0)).shape[0] == 1


def test_scale_swizzle_rejects_rank1():
    with pytest.raises(ValueError, match="at least 2"):
        ts_scale.swizzle_cdna4_mxfp4_scale(torch.zeros(8, dtype=torch.uint8))


def test_scale_swizzle_lane0_block_matches_documented_permutation():
    # The docstring pins the first eight bytes lane 0 reads for a 32x8 tile:
    # (n=0,kb=0) (n=16,kb=0) (n=0,kb=4) (n=16,kb=4) (n=1,kb=0) (n=17,kb=0) (n=1,kb=4) (n=17,kb=4)
    n, k_scale = 32, 8
    scale = torch.zeros(n, k_scale, dtype=torch.uint8)
    for n_i in range(n):
        for k_i in range(k_scale):
            scale[n_i, k_i] = n_i * k_scale + k_i  # unique bytes
    out = ts_scale.swizzle_cdna4_mxfp4_scale(scale)
    first8 = out.reshape(-1)[:8].tolist()
    expected = [
        int(scale[0, 0]), int(scale[16, 0]), int(scale[0, 4]), int(scale[16, 4]),
        int(scale[1, 0]), int(scale[17, 0]), int(scale[1, 4]), int(scale[17, 4]),
    ]
    assert first8 == expected


# --- n16 weights (pure-torch shuffles + gluon offset helpers) ------------------


def _n16_offset_ref(n: int, k_packed: int, packed_k: int) -> int:
    return (
        (n // 16 * (packed_k // 64) + k_packed // 64) * 1024
        + (k_packed // 16 % 4) * 256
        + n % 16 * 16
        + k_packed % 16
    )


def test_n16_shuffle_weight_matches_gluon_offset_formula():
    torch.manual_seed(1)
    experts, columns, packed_k = 2, 64, 128
    weight = torch.randint(0, 255, (experts, columns, packed_k), dtype=torch.uint8)
    shuffled = ts_n16._shuffle_weight(weight, gate_up=False)
    flat = shuffled.reshape(experts, -1)
    for n in (0, 1, 15, 16, 31, 63):
        for k in (0, 1, 15, 16, 63, 64, 127):
            assert int(flat[1, _n16_offset_ref(n, k, packed_k)]) == int(weight[1, n, k])


def test_n16_shuffle_scale_gate_up_split():
    torch.manual_seed(2)
    experts, columns, groups = 1, 64, 16
    scale = torch.randint(0, 255, (experts, columns, groups), dtype=torch.uint8)
    out = ts_n16._shuffle_scale(scale, gate_up=False)
    assert out.shape == (experts, columns // 32, groups // 8, 4, 16, 4)
    assert out.numel() == scale.numel()


def _make_n16_module(hidden=256, intermediate=256, experts=2):
    class _W(torch.nn.Module):
        pass

    w = _W()
    two_i, packed_d = 2 * intermediate, hidden // 2
    w.w13_weight = torch.nn.Parameter(
        torch.randint(0, 255, (experts, two_i, packed_d), dtype=torch.uint8),
        requires_grad=False,
    )
    w.w13_weight_scale = torch.nn.Parameter(
        torch.randint(127, 130, (experts, two_i, hidden // 32), dtype=torch.uint8),
        requires_grad=False,
    )
    w.w2_weight = torch.nn.Parameter(
        torch.randint(0, 255, (experts, hidden, intermediate // 2), dtype=torch.uint8),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        torch.randint(127, 130, (experts, hidden, intermediate // 32), dtype=torch.uint8),
        requires_grad=False,
    )
    return w


def test_n16_preprocess_produces_valid_bank():
    w = _make_n16_module()
    ts_n16.preprocess_n16_mxfp4_weights(w)
    e, d, i = ts_n16.n16_mxfp4_shape(
        w.w13_weight, w.w13_weight_scale, w.w2_weight, w.w2_weight_scale
    )
    assert (e, d, i) == (2, 256, 256)
    with pytest.raises(ValueError, match="already prepared"):
        ts_n16.preprocess_n16_mxfp4_weights(w)


def test_n16_preprocess_rejects_bad_shapes():
    w = _make_n16_module(hidden=200)  # not divisible by 256
    with pytest.raises(ValueError):
        ts_n16.preprocess_n16_mxfp4_weights(w)


# --- weight_preprocess (pure torch) -------------------------------------------


def _make_moe_module(experts=2, hidden=256, intermediate=256, *, bias=True):
    class _W(torch.nn.Module):
        pass

    w = _W()
    two_i = 2 * intermediate
    w.w13_weight = torch.nn.Parameter(
        torch.randint(0, 255, (experts, two_i, hidden // 2), dtype=torch.uint8),
        requires_grad=False,
    )
    w.w13_weight_scale = torch.nn.Parameter(
        torch.randint(126, 129, (experts, two_i, hidden // 32), dtype=torch.uint8),
        requires_grad=False,
    )
    w.w2_weight = torch.nn.Parameter(
        torch.randint(0, 255, (experts, hidden, intermediate // 2), dtype=torch.uint8),
        requires_grad=False,
    )
    w.w2_weight_scale = torch.nn.Parameter(
        torch.randint(126, 129, (experts, hidden, intermediate // 32), dtype=torch.uint8),
        requires_grad=False,
    )
    if bias:
        w.w13_weight_bias = torch.zeros(experts, two_i, dtype=torch.bfloat16)
        w.w2_weight_bias = torch.zeros(experts, hidden, dtype=torch.bfloat16)
    return w


def test_interleave_gate_up_rows():
    x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    out = ts_wp._interleave_gate_up_rows(x, dim=1)
    assert out.tolist() == [[0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]]
    with pytest.raises(ValueError, match="even"):
        ts_wp._interleave_gate_up_rows(torch.zeros(3, 4), dim=0)


def test_weight_preprocess_swizzle_and_configs():
    w = _make_moe_module()
    raw13 = w.w13_weight.data.clone()
    ts_wp.preprocess_gluon_mxfp4_gfx950_moe_weights({}, w, preshuffle=False)
    # gate/up halves interleaved row-wise, then stored K-packed
    # (``_make_k_packed_mxfp4_weight`` view: stride(-2) == 1)
    interleaved = ts_wp._interleave_gate_up_rows(raw13, dim=-2)
    stored = w.w13_weight_triton_tensor
    assert stored.stride(-2) == 1 and stored.stride(-1) == interleaved.shape[-1]
    assert torch.equal(stored.contiguous(), interleaved.transpose(-2, -1).contiguous())
    assert ts_scale.is_swizzled_cdna4_mxfp4_scale(w.w13_precision_config.b_mx_scale)
    assert w.w13_precision_config.b_microblock_size == ts_scale.MXFP4_BLOCK
    assert w.w13_precision_config.out_dtype == torch.bfloat16
    assert w._gluon_w13_bias_is_zero is True
    assert w._gluon_w2_bias_is_zero is True
    # original parameter slots released
    assert not hasattr(w, "w13_weight")


def test_weight_preprocess_pads_w2_to_block_n():
    # intermediate=200 -> w2 N (=hidden 256) is fine; use odd intermediate to
    # force nothing; instead force via hidden not multiple of 128 for w13 K.
    w = _make_moe_module(experts=1, hidden=256, intermediate=192)
    # w2_weight N = hidden = 256 (divisible); make it non-divisible manually:
    w.w2_weight = torch.nn.Parameter(
        w.w2_weight.data[:, :200, :].contiguous(), requires_grad=False
    )
    w.w2_weight_scale = torch.nn.Parameter(
        w.w2_weight_scale.data[:, :200, :].contiguous(), requires_grad=False
    )
    ts_wp._pad_w2_to_block_n(w, ts_wp._GLUON_COMBINE_BLOCK_N)
    assert w.w2_weight.shape[-2] == 256
    assert w._w2_logical_n == 200
    # padded scale rows are identity (127)
    assert bool((w.w2_weight_scale.data[:, 200:, :] == 127).all())


def test_shuffle_weight_for_gluon_dot_layout_contract():
    torch.manual_seed(3)
    k_pk, n = 128, 128
    weight = torch.randint(0, 255, (k_pk, n), dtype=torch.uint8)
    out = ts_wp.shuffle_weight_for_gluon_dot_layout(weight)
    assert out.shape == (k_pk, n)  # leading dims preserved; 2-D in, 2-D out
    assert out.is_shuffled_for_gluon_dot is True
    assert out.original_k_pk == k_pk
    # P must be a bijection over K*N bytes
    b = ts_wp._GLUON_DOT_SUB_TILE_K * 2  # block_k_pk=128
    block_n = 128
    stride_k_quad = ts_wp._GLUON_DOT_N_LANE * ts_wp._GLUON_DOT_K_WIDTH
    stride_k_block = ts_wp._GLUON_DOT_K_QUAD * stride_k_quad
    stride_n_block = (b // ts_wp._GLUON_DOT_SUB_TILE_K) * stride_k_block
    k = torch.arange(b).view(-1, 1)
    nn = torch.arange(block_n).view(1, -1)
    p = (
        (nn // 16) * stride_n_block
        + (k // 64) * stride_k_block
        + ((k // 16) % 4) * stride_k_quad
        + (nn % 16) * 16
        + (k % 16)
    )
    assert len(torch.unique(p.flatten())) == b * block_n
    # out[P[j]] = w[k, n] for j = k*N + n (k-major source flatten)
    flat = out.reshape(-1)
    recovered = torch.empty_like(flat)
    recovered[p.flatten()] = weight.reshape(-1)
    assert torch.equal(flat, recovered)
    with pytest.raises(ValueError, match="block_k_pk"):
        ts_wp.shuffle_weight_for_gluon_dot_layout(weight, block_k_pk=32)
    with pytest.raises(ValueError, match="block_n"):
        ts_wp.shuffle_weight_for_gluon_dot_layout(weight, block_n=24)
    with pytest.raises(ValueError, match="divisible by block_n"):
        ts_wp.shuffle_weight_for_gluon_dot_layout(weight[:, :100].contiguous())


# --- routing / sorting host logic ---------------------------------------------


def test_routing_support_gates_on_metadata():
    logits_cpu = torch.randn(4, 128)
    assert ts_routing.gluon_topk_route_supported(logits_cpu, 8) is False  # CPU
    with pytest.raises(ValueError, match="unsupported"):
        ts_routing.invoke_softmax_topk_route_gluon(logits_cpu, 8)
    assert ts_routing._next_pow2(1) == 1
    assert ts_routing._next_pow2(5) == 8


def test_moe_sorting_capacity_bound():
    assert ts_sorting._max_padded_route_capacity(8, 4, 16) == 64
    assert ts_sorting._max_padded_route_capacity(0, 8, 32) == 0
    assert ts_sorting._max_padded_route_capacity(16, 4, 8) == 16 + 4 * 7 // 8 * 8
    with pytest.raises(ValueError):
        ts_sorting._max_padded_route_capacity(-1, 4, 8)
    with pytest.raises(ValueError):
        ts_sorting._max_padded_route_capacity(4, 4, 0)


# --- argmax torch fallback ------------------------------------------------------


def test_argmax_fallback_matches_torch():
    torch.manual_seed(4)
    logits = torch.randn(7, 4096)
    out = ts_argmax.launch_gluon_argmax_gfx950(logits)
    assert torch.equal(out, torch.argmax(logits, dim=-1))
    pre = torch.empty(7, dtype=torch.int64)
    assert ts_argmax.launch_gluon_argmax_gfx950(logits, out=pre) is pre
    assert torch.equal(pre, torch.argmax(logits, dim=-1))


def test_argmax_pair_and_validation():
    logits = torch.randn(3, 128)
    pair = ts_argmax.argmax_pair(logits)
    assert pair.shape == (3, 2) and pair.dtype == torch.float32
    vals, idxs = torch.max(logits, dim=-1)
    assert torch.equal(pair[:, 0], vals)
    assert torch.equal(pair[:, 1], idxs.to(torch.float32))
    with pytest.raises(ValueError, match="out must have shape"):
        ts_argmax.launch_gluon_argmax_gfx950(
            logits, out=torch.empty(4, dtype=torch.int64)
        )
    with pytest.raises(ValueError, match="int32 or int64"):
        ts_argmax.launch_gluon_argmax_gfx950(
            logits, out=torch.empty(3, dtype=torch.float32)
        )


def test_argmax_select_config_table():
    assert ts_argmax._select_config(1, 200_000) == (8192, 4, True, 32)
    assert ts_argmax._select_config(4, 200_000) == (16384, 4, True, 16)
    assert ts_argmax._select_config(8, 100_000) == (16384, 4, True, None)
    assert ts_argmax._select_config(64, 200_000) == (32768, 4, True, None)
    assert ts_argmax._select_config(128, 100_000) == (8192, 4, False, None)
    assert ts_argmax._load_elements_per_thread(torch.float16) == 8


# --- registry -------------------------------------------------------------------


def _request(op, *tensors):
    return KernelRequest(
        operation=op,
        tensors=tuple(
            TensorMetadata(
                shape=t.shape, dtype=str(t.dtype).replace("torch.", ""), device="cuda"
            )
            for t in tensors
        ),
        architecture="gfx942",
        backends=frozenset({"gluon"}),
    )


def test_registry_selects_and_gates():
    logits = torch.zeros(8, 128, dtype=torch.bfloat16)
    assert ROUTER_TOPK_REGISTRY.select(_request("router_topk", logits)).name in (
        "gluon-softmax",
        "gluon-sigmoid-bias",
    )
    with pytest.raises(Exception):
        ROUTER_TOPK_REGISTRY.select(
            _request("router_topk", torch.zeros(8, 2048))
        )  # E > 1024
    ids = torch.zeros(8, 4, dtype=torch.int32)
    weights = torch.zeros(8, 4, dtype=torch.float32)
    assert MOE_SORTING_REGISTRY.select(_request("moe_sorting", ids, weights)).name == "gluon"
    assert (
        SAMPLING_ARGMAX_REGISTRY.select(_request("sampling_argmax", logits)).name
        == "gluon"
    )
    rows = torch.zeros(64, 128, dtype=torch.bfloat16)
    assert (
        HADAMARD_TRANSFORM_REGISTRY.select(_request("hadamard_transform", rows)).name
        == "gluon-128"
    )
    with pytest.raises(Exception):
        HADAMARD_TRANSFORM_REGISTRY.select(
            _request("hadamard_transform", torch.zeros(64, 64))
        )


def test_mhc_registry_shape_gate():
    mul = torch.zeros(64, 8, 24)
    sqrsum = torch.zeros(64, 8)
    residual = torch.zeros(8, 4, 4096)
    layer_input = torch.zeros(8, 4096, dtype=torch.bfloat16)
    ok = _request("mhc_pre", mul, sqrsum, residual, layer_input)
    assert MHC_PRE_REGISTRY.select(ok).name == "gluon"
    bad_residual = torch.zeros(8, 4, 2048)
    with pytest.raises(Exception):
        MHC_PRE_REGISTRY.select(
            _request("mhc_pre", mul, sqrsum, bad_residual, layer_input)
        )
    with pytest.raises(Exception):
        MHC_PRE_REGISTRY.select(
            _request("mhc_pre", mul, sqrsum, torch.zeros(65, 4, 4096), layer_input)
        )


def test_combined_registry_view():
    ops = {impl.operation for impl in TOKENSPEED_AMD_REGISTRY.implementations}
    assert ops == {
        "router_topk",
        "moe_sorting",
        "sampling_argmax",
        "hadamard_transform",
        "mhc_pre",
    }


def test_registry_rejects_cpu_tensors():
    explanation = ROUTER_TOPK_REGISTRY.explain(
        KernelRequest(
            operation="router_topk",
            tensors=(TensorMetadata(shape=(8, 128), dtype="bfloat16", device="cpu"),),
            backends=frozenset({"gluon"}),
        )
    )
    assert explanation.selected is None
    reasons = [
        r
        for candidate in explanation.candidates
        for r in candidate.reasons
    ]
    assert any("GPU" in r for r in reasons)


# --- GPU parity (skipped without a working gluon driver) ------------------------


def _gluon_driver():
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm required")
    if not ts_amd.gluon_available():
        pytest.skip("gluon unavailable on this triton")
    try:
        import triton

        triton.runtime.driver.active  # noqa: B018 — probes the runtime driver
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"triton driver unavailable: {exc}")


@pytest.mark.gpu
def test_gpu_softmax_routing_matches_reference():
    _gluon_driver()
    torch.manual_seed(5)
    for m, e, topk in ((1, 128, 8), (8, 256, 16), (64, 384, 2)):
        logits = torch.randn(m, e, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(e, device="cuda", dtype=torch.float32)
        scale = 2.5
        ids, weights = ts_routing.invoke_softmax_topk_route_gluon(
            logits, topk, correction_bias=bias,
            routed_scaling_factor=scale, normalize_topk_weights=True,
        )
        # reference: softmax + bias selects; stored weights are unbiased
        # full-row softmax scores, renormalized, then scaled
        probs = torch.softmax(logits.float(), dim=-1)
        choice = probs + bias
        sel = torch.topk(choice, topk, dim=-1).indices
        rows = torch.arange(m, device="cuda")[:, None]
        w = probs[rows, sel]
        w = w / w.sum(dim=-1, keepdim=True) * scale
        assert torch.equal(ids.to(torch.int64), sel)
        assert torch.allclose(weights, w.to(torch.float32), atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_gpu_moe_sorting_matches_reference_sort():
    _gluon_driver()
    torch.manual_seed(6)
    m, e, topk, block = 512, 128, 4, 16
    ids = torch.randint(0, e, (m, topk), device="cuda", dtype=torch.int32)
    weights = torch.rand(m, topk, device="cuda", dtype=torch.float32)
    (sorted_ids, sorted_weights, sorted_expert_ids, num_valid, out) = (
        ts_sorting.gluon_moe_sorting(
            ids, weights, e, 64, torch.bfloat16, block,
            compact_route_programs=False,
        )
    )
    # Reference block-aligned sort. sorted_ids are packed
    # ``(topk_id << 24) | token_id``; stage-4 atomics make intra-chunk order
    # nondeterministic, so compare per-expert multisets, block alignment,
    # padding sentinels, and the weight<->id pairing.
    flat = ids.reshape(-1)
    counts = torch.bincount(flat, minlength=e)
    starts = [0]
    for c in counts.tolist():
        starts.append(starts[-1] + -(-c // block) * block)
    padded_total = starts[-1]
    assert int(num_valid[0]) == padded_total
    assert int(num_valid[1]) == m
    assert out.shape == (m, 64)
    for exp in range(0, e, 17):  # sample experts
        s, c = starts[exp], int(counts[exp])
        got = sorted_ids[s : s + c]
        got_pairs = sorted(zip((got >> 24).tolist(), (got & 0xFFFFFF).tolist()))
        ref_idx = (flat == exp).nonzero().flatten()
        ref_pairs = sorted(
            zip((ref_idx % topk).tolist(), (ref_idx // topk).tolist())
        )
        assert got_pairs == ref_pairs, f"expert {exp} route mismatch"
        # weights follow their packed ids (pair each actual slot value)
        for slot in range(s, s + c):
            k = int(sorted_ids[slot] >> 24)
            tok = int(sorted_ids[slot] & 0xFFFFFF)
            assert float(sorted_weights[slot]) == pytest.approx(
                float(weights[tok, k]), abs=1e-6
            )
        # padding sentinel: (topk << 24) | m
        pad = sorted_ids[s + c : starts[exp + 1]]
        if pad.numel():
            assert bool((pad == ((topk << 24) | m)).all())
        # block map consistency
        for b in range(s // block, starts[exp + 1] // block):
            assert int(sorted_expert_ids[b]) == exp


@pytest.mark.gpu
def test_gpu_hadamard_matches_torch_matrix():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import hadamard as ts_hadamard

    torch.manual_seed(7)
    rows = 4096
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    # build the unnormalized 128x128 Hadamard matrix via Kronecker powers
    base = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    mat = base
    for _ in range(6):
        mat = torch.kron(mat, base)
    mat = mat.to(device="cuda", dtype=torch.float32)
    scale = 128 ** -0.5
    out = ts_hadamard.launch_gluon_hadamard_transform_128_gfx950(x, scale=scale)
    ref = (x.float() @ mat.T) * scale
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=1e-2)
    # non-contiguous trailing dim is rejected by reshape -> verify empty rows
    empty = ts_hadamard.launch_gluon_hadamard_transform_128_gfx950(
        x[:0], scale=scale
    )
    assert empty.shape == (0, 128)


@pytest.mark.gpu
def test_gpu_n16_shuffle_cpu_gpu_agree():
    _gluon_driver()
    torch.manual_seed(8)
    weight = torch.randint(0, 255, (2, 64, 128), dtype=torch.uint8)
    cpu = ts_n16._shuffle_weight(weight, gate_up=False)
    gpu = ts_n16._shuffle_weight(weight.cuda(), gate_up=False)
    assert torch.equal(cpu.cuda(), gpu)


def _kda_fused_decode_reference(
    mixed_qkv, conv_weights, conv_states, f_a_out, f_b_weight, beta_logits,
    a_log, dt_bias, output_gate, norm_weight, state_pool, read_indices,
    write_indices, norm_eps=1e-5, lower_bound=None,
):
    """Eager fp32 oracle for ``gluon_kda_fused_decode_gfx950`` (K3 shape).

    Mirrors the kernel's op order exactly: silu conv on bf16-moved history,
    bf16-rounded low-rank decay gate, delta rule on the L2-normalized q/k
    with the V-major state, gated RMSNorm output. Fresh tensors — the
    kernel mutates the pools in place.
    """
    tokens = mixed_qkv.shape[0]
    heads, dim = 12, 128
    pw = heads * dim
    out = torch.zeros(1, tokens, heads, dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    next_conv = conv_states.clone()
    next_state = state_pool.clone()
    for n in range(tokens):
        ridx = int(read_indices[n])
        widx = int(write_indices[n])
        if ridx < 0 or ridx >= state_pool.shape[0]:
            continue
        beta = torch.sigmoid(beta_logits[n].float())
        a_value = torch.exp(a_log.float())
        qkv = torch.empty(3 * pw, dtype=torch.float32, device=mixed_qkv.device)
        for c in range(3 * pw):
            hist = conv_states[ridx, c].float()
            x = mixed_qkv[n, c].float()
            w = conv_weights[c].float()
            value = hist[0] * w[0] + hist[1] * w[1] + hist[2] * w[2] + x * w[3]
            qkv[c] = value * (1.0 / (1.0 + torch.exp(-value)))
        if widx >= 0:
            next_conv[widx, :, 0] = conv_states[ridx, :, 1]
            next_conv[widx, :, 1] = conv_states[ridx, :, 2]
            next_conv[widx, :, 2] = mixed_qkv[n, :]
        for h in range(heads):
            q = qkv[h * dim : (h + 1) * dim]
            k = qkv[pw + h * dim : pw + (h + 1) * dim]
            v = qkv[2 * pw + h * dim : 2 * pw + (h + 1) * dim]
            raw_gate = (
                f_b_weight[h * dim : (h + 1) * dim].float()
                @ f_a_out[n].float()
            ).to(f_a_out.dtype).float()
            gate = raw_gate + dt_bias[h * dim : (h + 1) * dim].float()
            if lower_bound is not None:
                log_decay = lower_bound / (1.0 + torch.exp(-(a_value[h] * gate)))
            else:
                softplus = torch.maximum(gate, torch.zeros_like(gate)) + torch.log(
                    1.0 + torch.exp(-torch.abs(gate))
                )
                log_decay = -a_value[h] * softplus
            decay = torch.exp(log_decay)
            q_scale = torch.rsqrt((q * q).sum() + 1e-6) * (dim**-0.5)
            k_scale = torch.rsqrt((k * k).sum() + 1e-6)
            qh = q * q_scale
            kh = k * k_scale
            key_query = (q * k).sum() * q_scale * k_scale
            running = state_pool[ridx, h].float() * decay[None, :]
            prediction = (running * kh[None, :]).sum(dim=1)
            delta = beta[h] * (v - prediction)
            prior_output = (running * qh[None, :]).sum(dim=1)
            panel_out = prior_output + delta * key_query
            updated = running + delta[:, None] * kh[None, :]
            if widx >= 0:
                next_state[widx, h] = updated
            out_sumsq = (panel_out * panel_out).sum()
            inverse_rms = torch.rsqrt(out_sumsq / dim + norm_eps)
            gate_v = output_gate[n, h * dim : (h + 1) * dim].float()
            weight = norm_weight.float()
            out[0, n, h] = (
                panel_out * inverse_rms * weight * torch.sigmoid(gate_v)
            ).to(mixed_qkv.dtype)
    return out, next_conv, next_state


@pytest.mark.gpu
def test_gpu_kda_fused_decode_matches_reference():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import kda_decode as ts_kda

    torch.manual_seed(13)
    batch, heads, dim, pages = 5, 12, 128, 9
    pw = heads * dim
    mixed_qkv = torch.randn(batch, 3 * pw, device="cuda", dtype=torch.bfloat16) * 0.5
    conv_weights = torch.randn(3 * pw, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    conv_states = torch.randn(
        pages, 3 * pw, 3, device="cuda", dtype=torch.bfloat16
    ) * 0.3
    f_a_out = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)
    f_b_weight = torch.randn(pw, dim, device="cuda", dtype=torch.bfloat16) * 0.05
    beta_logits = torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(pw, device="cuda", dtype=torch.float32) * 0.1
    output_gate = torch.randn(batch, pw, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    state_pool = torch.randn(
        pages, heads, dim, dim, device="cuda", dtype=torch.float32
    ) * 0.05
    read_indices = torch.tensor([0, 1, 2, -1, 4], dtype=torch.int32, device="cuda")
    write_indices = torch.tensor([0, 1, -1, 3, 4], dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device="cuda")

    ref_out, ref_conv, ref_state = _kda_fused_decode_reference(
        mixed_qkv, conv_weights, conv_states, f_a_out, f_b_weight, beta_logits,
        a_log, dt_bias, output_gate, norm_weight, state_pool, read_indices,
        write_indices,
    )
    conv_pool = conv_states.clone()
    state_pool_k = state_pool.clone()
    out = ts_kda.gluon_kda_fused_decode_gfx950(
        mixed_qkv, conv_weights, conv_pool, f_a_out, f_b_weight, beta_logits,
        a_log, dt_bias, output_gate, norm_weight, 1e-5,
        state_pool=state_pool_k, read_indices=read_indices,
        write_indices=write_indices, num_heads=heads, head_dim=dim,
        cu_seqlens=cu_seqlens, lower_bound=None,
    )
    # padded read row (index 3) must produce zeros
    assert torch.equal(out[0, 3], torch.zeros_like(out[0, 3]))
    valid = torch.tensor([True, True, True, False, True])
    assert torch.allclose(
        out[0][valid].float(), ref_out[0][valid].float(), atol=3e-2, rtol=3e-2
    )
    assert torch.equal(conv_pool, ref_conv)
    assert torch.allclose(state_pool_k, ref_state, atol=2e-3, rtol=1e-3)


# --- DSA logical top-k (vendored sparse_mla; radix selection) ---------------


def test_dsa_logical_topk_contract_gates():
    logits = torch.zeros(4, 4096)
    starts = torch.zeros(4, dtype=torch.int32)
    ends = torch.full((4,), 4096, dtype=torch.int32)
    with pytest.raises(ValueError, match="topk"):
        ts_dsa_topk.gluon_dsa_logical_topk_gfx950(
            logits, starts, ends, topk=256
        )
    with pytest.raises(RuntimeError, match="CUDA"):
        ts_dsa_topk.gluon_dsa_logical_topk_gfx950(
            logits, starts, ends, topk=512
        )
    fp16 = torch.zeros(4, 4096, dtype=torch.float16)
    with pytest.raises(TypeError, match="float32"):
        ts_dsa_topk.gluon_dsa_logical_topk_gfx950(
            fp16, starts, ends, topk=512
        )


def test_dsa_standard_cache_stub_raises():
    for name in (
        "gluon_dsa_decode_topk_standard_gfx950",
        "gluon_dsa_prefill_topk_standard_gfx950",
        "gluon_kpool_prefill_topk_fp8_gfx950",
        "gluon_kpool_prefill_topk_fp8_plan_gfx950",
    ):
        with pytest.raises(RuntimeError, match="CDNA4"):
            getattr(ts_dsa_topk, name)()


@pytest.mark.gpu
def test_gpu_dsa_logical_topk_matches_torch():
    _gluon_driver()
    torch.manual_seed(15)
    rows, cols, topk = 9, 4096, 512
    logits = torch.randn(rows, cols, device="cuda", dtype=torch.float32) * 3
    # one empty row and one window shorter than topk
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), cols, dtype=torch.int32, device="cuda")
    starts[3], ends[3] = 100, 100  # empty
    ends[5] = 100 + 300  # 300 < topk
    idx, lens = ts_dsa_topk.gluon_dsa_logical_topk_gfx950(
        logits, starts, ends, topk=topk
    )
    assert idx.shape == (rows, topk) and idx.dtype == torch.int32
    for r in range(rows):
        lo, hi = int(starts[r]), int(ends[r])
        n = min(hi - lo, topk)
        assert int(lens[r]) == n
        got = idx[r, :n]
        assert bool((got >= lo).all() and (got < hi).all()), r
        # same selection as torch.topk (compare as sorted index sets; the
        # radix order and torch's order may differ on ties)
        ref_sel = torch.topk(logits[r, lo:hi], n).indices + lo
        got_sorted = torch.sort(logits[r, got.long()]).values
        ref_sorted = torch.sort(logits[r, ref_sel.long()]).values
        assert torch.equal(got_sorted, ref_sorted), r
        assert bool((idx[r, n:] == -1).all()), r


@pytest.mark.gpu
def test_gpu_dsa_fp8_decode_topk_smoke():
    """Compile/run smoke for the packed-FP8 decode top-k (dsa_indexing)."""
    _gluon_driver()
    torch.manual_seed(16)
    tokens, heads, dim, page_size = 4, 2, 128, 64
    seq_lens = torch.tensor([1000, 700, 300, 64], dtype=torch.int32, device="cuda")
    q = torch.randn(tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(tokens, heads, device="cuda", dtype=torch.bfloat16) + 0.5
    slots, row_bytes = page_size * 8, dim + 4  # 8 pages of 64 slots
    cache = torch.randint(
        0x38, 0x48, (slots, row_bytes), dtype=torch.uint8, device="cuda"
    )
    cache[:, dim:] = 127  # identity e8m0 scales
    pages = slots // page_size
    table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(
        tokens, 1
    )[:, : max(1, (int(seq_lens.max()) + page_size - 1) // page_size)]
    if table.shape[1] == 0:
        table = torch.zeros(tokens, 1, dtype=torch.int32, device="cuda")
    out, lens = ts_dsa_topk.launch_gluon_dsa_decode_topk_fp8_gfx950(
        q, weights, seq_lens, table.contiguous(),
        page_size=page_size, topk=512, softmax_scale=dim**-0.5,
        index_k_cache=cache,
    )
    assert out.shape == (tokens, 512) and out.dtype == torch.int32
    for t in range(tokens):
        n = min(int(seq_lens[t]), 512)
        assert int(lens[t]) == n, t
        assert bool((out[t, :n] >= 0).all() and (out[t, :n] < int(seq_lens[t])).all())
        assert bool((out[t, n:] == -1).all())


def _kda_fused_verify_reference(
    mixed_qkv, conv_weights, conv_pool, raw_g, beta_logits, a_log, dt_bias,
    state_pool, read_indices, draft_token_num, lower_bound=None,
):
    """Eager fp32 oracle for ``gluon_kda_fused_verify_gfx950`` (no-store)."""
    batch = read_indices.numel()
    tokens = batch * draft_token_num
    heads, dim = 12, 128
    pw = heads * dim
    out = torch.zeros(1, tokens, heads, dim, dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    a_value = torch.exp(a_log.float())
    for n in range(batch):
        ridx = int(read_indices[n])
        if ridx < 0 or ridx >= state_pool.shape[0]:
            continue
        qkv_rows = []
        h0 = conv_pool[ridx, :, 0].float()
        h1 = conv_pool[ridx, :, 1].float()
        h2 = conv_pool[ridx, :, 2].float()
        w = conv_weights.float()  # [3*pw, 4]
        for t in range(draft_token_num):
            tok = n * draft_token_num + t
            x = mixed_qkv[tok].float()
            value = h0 * w[:, 0] + h1 * w[:, 1] + h2 * w[:, 2] + x * w[:, 3]
            qkv_rows.append(value * (1.0 / (1.0 + torch.exp(-value))))
            h0, h1, h2 = h1, h2, x
        for h in range(heads):
            running = state_pool[ridx, h].float().clone()
            for t in range(draft_token_num):
                tok = n * draft_token_num + t
                row = qkv_rows[t]
                q = row[h * dim : (h + 1) * dim]
                k = row[pw + h * dim : pw + (h + 1) * dim]
                v = row[2 * pw + h * dim : 2 * pw + (h + 1) * dim]
                gate = raw_g[tok, h * dim : (h + 1) * dim].float() + dt_bias[
                    h * dim : (h + 1) * dim
                ].float()
                if lower_bound is not None:
                    log_decay = lower_bound / (1.0 + torch.exp(-(a_value[h] * gate)))
                else:
                    softplus = torch.maximum(gate, torch.zeros_like(gate)) + torch.log(
                        1.0 + torch.exp(-torch.abs(gate))
                    )
                    log_decay = -a_value[h] * softplus
                decay = torch.exp(log_decay)
                q_scale = torch.rsqrt((q * q).sum() + 1e-6) * (dim**-0.5)
                k_scale = torch.rsqrt((k * k).sum() + 1e-6)
                qh = q * q_scale
                kh = k * k_scale
                key_query = (qh * kh).sum()
                beta = torch.sigmoid(beta_logits[tok, h].float())
                running *= decay[None, :]
                prediction = (running * kh[None, :]).sum(dim=1)
                delta = beta * (v - prediction)
                prior_output = (running * qh[None, :]).sum(dim=1)
                running += delta[:, None] * kh[None, :]
                out[0, tok, h] = (prior_output + delta * key_query).to(mixed_qkv.dtype)
    return out


@pytest.mark.gpu
def test_gpu_kda_fused_verify_matches_reference():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import kda_decode as ts_kda

    torch.manual_seed(17)
    batch, heads, dim, draft = 4, 12, 128, 3
    pw = heads * dim
    pages = batch + 3
    tokens = batch * draft
    mixed_qkv = torch.randn(tokens, 3 * pw, device="cuda", dtype=torch.bfloat16) * 0.5
    conv_weights = torch.randn(3 * pw, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    conv_pool = torch.randn(pages, 3 * pw, 3, device="cuda", dtype=torch.bfloat16) * 0.3
    raw_g = torch.randn(tokens, pw, device="cuda", dtype=torch.bfloat16) * 0.5
    beta_logits = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda")
    dt_bias = torch.randn(pw, device="cuda") * 0.1
    state_pool = torch.randn(pages, heads, dim, dim, device="cuda") * 0.05
    read_indices = torch.tensor([0, 1, -1, 3], dtype=torch.int32, device="cuda")
    ref = _kda_fused_verify_reference(
        mixed_qkv, conv_weights, conv_pool, raw_g, beta_logits, a_log, dt_bias,
        state_pool, read_indices, draft,
    )
    pool_before = state_pool.clone()
    out = ts_kda.gluon_kda_fused_verify_gfx950(
        mixed_qkv, conv_weights, conv_pool, None, raw_g, beta_logits, a_log,
        dt_bias, state_pool=state_pool, state_scratch=None,
        read_indices=read_indices, write_indices=read_indices,
        num_heads=heads, head_dim=dim, draft_token_num=draft,
        lower_bound=None,
    )
    # no-store contract: the pools must be untouched
    assert torch.equal(state_pool, pool_before)
    assert out.shape == (1, tokens, heads, dim)
    valid = read_indices >= 0
    valid_tok = valid.repeat_interleave(draft)
    assert torch.equal(out[0, ~valid_tok], torch.zeros_like(out[0, ~valid_tok]))
    assert torch.allclose(
        out[0][valid_tok].float(), ref[0][valid_tok].float(), atol=3e-2, rtol=3e-2
    )


def _kda_replay_reference(
    per_layer, group_indices, read_indices, write_indices, accepted_length,
    draft_token_num, lower_bound=None,
):
    """Eager fp32 oracle for ``launch_gluon_kda_fused_replay_gfx950``.

    ``per_layer[l]`` holds one layer's tensors (mixed_qkv, conv_weights,
    conv_pool, raw_g, beta_logits, a_log, dt_bias, state_pool); the pools
    are updated IN PLACE. ``read/write_indices [groups, batch]`` hold
    ABSOLUTE page ids; ``group_indices[layer]`` selects each layer's row.
    """
    device = per_layer[0][0].device
    batch = accepted_length.numel()
    heads, dim = 12, 128
    pw = heads * dim
    for layer, tensors in enumerate(per_layer):
        (mixed_qkv, conv_weights, conv_pool, raw_g, beta_logits, a_log,
         dt_bias, state_pool) = tensors
        group = int(group_indices[layer])
        a_value = torch.exp(a_log.float())
        for n in range(batch):
            ridx = int(read_indices[group, n])
            widx = int(write_indices[group, n])
            steps = int(accepted_length[n])
            if ridx < 0 or steps <= 0:
                continue
            conv_rows = []
            h0 = conv_pool[ridx, :, 0].float()
            h1 = conv_pool[ridx, :, 1].float()
            h2 = conv_pool[ridx, :, 2].float()
            w_all = conv_weights.float()
            for t in range(draft_token_num):
                tok = n * draft_token_num + t
                active = t < steps
                x = mixed_qkv[tok].float() if active else torch.zeros(3 * pw, device=device)
                value = h0 * w_all[:, 0] + h1 * w_all[:, 1] + h2 * w_all[:, 2] + x * w_all[:, 3]
                silu = value * (1.0 / (1.0 + torch.exp(-value)))
                if t == steps - 1 and widx >= 0:
                    conv_pool[widx, :, 0] = h1.to(conv_pool.dtype)
                    conv_pool[widx, :, 1] = h2.to(conv_pool.dtype)
                    conv_pool[widx, :, 2] = x.to(conv_pool.dtype)
                if active:
                    conv_rows.append(silu)
                    h0, h1, h2 = h1, h2, x
            for h in range(heads):
                running = state_pool[ridx, h].float().clone()
                for t in range(steps):
                    tok = n * draft_token_num + t
                    row = conv_rows[t]
                    q = row[h * dim : (h + 1) * dim]
                    k = row[pw + h * dim : pw + (h + 1) * dim]
                    v = row[2 * pw + h * dim : 2 * pw + (h + 1) * dim]
                    gate = raw_g[tok, h * dim : (h + 1) * dim].float() + dt_bias[
                        h * dim : (h + 1) * dim
                    ].float()
                    if lower_bound is not None:
                        log_decay = lower_bound / (1.0 + torch.exp(-(a_value[h] * gate)))
                    else:
                        softplus = torch.maximum(gate, torch.zeros_like(gate)) + torch.log(
                            1.0 + torch.exp(-torch.abs(gate))
                        )
                        log_decay = -a_value[h] * softplus
                    decay = torch.exp(log_decay)
                    k_scale = torch.rsqrt((k * k).sum() + 1e-6)
                    kh = k * k_scale
                    beta = torch.sigmoid(beta_logits[tok, h].float())
                    running *= decay[None, :]
                    prediction = (running * kh[None, :]).sum(dim=1)
                    delta = beta * (v - prediction)
                    running += delta[:, None] * kh[None, :]
                if widx >= 0:
                    state_pool[widx, h] = running


@pytest.mark.gpu
def test_gpu_kda_fused_replay_matches_reference():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import kda_decode as ts_kda

    torch.manual_seed(18)
    batch, heads, dim, draft, layers, groups = 3, 12, 128, 4, 2, 2
    pw = heads * dim
    tokens = batch * draft
    per_layer = []
    descriptors = torch.zeros(layers, 10, dtype=torch.uint64, device="cuda")
    group_indices = torch.arange(layers, dtype=torch.int32, device="cuda")
    # each group's row holds its absolute page ids; group_indices picks the row
    read_indices = torch.arange(
        groups * batch, dtype=torch.int32, device="cuda"
    ).reshape(groups, batch)
    write_indices = read_indices.clone()
    accepted = torch.tensor([2, 4, 1], dtype=torch.int32, device="cuda")
    for layer in range(layers):
        pages = groups * batch + 2
        mixed_qkv = torch.randn(tokens, 3 * pw, device="cuda", dtype=torch.bfloat16) * 0.5
        conv_weights = torch.randn(3 * pw, 4, device="cuda", dtype=torch.bfloat16) * 0.2
        conv_pool = torch.randn(pages, 3 * pw, 3, device="cuda", dtype=torch.bfloat16) * 0.3
        raw_g = torch.randn(tokens, pw, device="cuda", dtype=torch.bfloat16) * 0.5
        beta_logits = torch.randn(tokens, heads, device="cuda", dtype=torch.bfloat16)
        a_log = torch.randn(heads, device="cuda")
        dt_bias = torch.randn(pw, device="cuda") * 0.1
        state_pool = torch.randn(pages, heads, dim, dim, device="cuda") * 0.05
        per_layer.append(
            (mixed_qkv, conv_weights, conv_pool, raw_g, beta_logits, a_log,
             dt_bias, state_pool)
        )
        for slot, tensor in zip(
            (0, 1, 2, 5, 6, 7, 8, 9),
            (mixed_qkv, conv_weights, conv_pool, beta_logits, a_log, dt_bias,
             state_pool, raw_g),
        ):
            descriptors[layer, slot] = int(tensor.data_ptr()) & 0xFFFFFFFFFFFFFFFF

    # reference runs on fresh clones of the pre-kernel pools
    ref_layer = [
        (t[0], t[1], t[2].clone(), t[3], t[4], t[5], t[6], t[7].clone())
        for t in per_layer
    ]
    _kda_replay_reference(
        ref_layer, group_indices.cpu(), read_indices.cpu(), write_indices.cpu(),
        accepted.cpu(), draft, lower_bound=0.05,
    )
    ts_kda.launch_gluon_kda_fused_replay_gfx950(
        descriptors, group_indices, read_indices, write_indices, accepted,
        draft_token_num=draft, num_heads=heads, head_dim=dim, f_a_dim=dim,
        qkv_stride=per_layer[0][0].stride(0), conv_stride=per_layer[0][2].stride(0),
        f_a_stride=dim, beta_stride=per_layer[0][4].stride(0),
        state_stride=per_layer[0][7].stride(0), gate_stride=per_layer[0][3].stride(0),
        conv_width=4, lower_bound=0.05,
    )
    torch.cuda.synchronize()
    for layer in range(layers):
        assert torch.equal(per_layer[layer][2], ref_layer[layer][2]), (
            "conv pool",
            layer,
        )
        assert torch.allclose(
            per_layer[layer][7].float(), ref_layer[layer][7].float(),
            atol=2e-3, rtol=1e-3,
        ), ("state pool", layer)


@pytest.mark.gpu
def test_gpu_kda_fused_decode_smoke():
    """Compile/launch smoke for the vendored KDA fused decode on gfx942.

    The kernel is the bucket-2 first target (cdna3 buffer-op binding).
    Full parity against ``glm_kda_fused_decode_reference`` needs the
    low-rank decay-gate input conversion (``a = f_a_out @ f_b_weight.T``
    rounded to bf16) — tracked as the bucket-2 follow-up; this test pins
    the launch contract and sane outputs meanwhile.
    """
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import kda_decode as ts_kda

    torch.manual_seed(12)
    batch, heads, dim = 4, 12, 128  # kernel hard-specializes K3: 12 heads x 128
    state_pages = 8
    mixed_qkv = torch.randn(batch, 3 * heads * dim, device="cuda", dtype=torch.bfloat16)
    conv_weights = torch.randn(3 * heads * dim, 4, device="cuda", dtype=torch.bfloat16) * 0.2
    conv_states = torch.randn(
        state_pages, 3 * heads * dim, 3, device="cuda", dtype=torch.bfloat16
    )
    f_a_out = torch.randn(batch, dim, device="cuda", dtype=torch.bfloat16)
    f_b_weight = torch.randn(
        heads * dim, dim, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    beta_logits = torch.randn(batch, heads, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.randn(heads * dim, device="cuda", dtype=torch.float32) * 0.1
    output_gate = torch.randn(batch, heads * dim, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(dim, device="cuda", dtype=torch.bfloat16)
    state_pool = torch.randn(
        state_pages, heads, dim, dim, device="cuda", dtype=torch.float32
    ) * 0.05
    before = state_pool.clone()
    read_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    write_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.arange(
        0, batch + 1, dtype=torch.int32, device="cuda"
    )
    out = ts_kda.gluon_kda_fused_decode_gfx950(
        mixed_qkv, conv_weights, conv_states, f_a_out, f_b_weight,
        beta_logits, a_log, dt_bias, output_gate, norm_weight, 1e-5,
        state_pool=state_pool, read_indices=read_indices,
        write_indices=write_indices, num_heads=heads, head_dim=dim,
        cu_seqlens=cu_seqlens, lower_bound=None,
    )
    assert out.shape == (1, batch, heads, dim)
    assert torch.isfinite(out.float()).all()
    # state rows written by batch slots changed; untouched pages did not
    changed = (state_pool != before).any(dim=(1, 2, 3))
    assert bool(changed[:batch].any())
    assert not bool(changed[batch:].any())
    assert torch.isfinite(state_pool).all()


@pytest.mark.gpu
def test_gpu_mhc_prenorm_gemm_matches_reference():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import mhc_gemm as ts_mhc_gemm

    torch.manual_seed(10)
    for tokens, hidden in ((8, 4096), (64, 4096), (3, 7168)):
        hc, splits = 4, hidden * 4 // 64
        x = torch.randn(tokens, hc * hidden, device="cuda", dtype=torch.bfloat16)
        fn = torch.randn(24, hc * hidden, device="cuda", dtype=torch.float32)
        out_mul = torch.empty(splits, tokens, 24, device="cuda")
        out_sqrsum = torch.empty(splits, tokens, device="cuda")
        ts_mhc_gemm.mhc_prenorm_gemm(x, fn, out_mul, out_sqrsum, splits)
        ref = x.float() @ fn.T
        assert torch.allclose(out_mul.sum(0), ref, atol=2e-2 * hidden, rtol=1e-3), (
            tokens,
            hidden,
        )
        assert torch.allclose(
            out_sqrsum.sum(0), (x.float() ** 2).sum(-1), atol=2.0, rtol=1e-3
        ), (tokens, hidden)
        # per-split partials must sum to the reference (split coverage exact)
        k_per_split = (hc * hidden + splits - 1) // splits
        k_per_split = (k_per_split + 63) // 64 * 64
        for s in (0, splits - 1):
            lo, hi = s * k_per_split, min((s + 1) * k_per_split, hc * hidden)
            if lo >= hc * hidden:
                continue
            part = x.float()[:, lo:hi] @ fn[:, lo:hi].T
            assert torch.allclose(out_mul[s], part, atol=2e-2 * hidden, rtol=1e-3)


@pytest.mark.gpu
def test_gpu_mhc_full_chain_matches_reference():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import mhc_gemm as ts_mhc_gemm

    torch.manual_seed(11)
    tokens, hidden = 8, 4096
    residual = torch.randn(tokens, 4, hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 4 * hidden, device="cuda", dtype=torch.float32)
    hc_scale = torch.randn(3, device="cuda")
    hc_base = torch.randn(24, device="cuda")
    post_mix, _comb_mix, layer_input = ts_mhc_gemm.gluon_mhc_pre(
        residual, fn, hc_scale, hc_base
    )
    # reference: proj = (residual @ fn.T) then the verified mhc chain
    proj = residual.reshape(tokens, -1).float() @ fn.T
    inv_rms = torch.rsqrt(
        (residual.float() ** 2).sum((1, 2)) / (4 * hidden) + 1e-6
    )
    pre = (
        1.0
        / (
            1.0
            + torch.exp(
                -(proj[:, :4] * inv_rms[:, None] * hc_scale[0] + hc_base[:4])
            )
        )
        + 1e-4
    )
    want_layer = (pre[:, :, None] * residual.float()).sum(1)
    assert torch.allclose(
        layer_input.float(), want_layer.to(torch.bfloat16).float(), atol=5e-2, rtol=2e-2
    )
    post = 2.0 / (
        1.0
        + torch.exp(-(proj[:, 4:8] * inv_rms[:, None] * hc_scale[1] + hc_base[4:8]))
    )
    assert torch.allclose(post_mix, post, atol=5e-3, rtol=1e-2)


@pytest.mark.gpu
def test_gpu_mhc_pre_matches_reference_chain():
    _gluon_driver()
    from vkernels.torch_ops.tokenspeed_amd import mhc as ts_mhc

    torch.manual_seed(9)
    tokens, hidden, splits = 8, 4096, 64
    mul = torch.randn(splits, tokens, 24, device="cuda", dtype=torch.float32)
    sqrsum = torch.rand(splits, tokens, device="cuda", dtype=torch.float32)
    hc_scale = torch.randn(3, device="cuda", dtype=torch.float32)
    hc_base = torch.randn(24, device="cuda", dtype=torch.float32)
    residual = torch.randn(tokens, 4, hidden, device="cuda", dtype=torch.float32)
    layer_input = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    post_mix = torch.empty(tokens, 4, device="cuda", dtype=torch.float32)
    comb_mix = torch.empty(tokens, 16, device="cuda", dtype=torch.float32)
    ts_mhc.gluon_mhc_pre_reduce_apply_gfx950(
        mul, sqrsum, hc_scale, hc_base, residual, layer_input,
        post_mix, comb_mix, hidden, rms_eps=1e-6, hc_eps=1e-4,
        sinkhorn_iters=20, n_splits=splits, num_tokens=tokens,
    )
    # torch reference from the kernel's math
    total = sqrsum.sum(0)
    inv_rms = torch.rsqrt(total / (4 * hidden) + 1e-6)
    proj = mul.sum(0)  # [tokens, 24]
    pre = (
        1.0 / (1.0 + torch.exp(-(proj[:, :4] * inv_rms[:, None] * hc_scale[0] + hc_base[:4])))
        + 1e-4
    )
    want_layer = (pre[:, :, None] * residual).sum(1)
    assert torch.allclose(
        layer_input.float(), want_layer.to(torch.bfloat16).float(), atol=2e-2, rtol=1e-2
    )
    post = 2.0 / (
        1.0 + torch.exp(-(proj[:, 4:8] * inv_rms[:, None] * hc_scale[1] + hc_base[4:8]))
    )
    assert torch.allclose(post_mix, post, atol=2e-3, rtol=1e-3)
    # comb: softmax (row-max stable, +eps after row normalize), then 20
    # Sinkhorn iterations of row normalize (+eps) / col normalize (+eps).
    eps = 1e-4
    comb = proj[:, 8:].reshape(tokens, 4, 4) * inv_rms[:, None, None] * hc_scale[2]
    comb = comb + hc_base[8:].reshape(4, 4)
    comb = torch.exp(comb - comb.max(dim=-1, keepdim=True).values)
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    # Verified on MI300A (triton 3.8): the gluon kernel places HC_EPS in the
    # denominator during the Sinkhorn loop (matching the upstream Triton
    # reference), not as a post-division add.
    for _ in range(19):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    got_comb = comb_mix.reshape(tokens, 4, 4)
    assert torch.allclose(got_comb, comb, atol=5e-3, rtol=1e-2), (
        (got_comb - comb).abs().max()
    )
