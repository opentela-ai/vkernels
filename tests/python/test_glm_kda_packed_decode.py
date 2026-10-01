"""glm_kda_packed_decode: the vendored SGLang/K3 CUDA packed-decode vs oracles.

The CUDA kernel (torch_ops/glm_kda_packed_decode.py, SGLang v0.5.20 /
NVIDIA x Moonshot K3 package, dependency-free CUDA-JIT port) replaces the
Triton fused-recurrent decode at batched decode: row-streaming state
update (one warp per V-row group) reaching the in-place state bandwidth
(~9.6 TB/s on H100 vs ~5 for the register-tile Triton class).

Contracts tested:
- vs the eager fp32 oracle (module reference): output within bf16 ULPs and
  state within fp32 ULPs (the donor's ULP class — warp-shuffle vs tl.sum
  reduction order; observed tighter: outputs usually bit-equal);
- vs the incumbent Triton ``glm_kda_decode`` fed with the equivalent
  floe-chain inputs (log-gates + post-sigmoid beta, K-major state) — the
  cross-kernel equivalence the floe wiring will rely on;
- in-place pool semantics: selected rows updated, untouched rows
  bit-identical, ``-1`` padded graph slots zero the output and skip the
  pool row;
- both gate branches (softplus no-lower-bound / lower-bound sigmoid), GQA
  head mapping (HV > H), state layout adapters round-trip, graph capture +
  replay, eligibility contracts, and the JIT off-switch.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.glm_kda_packed_decode import (  # noqa: E402
    glm_kda_packed_decode,
    glm_kda_packed_decode_eligible,
    glm_kda_packed_decode_reference,
    kda_state_kmajor_to_vmajor,
    kda_state_vmajor_to_kmajor,
)


def _inputs(device, b=3, h=16, hv=None, slots=8, seed=0, qkv_scale=4.0):
    """GLM-5.3-Flash TP4 shaped (H=HV=16) plus GQA variants."""
    hv = hv or h
    gen = torch.Generator().manual_seed(seed)
    dev = torch.device(device)

    def r(*shape, s=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=gen) * s).to(dtype)
    mixed = r(b, 2 * h * 128 + hv * 128, s=1 / qkv_scale, dtype=torch.bfloat16).to(dev)
    a = r(b, hv * 128, s=0.125, dtype=torch.bfloat16).to(dev)
    bb = r(b, hv, s=0.125, dtype=torch.bfloat16).to(dev)
    a_log = r(hv, s=0.1).to(dev)
    dt = r(hv * 128, s=0.1).to(dev)
    pool = r(slots, hv, 128, 128, s=0.125).to(dev)
    idx = torch.randperm(slots, generator=gen)[:b].to(torch.int32).to(dev)
    return mixed, a, bb, a_log, dt, pool, idx


# ---------------------------------------------------------------------------
# JIT availability gate for the whole file
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")


def _jit_available():
    if torch.cuda.is_available() and os.environ.get("VK_CUDA_JIT", "1") != "0":
        try:
            from vkernels.torch_ops.glm_kda_packed_decode import _lib_for

            _lib_for(torch.cuda.current_device())
            return True
        except Exception:
            return False
    return False


requires_jit = pytest.mark.skipif(not _jit_available(), reason="nvcc/CUDA-JIT unavailable")


# ---------------------------------------------------------------------------
# parity vs the eager oracle
# ---------------------------------------------------------------------------


@requires_jit
@pytest.mark.parametrize("lower_bound", [None, -0.05], ids=["softplus", "lb-sigmoid"])
@pytest.mark.parametrize("b,h,hv", [(1, 16, 16), (3, 16, 16), (8, 16, 16), (5, 4, 16)])
def test_parity_vs_reference(b, h, hv, lower_bound, seed=1):
    """fp32-oracle parity: out within bf16 ULPs, state within fp32 ULPs."""
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cuda", b=b, h=h, hv=hv, seed=seed)
    scale = 128 ** -0.5
    pool2 = pool.clone()
    out = glm_kda_packed_decode(
        mixed, a, bb, a_log, dt, scale, pool2, idx, h, lower_bound=lower_bound
    )
    ref_out, ref_state = glm_kda_packed_decode_reference(
        mixed, a, bb, a_log, dt, scale, pool, idx, h, lower_bound=lower_bound
    )
    # ULP-class gates (donor: "match to ULPs, not bits")
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool2, ref_state, rtol=1e-5, atol=1e-5)
    # untouched rows stay bit-identical
    touched = torch.zeros(pool.shape[0], dtype=torch.bool, device=pool.device)
    touched[torch.as_tensor(idx.tolist(), device=pool.device)] = True
    assert torch.equal(pool2[~touched], pool[~touched])


@requires_jit
def test_parity_vs_triton_kda_decode(seed=2):
    """Cross-kernel equivalence: the CUDA kernel vs the incumbent Triton
    glm_kda_decode fed the equivalent floe-chain inputs (log-gates g with
    decay=exp(g), POST-sigmoid beta, K-major state)."""
    from vkernels.torch_ops.glm_kda_decode import kda_decode

    b, h, hv = 4, 16, 16
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cuda", b=b, h=h, hv=hv, seed=seed)
    scale = 128 ** -0.5

    # floe-chain equivalents of the raw donor inputs: the CUDA kernel derives
    # decay from RAW a (bf16) + dt_bias in fp32; the Triton kernel takes
    # bf16 LOG-gates (decay = exp(g)) — the extra bf16 round of g here is the
    # one intentional rounding-point difference between the two paths
    x = a.float().view(b, hv, 128) + dt.float().view(hv, 128)  # [b, hv, k]
    exp_a = torch.exp(a_log.float()).view(1, hv, 1)
    sp = torch.where(x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x)
    g_log = (-exp_a * sp).to(torch.bfloat16).view(b, 1, hv, 128)
    beta_post = torch.sigmoid(bb.float()).view(b, 1, hv).to(torch.bfloat16)

    q = mixed[:, : h * 128].view(b, 1, h, 128)
    k = mixed[:, h * 128 : 2 * h * 128].view(b, 1, h, 128)
    v = mixed[:, 2 * h * 128 :].view(b, 1, hv, 128)

    pool2 = pool.clone()
    out_cuda = glm_kda_packed_decode(mixed, a, bb, a_log, dt, scale, pool2, idx, h)

    # Triton path: per-request states, K-major
    kmajor = kda_state_vmajor_to_kmajor(pool)
    outs_t, nexts = [], []
    for n in range(b):
        o_t, st = kda_decode(
            q[n : n + 1].contiguous(), k[n : n + 1].contiguous(), v[n : n + 1].contiguous(),
            g_log[n : n + 1].contiguous(), beta_post[n : n + 1].contiguous(),
            kmajor[idx[n] : idx[n] + 1].contiguous(),
        )
        outs_t.append(o_t)
        nexts.append(st)
    out_t = torch.cat(outs_t, 0)
    next_t = torch.cat(nexts, 0)

    torch.testing.assert_close(out_cuda.float(), out_t.float(), rtol=1e-2, atol=1e-2)
    # state: the Triton path rounds g through bf16 (decay=exp(g_bf16)) while
    # the CUDA path keeps decay fp32 — a ~0.4% decay error accumulates into
    # the state; gate at that documented class (vs-reference stays 1e-5)
    cuda_next = kda_state_vmajor_to_kmajor(pool2)[[int(i) for i in idx.tolist()]]
    torch.testing.assert_close(cuda_next, next_t, rtol=2e-2, atol=2e-3)


# ---------------------------------------------------------------------------
# pool semantics: padded slots, in-place, adapters
# ---------------------------------------------------------------------------


@requires_jit
def test_padded_graph_slots(seed=3):
    """-1 indices: zero output, pool row untouched (the CUDA-graph contract)."""
    b, h = 4, 16
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cuda", b=b, h=h, seed=seed)
    idx = idx.clone()
    idx[1] = -1
    idx[3] = -1
    pool2 = pool.clone()
    out = glm_kda_packed_decode(mixed, a, bb, a_log, dt, 128 ** -0.5, pool2, idx, h)
    assert torch.count_nonzero(out[1]).item() == 0
    assert torch.count_nonzero(out[3]).item() == 0
    assert torch.count_nonzero(out[0]).item() > 0
    touched = torch.zeros(pool.shape[0], dtype=torch.bool, device=pool.device)
    for i in idx.tolist():
        if i >= 0:
            touched[i] = True
    assert torch.equal(pool2[~touched], pool[~touched])


@requires_jit
def test_state_adapters_roundtrip():
    pool = torch.randn(4, 16, 128, 128, dtype=torch.float32)
    vmaj = kda_state_kmajor_to_vmajor(pool)
    assert vmaj.shape == (4, 16, 128, 128) and vmaj.is_contiguous()
    assert torch.equal(kda_state_vmajor_to_kmajor(vmaj), pool)
    # the transposition really flips axes, not just copies
    assert torch.equal(vmaj[0, 0], pool[0, 0].T)
    with pytest.raises(OpNotEligible):
        kda_state_kmajor_to_vmajor(torch.zeros(4, 16, 128, 128, dtype=torch.bfloat16))


def test_duplicate_indices_are_ub():
    """Two requests hitting the same pool slot in ONE launch race: the grid
    is one CTA per (n, hv) pair with no cross-CTA ordering, so duplicate
    slot indices have unspecified state results. floe's pool allocator
    never routes duplicates (one slot per live request); documented here so
    the wiring side never assumes sequential semantics."""
    pytest.skip("duplicate slots are UB by design (racing CTAs); documentation")


# ---------------------------------------------------------------------------
# capture + PDL + eligibility
# ---------------------------------------------------------------------------


@requires_jit
def test_graph_capture_replay(seed=5):
    b, h = 3, 16
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cuda", b=b, h=h, seed=seed)
    pool2 = pool.clone()
    scale = 128 ** -0.5
    out = torch.empty(b, 1, h, 128, dtype=torch.bfloat16, device="cuda")
    glm_kda_packed_decode(mixed, a, bb, a_log, dt, scale, pool2, idx, h, out=out)  # warm + compile

    pool3 = pool.clone()
    out3 = torch.empty_like(out)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        glm_kda_packed_decode(mixed, a, bb, a_log, dt, scale, pool3, idx, h, out=out3)
    torch.cuda.current_stream().wait_stream(s)
    # the side-stream warmup MUTATED pool3 (in-place op) — reset so the
    # first replay starts from the pristine state like the eager reference
    pool3.copy_(pool)
    with torch.cuda.graph(g):
        glm_kda_packed_decode(mixed, a, bb, a_log, dt, scale, pool3, idx, h, out=out3)
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out3.float(), out.float(), rtol=0, atol=0)
    torch.testing.assert_close(pool3, pool2, rtol=0, atol=0)

    # replay mutates inputs in place and tracks them
    pool3.copy_(pool)
    mixed.mul_(2.0)
    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_state = glm_kda_packed_decode_reference(mixed, a, bb, a_log, dt, scale, pool, idx, h)
    torch.testing.assert_close(out3.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool3, ref_state, rtol=1e-5, atol=1e-5)


@requires_jit
def test_pdl_launch(seed=6):
    """PDL-enabled launch: same outputs (sm_90+; fails cleanly elsewhere)."""
    b, h = 2, 16
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cuda", b=b, h=h, seed=seed)
    scale = 128 ** -0.5
    pool_plain = pool.clone()
    pool_pdl = pool.clone()
    out_p = glm_kda_packed_decode(mixed, a, bb, a_log, dt, scale, pool_plain, idx, h)
    try:
        out_d = glm_kda_packed_decode(mixed, a, bb, a_log, dt, scale, pool_pdl, idx, h, use_pdl=True)
    except RuntimeError as e:
        pytest.skip(f"PDL launch unsupported here: {e}")
    assert torch.equal(out_p, out_d)
    assert torch.equal(pool_plain, pool_pdl)


def test_eligibility_cpu_and_contracts():
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cpu")
    if os.environ.get("VK_CUDA_JIT", "1") == "0":
        pytest.skip("VK_CUDA_JIT=0 flips these checks")
    assert not glm_kda_packed_decode_eligible(mixed, a, bb, a_log, dt, pool, idx, 16)
    if torch.cuda.is_available():
        mixed_g, a_g, bb_g, a_log_g, dt_g, pool_g, idx_g = _inputs("cuda")
        assert glm_kda_packed_decode_eligible(mixed_g, a_g, bb_g, a_log_g, dt_g, pool_g, idx_g, 16)
        # wrong head division
        assert not glm_kda_packed_decode_eligible(mixed_g, a_g, bb_g, a_log_g, dt_g, pool_g, idx_g, 12)
        # NOTE: a K-major pool (floe's incumbent layout) is STRIDE-IDENTICAL
        # to V-major (square 128x128 per head) — eligibility cannot detect a
        # wrong-layout pool; the layout is a caller contract enforced at the
        # single wiring site (the adapters exist for exactly that boundary).
        # int64 indices reject
        assert not glm_kda_packed_decode_eligible(mixed_g, a_g, bb_g, a_log_g, dt_g, pool_g, idx_g.long(), 16)
        # fp16 mixed rejects (kernel specialized bf16)
        assert not glm_kda_packed_decode_eligible(
            mixed_g.half(), a_g, bb_g, a_log_g, dt_g, pool_g, idx_g, 16
        )


def test_reference_runs_on_cpu():
    mixed, a, bb, a_log, dt, pool, idx = _inputs("cpu", b=2, h=4, hv=4, seed=7)
    out, st = glm_kda_packed_decode_reference(mixed, a, bb, a_log, dt, 128 ** -0.5, pool, idx, 4)
    assert out.shape == (2, 1, 4, 128) and out.dtype == torch.bfloat16
    assert st.shape == pool.shape
    assert not torch.equal(st, pool)  # rows really advanced
