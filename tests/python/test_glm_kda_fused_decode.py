"""glm_kda_fused_decode: the vendored SGLang/K3 fused CUDA decode vs oracles.

The CUDA kernel (torch_ops/glm_kda_fused_decode.py, SGLang v0.5.20 /
NVIDIA x Moonshot K3 package, dependency-free CUDA-JIT port) replaces the
whole three-kernel GLM-5.3 KDA decode chain — conv-update + packed-decode +
gated RMSNorm — with ONE launch that also consumes the PRE-conv projection
rows directly (conv, SiLU and the conv-pool shift are in-kernel).

Contracts tested:
- vs the eager fp32 oracle (module reference): output within bf16 ULPs,
  state within fp32 ULPs, conv pool shift BIT-EXACT (pure bf16 moves);
- vs the unfused incumbent chain — conv rows rounded through bf16 (the
  conv-update store contract) -> glm_kda_packed_decode_reference -> eager
  sigmoid-gated RMSNorm — the cross-path equivalence class the floe wiring
  will rely on (the bf16 round of the conv rows is the intentional
  difference: the fused kernel keeps them fp32);
- pool semantics: selected rows updated, untouched rows bit-identical,
  ``-1`` padded graph slots zero the output row and skip BOTH pools, the
  conv shift lands the raw new token at w=2;
- row-strided mixed_qkv/a/b/onorm_g rows (the no-copy fused-GEMM-slice
  contract), envelope-strided state pools (TMA path with a non-dense slot
  pitch), the 3-stage TMA staging variant (B*HV >= 512), B == 0, graph
  capture + replay, PDL launch, eligibility contracts, adapters.
"""

import os

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.glm_kda_fused_decode import (  # noqa: E402
    glm_kda_fused_decode,
    glm_kda_fused_decode_eligible,
    glm_kda_fused_decode_reference,
    kda_conv_state_cmajor_to_wmajor,
    kda_conv_state_wmajor_to_cmajor,
    kda_conv_weight_to_taps,
)
from vkernels.torch_ops.glm_kda_packed_decode import (  # noqa: E402
    glm_kda_packed_decode_reference,
)

_SCALE = 128 ** -0.5
_EPS = 1.0e-5


def _inputs(device, b=3, h=16, slots=8, seed=0, biased=False):
    """GLM-5.3-Flash TP4 shaped (H = HV = 16, seg = 2048, conv_dim = 6144)."""
    seg = h * 128
    gen = torch.Generator().manual_seed(seed)
    dev = torch.device(device)

    def r(*shape, s=1.0, dtype=torch.float32):
        return (torch.randn(*shape, generator=gen) * s).to(dtype).to(dev)

    return dict(
        mixed=r(b, 3 * seg, s=0.5, dtype=torch.bfloat16),       # RAW pre-conv rows
        a=r(b, seg, s=0.125, dtype=torch.bfloat16),             # RAW f_b dots
        b=r(b, h, s=0.125, dtype=torch.bfloat16),               # RAW b_proj dots
        conv=r(slots, 3, 3 * seg, s=0.5, dtype=torch.bfloat16),  # TIME-major pool
        wq=r(4, seg, s=0.25),
        wk=r(4, seg, s=0.25),
        wv=r(4, seg, s=0.25),
        cbias=(r(3 * seg, s=0.25) if biased else torch.zeros(3 * seg).to(dev)),
        alog=r(h, s=0.1),
        dt=r(seg, s=0.1),
        og=r(b, seg, s=0.5, dtype=torch.bfloat16),              # RAW o-norm gate dots
        ow=r(128, s=0.25),
        pool=r(slots, h, 128, 128, s=0.125),                    # V-major ssm pool
        idx=torch.randperm(slots, generator=gen)[:b].to(torch.int32).to(dev),
    )


def _call(op, inp, pool, conv=None, lower_bound=None, out=None, use_pdl=False):
    kw = {"lower_bound": lower_bound}
    if op is glm_kda_fused_decode:
        kw.update(out=out, use_pdl=use_pdl)
    return op(
        inp["mixed"], inp["a"], inp["b"], inp["conv"] if conv is None else conv,
        inp["wq"], inp["wk"],
        inp["wv"], inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"],
        pool, inp["idx"], _SCALE, _EPS, **kw,
    )


def _eligible(inp, pool, **kw):
    return glm_kda_fused_decode_eligible(
        inp["mixed"], inp["a"], inp["b"], inp["conv"], inp["wq"], inp["wk"],
        inp["wv"], inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"],
        pool, inp["idx"], **kw,
    )


def _chain_oracle(inp, pool, lower_bound=None):
    """The unfused incumbent chain: conv rows with ONE bf16 store round
    (the conv-update contract) -> packed-decode reference (bf16 rows ABI)
    -> eager sigmoid-gated RMSNorm. Any device, pure torch."""
    mixed = inp["mixed"]
    b = mixed.shape[0]
    h = pool.shape[-3]
    w_all = torch.cat([inp["wq"], inp["wk"], inp["wv"]], dim=1).float()
    rows = torch.zeros_like(mixed)
    for n in range(b):
        sidx = int(inp["idx"][n])
        if sidx < 0:
            continue
        window = torch.cat([inp["conv"][sidx].float(), mixed[n].float().unsqueeze(0)], 0)
        acc = inp["cbias"] + window[0] * w_all[0] + window[1] * w_all[1] \
            + window[2] * w_all[2] + window[3] * w_all[3]
        # the conv-update store contract: one bf16 round, SiLU on the stored
        # value, one more round
        rows[n] = torch.nn.functional.silu(acc.to(torch.bfloat16).float()).to(torch.bfloat16)
    out_p, next_s = glm_kda_packed_decode_reference(
        rows, inp["a"], inp["b"], inp["alog"], inp["dt"], _SCALE, pool,
        inp["idx"], h, lower_bound=lower_bound,
    )
    o = out_p.float().squeeze(1)  # [B, HV, V]
    gate = torch.sigmoid(inp["og"].float().view(b, h, 128))
    y = o * torch.rsqrt((o * o).mean(-1, keepdim=True) + _EPS) * inp["ow"].float() * gate
    return y.to(torch.bfloat16), next_s


def _touched_mask(idx, slots, device):
    touched = torch.zeros(slots, dtype=torch.bool, device=device)
    touched[torch.as_tensor([int(i) for i in idx.tolist() if i >= 0], device=device)] = True
    return touched


# ---------------------------------------------------------------------------
# JIT availability gate for the GPU section
# ---------------------------------------------------------------------------

def _jit_available():
    if torch.cuda.is_available() and os.environ.get("VK_CUDA_JIT", "1") != "0":
        try:
            from vkernels.torch_ops.glm_kda_fused_decode import _lib_for

            _lib_for(torch.cuda.current_device())
            return True
        except Exception:
            return False
    return False


requires_jit = pytest.mark.skipif(not _jit_available(), reason="nvcc/CUDA-JIT unavailable")


# ---------------------------------------------------------------------------
# CPU-safe: the reference executor and the chain oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lower_bound", [None, -0.05], ids=["softplus", "lb-sigmoid"])
@pytest.mark.parametrize("biased", [False, True], ids=["nobias", "bias"])
def test_reference_runs_on_cpu(lower_bound, biased):
    inp = _inputs("cpu", b=2, h=4, seed=7, biased=biased)
    out, nconv, nstate = _call(glm_kda_fused_decode_reference, inp, inp["pool"], lower_bound=lower_bound)
    assert out.shape == (1, 2, 4, 128) and out.dtype == torch.bfloat16
    assert nconv.shape == inp["conv"].shape and nconv.dtype == torch.bfloat16
    assert nstate.shape == inp["pool"].shape
    assert not torch.equal(nstate, inp["pool"])  # rows really advanced
    assert not torch.equal(nconv, inp["conv"])  # shift really happened


def test_reference_conv_shift_is_pure_moves():
    """The conv pool shift must be bit-exact bf16 moves: [old w1, old w2, raw x]."""
    inp = _inputs("cpu", b=2, h=4, seed=8)
    _, nconv, _ = _call(glm_kda_fused_decode_reference, inp, inp["pool"])
    for n in range(2):
        sidx = int(inp["idx"][n])
        expect = torch.cat([inp["conv"][sidx][1:3], inp["mixed"][n].unsqueeze(0)], 0)
        assert torch.equal(nconv[sidx], expect)


def test_reference_pad_slots():
    inp = _inputs("cpu", b=4, h=4, seed=9)
    inp["idx"] = inp["idx"].clone()
    inp["idx"][1] = -1
    inp["idx"][3] = -1
    out, nconv, nstate = _call(glm_kda_fused_decode_reference, inp, inp["pool"])
    assert torch.count_nonzero(out[0, 1]).item() == 0
    assert torch.count_nonzero(out[0, 3]).item() == 0
    assert torch.count_nonzero(out[0, 0]).item() > 0
    assert torch.equal(nconv[1], inp["conv"][1]) and torch.equal(nconv[3], inp["conv"][3])
    assert torch.equal(nstate[1], inp["pool"][1]) and torch.equal(nstate[3], inp["pool"][3])


@pytest.mark.parametrize("lower_bound", [None, -0.05], ids=["softplus", "lb-sigmoid"])
def test_reference_vs_incumbent_chain(lower_bound):
    """The fused reference vs the unfused chain (conv rows through bf16 ->
    packed-decode reference -> eager norm). The bf16 round(s) of the conv
    rows are the intentional difference class (the fused kernel keeps the
    conv output fp32), so gate at the packed kernel's cross-path band."""
    inp = _inputs("cpu", b=3, h=8, seed=10)
    out_f, _, nstate_f = _call(glm_kda_fused_decode_reference, inp, inp["pool"], lower_bound=lower_bound)
    out_c, nstate_c = _chain_oracle(inp, inp["pool"], lower_bound)
    torch.testing.assert_close(out_f.float(), out_c.float().unsqueeze(0).float(), rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(nstate_f, nstate_c, rtol=2e-2, atol=2e-3)


def test_adapters():
    """floe-side layout adapters: conv weight [C,1,4] -> taps [4, C/3]; conv
    pool [slots, C, 3] <-> [slots, 3, C]."""
    c = 6144
    w = torch.randn(c, 1, 4).to(torch.bfloat16)
    wq, wk, wv = kda_conv_weight_to_taps(w)
    assert wq.shape == wk.shape == wv.shape == (4, 2048)
    assert wq.dtype == torch.float32 and wq.is_contiguous()
    # exact widen + transpose: taps[t, ch] == weight[ch, 0, t]
    assert torch.equal(wq, w[:2048].squeeze(1).t().float())
    assert torch.equal(wv, w[4096:].squeeze(1).t().float())
    with pytest.raises(OpNotEligible):
        kda_conv_weight_to_taps(torch.randn(c, 2, 4))
    pool = torch.randn(4, c, 3).to(torch.bfloat16)
    wm = kda_conv_state_cmajor_to_wmajor(pool)
    assert wm.shape == (4, 3, c) and wm.is_contiguous()
    assert torch.equal(wm[0, 1], pool[0, :, 1])
    assert torch.equal(kda_conv_state_wmajor_to_cmajor(wm), pool)


# ---------------------------------------------------------------------------
# GPU: parity vs the reference and the chain
# ---------------------------------------------------------------------------


@requires_jit
@pytest.mark.parametrize("lower_bound", [None, -0.05], ids=["softplus", "lb-sigmoid"])
@pytest.mark.parametrize("b,h", [(1, 16), (3, 16), (8, 16), (2, 4)])
@pytest.mark.parametrize("biased", [False, True], ids=["nobias", "bias"])
def test_parity_vs_reference(b, h, lower_bound, biased):
    """fp32-oracle parity: out within bf16 ULPs, state within fp32 ULPs,
    conv pool shift bit-exact."""
    inp = _inputs("cuda", b=b, h=h, seed=1, biased=biased)
    conv2 = inp["conv"].clone()
    pool2 = inp["pool"].clone()
    out = glm_kda_fused_decode(
        inp["mixed"], inp["a"], inp["b"], conv2, inp["wq"], inp["wk"], inp["wv"],
        inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool2,
        inp["idx"], _SCALE, _EPS, lower_bound=lower_bound,
    )
    ref_out, ref_conv, ref_state = glm_kda_fused_decode_reference(
        inp["mixed"], inp["a"], inp["b"], inp["conv"], inp["wq"], inp["wk"],
        inp["wv"], inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"],
        inp["pool"], inp["idx"], _SCALE, _EPS, lower_bound=lower_bound,
    )
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool2, ref_state, rtol=1e-5, atol=1e-5)
    assert torch.equal(conv2, ref_conv)  # the shift is pure bf16 moves
    touched = _touched_mask(inp["idx"], inp["conv"].shape[0], inp["conv"].device)
    assert torch.equal(conv2[~touched], inp["conv"][~touched])
    assert torch.equal(pool2[~touched], inp["pool"][~touched])


@requires_jit
@pytest.mark.parametrize("lower_bound", [None, -0.05], ids=["softplus", "lb-sigmoid"])
def test_parity_vs_incumbent_chain(lower_bound):
    """End-to-end kernel vs the unfused chain oracle: the bf16 round of the
    conv rows (chain) vs fp32 conv (kernel) is the documented band."""
    inp = _inputs("cuda", b=4, h=16, seed=2)
    conv2 = inp["conv"].clone()
    pool2 = inp["pool"].clone()
    out = glm_kda_fused_decode(
        inp["mixed"], inp["a"], inp["b"], conv2, inp["wq"], inp["wk"], inp["wv"],
        inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool2,
        inp["idx"], _SCALE, _EPS, lower_bound=lower_bound,
    )
    out_c, nstate_c = _chain_oracle(inp, inp["pool"], lower_bound)
    torch.testing.assert_close(out.float(), out_c.float().unsqueeze(0).float(), rtol=2e-2, atol=1e-2)
    torch.testing.assert_close(pool2, nstate_c, rtol=2e-2, atol=2e-3)


@requires_jit
def test_padded_graph_slots(seed=3):
    """-1 indices: zero output row, BOTH pools untouched."""
    inp = _inputs("cuda", b=4, h=16, seed=seed)
    inp["idx"] = inp["idx"].clone()
    inp["idx"][1] = -1
    inp["idx"][3] = -1
    conv2 = inp["conv"].clone()
    pool2 = inp["pool"].clone()
    out = glm_kda_fused_decode(
        inp["mixed"], inp["a"], inp["b"], conv2, inp["wq"], inp["wk"], inp["wv"],
        inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool2,
        inp["idx"], _SCALE, _EPS,
    )
    assert torch.count_nonzero(out[0, 1]).item() == 0
    assert torch.count_nonzero(out[0, 3]).item() == 0
    assert torch.count_nonzero(out[0, 0]).item() > 0
    touched = _touched_mask(inp["idx"], inp["conv"].shape[0], inp["conv"].device)
    assert torch.equal(conv2[~touched], inp["conv"][~touched])
    assert torch.equal(pool2[~touched], inp["pool"][~touched])


@requires_jit
def test_row_strided_rows_no_copy():
    """The fused-GEMM-slice contract: mixed_qkv/a/onorm_g rows may be wider
    row-strided views (stride(-1) == 1, stride(0) arbitrary) — no copies."""
    inp = _inputs("cuda", b=3, h=16, seed=4)
    pad = 512
    wide = torch.zeros(3, inp["mixed"].shape[1] + pad, dtype=torch.bfloat16, device="cuda")
    wide[:, : inp["mixed"].shape[1]] = inp["mixed"]
    mixed_v = wide[:, : inp["mixed"].shape[1]]
    assert mixed_v.stride(0) == inp["mixed"].shape[1] + pad
    og_wide = torch.zeros(3, inp["og"].shape[1] + pad, dtype=torch.bfloat16, device="cuda")
    og_wide[:, : inp["og"].shape[1]] = inp["og"]
    og_v = og_wide[:, : inp["og"].shape[1]]
    inp2 = dict(inp, mixed=mixed_v, og=og_v)
    assert _eligible(inp2, inp["pool"])
    conv2 = inp["conv"].clone()
    pool2 = inp["pool"].clone()
    out = _call(glm_kda_fused_decode, inp2, pool2, conv2)
    ref_out, ref_conv, ref_state = _call(glm_kda_fused_decode_reference, inp2, inp["pool"])
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool2, ref_state, rtol=1e-5, atol=1e-5)
    assert torch.equal(conv2, ref_conv)


@requires_jit
def test_envelope_strided_state_pool():
    """A non-dense slot pitch (the multi-layer envelope layout): the kernel
    reads/writes through state.stride(0) in place (TMA path — pitch % 4 == 0)."""
    inp = _inputs("cuda", b=3, h=16, seed=5)
    slots, hv, v, k = inp["pool"].shape
    dense = hv * v * k
    buf = torch.zeros(slots * 2 * dense, dtype=torch.float32, device="cuda")
    pool = buf.as_strided((slots, hv, v, k), (2 * dense, v * k, k, 1))
    pool.copy_(inp["pool"])
    assert pool.stride(0) == 2 * dense and _eligible(inp, pool)
    conv2 = inp["conv"].clone()
    out = _call(glm_kda_fused_decode, inp, pool, conv2)
    ref_out, _ref_conv, ref_state = _call(glm_kda_fused_decode_reference, inp, inp["pool"])
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool.contiguous(), ref_state, rtol=1e-5, atol=1e-5)
    # untouched envelope slots (the odd multiples) stay zero
    assert torch.count_nonzero(buf.view(slots, 2, dense)[:, 1]).item() == 0


@requires_jit
def test_three_stage_tma_path():
    """B*HV >= 512 selects the 3-stage TMA staging variant (one stage reuse
    behind an extra __syncthreads) — parity with the 4-stage default."""
    b, h = 32, 16
    inp = _inputs("cuda", b=b, h=h, slots=40, seed=6)
    assert b * h >= 512
    conv2 = inp["conv"].clone()
    pool2 = inp["pool"].clone()
    out = _call(glm_kda_fused_decode, inp, pool2, conv2)
    ref_out, ref_conv, ref_state = _call(glm_kda_fused_decode_reference, inp, inp["pool"])
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool2, ref_state, rtol=1e-5, atol=1e-5)
    assert torch.equal(conv2, ref_conv)


@requires_jit
def test_graph_capture_replay(seed=11):
    b, h = 3, 16
    inp = _inputs("cuda", b=b, h=h, seed=seed)
    conv2 = inp["conv"].clone()
    pool2 = inp["pool"].clone()
    out = glm_kda_fused_decode(
        inp["mixed"], inp["a"], inp["b"], conv2, inp["wq"], inp["wk"], inp["wv"],
        inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool2,
        inp["idx"], _SCALE, _EPS,
    )  # warm + compile

    conv3 = inp["conv"].clone()
    pool3 = inp["pool"].clone()
    out3 = torch.empty_like(out)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        glm_kda_fused_decode(
            inp["mixed"], inp["a"], inp["b"], conv3, inp["wq"], inp["wk"], inp["wv"],
            inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool3,
            inp["idx"], _SCALE, _EPS, out=out3.view(b, -1),
        )
    torch.cuda.current_stream().wait_stream(s)
    # the side-stream warmup MUTATED the pools (in-place op) — reset so the
    # first replay starts from the pristine state like the eager reference
    conv3.copy_(inp["conv"])
    pool3.copy_(inp["pool"])
    with torch.cuda.graph(g):
        glm_kda_fused_decode(
            inp["mixed"], inp["a"], inp["b"], conv3, inp["wq"], inp["wk"], inp["wv"],
            inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool3,
            inp["idx"], _SCALE, _EPS, out=out3.view(b, -1),
        )
    g.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out3.float(), out.float(), rtol=0, atol=0)
    torch.testing.assert_close(pool3, pool2, rtol=0, atol=0)
    torch.testing.assert_close(conv3.float(), conv2.float(), rtol=0, atol=0)

    # replay mutates inputs in place and tracks them
    pool3.copy_(inp["pool"])
    conv3.copy_(inp["conv"])
    inp["mixed"].mul_(2.0)
    g.replay()
    torch.cuda.synchronize()
    ref_out, ref_conv, ref_state = _call(glm_kda_fused_decode_reference, inp, inp["pool"])
    torch.testing.assert_close(out3.float(), ref_out.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(pool3, ref_state, rtol=1e-5, atol=1e-5)
    assert torch.equal(conv3, ref_conv)


@requires_jit
def test_pdl_launch(seed=12):
    """PDL-enabled launch: same outputs (sm_90+; fails cleanly elsewhere)."""
    inp = _inputs("cuda", b=2, h=16, seed=seed)
    conv_p, pool_p = inp["conv"].clone(), inp["pool"].clone()
    conv_d, pool_d = inp["conv"].clone(), inp["pool"].clone()
    out_p = glm_kda_fused_decode(
        inp["mixed"], inp["a"], inp["b"], conv_p, inp["wq"], inp["wk"], inp["wv"],
        inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool_p,
        inp["idx"], _SCALE, _EPS,
    )
    try:
        out_d = glm_kda_fused_decode(
            inp["mixed"], inp["a"], inp["b"], conv_d, inp["wq"], inp["wk"], inp["wv"],
            inp["cbias"], inp["alog"], inp["dt"], inp["og"], inp["ow"], pool_d,
            inp["idx"], _SCALE, _EPS, use_pdl=True,
        )
    except RuntimeError as e:
        pytest.skip(f"PDL launch unsupported here: {e}")
    assert torch.equal(out_p, out_d)
    assert torch.equal(pool_p, pool_d)
    assert torch.equal(conv_p, conv_d)


@requires_jit
def test_b0_empty_batch():
    inp = _inputs("cuda", b=3, h=16, seed=13)
    inp0 = dict(inp, mixed=inp["mixed"][:0], a=inp["a"][:0], b=inp["b"][:0],
                og=inp["og"][:0], idx=inp["idx"][:0])
    assert _eligible(inp0, inp["pool"])
    conv2, pool2 = inp["conv"].clone(), inp["pool"].clone()
    out = _call(glm_kda_fused_decode, inp0, pool2, conv2)
    assert out.shape == (1, 0, 16, 128)
    assert torch.equal(conv2, inp["conv"]) and torch.equal(pool2, inp["pool"])


def test_duplicate_indices_are_ub():
    """Two requests hitting the same pool slot in ONE launch race: the grid
    is one CTA per (n, hv) pair with no cross-CTA ordering, so duplicate
    slot indices have unspecified conv/ssm state results. floe's pool
    allocator never routes duplicates; documented here so the wiring side
    never assumes sequential semantics."""
    pytest.skip("duplicate slots are UB by design (racing CTAs); documentation")


# ---------------------------------------------------------------------------
# eligibility contracts
# ---------------------------------------------------------------------------


def test_eligibility_cpu_and_contracts():
    inp = _inputs("cpu", b=3, h=16)
    if os.environ.get("VK_CUDA_JIT", "1") == "0":
        pytest.skip("VK_CUDA_JIT=0 flips these checks")
    assert not _eligible(inp, inp["pool"])  # CPU tensors
    if not torch.cuda.is_available():
        return
    inp = _inputs("cuda", b=3, h=16)
    assert _eligible(inp, inp["pool"])
    # GQA-shaped rows (H != HV) reject: the K3 static decode layout is MHA
    gqa = _inputs("cuda", b=3, h=8)
    assert not _eligible(gqa, inp["pool"])
    # wrong pool head count rejects
    assert not _eligible(inp, _inputs("cuda", b=3, h=8, seed=1)["pool"])
    # int64 indices reject
    bad = dict(inp, idx=inp["idx"].long())
    assert not _eligible(bad, inp["pool"])
    # fp16 rows reject (kernel specialized bf16)
    bad = dict(inp, mixed=inp["mixed"].half())
    assert not _eligible(bad, inp["pool"])
    # fp32 taps reject
    bad = dict(inp, wq=inp["wq"].to(torch.bfloat16))
    assert not _eligible(bad, inp["pool"])
    # channel-major conv pool (floe's incumbent layout) rejects — unlike the
    # square ssm pool, the conv layout is shape-detectable
    bad = dict(inp, conv=kda_conv_state_cmajor_to_wmajor(inp["conv"]).transpose(1, 2))
    assert not _eligible(bad, inp["pool"])
    # non-16B-alignable ssm slot pitch rejects (cp.async/TMA 16B loads)
    slots, hv, v, k = inp["pool"].shape
    dense = hv * v * k
    buf = torch.zeros(slots * (dense + 1), dtype=torch.float32, device="cuda")
    pool = buf.as_strided((slots, hv, v, k), (dense + 1, v * k, k, 1))
    pool.copy_(inp["pool"])
    assert not _eligible(inp, pool)


@requires_jit
def test_call_rejects_bad_inputs_with_op_not_eligible():
    inp = _inputs("cuda", b=3, h=16, seed=14)
    with pytest.raises(OpNotEligible):
        _call(glm_kda_fused_decode, inp, inp["pool"].to(torch.bfloat16))
    with pytest.raises(OpNotEligible):
        _call(glm_kda_fused_decode, dict(inp, idx=inp["idx"].long()), inp["pool"])
