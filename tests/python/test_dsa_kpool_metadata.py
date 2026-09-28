"""fused_dsa_{decode,target_verify}_metadata: the vendored SGLang kpool
metadata fusion vs the eager oracle.

The two Triton kernels (torch_ops/dsa_kpool_metadata.py, vendored from
SGLang v0.5.20 dsa_kpool_metadata) fuse the decode-step metadata block —
seqlens, zero-prefixed cumsums, pool-aware indexer-visible lengths, and
the wide/compact page tables — into ONE launch. Contract: pure integer
metadata, so parity with the eager oracle is EXACT (no tolerance gate).
Page-table rows are written only over each row's live prefix (rounded up
to the 128-column tile); the tail keeps whatever was there. The tests
prefill output tables with -1 (the reference's "unwritten" sentinel) so a
full-tensor compare also catches over-writes past the live prefix.

Capture safety is the module's second contract: max_len/num_splits and
the table strides are do_not_specialize, and every data-dependent bound
is read from device memory at replay time. The graph test records a
launch under static shapes, then mutates seq_lens on the host-visible
device tensors and replays — outputs must track the new lengths.
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.dsa_kpool_metadata import (  # noqa: E402
    bounded_scan_num_splits,
    fused_dsa_decode_metadata,
    fused_dsa_decode_metadata_eligible,
    fused_dsa_decode_metadata_reference,
    fused_dsa_target_verify_metadata,
    fused_dsa_target_verify_metadata_eligible,
    fused_dsa_target_verify_metadata_reference,
)


def _pools(device, bs, max_len, pool_rows=None, seed=0, dtype=torch.int32):
    """Random live pools: seq_lens in [0, max_len], unique kv slots."""
    gen = torch.Generator().manual_seed(seed)  # CPU generator; moved below
    pool_rows = pool_rows or max(bs + 4, 8)
    seq = torch.randint(0, max_len + 1, (bs,), generator=gen, dtype=dtype)
    rpi = torch.randperm(pool_rows, generator=gen)[:bs].to(dtype)
    # token ids in [1, pool_rows*max_len): unique-enough slots, never 0/-1.
    req_to_token = torch.randint(
        1, pool_rows * max_len + 1, (pool_rows, max_len), generator=gen, dtype=dtype
    )
    if device.type == "cpu":
        return seq, rpi, req_to_token
    return seq.to(device), rpi.to(device), req_to_token.to(device)


def _alloc_outputs(bs, max_len, device, real_page_size, with_pt1=True, expanded=None):
    """Caller-owned int32 buffers, page tables prefilled with -1."""
    rows = expanded if expanded is not None else bs
    o = {
        "cache_seqlens": torch.full((bs,), -1, dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.full((bs + 1,), -1, dtype=torch.int32, device=device),
        "dsa_cache_seqlens": torch.full((bs,), -1, dtype=torch.int32, device=device),
        "dsa_cu_seqlens_k": torch.full((bs + 1,), -1, dtype=torch.int32, device=device),
        "page_table_1": (
            torch.full((rows, max_len), -1, dtype=torch.int32, device=device)
            if with_pt1
            else None
        ),
    }
    pages = (max_len + real_page_size - 1) // real_page_size
    if real_page_size > 1:
        o["real_page_table"] = torch.full((rows, pages), -1, dtype=torch.int32, device=device)
    else:
        o["real_page_table"] = None
    return o


def _run_decode(seq, rpi, r2t, o, *, topk, kpool, rps, bs=None, max_len=None):
    fused_dsa_decode_metadata(
        seq, rpi, r2t,
        o["cache_seqlens"], o["cu_seqlens_k"], o["page_table_1"],
        o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"], o["real_page_table"],
        dsa_index_topk=topk, real_page_size=rps, index_kpool=kpool,
        bs=bs, max_len=max_len,
    )


def _assert_decode(o, ref, rps):
    assert torch.equal(o["cache_seqlens"], ref["cache_seqlens"])
    assert torch.equal(o["cu_seqlens_k"], ref["cu_seqlens_k"])
    assert torch.equal(o["dsa_cache_seqlens"], ref["dsa_cache_seqlens"])
    assert torch.equal(o["dsa_cu_seqlens_k"], ref["dsa_cu_seqlens_k"])
    assert torch.equal(o["page_table_1"], ref["page_table_1"])
    if rps > 1:
        assert torch.equal(o["real_page_table"], ref["real_page_table"])


# ---------------------------------------------------------------------------
# bounded_scan_num_splits (pure host math)
# ---------------------------------------------------------------------------


def test_bounded_scan_num_splits_math():
    assert bounded_scan_num_splits(1, 10) == 10  # tiny batch: every block splits
    assert bounded_scan_num_splits(8, 4) == 4
    # ~_TILE_PROGRAM_TARGET program budget: rows*splits <= 8192
    n = bounded_scan_num_splits(3000, 128)
    assert n == 2 and 3000 * n <= 8192
    n = bounded_scan_num_splits(16384, 100)
    assert n == 1
    assert bounded_scan_num_splits(5, 0) == 1  # degenerate width, clamped


# ---------------------------------------------------------------------------
# decode metadata: exact parity vs the eager oracle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("kpool", [1, 4], ids=["plain-dsa", "pool-aware"])
@pytest.mark.parametrize("rps", [1, 64], ids=["ps1", "ps64"])
@pytest.mark.parametrize("bs,max_len", [(1, 512), (4, 300), (7, 1024), (3, 129)])
def test_decode_metadata_parity(bs, max_len, rps, kpool, seed=0):
    dev = torch.device("cuda")
    seq, rpi, r2t = _pools(dev, bs, max_len, seed=seed)
    topk = 2048
    o = _alloc_outputs(bs, max_len, dev, rps)
    _run_decode(seq, rpi, r2t, o, topk=topk, kpool=kpool, rps=rps)
    ref = fused_dsa_decode_metadata_reference(
        seq, rpi, r2t, dsa_index_topk=topk, real_page_size=rps, index_kpool=kpool,
        page_table_1=o["page_table_1"] if o["page_table_1"] is not None else None,
        real_page_table=o["real_page_table"],
    )
    if rps == 1:
        # reference returns the wide table as the real table (kernel contract:
        # with page_size==1 real IS page_table_1 — already compared above)
        assert ref["real_page_table"] is ref["page_table_1"]
    _assert_decode(o, ref, rps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64], ids=["i32", "i64"])
def test_decode_metadata_input_dtypes(dtype):
    """Index inputs accept int32/int64; outputs stay pinned to int32."""
    dev = torch.device("cuda")
    bs, max_len, rps, kpool = 5, 256, 16, 4
    seq, rpi, r2t = _pools(dev, bs, max_len, dtype=dtype)
    o = _alloc_outputs(bs, max_len, dev, rps)
    _run_decode(seq, rpi, r2t, o, topk=64, kpool=kpool, rps=rps)
    ref = fused_dsa_decode_metadata_reference(
        seq, rpi, r2t, dsa_index_topk=64, real_page_size=rps, index_kpool=kpool,
        page_table_1=o["page_table_1"], real_page_table=o["real_page_table"],
    )
    _assert_decode(o, ref, rps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_decode_metadata_topk_cap():
    """index_topk well below max_len: dsa lengths cap but tails survive."""
    dev = torch.device("cuda")
    bs, max_len, rps, kpool = 4, 512, 64, 4
    _, rpi, r2t = _pools(dev, bs, max_len, seed=3)
    # force long sequences so the cap engages on every row
    seq = torch.tensor([2051, 3000, 2049, 512], device=dev, dtype=torch.int32)
    o = _alloc_outputs(bs, max_len, dev, rps)
    _run_decode(seq, rpi, r2t, o, topk=2048, kpool=kpool, rps=rps)
    ref = fused_dsa_decode_metadata_reference(
        seq, rpi, r2t, dsa_index_topk=2048, real_page_size=rps, index_kpool=kpool,
        page_table_1=o["page_table_1"], real_page_table=o["real_page_table"],
    )
    _assert_decode(o, ref, rps)
    # spot-check the pool-aware formula: seq=2051 -> 2048+3, 3000 -> 2048+0? no:
    # 3000//4*4=3000 -> cap 2048, tail 0 -> 2048; 2049 -> 2048+1; 512 -> 512.
    exp = torch.tensor([2051, 2048, 2049, 512], device=dev, dtype=torch.int32)
    assert torch.equal(o["dsa_cache_seqlens"], exp)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_decode_metadata_bucket_override_and_stale_rows():
    """The CUDA-graph bucket pattern: buffers larger than the live batch.

    Rows beyond the live bs must be untouched (stale) in every output.
    """
    dev = torch.device("cuda")
    live_bs, bucket_bs, max_len, rps = 2, 8, 256, 16
    seq, rpi, r2t = _pools(dev, bucket_bs, max_len, seed=5)
    seq = seq.clone(); seq[live_bs:] = 12345  # garbage the kernel must not read
    o = _alloc_outputs(bucket_bs, max_len, dev, rps)
    _run_decode(seq, rpi, r2t, o, topk=64, kpool=4, rps=rps, bs=live_bs)
    ref = fused_dsa_decode_metadata_reference(
        seq[:live_bs], rpi[:live_bs], r2t, dsa_index_topk=64, real_page_size=rps,
        index_kpool=4, page_table_1=o["page_table_1"], real_page_table=o["real_page_table"],
    )
    for k in ("cache_seqlens", "dsa_cache_seqlens"):
        assert torch.equal(o[k][:live_bs], ref[k])
        assert (o[k][live_bs:] == -1).all()
    for k in ("cu_seqlens_k", "dsa_cu_seqlens_k"):
        assert torch.equal(o[k][: live_bs + 1], ref[k])
        assert (o[k][live_bs + 1 :] == -1).all()
    assert torch.equal(o["page_table_1"][:live_bs], ref["page_table_1"])
    assert (o["page_table_1"][live_bs:] == -1).all()
    assert torch.equal(o["real_page_table"][:live_bs], ref["real_page_table"])
    assert (o["real_page_table"][live_bs:] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_decode_metadata_drop_wide_table():
    """page_table_1=None: only the compact table is written (fused decode graph)."""
    dev = torch.device("cuda")
    bs, max_len, rps = 3, 256, 64
    seq, rpi, r2t = _pools(dev, bs, max_len, seed=7)
    o = _alloc_outputs(bs, max_len, dev, rps, with_pt1=False)
    assert o["page_table_1"] is None
    _run_decode(seq, rpi, r2t, o, topk=64, kpool=4, rps=rps)
    ref = fused_dsa_decode_metadata_reference(
        seq, rpi, r2t, dsa_index_topk=64, real_page_size=rps, index_kpool=4,
        page_table_1=None, real_page_table=o["real_page_table"],
    )
    assert torch.equal(o["cache_seqlens"], ref["cache_seqlens"])
    assert torch.equal(o["dsa_cache_seqlens"], ref["dsa_cache_seqlens"])
    assert torch.equal(o["real_page_table"], ref["real_page_table"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_decode_metadata_bs0_zeroes_heads_only():
    dev = torch.device("cuda")
    seq = torch.empty(0, dtype=torch.int32, device=dev)
    rpi = torch.empty(0, dtype=torch.int32, device=dev)
    r2t = torch.zeros(4, 8, dtype=torch.int32, device=dev)
    cu = torch.full((1,), -7, dtype=torch.int32, device=dev)
    dsa_cu = torch.full((1,), -7, dtype=torch.int32, device=dev)
    cs = torch.empty(0, dtype=torch.int32, device=dev)
    fused_dsa_decode_metadata(
        seq, rpi, r2t, cs, cu, None, cs, dsa_cu, None,
        dsa_index_topk=64, real_page_size=64,
    )
    assert cu.item() == 0 and dsa_cu.item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_decode_metadata_graph_replay_tracks_lengths():
    """Record once, replay after mutating seq_lens: outputs track the new data.

    This is the capture-safety contract (do_not_specialize + device-read
    bounds) end to end.
    """
    dev = torch.device("cuda")
    bs, max_len, rps = 3, 512, 16
    seq, rpi, r2t = _pools(dev, bs, max_len, seed=11)
    o = _alloc_outputs(bs, max_len, dev, rps)
    _run_decode(seq, rpi, r2t, o, topk=128, kpool=4, rps=rps)  # warm/compile

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run_decode(seq, rpi, r2t, o, topk=128, kpool=4, rps=rps)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        _run_decode(seq, rpi, r2t, o, topk=128, kpool=4, rps=rps)

    seq.copy_(torch.tensor([17, 512, 300], device=dev, dtype=torch.int32))
    o["page_table_1"].fill_(-1)
    o["real_page_table"].fill_(-1)
    for t in (o["cache_seqlens"], o["cu_seqlens_k"], o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"]):
        t.fill_(-1)
    g.replay()
    torch.cuda.synchronize()

    ref = fused_dsa_decode_metadata_reference(
        seq, rpi, r2t, dsa_index_topk=128, real_page_size=rps, index_kpool=4,
        page_table_1=o["page_table_1"], real_page_table=o["real_page_table"],
    )
    _assert_decode(o, ref, rps)


# ---------------------------------------------------------------------------
# target-verify metadata (speculative decoding variant)
# ---------------------------------------------------------------------------


def _run_verify(seq, rpi, r2t, o, *, next_n, topk, kpool, rps, bs=None, max_seqlen_k=None, pmqa=None):
    fused_dsa_target_verify_metadata(
        seq, rpi, r2t,
        o["cache_seqlens"], o["cu_seqlens_k"], o["page_table_1"],
        o["seqlens_expanded"], o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"],
        o["real_page_table"],
        next_n=next_n, dsa_index_topk=topk, real_page_size=rps, index_kpool=kpool,
        bs=bs, max_seqlen_k=max_seqlen_k, paged_mqa_ctx_lens_2d=pmqa,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
@pytest.mark.parametrize("rps", [1, 64], ids=["ps1", "ps64"])
@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("bs,max_len", [(1, 256), (4, 300)])
def test_verify_metadata_parity(bs, max_len, next_n, rps, seed=13):
    dev = torch.device("cuda")
    topk, kpool = 128, 4
    seq, rpi, r2t = _pools(dev, bs, max_len, seed=seed)
    expanded = bs * next_n
    o = _alloc_outputs(bs, max_len, dev, rps, expanded=expanded)
    o["seqlens_expanded"] = torch.full((expanded,), -1, dtype=torch.int32, device=dev)
    o["dsa_cache_seqlens"] = torch.full((expanded,), -1, dtype=torch.int32, device=dev)
    o["dsa_cu_seqlens_k"] = torch.full((expanded + 1,), -1, dtype=torch.int32, device=dev)
    _run_verify(seq, rpi, r2t, o, next_n=next_n, topk=topk, kpool=kpool, rps=rps)
    ref = fused_dsa_target_verify_metadata_reference(
        seq, rpi, r2t, next_n=next_n, dsa_index_topk=topk, real_page_size=rps,
        index_kpool=kpool, page_table_1=o["page_table_1"], real_page_table=o["real_page_table"],
    )
    assert torch.equal(o["cache_seqlens"], ref["cache_seqlens"])
    assert torch.equal(o["cu_seqlens_k"], ref["cu_seqlens_k"])
    assert torch.equal(o["seqlens_expanded"], ref["seqlens_expanded"])
    assert torch.equal(o["dsa_cache_seqlens"], ref["dsa_cache_seqlens"])
    assert torch.equal(o["dsa_cu_seqlens_k"], ref["dsa_cu_seqlens_k"])
    assert torch.equal(o["page_table_1"], ref["page_table_1"])
    if rps > 1:
        assert torch.equal(o["real_page_table"], ref["real_page_table"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_verify_metadata_paged_mqa_ctx_lens():
    dev = torch.device("cuda")
    bs, max_len, next_n, rps = 3, 256, 2, 64
    seq, rpi, r2t = _pools(dev, bs, max_len, seed=17)
    expanded = bs * next_n
    o = _alloc_outputs(bs, max_len, dev, rps, expanded=expanded)
    o["seqlens_expanded"] = torch.full((expanded,), -1, dtype=torch.int32, device=dev)
    o["dsa_cache_seqlens"] = torch.full((expanded,), -1, dtype=torch.int32, device=dev)
    o["dsa_cu_seqlens_k"] = torch.full((expanded + 1,), -1, dtype=torch.int32, device=dev)
    pmqa = torch.full((bs, next_n), -1, dtype=torch.int32, device=dev)
    _run_verify(seq, rpi, r2t, o, next_n=next_n, topk=128, kpool=4, rps=rps, pmqa=pmqa)
    ref = fused_dsa_target_verify_metadata_reference(
        seq, rpi, r2t, next_n=next_n, dsa_index_topk=128, real_page_size=rps,
        index_kpool=4, page_table_1=o["page_table_1"], real_page_table=o["real_page_table"],
        paged_mqa_ctx_lens_2d=pmqa,
    )
    assert torch.equal(pmqa, ref["paged_mqa_ctx_lens_2d"])
    exp = (seq.to(torch.int32).unsqueeze(1) + next_n).expand(bs, next_n)
    assert torch.equal(pmqa, exp.contiguous())


# ---------------------------------------------------------------------------
# eligibility contracts (run on any device; CPU tensor lists must reject)
# ---------------------------------------------------------------------------


def test_eligibility_cpu_rejects():
    dev = torch.device("cpu")
    bs, max_len, rps = 2, 64, 16
    seq, rpi, r2t = _pools(dev, bs, max_len)
    o = _alloc_outputs(bs, max_len, dev, rps)
    assert not fused_dsa_decode_metadata_eligible(
        seq, rpi, r2t, o["cache_seqlens"], o["cu_seqlens_k"], o["page_table_1"],
        o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"], o["real_page_table"],
        dsa_index_topk=64, real_page_size=rps,
    )
    with pytest.raises(OpNotEligible):
        _run_decode(seq, rpi, r2t, o, topk=64, kpool=4, rps=rps)


def test_eligibility_reference_runs_on_cpu():
    """The eager oracle is device-agnostic (used as the fallback oracle)."""
    dev = torch.device("cpu")
    bs, max_len, rps = 3, 64, 16
    seq, rpi, r2t = _pools(dev, bs, max_len, seed=1)
    ref = fused_dsa_decode_metadata_reference(
        seq, rpi, r2t, dsa_index_topk=32, real_page_size=rps, index_kpool=4,
    )
    assert ref["cache_seqlens"].dtype == torch.int32
    assert ref["cu_seqlens_k"][0].item() == 0
    assert ref["dsa_cu_seqlens_k"][0].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_eligibility_bad_shapes_and_dtypes():
    dev = torch.device("cuda")
    bs, max_len, rps = 2, 64, 16
    seq, rpi, r2t = _pools(dev, bs, max_len)

    good = _alloc_outputs(bs, max_len, dev, rps)
    args = lambda o: (  # noqa: E731
        seq, rpi, r2t, o["cache_seqlens"], o["cu_seqlens_k"], o["page_table_1"],
        o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"], o["real_page_table"],
    )
    kw = dict(dsa_index_topk=64, real_page_size=rps, index_kpool=4)

    # float seqlens
    bad_seq = seq.to(torch.float32)
    assert not fused_dsa_decode_metadata_eligible(bad_seq, *args(good)[1:], **kw)
    with pytest.raises(OpNotEligible):
        fused_dsa_decode_metadata(bad_seq, *args(good)[1:], **kw)

    # int64 outputs (kernel stores int32)
    bad = _alloc_outputs(bs, max_len, dev, rps)
    bad["cu_seqlens_k"] = bad["cu_seqlens_k"].to(torch.int64)
    assert not fused_dsa_decode_metadata_eligible(*args(bad), **kw)
    with pytest.raises(OpNotEligible):
        fused_dsa_decode_metadata(*args(bad), **kw)

    # undersized table (width < max_len) — pin max_len so the derived default
    # (narrowest table) cannot absorb the narrower table
    bad = _alloc_outputs(bs, max_len, dev, rps)
    bad["page_table_1"] = bad["page_table_1"][:, : max_len - 1].contiguous()
    assert not fused_dsa_decode_metadata_eligible(*args(bad), max_len=max_len, **kw)

    # real_page_size == 1 without the wide table
    assert not fused_dsa_decode_metadata_eligible(
        seq, rpi, r2t, good["cache_seqlens"], good["cu_seqlens_k"], None,
        good["dsa_cache_seqlens"], good["dsa_cu_seqlens_k"], None,
        dsa_index_topk=64, real_page_size=1,
    )

    # rps > 1 without the compact table
    assert not fused_dsa_decode_metadata_eligible(
        seq, rpi, r2t, good["cache_seqlens"], good["cu_seqlens_k"], good["page_table_1"],
        good["dsa_cache_seqlens"], good["dsa_cu_seqlens_k"], None,
        dsa_index_topk=64, real_page_size=rps,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel test")
def test_verify_eligibility_rejects():
    dev = torch.device("cuda")
    bs, max_len, next_n, rps = 2, 64, 2, 16
    seq, rpi, r2t = _pools(dev, bs, max_len)
    expanded = bs * next_n
    o = _alloc_outputs(bs, max_len, dev, rps, expanded=expanded)
    o["seqlens_expanded"] = torch.full((expanded,), -1, dtype=torch.int32, device=dev)
    o["dsa_cache_seqlens"] = torch.full((expanded,), -1, dtype=torch.int32, device=dev)
    o["dsa_cu_seqlens_k"] = torch.full((expanded + 1,), -1, dtype=torch.int32, device=dev)
    kw = dict(next_n=next_n, dsa_index_topk=64, real_page_size=rps, index_kpool=4)

    # CPU inputs reject
    assert not fused_dsa_target_verify_metadata_eligible(
        seq.cpu(), rpi, r2t, o["cache_seqlens"], o["cu_seqlens_k"], o["page_table_1"],
        o["seqlens_expanded"], o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"],
        o["real_page_table"], **kw,
    )
    # wrong paged_mqa_ctx_lens_2d shape
    assert not fused_dsa_target_verify_metadata_eligible(
        seq, rpi, r2t, o["cache_seqlens"], o["cu_seqlens_k"], o["page_table_1"],
        o["seqlens_expanded"], o["dsa_cache_seqlens"], o["dsa_cu_seqlens_k"],
        o["real_page_table"],
        paged_mqa_ctx_lens_2d=torch.zeros(bs, next_n + 1, dtype=torch.int32, device=dev),
        **kw,
    )
