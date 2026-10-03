"""dsa_kpool_compress: the vendored vLLM kpool Triton kernels vs the eager
torch oracles (torch_ops/dsa_kpool_compress.py, vendored from vLLM PR
#53906 ``vllm/models/glm5next/nvidia/ops/kpool_compress.py``).

Contract mirrors the donor's own kernel tests plus the vkernels house
pattern:

* fp8 parity is asserted at the BYTE level (``torch.equal`` on the written
  cache) — the kernels and the references share the exact fp32 butterfly /
  softmax op order, and the absmax power-of-two quantization absorbs the
  last-ulp exp/log2 differences;
* the decode-update kernel's per-request, in-position-order semantics (the
  read-after-stash dependency) are validated against an independent
  reference, PLUS the production invariant ``decode writer == prefill
  writer`` (a hand-written reference can drift to match a buggy kernel);
* the pure-torch pieces (the expand/append-tail twins, all ``*_reference``
  oracles) run on CPU — that is the CPU-safe subset exercised on machines
  without a GPU;
* eligibility: CPU tensors / wrong dtypes / wrong shapes raise
  ``OpNotEligible`` (floe catches and falls back); the capture-safety test
  records a decode-update launch under static shapes and replays it after
  mutating positions / slot mappings in place — outputs must track.
"""

import pytest

torch = pytest.importorskip("torch")

from vkernels.torch_ops._dispatch import OpNotEligible  # noqa: E402
from vkernels.torch_ops.dsa_kpool_compress import (  # noqa: E402
    FP8_DTYPE,
    FP8_MAX,
    INDEX_HEAD_DIM,
    append_tail_to_topk,
    expand_pools_and_append_tail,
    expand_pools_to_tokens,
    fwht128_quant_fp8,
    fwht128_quant_fp8_reference,
    history_group_budget_for_topk,
    kpool_compress_and_write_cache,
    kpool_compress_and_write_cache_eligible,
    kpool_compress_and_write_cache_reference,
    kpool_decode_update_and_maybe_write_cache_batched,
    kpool_decode_update_eligible,
    kpool_decode_update_reference,
    kpool_seed_tail_cache,
    kpool_seed_tail_cache_eligible,
    kpool_seed_tail_cache_reference,
)

HEAD_DIM = INDEX_HEAD_DIM
POOL_SIZE = 16  # donor default; GLM-5.3 uses 4 (covered separately)
PAGE_SIZE = 64
NUM_BLOCKS = 32
CUDA = torch.cuda.is_available()
if CUDA:
    DEV = torch.device("cuda")


def _make_caches(device, num_blocks=NUM_BLOCKS, pool=None):
    pool = pool or POOL_SIZE
    kv = torch.zeros(num_blocks, PAGE_SIZE, HEAD_DIM + 4, dtype=torch.uint8, device=device)
    tail = torch.zeros(num_blocks, 2, pool, HEAD_DIM, dtype=torch.bfloat16, device=device)
    return kv, tail


def _randn(shape, device, gen=None, dtype=torch.bfloat16):
    g = gen or torch.Generator(device="cpu").manual_seed(0)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype).to(device)


def _tail_slot_for(blocks, pos, pool=None):
    """tail_slot = block*pool + pos%pool; each request owns a tail block."""
    pool = pool or POOL_SIZE
    blk = torch.tensor(blocks, device=pos.device, dtype=torch.int32).unsqueeze(1)
    return (blk * pool + pos % pool).to(torch.int32)


def _seed_prior(tail, blocks, n_prior, seed=42):
    if n_prior <= 0:
        return
    g = torch.Generator().manual_seed(seed)
    prior_k = _randn((len(blocks), n_prior, HEAD_DIM), tail.device, g)
    prior_s = _randn((len(blocks), n_prior, HEAD_DIM), tail.device, g)
    for i, blk in enumerate(blocks):
        tail[blk, 0, :n_prior, :] = prior_k[i]
        tail[blk, 1, :n_prior, :] = prior_s[i]


def _pool_bytes(kv, loc):
    """One pool's K bytes + scale bytes as the cache stores them."""
    page, tok = loc // PAGE_SIZE, loc % PAGE_SIZE
    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    flat = kv.view(-1)
    k = flat[page * page_bytes + tok * HEAD_DIM : page * page_bytes + (tok + 1) * HEAD_DIM]
    s = flat[
        page * page_bytes + PAGE_SIZE * HEAD_DIM + 4 * tok : page * page_bytes + PAGE_SIZE * HEAD_DIM + 4 * (tok + 1)
    ]
    return torch.cat([k, s])


# ===========================================================================
# CPU-safe subset: the pure-torch pieces
# ===========================================================================


def test_history_group_budget():
    assert history_group_budget_for_topk(2048, 4) == 512  # GLM-5.3: 2051-wide
    assert history_group_budget_for_topk(2048, 1) == 2048
    assert history_group_budget_for_topk(64, 16) == 4


def test_expand_twins_identity_path_cpu():
    """The donor torch twins vs a manual construction (pure torch, CPU)."""
    pool, n_groups, rows = 4, 3, 5
    seq_lens = torch.tensor([100, 13, 7, 4, 0], dtype=torch.int32)
    pool_ids = torch.tensor([[0, 1, 5], [0, -1, 2], [3, 1, 0], [-1, -1, -1], [0, 2, 4]])
    valid = pool_ids >= 0
    topk = n_groups * pool

    hist = expand_pools_to_tokens(pool_ids, valid, topk, pool)
    assert hist.shape == (rows, topk) and hist.dtype == torch.int32
    # manual: pool g, slot j -> id g*pool + j when g valid else -1
    for r in range(rows):
        for g in range(n_groups):
            for j in range(pool):
                want = pool_ids[r, g] * pool + j if valid[r, g] else -1
                assert hist[r, g * pool + j] == want

    pool_lens = seq_lens // pool
    out = append_tail_to_topk(hist, seq_lens, pool_lens, pool)
    assert out.shape == (rows, topk + pool - 1)
    for r in range(rows):
        tail_start = int(pool_lens[r]) * pool
        tail_count = int(seq_lens[r]) - tail_start
        for c in range(topk + pool - 1):
            if c < topk:
                want = hist[r, c]
            elif c - topk < tail_count:
                want = tail_start + (c - topk)
            else:
                want = -1
            assert out[r, c] == want, (r, c)


def test_expand_tail_kernel_matches_twins_cuda():
    """The fused kernel vs the donor torch twins (identity path)."""
    if not CUDA:
        pytest.skip("CUDA kernel test")
    torch.manual_seed(0)
    pool = 4
    for rows, n_groups, seq_range in [(1, 512, 100_000), (5, 8, 40), (3, 33, 137)]:
        pool_ids = torch.randint(0, seq_range // pool, (rows, n_groups), device=DEV)
        # sprinkle invalid pools (the -1 mask path)
        pool_ids.view(-1)[:: max(rows * n_groups // 5, 1)] = -1
        valid = pool_ids >= 0
        seq_lens = torch.randint(1, seq_range, (rows,), device=DEV, dtype=torch.int32)
        topk = n_groups * pool
        want = append_tail_to_topk(
            expand_pools_to_tokens(pool_ids.to(torch.int64), valid, topk, pool),
            seq_lens,
            seq_lens // pool,
            pool,
        )
        got = expand_pools_and_append_tail(pool_ids.to(torch.int64), seq_lens, pool)
        assert got.shape == want.shape and got.dtype == torch.int32
        assert torch.equal(got, want), (rows, n_groups)


def test_decode_reference_matches_compress_reference_cpu():
    """Production invariant, reference level: feeding tokens one at a time
    through the decode-update oracle builds the SAME pools as the prefill
    compress oracle (pure torch, CPU)."""
    pool, nblk = 4, 2
    n_pools = 6
    n_tok = n_pools * pool
    g = torch.Generator().manual_seed(3)
    k = _randn((n_tok, HEAD_DIM), "cpu", g)
    score = _randn((n_tok, HEAD_DIM), "cpu", g)
    ape = torch.randn(pool, HEAD_DIM, generator=g, dtype=torch.float32)

    kv0, tail = _make_caches("cpu", nblk, pool)
    loc = torch.arange(n_pools, dtype=torch.int64)
    prefill = kpool_compress_and_write_cache_reference(
        k.view(n_pools, pool, HEAD_DIM), score.view(n_pools, pool, HEAD_DIM),
        ape, loc, head_dim=HEAD_DIM, kv_cache=kv0, round_scale=True,
    )

    kv = kv0.clone()
    for t in range(n_tok):
        out = kpool_decode_update_reference(
            kv, tail,
            torch.tensor([[t % pool]], dtype=torch.int32),   # tail slot
            k[t].view(1, 1, HEAD_DIM), score[t].view(1, 1, HEAD_DIM), ape,
            torch.tensor([[t // pool if t % pool == pool - 1 else -1]], dtype=torch.int32),
            torch.tensor([[t]], dtype=torch.int32),
            pool_size=pool, round_scale=True,
        )
        kv, tail = out["kv_cache"], out["tail_kv_cache"]
    assert torch.equal(kv, prefill["kv_cache"])


def test_seed_reference_semantics_cpu():
    """Tail seeding: a token whose kpool-ahead token sits in the SAME tail
    block is mid-pool and skipped (a later seed overwrites its ring slot);
    block-crossing or batch-end tokens seed."""
    kpool, ring = 4, 8  # two pools per tail block
    tail = torch.zeros(2, 2, ring, HEAD_DIM, dtype=torch.bfloat16)
    key = torch.randn(12, HEAD_DIM).to(torch.bfloat16)
    gate = torch.randn(12, HEAD_DIM).to(torch.bfloat16)
    # two requests, six tokens each: positions 0..5 in blocks 0 and 1
    # tslot = block*ring + pos
    tslot = torch.tensor([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13], dtype=torch.int32)
    out = kpool_seed_tail_cache_reference(tail, key, gate, tslot, kpool)
    # block 0: token 0 (ahead=tslot[4]=4, same block 0) SKIP; token 1
    # (ahead=5, same block) SKIP; tokens 2..5 (ahead out of batch -> -1) SEED.
    assert torch.equal(out[0, 0, 0], torch.zeros(HEAD_DIM, dtype=torch.bfloat16))
    assert torch.equal(out[0, 1, 1], torch.zeros(HEAD_DIM, dtype=torch.bfloat16))
    assert torch.equal(out[0, 0, 2], key[2]) and torch.equal(out[0, 1, 2], gate[2])
    assert torch.equal(out[0, 0, 5], key[5])
    # block 1: token 6 (tslot=8, ahead=tslot[10]=12, same block 1) SKIP;
    # token 7 (ahead=13, same block) SKIP; tokens 8..11 SEED at pos%ring.
    assert torch.equal(out[1, 0, 0], torch.zeros(HEAD_DIM, dtype=torch.bfloat16))
    assert torch.equal(out[1, 0, 1], torch.zeros(HEAD_DIM, dtype=torch.bfloat16))
    assert torch.equal(out[1, 0, 2], key[8])
    assert torch.equal(out[1, 1, 5], gate[11])


# ===========================================================================
# eligibility: contract misses raise OpNotEligible (CPU-visible paths)
# ===========================================================================


def _cpu_inputs():
    kv, tail = _make_caches("cpu", 4, 4)
    k = torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16)
    score = torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16)
    ape = torch.zeros(4, HEAD_DIM)
    loc = torch.zeros(2, dtype=torch.int64)
    return kv, tail, k, score, ape, loc


def test_eligibility_cpu_rejects_launch_ops():
    kv, tail, k, score, ape, loc = _cpu_inputs()
    with pytest.raises(OpNotEligible, match="CUDA-resident"):
        kpool_compress_and_write_cache(kv, k, score, ape, loc, 4)
    assert not kpool_compress_and_write_cache_eligible(kv, k, score, ape, loc, pool_size=4)

    tslot = torch.zeros(2, dtype=torch.int32)
    with pytest.raises(OpNotEligible, match="CUDA-resident"):
        kpool_seed_tail_cache(tail, k[0], score[0], tslot, 4)
    assert not kpool_seed_tail_cache_eligible(tail, k[0], score[0], tslot, kpool=4)

    key = torch.zeros(2, 1, HEAD_DIM, dtype=torch.bfloat16)
    sm = torch.full((2, 1), -1, dtype=torch.int32)
    pos = torch.zeros(2, 1, dtype=torch.int32)
    with pytest.raises(OpNotEligible, match="CUDA-resident"):
        kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, sm, key, key, ape, sm, pos, 4)
    assert not kpool_decode_update_eligible(
        kv, tail, sm, key, key, ape, sm, pos, pool_size=4)

    with pytest.raises(OpNotEligible, match="CUDA-resident"):
        expand_pools_and_append_tail(k[:2, 0].to(torch.int64), pos[:, 0].to(torch.int32), 4)
    with pytest.raises(OpNotEligible, match="CUDA-resident"):
        fwht128_quant_fp8(k[0])


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
def test_eligibility_dtype_and_shape_violations():
    kv, tail, k, score, ape, loc = _cuda_inputs()
    # slot_k fp16 instead of bf16
    with pytest.raises(OpNotEligible, match="slot_k must be bfloat16"):
        kpool_compress_and_write_cache(kv, k.to(torch.float16), score, ape, loc, 4)
    # ape fp64
    with pytest.raises(OpNotEligible, match="ape must be float32"):
        kpool_compress_and_write_cache(kv, k, score, ape.double(), loc, 4)
    # loc int32
    with pytest.raises(OpNotEligible, match="loc must be int64"):
        kpool_compress_and_write_cache(kv, k, score, ape, loc.to(torch.int32), 4)
    # write_mask + return_compressed mutually exclusive
    wm = torch.ones(2, dtype=torch.bool, device=DEV)
    with pytest.raises(OpNotEligible, match="mutually exclusive"):
        kpool_compress_and_write_cache(kv, k, score, ape, loc, 4, write_mask=wm,
                                       return_compressed=True)
    # kv cache last dim wrong
    bad_kv = torch.zeros(2, PAGE_SIZE, HEAD_DIM + 8, dtype=torch.uint8, device=DEV)
    with pytest.raises(OpNotEligible, match=r"128\+4"):
        kpool_compress_and_write_cache(bad_kv, k, score, ape, loc, 4)

    key = torch.zeros(2, 1, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    sm = torch.full((2, 1), -1, dtype=torch.int32, device=DEV)
    pos = torch.zeros(2, 1, dtype=torch.int32, device=DEV)
    # positions int64
    with pytest.raises(OpNotEligible, match="positions must be int32"):
        kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, sm, key, key, ape, sm, pos.to(torch.int64), 4)
    # ring not a multiple of pool
    bad_tail = torch.zeros(4, 2, 6, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(OpNotEligible, match="multiple"):
        kpool_decode_update_and_maybe_write_cache_batched(
            kv, bad_tail, sm, key, key, ape, sm, pos, 4)
    assert kpool_decode_update_eligible(kv, tail, sm, key, key, ape, sm, pos, pool_size=4)


def _cuda_inputs():
    kv, tail = _make_caches(DEV, 4, 4)
    k = torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    score = torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    ape = torch.zeros(4, HEAD_DIM, dtype=torch.float32, device=DEV)
    loc = torch.zeros(2, dtype=torch.int64, device=DEV)
    return kv, tail, k, score, ape, loc


# ===========================================================================
# CUDA parity: the kernels vs the eager oracles
# ===========================================================================


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize("rows", [1, 7, 32, 100])
def test_fwht128_quant_parity(rows):
    g = torch.Generator().manual_seed(rows)
    q = _randn((rows, HEAD_DIM), DEV, g)
    q_fp8, q_scale = fwht128_quant_fp8(q)
    r_fp8, r_scale = fwht128_quant_fp8_reference(q)
    assert q_fp8.dtype == FP8_DTYPE and q_scale.shape == (rows, 1)
    assert torch.equal(q_scale, r_scale.reshape(rows, 1))
    assert torch.equal(q_fp8.view(torch.uint8), r_fp8.view(torch.uint8))


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("masked", [False, True], ids=["nomask", "mask"])
@pytest.mark.parametrize("round_scale", [True, False], ids=["po2", "linear"])
@pytest.mark.parametrize("return_compressed", [False, True], ids=["write", "ret"])
def test_compress_parity(pool_size, masked, round_scale, return_compressed):
    g = torch.Generator().manual_seed(1)
    n_pools = 9
    # scattered locs across pages and blocks (not just [0..n))
    loc = torch.tensor([0, 3, 63, 64, 65, 127, 128, 129, 500], dtype=torch.int64, device=DEV)
    k = _randn((n_pools, pool_size, HEAD_DIM), DEV, g)
    score = _randn((n_pools, pool_size, HEAD_DIM), DEV, g)
    ape = torch.randn(pool_size, HEAD_DIM, generator=g, dtype=torch.float32).to(DEV)
    kv0, _ = _make_caches(DEV, 8, pool_size)
    if masked and return_compressed:
        # donor forbids write_mask + return_compressed (checked in the
        # eligibility test); do not exercise the combination here.
        return
    write_mask = (
        torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 0], dtype=torch.bool, device=DEV)
        if masked else None
    )
    ref = kpool_compress_and_write_cache_reference(
        k, score, ape, loc, head_dim=HEAD_DIM,
        kv_cache=kv0, write_mask=write_mask, round_scale=round_scale,
        return_compressed=True,
    )

    kv = kv0.clone()
    ret = kpool_compress_and_write_cache(
        kv, k, score, ape, loc, pool_size, HEAD_DIM,
        write_mask=write_mask, round_scale=round_scale,
        return_compressed=return_compressed,
    )
    assert torch.equal(kv.cpu(), ref["kv_cache"])
    if return_compressed:
        ck, cs = ret
        assert torch.equal(ck.view(torch.uint8).cpu(), ref["compressed_k"].view(torch.uint8))
        assert torch.equal(cs.cpu(), ref["compressed_scale"])
    else:
        assert ret is None


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
def test_compress_zero_pools():
    kv, _ = _make_caches(DEV, 2, 4)
    k = torch.zeros(0, 4, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    ret = kpool_compress_and_write_cache(
        kv, k, k, torch.zeros(4, HEAD_DIM, device=DEV),
        torch.zeros(0, dtype=torch.int64, device=DEV), 4, return_compressed=True)
    assert ret[0].shape == (0, HEAD_DIM) and ret[1].shape == (0,)


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize("pool", [4, 16])
def test_seed_parity(pool):
    g = torch.Generator().manual_seed(2)
    _, tail = _make_caches(DEV, 4, pool)
    n = 3 * pool
    key = _randn((n, HEAD_DIM), DEV, g)
    gate = _randn((n, HEAD_DIM), DEV, g)
    # three requests of one pool each: tslot = block*pool + pos%pool, 1-D [n]
    blocks = [0, 1, 2]
    tslot = torch.cat([
        torch.arange(pool, dtype=torch.int32) + b * pool for b in blocks
    ]).to(DEV)
    want = kpool_seed_tail_cache_reference(tail, key, gate, tslot, pool)
    kpool_seed_tail_cache(tail, key, gate, tslot, pool)
    torch.cuda.synchronize()
    assert torch.equal(tail, want)


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
def test_seed_padded_tail_block_stride():
    """The tail aliases a padded indexer allocation in production (donor
    test): blocks must be addressed via stride(0)/stride(1), never dense."""
    kpool, num_blocks = 4, 6
    logical = 2 * kpool * HEAD_DIM
    padded = logical + 256
    sentinel = -123.0
    backing = torch.full((num_blocks * padded,), sentinel, dtype=torch.bfloat16, device=DEV)
    tail = torch.as_strided(
        backing, size=(num_blocks, 2, kpool, HEAD_DIM),
        stride=(padded, kpool * HEAD_DIM, HEAD_DIM, 1),
    )
    block, ring_slot = 3, 2
    key = torch.arange(HEAD_DIM, dtype=torch.bfloat16, device=DEV).unsqueeze(0)
    score = (key + 256).to(torch.bfloat16)
    tslot = torch.tensor([block * kpool + ring_slot], dtype=torch.int32, device=DEV)
    kpool_seed_tail_cache(tail, key, score, tslot, kpool)
    torch.cuda.synchronize()
    assert torch.equal(tail[block, 0, ring_slot], key[0])
    assert torch.equal(tail[block, 1, ring_slot], score[0])
    compact = (block * 2 * kpool + ring_slot) * HEAD_DIM
    assert torch.all(backing[compact : compact + HEAD_DIM] == sentinel)


def _decode_case(case_id):
    """(kv, tail, tail_slot, key, score, ape, slot_map, pos) per donor case."""
    g = torch.Generator().manual_seed(0)
    if case_id == "no_completion":
        B, next_n, blocks = 3, 4, [0, 1, 2]
        pos = torch.arange(next_n, dtype=torch.int32).unsqueeze(0).expand(B, -1).contiguous()
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32)
        n_prior = 0
    elif case_id == "completion_at_end":
        B, next_n, blocks = 2, 4, [0, 1]
        pos = torch.tensor([[12, 13, 14, 15]] * B, dtype=torch.int32)
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32)
        slot_map[:, 3] = torch.tensor([15, PAGE_SIZE + 15], dtype=torch.int32)
        n_prior = POOL_SIZE - next_n
    elif case_id == "completion_mid_batch":
        B, next_n, blocks = 3, 4, [0, 1, 2]
        pos = torch.tensor([[13, 14, 15, 16]] * B, dtype=torch.int32)
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32)
        slot_map[:, 2] = torch.tensor([15, PAGE_SIZE + 15, 2 * PAGE_SIZE + 15], dtype=torch.int32)
        n_prior = 13
    elif case_id == "non_uniform_padding":
        B, next_n, blocks = 2, 4, [0, 1]
        pos = torch.tensor([[12, 13, 14, 15], [12, 13, -1, -1]], dtype=torch.int32)
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32)
        slot_map[0, 3] = 15
        n_prior = POOL_SIZE - 4
    else:  # plain_decode
        B, next_n, blocks = 4, 1, [0, 1, 2, 3]
        pos = torch.tensor([[5], [6], [7], [8]], dtype=torch.int32)
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32)
        n_prior = 0

    if case_id == "non_uniform_padding":
        safe_pos = torch.where(pos >= 0, pos, 0)
        tail_slot = torch.where(pos >= 0, _tail_slot_for(blocks, safe_pos), 0)
    else:
        tail_slot = _tail_slot_for(blocks, pos)

    key = _randn((B, next_n, HEAD_DIM), DEV, g)
    score = _randn((B, next_n, HEAD_DIM), DEV, g)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=g, dtype=torch.float32).to(DEV)
    kv, tail = _make_caches(DEV)
    _seed_prior(tail, blocks, n_prior)
    return (kv, tail, tail_slot.to(DEV), key, score, ape, slot_map.to(DEV), pos.to(DEV))


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
def test_leading_invalid_tail_slot():
    """A request whose FIRST token carries an invalid (-1) tail slot while a
    later token is a real pool completion (the tail block must be derived per
    token, not from token 0)."""
    B, next_n, blocks = 2, 4, [3, 5]
    pos = torch.tensor([[-1, 13, 14, 15], [4, 5, 6, 7]], dtype=torch.int32)
    safe_pos = torch.where(pos >= 0, pos, 0)
    tail_slot = _tail_slot_for(blocks, safe_pos)
    tail_slot[0, 0] = -1  # the -1 sentinel the scatter path emits
    slot_map = torch.full((B, next_n), -1, dtype=torch.int32)
    slot_map[0, 3] = 15
    g = torch.Generator().manual_seed(0)
    key = _randn((B, next_n, HEAD_DIM), DEV, g)
    score = _randn((B, next_n, HEAD_DIM), DEV, g)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=g, dtype=torch.float32).to(DEV)
    kv, tail = _make_caches(DEV)
    _seed_prior(tail, blocks, 13)

    args = (kv, tail, tail_slot.to(DEV), key, score, ape, slot_map.to(DEV), pos.to(DEV))
    ref = kpool_decode_update_reference(*args, pool_size=POOL_SIZE)
    kv_k, tail_k = kv.clone(), tail.clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        kv_k, tail_k, *args[2:], POOL_SIZE)
    assert torch.equal(kv_k.cpu(), ref["kv_cache"])
    assert torch.equal(tail_k.cpu(), ref["tail_kv_cache"])


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize(
    "case_id",
    ["no_completion", "completion_at_end", "completion_mid_batch",
     "non_uniform_padding", "plain_decode"],
)
def test_decode_update_matches_reference(case_id):
    args = _decode_case(case_id)
    kv, tail, tail_slot, key, score, ape, slot_map, pos = args
    ref = kpool_decode_update_reference(
        kv, tail, tail_slot, key, score, ape, slot_map, pos, pool_size=POOL_SIZE)
    kv_k, tail_k = kv.clone(), tail.clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        kv_k, tail_k, tail_slot, key, score, ape, slot_map, pos, POOL_SIZE)
    assert torch.equal(kv_k.cpu(), ref["kv_cache"]), (
        f"kv differs: {(kv_k.int() - ref['kv_cache'].to(kv_k.device).int()).abs().max()}")
    assert torch.equal(tail_k.cpu(), ref["tail_kv_cache"]), (
        f"tail differs: {(tail_k.float() - ref['tail_kv_cache'].to(tail_k.device).float()).abs().max()}")


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize("seed", list(range(10)))
def test_decode_update_fuzz(seed):
    g0 = torch.Generator().manual_seed(seed)
    B = int(torch.randint(1, 6, (1,), generator=g0))
    next_n = int(torch.randint(1, 8, (1,), generator=g0))
    blocks = list(range(B))
    starts = torch.randint(0, 33, (B,), generator=g0, dtype=torch.int32)
    pos = starts.unsqueeze(1) + torch.arange(next_n, dtype=torch.int32).unsqueeze(0)
    is_completion = pos % POOL_SIZE == POOL_SIZE - 1
    blk = torch.tensor(blocks, dtype=torch.int32).unsqueeze(1)
    pool_slot = blk * PAGE_SIZE + (POOL_SIZE - 1)
    slot_map = torch.where(is_completion, pool_slot, torch.full_like(pos, -1))
    tail_slot = _tail_slot_for(blocks, pos)

    g = torch.Generator().manual_seed(seed + 1)
    key = _randn((B, next_n, HEAD_DIM), DEV, g)
    score = _randn((B, next_n, HEAD_DIM), DEV, g)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, generator=g, dtype=torch.float32).to(DEV)
    kv, tail = _make_caches(DEV)
    pg = torch.Generator().manual_seed(seed + 1000)
    for b in range(B):
        n_prior = int(starts[b]) % POOL_SIZE
        if n_prior > 0:
            tail[blocks[b], 0, :n_prior, :] = _randn((n_prior, HEAD_DIM), DEV, pg)
            tail[blocks[b], 1, :n_prior, :] = _randn((n_prior, HEAD_DIM), DEV, pg)

    args = (kv, tail, tail_slot.to(DEV), key, score, ape, slot_map.to(DEV), pos.to(DEV))
    ref = kpool_decode_update_reference(*args, pool_size=POOL_SIZE)
    kv_k, tail_k = kv.clone(), tail.clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        kv_k, tail_k, *args[2:], POOL_SIZE)
    assert torch.equal(kv_k.cpu(), ref["kv_cache"])
    assert torch.equal(tail_k.cpu(), ref["tail_kv_cache"])


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("ring_pools", [1, 2])
def test_decode_writer_matches_prefill_writer(pool_size, ring_pools):
    """The production invariant: one-token-per-step decode writes byte-equal
    pools to the prefill compress writer."""
    ring = ring_pools * pool_size
    n_pools, page, nblk = 8, 64, 4
    n_tok = n_pools * pool_size
    g = torch.Generator().manual_seed(0)
    k = _randn((n_tok, HEAD_DIM), DEV, g)
    score = _randn((n_tok, HEAD_DIM), DEV, g)
    ape = torch.randn(pool_size, HEAD_DIM, generator=g, dtype=torch.float32).to(DEV)

    kv_prefill = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEV)
    kpool_compress_and_write_cache(
        kv_prefill, k.view(n_pools, pool_size, HEAD_DIM),
        score.view(n_pools, pool_size, HEAD_DIM), ape,
        torch.arange(n_pools, dtype=torch.int64, device=DEV), pool_size=pool_size)

    kv_decode = torch.zeros_like(kv_prefill)
    tail = torch.zeros(nblk, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    for t in range(n_tok):
        completes = t % pool_size == pool_size - 1
        kpool_decode_update_and_maybe_write_cache_batched(
            kv_decode, tail,
            torch.tensor([[t % ring]], dtype=torch.int32, device=DEV),
            k[t].view(1, 1, HEAD_DIM), score[t].view(1, 1, HEAD_DIM), ape,
            torch.tensor([[t // pool_size if completes else -1]], dtype=torch.int32, device=DEV),
            torch.tensor([[t]], dtype=torch.int32, device=DEV),
            pool_size, HEAD_DIM,
        )
    differing = [
        p for p in range(n_pools)
        if not torch.equal(kv_prefill[p // page, p % page], kv_decode[p // page, p % page])
    ]
    assert not differing, f"{len(differing)}/{n_pools} pools differ: {differing[:5]}"


@pytest.mark.skipif(not CUDA, reason="CUDA kernel test")
@pytest.mark.parametrize("ring_pools", [1, 2])
def test_rejected_draft_redo_needs_ring_slots(ring_pools):
    """A one-pool ring corrupts redo pools after a rejected draft batch — the
    documented sizing constraint (ring >= 2 pools under spec verify)."""
    pool, spec, page, nblk = 4, 3, 64, 2
    ring = ring_pools * pool
    g = torch.Generator().manual_seed(1)
    n_tok = 3 * pool
    k = _randn((n_tok, HEAD_DIM), DEV, g)
    score = _randn((n_tok, HEAD_DIM), DEV, g)
    ape = torch.randn(pool, HEAD_DIM, generator=g, dtype=torch.float32).to(DEV)
    kv_ref = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=DEV)
    kpool_compress_and_write_cache(
        kv_ref, k.view(3, pool, HEAD_DIM), score.view(3, pool, HEAD_DIM), ape,
        torch.arange(3, dtype=torch.int64, device=DEV), pool_size=pool)

    kv = torch.zeros_like(kv_ref)
    tail = torch.zeros(nblk, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=DEV)

    def step(positions, keys, scores):
        pos = torch.tensor([positions], dtype=torch.int32, device=DEV)
        slots = [(p // pool) if p % pool == pool - 1 else -1 for p in positions]
        kpool_decode_update_and_maybe_write_cache_batched(
            kv, tail, pos % ring, keys.view(1, -1, HEAD_DIM), scores.view(1, -1, HEAD_DIM),
            ape, torch.tensor([slots], dtype=torch.int32, device=DEV), pos, pool)

    drafts = _randn((spec, HEAD_DIM), DEV, g)
    draft_scores = _randn((spec, HEAD_DIM), DEV, g)
    for t in range(6):
        step([t], k[t], score[t])
    step([6, 7, 8, 9], torch.cat([k[6:7], drafts]), torch.cat([score[6:7], draft_scores]))
    step([7, 8, 9, 10], k[7:11], score[7:11])  # all drafts rejected -> redo
    pool1_ok = torch.equal(_pool_bytes(kv, 1), _pool_bytes(kv_ref, 1))
    assert pool1_ok == (ring_pools >= 2)
