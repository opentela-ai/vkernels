"""Correctness gates for the ported DFLASH speculative-decoding kernels.

These mirror the SGLang reference (python/sglang/kernels/ops/speculative/
dflash.py @ 30e7a30) and validate the three kernels -- ``accept_bonus``,
``prepare_draft_block``, ``selector_walk`` -- against trivial Python
references. The kernels are pure (no engine/scheduler coupling), so the
tests stand alone.

triton is CUDA-only on linux, so the whole module is skipped without it;
the numeric gates additionally require CUDA (matching the convention in
``test_elementwise.py``). The validation tests below launch no
kernel -- they raise before any device call -- so they run on any
triton-capable box.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")
import torch

from vkernels.torch_ops.dflash import (
    accept_bonus,
    prepare_draft_block,
    selector_walk,
)
from vkernels.torch_ops.dflash import (
    _is_row_major_contiguous_2d,
    _pick_num_warps,
)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# ---------------------------------------------------------------------------
# pure helpers (no kernel launch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block,warps",
    [
        (16, 1),
        (17, 2),  # next_pow2(17)=32 -> 2 warps
        (32, 2),
        (64, 4),
        (65, 8),  # next_pow2(65)=128 -> 8 warps
        (128, 8),
    ],
)
def test_pick_num_warps(block, warps):
    assert _pick_num_warps(block) == warps


def test_is_row_major_contiguous_2d():
    a = torch.arange(6).reshape(2, 3)
    assert _is_row_major_contiguous_2d(a)
    assert not _is_row_major_contiguous_2d(a.t())  # column-major view
    assert not _is_row_major_contiguous_2d(torch.arange(3))  # 1D, not 2D
    assert not _is_row_major_contiguous_2d(a[:, ::2])  # strided


# ---------------------------------------------------------------------------
# accept_bonus
# ---------------------------------------------------------------------------


def _ref_accept_bonus(candidates, target_top1, prefix_lens):
    """Trivial per-row longest-prefix match mirroring the Triton kernel.

    Drafted tokens are ``candidates[:, 1:]`` compared against
    ``target_top1[:, col]`` at positions ``0..block_size-2``; the bonus is
    the target's token just past the last accepted draft; the committed
    block is the accepted drafts left-shifted with the bonus placed at
    index ``accept_len``.
    """
    B, bs = candidates.shape
    accept = torch.zeros(B, dtype=torch.int32)
    commit = torch.zeros(B, dtype=torch.int32)
    bonus = torch.zeros(B, dtype=target_top1.dtype)
    new_seq = torch.zeros(B, dtype=prefix_lens.dtype)
    out_tokens = torch.zeros(B, bs, dtype=candidates.dtype)
    for r in range(B):
        a = 0
        for i in range(bs - 1):
            if int(candidates[r, i + 1]) == int(target_top1[r, i]):
                a += 1
            else:
                break
        c = a + 1
        b = int(target_top1[r, a])
        accept[r] = a
        commit[r] = c
        bonus[r] = b
        new_seq[r] = int(prefix_lens[r]) + c
        for col in range(bs):
            if col < bs - 1:
                out_tokens[r, col] = candidates[r, col + 1]
            if col == a:
                out_tokens[r, col] = b
    return accept, commit, bonus, out_tokens, new_seq


@gpu
@pytest.mark.parametrize("B,bs", [(1, 8), (4, 16), (3, 17), (2, 1)])
def test_accept_bonus_matches_reference(B, bs):
    torch.manual_seed(7)
    dev = "cuda"
    candidates = torch.randint(0, 100, (B, bs), device=dev, dtype=torch.int64)
    target_top1 = torch.randint(0, 100, (B, bs), device=dev, dtype=torch.int64)
    prefix_lens = torch.randint(0, 50, (B,), device=dev, dtype=torch.int64)
    accept = torch.empty(B, dtype=torch.int32, device=dev)
    commit = torch.empty(B, dtype=torch.int32, device=dev)
    bonus = torch.empty(B, dtype=torch.int64, device=dev)
    out_tokens = torch.empty(B, bs, dtype=torch.int64, device=dev)
    new_seq = torch.empty(B, dtype=torch.int64, device=dev)
    accept_bonus(
        candidates,
        target_top1,
        accept,
        commit,
        bonus,
        out_tokens,
        prefix_lens,
        new_seq,
    )
    r_acc, r_com, r_bon, r_out, r_new = _ref_accept_bonus(candidates.cpu(), target_top1.cpu(), prefix_lens.cpu())
    torch.testing.assert_close(accept.cpu(), r_acc, rtol=0, atol=0)
    torch.testing.assert_close(commit.cpu(), r_com, rtol=0, atol=0)
    torch.testing.assert_close(bonus.cpu(), r_bon, rtol=0, atol=0)
    torch.testing.assert_close(out_tokens.cpu(), r_out, rtol=0, atol=0)
    torch.testing.assert_close(new_seq.cpu(), r_new, rtol=0, atol=0)


@gpu
def test_accept_bonus_full_match_emits_block_minus_one():
    """Every draft matches: accept_len = bs-1, commit = bs, bonus is the
    target's last position, and the whole committed block is accepted
    drafts followed by the bonus."""
    dev = "cuda"
    bs = 8
    # drafted tail equals target positions 0..bs-2
    candidates = torch.zeros(1, bs, dtype=torch.int64, device=dev)
    target = torch.arange(1, bs + 1, dtype=torch.int64, device=dev).unsqueeze(0)
    candidates[:, 1:] = target[:, : bs - 1]  # pos 0 seed is unused by the walk
    prefix_lens = torch.tensor([10], dtype=torch.int64, device=dev)
    accept = torch.empty(1, dtype=torch.int32, device=dev)
    commit = torch.empty(1, dtype=torch.int32, device=dev)
    bonus = torch.empty(1, dtype=torch.int64, device=dev)
    out_tokens = torch.empty(1, bs, dtype=torch.int64, device=dev)
    new_seq = torch.empty(1, dtype=torch.int64, device=dev)
    accept_bonus(candidates, target, accept, commit, bonus, out_tokens, prefix_lens, new_seq)
    assert int(accept[0]) == bs - 1
    assert int(commit[0]) == bs
    assert int(bonus[0]) == int(target[0, bs - 1])
    assert int(new_seq[0]) == 10 + bs
    # committed block: accepted drafts (== target[:bs-1]) then bonus
    expected = torch.cat([target[0, : bs - 1], target[0, bs - 1 :]]).cpu()
    torch.testing.assert_close(out_tokens[0].cpu(), expected, rtol=0, atol=0)


@gpu
def test_accept_bonus_zero_match_bonus_is_first_target():
    """First draft mismatches: accept_len = 0, commit = 1, bonus is the
    target's position-0 token (the only committed token)."""
    dev = "cuda"
    bs = 8
    candidates = torch.zeros(1, bs, dtype=torch.int64, device=dev)
    candidates[:, 1] = 999  # guaranteed mismatch vs target[0]
    target = torch.arange(1, bs + 1, dtype=torch.int64, device=dev).unsqueeze(0)
    prefix_lens = torch.tensor([5], dtype=torch.int64, device=dev)
    accept = torch.empty(1, dtype=torch.int32, device=dev)
    commit = torch.empty(1, dtype=torch.int32, device=dev)
    bonus = torch.empty(1, dtype=torch.int64, device=dev)
    out_tokens = torch.empty(1, bs, dtype=torch.int64, device=dev)
    new_seq = torch.empty(1, dtype=torch.int64, device=dev)
    accept_bonus(candidates, target, accept, commit, bonus, out_tokens, prefix_lens, new_seq)
    assert int(accept[0]) == 0
    assert int(commit[0]) == 1
    assert int(bonus[0]) == int(target[0, 0])
    assert int(out_tokens[0, 0]) == int(target[0, 0])


def test_accept_bonus_validates_inputs():
    # Non-contiguous candidates: must raise before any kernel launch.
    a = torch.arange(16, dtype=torch.int64).reshape(2, 8)[:, ::2]
    out = torch.empty(2, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="contiguous candidates"):
        accept_bonus(
            a,
            a.contiguous(),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int64),
            out,
            torch.empty(2, dtype=torch.int64),
            torch.empty(2, dtype=torch.int64),
        )


def test_accept_bonus_requires_1d_prefix_lens():
    a = torch.zeros(2, 8, dtype=torch.int64)
    with pytest.raises(ValueError, match="1D prefix_lens"):
        accept_bonus(
            a,
            a,
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int64),
            torch.empty(2, 8, dtype=torch.int64),
            torch.empty(2, 1, dtype=torch.int64),  # 2D, not 1D
            torch.empty(2, dtype=torch.int64),
        )


# ---------------------------------------------------------------------------
# prepare_draft_block
# ---------------------------------------------------------------------------


@gpu
@pytest.mark.parametrize("B,bs,width", [(1, 8, 64), (3, 16, 64), (2, 17, 40)])
def test_prepare_draft_block_matches_reference(B, bs, width):
    torch.manual_seed(11)
    dev = "cuda"
    mask_token_id = 32000
    bonus_tokens = torch.randint(1, 100, (B,), device=dev, dtype=torch.int64)
    prefix_lens = torch.randint(0, width - bs, (B,), device=dev, dtype=torch.int64)
    req_pool_indices = torch.arange(B, device=dev, dtype=torch.int32)
    req_to_token = torch.randint(1, 4096, (B, width), device=dev, dtype=torch.int32)
    block_ids = torch.empty(B, bs, dtype=torch.int64, device=dev)
    positions = torch.empty(B, bs, dtype=torch.int64, device=dev)
    cache_loc = torch.empty(B, bs, dtype=torch.int64, device=dev)
    prepare_draft_block(
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids,
        positions,
        cache_loc,
        mask_token_id,
    )

    g_block_ids = torch.full((B, bs), mask_token_id, dtype=torch.int64)
    g_positions = torch.zeros((B, bs), dtype=torch.int64)
    g_cache_loc = torch.zeros((B, bs), dtype=torch.int64)
    for r in range(B):
        for col in range(bs):
            pos = int(prefix_lens[r]) + col
            if col == 0:
                g_block_ids[r, col] = int(bonus_tokens[r])
            g_positions[r, col] = pos
            if pos < width:
                g_cache_loc[r, col] = int(req_to_token[int(req_pool_indices[r]), pos])

    torch.testing.assert_close(block_ids.cpu(), g_block_ids, rtol=0, atol=0)
    torch.testing.assert_close(positions.cpu(), g_positions, rtol=0, atol=0)
    torch.testing.assert_close(cache_loc.cpu(), g_cache_loc, rtol=0, atol=0)


@gpu
def test_prepare_draft_block_clamps_past_width_to_zero_cache_loc():
    """Logical positions past the table width write a zero cache location
    (the kernel masks those loads to 0)."""
    dev = "cuda"
    B, bs, width = 1, 8, 5  # prefix 0 => positions 0..7, but width is 5
    bonus_tokens = torch.tensor([42], dtype=torch.int64, device=dev)
    prefix_lens = torch.tensor([0], dtype=torch.int64, device=dev)
    req_pool_indices = torch.tensor([0], dtype=torch.int32, device=dev)
    req_to_token = torch.arange(1, width + 1, dtype=torch.int32, device=dev).unsqueeze(0)
    block_ids = torch.empty(B, bs, dtype=torch.int64, device=dev)
    positions = torch.empty(B, bs, dtype=torch.int64, device=dev)
    cache_loc = torch.empty(B, bs, dtype=torch.int64, device=dev)
    prepare_draft_block(
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids,
        positions,
        cache_loc,
        32000,
    )
    assert int(block_ids[0, 0]) == 42
    assert int(block_ids[0, 1]) == 32000
    # positions 0..4 read real slot ids; 5..7 clamp to 0
    real = [int(req_to_token[0, p]) for p in range(5)] + [0, 0, 0]
    torch.testing.assert_close(cache_loc[0].cpu(), torch.tensor(real), rtol=0, atol=0)


def test_prepare_draft_block_requires_row_major_req_to_token():
    # A transposed [rows>1] matrix has stride(1) != 1 (column-major), so the
    # kernel rejects it before any device call.
    req_to_token = torch.arange(6, dtype=torch.int32).reshape(2, 3).t()  # [3,2], strides (1,3)
    with pytest.raises(ValueError, match="row-major req_to_token"):
        prepare_draft_block(
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int32),
            req_to_token,
            torch.empty(1, 8, dtype=torch.int64),
            torch.empty(1, 8, dtype=torch.int64),
            torch.empty(1, 8, dtype=torch.int64),
            32000,
        )


# ---------------------------------------------------------------------------
# selector_walk
# ---------------------------------------------------------------------------


def _ref_selector_walk(candidate_ids, scores, uniforms, temperatures, greedy_mask):
    B, slots, top_k = candidate_ids.shape
    tokens = torch.zeros((B, slots), dtype=torch.int64)
    q = torch.zeros((B, slots, top_k), dtype=torch.float32)
    offs = torch.arange(top_k)
    for r in range(B):
        temp = float(temperatures[r])
        greedy = int(greedy_mask[r]) != 0
        previous = 0
        for s in range(slots):
            sc = scores[r, s, previous, :].float()
            if greedy:
                best = sc.max()
                idx = int(torch.where(sc == best, offs, top_k).min().item())
                probs = torch.where(offs == idx, 1.0, 0.0).float()
            else:
                scaled = sc / temp
                exps = torch.exp(scaled - scaled.max())
                probs = exps / exps.sum()
                u = float(uniforms[r, s])
                idx = int(torch.sum(torch.where(torch.tensor(u) >= torch.cumsum(probs, 0), 1, 0)).item())
                idx = min(idx, top_k - 1)
            q[r, s] = probs
            tokens[r, s] = int(candidate_ids[r, s, idx])
            previous = idx
    return tokens, q


@gpu
@pytest.mark.parametrize("B,slots,top_k", [(1, 4, 4), (2, 8, 8), (3, 2, 16)])
def test_selector_walk_greedy_matches_reference(B, slots, top_k):
    torch.manual_seed(13)
    dev = "cuda"
    candidate_ids = torch.randint(0, 32000, (B, slots, top_k), device=dev, dtype=torch.int64)
    scores = torch.randn(B, slots, top_k, top_k, device=dev, dtype=torch.float32)
    uniforms = torch.rand(B, slots, device=dev, dtype=torch.float32)
    temperatures = torch.ones(B, device=dev, dtype=torch.float32)
    greedy_mask = torch.ones(B, dtype=torch.int32, device=dev)  # all greedy
    tokens, q = selector_walk(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    r_tok, r_q = _ref_selector_walk(candidate_ids.cpu(), scores.cpu(), uniforms.cpu(), temperatures.cpu(), greedy_mask.cpu())
    torch.testing.assert_close(tokens.cpu(), r_tok, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), r_q, rtol=0, atol=1e-6)


@gpu
@pytest.mark.parametrize("temperature", [1.0, 0.7, 2.0])
def test_selector_walk_sampling_matches_reference(temperature):
    torch.manual_seed(17)
    dev = "cuda"
    B, slots, top_k = 2, 6, 8
    candidate_ids = torch.randint(0, 32000, (B, slots, top_k), device=dev, dtype=torch.int64)
    scores = torch.randn(B, slots, top_k, top_k, device=dev, dtype=torch.float32)
    uniforms = torch.rand(B, slots, device=dev, dtype=torch.float32)
    temperatures = torch.full((B,), temperature, device=dev, dtype=torch.float32)
    greedy_mask = torch.zeros(B, dtype=torch.int32, device=dev)  # all sampled
    tokens, q = selector_walk(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    r_tok, r_q = _ref_selector_walk(candidate_ids.cpu(), scores.cpu(), uniforms.cpu(), temperatures.cpu(), greedy_mask.cpu())
    torch.testing.assert_close(tokens.cpu(), r_tok, rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), r_q, rtol=0, atol=1e-6)


@gpu
def test_selector_walk_clamps_uniform_one_to_last_candidate():
    """A uniform of 1.0 (>= every cumsum entry) would index top_k; the kernel
    clamps to top_k-1, so the last candidate is always selected."""
    dev = "cuda"
    B, slots, top_k = 1, 3, 4
    candidate_ids = torch.arange(top_k, dtype=torch.int64, device=dev).view(1, 1, top_k).expand(B, slots, top_k).contiguous()
    scores = torch.zeros(B, slots, top_k, top_k, device=dev, dtype=torch.float32)
    # uniform softmax: every distribution is uniform, so cumsum = [0.25,0.5,0.75,1]
    uniforms = torch.ones(B, slots, device=dev, dtype=torch.float32)
    temperatures = torch.ones(B, device=dev, dtype=torch.float32)
    greedy_mask = torch.zeros(B, dtype=torch.int32, device=dev)
    tokens, _ = selector_walk(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=greedy_mask,
    )
    # u=1.0 >= 1.0 (last cumsum) -> index 4 -> clamped to 3 -> last candidate
    assert torch.equal(tokens.cpu(), torch.full((B, slots), top_k - 1, dtype=torch.int64))
