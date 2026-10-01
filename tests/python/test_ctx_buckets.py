"""CPU-only unit tests for the ctx bucket ladder (fix/04-jit-bucket).

The JIT-cliff fix has two arms: (1) ctx-derived lengths become true runtime
kernel arguments (the Triton kernels themselves — GPU-gated, NOT tested
here), and (2) growing geometry snaps UP to a small, bounded bucket ladder
(``vkernels.torch_ops.ctx_buckets``). Arm (2) is pure host logic — tested
here with no torch, no triton, no CUDA.
"""

from __future__ import annotations

import importlib

import pytest

from vkernels.torch_ops import ctx_buckets


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VK_CTX_BUCKETS", raising=False)
    ctx_buckets.buckets.cache_clear()
    yield
    ctx_buckets.buckets.cache_clear()


def test_default_ladder_is_bounded_and_ascending():
    ladder = ctx_buckets.buckets()
    assert ladder == ctx_buckets.DEFAULT_CTX_BUCKETS
    assert len(ladder) <= 16  # small and bounded: at most |ladder| compile keys
    assert list(ladder) == sorted(set(ladder))
    assert all(b > 0 for b in ladder)


def test_bucket_for_snaps_up_to_nearest_rung():
    assert ctx_buckets.bucket_for(1) == 512
    assert ctx_buckets.bucket_for(511) == 512
    assert ctx_buckets.bucket_for(512) == 512  # boundary: exact hit, no change
    assert ctx_buckets.bucket_for(513) == 1024
    assert ctx_buckets.bucket_for(6934) == 8192  # the sgs 8K operating point


def test_bucket_for_beyond_ladder_clamps_to_top_rung():
    top = ctx_buckets.DEFAULT_CTX_BUCKETS[-1]
    assert ctx_buckets.bucket_for(top + 1) == top
    assert ctx_buckets.bucket_for(10**9) == top


def test_bucket_for_coverage_never_undershoots():
    # Coverage-preserving snap: buffer/loop sizing that must cover every
    # real token never gets a bucket below the true length. Beyond the top
    # rung the true length passes through unchanged.
    assert ctx_buckets.bucket_for_coverage(513) == 1024
    top = ctx_buckets.DEFAULT_CTX_BUCKETS[-1]
    assert ctx_buckets.bucket_for_coverage(top + 1) == top + 1
    assert ctx_buckets.bucket_for_coverage(top * 3) == top * 3


def test_bucket_for_degenerate_inputs():
    assert ctx_buckets.bucket_for(0) == 0
    assert ctx_buckets.bucket_for(-5) == 0
    assert ctx_buckets.bucket_for_coverage(0) == 0


def test_env_custom_ladder(monkeypatch):
    monkeypatch.setenv("VK_CTX_BUCKETS", "128, 512,2048")
    ctx_buckets.buckets.cache_clear()
    assert ctx_buckets.buckets() == (128, 512, 2048)
    assert ctx_buckets.bucket_for(600) == 2048
    assert ctx_buckets.bucket_for(100) == 128


def test_env_off_disables_bucketing(monkeypatch):
    for off in ("0", "off", "false", "none"):
        monkeypatch.setenv("VK_CTX_BUCKETS", off)
        ctx_buckets.buckets.cache_clear()
        assert not ctx_buckets.bucketing_enabled()
        # pass-through: every helper is the identity
        assert ctx_buckets.bucket_for(6934) == 6934
        assert ctx_buckets.bucket_for_coverage(6934) == 6934


def test_env_garbage_tokens_tolerated(monkeypatch):
    monkeypatch.setenv("VK_CTX_BUCKETS", "256, abc, , 1024, -3, 256")
    ctx_buckets.buckets.cache_clear()
    # garbage/dup/negative tokens dropped, valid ones kept ascending
    assert ctx_buckets.buckets() == (256, 1024)


def test_env_cache_respects_lru_clear(monkeypatch):
    monkeypatch.setenv("VK_CTX_BUCKETS", "64,128")
    ctx_buckets.buckets.cache_clear()
    assert ctx_buckets.bucket_for(70) == 128
    monkeypatch.setenv("VK_CTX_BUCKETS", "256")
    # cached until explicitly cleared (callers clear once at config load)
    assert ctx_buckets.bucket_for(70) == 128
    ctx_buckets.buckets.cache_clear()
    assert ctx_buckets.bucket_for(70) == 256


def test_split_geometry_is_step_stable_within_a_rung():
    """The decode_attention_split fix, as a pure-geometry property: with the
    bucketed bound, the (splits, split_len) compile key repeats across the
    per-step growth of the true max length inside one rung."""
    from vkernels.torch_ops.triton_attn import _MAX_KV_SPLITS

    def geometry(max_len, block_n=64):
        max_len = ctx_buckets.bucket_for_coverage(max_len)
        splits = max(1, min(_MAX_KV_SPLITS, (max_len + block_n - 1) // block_n))
        split_len = (max_len + splits - 1) // splits
        return splits, split_len

    # 2050..3071 is strictly inside the 4096 rung: identical geometry every
    # step (2048 itself is a rung boundary and snaps to its own rung)
    keys = {geometry(n) for n in range(2050, 3072)}
    assert len(keys) == 1
    # the sub-first-rung growth (1..512) snaps to the first rung: one key
    assert len({geometry(n) for n in range(1, 512)}) == 1
    # coverage: every geometry covers the true length
    for n in (1, 511, 6934, 60000, 70000):
        splits, split_len = geometry(n)
        assert splits * split_len >= n


def test_split_geometry_key_is_bounded_by_ladder_plus_triton_int_classes():
    """End-to-end boundedness of the compile key: within the ladder, one
    (splits, split_len) key per rung; beyond the top rung the true length
    passes through (coverage), but the key only varies through Triton's
    automatic int specialization classes (== 1, % 16 == 0, other) — so the
    TOTAL distinct key classes stay bounded by |ladder| + 3, independent of
    the context length."""
    from vkernels.torch_ops.triton_attn import _MAX_KV_SPLITS

    def key(max_len, block_n=64):
        max_len = ctx_buckets.bucket_for_coverage(max_len)
        splits = max(1, min(_MAX_KV_SPLITS, (max_len + block_n - 1) // block_n))
        split_len = (max_len + splits - 1) // splits
        return (splits, split_len == 1, split_len % 16 == 0)

    keys = {key(n) for n in range(1, 70000)}
    assert len(keys) <= len(ctx_buckets.buckets()) + 3
