"""A reference batch must preserve independent request histories."""

from dataclasses import replace

import numpy as np
import pytest

from vkernels.compiler.qwen3_arch import (
    Qwen3KVCache,
    qwen3_reference_forward,
    random_qwen3_weights,
    tiny_qwen3_config,
)


@pytest.mark.parametrize("batch", [2, 4])
def test_batched_oracle_matches_independent_rows(batch):
    cfg = replace(
        tiny_qwen3_config(layers=2, batch=batch, cache_capacity=4),
        vocab=32,
        hidden=16,
        heads=2,
        kv_heads=1,
        head_dim=8,
        intermediate=32,
    )
    single = replace(cfg, batch=1)
    weights = random_qwen3_weights(cfg, seed=31)
    batched_cache = Qwen3KVCache(cfg)
    row_caches = [Qwen3KVCache(single) for _ in range(batch)]
    tokens = np.random.default_rng(19).integers(0, cfg.vocab, (cfg.cache_capacity, batch))
    for position, ids in enumerate(tokens):
        actual, _ = qwen3_reference_forward(weights, ids, batched_cache, position, cfg)
        for row, cache in enumerate(row_caches):
            expected, _ = qwen3_reference_forward(weights, ids[row : row + 1], cache, position, single)
            np.testing.assert_allclose(actual[row], expected[0], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(
                batched_cache.k[:, row : row + 1], cache.k, atol=1e-12, rtol=1e-12, equal_nan=True
            )
            np.testing.assert_allclose(
                batched_cache.v[:, row : row + 1], cache.v, atol=1e-12, rtol=1e-12, equal_nan=True
            )
