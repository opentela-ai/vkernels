"""Op-level tuning: the tune-once-persist-always layer above the kernel tier.

:mod:`.cache` is the op-config cache (``VKERNELS_CACHE``): per-op launch
configs keyed by (op, kernel-source fingerprint, arch identity, coarse
shape class), persisted as human-readable JSON and resolved once per
process before any graph capture can observe the op.
"""

from .cache import op_config, seed, stored_records, reset_memo

__all__ = ["op_config", "seed", "stored_records", "reset_memo"]
