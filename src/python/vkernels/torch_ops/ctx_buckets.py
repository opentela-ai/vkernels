"""Context-length buckets: keep the ctx dimension OUT of Triton compile keys.

The bug this module exists for (sgs-gpu07 8K campaign, 2025-09-29/30): on the
GLM-5.3-Flash decode path, every context-length-derived quantity that reached
a ``@triton.jit`` kernel as a ``tl.constexpr`` (or as a plain int that Triton
auto-specializes) grew by one token per decode step — so the Triton JIT cache
gained ~1 entry per step, 86% of engine time landed in make_llir/make_ptx,
and decode ran ~0.5-1.0 s/step forever beyond the ~2K ctx graph regime (same
engine: 12.6 ms/step at 560 ctx; sglang: 8.78 ms/step at 8K ctx).

The fix pattern, per kernel, is one of:

1. **True runtime argument** (preferred — bit-identical by construction):
   the length only feeds loop bounds and lane masks, so it can be a plain
   kernel argument (``do_not_specialize`` where Triton's int specialization
   — ``== 1`` and ``% 16 == 0`` buckets — would otherwise churn). One
   compiled kernel then serves EVERY context. In-tree precedent:
   ``dsa_kpool_metadata`` (``do_not_specialize=["max_len", "num_splits"]``).
2. **Context buckets** (when a constexpr buys real perf — e.g. it sizes a
   partial-buffer axis or a static loop): snap the ctx-derived length UP to
   the smallest bucket of a small, bounded ladder before it enters the
   launch geometry. Padded work is masked to exact neutrality (see below),
   and the compile key takes one of ~|ladder| values instead of |ctx|.

Bucket ladder
-------------
The default ladder mirrors the batch-size bucket idea (the decode graph bank
captures [1, 2, 4]): geometric x2 steps, spanning the serving envelope, and
BOUNDED — a step at any context compiles at most one new kernel per
ctx-taking kernel per ladder rung, ever.

Config
------
``VK_CTX_BUCKETS`` — comma-separated ascending ints (the full ladder), or
``0`` / ``off`` to disable bucketing entirely (every helper becomes
pass-through; use as the numerics escape hatch). Unset -> the default
ladder below. Parsed once per process and cached.

Bit-compatibility of padding
----------------------------
Every consumer of a bucketed length MUST mask the padded region so results
for the active window are unchanged:

* score/attention kernels: padded candidate lanes load with
  ``mask=offs < true_len, other=0`` and their logits are forced to the
  masked sentinel (``-inf``, or the kernel's ``-1.0e30`` sentinel before the
  sink correction), contributing exactly zero to the online-softmax
  accumulator — ``exp(-inf - m) = 0`` and the running max/denominator are
  unchanged (the ``m == -inf`` fully-masked-prefix corner is already handled
  by the in-kernel sentinels these kernels use).
* split-KV partials: splits whose slice starts at or beyond the true length
  write the neutral partial (``l = 0``, ``m = -inf``/sentinel, ``acc = 0``);
  the stage-2 LSE merge of a neutral partial is the identity.
* top-k/index buffers: padded entries keep the up-front fill (``-1`` /
  ``0.0``) exactly as the eager tail already emits.

What masking does NOT guarantee: a *different partition* of the same tokens
(e.g. different split boundaries) reorders fp32 additions inside the online
softmax. Mathematically identical, and the same perf-only rounding class as
retuning ``BLOCK_N`` — but not bit-identical to a run with a different
partition. Within one bucket the partition is FIXED, so results are
step-stable (which the per-step-growing geometry never was). Callers that
need the exact legacy partition pass their static length (it lands on a
bucket boundary or keeps the pass-through hint path).
"""

from __future__ import annotations

import os
from functools import lru_cache

__all__ = [
    "DEFAULT_CTX_BUCKETS",
    "buckets",
    "bucket_for",
    "bucket_for_coverage",
    "bucketing_enabled",
]

DEFAULT_CTX_BUCKETS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)

_ENV_VAR = "VK_CTX_BUCKETS"


@lru_cache(maxsize=1)
def buckets() -> tuple[int, ...]:
    """The configured ctx bucket ladder (ascending, deduplicated)."""
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return DEFAULT_CTX_BUCKETS  # unset -> the sane default ladder
    raw = raw.strip()
    if not raw or raw.lower() in ("0", "off", "false", "none"):
        return ()  # disabled: every helper is pass-through
    out: list[int] = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError:
            continue  # tolerate garbage tokens; config must never crash serving
        if v > 0:
            out.append(v)
    return tuple(sorted(set(out)))


def bucketing_enabled() -> bool:
    """False when ``VK_CTX_BUCKETS`` disables the ladder."""
    return bool(buckets())


def bucket_for(n: int) -> int:
    """Smallest configured bucket >= n (n <= 0 -> 0; disabled/oversized -> n).

    Overshoot clamps to the LAST bucket: a context beyond the ladder's top
    rung is served by the top bucket's kernel shape with the true length
    carried at runtime/masked — never by a new compile. That keeps the
    bucket count exactly ``len(buckets())`` for every possible context.
    """
    if n <= 0:
        return 0
    ladder = buckets()
    if not ladder:
        return n
    for b in ladder:
        if n <= b:
            return b
    return ladder[-1]


def bucket_for_coverage(n: int) -> int:
    """``bucket_for(n)``, except never below ``n``: contexts beyond the top
    rung pass through unchanged (coverage-preserving snap for lengths that
    size buffers/loops which MUST cover every real token). Pass-through
    beyond the top rung re-opens the unbounded key only past the ladder's
    envelope — by design the envelope is the serving max context; kernels
    whose padded length may also undershoot must instead carry the true
    length at runtime and use the bucket for loop/buffer geometry only."""
    b = bucket_for(n)
    return b if b >= n else n
    
