"""SGLang v0.5.20 DSA sparse-fwd donor: ``flash_mla_sparse_fwd`` + pinned
configs through the op-config cache (lane 32, GB10 engagement).

DONOR
-----
SGLang v0.5.20's DSA sparse stack — ``--attention-backend dsa
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang`` and
``sgl_kernel.flash_mla.flash_mla_sparse_fwd`` (the FlashMLA token-level
sparse kernels). Boundary contract taken verbatim from
``python/sglang/kernels/aot/python/sgl_kernel/flash_mla.py`` @ v0.5.20
(validated boundary vs SGLang 0.5.18.post1 per the KB doc
glm53-sparse-residency-path):

* ``q``       ``[s_q, h_q, d_qk]`` bf16
* ``kv``      ``[s_kv, 1, d_qk]`` bf16 (MQA — one shared KV head)
* ``indices`` ``[s_q, 1, topk]``  int32; invalid slots are ``-1`` or
  ``>= s_kv`` (masked, never read)
* ``sm_scale`` float, natural-exp units (NO log2(e) prefactor)
* ``d_v``     value width (donor kernels: 512 only)
* ``attn_sink`` optional per-head fp32 sinks; ``topk_length`` optional
  per-row valid counts overriding the ``>= 0`` census
* returns ``(out [s_q, h_q, d_v] bf16, max_logits [s_q, h_q] fp32,
  lse [s_q, h_q] fp32 base-2)``

BACKENDS
--------
``backend`` selects (or ``None`` auto-resolves once per process):

* ``"sgl_kernel"`` — the donor native kernel, when ``sgl_kernel`` is
  importable AND its probe runs on this device. Zero re-implementation:
  arguments are validated then delegated; the donor returns the exact
  SGLang boundary.
* ``"triton"`` — the in-tree vendored vLLM Triton sparse-MLA kernels
  (:mod:`vkernels.torch_ops.vllm_sparse_mla`), the portable backend and
  the one that runs on GB10 today (sgl_kernel publishes no sm_121 /
  aarch64 wheels; FlashMLA compiles for SM90/SM100 only). The flat-cache
  layout difference is pure view arithmetic: the donor's
  ``[s_kv, 1, d_qk]`` MQA cache IS the vendored kernels' ``[rows, d]``
  flat cache. Deviations (documented, none silent): the Triton kernels
  do not produce max_logits/lse — those come back empty unless
  ``return_stats`` forces the fp32 recompute; d_v must equal d_qk
  (the vendored kernels have no separate value width).
* ``"torch"`` — fp32 reference oracle (tests / CPU). Not the fast path.

PINNED CONFIGS (op-config cache)
--------------------------------
Launch configs are pinned per coarse shape class through the op tier
(:func:`vkernels.tuning.cache.op_config`, ``VKERNELS_CACHE``): decode
``{num_splits, num_stages, num_warps}`` over the split-K partial+reduce
pair, prefill ``{block_k, num_warps}``. Defaults are the in-tree pins
(the ``_decode_num_splits`` fill heuristic; block_k 16 at head_dim >=
256); on a miss the bounded sweep benches the candidate space at the
live shape once and persists the winner. During a CUDA graph capture
``op_config`` serves the memo or the declared default instantly — it
never benches, never writes — and :func:`warmup` resolves every serve
shape class BEFORE capture runs (the floe ordering contract's other
half).

CAPTURE SAFETY
--------------
Every device op on the Triton backend is shape-static per call: no host
syncs, no ``.item()``, no data-dependent branching; allocations and
index compaction ride the caller's stream. The only capture hazards the
op tier names — a tune-on-miss sweep and a store write — are excluded by
construction (capture contract above) and by the warmup ordering.

GB10 STATUS (lane 32)
---------------------
Engaged behind floe's opt-in ``dsa_sparse_fwd`` dispatch knob as the
device path for DSA decode + prefill; the incumbent arms (HIP
split-kernel, vendored-Triton ``sparse_mla``, torch body, CPU oracle)
stay untouched as the escape hatch. Short-context (the 22-token bench
class) is NOT expected to beat the dense-window incumbent — the sparse
path's value is path parity with SGLang at every context and the
long-context regime (see .agents/runs/dsa-sparse/ in floe).
"""

from __future__ import annotations

import math

__all__ = [
    "flash_mla_sparse_fwd",
    "sparse_decode_splits_for",
    "decode_shape_class",
    "prefill_shape_class",
    "warmup",
    "active_backend",
    "BACKENDS",
]

BACKENDS = ("sgl_kernel", "triton", "torch")

_OP_DECODE = "sgl_sparse_mla.decode"
_OP_PREFILL = "sgl_sparse_mla.prefill"

_active: dict[str, str] = {}   # device type -> resolved backend
_sgl_probe: bool | None = None


# ---------------------------------------------------------------------------
# backend resolution
# ---------------------------------------------------------------------------


def _sgl_kernel_available() -> bool:
    """Probe the donor kernel once per process.

    Import alone is not enough: the wheel must carry this device's arch
    (sm_121/aarch64 has no published sgl_kernel). The probe is a tiny
    launch on the current device — run it only on CUDA.
    """
    global _sgl_probe
    if _sgl_probe is not None:
        return _sgl_probe
    _sgl_probe = False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd  # ty: ignore[unresolved-import]  # noqa: F401  # optional dep
    except Exception:
        return False
    try:
        dev = torch.device("cuda")
        q = torch.zeros(1, 1, 512, device=dev, dtype=torch.bfloat16)
        kv = torch.zeros(1, 1, 512, device=dev, dtype=torch.bfloat16)
        idx = torch.zeros(1, 1, 64, device=dev, dtype=torch.int32)
        flash_mla_sparse_fwd(q, kv, idx, 0.04, d_v=512)
        _sgl_probe = True
    except Exception:
        _sgl_probe = False
    return _sgl_probe


def active_backend(device=None) -> str:
    """The resolved backend for ``device`` (auto: sgl_kernel > triton > torch).

    Resolved once per device TYPE per process — a CPU-box resolution never
    poisons the CUDA one and vice versa."""
    import torch

    dev = torch.device(device) if device is not None else torch.device("cuda")
    kind = dev.type
    hit = _active.get(kind)
    if hit is not None:
        return hit
    if kind == "cuda" and _sgl_kernel_available():
        resolved = "sgl_kernel"
    elif kind == "cuda":
        resolved = "triton"
    else:
        resolved = "torch"
    _active[kind] = resolved
    return resolved


def _resolve_backend(backend, device=None) -> str:
    if backend is None:
        return active_backend(device)
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if backend == "sgl_kernel" and not _sgl_kernel_available():
        raise RuntimeError(
            "backend='sgl_kernel': sgl_kernel.flash_mla.flash_mla_sparse_fwd "
            "is not importable/usable on this device")
    if backend == "triton":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("backend='triton' requires CUDA")
    return backend


# ---------------------------------------------------------------------------
# pinned configs — the op-config cache tier
# ---------------------------------------------------------------------------


def decode_shape_class(num_queries: int, topk: int) -> str:
    """Coarse decode tier (tree gating conventions, never exact shapes).

    Row tier first (t1 / t8 / t64 — the decode-graph batch ladder), then
    the topk tier (the GLM index width class). One tuned config serves
    the whole bucket.
    """
    if num_queries <= 1:
        rows = "t1"
    elif num_queries <= 8:
        rows = "t8"
    else:
        rows = "t64"
    return f"{rows}.topk{max(1, (topk + 1023) // 1024)}k"


def prefill_shape_class(head_dim: int) -> str:
    """Coarse prefill tier: the head-dim class drives the tile pins."""
    return f"prefill.d{head_dim}"


def _decode_default(num_queries: int, num_heads: int, topk: int, block_k: int):
    """The declared default: the in-tree fill heuristic's pins."""
    from .vllm_sparse_mla import _decode_num_splits

    heads_blocks = -(-num_heads // 16)  # ceil; matches the wrapper's block_h
    num_splits = _decode_num_splits(num_queries, heads_blocks, float(topk), block_k)
    return {"num_splits": int(num_splits), "num_stages": 1, "num_warps": 4}


def _decode_candidates(default):
    """Split/stage space around the default. 16 splits caps the search:
    beyond that the reduce/HBM overhead dominates (the heuristic's own
    bound)."""
    cands = []
    for splits in range(1, 17):
        for stages in (1, 2):
            cands.append({"num_splits": splits, "num_stages": stages,
                          "num_warps": default["num_warps"]})
    return cands


def _decode_config(num_queries, num_heads, topk, block_k, q, kv_cache, indices):
    """Cache-first pinned decode config for this shape class.

    Memoized per exact shape on top of the per-bucket store (the
    ``dense_gemv`` adoption pattern). Capture-safe: ``op_config`` serves
    the memo or the declared default while a stream is capturing.
    """
    hit = _DECODE_CFG.get((num_queries, num_heads, topk, block_k))
    if hit is not None:
        return hit
    from ..tuning.cache import op_config
    from triton.testing import do_bench

    shape_class = decode_shape_class(num_queries, topk)
    default = _decode_default(num_queries, num_heads, topk, block_k)

    def bench(cfg):
        def launch():
            from .vllm_sparse_mla import sparse_attn_decode

            sparse_attn_decode(
                q, kv_cache, indices,
                0.04, num_splits=cfg["num_splits"],
                num_stages=cfg["num_stages"], num_warps=cfg["num_warps"],
                block_k=block_k,
            )

        return do_bench(launch, return_mode="median")

    cfg, _status = op_config(
        _OP_DECODE, shape_class,
        default=default, candidates=_decode_candidates(default),
        bench=bench, source_files=(__file__,),
    )
    resolved = (int(cfg["num_splits"]), int(cfg["num_stages"]), int(cfg["num_warps"]))
    _DECODE_CFG[(num_queries, num_heads, topk, block_k)] = resolved
    return resolved


_DECODE_CFG: dict = {}
_PREFILL_CFG: dict = {}


def _prefill_config(head_dim, q, kv, indices):
    """Cache-first pinned prefill config (block_k / num_warps)."""
    hit = _PREFILL_CFG.get(head_dim)
    if hit is not None:
        return hit
    from ..tuning.cache import op_config
    from triton.testing import do_bench

    default = {"block_k": 16 if head_dim >= 256 else 32, "num_warps": 4}
    candidates = [{"block_k": bk, "num_warps": w}
                  for bk in (16, 32, 64) if bk <= head_dim
                  for w in (4, 8)]

    def bench(cfg):
        from .vllm_sparse_mla import sparse_attn_prefill

        def launch():
            sparse_attn_prefill(q, kv, indices, 0.04,
                                block_k=cfg["block_k"], num_warps=cfg["num_warps"])

        return do_bench(launch, return_mode="median")

    cfg, _status = op_config(
        _OP_PREFILL, prefill_shape_class(head_dim),
        default=default, candidates=candidates,
        bench=bench, source_files=(__file__,),
    )
    resolved = (int(cfg["block_k"]), int(cfg["num_warps"]))
    _PREFILL_CFG[head_dim] = resolved
    return resolved


def sparse_decode_splits_for(num_queries: int, num_heads: int, topk: int,
                             block_k: int = 32) -> int:
    """The pinned decode split count for a shape (host arithmetic only).

    The store-consulting seam for callers that own their own launch
    (floe wiring reports, the crossover bench): falls to the declared
    default's heuristic on any miss without touching the GPU.
    """
    return _decode_default(num_queries, num_heads, topk, block_k)["num_splits"]


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


def _validate(q, kv, indices, d_v, topk_length):
    import torch

    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError(
            f"expected q [s_q,h,d] kv [s_kv,1,d] indices [s_q,1,topk]; got "
            f"{tuple(q.shape)} {tuple(kv.shape)} {tuple(indices.shape)}")
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, kv_d = kv.shape
    if h_kv != 1 or indices.shape[1] != 1:
        raise ValueError("MQA contract: kv and indices carry ONE shared head")
    if indices.shape[0] != s_q:
        raise ValueError(f"indices rows {indices.shape[0]} != s_q {s_q}")
    if d_qk != kv_d:
        raise ValueError(f"d_qk {d_qk} != kv head dim {kv_d}")
    if indices.dtype != torch.int32:
        raise ValueError(f"indices must be int32, got {indices.dtype}")
    if topk_length is not None and (
            topk_length.shape != (s_q,) or topk_length.dtype != torch.int32):
        raise ValueError(f"topk_length must be int32 [{s_q}], got "
                         f"{tuple(topk_length.shape)} {topk_length.dtype}")
    if d_v <= 0 or d_v > d_qk:
        raise ValueError(f"d_v {d_v} outside (0, d_qk {d_qk}]")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("donor contract: bf16 q/kv")
    return s_q, h_q, d_qk, s_kv


def _stats_from_output(q, kv, indices, scale, out, topk_length):
    """fp32 max_logits / base-2 lse recompute for the Triton backend.

    The vendored kernels fuse the softmax internally and expose no split
    state, so the stats are recomputed from the selected rows (fp32).
    Off by default: SGLang callers discard them (``o, _, _ =``).
    """
    import torch

    safe = indices.clamp(0, max(kv.shape[0] - 1, 0)).long()  # never OOB
    gathered = kv[:, 0, :][safe[:, 0]]       # [s_q, topk, d]
    scores = torch.einsum("shd,swd->shw", q.float(), gathered.float()) * scale
    valid = (indices >= 0)
    if kv.shape[0] > 0:
        valid = valid & (indices < kv.shape[0])
    if topk_length is not None:
        ar = torch.arange(indices.shape[-1], device=indices.device)
        valid = valid & (ar[None, None, :] < topk_length[:, None, None].to(torch.int32))
    scores = scores.masked_fill(~valid, float("-inf"))
    max_logits = scores.amax(dim=-1)         # [s_q, h]
    lse_nat = torch.logsumexp(scores.where(valid, float("-inf")), dim=-1)
    lse = lse_nat / math.log(2.0)            # donor lse is base-2
    return max_logits, lse


def _run_triton(q, kv, indices, scale, d_v, attn_sink, topk_length, return_stats):
    import torch

    from .vllm_sparse_mla import sparse_attn_decode, sparse_attn_prefill

    s_q, h_q, d_qk, s_kv = _validate(q, kv, indices, d_v, topk_length)
    if d_v != d_qk:
        raise ValueError(
            f"triton backend has no separate value width (d_v {d_v} != d_qk {d_qk})")
    if attn_sink is not None:
        raise ValueError("triton backend does not support attn_sink")
    if s_q == 1:
        block_k = 32
        num_splits, num_stages, num_warps = _decode_config(
            1, h_q, indices.shape[-1], block_k, q, kv, indices)
        out = sparse_attn_decode(
            q, kv[:, 0, :], indices[:, 0, :], scale,
            topk_length=topk_length,
            block_k=block_k, num_splits=num_splits,
            num_stages=num_stages, num_warps=num_warps,
        )
    else:
        block_k, num_warps = _prefill_config(d_qk, q, kv[:, 0, :], indices[:, 0, :])
        out = sparse_attn_prefill(
            q, kv[:, 0, :], indices[:, 0, :], scale,
            topk_length=topk_length,
            block_k=block_k, num_warps=num_warps,
        )
    if return_stats:
        ml, lse = _stats_from_output(q, kv, indices, scale, out, topk_length)
        return out, ml, lse
    empty = torch.empty(0, device=q.device, dtype=torch.float32)
    return out, empty, empty.clone()


def _run_torch(q, kv, indices, scale, d_v, attn_sink, topk_length, return_stats):
    """fp32 reference oracle — every device, the parity authority."""
    import torch

    s_q, h_q, d_qk, s_kv = _validate(q, kv, indices, d_v, topk_length)
    safe = indices.clamp(0, max(kv.shape[0] - 1, 0)).long()  # never OOB
    gathered = kv[:, 0, :][safe[:, 0]].float()             # [s_q, topk, d]
    scores = torch.einsum("shd,swd->shw", q.float(), gathered) * scale
    valid = (indices >= 0) & (indices < max(s_kv, 1))
    if topk_length is not None:
        ar = torch.arange(indices.shape[-1], device=indices.device)
        valid = valid & (ar[None, None, :] < topk_length[:, None, None].to(torch.int32))
    scores = scores.masked_fill(~valid, float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.float().view(1, h_q, 1)
        m = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        scores = scores - m
        p = torch.exp(scores)
        denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
    else:
        p = torch.softmax(scores, dim=-1)
        denom = torch.ones_like(p[..., :1])
    out = torch.einsum("shw,swd->shd", p, gathered)
    has_row = valid.any(dim=-1, keepdim=True)
    out = torch.where(has_row, out / denom, torch.zeros_like(out))
    out = out.to(q.dtype)
    if return_stats:
        ml = scores.masked_fill(~valid, float("-inf")).amax(dim=-1)
        if attn_sink is not None:
            ml = torch.maximum(ml, attn_sink.float().view(1, h_q).expand(s_q, h_q))
        lse = ((scores.masked_fill(~valid, float("-inf")).logsumexp(dim=-1)
                + m.squeeze(-1)) / math.log(2.0)) if attn_sink is not None \
            else (torch.logsumexp(scores.where(valid, float("-inf")), dim=-1)
                  / math.log(2.0))
        return out, ml, lse
    empty = torch.empty(0, device=q.device, dtype=torch.float32)
    return out, empty, empty.clone()


def _run_sgl(q, kv, indices, scale, d_v, attn_sink, topk_length, return_stats):
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd as _donor  # ty: ignore[unresolved-import]  # optional dep

    _validate(q, kv, indices, d_v, topk_length)
    out, max_logits, lse = _donor(
        q, kv, indices, float(scale), d_v=d_v,
        attn_sink=attn_sink, topk_length=topk_length,
    )
    if not return_stats:
        import torch

        empty = torch.empty(0, device=q.device, dtype=torch.float32)
        return out, empty, empty.clone()
    return out, max_logits, lse


def flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale: float,
    d_v: int = 512,
    attn_sink=None,
    topk_length=None,
    backend: str | None = None,
    return_stats: bool = False,
):
    """SGLang v0.5.20 ``flash_mla_sparse_fwd`` boundary, multi-backend.

    Args mirror the donor (see the module docstring for the full
    contract). ``backend=None`` auto-resolves once per process
    (sgl_kernel > triton > torch); an explicit name pins the backend and
    raises when it cannot run. ``return_stats=True`` fills max_logits /
    base-2 lse (the Triton backend recomputes them in fp32; the donor
    returns its own). Returns ``(out, max_logits, lse)`` — the stats
    tensors are EMPTY (0-element) when not requested, so callers that
    discard them pay nothing.
    """
    backend = _resolve_backend(backend, getattr(q, "device", None))
    if backend == "sgl_kernel":
        return _run_sgl(q, kv, indices, sm_scale, d_v, attn_sink,
                        topk_length, return_stats)
    if backend == "triton":
        return _run_triton(q, kv, indices, sm_scale, d_v, attn_sink,
                           topk_length, return_stats)
    return _run_torch(q, kv, indices, sm_scale, d_v, attn_sink,
                      topk_length, return_stats)


# ---------------------------------------------------------------------------
# warmup — the capture contract's ordering half
# ---------------------------------------------------------------------------


def warmup(device="cuda", num_heads: int = 64, head_dim: int = 512,
           topk: int = 2051, prefill_rows=(16, 256)) -> dict:
    """Resolve every serve shape class and pre-compile BEFORE capture.

    Runs each backend arm once outside any capture: the sgl_kernel probe
    (when the wheel exists), the pinned-config resolution + bounded sweep
    per (op, shape class) — decode t1/t8/t64 at the serve's index width,
    prefill per head-dim class — and the Triton JIT compiles those
    resolutions imply. Idempotent; safe to call at every serve boot.
    Returns a report dict (backend, per-class statuses).
    """
    import torch

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return {"status": "skipped", "reason": "non-CUDA device",
                "backend": active_backend(dev)}
    backend = active_backend(dev)
    report: dict = {"status": "ok", "backend": backend, "classes": {}}
    if backend == "sgl_kernel":
        return report  # donor kernel: no op-tier configs, no Triton compiles
    if backend != "triton":
        return {"status": "skipped", "reason": "torch backend: nothing to pin",
                "backend": backend}

    rows_probe = 1
    q = torch.zeros(rows_probe, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
    kv = torch.zeros(topk + 1, 1, head_dim, device=dev, dtype=torch.bfloat16)
    idx = torch.zeros(rows_probe, 1, topk, device=dev, dtype=torch.int32)
    # decode classes: the bounded sweep for t1 runs here; t8/t64 resolve
    # through the same bucket when their rows tier was already tuned, or
    # sweep on their first live launch (still outside a capture — eager).
    for rows, tag in ((1, "t1"), (8, "t8"), (64, "t64")):
        qr = torch.zeros(rows, num_heads, head_dim, device=dev, dtype=torch.bfloat16)
        ir = torch.zeros(rows, 1, topk, device=dev, dtype=torch.int32)
        try:
            block_k = 32
            num_splits, num_stages, num_warps = _decode_config(
                rows, num_heads, topk, block_k, qr, kv, ir)
            from .vllm_sparse_mla import sparse_attn_decode

            sparse_attn_decode(qr, kv[:, 0, :], ir[:, 0, :], 0.04,
                               block_k=block_k, num_splits=num_splits,
                               num_stages=num_stages, num_warps=num_warps)
            report["classes"][f"decode_{tag}"] = {
                "num_splits": num_splits, "num_stages": num_stages,
                "num_warps": num_warps}
        except Exception as exc:  # a failed class warms nothing, blocks nothing
            report["classes"][f"decode_{tag}"] = {"status": "failed", "reason": repr(exc)}
    torch.cuda.synchronize(dev)

    qp = torch.zeros(max(prefill_rows), num_heads, head_dim, device=dev,
                     dtype=torch.bfloat16)
    ip = torch.zeros(max(prefill_rows), 1, topk, device=dev, dtype=torch.int32)
    try:
        block_k, num_warps = _prefill_config(head_dim, qp, kv[:, 0, :], ip[:, 0, :])
        from .vllm_sparse_mla import sparse_attn_prefill

        sparse_attn_prefill(qp, kv[:, 0, :], ip[:, 0, :], 0.04,
                            block_k=block_k, num_warps=num_warps)
        report["classes"]["prefill"] = {"block_k": block_k, "num_warps": num_warps}
    except Exception as exc:
        report["classes"]["prefill"] = {"status": "failed", "reason": repr(exc)}
    torch.cuda.synchronize(dev)
    return report
