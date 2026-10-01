"""GLM-5.3 mHC chain-fuse: compose + unweighted-RMSNorm + mix GEMV, ONE launch.

The decode mHC chain between two hyper-connection sites is three
launch-bound kernels (the sgs-gpu07 moegrp2 census, per site, 90 sites per
decode step)::

    _compose   streams' = post·sub + combᵀ @ residual     (4.88 µs med)
    _norm_uw   y = flat' · rsqrt(mean(flat'²) + eps)      (2.52 µs med)
    GEMV       logits = F.linear(y, fn)                   (BLAS / mhc_pre_gemv)

``mhc_compose_pre`` collapses the three into ONE Triton launch: the previous
site's compose tail and the next site's pre head share the composed streams
in registers, so the ``[tokens, hc·d]`` norm input is never written to
memory only to be re-read by the GEMV (grid ``(tokens, mix)`` mirrors the
pinned ``mhc_pre_gemv`` launch; the ``row == 0`` program alone stores the
composed streams, which the site's ``mhc_pre_big_fuse`` epilogue still
reads). Per decode step this removes one launch per mHC site (90 nodes)
plus the ``_norm_uw`` read/write traffic — on top of the ``mhc_big_fuse``
head fusion (``mhc_pre_gemv``), taking a site from the eager 4-kernel chain
to 2 launches (``mhc_compose_pre`` + ``_big_fuse``).

Adopted from floe (``floe/engine/runner/kernels/mhc_chain.py``, the
sgs-gpu07 perf campaign) under the #64/#65 re-export model; the module
sits beside the ``mhc_compose`` / ``mhc_pre_gemv`` ops whose contracts it
fuses. Adoption deltas besides the move: ``OpNotEligible`` now comes from
this package's ``_dispatch`` directly (no vkernels-optional fallback
class), and the row-cap override env var is ``VK_MHC_COMPOSE_PRE_ROWS``
(the floe-local original read ``FLOE_MHC_COMPOSE_PRE_ROWS``; sweep
tooling that set the old name must set the new one).

Numerics contract (the parity-oracle pattern; the eager chain stays the
fallback and the oracle):

* compose — the exact ``mhc_compose._compose`` association: fp32
  ``static_range`` k-sum of ``comb[k, j] · residual[k, :]`` (ascending k,
  no FMA — ``enable_fp_fusion=False``), the ``post·sub`` product added
  AFTER the k-sum, one round to the stream dtype on the (single) store.
* unweighted-RMSNorm — the exact ``Glm53UnweightedRMSNorm`` /
  ``mhc_pre_gemv`` rounding points: the fp32 statistic is computed on the
  bf16-rounded compose store re-upcast (the eager chain reads the compose
  output from memory), ``rsqrt`` rounded to the stream dtype before the
  multiply, the normalized value rounded to the stream dtype again.
* mix GEMV — ``mhc_pre_gemv``'s contract: fp32 accumulation over the whole
  ``hc·d`` row, one round on the logits store.

Against the VERBATIM eager chain (cuBLAS matmul/GEMV blocking) the composed
streams and logits can differ by at most one storage-dtype ulp (fp32
reduction order — the same documented class ``mhc_compose`` and
``mhc_pre_gemv`` already carry); against the fused incumbent chain
(``mhc_compose`` → ``mhc_pre_gemv``) the element-wise structure is
reproduced verbatim and the reduction trees share the flat ``hc·d``
vector shape, pinned by the unit test.

Decode-shaped: rows ≤ the policy cap (2 by default — see :func:`_max_rows`;
the kernel's correctness envelope is 8), static per-bucket shapes, plain
current-stream launches, no host syncs, no device scalars — CUDA
graph-capture safe.

Microbench (dev-box GB10, graph-replayed, per site, hc=4 d=4096
fn [24,16384] bf16; the sgs H100 run is the integration gate):

=====  ========  ========  ========  ========
rows   eager3   moegrp    sglfuse   fused
=====  ========  ========  ========  ========
1      26.7 µs  10.7 µs   6.6 µs    **6.2 µs**
2      28.7 µs  12.4 µs   8.3 µs    **8.2 µs**
4      33.7 µs  12.4 µs   **8.3 µs** 11.7 µs
8      34.2 µs  12.4 µs   **12.5 µs** 18.9 µs
=====  ========  ========  ========  ========

(the arms: the verbatim eager chain / the moegrp arm's 3-launch chain /
the sgl-fuse2 arm's 2-launch incumbent this lane stacks on / this kernel.)
The fused launch wins at rows ≤ 2 and loses at rows ≥ 4 — one CTA per
(token, mix-row) holds the full [hc, d] compose tile plus the fn row, so
at higher row counts the machine fills with fat CTAs and the single
launch pays more than the second launch it saves. The policy cap keeps
the fusion to the buckets where it wins.

Stdlib+torch at import (triton lazily — CPU-only hosts import the module
for the eligibility surface and the reference). Inference-only, no
autograd backward.
"""

from __future__ import annotations

import functools

import math
import os

import torch

from ._dispatch import OpNotEligible

__all__ = ["mhc_compose_pre", "mhc_compose_pre_eligible",
           "mhc_compose_pre_reference"]

# The decode-graph ladder buckets are 1/2/4 with an 8-row envelope — but
# the FUSION only pays at low row counts: the one-CTA-per-(token, mix-row)
# programs hold the full [hc, d] compose tile plus the fn row (a far larger
# register set than mhc_pre_gemv's flat vector), so at T >= 4 the machine
# fills with fat CTAs and the single launch LOSES to the two-launch
# incumbent arm (measured, graph-replayed, dev-box GB10: fused 10.3-11.3 us
# vs mhc_compose+mhc_pre_gemv 8.2 us at T=4; fused 6.4 vs 6.7 us — a win —
# at T=1, parity at T=2). The eligibility cap therefore keeps the fusion
# to rows <= 2 (the B=1/B=2 decode buckets where it wins);
# VK_MHC_COMPOSE_PRE_ROWS overrides for the H100 microbench/sweep (the
# sgs census prices the incumbent chain at ~4.9 + ~2.5 + GEMV us/site at
# its batch-4 mix, so the crossover is worth re-measuring on the serving
# part).
def _max_rows() -> int:
    env = os.environ.get("VK_MHC_COMPOSE_PRE_ROWS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 2


@functools.lru_cache(maxsize=1)
def _kernel():
    """JIT-scoped kernel definition (import triton lazily, moe_combine style)."""
    import triton
    import triton.language as tl

    @triton.jit
    def _compose_pre(
        POST,     # [T, HC] fp32 (previous site's post gates)
        COMB,     # [T, HC, HC] fp32 (previous site's combiner)
        SUB,      # [T, D] stream dtype (the previous sublayer's output)
        RES,      # [T, HC, D] stream dtype (the site's input streams)
        FN,       # [MIX, HC*D] stream dtype (this site's mix weights)
        OUT_STREAMS,  # [T, HC, D] stream dtype (the composed streams)
        OUT_LOGITS,   # [T, MIX] stream dtype (the mix logits)
        MIX,
        HC: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,   # HC * D
        EPS: tl.constexpr,
    ):
        token = tl.program_id(0)
        row = tl.program_id(1)
        j = tl.arange(0, HC)
        offs = tl.arange(0, D)
        # -- compose (mhc_compose._compose order): fp32 k-sum, then + post·sub
        acc = tl.zeros([HC, D], dtype=tl.float32)
        for k in tl.static_range(HC):
            comb_k = tl.load(COMB + token * HC * HC + k * HC + j)
            value = tl.load(RES + token * HC * D + k * D + offs)
            acc += comb_k[:, None] * value[None, :].to(tl.float32)
        sub = tl.load(SUB + token * D + offs).to(tl.float32)
        post = tl.load(POST + token * HC + j)
        acc += post[:, None] * sub[None, :]
        # the eager chain rounds the compose once on store, and the norm/GEMV
        # read that rounded value back — round here, keep it in registers.
        streams = acc.to(OUT_STREAMS.dtype.element_ty)
        if row == 0:
            tl.store(
                OUT_STREAMS + token * HC * D + j[:, None] * D + offs[None, :],
                streams,
            )
        # -- pre head (mhc_pre_gemv order) on the in-register compose store
        streams_f = streams.to(tl.float32)
        ss = tl.sum(streams_f * streams_f)
        rstd = tl.rsqrt(ss / K + EPS).to(OUT_LOGITS.dtype.element_ty).to(tl.float32)
        # the fn row shares the streams' [HC, D] layout — no reshape, the
        # reduction trees stay identical to mhc_pre_gemv's flat vector
        # (bit-equality with the incumbent arm is pinned by the unit test)
        w = tl.load(FN + row * K + j[:, None] * D + offs[None, :]).to(tl.float32)
        xn = (streams_f * rstd).to(OUT_LOGITS.dtype.element_ty).to(tl.float32)
        logits = tl.sum(xn * w)
        tl.store(OUT_LOGITS + token * MIX + row, logits.to(OUT_LOGITS.dtype.element_ty))

    return _compose_pre


def mhc_compose_pre_eligible(post, comb, sublayer_out, streams, fn,
                             hc=4, hidden_size=4096) -> None:
    """Raise :class:`OpNotEligible` unless the one-launch contract holds.

    Mirrors the ``mhc_pre_gemv`` + ``mhc_compose`` envelopes (GPU-resident
    contiguous bf16/fp16 streams, fp32 controls, decode-sized rows) so a
    miss falls back to the verbatim two-kernel chain per site.
    """
    if not isinstance(hc, int) or hc < 2 or hc & (hc - 1):
        raise OpNotEligible("hc must be a power of two >= 2")
    k = hc * hidden_size
    width = hc * (hc + 2)
    if streams.ndim < 2 or streams.shape[-2:] != (hc, hidden_size):
        raise OpNotEligible("streams must be [..., hc, hidden] rows")
    lead = streams.shape[:-2]
    if sublayer_out.shape != (*lead, hidden_size):
        raise OpNotEligible("sublayer_out must be [..., hidden] matching streams' leading dims")
    if post.shape != (*lead, hc) or comb.shape != (*lead, hc, hc):
        raise OpNotEligible("expected post [..., hc] and comb [..., hc, hc] matching streams")
    if fn.ndim != 2 or fn.shape != (width, k):
        raise OpNotEligible("expected fn [hc*(hc+2), hc*hidden]")
    if streams.numel() // k > _max_rows():
        raise OpNotEligible(f"decode-sized rows only (<= {_max_rows()})")
    if streams.dtype not in (torch.bfloat16, torch.float16):
        raise OpNotEligible("streams must be bf16/fp16")
    if sublayer_out.dtype != streams.dtype or fn.dtype != streams.dtype:
        raise OpNotEligible("sublayer_out/fn must share the streams' dtype")
    if post.dtype != torch.float32 or comb.dtype != torch.float32:
        raise OpNotEligible("mHC control tensors must be FP32")
    if hidden_size != 1 << (hidden_size.bit_length() - 1) or not 512 <= hidden_size <= 8192:
        raise OpNotEligible("hidden must be a power of two in [512, 8192]")
    for x in (post, comb, sublayer_out, streams, fn):
        if not x.is_cuda or not x.is_contiguous():
            raise OpNotEligible("inputs must be contiguous tensors on a GPU")
    if not (post.device == comb.device == sublayer_out.device == streams.device == fn.device):
        raise OpNotEligible("inputs must live on one GPU")


def mhc_compose_pre(post, comb, sublayer_out, streams, fn,
                    hc=4, hidden_size=4096, eps=1e-6):
    """One-launch compose + unweighted-RMSNorm + mix GEMV.

    Returns ``(streams_out, logits)``: the composed streams ``[..., hc,
    hidden]`` in the stream dtype (the value the site's ``mhc_pre_big_fuse``
    epilogue collapses — identical to ``mhc_compose``'s output) and the mix
    logits ``[..., hc*(hc+2)]`` in the stream dtype (identical rounding
    class to ``mhc_pre_gemv``'s output). ``eps`` is the site's
    ``Glm53UnweightedRMSNorm`` epsilon.
    """
    mhc_compose_pre_eligible(post, comb, sublayer_out, streams, fn,
                             hc=hc, hidden_size=hidden_size)
    if not math.isfinite(eps):
        raise OpNotEligible("eps must be finite")

    lead = streams.shape[:-2]
    tokens = streams.numel() // (hc * hidden_size)
    width = hc * (hc + 2)
    out_streams = torch.empty_like(streams)
    out_logits = torch.empty(*lead, width, device=streams.device, dtype=streams.dtype)
    if tokens:
        kernel = _kernel()
        # num_warps mirrors the pinned mhc_pre_gemv launch at K = hc*d
        # (measured flat-in-rows optimum for this working set: warps=8 keeps
        # the compose tile + fn row under the register file at 1 CTA/SM —
        # the T-scaling cap above is what guards the fat-CTA regime).
        k = hc * hidden_size
        warps = 2 if k <= 512 else 4 if k <= 2048 else 8
        with torch.cuda.device(streams.device):
            kernel[(tokens, width)](
                post, comb, sublayer_out, streams, fn,
                out_streams, out_logits,
                width,
                hc,
                hidden_size,
                k,
                eps,
                num_warps=warps,
                enable_fp_fusion=False,
            )
    return out_streams, out_logits


def mhc_compose_pre_reference(post, comb, sublayer_out, streams, fn,
                              hc=4, hidden_size=4096, eps=1e-6):
    """Eager oracle: the VERBATIM base chain the kernel replaces.

    Composed from the production eager expressions (``_mhc_compose``'s
    torch form, ``Glm53UnweightedRMSNorm``'s eager math, ``F.linear``), so
    parity here is parity with the shipped eager chain, not a hand-rolled
    restatement.
    """
    k = hc * hidden_size
    residual = streams.reshape(-1, hc, hidden_size)
    sub = sublayer_out.reshape(-1, hidden_size)
    p = post.reshape(-1, hc).float()
    c = comb.reshape(-1, hc, hc).float()
    composed = p.unsqueeze(-1) * sub.unsqueeze(-2).float() + torch.matmul(
        c.transpose(-1, -2), residual.float())
    composed = composed.to(streams.dtype)
    flat = composed.reshape(-1, k).float()
    normalized = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + eps).to(streams.dtype)
    logits = torch.nn.functional.linear(normalized.to(streams.dtype), fn)
    return composed.view_as(streams), logits.view(*streams.shape[:-2], -1)
