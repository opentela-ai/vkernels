"""Fused per-dimension-gated KDA chunked prefill (optional, NVIDIA Triton).

Drop-in for the eager torch chunked delta-rule reference (floe
``glm53flash/forward.py::_kda_chunk``, a port of HF's
``chunk_kimi_delta_attention``): chunked WY forward with per-(head, dim)
log gates ``g [B, H, S, D]``, returning ``(core_attn_out [B, S, H, V]`` in
the input dtype, ``final_state [B, H, K, V]`` fp32 ``| None)``.

The heavy lifting is the vendored fla pipeline in
:mod:`vkernels.torch_ops.vllm_kda` (chunk-local gate cumsum → decayed
K·Kᵀ/q·Kᵀ tiles → blockwise ``(I+A)^{-1}`` → W/U recompute → chunkwise
state recurrence → fused output). Versus the eager reference it never
materializes the ``[B, H, nc, cs, cs, D]`` fp32 decay mask (≈2.9 GB per
layer at the serving shape), folds the 63-iteration intra-chunk
tri-solve into one in-kernel blockwise inversion, and runs the whole
chunk scan as ~7 launches per layer instead of ~13k.

Numerics (the fp32 contract)
----------------------------
The KDA core is fp32-by-contract: the op accepts **fp32 inputs only**
(the caller widens, exactly as it did for the reference) and the vendored
kernels run every fp32-operand ``tl.dot`` with ``input_precision="ieee"``
(see ``vllm_kda.NV_DOT_PRECISION`` — the tf32 default drifts ~1e-3 against
the fp32 reference, two orders over the 1e-4 parity bar). Remaining
deviation vs the eager reference is fp32 summation-order noise (blockwise
``(I+A)^{-1}`` vs row-by-row forward substitution, exp vs exp2·log2e gate
evaluation, blocked GEMM orders, mma accumulation order).
STATUS (round-7 lane I): the 7.3e-4 GH200 "final-state drift" of round-7
lane C was NOT sm90 codegen — it was the NGC container defaulting torch
matmuls to tf32 (``fp32_precision='tf32'``, ``allow_tf32=True``), which
silently degraded the *eager oracle's* own matmuls; the fused ieee kernel
was the more accurate side. :func:`kda_chunk_reference` now pins true-fp32
(ieee) matmuls for the oracle, and the rig re-run closes state parity at
the 1e-5 level (see ``tests/python/test_glm_kda_chunk.py::
test_gpu_final_state_parity``). NOTE for the campaign: any other fp32
reference computed inside this container (including floe's own eager
``_kda_chunk`` serving path) is tf32-degraded the same way unless pinned.
The gate cumsum fed to the pipeline is bit-identical to the reference's
(torch.cumsum on the same chunked view), so gates are ruled out as a
drift source.

Ragged S is handled by zero-padding to the next chunk multiple BEFORE the
pipeline (the reference pads identically, and padding to the chunk size
lets every fla boundary check pass fully in-bounds — no masked-row
garbage can leak through the state recurrence).

Torch and Triton load lazily. Inference-only, no autograd backward.
AMD (``torch.version.hip``) stays on the HIP WY path (floe knob
``kda_chunk_hip``): this module raises :class:`OpNotEligible` there so the
caller's fallback keeps today's behavior until the AMD twin
(``vllm_kda_amd``) gets the same adapter.
"""

from ._dispatch import OpNotEligible

_HEAD_DIMS = (64, 128, 256)  # fwd_h splits K into 64-wide register blocks
_CHUNK_SIZES = (16, 32, 64)  # solve_tril / kkt block structure supports these


def _pin_fp32_matmul():
    """Force torch matmuls to true fp32 (ieee); returns a restore callable.

    The fp32 contract of the KDA core is meaningless if the *reference*
    computes in tf32: NGC/NVIDIA containers set the torch matmul default to
    tf32 (``fp32_precision='tf32'``, ``allow_tf32=True``) — stock torch
    defaults to ieee — which silently degrades every fp32-by-contract torch
    reference run inside them (round-7 lane C's 7.3e-4 "kernel drift" was
    exactly this, on the oracle side). The fused Triton pipeline pins its
    own dots to ieee (``vllm_kda.NV_DOT_PRECISION``); this pins the oracle.
    """
    import torch

    matmul = torch.backends.cuda.matmul
    prev = (matmul.allow_tf32, getattr(matmul, "fp32_precision", None))
    matmul.allow_tf32 = False
    if prev[1] is not None:
        matmul.fp32_precision = "ieee"

    def _restore():
        matmul.allow_tf32 = prev[0]
        if prev[1] is not None:
            matmul.fp32_precision = prev[1]

    return _restore


def kda_chunk(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
):
    """Chunked per-dim-gate delta rule on the fused NVIDIA Triton pipeline.

    Contract (mirrors the eager reference): fp32 ``query``/``key``
    L2-normalized and UNSCALED by the caller (the kernel applies the
    ``D**-0.5`` scale), ``value`` fp32, ``g`` per-(head, dim) natural-log
    log-gates ``[B, H, S, K]``, ``beta`` ``[B, H, S]``, ``initial_state``
    ``[B, H, K, V]`` fp32 or ``None``. Returns ``(out [B, S, H, V]`` in the
    input dtype, ``final_state [B, H, K, V]`` fp32 or ``None)``.

    Raises :class:`OpNotEligible` on any contract miss (CPU tensors, AMD
    build, dtype, shape, layout) so the caller falls back to its eager
    path.
    """
    import torch

    tensors = (query, key, value, g)
    if any(t.ndim != 4 for t in tensors) or beta.ndim != 3:
        raise OpNotEligible("expected q/k/v/g [B, H, S, D] and beta [B, H, S]")
    bsz, heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    if query.shape != key.shape or g.shape != key.shape:
        raise OpNotEligible("query/key/gate shapes must match")
    if value.shape[:3] != key.shape[:3]:
        raise OpNotEligible("value must share [B, H, S] with key")
    if beta.shape != (bsz, heads, seq_len):
        raise OpNotEligible("beta must be [B, H, S]")
    if k_dim not in _HEAD_DIMS or v_dim not in _HEAD_DIMS:
        raise OpNotEligible(
            f"supported head dimensions are {_HEAD_DIMS}; got K={k_dim}, V={v_dim}"
        )
    if chunk_size not in _CHUNK_SIZES:
        raise OpNotEligible(
            f"supported chunk sizes are {_CHUNK_SIZES}; got {chunk_size}"
        )
    if any(t.dtype != torch.float32 for t in (*tensors, beta)):
        raise OpNotEligible("KDA core numerics are fp32-by-contract: fp32 inputs only")
    if initial_state is not None and (
        initial_state.shape != (bsz, heads, k_dim, v_dim)
        or initial_state.dtype != torch.float32
    ):
        raise OpNotEligible("initial_state must be [B, H, K, V] fp32")
    for t in (*tensors, beta, initial_state):
        if t is None:
            continue
        if not t.is_cuda or not t.is_contiguous():
            raise OpNotEligible(
                "inputs must be contiguous CUDA tensors (caller owns "
                ".contiguous())"
            )
    if len({t.device for t in tensors} | {beta.device}) != 1:
        raise OpNotEligible("inputs must live on one GPU")
    if torch.version.hip is not None:
        raise OpNotEligible(
            "the fused Triton chunk pipeline is wired for NVIDIA; AMD uses the "
            "kda_chunk_hip WY path"
        )

    # Zero-pad S to the next chunk multiple (exactly the reference's
    # padding): every fla boundary check then passes fully in-bounds, so no
    # masked-row garbage can reach the state recurrence.
    pad = (-seq_len) % chunk_size
    if pad:
        import torch.nn.functional as F

        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        g = F.pad(g, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))

    # lazy: importing torch_ops.vllm_kda pulls torch + triton at module scope
    from vkernels.torch_ops.vllm_kda import kda_chunk_floe

    out, final_state = kda_chunk_floe(
        query,
        key,
        value,
        g,
        beta,
        chunk_size=chunk_size,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )
    if pad:
        # the padded tail rows must not leak: the reference slices its output
        # to seq_len after padding, and so does the contract (o is [B, S, H, V]).
        out = out[:, :seq_len]
    return out, final_state


def kda_chunk_reference(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
):
    """Eager FP32 oracle mirroring floe ``forward.py::_kda_chunk`` (no mutation).

    Same contract as :func:`kda_chunk`; this is the parity bar the kernel
    tests compare against (kept here so the vkernels suite is
    self-contained — floe's copy is the upstream original).

    Matmuls run under a true-fp32 (ieee) pin for the duration of the call
    (see :func:`_pin_fp32_matmul`): an fp32 oracle computed with the
    container's tf32 matmul default is not an fp32 oracle.
    """
    restore = _pin_fp32_matmul()
    try:
        return _kda_chunk_reference_impl(
            query, key, value, g, beta, chunk_size, initial_state, output_final_state
        )
    finally:
        restore()


def _kda_chunk_reference_impl(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
):
    """Eager FP32 oracle body (called under the ieee matmul pin)."""
    import math

    import torch
    import torch.nn.functional as F

    initial_dtype = query.dtype
    bsz, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(query.shape[-1])
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    total = seq_len + pad

    query = F.pad(query, (0, 0, 0, pad)) * scale
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    shape = lambda t: t.reshape(bsz, num_heads, -1, chunk_size, t.shape[-1])  # noqa: E731
    query, key, value, g, k_beta, v_beta = (
        shape(query),
        shape(key),
        shape(value),
        shape(g),
        shape(k_beta),
        shape(v_beta),
    )
    beta = beta.reshape(bsz, num_heads, -1, chunk_size)

    # intra-chunk: per-dim decayed triangular solve (KDA vs GDN difference)
    g = g.cumsum(dim=-2)  # [B, H, nc, cs, D] per-chunk prefix log-gates
    tri0 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 0
    )
    # .float() was a no-op on the fp32 CPU reference and would re-widen the
    # low-precision device path — keep the decay in the working dtype.
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).exp()  # [B,H,nc,cs,cs,D]
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1).masked_fill(tri0, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    state = (
        torch.zeros(bsz, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value.dtype)
    )
    core_attn_out = torch.zeros_like(value)
    tri1 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), 1
    )
    for c in range(total // chunk_size):
        q_i, k_i, v_i, g_i = query[:, :, c], key[:, :, c], value[:, :, c], g[:, :, c]
        attn_inter = (q_i * g_i.exp()) @ state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, c]).sum(dim=-1).masked_fill(tri1, 0)
        v_prime = k_cumdecay[:, :, c] @ state
        v_new = v_i - v_prime
        core_attn_out[:, :, c] = attn_inter + attn_intra @ v_new
        state = state * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new

    core_attn_out = core_attn_out.reshape(bsz, num_heads, total, v_dim)[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, (state if output_final_state else None)


def kda_recurrent_reference(query, key, value, g, beta, initial_state=None,
                            output_final_state=False):
    """Per-token recurrent oracle (floe ``_kda_recurrent`` contract).

    Inputs ``[B, S, H, …]`` fp32 with per-dim ``g [B, S, H, D]``; q/k
    pre-normalized AND pre-scaled (the chunked oracle applies the scale
    internally, so pass already-scaled q here — see the test). Used by the
    tests as an independent check of :func:`kda_chunk_reference` (chunked
    WY math must equal the token-recurrent delta rule).
    """
    import torch

    initial_dtype = query.dtype
    bsz, seq_len, num_heads, k_dim = key.shape
    v_dim = value.shape[-1]

    core_attn_out = torch.zeros(
        bsz, seq_len, num_heads, v_dim, dtype=value.dtype, device=value.device
    )
    state = (
        torch.zeros(bsz, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value.dtype)
    )
    for i in range(seq_len):
        q_i, k_i, v_i = query[:, i], key[:, i], value[:, i]
        g_i = g[:, i][..., None].exp()  # [B, H, 1, D] — element-wise row decay
        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(dim=-2)
        delta = (v_i - kv_mem) * beta[:, i][..., None]
        state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, i] = (state * q_i.unsqueeze(-1)).sum(dim=-2)
    return core_attn_out.to(initial_dtype), (state if output_final_state else None)
