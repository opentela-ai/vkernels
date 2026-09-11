"""Single-token FP32 per-dimension-gated KDA inference kernel (optional).

Normalizes raw q/k (L2 with eps, then q scaled by ``D**-0.5``) and advances
the per-dimension-gated delta-rule state by one decode token, returning
``(output, next_state)`` without mutating any input.

Adopted from floe's ``engine/runner/kernels/glm5_kda_decode.py`` (vkernels
owns the kernel; floe imports it back through a thin adapter — the #64/#65
thin-adapter model extended to the whole GLM-5 Triton set).

Torch and Triton load lazily. Inference-only, no autograd backward.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def _decode(
        Q,
        K,
        V,
        G,
        BETA,
        STATE,
        OUT,
        NEXT,
        D: tl.constexpr,
        EPS: tl.constexpr,
        BV: tl.constexpr,
    ):
        head = tl.program_id(0)
        block = tl.program_id(1)
        rows = tl.arange(0, D)
        cols = block * BV + tl.arange(0, BV)
        q = tl.load(Q + head * D + rows)
        k = tl.load(K + head * D + rows)
        # Reference divides by sqrt(sum(x*x) + eps), then scales q.
        q = tl.div_rn(q, tl.sqrt(tl.sum(q * q, 0) + EPS))
        k = tl.div_rn(k, tl.sqrt(tl.sum(k * k, 0) + EPS))
        q = q * (D**-0.5)
        decay = tl.exp(tl.load(G + head * D + rows))
        state_offset = head * D * D + rows[:, None] * D + cols[None, :]
        state = tl.load(STATE + state_offset, cols[None, :] < D, 0)
        state = state * decay[:, None]
        memory = tl.sum(state * k[:, None], 0)
        value = tl.load(V + head * D + cols, cols < D, 0)
        delta = (value - memory) * tl.load(BETA + head)
        state = state + k[:, None] * delta[None, :]
        output = tl.sum(state * q[:, None], 0)
        tl.store(NEXT + state_offset, state, cols[None, :] < D)
        tl.store(OUT + head * D + cols, output, cols < D)

    return _decode


def kda_decode(query, key, value, gate, beta, initial_state):
    """Normalize raw q/k and return ``(output, next_state)`` without mutation.

    Vectors must be contiguous FP32 GPU tensors [B,1,H,D], beta [B,1,H],
    state [B,H,D,D]. Output has the vector shape. Inference only; no autograd.
    The normalization epsilon is pinned to 1e-6 (kda_decode_reference's
    default); experiment via the reference's ``eps`` argument.
    """
    import torch

    if query.ndim != 4 or query.shape[1] != 1:
        raise ValueError("expected vectors [B,1,H,D]")
    batch, _, heads, dim = query.shape
    if dim not in (32, 64, 128):
        raise ValueError("supported head dimensions are 32, 64, 128")
    if any(x.shape != query.shape for x in (key, value, gate)):
        raise ValueError("query/key/value/gate shapes must match")
    if beta.shape != (batch, 1, heads) or initial_state.shape != (
        batch,
        heads,
        dim,
        dim,
    ):
        raise ValueError("incorrect beta or state shape")
    for x in (query, key, value, gate, beta, initial_state):
        if x.dtype != torch.float32:
            raise TypeError("all inputs must be FP32")
        if not x.is_cuda or x.device != query.device or not x.is_contiguous():
            raise ValueError("inputs must be contiguous on the same GPU")
    out = torch.empty_like(query)
    state = torch.empty_like(initial_state)
    if batch * heads:
        import triton  # lazy: validation above needs only torch

        with torch.cuda.device(query.device):
            _kernel()[(batch * heads, triton.cdiv(dim, 32))](
                query,
                key,
                value,
                gate,
                beta,
                initial_state,
                out,
                state,
                dim,
                1e-6,
                32,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out, state


def kda_decode_reference(query, key, value, gate, beta, initial_state, eps=1e-6):
    """Eager FP32 oracle mirroring the kernel's exact math (no mutation)."""
    import torch

    q = query[:, 0]
    k = key[:, 0]
    v = value[:, 0]
    g = gate[:, 0]
    b = beta[:, 0]
    # Reference divides by sqrt(sum(x*x) + eps), then scales q.
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + eps) * query.shape[-1] ** -0.5
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + eps)
    state = initial_state * g.exp()[..., None]
    memory = (state * k[..., :, None]).sum(-2)
    state = state + k[..., :, None] * ((v - memory) * b[..., None])[..., None, :]
    out = (state * q[..., :, None]).sum(-2)
    return out.unsqueeze(1), state
