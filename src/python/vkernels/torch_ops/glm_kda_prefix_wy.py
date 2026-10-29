"""Prefix state using the same 32-token WY arithmetic as short HIP KDA.

The FP32 sequential recurrence is mathematically equivalent but does not
reproduce the serving WY pipeline's rounding on trained states. This is a
separate explicit operation for that pipeline; it enables no model policy.
"""
from ._dispatch import OpNotEligible, same_gpu_contiguous


def kda_prefix_state_wy(key, value, gate, beta, initial_state, prefix_lengths):
    """Restore FP32 [B,16,128,128] from a captured block of 2–4 tokens.

    Inputs are contiguous GPU FP32 core tensors: key/value/gate [B,16,S,128],
    beta [B,16,S], initial_state [B,16,128,128], plus GPU int32/int64 lengths
    [B]. Lengths clip to [0,S]. Inactive tokens contribute zero; prefix zero
    preserves initial-state bytes. Query/output computation is unused.

    This matches chunked WY state, including its rounding. It does not
    assert equivalence to the single-token decode algorithm or select a
    runner's backend. No inputs mutate; GPU lengths may change on replay.
    """
    import torch

    if key.ndim != 4:
        raise OpNotEligible("WY prefix key must be [B,16,S,128]")
    batch, heads, tokens, width = key.shape
    if (heads != 16 or width != 128 or not 2 <= tokens <= 4
            or value.shape != key.shape or gate.shape != key.shape
            or beta.shape != (batch, heads, tokens)
            or initial_state.shape != (batch, heads, width, width)):
        raise OpNotEligible("WY prefix requires H16/K128/V128 and S2–4")
    if prefix_lengths.shape != (batch,) or prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise OpNotEligible("prefix lengths must be GPU int32/int64 [B]")
    if any(t.dtype != torch.float32 for t in (key, value, gate, beta, initial_state)):
        raise OpNotEligible("WY prefix core inputs and state must be FP32")
    same_gpu_contiguous(key, value, gate, beta, initial_state, prefix_lengths)
    if batch == 0:
        return initial_state.clone()

    from torch.nn import functional as F
    from .vllm_kda import kda_chunk_floe

    lengths = prefix_lengths.clamp(0, tokens)
    valid = torch.arange(tokens, device=key.device)[None, None, :] < lengths[:, None, None]
    pad = 32 - tokens
    k = F.pad(torch.where(valid[..., None], key, 0), (0, 0, 0, pad))
    v = F.pad(torch.where(valid[..., None], value, 0), (0, 0, 0, pad))
    g = F.pad(torch.where(valid[..., None], gate, 0), (0, 0, 0, pad))
    b = F.pad(torch.where(valid, beta, 0), (0, pad))
    _, state = kda_chunk_floe(torch.zeros_like(k), k, v, g, b, chunk_size=32,
        initial_state=initial_state, output_final_state=True)
    return torch.where(lengths[:, None, None, None] == 0, initial_state, state)
