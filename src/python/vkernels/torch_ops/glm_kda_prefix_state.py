"""State-only KDA replay of a verified prefix, without recomputing projections.

Keys are already L2-normalized. All arithmetic and state storage are FP32,
matching the chunk adapter's core contract. Different batch rows may accept
different prefixes. This op does not compute attention output or mutate input
state. The companion convolution-history op retains the raw projection
columns for the same accepted prefix, without rerunning convolution.
"""
from functools import lru_cache

from ._dispatch import OpNotEligible, same_gpu_contiguous


@lru_cache(maxsize=1)
def _kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def replay(KEY, VALUE, GATE, BETA, INITIAL, LENGTHS, FINAL,
               HEADS: tl.constexpr, TOKENS: tl.constexpr,
               KD: tl.constexpr, VD: tl.constexpr, BV: tl.constexpr):
        head = tl.program_id(0)
        block = tl.program_id(1)
        row = head // HEADS
        ki = tl.arange(0, KD)
        vi = block * BV + tl.arange(0, BV)
        offsets = head * KD * VD + ki[:, None] * VD + vi[None, :]
        state = tl.load(INITIAL + offsets, vi[None, :] < VD, 0)
        count = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), TOKENS)
        for token in tl.range(0, count):
            key = tl.load(KEY + head * TOKENS * KD + token * KD + ki)
            gate = tl.load(GATE + head * TOKENS * KD + token * KD + ki)
            state = state * tl.exp(gate)[:, None]
            memory = tl.sum(state * key[:, None], 0)
            value = tl.load(VALUE + head * TOKENS * VD + token * VD + vi, vi < VD, 0)
            beta = tl.load(BETA + head * TOKENS + token)
            delta = (value - memory) * beta
            state = state + key[:, None] * delta[None, :]
        tl.store(FINAL + offsets, state, vi[None, :] < VD)

    return replay


def kda_prefix_state(key, value, gate, beta, initial_state, prefix_lengths):
    """Return FP32 state after each row's prefix; lengths are clipped to [0,S].

    Contiguous GPU inputs: key/gate [B,H,S,K], value [B,H,S,V],
    beta [B,H,S], initial_state [B,H,K,V], all FP32; prefix_lengths
    [B], int32 or int64. K and V are powers of two in [32,256].
    A zero prefix preserves the input state's bytes. No host scalar reads;
    length values may change between graph replays. Inference only.
    """
    import torch

    if key.ndim != 4 or value.ndim != 4 or gate.shape != key.shape:
        raise OpNotEligible("expected key/gate [B,H,S,K] and value [B,H,S,V]")
    batch, heads, tokens, kd = key.shape
    vd = value.shape[-1]
    if kd not in (32, 64, 128, 256) or vd not in (32, 64, 128, 256):
        raise OpNotEligible("K/V dimensions must be powers of two in [32,256]")
    if value.shape[:3] != key.shape[:3] or beta.shape != (batch, heads, tokens):
        raise OpNotEligible("key/value/gate/beta must share [B,H,S]")
    if initial_state.shape != (batch, heads, kd, vd):
        raise OpNotEligible("initial_state must be [B,H,K,V]")
    if prefix_lengths.shape != (batch,) or prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise OpNotEligible("prefix_lengths must be int32/int64 [B]")
    if any(t.dtype != torch.float32 for t in (key, value, gate, beta, initial_state)):
        raise OpNotEligible("KDA core inputs and state must be FP32")
    same_gpu_contiguous(key, value, gate, beta, initial_state, prefix_lengths)
    result = torch.empty_like(initial_state)
    if batch * heads:
        import triton

        with torch.cuda.device(key.device):
            _kernel()[(batch * heads, triton.cdiv(vd, 32))](
                key, value, gate, beta, initial_state, prefix_lengths, result,
                heads, tokens, kd, vd, 32, num_warps=4, enable_fp_fusion=False)
    return result


@lru_cache(maxsize=1)
def _conv_kernel():
    global tl
    import triton
    import triton.language as tl

    @triton.jit
    def restore(RAW, INITIAL, LENGTHS, FINAL,
                CHANNELS: tl.constexpr, TOKENS: tl.constexpr,
                HISTORY: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        channel = offsets // HISTORY
        history = offsets % HISTORY
        count = tl.minimum(tl.maximum(tl.load(LENGTHS + row), 0), TOKENS)
        column = count + history
        valid = channel < CHANNELS
        prior = tl.load(INITIAL + row * CHANNELS * HISTORY + channel * HISTORY + column,
                        valid & (column < HISTORY), 0)
        fresh = tl.load(RAW + row * CHANNELS * TOKENS + channel * TOKENS + column - HISTORY,
                        valid & (column >= HISTORY), 0)
        value = tl.where(column < HISTORY, prior, fresh)
        tl.store(FINAL + row * CHANNELS * HISTORY + offsets, value, valid)

    return restore


def kda_prefix_conv_state(raw, initial_state, prefix_lengths):
    """Return raw convolution history after each row's accepted prefix.

    Contiguous GPU raw projections [B,C,S], initial history [B,C,W],
    matching FP32/FP16/BF16 dtype; GPU int32/int64 prefix lengths [B].
    Lengths are clipped to [0,S]. Output is the last W columns of
    concat(initial_state, raw[..., :length]) for each row, including
    history retained when a prefix is shorter than W. No inputs mutate,
    convolution is not recomputed, and lengths may change on graph replay.
    Time-major pool owners must adapt to this channel-major boundary.
    """
    import torch

    if raw.ndim != 3 or initial_state.ndim != 3 or raw.shape[:2] != initial_state.shape[:2]:
        raise OpNotEligible("expected raw [B,C,S] and initial_state [B,C,W]")
    batch, channels, tokens = raw.shape
    history = initial_state.shape[-1]
    if raw.dtype not in (torch.float16, torch.bfloat16, torch.float32) or initial_state.dtype != raw.dtype:
        raise OpNotEligible("raw and initial_state must share FP32/FP16/BF16 dtype")
    if prefix_lengths.shape != (batch,) or prefix_lengths.dtype not in (torch.int32, torch.int64):
        raise OpNotEligible("prefix_lengths must be int32/int64 [B]")
    same_gpu_contiguous(raw, initial_state, prefix_lengths)
    result = torch.empty_like(initial_state)
    if batch * channels * history:
        import triton

        with torch.cuda.device(raw.device):
            _conv_kernel()[(batch, triton.cdiv(channels * history, 256))](
                raw, initial_state, prefix_lengths, result,
                channels, tokens, history, 256, num_warps=4)
    return result
