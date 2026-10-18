"""Experimental split projection that moves one FP32 multiply before scratch.

The incumbent evaluates ``acc += (dot * activation_scale) * weight_scale``.
Only the parenthesized product moves into the producer. The ordered reduction
and its final multiply/add remain together, including compiler FMA contraction.
The caller must explicitly select this variant; the default stays unchanged.
"""
import triton
import triton.language as tl


@triton.jit
def scaled_group_products(A, W, SA, P, M: tl.constexpr, N: tl.constexpr,
                          K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, 128)
    group = tl.program_id(2)
    a = tl.load(A + rows[:, None] * K + group * 128 + reduction[None, :],
                rows[:, None] < M, other=0.0)
    w = tl.load(W + columns[None, :] * K + group * 128 + reduction[:, None],
                columns[None, :] < N, other=0.0)
    product = tl.dot(a, w)
    sa = tl.load(SA + rows * (K // 128) + group, rows < M, other=0.0)
    tl.store(P + (group * M + rows[:, None]) * N + columns[None, :],
             product * sa[:, None],
             (rows[:, None] < M) & (columns[None, :] < N))


@triton.jit
def accumulate_scaled_groups(P, SW, Y, M: tl.constexpr, N: tl.constexpr,
                             GROUPS: tl.constexpr, BLOCK: tl.constexpr,
                             STAGES: tl.constexpr = 1):
    COUNT: tl.constexpr = M * N
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    columns = offsets % N
    accumulator = tl.zeros((BLOCK,), tl.float32)
    for group in tl.range(GROUPS, num_stages=STAGES):
        product = tl.load(P + group * COUNT + offsets, offsets < COUNT, other=0.0)
        sw = tl.load(SW + (columns // 128) * GROUPS + group,
                     offsets < COUNT, other=0.0)
        accumulator += product * sw
    tl.store(Y + offsets, accumulator, offsets < COUNT)
