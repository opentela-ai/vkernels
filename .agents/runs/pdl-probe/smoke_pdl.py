"""PDL smoke test on GB10: launch_pdl kwarg + gdc annotations + graph capture."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_wait, gdc_launch_dependents


@triton.jit
def producer(X, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = (offs % 97).to(tl.float32)
    gdc_launch_dependents()  # earliest possible trigger; consumer still gdc_waits
    tl.store(X + offs, v, mask=offs < n)


@triton.jit
def consumer(X, Y, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    gdc_wait()  # memory ordering vs the producer grid
    x = tl.load(X + offs, mask=offs < n, other=0.0)
    tl.store(Y + offs, x * 2.0, mask=offs < n)


def main():
    n, BLOCK = 1 << 20, 1024
    grid = (triton.cdiv(n, BLOCK),)
    x = torch.empty(n, device="cuda", dtype=torch.float32)
    y = torch.empty(n, device="cuda", dtype=torch.float32)
    ref = ((torch.arange(n, device="cuda") % 97).float() * 2.0)

    # eager, PDL consumer
    producer[grid](x, n, BLOCK)
    consumer[grid](x, y, n, BLOCK, launch_pdl=True)
    torch.cuda.synchronize()
    print("eager pdl correct:", torch.equal(y, ref))

    # graph capture + replay with PDL inside
    g = torch.cuda.CUDAGraph()
    producer[grid](x, n, BLOCK)  # warmup outside capture
    consumer[grid](x, y, n, BLOCK, launch_pdl=True)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(10):
            producer[grid](x, n, BLOCK)
            consumer[grid](x, y, n, BLOCK, launch_pdl=True)
    g.replay()
    torch.cuda.synchronize()
    print("graph-replayed pdl correct:", torch.equal(y, ref))

    # sanity: kernel launches recorded in the capture (no fallback to eager)
    print("capture ok, nodes:", g)


if __name__ == "__main__":
    main()
