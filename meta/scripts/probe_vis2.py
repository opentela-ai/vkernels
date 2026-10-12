"""Race-free visibility tests for the grid barrier (MI300A bring-up probe).

Exactly one writer, one address, per-round slots.  No block can overwrite an
address another block is about to read, so *any* deviation from the exact
expected value is a genuine release/acquire visibility failure, not the
single-slot microprogram's documented next-round race.

A: plain load after the barrier -- what the megakernel itself does
B: same, but the reader uses a coherent .cv load
C: race-free M0 -- every round gets its own slot
"""
import sys

import triton
import triton.language as tl
import torch

from vkernels.compiler.device_triton import grid_barrier

# The barrier is a co-residency spin, so P must not exceed the number of
# resident blocks.  GB10: 48 SMs.  MI300A: 228 CUs.  Pass the cap in.
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 48


@triton.jit
def vis_kernel(out_ptr, a_ptr, bar_ptr, P, ROUNDS, base: tl.int64, CV: tl.constexpr):
    pid = tl.program_id(0)
    r = 0
    while r < ROUNDS:
        if pid == 0:
            tl.store(a_ptr, r.to(tl.float32))
        grid_barrier(bar_ptr, base + (r + 1) * P)
        if CV:
            v = tl.load(a_ptr, cache_modifier=".cv")
        else:
            v = tl.load(a_ptr)
        tl.store(out_ptr + r * P + pid, v)
        r += 1


@triton.jit
def m0_racefree(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64):
    """Per-round slots: round r writes out[r*P + pid] and reads out[r*P + nb]."""
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        tl.store(out_ptr + r * P + pid, pid * 1000.0 + r)
        grid_barrier(bar_ptr, base + (r + 1) * P)
        nb = (pid + 1) % P
        acc += tl.load(out_ptr + r * P + nb)
        r += 1
    tl.store(out_ptr + ROUNDS * P + pid, acc)


def run(P, R, nw=8, cv=False):
    out = torch.full((R * P + 1,), -1.0, device="cuda")
    a = torch.full((1,), -1.0, device="cuda")
    bar = torch.zeros(1, dtype=torch.int64, device="cuda")
    vis_kernel[(P,)](out, a, bar, P, R, 0, CV=cv, num_warps=nw)
    torch.cuda.synchronize()
    got = out[: R * P].reshape(R, P).cpu()
    lag = []
    for r in range(R):
        for pid in range(P):
            if float(got[r, pid]) != float(r):
                lag.append((r, pid, float(got[r, pid])))
    return lag, int(bar[0]), R * P


def run_m0(P, R, nw=8):
    out = torch.full((R * P + P + 1,), -1.0, device="cuda")
    bar = torch.zeros(1, dtype=torch.int64, device="cuda")
    m0_racefree[(P,)](out, bar, P, R, 0, num_warps=nw)
    torch.cuda.synchronize()
    got = out[R * P : R * P + P].cpu()
    bad = []
    for pid in range(P):
        nb = (pid + 1) % P
        expect = float(sum(nb * 1000 + r for r in range(R)))
        if float(got[pid]) != expect:
            bad.append((pid, float(got[pid]), expect))
    return bad, int(bar[0]), R * P


def _pairs(pairs):
    return [(p, r) for p, r in pairs if p <= CAP]


print("=== A: plain load, single writer, same address (staleness) ===", flush=True)
for P, R in _pairs(((1, 16), (8, 16), (48, 16), (228, 16), (8, 256))):
    lag, ctr, want = run(P, R)
    tail = f" first={lag[:4]}" if lag else "  ALL EXACT"
    print(f"  P={P:3d} R={R:3d} counter={ctr}/{want} mismatches={len(lag)}{tail}", flush=True)

print("=== B: same, reader uses .cv ===", flush=True)
for P, R in _pairs(((8, 16), (48, 16))):
    lag, ctr, want = run(P, R, cv=True)
    tail = f" first={lag[:4]}" if lag else "  ALL EXACT"
    print(f"  P={P:3d} R={R:3d} counter={ctr}/{want} mismatches={len(lag)}{tail}", flush=True)

print("=== C: race-free M0, per-round slots ===", flush=True)
for P, R in _pairs(((1, 32), (8, 32), (48, 32), (228, 16))):
    bad, ctr, want = run_m0(P, R)
    tail = f" {bad[:4]}" if bad else "  ALL EXACT"
    print(f"  P={P:3d} R={R:3d} counter={ctr}/{want} bad_pids={len(bad)}{tail}", flush=True)
