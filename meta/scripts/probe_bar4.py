"""Which helper shape makes the SM-form barrier survive the AMD backend?

Judge: race-free per-round-slot microprogram (same as probe_bar2/3).  V=0
inlines the known-good SM body; V=1..4 call it through @triton.jit helpers of
different shapes.  Variant 1 is the shipped form and is expected to hang, so
it runs last.

  0  inline SM body                        (control, known exact)
  2  helper: SM form, no tl.debug_barrier
  3  helper: debug_barrier + release arrive + volatile-load poll + acq RMW
  4  helper: SM form but the poll RMW is acq_rel and starts from a zero read
  1  helper: shipped SM form (debug_barrier + acquire-RMW poll)
"""
import sys

import triton
import triton.language as tl
import torch

CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 48
VLIST = ([int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2
         else [0, 2, 3, 4, 1])


@triton.jit
def h_sm_nofence(bar_ptr, target):
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def h_vol_poll(bar_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
    c = tl.load(bar_ptr, volatile=True)
    while c < target:
        c = tl.load(bar_ptr, volatile=True)
    tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def h_sm_acqrel(bar_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acq_rel", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acq_rel", scope="gpu")


@triton.jit
def h_sm_shipped(bar_ptr, target):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")


@triton.jit
def judge(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64, V: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        tl.store(out_ptr + r * P + pid, pid * 1000.0 + r)
        t = base + (r + 1) * P
        if V == 0:
            tl.debug_barrier()
            tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
            c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            while c < t:
                c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
        elif V == 1:
            h_sm_shipped(bar_ptr, t)
        elif V == 2:
            h_sm_nofence(bar_ptr, t)
        elif V == 3:
            h_vol_poll(bar_ptr, t)
        elif V == 4:
            h_sm_acqrel(bar_ptr, t)
        nb = (pid + 1) % P
        acc += tl.load(out_ptr + r * P + nb)
        r += 1
    tl.store(out_ptr + ROUNDS * P + pid, acc)


def run(P, R, V):
    out = torch.full((R * P + P + 1,), -1.0, device="cuda")
    bar = torch.zeros(1, dtype=torch.int64, device="cuda")
    judge[(P,)](out, bar, P, R, 0, V=V, num_warps=8)
    torch.cuda.synchronize()
    got = out[R * P : R * P + P].cpu()
    bad = []
    for pid in range(P):
        nb = (pid + 1) % P
        expect = float(sum(nb * 1000 + r for r in range(R)))
        d = float(got[pid]) - expect
        if d != 0.0:
            bad.append((pid, d))
    return bar, bad


NAMES = {0: "inline SM (control)", 1: "helper shipped SM",
         2: "helper SM no debug_barrier", 3: "helper volatile-load poll",
         4: "helper SM acq_rel poll"}

for V in VLIST:
    for P, R in [(1, 32), (8, 32), (48, 32), (228, 32)]:
        if P > CAP:
            continue
        try:
            bar, bad = run(P, R, V)
        except Exception as exc:
            print(f"  V={V} P={P:3d}: {type(exc).__name__}: {str(exc)[:110]}", flush=True)
            continue
        tail = f" first={bad[:3]}" if bad else ""
        print(f"  V={V} [{NAMES[V]:28s}] P={P:3d} R={R} counter={int(bar[0])}/{P * R} "
              f"bad_pids={len(bad)}{tail}", flush=True)
    print("", flush=True)
