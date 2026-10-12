"""Hang-localiser: one barrier shape per process, so a hang is contained.

Driver runs each CASE as a separate `timeout`-wrapped subprocess.  All cases
use the race-free per-round-slot judge; the point is not the values but
whether the kernel *returns*.

  A  barrier body written inline in the kernel
  B  the shipped helper (`h_ship`, target: tl.int64)
  C  helper with a tl.int32 target
  D  helper with the .cv-load poll (known to release; wrong values)
  E  inline body, but with no trailing tl.debug_barrier()
"""
import sys

import triton
import triton.language as tl
import torch

CASE = sys.argv[1]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 1
R = int(sys.argv[3]) if len(sys.argv) > 3 else 32


@triton.jit
def h_ship(bar_ptr, target: tl.int64):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_ship_i32(bar_ptr, target: tl.int32):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_ge(bar_ptr, target: tl.int64):
    """F: int64 target, guard written as `not (c >= target)`."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while not (c >= target):
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_diff(bar_ptr, target: tl.int64):
    """G: int64 target, spin on the (small) remaining difference."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while target - c > 0:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_inner(bar_ptr, target: tl.int64):
    """H: int64 target, reconvergence point inside the spin body."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        tl.debug_barrier()
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_i32widen(bar_ptr, target: tl.int32):
    """I: int32 argument, widened back to int64 for the comparison."""
    t64 = target.to(tl.int64)
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < t64:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_j_i64arg(bar_ptr, target: tl.int64):
    """J: int64 argument, but a 32-bit guard (wrapping difference)."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    t32 = target.to(tl.int32)
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu").to(tl.int32)
    while t32 - c > 0:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu").to(tl.int32)
    tl.debug_barrier()


@triton.jit
def h_k_i32arg(bar_ptr, target: tl.int32):
    """K: int32 argument and a 32-bit guard (wrapping difference)."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu").to(tl.int32)
    while target - c > 0:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu").to(tl.int32)
    tl.debug_barrier()


@triton.jit
def h_l_i64arg_cguard(bar_ptr, target: tl.int64):
    """L: int64 argument, truncated inside, case-C guard ``c < t32``."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    t32 = target.to(tl.int32)
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < t32:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_m_i64arg_wide(bar_ptr, target: tl.int64):
    """M: int64 argument, case-C guard against the raw int64 target."""
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    while c < target:
        c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def h_cv(bar_ptr, target: tl.int64):
    tl.debug_barrier()
    tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
    c = tl.load(bar_ptr, cache_modifier=".cv")
    while c < target:
        c = tl.load(bar_ptr, cache_modifier=".cv")
    tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
    tl.debug_barrier()


@triton.jit
def k_A(out_ptr, bar_ptr, P, ROUNDS, base: tl.int64, CASE: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((), tl.float32)
    r = 0
    while r < ROUNDS:
        tl.store(out_ptr + r * P + pid, pid * 1000.0 + r)
        if CASE == "A" or CASE == "E":
            tl.debug_barrier()
            tl.atomic_add(bar_ptr, 1, sem="acq_rel", scope="gpu")
            c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            while c < base + (r + 1) * P:
                c = tl.atomic_add(bar_ptr, 0, sem="acquire", scope="gpu")
            if CASE == "A":
                tl.debug_barrier()
        elif CASE == "B":
            h_ship(bar_ptr, base + (r + 1) * P)
        elif CASE == "C":
            h_ship_i32(bar_ptr, (base + (r + 1) * P).to(tl.int32))
        elif CASE == "D":
            h_cv(bar_ptr, base + (r + 1) * P)
        elif CASE == "F":
            h_ge(bar_ptr, base + (r + 1) * P)
        elif CASE == "G":
            h_diff(bar_ptr, base + (r + 1) * P)
        elif CASE == "H":
            h_inner(bar_ptr, base + (r + 1) * P)
        elif CASE == "I":
            h_i32widen(bar_ptr, (base + (r + 1) * P).to(tl.int32))
        elif CASE == "J":
            h_j_i64arg(bar_ptr, base + (r + 1) * P)
        elif CASE == "K":
            h_k_i32arg(bar_ptr, (base + (r + 1) * P).to(tl.int32))
        elif CASE == "L":
            h_l_i64arg_cguard(bar_ptr, base + (r + 1) * P)
        elif CASE == "M":
            h_m_i64arg_wide(bar_ptr, base + (r + 1) * P)
        nb = (pid + 1) % P
        acc += tl.load(out_ptr + r * P + nb)
        r += 1
    tl.store(out_ptr + ROUNDS * P + pid, acc)


out = torch.full((R * P + P + 1,), -1.0, device="cuda")
bar = torch.zeros(1, dtype=torch.int64, device="cuda")
k_A[(P,)](out, bar, P, R, 0, CASE=CASE, num_warps=8)
torch.cuda.synchronize()
got = out[R * P : R * P + P].cpu()
bad = [(pid, float(got[pid]) - float(sum(((pid + 1) % P) * 1000 + r for r in range(R))))
       for pid in range(P)]
bad = [b for b in bad if b[1] != 0.0]
print(f"OK case={CASE} P={P} R={R} counter={int(bar[0])}/{P*R} bad={bad[:3]}", flush=True)
