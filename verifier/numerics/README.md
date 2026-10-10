# numerics — certified floating-point error bounds

Testing can show the device kernel is *close* to the CPU oracle on the inputs
you ran.  It cannot tell you the worst-case gap.  This family derives a
**certified bound** on `|device - real|` for the numerically sensitive
computation, using [Gappa](https://gappa.gitlabpages.inria.fr/) (or FPTaylor /
Rosa), and lets the parity tolerance be read off a proven number instead of an
observed one.

## Why this matters for vkernels

The kernels that are hardest to test — MLA/DSA online softmax, the KDA
chunked recurrence, MoE fp8/bf16 dequant, multi-stage reductions — accumulate
in a different order on the device than in the oracle, so bit-exactness is
not achievable and never the goal.  The docs say "matches to within fp32
round-off"; this family is how that phrase becomes precise.

## The pattern

A bound script declares the input ranges, the exact (real-number) reference
expression, and the floating-point expression, then asks the prover for the
relative error.  For a length-`N` sequential sum of terms bounded by `M`,
with unit round-off `u = 2^-24` for binary32 round-to-nearest:

```gappa
# bounds/sum_error.gappa (illustrative pattern)
r = 0 -> 1;                      # the reduction order, as a free variable
...
```

The concrete script for the sequential-sum accumulation is:

```gappa
# Sequential accumulation s_{i+1} = fl(s_i + x_i), |x_i| <= M, |s_0| <= M.
# Prove |s_N - sum x_i| <= ((1+u)^N - 1) * N * M / ... via Gappa's
# relative-error machinery:
```

Write each bound as a `.gappa` file under `bounds/`; `./run.sh` checks all of
them.  The scripts are *not* shipped until they have been checked in, because
a wrong bound is worse than an unstated one.

## Relationship to the tests

| Question | Where answered |
|---|---|
| Does the device agree with the oracle on these inputs? | `tests/python/` GPU-parity tests |
| What is the worst-case bound for *all* inputs? | this family |
| Are the oracle's element ops exactly right? | `verifier/bmc/` |
| Is the tiling equivalent to the naive loop? | `verifier/smt/` |

## Install

Gappa needs only GMP, MPFR, Boost **headers**, Flex and Bison — no OCaml.
There is no root on this host, so it is built from source against
dependencies from conda-forge (Gappa is not packaged for linux-aarch64):

```bash
export MAMBA_ROOT_PREFIX=~/.local/opt/mamba
micromamba create -y -p ~/.local/opt/gappa-deps -c conda-forge \
    gmp mpfr libboost-devel
cd /tmp && curl -LO https://gappa.gitlabpages.inria.fr/releases/gappa-1.8.3.tar.gz
# ./configure CXX=g++ \
#   CPPFLAGS=-I~/.local/opt/gappa-deps/include \
#   LDFLAGS="-L~/.local/opt/gappa-deps/lib -Wl,-rpath,~/.local/opt/gappa-deps/lib"
# ./remake            # produces src/gappa, a standalone executable
```

The standalone `src/gappa` is copied to `~/.local/bin/gappa`.  (FPTaylor and
Rosa are equally acceptable if already present.)

## Shipped bounds

`bounds/sequential_sum_fp32.gappa` certifies the sequential four-term fp32
accumulation against the exact sum: for all `x_i` in `[-1,1]`,
`|s3 - z| <= 5*2^-24` (~2.98e-7), where `s3 = fl(fl(fl(x0+x1)+x2)+x3)`.
That is the reduction-side half of the device-vs-oracle gap; the oracle
reorders the sum, so this turns "within fp32 round-off" into a number.
