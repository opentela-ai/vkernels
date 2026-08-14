#!/usr/bin/env bash
# meta/benchmarks/bench_kda.sh
#
# Driver for bench_kda on beverin MI300A. Runs each (H,S,D) delta_rule_fwd
# shape as a SEPARATE short-lived process (bench_kda H S D), because the
# MI300A runtime faults a long-lived single-context sweep non-
# deterministically ("Memory access fault, Reason: Unknown") even though the
# identical kernel runs cleanly in the one-shot correctness harness. Each
# invocation owns its own HIP context, does ~80 ms of work, and exits, so the
# per-context fault never triggers. Then the short supporting kernels
# (layer_norm_gated, gate_chunk_cumsum) run once in their own process.
#
# Usage (from a Slurm mi300 allocation):
#   bench_kda.sh /path/to/build/meta/benchmarks/kda_bench
set -euo pipefail
BIN="${1:?usage: bench_kda.sh /path/to/kda_bench}"
ulimit -c 0   # the faulting runs dump multi-hundred-MB GPU cores otherwise

printf '=== hip::kda_delta_rule_fwd (D x D state in gmem, row-parallel) ===\n'
printf '  %4s %6s %4s %9s %9s %9s %9s %5s  %s\n' \
       H S D us\(min\) us\(med\) TFLOP/s GB/s AI bound
for cfg in "1 64 16" "1 64 32" "1 64 64" "16 64 64" \
           "1 512 64" "1 512 128"; do
  # bench_kda H S D prints only its data row (driver owns the table).
  "$BIN" $cfg
done
printf '  Roof: 1307 TFLOP/s bf16, 5300 GB/s HBM, ridge ~247 FLOP/B\n'

printf '\n'
"$BIN"   # no args: layer_norm_gated + gate_chunk_cumsum
