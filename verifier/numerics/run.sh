#!/usr/bin/env bash
# verifier/numerics/run.sh — certified floating-point error bounds.
#
# The parity tests assert the device path matches the oracle "within fp32
# round-off".  That tolerance is currently empirical.  This family turns it
# into a theorem: a certified bound on |device - real| derived from the
# reduction/softmax/dequant structure, checked by Gappa / FPTaylor / Rosa.
#
# Drop `.gappa` (or `.fptaylor`) files into ./bounds/ and they are checked
# here.  With no prover installed the runner skips (exit 77) rather than
# pretending the bound holds.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bounds="$here/bounds"
shopt -s nullglob
scripts=("$bounds"/*.gappa)

if ! command -v gappa >/dev/null 2>&1; then
  echo "SKIP: gappa not found."
  echo "      Install Gappa (https://gappa.gitlabpages.inria.fr/) to check"
  echo "      the bound scripts under verifier/numerics/bounds/."
  exit 77
fi

if [[ ${#scripts[@]} -eq 0 ]]; then
  echo "SKIP: no bound scripts in ${bounds}/ yet (see README.md for the pattern)."
  exit 77
fi

for s in "${scripts[@]}"; do
  echo "==> gappa: $(basename "$s")"
  gappa "$s"
done
echo "All numeric error bounds discharged."
