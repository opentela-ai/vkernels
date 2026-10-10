#!/usr/bin/env bash
# verifier/smt/run.sh — index/tiling equivalence checks.
#
# The exhaustive layer needs only Python 3 and always runs.  The symbolic
# layer needs z3 (pip install z3-solver); its absence is a skip (exit 77)
# only when --require-z3 is passed, otherwise it is a printed note.
#
# The interpreter is chosen as the first of $PYTHON, the repo's .venv, and
# python3 that can `import z3`, so a `uv pip install z3-solver` is picked up
# automatically.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"

pick_python() {
  local cand
  for cand in "${PYTHON:-}" "$root/.venv/bin/python" python3; do
    [[ -n "$cand" ]] || continue
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import z3' >/dev/null 2>&1; then
      printf '%s' "$cand"
      return 0
    fi
  done
  printf '%s' "${PYTHON:-python3}"
}

py="$(pick_python)"
exec "$py" "$here/gemm_index_equivalence.py" "$@"
