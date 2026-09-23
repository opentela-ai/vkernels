#!/bin/bash
# Fast SGPR/VGPR/spill audit for a HIP source's device code (used to tune
# issue #151: the fused-head prefill kernels must not spill — a 57-SGPR
# spill cost ~50x per-key throughput on gfx942).
# Usage: issue151_checkregs.sh [source-file]   (default: src/c mla.hip)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC=${1:-$ROOT/src/c/vkernels/kernels/mla.hip}
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
hipcc -O3 -DNDEBUG -std=c++17 --offload-arch=gfx942 -fPIC \
  -I"$ROOT/src/c" -c "$SRC" -o $OUT/k.o 2>/dev/null
llvm-objcopy --only-section=.hip_fatbin -O binary $OUT/k.o $OUT/fb.bin
python3 - "$OUT" <<'PYEOF'
import sys, struct
out = sys.argv[1]
data = open(out + "/fb.bin", "rb").read()
locs = []
i = 0
while True:
    i = data.find(b"\x7fELF", i)
    if i < 0: break
    locs.append(i); i += 4
locs.append(len(data))
best = max(range(len(locs) - 1), key=lambda k: locs[k + 1] - locs[k])
open(out + "/code.so", "wb").write(data[locs[best]:locs[best + 1]])
PYEOF
llvm-readobj --notes $OUT/code.so 2>/dev/null > $OUT/kernels.notes
python3 - "$OUT" <<'PYEOF'
import sys, re
text = open(sys.argv[1] + "/kernels.notes").read()
for blk in re.split(r"\n  - \.agpr", text):
    m = re.search(r"\.name:\s+(\S+)", blk)
    if not m: continue
    short = re.sub(r"_ZN8vkernels7kernels3hip\d*", "", m.group(1))
    short = re.sub(r"E[vPKfiSHLNj]+.*", "", short)
    def g(k, blk=blk):
        mm = re.search(r"\." + k + r":\s+(\d+)", blk)
        return mm.group(1) if mm else "-"
    print("%-56s vgpr=%4s vgpr_spill=%3s sgpr=%4s sgpr_spill=%3s lds=%6s" %
          (short[:56], g("vgpr_count"), g("vgpr_spill_count"),
           g("sgpr_count"), g("sgpr_spill_count"),
           g("group_segment_fixed_size")))
PYEOF
