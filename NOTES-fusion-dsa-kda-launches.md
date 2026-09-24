# NOTES-fusion-dsa-kda-launches — launch-count fusion pass (dsa_topk / kda)

Fusion lane (one of five). Scope: `src/c/vkernels/kernels/dsa_topk.*` and/or
`kda.*`. Two candidates from the brief; one implemented, one assessed and
deferred with reasoning.

## Candidate (ii) — kda chunked-WY: fold the per-key cumsum launch into the
## gram-kernel prologue (IMPLEMENTED)

### Baseline

`hip::kda_delta_rule_fwd_chunked_with_scratch` (kda.hip #L8, issue #70) —
the K3 prefill path where the WY form wins 1.95–3.91x over the cooperative
kernel (docs/kernels/kda.md) — launches FOUR kernels per call:

1. `kda_k3_cumsum_kernel` — per-key within-chunk log-cumsum `L`
   (grid (B·H, nc), one thread per key column, serial 64-step `logf` chain);
2. `kda_k3_gram_kernel<D>` — gate-weighted grams M/N (stages `sk`/`sL`
   from gmem);
3. `kda_k3_local_kernel<D>` — Ainv/U_v/W/T/Opar (reads `L` again);
4. `kda_k3_state_kernel<D,Rb>` — serial state pass (does NOT read `L`).

Phase decomposition (issue #83, job 640235, MI300A): the cumsum launch is
23–36 us of 748–2922 us; at the decode shapes (S=64, nc=1) the four fixed
launches are exactly what makes the chunked path lose to the cooperative
kernel.

Key observation: `L` is **per-chunk local** — the cumsum kernel has no
cross-chunk dependency, and kernels 2 and 3 each read only their own
chunk's `L`. Kernel 4 does not read `L` at all.

### Change

Additive, env-gated (default = proven path, per repo convention):

- `kda_k3_gram_cumsum_kernel<D>` (new, kda.hip): the gram kernel with the
  cumsum statements as a prologue. One thread per key column (tid < D,
  kThL=512 threads — D ≤ 128 fits), accumulating in the SAME ascending
  token order with the SAME `(gg <= 0) ? -1e9f : logf(gg)` clamp as the
  standalone kernel → every `L` value is **bit-identical**, so every
  downstream float is. Writes the `L` gmem scratch (kernel 3 still reads
  it) and stages `sL` directly, eliminating the gram kernel's `L` gmem
  re-read. The rest of the kernel body is the proven gram kernel verbatim.
- `launch_chunked<D>` gains a `bool fused_cumsum` tail parameter (internal,
  anonymous-namespace helper; no ABI change).
- `kda_delta_rule_fwd_chunked_with_scratch` (public entry, signature
  unchanged) reads `VK_KDA_CHUNKED_FUSED` per call (mhc.hip's
  `VK_MHC_PRE_STRICT` getenv pattern): `=1` → fused chain, default → proven
  4-launch chain.
- kda.hpp: contract comment on the entry documents the gate (comment-only).

### Validation (GB10, HIP-on-NVIDIA shim — no ROCm on this box)

`kda.hip` joins the CUDA shim list (`VKERNELS_CUDA_HIP_SOURCES` in
src/c/CMakeLists.txt — SHARED FILE, 1 line, flagged as a merge-conflict
point), which makes the device path compilable AND runnable here. Shim
additions (cuda_compat/hip/hip_runtime.h, shared shim — additive defines,
per its own "add the mapping here" contract): `hipMemsetAsync`,
`__shfl_xor(var, mask) → __shfl_xor_sync(0xFFFFFFFFu, var, mask)` (CUDA 13
removed the unsynced intrinsic; every kda.hip call site is a full-warp
32-lane butterfly with all lanes active).

- **Backend capability gate** (`VKERNELS_KDA_CUDA_SHIM`, defined for kda.hip
  and the new test TU on CUDA builds): the D=128 gram kernels stage
  2×64×128×4 B = 64 KB of STATIC shared — within gfx942's budget but over
  NVIDIA's 48 KB static cap → the shim build serves D ≤ 64 only and errors
  loudly for D > 64; `kda_chunked_phase_times`' D=128 arm is likewise
  excluded. D=128 stays validated on HIP/gfx942 as before.
- **New committed GPU test** `tests/kernels/attn/test_kda_chunked_fused_gpu.cpp`
  (registered in tests/CMakeLists.txt — SHARED FILE, flagged): two gates.
  (1) BIT-IDENTITY: fused-vs-proven out AND final state must be
  `memcmp`-equal (a single ULP is a failure). (2) ORACLE PARITY: both
  chains vs `kda_delta_rule_fwd_state_cpu` < 2e-2 (the
  test_kda_chunked.hip tolerance). On host-only builds the TU compiles to
  an honest skip (device path not compiled), keeping host CI green.
  On host build: `build/host` 46/46 PASS. On `build/cuda`: 56/56 PASS,
  with the device gate bit-identical at {1,1,64,16}, {1,2,128,64},
  {2,1,128,32} and oracle max_rel ≤ 4.4e-7 (D=128 shapes self-skip on the
  shim; covered on gfx942).
- **Ad-hoc wall A/B** (uncommitted driver, sync-per-call wall, 200 iters,
  warmup 10): H=16 S=64 D=64 607.8 → 412.3 us (−32%); H=16 S=512 D=64
  1039.1 → 988.5 us (−4.9%); H=32 S=512 D=64 1711.0 → 1692.3 us (−1.1%).
  GB10 is launch-overhead-dominated so the decode-shape delta is inflated
  vs what MI300A should show (~5–40 us: one launch ≈ 3–5 us + the 23–36 us
  cumsum phase partially hidden under the gram kernel's staging latency).

### Pending (for the MI300A lane)

- Extend `meta/scripts/ab_kda_chunked_mi300.sh` (or bench_kda_chunked.hip)
  to time the entry with `VK_KDA_CHUNKED_FUSED=0/1` at the K3 six-config set
  + H sweep; then run `test_kda_chunked_fused_gpu` incl. the D=128 shapes.
- If the A/B holds (expected: strictly ≥, since bit-identical work minus one
  launch), flip the env default to fused in a follow-up.

### Incidents

- nvcc: `__shfl_xor` undefined (CUDA 13 removal) — fixed in the shim, not
  in kda.hip (the gfx942 source must not change for the port's sake).
- nvlink: 64 KB static shared at D=128 over NVIDIA's 48 KB static cap —
  resolved with the `VKERNELS_KDA_CUDA_SHIM` capability gate, NOT by
  touching the proven kernel's shared-memory layout.
- No clang-format binary on this box; the edited regions were
  hand-formatted to the surrounding style (2-space indent, ≤100 cols,
  trailing-comment alignment).

## Candidate (i) — dsa_topk: fold the transform into the logits kernel tail
## (ASSESSED, DEFERRED — out of scope + cross-block dependency)

The brief suggested folding `dsa_topk_transform` into `dsa_topk_logits`'
tail to drop one launch from the indexer chain. Assessed, not implemented:

1. **Out of file scope.** `dsa_topk_logits` lives in `dsa.hip`/`dsa.cpp`
   (the MFMA/wmma launcher with four variants), NOT in `dsa_topk.*` —
   this lane's scope. Any fold means editing the most perf-critical,
   most heavily gated kernel in the repo.
2. **Cross-block dependency.** The logits kernels split one score row
   across `split_kv` blocks (block = (batch, split_kv), lane = KV token);
   the transform's radix selection needs the COMPLETE score row. A tail
   fold therefore needs a grid-wide barrier (cooperative launch or
   grid-sync) or a second phase in the same kernel — a rewrite of the
   proven launch geometry, not a tail stitch. Bit-exactness of the
   selection (value desc, index asc tie-break) is contract-critical here.
3. **Value.** The transform kernel is one small launch per call, but it
   runs once per decode step per layer — the launch saving is real but
   strictly smaller than the risk above.

Recommended follow-up for whoever owns dsa.hip: a cooperative-launch
variant of the AUTO dispatcher (`dsa_topk_logits_with_variant`) that runs
the transform as a phase-2 kernel in the SAME `hipLaunchCooperativeKernel`
grid after a `grid.sync()`, gated `VK_DSA_TOPK_FUSED=1` default-off, with
the transform outputs compared bit-exact against the standalone kernel.
