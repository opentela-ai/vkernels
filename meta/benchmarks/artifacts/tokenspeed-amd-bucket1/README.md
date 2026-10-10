# TokenSpeed-Kernel-AMD "bucket 1" port: beverin (MI300A) measurements

Beverin (gfx942/MI300A, x86_64 container, Slurm `mi300` partition,
account `a-infra02`), 2026-10-10. Jobs 675311 (stock env), 675350/675351
(triton 3.8 leg). Staging:
`/iopsstor/scratch/cscs/xyao/floe-beverin/tokenspeed-amd-bucket1/`.

## What was ported

`src/python/vkernels/torch_ops/tokenspeed_amd/` — MIT-vendored from
lightseekorg/tokenspeed `tokenspeed-kernel-amd` (fetched 2026-11, `main`):

| component | kind | needs |
|---|---|---|
| `scale_layout.py` — CDNA4 MXFP4 e8m0 scale swizzle | pure torch | any device |
| `weight_preprocess.py` — gate/up interleave, W2 block-N padding, gdot128 preshuffle | pure torch | any device |
| `n16_weights.py` — N16 packed-cell weight banks | pure torch + gluon offset helpers | any device |
| `routing.py` — softmax top-k (generic) + sigmoid-bias (cdna4) fused routers | gluon | gluon / gluon+cdna4 |
| `moe_sorting.py` — 4-stage block-aligned expert sort | gluon | gluon |
| `hadamard.py` — length-128 register butterfly | gluon | gluon |
| `mhc.py` — decode-shaped hyperconnection pre reduce/apply | gluon | gluon |
| `argmax.py` — split-atomic sampling argmax + torch fallback | gluon+cdna4 | gluon+cdna4 |

Registry: `tokenspeed_amd_registry.py` (ops `router_topk`, `moe_sorting`,
`sampling_argmax`, `hadamard_transform`, `mhc_pre`; PORTABLE band, format
signatures). Tests: `tests/python/test_tokenspeed_amd.py` — **29/29 pass on
MI300A** (24 CPU-side + 5 GPU parity, job 675350), 24 pass / 5 GPU-skip on
the CPU-only dev box.

## Environment finding: gluon needs triton ≥ 3.8

The container's stock triton **3.4.0** ships the first experimental gluon —
feature-incomplete for *every* vendored kernel: no
`gluon.constexpr_function` (hadamard/argmax), no `gl.histogram` (sorting
stage 1/2), no `gluon.language.amd.cdna4` (sigmoid routing, argmax loads),
and routing/mhc hit `CompilationError` on the 3.4 stdlib. The vendored
`_gluon.py` shim degrades to inert stubs (import-safe on any host) and the
probes (`gluon_features()`) report the per-kernel availability; with stock
3.4 only the pure-torch components run (measured: `results-stock-triton34.json`).

Fix: install triton 3.8.0 from the PyTorch rocm7.0 nightly index into scratch
(`triton-3.8.0+gitdf3f91dd-cp310-cp310-manylinux_2_27_x86_64...whl`, pip
`--target`; the container is **x86_64** — aarch64 wheels fail with a
misleading `cannot open shared object file` dlopen error) and prepend it to
`PYTHONPATH`. All features green (`constexpr_function`, `associative_scan`,
`amd.cdna4`); torch 2.9.0a+git7bcbafe (hip 7.0.51831) needs no change.
Sbatch: `bench-bucket1-t38.sbatch` in the staging dir.

## Results (MI300A, triton 3.8, medians of 50 iters after 10 warmup)

`results-triton38.json`. Timings in µs.

### Routing — fused softmax top-k vs torch chain

Gluon single-CTA kernel (limit M·topk ≤ 128, E ≤ 1024) vs the
vLLM-style torch chain (softmax → bias → topk → gather → normalize → scale):

| M, E, topk | gluon | torch | speedup |
|---|---:|---:|---:|
| 1, 128, 16 | 68.3 | 139.9 | 2.0× |
| 1, 384, 16 | 106.3 | 137.5 | 1.3× |
| 8, 256, 16 | 47.1 | 141.9 | 3.0× |
| 16, 128, 8 | 107.9 | 142.3 | 1.3× |
| 64, 384, 2 | 53.8 | 139.7 | 2.6× |

The 47–108 µs spread tracks `NUM_WARPS = next_pow2(M)` respecialization
(M=1/E=384 and M=16 pick worse wave counts on gfx942 than MI355X tuning).
Still 1.3–3× over the 5-op torch chain, and it removes the chain's extra
kernel launches.

### MoE sorting — 4 gluon stages vs aiter CK vs torch reference

| M, E, topk, block | gluon | aiter | torch ref |
|---|---:|---:|---:|
| 8, 128, 4, 16 (decode) | 96.2 | **30.5** | 174.7 |
| 8, 256, 8, 16 (decode) | 90.1 | **41.0** | 174.3 |
| 64, 128, 8, 64 | 89.8 | **31.2** | 162.8 |
| 512, 128, 4, 64 (prefill) | 114.0 | **36.0** | 165.3 |
| 4096, 256, 8, 128 (prefill) | **125.5** | 165.9 | 221.0 |

The aiter `moe_align_block_size` we already wrap stays the decode/prefill
default on gfx942; the gluon sort only wins at the largest prefill shape
(1.3× vs aiter) and beats the torch reference everywhere. Note the outputs
differ in contract: gluon packs `sorted_ids = (k << 24) | token` with
weights, aiter emits flat `token*topk + slot` with per-block experts.

### Hadamard-128 (bf16 rows, fp32 butterfly, scale = 1/√128)

| rows | gluon | effective BW | torch fp32 matmul |
|---|---:|---:|---:|
| 4 096 | 40.6 µs | 52 GB/s | 727 ms |
| 32 768 | 42.7 µs | 336 GB/s | 713 ms |
| 262 144 | 157.6 µs | 829 GB/s | 700 ms |

Single-wave-per-row register butterfly; the torch baseline materializes an
fp32 copy + GEMM (≈4 orders slower). 829 GB/s at the large shape is
HBM-roofline territory for a 33.5 MB×2 streaming op on MI300A.

### mHC pre reduce/apply (decode-shaped, hidden 4096, 64 split-K partials)

| tokens | gluon | torch ref | speedup |
|---|---:|---:|---:|
| 1 | 42.3 | 1237.8 | 29× |
| 8 | 41.8 | 1234.9 | 30× |
| 64 | 43.0 | 1250.4 | 29× |

Parity verified bit-tight on layer_input/post and to 5e-3 on the 20-iteration
Sinkhorn comb (the kernel places `HC_EPS` in the denominator during the loop,
matching the upstream Triton reference — verified empirically, see the test
comment). Needs the split-K partial GEMM (`gemm_out_mul` [splits, T, 24])
upstream of it to be useful in serving.

### Sampling argmax

| M, N | torch | gluon |
|---|---:|---:|
| 1, 32 768 | **21.7** | 101.4 |
| 8, 131 072 | 44.9 | 51.9 |
| 64, 151 936 | **55.8** | 57.6 |
| 8, 262 144 | 102.5 | **50.2** (2.0×) |

Split-atomic paths win at large vocab with few rows; torch wins small-M and
M=64 (config table tuned for MI355X CU counts). The launcher's fallback
keeps torch semantics where the probe says gluon is unavailable.

### Host-side preprocessing (pure torch, any GPU)

CDNA4 scale swizzle on a GPT-OSS-120B-sized scale tensor
(368 640 × 90 = 33 MB): **942 µs** — one-time load cost, not on the
serving path.

## Artifacts

- `results-triton38.json` — full run, all sections error-free (job 675351)
- `results-triton38-partial.json` — earlier triton-3.8 run before the
  parity-reference fixes (kept for provenance)
- `results-stock-triton34.json` — stock env: only pure-torch + argmax-torch
  baselines measurable; every gluon kernel feature-blocked on triton 3.4
- Repro: staging `bench-bucket1.sbatch` (stock) / `bench-bucket1-t38.sbatch`
  (triton-3.8 scratch install), bench script
  `bench/bench_tokenspeed_amd_bucket1.py` in-tree.

## Follow-ups

- The triton-3.8 scratch install is a benchmark leg, not a serving change;
  getting these into production needs the container env bumped (or the
  scratch wheel blessed by infra).
- Routing `NUM_WARPS` tuning for gfx942 (the M=1/E=384 and M=16 shapes are
  2× off the best config) — small tuning table, upstream tuned MI355X.
- Sorting stays behind aiter on gfx942 except huge-prefill shapes; the
  packed-id + weights contract differs from aiter's, so a consumer switch
  is not drop-in.
- Buckets 2/3 of the borrowing plan (CDNA3 attention ports; MI350 day-0
  mxfp4 MoE + dense MLA/DSA/KDA Gluon set) per
  `docs/borrowing-tokenspeed-deepgemm.md`.
