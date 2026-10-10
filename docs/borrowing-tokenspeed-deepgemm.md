# Learning from tokenspeed-kernel and DeepGEMM

Survey of two production kernel libraries — [lightseekorg/tokenspeed]
(`tokenspeed-kernel/` subpackage) and [deepseek-ai/DeepGEMM] — answering two
questions: how do they manage the kernel-development scaffold (and what of
that design should we borrow), and which kernels are worth importing into
vkernels. Both repos are MIT-licensed, so code-level borrowing is fine with
attribution.

Both repos were surveyed at shallow-clone depth; paths below are relative to
each repo root. clones lived at `/tmp/tokenspeed` and `/tmp/DeepGEMM`.

[tokenspeed]: https://github.com/lightseekorg/tokenspeed
[DeepGEMM]: https://github.com/deepseek-ai/DeepGEMM

## 1. How each repo manages the scaffold

### tokenspeed-kernel — Python-first, layered registry

A pip-installable standalone kernel library (`tokenspeed-kernel/` inside the
tokenspeed inference engine). Architecture:

```
public API  (attention.mha.mha_prefill, mm, ...)
    │
select_kernel   (family, mode, format_signature, traits, ...)
    │ queries
KernelRegistry ← @register_kernel(...) populates
    │
ops/<family>/<variant>/<solution>.py   (triton / gluon / cute_dsl / vendor lib)
    └ reference (PyTorch ground truth, never auto-selected)
```

Scaffold components (file sizes at survey time):

| Component | File | What it does |
|---|---|---|
| Registry | `registry.py` (509 LOC) | `@register_kernel(family, mode, features, solution, capability, signatures, traits, priority)` |
| Priority bands | `registry.py` | `REFERENCE=0 / PORTABLE=4..7 / PERFORMANT=8..11 / SPECIALIZED=12..15 / PLUGIN=16..19` |
| Format contracts | `signature.py` (293 LOC) | Typed `TensorFormat` / `ScaleFormat` / `FormatSignature` (dense vs paged, fp8+block-scale, …) |
| Selection | `selection.py` (867 LOC) | Capability+trait filtering, per-family `SelectionOracle` ranking, per-call `solution=`/`override=` + config-file overrides |
| Platform | `platform.py` (829 LOC) | `PlatformInfo`, capability detection, `CapabilityRequirement(min_arch_version, required_features)` |
| Numerics | `numerics/` | Reference impls, dtype-aware tolerances, comparison + **bisect** CLI |
| Benchmark | `benchmark/` | `KernelBenchmarkHarness` — graph-replay device timing, JSON suites per `<vendor>/<arch>.json`, **PR CI compares against merge base** |
| Profiling | `profiling.py` (421 LOC) | ShapeCapture, `kernel_scope` for traces, Proton bootstrap |
| Compile monitor | `compile_monitor` | JIT watchdog (see below) |
| Plugins | `plugins/` | Out-of-tree backends register via the same decorator and win via the reserved PLUGIN band |

Design decisions worth internalizing:

* **Priority bands with reserved plugin headroom.** Out-of-tree plugin
  authors can always override in-tree defaults without auditing every
  registration. Bands also encode a portability/performance *contract*
  (PORTABLE = works everywhere, SPECIALIZED = narrowly gated on arch+shape).
* **Declarative gating.** A kernel declares its format signatures, arch
  capability, and traits (head_dim, GQA factor, …). Selection is then data,
  explainable, and testable — no predicate logic buried in launcher code.
* **The compile monitor.** Compile-time parameters (`tl.constexpr`,
  `gl.constexpr`) key the JIT cache; a per-batch value passed as one
  recompiles on the forward thread (100 ms–seconds). Their monitor hooks
  Triton's JIT, marks end-of-startup, then logs every later compilation
  with duration, what changed in the compile key, and the launching call
  site. CI serving runs with `TOKENSPEED_JIT_COMPILE_CHECK=error`; kernel
  tests guard with `assert_no_triton_compile`. Rules: batch-varying counts
  go in `do_not_specialize`, bucketed (`next_power_of_2`) when a
  compile-time bound is needed; pointers sliced at per-batch offsets go in
  `do_not_specialize_on_alignment`.
* **File naming is CI gating.** Vendor-only code lives in files named
  after the solution (`gluon`, `cute_dsl`, …) so CI can skip the other
  vendor's GPU jobs; vendor-neutral code uses `triton`. Vendor tests under
  `test/<vendor>/`.
* **Per-op READMEs.** Top-level README holds system design only; details
  live in `ops/<family>/README.md`.
* **`weight_preprocessor` hook** on registrations — e.g. BF16 expert copies
  built at load time while compact FP8 experts stay for decode.

### DeepGEMM — C++/CUDA-first, runtime JIT, header layering

No CUDA compilation at install: everything JIT-compiles at runtime per shape
via **DeepJIT** (`third-party/deep_jit` submodule, plus CUTLASS submodule).
Library layout:

```
include/deep_gemm/
  common/    tma_copy.cuh, packing.cuh, ring_pipeline.cuh, exception.cuh, compile.cuh
  mma/       sm90.cuh, sm100.cuh           ← warp-level MMA wrappers
  epilogue/  operators.cuh, store variants  ← composable epilogue operators
  scheduler/ gemm.cuh, mega_*.cuh          ← persistent-tile schedulers
  impls/     sm90_*.cuh, sm100_*.cuh       ← kernel bodies, one per kernel
csrc/
  apis/        public C++ API per domain (gemm.hpp, attention.hpp, mega_moe.hpp, …)
  jit_kernels/ impls/   compile_and_launch — std::format writes the template
                       instantiation from the chosen config
               heuristics/  sm90.hpp / sm100.hpp — enumerate config candidates
                           (block_m/n/k, cluster, swap_ab, stages) as data
  runtime/     jit.hpp  (LazyInit<Runtime<CUDA>>, include paths, nvcc flags)
deep_gemm/
  testing/     numeric.py, bench.py        ← see §3.9
  tests/       flat pytest per domain + generators.py for shape enumeration
```

The load-bearing split: **heuristics choose a config as data →
`jit_kernels/impls` formats it into a template instantiation string →
`include/impls` holds the kernel**. Runtime knobs (`set_num_sms`,
`set_tc_util`, `set_pdl`, `set_block_size_multiple_of`) inject without
rebuilding the library. Kernels are a small closed set (FP8/FP4/BF16/TF32
GEMMs in all layouts, m/k-grouped contiguous+masked, MQA logits for the
lightning indexer, mega MoE/gate/mHC, einsum, locality domain) — explicitly
kept minimal as "a clean resource for learning GPU kernel optimization".

## 2. What vkernels already has

Don't over-borrow; we are ahead in places:

* **CPU-oracle + CUDA two-implementation model** — host CI, 100% line
  coverage target, GPU tests compare against the oracle. tokenspeed uses
  PyTorch references; DeepGEMM has no reference tier at all.
* **Formal verification** (GPUVerify data-race freedom, ESBMC oracle
  proofs) — unique to us.
* **Metadata-only registry** (`src/python/vkernels/registry.py`) with
  frozen `catalog.json`, deterministic selection, `explain()` with
  per-candidate reasons, `override=`, reference-impl opt-in. Same spirit as
  tokenspeed's registry, smaller: plain `int priority`, ad-hoc `check()`
  predicates, no format-signature types, no plugin band.
* **Tuning cache** (`tuning/`, `VKERNELS_CACHE`; `vkl tune`) — DeepGEMM's
  heuristics persistence equivalent, plus Triton autotune persistence that
  neither repo has in this form.
* **Per-op docs, minitest harness, coverage/asan presets, HIP support,
  Rust/C-API bindings.**

Current kernel inventory (catalog categories): dsa (18), dsa_kpool (10),
dsa_topk (4), mla (7), moe_fused (14), moe_aux (6), glm_moe (6), moe (4),
mhc (6), kda (19), gemm_bf16 (8), elementwise (3), reduce (2) — plus the
comm layer (allreduce, cross-node KV, kv gather/scatter, p2p donate/restore,
fabric import, pipeline boundary, rccl).

## 3. Scaffold designs worth borrowing

Ordered by (value / effort). Items marked **[done 2026-11]** are
implemented in this repo; see the referenced modules.

### From tokenspeed-kernel

1. **Priority bands + PLUGIN headroom** **[done 2026-11]** —
   `vkernels.registry.Priority` (REFERENCE/PORTABLE/PERFORMANT/SPECIALIZED/
   PLUGIN, each four wide in `[0, 20)`), validated at registry construction;
   in-tree registries migrated (`moe_combine_registry`, `expert_gemv_registry`).
2. **Typed format-signature gating** **[done 2026-11]** —
   `vkernels.signature.FormatSignature` (per-role dtype sets + layout tag,
   rejection reasons in the registry vocabulary); `KernelImplementation.signature`
   filters alongside ``check`` in ``explain()``. Both in-tree registries
   migrated (dtype branches moved out of their ``check`` predicates into
   signatures).
3. **Generalize the numerics CLI** **[done 2026-11]** — `vkernels.numerics`
   package: dtype-aware tolerance policy (`tolerance.py`), DeepGEMM-ported
   comparison utils (`compare.py`: `calc_diff`, `count_bytes`,
   `assert_bitwise_equal`, NaN-aware `assert_close` with first-mismatch
   reporting), seeded shape-parameterized generators (`generators.py`),
   kernel-vs-oracle `verify()` (`verify.py`), smallest-failing-size
   `bisect_size()` (`bisect.py`), and `python -m vkernels.numerics` CLI.
   The ad-hoc `bench/kda_*_bisect.py` scripts remain as GPU-fault special
   cases (one process per config).
4. **`KernelBenchmarkHarness`** **[done 2026-11]** — `vkernels.benchmark`
   package: pluggable timers (`HostTimer`, `CudaEventTimer`, `GraphTimer`
   with warmed graph replay and capture-failure fallback to events),
   FLOPs/bytes throughput models per op (the roofline denominators),
   `run_benchmark` requests with an optional oracle-verification gate,
   versioned JSON reports, and the `compare` PR check (median-runtime
   regression beyond a threshold against a merge-base baseline; new/
   missing cases reported, not failed). CLI: `python -m vkernels.benchmark
   run|compare`. torch_ops serving kernels plug in via
   `register_benchmark_case`; the GitHub-Actions job lives in
   `.github/workflows/ci.yml` (`benchmark`: merge-base vs candidate over
   `meta/benchmarks/python/host-suite.json`, failing on >10 % median
   regression; a baseline predating the harness degrades to an empty
   report so every case counts as new, not failed).
5. **Compile monitor + `assert_no_triton_compile`** **[done 2026-11]** —
   `vkernels.compile_monitor`: hooks `triton.knobs.runtime`
   JIT hooks when Triton is present (import stays dependency-free), splits
   startup/serving compile stats at `mark_serving()`, names the
   compile-time parameter and call site that keeps taking new values
   (powers of two excluded), reacts per `VKERNELS_JIT_COMPILE_CHECK`
   (`warn` default / `error`), and ships the `assert_no_triton_compile`
   test guard. Serving integration (calling `mark_serving` from the
   engine) is still pending.
6. **`weight_preprocessor` registration hook** **[done 2026-11]** —
   `KernelImplementation.weight_preprocessor` (entrypoint string, keeps
   selection import-free) + `load_weight_preprocessor()`; the
   `expert_gemv` fnuz-storage variant now declares
   `e4m3fn_to_fnuz_inplace`, so model-load code can discover the one-time
   checkpoint conversion from the selection instead of hard-coding it.
7. **Out-of-tree plugin mechanism** (defer until needed).

### From DeepGEMM

8. **`common/` layering in `src/c/vkernels/kernels/`** **[first cut done
   2026-11]** — `kernels/common/` now exists with `launch.hpp` (the
   launch-and-check idiom + `ceil_div`, so the `cudaGetLastError` funnel
   cannot be forgotten by construction; `elementwise.cu`, `gemm.cu`,
   `mqa_logits.cu`, `gemm_grouped.cu` migrated), `epilogue.hpp` (§3.10)
   and `heuristics.hpp` (§3.11). Further extraction stays incremental —
   helpers move in as they appear a second time.
9. **Drop-in testing utils** **[done 2026-11]** — ported into
   `vkernels.numerics.compare` (see §3.3): `calc_diff` (cosine-similarity
   relative diff), `count_bytes` for bandwidth accounting,
   `assert_bitwise_equal` reporting the first mismatching byte with
   element coordinates.
10. **Epilogue operator composition** **[done 2026-11]** —
    `kernels/common/epilogue.hpp`: `epilogue::Linear` (BLAS beta=0
    semantics) and `epilogue::Relu<Base>` composition, instantiated by BOTH
    the CPU oracle (`gemm.cpp`) and the CUDA kernel (`gemm.cu` templates on
    the epilogue type) — a fused epilogue is now one type, not a kernel
    copy.
11. **Heuristics as first-class arch modules** **[done 2026-11]** —
    `kernels/common/heuristics.hpp`: the swept tile tables as data
    (`gemm_bf16_tile_candidates()`), with `gemm_bf16_config_for` now
    selecting entries from the table (a returned tile outside the swept set
    fails the `is_gemm_bf16_candidate` contract test). Pairs with the
    tuning cache: heuristics = the search space, cache = the persisted
    winner.
12. **DeepJIT runtime compilation** (defer; heavier lift). Would remove
    per-arch CMake preset rebuilds and enable per-shape specialization at
    runtime; evaluate once the include layering exists.

### Also: devflow rules worth adopting verbatim

From their `AGENTS.md`: batch-varying values must be runtime arguments or
bucketed compile-time; reviews check every new/changed kernel signature;
per-op README under the op directory; vendor-specific filenames double as
CI skip filters.

## 4. Kernels to borrow

**Competing implementations of what we already have — cross-pollinate:**

* **DSA** — tokenspeed `ops/attention/dsa/`: `triton.py` (891 LOC),
  `gluon.py` (555 LOC, AMD), `cute_dsl.py`, `deep_gemm.py`,
  `flashinfer.py` wrappers. Ours is CUDA + CPU oracle; their Triton/Gluon
  versions give a portable path and an AMD story.
* **MLA** — tokenspeed `ops/attention/mla/`: `gluon.py` (779 LOC,
  claimed among the fastest MLA on Blackwell), `triton.py`,
  `tokenspeed_mla.py`. Ours (`kernels/mla.{cpp,cu,hpp}`) is CUDA-only.
* **KDA** — tokenspeed `ops/attention/kda/`: `triton.py` (~19K),
  `gluon.py`, `cute_dsl.py`; their `numerics/reference/kda.py` is a second
  oracle to test against.
* **MHC/hyperconnection** — DeepGEMM `mega_mhc` + `tf32_hc_prenorm_gemm`
  (fused pre-norm into the hyperconnection GEMM). We have
  `kernels/mhc.{cpp,hpp}`; the fused-pre-norm design is the interesting
  part.

**Gaps they fill:**

* **MQA-logits / lightning-indexer kernels** (DeepGEMM):
  `fp8_fp4_mqa_logits`, paged + sparse variants, metadata builders. Pairs
  directly with our `dsa_topk` / `dsa_kpool` — this is the DeepSeek-V3.2
  indexer stack; we have the selection side, not the scoring side.
* **m-grouped masked GEMM** (DeepGEMM `m_grouped_fp8_gemm_nt_masked`) —
  CUDA-graph-safe MoE decode GEMM when per-expert token counts are unknown
  to the CPU; also `k_grouped_*` weight-gradient GEMMs.
* **MoE plumbing** (tokenspeed): `moe_topk`, `moe_route/dispatch/combine`,
  precomputed-routing Triton MoE with MXFP4 weights, marlin path.
* **Fused norm/residual/activation/quantization tier** (tokenspeed
  `residual/` ~63K LOC Triton, `layernorm/`, `activation/`,
  `quantization/`) — we lack this op tier entirely (only elementwise add /
  scale / relu).
* **Portable MHA/GQA** (tokenspeed `ops/attention/mha/`: prefill +
  `mha_decode_with_kvcache`) — we only have MLA/DSA-class attention.
* **Reusable CUDA helpers** (DeepGEMM `include/deep_gemm/common/`):
  `tma_copy.cuh`, `packing.cuh`, `ring_pipeline.cuh`; and
  `scheduler/gemm.cuh` (persistent-tile scheduler) → feed
  `kernels/common/`.

**Caveats:**

* tokenspeed Triton kernels import their vendored `tokenspeed_triton`
  fork — porting means swapping imports and re-validating numerics against
  our oracles.
* DeepGEMM kernel bodies require CUTLASS includes; borrow the `.cuh`
  impls and compile AOT through our CMake — adopting DeepJIT is optional
  and separate.
* Their numerics assume PyTorch ground truth; our CPU-oracle contracts
  stay authoritative.

**Suggested order of attack:**

1. Cheap scaffold wins: testing utils (§3.9), priority bands (§3.1),
   compile monitor (§3.5), numerics CLI (§3.3). — **done 2026-11**
   (plus §3.2, §3.4 incl. the CI benchmark job in
   `.github/workflows/ci.yml` + `meta/benchmarks/python/host-suite.json`,
   §3.6, §3.8, §3.10, §3.11)
2. DSA/MLA Triton ports for portability + AMD coverage. — **done
   2026-11**: `torch_ops/tokenspeed_mla.py` (varlen prefill + paged decode
   + page-table helpers, registered via `mla_registry.py`) and
   `torch_ops/tokenspeed_dsa.py` (packed-FP8 + dense-KV sparse attention,
   registered via `dsa_registry.py`); import swaps verified, CPU-side
   paths unit-tested, GPU parity tests ship behind the `gpu` marker.
3. MQA-logits + masked grouped GEMM for the indexer/MoE decode story. —
   **done 2026-11**: `kernels/mqa_logits.{hpp,cpp,cu}` (weighted-ReLU MQA
   logits, oracle + device parity verified on GB10) and
   `kernels/gemm_grouped.{hpp,cpp,cu}` (masked m-grouped NT GEMM,
   untouched-rows contract + device parity); both in the shipped catalog
   (`mqa_logits`, `gemm_grouped` categories).

**Still open:** §3.7 plugins (defer until needed) and §3.12 DeepJIT
(deferred until the include layering grows); `mark_serving()` wiring from
the serving engine (floe-side integration, cross-repo); fp8 variants of
the borrowed kernels (these first cuts are fp32 — the quantized formats
follow the DeepGEMM layouts when the indexer path goes fp8); the tuned
tile/pipeline ladders for the new CUDA kernels (oracle parity is the
deliverable here; profiling drives the ladder).
