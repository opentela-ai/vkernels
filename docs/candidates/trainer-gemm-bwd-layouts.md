# Trainer K1 — `gemm_bf16` training layout variants (dgrad / wgrad / bias-grad) + the missing device C ABI

> **Status: issue draft.** This is the vkernels-side spec for the first
> training-driven kernel work, cut from the vtrainer design
> (`/local/home/xiayao/Documents/code/trainer/docs/DESIGN.md` §3, kernel
> inventory item K1). The consumer-side skeleton is already written and
> probing for exactly the symbol names frozen below
> (`trainer/src/vtrainer/ops/_device.py`) — landing this spec lights up the
> trainer's L1→L2 device paths with **zero trainer changes**.

## 1. Why

vtrainer (the LLM-training repo) consumes vkernels ops through
`torch.autograd.Function` wrappers. Its linear op is the first one live:
`Y = X @ Wᵀ (+ b)`, `X [M, K]`, `W [N, K]` — the shape of every projection
in the model. Backward needs three GEMM-shaped computations the current
`gemm_bf16` family cannot express:

| grad | computation | today |
|---|---|---|
| `dX` (input) | `dX[M,K] = dY[M,N] · W[N,K]` | expressible via dims-swap of the NN kernel, but no entry point, no config, no ABI |
| `dW` (weight) | `dW[N,K] = dYᵀ[M,N]ᵀ · X[M,K]` | **not expressible**: A operand arrives transposed (no A-transposed loader), reduction is over the token dim `M` (needs split-K), and training wants an **fp32 output** |
| `db` (bias) | `db[N] = Σₘ dY[m,n]` | no column-sum kernel |

Backward is ~2/3 of training FLOPs; dgrad + wgrad are ~80% of that. This
issue unblocks the L2 rung of the trainer's op ladder (device fwd+bwd) for
the single most valuable op family.

**Scope note.** This is a *training-side* addition to an inference-tuned
kernel family. The serving recipes are untouched: no existing symbol,
layout, or tuning-table entry changes. Everything below is additive.

## 2. Current state (inventory)

From `src/c/vkernels/kernels/gemm_bf16.hpp`, `src/c/vkernels/capi/`,
`docs/kernels/gemm_bf16.md`, `docs/performance/gemm-bf16/gfx942.md`:

* `gemm_bf16_cpu` — oracle: `C[M,N] = α·A[M,K]·B[K,N] + β·C`, bf16
  in/out, fp32 accumulate, single RNE on store.
* `hip::gemm_bf16` — the NN MFMA kernel (gfx942). **B is `[K, N]` = the
  transposed projection weight `W[N,K].T` materialized row-major** — the
  serving recipe. Routes `M <= 64 && K >= 4·BK` to split-K
  (`gemm_bf16_splitk_wide_kernel` + fixed-order `..._combine_kernel`,
  issue #156's fp32 partial planes). N is bounds-checked per BN-tile
  (every K3 N is a multiple of 16); K is bounds-checked defensively
  (every K3 K is a multiple of 64 so it never fires today).
* Issue #156 family: `gemm_fp8_block_splitk_with_config`,
  `gemm_bf16_splitk_fused_with_config` (single-launch bit-exact fused
  combine, gated `VK_GEMM_SPLITK_FUSED`), `gemv_decode_{bf16,fp8}_splitk`
  (tiny-M, no staging). Split-K machinery over a token-free reduction dim
  already exists and is proven.
* `gemm_bf16_config_for(M, N, K, ...)` — per-shape tile table
  (analytically chosen vs the MI300A roofline; the autotuner in
  `meta/benchmarks/bench_gemm_bf16.hip` regenerates on device). The
  header already contemplates **M = 8192 warmup shapes** ("bf16
  compute-bound (warmup, M >= 1024)") — large-M tiling is not new ground.
* C ABI: host wrappers `vk_gemm_bf16`, `vk_gemm_bf16_config`
  (`capi.hpp`) — but **no `vk_hip_gemm_bf16*` device symbols at all**.
  The device path is reachable today only from C++ harnesses
  (`bench_gemm_bf16.hip`, `test_gemm_bf16_correct.hip`). Every
  Python-reachable device kernel in this repo (dsa/mhc/kda/moe/mla) has
  `vk_hip_*[_stream]` entries; `gemm_bf16` is the outlier.
* Python: the `hip_dsa_mhc.py` pattern (torch-tensor ctypes over
  `libvkernels_hip.so`, lazy lib load via
  `vkernels.vllm_experts.load_libvkernels_hip`, stream-resolved to
  torch's current stream per issue #69, capture-safe, device-guarded) is
  the established consumption mechanism.

## 3. Spec

### 3.1 The key observation: dgrad is the NN kernel with dims swapped

Write the GEMM dims of each training computation as `C[Mc,Nc] =
A[Mc,Kc]·B[Kc,Nc]` and compare storage layouts:

| op | output | A operand | B operand | Kc (reduction) |
|---|---|---|---|---|
| fwd (today) | `Y[M,N]` | `X[M,K]` row-major, normal | `Wᵀ` **materialized** `[K,N]` row-major, normal | K |
| **dgrad** | `dX[M,K]` | `dY[M,N]` row-major, normal | `W[N,K]` row-major, **normal** (no transpose!) | N |
| **wgrad** | `dW[N,K]` | `dYᵀ` — dY `[M,N]` read **transposed** | `X[M,K]` row-major, normal | M (tokens) |
| fwd_wt (optional) | `Y[M,N]` | `X[M,K]` row-major, normal | `W[N,K]` row-major read **transposed** | K |

dgrad with `(Mc, Nc, Kc) = (M, K, N)`: A is dY `[M, N]` = `[Mc, Kc]`
row-major — normal load; B is W `[N, K]` = `[Kc, Nc]` row-major — **normal
load, no materialized transpose**. The existing `hip::gemm_bf16` computes
it verbatim when called as `gemm_bf16(M, K, N, 1.0, dY, W, 0.0, dX)`.
The only real work is config entries and the ABI:

* `Kc = N` is a multiple of 16 but not always of 64 (e.g. QKV N=6288:
  6288 % 64 = 16) → the **defensive K bounds-check fires**. Correct
  today, but untuned; the config entry should size `BK`-tail handling
  for it (a K-padding variant is a follow-up only if the bench says it
  matters).
* `Nc = K` is a multiple of 64 (all K3 Ks are) → no N bounds-check.
* Training `M` is large (512–8192), so the `M <= 64` split-K route never
  fires — plain tiling, compute-bound regime (already contemplated by
  the warmup shapes).

**wgrad is the new kernel body**: an A-transposed loader (dY read as
`[N, M]` from row-major `[M, N]` storage — same transposed-global-read
pattern as the fwd_wt B load below, so the two loaders share a review),
B normal, and split-K over the token dim `M` reusing #156's fp32 partial
planes + fixed-order combine unchanged (determinism preserved: reruns
bit-identical — required by the trainer's parity tests and friendly to
DDP bucket allreduce).

### 3.2 Output dtypes (the deliberate deviation)

`dgrad` output is bf16 (it feeds the next layer like an activation).
`wgrad` and `bias_grad` outputs are **fp32**:

* wgrad accumulates across microbatches (grad accumulation). A bf16
  `β=1` accumulate-into-C would round once **per microbatch**; training
  keeps master grads in fp32 buckets and rounds once per optimizer step
  (vtrainer DESIGN §7). The kernel therefore writes fp32 directly —
  `alpha`/`beta` applied in the fixed-order combine, no bf16 store.
* bias_grad likewise writes fp32 (it lands in the same bucket).
* The oracles mirror this exactly (fp32 accumulate, fp32 store — no RNE
  round on the wgrad/bias path; the bf16-out contract of the serving
  family is untouched).

### 3.3 New C++ surface (additive; serving names untouched)

```cpp
// gemm_bf16.hpp — oracles (always compiled), mirroring gemm_bf16_cpu:
namespace vkernels::kernels {

// dX[M,K] = alpha * dY[M,N] @ W[N,K] + beta * dX   (bf16 in/out)
void gemm_bf16_dgrad_cpu(std::size_t M, std::size_t N, std::size_t K,
                         float alpha, const uint16_t* dY,
                         const uint16_t* W, float beta, uint16_t* dX);

// dW[N,K] (fp32 out) = alpha * dY^T @ X[M,K] + beta * dW
void gemm_bf16_wgrad_cpu(std::size_t M, std::size_t N, std::size_t K,
                         float alpha, const uint16_t* X,
                         const uint16_t* dY, float beta, float* dW);

// db[N] (fp32 out) = sum_m dY[m,n]
void gemm_bf16_bias_grad_cpu(std::size_t M, std::size_t N,
                             const uint16_t* dY, float* db);

// per-shape tiles for the training regime (M >= 512); wgrad also picks
// the split count S over the token dim M.
void gemm_bf16_dgrad_config_for(std::size_t M, std::size_t N,
                                std::size_t K, int* bm, int* bn,
                                int* bk, int* threads);
void gemm_bf16_wgrad_config_for(std::size_t M, std::size_t N,
                                std::size_t K, int* bm, int* bn,
                                int* bk, int* threads, int* splits);
}  // namespace vkernels::kernels

#if VKERNELS_HAS_HIP
namespace vkernels::kernels::hip {
// dims-swap of gemm_bf16 (see 3.1); no new kernel body.
void gemm_bf16_dgrad(std::size_t M, std::size_t N, std::size_t K,
                     float alpha, const uint16_t* dY, const uint16_t* W,
                     float beta, uint16_t* dX);
// A-transposed load + split-K over M + fp32 partials/combine/output.
void gemm_bf16_wgrad(std::size_t M, std::size_t N, std::size_t K,
                     float alpha, const uint16_t* X, const uint16_t* dY,
                     float beta, float* dW);
// deterministic two-stage column sum (block partials + fixed-order
// combine, same determinism discipline as the split-K family).
void gemm_bf16_bias_grad(std::size_t M, std::size_t N,
                         const uint16_t* dY, float* db);
// optional: Y[M,N] = alpha * X[M,K] @ W[N,K]^T + beta * Y — B-transposed
// load; kills the trainer's per-call W.T materialization. Same loader
// family as wgrad's A-transposed load.
void gemm_bf16_fwd_wt(std::size_t M, std::size_t N, std::size_t K,
                      float alpha, const uint16_t* X, const uint16_t* W,
                      float beta, uint16_t* Y);
}  // namespace vkernels::kernels::hip
#endif
```

### 3.4 New C ABI (the names are frozen — vtrainer probes them verbatim)

Host oracle wrappers (`capi.hpp`, mirroring `vk_gemm_bf16`):
`vk_gemm_bf16_dgrad`, `vk_gemm_bf16_wgrad`, `vk_gemm_bf16_bias_grad`,
`vk_gemm_bf16_dgrad_config`, `vk_gemm_bf16_wgrad_config`.

Device entries (`hip_capi.hpp`, issue #69 stream discipline — `int`
dims, opaque pointers, `stream` last, `int` return = the launch error,
never silently absorbed; **no alloc/free/sync**; explicit `stream`
resolves to the caller's current — under capture, capturing — stream):

```c
int vk_hip_gemm_bf16_fwd_stream(int M, int N, int K, float alpha,
    const void* X, const void* B /* [K,N] pre-transposed W */, float beta,
    void* Y, void* stream);
int vk_hip_gemm_bf16_fwd_wt_stream(int M, int N, int K, float alpha,
    const void* X, const void* W /* [N,K] direct */, float beta,
    void* Y, void* stream);
int vk_hip_gemm_bf16_dgrad_stream(int M, int N, int K, float alpha,
    const void* dY, const void* W, float beta, void* dX, void* stream);
int vk_hip_gemm_bf16_wgrad_stream(int M, int N, int K, float alpha,
    const void* X, const void* dY, float beta,
    void* dW /* fp32 [N,K] */, void* stream);
int vk_hip_gemm_bf16_bias_grad_stream(int M, int N, const void* dY,
    void* db /* fp32 [N] */, void* stream);
```

`vk_hip_gemm_bf16_fwd_stream` is the plain existing NN kernel behind the
stream ABI — the baseline forward the trainer uses when `fwd_wt` is
absent. Non-`_stream` legacy entries follow the usual pattern
(`vk_hip_gemm_bf16_fwd` etc., stream 0).

Autotuner hooks (`..._with_config`, taking the explicit tile + `S`) are
harness-only like `gemm_bf16_splitk_with_config` — not routed from the
plain entries.

### 3.5 Python surface

New `src/python/vkernels/hip_train.py`, the `hip_dsa_mhc.py` pattern
verbatim: lazy `load_libvkernels_hip()`, `_stream_ptr` resolving
torch's current stream, `_device_guard`, nonzero-rc → `RuntimeError`,
and `available()`-style probes. This is the module vtrainer's
`ops/_device.py` mirrors; keeping both sides in the house pattern means
the trainer's ctypes bridge and any direct `vkernels` user see the same
contract. (The trainer also probes the raw symbols directly, so it works
against a `libvkernels_hip.so` without the `vkernels` package — both
paths must stay in sync on the names in 3.4.)

## 4. Tests

1. **Oracle units** (`test_gemm_bf16_train_correct.cc`, host, CI-green
   without HIP): dgrad/wgrad/bias_grad vs a naive fp64 loop; edge shapes
   `M=1`, odd `N` (N bounds-check), `N % 64 != 0` (the defensive K-tail
   dgrad path), non-multiple-of-16 M tail; `beta=1` accumulate;
   determinism: two wgrad runs bit-identical (fixed-order combine).
2. **Device parity** (`test_gemm_bf16_train_correct.hip`, gfx942):
   each op vs its oracle — bf16 outputs `max_abs_rel < 2e-2` (house
   bf16 gate), fp32 outputs `max_abs_rel < 1e-5` (split-K reorder vs
   oracle order) and bit-identical across reruns; uneven split counts
   (`S` not dividing `M`).
3. **Trainer gradcheck parity** (vtrainer side, already written):
   `tests/test_linear.py` — fp64 finite-difference gradcheck through the
   full `autograd.Function` (reference path), bf16 fwd+bwd vs the fp32
   oracle path, dispatch fallback when the symbols are absent. On a
   gfx942 host with this issue landed, the device path runs the same
   suite (L2 gate).

## 5. Bench plan

`meta/benchmarks/bench_gemm_bf16_train.hip` (the `bench_gemm_bf16.hip`
autotuner extended to the training ops): latency, TFLOP/s vs the 1307
TFLOP/s bf16 MFMA roof, effective GB/s vs the 5.3 TB/s HBM roof, per the
`docs/kernels-reference.md` gap-to-SOL methodology. Shapes:

* nano-K3 projections (`d_model=1024`): `(N,K) ∈ {(3072,1024) QKV,
  (1024,3072) o-proj, (512,1024) + (1024,512) MoE ispp, (32768,1024)
  LM head}` at `M ∈ {512, 1024, 2048, 4096, 8192}` (the last is already
  a warmup shape in the serving table);
* one roofline anchor `(4096, 4096)` to bound achievable TFLOP/s;
* K3 serving shapes cross-check (regression guard: the dims-swap must
  not disturb existing tuning — same table entries, different call).

Expected classification: all `M ≥ 2048` shapes compute-bound; the gate
is **recorded and classified** (roofline table in
`docs/performance/gemm-bf16/`), escalating only if any `M ≥ 2048`
compute-bound shape lands `< 50%` of roof — that would indicate a tiling
problem worth its own issue, not a blocker for wiring the ABI.

## 6. Phasing & acceptance

| phase | content | gate |
|---|---|---|
| 0a | oracles + host ABI + oracle units | host CI green |
| 0b | `fwd`/`dgrad` device ABI over the existing kernel + config entries + device parity | test 2 green on gfx942 |
| 0c | `wgrad` kernel (A-transposed loader, split-K over M, fp32 out) | test 2 + determinism green |
| 0d | `bias_grad`; optional `fwd_wt` | test 2 green |
| 0e | bench + tuning table + roofline doc | §5 recorded |

## 7. Non-goals

* No fused optimizer epilogues (K5, later), no fp8/block-fp8 wgrad
  (training keeps bf16 weights), no fused bias-grad into wgrad.
* No autograd/`torch.autograd.Function` glue here — that is the
  vtrainer repo's side of the contract (already written).
* No serving-path changes: existing symbols, layouts, tuning tables and
  the `M <= 64` split-K route are untouched.
* Graph capture: nothing to do — the `_stream` ABI is capture-safe by
  construction (#69 discipline); no new host-side sync is introduced.
