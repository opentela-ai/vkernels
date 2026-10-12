# Five-launch FP8 decode MoE

`vkernels.torch_ops.sgl_moe_decode_glue.sgl_fused_moe_decode_glue` reduces the
vendored weighted-down SGLang MoE wrapper's ten CUDA launches to five when
zero-route skipping is enabled:

1. Stable expert alignment and input FP8 quantization.
2. Unchanged vendored gate/up GEMM.
3. BF16 SwiGLU, activation FP8 quantization, and down-buffer initialization.
4. Unchanged vendored weighted-down GEMM.
5. Unchanged torch BF16 reduction over routes.

The two fusions use independent CTAs within each launch. Dependencies between
phases use ordinary kernel boundaries. This operator has no grid spin barrier
or cooperative residency requirement. It is an intermediate implementation
toward a decode megakernel, not a one-launch operator.

Inputs must be contiguous, same-device CUDA tensors, without gradients: BF16
`x[T,H]`, E4M3FN weights `[E,2I,H]` and `[E,H,I]`, FP32 block-128 scales,
int32/int64 expert IDs `[T,K]`, and FP32 route weights `[T,K]`. Tokens are
bounded to 1–8 and route slots to 64. H and I are multiples of 128. Caller
preconditions include finite inputs/intermediates and valid local expert IDs,
including IDs in zero-weight slots; no host scan checks GPU values. ROCm/FNUZ
and training decline with `OpNotEligible`.

Precision follows `sgl_moe.sgl_fused_moe(route_weight_in_down=True)`:
input and activation quantization use amax floor `1e-10`, FP32 division by
448 and reciprocal multiplication. Clamped gate/up, SiLU and activation
products retain each BF16 rounding boundary. Route weighting remains before
the down GEMM's BF16 store. Final reduction retains torch's implementation.
This differs from the native grouped-prefill quantizer; the prefill persistent
region cannot be substituted without numerical qualification.

Every invocation allocates only scratch and output; no model weights are
converted or cached. Warm the operator eagerly before graph capture. Captured
allocations use the existing CUDA graph pool, and replay rewrites all live
metadata, quantized activations and required down zeros. Skipped routes never
read uninitialized gate outputs. All-zero and revived routing are supported
without a host scalar read or data-dependent allocation. The caller retains
inputs, weights, outputs and the graph through physical completion.

Floe's default-off `sgl_moe_decode_glue` selects this path only alongside
`sgl_fused_moe` and `sgl_moe_weighted_down`. Unsupported contracts retain the
incumbent path. Shared experts, local EP remapping, collectives, failure
consensus, KV ownership and publication remain outside the operator.

The parent workspace records component evidence at
`optimization/2026-10-12-decode-moe-glue/report.md` and full-model qualification
at `optimization/2026-10-12-decode-moe-glue-floe-clariden/report.md`.

An additional default-off `tuned_gemm` argument independently selects gate/up
and down tiles. Its measured scope is GH200 SM90, T1/T4, E72/H4096/I2048,
top8, zero-route skipping, and the exact donor default configuration. Gate/up
uses N64/groupM4/stages3; down uses N64/groupM16/stages4. Both retain M16/K128
and four warps. Custom donor configurations, other accelerators and other
shapes keep the original configurations. Configurations are copied rather
than mutating the cached donor policy.

Floe exposes this through default-off `sgl_moe_decode_tuned`, which requires
`sgl_moe_decode_glue` and namespaces graph configuration. It adds no launches
or persistent workspace. The component screen rejected N256 on intermediate
bitwise parity and excluded B2/B8 from the narrower-tile policy after observed
regressions. A separate bounded-worker persistent GEMM prototype also regressed
denser routing and was removed from product code. Component evidence is in
`optimization/2026-10-12-decode-moe-tuned/report.md`; the full-model A/B is in
`optimization/2026-10-12-decode-moe-gemm-floe-clariden/report.md`.
