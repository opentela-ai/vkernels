# issue #74 — VKERNELS_MXFP4_BF16 degenerate serving output: root cause, fix, gate

Beverin (gfx942/MI300A), 2026-09-17. Branch `issue-74-mxfp4-serving-fix`
(base `4c58c58`, commits `fb34572`, `1947499`).

## Root cause (serving integration, as suspected in the issue)

The K3 serving shim (`home/pylib/sitecustomize.py`, the issue-#78 block)
registers `VkernelFusedExperts` by hijacking
`vllm.model_executor.layers.fused_moe.oracle.mxfp4.backend_to_kernel_cls` for
`Mxfp4MoeBackend.AITER_MXFP4_BF16`. The K3 selector
(`select_deepseek_v4_mxfp4_moe_backend`) then selects the **AITER_MXFP4_BF16
enum** on gfx942 — and `Mxfp4MoEMethod._setup_kernel`
(`quantization/mxfp4.py:723`) calls
`convert_weight_to_mxfp4_moe_kernel_format` with that enum, which runs the
oracle's **AITER branch**: `aiter.ops.shuffle.shuffle_weight(is_guinterleave=
True, gate_up=True)` + `shuffle_scale` — the gfx950 CK shuffle layout.

`vk_hip_fused_moe_mxfp4` consumes the **raw `create_weights()` layout**
(w13 `[E, 2*ispp, hidden/2]` uint8 with gate rows `[0, ispp)` / up rows
`[ispp, 2*ispp)` — separated halves; w2 `[E, hidden, ispp/2]`; N-major ue8m0
scales; `float*` biases, `src/c/vkernels/capi/hip_capi.cpp:40`). It received
CK-shuffled weights *and* shuffled scales instead → the experts read garbage
(finite-wrong, `.dartampionship…`) or NaN (`!!!!…`, token 0), exactly the two
degenerate modes in the issue table, while `TRITON_UNFUSED` (whose oracle
conversion matches its own kernel) stayed coherent.

Note: no pass-through patch of the conversion ever existed in the deploy
stack (`deploy*/k3_patch.py`, `deploy*/sitecustomize.py`,
`home/pylib/*` — none reference the convert functions); the issue's
candidate #1 wording ("patched to pass through") described the AITER branch's
*lack of* `_swizzle_mxfp4`/`PrecisionConfig`, but that branch still applies
the aiter CK shuffle — which is the defect.

## Fix (repo)

`src/python/vkernels/vllm_experts.py`:

- `convert_weights_for_vkernel(...)` — the serving contract: weights/scales
  pass through **unchanged**; bf16 biases cast to the C ABI's fp32. Accepts
  both oracle call shapes (MoE + gpt-oss positional variants).
- `register_vkernel_backend_shim()` — the *coupled* registration the
  sitecustomize previously improvised by hand: (1)
  `backend_to_kernel_cls(AITER_MXFP4_BF16) → [VkernelFusedExperts]`, and
  (2) interception of both oracle `convert_*_to_mxfp4_moe_kernel_format`
  functions so the `AITER_MXFP4_BF16` enum routes to the vkernels
  pass-through conversion.

Deployment (`/capstor/scratch/cscs/xyao/kimi-k3-vllm-beverin/home/pylib/`):
`vkernels_experts.py` + `sitecustomize.py` patched to call
`install_vkernel_weight_conversion()` right after the experts-class hijack
(backups: `*.bak-issue74`; idempotent marker
`oracle._vkernels_convert_patched`).

## Gate (the correctness gate that never existed)

`tests/python/test_mxfp4_serving_parity.py`:

- `ConvertWeightsForVkernelTest` (CPU): pass-through + fp32-bias contract.
- `VkernelBackendShimTest` (vLLM host): the enum really resolves to
  `VkernelFusedExperts`, and the oracle conversion returns raw uint8 weights
  (not a CK view) with fp32 biases.
- `Mxfp4ServingParityTest` (gfx942 HIP): `vk_hip_fused_moe_mxfp4` vs a naive
  bf16 dequant reference (single bf16 rounding, fp32 accumulation, K3 SiTU;
  cancellation-aware tolerance `e > 0.05|ref| + 0.03·rms(ref)`, mirroring
  `test_moe_fused_bigshape_correct.hip`) at decode (M=8, bs16) and prefill
  (M=64, bs64) shapes, plus the **real K3 serving shape**
  (h=7168, ispp=3072, E=112, topk=8 — bracketed but never covered by the
  in-tree sweep), env-gated `VK74_K3_SHAPE=1`.
- Negative control: pair-interleaved gate/up rows (the AITER-input
  convention) must FAIL parity — proving the gate can catch a layout bug.

## Evidence (Slurm `vk-i74-parity`, jobs 640417 / 640420, nid002714)

```
test_biases_cast_to_c_abi_fp32 ... ok          test_shim_installed ... ok
test_weights_and_scales_...     ... ok          test_aiter_enum_resolves_to_vkernel_experts ... ok
test_negative_control_...       ... ok          test_conversion_is_raw_pass_through ... ok
test_parity_decode_shape        ... ok          (raw uint8, fp32 biases)
test_parity_prefill_shape       ... ok
test_parity_k3_serving_shape    ... ok   (h=7168, ispp=3072, E=112, topk=8; ~115 s)
→ 9/9 OK (job 640420); 6/6 OK + K3 shape OK (job 640417)
```

Harness: `/capstor/scratch/cscs/xyao/tmp/vk-i74/`
(`vk_i74_parity.sbatch`, repo package staged under `vkernels/`).

## Remaining risk

- The end-to-end K3 serve (`capital -> Paris` with real weights,
  `VKERNELS_MXFP4_BF16` selected) has not been re-run yet — the fix changes
  only the weight conversion, which the parity gate now covers
  serving-independently at the exact serving shape; the next scheduled serve
  should confirm with `gen_correctness_*.json`.
- `test_parity_k3_serving_shape` is env-gated (≈4 GB weights, ~2 min) — run
  it on gfx942 hosts when touching the MoE serving path.
