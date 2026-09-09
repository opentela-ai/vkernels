# Device-native GLM Q/K/V projection

```python
from vkernels.torch_ops.qkv_projection import qkv_projection

y = qkv_projection(x, q_weight, k_weight, v_weight)
```

Inputs are contiguous BF16 tensors on one GPU: `x [...,4096]` with one or two
activation rows, and three separate `[8192,4096]` weights. Output is contiguous
BF16 `[...,24576]`, ordered Q, K, V. No weight packing, replication, activation
normalization, or intermediate projection outputs are required. The operator
is inference-only and has no backward implementation.

The Triton kernel selects one weight pointer per workgroup, reduces BF16
products in FP32, and rounds each final output to BF16. Reduction order is not
guaranteed to match BLAS. `qkv_projection_reference` provides a CPU/GPU Torch
FP32-accumulating oracle. Torch and Triton imports remain lazy and optional.

Warm up eagerly for every device and row count before graph capture. The
kernel autotunes eight configurations: rows/workgroup 1, 2, 4, 8 and warps 4,
8. Cold capture raises an error. Tuning is cached in process by device and row
count, not persisted as a portable configuration. Retune after stack/hardware
changes. `qkv_projection_tuning_metadata()` records the choices for reports.

## Reproduce

```bash
PYTHONPATH=src/python python -m pytest tests/python/test_qkv_projection.py -q
PYTHONPATH=src/python python meta/benchmarks/bench_qkv_projection.py --output /tmp/qkv.json
```

On a **beverin `mi300`** node the pytest + this benchmark + the TunableOp
preflight (#67) + roofline/counter references are driven by one
self-bootstrapping script that also papers over the missing
`python3.11-devel` package — see [`docs/torch-ops-mi300.md`](torch-ops-mi300.md):

```bash
VK63_SECTIONS='2. torch_ops* 4. BENCH QKV* 5. CHECK frozen*' sbatch meta/scripts/run_issue63_torchops_mi300.sh
```

The benchmark compares three separate BLAS projections plus concatenation,
the same operations using PyTorch TunableOp, and fused Triton. Eight distinct
weight triplets (~1.5 GiB) rotate through each graph. All tuning precedes
capture; captured outputs are validated before 15 GPU timing samples.
Variant order is fixed, so small differences need paired end-to-end follow-up.
The shared graph timing helper lives in `bench_mhc_projection.py`.

## MI300A measurement, 2026-09-08

Beverin job 628602; Torch 2.9.0a0+git7bcbafe, HIP 7.0.51831-a3e329ad8,
Triton 3.4.0. Times include all three projections and baseline concatenation:

| Activation rows | Default BLAS | Tuned BLAS | Fused Triton |
| --- | ---: | ---: | ---: |
| 1 | 517.538 us | 75.976 us | 70.106 us |
| 2 | 500.408 us | 76.966 us | 119.076 us |

Triton is fastest here for one-row decode; tuned BLAS wins at two rows. Most
of the one-row gain comes from replacing the default BLAS algorithm, with an
additional ~8% microbenchmark improvement from fusion. The one-row TunableOp
choice was `Gemm_Hipblaslt_208015`; Triton chose one output row/workgroup and
four warps. These are observed choices, not portable optimal settings.

Twelve GPU tests passed, including floe integration, graph replay, read-only
inputs, projection order, cancellation, cold-capture refusal, and restoration
of a non-current device. Maximum synthetic relative L2 error against default
BLAS was 8.72e-5, maximum absolute error 0.0078125. This does not establish
broad real-model quality equivalence.
