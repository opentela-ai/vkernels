# Device-native mHC projection

`vkernels.torch_ops.mhc_projection` is an optional Torch/Triton inference
operator. The existing NumPy/C++ bindings are unchanged, and importing
`vkernels` or `vkernels.torch_ops` does not import Torch or Triton.

```python
from vkernels.torch_ops import mhc_projection

# Contiguous BF16 GPU tensors: normalized_x [...,16384], weight [24,16384].
# Exactly one or two activation rows; normalization remains the caller's job.
y = mhc_projection(normalized_x, weight)
```

The implementation uses 32 deterministic FP32 split-K partials followed by a
fixed reduction and BF16 output conversion. It has no atomics, modifies neither
input, and does not implement backward. Reduction order differs from BLAS.
`mhc_projection_reference` supplies a Torch FP32-accumulating CPU/GPU oracle.

Call eagerly on every device and token count before graph capture. Triton
autotunes rows/program (1, 2, 4) and warps (4, 8); split-K is currently fixed.
Cold capture raises an error. Warm capture allocates scratch/output from the
graph memory pool. Tuning choices are cached in process, not persisted as a
portable deployment artifact. A new process retunes; do not change hardware or
software underneath a running process. The metadata helper records selected
configurations for benchmark reports.

## Reproduce

Use an environment that already provides compatible Torch and Triton:

```bash
PYTHONPATH=src/python python -m pytest tests/python/test_mhc_projection.py -q
PYTHONPATH=src/python python meta/benchmarks/bench_mhc_projection.py --output /tmp/mhc.json
```

On a **beverin `mi300`** node the whole suite (pytest + this benchmark + the
roofline and counter references it relies on) is driven by a single
self-bootstrapping script that also papers over the missing `python3.11-devel`
package — see [`docs/torch-ops-mi300.md`](torch-ops-mi300.md):

```bash
VK63_SECTIONS='2. torch_ops* 3. BENCH mHC*' sbatch meta/scripts/run_issue63_torchops_mi300.sh
```

The benchmark separately measures default BLAS, PyTorch TunableOp-selected
BLAS, and Triton. All tuning/warmup precedes graph capture. Each graph rotates
through 90 independent weights (~68 MiB), with 15 timing samples. This is a
synthetic projection benchmark, not a whole-model throughput measurement.
TunableOp writes a version-validated CSV next to the JSON. The harness does
not install dependencies or enable TunableOp in serving.

## MI300A evidence, 2026-09-08

Beverin job 628593; Torch 2.9.0a0+git7bcbafe, ROCm 7.0.51831-a3e329ad8,
Triton 3.4.0. Captured outputs of every variant were validated before timing.
Median graph GPU latency per projection:

| Activation rows | Default BLAS | Tuned BLAS | Triton |
| --- | ---: | ---: | ---: |
| 1 | 146.166 us | 6.647 us | 3.936 us |
| 2 | 146.018 us | 9.288 us | 3.946 us |

The earlier job 628588 measured 146.138 / 11.542 / 4.223 us for one row:
TunableOp solution selection and timings vary between runs, so these are
observations, not universally optimal configurations. Twelve GPU tests
passed, including floe integration, graph replay, cancellation, cold-capture
guard, and non-current-device restoration. One-row synthetic outputs matched
BLAS exactly; two-row maximum relative L2 error was 2.12e-6. These measurements do
not establish general model-quality equivalence. Floe's separate full-model
A/B harness validates logits/NLL/greedy outputs before timing.
