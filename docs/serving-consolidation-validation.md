# Serving consolidation validation

This integration retains qualified serving changes and the existing numerical
assertions. It does not establish new end-to-end serving performance.

## Pre-existing GB10 numerical failures

On 2026-10-06, both tests below failed on the clean, detached pre-consolidation
`main` commit `223a762dbfbc162a60dbda59e47f415c0496200a` and on the
consolidation branch. The baseline reproduction used the same GB10 GPU and
Torch/Triton environment. No test tolerance, skip, or expected-failure marker
was added to hide these failures.

- `test_glm_moe_grouped.py::test_fnuz_storage_bit_exact_vs_ladder[shape2]`:
  the original grouped decode differs from the original ladder before FNUZ
  conversion; the exact-equality assertion fails for shape
  `(4, 8, 4096, 512, 4096)` with seed 204.
- `test_sgl_fused_moe.py::test_sgl_fused_moe_matches_eager_serving_shape[8]`:
  the quant-aware oracle gate fails for 58 of 32,768 elements, with maximum
  absolute difference `0.035888671875` against tolerance `0.02`. This kernel
  remains opt-in. This integration does not promote its numerical policy.

Reproduce from the baseline checkout with CUDA-capable Torch and Triton:

```sh
PYTHONPATH=src/python python -m pytest -q \
  'tests/python/test_glm_moe_grouped.py::test_fnuz_storage_bit_exact_vs_ladder[shape2]' \
  'tests/python/test_sgl_fused_moe.py::test_sgl_fused_moe_matches_eager_serving_shape[8]'
```

The [integration CUDA run](https://github.com/opentela-ai/vkernels/actions/runs/37491734019)
also exposed subprocess import failures. CUDA CI now exports the checkout's
Python source path so subprocesses resolve the tested package. DeepGEMM now
reports unsupported device architecture before probing its optional import.
Those fixes change environment discovery, not kernel arithmetic.

Host coverage, installed-wheel parity, and Rust CI passed on integration commit
`646ce2fe7c8f59cbedb593dd7d8612ac86842e9b`. The expanded focused serving
kernel suite passed 68 tests with two hardware/optional-backend skips.
The full CUDA suite retains the two failures above and must not be described
as completely passing.
