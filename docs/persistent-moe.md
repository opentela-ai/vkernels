# Prepared routed-expert megakernel

`vkernels.torch_ops.persistent_moe.PersistentMoE` executes a bounded,
single-device block-FP8 routed MLP in one persistent CUDA kernel. This is an
experimental compute region. It does not replace the distributed GLM runner,
attention, shared experts, or collectives.

```python
from vkernels.torch_ops.persistent_moe import PersistentMoE

# Prepare on the stream that will execute this region, before graph capture.
region = PersistentMoE(
    x, gate_up_fp8, gate_up_scales, down_fp8, down_scales,
    local_expert_ids, route_weights,
    skip_zero_weights=True,
    round_products_to_bf16=True,
)
y = region.run(x, local_expert_ids, route_weights)
# Consume y before the next run: it is borrowed workspace, not a new allocation.
```

The owner must retain the region, weights, inputs and output consumers through
physical completion. Submissions and graph replays are serialized on the
preparation stream. Create separate instances for concurrent regions. CUDA
graph capture must explicitly use that stream; graph replay bypasses Python
checks, so its owner must enforce the same stream and lifetime rules.

The region has five phases and four GPU-scope acquire/release grid barriers:

1. Stable expert permutation, counts, segments, tile map and inverse
   permutation; independent per-token input quantization.
2. Grouped gate/up FP8 GEMM with per-128 block scaling, BF16 output.
3. Clamped FP32 SwiGLU, virtual BF16 activation store, group FP8 quantization.
4. Grouped down FP8 GEMM, BF16 output.
5. Original-route-order FP32 reduction, with optional BF16 product rounding.

Inactive routes never read uninitialized GEMM tails. Every invocation rewrites
all metadata and required activation payloads. Workspace shapes and capacity
bounds are static. Activation quantization visits only live rows, four groups
per task, using a device-side loop bound. Changing routes causes no host scalar read,
data-dependent allocation, permutation sort launch or barrier-reset launch.
The monotonic barrier counter/base advance on device, including graph replay.
Cross-phase workspace loads bypass L1 to observe other workers' writes.

Preparation compiles without launching, queries actual compiled occupancy via
the existing compiler residency checker, and requests cooperative CUDA grid
admission (`launch_cooperative_grid=True`). The default is at most one block
per SM. A compute window may request `blocks_per_sm=2`; preparation rejects
this if the compiled kernel cannot actually keep two blocks resident.
Cooperative admission prevents a partially admitted grid from spinning while
its remaining blocks wait for resources occupied by another stream. It does
not remove dependency cycles created by the caller. See the [CUDA execution
control contract](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EXEC.html).
Never run a spin-barrier region concurrently with a kernel that
occupies its required residency and waits for that region. Distributed
integration must establish an independent compute window and keep failure
consensus and collectives outside it. An occupancy check alone does not prove
the absence of dependency cycles with other streams or devices.

Supported inputs are contiguous, same-device CUDA tensors: BF16 `x[T,H]`,
E4M3FN expert stacks `[E,2I,H]` and `[E,H,I]`, FP32 block scales, int64
local expert ids `[T,K]` and FP32 route weights. `H` and `I` are multiples of
128; `1 <= T*K <= 4096`, `1 <= E <= 1024`, `1 <= K <= 32`. GEMM tiles are fixed
at 64 by 128 to retain the qualified arithmetic. Smaller row tiles changed
outputs on GH200 and were rejected. Inputs and intermediate activations must
remain finite. Local
expert ids must be in range; negative sentinels require skip-zero mode. No
device value scan validates these caller preconditions. ROCm/FNUZ, arbitrary
strides, training, unbounded routes and inter-device synchronization are not
implemented.

Numerical boundaries retain block order, input group amax floor `1e-4`, scale
floor `1e-12`, FP32 scalar-reciprocal rounding, tensor division rounding,
BF16 gate/up and activation stores, down store and fixed route reduction.
The fused activation uses libdevice exponential and round-to-nearest division
and multiplication. Tests compare complete region outputs and intermediate
FP8 bytes/scales against the existing native grouped operator; target-device
qualification remains required. This does not repair an existing model-level
NLL difference against another engine.

`tests/python/test_persistent_moe.py` checks numerical equality, metadata,
multiple resident/idle workers, changed-input CUDA graph replay, all-dropped
replay, stream ownership, residency bounds and exactly one compute launch.
The workspace preparation and its initial counter fills are excluded from
the one-launch claim. Performance must be compared to a graph baseline:
fewer launches can lose to lower tensor-core occupancy or barrier imbalance.


Floe's experimental `fp8_moe_megakernel` adapter admits eager prefill only and
shares one active plus at most one retiring workspace across serialized routed
layers of a model. Passing `expert_weights=(gate_up, gate_scales, down, down_scales)`
to `run` uses the current layer's tensors, with geometry/alignment validation
and no weight-derived cache. Callers retain runtime weights through completion.
A CUDA completion event protects replaced workspaces and prior output consumers.
Shape changes decline while a retired workspace remains incomplete. Graph
capture and decode retain their previous path. Shared experts, collectives, failure
consensus and kvaas ownership remain outside this region. Full-model evidence
is recorded in `optimization/2026-10-11-moe-megakernel-floe-shared-clariden/report.md`
in the parent workspace. Earlier isolated timing results predate cooperative
admission; qualify this launch mode separately before attributing those gains
to serving.
