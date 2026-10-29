# GLM prefix WY restoration candidate

This branch adds the explicit `kda_prefix_state_wy` operation. It changes no
default dispatch and leaves sequential restoration unchanged. The caller must
select it only when its target uses the corresponding 32-token WY pipeline.
Prefix-one decode remains a separate numerical contract.

Full-serving prefix candidate 694534 failed its state shadows and activated
no ranks. Trained eight-layer diagnostic 694973 retained four rank fixtures:
sequential restoration violates the existing state tolerance in 41 values.
FP64 sequential replay exhibits the same discrepancy. This is insufficient
numerical equivalence to the serving WY implementation, rather than evidence
that more sequential precision alone fixes the mismatch.

Prototype 694974 on nid002668 reuses that serving pipeline with masked
unaccepted tokens. All four saved fixtures meet unchanged atol1e-5/rtol1e-4;
the maximum difference is 2.98e-8. Recomputing the original short inputs is
bitwise exact. Changing GPU prefix lengths across capture/replay also passes.
This is prototype evidence; the exact new module still needs its AMD check.

The explicit operation admits contiguous FP32 H16/K128/V128 blocks with
2–4 tokens. Lengths clip to the block; zero retains signed-zero bytes of the
initial state. It masks future contributions, pads to 32, and calls the
existing WY pipeline. It computes unused query outputs; no new device kernel
or performance claim is introduced. Model integration and full serving
remain unqualified, including the separate convolution/DSA discrepancy.

Local NVIDIA GB10 validation: 24 tests pass, including six new lazy-import,
CPU-decline and changing-input GPU replay cases. The new cases compare
against independent per-row truncated WY forwards at the same strict state
tolerance. The existing 18 sequential/convolution tests remain passing.
Use the vkernels virtual environment: the floe environment has an editable
import redirector that binds the parent checkout before path resolution.
