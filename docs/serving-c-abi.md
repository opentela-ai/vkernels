# Serving C ABI — library map, entry points, stream/capture safety

The serving stacks (the cookbook vLLM shim, `vkernels_attn.py`, and its
SGLang counterpart) consume vkernels exclusively through two extern "C"
ABIs built from `src/c/`. This page is the reference for what each shared
library contains, which entry point to call from which execution mode, and
which calls are safe under CUDA/HIP graph capture (issue #45).

## The two shared libraries

| Library | CMake target | Built | Contents |
|---|---|---|---|
| `libvkernels_hip.so` | `vkernels_hip` (src/c/CMakeLists.txt, HIP builds) | when ROCm is found | `vk_hip_*` device entry points (thin wrappers over the C++ `vkernels::kernels::hip` kernels) |
| `libvkernels.so` | `vkernels_hostc` (HIP builds, commit `647d0d3`) | when ROCm is found | host CPU oracles `vk_mla_fwd`, `vk_kda_naive_delta_rule_fwd` etc. for in-process cross-validation (`VKERNELS_MLA_VALIDATE=1`, `VKERNELS_KDA_VALIDATE=1`) |

Both share the headers:

- `src/c/vkernels/capi/hip_capi.hpp` — device ABI (`vk_hip_*`, incl. the
  issue-#69 stream-safe variants `vk_hip_dsa_sparse_fwd_stream`(+split),
  `vk_hip_mhc_pre_gemm_sqrsum_stream` / `_post_stream`, and the KDA
  with-scratch `_stream` entries)
- `src/c/vkernels/capi/capi.hpp` — host oracle ABI (`vk_*`, incl. the
  cross-validation entries `vk_mla_fwd` and `vk_kda_naive_delta_rule_fwd`)

The serving-side resolver globs `libvkernels.so` in the HIP build tree
(`build/hip/src/c/`), which is why `vkernels_hostc` sets
`OUTPUT_NAME vkernels`.

## Stream / capture-safety matrix (issues #69, #45)

"Stream-safe" means: every kernel is enqueued on a caller-provided stream
(passed as `void*`, e.g. `torch.cuda.current_stream().cuda_stream`) and the
call performs **no allocation, free, or synchronisation** — required
invariants for `hipStreamBeginCapture` / `cudaStreamBeginCapture`. Error
status is returned as `(int)hipGetLastError()`, never swallowed.

| Entry point | Status | Capture-safe? | Notes |
|---|---|---|---|
| `vk_hip_dsa_sparse_fwd_stream` (+ `_split_stream`) | `int`, stream arg | yes (single-launch either way) | DSA sparse-MLA, issue #69 |
| `vk_hip_mhc_pre_gemm_sqrsum_stream` / `vk_hip_mhc_post_stream` | `int`, stream arg | yes | MHC pre/post, issue #69 |
| `vk_hip_kda_delta_rule_fwd_with_scratch_stream` / `..._chunked_with_scratch_stream` | `int`, stream arg + caller-owned scratch | yes (scratch sized before capture) | KDA, issue #69 |
| `vk_hip_mla_fwd_stream` | `int`, stream arg | yes (see workspace rule below) | MLA, added for issue #45 |
| `vk_hip_mla_fwd` | `void` (legacy) | **no guarantee** | default stream, silent errors; kept for parity with older shim revisions |
| `vk_hip_dsa_sparse_fwd`, `vk_hip_mhc_attn`, `vk_hip_kda_delta_rule_fwd` | `int` | not audited for capture | eager-only entries |

### MLA split-K workspace rule

`mla_fwd`'s split-K decode path (issue #82) uses a process-global, grow-only
workspace. Resizing it calls `hipMalloc`/`hipFree` — `hipFree`
device-synchronises, which is **illegal mid-capture**. Therefore:

- While the caller stream is capturing, the entry point never touches the
  workspace. If it is already big enough (steady state: the decode shape was
  run eagerly at least once before capture), the split path is captured as
  usual; otherwise it falls back to the capture-safe non-split kernel.
- **Serving pattern:** run one eager decode step per shape before
  `BeginCapture` (vLLM does this anyway during profile/warmup runs) to keep
  the split path inside the graph.

## MLA ABI (Kimi-K3 / DeepSeek absorbed form)

Semantics absorbed from vLLM's TRITON_MLA / AITER: `q` carries its own RoPE
slice. Layouts (identical in the CPU oracle and the HIP kernel; index form
`((b*H + h)*S_q + row)*D + d`):

```
q     [B, H, S_q, kv_lora_rank + qk_rope_head_dim]  float32
k_c   [B, S_kv, kv_lora_rank]                       float32
k_pe  [B, S_kv, qk_rope_head_dim]                   float32
v_c   [B, S_kv, kv_lora_rank]                       float32
out   [B, H, S_q, kv_lora_rank]                     float32
q_start / kv_start   row offsets into the (B, S) batch
scale                1/sqrt(kv_lora_rank + qk_rope_head_dim)
```

`vk_hip_mla_fwd_stream(B, H, S_q, S_kv, q_start, kv_start, kv_lora_rank,
qk_rope_head_dim, scale, q, k_c, k_pe, v_c, out, stream)` returns
`(int)hipGetLastError()` — `0` on success. `stream == nullptr` means legacy
stream 0.

## Cluster validation (issues #42/#45)

On a HIP build, `libvkernels.so` (target `vkernels_hostc`) exposes the CPU
oracles, so a serving validator can cross-check device output in-process:

```bash
cmake -S . -B build/hip -DVKERNELS_BUILD_HIP=ON
cmake --build build/hip -j
# shim-side: VKERNELS_MLA_VALIDATE=1 / VKERNELS_KDA_VALIDATE=1 globs
# build/hip/src/c/libvkernels.so and compares against vk_hip_* output.
```

On-device regression coverage:

- `meta/benchmarks/test_mla_correct.hip` — MLA numerics vs the CPU oracle,
  incl. split-K decode shapes (issue #82).
- `meta/benchmarks/test_mla_capture.hip` — MLA stream/graph-capture safety:
  cold-workspace capture (fallback path), warm-workspace capture (split path
  inside the graph), re-feed replay, and legacy/stream parity.
