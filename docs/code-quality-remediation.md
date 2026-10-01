# Runtime and compiler contracts

This change addresses the twelve repository audit findings and three follow-up
lifetime findings. The implementation boundaries are:

| Audit finding | Implementation |
| --- | --- |
| GPU scratch ownership | `execution_workspace.hpp`: stream-ordered invocation buffers and explicit `PreparedWorkspace`/`WorkspaceScope`; DSA, MLA, and GEMM no longer share global writable scratch or abandon allocations. |
| Incorrect/incomplete device capabilities | Real CUDA sum/max; ragged GEMM tile loads; mandatory CUDA correctness tests; unfinished OFI plugin is not built or installed. |
| Rust communication safety | Raw-address gather/batch-copy entry points are `unsafe`; borrowed-slice `gather` completes synchronously. |
| Host task failures | Streams drain callbacks and release captures before reporting the first error; counter reads synchronize; overlap drains both queues on every exit. |
| ctypes validation | Shared device/stream/lifetime boundary; kpool output dtype/contiguity/page geometry checks; conversion work runs on the launch stream. |
| Persistent residency | Compile before execution; query compiled CUDA kernel occupancy; reject grids above the conservative one-block-per-SM bound. |
| Compiler duplication | One operator registry owns supported kinds and lowering/reference/codegen dispatch. Model adapters select capture. Qwen3 consumes the compiler schedule and packed workspace; implementation groups are separate modules. |
| Backend discovery | Explicit extension/library selection, packaged API catalog, and `vkl doctor`; no modification-time selection or silently ignored explicit-path failures. |
| Packaging | PEP 517/scikit-build-core staging, interpreter-aware CMake build, native wheel tags and RECORD; GPU wheels install serving libraries. |
| Tuning identity/persistence | Current execution-device fingerprints; native architecture mismatches are misses; configuration version 2; locked read/merge/replace with unique temporary files. |
| Test reproducibility | Opt-in integration/checkpoint suites, no synthetic kvaas modules or personal paths; Rust and installed-wheel CI; gated real HIP runner; device placeholders disabled on host. |
| Execution cost | `prepare_device` shares compiler planning; `run_device` accepts prepared device inputs; graph replay advances barrier state on-device; separate latency benchmark. |
| Python stream destruction | Custom holder releases the GIL while joining; subprocess timeout regressions. |
| Partial overlap submission | Queued tasks own shared callbacks; allocation-failure injection verifies draining and reclamation. |
| Explicit-stream temporaries | Stream dependency, device guard, allocator lifetime recording, delayed-stream/churn and graph regressions. |

## Selecting a build

Installed packages use their installed extension and serving libraries. For a
checkout, select a build explicitly:

```sh
export VKERNELS_EXTENSION="$PWD/build/python/python/vkernels/_core.cpython-312-aarch64-linux-gnu.so"
export VKERNELS_BUILD_DIR="$PWD/build/cuda"
export VKERNELS_BACKEND=compiled
vkl doctor
```

Use the extension filename produced by your interpreter/build. `VKERNELS_LIB`
can select an exact serving-library path. `VKERNELS_BACKEND=fallback` explicitly
selects the NumPy reference. An invalid explicit native path is an error.

The API catalog describes implemented entry points. Build metadata, library
loadability, and device validation are separate facts. A successful load does
not attest numerical validation on that device. `doctor` leaves that status
unknown; run the device contract tests on the target hardware.

Build wheels with `python -m build --wheel` (host) or
`meta/scripts/build_wheel.sh --cuda/--hip`. They retain the native `linux_*` tag.
Use `auditwheel show` and a compatible manylinux build/repair environment when
publishing manylinux wheels; naming a wheel does not establish compatibility.

## Execution ownership

Each native `Scratch<T>` allocates and releases in stream order. Runtimes that
do not support stream-ordered allocation must use a prepared workspace. Prepare
buffers outside capture, enter a `WorkspaceScope` for each invocation, and keep
the workspace alive through execution and the lifetime of captured graphs.
Buffers are consumed in allocation order; undersized/missing buffers and
mismatched device/stream scopes fail explicitly. Replay graphs on the workspace's
stream. Use distinct workspaces for concurrent streams/devices.

Python kpool inputs must be ready on the caller's current stream. Explicit
launch streams wait for that stream; consumers on other streams must wait for
the launch stream. In-place outputs/tails never undergo implicit conversion.
The bf16 output accepts `[pages, slots, head_dim]` or the legacy flattened shape.

Streams consume their first task failure in `wait()` after draining all work.
A subsequent wait succeeds unless new work fails. The legacy void C wait reports
failure through `vk_last_error_code`/`vk_last_error`.

Rust callers of raw-address copies must prove source validity and exclusive
destination access through completion, including submission-error paths.
`gather` provides the ordinary borrowed-buffer synchronous interface.

K3's native naive recurrence takes `g[B,H,S,D]` and predicts after decay.
The standard chunked recurrence takes `g[B,H,S]` and predicts before decay.
Their Python/Rust tests use separate numerical oracles.

## Compiler and performance work

Add an operator contract in `compiler/contracts.py`, its capture and region/tile
rules, and its mathematical reference/device implementation. Registration sets
are derived from the contract rather than maintained separately. Dense,
attention, and recurrent implementations have distinct modules.

`executable.prepare_device()` currently supports dense Qwen3. It returns a
runner using the exact compiled schedule and workspace; other model adapters
remain reference-only until their backend is implemented. `run_device` requires
correctly shaped device-resident token/position inputs with valid values. The
runner owns one stream and reusable output storage. `run` retains checked host
inputs for ordinary use. Barrier state remains on-device across graph replay.

Measure complete costs with:

```sh
python meta/benchmarks/bench_decode_costs.py --output /tmp/decode-costs.json
```

The benchmark reports preparation/compilation/allocation, checked host decode,
prepared decode, GPU-event latency, graph replay, and workspace size separately.
It validates output after timing and reports dispersion/stability. It does not
claim throughput or speedup; use the existing `bench_roofline.py` calibration
and an identical before/after harness for those claims.

Native tuning version 2 intentionally misses old records. Re-run tuning after
upgrading. Writers retain `.lock` files because unlinking a lock while another
process holds it can create two independent locks for the same store.

External oracle tests require `--run-integration` and installed dependencies
(or explicit `FLOE_ROOT`). Checkpoint tests require `--run-checkpoint` and
`VKERNELS_TEST_CHECKPOINT`. HIP CI is enabled with `VKERNELS_HIP_CI=true` only
when a real runner exists; an unavailable/skipped HIP job is not validation.
