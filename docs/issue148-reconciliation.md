# Issue 148: reconciled experiment, pending target qualification

Source: `fix/issue-148@2268cf0`; base: `main@48b8517`.

This draft recovers the distinct native FNUZ-conversion and MFMA kernels,
CPU oracles, Triton fused activation quantization/decode GEMM, and diagnostic
benchmarks. Existing main tuning-store lookup, registry selection, route-aware
zero skipping, geometry, and benchmark `--persist` parity gate are retained.
The new operators are explicit experiments; serving dispatch does not select them.

Review corrections:

- Allow either optional MFMA diagnostic output to be omitted independently.
- Fix the CPU quantizer's absolute-value search across both signs. An even
  negative encoding could overwrite the matching positive encoding. Search
  positive magnitudes, apply the sign once, canonicalize underflow to unsigned
  zero, and handle NaN explicitly. Exhaustive finite-code tests independently
  check decoding and encoding, plus positive/negative ties and saturation.
- Require contiguous inference operands on the same device, the documented
  dtypes/scale grid, positive bounded shapes, valid split/tile configurations,
  and ROCm gfx942 for the new Python API. Launch on the operand device.
- Keep the native-conversion switch opt-in and preserve current registry and
  zero-route launch handling.
- Check every timing candidate and the selected configuration against the
  quantized oracle; fail if the native-conversion A/B changes bits. A failing
  native correctness executable now stops the job script. Remove its automatic
  Python/header download; callers provide a prepared environment via `PY`.

Validation on the local GB10 host:

- `cmake --preset host`, `cmake --build --preset host -j 8`,
  `ctest --preset host --output-on-failure`: **47 passed**, 2 GPU tests disabled.
- Issue 148, registry, expert-GEMV and blockwise-GEMM Python suites:
  **47 passed, 7 skipped**. The skips include four new gfx942 graph cases.
- Ruff, Python compilation and `bash -n` pass.
- Native changes formatted using the checked-in C++ style. The existing
  `.clang-format` also contains a `Language: Cuda` stanza rejected by
  clang-format 18; formatting used its C++ stanza without changing that file.

Required before promoting this draft:

1. Compile HIP and the diagnostic/benchmark targets on ROCm gfx942; run the
   native FNUZ and MFMA correctness checks, including both token counts.
2. Run the new changed-input graph cases and explicit stream checks on gfx942.
   Native convenience wrappers allocate/free scratch and launch on the default
   stream; they are standalone experiments, not serving graph adapters. A
   serving integration needs caller-owned scratch and explicit stream plumbing.
3. Re-measure against current main on the same MI300A node. Original comments
   contain historical measurements; they are not results for this merged tree.
4. Qualify held-out full-model histories against SGLang at matched parallelism
   with the existing **+0.01 nats/token** budget. MFMA activation quantization
   changes the numerical recipe; operator tolerance cannot establish that gate.

No fresh MI300A performance or full-model quality result is claimed here.
