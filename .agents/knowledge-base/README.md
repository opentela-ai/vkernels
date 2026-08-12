# Knowledge Base Index

The `.agents/knowledge-base/` directory contains fact sheets and quick-reference
documents for GPU kernel programming across AMD (ROCm/HIP) and NVIDIA (CUDA)
ecosystems. These are **reference documents** — dense, lookup-oriented — not
tutorials. For process and methodology, use the skills in `../skills/`.

## Reading the knowledge base

Use these documents when you need a **specific fact**:

- "What's the LDS size on MI300X?" → `amd-gpu-specs.md`
- "What's the HBM bandwidth on H100?" → `nvidia-gpu-specs.md`
- "How many VGPRs per thread on CDNA3?" → `amd-gpu-specs.md`
- "What's the HIP equivalent of `cudaMemcpyAsync`?" → `hip-api-reference.md`
- "What does `v_mfma_f32_16x16x16f16` do?" → `amdgpu-isa-reference.md`
- "What's the ridge point on MI300X?" → `performance-roofs.md`
- "How do I wait for LDS writes in AMD assembly?" → `amdgpu-isa-reference.md`
- "What's a wavefront called in NVIDIA?" → `cross-vendor-reference.md`
- "What counters tell me if a kernel is memory-bound?" → `profiling-tools.md`
- "How do I compute FLOPs for a GEMM?" → `performance-roofs.md`

## Document map

| Document | Contents | When to use |
|---|---|---|
| `amd-gpu-specs.md` | MI300X, MI250X, MI210, MI100 specs | Device capacities, peak rates |
| `nvidia-gpu-specs.md` | B200, H100, H200, A100 specs | Device capacities, peak rates |
| `cross-vendor-reference.md` | AMD↔NVIDIA terminology, memory, sync | Translating between platforms |
| `amdgpu-isa-reference.md` | AMDGPU instructions, registers, waitcnt | Writing inline assembly |
| `hip-api-reference.md` | HIP runtime API ↔ CUDA mappings | Writing HIP host code |
| `profiling-tools.md` | omniperf, omnitrace, ncu, nsys commands | Profiling a kernel |
| `performance-roofs.md` | AI formulas, ridge points, FLOP/byte calcs | Analyzing kernel bounds |

## Knowledge base vs skills

| Knowledge base | Skills |
|---|---|
| Dense facts and tables | Process and methodology |
| "What is X?" | "How do I optimize X?" |
| Lookup-oriented | Action-oriented |
| Read when you need a number | Read when you need a method |
| Static reference | Step-by-step workflow |
| Cross-linked from skills | Cross-reference to knowledge base |

## How skills use the knowledge base

Skills reference the knowledge base for concrete numbers. For example:

- `hip-efficient-kernels` says "On MI300X the ridge is ≈245 FLOP/byte" — the
  actual peak numbers come from `amd-gpu-specs.md`.
- `hip-kernel-profiling` says "use `omniperf analyze`" — the exact command
  syntax is in `profiling-tools.md`.
- `amdgpu-isa` describes the instruction model — the cheat sheet of
  instructions is in `amdgpu-isa-reference.md`.

When a skill says "see [references/X](references/X.md)" or gives a number,
verify it against the knowledge base if precision matters.

## Updating the knowledge base

When a new GPU launches or a new ROCm/CUDA version changes an API:

1. Update the relevant fact sheet with new numbers.
2. If the new hardware changes the ridge point significantly, update any
   skill text that quotes the old number.
3. Cross-check the cross-vendor reference for new ISA features.

Numbers should be pulled from official datasheets, not inference. Round to
3 significant figures.
