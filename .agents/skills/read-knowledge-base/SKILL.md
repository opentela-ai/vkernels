---
name: read-knowledge-base
description: Navigate and use the GPU kernel programming knowledge base effectively. Use when you need a quick fact (hardware spec, API mapping, ISA instruction, profiling command, performance formula), when translating between AMD and NVIDIA ecosystems, when looking up a constant before making an optimization claim, or when you need a number to feed into a skill's methodology.
---

# Reading the Knowledge Base

The knowledge base at `.agents/knowledge-base/` contains dense, structured fact
sheets for GPU kernel programming across AMD and NVIDIA. This skill teaches you
how to find what you need quickly and how to use it with the skills system.

## When to use the knowledge base vs a skill

- **Knowledge base:** you need a **fact** — a number, an API mapping, an ISA
  instruction format, a command flag.
- **Skill:** you need a **method** — how to optimize, how to profile, how to
  classify a kernel, how to measure.

The knowledge base answers "what is X?"; skills answer "how do I do Y?"

Examples:
- "What's the fp16 peak on MI300X?" → `amd-gpu-specs.md`
- "How do I optimize a kernel on MI300X?" → load `hip-efficient-kernels`
- "What's the HIP equivalent of `cudaMalloc`?" → `hip-api-reference.md`
- "How do I allocate managed memory?" → the HIP runtime API section of `hip-api-reference.md`
- "What was that `s_waitcnt` syntax again?" → `amdgpu-isa-reference.md`
- "How do I pipeline LDS loads?" → load `hip-async-coordination`

## Quick lookup by question type

| Your question | Open this |
|---|---|
| How many CUs/TFLOPS/GB on GPU X? | `amd-gpu-specs.md` or `nvidia-gpu-specs.md` |
| What does this AMDGPU instruction do? | `amdgpu-isa-reference.md` |
| What's the HIP version of `cudaFoo`? | `hip-api-reference.md` |
| What's AMD's terminology for X? | `cross-vendor-reference.md` |
| How do I run the profiler to check Y? | `profiling-tools.md` |
| Is this kernel memory-bound or compute-bound? | `performance-roofs.md` for formulas, then `...-specs.md` for peaks |
| How do I compute FLOPs for a GEMM? | `performance-roofs.md` |
| What does `vmcnt(0)` wait for? | `amdgpu-isa-reference.md` (Wait counters section) |

## Reading strategy

### Step 1 — Identify the domain

Which ecosystem? AMD (ROCm/HIP), NVIDIA (CUDA), or both?

- AMD-only questions: start with `hip-*` or `amd-*` files
- NVIDIA-only questions: the existing skills have NVIDIA data; knowledge base
  provides the numbers in `nvidia-gpu-specs.md`
- Cross-platform: `cross-vendor-reference.md` maps between them

### Step 2 — Identify the precision level

Are you looking for:
- **Hardware capacity** (how much memory, how many TFLOPS)? → `*-gpu-specs.md`
- **Software API** (function signature, arguments)? → `hip-api-reference.md`
- **Low-level ISA** (instruction, register, wait counter)? → `amdgpu-isa-reference.md`
- **Performance number** (ridge point, AI formula)? → `performance-roofs.md`
- **Tool command** (profiler flag, metric name)? → `profiling-tools.md`

### Step 3 — Read the fact, then feed it to the skill

The knowledge base rarely tells you what to **do** with a fact. The pattern is:

1. Find the number/fact in the knowledge base
2. Load the relevant skill for methodology
3. Apply the methodology using the fact from step 1

Example workflow:
```
Q: "Is my 64×64 fp16 GEMM on MI300X compute-bound?"

1. Knowledge base lookup:
   - performance-roofs.md: AI = B/sizeof = 64/2 = 32 FLOP/byte
   - amd-gpu-specs.md: MI300X ridge = ~245 FLOP/byte
   - 32 < 245 → memory-bound

2. Load hip-efficient-kernels skill:
   - Step 2 (memory-bound): fuse, tile larger, use smaller dtype
   - Try B=128: AI = 64 → still memory-bound
   - Try B=256: AI = 128 → still below ridge for MI300X
   - Conclusion: this GEMM is memory-bound; optimize HBM traffic
```

## Cross-referencing between knowledge base and skills

When a skill makes a claim with a number, the knowledge base is the source of
truth:

```
Skill says:  "MI300X has ~245 FLOP/byte ridge point"
Verify via:  amd-gpu-specs.md (peak fp16 / HBM BW = 1307/5.3 ≈ 245)
             performance-roofs.md (ridge = peak_compute / peak_bandwidth)
```

When the knowledge base is updated (new GPU, new ROCm version), check the
skills for hardcoded numbers that reference the old value.

## Document structure within each fact sheet

Fact sheets are organized for scanning:

- **Tables first:** the fastest way to find a number. Scan the left column
  for the parameter, read the right column for the value.
- **Code blocks:** for API signatures, ISA syntax, and bash commands. Copy
  and paste into your work.
- **Commentary text:** explains caveats, edge cases, and gotchas. Read after
  you've found the basic fact.

## The knowledge base is not exhaustive

The knowledge base covers the most commonly needed facts. For edge cases:

- AMD ROCm full documentation: https://rocm.docs.amd.com
- AMD GPU ISA manual: `amdgpu-isa-manual` (from ROCm install)
- NVIDIA CUDA programming guide: https://docs.nvidia.com/cuda
- PTX ISA reference: https://docs.nvidia.com/cuda/parallel-thread-execution

## Completion criterion

You found the specific fact you needed, fed it to the right skill's
methodology, and arrived at an actionable decision backed by a number from
the knowledge base.
