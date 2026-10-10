#!/usr/bin/env python3
"""Script-transform: upstream gfx950 Gluon kernel -> CDNA3 (gfx942) port.

The template proven by ``mha_extend.py`` (first-run parity). Applies the
mechanical CDNA3 substitutions to an upstream kernel file:

  1. ``_triton``/``_common`` imports -> the local ``_gluon`` shim with the
     ``_common`` pieces inlined (InputStrides, max/maximum, constants, and a
     CDNA3 ``attention_layouts`` returning v2 [32,32,8] non-transposed layouts
     with plain swizzled shared layouts).
  2. ``cdna4 = gl.amd.cdna4`` -> CDNA3-first alias; ``cdna4.mfma`` -> cdna3.
  3. K smem allocations transposed ([D, N]) + permuted stores + no-permuted
     operand reads (dual-orientation staging); async DMA pairs collapsed to
     synchronous loads; ``async_copy`` waits deleted.
  4. SM-count hardcodes -> device CU count.
  5. Module-scope ``gl.constexpr`` literals -> plain ints.

Kernel-specific logic (schedulers, masking ladders, reduce kernels, host
heuristics) passes through verbatim. Usage:

  python3 tools/cdna3_transform.py <upstream.py> <out.py> [--instr 32x32x8]

Not a blind tool: the output needs its module docstring, launcher rename,
and FP8 gate completed by hand, plus a parity test.
"""

from __future__ import annotations

import argparse
import re

HEADER = '''from vkernels.torch_ops.tokenspeed_amd._gluon import gl, gluon

# --- PORT (CDNA3/gfx942): inlined upstream ``_common`` pieces ---

from triton.experimental.gluon.language import amd

# Upstream binds ``gl.amd.cdna4``; CDNA3 exposes identical buffer ops.
_amd_ns = getattr(gl, "amd", None)
cdna4 = getattr(_amd_ns, "cdna3", None) or getattr(_amd_ns, "cdna4", None)

_INV_LN2_VALUE = 1.4426950408889634
_LN2_VALUE = 0.6931471805599453
_INV_LN2 = gl.constexpr(_INV_LN2_VALUE)
_LN2 = gl.constexpr(_LN2_VALUE)
MAX_KV_SPLITS = 32


@gluon.aggregate
class InputStrides:
    stride_t: gl.constexpr
    stride_h: gl.constexpr
    stride_d: gl.constexpr

    @gluon.jit
    def offsets(self, token, head, dim):
        return (token * self.stride_t + head * self.stride_h + dim * self.stride_d).to(
            gl.int32
        )


@gluon.jit
def maximum(a, b):
    return gl.maximum(a, b)


@gluon.jit
def max(input, axis=None, keep_dims=False):
    return gl.reduce(input, axis, maximum, keep_dims=keep_dims)


def select_kv_splits(*, base_ctas: int, num_pages: int, sm_count: int) -> int:
    wave_target = 2
    min_pages_per_split = 2
    min_page_splits = 8
    max_page_splits = MAX_KV_SPLITS
    target_ctas = sm_count * wave_target
    splits_for_occupancy = (target_ctas + base_ctas - 1) // base_ctas
    splits_for_pages = num_pages // min_pages_per_split
    min_page_splits = min(min_page_splits, num_pages)
    if splits_for_pages < min_page_splits:
        splits_for_pages = min_page_splits
    if splits_for_pages > max_page_splits:
        splits_for_pages = max_page_splits
    return min(splits_for_occupancy, splits_for_pages)


@gluon.constexpr_function
def padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True):
    """PORT: CDNA3 - padded (CDNA4 DMA-legal) shared layouts are replaced by
    plain swizzled staging (shape comes from the allocation)."""
    return gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])


@gluon.jit
def _mfma_unscaled_fp8(a, b, acc):
    # PORT: fp8 needs dot-scaled MFMA (gfx950-only); never traced on CDNA3
    # because IS_FP8 is constexpr-False and the launcher rejects fp8 dtypes.
    raise NotImplementedError("fp8 dot-scaled MFMA is not ported to CDNA3")


@gluon.constexpr_function
def attention_layouts(head_dim, block_n, is_fp8, dtype, num_warps, instr_shape):
    """CDNA3 replacement: v2 non-transposed (upstream: v4 transposed);
    operand k-widths follow the 128-bit-load rule; plain swizzled shared
    layouts replace the CDNA4 DMA-legal padded ones."""
    mfma = gl.amd.AMDMFMALayout(
        version=2,
        instr_shape=[32, 32, 8],
        transposed=False,
        warps_per_cta=[num_warps, 1],
    )
    qk_layout = mfma
    pv_layout = mfma
    qk_kw = 8
    pv_kw = 4
    q_layout = gl.DotOperandLayout(0, qk_layout, k_width=qk_kw)
    k_layout = gl.DotOperandLayout(1, qk_layout, k_width=qk_kw)
    p_layout = gl.DotOperandLayout(0, pv_layout, k_width=pv_kw)
    v_layout = gl.DotOperandLayout(1, pv_layout, k_width=pv_kw)
    load_vec = 8
    load_threads = head_dim // load_vec
    load_layout = gl.BlockedLayout(
        [1, load_vec], [64 // load_threads, load_threads], [num_warps, 1], [1, 0]
    )
    store_vec = 8
    store_threads = head_dim // store_vec
    store_layout = gl.BlockedLayout(
        [1, store_vec], [64 // store_threads, store_threads], [num_warps, 1], [1, 0]
    )
    k_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    v_smem_layout = gl.SwizzledSharedLayout(1, 1, 1, order=[0, 1])
    return (
        qk_layout,
        pv_layout,
        q_layout,
        k_layout,
        p_layout,
        v_layout,
        load_layout,
        store_layout,
        k_smem_layout,
        v_smem_layout,
    )
'''

IMPORT_RE = re.compile(
    r"from tokenspeed_kernel_amd\._triton import [^\n]*\n"
    r"(?:from tokenspeed_kernel_amd[^\n]*\n(?:\s+[^\n]*\n)*?\n?)"
)


def transform(src: str) -> str:
    m = IMPORT_RE.search(src)
    if not m:
        raise SystemExit("upstream import block not found")
    src = src[: m.start()] + HEADER + src[m.end() :]

    src = re.sub(r"cdna4 = gl\.amd\.cdna4\n", "", src)
    src = re.sub(r"async_copy = cdna4\.async_copy\n", "", src)

    # mfma -> cdna3
    src = src.replace("cdna4.mfma(", "amd.cdna3.mfma(")

    # K smem: physically transposed staging ([BLOCK_N, HEAD_DIM] -> [HEAD_DIM, BLOCK_N])
    src = re.sub(
        r"k_smem = gl\.allocate_shared_memory\(\s*"
        r"([a-z_]+\.dtype\.element_ty),\s*\[cfg\.BLOCK_N, cfg\.HEAD_DIM\],\s*"
        r"cfg\.k_smem_layout,\s*\)",
        r"k_smem = gl.allocate_shared_memory(\n"
        r"        \1, [cfg.HEAD_DIM, cfg.BLOCK_N],\n"
        r"        cfg.k_smem_layout\n"
        r"    )  # PORT: K stages pre-permuted [HEAD_DIM, BLOCK_N]",
        src,
    )

    # multi-buffered K-class smem: swap to [.., HEAD/ROPE DIM, BLOCK_N]
    src = src.replace("[2, cfg.BLOCK_N, cfg.HEAD_DIM]", "[2, cfg.HEAD_DIM, cfg.BLOCK_N]")
    src = src.replace("[2, cfg.BLOCK_N, cfg.ROPE_DIM]", "[2, cfg.ROPE_DIM, cfg.BLOCK_N]")
    src = src.replace(
        "[cfg.BLOCK_N, cfg.ROPE_DIM]", "[cfg.ROPE_DIM, cfg.BLOCK_N]"
    )  # single-buffer k_pe

    # issue_load_k/v bodies: synchronous permuted/natural stores
    src = src.replace(
        "async_copy.buffer_load_to_shared(k_smem, self.k_cache_ptr, offsets)",
        "k_smem.store(gl.load(self.k_cache_ptr + offsets).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(k_smem, self.k_cache_ptr, offsets, mask=mask)",
        "k_smem.store(gl.load(self.k_cache_ptr + offsets, mask=mask).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(k_smem, self.k_cache_ptr, offsets, mask=mask, other=other)",
        "k_smem.store(gl.load(self.k_cache_ptr + offsets, mask=mask, other=other).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(v_smem, self.v_cache_ptr, offsets)",
        "v_smem.store(gl.load(self.v_cache_ptr + offsets))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(v_smem, self.v_cache_ptr, offsets, mask=mask)",
        "v_smem.store(gl.load(self.v_cache_ptr + offsets, mask=mask))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(v_smem, self.v_cache_ptr, offsets, mask=mask, other=other)",
        "v_smem.store(gl.load(self.v_cache_ptr + offsets, mask=mask, other=other))",
    )
    src = src.replace(
        "async_copy.global_load_to_shared(k_smem, self.k_cache_ptr + offsets)",
        "k_smem.store(gl.load(self.k_cache_ptr + offsets).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.global_load_to_shared(v_smem, self.v_cache_ptr + offsets)",
        "v_smem.store(gl.load(self.v_cache_ptr + offsets))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(smem, self.k_ptr, offsets)",
        "smem.store(gl.load(self.k_ptr + offsets).permute([1, 0]))",
    )
    src = src.replace(
        """async_copy.buffer_load_to_shared(
                smem, self.k_ptr, offsets, mask=mask, other=other
            )""",
        "smem.store(gl.load(self.k_ptr + offsets, mask=mask, other=other).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(k_smem, self.k_ptr, offsets)",
        "k_smem.store(gl.load(self.k_ptr + offsets).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(k_smem, self.k_ptr, offsets, mask=mask)",
        "k_smem.store(gl.load(self.k_ptr + offsets, mask=mask).permute([1, 0]))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(v_smem, self.v_ptr, offsets)",
        "v_smem.store(gl.load(self.v_ptr + offsets))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(v_smem, self.v_ptr, offsets, mask=mask)",
        "v_smem.store(gl.load(self.v_ptr + offsets, mask=mask))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(v_smem, self.v_ptr, offsets, mask=mask, other=other)",
        "v_smem.store(gl.load(self.v_ptr + offsets, mask=mask, other=other))",
    )
    src = src.replace(
        "async_copy.buffer_load_to_shared(k_smem, self.k_ptr, offsets, mask=mask, other=other)",
        "k_smem.store(gl.load(self.k_ptr + offsets, mask=mask, other=other).permute([1, 0]))",
    )
    src = src.replace("async_copy.commit_group()\n", "")

    # shared_load_k: drop the permuted view (physically transposed now)
    src = src.replace(
        """k_buffer = k_smem.permute([1, 0])
        return k_buffer.load(cfg.k_layout)""",
        "return k_smem.load(cfg.k_layout)",
    )
    src = src.replace(
        "return k_smem.permute([1, 0]).load(cfg.k_layout)",
        "return k_smem.load(cfg.k_layout)",
    )

    # K-class smem reads: physically transposed staging -> no permuted view
    src = re.sub(r"(\w+)\.permute\(\[1, 0\]\)\.load\(", r"\1.load(", src)

    # drop the async waits
    src = re.sub(r"[ \t]*async_copy\.wait_group\(\d+\)\n", "", src)

    # SM count -> device CUs
    src = re.sub(r"_GFX950_SM_COUNT = 256", "_GFX950_SM_COUNT = None  # PORT: device CU count", src)
    src = src.replace(
        "sm_count=_GFX950_SM_COUNT,",
        "sm_count=torch.cuda.get_device_properties(q.device).multi_processor_count,",
    )

    # module-scope constexpr literals -> ints
    src = re.sub(r"gl\.constexpr\((\d+)\)", r"\1", src)

    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    args = ap.parse_args()
    src = open(args.src).read()
    out = transform(src)
    open(args.out, "w").write(out)
    residual = [l for l in out.splitlines() if "async_copy" in l or "gl.amd.cdna4" in l]
    print(f"transformed {args.src} -> {args.out} ({len(out)} chars)")
    if residual:
        print("RESIDUAL upstream refs (need hand-patching):")
        for l in residual[:10]:
            print("  ", l.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
