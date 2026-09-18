"""Standalone Triton launchers for the issue-#99 device templates.

Imported only by the CUDA-gated device test (``test_triton_device_templates_match_mirror``);
the whole module is a no-op when triton is absent so the bare environment
collects cleanly. The launchers supply ``worker``/``P`` per program exactly
as the megakernel composes task templates (``tl.program_id(0)``).
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl

    from vkernels.compiler.device_triton import _t_mhc_post, _t_mhc_pre

    @triton.jit
    def _mhc_pre_launch(streams_ptr, fn_ptr, base_ptr, scale_ptr, hin_ptr, post_ptr,
                        comb_ptr, B: tl.constexpr, HC: tl.constexpr, C: tl.constexpr,
                        MIX: tl.constexpr, EPS: tl.constexpr, RMS_EPS: tl.constexpr,
                        ITERS: tl.constexpr, MIXP: tl.constexpr, HCP: tl.constexpr,
                        BLOCK_K: tl.constexpr):
        _t_mhc_pre(tl.program_id(0), tl.num_programs(0), streams_ptr, fn_ptr, base_ptr,
                   scale_ptr, hin_ptr, post_ptr, comb_ptr, B, HC, C, MIX, EPS, RMS_EPS,
                   ITERS, MIXP, HCP, BLOCK_K)

    @triton.jit
    def _mhc_post_launch(streams_ptr, body_out_ptr, post_ptr, comb_ptr, out_ptr,
                         B: tl.constexpr, HC: tl.constexpr, C: tl.constexpr,
                         HCP: tl.constexpr, BLOCK_C: tl.constexpr):
        _t_mhc_post(tl.program_id(0), tl.num_programs(0), streams_ptr, body_out_ptr,
                    post_ptr, comb_ptr, out_ptr, B, HC, C, HCP, BLOCK_C)

except ImportError:  # pragma: no cover - bare environment (no triton)
    _mhc_pre_launch = None
    _mhc_post_launch = None
