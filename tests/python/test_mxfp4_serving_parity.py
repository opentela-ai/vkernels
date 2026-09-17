"""Serving-layout parity gate for the VKERNELS_MXFP4_BF16 MoE backend.

Issue #74: the fused ``vk_hip_fused_moe_mxfp4`` serving backend never had a
correctness gate — the K3 "smoke" check was non-empty-only, so an
incompatible serving weight format (the AITER gfx950 CK shuffle fed to a
kernel expecting the raw ``create_weights()`` layout) shipped as degenerate
serving output (``.dartampionship…`` / all-NaN ``!!!!…``).

This gate closes that hole with a serving-independent oracle:

* :class:`ConvertWeightsForVkernelTest` — the
  :func:`vkernels.vllm_experts.convert_weights_for_vkernel` contract
  (weights/scales pass through unchanged, biases cast to the C ABI's fp32).
* :class:`Mxfp4ServingParityTest` — the HIP kernel on gfx942 vs a naive
  bf16 dequant reference (torch-free numpy oracle, fp32 accumulation) at
  the exact weight layout the kernel documents, at both a CI-cheap shape
  and (env-gated) the real K3 serving shape (h=7168, ispp=3072, E=112)
  that the in-tree bigshape sweep brackets but does not include.
* a negative control proving the gate can actually catch a layout
  mismatch: the same weights pair-interleaved (the AITER-input gate/up
  convention) must FAIL parity.

The HIP tests are skipped unless a gfx942 device and
``libvkernels_hip.so`` are available (CI / non-ROCm hosts run only the CPU
contract tests).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional
    torch = None

from vkernels._fallback import _FP4_NIBBLE_VALUES
from vkernels.vllm_experts import (
    find_libvkernels_hip,
    load_libvkernels_hip,
    moe_align_block_size_with_map,
    resolve_moe_fn,
)


# --- bf16 / mxfp4 bit helpers (mirror src/c/vkernels/kernels/moe_fused.hip) --


def _pack_bf16_np(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (uint16), round-to-nearest-even."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def _bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """bf16 bit patterns (uint16) -> float32 values."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _ue8m0_scale_floats(scale_bytes: np.ndarray) -> np.ndarray:
    """Decode ue8m0 scale bytes (2^(s-127); 0xFF -> 0) to float32."""
    out = np.ones_like(scale_bytes, dtype=np.float32)
    finite = scale_bytes != 0xFF
    out[finite] = np.exp2(scale_bytes[finite].astype(np.int32) - 127)
    out[~finite] = 0.0
    return out


def _dequant_mxfp4(packed: np.ndarray, scale: np.ndarray,
                   group_size: int = 32) -> np.ndarray:
    """[E, N, K/2] uint8 + [E, N, K/32] uint8 -> bf16 bits [E, N, K].

    Mirrors the HIP dequant: low nibble is the even k, high nibble the odd
    k; value = nibble * 2^(s-127); a single bf16 rounding.  Processed
    expert-by-expert so the K3 serving shape stays memory-bounded.
    """
    E, N, Kh = packed.shape
    K = Kh * 2
    out = np.empty((E, N, K), dtype=np.uint16)
    for e in range(E):
        nib = np.empty((N, K), dtype=np.int32)
        nib[:, 0::2] = packed[e] & 0x0F
        nib[:, 1::2] = (packed[e] >> 4) & 0x0F
        vals = np.array(_FP4_NIBBLE_VALUES, dtype=np.float32)[nib]
        vals *= np.repeat(_ue8m0_scale_floats(scale[e]), group_size, axis=-1)
        out[e] = _pack_bf16_np(vals)
    return out


def _make_weights(E, hidden, ispp, seed, scale_lo=124, scale_hi=130):
    """Random MXFP4 weights in the RAW create_weights() layout.

    w13 [E, 2*ispp, hidden/2] uint8 (gate rows [0, ispp), up rows
    [ispp, 2*ispp)), w2 [E, hidden, ispp/2] uint8, matching ue8m0 scales.
    Nibbles restricted to finite values; scales to a benign exponent band.
    """
    rng = np.random.default_rng(seed)
    finite_nibbles = np.array(
        [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13], dtype=np.uint8
    )

    def rand_packed(shape):
        # shape is the PACKED shape (..., K/2); draw nibbles at element
        # resolution (..., K) then pack pairs (even k -> low nibble).
        elem_shape = (*shape[:-1], shape[-1] * 2)
        nib = finite_nibbles[rng.integers(0, len(finite_nibbles), elem_shape)]
        return (nib[..., 0::2] | (nib[..., 1::2] << 4)).astype(np.uint8)

    def rand_scale(shape):
        return rng.integers(scale_lo, scale_hi + 1, shape).astype(np.uint8)

    w13 = rand_packed((E, 2 * ispp, hidden // 2))
    w13_scale = rand_scale((E, 2 * ispp, hidden // 32))
    w2 = rand_packed((E, hidden, ispp // 2))
    w2_scale = rand_scale((E, hidden, ispp // 32))
    return w13, w13_scale, w2, w2_scale


def _interleave_pairs(packed, scale):
    """Raw layout -> gate/up pair-interleaved rows (the AITER-input
    convention): row 2i = gate_i, row 2i+1 = up_i."""
    E, N, Kh = packed.shape
    ispp = N // 2
    p = np.empty_like(packed)
    p[:, 0::2] = packed[:, :ispp]
    p[:, 1::2] = packed[:, ispp:]
    s = np.empty_like(scale)
    s[:, 0::2] = scale[:, :ispp]
    s[:, 1::2] = scale[:, ispp:]
    return p, s


def _bf16_reference(hs_bits, w13_bits, w2_bits, topk_ids, topk_w,
                    b13, b2, ispp, beta=1.0, linear_beta=25.0):
    """Naive bf16 MoE oracle: bf16 weights (single dequant rounding, same
    as the kernel), fp32-accumulated matmuls, K3 SiTU activation.

    hs_bits: [M, hidden] uint16 bf16 patterns.  w13_bits: [E, 2*ispp,
    hidden] uint16 (gate rows [0, ispp)).  Returns fp32 [M, hidden].
    """
    M = hs_bits.shape[0]
    hidden = hs_bits.shape[1]
    top_k = topk_ids.shape[1]
    E = w13_bits.shape[0]
    out = np.zeros((M, hidden), dtype=np.float32)
    x = _bf16_bits_to_f32(hs_bits)  # [M, hidden]

    for e in range(E):
        rows = np.argwhere(topk_ids == e)  # [n_e, 2] = (token, slot)
        if rows.size == 0:
            continue
        gate_w = _bf16_bits_to_f32(w13_bits[e, :ispp])  # [ispp, hidden]
        up_w = _bf16_bits_to_f32(w13_bits[e, ispp:])
        down_w = _bf16_bits_to_f32(w2_bits[e])  # [hidden, ispp]
        toks = rows[:, 0]
        slots = rows[:, 1]
        a = x[toks]  # [n_e, hidden]
        g = a @ gate_w.T
        u = a @ up_w.T
        if b13 is not None:
            g = g + b13[e, :ispp]
            u = u + b13[e, ispp:]
        gate_out = beta * np.tanh(g / beta) / (1.0 + np.exp(-np.clip(g, -60, 60)))
        up_out = linear_beta * np.tanh(u / linear_beta)
        act = gate_out * up_out  # [n_e, ispp]
        y = act @ down_w.T
        if b2 is not None:
            y = y + b2[e]
        w = topk_w[toks * top_k + slots]
        out[toks] += w[:, None] * y
    return out


# --- HIP harness (the same call the serving integration makes) --------------

_HIP_OK, _HIP_WHY = (False, "torch unavailable")
if torch is not None:
    if not torch.cuda.is_available():
        _HIP_WHY = "torch.cuda unavailable"
    else:
        try:
            gcn = getattr(torch.cuda.get_device_properties(0),
                          "gcnArchName", "")
            if "gfx942" not in gcn:
                _HIP_WHY = f"device {gcn} is not gfx942"
            elif find_libvkernels_hip() is None:
                _HIP_WHY = "libvkernels_hip.so not found"
            else:
                _HIP_OK, _HIP_WHY = True, ""
        except Exception as exc:  # pragma: no cover
            _HIP_WHY = f"device probe failed: {exc}"


# C ABI of vk_hip_fused_moe_mxfp4 (src/c/vkernels/capi/hip_capi.cpp).
_MOE_FN_ARGTYPES = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # w13
    ctypes.c_void_p,  # w13_scale
    ctypes.c_void_p,  # w2
    ctypes.c_void_p,  # w2_scale
    ctypes.c_void_p,  # topk_ids
    ctypes.c_void_p,  # topk_w
    ctypes.c_void_p,  # act_scratch
    ctypes.c_void_p,  # out
    ctypes.c_int,     # M
    ctypes.c_int,     # hidden
    ctypes.c_int,     # ispp
    ctypes.c_int,     # top_k
    ctypes.c_void_p,  # sorted_ids
    ctypes.c_void_p,  # expert_ids
    ctypes.c_int,     # EM
    ctypes.c_float,   # swiglu_limit
    ctypes.c_int,     # activation (1 = SiTU)
    ctypes.c_float,   # beta
    ctypes.c_float,   # linear_beta
    ctypes.c_void_p,  # b13 (fp32) or NULL
    ctypes.c_void_p,  # b2 (fp32) or NULL
    ctypes.c_int,     # block_size
    ctypes.c_void_p,  # stream
]


def _run_hip_kernel(hidden_states, w13, w13_scale, w2, w2_scale,
                    topk_ids, topk_w, b13=None, b2=None, beta=1.0,
                    linear_beta=25.0):
    """Run vk_hip_fused_moe_mxfp4 on the current stream; returns fp32
    [M, hidden] numpy."""
    M, hidden = hidden_states.shape
    E, N, _ = w13.shape
    ispp = N // 2
    top_k = topk_ids.shape[1]
    block_size = 16 if M <= 32 else 64

    dev = hidden_states.device
    sids_np, eids_np, EM = moe_align_block_size_with_map(
        topk_ids.cpu().numpy().astype(np.int32).ravel(), E, block_size, None,
    )
    d_sids = torch.from_numpy(sids_np).to(dev)
    d_eids = torch.from_numpy(eids_np).to(dev)
    act = torch.zeros(EM * ispp, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(M * hidden, dtype=torch.float32, device=dev)

    moe_fn = resolve_moe_fn(load_libvkernels_hip())
    moe_fn.argtypes = _MOE_FN_ARGTYPES
    moe_fn.restype = None
    stream = torch.cuda.current_stream().cuda_stream
    moe_fn(
        ctypes.c_void_p(hidden_states.contiguous().data_ptr()),
        ctypes.c_void_p(w13.contiguous().data_ptr()),
        ctypes.c_void_p(w13_scale.contiguous().data_ptr()),
        ctypes.c_void_p(w2.contiguous().data_ptr()),
        ctypes.c_void_p(w2_scale.contiguous().data_ptr()),
        ctypes.c_void_p(topk_ids.contiguous().data_ptr()),
        ctypes.c_void_p(topk_w.contiguous().data_ptr()),
        ctypes.c_void_p(act.data_ptr()),
        ctypes.c_void_p(out.data_ptr()),
        ctypes.c_int(M), ctypes.c_int(hidden), ctypes.c_int(ispp),
        ctypes.c_int(top_k),
        ctypes.c_void_p(d_sids.data_ptr()),
        ctypes.c_void_p(d_eids.data_ptr()),
        ctypes.c_int(EM),
        ctypes.c_float(4.0),  # swiglu_limit (unused for SiTU)
        ctypes.c_int(1),      # SiTU
        ctypes.c_float(beta), ctypes.c_float(linear_beta),
        ctypes.c_void_p(b13.contiguous().data_ptr()) if b13 is not None
        else None,
        ctypes.c_void_p(b2.contiguous().data_ptr()) if b2 is not None
        else None,
        ctypes.c_int(block_size),
        ctypes.c_void_p(stream),
    )
    torch.cuda.synchronize()
    return out.view(M, hidden).cpu().numpy()


def _assert_parity(testcase, got, ref, label):
    """Cancellation-aware tolerance (mirrors the in-tree bigshape gate:
    bad if e > 0.05*|ref| + 0.03*rms(ref))."""
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    abs_tol = 0.03 * max(ref_rms, 1e-6)
    err = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    bad = err > (0.05 * np.abs(ref) + abs_tol)
    testcase.assertLess(
        float(bad.mean()), 0.02,
        f"{label}: {int(bad.sum())}/{bad.size} elements outside tolerance "
        f"(max_abs_err={float(err.max()):.4g}, ref_rms={ref_rms:.4g})",
    )


@unittest.skipIf(torch is None, "torch is required")
class ConvertWeightsForVkernelTest(unittest.TestCase):
    """Contract of the issue-#74 serving weight conversion (CPU-only)."""

    def test_weights_and_scales_pass_through_unchanged(self):
        from vkernels.vllm_experts import convert_weights_for_vkernel

        E, hidden, ispp = 3, 128, 64
        w13, w13_scale, w2, w2_scale = _make_weights(E, hidden, ispp, seed=0)
        w13b = torch.from_numpy(w13)
        w2b = torch.from_numpy(w2)
        s13b = torch.from_numpy(w13_scale)
        s2b = torch.from_numpy(w2_scale)

        out = convert_weights_for_vkernel(w13b, w2b, s13b, s2b)

        # .data returns a new tensor object sharing the same storage.
        self.assertEqual(out[0].data_ptr(), w13b.data_ptr())
        self.assertEqual(out[1].data_ptr(), w2b.data_ptr())
        self.assertEqual(out[2].data_ptr(), s13b.data_ptr())
        self.assertEqual(out[3].data_ptr(), s2b.data_ptr())
        self.assertTrue(torch.equal(out[0], w13b))
        self.assertTrue(torch.equal(out[2], s13b))

    def test_biases_cast_to_c_abi_fp32(self):
        from vkernels.vllm_experts import convert_weights_for_vkernel

        E, hidden, ispp = 2, 128, 64
        w13, w13_scale, w2, w2_scale = _make_weights(E, hidden, ispp, seed=1)
        b13 = torch.randn(E, 2 * ispp, dtype=torch.bfloat16)
        b2 = torch.randn(E, hidden, dtype=torch.bfloat16)

        *_, b13_out, b2_out = convert_weights_for_vkernel(
            torch.from_numpy(w13), torch.from_numpy(w2),
            torch.from_numpy(w13_scale), torch.from_numpy(w2_scale),
            b13, b2,
        )
        self.assertEqual(b13_out.dtype, torch.float32)
        self.assertEqual(b2_out.dtype, torch.float32)
        self.assertTrue(torch.equal(b13_out, b13.to(torch.float32)))


@unittest.skipIf(torch is None, "torch is required")
@unittest.skipIf(
    importlib.util.find_spec("vllm") is None,
    "vLLM is required (oracle import) — runs on the serving image",
)
class VkernelBackendShimTest(unittest.TestCase):
    """Integration gate: register_vkernel_backend_shim must make the
    ``AITER_MXFP4_BF16`` enum — the enum the K3 serving shim selects —
    resolve to VkernelFusedExperts AND convert weights with the raw
    pass-through contract, not the AITER gfx950 CK shuffle (issue #74)."""

    @classmethod
    def setUpClass(cls):
        from vkernels.vllm_experts import register_vkernel_backend_shim

        cls.installed = register_vkernel_backend_shim()
        from vllm.model_executor.layers.fused_moe.oracle import (
            mxfp4 as oracle_mxfp4,
        )
        cls.oracle = oracle_mxfp4

    def test_shim_installed(self):
        self.assertTrue(self.installed)

    def test_aiter_enum_resolves_to_vkernel_experts(self):
        from vkernels.vllm_experts import VkernelFusedExperts

        cls_list = self.oracle.backend_to_kernel_cls(
            self.oracle.Mxfp4MoeBackend.AITER_MXFP4_BF16
        )
        self.assertEqual(cls_list, [VkernelFusedExperts])

    def test_conversion_is_raw_pass_through(self):
        E, hidden, ispp = 2, 256, 128
        w13, w13_scale, w2, w2_scale = _make_weights(E, hidden, ispp, seed=3)
        b13 = torch.randn(E, 2 * ispp, dtype=torch.bfloat16)
        b2 = torch.randn(E, hidden, dtype=torch.bfloat16)
        out = self.oracle.convert_weight_to_mxfp4_moe_kernel_format(
            mxfp4_backend=self.oracle.Mxfp4MoeBackend.AITER_MXFP4_BF16,
            layer=None,
            w13_weight=torch.from_numpy(w13),
            w2_weight=torch.from_numpy(w2),
            w13_weight_scale=torch.from_numpy(w13_scale),
            w2_weight_scale=torch.from_numpy(w2_scale),
            w13_bias=b13,
            w2_bias=b2,
        )
        w13o, w2o, s13o, s2o, b13o, b2o = out
        self.assertTrue(torch.equal(w13o, torch.from_numpy(w13)))
        self.assertTrue(torch.equal(s13o, torch.from_numpy(w13_scale)))
        self.assertTrue(torch.equal(w2o, torch.from_numpy(w2)))
        self.assertEqual(w13o.dtype, torch.uint8)  # NOT a fp4x2/CK view
        self.assertEqual(b13o.dtype, torch.float32)


@unittest.skipUnless(_HIP_OK, _HIP_WHY)
class Mxfp4ServingParityTest(unittest.TestCase):
    """HIP vk_hip_fused_moe_mxfp4 vs the naive bf16 reference, on the raw
    create_weights() layout the serving integration must feed it."""

    # (M, hidden, ispp, E, top_k, seed)
    SHAPES = [
        (8, 256, 128, 4, 2, 11),   # decode path (block_size 16)
        (64, 512, 256, 8, 4, 12),  # prefill path (block_size 64)
    ]
    K3_SHAPE = (4, 7168, 3072, 112, 8, 74)  # env-gated (~4 GB of weights)

    def _case(self, M, hidden, ispp, E, top_k, seed, wrong_layout=False):
        w13, w13_scale, w2, w2_scale = _make_weights(E, hidden, ispp, seed)
        # The bf16 reference is ALWAYS computed from the raw (correct)
        # layout — snapshot before the wrong-layout mutation.
        ref_w13, ref_w13_scale = w13.copy(), w13_scale.copy()
        if wrong_layout:
            w13, w13_scale = _interleave_pairs(w13, w13_scale)
        rng = np.random.default_rng(seed + 1)
        # Distinct experts per token, random normalized routing weights.
        topk_ids = np.array(
            [rng.permutation(E)[:top_k] for _ in range(M)], dtype=np.int32
        )
        raw_w = rng.random((M, top_k)).astype(np.float32) + 0.5
        topk_w = (raw_w / raw_w.sum(axis=1, keepdims=True)).ravel()
        b13 = (rng.standard_normal((E, 2 * ispp)) * 0.01).astype(np.float32)
        b2 = (rng.standard_normal((E, hidden)) * 0.01).astype(np.float32)

        hs = (rng.standard_normal((M, hidden)) * 0.05).astype(np.float32)
        hs_bits = _pack_bf16_np(hs)
        dev = torch.device("cuda")
        hs_t = (
            torch.from_numpy(hs_bits.copy())
            .view(torch.bfloat16).to(dev)
        )

        got = _run_hip_kernel(
            hs_t,
            torch.from_numpy(w13).to(dev),
            torch.from_numpy(w13_scale).to(dev),
            torch.from_numpy(w2).to(dev),
            torch.from_numpy(w2_scale).to(dev),
            torch.from_numpy(topk_ids).to(dev),
            torch.from_numpy(topk_w).to(dev),
            torch.from_numpy(b13).to(dev),
            torch.from_numpy(b2).to(dev),
        )

        ref = _bf16_reference(
            hs_bits,
            _dequant_mxfp4(ref_w13, ref_w13_scale),
            _dequant_mxfp4(w2, w2_scale),
            topk_ids, topk_w, b13, b2, ispp,
        )
        return got, ref

    def test_parity_decode_shape(self):
        got, ref = self._case(*self.SHAPES[0])
        _assert_parity(self, got, ref, "decode M=8")

    def test_parity_prefill_shape(self):
        got, ref = self._case(*self.SHAPES[1])
        _assert_parity(self, got, ref, "prefill M=64")

    @unittest.skipUnless(
        os.environ.get("VK74_K3_SHAPE") == "1",
        "set VK74_K3_SHAPE=1 for the real K3 serving shape (h=7168, "
        "ispp=3072, E=112; ~4 GB of weights)",
    )
    def test_parity_k3_serving_shape(self):
        got, ref = self._case(*self.K3_SHAPE)
        _assert_parity(self, got, ref, "K3 serving shape")

    def test_negative_control_interleaved_layout_fails(self):
        """The gate must catch the issue-#74 failure mode: the same weights
        in the pair-interleaved (AITER-input) layout must NOT pass."""
        got, ref = self._case(*self.SHAPES[0], wrong_layout=True)
        ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
        abs_tol = 0.03 * max(ref_rms, 1e-6)
        err = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        frac_bad = float((err > 0.05 * np.abs(ref) + abs_tol).mean())
        self.assertGreater(
            frac_bad, 0.5,
            "interleaved gate/up layout unexpectedly matched the bf16 "
            "reference — the negative control is toothless",
        )


if __name__ == "__main__":
    unittest.main()
