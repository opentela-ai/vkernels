# tests/python/test_moe_align_device.py
#
# VK_MOE_ALIGN_DEVICE kill-switch (fusion-candidate moe-align-device lane).
#
# The on-device moe_align_block_size is the serving DEFAULT (issue #78 —
# the topk_ids.cpu() host round-trip was 97-100% of PP0's per-call
# moe:vkernel_apply). There are two independent env kill-switches at the
# binding layer, both default ON:
#
#   VK_MOE_ALIGN_DEVICE=0  (this lane's naming, VK_DSA_DECODE_SPLIT /
#                           VK_MHC_PRE_STRICT precedent)
#   VKERNELS_GPU_ALIGN=0   (the original issue-#78 switch)
#
# Either switch forcing 0 must select the CPU align path. These tests only
# check the gate flags (no GPU required): the numerics parity of the device
# path itself is covered by meta/benchmarks/test_capi_moe_align.hip on
# gfx942, and the CPU path by tests/kernels/moe/.
from __future__ import annotations

import os
import subprocess
import sys
import unittest


def _flag_probe_code() -> str:
    return (
        "import vkernels.vllm_experts as ve; "
        "print(int(ve._MOE_ALIGN_DEVICE), int(ve._GPU_ALIGN), "
        "int(ve._MOE_ALIGN_DEVICE and ve._GPU_ALIGN))"
    )


class MoeAlignDeviceSwitchTest(unittest.TestCase):
    """VK_MOE_ALIGN_DEVICE defaults ON and 0 forces the CPU path."""

    def test_default_is_on(self):
        import vkernels.vllm_experts as ve

        if os.environ.get("VK_MOE_ALIGN_DEVICE") is not None:
            self.skipTest("VK_MOE_ALIGN_DEVICE is set in this environment")
        self.assertTrue(ve._MOE_ALIGN_DEVICE)

    def test_env_zero_disables(self):
        env = dict(os.environ, VK_MOE_ALIGN_DEVICE="0")
        out = subprocess.run(
            [sys.executable, "-c", _flag_probe_code()],
            env=env, capture_output=True, text=True, check=True,
        )
        moe_dev, _gpu, _both = out.stdout.split()
        self.assertEqual(moe_dev, "0")
        # The combined gate (what use_gpu consumes) must be closed too.
        self.assertEqual(_both, "0")

    def test_false_string_disables(self):
        env = dict(os.environ, VK_MOE_ALIGN_DEVICE="false")
        out = subprocess.run(
            [sys.executable, "-c", _flag_probe_code()],
            env=env, capture_output=True, text=True, check=True,
        )
        self.assertEqual(out.stdout.split()[0], "0")

    def test_other_values_keep_device_on(self):
        for v in ("1", "", "on"):
            env = dict(os.environ)
            if v == "":
                env.pop("VK_MOE_ALIGN_DEVICE", None)
            else:
                env["VK_MOE_ALIGN_DEVICE"] = v
            out = subprocess.run(
                [sys.executable, "-c", _flag_probe_code()],
                env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(out.stdout.split()[0], "1")

    def test_switches_are_independent(self):
        # The new switch must work regardless of the issue-#78 switch, and
        # vice versa: either one alone closes the combined gate.
        for env_over in (
            {"VK_MOE_ALIGN_DEVICE": "0"},
            {"VKERNELS_GPU_ALIGN": "0"},
            {"VK_MOE_ALIGN_DEVICE": "0", "VKERNELS_GPU_ALIGN": "0"},
        ):
            env = dict(os.environ, **env_over)
            out = subprocess.run(
                [sys.executable, "-c", _flag_probe_code()],
                env=env, capture_output=True, text=True, check=True,
            )
            _moe_dev, _gpu, both = out.stdout.split()
            self.assertEqual(both, "0", f"gate open with {env_over}")


if __name__ == "__main__":
    unittest.main()
