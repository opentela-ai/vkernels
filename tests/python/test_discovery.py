"""Contract tests for vkernels discovery and the ``vkl`` CLI.

Run from the repository root (uv-managed project, see pyproject.toml):

    uv run python -m unittest discover -s tests/python -v
    uv run pytest              # same suite via pytest
    # or: make py-test
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
sys.path.insert(0, str(_SRC))

from vkernels import discovery  # noqa: E402
from vkernels.cli import main as cli_main  # noqa: E402

ROOT = discovery.find_repo_root(_SRC / "vkernels")

# Exact contract for the kernels/ headers (files sorted alphabetically,
# declarations in file order): name -> category.
EXPECTED_KERNELS = [
    ("add", "elementwise"),
    ("scale", "elementwise"),
    ("relu", "elementwise"),
    ("gemm", "gemm"),
    ("direct_lds_fill_bf16", "moe"),
    ("fp4_to_bf16_dequant", "moe"),
    ("use_async_copy_default", "moe"),
    ("mfma_f32_16x16x16bf16", "moe"),
    ("fused_moe_mxfp4_cpu", "moe_fused"),
    ("moe_align_block_size", "moe_fused"),
    ("fused_moe_mxfp4", "moe_fused"),
    ("sum", "reduce"),
    ("max", "reduce"),
]

# Exact contract for the comm/ headers, in file then declaration order.
EXPECTED_COMM = [
    ("ring_allreduce_rank", "function"),
    ("ring_allreduce", "function"),
    ("Channel", "class"),
    ("BlockingQueue", "class"),
    ("MockChannel", "class"),
    ("make_ring_channels", "function"),
    ("OverlapExecutor", "class"),
    ("p2p_gather_runs", "function"),
    ("Gather2DRun", "struct"),
    ("StagedRun1D", "struct"),
    ("StagedRun2D", "struct"),
    ("stage_runs_1d", "function"),
    ("stage_runs_2d", "function"),
    ("p2p_gather_runs_2d", "function"),
    ("memcpy_peer_batch_async", "function"),
    ("set_gather_dispatch", "function"),
    ("gather_dispatch_config", "function"),
    ("est_copy_engine_us", "function"),
    ("est_gather_kernel_us", "function"),
    ("prefer_gather_kernel", "function"),
    ("P2PGatherPlan1D", "class"),
    ("P2PGatherPlan2D", "class"),
    ("p2p_kv_restore_layer", "function"),
    ("p2p_kv_restore_layer_twostage", "function"),
    ("Topology", "struct"),
    ("ring_rank", "function"),
    ("build_ring_topology", "function"),
]


class DiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.disc = discovery.discover(ROOT)
        self.kernels = {e.name: e for e in self.disc.kernels}
        self.comm = {e.name: e for e in self.disc.comm}

    def test_kernel_names_and_order(self):
        self.assertEqual(
            [(e.name, e.category) for e in self.disc.kernels],
            EXPECTED_KERNELS,
        )

    def test_kernel_kinds_and_backends(self):
        for name, _ in EXPECTED_KERNELS:
            e = self.kernels[name]
            self.assertEqual(e.kind, "kernel", name)
            self.assertTrue(e.host, f"{name} must have a CPU reference")
            self.assertTrue(e.cuda or e.hip,
                            f"{name} must have a GPU (CUDA or HIP) source")

    def test_moe_kernels_are_hip(self):
        for name in ("direct_lds_fill_bf16", "fp4_to_bf16_dequant",
                     "use_async_copy_default", "mfma_f32_16x16x16bf16",
                     "fused_moe_mxfp4_cpu", "moe_align_block_size",
                     "fused_moe_mxfp4"):
            self.assertTrue(self.kernels[name].hip, name)
            self.assertFalse(self.kernels[name].cuda, name)

    def test_comm_names_kinds_and_order(self):
        self.assertEqual(
            [(e.name, e.kind) for e in self.disc.comm],
            EXPECTED_COMM,
        )

    def test_comm_backends(self):
        self.assertTrue(self.comm["ring_allreduce"].cuda)
        self.assertTrue(self.comm["p2p_gather_runs"].cuda)
        self.assertFalse(self.comm["ring_rank"].cuda)
        self.assertFalse(self.comm["make_ring_channels"].cuda)
        self.assertFalse(self.comm["OverlapExecutor"].cuda)
        # Inline-in-header functions count as host implementations.
        self.assertTrue(self.comm["ring_rank"].host)
        self.assertTrue(self.comm["build_ring_topology"].host)
        self.assertTrue(self.comm["OverlapExecutor"].host)

    def test_cuda_only_header_skipped(self):
        names = [e.name for e in self.disc.comm]
        self.assertNotIn("p2p_gather_cuda", names)

    def test_descriptions(self):
        self.assertEqual(self.kernels["add"].description, "out = a + b")
        self.assertEqual(self.kernels["scale"].description, "out = alpha * x")
        self.assertEqual(self.kernels["relu"].description, "out = max(x, 0)")
        self.assertTrue(self.kernels["sum"].description.startswith(
            "Sum of all elements"))
        self.assertTrue(self.kernels["gemm"].description.startswith("SGEMM"))
        self.assertTrue(self.comm["ring_allreduce"].description.lower().find(
            "all-reduce") >= 0)
        self.assertTrue(self.comm["make_ring_channels"].description.lower().find(
            "mock channels") >= 0)

    def test_signatures(self):
        self.assertEqual(
            self.kernels["add"].signature,
            "add(Span<const float> a, Span<const float> b, Span<float> out)",
        )
        self.assertTrue(self.kernels["gemm"].signature.startswith("gemm("))
        self.assertIn("float alpha", self.kernels["gemm"].signature)
        self.assertTrue(
            self.comm["p2p_gather_runs"].signature.startswith("p2p_gather_runs("))

    def test_version_reads_cpp_header(self):
        self.assertRegex(discovery.version(ROOT), r"^\d+\.\d+\.\d+$")


def _run_vkl(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    return subprocess.run(
        [sys.executable, "-m", "vkernels.cli", *args],
        cwd=_SRC, env=env, capture_output=True, text=True,
    )


class CliTest(unittest.TestCase):
    def test_list_plain(self):
        p = _run_vkl("list")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("KERNELS", p.stdout)
        self.assertIn("gemm", p.stdout)
        self.assertIn("ring_allreduce", p.stdout)
        self.assertIn("out = a + b", p.stdout)
        self.assertIn("COMMUNICATION PRIMITIVES", p.stdout)

    def test_list_kernels_only(self):
        p = _run_vkl("list", "--kernels")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("gemm", p.stdout)
        self.assertNotIn("ring_allreduce", p.stdout)

    def test_list_comm_only(self):
        p = _run_vkl("list", "--comm")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ring_allreduce", p.stdout)
        self.assertNotIn("gemm", p.stdout)

    def test_list_filters(self):
        p = _run_vkl("list", "--cuda-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("gemm", p.stdout)
        self.assertNotIn("ring_rank", p.stdout)  # no .cu for topology
        p = _run_vkl("list", "--host-only")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("ring_rank", p.stdout)  # inline host impl

    def test_list_json(self):
        p = _run_vkl("list", "--json")
        self.assertEqual(p.returncode, 0, p.stderr)
        payload = json.loads(p.stdout)
        self.assertRegex(payload["version"], r"^\d+\.\d+\.\d+$")
        self.assertEqual(payload["root"], str(ROOT))
        by_name = {e["name"]: e for e in payload["kernels"]}
        self.assertIn("gemm", by_name)
        self.assertTrue(by_name["gemm"]["cuda"])
        self.assertTrue(by_name["gemm"]["host"])
        self.assertIn("description", by_name["gemm"])
        self.assertIn("signature", by_name["gemm"])
        comm_names = {e["name"] for e in payload["comm"]}
        self.assertIn("ring_allreduce", comm_names)
        self.assertIn("OverlapExecutor", comm_names)

    def test_info(self):
        p = _run_vkl("info", "gemm")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("name:        gemm", p.stdout)
        self.assertIn("category:    gemm", p.stdout)
        self.assertIn("header:      src/c/vkernels/kernels/gemm.hpp", p.stdout)

    def test_info_class_lookup(self):
        p = _run_vkl("info", "OverlapExecutor")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("kind:        class", p.stdout)

    def test_info_unknown(self):
        p = _run_vkl("info", "definitely_not_a_kernel")
        self.assertEqual(p.returncode, 2)
        self.assertIn("unknown kernel or primitive", p.stderr)

    def test_info_unknown_suggests(self):
        p = _run_vkl("info", "gema")
        self.assertEqual(p.returncode, 2)
        self.assertIn("did you mean", p.stderr)
        self.assertIn("gemm", p.stderr)

    def test_version(self):
        p = _run_vkl("--version")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertRegex(p.stdout.strip(), r"^vkl \d+\.\d+\.\d+$")

    def test_bare_invocation_prints_help(self):
        p = _run_vkl()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("usage: vkl", p.stdout)

    def test_main_returns_codes(self):
        self.assertEqual(cli_main.main(["info", "gemm"]), 0)
        self.assertEqual(cli_main.main(["info", "nope"]), 2)
        self.assertEqual(cli_main.main([]), 0)


if __name__ == "__main__":
    unittest.main()
