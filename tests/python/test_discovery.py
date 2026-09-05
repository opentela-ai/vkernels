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
    # DSA sparse-MLA forward (issue #51): the two-implementation model is
    # dsa_sparse_fwd_cpu (always compiled) + the HIP dsa_sparse_fwd[_with_tile].
    ("dsa_sparse_fwd_cpu", "dsa"),
    ("dsa_config_for", "dsa"),
    # DSA paged-MQA gated top-k logits (issue #51, the kpool>1 indexer path):
    # the FIRST stage feeding dsa_sparse_fwd. dsa_topk_logits_cpu is the
    # always-compiled host reference; dsa_topk_logits_fits_lds is the
    # host-callable 64 KB non-optin LDS-cap guard selecting the launcher's
    # fp32-Q fast path (verified at H=32); dsa_topk_logits_fits_lds_fp8q is
    # the fallback guard for larger H (Q staged raw fp8, dequantised on the
    # fly -- bit-identical output); dsa_topk_logits_fits_lds_mfma is the
    # bf16-MFMA fast path guard (smallest footprint, Matrix-Core). The HIP
    # dsa_topk_logits and dsa_topk_logits_with_variant are declared in the
    # #if VKERNELS_HAS_HIP block, so they sort after dsa_sparse_fwd_with_tile.
    ("dsa_topk_logits_cpu", "dsa"),
    ("dsa_topk_logits_fits_lds", "dsa"),
    ("dsa_topk_logits_fits_lds_fp8q", "dsa"),
    ("dsa_topk_logits_fits_lds_mfma", "dsa"),
    # The optimal split_kv for the HIP dsa_topk_logits indexer (issue
    # #51), the single source of truth for the formula the hip_capi.hpp
    # docstring used to restate (with the wrong NUM_CU=256 -> 228).
    ("dsa_topk_logits_split_for", "dsa"),
    ("dsa_sparse_fwd", "dsa"),
    ("dsa_sparse_fwd_with_tile", "dsa"),
    ("dsa_topk_logits", "dsa"),
    ("dsa_topk_logits_with_variant", "dsa"),
    # DSA kpool-cache compress/write (issue #60), declared in dsa_kpool.hpp
    # (file sorts after dsa.hpp and before dsa_topk.hpp, so the dsa_kpool
    # block sits between the dsa and dsa_topk blocks):
    ("dsa_kpool_group_topk_supported", "dsa_kpool"),
    ("dsa_kpool_max_closed_pools", "dsa_kpool"),
    ("dsa_kpool_assemble_cpu", "dsa_kpool"),
    ("dsa_kpool_decode_update_cpu", "dsa_kpool"),
    # Issue #61: fp8+scale store variants (CPU references + HIP kernels).
    ("dsa_kpool_assemble_fp8_cpu", "dsa_kpool"),
    ("dsa_kpool_decode_update_fp8_cpu", "dsa_kpool"),
    ("dsa_kpool_assemble", "dsa_kpool"),
    ("dsa_kpool_decode_update", "dsa_kpool"),
    ("dsa_kpool_assemble_fp8", "dsa_kpool"),
    ("dsa_kpool_decode_update_fp8", "dsa_kpool"),
    # DSA radix top-k transform (issue #56), declared in dsa_topk.hpp
    # (file sorts after dsa.hpp, so the dsa_topk block follows the dsa block):
    ("dsa_topk_transform_group_topk_supported", "dsa_topk"),
    ("dsa_topk_transform_cpu", "dsa_topk"),
    ("dsa_topk_transform", "dsa_topk"),
    ("add", "elementwise"),
    ("scale", "elementwise"),
    ("relu", "elementwise"),
    ("gemm", "gemm"),
    ("gemm_bf16_cpu", "gemm_bf16"),
    ("gemm_bf16_config_for", "gemm_bf16"),
    ("gemm_bf16", "gemm_bf16"),
    ("kda_layer_norm_gated_cpu", "kda"),
    ("kda_gate_chunk_cumsum_cpu", "kda"),
    ("kda_naive_delta_rule_fwd_cpu", "kda"),
    ("kda_delta_rule_intra_cpu", "kda"),
    ("kda_delta_rule_inter_cpu", "kda"),
    ("kda_gla_fwd_o_cpu", "kda"),
    ("kda_delta_rule_fwd_cpu", "kda"),
    ("kda_pack_bitmatrix_cpu", "kda"),
    ("kda_layer_norm_gated", "kda"),
    ("kda_gate_chunk_cumsum", "kda"),
    ("kda_delta_rule_fwd", "kda"),
    ("kda_delta_rule_fwd_with_scratch", "kda"),
    ("kda_pack_bitmatrix", "kda"),
    ("mhc_pre_gemm_sqrsum_cpu", "mhc"),
    ("mhc_post_cpu", "mhc"),
    ("mhc_pre_gemm_sqrsum", "mhc"),
    ("mhc_post", "mhc"),
    ("mla_fwd_cpu", "mla"),
    ("mla_config_for", "mla"),
    ("mla_fwd", "mla"),
    ("mla_fwd_with_tile", "mla"),
    ("direct_lds_fill_bf16", "moe"),
    ("fp4_to_bf16_dequant", "moe"),
    ("use_async_copy_default", "moe"),
    ("mfma_f32_16x16x16bf16", "moe"),
    ("mxfp4_moe_quant", "moe_aux"),
    ("mxfp4_moe_sort", "moe_aux"),
    ("mxfp4_moe_sort_scales", "moe_aux"),
    ("mxfp4_moe_scatter_reduce", "moe_aux"),
    ("mxfp4_moe_scatter_reduce_q", "moe_aux"),
    ("fused_moe_mxfp4_cpu", "moe_fused"),
    ("moe_gateup_cpu", "moe_fused"),
    ("moe_act_epilogue_cpu", "moe_fused"),
    ("moe_down_cpu", "moe_fused"),
    ("moe_combine_cpu", "moe_fused"),
    ("moe_align_block_size", "moe_fused"),
    ("dequant_weight_tile", "moe_fused"),
    ("dequant_weight_tile_ref", "moe_fused"),
    ("moe_gateup_preact", "moe_fused"),
    ("moe_act_epilogue", "moe_fused"),
    ("moe_down_preact", "moe_fused"),
    ("moe_combine", "moe_fused"),
    ("fused_moe_mxfp4", "moe_fused"),
    # on-device moe_align_block_size (issue #46 follow-up): declared after
    # fused_moe_mxfp4 in the VKERNELS_HAS_HIP block at the end of
    # moe_fused.hpp, so discovery emits it as the last moe_fused entry.
    ("moe_align_block_size_hip", "moe_fused"),
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
    # --- cross-node KV transfer plans + fabric import (issue #49) ---
    # cross_node_kv.hpp: the byte channel the host-bounce path runs over
    # (discovery scans *.hpp in comm/ sorted by filename, so this block
    # lands between channel.hpp and overlap.hpp).
    ("ByteBlockingQueue", "class"),
    ("ByteChannel", "class"),
    ("MockByteChannel", "class"),
    ("make_byte_link", "function"),
    # cross_node_kv.hpp (issue #49, all-gather + access-pattern routing):
    # the access pattern + route selection are declared ABOVE the
    # restore/donate plans in the header, so discovery reports them first.
    ("CrossNodeKvAccess", "struct"),
    ("CrossNodeKvRoute", "struct"),
    ("select_cross_node_kv_route", "function"),
    ("CrossNodeKvRestorePlan", "class"),
    ("CrossNodeKvDonatePlan", "class"),
    # fabric_import.hpp: the transport classification + FabricHandle /
    # FabricImport + the per-hop cost model.
    ("fabric_import_transport_name", "function"),
    ("FabricImportConfig", "struct"),
    ("classify_fabric_import", "function"),
    ("is_import_graph_capturable", "function"),
    ("eager_break_fabric_import", "function"),
    ("FabricHandle", "class"),
    ("FabricImport", "class"),
    ("CrossNodeHopCost", "struct"),
    ("cross_node_kv_throughput", "function"),
    ("same_node_fabric_roof_gbps", "function"),
    ("kv_gather_layer", "function"),
    ("kv_scatter_layer", "function"),
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
    # --- fused indexed-KV-to-peer donation (issue #36) ---
    ("p2p_kv_donate_layer", "function"),
    ("p2p_kv_donate_layer_twostage", "function"),
    ("kv_gather", "function"),
    ("set_donate_dispatch", "function"),
    ("donate_dispatch_config", "function"),
    ("est_direct_store_us", "function"),
    ("est_copy_engine_donate_us", "function"),
    ("prefer_direct_store", "function"),
    ("P2PKvDonatePlan", "class"),
    ("p2p_kv_restore_layer", "function"),
    ("p2p_kv_restore_layer_twostage", "function"),
    ("kv_scatter", "function"),
    ("from_device_slots_t", "struct"),
    ("from_device_slots_int64_t", "struct"),
    ("P2PKvRestorePlan", "class"),
    # --- graph-capturable PP-boundary transfer (issue #10) ---
    ("pipeline_transport_name", "function"),
    ("PipelineBoundaryConfig", "struct"),
    ("classify_boundary", "function"),
    ("is_graph_capturable", "function"),
    ("eager_break_during_capture", "function"),
    ("GraphCapture", "class"),
    ("PipelineBoundaryPlan", "class"),
    # --- HIP/RCCL transport + OFI/CXI plugin (issue #19) ---
    # Host reference (rccl.hpp, always compiled) then the HIP-gated
    # declarations (rccl_hip.hpp); discovery pins both since it scans every
    # *.hpp in comm/ regardless of VKERNELS_HAS_HIP (mirrors the existing
    # P2PGatherPlan1D/2D pin which live in the CUDA-gated p2p_gather_cuda.hpp).
    ("transport_name", "function"),
    ("RcclTransportConfig", "struct"),
    ("is_cuda_built_plugin", "function"),
    ("resolve_rccl_transport", "function"),
    ("est_rccl_socket_us", "function"),
    ("est_rccl_ofi_us", "function"),
    ("prefer_slingshot_rccl", "function"),
    ("resolve_transport", "function"),
    ("NodeTopology", "struct"),
    ("build_cross_node_ring", "function"),
    ("inter_node_ring_edges", "function"),
    ("cross_node_hops", "function"),
    ("OfiCxiInfo", "struct"),
    ("discover_ofi_cxi", "function"),
    ("RcclAllreducePlan", "class"),
    ("RcclComm", "class"),
    ("make_rccl_comms", "function"),
    ("make_rccl_comms_from_uniqueid", "function"),
    ("rccl_unique_id", "function"),
    ("RcclChannel", "class"),
    ("RcclRingChannels", "struct"),
    ("make_rccl_ring_channels", "function"),
    ("HipStream", "class"),
    ("RcclAllreducePlanHip", "class"),
    # --- shared slot-map validators + byte accounting (issue #49) ---
    # slot_map.hpp (lifted in 181d9f9 to dedup the restore/donate/all-gather
    # plan contracts) sorts after rccl_hip.hpp and before topology.hpp, so
    # its six inline helpers land here.
    ("per_slot_bytes", "function"),
    ("token_stride_bytes", "function"),
    ("checked_mul", "function"),
    ("validate_kv_plan_shape", "function"),
    ("validate_unique_slots", "function"),
    ("validate_slot_bounds", "function"),
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
            self.assertTrue(
                e.cuda or e.hip, f"{name} must have a GPU (CUDA or HIP) source"
            )

    def test_moe_kernels_are_hip(self):
        for name in (
            "direct_lds_fill_bf16",
            "fp4_to_bf16_dequant",
            "use_async_copy_default",
            "mfma_f32_16x16x16bf16",
            "fused_moe_mxfp4_cpu",
            "moe_align_block_size",
            "moe_align_block_size_hip",
            "fused_moe_mxfp4",
            "mxfp4_moe_quant",
            "mxfp4_moe_sort",
            "mxfp4_moe_sort_scales",
            "mxfp4_moe_scatter_reduce",
            "mxfp4_moe_scatter_reduce_q",
        ):
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
        self.assertTrue(
            self.kernels["sum"].description.startswith("Sum of all elements")
        )
        self.assertTrue(self.kernels["gemm"].description.startswith("SGEMM"))
        self.assertTrue(
            self.comm["ring_allreduce"].description.lower().find("all-reduce") >= 0
        )
        self.assertTrue(
            self.comm["make_ring_channels"].description.lower().find("mock channels")
            >= 0
        )

    def test_signatures(self):
        self.assertEqual(
            self.kernels["add"].signature,
            "add(Span<const float> a, Span<const float> b, Span<float> out)",
        )
        self.assertTrue(self.kernels["gemm"].signature.startswith("gemm("))
        self.assertIn("float alpha", self.kernels["gemm"].signature)
        self.assertTrue(
            self.comm["p2p_gather_runs"].signature.startswith("p2p_gather_runs(")
        )

    def test_version_reads_cpp_header(self):
        self.assertRegex(discovery.version(ROOT), r"^\d+\.\d+\.\d+$")


def _run_vkl(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(_SRC)}
    return subprocess.run(
        [sys.executable, "-m", "vkernels.cli", *args],
        cwd=_SRC,
        env=env,
        capture_output=True,
        text=True,
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
