"""Regression coverage for capability selection and shared compiler planning."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_requested_extension_never_silently_falls_back():
    env = dict(os.environ, VKERNELS_EXTENSION="/missing/_core.so", VKERNELS_BACKEND="compiled")
    result = subprocess.run(
        [sys.executable, "-c", "from vkernels._backend import load_extension; load_extension()"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "requested vkernels extension" in result.stderr


def test_catalog_works_without_repository(tmp_path):
    import shutil
    import vkernels

    package = Path(vkernels.__file__).parent
    shutil.copytree(package, tmp_path / "vkernels", ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    env = dict(os.environ, PYTHONPATH=str(tmp_path), VKERNELS_BACKEND="fallback")
    env.pop("VKERNELS_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-m", "vkernels.cli", "list", "--json"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    assert any(e["name"] == "gemm" for e in data["kernels"])


def test_k3_rejects_scalar_gate_before_native_access():
    from vkernels.kernels import kda_naive_delta_rule_fwd

    q = np.ones((1, 1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="g must have shape"):
        kda_naive_delta_rule_fwd(q, q, q, np.ones((1, 1, 2)), np.ones((1, 1, 2)))


def test_compiler_contract_registry_covers_all_execution_paths():
    from vkernels.compiler.contracts import CONTRACTS
    from vkernels.compiler.lowerings import LOWERINGS
    from vkernels.compiler.operator_ir import ARITHMETIC_OP_KINDS
    from vkernels.compiler.reference_exec import ReferenceExecutor
    from vkernels.compiler.codegen_cute import TEMPLATE_NAMES

    assert set(CONTRACTS) == set(LOWERINGS) == set(ARITHMETIC_OP_KINDS)
    for spec in CONTRACTS.values():
        assert callable(getattr(ReferenceExecutor, spec.reference_body))
        assert spec.task_kind in TEMPLATE_NAMES


def test_prepared_device_uses_compiler_workspace_and_rejects_oversubscription():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vkernels.compiler import compile_model
    from vkernels.compiler.model_qwen3 import tiny_qwen3_config
    from vkernels.compiler.device_triton import TritonMegakernel

    exe = compile_model(model_config=tiny_qwen3_config(), workers=2)
    runner = exe.prepare_device()
    assert runner.ws.numel() == exe.workspace_plan.total_elements
    ids = torch.tensor([3], dtype=torch.int64, device="cuda")
    positions = torch.tensor([0], dtype=torch.int32, device="cuda")
    fast = runner.run_device(ids, positions).clone()
    checked = runner.run([3], [0]).clone()
    torch.testing.assert_close(fast, checked)
    with pytest.raises(ValueError, match="residency bound"):
        TritonMegakernel(
            exe.config,
            exe.weights,
            capacity=exe.config.cache_capacity,
            workers=torch.cuda.get_device_properties(0).multi_processor_count + 1,
        )


def test_graph_replays_refresh_barrier_base():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from vkernels.compiler import compile_model
    from vkernels.compiler.model_qwen3 import tiny_qwen3_config

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        exe = compile_model(model_config=tiny_qwen3_config(), workers=2)
        runner = exe.prepare_device()
        ids = torch.tensor([3], dtype=torch.int64, device="cuda")
        positions = torch.tensor([0], dtype=torch.int32, device="cuda")
        expected = runner.run_device(ids, positions).clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            runner.run_device(ids, positions)
        for _ in range(5):
            graph.replay()
            torch.testing.assert_close(runner.logits, expected)
        assert runner.barrier_base == runner.num_barriers * runner.workers * 6


def test_concurrent_tuning_writers_merge_records(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from vkernels.torch_ops.tuning_cache import TuningCache
    from vkernels.torch_ops.tuner import NativeStore

    fingerprint = {"arch": "test", "name": "test", "cu_count": 1, "software": {}}

    def write(index):
        TuningCache("parallel", store_dir=tmp_path, device=fingerprint).record(
            (index,), kwargs={"BLOCK": 64}, num_warps=1, num_stages=1, time_ms=1.0
        )
        NativeStore("parallel", store_dir=tmp_path, arch="test").upsert((index,), {"split": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(32)))
    cache = TuningCache("parallel", store_dir=tmp_path, device=fingerprint)
    for index in range(32):
        assert cache.lookup((index,)) is not None
    assert len(NativeStore("parallel", store_dir=tmp_path, arch="test").records()) == 32
