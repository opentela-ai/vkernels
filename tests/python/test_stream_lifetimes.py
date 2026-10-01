"""Timeout-isolated regressions: a GIL/worker deadlock must fail, not hang CI."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from vkernels import _backend


@pytest.mark.parametrize("backend", ["fallback", "compiled"])
@pytest.mark.parametrize("scenario", ["drop_pending", "drop_after_wait", "failure", "overlap_failure"])
def test_worker_lifetimes(backend, scenario):
    extension = _backend.load_extension() if backend == "compiled" else None
    if backend == "compiled" and extension is None:
        pytest.skip("compiled Python backend required")
    source = Path(__file__).resolve().parents[2] / "src" / "python"
    script = textwrap.dedent('''
        import gc
        import importlib.util
        import sys
        import time
        import weakref

        if sys.argv[1] == "compiled":
            spec = importlib.util.spec_from_file_location("vkernels._core", sys.argv[3])
            core = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(core)
            Stream, Overlap = core.core.Stream, core.comm.OverlapExecutor
        else:
            from vkernels._fallback import Stream, OverlapExecutor as Overlap

        def fail():
            raise ValueError("callback failure")

        scenario = sys.argv[2]
        if scenario == "overlap_failure":
            for failing_stage in ("compute", "comm"):
                ex = Overlap()
                computed, communicated = [], []
                def compute(i):
                    computed.append(i)
                    if failing_stage == "compute" and i == 1:
                        fail()
                    return i
                def comm(i, value):
                    communicated.append(i)
                    if failing_stage == "comm" and i == 1:
                        fail()
                try:
                    ex.run(5, compute, comm)
                except ValueError as e:
                    assert str(e) == "callback failure"
                else:
                    raise AssertionError("callback failure was swallowed")
                assert len(computed) == 5
                assert len(communicated) == (4 if failing_stage == "compute" else 5)
                ex.run(1, lambda i: 0, lambda i, v: None)
        else:
            stream = Stream()
            completed = []
            class Callback:
                def __call__(self):
                    time.sleep(0.02)
                    completed.append(1)
            callback = Callback()
            ref = weakref.ref(callback)
            if scenario == "failure":
                stream.submit(fail)
            stream.submit(callback)
            del callback
            if scenario == "drop_pending":
                if sys.argv[1] == "fallback":
                    stream.close()
                del stream
            else:
                try:
                    stream.wait()
                except ValueError as e:
                    assert scenario == "failure"
                    assert str(e) == "callback failure"
                else:
                    assert scenario != "failure"
                assert ref() is None
                stream.wait()
                if sys.argv[1] == "fallback":
                    stream.close()
                del stream
            gc.collect()
            assert completed == [1]
            assert ref() is None
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script, backend, scenario,
         str(extension.__file__) if extension else ""],
        env={**os.environ, "PYTHONPATH": str(source)},
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
