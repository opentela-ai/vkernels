"""Fresh native default and composed-width/merge witnesses against incumbent."""
import json
import os
from pathlib import Path
from runpy import run_path
import subprocess
import sys

import pytest
import torch

# The public witness also runs as a standalone frozen pytest file on ROCm.
CHILD = run_path(str(Path(__file__).with_name('test_hip_dsa_combine.py')))['CHILD']

LIBRARY = os.environ.get('VKERNELS_TEST_HIP_COMBINE_LIB')
BASE = os.environ.get('VKERNELS_TEST_HIP_STACK_BASE')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None or not LIBRARY or not BASE,
                                reason='Explicit candidate and incumbent HIP libraries required')


@pytest.mark.parametrize('gpu', range(4))
def test_default_and_composed_stack_match_incumbent(gpu, tmp_path):
    records = []
    for name, library, enabled in (('incumbent', BASE, False), ('rebuilt', LIBRARY, False), ('stack', LIBRARY, True)):
        folder = tmp_path / name
        env = dict(os.environ, VKERNELS_LIB=str(Path(library).resolve()),
                   VK_DSA_DB_WIDTH=str(int(enabled)), VK_DSA_COMBINE_TILED=str(int(enabled)),
                   VKERNELS_CACHE='off', VKERNELS_TUNING_CACHE='off', PYTHONDONTWRITEBYTECODE='1')
        for key in ('VK_DSA_DB', 'VK_DSA_VEC', 'VK_DSA_KTI', 'VK_DSA_DBPIPE', 'VK_DSA_SKIP_INVALID'):
            env.pop(key, None)
        result = subprocess.run([sys.executable, '-c', CHILD, str(gpu), str(folder)],
                                env=env, text=True, capture_output=True, timeout=180)
        if folder.exists():
            (folder / 'child.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        kernels = [e['name'] for e in json.loads((folder / 'trace.json').read_text())['traceEvents'] if e.get('cat') == 'kernel']
        partials = [n for n in kernels if 'dsa_sparse_fwd_partial_db_kernel' in n]
        combines = [n for n in kernels if ('dsa_sparse_fwd_combine_tiled_kernel' if enabled else 'dsa_sparse_fwd_combine_kernel') in n]
        assert len(partials) == len(combines) == 5
        expected = ('<1, 512, 0>', 'ILi1ELi512ELi0E') if enabled else ('<1>', 'ILi1E', '<1, 0, -1>', 'ILi1ELi0ELin1E')
        assert all(any(tag in n for tag in expected) for n in partials)
        records.append(torch.load(folder / 'outputs.pt', map_location='cpu', weights_only=True))
    assert all(len(r) == 12 for r in records)
    for original, rebuilt, stack in zip(*records, strict=True):
        for result in (rebuilt, stack):
            assert (original['valid'], original['phase']) == (result['valid'], result['phase'])
            for field in ('out', 'lse', 'out_without_lse'):
                assert torch.equal(original[field].view(torch.uint8), result[field].view(torch.uint8))
