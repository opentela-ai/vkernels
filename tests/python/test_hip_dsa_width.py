"""Fresh-process public ABI parity, including proof of the selected GPU kernel.

Set VKERNELS_TEST_HIP_WIDTH_LIB to a candidate build or the isolated overlay
fixture; a library without the specialized kernel fails the trace assertion.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

LIBRARY = os.environ.get('VKERNELS_TEST_HIP_WIDTH_LIB')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is None or not LIBRARY,
                                reason='ROCm GPU and explicit candidate library required')

CHILD = r'''
import math,sys,time
from pathlib import Path
import torch
sys.meta_path[:]=[f for f in sys.meta_path if 'ScikitBuildRedirectingFinder' not in type(f).__name__]
from vkernels.hip_dsa_mhc import dsa_sparse_fwd_split
gpu,folder=int(sys.argv[1]),Path(sys.argv[2]);folder.mkdir()
torch.cuda.set_device(gpu);torch.manual_seed(5721);torch.set_num_threads(1)
q=torch.randn(1,1,16,512,dtype=torch.bfloat16).cuda()
kv=torch.randn(1,4352,1,512,dtype=torch.bfloat16).cuda()*.125
ids=torch.full((1,1,1,2051),-1,device='cuda',dtype=torch.int32)
outputs=[]
saved_q,saved_kv=q.clone(),kv.clone()
for valid in (128,640,2051):
 q.copy_(saved_q);kv.copy_(saved_kv)
 ids.fill_(-1);ids[...,:valid-3]=torch.arange(valid-3,device='cuda',dtype=torch.int32)
 ids[...,-3:]=torch.arange(4349,4352,device='cuda',dtype=torch.int32)
 saved_ids=ids.clone()
 def call():
  return dsa_sparse_fwd_split(q,kv,ids,dim=512,tail_dim=0,topk=2051,split_kv=32,
                              sm_scale=.0625*math.log2(math.e),return_lse=True)
 for _ in range(3):call()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out,lse=call()
 for phase in range(4):
  if phase==1:q.mul_(.875);kv.mul_(1.0625);ids.copy_(torch.roll(ids,37,-1))
  elif phase==2:ids.fill_(-1)
  elif phase==3:q.zero_();ids.copy_(saved_ids)
  graph.replay();torch.cuda.synchronize()
  assert torch.isfinite(out).all()
  outputs.append({'valid':valid,'phase':phase,'out':out.cpu().clone(),'lse':lse.cpu().clone()})
 if valid==128:
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
   time.sleep(.1)
   for _ in range(5):graph.replay()
   torch.cuda.synchronize();time.sleep(.1)
  profile.export_chrome_trace(str(folder/'trace.json'))
torch.save(outputs,folder/'outputs.pt')
'''


@pytest.mark.parametrize('gpu', range(4))
def test_public_width_dispatch_preserves_graph_results_and_selects_kernel(gpu, tmp_path):
    if gpu >= torch.cuda.device_count():
        pytest.skip('GPU index absent')
    library = Path(LIBRARY).resolve(strict=True)
    records = []
    for enabled in (False, True):
        folder = tmp_path / ('on' if enabled else 'off')
        env = dict(os.environ, VKERNELS_LIB=str(library), VK_DSA_DB_WIDTH=str(int(enabled)),
                   VKERNELS_CACHE='off', VKERNELS_TUNING_CACHE='off', PYTHONDONTWRITEBYTECODE='1')
        for key in ('VK_DSA_DB', 'VK_DSA_VEC', 'VK_DSA_KTI', 'VK_DSA_DBPIPE', 'VK_DSA_SKIP_INVALID'):
            env.pop(key, None)
        result = subprocess.run([sys.executable, '-c', CHILD, str(gpu), str(folder)],
                                env=env, text=True, capture_output=True, timeout=180)
        if folder.exists():
            (folder / 'child.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        events = json.loads((folder / 'trace.json').read_text())['traceEvents']
        names = [e['name'] for e in events if e.get('cat') == 'kernel'
                 and 'dsa_sparse_fwd_partial_db_kernel' in e.get('name', '')]
        assert len(names) == 5, 'Incomplete five-replay native GPU coverage: ' + repr(names)
        combines = [e for e in events if e.get('cat') == 'kernel'
                    and 'dsa_sparse_fwd_combine_kernel' in e.get('name', '')]
        assert len(combines) == 5, 'Incomplete five-replay combine coverage'
        # Template args may carry a trailing SkipInvalid flag, so match the prefix.
        expected = ('<1, 512, 0>', '<1, 512, 0,', 'ILi1ELi512ELi0E') if enabled else ('<1, 0, -1>', '<1, 0, -1,', 'ILi1ELi0ELin1E')
        assert all(any(part in name for part in expected) for name in names), names
        records.append(torch.load(folder / 'outputs.pt', map_location='cpu', weights_only=True))
    assert len(records[0]) == len(records[1]) == 12
    for old, new in zip(*records, strict=True):
        assert (old['valid'], old['phase']) == (new['valid'], new['phase'])
        for name in ('out', 'lse'):
            assert torch.equal(old[name].view(torch.uint8), new[name].view(torch.uint8))
