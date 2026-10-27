"""Growing MTP contexts reuse a binary while target static geometry stays fixed."""
import pytest
import torch
from vkernels.torch_ops.glm_indexer_finalize_native import _kernel, glm_indexer_native_finalize
from vkernels.torch_ops._dispatch import OpNotEligible


def test_dynamic_geometry_does_not_specialize_lengths_pools_or_strides():
    static = _kernel(False)
    dynamic = _kernel(True)
    assert static.constexprs == list(range(6, 21))
    assert dynamic.constexprs == [8, 19, 20]  # output width and reduction tiles
    assert set(dynamic.do_not_specialize) == {
        'K', 'L', 'IB', 'IK', 'IS', 'SB', 'SK', 'VB', 'VL', 'KB', 'KL', 'MB'}
    assert _kernel(False) is static and _kernel(True) is dynamic
    assert static is not dynamic


@pytest.mark.parametrize('option', [None, 1, 'true'])
def test_dynamic_geometry_requires_an_explicit_boolean(option):
    with pytest.raises(OpNotEligible, match='must be boolean'):
        glm_indexer_native_finalize(torch.empty(1, 1, 0, 4, dtype=torch.int64),
                                    torch.empty(1, 1, 0, dtype=torch.bool),
                                    torch.ones(1, 1, 1, dtype=torch.bool),
                                    torch.ones(1, 1, dtype=torch.bool),
                                    torch.ones(1, 1, dtype=torch.bool),
                                    dynamic_geometry=option)
