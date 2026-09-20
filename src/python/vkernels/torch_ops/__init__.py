"""Optional device-native Torch operators; importing this module loads no Torch.

Every public op validates its own eligibility (device/arch, dtype, shape,
contiguity — see :mod:`._dispatch`) and raises :class:`OpNotEligible` when the
fused path cannot take the inputs, so callers fall back instead of pre-gating.
"""

from ._dispatch import OpNotEligible
from .mhc_projection import mhc_projection, mhc_projection_reference

__all__ = ["OpNotEligible", "mhc_projection", "mhc_projection_reference"]
