"""Optional device-native Torch operators; importing this module loads no Torch."""

from .mhc_projection import mhc_projection, mhc_projection_reference

__all__ = ["mhc_projection", "mhc_projection_reference"]
