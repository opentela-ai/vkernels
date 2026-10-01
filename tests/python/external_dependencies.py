"""Explicit integration dependencies; no synthetic modules or personal paths."""
import os
import sys
from pathlib import Path

import pytest


def require_floe():
    if os.environ.get("VKERNELS_TEST_INTEGRATION") != "1":
        pytest.skip("external integration requires --run-integration")
    root = os.environ.get("FLOE_ROOT")
    if root:
        path = str(Path(root).resolve(strict=True))
        if path not in sys.path:
            sys.path.insert(0, path)
    # Explicit integration runs fail when a required dependency is missing.
    __import__("floe")
