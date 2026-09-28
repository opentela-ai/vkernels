"""One explicit resolution contract for packaged and development native libraries."""

from __future__ import annotations

import json
import os
from pathlib import Path


def find_library(names=("libvkernels_c.so", "libvkernels_hip.so")) -> str | None:
    override = os.environ.get("VKERNELS_LIB")
    if override:
        return str(Path(override).expanduser().resolve(strict=True))
    locations = [Path(__file__).parent]
    build = os.environ.get("VKERNELS_BUILD_DIR")
    if build:
        locations.append(Path(build) / "src" / "c")
    k3 = os.environ.get("K3")
    if k3:
        locations.append(Path(k3) / "home" / "pylib")
    for directory in locations:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate.resolve())
    return None


def doctor() -> dict:
    from ._backend import backend_status

    status = backend_status()
    info_path = (
        Path(status["path"]).parent / "build_info.json"
        if status["path"]
        else Path(__file__).with_name("build_info.json")
    )
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    library = find_library()
    loadable = False
    reason = None
    if library:
        import ctypes

        try:
            ctypes.CDLL(library)
            loadable = True
        except OSError as exc:
            reason = str(exc)
    return {
        "python": status,
        "native": {"path": library, "built": info, "loadable": loadable, "reason": reason},
        "device_validation": "unknown; run device contract tests on this device",
        "catalog": str(Path(__file__).with_name("catalog.json")),
        "ofi_transport": "not implemented; plugin disabled",
    }
