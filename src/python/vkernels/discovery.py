"""Read the shipped API/capability catalog, inside or outside a checkout.

Implementation flags describe individual entry points, not neighboring files.
Build and loadability status are reported separately by ``vkl doctor``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

_VERSION_RE = re.compile(r'#define\s+VKERNELS_VERSION_STRING\s+"([^"]+)"')


@dataclass(frozen=True)
class Entry:
    """One implemented kernel or communication primitive."""

    name: str
    kind: str          # "kernel" | "function" | "class" | "struct"
    category: str      # elementwise / reduce / gemm / comm
    description: str
    signature: str
    header: str        # repo-relative path to the declaring header
    host: bool         # CPU reference implementation exists (.cpp or inline)
    cuda: bool         # a CUDA implementation file (.cu) exists for this module
    hip: bool          # reviewed HIP entry point
    contract_test: str | None = None


@dataclass(frozen=True)
class Discovery:
    root: Path
    kernels: list[Entry]
    comm: list[Entry]


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this package) to the repo root."""
    start = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src" / "c" / "vkernels").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find a vkernels repository root from {start} "
        "(expected a directory containing src/c/vkernels)"
    )


def resolve_root(override: str | None = None) -> Path:
    """Repository root: ``--root`` flag, ``VKERNELS_ROOT`` env, or auto."""
    if override:
        root = Path(override).expanduser().resolve()
    else:
        env = os.environ.get("VKERNELS_ROOT")
        root = Path(env).expanduser().resolve() if env else _default_root()
    if not (root / "src" / "c" / "vkernels").is_dir() and root != Path(__file__).parent:
        raise FileNotFoundError(
            f"{root} is not a vkernels repository root (missing src/c/vkernels)"
        )
    return root


def _default_root() -> Path:
    try:
        return find_repo_root()
    except FileNotFoundError:
        return Path(__file__).parent


def discover(root: Path) -> Discovery:
    """Read the shipped, reviewed API catalog (also available outside a checkout)."""
    data = json.loads(Path(__file__).with_name("catalog.json").read_text())
    return Discovery(root, [Entry(**e) for e in data["kernels"]], [Entry(**e) for e in data["comm"]])


def version(root: Path | None = None) -> str:
    """Version string from src/c/vkernels/util/version.hpp (single source)."""
    try:
        header = (root or resolve_root()) / "src" / "c" / "vkernels" / "util" / "version.hpp"
        m = _VERSION_RE.search(header.read_text(encoding="utf-8"))
        return m.group(1) if m else "0.0.0"
    except OSError:
        return json.loads(Path(__file__).with_name("catalog.json").read_text())["version"]
