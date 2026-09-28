"""Deterministic backend selection, with inspectable failure reasons.

Installed extensions take precedence. Development builds are opt-in through
VKERNELS_EXTENSION=/absolute/path/_core...so. VKERNELS_BACKEND is auto,
compiled (failure is fatal), or fallback (never attempts a native load).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

_loaded = None
_tried = False
_status = {"backend": "fallback", "path": None, "reason": "not loaded"}


def load_extension():
    global _loaded, _tried
    if _tried:
        return _loaded
    mode = os.environ.get("VKERNELS_BACKEND", "auto")
    if mode not in {"auto", "compiled", "fallback"}:
        raise ValueError("VKERNELS_BACKEND must be auto, compiled, or fallback")
    if mode == "fallback":
        _tried = True
        _status["reason"] = "explicit fallback selection"
        return None
    path = os.environ.get("VKERNELS_EXTENSION")
    try:
        if path:
            path = str(Path(path).resolve(strict=True))
            spec = importlib.util.spec_from_file_location("vkernels._core", path)
            _loaded = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_loaded)
        else:
            _loaded = importlib.import_module("vkernels._core")
    except (ImportError, OSError, RuntimeError) as exc:
        _loaded = None
        _status.update(path=path, reason=f"{type(exc).__name__}: {exc}")
        if mode == "compiled" or path:
            raise ImportError(f"requested vkernels extension could not load: {exc}") from exc
    else:
        _status.update(backend="compiled", path=_loaded.__file__, reason=None)
    _tried = True
    return _loaded


def backend_name() -> str:
    return "compiled" if load_extension() is not None else "fallback"


def backend_status() -> dict:
    load_extension()
    return dict(_status)
