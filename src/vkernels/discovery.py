"""Discovery of implemented vkernels by scanning the C++/CUDA sources.

``vkl list`` answers "what is implemented in this repository?" by parsing the
public headers under ``csrc/vkernels/kernels/`` (kernels) and
``csrc/vkernels/comm/`` (communication primitives) and pairing each declared
function or type with its implementations: a ``.cpp`` (CPU reference, always
built) and/or a ``.cu`` (CUDA, built when a toolkit is present).

The parser is intentionally small and is kept in lock-step with the repo's
header style: it strips comments, reads namespace-scope declarations, and
skips ``class``/``struct`` bodies while recording the types themselves.
``tests/python/test_discovery.py`` pins the exact contract (names, order,
descriptions, backends).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Headers that only expose implementation detail (e.g. the CUDA-only
# declarations in p2p_gather_cuda.hpp) are not public API and are skipped.
_SKIP_HEADER_SUFFIXES = ("_cuda.hpp",)

_KERNELS_DIR = "csrc/vkernels/kernels"
_COMM_DIR = "csrc/vkernels/comm"

_TYPE_START = re.compile(r"^\s*(class|struct)\s+([A-Za-z_]\w*)")
_NAME = re.compile(r"[A-Za-z_]\w*")
_KEYWORDS = frozenset(
    {"for", "while", "if", "else", "switch", "case", "return", "sizeof",
     "new", "delete", "this", "static_cast", "const_cast",
     "reinterpret_cast", "dynamic_cast"}
)

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
    cuda: bool         # a CUDA implementation file exists for this module


@dataclass(frozen=True)
class Discovery:
    root: Path
    kernels: list[Entry]
    comm: list[Entry]


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove C/C++ comments while preserving line structure."""
    text = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)),
                  text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        idx = line.find("//")
        lines.append(line[:idx] if idx >= 0 else line)
    return "\n".join(lines)


def _find_decl_paren(s: str) -> int:
    """Index of the first '(' at angle-bracket depth 0, or -1."""
    angle = 0
    for i, ch in enumerate(s):
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "(" and angle == 0:
            return i
    return -1


def _split_decl(stmt: str) -> tuple[str, str] | None:
    """Return ``(name, params)`` if ``stmt`` is a function declaration."""
    s = stmt.strip()
    open_i = _find_decl_paren(s)
    if open_i < 0:
        return None
    depth = 0
    close_i = -1
    for i in range(open_i, len(s)):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
            if depth == 0:
                close_i = i
                break
    if close_i < 0:
        return None
    tokens = s[:open_i].rstrip().split()
    if len(tokens) < 2:
        return None
    name = tokens[-1]
    ret = " ".join(tokens[:-1])
    if not _NAME.fullmatch(name) or name in _KEYWORDS:
        return None
    # Member calls / inline-junk prefixes ("out.reserve(...)", a namespace
    # opener merged into a statement, "= default", ...) are not declarations.
    if not ret or any(ch in ret for ch in ".(=){}"):
        return None
    return name, s[open_i:close_i + 1]


def _preceding_comment(raw_lines: list[str], start_1based: int) -> str:
    """Collect the contiguous ``//`` comment block directly above a line."""
    parts = []
    i = start_1based - 2
    while i >= 0:
        s = raw_lines[i].strip()
        if not s.startswith("//"):
            break
        text = s[2:].strip()
        if text:
            parts.append(text)
        i -= 1
    return " ".join(reversed(parts)).strip()


def _file_header_comment(raw: str) -> str:
    """First comment block of the file (skips the ``// path`` banner)."""
    parts = []
    for line in raw.splitlines():
        s = line.strip()
        if s.startswith("//"):
            text = s[2:].strip()
            if text and not text.startswith("vkernels/"):
                parts.append(text)
        elif not s:
            continue
        else:
            break
    return " ".join(parts).strip()


def _parse_header(header: Path, root: Path) -> list[Entry]:
    raw = header.read_text(encoding="utf-8", errors="replace")
    text = _strip_comments(raw)
    lines = text.splitlines()
    raw_lines = raw.splitlines()
    n = len(lines)
    category = "comm" if header.parent.name == "comm" else header.stem
    cuda = header.with_suffix(".cu").exists()
    cpp = header.with_suffix(".cpp").exists()

    def make(name, kind, signature, start, raw_stmt="") -> Entry:
        desc = _preceding_comment(raw_lines, start)
        if not desc:
            desc = _file_header_comment(raw)
        host = cpp or bool(raw_stmt and "{" in raw_stmt) or kind in ("class", "struct")
        return Entry(
            name=name,
            kind=kind,
            category=category,
            description=desc,
            signature=signature,
            header=str(header.relative_to(root)),
            host=host,
            cuda=cuda,
        )

    entries: list[Entry] = []
    i = 0
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if stripped.startswith("namespace") and stripped.endswith("{"):
            i += 1
            continue

        m = _TYPE_START.match(line)
        if m and "{" in line:
            kw, type_name = m.group(1), m.group(2)
            entries.append(make(type_name, kw, f"{kw} {type_name}", i + 1))
            depth = line.count("{") - line.count("}")
            j = i
            while depth > 0 and j + 1 < n:
                j += 1
                depth += lines[j].count("{") - lines[j].count("}")
            i = j + 1
            continue

        # Accumulate a namespace-scope statement terminated by ';' or '}'.
        buf = [line]
        raw_buf = [raw_lines[i]]
        j = i
        terminated = False
        while True:
            s = lines[j].strip()
            if s.endswith(";") or s in ("}", "};"):
                terminated = True
                break
            j += 1
            if j >= n:
                break
            buf.append(lines[j])
            raw_buf.append(raw_lines[j])
        if terminated:
            parsed = _split_decl(" ".join(x.strip() for x in buf))
            if parsed:
                name, params = parsed
                fn_kind = "kernel" if category != "comm" else "function"
                entries.append(make(name, fn_kind, f"{name}{params}", i + 1,
                                    raw_stmt="\n".join(raw_buf)))
        i = j + 1
    return entries


# ---------------------------------------------------------------------------
# Repository discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (default: this package) to the repo root."""
    start = (start or Path(__file__).resolve().parent).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "csrc" / "vkernels").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not find a vkernels repository root from {start} "
        "(expected a directory containing csrc/vkernels)"
    )


def resolve_root(override: str | None = None) -> Path:
    """Repository root: ``--root`` flag, ``VKERNELS_ROOT`` env, or auto."""
    if override:
        root = Path(override).expanduser().resolve()
    else:
        env = os.environ.get("VKERNELS_ROOT")
        root = Path(env).expanduser().resolve() if env else find_repo_root()
    if not (root / "csrc" / "vkernels").is_dir():
        raise FileNotFoundError(
            f"{root} is not a vkernels repository root (missing csrc/vkernels)"
        )
    return root


def discover(root: Path) -> Discovery:
    """Scan the kernels/ and comm/ headers under ``root``."""
    kernels: list[Entry] = []
    comm: list[Entry] = []
    for header in sorted((root / _KERNELS_DIR).glob("*.hpp")):
        if header.name.endswith(_SKIP_HEADER_SUFFIXES):
            continue
        kernels.extend(_parse_header(header, root))
    for header in sorted((root / _COMM_DIR).glob("*.hpp")):
        if header.name.endswith(_SKIP_HEADER_SUFFIXES):
            continue
        comm.extend(_parse_header(header, root))
    return Discovery(root=root, kernels=kernels, comm=comm)


def version(root: Path | None = None) -> str:
    """Version string from csrc/vkernels/util/version.hpp (single source)."""
    try:
        header = (root or resolve_root()) / "csrc" / "vkernels" / "util" / "version.hpp"
        m = _VERSION_RE.search(header.read_text(encoding="utf-8"))
        return m.group(1) if m else "0.0.0"
    except OSError:
        return "0.0.0"
