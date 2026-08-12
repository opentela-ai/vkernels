"""vkl — list and inspect the kernels implemented in the vkernels repository.

Usage (from the repo):

    python3 -m vkernels.cli list            # list kernels + comm primitives
    python3 -m vkernels.cli info gemm       # details for one entry
    python3 -m vkernels.cli --version

``pip install -e ./src`` installs a ``vkl`` console script with the same
behaviour. The repository root is auto-detected (override with ``--root`` or
the ``VKERNELS_ROOT`` environment variable).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

from vkernels import discovery

_DESC_WIDTH = 72


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vkl",
        description="List and inspect the kernels implemented in the "
                    "vkernels repository (src/c/vkernels).",
    )
    p.add_argument("--version", action="version",
                   version=f"vkl {discovery.version()}")
    p.add_argument("--root", metavar="DIR", default=None,
                   help="vkernels repository root (default: auto-detected)")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    list_p = sub.add_parser(
        "list", help="list implemented kernels and communication primitives")
    group = list_p.add_mutually_exclusive_group()
    group.add_argument("--kernels", action="store_true",
                       help="only kernels (src/c/vkernels/kernels)")
    group.add_argument("--comm", action="store_true",
                       help="only communication primitives (src/c/vkernels/comm)")
    list_p.add_argument("--cuda-only", action="store_true",
                        help="only entries with a CUDA implementation")
    list_p.add_argument("--host-only", action="store_true",
                        help="only entries with a host implementation")
    list_p.add_argument("--json", action="store_true",
                        help="machine-readable JSON output")

    info_p = sub.add_parser(
        "info", help="show details for one kernel or primitive")
    info_p.add_argument("name", help="kernel or primitive name")
    return p


def _truncate(s: str, width: int) -> str:
    if len(s) <= width:
        return s
    return s[: width - 3].rstrip() + "..."


def _print_section(title: str, entries: list, extra_label: str | None,
                   extra_getter) -> None:
    name_w = max(len(e.name) for e in entries)
    extra_w = max(len(extra_getter(e)) for e in entries) if extra_label else 0
    descs = [_truncate(e.description, _DESC_WIDTH) for e in entries]
    desc_w = max(len(d) for d in descs)

    labels = ["NAME", extra_label if extra_label else None, "CPU", "CUDA",
              "HIP", "DESCRIPTION"]
    widths = [name_w, extra_w if extra_label else 0, 3, 4, 3, desc_w]
    cols = [(lab, w) for lab, w in zip(labels, widths) if lab is not None]

    header = "  ".join(f"{lab:<{w}}" for lab, w in cols)
    print(title)
    print(header)
    print("-" * len(header))
    for e, d in zip(entries, descs):
        cells = [e.name]
        if extra_label:
            cells.append(extra_getter(e))
        cells += ["yes" if e.host else "no", "yes" if e.cuda else "no",
                  "yes" if e.hip else "no", d]
        print("  ".join(f"{c:<{w}}" for c, w in zip(cells, [w for _, w in cols])))


def _entry_dict(e: discovery.Entry) -> dict:
    return {
        "name": e.name,
        "kind": e.kind,
        "category": e.category,
        "description": e.description,
        "signature": e.signature,
        "host": e.host,
        "cuda": e.cuda,
        "hip": e.hip,
        "header": e.header,
    }


def _cmd_list(root: Path, args: argparse.Namespace) -> int:
    disc = discovery.discover(root)

    def keep(entries: list) -> list:
        out = []
        for e in entries:
            if args.host_only and not e.host:
                continue
            if args.cuda_only and not e.cuda:
                continue
            out.append(e)
        return out

    kernels = keep(disc.kernels) if not args.comm else []
    comm = keep(disc.comm) if not args.kernels else []

    if args.json:
        payload = {
            "version": discovery.version(root),
            "root": str(root),
            "kernels": [_entry_dict(e) for e in kernels],
            "comm": [_entry_dict(e) for e in comm],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not kernels and not comm:
        print(f"vkl: no kernels found under {root} (src/c/vkernels missing?)",
              file=sys.stderr)
        return 1

    printed = 0
    if kernels:
        _print_section("KERNELS", kernels, "CATEGORY",
                       lambda e: e.category)
        printed += len(kernels)
    if comm:
        if printed:
            print()
        _print_section("COMMUNICATION PRIMITIVES", comm, "KIND",
                       lambda e: e.kind)
        printed += len(comm)
    noun = "entry" if printed == 1 else "entries"
    print(f"\n{printed} {noun} in vkernels {discovery.version(root)}")
    return 0


def _cmd_info(root: Path, name: str) -> int:
    disc = discovery.discover(root)
    by_name = {e.name: e for e in [*disc.kernels, *disc.comm]}
    entry = by_name.get(name)
    if entry is None:
        print(f"vkl: unknown kernel or primitive: {name}", file=sys.stderr)
        suggestions = difflib.get_close_matches(name, by_name, n=3, cutoff=0.4)
        if suggestions:
            print(f"vkl: did you mean: {', '.join(suggestions)}?",
                  file=sys.stderr)
        return 2
    print(f"name:        {entry.name}")
    print(f"kind:        {entry.kind}")
    print(f"category:    {entry.category}")
    print(f"description: {entry.description}")
    print(f"signature:   {entry.signature}")
    print(f"host:        {'yes' if entry.host else 'no'}")
    print(f"cuda:        {'yes' if entry.cuda else 'no'}")
    print(f"hip:         {'yes' if entry.hip else 'no'}")
    print(f"header:      {entry.header}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0
    try:
        root = discovery.resolve_root(args.root)
    except FileNotFoundError as e:
        print(f"vkl: error: {e}", file=sys.stderr)
        return 2
    if args.command == "list":
        return _cmd_list(root, args)
    return _cmd_info(root, args.name)
