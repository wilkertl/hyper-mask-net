"""SHA-256 fingerprints of files and directory trees (plan F1, research R6)."""

from __future__ import annotations

import hashlib
from pathlib import Path

_BLOCK = 1 << 20


def file_sha256(path: Path) -> str:
    """Streamed SHA-256 of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """SHA-256 over sorted ``"<relative posix path>\\t<file sha256>\\n"`` lines.

    Independent of enumeration order and file timestamps, so the same content hashes the same on
    every machine.
    """
    lines = sorted(
        f"{path.relative_to(root).as_posix()}\t{file_sha256(path)}\n"
        for path in root.rglob("*")
        if path.is_file()
    )
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
