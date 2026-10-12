"""SHA-256 fingerprints of configs, files, and directory trees (research R5, R6)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from hyperdime.contracts.validation import ContractError

NO_DATASET: Final = "none"
_BLOCK = 1 << 20


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, UTF-8 text; non-finite floats are rejected."""
    try:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except ValueError as error:
        raise ContractError(f"cannot fingerprint non-finite values: {error}") from error


def config_hash(config: BaseModel | Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a validated config, independent of key order."""
    data = config.model_dump(mode="json") if isinstance(config, BaseModel) else dict(config)
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


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
