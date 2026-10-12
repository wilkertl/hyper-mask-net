"""Run manifests for spike outputs, written as ``RunManifest`` records (F1 contracts).

The spike keeps its in-place, resumable output directories rather than ``ArtifactWriter``
(ADR-0005), so a step writes its directory's manifest twice: when it starts (``finished_at``
unset, meaning running) and when it completes. Manifests written before F1, with a ``"status"``
field, stay readable so the F0 artifacts remain valid inputs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import (
    MANIFEST,
    RunManifest,
    finish_manifest,
    start_manifest,
)
from hyperdime.contracts.schemas import dump_record
from hyperdime.contracts.validation import ContractError

# Every spike seed (split, negatives, diagnostics) is 0; per-step seed lists are in the config.
BASE_SEED = 0


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically with sorted keys, so reruns produce identical bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.rename(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    directory: Path, command: str, config: Mapping[str, Any], status: str = "complete"
) -> None:
    """Record provenance next to a spike output: ``running`` before outputs, ``complete`` after.

    A complete manifest keeps the run id and start time of the running one it replaces and lists
    the files directly in ``directory`` with their SHA-256.
    """
    if status not in ("running", "complete"):
        raise ValueError(f"status must be 'running' or 'complete', got {status!r}.")
    manifest = _running(directory, command) or start_manifest(command, config, seed=BASE_SEED)
    if manifest.config != dict(config):
        manifest = start_manifest(command, config, seed=BASE_SEED)
    if status == "complete":
        manifest = finish_manifest(manifest, _files(directory))
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / f"{MANIFEST}.tmp"
    dump_record(manifest, tmp)
    tmp.rename(directory / MANIFEST)


def is_complete(manifest: Mapping[str, Any]) -> bool:
    """A finished step: ``finished_at`` set, or ``status: complete`` in a pre-F1 manifest."""
    if "status" in manifest:
        return bool(manifest["status"] == "complete")
    return manifest.get("finished_at") is not None


def require_complete(directory: Path) -> dict[str, Any]:
    """Return the manifest of a finished step, or raise if it is missing or was interrupted."""
    path = directory / MANIFEST
    if not path.exists():
        raise ValueError(f"{directory} has no manifest; run the step that produces it.")
    manifest: dict[str, Any] = read_json(path)
    if "status" not in manifest:
        try:
            RunManifest.model_validate(manifest)
        except ValueError as error:
            raise ContractError(f"{path} is not a valid RunManifest: {error}") from error
    if not is_complete(manifest):
        state = manifest.get("status", "running")
        raise ValueError(f"{directory} was not completed (status {state!r}); rerun its step.")
    return manifest


def _running(directory: Path, command: str) -> RunManifest | None:
    path = directory / MANIFEST
    if not path.exists():
        return None
    try:
        manifest = RunManifest.model_validate(read_json(path))
    except ValueError:
        return None  # a pre-F1 manifest or an unrelated file: start a new run
    return manifest if manifest.finished_at is None and manifest.command == command else None


def _files(directory: Path) -> list[tuple[str, str]]:
    if not directory.exists():
        return []
    return [
        (path.name, file_sha256(path))
        for path in sorted(directory.iterdir())
        if path.is_file() and not path.name.startswith(MANIFEST)
    ]
