"""Find published artifacts by kind and environment (``<root>/<kind>/<env>/<run_id>/``)."""

from __future__ import annotations

from pathlib import Path

from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.validation import ContractError


def latest_artifact(root: Path, kind: str, environment: str) -> Path:
    """The most recently finished published artifact of ``kind`` for ``environment``.

    Ordered by the manifest's ``finished_at`` (run ids only have one-second resolution);
    directories without a complete manifest are not artifacts and are skipped.
    """
    directory = root / kind / environment
    finished = []
    if directory.is_dir():
        for candidate in directory.iterdir():
            if not candidate.is_dir() or candidate.name.startswith("."):
                continue
            try:
                manifest = read_manifest(candidate)
            except ContractError:
                continue
            if manifest.finished_at is not None:
                finished.append((manifest.finished_at, candidate.name, candidate))
    if not finished:
        raise ContractError(f"no published {kind} artifact for {environment!r} under {root}")
    return max(finished)[2]
