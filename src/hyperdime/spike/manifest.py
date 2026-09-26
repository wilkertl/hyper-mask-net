"""Lightweight run manifests for spike outputs (stand-in for the F1 ``RunManifest``)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


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
    """Record code version, config, and library versions next to a spike output.

    Steps write a ``"running"`` manifest before their first output and a ``"complete"`` one at
    the end, so an interrupted run never leaves outputs without a manifest.
    """
    if status not in ("running", "complete"):
        raise ValueError(f"status must be 'running' or 'complete', got {status!r}.")
    write_json(
        directory / "manifest.json",
        {
            "command": command,
            "status": status,
            "config": dict(config),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": _git("status", "--porcelain", "--untracked-files=no") != "",
            "versions": _versions(),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )


def require_complete(directory: Path) -> dict[str, Any]:
    """Return the manifest of a finished step, or raise if it is missing or was interrupted."""
    path = directory / "manifest.json"
    if not path.exists():
        raise ValueError(f"{directory} has no manifest; run the step that produces it.")
    manifest: dict[str, Any] = read_json(path)
    if manifest.get("status") != "complete":
        raise ValueError(
            f"{directory} was not completed (status {manifest.get('status')!r}); rerun its step."
        )
    return manifest


def _git(*args: str) -> str:
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
    transformers = sys.modules.get("transformers")
    if transformers is not None:
        versions["transformers"] = str(transformers.__version__)
    return versions
