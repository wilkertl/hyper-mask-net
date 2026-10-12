"""Run manifests and the artifact writer: no artifact without a manifest (research R8, R9).

An artifact is a directory ``<root>/<kind>/<run_id>/``. ``ArtifactWriter`` stages files in a
hidden sibling directory, writes ``manifest.json`` last, and publishes with one directory rename,
so an interrupted write never leaves a visible artifact without its manifest.
"""

from __future__ import annotations

import os
import platform
import secrets
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path, PurePosixPath
from types import TracebackType
from typing import Annotated, Any, Literal, Self

import numpy as np
import torch
from pydantic import AwareDatetime, BaseModel, ConfigDict, StringConstraints, model_validator

from hyperdime.contracts.config import RunConfig
from hyperdime.contracts.hashing import NO_DATASET, config_hash, file_sha256
from hyperdime.contracts.schemas import Dtype, Record, Sha256, dump_record, load_record
from hyperdime.contracts.validation import ContractError

MANIFEST = "manifest.json"
RunId = Annotated[str, StringConstraints(pattern=r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")]
GitCommit = Annotated[str, StringConstraints(pattern=r"^([0-9a-f]{40}|unknown)$")]


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, StringConstraints(min_length=1)]
    sha256: Sha256


class RunManifest(Record):
    """Provenance attached to every artifact (spec FR-006)."""

    schema_version: Literal[1] = 1
    run_id: RunId
    command: Annotated[str, StringConstraints(min_length=1)]
    git_commit: GitCommit
    git_dirty: bool | None
    config_hash: Sha256
    config: dict[str, Any]
    dataset_hash: Sha256 | Literal["none"]
    model_revision: Annotated[str, StringConstraints(min_length=1)]
    seed: int
    determinism: dict[str, str | bool | None]
    device: Annotated[str, StringConstraints(min_length=1)]
    dtype: Dtype
    versions: dict[str, str]
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    artifacts: list[ArtifactEntry] = []

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.seed < 0:
            raise ValueError("seed: must be >= 0")
        if self.finished_at is not None and self.finished_at < self.started_at:
            raise ValueError("finished_at: must not be before started_at")
        return self


def resolve_device(requested: str) -> str:
    """The device a run will actually use: ``auto`` or an unavailable GPU fall back to CPU."""
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return "cuda:0" if requested == "cuda" else requested


def start_manifest(
    command: str,
    config: RunConfig | Mapping[str, Any],
    *,
    dataset_hash: str = NO_DATASET,
    model_revision: str | None = None,
    device: str | None = None,
    seed: int | None = None,
    dtype: str | None = None,
    determinism: Mapping[str, str | bool | None] | None = None,
) -> RunManifest:
    """A manifest for a run that is starting; ``finish_manifest`` completes it.

    A ``RunConfig`` supplies the seed, dtype, device, and model revision unless they are given;
    a plain mapping (for tools outside the config system) needs an explicit ``seed``.
    """
    if isinstance(config, RunConfig):
        payload = config.model_dump(mode="json")
        seed = config.run.seed if seed is None else seed
        dtype = dtype or config.run.dtype
        device = device or config.run.device
        model_revision = model_revision or config.model.revision
    else:
        payload = dict(config)
        if seed is None:
            raise ContractError("start_manifest: a seed is required when config is a mapping")
    commit, dirty = _git_state()
    now = datetime.now(UTC)
    record = {
        "run_id": f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}",
        "command": command,
        "git_commit": commit,
        "git_dirty": dirty,
        "config_hash": config_hash(payload),
        "config": payload,
        "dataset_hash": dataset_hash,
        "model_revision": model_revision or NO_DATASET,
        "seed": seed,
        "determinism": dict(determinism or {"PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED")}),
        "device": resolve_device(device or "cpu"),
        "dtype": dtype or "float32",
        "versions": _versions(),
        "started_at": now,
    }
    return _validated(record)


def finish_manifest(manifest: RunManifest, artifacts: Sequence[tuple[str, str]]) -> RunManifest:
    """Set ``finished_at`` and the list of published files with their SHA-256."""
    record = manifest.model_dump()
    record["finished_at"] = datetime.now(UTC)
    record["artifacts"] = [{"name": name, "sha256": digest} for name, digest in artifacts]
    return _validated(record)


def read_manifest(artifact_dir: Path) -> RunManifest:
    path = artifact_dir / MANIFEST
    if not path.is_file():
        raise ContractError(f"{artifact_dir} has no {MANIFEST}; it is not a valid artifact")
    return load_record(RunManifest, path)


class ArtifactWriter:
    """Stage files, then publish them atomically together with their manifest."""

    def __init__(self, root: Path, kind: str, manifest: RunManifest) -> None:
        if not isinstance(manifest, RunManifest):
            raise ContractError("refusing to write an artifact without a run manifest")
        self.manifest = manifest
        self._kind_dir = root / kind
        self._final = self._kind_dir / manifest.run_id
        self._staging = self._kind_dir / f".{manifest.run_id}.tmp-{uuid.uuid4().hex}"
        self._names: set[str] = set()
        self._committed = False

    def path(self, name: str) -> Path:
        """Where to write a payload file; it is listed in the manifest on commit."""
        parts = PurePosixPath(name).parts
        if not parts or name.startswith("/") or ".." in parts or name == MANIFEST:
            raise ContractError(f"invalid artifact file name {name!r}")
        target = self._staging.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._names.add(name)
        return target

    def write_record(self, name: str, record: Record) -> None:
        dump_record(record, self.path(name))

    def commit(self) -> Path:
        if self._committed:
            raise ContractError(f"{self._final} was already committed")
        if self._final.exists():
            raise ContractError(f"{self._final} already exists; artifacts are never overwritten")
        self._staging.mkdir(parents=True, exist_ok=True)
        entries = []
        for name in sorted(self._names):
            path = self._staging / name
            if not path.is_file():
                raise ContractError(f"staged file {name!r} was never written")
            entries.append((name, file_sha256(path)))
        self.manifest = finish_manifest(self.manifest, entries)
        dump_record(self.manifest, self._staging / MANIFEST)
        _fsync_tree(self._staging)
        try:
            os.rename(self._staging, self._final)
        except OSError as error:
            raise ContractError(
                f"{self._final} already exists; artifacts are never overwritten"
            ) from error
        self._committed = True
        return self._final

    def __enter__(self) -> ArtifactWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
        elif not self._committed:
            try:
                self.commit()
            except BaseException:
                shutil.rmtree(self._staging, ignore_errors=True)
                raise


def _validated(record: Mapping[str, Any]) -> RunManifest:
    try:
        return RunManifest.model_validate(record)
    except ValueError as error:
        raise ContractError(f"RunManifest: {error}") from error


def _git_state() -> tuple[str, bool | None]:
    """Commit and dirty flag of the working directory's checkout; untracked files do not count."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit, status.strip() != ""


def _versions() -> dict[str, str]:
    try:
        hyperdime = metadata.version("hyperdime-qwen")
    except metadata.PackageNotFoundError:
        hyperdime = "unknown"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "hyperdime": hyperdime,
    }


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
