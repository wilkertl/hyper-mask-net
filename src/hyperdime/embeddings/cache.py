"""Embedding caches: the spike's in-place cache and F2's sharded, published embeddings.

``embed_cached``/``load_cached`` serve the F0 spike unchanged. ``embed_sharded``,
``publish_embeddings``, and ``load_embeddings`` are F2's: resumable shards in a work directory,
published atomically through ``ArtifactWriter`` as ``EmbeddingArtifact`` records.
"""

from __future__ import annotations

import itertools
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hyperdime.contracts.config import ModelConfig
from hyperdime.contracts.hashing import config_hash, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter
from hyperdime.contracts.schemas import EmbeddingArtifact, PayloadRef, load_record
from hyperdime.contracts.validation import ContractError
from hyperdime.embeddings.qwen import EMBEDDING_DIM


def embed_cached(
    ids: Sequence[str],
    texts: Sequence[str],
    encode: Callable[[Sequence[str]], Tensor],
    directory: Path,
    chunk_size: int = 2048,
    settings: Mapping[str, Any] | None = None,
) -> tuple[list[str], Tensor]:
    """Return ``(ids, embeddings)`` from ``directory``, encoding missing chunks first.

    Each chunk is saved as soon as it is encoded, so an interrupted run resumes where it stopped.
    ``settings`` (model revision, token limit, dtype, ...) are stored with the cache; resuming or
    reusing it under different settings raises instead of mixing incompatible vectors.
    """
    if len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError("ids must be distinct and align with texts.")
    _check_settings(directory, dict(settings or {}))
    final = directory / "embeddings.npy"
    if final.exists():
        return load_cached(directory)
    parts: list[np.ndarray] = []
    for index, start in enumerate(range(0, len(ids), chunk_size)):
        part = directory / f"part-{index:05d}.npy"
        if not part.exists():
            chunk = encode(texts[start : start + chunk_size]).numpy().astype(np.float16)
            np.save(directory / f"part-{index:05d}.tmp.npy", chunk)
            (directory / f"part-{index:05d}.tmp.npy").rename(part)
        parts.append(np.load(part))
    np.save(directory / "embeddings.tmp.npy", np.concatenate(parts))
    (directory / "ids.json").write_text(json.dumps(list(ids)), encoding="utf-8")
    (directory / "embeddings.tmp.npy").rename(final)
    for part in directory.glob("part-*.npy"):
        part.unlink()
    return load_cached(directory)


def _check_settings(directory: Path, settings: dict[str, Any]) -> None:
    path = directory / "settings.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != settings:
            raise ValueError(
                f"{directory} was built with {stored}, not {settings}; delete it to re-embed."
            )
        return
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, sort_keys=True), encoding="utf-8")


def load_cached(directory: Path) -> tuple[list[str], Tensor]:
    """Load cached ``(ids, float32 embeddings)``."""
    ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
    embeddings = torch.from_numpy(np.load(directory / "embeddings.npy").astype(np.float32))
    if embeddings.shape[0] != len(ids):
        raise ValueError(f"{directory} has {embeddings.shape[0]} rows for {len(ids)} ids.")
    return list(ids), embeddings


# --- F2: sharded, resumable embedding published through ArtifactWriter (research D5) ---------


def check_settings(stored: Mapping[str, Any], requested: Mapping[str, Any]) -> None:
    """Refuse to mix vectors made with different settings, naming every differing key."""
    keys = sorted(k for k in set(stored) | set(requested) if stored.get(k) != requested.get(k))
    if keys:
        details = "; ".join(
            f"{k}: stored {stored.get(k)!r}, requested {requested.get(k)!r}" for k in keys
        )
        raise ContractError(f"embedding settings differ ({', '.join(keys)}): {details}")


def embed_sharded(
    ids: Sequence[str],
    texts: Iterable[str],
    encode: Callable[[Sequence[str]], Tensor],
    work_dir: Path,
    settings: Mapping[str, Any],
    shard_size: int = 65_536,
) -> list[Path]:
    """Encode ``texts`` (aligned with ``ids``) in shards saved under ``work_dir``; return them.

    A shard is saved as soon as it is encoded and skipped on a restart, so an interruption loses
    at most one shard. ``texts`` is consumed as a stream, so only one shard of text is in memory.
    The work directory records the settings and the ids; reusing it for anything else is refused.
    """
    if shard_size < 1:
        raise ContractError(f"shard_size must be >= 1, got {shard_size}")
    stored = {**dict(settings), "ids_sha256": config_hash({"ids": list(ids)}), "rows": len(ids)}
    _guard_work_dir(work_dir, stored)
    stream = iter(texts)
    parts: list[Path] = []
    for index, start in enumerate(range(0, len(ids), shard_size)):
        size = min(shard_size, len(ids) - start)
        part = work_dir / f"part-{index:05d}.npy"
        if part.exists():
            _skip(stream, size)
        else:
            chunk = list(itertools.islice(stream, size))
            if len(chunk) != size:
                raise ContractError(f"texts ended after {start + len(chunk)} of {len(ids)} ids")
            vectors = encode(chunk).float().numpy().astype(np.float16)
            if vectors.shape != (size, EMBEDDING_DIM):
                raise ContractError(f"encoder returned {vectors.shape}, expected ({size}, 1024)")
            tmp = work_dir / f"part-{index:05d}.tmp.npy"
            np.save(tmp, vectors)
            tmp.rename(part)
            print(
                f"{work_dir.name}: shard {index + 1} ({start + size}/{len(ids)})", file=sys.stderr
            )
        parts.append(part)
    if next(stream, None) is not None:
        raise ContractError(f"more texts than the {len(ids)} ids")
    return parts


def publish_embeddings(
    writer: ArtifactWriter,
    name: str,
    ids: Sequence[str],
    parts: Sequence[Path],
    role: str,
    environment: str,
    instruction: str | None,
    model: ModelConfig,
) -> EmbeddingArtifact:
    """Concatenate shards into ``<name>.npy`` (via a memmap, never fully in RAM) and write the
    ``EmbeddingArtifact`` record ``<name>.json`` next to it."""
    target = writer.path(f"{name}.npy")
    matrix = np.lib.format.open_memmap(
        target, mode="w+", dtype=np.float16, shape=(len(ids), EMBEDDING_DIM)
    )
    row = 0
    for part in parts:
        block = np.load(part, mmap_mode="r")
        matrix[row : row + block.shape[0]] = block
        row += block.shape[0]
    matrix.flush()
    del matrix
    if row != len(ids):
        raise ContractError(f"{name}: shards hold {row} rows for {len(ids)} ids")
    record = EmbeddingArtifact(
        role=role,
        environment=environment,
        ids=list(ids),
        num_rows=len(ids),
        dtype="float16",
        normalized=model.normalize,
        model_revision=model.revision,
        instruction=instruction if role == "queries" else None,
        payload=PayloadRef(path=f"{name}.npy", sha256=file_sha256(target), format="npy"),
    )
    writer.write_record(f"{name}.json", record)
    return record


def load_embeddings(
    artifact_dir: Path, name: str, *, mmap: bool = True
) -> tuple[EmbeddingArtifact, np.ndarray]:
    """An embedding record and its float16 matrix (memory-mapped by default)."""
    record = load_record(EmbeddingArtifact, artifact_dir / f"{name}.json")
    vectors = np.load(artifact_dir / record.payload.path, mmap_mode="r" if mmap else None)
    if vectors.shape != (record.num_rows, record.dim):
        raise ContractError(f"{artifact_dir / record.payload.path} has shape {vectors.shape}")
    return record, vectors


def _guard_work_dir(work_dir: Path, stored: Mapping[str, Any]) -> None:
    path = work_dir / "settings.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("ids_sha256") != stored["ids_sha256"]:
            raise ContractError(f"{work_dir} holds shards for different ids; delete it to restart")
        check_settings(previous, stored)
        return
    work_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(stored), sort_keys=True, indent=2), encoding="utf-8")


def _skip(stream: Iterator[str], count: int) -> None:
    skipped = sum(1 for _ in itertools.islice(stream, count))
    if skipped != count:
        raise ContractError("texts ended before the shards already on disk")
