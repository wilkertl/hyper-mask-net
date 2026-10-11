"""Resumable on-disk embedding cache: float16 ``.npy`` rows plus an ID list."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


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
