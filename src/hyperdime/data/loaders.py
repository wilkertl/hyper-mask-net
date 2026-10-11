"""BEIR dataset download and parsing."""

from __future__ import annotations

import csv
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hyperdime.contracts.hashing import file_sha256, tree_sha256

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

Qrels = dict[str, dict[str, int]]


@dataclass(frozen=True)
class BeirDataset:
    """A BEIR dataset: documents as ``title text``, queries, and qrels per split."""

    name: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, Qrels]


def download_beir(name: str, root: Path) -> Path:
    """Download and extract a BEIR dataset under ``root`` once; return its directory.

    The archive's SHA-256 and the extracted tree's SHA-256 are recorded in
    ``root/<name>.sha256.json`` (see ``beir_fingerprint``).
    """
    directory = root / name
    if (directory / "corpus.jsonl").exists():
        return directory
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{name}.zip"
    partial = archive.with_suffix(".zip.part")
    with urllib.request.urlopen(BEIR_URL.format(name=name)) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out)
    partial.rename(archive)
    zip_sha256 = file_sha256(archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(root)
    archive.unlink()
    if not (directory / "corpus.jsonl").exists():
        raise FileNotFoundError(f"{archive.name} did not contain {name}/corpus.jsonl.")
    _write_fingerprint(root, name, {"url": BEIR_URL.format(name=name), "zip_sha256": zip_sha256})
    return directory


def beir_fingerprint(root: Path, name: str) -> dict[str, str]:
    """Provenance of a downloaded dataset: source URL, archive SHA-256, and extracted-tree SHA-256.

    Datasets extracted before fingerprints were recorded get ``zip_sha256 = "unknown"`` and a
    freshly computed tree hash.
    """
    path = root / f"{name}.sha256.json"
    if not path.exists():
        _write_fingerprint(root, name, {"url": BEIR_URL.format(name=name), "zip_sha256": "unknown"})
    record: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return record


def _write_fingerprint(root: Path, name: str, record: dict[str, str]) -> None:
    record = {**record, "tree_sha256": tree_sha256(root / name)}
    path = root / f"{name}.sha256.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_beir(directory: Path) -> BeirDataset:
    """Parse ``corpus.jsonl``, ``queries.jsonl``, and every ``qrels/<split>.tsv``."""
    corpus: dict[str, str] = {}
    for record in _read_jsonl(directory / "corpus.jsonl"):
        title, text = str(record.get("title") or "").strip(), str(record.get("text") or "").strip()
        corpus[str(record["_id"])] = f"{title} {text}".strip() if title else text
    queries = {
        str(record["_id"]): str(record["text"])
        for record in _read_jsonl(directory / "queries.jsonl")
    }
    qrels: dict[str, Qrels] = {}
    for path in sorted((directory / "qrels").glob("*.tsv")):
        split: Qrels = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                split.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
        qrels[path.stem] = split
    return BeirDataset(directory.name, corpus, queries, qrels)


def judged_queries(qrels: Mapping[str, Mapping[str, int]], queries: Mapping[str, str]) -> list[str]:
    """Sorted IDs of queries that exist and have at least one positive judgment."""
    return sorted(
        qid
        for qid, labels in qrels.items()
        if qid in queries and any(g > 0 for g in labels.values())
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
