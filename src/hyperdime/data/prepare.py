"""Prepare an environment into shared records with leakage-safe splits (F2 research D2, D3).

A prepared environment is one published artifact: documents, queries, one judgments file per
split and per evaluation query set, and a ``DatasetManifest``. Test and evaluation-set judgments
live only in their own files, and ``load_environment`` opens them only when evaluation asks.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from hyperdime.contracts.config import DataConfig
from hyperdime.contracts.hashing import config_hash, file_sha256
from hyperdime.contracts.manifests import ArtifactWriter
from hyperdime.contracts.schemas import (
    DatasetManifest,
    DocumentRecord,
    QrelsRecord,
    QueryRecord,
    load_record,
    load_records,
)
from hyperdime.contracts.validation import ContractError
from hyperdime.data.environments import EnvironmentSpec
from hyperdime.data.sources import Qrels, QuerySetReader, SourceReader
from hyperdime.data.splits import split_ids

SPLIT_SEED = 0
DOCUMENTS = "documents.jsonl"
QUERIES = "queries.jsonl"
MANIFEST = "dataset.json"
TRAINING_SPLITS = ("train", "validation")


def qrels_file(name: str) -> str:
    return f"qrels-{name}.jsonl"


def queries_file(eval_set: str) -> str:
    return f"queries-{eval_set}.jsonl"


def document_text(document: DocumentRecord) -> str:
    """The text that is embedded: ``title text`` when there is a title (as BEIR and the spike)."""
    title, text = document.title.strip(), document.text.strip()
    return f"{title} {text}".strip() if title else text


@dataclass(frozen=True)
class DatasetBundle:
    """A prepared environment as loaded by a step; test judgments only if asked for."""

    environment: str
    manifest: DatasetManifest
    queries: dict[str, QueryRecord]
    eval_queries: dict[str, dict[str, QueryRecord]]
    qrels: dict[str, list[QrelsRecord]]
    documents: list[DocumentRecord] = field(default_factory=list)

    def query_ids(self, split: str) -> list[str]:
        return sorted(q for q, record in self.queries.items() if record.split == split)

    def positives(self, name: str) -> dict[str, set[str]]:
        """``{query_id: relevant doc ids}`` from one judgments file (relevance > 0)."""
        found: dict[str, set[str]] = {}
        for judgment in self.qrels[name]:
            if judgment.relevance > 0:
                found.setdefault(judgment.query_id, set()).add(judgment.doc_id)
        return found


def prepare_environment(
    spec: EnvironmentSpec,
    data: DataConfig,
    writer: ArtifactWriter,
    source: SourceReader,
    eval_sets: Mapping[str, QuerySetReader] | None = None,
) -> DatasetManifest:
    """Write the environment's records through ``writer``; returns its ``DatasetManifest``."""
    eval_sets = dict(eval_sets or {})
    missing = {s.name for s in spec.eval_query_sets} - set(eval_sets)
    if missing:
        raise ContractError(f"no reader for evaluation query sets {sorted(missing)}")

    with writer.path(DOCUMENTS).open("w", encoding="utf-8") as handle:
        doc_ids, dropped_documents = _write_documents(spec, source.iter_corpus(), handle)
    texts = _unique("query", source.iter_queries())

    dropped_qrels = 0
    excluded: dict[str, int] = {}
    judged: dict[str, Qrels] = {}
    names = ([spec.train_split] if spec.train_split else []) + [spec.test_split]
    for split_name in names:
        kept, dropped, without_positive = _clean(source.qrels(split_name), doc_ids, texts)
        dropped_qrels += dropped
        role = "train" if split_name == spec.train_split else "test"
        excluded[role] = without_positive
        judged[role] = kept

    train_ids = sorted(judged.get("train", {}))
    if spec.train_query_cap is not None and len(train_ids) > spec.train_query_cap:
        train_ids = sorted(random.Random(SPLIT_SEED).sample(train_ids, spec.train_query_cap))
    splits: dict[str, list[str]] = {"test": sorted(judged["test"])}
    if train_ids:
        splits["train"], splits["validation"] = split_ids(
            train_ids, data.validation_fraction, SPLIT_SEED
        )
    overlap = set(splits.get("train", [])) & set(splits["test"])
    if overlap:
        raise ContractError(f"queries in both train and test: {sorted(overlap)[:5]}")

    records = [
        QueryRecord(
            query_id=qid,
            text=texts[qid],
            instruction=spec.instruction,
            environment=spec.environment,
            split=split,
        )
        for split in ("train", "validation", "test")
        for qid in splits.get(split, [])
    ]
    _write_jsonl(writer.path(QUERIES), records)
    num_queries = {split: len(ids) for split, ids in splits.items()}
    num_qrels: dict[str, int] = {}
    for split, ids in splits.items():
        labels = judged["train" if split in TRAINING_SPLITS else "test"]
        num_qrels[split] = _write_qrels(writer.path(qrels_file(split)), spec, split, ids, labels)

    for query_set in spec.eval_query_sets:
        reader = eval_sets[query_set.name]
        set_texts = _unique(f"{query_set.name} query", reader.iter_queries())
        kept, dropped, without_positive = _clean(reader.qrels(), doc_ids, set_texts)
        dropped_qrels += dropped
        excluded[query_set.name] = without_positive
        ids = sorted(kept)
        _write_jsonl(
            writer.path(queries_file(query_set.name)),
            [
                QueryRecord(
                    query_id=qid,
                    text=set_texts[qid],
                    instruction=spec.instruction,
                    environment=spec.environment,
                    split="test",
                )
                for qid in ids
            ],
        )
        num_queries[query_set.name] = len(ids)
        num_qrels[query_set.name] = _write_qrels(
            writer.path(qrels_file(query_set.name)), spec, "test", ids, kept
        )

    files = {name: file_sha256(writer.path(name)) for name in sorted(_written(spec, splits))}
    manifest = DatasetManifest(
        environment=spec.environment,
        spec=spec.model_dump(mode="json"),
        num_documents=len(doc_ids),
        dropped_documents=dropped_documents,
        num_queries=num_queries,
        num_qrels=num_qrels,
        dropped_qrels=dropped_qrels,
        excluded_queries=excluded,
        split_seed=SPLIT_SEED,
        validation_fraction=data.validation_fraction,
        files=files,
        fingerprint=config_hash(files),
    )
    writer.write_record(MANIFEST, manifest)
    return manifest


def load_environment(
    dataset_dir: Path, *, include_test_qrels: bool = False, with_documents: bool = False
) -> DatasetBundle:
    """Load a prepared environment for one step.

    Without ``include_test_qrels`` the test and evaluation-set judgments files are never opened;
    only evaluation code passes ``True``. Evaluation query texts are always returned (they carry
    no labels).
    """
    manifest = load_record(DatasetManifest, dataset_dir / MANIFEST)
    queries = {q.query_id: q for q in load_records(QueryRecord, dataset_dir / QUERIES)}
    eval_names = [s["name"] for s in manifest.spec.get("eval_query_sets", [])]
    eval_queries = {
        name: {q.query_id: q for q in load_records(QueryRecord, dataset_dir / queries_file(name))}
        for name in eval_names
    }
    readable = [s for s in TRAINING_SPLITS if (dataset_dir / qrels_file(s)).exists()]
    if include_test_qrels:
        readable += ["test", *eval_names]
    qrels = {name: load_records(QrelsRecord, dataset_dir / qrels_file(name)) for name in readable}
    documents = list(iter_documents(dataset_dir)) if with_documents else []
    return DatasetBundle(manifest.environment, manifest, queries, eval_queries, qrels, documents)


def iter_documents(dataset_dir: Path) -> Iterator[DocumentRecord]:
    """Stream the corpus in its stored order without holding it in memory."""
    with (dataset_dir / DOCUMENTS).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield DocumentRecord.model_validate_json(line)


def _write_documents(
    spec: EnvironmentSpec, corpus: Iterable[tuple[str, str, str]], handle: TextIO
) -> tuple[set[str], int]:
    """Stream documents to ``handle``; empty ones are dropped and counted, duplicates refused."""
    seen: set[str] = set()
    empty: set[str] = set()
    for doc_id, title, text in corpus:
        if doc_id in seen:
            raise ContractError(f"duplicate document id {doc_id!r} in {spec.source}")
        seen.add(doc_id)
        if not (title + text).strip():
            empty.add(doc_id)
            continue
        record = DocumentRecord(doc_id=doc_id, title=title, text=text, environment=spec.environment)
        handle.write(record.model_dump_json() + "\n")
    return seen - empty, len(empty)


def _unique(kind: str, pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    found: dict[str, str] = {}
    for key, text in pairs:
        if key in found:
            raise ContractError(f"duplicate {kind} id {key!r}")
        found[key] = text
    return found


def _clean(qrels: Qrels, doc_ids: set[str], texts: Mapping[str, str]) -> tuple[Qrels, int, int]:
    """Drop judgments of missing documents; keep queries that exist and still have a positive."""
    kept: Qrels = {}
    dropped = without_positive = 0
    for qid, labels in qrels.items():
        present = {doc: grade for doc, grade in labels.items() if doc in doc_ids}
        dropped += len(labels) - len(present)
        if qid in texts and texts[qid].strip() and any(g > 0 for g in present.values()):
            kept[qid] = present
        else:
            without_positive += 1
    return kept, dropped, without_positive


def _write_qrels(
    path: Path, spec: EnvironmentSpec, split: str, ids: list[str], labels: Qrels
) -> int:
    records = [
        QrelsRecord(
            query_id=qid,
            doc_id=doc,
            relevance=grade,
            environment=spec.environment,
            split=split,
        )
        for qid in ids
        for doc, grade in sorted(labels[qid].items())
    ]
    _write_jsonl(path, records)
    return len(records)


def _write_jsonl(path: Path, records: Iterable[QueryRecord | QrelsRecord]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _written(spec: EnvironmentSpec, splits: Mapping[str, list[str]]) -> list[str]:
    names = [DOCUMENTS, QUERIES, *(qrels_file(s) for s in splits)]
    for query_set in spec.eval_query_sets:
        names += [queries_file(query_set.name), qrels_file(query_set.name)]
    return names
