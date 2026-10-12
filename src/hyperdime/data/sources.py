"""Readers for pinned dataset sources (F2 research D1).

``HuggingFaceSource`` reads a BEIR dataset from the Hugging Face Hub at a pinned commit: corpus and
queries from parquet (streamed in batches), judgments from TSV. ``IrDatasetsQuerySet`` reads an
evaluation-only query set from ``ir_datasets``. Both are behind small protocols so tests inject
in-memory fakes and never touch the network.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from hyperdime.data.environments import EnvironmentSpec, EvalQuerySet

Qrels = dict[str, dict[str, int]]

CORPUS_FILE = "corpus/corpus-00000-of-00001.parquet"
QUERIES_FILE = "queries/queries-00000-of-00001.parquet"
BATCH_ROWS = 50_000


class SourceReader(Protocol):
    def iter_corpus(self) -> Iterator[tuple[str, str, str]]:
        """``(doc_id, title, text)`` in source order; may yield duplicate ids (checked later)."""
        ...

    def iter_queries(self) -> Iterator[tuple[str, str]]:
        """``(query_id, text)`` in source order."""
        ...

    def qrels(self, split: str) -> Qrels:
        """``{query_id: {doc_id: relevance}}`` for one source split."""
        ...


class QuerySetReader(Protocol):
    def iter_queries(self) -> Iterator[tuple[str, str]]: ...

    def qrels(self) -> Qrels: ...


class HuggingFaceSource:
    """A BEIR dataset on the Hugging Face Hub, pinned by repository commits."""

    def __init__(self, spec: EnvironmentSpec, cache_dir: Path | None = None) -> None:
        self.spec = spec
        self.cache_dir = cache_dir

    def _file(self, repo: str, filename: str, revision: str) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=self.cache_dir,
            )
        )

    def iter_corpus(self) -> Iterator[tuple[str, str, str]]:
        path = self._file(self.spec.source, CORPUS_FILE, self.spec.source_revision)
        for row in _parquet_rows(path):
            yield str(row["_id"]), str(row.get("title") or ""), str(row.get("text") or "")

    def iter_queries(self) -> Iterator[tuple[str, str]]:
        path = self._file(self.spec.source, QUERIES_FILE, self.spec.source_revision)
        for row in _parquet_rows(path):
            yield str(row["_id"]), str(row.get("text") or "")

    def qrels(self, split: str) -> Qrels:
        path = self._file(self.spec.qrels_source, f"{split}.tsv", self.spec.qrels_revision)
        return read_qrels_tsv(path)


class IrDatasetsQuerySet:
    """An evaluation-only query set and its judgments from ``ir_datasets``."""

    def __init__(self, query_set: EvalQuerySet) -> None:
        self.query_set = query_set

    def _dataset(self) -> Any:
        import ir_datasets

        return ir_datasets.load(self.query_set.source)

    def iter_queries(self) -> Iterator[tuple[str, str]]:
        for query in self._dataset().queries_iter():
            yield str(query.query_id), str(query.text)

    def qrels(self) -> Qrels:
        judgments: Qrels = {}
        for row in self._dataset().qrels_iter():
            judgments.setdefault(str(row.query_id), {})[str(row.doc_id)] = int(row.relevance)
        return judgments


def read_qrels_tsv(path: Path) -> Qrels:
    """BEIR judgments: a TSV with ``query-id``, ``corpus-id``, and ``score`` columns."""
    judgments: Qrels = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            judgments.setdefault(row["query-id"], {})[row["corpus-id"]] = int(row["score"])
    return judgments


def _parquet_rows(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    with pq.ParquetFile(path) as handle:
        for batch in handle.iter_batches(batch_size=BATCH_ROWS):
            yield from batch.to_pylist()
