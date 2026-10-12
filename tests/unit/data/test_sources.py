"""Pinned source readers, with local files and fakes in place of the network."""

import sys
import types
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from hyperdime.data.environments import EvalQuerySet, environment_spec
from hyperdime.data.sources import HuggingFaceSource, IrDatasetsQuerySet, read_qrels_tsv


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    """A fake ``hf_hub_download`` serving local files and recording each request."""
    pq.write_table(
        pa.table({"_id": ["d1", "d2"], "title": ["T", None], "text": ["one", "two"]}),
        tmp_path / "corpus.parquet",
    )
    pq.write_table(
        pa.table({"_id": ["q1"], "title": [""], "text": ["query"]}), tmp_path / "queries.parquet"
    )
    (tmp_path / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\n")
    files = {"corpus": "corpus.parquet", "queries": "queries.parquet", "train.tsv": "train.tsv"}
    requests: list[tuple[str, str, str]] = []

    def download(repo: str, filename: str, **kwargs: Any) -> str:
        requests.append((repo, filename, kwargs["revision"]))
        key = next(k for k in files if filename.startswith(k) or filename == k)
        return str(tmp_path / files[key])

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return requests


def test_hugging_face_source_reads_pinned_files(hub: list[tuple[str, str, str]]) -> None:
    spec = environment_spec("scifact")
    source = HuggingFaceSource(spec)
    assert list(source.iter_corpus()) == [("d1", "T", "one"), ("d2", "", "two")]
    assert list(source.iter_queries()) == [("q1", "query")]
    assert source.qrels("train") == {"q1": {"d1": 1, "d2": 0}}
    assert {revision for _, _, revision in hub} == {spec.source_revision, spec.qrels_revision}
    assert ("BeIR/scifact-qrels", "train.tsv", spec.qrels_revision) in hub


def test_ir_datasets_query_set(monkeypatch: pytest.MonkeyPatch) -> None:
    query = types.SimpleNamespace(query_id="1", text="how far")
    judgment = types.SimpleNamespace(query_id="1", doc_id="7", relevance=2)
    dataset = types.SimpleNamespace(
        queries_iter=lambda: iter([query]), qrels_iter=lambda: iter([judgment])
    )
    fake = types.ModuleType("ir_datasets")
    fake.load = lambda source: dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ir_datasets", fake)
    reader = IrDatasetsQuerySet(EvalQuerySet(name="dl", loader="ir_datasets", source="x/y"))
    assert list(reader.iter_queries()) == [("1", "how far")]
    assert reader.qrels() == {"1": {"7": 2}}


def test_qrels_tsv(tmp_path: Path) -> None:
    path = tmp_path / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\na\tx\t2\na\ty\t0\nb\tx\t1\n")
    assert read_qrels_tsv(path) == {"a": {"x": 2, "y": 0}, "b": {"x": 1}}
