"""Preparing an environment into shared records with leakage-safe splits (F2 US1)."""

from pathlib import Path

import pytest

from hyperdime.contracts.manifests import ArtifactWriter, start_manifest
from hyperdime.contracts.schemas import DatasetManifest
from hyperdime.contracts.validation import ContractError
from hyperdime.data.prepare import (
    document_text,
    iter_documents,
    load_environment,
    prepare_environment,
)

from conftest import FakeSource, toy_data, toy_eval_sets, toy_source, toy_spec


def _prepare(
    root: Path, source: FakeSource, *, cap: int | None = None
) -> tuple[Path, DatasetManifest]:
    spec = toy_spec(cap=cap)
    with ArtifactWriter(root, "datasets/toy", start_manifest("prepare-data", {}, seed=0)) as writer:
        manifest = prepare_environment(spec, toy_data(spec), writer, source, toy_eval_sets())
        return writer.commit(), manifest


def test_counts_and_exclusions(tmp_path: Path, source: FakeSource) -> None:
    _, manifest = _prepare(tmp_path, source)
    assert manifest.num_documents == 12 and manifest.dropped_documents == 1
    # 19 kept train queries (q0-q18): 2 validation (10% of 19, rounded), 17 train; 10 test; 2 eval
    assert manifest.num_queries == {"train": 17, "validation": 2, "test": 10, "evalset": 2}
    assert manifest.dropped_qrels == 3  # q0 -> missing, q19 -> missing, e2 -> missing
    assert manifest.excluded_queries == {"train": 1, "test": 0, "evalset": 1}
    assert manifest.num_qrels["test"] == 20  # two judgments per test query, grade 0 kept


def test_documents_keep_title_and_text(tmp_path: Path, source: FakeSource) -> None:
    directory, _ = _prepare(tmp_path, source)
    documents = {d.doc_id: d for d in iter_documents(directory)}
    assert "d12" not in documents
    assert document_text(documents["d1"]) == "Title 1 text of document 1"
    assert document_text(documents["d0"]) == "text of document 0"


def test_splits_are_disjoint_and_seeded(tmp_path: Path, source: FakeSource) -> None:
    directory, first = _prepare(tmp_path / "a", source)
    bundle = load_environment(directory)
    train, validation, test = (bundle.query_ids(s) for s in ("train", "validation", "test"))
    assert not (set(train) & set(validation)) and not (set(train + validation) & set(test))
    assert "q19" not in train + validation
    _, second = _prepare(tmp_path / "b", toy_source())
    assert first.fingerprint == second.fingerprint


def test_training_cap_samples_deterministically(tmp_path: Path) -> None:
    _, capped = _prepare(tmp_path / "a", toy_source(), cap=10)
    assert capped.num_queries["train"] + capped.num_queries["validation"] == 10
    _, again = _prepare(tmp_path / "b", toy_source(), cap=10)
    assert capped.fingerprint == again.fingerprint


@pytest.mark.parametrize("kind", ["document", "query"])
def test_duplicate_ids_fail_naming_the_id(tmp_path: Path, kind: str) -> None:
    source = toy_source()
    if kind == "document":
        source.corpus.append(("d3", "", "again"))
    else:
        source.queries.append(("q3", "again"))
    with pytest.raises(ContractError, match=f"duplicate {kind} id 'd3'|duplicate {kind} id 'q3'"):
        _prepare(tmp_path, source)
    assert not list((tmp_path / "datasets" / "toy").glob("2*"))


def test_evaluation_query_sets_get_their_own_files(tmp_path: Path, source: FakeSource) -> None:
    directory, manifest = _prepare(tmp_path, source)
    assert (directory / "queries-evalset.jsonl").exists()
    assert (directory / "qrels-evalset.jsonl").exists()
    assert set(manifest.files) >= {"qrels-evalset.jsonl", "qrels-test.jsonl", "documents.jsonl"}
