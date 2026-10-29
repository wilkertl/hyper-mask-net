"""The label-free loader of descriptors and targets opens no judgments file (F10 research D2)."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from hyperdime.cli.__main__ import main
from hyperdime.contracts.validation import ContractError
from hyperdime.space.inputs import load_unlabeled


@pytest.fixture
def prepared(toy_configs: Path, tmp_path: Path) -> Path:
    out = tmp_path / "out"
    for command in ("prepare-data", "embed"):
        args = [command, "--config-dir", str(toy_configs), "--output", str(out), "data=toy"]
        assert main(args) == 0
    return out


def _qrels(out: Path) -> list[Path]:
    found = sorted((out / "datasets" / "toy").glob("*/qrels-*.jsonl"))
    assert found  # train, validation, test, and the evaluation set
    return found


def _snapshot(out: Path) -> tuple[list[str], np.ndarray, dict[str, list[str]], np.ndarray]:
    inputs = load_unlabeled(out, "toy")
    validation = inputs.splits["validation"]
    queries = inputs.queries(validation).numpy()
    return inputs.doc_ids, np.asarray(inputs.documents), inputs.splits, queries


def test_it_returns_documents_and_queries_by_split(prepared: Path) -> None:
    inputs = load_unlabeled(prepared, "toy")
    assert inputs.documents.shape == (12, 1024) and len(inputs.doc_ids) == 12
    assert set(inputs.splits) == {"train", "validation", "test"}
    assert all(ids == sorted(ids) for ids in inputs.splits.values())
    queries = inputs.queries(inputs.splits["train"][:3])
    assert queries.shape == (3, 1024) and str(queries.dtype) == "torch.float32"


def test_it_is_unchanged_with_every_judgments_file_deleted(prepared: Path) -> None:
    before = _snapshot(prepared)
    for path in _qrels(prepared):
        path.unlink()
    after = _snapshot(prepared)
    assert before[0] == after[0] and before[2] == after[2]
    assert np.array_equal(before[1], after[1]) and np.array_equal(before[3], after[3])


def test_it_never_opens_a_judgments_file(prepared: Path) -> None:
    before = _snapshot(prepared)
    for path in _qrels(prepared):
        path.write_text("{not json")
    after = _snapshot(prepared)
    assert before[2] == after[2] and np.array_equal(before[3], after[3])


def test_it_refuses_embeddings_made_from_another_dataset(prepared: Path) -> None:
    embeddings = next((prepared / "embeddings" / "toy").iterdir())
    manifest_path = embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_hash"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="was not made from"):
        load_unlabeled(prepared, "toy")


def test_it_needs_a_published_dataset(tmp_path: Path) -> None:
    shutil.rmtree(tmp_path, ignore_errors=True)
    with pytest.raises(ContractError):
        load_unlabeled(tmp_path, "toy")
