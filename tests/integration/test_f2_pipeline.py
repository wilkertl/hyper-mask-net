"""F2 commands end to end on a toy environment: fake sources, stub encoder, no network."""

import json
from pathlib import Path

import numpy as np
import pytest

from hyperdime.cli import embed, prepare_data
from hyperdime.cli.__main__ import main
from hyperdime.embeddings.cache import load_embeddings

from conftest import FakeQuerySet, StubEncoder, toy_eval_sets


def _run(configs: Path, output: Path, command: str, *extra: str) -> int:
    return main(
        [command, "--config-dir", str(configs), "--output", str(output), "data=toy", *extra]
    )


def test_prepare_data_publishes_a_dataset(
    toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(toy_configs, tmp_path / "out", "prepare-data") == 0
    directory = Path(capsys.readouterr().out.strip())
    assert directory.parent == tmp_path / "out" / "datasets" / "toy"
    dataset = json.loads((directory / "dataset.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert dataset["num_documents"] == 12
    names = {a["name"] for a in manifest["artifacts"]}
    assert {"documents.jsonl", "queries.jsonl", "qrels-test.jsonl", "dataset.json"} <= names


def test_prepare_data_refuses_a_config_that_disagrees_with_the_registry(
    toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(toy_configs, tmp_path / "out", "prepare-data", "data.source=BeIR/other") == 2
    assert "error: invalid config: data.source" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_failed_preparation_leaves_nothing(
    toy_configs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = toy_eval_sets()
    broken["evalset"] = FakeQuerySet([("e0", "x"), ("e0", "y")])
    monkeypatch.setattr(prepare_data, "QUERY_SET_FACTORY", lambda s: broken[s.name])
    assert _run(toy_configs, tmp_path / "out", "prepare-data") == 1
    assert "duplicate evalset query id 'e0'" in capsys.readouterr().err
    assert list((tmp_path / "out" / "datasets" / "toy").iterdir()) == []


def _prepared(configs: Path, output: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    assert _run(configs, output, "prepare-data") == 0
    return Path(capsys.readouterr().out.strip())


def test_embed_publishes_documents_queries_and_each_eval_set(
    toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    _prepared(toy_configs, out, capsys)
    assert _run(toy_configs, out, "embed", "--shard-size", "5") == 0
    directory = Path(capsys.readouterr().out.strip())
    documents, vectors = load_embeddings(directory, "documents")
    assert documents.num_rows == 12 and documents.instruction is None
    assert np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1.0, atol=1e-2)
    queries, _ = load_embeddings(directory, "queries")
    assert queries.num_rows == 29 and queries.instruction == "Find it"  # 17 + 2 + 10
    evalset, _ = load_embeddings(directory, "queries-evalset")
    assert evalset.ids == ["e0", "e1"]
    manifest = json.loads((directory / "manifest.json").read_text())
    dataset = json.loads((next((out / "datasets" / "toy").iterdir()) / "dataset.json").read_text())
    assert manifest["dataset_hash"] == dataset["fingerprint"]
    assert not list((out / ".work").rglob("part-*.npy"))  # work shards removed after publishing


def _embedded(configs: Path, out: Path, capsys: pytest.CaptureFixture[str]) -> str:
    _prepared(configs, out, capsys)
    assert _run(configs, out, "embed") == 0
    return capsys.readouterr().out.strip()


def test_embed_reuses_an_embedding_with_the_same_settings(
    toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _embedded(toy_configs, tmp_path / "out", capsys)
    assert _run(toy_configs, tmp_path / "out", "embed") == 0
    assert capsys.readouterr().out.strip() == first


def test_embed_refuses_an_embedding_with_other_settings(
    toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _embedded(toy_configs, tmp_path / "out", capsys)
    assert _run(toy_configs, tmp_path / "out", "embed", "model.max_length=256") == 1
    assert "embedding settings differ (max_length)" in capsys.readouterr().err


def test_an_interrupted_embed_resumes(
    toy_configs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    _prepared(toy_configs, out, capsys)
    monkeypatch.setattr(embed, "ENCODER_FACTORY", lambda *args: StubEncoder(fail_after=1))
    assert _run(toy_configs, out, "embed", "--shard-size", "5") == 1
    assert len(list((out / ".work").rglob("part-*.npy"))) == 1
    assert not (out / "embeddings" / "toy").exists() or not any(
        (out / "embeddings" / "toy").glob("2*")
    )
    resumed = StubEncoder()
    monkeypatch.setattr(embed, "ENCODER_FACTORY", lambda *args: resumed)
    assert _run(toy_configs, out, "embed", "--shard-size", "5") == 0
    assert len(resumed.encoded) == 12 - 5 + 29 + 2  # the first document shard was kept


def test_mine_negatives_never_reads_test_judgments(
    toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    dataset = _prepared(toy_configs, out, capsys)
    _run(toy_configs, out, "embed")
    capsys.readouterr()
    options = ("--depth", "8", "--count", "4")
    assert _run(toy_configs, out, "mine-negatives", *options) == 0
    first = Path(capsys.readouterr().out.strip())
    pools = [json.loads(line) for line in (first / "negatives.jsonl").read_text().splitlines()]
    assert {p["split"] for p in pools} == {"train", "validation"} and len(pools) == 19
    assert all(len(p["doc_ids"]) == 4 for p in pools)
    (dataset / "qrels-test.jsonl").unlink()
    (dataset / "qrels-evalset.jsonl").unlink()
    assert _run(toy_configs, out, "mine-negatives", *options) == 0
    second = Path(capsys.readouterr().out.strip())
    assert (second / "negatives.jsonl").read_bytes() == (first / "negatives.jsonl").read_bytes()
