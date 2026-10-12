"""F2 commands end to end on a toy environment: fake sources, stub encoder, no network."""

import json
import shutil
from pathlib import Path

import pytest

from hyperdime.cli import prepare_data
from hyperdime.cli.__main__ import main
from hyperdime.data import environments

from conftest import FakeQuerySet, toy_eval_sets, toy_source, toy_spec

CONFIGS = Path(__file__).parents[1] / "fixtures" / "configs" / "valid"


@pytest.fixture
def toy_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config tree whose `data=toy` is the registered toy environment with fake sources."""
    spec = toy_spec()
    monkeypatch.setitem(environments.ENVIRONMENTS, "toy", spec)  # type: ignore[index]
    monkeypatch.setattr(prepare_data, "SOURCE_FACTORY", lambda s: toy_source())
    eval_sets = toy_eval_sets()
    monkeypatch.setattr(prepare_data, "QUERY_SET_FACTORY", lambda s: eval_sets[s.name])
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    (configs / "data" / "toy.yaml").write_text(
        f"environment: toy\nsource: {spec.source}\ninstruction: {spec.instruction}\n"
        "validation_fraction: 0.1\n"
    )
    shutil.copy(configs / "experiment" / "diagnose.yaml", configs / "experiment" / "prepare.yaml")
    return configs


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
