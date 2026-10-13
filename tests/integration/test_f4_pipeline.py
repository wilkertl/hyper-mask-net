"""F4 commands end to end on three toy environments: F2 and F3 first, then seeded selectors."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.methods import selector_importance
from hyperdime.cli.__main__ import main
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import (
    EvaluationResult,
    SelectorProperties,
    TransferCell,
    load_record,
    load_records,
)
from hyperdime.training import selection
from hyperdime.training.selection import load_selector

from conftest import CONFIGS, StubEncoder, toy_eval_sets, toy_source, toy_spec

TOY_ENVS = ("toya", "toyb", "toyc")
SEEDS = (0, 1, 2)


def _shrink_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selection, "OFFICIAL_TEMPERATURES", (0.05, 0.2))
    monkeypatch.setattr(selection, "OFFICIAL_EPOCHS", (2, 4))
    monkeypatch.setattr(selection, "WIDE_TEMPERATURES", (0.02, 0.5))
    monkeypatch.setattr(selection, "WIDE_EPOCHS", (1, 3))
    monkeypatch.setattr(selection, "STOP_EPOCHS", 6)
    monkeypatch.setattr(selection, "ANCHOR_WEIGHTS", (1e-2,))


def register_toy_envs(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Toy environments sharing the toy corpus; their instructions differ, so their query
    embeddings differ. Returns the config directory."""
    from hyperdime.cli import embed, prepare_data
    from hyperdime.data import environments

    configs = root / "configs"
    shutil.copytree(CONFIGS, configs)
    shutil.copy(configs / "experiment" / "diagnose.yaml", configs / "experiment" / "prepare.yaml")
    for name in TOY_ENVS:
        spec = toy_spec().model_copy(
            update={"environment": name, "corpus_group": name, "instruction": f"Find {name}"}
        )
        monkeypatch.setitem(environments.ENVIRONMENTS, name, spec)  # type: ignore[index]
        (configs / "data" / f"{name}.yaml").write_text(
            f"environment: {name}\nsource: {spec.source}\ninstruction: {spec.instruction}\n"
            "validation_fraction: 0.1\n"
        )
    monkeypatch.setattr(prepare_data, "SOURCE_FACTORY", lambda s: toy_source())
    eval_sets = toy_eval_sets()
    monkeypatch.setattr(prepare_data, "QUERY_SET_FACTORY", lambda s: eval_sets[s.name])
    monkeypatch.setattr(embed, "ENCODER_FACTORY", lambda *args: StubEncoder())
    return configs


def run_command(configs: Path, out: Path, command: str, *extra: str) -> int:
    return main([command, "--config-dir", str(configs), "--output", str(out), *extra])


@pytest.fixture(scope="module")
def f4_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, Path]]:
    """F2 artifacts, F3 official, global, and candidate selectors, and three seeded selectors
    for each toy environment; built once per module."""
    root = tmp_path_factory.mktemp("f4")
    out = root / "out"
    with pytest.MonkeyPatch.context() as monkeypatch:
        _shrink_grids(monkeypatch)
        configs = register_toy_envs(monkeypatch, root)
        for env in TOY_ENVS:
            for command, extra in (
                ("prepare-data", ()),
                ("embed", ()),
                ("mine-negatives", ("--depth", "8", "--count", "4")),
                ("train-selector", ("--recipe", "official")),
            ):
                assert run_command(configs, out, command, f"data={env}", *extra) == 0
        assert run_command(configs, out, "train-global", "--envs", *TOY_ENVS) == 0
        for env in TOY_ENVS:
            assert (
                run_command(configs, out, "train-selector", f"data={env}", "--recipe", "candidates")
                == 0
            )
            for seed in SEEDS:
                assert (
                    run_command(
                        configs,
                        out,
                        "train-selector",
                        "--recipe",
                        "seeded",
                        f"data={env}",
                        f"run.seed={seed}",
                    )  # fmt: skip
                    == 0
                )
        yield configs, out


def _seeded(out: Path, env: str, seed: int) -> Path:
    return latest_artifact(out, f"selectors/seeded/{env}", f"seed-{seed}")


def test_each_seed_is_its_own_run_of_the_wide_grid_cell(f4_stack: tuple[Path, Path]) -> None:
    _, out = f4_stack
    for env in TOY_ENVS:
        candidates = json.loads(
            (latest_artifact(out, "selectors/candidates", env) / "selection.json").read_text()
        )
        for seed in SEEDS:
            directory = _seeded(out, env, seed)
            assert read_manifest(directory).seed == seed
            record = json.loads((directory / "selection.json").read_text())
            assert record["seed"] == seed and record["family"] == "wide-grid"
            assert record["source"]["cell"] == candidates["family_best"]["wide-grid"]
            checkpoint, _ = load_selector(directory, "selector")
            assert checkpoint.kind == "per_domain" and checkpoint.environment == env


def test_a_seeded_selector_reloads_with_identical_outputs(f4_stack: tuple[Path, Path]) -> None:
    _, out = f4_stack
    directory = _seeded(out, "toya", 1)
    queries = torch.randn(5, 1024)
    first, second = load_selector(directory, "selector")[1], load_selector(directory, "selector")[1]
    assert torch.equal(selector_importance(queries, first), selector_importance(queries, second))


def _copy(f4_stack: tuple[Path, Path], tmp_path: Path) -> Path:
    out = tmp_path / "out"
    shutil.copytree(f4_stack[1], out)
    return out


def _retrain(configs: Path, out: Path, env: str, seed: int) -> int:
    return run_command(
        configs, out, "train-selector", "--recipe", "seeded", f"data={env}", f"run.seed={seed}"
    )


def test_seeded_selectors_never_see_test_judgments(
    f4_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs, original = f4_stack
    out = _copy(f4_stack, tmp_path)
    dataset = latest_artifact(out, "datasets", "toya")
    (dataset / "qrels-test.jsonl").unlink()
    (dataset / "qrels-evalset.jsonl").unlink()
    assert _retrain(configs, out, "toya", 0) == 0
    retrained = Path(capsys.readouterr().out.strip().splitlines()[-1])
    a = load_selector(_seeded(original, "toya", 0), "selector")[1].state_dict()
    b = load_selector(retrained, "selector")[1].state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_seeded_training_refuses_other_inputs_than_f3s(
    f4_stack: tuple[Path, Path], tmp_path: Path
) -> None:
    configs = f4_stack[0]
    out = _copy(f4_stack, tmp_path)
    selection_file = latest_artifact(out, "selectors/candidates", "toyb") / "selection.json"
    record = json.loads(selection_file.read_text())
    record["inputs"]["negatives"] = "another-run"
    selection_file.write_text(json.dumps(record))
    assert _retrain(configs, out, "toyb", 0) == 1


def _matrix(configs: Path, out: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    capsys.readouterr()
    assert run_command(configs, out, "transfer-matrix", "--envs", *TOY_ENVS) == 0
    return Path(capsys.readouterr().out.strip().splitlines()[-1])


def _files(directory: Path) -> dict[str, bytes]:
    return {
        p.name: p.read_bytes()
        for p in directory.iterdir()
        if p.is_file() and p.name != "manifest.json"
    }


def test_the_transfer_matrix_has_every_ordered_pair_and_trains_nothing(
    f4_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs = f4_stack[0]
    out = _copy(f4_stack, tmp_path)
    selectors = {p: p.read_bytes() for p in (out / "selectors").rglob("*") if p.is_file()}
    first = _matrix(configs, out, capsys)
    assert {p: p.read_bytes() for p in (out / "selectors").rglob("*") if p.is_file()} == selectors
    cells = load_records(TransferCell, first / "cells.jsonl")
    test_cells = [c for c in cells if c.query_set == "test"]
    assert {(c.train_environment, c.test_environment) for c in test_cells} == {
        (a, b) for a in TOY_ENVS for b in TOY_ENVS
    }
    assert all(c.seeds == list(SEEDS) for c in cells)
    results = load_records(EvaluationResult, first / "results-toyb-test.jsonl")
    protocols = {r.selector: r.protocol for r in results}
    assert protocols["selector:toya:s0"] == "transfer" and protocols["fixed:toyc"] == "transfer"
    assert protocols["selector:toyb:s2"] == "in-domain" and protocols["prefix"] == "in-domain"
    assert _files(_matrix(configs, out, capsys)) == _files(first)


def test_behavior_compares_every_selector_pair_on_one_battery(
    f4_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _matrix(f4_stack[0], _copy(f4_stack, tmp_path), capsys)
    behavior = json.loads((directory / "behavior.json").read_text())
    assert len(behavior["pairs"]) == 36 and behavior["assignments"] == 1680
    assert sum(p["within"] for p in behavior["pairs"]) == 9
    battery = json.loads((directory / "battery.json").read_text())
    assert battery["n"] == 10 and set(battery["ids"]) == set(TOY_ENVS)
    for env in TOY_ENVS:
        for seed in SEEDS:
            record = load_record(SelectorProperties, directory / f"properties-{env}-s{seed}.json")
            assert record.battery_sha256 == battery["sha256"]


def test_behavior_fit_and_subsets_never_read_test_judgments(
    f4_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs = f4_stack[0]
    out = _copy(f4_stack, tmp_path)
    first = _matrix(configs, out, capsys)
    for env in TOY_ENVS:
        qrels = latest_artifact(out, "datasets", env) / "qrels-test.jsonl"
        moved = [json.loads(line) for line in qrels.read_text().splitlines()]
        for record in moved:
            record["doc_id"] = "d0" if record["doc_id"] != "d0" else "d1"
        qrels.write_text("".join(json.dumps(r) + "\n" for r in moved))
    second = _matrix(configs, out, capsys)
    for name in ("behavior.json", "battery.json", "fit.json", "fixed_subsets.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert (first / "cells.jsonl").read_bytes() != (second / "cells.jsonl").read_bytes()


def test_the_transfer_matrix_refuses_a_single_seed(
    f4_stack: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    configs, out = f4_stack
    assert run_command(configs, out, "transfer-matrix", "--envs", *TOY_ENVS, "--seeds", "0") == 1
    assert "at least two seeds" in capsys.readouterr().err
