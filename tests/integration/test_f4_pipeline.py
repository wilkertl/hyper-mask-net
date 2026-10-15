"""F4 commands end to end on three toy environments: F2 and F3 first, then seeded selectors."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from safetensors.torch import save_file

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.methods import selector_importance
from hyperdime.cli import train_selector
from hyperdime.cli.__main__ import main
from hyperdime.contracts.hashing import file_sha256
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import (
    EvaluationResult,
    GateDecision,
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


def _matrix(configs: Path, out: Path) -> Path:
    assert run_command(configs, out, "transfer-matrix", "--envs", *TOY_ENVS) == 0
    return latest_artifact(out, "evaluation/E1", "all")


def _selector_bytes(out: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in (out / "selectors").rglob("*") if p.is_file()}


class Matrix(NamedTuple):
    out: Path
    run: Path
    selectors_before: dict[Path, bytes]


@pytest.fixture(scope="module")
def matrix(f4_stack: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Matrix:
    """One transfer-matrix run on a copy of the stack, shared by the read-only tests below."""
    out = tmp_path_factory.mktemp("matrix") / "out"
    shutil.copytree(f4_stack[1], out)
    before = _selector_bytes(out)
    return Matrix(out, _matrix(f4_stack[0], out), before)


def test_the_transfer_matrix_trains_nothing(matrix: Matrix) -> None:
    assert _selector_bytes(matrix.out) == matrix.selectors_before


def test_the_transfer_matrix_has_every_ordered_pair(matrix: Matrix) -> None:
    cells = load_records(TransferCell, matrix.run / "cells.jsonl")
    pairs = {(c.train_environment, c.test_environment) for c in cells if c.query_set == "test"}
    assert pairs == {(a, b) for a in TOY_ENVS for b in TOY_ENVS}


def test_every_cell_averages_all_seeds(matrix: Matrix) -> None:
    cells = load_records(TransferCell, matrix.run / "cells.jsonl")
    assert {tuple(c.seeds) for c in cells} == {SEEDS}


@pytest.mark.parametrize(
    ("method", "protocol"),
    [
        ("selector:toya:s0", "transfer"),
        ("fixed:toyc", "transfer"),
        ("selector:toyb:s2", "in-domain"),
        ("prefix", "in-domain"),
    ],
)
def test_results_from_other_environments_are_transfer_results(
    matrix: Matrix, method: str, protocol: str
) -> None:
    results = load_records(EvaluationResult, matrix.run / "results-toyb-test.jsonl")
    assert {r.selector: r.protocol for r in results}[method] == protocol


BUDGETS = [102, 204, 307, 409, 512, 614, 716, 819, 921]


@pytest.mark.parametrize("method", ["selector:toya:s1", "fixed:toyb", "random", "global-selector"])
def test_every_masked_method_is_scored_at_every_10_percent_budget(
    matrix: Matrix, method: str
) -> None:
    results = load_records(EvaluationResult, matrix.run / "results-toyc-test.jsonl")
    assert sorted(r.k for r in results if r.selector == method) == BUDGETS
    assert [r.k for r in results if r.selector == "full"] == [1024]


def test_each_budget_has_its_own_fixed_subset_of_k_dimensions(matrix: Matrix) -> None:
    subsets = json.loads((matrix.run / "fixed_subsets.json").read_text())
    assert {env: sorted(int(k) for k in by_k) for env, by_k in subsets.items()} == {
        env: BUDGETS for env in TOY_ENVS
    }
    assert all(len(s) == int(k) for by_k in subsets.values() for k, s in by_k.items())


def test_cells_stay_at_the_headline_budget(matrix: Matrix) -> None:
    assert {c.k for c in load_records(TransferCell, matrix.run / "cells.jsonl")} == {307}


def _result_files(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in directory.iterdir() if p.name != "manifest.json"}


def test_a_second_transfer_matrix_run_gives_identical_result_files(
    f4_stack: tuple[Path, Path], tmp_path: Path
) -> None:
    out = _copy(f4_stack, tmp_path)
    first = _result_files(_matrix(f4_stack[0], out))
    assert _result_files(_matrix(f4_stack[0], out)) == first


def test_behavior_compares_all_36_selector_pairs_9_within_an_environment(matrix: Matrix) -> None:
    pairs = json.loads((matrix.run / "behavior.json").read_text())["pairs"]
    assert (len(pairs), sum(p["within"] for p in pairs)) == (36, 9)


def test_criterion_2_enumerates_1680_assignments(matrix: Matrix) -> None:
    assert json.loads((matrix.run / "behavior.json").read_text())["assignments"] == 1680


def test_the_battery_takes_every_toy_test_query(matrix: Matrix) -> None:
    battery = json.loads((matrix.run / "battery.json").read_text())
    assert (battery["n"], sorted(battery["ids"])) == (10, sorted(TOY_ENVS))


@pytest.mark.parametrize(("env", "seed"), [(env, seed) for env in TOY_ENVS for seed in SEEDS])
def test_every_selector_property_names_the_battery(matrix: Matrix, env: str, seed: int) -> None:
    battery = json.loads((matrix.run / "battery.json").read_text())
    record = load_record(SelectorProperties, matrix.run / f"properties-{env}-s{seed}.json")
    assert record.battery_sha256 == battery["sha256"]


def _move_test_judgments(out: Path) -> None:
    """Point every test judgment at another document."""
    for env in TOY_ENVS:
        qrels = latest_artifact(out, "datasets", env) / "qrels-test.jsonl"
        moved = [json.loads(line) for line in qrels.read_text().splitlines()]
        for record in moved:
            record["doc_id"] = "d0" if record["doc_id"] != "d0" else "d1"
        qrels.write_text("".join(json.dumps(r) + "\n" for r in moved))


LABEL_FREE = ("behavior.json", "battery.json", "fit.json", "fixed_subsets.json")


def test_behavior_fit_and_subsets_never_read_test_judgments(
    f4_stack: tuple[Path, Path], matrix: Matrix, tmp_path: Path
) -> None:
    out = _copy((f4_stack[0], matrix.out), tmp_path)
    _move_test_judgments(out)
    second = _matrix(f4_stack[0], out)
    assert (second / "cells.jsonl").read_bytes() != (matrix.run / "cells.jsonl").read_bytes()
    assert {n: (second / n).read_bytes() for n in LABEL_FREE} == {
        n: (matrix.run / n).read_bytes() for n in LABEL_FREE
    }


def test_the_transfer_matrix_refuses_a_single_seed(
    f4_stack: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    configs, out = f4_stack
    assert run_command(configs, out, "transfer-matrix", "--envs", *TOY_ENVS, "--seeds", "0") == 1
    assert "at least two seeds" in capsys.readouterr().err


def _report(matrix: Matrix, report_dir: Path) -> dict[str, bytes]:
    args = ["report", "E1", "--artifacts", str(matrix.out), "--report-dir", str(report_dir)]
    assert main(args) == 0
    return {p.name: p.read_bytes() for p in report_dir.rglob("*") if p.is_file()}


def test_report_e1_regenerates_identically(matrix: Matrix, tmp_path: Path) -> None:
    assert _report(matrix, tmp_path / "E1") == _report(matrix, tmp_path / "E1")


def test_report_e1_names_its_transfer_matrix_run(matrix: Matrix, tmp_path: Path) -> None:
    readme = _report(matrix, tmp_path / "E1")["README.md"].decode()
    assert f"transfer-matrix `{matrix.run.name}`" in readme


def test_decide_g2_writes_a_record_and_a_proposed_adr(matrix: Matrix, tmp_path: Path) -> None:
    adr = tmp_path / "ADR-0007-g2-selector-shift.md"
    assert main(["decide", "G2", "--artifacts", str(matrix.out), "--adr", str(adr)]) == 0
    record = load_record(
        GateDecision, latest_artifact(matrix.out, "decisions", "G2") / "gate_decision.json"
    )
    assert (record.gate, record.evidence[0]) == ("G2", matrix.run.name)
    assert "- **Status:** proposed" in adr.read_text()


def test_the_transfer_matrix_refuses_a_seed_that_was_not_trained(
    f4_stack: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    configs, out = f4_stack
    seeds = ("--seeds", "0", "1", "2", "3")
    assert run_command(configs, out, "transfer-matrix", "--envs", *TOY_ENVS, *seeds) == 1
    assert "seed-3" in capsys.readouterr().err


def test_the_transfer_matrix_refuses_a_selector_that_differs_from_its_record(
    f4_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _copy(f4_stack, tmp_path)
    payload = _seeded(out, "toyc", 2) / "selector.safetensors"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0xFF  # one weight byte; the file still parses
    payload.write_bytes(bytes(data))
    assert run_command(f4_stack[0], out, "transfer-matrix", "--envs", *TOY_ENVS) == 1
    assert "does not match its record" in capsys.readouterr().err


def test_the_transfer_matrix_stops_on_non_finite_selector_scores(
    f4_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _copy(f4_stack, tmp_path)
    directory = _seeded(out, "toya", 1)
    _, model = load_selector(directory, "selector")
    weights = {k: torch.full_like(v, float("nan")) for k, v in model.state_dict().items()}
    save_file(weights, str(directory / "selector.safetensors"))
    record = json.loads((directory / "selector.json").read_text())
    record["payload"]["sha256"] = file_sha256(directory / "selector.safetensors")
    (directory / "selector.json").write_text(json.dumps(record))
    assert run_command(f4_stack[0], out, "transfer-matrix", "--envs", *TOY_ENVS) == 1
    assert "selector:toya:s1 gives non-finite scores" in capsys.readouterr().err


def test_seeded_training_stops_on_a_non_finite_validation_result(
    f4_stack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def nan_scores(trained: list[selection.Trained], *args: object, **kwargs: object) -> None:
        for t in trained:
            t.score = float("nan")

    monkeypatch.setattr(train_selector, "score_on_validation", nan_scores)
    out = _copy(f4_stack, tmp_path)
    assert _retrain(f4_stack[0], out, "toyb", 1) == 1
    assert "non-finite validation result" in capsys.readouterr().err
