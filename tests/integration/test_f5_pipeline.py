"""F5 commands end to end on the F4 toy stack: the retrieval grid, then seeded selectors."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest
import torch
from test_f4_pipeline import TOY_ENVS, f4_stack, run_command  # noqa: F401 - f4_stack is a fixture

from hyperdime.artifacts import latest_artifact
from hyperdime.baselines.learning_to_select import GeometricSelector
from hyperdime.cli.__main__ import main
from hyperdime.cli.transfer_matrix import BUDGETS
from hyperdime.contracts.manifests import read_manifest
from hyperdime.contracts.schemas import EvaluationResult, GateDecision, load_record, load_records
from hyperdime.training.retrieval_selection import retrieval_cells
from hyperdime.training.selection import load_selector

STEPS = "3"


def _grid(configs: Path, out: Path, env: str) -> int:
    return run_command(
        configs,
        out,
        "train-selector",
        "--recipe",
        "retrieval-grid",
        f"data={env}",
        "--steps",
        STEPS,
    )


@pytest.fixture(scope="module")
def f5_stack(
    f4_stack: tuple[Path, Path],  # noqa: F811
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, Path]]:
    """The F4 stack plus a retrieval grid per toy environment, on a copy."""
    configs, original = f4_stack
    out = tmp_path_factory.mktemp("f5") / "out"
    shutil.copytree(original, out)
    for env in TOY_ENVS:
        assert _grid(configs, out, env) == 0
    yield configs, out


def _grid_dir(out: Path, env: str) -> Path:
    return latest_artifact(out, "selectors/retrieval-grid", env)


def test_the_grid_trains_every_cell_and_records_its_selection(
    f5_stack: tuple[Path, Path],
) -> None:
    _, out = f5_stack
    directory = _grid_dir(out, "toya")
    record = json.loads((directory / "selection.json").read_text())
    keys = [cell.key for cell in retrieval_cells()]
    assert sorted(record["cells"]) == sorted(keys)
    assert record["objective"] == "retrieval" and record["selected"] in keys
    assert set(record["family_best"]) <= {
        "fixed-307-scratch", "fixed-307-l2s", "multi-budget-scratch", "multi-budget-l2s",
    }  # fmt: skip
    assert (
        record["l2s_start"]["run"] == latest_artifact(out, "selectors/seeded/toya", "seed-0").name
    )
    assert sorted(json.loads((directory / "history.json").read_text())) == sorted(keys)
    assert read_manifest(directory).seed == 0
    checkpoint, _ = load_selector(directory, record["selected"])
    assert checkpoint.kind == "per_domain" and checkpoint.environment == "toya"


def test_every_cell_is_scored_at_the_nine_budgets(f5_stack: tuple[Path, Path]) -> None:
    _, out = f5_stack
    record = json.loads((_grid_dir(out, "toyb") / "selection.json").read_text())
    for cell in record["cells"].values():
        assert list(cell["validation_ndcg@10"]) == [
            "102", "204", "307", "409", "512", "614", "716", "819", "921",
        ]  # fmt: skip


def _copy(f5_stack: tuple[Path, Path], tmp_path: Path) -> Path:
    out = tmp_path / "out"
    shutil.copytree(f5_stack[1], out)
    return out


def test_the_grid_never_sees_test_judgments(
    f5_stack: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs, original = f5_stack
    out = _copy(f5_stack, tmp_path)
    dataset = latest_artifact(out, "datasets", "toya")
    (dataset / "qrels-test.jsonl").unlink()
    (dataset / "qrels-evalset.jsonl").unlink()
    capsys.readouterr()
    assert _grid(configs, out, "toya") == 0
    rerun = Path(capsys.readouterr().out.strip().splitlines()[-1])
    selected = json.loads((_grid_dir(original, "toya") / "selection.json").read_text())["selected"]
    a = load_selector(_grid_dir(original, "toya"), selected)[1].state_dict()
    b = load_selector(rerun, selected)[1].state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)


def test_the_grid_refuses_an_l2s_start_that_changed(
    f5_stack: tuple[Path, Path], tmp_path: Path
) -> None:
    configs = f5_stack[0]
    out = _copy(f5_stack, tmp_path)
    payload = latest_artifact(out, "selectors/seeded/toyc", "seed-0") / "selector.safetensors"
    payload.write_bytes(payload.read_bytes()[:-1] + b"\0")
    assert _grid(configs, out, "toyc") == 1


SEEDS = (0, 1, 2)


def _seeded(configs: Path, out: Path, env: str, seed: int) -> int:
    return run_command(
        configs, out, "train-selector", "--recipe", "retrieval-seeded", f"data={env}",
        f"run.seed={seed}",
    )  # fmt: skip


def _compare(configs: Path, out: Path) -> int:
    return run_command(configs, out, "compare-selectors", "--envs", *TOY_ENVS)


def _selector_bytes(out: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in (out / "selectors").rglob("*") if p.is_file()}


class Comparison(NamedTuple):
    configs: Path
    out: Path
    run: Path
    selectors_before: dict[Path, bytes]


@pytest.fixture(scope="module")
def comparison(f5_stack: tuple[Path, Path], tmp_path_factory: pytest.TempPathFactory) -> Comparison:
    """Seeded retrieval selectors and one compare-selectors run, on a copy of the F5 stack."""
    configs, original = f5_stack
    out = tmp_path_factory.mktemp("s1") / "out"
    shutil.copytree(original, out)
    for env in TOY_ENVS:
        for seed in SEEDS:
            assert _seeded(configs, out, env, seed) == 0
    before = _selector_bytes(out)
    assert _compare(configs, out) == 0
    return Comparison(configs, out, latest_artifact(out, "evaluation/S1", "all"), before)


def test_each_seed_retrains_the_grids_selected_cell(comparison: Comparison) -> None:
    for env in TOY_ENVS:
        grid = json.loads((_grid_dir(comparison.out, env) / "selection.json").read_text())
        for seed in SEEDS:
            directory = latest_artifact(
                comparison.out, f"selectors/retrieval-seeded/{env}", f"seed-{seed}"
            )
            record = json.loads((directory / "selection.json").read_text())
            assert record["source"]["cell"] == grid["selected"] and record["seed"] == seed
            assert read_manifest(directory).seed == seed


def test_every_method_is_scored_at_every_budget_and_full_only_at_1024(
    comparison: Comparison,
) -> None:
    records = load_records(EvaluationResult, comparison.run / "results-toya-test.jsonl")
    by_method: dict[str, set[int]] = {}
    for r in records:
        by_method.setdefault(r.selector, set()).add(r.k)
        assert r.protocol == "in-domain"
    assert by_method.pop("full") == {1024}
    expected = {f"retrieval:toya:s{s}" for s in SEEDS} | {f"l2s:toya:s{s}" for s in SEEDS}
    assert expected <= set(by_method)
    assert any(m.startswith("family:") for m in by_method)
    assert all(ks == set(BUDGETS) for ks in by_method.values())


def test_compare_selectors_trains_nothing(comparison: Comparison) -> None:
    assert _selector_bytes(comparison.out) == comparison.selectors_before


def test_a_second_comparison_gives_identical_result_files(
    comparison: Comparison, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    shutil.copytree(comparison.out, out)
    assert _compare(comparison.configs, out) == 0
    rerun = sorted(p for p in latest_artifact(out, "evaluation/S1", "all").iterdir())
    second = {p.name: p.read_bytes() for p in rerun if p.name != "manifest.json"}
    first = {p.name: p.read_bytes() for p in comparison.run.iterdir() if p.name != "manifest.json"}
    assert second == first


def test_report_s1_and_decide_s1_run_on_the_comparison(
    comparison: Comparison, tmp_path: Path
) -> None:
    adr = tmp_path / "ADR-0009.md"
    assert main(["report", "S1", "--artifacts", str(comparison.out),
                 "--report-dir", str(tmp_path / "S1")]) == 0  # fmt: skip
    assert comparison.run.name in (tmp_path / "S1" / "README.md").read_text()
    assert main(["decide", "S1", "--artifacts", str(comparison.out), "--adr", str(adr)]) == 0
    record = load_record(
        GateDecision, latest_artifact(comparison.out, "decisions/S1", "") / "gate_decision.json"
    )
    assert record.gate == "S1" and record.evidence[0] == comparison.run.name
    assert "- **Status:** proposed" in adr.read_text()


def test_compare_selectors_refuses_a_seed_from_another_grid_cell(
    comparison: Comparison, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    shutil.copytree(comparison.out, out)
    path = latest_artifact(out, "selectors/retrieval-seeded/toyb", "seed-1") / "selection.json"
    record = json.loads(path.read_text())
    record["source"]["cell"] = "another-cell"
    path.write_text(json.dumps(record))
    assert _compare(comparison.configs, out) == 1


def test_probe_geometry_and_report_p1_run_on_the_comparison(
    comparison: Comparison, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    shutil.copytree(comparison.out, out)
    assert run_command(comparison.configs, out, "probe-geometry", "--envs", *TOY_ENVS) == 0
    run = latest_artifact(out, "evaluation/P1", "all")
    records = load_records(EvaluationResult, run / "results-toya-test.jsonl")
    assert {(r.selector, r.k) for r in records} == {("full", 1024)} | {
        ("variance", k) for k in BUDGETS
    }
    assert json.loads((run / "inputs.json").read_text())["s1"] == comparison.run.name
    assert (
        main(["report", "P1", "--artifacts", str(out), "--report-dir", str(tmp_path / "P1")]) == 0
    )
    assert run.name in (tmp_path / "P1" / "README.md").read_text()


@pytest.fixture(scope="module")
def s2_comparison(comparison: Comparison, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Geometric grids and seeds, then compare-selectors --experiment S2, on a copy of S1's."""
    out = tmp_path_factory.mktemp("s2") / "out"
    shutil.copytree(comparison.out, out)
    for env in TOY_ENVS:
        recipe = ("train-selector", "--recipe")
        assert run_command(comparison.configs, out, *recipe, "geometric-grid", f"data={env}",
                           "--steps", STEPS) == 0  # fmt: skip
        for seed in SEEDS:
            assert run_command(comparison.configs, out, *recipe, "geometric-seeded",
                               f"data={env}", f"run.seed={seed}") == 0  # fmt: skip
    assert (
        run_command(
            comparison.configs, out, "compare-selectors", "--envs", *TOY_ENVS, "--experiment", "S2"
        )
        == 0
    )
    return out


def test_the_geometric_grid_starts_every_cell_from_the_prior(s2_comparison: Path) -> None:
    record = json.loads(
        (
            latest_artifact(s2_comparison, "selectors/geometric-grid", "toya") / "selection.json"
        ).read_text()
    )
    assert len(record["cells"]) == 16
    assert {c["cell"]["init"] for c in record["cells"].values()} == {"prior"}
    _, model = load_selector(latest_artifact(s2_comparison, "selectors/geometric-grid", "toya"),
                             record["selected"])  # fmt: skip
    assert isinstance(model, GeometricSelector)


def test_s2_scores_the_geometric_selector_variance_and_the_s1_selector(
    s2_comparison: Path,
) -> None:
    run = latest_artifact(s2_comparison, "evaluation/S2", "all")
    methods = {r.selector for r in load_records(EvaluationResult, run / "results-toyb-test.jsonl")}
    assert {"variance", "geometric:toyb:s0", "retrieval:toyb:s0", "l2s:toyb:s0"} <= methods
    assert set(json.loads((run / "fit.json").read_text())) == {"l2s", "retrieval", "geometric"}


def test_report_s2_and_decide_s2_run_and_leave_s1_unchanged(
    s2_comparison: Path, comparison: Comparison, tmp_path: Path
) -> None:
    adr = tmp_path / "ADR-0010.md"
    assert main(["report", "S2", "--artifacts", str(s2_comparison),
                 "--report-dir", str(tmp_path / "S2")]) == 0  # fmt: skip
    assert main(["decide", "S2", "--artifacts", str(s2_comparison), "--adr", str(adr)]) == 0
    assert "- **Status:** proposed" in adr.read_text()
    assert main(["report", "S1", "--artifacts", str(s2_comparison),
                 "--report-dir", str(tmp_path / "S1a")]) == 0  # fmt: skip
    assert main(["report", "S1", "--artifacts", str(comparison.out),
                 "--report-dir", str(tmp_path / "S1b")]) == 0  # fmt: skip
    assert (tmp_path / "S1a" / "README.md").read_bytes() == (
        tmp_path / "S1b" / "README.md"
    ).read_bytes()
