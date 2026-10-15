"""F5 commands end to end on the F4 toy stack: the retrieval grid, then seeded selectors."""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from test_f4_pipeline import TOY_ENVS, f4_stack, run_command  # noqa: F401 - f4_stack is a fixture

from hyperdime.artifacts import latest_artifact
from hyperdime.contracts.manifests import read_manifest
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
