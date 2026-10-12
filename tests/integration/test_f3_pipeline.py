"""F3 commands end to end on the toy environment: F2 artifacts first, then selectors."""

import json
from pathlib import Path

import pytest
import torch

from hyperdime.cli.__main__ import main
from hyperdime.training import selection
from hyperdime.training.selection import load_selector


@pytest.fixture(autouse=True)
def tiny_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selection, "OFFICIAL_TEMPERATURES", (0.05, 0.2))
    monkeypatch.setattr(selection, "OFFICIAL_EPOCHS", (2, 4))
    monkeypatch.setattr(selection, "WIDE_TEMPERATURES", (0.02, 0.5))
    monkeypatch.setattr(selection, "WIDE_EPOCHS", (1, 3))
    monkeypatch.setattr(selection, "STOP_EPOCHS", 6)
    monkeypatch.setattr(selection, "ANCHOR_WEIGHTS", (1e-2,))


def _run(configs: Path, out: Path, command: str, *extra: str) -> int:
    return main([command, "--config-dir", str(configs), "--output", str(out), "data=toy", *extra])


def _stdout(capsys: pytest.CaptureFixture[str]) -> Path:
    return Path(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.fixture
def f2(toy_configs: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, Path]:
    out = tmp_path / "out"
    for command, extra in (
        ("prepare-data", ()),
        ("embed", ()),
        ("mine-negatives", ("--depth", "8", "--count", "4")),
    ):
        assert _run(toy_configs, out, command, *extra) == 0
    capsys.readouterr()
    return toy_configs, out


def _official(f2: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> Path:
    assert _run(*f2, "train-selector", "--recipe", "official") == 0
    return _stdout(capsys)


def test_official_selector_is_the_validation_best_cell(
    f2: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _official(f2, capsys)
    record = json.loads((directory / "selection.json").read_text())
    scores = {key: cell["validation_ndcg@10"] for key, cell in record["cells"].items()}
    assert len(scores) == 4 and record["chosen"] == max(scores, key=scores.get)
    # G1 (a) compares these on real data; on toy data both must simply be recorded.
    assert all(c["val_kl"] > 0 and c["uniform_kl"] > 0 for c in record["cells"].values())
    checkpoint, _ = load_selector(directory, "official")
    assert checkpoint.kind == "per_domain" and checkpoint.environment == "toy"


def test_candidates_record_every_family_and_the_selected_best(
    f2: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _official(f2, capsys)
    assert _run(*f2, "train-global", "--envs", "toy") == 0
    capsys.readouterr()
    assert _run(*f2, "train-selector", "--recipe", "candidates") == 0
    directory = _stdout(capsys)
    record = json.loads((directory / "selection.json").read_text())
    assert set(record["family_best"]) == {"wide-grid", "val-stopped", "global-regularized"}
    scores = record["family_scores"]
    assert record["selected"] == max(scores, key=lambda f: (scores[f], f == "official"))
    for name in ("wide-grid", "val-stopped", "global-regularized", "selected"):
        assert (directory / f"{name}.safetensors").exists()


def test_selectors_never_see_test_judgments(
    f2: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    first = _official(f2, capsys)
    dataset = next((f2[1] / "datasets" / "toy").iterdir())
    (dataset / "qrels-test.jsonl").unlink()
    (dataset / "qrels-evalset.jsonl").unlink()
    second = _official(f2, capsys)
    a, b = load_selector(first, "official")[1], load_selector(second, "official")[1]
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_evaluate_and_report_e0(
    f2: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configs, out = f2
    _official(f2, capsys)
    assert _run(configs, out, "evaluate") == 0
    directory = _stdout(capsys)
    checks = json.loads((directory / "checks.json").read_text())
    assert set(checks) == {"test", "evalset"}
    assert all(all(v.values()) for v in (c["k1024_identical"] for c in checks.values()))
    assert checks["evalset"]["relevance_threshold"] == 2
    report_dir = tmp_path / "E0"
    args = [
        "report",
        "E0",
        "--artifacts",
        str(out),
        "--envs",
        "toy",
        "--report-dir",
        str(report_dir),
    ]
    assert main(args) == 0
    text = (report_dir / "README.md").read_text()
    assert "## Gate G1 (research R7)" in text and "(d) the report regenerates" in text
    assert "DIME format: evalset (toy), ap by budget" in text
    assert (report_dir / "tables" / "l2s_style.csv").exists()
    first = {p.name: p.read_bytes() for p in report_dir.rglob("*") if p.is_file()}
    assert main(args) == 0
    assert {p.name: p.read_bytes() for p in report_dir.rglob("*") if p.is_file()} == first
