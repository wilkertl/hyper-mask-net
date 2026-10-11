"""SCRAP for pytest: extraction pressure, file pressure, modes, verdicts, and the CLI."""

import ast
import json
import textwrap
from pathlib import Path
from typing import Any

import pytest

from scrap.analyze import analyze_tree
from scrap.cli import main
from scrap.report import verdict
from scrap.summary import (
    duplicate_clusters,
    duplication_cost,
    pressure_level,
    pressure_score,
    remediation_mode,
    stable,
)


def _summary(**overrides: Any) -> dict[str, Any]:
    summary = {
        "example_count": 5,
        "avg_scrap": 10.0,
        "max_scrap": 20.0,
        "effective_duplication_score": 0.0,
        "low_assertion_examples": 1,
        "zero_assertion_examples": 0,
        "branching_examples": 0,
        "with_redefs_examples": 0,
        "helper_hidden_example_count": 0,
        "case_matrix_repetition": 0,
        "coverage_matrix_candidates": 0,
        "subject_repetition_score": 0,
    }
    summary.update(overrides)
    return summary


def test_duplication_cost_by_hand() -> None:
    # (shared - 3) * (instances - 1) ** 1.5 / (variable + 1) = 2 * 2 ** 1.5 / 2
    assert duplication_cost(5, 3, 1) == pytest.approx(2**1.5)
    assert duplication_cost(3, 5, 0) == 0.0  # too few shared forms
    assert duplication_cost(6, 2, 5) == 0.0  # too many variable points


def test_pressure_score_and_level_by_hand() -> None:
    # 5 tests (size factor 1.0): 1.2 * 10 + 0.6 * 20 + 20 * (1 / 5) = 28
    summary = _summary()
    assert pressure_score(summary) == pytest.approx(28.0)
    assert pressure_level(summary) == "MEDIUM"


def test_small_files_and_coverage_matrices_lower_the_pressure() -> None:
    # Two tests get size factor 0.40: 0.40 * (1.2 * 10 + 0.6 * 20) - 1.5 matrix credit
    small = _summary(example_count=2, low_assertion_examples=0, case_matrix_repetition=1)
    assert pressure_score(small) == pytest.approx(0.40 * 24 - 1.5)


def test_small_clean_files_are_stable() -> None:
    assert stable(_summary(example_count=2, max_scrap=5.0, low_assertion_examples=0))


@pytest.mark.parametrize(
    ("overrides", "mode"),
    [
        ({"example_count": 2, "max_scrap": 5.0}, "STABLE"),
        ({}, "LOCAL"),
        ({"example_count": 12, "max_scrap": 40.0}, "SPLIT"),
    ],
)
def test_remediation_mode(overrides: dict[str, Any], mode: str) -> None:
    assert remediation_mode(_summary(**overrides), []) == mode


def test_structurally_identical_tests_form_one_duplicate_cluster() -> None:
    body = "\n".join(
        f"def test_{n}():\n    x = build({n}, 'a')\n    y = run(x)\n    assert y == {n}\n"
        for n in range(3)
    )
    examples = analyze_tree(ast.parse(body)).examples
    clusters = duplicate_clusters(examples)
    assert [[e.name for e in cluster] for cluster in clusters] == [["test_0", "test_1", "test_2"]]


@pytest.mark.parametrize(
    ("deltas", "expected"),
    [
        ({"file_score_delta": -6.0}, "improved"),
        ({"helper_hidden_delta": 1}, "worse"),
        ({"max_scrap_delta": 1.0}, "worse"),
        ({}, "unchanged"),
        ({"file_score_delta": 2.0}, "mixed"),
    ],
)
def test_comparison_verdicts(deltas: dict[str, float], expected: str) -> None:
    comparison = {
        "file_score_delta": 0.0,
        "extraction_pressure_delta": 0.0,
        "max_scrap_delta": 0.0,
        "helper_hidden_delta": 0,
        "case_matrix_delta": 0,
        **deltas,
    }
    assert verdict(comparison) == expected


def test_cli_writes_json_and_an_identical_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "test_good.py"
    good.write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    baseline = tmp_path / "baseline.json"
    assert main([str(good), "--json", "--write-baseline", str(baseline)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["reports"][0]["remediation_mode"] == "STABLE"
    assert json.loads(baseline.read_text()) == doc


def test_cli_compares_with_a_baseline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "test_good.py"
    good.write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    baseline = tmp_path / "baseline.json"
    main([str(good), "--write-baseline", str(baseline)])
    capsys.readouterr()
    assert main([str(good), "--compare", str(baseline)]) == 0
    assert "vs baseline: unchanged" in capsys.readouterr().out


def test_cli_fails_on_structural_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "test_bad.py"
    bad.write_text(textwrap.dedent("def test_x():\n    assert 1\n\ndef test_x():\n    assert 2\n"))
    assert main([str(tmp_path)]) == 1
    assert "shadows" in capsys.readouterr().out
