"""The drafted G2 decision and its proposed ADR (F4 US4)."""

import json
from pathlib import Path

import pytest
from synthetic_e1 import build

from hyperdime.reports import e1, g2

ADR = Path("ADR-0007-g2-selector-shift.md")


@pytest.mark.parametrize(
    ("options", "outcome"),
    [
        ({}, "pass"),
        ({"shift": 0.0}, "fail"),  # criterion 1 not met
        ({"p_value": 0.5}, "fail"),  # criterion 2 not met
        ({"within": 0.3, "cross": 0.8}, "fail"),  # cross above within
    ],
)
def test_the_outcome_passes_only_when_both_criteria_hold(
    tmp_path: Path, options: dict[str, float], outcome: str
) -> None:
    build(tmp_path, **options)  # type: ignore[arg-type]
    assert g2.decision(e1.load(tmp_path), ADR).outcome == outcome


def test_context_never_changes_the_outcome(tmp_path: Path) -> None:
    directory = build(tmp_path)
    before = g2.decision(e1.load(tmp_path), ADR)
    fit = json.loads((directory / "fit.json").read_text())
    for row in fit.values():
        row.update(fits=False, mean_gain=-0.2)
    (directory / "fit.json").write_text(json.dumps(fit))
    after = g2.decision(e1.load(tmp_path), ADR)
    assert after.outcome == before.outcome == "pass"
    assert after.context["fit"] != before.context["fit"]


def test_the_proposed_adr_names_its_evidence(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    results = e1.load(tmp_path / "artifacts")
    record = g2.decision(results, ADR)
    path = g2.write_adr(results, record, tmp_path / ADR)
    text = path.read_text()
    assert "- **Status:** proposed" in text and f"`{results.run}`" in text
    assert record.evidence[0] == results.run and len(record.evidence) == 1 + 9
    assert "read **pass**" in text


def test_an_adr_that_is_no_longer_proposed_is_not_overwritten(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    results = e1.load(tmp_path / "artifacts")
    path = tmp_path / ADR
    path.write_text("# ADR-0007\n\n- **Status:** accepted\n")
    with pytest.raises(ValueError, match="no longer proposed"):
        g2.write_adr(results, g2.decision(results, ADR), path)
    assert "accepted" in path.read_text()
