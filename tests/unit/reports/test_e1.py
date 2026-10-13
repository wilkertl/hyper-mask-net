"""The E1 report from synthetic result files (F4 US4)."""

from pathlib import Path

import pytest
from synthetic_e1 import build

from hyperdime.reports import e1

TABLES = {
    "g2", "transfer", "trec", "methods", "seeds", "behavior", "noise_floor",
    "environment_pairs", "fit", "payoff", "fixed_subset",
}  # fmt: skip


def _report(root: Path, output: Path) -> dict[str, bytes]:
    e1.write(root, output)
    return {
        p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }


def test_every_table_is_generated_and_regenerates_identically(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    first = _report(tmp_path / "artifacts", tmp_path / "E1")
    assert {name.removeprefix("tables/").removesuffix(".csv") for name in first} == TABLES | {
        "README.md"
    }
    assert _report(tmp_path / "artifacts", tmp_path / "E1") == first


def test_planted_shift_meets_both_criteria(tmp_path: Path) -> None:
    build(tmp_path)
    criteria = e1.criteria(e1.load(tmp_path))
    assert [c["met"] for c in criteria] == [True, True]
    assert criteria[0]["shown"] == "6 of 6 ordered pairs"


def test_comparison_tables_carry_full_and_prefix_and_state_protocols(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    report = _report(tmp_path / "artifacts", tmp_path / "E1")
    transfer = report["tables/transfer.csv"].decode()
    assert "full nDCG@10,prefix nDCG@10" in transfer.splitlines()[0]
    assert ",transfer," in transfer and ",in-domain," in transfer
    for name in ("payoff", "fixed_subset"):
        assert "full,prefix" in report[f"tables/{name}.csv"].decode().splitlines()[0]
    text = report["README.md"].decode()
    assert "Protocols: in-domain and transfer" in text and "Scoring: exact" in text


def test_notes_and_the_wide_grid_departure_are_stated(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    text = _report(tmp_path / "artifacts", tmp_path / "E1")["README.md"].decode()
    assert e1.NOTES["msmarco"] in text and e1.NOTES["nfcorpus"] in text
    assert "`msmarco`: F3 selected `global-regularized` (validation nDCG@10 0.5100)" in text
    assert "`scifact`: F3 selected" not in text


def test_the_report_refuses_fewer_than_three_seeds(tmp_path: Path) -> None:
    build(tmp_path, seeds=(0, 1))
    with pytest.raises(ValueError, match="at least 3 seeds"):
        e1.load(tmp_path)
