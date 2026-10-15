"""The P1 report from synthetic P1 and S1 result files (F6)."""

import csv
import io
from pathlib import Path

import pytest
from synthetic_p1 import build

from hyperdime.reports import p1

TABLES = {"p1", "comparison", "promise", "dimensionality", "progression"}


def _rows(output: Path, table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((output / "tables" / f"{table}.csv").read_text())))


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p1")
    build(root / "artifacts", {"scifact": 0.8, "nfcorpus": 0.6, "msmarco": 0.1})
    p1.write(root / "artifacts", root / "P1")
    return root / "P1"


def test_every_table_is_generated_and_regenerates_identically(report: Path, tmp_path: Path) -> None:
    names = {p.stem for p in (report / "tables").iterdir()}
    assert names == TABLES
    p1.write(report.parent / "artifacts", tmp_path)
    assert (tmp_path / "README.md").read_bytes() == (report / "README.md").read_bytes()


def test_the_share_of_l2s_gain_is_recovered_per_environment(report: Path) -> None:
    shares = {r["test"]: float(r["share of L2S's gain"]) for r in _rows(report, "promise")}
    # The synthetic L2S carries seed noise (mean about +0.005), so shares sit a little low.
    assert shares == pytest.approx({"scifact": 0.8, "nfcorpus": 0.6, "msmarco": 0.1}, abs=0.05)


def test_two_of_three_recovering_half_reads_promising(report: Path) -> None:
    reading = _rows(report, "p1")[0]
    assert (reading["value"], reading["holds"]) == ("2 of 3 environments", "yes")


def test_one_of_three_recovering_half_does_not(tmp_path: Path) -> None:
    build(tmp_path, {"scifact": 0.8})
    p1.write(tmp_path, tmp_path / "P1")
    assert _rows(tmp_path / "P1", "p1")[0]["holds"] == "no"


def test_each_reference_is_holm_corrected_over_the_three_environments(report: Path) -> None:
    rows = _rows(report, "comparison")
    assert sorted({r["against"] for r in rows}) == ["L2S", "S1", "global"]
    assert all(sum(r["against"] == a for r in rows) == 3 for a in ("L2S", "S1", "global"))


def test_the_dimensionality_table_orders_environments_by_participation_ratio(report: Path) -> None:
    ratios = {r["environment"]: r["participation ratio"] for r in _rows(report, "dimensionality")}
    assert ratios == {"msmarco": "300.0", "nfcorpus": "200.0", "scifact": "100.0"}
    assert "scifact 100.0" in _rows(report, "p1")[1]["value"]


def test_the_progression_has_full_prefix_variance_and_both_selectors(report: Path) -> None:
    methods = {r["method"] for r in _rows(report, "progression") if r["test"] == "scifact"}
    assert {"full", "prefix", "variance", "l2s:scifact", "retrieval:scifact"} <= methods


def test_the_report_refuses_an_s1_run_other_than_the_one_it_was_scored_against(
    tmp_path: Path,
) -> None:
    build(tmp_path, {}, s1_run="another-run")
    with pytest.raises(ValueError, match="another-run"):
        p1.load(tmp_path)
