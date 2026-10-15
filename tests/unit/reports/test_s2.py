"""The S2 report and its drafted decision from synthetic result files (F7)."""

import csv
import io
from pathlib import Path

import pytest
from synthetic_s2 import build

from hyperdime.reports import s2

TABLES = {
    "s2", "comparison", "progression", "vs_full", "trec", "families", "grid", "overlap", "fit",
    "curves",
}  # fmt: skip


def _rows(output: Path, table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((output / "tables" / f"{table}.csv").read_text())))


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("s2")
    build(root / "artifacts", {"scifact": 0.05, "nfcorpus": 0.05}, {})
    s2.write(root / "artifacts", root / "S2")
    return root / "S2"


def test_every_table_is_generated_and_regenerates_identically(report: Path, tmp_path: Path) -> None:
    assert {p.stem for p in (report / "tables").iterdir()} == TABLES
    s2.write(report.parent / "artifacts", tmp_path)
    assert (tmp_path / "README.md").read_bytes() == (report / "README.md").read_bytes()


@pytest.mark.parametrize(
    ("over_retrieval", "over_variance", "met"),
    [
        ({"scifact": 0.05, "nfcorpus": 0.05}, {}, [True, True]),
        ({"scifact": 0.05}, {}, [False, True]),
        ({"scifact": 0.05, "nfcorpus": 0.05}, {"msmarco": -0.05}, [True, False]),
    ],
)
def test_the_criteria_read_the_planted_differences(
    tmp_path: Path,
    over_retrieval: dict[str, float],
    over_variance: dict[str, float],
    met: list[bool],
) -> None:
    build(tmp_path, over_retrieval, over_variance)
    assert [c["met"] for c in s2.criteria(s2.load(tmp_path))] == met


def test_each_reference_and_summary_is_holm_corrected_over_the_three_environments(
    report: Path,
) -> None:
    rows = _rows(report, "comparison")
    groups = {(r["against"], r["summary"]) for r in rows}
    assert {a for a, _ in groups} == {"S1 selector", "variance", "L2S"}
    assert all(sum((r["against"], r["summary"]) == g for r in rows) == 3 for g in groups)


def test_the_trec_sets_are_compared_without_holm(report: Path) -> None:
    trec = _rows(report, "trec")
    assert {r["query set"] for r in trec} == {"trec-dl-2019"} and "p_holm" not in trec[0]


def test_the_progression_has_full_prefix_variance_and_all_three_selectors(report: Path) -> None:
    methods = {r["method"] for r in _rows(report, "progression") if r["test"] == "scifact"}
    assert {"full", "prefix", "variance", "l2s:scifact", "retrieval:scifact"} <= methods
    assert {"geometric:scifact", "family:fixed-307-prior:scifact"} <= methods


def test_a_geometric_selector_that_kept_its_prior_is_named_as_the_variance_method(
    tmp_path: Path,
) -> None:
    build(tmp_path, {}, {}, unchanged=("nfcorpus",))
    results = s2.load(tmp_path)
    assert "unchanged_nfcorpus" in s2.decision(results, s2.DEFAULT_ADR).context
    s2.write(tmp_path, tmp_path / "S2")
    assert (
        "`geometric:nfcorpus` is the `variance` method itself"
        in (tmp_path / "S2" / "README.md").read_text()
    )


def test_the_decision_records_s2_and_an_accepted_adr_is_kept(report: Path, tmp_path: Path) -> None:
    results = s2.load(report.parent / "artifacts")
    record = s2.decision(results, tmp_path / "adr.md")
    assert (record.gate, record.outcome) == ("S2", "pass")
    path = s2.write_adr(results, record, tmp_path / "adr.md")
    path.write_text(path.read_text().replace("proposed", "accepted", 1))
    with pytest.raises(ValueError, match="no longer proposed"):
        s2.write_adr(results, record, path)
