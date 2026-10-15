"""The E2 report and its drafted decision from synthetic result files (F5 US3, US4)."""

import csv
import io
from pathlib import Path

import pytest
from synthetic_e2 import BUDGETS, ENVS, build

from hyperdime.reports import e2

TABLES = {
    "e2", "comparison", "progression", "vs_full", "trec", "families", "grid", "overlap", "fit",
    "curves",
}  # fmt: skip


def _report(root: Path, output: Path) -> dict[str, bytes]:
    e2.write(root, output)
    return {
        p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }


def _rows(output: Path, table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((output / "tables" / f"{table}.csv").read_text())))


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One report from synthetic files with a gain planted in SciFact, shared by read-only tests."""
    root = tmp_path_factory.mktemp("e2")
    build(root / "artifacts", {"scifact": 0.05})
    e2.write(root / "artifacts", root / "E2")
    return root / "E2"


def test_every_table_is_generated_and_regenerates_identically(tmp_path: Path) -> None:
    build(tmp_path / "artifacts", {"scifact": 0.05})
    first = _report(tmp_path / "artifacts", tmp_path / "E2")
    assert {n.removeprefix("tables/").removesuffix(".csv") for n in first} == TABLES | {"README.md"}
    assert _report(tmp_path / "artifacts", tmp_path / "E2") == first


def test_the_budget_mean_is_the_per_query_mean_over_the_masked_budgets(tmp_path: Path) -> None:
    build(tmp_path, {})
    results = e2.load(tmp_path)
    key = ("scifact", "test")
    by_k = [e2.per_query(results, key, "prefix", k) for k in BUDGETS]
    mean = e2.summary(results, key, "prefix", e2.SUMMARIES[0])
    assert results.budgets == list(BUDGETS)
    assert mean["q3"] == pytest.approx(sum(v["q3"] for v in by_k) / len(BUDGETS))


def test_selectors_are_averaged_over_seeds_per_query(tmp_path: Path) -> None:
    build(tmp_path, {})
    results = e2.load(tmp_path)
    key = ("nfcorpus", "test")
    seeds = [results.results[key][(f"l2s:nfcorpus:s{s}", 307)].per_query["q0"] for s in (0, 1, 2)]
    averaged = e2.per_query(results, key, "l2s:nfcorpus", 307)["q0"]
    assert averaged == pytest.approx(sum(s["ndcg@10"] for s in seeds) / 3)


@pytest.mark.parametrize(
    ("gains", "met"),
    [
        ({"scifact": 0.05}, [False, True]),
        ({"scifact": 0.05, "nfcorpus": 0.05}, [True, True]),
        ({"scifact": 0.05, "nfcorpus": 0.05, "msmarco": -0.05}, [True, False]),
        ({}, [False, True]),
    ],
)
def test_the_criteria_find_planted_differences(
    tmp_path: Path, gains: dict[str, float], met: list[bool]
) -> None:
    build(tmp_path, gains)
    assert [c["met"] for c in e2.criteria(e2.load(tmp_path))] == met


def test_a_planted_gain_is_found_only_where_it_was_planted(tmp_path: Path) -> None:
    build(tmp_path, {"scifact": 0.05})
    rows = [r for r in e2.comparisons(e2.load(tmp_path)) if r["summary"] == e2.SUMMARIES[0]]
    significant = {r["environment"] for r in rows if r["p_holm"] <= 0.05}
    assert significant == {"scifact"}
    assert next(r for r in rows if r["environment"] == "scifact")["diff"] == pytest.approx(0.05)


def test_holm_counts_the_three_test_sets_once_per_summary(report: Path) -> None:
    rows = _rows(report, "comparison")
    assert {r["query set"] for r in rows} == {"test"}
    assert sorted((r["test"], r["summary"]) for r in rows) == sorted(
        (env, which) for env in ENVS for which in e2.SUMMARIES
    )


def test_the_trec_sets_are_tested_without_holm(report: Path) -> None:
    trec = _rows(report, "trec")
    assert {r["query set"] for r in trec} == {"trec-dl-2019"} and "p_holm" not in trec[0]


def test_the_progression_has_full_prefix_and_every_selector(report: Path) -> None:
    methods = {r["method"] for r in _rows(report, "progression") if r["test"] == "scifact"}
    assert {"full", "prefix", "l2s:scifact", "retrieval:scifact"} <= methods
    assert {"family:fixed-307-scratch:scifact", "family:multi-budget-l2s:scifact"} <= methods


def test_the_report_states_its_protocol_and_scoring(report: Path) -> None:
    assert "Protocol: in-domain. Scoring: exact" in (report / "README.md").read_text()


def test_overlap_has_one_row_per_environment_and_budget(report: Path) -> None:
    assert len(_rows(report, "overlap")) == len(ENVS) * len(BUDGETS)


def test_fit_is_reported_for_both_selectors(report: Path) -> None:
    assert {r["selector"] for r in _rows(report, "fit")} == {"l2s", "retrieval"}


def test_training_curves_report_the_best_step_and_final_soft_mass(report: Path) -> None:
    first = _rows(report, "curves")[0]
    assert (first["best step"], first["final soft mass"]) == ("1000", "0.9600")


def test_the_report_refuses_fewer_than_three_seeds(tmp_path: Path) -> None:
    build(tmp_path, {}, seeds=(0, 1))
    with pytest.raises(ValueError, match="at least 3 seeds"):
        e2.load(tmp_path)


def test_the_decision_passes_only_when_both_criteria_hold(tmp_path: Path) -> None:
    build(tmp_path, {"scifact": 0.05, "nfcorpus": 0.05})
    results = e2.load(tmp_path)
    record = e2.decision(results, e2.DEFAULT_ADR)
    assert record.gate == "E2" and record.outcome == "pass"
    assert record.evidence[0] == results.run


def test_an_accepted_adr_is_not_overwritten(tmp_path: Path) -> None:
    build(tmp_path, {})
    results = e2.load(tmp_path)
    record = e2.decision(results, tmp_path / "adr.md")
    path = e2.write_adr(results, record, tmp_path / "adr.md")
    assert "- **Status:** proposed" in path.read_text()
    path.write_text(path.read_text().replace("proposed", "accepted", 1))
    with pytest.raises(ValueError, match="no longer proposed"):
        e2.write_adr(results, record, path)
