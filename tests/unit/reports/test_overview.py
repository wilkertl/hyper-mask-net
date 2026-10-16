"""The overview report from synthetic S3 result files (F8)."""

import csv
import io
from pathlib import Path

import pytest
from synthetic_s2 import build_track

from hyperdime.reports import overview

FAMILIES = ("fixed-307-weighted", "multi-budget-weighted")


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("overview")
    build_track(root / "artifacts", "S3", "weighted", "weighted", FAMILIES,
                {"retrieval": {"scifact": 0.05}}, others=("retrieval", "geometric"))  # fmt: skip
    overview.write(root / "artifacts", root / "overview")
    return root / "overview"


def _rows(report: Path, table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((report / "tables" / f"{table}.csv").read_text())))


def test_every_strategy_is_reported_on_every_test_set(report: Path) -> None:
    rows = _rows(report, "test")
    assert len(rows) == 3 * len(overview.STRATEGIES)
    assert {r["method"] for r in rows if r["test"] == "scifact"} >= {
        "full", "variance", "l2s:scifact", "retrieval:scifact", "geometric:scifact",
        "weighted:scifact",
    }  # fmt: skip


def test_the_full_query_has_one_value_at_every_budget(report: Path) -> None:
    full = next(r for r in _rows(report, "test") if r["method"] == "full")
    assert full["mean 10%-90%"] == full["30% (k=307)"] == full["nDCG@10 at best"]


def test_the_leader_is_the_masked_strategy_with_the_highest_mean(report: Path) -> None:
    rows = {r["test"]: r for r in _rows(report, "leaders")}
    test = [r for r in _rows(report, "test") if r["test"] == "scifact" and r["method"] != "full"]
    assert float(rows["scifact"]["mean 10%-90%"]) == max(float(r["mean 10%-90%"]) for r in test)


def test_the_trec_sets_are_reported_apart(report: Path) -> None:
    assert {r["query set"] for r in _rows(report, "trec")} == {"trec-dl-2019"}


def test_the_report_regenerates_identically_and_links_every_experiment(
    report: Path, tmp_path: Path
) -> None:
    overview.write(report.parent / "artifacts", tmp_path)
    assert (tmp_path / "README.md").read_bytes() == (report / "README.md").read_bytes()
    text = (report / "README.md").read_text()
    assert all(f"[{e}](../{e}/README.md)" in text for e, _, _ in overview.REPORTS)
