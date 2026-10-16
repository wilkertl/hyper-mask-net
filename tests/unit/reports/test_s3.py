"""The S3 report and its drafted decision from synthetic result files (F8)."""

import csv
import io
from collections.abc import Mapping
from pathlib import Path

import pytest
from synthetic_s2 import build_track

from hyperdime.reports import s3

TABLES = {
    "s3", "comparison", "weights", "progression", "vs_full", "trec", "families", "grid",
    "overlap", "fit", "curves",
}  # fmt: skip
FAMILIES = ("fixed-307-weighted", "multi-budget-weighted")


def _build(root: Path, over: Mapping[str, Mapping[str, float]], **kwargs: object) -> Path:
    return build_track(
        root, "S3", "weighted", "weighted", FAMILIES, over,
        others=("retrieval", "geometric"), **kwargs,  # type: ignore[arg-type]
    )  # fmt: skip


def _rows(output: Path, table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((output / "tables" / f"{table}.csv").read_text())))


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("s3")
    _build(root / "artifacts", {"retrieval": {"scifact": 0.05, "nfcorpus": 0.05}})
    s3.write(root / "artifacts", root / "S3")
    return root / "S3"


def test_every_table_is_generated_and_regenerates_identically(report: Path, tmp_path: Path) -> None:
    assert {p.stem for p in (report / "tables").iterdir()} == TABLES
    s3.write(report.parent / "artifacts", tmp_path)
    assert (tmp_path / "README.md").read_bytes() == (report / "README.md").read_bytes()


@pytest.mark.parametrize(
    ("over", "met"),
    [
        ({"retrieval": {"scifact": 0.05, "nfcorpus": 0.05}}, [True, True]),
        ({"retrieval": {"scifact": 0.05}}, [True, False]),
        ({"retrieval": {"scifact": 0.05, "nfcorpus": 0.05}, "geometric": {"msmarco": -0.05}},
         [False, True]),
        ({"retrieval": {"scifact": 0.05, "nfcorpus": 0.05}, "variance": {"scifact": -0.05}},
         [False, True]),
    ],
)  # fmt: skip
def test_the_criteria_read_the_planted_differences(
    tmp_path: Path, over: Mapping[str, Mapping[str, float]], met: list[bool]
) -> None:
    _build(tmp_path, over)
    assert [c["met"] for c in s3.criteria(s3.load(tmp_path))] == met


def test_a_loss_names_the_environment_and_reference(tmp_path: Path) -> None:
    _build(tmp_path, {"geometric": {"msmarco": -0.05}})
    assert s3.criteria(s3.load(tmp_path))[0]["shown"] == "msmarco vs S2 selector"


def test_every_reference_is_compared_and_holm_corrected_over_three_environments(
    report: Path,
) -> None:
    rows = _rows(report, "comparison")
    groups = {(r["against"], r["summary"]) for r in rows}
    assert {a for a, _ in groups} == {"S1 selector", "S2 selector", "variance", "L2S"}
    assert all(sum((r["against"], r["summary"]) == g for r in rows) == 3 for g in groups)


def test_the_weight_table_reads_the_weight_at_the_kept_checkpoint(report: Path) -> None:
    rows = _rows(report, "weights")
    assert len(rows) == 9
    assert [r["prior weight at best step"] for r in rows[:3]] == ["0.2500", "0.5000", "0.7500"]


def test_the_progression_has_all_four_selectors(report: Path) -> None:
    methods = {r["method"] for r in _rows(report, "progression") if r["test"] == "nfcorpus"}
    assert {"variance", "l2s:nfcorpus", "retrieval:nfcorpus", "geometric:nfcorpus"} <= methods
    assert {"weighted:nfcorpus", "family:fixed-307-weighted:nfcorpus"} <= methods


def test_a_weighted_selector_that_kept_step_0_is_named_as_the_variance_method(
    tmp_path: Path,
) -> None:
    _build(tmp_path, {}, unchanged=("scifact",))
    assert "unchanged_scifact" in s3.decision(s3.load(tmp_path), s3.DEFAULT_ADR).context


def test_the_decision_records_s3_and_an_accepted_adr_is_kept(report: Path, tmp_path: Path) -> None:
    results = s3.load(report.parent / "artifacts")
    record = s3.decision(results, tmp_path / "adr.md")
    assert (record.gate, record.outcome) == ("S3", "pass")
    path = s3.write_adr(results, record, tmp_path / "adr.md")
    path.write_text(path.read_text().replace("proposed", "accepted", 1))
    with pytest.raises(ValueError, match="no longer proposed"):
        s3.write_adr(results, record, path)
