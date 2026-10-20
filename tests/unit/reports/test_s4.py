"""The S4 report and its drafted decision from synthetic result files (F9)."""

import csv
import io
from collections.abc import Mapping
from pathlib import Path

import pytest
from synthetic_s4 import build

from hyperdime.reports import s4

TABLES = {
    "s4", "noninferiority", "comparison", "per_budget", "weights", "prior_share", "ablation",
    "kept_steps", "progression", "vs_full", "trec",
}  # fmt: skip
PASS = {"weighted-aligned": {"scifact": 0.05}}


def _rows(output: Path, table: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO((output / "tables" / f"{table}.csv").read_text())))


@pytest.fixture(scope="module")
def report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("s4")
    build(root / "artifacts", PASS)
    s4.write(root / "artifacts", root / "S4")
    return root / "S4"


def test_every_table_is_generated_and_regenerates_identically(report: Path, tmp_path: Path) -> None:
    assert {p.stem for p in (report / "tables").iterdir()} == TABLES
    s4.write(report.parent / "artifacts", tmp_path)
    assert (tmp_path / "README.md").read_bytes() == (report / "README.md").read_bytes()
    for name in TABLES:
        assert (tmp_path / "tables" / f"{name}.csv").read_bytes() == (
            report / "tables" / f"{name}.csv"
        ).read_bytes()


@pytest.mark.parametrize(
    ("over", "met"),
    [
        (PASS, [True, True]),
        ({}, [True, False]),  # equal to every reference: non-inferior, not better
        # 0.002 below aligned S3 in NFCorpus is inside the margin of 0.005
        ({"weighted-aligned": {"scifact": 0.05, "nfcorpus": -0.002}}, [True, True]),
        ({"weighted-aligned": {"scifact": 0.05, "nfcorpus": -0.01}}, [False, True]),
        ({**PASS, "geometric": {"msmarco": -0.01}}, [False, True]),
        ({**PASS, "variance": {"scifact": -0.01}}, [False, True]),
        # far below S1 and L2S is context, not a criterion
        ({**PASS, "retrieval": {"scifact": -0.05}, "l2s": {"scifact": -0.05}}, [True, True]),
    ],
)  # fmt: skip
def test_the_criteria_read_the_planted_differences(
    tmp_path: Path, over: Mapping[str, Mapping[str, float]], met: list[bool]
) -> None:
    build(tmp_path, over)
    assert [c["met"] for c in s4.criteria(s4.load(tmp_path))] == met


def test_a_failed_noninferiority_names_the_environment_and_reference(tmp_path: Path) -> None:
    build(tmp_path, {"weighted-aligned": {"nfcorpus": -0.01}})
    assert s4.criteria(s4.load(tmp_path))[0]["shown"] == "nfcorpus vs aligned S3"


def test_noninferiority_is_one_sided_with_the_margin_and_holm_over_three_environments(
    report: Path,
) -> None:
    rows = _rows(report, "noninferiority")
    assert {r["against"] for r in rows} == {"aligned S3", "S2 selector", "variance"}
    assert all(sum(r["against"] == a for r in rows) == 3 for a in {r["against"] for r in rows})
    assert {r["margin"] for r in rows} == {"0.0050"} and s4.MARGIN == 0.005
    assert all(r["non-inferior"] == "yes" for r in rows)


def test_every_reference_and_secondary_metric_is_compared(report: Path) -> None:
    rows = _rows(report, "comparison")
    assert {r["metric"] for r in rows} == {"ndcg@10", "mrr@10", "recall@100"}
    assert {(r["candidate"], r["against"]) for r in rows} == {
        *(("budget", a) for a in s4.AGAINST), ("weighted-aligned", "S3 selector")
    }  # fmt: skip


def test_the_per_budget_differences_cover_every_budget(report: Path) -> None:
    rows = _rows(report, "per_budget")
    assert len(rows) == 9 and {r["k"] for r in rows} == {"102", "307", "512"}
    assert {r["diff"] for r in rows if r["test"] == "scifact"} == {"+0.0500"}


def test_the_weights_diagnostics_and_kept_steps_are_read_from_the_records(report: Path) -> None:
    weights = _rows(report, "weights")
    budget = next(r for r in weights if r["track"] == "budget" and r["seed"] == "0")
    assert [budget[f"a at k={k}"] for k in (102, 307, 512)] == ["0.2000", "0.4000", "0.6000"]
    control = next(r for r in weights if r["track"] == "weighted-aligned" and r["seed"] == "1")
    assert {control[f"a at k={k}"] for k in (102, 307, 512)} == {"0.6000"}
    share = _rows(report, "prior_share")
    assert [r["prior share"] for r in share[:3]] == ["0.6000", "0.5000", "0.4000"]
    assert {r["removed - kept"] for r in _rows(report, "ablation")} == {"-0.2000"}
    kept = _rows(report, "kept_steps")
    assert {(r["kept step"], r["rerank-best step"]) for r in kept} == {("1000", "0")}


def test_the_readme_states_the_margin_and_the_test_set_reuse(report: Path) -> None:
    text = (report / "README.md").read_text()
    assert "δ = 0.005" in text and "informed the S4 proposal" in text
    assert "Protocol: in-domain" in text


@pytest.mark.parametrize("track", ["budget", "weighted-aligned"])
def test_fewer_than_three_seeds_per_track_is_refused(tmp_path: Path, track: str) -> None:
    build(tmp_path, PASS, seeds={track: (0, 1)})
    with pytest.raises(ValueError, match="at least 3 seeds"):
        s4.load(tmp_path)


def test_the_decision_records_s4_and_an_accepted_adr_is_kept(report: Path, tmp_path: Path) -> None:
    results = s4.load(report.parent / "artifacts")
    record = s4.decision(results, tmp_path / "adr.md")
    assert (record.gate, record.outcome) == ("S4", "pass")
    assert {"prior_weights", "prior_share", "ablation", "kept_steps", "test_set_reuse",
            "margin"} <= set(record.context)  # fmt: skip
    path = s4.write_adr(results, record, tmp_path / "adr.md")
    assert path.read_text().startswith("# ADR-0012")
    path.write_text(path.read_text().replace("proposed", "accepted", 1))
    with pytest.raises(ValueError, match="no longer proposed"):
        s4.write_adr(results, record, path)
