"""The E1 report from synthetic result files (F4 US4)."""

from pathlib import Path

import pytest
from synthetic_e1 import build

from hyperdime.reports import e1

TABLES = {
    "g2", "transfer", "trec", "methods", "seeds", "behavior", "noise_floor",
    "environment_pairs", "fit", "payoff", "fixed_subset", "progression",
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


def _table(report: dict[str, bytes], name: str) -> list[list[str]]:
    import csv

    return list(csv.reader(report[f"tables/{name}.csv"].decode().splitlines()))


def test_the_trec_table_carries_seed_averaged_ap(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    results = e1.load(tmp_path / "artifacts")
    header, *rows = _table(_report(tmp_path / "artifacts", tmp_path / "E1"), "trec")
    cell = next(
        c for c in results.cells if c.query_set != "test" and c.train_environment == "scifact"
    )
    row = dict(zip(header, next(r for r in rows if r[0] == "scifact"), strict=True))
    assert row["selector AP"] == f"{cell.metrics['ap']:.4f}"


@pytest.mark.parametrize(
    ("table", "column", "value", "note"),
    [
        ("transfer", "train", "msmarco", "see note: msmarco, nfcorpus"),  # msmarco -> nfcorpus
        ("fit", "environment", "scifact", "—"),
        ("fit", "environment", "nfcorpus", "see note: nfcorpus"),
        ("seeds", "environment", "msmarco", "see note: msmarco"),
    ],
)
def test_rows_of_noted_environments_reference_their_note(
    tmp_path: Path, table: str, column: str, value: str, note: str
) -> None:
    build(tmp_path / "artifacts")
    header, *rows = _table(_report(tmp_path / "artifacts", tmp_path / "E1"), table)
    notes = {r[header.index("note")] for r in rows if r[header.index(column)] == value}
    assert note in notes


def test_the_nfcorpus_caveat_sits_beside_the_fit_table(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    text = _report(tmp_path / "artifacts", tmp_path / "E1")["README.md"].decode()
    fit = text.split("## Selector fit as a ranking")[1].split("## Payoff")[0]
    assert e1.NOTES["nfcorpus"] in fit


def test_the_global_selector_is_stated_to_use_every_environments_training_labels(
    tmp_path: Path,
) -> None:
    build(tmp_path / "artifacts")
    text = _report(tmp_path / "artifacts", tmp_path / "E1")["README.md"].decode()
    for section in ("## Payoff", "## Every method"):
        assert e1.GLOBAL_NOTE in text.split(section)[1]


def test_the_progression_table_has_every_budget_and_ends_at_full(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    results = e1.load(tmp_path / "artifacts")
    header, *rows = _table(_report(tmp_path / "artifacts", tmp_path / "E1"), "progression")
    assert header[4:-1] == ["10% (k=102)", "30% (k=307)", "50% (k=512)", "100% (k=1024)"]
    row = dict(zip(header, rows[0], strict=True))
    assert (row["test"], row["query set"], row["method"]) == ("msmarco", "test", "full")
    full = results.results[("msmarco", "test")]["full"].metrics["ndcg@10"]
    assert {row[h] for h in header[4:-1]} == {f"{full:.4f}"}


def test_the_progression_averages_seeds_and_matches_the_cell_at_307(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    results = e1.load(tmp_path / "artifacts")
    header, *rows = _table(_report(tmp_path / "artifacts", tmp_path / "E1"), "progression")
    cell = next(
        c
        for c in results.cells
        if (c.train_environment, c.test_environment, c.query_set) == ("scifact", "nfcorpus", "test")
    )
    found = next(r for r in rows if r[:3] == ["nfcorpus", "test", "selector:scifact"])
    row = dict(zip(header, found, strict=True))
    assert row["30% (k=307)"] == f"{cell.metrics['ndcg@10']:.4f}"
    assert row["protocol"] == "transfer"
    assert float(row["10% (k=102)"]) < float(row["30% (k=307)"]) < float(row["50% (k=512)"])


def test_the_headline_tables_ignore_other_budgets(tmp_path: Path) -> None:
    build(tmp_path / "artifacts")
    header, *rows = _table(_report(tmp_path / "artifacts", tmp_path / "E1"), "methods")
    assert {r[header.index("k")] for r in rows} == {"307", "1024"}
