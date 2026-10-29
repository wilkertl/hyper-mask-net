"""The chart of each experiment's metric by retained dimensions, generated from report tables."""

import json
import re
from pathlib import Path

import pytest

from hyperdime.cli.__main__ import main
from hyperdime.reports import dimensions

PROGRESSION_HEAD = ["test", "query set", "method", *(f"{p}% (k={k})" for p, k in (
    (10, 102), (20, 204), (30, 307), (40, 409), (50, 512), (60, 614), (70, 716), (80, 819),
    (90, 921), (100, 1024)))]  # fmt: skip
E0_HEAD = ["method", *(f"k={k}" for k in (102, 204, 256, 307, 409, 512, 614, 716, 819, 921, 1024))]


def _tables(root: Path) -> Path:
    """E0 tables whose k=256 column is 9.0, and one progression per experiment whose values
    are 0.1, 0.2, ..., 1.0 at 10%, ..., 100%."""
    for query_set in dimensions.E0_QUERY_SETS:
        for _, stem in dimensions.E0_METRICS:
            path = root / "E0" / "tables" / f"dime_{query_set}_{stem}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            values = ["0.1", "0.2", "9.0", *(f"{v / 10:.1f}" for v in range(3, 11))]
            path.write_text(",".join(E0_HEAD) + "\nl2s," + ",".join(values) + "\n")
    for experiment in dimensions.PROGRESSIONS:
        path = root / experiment / "tables" / "progression.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = ["scifact", "test", "variance", *(f"{v / 10:.1f}" for v in range(1, 11))]
        path.write_text(",".join([*PROGRESSION_HEAD, "note"]) + "\n" + ",".join([*row, "—"]) + "\n")
    return root


def _data(page: str) -> dict:  # type: ignore[type-arg]
    found = re.search(r'id="chart-data" type="application/json">(.*?)</script>', page, re.S)
    assert found is not None
    return json.loads(found.group(1))  # type: ignore[no-any-return]


def test_the_chart_data_is_read_from_the_report_tables(tmp_path: Path) -> None:
    data = _data(dimensions.render(_tables(tmp_path)))
    tenths = [round(v / 10, 1) for v in range(1, 11)]
    assert list(data) == ["E0", *dimensions.PROGRESSIONS]
    assert data["S2"]["groups"]["scifact|test"]["nDCG@10"] == [
        {"method": "variance", "values": tenths}
    ]
    # E0's extra k = 256 point is dropped, so every curve has the ten 10% steps.
    assert data["E0"]["groups"]["msmarco|trec-dl-hard"]["AP"][0]["values"] == tenths
    assert data["E0"]["metrics"] == ["nDCG@10", "AP"]


def test_the_page_regenerates_identically_through_the_cli(tmp_path: Path) -> None:
    root = _tables(tmp_path)
    assert main(["report", "dimensions", "--report-dir", str(root)]) == 0
    first = (root / dimensions.OUTPUT).read_bytes()
    assert main(["report", "dimensions", "--report-dir", str(root)]) == 0
    assert (root / dimensions.OUTPUT).read_bytes() == first
    assert dimensions.PLACEHOLDER.encode() not in first


def test_the_committed_page_matches_the_committed_tables() -> None:
    reports = Path(__file__).resolve().parents[3] / "reports"
    if not (reports / dimensions.OUTPUT).exists():
        pytest.skip("no committed chart")
    assert dimensions.render(reports) == (reports / dimensions.OUTPUT).read_text(encoding="utf-8")
