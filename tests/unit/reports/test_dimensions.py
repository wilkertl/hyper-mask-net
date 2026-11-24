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
BUDGETS = dimensions.BUDGETS
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
    _p3_tables(root)
    return root


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _cells(value: float) -> str:
    return ",".join([f"{value} ({value} to {value})"] * len(dimensions.RANKING_METRICS))


def _p3_tables(root: Path) -> None:
    """P3-mask-ranking: the full query 0.5 at k = 1024 on 10 queries, prefix k/1000 on the same
    queries, and a 2-2 arm on 4 queries. Each oracle run, one environment each: the full query
    0.3 at k = 1024 and an oracle arm at k/1000; the SciFact run is uncapped, the others pilots."""
    budgets = BUDGETS[:-1]
    lines = ['environment,arm,k,metric,queries,mean,"full query, same queries",difference,ci']
    for metric, _ in dimensions.RANKING_METRICS:
        lines.append(f"scifact,full query,1024,{metric},10,0.5,,,")
        lines += [f"scifact,prefix,{k},{metric},10,{k / 1000},0.5,0,0 to 0" for k in budgets]
        lines += [f"scifact,2-2,{k},{metric},4,0.9,0.6,0,0 to 0" for k in budgets]
    _write(root / dimensions.MASK_RANKING / "tables/list-metrics.csv", "\n".join(lines) + "\n")
    columns = ",".join(c for _, c in dimensions.RANKING_METRICS)
    runs = zip(dimensions.ORACLE_L2S, ("scifact", "nfcorpus", "msmarco"), strict=True)
    for report, env in runs:
        rows = [f"environment,cohort,k,ratio,arm,{columns}"]
        rows.append(f"{env},multi-positive,1024,100%,full query,{_cells(0.3)}")
        rows += [f"{env},multi-positive,{k},x,oracle,{_cells(k / 1000)}" for k in BUDGETS]
        _write(root / report / "tables/metrics.csv", "\n".join(rows) + "\n")
        capped = env != "scifact"
        _write(
            root / report / "tables/coverage.csv",
            f"environment,single-positive,multi-positive,capped\n{env},5,7,{capped}\n",
        )


def _data(page: str) -> dict:  # type: ignore[type-arg]
    found = re.search(r'id="chart-data" type="application/json">(.*?)</script>', page, re.S)
    assert found is not None
    return json.loads(found.group(1))  # type: ignore[no-any-return]


def test_the_chart_data_is_read_from_the_report_tables(tmp_path: Path) -> None:
    data = _data(dimensions.render(_tables(tmp_path)))
    tenths = [round(v / 10, 1) for v in range(1, 11)]
    assert list(data) == ["E0", *dimensions.PROGRESSIONS, "P3R", "P3O"]
    assert data["S2"]["groups"]["scifact|test"]["nDCG@10"] == [
        {"method": "variance", "values": tenths}
    ]
    # E0's extra k = 256 point is dropped, so every curve has the ten 10% steps.
    assert data["E0"]["groups"]["msmarco|trec-dl-hard"]["AP"][0]["values"] == tenths
    assert data["E0"]["metrics"] == ["nDCG@10", "AP"]


def test_p3_mask_ranking_curves_end_at_the_full_query_on_one_cohort(tmp_path: Path) -> None:
    data = _data(dimensions.render(_tables(tmp_path)))["P3R"]
    group = data["groups"]["scifact|multi-positive training queries (10)"]
    assert data["metrics"] == ["nDCG@10", "MRR@10", "MRR@100", "Recall@100"]
    # The 2-2 arm covers 4 of the 10 queries, so it is left out. The full query is flat and is
    # every arm's 100% point.
    assert [line["method"] for line in group["Recall@100"]] == ["full query", "prefix"]
    assert group["nDCG@10"][0]["values"] == [0.5] * 10
    assert group["nDCG@10"][1]["values"] == [k / 1000 for k in dimensions.BUDGETS[:-1]] + [0.5]


def test_p3_oracle_runs_chart_point_estimates_and_mark_pilots(tmp_path: Path) -> None:
    groups = _data(dimensions.render(_tables(tmp_path)))["P3O"]["groups"]
    assert list(groups) == [
        "scifact|multi-positive training queries (7)",
        "nfcorpus|multi-positive training queries (bounded pilot, 7)",
        "msmarco|multi-positive training queries (bounded pilot, 7)",
    ]
    lines = groups["scifact|multi-positive training queries (7)"]["MRR@100"]
    # Only the estimate before the interval is charted; the full query comes first.
    assert lines[0] == {"method": "full query", "values": [0.3] * 10}
    assert lines[1] == {"method": "oracle", "values": [k / 1000 for k in dimensions.BUDGETS]}


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
