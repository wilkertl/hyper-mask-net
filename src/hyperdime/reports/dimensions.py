"""The interactive chart of each experiment's metric by retained dimensions.

Every value is read from a generated report table: E0's TREC DL tables (nDCG@10 and AP; its extra
k = 256 point is left out to keep 10% steps) and the ``progression.csv`` tables of E1, P1, and
S1-S3. The page itself (layout and plotting script) is a fixed template; only the data block is
generated, so regenerating from the same tables gives identical bytes.
"""

from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path
from typing import Any

OUTPUT = "experiments-dimensions.html"
PROGRESSIONS = ("E1", "P1", "S1", "S2", "S3")
E0_QUERY_SETS = ("trec-dl-2019", "trec-dl-2020", "trec-dl-hard")
E0_METRICS = (("nDCG@10", "ndcg10"), ("AP", "ap"))
E0_DROPPED = "k=256"
PLACEHOLDER = "{{CHART_DATA}}"


def chart_data(reports: Path) -> dict[str, Any]:
    """Per experiment: its metrics, and per ``<test>|<query set>`` group, each method's values at
    10%, 20%, ..., 100% of the dimensions, with the table each came from."""
    data = {"E0": _e0(reports)}
    data.update({experiment: _progression(reports, experiment) for experiment in PROGRESSIONS})
    return data


def render(reports: Path) -> str:
    template = resources.files("hyperdime.reports").joinpath("templates/dimensions.html")
    page = template.read_text(encoding="utf-8")
    if page.count(PLACEHOLDER) != 1:
        raise ValueError(f"the chart template must hold {PLACEHOLDER} exactly once")
    # "</" cannot appear inside the script block; the tables hold no markup, so refuse it.
    blob = json.dumps(chart_data(reports), separators=(",", ":"), ensure_ascii=False)
    if "</" in blob:
        raise ValueError("a report table holds markup that cannot be embedded in the chart")
    return page.replace(PLACEHOLDER, blob)


def write(reports: Path, output: Path | None = None) -> Path:
    path = output or reports / OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(reports), encoding="utf-8")
    return path


def _progression(reports: Path, experiment: str) -> dict[str, Any]:
    source = f"{experiment}/tables/progression.csv"
    rows = _rows(reports / source)
    columns = [c for c in rows[0] if "(k=" in c]
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        group = groups.setdefault(
            f"{row['test']}|{row['query set']}", {"nDCG@10": [], "_source": {"nDCG@10": source}}
        )
        group["nDCG@10"].append(
            {"method": row["method"], "values": [float(row[c]) for c in columns]}
        )
    return {
        "metrics": ["nDCG@10"],
        "groups": groups,
        "source": source,
        "report": f"{experiment}/README.md",
    }


def _e0(reports: Path) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for query_set in E0_QUERY_SETS:
        group: dict[str, Any] = {"_source": {}}
        for metric, stem in E0_METRICS:
            source = f"E0/tables/dime_{query_set}_{stem}.csv"
            group["_source"][metric] = source
            group[metric] = [
                {
                    "method": row["method"],
                    "values": [
                        float(v) for c, v in row.items() if c.startswith("k=") and c != E0_DROPPED
                    ],
                }  # fmt: skip
                for row in _rows(reports / source)
            ]
        groups[f"msmarco|{query_set}"] = group
    return {
        "metrics": [metric for metric, _ in E0_METRICS],
        "groups": groups,
        "source": f"E0/tables/dime_{E0_QUERY_SETS[0]}_{E0_METRICS[0][1]}.csv",
        "report": "E0/README.md",
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
