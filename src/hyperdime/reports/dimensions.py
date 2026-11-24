"""The interactive chart of each experiment's metric by retained dimensions.

Every value is read from a generated report table: E0's TREC DL tables (nDCG@10 and AP; its extra
k = 256 point is left out to keep 10% steps) and the ``progression.csv`` tables of E1, P1, and
S1-S4, and the P3 ranking tables: ``list-metrics.csv`` of P3-mask-ranking and ``metrics.csv`` of
the P3 oracle-against-L2S runs (SciFact and the NFCorpus and MS MARCO pilots). The page itself
(layout, plotting script, and the description of each model) is a fixed template; only the data
block is generated, so regenerating from the same tables gives identical bytes.
"""

from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path
from typing import Any

OUTPUT = "experiments-dimensions.html"
PROGRESSIONS = ("E1", "P1", "S1", "S2", "S3", "S4")
E0_QUERY_SETS = ("trec-dl-2019", "trec-dl-2020", "trec-dl-hard")
E0_METRICS = (("nDCG@10", "ndcg10"), ("AP", "ap"))
E0_DROPPED = "k=256"
PLACEHOLDER = "{{CHART_DATA}}"
BUDGETS = (102, 204, 307, 409, 512, 614, 716, 819, 921, 1024)
FULL = 1024
RANKING_METRICS = (
    ("nDCG@10", "ndcg@10"),
    ("MRR@10", "mrr@10"),
    ("MRR@100", "mrr@100"),
    ("Recall@100", "recall@100"),
)
MASK_RANKING = "P3-mask-ranking"
ORACLE_L2S = ("P3-oracle-l2s", "P3-oracle-l2s-nfcorpus-pilot", "P3-oracle-l2s-msmarco-pilot")


def chart_data(reports: Path) -> dict[str, Any]:
    """Per experiment: its metrics, and per ``<test>|<query set>`` group, each method's values at
    10%, 20%, ..., 100% of the dimensions, with the table each came from."""
    data = {"E0": _e0(reports)}
    data.update({experiment: _progression(reports, experiment) for experiment in PROGRESSIONS})
    data["P3R"] = _mask_ranking(reports)
    data["P3O"] = _oracle_l2s(reports)
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


def _mask_ranking(reports: Path) -> dict[str, Any]:
    """P3-mask-ranking's all-label metrics on multi-positive training queries. The full query is
    measured at k = 1024 only, which is every arm's 100% point; arms scored on a smaller cohort
    (k-k splits that need more positives) are left out, so every curve covers the same queries."""
    source = f"{MASK_RANKING}/tables/list-metrics.csv"
    rows = _rows(reports / source)
    full = {(r["environment"], r["metric"]): r for r in rows if r["arm"] == "full query" and r["k"]}
    curves: dict[tuple[str, str, str], dict[int, float]] = {}
    for row in rows:
        reference = full[(row["environment"], row["metric"])]
        if row["arm"] == "full query" or row["queries"] != reference["queries"]:
            continue
        curves.setdefault((row["environment"], row["metric"], row["arm"]), {})[int(row["k"])] = (
            float(row["mean"])
        )
    groups: dict[str, dict[str, Any]] = {}
    for (environment, metric), reference in full.items():
        name = f"{environment}|multi-positive training queries ({reference['queries']})"
        group = groups.setdefault(name, {"_source": {}})
        group["_source"][metric] = source
        value = float(reference["mean"])
        group[metric] = [{"method": "full query", "values": [value] * len(BUDGETS)}] + [
            {"method": arm, "values": [*(points[k] for k in BUDGETS[:-1]), value]}
            for (env, met, arm), points in curves.items()
            if env == environment and met == metric
        ]
    return {
        "metrics": [m for m, _ in RANKING_METRICS],
        "groups": groups,
        "source": source,
        "report": f"{MASK_RANKING}/README.md",
    }


def _oracle_l2s(reports: Path) -> dict[str, Any]:
    """The P3 oracle-against-L2S runs, one group per environment and positive-count cohort. Each
    cell holds a point estimate and its interval; only the estimate is charted."""
    groups: dict[str, dict[str, Any]] = {}
    for report in ORACLE_L2S:
        source = f"{report}/tables/metrics.csv"
        coverage = {r["environment"]: r for r in _rows(reports / report / "tables/coverage.csv")}
        curves: dict[tuple[str, str], dict[str, dict[int, float]]] = {}
        for row in _rows(reports / source):
            arms = curves.setdefault((row["environment"], row["cohort"]), {})
            for metric, column in RANKING_METRICS:
                arms.setdefault(f"{metric}|{row['arm']}", {})[int(row["k"])] = float(
                    row[column].split()[0]
                )
        for (environment, cohort), arms in curves.items():
            info = coverage[environment]
            queries = info[cohort]
            scope = "bounded pilot, " if info["capped"] == "True" else ""
            group: dict[str, Any] = {"_source": {m: source for m, _ in RANKING_METRICS}}
            for metric, _ in RANKING_METRICS:
                lines: list[dict[str, Any]] = []
                for key, points in arms.items():
                    name, arm = key.split("|", 1)
                    if name != metric:
                        continue
                    values = [points.get(k, points[FULL]) for k in BUDGETS]
                    line = {"method": arm, "values": values}
                    if set(points) == {FULL}:  # measured at 1024 only: the full query
                        lines.insert(0, line)
                    else:
                        lines.append(line)
                group[metric] = lines
            groups[f"{environment}|{cohort} training queries ({scope}{queries})"] = group
    return {
        "metrics": [m for m, _ in RANKING_METRICS],
        "groups": groups,
        "source": f"{ORACLE_L2S[0]}/tables/metrics.csv",
        "report": f"{ORACLE_L2S[0]}/README.md",
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
